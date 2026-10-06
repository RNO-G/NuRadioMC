"""Preprocessing cost changes that must not change any result.

Two invariants: the CW peak-search pre-screen returns exactly the peaks of a search over
every window (so the subtracted trace is bit-identical), and restricting the preprocessor
to the reconstruction channels leaves the other channels untouched and the reconstruction
result unchanged.
"""

import functools
import os

import numpy as np
import pytest

import NuRadioReco.modules.channelSinewaveSubtraction as cws
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.parameters import channelParameters, channelParametersRNOG
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.RNO_G.channelPreprocessor import channelPreprocessor
from NuRadioReco.utilities import fft, units

from conftest import STATION
from synthetic import (HPOL_CHANNELS, N_NATIVE, NATIVE_RATE, SAMPLING_RATE, VPOL_CHANNELS,
                       antenna_locations, cylindrical_to_enu, filtered_trace)

FREQ_BAND = (0.1, 0.6)
PROMINENCE = 4.0
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _spectrum(trace):
    """Return the amplitude spectrum, band mask and bin width as sinewave_subtraction computes them."""
    trace = trace - np.mean(trace)
    spec = np.abs(fft.time2freq(trace, NATIVE_RATE))
    freqs = fft.freqs(len(trace), NATIVE_RATE)
    band_mask = (freqs >= FREQ_BAND[0]) & (freqs <= FREQ_BAND[1])
    return spec, band_mask, freqs[1] - freqs[0]


def _cw_trace(rng, lines, n=N_NATIVE):
    """Return unit white noise plus sinusoids given as (frequency in GHz, amplitude) pairs."""
    t = np.arange(n) / NATIVE_RATE
    trace = rng.normal(0.0, 1.0, n)
    for f, a in lines:
        trace += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return trace


def _peaks(trace, algorithm, prescreen):
    """Run the module's peak search on a trace."""
    spec, band_mask, delta_f = _spectrum(trace)
    return np.asarray(cws.find_cw_peaks(spec, band_mask, delta_f, algorithm, PROMINENCE,
                                        prescreen=prescreen))


def _count_find_peaks(monkeypatch):
    """Wrap scipy's find_peaks inside the module and return the call counter."""
    calls = [0]
    real = cws.signal.find_peaks

    def counted(*args, **kwargs):
        """Count the call, then defer to scipy."""
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cws.signal, 'find_peaks', counted)
    return calls


@pytest.mark.parametrize('algorithm', ['sliding', 'simple'])
def test_prescreen_matches_full_search_on_traces(algorithm):
    """Lines at amplitudes from far below to far above the height rule give the same peaks."""
    rng = np.random.default_rng(1)
    amplitudes = np.concatenate([[0.0], np.geomspace(0.02, 2.0, 16)])
    n_with_peaks = 0
    for amp in amplitudes:
        for n_lines in (1, 2, 3):
            lines = [(rng.uniform(0.12, 0.58), amp) for _ in range(n_lines)]
            trace = _cw_trace(rng, lines)
            screened = _peaks(trace, algorithm, True)
            full = _peaks(trace, algorithm, False)
            assert np.array_equal(screened, full), (algorithm, amp, lines)
            n_with_peaks += len(full) > 0
    assert 0 < n_with_peaks < len(amplitudes) * 3


def test_prescreen_matches_full_search_at_trace_threshold():
    """A single line bisected onto the height rule agrees on both sides of it, ulp-close."""
    rng = np.random.default_rng(2)
    noise = rng.normal(0.0, 1.0, N_NATIVE)
    t = np.arange(N_NATIVE) / NATIVE_RATE
    line = np.sin(2 * np.pi * 0.2371 * t + 0.4)

    def n_peaks(amp):
        """Number of peaks the full search finds for a line of this amplitude."""
        return len(_peaks(noise + amp * line, 'sliding', False))

    lo, hi = 0.0, 2.0
    assert n_peaks(lo) == 0 and n_peaks(hi) > 0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if n_peaks(mid) > 0:
            hi = mid
        else:
            lo = mid
    outcomes = set()
    for rel in (-1e-2, -1e-4, -1e-6, -1e-9, 0.0, 1e-9, 1e-6, 1e-4, 1e-2):
        trace = noise + hi * (1.0 + rel) * line
        screened = _peaks(trace, 'sliding', True)
        full = _peaks(trace, 'sliding', False)
        assert np.array_equal(screened, full), rel
        outcomes.add(len(full) > 0)
    assert outcomes == {False, True}


def test_prescreen_matches_full_search_at_spectrum_threshold():
    """A constructed spike scanned ulp by ulp around 4 x the window RMS agrees on every step."""
    n_bins = N_NATIVE // 2 + 1
    freqs = fft.freqs(N_NATIVE, NATIVE_RATE)
    delta_f = freqs[1] - freqs[0]
    band_mask = (freqs >= FREQ_BAND[0]) & (freqs <= FREQ_BAND[1])
    spike_bin = np.flatnonzero(band_mask)[150]
    window = int(50 * units.MHz / delta_f)
    # a spike A over a unit floor crosses 4 x RMS of a window of `window` bins at A^2 = 16 (window - 1) / (window - 16)
    a_cross = np.sqrt(16.0 * (window - 1) / (window - 16))
    outcomes = set()
    for k in list(range(-64, 65)) + [-4096, -512, 512, 4096]:
        spec = np.ones(n_bins)
        spec[spike_bin] = a_cross * (1.0 + k * np.finfo(float).eps)
        screened = np.asarray(cws.find_cw_peaks(spec, band_mask, delta_f, 'sliding', PROMINENCE, prescreen=True))
        full = np.asarray(cws.find_cw_peaks(spec, band_mask, delta_f, 'sliding', PROMINENCE, prescreen=False))
        assert np.array_equal(screened, full), k
        outcomes.add(len(full) > 0)
        if len(full) > 0:
            assert list(full) == [spike_bin]
    assert outcomes == {False, True}


def test_prescreen_skips_windows_without_a_candidate(monkeypatch):
    """Noise-only traces skip every window; a loud line visits only the windows around it."""
    rng = np.random.default_rng(3)
    spec, band_mask, delta_f = _spectrum(rng.normal(0.0, 1.0, N_NATIVE))
    n_windows = int(np.count_nonzero(band_mask)) - int(50 * units.MHz / delta_f) + 1
    calls = _count_find_peaks(monkeypatch)
    cws.find_cw_peaks(spec, band_mask, delta_f, 'sliding', PROMINENCE, prescreen=False)
    assert calls[0] == n_windows
    calls[0] = 0
    cws.find_cw_peaks(spec, band_mask, delta_f, 'sliding', PROMINENCE, prescreen=True)
    assert calls[0] == 0
    calls[0] = 0
    spec, band_mask, delta_f = _spectrum(_cw_trace(rng, [(0.3, 2.0)]))
    cws.find_cw_peaks(spec, band_mask, delta_f, 'sliding', PROMINENCE, prescreen=True)
    assert 0 < calls[0] <= 3 * int(50 * units.MHz / delta_f)


def test_sinewave_subtraction_identical_with_and_without_prescreen(monkeypatch):
    """The subtracted trace and the removed frequencies are bit-identical with the pre-screen."""
    rng = np.random.default_rng(4)
    cases = [[], [(0.15, 0.05)], [(0.2, 0.3)], [(0.151, 1.0), (0.4033, 0.5)],
             [(0.25, 3.0), (0.33, 0.12), (0.52, 0.8)]]
    traces = [_cw_trace(rng, lines) for lines in cases]
    screened = [cws.sinewave_subtraction(tr, 'sliding', PROMINENCE, NATIVE_RATE, FREQ_BAND) for tr in traces]
    monkeypatch.setattr(cws, 'find_cw_peaks', functools.partial(cws.find_cw_peaks, prescreen=False))
    full = [cws.sinewave_subtraction(tr, 'sliding', PROMINENCE, NATIVE_RATE, FREQ_BAND) for tr in traces]
    n_removed = 0
    for (wf_s, freqs_s), (wf_f, freqs_f) in zip(screened, full):
        assert np.array_equal(wf_s, wf_f)
        assert freqs_s == freqs_f
        n_removed += len(freqs_f)
    assert n_removed >= 3


def _native_event(det, tables, src_enu, seed, channel_order):
    """Build a native-rate event with pulses on the VPol channels, noise elsewhere and one CW line on every channel."""
    rng = np.random.default_rng(seed)
    ant_locs = antenna_locations(det, STATION)
    tts = {ch: tables.travel_time(ch, src_enu, ant_locs[ch]) for ch in VPOL_CHANNELS}
    t_ref = min(tts.values())
    t = np.arange(N_NATIVE) / NATIVE_RATE
    evt = Event(0, seed)
    stn = Station(STATION)
    for ch in channel_order:
        if ch in tts:
            trace = filtered_trace(150.0 * units.ns + tts[ch] - t_ref, 1.0, 0.05, rng)
        else:
            trace = filtered_trace(0.0, 0.0, 0.05, rng)
        trace = trace + 0.05 * np.sin(2 * np.pi * 0.2371 * t + rng.uniform(0, 2 * np.pi))
        c = Channel(ch)
        c.set_trace(trace, NATIVE_RATE)
        stn.add_channel(c)
    evt.set_station(stn)
    return evt, stn


def _snapshot(stn):
    """Copy every channel's trace, start time, sampling rate and parameter dict."""
    out = {}
    for ch in stn.iter_channels():
        out[ch.get_id()] = (ch.get_trace().copy(), ch.get_trace_start_time(), ch.get_sampling_rate(),
                            dict(ch.get_parameters()))
    return out


@pytest.mark.slow
def test_channel_restriction_leaves_other_channels_and_result_unchanged(det, tables, pa, reco, base_config):
    """Restricting the chain to the VPol channels changes nothing the reconstruction sees."""
    channel_order = [0, 4, 12, 1, 2, 8, 13, 3, 5, 14, 6, 11, 7, 15, 9, 16, 10, 21, 17, 22, 18, 23, 19, 20]
    src = cylindrical_to_enu(80.0, 210.0, -30.0, pa)
    cfg = dict(apply_glitch_detection=True, apply_cable_delay=True, apply_hw_phase_removal=True,
               apply_cw_removal=True, apply_bandpass=True, bandpass_band=(0.1 * units.GHz, 0.7 * units.GHz))
    evt_all, stn_all = _native_event(det, tables, src, 7, channel_order)
    evt_sel, stn_sel = _native_event(det, tables, src, 7, channel_order)
    before = _snapshot(stn_sel)

    pre_all = channelPreprocessor()
    pre_all.begin(config=dict(cfg))
    pre_all.run(evt_all, stn_all, det)
    pre_sel = channelPreprocessor()
    pre_sel.begin(config=dict(cfg, channels=list(VPOL_CHANNELS)))
    pre_sel.run(evt_sel, stn_sel, det)

    assert stn_sel.get_channel_ids() == channel_order
    assert stn_all.get_channel_ids() == channel_order
    after_sel = _snapshot(stn_sel)
    after_all = _snapshot(stn_all)
    for ch in channel_order:
        trace_sel, t0_sel, rate_sel, params_sel = after_sel[ch]
        trace_all, t0_all, rate_all, params_all = after_all[ch]
        if ch in VPOL_CHANNELS:
            assert np.array_equal(trace_sel, trace_all) and t0_sel == t0_all and rate_sel == rate_all
            assert channelParametersRNOG.glitch in params_sel and channelParameters.block_offsets in params_sel
        else:
            trace_0, t0_0, rate_0, params_0 = before[ch]
            assert np.array_equal(trace_sel, trace_0) and t0_sel == t0_0 and rate_sel == rate_0
            assert params_sel == params_0 == {}
            assert not np.array_equal(trace_all, trace_0) and t0_all != t0_0

    resampler = channelResampler()
    resampler.begin()
    resampler.run(evt_all, stn_all, det, sampling_rate=SAMPLING_RATE)
    resampler.run(evt_sel, stn_sel, det, sampling_rate=SAMPLING_RATE)
    res_all = reco.run(evt_all, stn_all, det, base_config)
    res_sel = reco.run(evt_sel, stn_sel, det, base_config)
    for key in ('rho', 'phi', 'z', 'max_corr'):
        assert res_sel[key] == res_all[key], key


def test_default_config_processes_every_channel():
    """The shipped default leaves the channel list unset."""
    assert channelPreprocessor._DEFAULT_CONFIG['channels'] is None
    pre = channelPreprocessor()
    pre.begin(config={'apply_cw_removal': True})
    assert pre._config['channels'] is None


def test_driver_unions_reconstruction_channels(monkeypatch):
    """The driver fills the preprocessor list with every channel the reconstruction reads."""
    monkeypatch.syspath_prepend(EXAMPLE_DIR)
    from reco_validation import preprocessing_channels
    cfg = {'channels': [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23, 4, 8, 11, 21],
           'polarization_groups': {'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS}}
    assert preprocessing_channels(cfg) == sorted(VPOL_CHANNELS + HPOL_CHANNELS)
    assert preprocessing_channels({'channels': [3, 0, 1]}) == [0, 1, 3]
    fallback = preprocessing_channels({'channels': [0, 1, 2, 3], 'plane_wave_fallback': True})
    assert fallback == sorted(set([0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]))
