"""Time-window SNR estimator and pair weights on a heterogeneous-SNR synthetic family.

The record SNR is the peak-to-peak amplitude within a 3-sample window over twice the
split-trace noise RMS, measured on the 10 GHz traces the reconstruction sees, so it reads
a fixed fraction of the pulse amplitude (about a third for the reference bandpass) and
about 1 on pure noise. With `snr_window_ns` the window is a time span (1.5 ns is 15
samples at 10 GHz and 5 at the native 3.2 GHz), the estimator reads the injected pulse
amplitude and about 3 on pure noise, and the values are written under versioned columns
(`ch{N}_snr_w15` and the `_w15` summaries) beside the unchanged record columns. The
family here has the phased array at SNR 40, the helpers between 5 and 15, channel 5 at 5
and channels 6 and 7 at noise. `pair_weight_mode: information` replaces the record
geometric-mean weights by inverse timing-variance weights built from the windowed SNR
(`1 / (floor^2 + (k / SNR_i)^2 + (k / SNR_j)^2)`, maximum 1); with the defaults (k 12.65 ns,
floor 2 ns) a pair of two noise channels keeps about an eighth of a phased-array pair's
weight because the windowed estimator reads about 3 on noise, and k = 25 ns brings it
below a twentieth.
"""

import itertools

import numpy as np
import pytest

from conftest import STATION, reference_config
from synthetic import VPOL_CHANNELS, cylindrical_to_enu, make_event
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

SRC = (80.0, 120.0, -40.0)
PA = [0, 1, 2, 3]
HELPERS = [9, 10, 22, 23]
NOISE = [6, 7]
HET_SNR = {0: 40.0, 1: 40.0, 2: 40.0, 3: 40.0, 9: 15.0, 10: 10.0, 22: 8.0, 23: 5.0,
           5: 5.0, 6: 0.0, 7: 0.0}
WINDOW_NS = 1.5
TIMING_KEYS = ('coarse_time', 'refine_time', 'opt_time', 'post_time', 'raw_refine_time',
               'peak_time')


@pytest.fixture(scope='module')
def het_event(det, tables, pa):
    """Heterogeneous-SNR event at the mid-depth source."""
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*SRC, pa), VPOL_CHANNELS, tables,
                             snr=HET_SNR, seed=3)
    return evt, stn


@pytest.fixture(scope='module')
def window_config(table_dir):
    """Reference configuration with the 1.5 ns SNR window and validation columns."""
    return reference_config(table_dir, snr_window_ns=WINDOW_NS, validation=True)


@pytest.fixture(scope='module')
def window_result(reco, det, het_event, window_config):
    """Reconstruction of the heterogeneous event with the SNR window."""
    return reco.run(het_event[0], het_event[1], det, window_config)


def _windowed(res):
    """Return the windowed SNR (``ch{ch}_snr_w15``) of every VPol channel, keyed by channel."""
    return {ch: res[f'ch{ch}_snr_w15'] for ch in VPOL_CHANNELS}


@pytest.mark.slow
def test_windowed_snr_orders_channels(window_result):
    """The windowed SNR ranks the phased array above the helpers above the noise channels and reads the injected amplitude."""
    w = _windowed(window_result)
    assert min(w[ch] for ch in PA) > max(w[ch] for ch in HELPERS + [5]), w
    assert w[9] > w[10] > w[22] > max(w[ch] for ch in NOISE), w
    assert min(w[23], w[5]) > max(w[ch] for ch in NOISE), w
    assert 0.8 * 40 < np.mean([w[ch] for ch in PA]) < 1.2 * 40, w
    for ch in NOISE:
        assert 2.0 < w[ch] < 5.0, w
    for ch in PA:
        assert window_result[f'ch{ch}_snr'] < 0.5 * w[ch], (ch, window_result[f'ch{ch}_snr'], w[ch])


@pytest.mark.slow
def test_windowed_summaries_use_their_own_threshold(window_result, window_config):
    """The `_w15` summaries repeat the record summaries on the windowed SNR with `helper_snr_threshold_windowed`."""
    w = _windowed(window_result)
    threshold = InterferometricReco3D.HELPER_SNR_THRESHOLD_WINDOWED
    assert window_result['n_helpers_above_w15'] == sum(w[ch] > threshold for ch in HELPERS)
    assert window_result['has_helper_signal_w15'] == (window_result['n_helpers_above_w15'] > 0)
    assert window_result['pa_max_snr_w15'] == max(w[ch] for ch in PA)
    assert window_result['helper_c_min_snr_w15'] == min(w[22], w[23])
    assert window_result['n_helpers_above'] == sum(
        window_result[f'ch{ch}_snr'] > window_config.get('helper_snr_threshold', 5.0) for ch in HELPERS)


@pytest.mark.slow
def test_record_columns_unchanged_by_snr_window(reco, det, het_event, table_dir, window_result):
    """Setting `snr_window_ns` adds the `_w15` columns and leaves every record output bit-identical."""
    ref = reco.run(het_event[0], het_event[1], det, reference_config(table_dir, validation=True))
    for key, val in ref.items():
        if key in TIMING_KEYS or key == 'coarse_peaks':
            continue
        assert key in window_result, key
        assert window_result[key] == val, (key, val, window_result[key])
    added = set(window_result) - set(ref)
    assert added == {f'ch{ch}_snr_w15' for ch in VPOL_CHANNELS} | {
        'pa_avg_snr_w15', 'pa_max_snr_w15', 'helper_b_max_snr_w15', 'helper_b_min_snr_w15',
        'helper_c_max_snr_w15', 'helper_c_min_snr_w15', 'n_helpers_above_w15',
        'n_channels_above_w15', 'has_helper_signal_w15'}, added


@pytest.mark.slow
def test_window_matches_native_rate(det, tables, pa):
    """15 samples at 10 GHz and 5 at 3.2 GHz measure the same 1.5 ns window on the same pulse."""
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*SRC, pa), VPOL_CHANNELS, tables,
                             snr=HET_SNR, seed=3)
    reco = InterferometricReco3D()
    up = [stn.get_channel(ch).get_trace() for ch in VPOL_CHANNELS]
    snr_up = reco._channel_snrs(up, reco._snr_window_samples(WINDOW_NS, 0.1))
    for ch in VPOL_CHANNELS:
        stn.get_channel(ch).resample(3.2)
    native = [stn.get_channel(ch).get_trace() for ch in VPOL_CHANNELS]
    snr_native = reco._channel_snrs(native, reco._snr_window_samples(WINDOW_NS, 1.0 / 3.2))
    assert reco._snr_window_samples(WINDOW_NS, 0.1) == 15
    assert reco._snr_window_samples(WINDOW_NS, 1.0 / 3.2) == 5
    for a, b, ch in zip(snr_up, snr_native, VPOL_CHANNELS):
        assert abs(a - b) < 0.15 * max(a, b), (ch, a, b)


def _pair_weight_table(snrs, k_ns, floor_ns=InterferometricReco3D.PAIR_WEIGHT_FLOOR_NS):
    """Compute the information pair weights of the VPol channels.

    Args:
        snrs: Dict mapping channel id to SNR.
        k_ns: Timing error at unit SNR (ns).
        floor_ns: Timing error floor (ns).

    Returns:
        Dict mapping each channel pair, in ``itertools.combinations`` order, to its weight.
    """
    weights = InterferometricReco3D._information_pair_weights(
        [snrs[ch] for ch in VPOL_CHANNELS], k_ns, floor_ns)
    return dict(zip(itertools.combinations(VPOL_CHANNELS, 2), weights))


@pytest.mark.slow
def test_information_weights_suppress_noise_pairs(window_result):
    """Information weights put the noise pair far below a phased-array pair; the record rule is reported beside them."""
    w = _windowed(window_result)
    info = _pair_weight_table(w, InterferometricReco3D.PAIR_WEIGHT_K_NS)
    assert info[(0, 1)] == max(info.values())
    assert info[(6, 7)] < info[(0, 1)] / 6.0, (info[(6, 7)], info[(0, 1)])
    assert info[(0, 6)] < info[(0, 1)] / 4.0, (info[(0, 6)], info[(0, 1)])
    assert info[(0, 9)] > info[(0, 23)] > info[(0, 6)]
    for pair, val in info.items():
        assert 0.0 < val <= 1.0, (pair, val)
    steep = _pair_weight_table(w, 25.0)
    assert steep[(6, 7)] < steep[(0, 1)] / 20.0, (steep[(6, 7)], steep[(0, 1)])
    record = {ch: window_result[f'ch{ch}_snr'] for ch in VPOL_CHANNELS}
    rec = dict(zip(itertools.combinations(VPOL_CHANNELS, 2),
                   [np.sqrt(record[a] * record[b]) for a, b in itertools.combinations(VPOL_CHANNELS, 2)]))
    rec = {p: v / max(rec.values()) for p, v in rec.items()}
    noise_pairs = [p for p in info if p[0] in NOISE or p[1] in NOISE]
    frac_info = sum(info[p] for p in noise_pairs) / sum(info.values())
    frac_rec = sum(rec[p] for p in noise_pairs) / sum(rec.values())
    print(f'noise-pair weight fraction: record {frac_rec:.3f}, information {frac_info:.3f}')
    assert frac_info < frac_rec


def test_information_weights_zero_dead_channels():
    """A channel with SNR 0 zeroes its pairs and the remaining pairs stay normalised."""
    weights = InterferometricReco3D._information_pair_weights([40.0, 0.0, 10.0], 12.65, 2.0)
    assert weights[0] == 0.0 and weights[2] == 0.0 and weights[1] == 1.0
    flat = InterferometricReco3D._information_pair_weights([1e6, 1e6, 1e6], 12.65, 2.0)
    assert np.allclose(flat, 1.0)


@pytest.mark.slow
def test_information_weights_recover_source(reco, det, het_event, table_dir):
    """The information weights reconstruct the heterogeneous event as well as the record weights."""
    ref = reco.run(het_event[0], het_event[1], det, reference_config(table_dir))
    cfg = reference_config(table_dir, snr_window_ns=WINDOW_NS, pair_weight_mode='information')
    res = reco.run(het_event[0], het_event[1], det, cfg)
    for r in (ref, res):
        assert abs(r['rho'] - SRC[0]) < 2.0 and abs(r['z'] - SRC[2]) < 2.0, r
        assert abs((r['phi'] - SRC[1] + 180.0) % 360.0 - 180.0) < 0.5, r
    assert res['max_corr'] != ref['max_corr']
    assert InterferometricReco3D.objective_version(cfg) == \
        'pair_weight_mode=information(k_ns=12.65,floor_ns=2.0,snr_window_ns=1.5)'
    assert InterferometricReco3D.objective_version(reference_config(table_dir)) == 'record'


def test_snr_suffix_and_invalid_keys():
    """The column suffix follows the window and bad SNR or weight keys are rejected."""
    assert InterferometricReco3D._snr_suffix(1.5) == 'w15'
    assert InterferometricReco3D._snr_suffix(2.0) == 'w2'
    reco = InterferometricReco3D()
    for bad in ({'snr_window_ns': 0}, {'snr_window_ns': -1.5}, {'snr_window_ns': '1.5'},
                {'helper_snr_threshold_windowed': 'five'},
                {'pair_weight_mode': 'information'},
                {'pair_weight_mode': 'lofar', 'snr_window_ns': 1.5},
                {'pair_weight_mode': 'information', 'snr_window_ns': 1.5, 'pair_weight_k_ns': 0},
                {'pair_weight_mode': 'information', 'snr_window_ns': 1.5, 'pair_weight_floor_ns': -1},
                {'pair_weight_mode': 'information', 'snr_window_ns': 1.5},
                {'pair_weight_mode': 'information', 'snr_window_ns': 1.5, 'snr_pair_weighting': False}):
        with pytest.raises(ValueError):
            reco._validate_config(bad)
    reco._validate_config({'snr_window_ns': None})
    reco._validate_config({'snr_window_ns': 1.5, 'helper_snr_threshold_windowed': 12.0})
    reco._validate_config({'snr_window_ns': 1.5, 'pair_weight_mode': 'information',
                           'snr_pair_weighting': True, 'pair_weight_k_ns': 25.0,
                           'pair_weight_floor_ns': 0.0})
