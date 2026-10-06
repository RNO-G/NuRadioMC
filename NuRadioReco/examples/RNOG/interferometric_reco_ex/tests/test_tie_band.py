"""Candidate tie band: fall back to the raw chain's own answer when the candidate gain is small.

In candidate mode the raw chain's best optimizer output before the polish (R0) is the
default search's answer for the same parameters, and the ranked best candidate (R1) can
only raise the raw correlation, so `candidate_gain = R1 - R0` is non-negative up to float
round-off. With `candidate_tie_band` set, events whose gain is below the band keep R0 as the
primary result. A huge band must therefore reproduce the default search exactly (position
and max_corr, compared with ==) on signal and on pure-noise events, and a zero band must
reproduce the plain candidate search on these events. `candidate_tie_band_max_raw_corr`
restricts the fallback to events whose R0 is below it: an infinite ceiling is the plain
band, a zero ceiling is the unbanded candidate search.
"""

import functools
import os

import numpy as np
import pytest
import yaml

from conftest import STATION, reference_config
from synthetic import (HPOL_CHANNELS, RECORD_PREPROCESSOR, VPOL_CHANNELS, TravelTimeTables, cylindrical_to_enu,
                       make_event, make_noise_event)
from test_reco_known_answer import EXACT_SOURCES, SUMMARY_SOURCES
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

SNR = 20.0
HUGE_BAND = 1e9
NOISE_SEEDS = [1, 2, 3]
EXACT = [p.values[0] for p in EXACT_SOURCES]
EXACT_IDS = [p.id for p in EXACT_SOURCES]
GAIN_SOURCES = list(dict.fromkeys(EXACT + SUMMARY_SOURCES))
POSITION_KEYS = ('rho', 'phi', 'z', 'max_corr')
GAIN_ROUNDOFF = 1e-7
RECOMMENDED_BAND = 0.0054
CASES = [('signal', s) for s in EXACT[:2]] + [('noise', NOISE_SEEDS[0])]
CASE_IDS = EXACT_IDS[:2] + [f'noise{NOISE_SEEDS[0]}']
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
RECOMMENDED_CR_KEYS = {'candidate_search': ['envelope:traces', 'envelope:correlation', 'raw'],
                       'candidate_tie_band': RECOMMENDED_BAND, 'candidate_tie_band_max_raw_corr': 0.035,
                       'candidate_fill_saved_peaks': True}
# Keys of the station-23 VPol configuration the tie band and its ceiling were measured with; its
# preprocessor block is RECORD_PREPROCESSOR.
BENCHMARK_CR_KEYS = dict(
    hierarchical=True, multi_ray_types=False, multiray_combo_mode='grouped',
    coarse_limits=[1, 250, 0, 360, -100, 0], coarse_n_rho=30, coarse_n_z=100, coarse_step_sizes=[0, 3, 0],
    coarse_n_peaks=7, coarse_peak_separation=[50, 15, 50], refine_window=[15, 3, 15],
    refine_step_sizes=[1, 0.3, 1], limits=[1, 250, 0, 360, -100, 0], n_peaks_save=3,
    multiray_table_name_pattern='st{station_id}_ch{ch}_rz_table_{ray_type}.npz',
    apply_upsampling=True, apply_dedispersion=False, hilbert_envelope_mode=None, apply_hann_window=True,
    snr_pair_weighting=True, correlation_normalization='energy', interp_method='linear',
    candidate_search=['envelope:traces', 'envelope:correlation', 'raw'], candidate_tie_band=RECOMMENDED_BAND,
    candidate_tie_band_max_raw_corr=0.035)
DELAY_CORRECTION_KEYS = {'apply_delay_corrections', 'delay_corrections_file'}


@pytest.fixture(scope='module')
def cand_config(table_dir):
    """Reference configuration with the default candidate search and no tie band."""
    return reference_config(table_dir, candidate_search=['envelope', 'raw'])


@pytest.fixture(scope='module')
def huge_band_config(table_dir):
    """Candidate search with a band no gain can reach, so every event falls back."""
    return reference_config(table_dir, candidate_search=['envelope', 'raw'], candidate_tie_band=HUGE_BAND)


@pytest.fixture(scope='module')
def zero_band_config(table_dir):
    """Candidate search with a zero band, so no event falls back."""
    return reference_config(table_dir, candidate_search=['envelope', 'raw'], candidate_tie_band=0.0)


def _signal(det, src, tables, pa):
    """Return a fresh (event, station) pair with a pulse from a (rho, phi, z) source at SNR 20."""
    return make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables, snr=SNR, seed=5)[:2]


def _noise(seed):
    """Return a fresh pure-noise (event, station) pair."""
    return make_noise_event(STATION, VPOL_CHANNELS, seed=seed)


def _build(det, tables, pa, case):
    """Return a builder for a ('signal', source) or ('noise', seed) test case."""
    kind, arg = case
    if kind == 'signal':
        return functools.partial(_signal, det, arg, tables, pa)
    return functools.partial(_noise, arg)


def _compared_keys(res):
    """Return the position, peak and candidate keys of a result, timing keys excluded."""
    keys = [k for k in res if not k.endswith('_time') and (
        k in POSITION_KEYS or k.startswith(('peak_', 'candidate_origin_'))
        or k in ('n_candidates', 'n_saved_peaks', 'candidate_raw_chain_corr', 'candidate_gain'))]
    assert 'peak_0_rho' in keys and 'candidate_gain' in keys
    return keys


def _assert_same_answer(res, ref):
    """Check that every compared key of ``ref`` has the identical value in ``res``."""
    for key in _compared_keys(ref):
        assert res[key] == ref[key], (key, res[key], ref[key])


def _run_fresh(reco, det, config, build):
    """Reconstruct a newly built event so that no run sees another run's station state."""
    evt, stn = build()
    return reco.run(evt, stn, det, config)


def _assert_record_identity(rec, res):
    """Check that a fallback result equals the default search's answer exactly."""
    for key in POSITION_KEYS:
        assert res[key] == rec[key], (key, res[key], rec[key])
    for f in ('rho', 'phi', 'z', 'corr'):
        assert res[f'peak_0_{f}'] == rec[f'peak_0_{f}'], f
    assert res['candidate_fallback'] == 1
    assert res['candidate_origin_0'] == 0
    assert res['candidate_raw_chain_corr'] == rec['max_corr']
    assert res['candidate_gain'] >= -GAIN_ROUNDOFF


def test_tie_band_validation():
    """The band must be a non-negative number and needs the raw chain in the candidate search."""
    reco = InterferometricReco3D()
    for bad in ({'candidate_tie_band': 0.01},
                {'candidate_search': [], 'candidate_tie_band': 0.01},
                {'candidate_search': ['envelope'], 'candidate_tie_band': 0.01},
                {'candidate_search': ['envelope:traces', 'envelope:correlation'], 'candidate_tie_band': 0.01},
                {'candidate_search': ['envelope', 'raw'], 'candidate_tie_band': -0.01},
                {'candidate_search': ['envelope', 'raw'], 'candidate_tie_band': float('nan')},
                {'candidate_search': ['envelope', 'raw'], 'candidate_tie_band': '0.01'},
                {'candidate_search': ['envelope', 'raw'], 'candidate_tie_band': True},
                {'candidate_search': ['envelope', 'raw'], 'candidate_tie_band': [0.01]}):
        with pytest.raises(ValueError):
            reco._validate_config(bad)
    for band in (0, 0.0, 0.02, HUGE_BAND, float('inf'), np.float64(0.02)):
        reco._validate_config({'candidate_search': ['envelope', 'raw'], 'candidate_tie_band': band})
        assert reco._candidate_tie_band({'candidate_search': ['raw'], 'candidate_tie_band': band}) == float(band)
    reco._validate_config({'candidate_search': ['envelope', 'raw'], 'candidate_tie_band': None})
    assert reco._candidate_tie_band({'candidate_search': ['envelope', 'raw']}) is None
    assert reco._candidate_tie_band({}) is None
    assert 'candidate_tie_band' in InterferometricReco3D._KNOWN_CONFIG_KEYS


@pytest.mark.slow
def test_tie_band_rejected_at_run_without_raw_chain(reco, det, table_dir, tables, pa):
    """A configuration passed straight to run is checked as well."""
    cfg = reference_config(table_dir, candidate_search=['envelope'], candidate_tie_band=0.01)
    evt, stn = _signal(det, (80.0, 120.0, -40.0), tables, pa)
    with pytest.raises(ValueError):
        reco.run(evt, stn, det, cfg)


@pytest.mark.slow
@pytest.mark.parametrize('src', EXACT, ids=EXACT_IDS)
def test_huge_band_reproduces_record_chain(reco, det, base_config, huge_band_config, tables, pa, src):
    """With a huge band the candidate mode returns the default search's rho, phi, z and max_corr exactly."""
    build = functools.partial(_signal, det, src, tables, pa)
    rec = _run_fresh(reco, det, base_config, build)
    res = _run_fresh(reco, det, huge_band_config, build)
    _assert_record_identity(rec, res)


@pytest.mark.slow
@pytest.mark.parametrize('seed', NOISE_SEEDS)
def test_huge_band_reproduces_record_chain_on_noise(reco, det, base_config, huge_band_config, seed):
    """The identity also holds on pure-noise events, where the fallback is meant to act."""
    build = functools.partial(_noise, seed)
    rec = _run_fresh(reco, det, base_config, build)
    res = _run_fresh(reco, det, huge_band_config, build)
    _assert_record_identity(rec, res)


@pytest.mark.slow
@pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
def test_zero_band_matches_candidate_search(reco, det, cand_config, zero_band_config, tables, pa, case):
    """A zero band does not fall back on these events, so every position and peak key equals the plain candidate search's."""
    build = _build(det, tables, pa, case)
    plain = _run_fresh(reco, det, cand_config, build)
    zero = _run_fresh(reco, det, zero_band_config, build)
    assert zero['candidate_fallback'] == 0
    assert 'candidate_fallback' not in plain
    _assert_same_answer(zero, plain)


@pytest.mark.slow
def test_candidate_gain_non_negative(reco, det, cand_config, tables, pa):
    """The ranked best raw correlation is not below the raw chain's own answer beyond round-off, on signal and on noise."""
    builds = [functools.partial(_signal, det, s, tables, pa) for s in GAIN_SOURCES]
    builds += [functools.partial(_noise, seed) for seed in NOISE_SEEDS]
    gains = []
    for build in builds:
        res = _run_fresh(reco, det, cand_config, build)
        assert np.isfinite(res['candidate_raw_chain_corr']), res
        assert res['candidate_gain'] == res['max_corr'] - res['candidate_raw_chain_corr']
        gains.append(res['candidate_gain'])
    assert min(gains) >= -GAIN_ROUNDOFF, gains


@pytest.mark.slow
def test_polarization_groups_carry_suffixed_diagnostics(det, table_dir, pa):
    """With polarization groups each group reports its own diagnostics and, under a huge band, its default-search answer."""
    channels = VPOL_CHANNELS + HPOL_CHANNELS
    groups = {'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS}
    rec_cfg = reference_config(table_dir, channels=channels, polarization_groups=groups)
    cfg = reference_config(table_dir, channels=channels, polarization_groups=groups,
                           candidate_search=['envelope', 'raw'], candidate_tie_band=HUGE_BAND)
    all_tables = TravelTimeTables(table_dir, STATION, channels)
    reco = InterferometricReco3D()
    reco.begin(STATION, cfg, det)
    src = (80.0, 120.0, -40.0)
    build = functools.partial(make_event, det, STATION, cylindrical_to_enu(*src, pa), channels, all_tables,
                              snr=SNR, seed=3)
    evt, stn, _ = build()
    rec = reco.run(evt, stn, det, rec_cfg)
    evt, stn, _ = build()
    res = reco.run(evt, stn, det, cfg)
    for group in ('vpol', 'hpol'):
        assert res[f'candidate_fallback_{group}'] == 1
        assert res[f'candidate_gain_{group}'] >= -GAIN_ROUNDOFF
        assert res[f'candidate_raw_chain_corr_{group}'] == res[f'max_corr_{group}']
        for key in POSITION_KEYS:
            assert res[f'{key}_{group}'] == rec[f'{key}_{group}'], (group, key)


def test_ceiling_validation():
    """The ceiling must be a non-negative number and needs candidate_tie_band."""
    reco = InterferometricReco3D()
    cand = {'candidate_search': ['envelope', 'raw'], 'candidate_tie_band': RECOMMENDED_BAND}
    for bad in ({'candidate_tie_band_max_raw_corr': 0.035},
                {'candidate_search': ['envelope', 'raw'], 'candidate_tie_band_max_raw_corr': 0.035},
                {**cand, 'candidate_tie_band': None, 'candidate_tie_band_max_raw_corr': 0.035},
                {**cand, 'candidate_tie_band_max_raw_corr': -0.01},
                {**cand, 'candidate_tie_band_max_raw_corr': float('nan')},
                {**cand, 'candidate_tie_band_max_raw_corr': '0.035'},
                {**cand, 'candidate_tie_band_max_raw_corr': True},
                {**cand, 'candidate_tie_band_max_raw_corr': [0.035]}):
        with pytest.raises(ValueError):
            reco._validate_config(bad)
    for ceiling in (0, 0.0, 0.035, float('inf'), np.float64(0.035)):
        cfg = {**cand, 'candidate_tie_band_max_raw_corr': ceiling}
        reco._validate_config(cfg)
        assert reco._candidate_tie_band_max_raw_corr(cfg) == float(ceiling)
    reco._validate_config({**cand, 'candidate_tie_band_max_raw_corr': None})
    assert reco._candidate_tie_band_max_raw_corr(cand) is None
    assert reco._candidate_tie_band_max_raw_corr({}) is None
    assert 'candidate_tie_band_max_raw_corr' in InterferometricReco3D._KNOWN_CONFIG_KEYS


@pytest.mark.slow
def test_ceiling_rejected_at_run_without_band(reco, det, table_dir, tables, pa):
    """A ceiling without a band passed straight to run is checked as well."""
    cfg = reference_config(table_dir, candidate_search=['envelope', 'raw'], candidate_tie_band_max_raw_corr=0.035)
    evt, stn = _signal(det, (80.0, 120.0, -40.0), tables, pa)
    with pytest.raises(ValueError):
        reco.run(evt, stn, det, cfg)


@pytest.mark.slow
@pytest.mark.parametrize('band', [RECOMMENDED_BAND, HUGE_BAND])
@pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
def test_infinite_ceiling_reproduces_plain_band(reco, det, table_dir, tables, pa, case, band):
    """An infinite ceiling gives the plain band's result, fallback flag included."""
    build = _build(det, tables, pa, case)
    plain_cfg = reference_config(table_dir, candidate_search=['envelope', 'raw'], candidate_tie_band=band)
    inf_cfg = reference_config(table_dir, candidate_search=['envelope', 'raw'], candidate_tie_band=band,
                               candidate_tie_band_max_raw_corr=float('inf'))
    plain = _run_fresh(reco, det, plain_cfg, build)
    res = _run_fresh(reco, det, inf_cfg, build)
    assert res['candidate_fallback'] == plain['candidate_fallback']
    _assert_same_answer(res, plain)


@pytest.mark.slow
@pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
def test_zero_ceiling_matches_candidate_search(reco, det, table_dir, cand_config, tables, pa, case):
    """A zero ceiling never falls back, even under a huge band, so it equals the unbanded candidate search."""
    build = _build(det, tables, pa, case)
    cfg = reference_config(table_dir, candidate_search=['envelope', 'raw'], candidate_tie_band=HUGE_BAND,
                           candidate_tie_band_max_raw_corr=0.0)
    plain = _run_fresh(reco, det, cand_config, build)
    res = _run_fresh(reco, det, cfg, build)
    assert res['candidate_fallback'] == 0
    _assert_same_answer(res, plain)


@pytest.mark.slow
def test_ceiling_falls_back_on_noise_only(reco, det, base_config, table_dir, cand_config, tables, pa):
    """A ceiling between the noise and the signal raw-chain correlations falls back on noise and keeps the signal's ranked best."""
    signal = functools.partial(_signal, det, EXACT[0], tables, pa)
    noise = functools.partial(_noise, NOISE_SEEDS[0])
    plain_signal = _run_fresh(reco, det, cand_config, signal)
    plain_noise = _run_fresh(reco, det, cand_config, noise)
    r0_signal = plain_signal['candidate_raw_chain_corr']
    r0_noise = plain_noise['candidate_raw_chain_corr']
    assert r0_noise < r0_signal, (r0_noise, r0_signal)
    cfg = reference_config(table_dir, candidate_search=['envelope', 'raw'], candidate_tie_band=HUGE_BAND,
                           candidate_tie_band_max_raw_corr=0.5 * (r0_noise + r0_signal))
    res_signal = _run_fresh(reco, det, cfg, signal)
    assert res_signal['candidate_fallback'] == 0
    _assert_same_answer(res_signal, plain_signal)
    rec_noise = _run_fresh(reco, det, base_config, noise)
    _assert_record_identity(rec_noise, _run_fresh(reco, det, cfg, noise))


def _shipped_config(name):
    """Load one of the shipped configs in configs/."""
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def test_recommended_cr_config_matches_the_benchmark_configuration():
    """configs/reco3d_cr_candidate.yaml carries the search, driver, preprocessing and candidate keys the thresholds were measured with."""
    recommended = _shipped_config('reco3d_cr_candidate.yaml')
    differ = {k: (recommended.get(k), v) for k, v in BENCHMARK_CR_KEYS.items() if recommended.get(k) != v}
    assert not differ, differ
    assert recommended['candidate_fill_saved_peaks'] is True
    assert recommended['polarization_groups']['vpol'] == VPOL_CHANNELS
    preprocessor = recommended['preprocessor']
    assert {k: v for k, v in preprocessor.items() if k not in DELAY_CORRECTION_KEYS} == RECORD_PREPROCESSOR
    assert preprocessor['apply_delay_corrections'] is True
    assert os.path.basename(preprocessor['delay_corrections_file']) == 'delay_corrections_2022_v2.yaml'
    assert not set(RECORD_PREPROCESSOR) & set(recommended), 'preprocessing step keys at the top level are ignored'
    InterferometricReco3D()._validate_config(recommended)


def test_recommended_cr_config_is_record_plus_candidate_keys_and_corrections():
    """configs/reco3d_cr_candidate.yaml is configs/reco3d_cr.yaml plus the four candidate keys and the delay corrections."""
    record = _shipped_config('reco3d_cr.yaml')
    recommended = _shipped_config('reco3d_cr_candidate.yaml')
    assert not set(RECOMMENDED_CR_KEYS) & set(record)
    assert not DELAY_CORRECTION_KEYS & set(record['preprocessor'])
    expected = {**record, **RECOMMENDED_CR_KEYS}
    expected['preprocessor'] = {**record['preprocessor'],
                                **{k: recommended['preprocessor'][k] for k in DELAY_CORRECTION_KEYS}}
    assert recommended == expected
    InterferometricReco3D()._validate_config(record)
