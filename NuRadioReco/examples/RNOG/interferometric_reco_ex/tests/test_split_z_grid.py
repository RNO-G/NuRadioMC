"""Split z grid: an in-ice block and an air block that meet at the ice surface.

With ``z_grid_below`` and ``z_grid_above`` the coarse z vector is the grid an in-ice search
volume builds (for the reference configuration 100 linear points over [-100, 0]) followed
by a log grid above the surface that starts at the first tabulated air row. The two blocks
are searched separately (peaks, refine windows, optimizer and polish clamped at z = 0) and
merged by correlation. The identity test checks that an in-ice source reconstructs to
exactly the same (rho, phi, z, max_corr) under the two-sided split configuration as under
the in-ice configuration, both on the air-ice tables so that only the grid differs, on the
exact-recovery sources, for the default chain and for the candidate search. The above-surface test repeats the air-ice
known-answer test with the split grid. With the keys absent nothing changes:
test_golden_master.py is the check. On an in-ice volume the split grid holds the in-ice
block alone, and the block path (per-block refine, optimizer and candidate polish, the
merge, the tie band with its noise ceiling, the filled saved peaks and the candidate pool)
must then give every result field of the default search on the record tables exactly.
The flat search (``hierarchical: false``) accepts the split keys too; its travel-time and
delay caches are keyed by the z vector, so configurations with the same number of z
points never share them.
"""

import os

import numpy as np
import pytest

from conftest import STATION, reference_config
from synthetic import (VPOL_CHANNELS, TravelTimeTables, angular_separation, cylindrical_to_enu, make_event,
                       make_noise_event, same_value)
from test_reco_known_answer import EXACT_CORR_MIN, EXACT_SNR, EXACT_SOURCES
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.utilities.reco3d_kernels import _build_split_z_vec, _build_split_z_window, _build_z_vec

BELOW = {'n': 100, 'spacing': 'linear'}
ABOVE = {'n': 60, 'spacing': 'log', 'offset': 1.0, 'refine_spacing': 'linear'}
LOG_REFINE = dict(ABOVE, refine_spacing='log')
AIR_ANGLE_TOL_DEG = 1.0
AIR_CORR_MIN = 0.15
AIR_SNR = 20.0
AIR_SOURCES = [
    pytest.param((50.0, 30.0, 5.0), id='rho50_h5'),
    pytest.param((100.0, 200.0, 20.0), id='rho100_h20'),
    pytest.param((150.0, 200.0, 50.0), id='rho150_h50'),
    pytest.param((30.0, 300.0, 1.0), id='rho30_h1'),
    pytest.param((200.0, 45.0, 100.0), id='rho200_h100'),
]
CANDIDATE = {'candidate_search': ['envelope:traces', 'envelope:correlation', 'raw']}
RECOMMENDED_CANDIDATE = dict(CANDIDATE, candidate_tie_band=0.0054, candidate_tie_band_max_raw_corr=0.035,
                             candidate_fill_saved_peaks=True, candidate_diagnostics=True)
FLAT_SOURCE = (60.0, 120.0, -40.0)
FLAT_N_Z = 40
SINGLE_BLOCK_CASES = [
    pytest.param(('signal', (30.0, 300.0, -5.0)), id='near_shallow'),
    pytest.param(('signal', (113.3, 235.5, -49.9)), id='far_southwest'),
    pytest.param(('noise', 1), id='noise1'),
]


def split_config(table_dir, rho_max=250, **overrides):
    """Reference parameters with the volume extended to z = +300 m on the split grid."""
    cfg = reference_config(table_dir, coarse_limits=[1, rho_max, 0, 360, -100, 300],
                           limits=[1, rho_max, 0, 360, -100, 300], allow_above_surface=True,
                           z_grid_below=dict(BELOW), z_grid_above=dict(ABOVE))
    del cfg['coarse_n_z']
    cfg.update(overrides)
    return cfg


def flat_config(table_dir, **overrides):
    """Reference parameters for the flat search over rho 1 to 100 m and z -100 to 0 m on 5 m and 5 deg steps."""
    cfg = reference_config(table_dir, hierarchical=False, limits=[1, 100, 0, 360, -100, 0], step_sizes=[5, 5, 5],
                           **overrides)
    del cfg['coarse_n_z']
    return cfg


@pytest.fixture(scope='module')
def airice_dir():
    """Root of the air-ice tables, or skip when a VPol channel's table is missing."""
    path = os.environ.get('RECO3D_TEST_AIRICE_TABLES', '')
    needed = [os.path.join(path, f'station{STATION}', f'st{STATION}_ch{ch}_rz_table.npz') for ch in VPOL_CHANNELS]
    if not all(os.path.isfile(p) for p in needed):
        pytest.skip(f'air-ice tables for every VPol channel not found under {path}')
    return path


@pytest.fixture(scope='module')
def airice_tables(airice_dir):
    """Combined air-ice travel-time tables for the VPol channels."""
    return TravelTimeTables(airice_dir, STATION, VPOL_CHANNELS)


@pytest.fixture(scope='module')
def inice_airice_reco(det, airice_dir):
    """Reconstruction object initialised with the in-ice configuration on the air-ice tables."""
    reco = InterferometricReco3D()
    reco.begin(STATION, reference_config(airice_dir), det)
    return reco


@pytest.fixture(scope='module')
def split_reco(det, airice_dir):
    """Reconstruction object initialised with the two-sided split configuration."""
    reco = InterferometricReco3D()
    reco.begin(STATION, split_config(airice_dir), det)
    return reco


def test_split_z_vec_is_in_ice_grid_plus_air_block():
    """The coarse vector is the in-ice grid bit for bit followed by the air block."""
    below = dict(BELOW, offset=0.1)
    n_air = ABOVE['n']
    z = _build_split_z_vec(-100.0, 300.0, below, ABOVE)
    assert len(z) == 100 + n_air
    assert np.array_equal(z[:100], np.linspace(-100.0, 0.0, 100))
    assert np.array_equal(z[:100], _build_z_vec(-100.0, 0.0, 100))
    assert np.array_equal(z[100:], np.geomspace(1.0, 300.0, n_air))
    assert np.all(np.diff(z) > 0)
    linear_air = _build_split_z_vec(-100.0, 300.0, below, dict(ABOVE, spacing='linear'))
    assert np.array_equal(linear_air[100:], np.linspace(1.0, 300.0, n_air))
    log_ice = _build_split_z_vec(-100.0, 300.0, dict(below, spacing='log'), ABOVE)
    assert np.array_equal(log_ice[:100], _build_z_vec(-100.0, 0.0, 100, 'log', 0.1))
    assert np.array_equal(_build_split_z_vec(-100.0, 0.0, below, ABOVE), np.linspace(-100.0, 0.0, 100))
    assert np.array_equal(_build_split_z_vec(50.0, 300.0, below, ABOVE), np.geomspace(50.0, 300.0, n_air))
    with pytest.raises(ValueError):
        _build_split_z_vec(0.0, 0.5, below, ABOVE)


def test_split_z_window_splits_at_the_surface():
    """Windows crossing z = 0 keep the linear in-ice part and add an air part from the offset."""
    crossing = _build_split_z_window(-20.05, 9.95, 1.0, LOG_REFINE)
    ice = crossing[crossing <= 0]
    air = crossing[crossing > 0]
    assert np.array_equal(ice, np.arange(-20.05, 1.0, 1.0)[:-1])
    assert ice[-1] == pytest.approx(-0.05)
    assert air[0] == 1.0 and air[-1] == pytest.approx(9.95) and len(air) == 9
    assert np.allclose(air, np.geomspace(1.0, 9.95, 9))
    assert np.all(np.diff(crossing) > 0)
    in_ice = _build_split_z_window(-60.05, -30.05, 1.0, LOG_REFINE)
    assert np.array_equal(in_ice, np.arange(-60.05, -29.05, 1.0))
    in_air = _build_split_z_window(150.0, 180.0, 1.0, LOG_REFINE)
    assert np.allclose(in_air, np.geomspace(150.0, 180.0, 31))
    linear_air = _build_split_z_window(-5.0, 9.0, 1.0, ABOVE)
    assert np.array_equal(linear_air, np.concatenate((np.arange(-5.0, 1.0, 1.0), np.arange(1.0, 10.0, 1.0))))
    assert np.array_equal(_build_split_z_window(150.0, 180.0, 1.0, ABOVE), np.arange(150.0, 181.0, 1.0))
    below_offset = _build_split_z_window(-5.0, 0.5, 1.0, ABOVE)
    assert np.array_equal(below_offset, np.arange(-5.0, 1.0, 1.0))
    assert len(_build_split_z_window(0.2, 0.5, 1.0, ABOVE)) == 0


def test_invalid_split_keys_are_rejected():
    """One key alone, a replaced key, malformed blocks, bad counts, spacings and offsets raise."""
    reco = InterferometricReco3D()
    for bad in ({'z_grid_below': {'n': 100}},
                {'z_grid_above': {'n': 40}},
                {'z_grid_below': {'n': 100}, 'z_grid_above': {'n': 40}, 'coarse_n_z': 160},
                {'z_grid_below': {'n': 100}, 'z_grid_above': {'n': 40}, 'n_z': 50},
                {'z_grid_below': {'n': 100}, 'z_grid_above': {'n': 40}, 'z_spacing': 'linear'},
                {'z_grid_below': [100], 'z_grid_above': {'n': 40}},
                {'z_grid_below': {'n': 1}, 'z_grid_above': {'n': 40}},
                {'z_grid_below': {'n': 100.0}, 'z_grid_above': {'n': 40}},
                {'z_grid_below': {}, 'z_grid_above': {'n': 40}},
                {'z_grid_below': {'n': 100, 'refine_spacing': 'log'}, 'z_grid_above': {'n': 40}},
                {'z_grid_below': {'n': 100}, 'z_grid_above': {'n': 40, 'spacing': 'cubic'}},
                {'z_grid_below': {'n': 100}, 'z_grid_above': {'n': 40, 'refine_spacing': 'cubic'}},
                {'z_grid_below': {'n': 100}, 'z_grid_above': {'n': 40, 'offset': 0.0}},
                {'z_grid_below': {'n': 100, 'offset': -1.0}, 'z_grid_above': {'n': 40}}):
        with pytest.raises(ValueError):
            reco._validate_config(bad)
    assert reco._split_z_grid({}) is None
    below, above = reco._split_z_grid({'z_grid_below': dict(BELOW), 'z_grid_above': dict(LOG_REFINE)})
    assert below == {'n': 100, 'spacing': 'linear', 'offset': 0.1}
    assert above == {'n': ABOVE['n'], 'spacing': 'log', 'offset': 1.0, 'refine_spacing': 'log'}
    below, above = reco._split_z_grid({'z_grid_below': {'n': 50}, 'z_grid_above': {'n': 20},
                                       'z_surface_offset': 0.5, 'coarse_n_z': 0})
    assert below == {'n': 50, 'spacing': 'linear', 'offset': 0.5}
    assert above == {'n': 20, 'spacing': 'log', 'offset': 1.0, 'refine_spacing': 'linear'}


def test_coarse_grid_and_cache_key_follow_the_split_keys():
    """The coarse z vector and its cache key change with the split keys and not otherwise."""
    reco = InterferometricReco3D()
    z_ref, key_ref = reco._coarse_z_grid({'coarse_n_z': 100}, -100, 0, 0)
    assert np.array_equal(z_ref, np.linspace(-100.0, 0.0, 100))
    assert key_ref == (0, -100, 0, 100, 'linear', 0.1)
    z_split, key_split = reco._coarse_z_grid({'z_grid_below': dict(BELOW), 'z_grid_above': dict(ABOVE)}, -100, 300, 0)
    assert np.array_equal(z_split[:100], z_ref) and len(z_split) == 100 + ABOVE['n']
    assert key_split[:3] == (-100, 300, 'split') and len(key_split) == 5
    _, key_other = reco._coarse_z_grid({'z_grid_below': dict(BELOW), 'z_grid_above': dict(ABOVE, n=ABOVE['n'] + 1)}, -100, 300, 0)
    assert key_other != key_split
    _, key_refine = reco._coarse_z_grid({'z_grid_below': dict(BELOW), 'z_grid_above': dict(LOG_REFINE)}, -100, 300, 0)
    assert key_refine != key_split
    z_two_sided, key_two_sided = reco._coarse_z_grid({'coarse_n_z': 160, 'z_spacing': 'log'}, -100, 300, 0)
    assert np.array_equal(z_two_sided, _build_z_vec(-100, 300, 160, 'log', 0.1))
    assert key_two_sided == (0, -100, 300, 160, 'log', 0.1)


def test_z_blocks_partition_the_coarse_grid():
    """The split grid is searched as an in-ice block clamped at z = 0 and an air block clamped at z = 0 from below."""
    reco = InterferometricReco3D()
    limits = [1, 250, 0, 360, -100, 300]
    assert reco._z_blocks({'coarse_n_z': 100}, np.linspace(-100.0, 0.0, 100), limits) is None
    cfg = {'z_grid_below': dict(BELOW), 'z_grid_above': dict(ABOVE)}
    z_vec, _ = reco._coarse_z_grid(cfg, -100, 300, 0)
    (ice_slice, ice_limits, ice_above), (air_slice, air_limits, air_above) = reco._z_blocks(cfg, z_vec, limits)
    assert np.array_equal(z_vec[ice_slice], np.linspace(-100.0, 0.0, 100))
    assert np.all(z_vec[air_slice] >= ABOVE['offset']) and len(z_vec[air_slice]) == ABOVE['n']
    assert ice_limits == [1, 250, 0, 360, -100, 0.0] and ice_above is None
    assert air_limits == [1, 250, 0, 360, 0.0, 300] and air_above == reco._split_z_grid(cfg)[1]
    assert limits == [1, 250, 0, 360, -100, 300]
    in_ice_only = reco._z_blocks(cfg, np.linspace(-100.0, 0.0, 100), [1, 250, 0, 360, -100, 0])
    assert len(in_ice_only) == 1 and in_ice_only[0][2] is None
    air_only = reco._z_blocks(cfg, np.geomspace(50.0, 300.0, 10), [1, 250, 0, 360, 50, 300])
    assert len(air_only) == 1 and air_only[0][1][4] == 50


def test_default_config_has_no_split_keys(base_config):
    """The reference configuration carries neither key, so the golden master covers the default path."""
    assert 'z_grid_below' not in base_config and 'z_grid_above' not in base_config
    assert InterferometricReco3D._split_z_grid(base_config) is None


@pytest.mark.airice
@pytest.mark.slow
@pytest.mark.parametrize('mode', ['record', 'candidate'])
@pytest.mark.parametrize('src', EXACT_SOURCES)
def test_in_ice_source_identical_under_split_grid(inice_airice_reco, split_reco, det, airice_dir, airice_tables, pa, src, mode):
    """An in-ice source gives the same (rho, phi, z, max_corr) on the two-sided split grid as in ice.

    Both configurations use the air-ice tables, so only the grid differs. The in-ice block
    is the in-ice grid node for node and runs its own peak extraction, refine levels,
    optimizer and candidate polish within z <= 0, so it repeats the in-ice search exactly;
    the result can differ only when an air maximum has the higher correlation, which these
    sources do not produce. The values are therefore compared for equality.
    """
    overrides = CANDIDATE if mode == 'candidate' else {}
    cfg_ice = reference_config(airice_dir, **overrides)
    cfg_split = split_config(airice_dir, **overrides)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, airice_tables, snr=EXACT_SNR, seed=1)
    res_ice = inice_airice_reco.run(evt, stn, det, cfg_ice)
    res_split = split_reco.run(evt, stn, det, cfg_split)
    got = {k: (float(res_ice[k]), float(res_split[k])) for k in ('rho', 'phi', 'z', 'max_corr')}
    assert res_ice['max_corr'] > EXACT_CORR_MIN, got
    for k, (a, b) in got.items():
        assert a == b, (k, got)


@pytest.mark.airice
@pytest.mark.slow
@pytest.mark.parametrize('src', AIR_SOURCES)
def test_above_surface_source_recovered_on_split_grid(split_reco, det, airice_dir, airice_tables, pa, src):
    """Sources above the surface are reconstructed within one degree on the split grid."""
    cfg = split_config(airice_dir, rho_max=300)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, airice_tables, snr=AIR_SNR, seed=2)
    res = split_reco.run(evt, stn, det, cfg)
    got = {k: round(float(res[k]), 3) for k in ('rho', 'phi', 'z', 'max_corr')}
    sep = angular_separation((res['rho'], res['phi'], res['z']), src, pa)
    assert sep < AIR_ANGLE_TOL_DEG, (src, got, round(sep, 3))
    assert res['z'] > -2.0, ('reconstructed below the surface', src, got)
    assert res['max_corr'] > AIR_CORR_MIN, got


@pytest.mark.slow
@pytest.mark.parametrize('mode', ['record', 'candidate'])
@pytest.mark.parametrize('case', SINGLE_BLOCK_CASES)
def test_split_grid_without_air_block_repeats_default_search(reco, det, table_dir, tables, pa, case, mode):
    """On an in-ice volume the split keys give one in-ice block whose result equals the default search field by field."""
    overrides = RECOMMENDED_CANDIDATE if mode == 'candidate' else {}
    cfg_default = reference_config(table_dir, **overrides)
    cfg_split = reference_config(table_dir, z_grid_below=dict(BELOW), z_grid_above=dict(ABOVE), **overrides)
    del cfg_split['coarse_n_z']
    z_vec, _ = reco._coarse_z_grid(cfg_split, -100, 0, 0)
    assert np.array_equal(z_vec, np.linspace(-100.0, 0.0, 100))
    assert len(reco._z_blocks(cfg_split, z_vec, cfg_split['limits'])) == 1
    kind, value = case
    results = []
    for cfg in (cfg_default, cfg_split):
        if kind == 'signal':
            evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*value, pa), VPOL_CHANNELS, tables,
                                     snr=AIR_SNR, seed=4)
        else:
            evt, stn = make_noise_event(STATION, VPOL_CHANNELS, seed=value)
        results.append(reco.run(evt, stn, det, cfg))
    default, split = results
    assert set(default) == set(split), sorted(set(default) ^ set(split))
    differ = [k for k in default if 'time' not in k and not same_value(default[k], split[k])]
    assert not differ, [(k, default[k], split[k]) for k in differ]
    if mode == 'candidate' and kind == 'noise':
        assert split['candidate_fallback'] == 1


@pytest.mark.slow
@pytest.mark.parametrize('fused', [True, False], ids=['fused', 'delay_matrix'])
def test_flat_search_caches_follow_the_z_vector(det, table_dir, tables, pa, fused):
    """Flat configurations with equal z point counts run on one instance give the results of fresh instances.

    A plain linear n_z grid and two in-ice split grids with log blocks of different offsets
    all have FLAT_N_Z points and equal limits and steps, so a cache keyed by the point count
    would hand the later ones the first one's travel times.
    """
    log_block = {'n': FLAT_N_Z, 'spacing': 'log'}
    configs = [flat_config(table_dir, n_z=FLAT_N_Z, use_fused_correlator=fused),
               flat_config(table_dir, z_grid_below=log_block, z_grid_above=dict(ABOVE), use_fused_correlator=fused),
               flat_config(table_dir, z_grid_below=dict(log_block, offset=1.0), z_grid_above=dict(ABOVE),
                           use_fused_correlator=fused)]
    z_vecs = [InterferometricReco3D()._generate_coord_arrays(cfg)[2] for cfg in configs]
    assert all(len(z) == FLAT_N_Z for z in z_vecs)
    assert not np.array_equal(z_vecs[0], z_vecs[1]) and not np.array_equal(z_vecs[1], z_vecs[2])
    shared = InterferometricReco3D()
    shared.begin(STATION, configs[0], det)
    for cfg in configs:
        fresh = InterferometricReco3D()
        fresh.begin(STATION, cfg, det)
        results = []
        for instance in (shared, fresh):
            evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*FLAT_SOURCE, pa), VPOL_CHANNELS, tables,
                                     snr=AIR_SNR, seed=5)
            results.append(instance.run(evt, stn, det, cfg))
        got, want = results
        assert set(got) == set(want)
        differ = [k for k in want if 'time' not in k and not same_value(want[k], got[k])]
        assert not differ, [(k, want[k], got[k]) for k in differ]
