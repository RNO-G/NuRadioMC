"""Known-answer reconstruction of sources above the ice surface.

Needs the air-ice tables (RECO3D_TEST_AIRICE_TABLES) and the reconstruction option
`allow_above_surface`. Synthetic events are built from the air-ice combined tables the
same way the in-ice tests build theirs. An instance that ran on the in-ice tables and is
begun again on the air-ice tables must give a fresh instance's result, so no cache built
from the in-ice tables survives the second begin().
"""

import os

import numpy as np
import pytest

from conftest import STATION, reference_config
from synthetic import (VPOL_CHANNELS, TravelTimeTables, angular_separation, cylindrical_to_enu, make_event,
                       same_value)
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.utilities.reco3d_kernels import _build_z_vec

pytest.importorskip('NuRadioMC.SignalProp.airIceRayTracing')

ANGLE_TOL_DEG = 1.0
CORR_MIN = 0.15
SNR = 20.0
# Azimuths near 100 degrees are left out: at that azimuth the station geometry admits a mirror
# solution that the search picks for far sources (seen for in-ice sources as well).
SOURCES = [
    pytest.param((50.0, 30.0, 5.0), id='rho50_h5'),
    pytest.param((100.0, 200.0, 20.0), id='rho100_h20'),
    pytest.param((150.0, 200.0, 50.0), id='rho150_h50'),
    pytest.param((30.0, 300.0, 1.0), id='rho30_h1'),
    pytest.param((200.0, 45.0, 100.0), id='rho200_h100'),
]


def above_surface_config(table_dir):
    """Reference parameters with the search volume extended to z = +300 m."""
    return reference_config(table_dir, coarse_limits=[1, 300, 0, 360, -100, 300],
                            limits=[1, 300, 0, 360, -100, 300], coarse_n_z=160,
                            z_spacing='log', z_surface_offset=0.1, allow_above_surface=True)


@pytest.fixture(scope='module')
def airice_dir():
    path = os.environ.get('RECO3D_TEST_AIRICE_TABLES', '')
    needed = [os.path.join(path, f'station{STATION}', f'st{STATION}_ch{ch}_rz_table.npz') for ch in VPOL_CHANNELS]
    if not all(os.path.isfile(p) for p in needed):
        pytest.skip(f'air-ice tables for every VPol channel not found under {path}')
    return path


@pytest.fixture(scope='module')
def airice_tables(airice_dir):
    return TravelTimeTables(airice_dir, STATION, VPOL_CHANNELS)


@pytest.fixture(scope='module')
def airice_reco(det, airice_dir):
    reco = InterferometricReco3D()
    reco.begin(STATION, above_surface_config(airice_dir), det)
    return reco


@pytest.mark.airice
def test_z_grid_spans_both_sides_of_the_surface():
    """Log spacing with z_max > 0 produces an ascending grid dense near the surface on both sides."""
    z = _build_z_vec(-100.0, 300.0, 160, 'log', 0.1)
    assert len(z) == 160
    assert np.all(np.diff(z) > 0)
    assert z[0] == pytest.approx(-100.0) and z[-1] == pytest.approx(300.0)
    assert np.sum(np.abs(z) < 1.0) >= 6
    assert (z > 0).sum() > 20 and (z < 0).sum() > 20


@pytest.mark.airice
def test_reference_config_still_rejects_positive_z_without_flag(det, table_dir):
    """Without allow_above_surface the module refuses a search volume above the surface."""
    reco = InterferometricReco3D()
    cfg = reference_config(table_dir, limits=[1, 250, 0, 360, -100, 10])
    reco.begin(STATION, cfg, det)
    with pytest.raises(ValueError):
        reco._generate_coord_arrays(dict(cfg, step_sizes=[5, 5, 5]))


@pytest.mark.airice
@pytest.mark.slow
@pytest.mark.parametrize('src', SOURCES)
def test_above_surface_source_recovered(airice_reco, det, airice_dir, airice_tables, pa, src):
    """Sources above the surface are reconstructed to within one degree with the air-ice tables."""
    cfg = above_surface_config(airice_dir)
    src_enu = cylindrical_to_enu(*src, pa)
    evt, stn, _ = make_event(det, STATION, src_enu, VPOL_CHANNELS, airice_tables, snr=SNR, seed=2)
    res = airice_reco.run(evt, stn, det, cfg)
    got = {k: round(float(res[k]), 3) for k in ('rho', 'phi', 'z', 'max_corr')}
    sep = angular_separation((res['rho'], res['phi'], res['z']), src, pa)
    assert sep < ANGLE_TOL_DEG, (src, got, round(sep, 3))
    assert res['z'] > -2.0, ('reconstructed below the surface', src, got)
    assert res['max_corr'] > CORR_MIN, got


@pytest.mark.airice
@pytest.mark.slow
def test_in_ice_source_unchanged_with_extended_volume(airice_reco, det, airice_dir, airice_tables, pa):
    """An in-ice source still reconstructs when the volume also covers the air.

    The extended volume spreads 160 log-spaced coarse z points over both sides of the
    surface, about 3 m spacing at 30 m depth against 1 m in the reference configuration, so the tolerance
    is 2 degrees here rather than exact recovery.
    """
    src = (90.0, 65.0, -30.0)
    cfg = above_surface_config(airice_dir)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, airice_tables, snr=SNR, seed=2)
    res = airice_reco.run(evt, stn, det, cfg)
    got = {k: round(float(res[k]), 3) for k in ('rho', 'phi', 'z', 'max_corr')}
    sep = angular_separation((res['rho'], res['phi'], res['z']), src, pa)
    assert sep < 2.0, (src, got, round(sep, 3))
    assert res['z'] < 0, got


@pytest.mark.airice
@pytest.mark.slow
def test_second_begin_with_air_ice_tables_matches_a_fresh_instance(det, table_dir, tables, airice_dir, airice_tables, pa):
    """An instance that ran on the in-ice tables, begun again on the air-ice tables, gives a fresh instance's result.

    The optimizer geometry is cached per channel tuple, which both table sets share, so
    a cache kept across the second begin() would optimize the above-surface source on
    the in-ice tables, which hold no travel times above the surface.
    """
    cfg_ice = reference_config(table_dir)
    cfg_air = above_surface_config(airice_dir)
    reused = InterferometricReco3D()
    reused.begin(STATION, cfg_ice, det)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(90.0, 65.0, -30.0, pa), VPOL_CHANNELS, tables,
                             snr=SNR, seed=3)
    reused.run(evt, stn, det, cfg_ice)
    reused.begin(STATION, cfg_air, det)
    fresh = InterferometricReco3D()
    fresh.begin(STATION, cfg_air, det)
    results = []
    for instance in (reused, fresh):
        evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(50.0, 30.0, 5.0, pa), VPOL_CHANNELS, airice_tables,
                                 snr=SNR, seed=2)
        results.append(instance.run(evt, stn, det, cfg_air))
    got, want = results
    assert set(got) == set(want)
    differ = [k for k in want if 'time' not in k and not same_value(want[k], got[k])]
    assert not differ, [(k, want[k], got[k]) for k in differ]
