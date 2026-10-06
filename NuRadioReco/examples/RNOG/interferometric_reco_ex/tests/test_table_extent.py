"""Travel-time cells the kernels cannot read: NaN corners, queries outside a table, tables of differing extent.

Every kernel of the singleray search (the coarse map, the refine and polish grids, the
optimizer and the grading of candidates) reads a channel's table only where the query lies
inside the table and the four corners of its cell are finite, with a positive value; anywhere
else every pair with that channel is skipped at that point: it adds nothing to the weighted
sum and, under the default ``objective_normalisation: total``, its weight stays in the
divisor. The tests check that rule on a table whose rows above z = -50 m are NaN: below the
NaN rows the objective is unchanged bit for bit, above them it equals the sum over the other
pairs divided by the total weight, on the scalar objective and on a grid map; the lag
windows ignore the NaN cells and stay finite, and a channel with no readable cell leaves
its pairs a NaN window. A table padded with NaN rows beyond its range is read like the
table itself: the windows and the whole record and candidate search are identical (exactly,
here, because the cell fraction (z - z_min) * dz_inv rounds alike for both origins; in
general a different origin moves interpolated travel times by float rounding).
"""

import itertools
import os

import numpy as np
import pytest

from conftest import STATION, reference_config
from synthetic import VPOL_CHANNELS, TravelTimeTables, cylindrical_to_enu, make_event, same_value
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

NAN_CHANNEL = 6
NAN_ABOVE_Z = -50.0
CANDIDATE = dict(candidate_search=['envelope:traces', 'envelope:correlation', 'raw'],
                 candidate_tie_band=0.0054, candidate_tie_band_max_raw_corr=0.035,
                 candidate_fill_saved_peaks=True, candidate_diagnostics=True)


def _table_dir(tmp_path, table_dir, name, edit):
    """A table directory linking the VPol tables, with channel NAN_CHANNEL's table rewritten by ``edit``."""
    out = tmp_path / name / f'station{STATION}'
    out.mkdir(parents=True)
    for ch in VPOL_CHANNELS:
        src = os.path.join(table_dir, f'station{STATION}', f'st{STATION}_ch{ch}_rz_table.npz')
        dst = out / os.path.basename(src)
        if ch != NAN_CHANNEL:
            os.symlink(src, dst)
            continue
        with np.load(src) as d:
            r, z, data = edit(d['r_range_vals'], d['z_range_vals'], d['data'].copy())
        np.savez(dst, r_range_vals=r, z_range_vals=z, data=data)
    return str(tmp_path / name)


def _nan_above(r, z, data):
    """Set every row above NAN_ABOVE_Z to NaN."""
    data[:, z > NAN_ABOVE_Z] = np.nan
    return r, z, data


def _pad_below(r, z, data, n=100):
    """Prepend n NaN rows below the table's lowest row on the same grid."""
    dz = z[1] - z[0]
    z_pad = z[0] - dz * np.arange(n, 0, -1)
    return r, np.concatenate([z_pad, z]), np.concatenate([np.full((len(r), n), np.nan), data], axis=1)


def _begun(det, table_dir, **overrides):
    """Reconstruction object begun on a table directory with the reference VPol configuration."""
    config = reference_config(table_dir, **overrides)
    reco = InterferometricReco3D()
    reco.begin(STATION, config, det)
    return reco, config


@pytest.fixture(scope='module')
def event(det, tables, pa):
    """A VPol event of a source below the NaN rows."""
    return make_event(det, STATION, cylindrical_to_enu(70.0, 150.0, -70.0, pa), VPOL_CHANNELS,
                      tables, snr=15.0, seed=17)[:2]


@pytest.mark.slow
def test_unreadable_cell_drops_the_pair_and_keeps_its_weight(det, table_dir, tmp_path, event):
    """With channel 6 unreadable above z = -50 m its pairs add nothing there and the total weight is unchanged."""
    nan_dir = _table_dir(tmp_path, table_dir, 'nan', _nan_above)
    full, config = _begun(det, table_dir)
    cut, _ = _begun(det, nan_dir)
    pairs = full.compute_pairs(event[1], config)
    weights = full._group_pair_weights(pairs.snr, pairs.snr_windowed, VPOL_CHANNELS, config)
    corr_data, packed = full._group_inputs(pairs, VPOL_CHANNELS, config)[3](None)
    with_6 = np.array([NAN_CHANNEL in p for p in itertools.combinations(VPOL_CHANNELS, 2)])
    w = np.array(weights)
    zeroed = list(np.where(with_6, 0.0, w))
    caches = {name: reco._build_optimizer_cache(VPOL_CHANNELS, wts, corr_data, packed=packed)
              for name, reco, wts in (('full', full, weights), ('cut', cut, weights),
                                      ('zeroed', full, zeroed))}

    def value(name, point):
        """Objective of one cache at (rho, phi_deg, z)."""
        reco = cut if name == 'cut' else full
        return -reco._correlation_at_point(point, corr_data, VPOL_CHANNELS, None, _cache=caches[name])

    scale = w[~with_6].sum() / w.sum()
    for point in ([70.0, 150.0, -70.0], [40.0, 10.0, -50.5], [120.0, 200.0, -90.0]):
        assert value('cut', point) == value('full', point), point
    for point in ([70.0, 150.0, -30.0], [40.0, 10.0, -49.5], [90.0, 300.0, -25.0]):
        assert value('cut', point) == pytest.approx(value('zeroed', point) * scale, rel=1e-12, abs=1e-15)
        assert value('cut', point) != value('full', point)

    rho = np.linspace(20.0, 200.0, 7)
    phi = np.radians(np.linspace(0.0, 330.0, 12))
    z_below, z_above = np.linspace(-95.0, -51.0, 5), np.linspace(-49.5, -1.0, 5)
    for z, expect in ((z_below, 'full'), (z_above, 'zeroed')):
        got = cut._singleray_grid_maps(rho, phi, z, VPOL_CHANNELS, [packed], weights)
        ref = full._singleray_grid_maps(rho, phi, z, VPOL_CHANNELS, [packed],
                                        weights if expect == 'full' else zeroed)
        if expect == 'full':
            assert np.array_equal(got, ref)
        else:
            assert np.allclose(got, ref * scale, rtol=1e-12, atol=1e-15)

    windows = cut.pair_lag_windows(list(itertools.combinations(VPOL_CHANNELS, 2)), config)
    assert np.all(np.isfinite(windows))
    above = dict(config, coarse_limits=[1, 250, 0, 360, -40, 0], limits=[1, 250, 0, 360, -40, 0])
    windows = cut.pair_lag_windows(list(itertools.combinations(VPOL_CHANNELS, 2)), above)
    assert np.all(np.isnan(windows[with_6])) and np.all(np.isfinite(windows[~with_6]))
    result = cut.run(*event, det, dict(config, **CANDIDATE))
    assert np.isfinite(result['max_corr'])


@pytest.mark.slow
@pytest.mark.parametrize('candidate', [False, True], ids=['record', 'recommended'])
def test_nan_padded_table_equals_the_table(det, table_dir, tmp_path, event, candidate):
    """A table padded with NaN rows below its range gives the same lag windows and the same search."""
    padded_dir = _table_dir(tmp_path, table_dir, 'padded', _pad_below)
    extra = CANDIDATE if candidate else {}
    plain, config = _begun(det, table_dir, **extra)
    padded, padded_config = _begun(det, padded_dir, **extra)
    assert padded._interpolators[NAN_CHANNEL].nz == plain._interpolators[NAN_CHANNEL].nz + 100
    pairs = list(itertools.combinations(VPOL_CHANNELS, 2))
    assert np.array_equal(padded.pair_lag_windows(pairs, padded_config),
                          plain.pair_lag_windows(pairs, config), equal_nan=True)
    expected = plain.run(*event, det, config)
    got = padded.run(*event, det, padded_config)
    for key in expected:
        if '_time' not in key:
            assert same_value(expected[key], got[key]), (key, expected[key], got[key])
