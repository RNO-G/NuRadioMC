"""The compiled multi-ray point lookup and optimizer differences equal the Python objective and scipy bit for bit."""

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative

from conftest import STATION, reference_config
from synthetic import VPOL_CHANNELS
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D


@pytest.mark.slow
@pytest.mark.parametrize('mode', ['grouped', 'per_pair'])
def test_multiray_objective_equals_python_loop(det, table_dir, mode):
    """``_correlation_at_point`` with the compiled travel-time lookup ('mr_tables') equals the per-table loop."""
    cfg = reference_config(table_dir, multi_ray_types=True, table_scheme='solution_ordered',
                           multiray_combo_mode=mode)
    reco = InterferometricReco3D()
    reco.begin(STATION, cfg, det)
    channels = VPOL_CHANNELS
    rng = np.random.default_rng(3)
    corr_data = [(rng.normal(0, 0.3, 4001), 0.1, -200.0) for _ in itertools.combinations(channels, 2)]
    weights = list(rng.uniform(0.2, 1.5, len(corr_data)))
    cache = reco._build_optimizer_cache(channels, weights, corr_data=corr_data)
    assert 'mr_tables' in cache
    loop = {k: v for k, v in cache.items() if k != 'mr_tables'}
    for _ in range(1500):
        point = [rng.uniform(1, 1600), rng.uniform(0, 360), rng.uniform(-1500, 300)]
        a = reco._correlation_at_point(point, corr_data, channels, weights, _cache=cache)
        b = reco._correlation_at_point(point, corr_data, channels, weights, _cache=loop)
        assert np.float64(a).tobytes() == np.float64(b).tobytes(), (point, a, b)


@pytest.mark.slow
def test_grouped_optimizer_reproduces_scipy_lbfgsb(det, table_dir):
    """Grouped multi-ray ``_optimize_from_seed`` ends where scipy's L-BFGS-B differencing the Python objective ends.

    Seeds include points on and next to the bounds, where the steps turn one-sided; the compiled
    gradient also equals approx_derivative at every seed.
    """
    from NuRadioReco.utilities.reco3d_kernels import _grouped_fd_value_grad, _lbfgsb_fd_points
    cfg = reference_config(table_dir, multi_ray_types=True, table_scheme='solution_ordered',
                           multiray_combo_mode='grouped')
    reco = InterferometricReco3D()
    reco.begin(STATION, cfg, det)
    channels = VPOL_CHANNELS
    rng = np.random.default_rng(7)
    corr_data = [(np.convolve(rng.normal(0, 0.3, 4001), np.ones(25) / 25, mode='same'), 0.1, -200.0)
                 for _ in itertools.combinations(channels, 2)]
    weights = list(rng.uniform(0.2, 1.5, len(corr_data)))
    cache = reco._build_optimizer_cache(channels, weights, corr_data=corr_data)
    bounds = [(1.0, 1600.0), (0.0, 360.0), (-1500.0, 300.0)]
    seeds = [(rng.uniform(1, 1600), rng.uniform(0, 360), rng.uniform(-1500, 300)) for _ in range(40)]
    seeds += [(1.0, 30.0, -50.0), (300.0, 100.0, 300.0), (300.0, 200.0, -1500.0), (1.0 + 5e-9, 10.0, -5e-9)]
    for seed in seeds:
        phi_shift = seed[1] - 180.0

        def objective(params):
            return reco._correlation_at_point([params[0], (params[1] + phi_shift) % 360.0, params[2]],
                                              corr_data, channels, weights, _cache=cache)

        shifted = [bounds[0], (0.0, 360.0), bounds[2]]
        x0 = np.array([seed[0], 180.0, seed[2]])
        lb, ub = np.array([b[0] for b in shifted]), np.array([b[1] for b in shifted])
        c = cache
        points, phi_rad = _lbfgsb_fd_points(x0, phi_shift, lb, ub)
        args = ((float(c['pa_center'][0]), float(c['pa_center'][1]), c['ant_pos']) + c['mr_tables']
                + (reco._n_ray_slots, c['corr_packed'], c['corr_lengths'], c['corr_dts'], c['corr_offsets'],
                   c['pair_ch1'], c['pair_ch2'], c['pw'], c['ch_group'], c['group_rts'], c['group_nrt'], c['n_pairs'],
                   c['w_total'], np.zeros(2, dtype=np.int64)))
        f, g = _grouped_fd_value_grad(points, np.cos(phi_rad), np.sin(phi_rad), *args)
        assert f == objective(x0), seed
        assert np.array_equal(g, approx_derivative(objective, x0, method='2-point', abs_step=1e-8,
                                                   bounds=(lb, ub), f0=f)), seed
        ref = minimize(objective, x0, method='L-BFGS-B', bounds=shifted, options={'maxiter': 30, 'ftol': 1e-10})
        got = reco._optimize_from_seed(seed, corr_data, channels, bounds, weights, _cache=cache)
        assert got == (ref.x[0], (ref.x[1] + phi_shift) % 360.0, ref.x[2], -ref.fun), seed


@pytest.mark.slow
@pytest.mark.parametrize('seed', range(6))
def test_grouped_blocks_equal_combination_sums(seed):
    """The group-block, depth-first grouped objective equals the best per-combination pair sum to rounding.

    Random travel times with unusable entries, groups of one to three channels, one group restricted to one
    ray type, series with out-of-range delays.
    """
    from NuRadioReco.utilities.reco3d_kernels import _scalar_grouped_corr_numba
    rng = np.random.default_rng(seed)
    n_ch, n_rt = 12, 2
    ch_group = np.sort(rng.integers(0, 7, n_ch))
    ch_group = np.unique(ch_group, return_inverse=True)[1].astype(np.int64)
    n_groups = int(ch_group.max()) + 1
    group_opts = [[0, 1] for _ in range(n_groups)]
    group_opts[n_groups // 2] = [1]
    group_rts = np.full((n_groups, 2), -1, dtype=np.int64)
    for g, opts in enumerate(group_opts):
        group_rts[g, :len(opts)] = opts
    group_nrt = np.array([len(o) for o in group_opts], dtype=np.int64)
    tt_vals = 500.0 + 60.0 * rng.random((n_ch, n_rt))
    tt_valid = rng.random((n_ch, n_rt)) > 0.15
    pairs = list(itertools.combinations(range(n_ch), 2))
    n_pairs = len(pairs)
    corr = rng.normal(0, 0.3, (n_pairs, 1200))
    lengths = rng.integers(400, 1200, n_pairs).astype(np.int64)
    dts = 0.1 + 0.02 * rng.random(n_pairs)
    offs = -40.0 - 10.0 * rng.random(n_pairs)
    ch1 = np.array([p[0] for p in pairs], dtype=np.int64)
    ch2 = np.array([p[1] for p in pairs], dtype=np.int64)
    w = rng.uniform(0.2, 1.5, n_pairs)
    best = -np.inf
    for combo in itertools.product(*group_opts):
        total = 0.0
        for p, (a, b) in enumerate(pairs):
            r1, r2 = combo[ch_group[a]], combo[ch_group[b]]
            if not (tt_valid[a, r1] and tt_valid[b, r2]):
                continue
            kf = (tt_vals[a, r1] - tt_vals[b, r2] - offs[p]) / dts[p]
            k = int(np.floor(kf))
            if 0 <= k < lengths[p] - 1:
                total += (corr[p, k] + (corr[p, k + 1] - corr[p, k]) * (kf - k)) * w[p]
        best = max(best, total)
    walk = np.zeros(2, dtype=np.int64)
    got = _scalar_grouped_corr_numba(tt_vals, tt_valid, corr, lengths, dts, offs, ch1, ch2, w, ch_group,
                                     group_rts, group_nrt, n_pairs, float(w.sum()), walk)
    assert abs(got - (-best / w.sum())) <= 1e-12
    full_nodes = sum(int(np.prod(group_nrt[:d + 1])) for d in range(len(group_nrt)))
    assert 0 < walk[0] <= full_nodes and 0 <= walk[1] < walk[0]
