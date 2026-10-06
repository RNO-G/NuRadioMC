"""Gradients passed to L-BFGS-B: compiled finite differences (default) and exact gradients (optimizer_gradient: exact).

By default the point-source and far-field L-BFGS-B receive, with the objective, the forward
differences scipy would take itself (absolute step 1e-8, steps adjusted to the bounds): the steps
and the gradient equal ``scipy.optimize._numdiff.approx_derivative`` bit for bit, and the optimizer
end points equal those of scipy's own differencing of the Python objective bit for bit. The three
far-field points are evaluated in one ``plane_wave_times`` call, whose rows equal single calls bit
for bit. The exact gradients (point source: weighted sum of linearly interpolated pair correlations
at differences of bilinearly interpolated table travel times; far field: the same sum at plane-wave
arrival times) are checked against central differences at random points of synthetic events
wherever the two one-sided differences agree (no table or series cell boundary inside the step),
with record and valid-weight normalisation. The value either optimizer minimizes is the record
kernel's, so an end point carries exactly the objective that grading the same position gives. The
Gauss-Legendre profile of the plane-wave model is cached; the arrival times equal those with the
nodes computed per call, bit for bit, for the exponential and the layered (greenland_3exp_layered)
profile.
"""

import numpy as np
import pytest

from conftest import STATION, rng_sources
from synthetic import VPOL_CHANNELS, cylindrical_to_enu, make_event
from test_far_field import ICE, plane_wave_event
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative

from NuRadioReco.modules.interferometricDirectionReconstruction3D import (
    _C_M_PER_NS, _FAR_QUADRATURE_NODES, _lbfgsb_fd_steps, _plane_wave_profile, far_field_profile, plane_wave_times)

H = 1e-6
AGREE = 1e-6
REL_TOL = 1e-4


def central_check(f, x, grad, h=H):
    """Relative errors of grad against central differences, for the coordinates without a cell boundary in the step.

    Returns:
        List of |grad - central| / (|grad| + 1e-6) over the coordinates whose forward and backward
        differences agree to AGREE relative.
    """
    errs = []
    f0 = f(x)
    for i in range(len(x)):
        xp, xm = np.array(x, float), np.array(x, float)
        xp[i] += h
        xm[i] -= h
        fwd, bwd = (f(xp) - f0) / h, (f0 - f(xm)) / h
        if abs(fwd - bwd) < AGREE * (abs(fwd) + 1e-3):
            errs.append(abs(grad[i] - 0.5 * (fwd + bwd)) / (abs(grad[i]) + 1e-6))
    return errs


@pytest.fixture(scope='module')
def event_caches(reco, det, tables, pa):
    """Optimizer caches (raw correlation, SNR weights) of three synthetic VPol events and their sources."""
    out = []
    for i, src in enumerate(rng_sources(3, 20261003)):
        _, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables,
                               snr=20.0, seed=31 + i)
        times = [stn.get_channel(ch).get_times() for ch in VPOL_CHANNELS]
        volts = [stn.get_channel(ch).get_trace() for ch in VPOL_CHANNELS]
        _, packed = reco._prepare_corr_funcs(times, volts, None, True, 'energy')
        weights, _ = reco._compute_snr_pair_weights(volts, VPOL_CHANNELS)
        out.append((src, reco._build_optimizer_cache(VPOL_CHANNELS, weights, packed=packed)))
    return out


@pytest.mark.slow
@pytest.mark.parametrize('valid_norm', [False, True])
def test_point_source_gradient_matches_finite_differences(reco, event_caches, valid_norm):
    """The kernel gradient equals central differences to 1e-4 relative near and far from the source."""
    from NuRadioReco.utilities.reco3d_kernels import _scalar_singleray_corr_grad_numba, _scalar_singleray_corr_numba
    rng = np.random.default_rng(7)
    saved = reco._valid_norm
    reco._valid_norm = valid_norm
    try:
        errs, value_errs = [], []
        for src, cache in event_caches:
            args = reco._gradient_args(cache)
            near = np.array(src) + rng.normal(0.0, [3.0, 1.0, 3.0], (150, 3))
            far = np.column_stack([rng.uniform(2, 240, 150), rng.uniform(0, 360, 150), rng.uniform(-99, -1, 150)])
            for x in np.vstack([near, far]):
                value, *grad = _scalar_singleray_corr_grad_numba(*x, *args)
                record = _scalar_singleray_corr_numba(x[0], x[1] * (np.pi / 180.0), x[2], *args)
                value_errs.append(abs(value - record))
                errs += central_check(lambda p: _scalar_singleray_corr_numba(
                    p[0], p[1] * (np.pi / 180.0), p[2], *args), x, grad)
    finally:
        reco._valid_norm = saved
    assert max(value_errs) < 1e-12
    assert len(errs) > 1000
    assert np.max(errs) < REL_TOL


@pytest.mark.slow
@pytest.mark.parametrize('gradient', ['finite_difference', 'exact'])
def test_optimizer_returns_the_record_objective(reco, event_caches, gradient):
    """Every L-BFGS-B end point carries minus the record objective at that position and does not lose to its seed.

    The optimizer starts from the seed's azimuth shifted to 180 deg and back, which can move it by float
    rounding, hence the 1e-12 allowance against the seed's value.
    """
    bounds = [(1.0, 250.0), (0.0, 360.0), (-100.0, 0.0)]
    rng = np.random.default_rng(11)
    for src, cache in event_caches:
        for seed in np.array(src) + rng.normal(0.0, [5.0, 2.0, 5.0], (5, 3)):
            seed = (float(np.clip(seed[0], 1, 250)), float(seed[1] % 360.0), float(np.clip(seed[2], -100, 0)))
            rho, phi, z, corr = reco._optimize_from_seed(seed, None, VPOL_CHANNELS, bounds, _cache=cache,
                                                         gradient=gradient)
            at_end = -reco._correlation_at_point([rho, phi, z], None, VPOL_CHANNELS, _cache=cache)
            at_seed = -reco._correlation_at_point(list(seed), None, VPOL_CHANNELS, _cache=cache)
            assert corr == at_end
            assert corr >= at_seed - 1e-12


@pytest.mark.slow
def test_compiled_differences_reproduce_scipy_lbfgsb(reco, event_caches):
    """The default optimizer path ends where scipy's L-BFGS-B differencing the Python objective ends, bit for bit.

    Seeds include points on and next to the bounds (rho 1 m, z 0 m, z -100 m), where the steps turn
    one-sided; the compiled gradient also equals approx_derivative at every seed.
    """
    from NuRadioReco.utilities.reco3d_kernels import _lbfgsb_singleray_value_grad
    bounds = [(1.0, 250.0), (0.0, 360.0), (-100.0, 0.0)]
    rng = np.random.default_rng(5)
    for src, cache in event_caches:
        seeds = [tuple(np.array(src) + rng.normal(0.0, [5.0, 2.0, 5.0])) for _ in range(4)]
        seeds += [(1.0, src[1], src[2]), (src[0], src[1], 0.0), (src[0], src[1], -100.0),
                  (1.0 + 5e-9, src[1], -5e-9)]
        for seed in seeds:
            seed = (float(np.clip(seed[0], 1, 250)), float(seed[1] % 360.0), float(np.clip(seed[2], -100, 0)))
            phi_shift = seed[1] - 180.0

            def objective(params):
                return reco._correlation_at_point([params[0], (params[1] + phi_shift) % 360.0, params[2]],
                                                  None, VPOL_CHANNELS, _cache=cache)

            shifted = [bounds[0], (0.0, 360.0), bounds[2]]
            x0 = np.array([seed[0], 180.0, seed[2]])
            ref = minimize(objective, x0, method='L-BFGS-B', bounds=shifted,
                           options={'maxiter': 30, 'ftol': 1e-10})
            got = reco._optimize_from_seed(seed, None, VPOL_CHANNELS, bounds, _cache=cache)
            assert got == (ref.x[0], (ref.x[1] + phi_shift) % 360.0, ref.x[2], -ref.fun)
            lb, ub = np.array([b[0] for b in shifted]), np.array([b[1] for b in shifted])
            f, g = _lbfgsb_singleray_value_grad(x0, phi_shift, lb, ub, *reco._gradient_args(cache))
            assert f == objective(x0)
            assert np.array_equal(g, approx_derivative(objective, x0, method='2-point', abs_step=1e-8,
                                                       bounds=(lb, ub), f0=f))


def test_difference_steps_match_scipy():
    """_lbfgsb_fd_steps gives the steps approx_derivative takes, inside, on and next to the bounds."""
    lb, ub = np.array([0.0, -3.0, 1.0]), np.array([90.0, 3.0, 1e9])
    points = [np.array([45.0, 0.0, 5.0]), np.array([90.0, 3.0, 1.0]), np.array([0.0, -3.0, 1e9]),
              np.array([90.0 - 5e-9, -3.0 + 5e-9, 1e9 - 1.0]), np.array([1e-300, 2.9999999999, 3e8])]
    for x in points:
        steps = []
        approx_derivative(lambda p: (steps.append(p.copy()), float(np.sum(p)))[1], x, method='2-point',
                          abs_step=1e-8, bounds=(lb, ub), f0=float(np.sum(x)))
        expected = np.array([p[i] - x[i] for i, p in enumerate(steps)])
        h = _lbfgsb_fd_steps(x, lb, ub)
        assert np.array_equal(x + h, np.array([p[i] for i, p in enumerate(steps)]))
        assert np.array_equal((x + h) - x, expected)


def test_batched_plane_wave_times_equal_single_calls():
    """Rows of one plane_wave_times call over several directions equal one call per direction, bit for bit."""
    ant = np.array([[0.0, 0.0, -95.0], [10.0, -5.0, -40.0], [3.0, 7.0, -0.5], [1.0, 2.0, 3.0]])
    rng = np.random.default_rng(2)
    for _ in range(200):
        x = np.array([rng.uniform(0, 90), rng.uniform(-10, 370)])
        points = np.repeat(x[None, :], 3, axis=0) + np.vstack([np.zeros(2), np.diag([1e-8, -1e-8])])
        batch = plane_wave_times(np.radians(points[:, 0]), np.radians(points[:, 1]), ant, ICE)
        for p, row in zip(points, batch):
            assert np.array_equal(row, plane_wave_times(np.radians(p[0]), np.radians(p[1]), ant, ICE))


@pytest.mark.slow
@pytest.mark.parametrize('valid_norm', [False, True])
def test_far_field_gradient_matches_finite_differences(reco, det, valid_norm):
    """The plane-wave gradient equals central differences of the far-field objective to 1e-4 relative.

    The step is 1e-5 deg: at 1e-7 deg the rounding of the objective (about 1e-12) over the step
    exceeds 1e-4 of the smallest gradients (about 2e-5 per deg); at 1e-5 deg the central
    difference matches them to about 4e-6 relative.
    """
    from NuRadioReco.utilities.reco3d_kernels import _pairs_corr_numba, _plane_wave_corr_grad_numba
    evt, stn = plane_wave_event(det, 40.0, 130.0, seed=5)
    times = [stn.get_channel(ch).get_times() for ch in VPOL_CHANNELS]
    volts = [stn.get_channel(ch).get_trace() for ch in VPOL_CHANNELS]
    _, packed = reco._prepare_corr_funcs(times, volts, None, True, 'energy')
    weights, _ = reco._compute_snr_pair_weights(volts, VPOL_CHANNELS)
    saved = reco._valid_norm
    reco._valid_norm = valid_norm
    try:
        args = reco._singleray_corr_args(VPOL_CHANNELS, [packed], weights)
    finally:
        reco._valid_norm = saved
    ant, ice = reco._far_field_geometry(VPOL_CHANNELS)
    gl_weights, n2 = _plane_wave_profile(np.minimum(ant[:, 2], 0.0).tobytes(), tuple(float(v) for v in ice))
    ones = np.ones(len(VPOL_CHANNELS), dtype=np.bool_)

    def value(d):
        tts = plane_wave_times(np.radians(d[0]), np.radians(d[1]), ant, ice)
        return _pairs_corr_numba(tts, ones, args[0], 0, *args[1:])

    rng = np.random.default_rng(3)
    near = np.array([40.0, 130.0]) + rng.normal(0.0, 1.0, (200, 2))
    far = np.column_stack([rng.uniform(1, 89, 200), rng.uniform(0, 360, 200)])
    errs = []
    for d in np.vstack([near, far]):
        tts = plane_wave_times(np.radians(d[0]), np.radians(d[1]), ant, ice)
        grad = _plane_wave_corr_grad_numba(tts, float(np.radians(d[0])), float(np.radians(d[1])), ant, n2,
                                           gl_weights, _C_M_PER_NS, args[0], 0, *args[1:])
        errs += central_check(value, d, grad, h=1e-5)
    assert len(errs) > 200
    assert np.max(errs) < REL_TOL


def test_quadrature_nodes_computed_once_give_the_same_times():
    """Plane-wave times with the cached Gauss-Legendre profile equal those with nodes computed per call, bit for bit."""
    nodes, weights = np.polynomial.legendre.leggauss(_FAR_QUADRATURE_NODES)
    ant = np.array([[0.0, 0.0, -95.0], [10.0, -5.0, -40.0], [3.0, 7.0, -0.5], [1.0, 2.0, 3.0]])
    zen, az = np.meshgrid(np.radians([0.0, 5.0, 45.0, 89.5]), np.radians([0.0, 130.0, 300.0]), indexing='ij')
    times = plane_wave_times(zen, az, ant, ICE)
    sin2 = np.sin(zen[:, 0])[:, None, None] ** 2
    depth = np.minimum(ant[:, 2], 0.0)
    n = ICE[0] - ICE[1] * np.exp(0.5 * depth[:, None] * (1.0 - nodes[None, :]) / ICE[2])
    vertical = (0.5 * -depth[None, :] * np.sum(weights * np.sqrt(n[None] ** 2 - sin2), axis=-1)
                - np.maximum(ant[:, 2], 0.0)[None, :] * np.cos(zen[:, 0])[:, None]) / _C_M_PER_NS
    horizontal = -np.sin(zen)[..., None] * (np.cos(az)[..., None] * ant[:, 0] + np.sin(az)[..., None] * ant[:, 1])
    assert np.array_equal(times, horizontal / _C_M_PER_NS + vertical[:, None, :])


def test_cached_layered_profile_gives_the_same_times():
    """Plane-wave times of a layered profile with the cached layers equal the layer-by-layer quadrature per call, bit for bit."""
    from NuRadioMC.utilities import medium
    profile = far_field_profile(medium.greenland_3exp_layered())
    nodes, weights = np.polynomial.legendre.leggauss(_FAR_QUADRATURE_NODES)
    ant = np.array([[0.0, 0.0, -95.0], [10.0, -5.0, -40.0], [3.0, 7.0, -0.5], [1.0, 2.0, 3.0]])
    zen, az = np.meshgrid(np.radians([0.0, 5.0, 45.0, 89.5]), np.radians([0.0, 130.0, 300.0]), indexing='ij')
    times = plane_wave_times(zen, az, ant, profile)
    sin2 = np.sin(zen[:, 0])[:, None, None] ** 2
    depth = np.minimum(ant[:, 2], 0.0)
    in_ice = 0.0
    for z_min, z_max, n_ice, delta_n, z_0 in profile:
        lower = np.maximum(depth, z_min)
        length = np.maximum(min(z_max, 0.0) - lower, 0.0)
        n = n_ice - delta_n * np.exp((lower[:, None] + 0.5 * length[:, None] * (1.0 + nodes[None, :])) / z_0)
        in_ice = in_ice + 0.5 * length[None, :] * np.sum(weights * np.sqrt(np.maximum(n[None] ** 2 - sin2, 0.0)), axis=-1)
    vertical = (in_ice - np.maximum(ant[:, 2], 0.0)[None, :] * np.cos(zen[:, 0])[:, None]) / _C_M_PER_NS
    horizontal = -np.sin(zen)[..., None] * (np.cos(az)[..., None] * ant[:, 0] + np.sin(az)[..., None] * ant[:, 1])
    assert np.array_equal(times, horizontal / _C_M_PER_NS + vertical[:, None, :])
