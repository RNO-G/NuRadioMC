"""The lean L-BFGS-B driver equals scipy's minimize(jac=True, method='L-BFGS-B') bit for bit (synthetic functions)."""

import numpy as np
import pytest
from scipy.optimize import minimize

from NuRadioReco.modules.interferometricDirectionReconstruction3D import _minimize_lbfgsb


def _rosenbrock(x, scale):
    """Scaled Rosenbrock value and gradient."""
    a, b = x[0], x[1]
    f = scale * ((1 - a) ** 2 + 100 * (b - a * a) ** 2)
    g = scale * np.array([-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)])
    return f, g


def _wavy(x):
    """Bumpy 3D objective with many local minima, value and forward differences (as the reco objectives)."""
    def value(p):
        return float(-np.cos(0.3 * p[0]) * np.cos(0.05 * p[1]) - 0.2 * np.sin(0.7 * p[2]) + 1e-4 * np.sum(p ** 2))
    f0 = value(x)
    g = np.empty(3)
    for i in range(3):
        h = 1e-8 * max(1.0, abs(x[i]))
        p = x.copy()
        p[i] += h
        g[i] = (value(p) - f0) / h
    return f0, g


def _scipy(fun, x0, bounds, maxiter, ftol, args=()):
    """scipy's result of the same problem."""
    return minimize(fun, x0, args=args, jac=True, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': ftol})


@pytest.mark.parametrize('maxiter', [3, 30, 200])
@pytest.mark.parametrize('x0, bounds', [
    ([-1.2, 1.0], [(-2.0, 2.0), (-1.0, 3.0)]),
    ([0.5, 0.5], [(0.6, 2.0), (-1.0, 0.3)]),
    ([3.0, -2.0], [(None, 2.0), (-1.0, None)]),
])
def test_rosenbrock_equals_scipy(x0, bounds, maxiter):
    """Iterates, final point and value equal scipy's with active and inactive bounds and an iteration cap."""
    ref = _scipy(_rosenbrock, x0, bounds, maxiter, 1e-10, args=(3.0,))
    out = _minimize_lbfgsb(_rosenbrock, x0, bounds, maxiter, 1e-10, args=(3.0,))
    assert out.x.tobytes() == ref.x.tobytes()
    assert np.float64(out.fun).tobytes() == np.float64(ref.fun).tobytes()
    assert out.nit == ref.nit
    assert out.nfev == ref.nfev


@pytest.mark.parametrize('seed', range(8))
def test_bumpy_objective_equals_scipy(seed):
    """Random starts on a bumpy objective with forward-difference gradients (the shape of the reco objectives)."""
    rng = np.random.default_rng(seed)
    x0 = np.array([rng.uniform(1, 1600), 180.0, rng.uniform(-1500, 300)])
    bounds = [(1.0, 1600.0), (0.0, 360.0), (-1500.0, 300.0)]
    ref = _scipy(_wavy, x0, bounds, 30, 1e-10)
    out = _minimize_lbfgsb(_wavy, x0, bounds, 30, 1e-10)
    assert out.x.tobytes() == ref.x.tobytes()
    assert np.float64(out.fun).tobytes() == np.float64(ref.fun).tobytes()
