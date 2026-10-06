"""Coverage factor of the valid-weight normalisation against exact rational arithmetic.

``_coverage_ramp`` decides the knees of the factor exactly and evaluates the ramp between them within
one ulp of the exact value; the points are packed ulp by ulp around both knees, where the ramp's
numerator cancels (half the floor) or the factor meets 1 (the floor). ``_valid_mean`` is the IEEE
expression acc / w_valid * f. The GPU backend's copy of the ramp must give the CPU values bit for bit.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from NuRadioReco.utilities import reco3d_kernels as kernels

pytestmark = pytest.mark.skipif(not kernels.USE_NUMBA, reason="needs numba")


def knee_points(rng, n_totals=30, n_ulps=48):
    """(w_valid, w_total, valid_floor) triples packed around both knees of the ramp, plus random ramp points.

    For each floor and total weight the valid weight steps ulp by ulp through fl(floor * w_total) / 2 and
    fl(floor * w_total), the knee ties included, then sits at relative offsets of 1e-15 to 1e-3 on both
    sides of each knee.
    """
    floors = [0.6, 0.5, 0.3, 0.75, 1.0, 0.1] + list(rng.uniform(0.05, 1.0, 4))
    points = []
    for floor in floors:
        for wt in np.concatenate([[1.0, 2.0, 3.0, 66.0], rng.uniform(0.1, 100.0, n_totals - 4)]):
            p = floor * wt
            for knee in (0.5 * p, p):
                w = knee
                for _ in range(n_ulps):
                    w = np.nextafter(w, 0.0)
                for _ in range(2 * n_ulps + 1):
                    points.append((float(w), float(wt), floor))
                    w = np.nextafter(w, np.inf)
                for j in range(3, 16):
                    points += [(knee * (1.0 - 10.0 ** -j), float(wt), floor),
                               (knee * (1.0 + 10.0 ** -j), float(wt), floor)]
            points += [(float(w), float(wt), floor) for w in rng.uniform(0.5 * p, p, 20)]
    return points


def exact_ramp(w_valid, w_total, valid_floor):
    """Exact coverage factor of the float inputs, as a Fraction."""
    if w_valid <= 0.0 or w_total <= 0.0:
        return Fraction(0)
    wv = Fraction(w_valid)
    total = Fraction(valid_floor) * Fraction(w_total)
    if wv >= total:
        return Fraction(1)
    if 2 * wv <= total:
        return Fraction(0)
    return (2 * wv - total) / total


def test_coverage_ramp_within_one_ulp_of_exact():
    """The factor is exact at and beyond the knees and within one ulp of the exact ramp between them."""
    rng = np.random.default_rng(11)
    n_ramp = 0
    for wv, wt, floor in knee_points(rng):
        f = kernels._coverage_ramp(wv, wt, floor)
        exact = exact_ramp(wv, wt, floor)
        if exact in (0, 1):
            assert f == exact, (wv, wt, floor, f)
            continue
        n_ramp += 1
        assert abs(Fraction(f) - exact) <= Fraction(math.ulp(float(exact))), (wv, wt, floor, f, float(exact))
    assert n_ramp > 20000
    for wv, wt in ((0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)):
        assert kernels._coverage_ramp(wv, wt, 0.6) == 0.0


def test_valid_mean_is_the_ieee_expression():
    """acc / w_valid * f with IEEE rounding, and 0 when the factor is 0."""
    rng = np.random.default_rng(12)
    for acc, wv, f in zip(rng.normal(size=2000), rng.uniform(0.1, 50.0, 2000), rng.uniform(0.0, 1.0, 2000)):
        assert kernels._valid_mean(acc, wv, f) == acc / wv * f
    assert kernels._valid_mean(1.5, 0.0, 0.0) == 0.0


@pytest.mark.skipif(not kernels.USE_CUPY, reason="needs a CUDA device")
def test_gpu_coverage_ramp_equals_cpu():
    """The GPU backend's factor and normalised value equal the CPU kernels' bit for bit at the knee-packed points."""
    import cupy as cp
    from NuRadioReco.modules import reco3d_batch_gpu

    kernel = cp.RawModule(code=reco3d_batch_gpu._SRC + r'''
extern "C" __global__ void ramp_points(const double* acc, const double* wv, const double* wt, const double* vfloor,
                                       long long n, double* f, double* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    f[i] = coverage_ramp(wv[i], wt[i], vfloor[i]);
    out[i] = normalised(acc[i], wv[i], wt[i], 1, vfloor[i]);
}
''').get_function('ramp_points')
    rng = np.random.default_rng(13)
    wv, wt, floor = (np.ascontiguousarray(c) for c in np.array(knee_points(rng)).T)
    acc = rng.normal(size=wv.size) * wv
    n = wv.size
    f_gpu = cp.empty(n)
    out_gpu = cp.empty(n)
    kernel(((n + 255) // 256,), (256,), (cp.asarray(acc), cp.asarray(wv), cp.asarray(wt), cp.asarray(floor),
                                         np.int64(n), f_gpu, out_gpu))
    f_cpu = np.array([kernels._coverage_ramp(a, b, c) for a, b, c in zip(wv, wt, floor)])
    out_cpu = np.array([kernels._valid_mean(x, a, f) for x, a, f in zip(acc, wv, f_cpu)])
    assert cp.asnumpy(f_gpu).tobytes() == f_cpu.tobytes()
    assert cp.asnumpy(out_gpu).tobytes() == out_cpu.tobytes()
