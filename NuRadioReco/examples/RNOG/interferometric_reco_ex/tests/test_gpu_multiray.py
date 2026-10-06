"""The GPU multi-ray maps equal the numba per-pair and grouped maps to rounding (synthetic series; travel times synthetic or from the test tables)."""

import itertools

import numpy as np
import pytest

from fast_grouped_multiray import grouped_multiray_numba, perpair_multiray_numba


def _case(seed, n_ch=6, shape=(7, 9, 11), ray_types=('direct', 'refracted', 'reflected')):
    """Random multi-ray travel times (NaN holes, channel 2 without the last ray type) and pair series."""
    rng = np.random.default_rng(seed)
    tt_all = {}
    for ch in range(n_ch):
        tt_all[ch] = {}
        for k, rt in enumerate(ray_types):
            if rt == ray_types[-1] and ch == 2:
                continue
            grid = 500.0 + 40.0 * rng.random(shape) + 15.0 * k
            grid[rng.random(shape) < 0.1] = np.nan
            tt_all[ch][rt] = grid
    corr_data = []
    for _ in itertools.combinations(range(n_ch), 2):
        n = int(rng.integers(300, 500))
        corr_data.append((rng.normal(0, 0.3, n), 0.1 + 0.05 * rng.random(), -20.0 - 10 * rng.random()))
    weights = list(rng.uniform(0.1, 2.0, len(corr_data)))
    return tt_all, corr_data, weights


@pytest.fixture(scope='module')
def backend():
    """GpuMultiray on the CUDA device (skips without one)."""
    cp = pytest.importorskip('cupy')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as err:
        pytest.skip(f'no CUDA device: {err}')
    from NuRadioReco.modules.reco3d_batch_gpu import GpuMultiray
    return GpuMultiray()


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('weighted', [False, True])
def test_perpair_matches_numba(backend, seed, weighted):
    """Per-pair map: equal to the numba kernel to 1e-12, twice (second call from the device cache)."""
    tt_all, corr_data, weights = _case(seed)
    channels = list(tt_all)
    w = weights if weighted else None
    ref, ref_max = perpair_multiray_numba(corr_data, tt_all, channels, pair_weights=w)
    for _ in range(2):
        out, out_max = backend.perpair(corr_data, tt_all, channels, pair_weights=w)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)
        assert abs(out_max - ref_max) <= 1e-12


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('ray_types', [('direct', 'refracted', 'reflected'), ('solution_0', 'solution_1')])
@pytest.mark.parametrize('ch_to_group', [{0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}, {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4}])
def test_grouped_matches_numba(backend, seed, ray_types, ch_to_group):
    """Grouped map (group-block kernel, two or three ray types): equal to the numba kernel to 1e-12."""
    tt_all, corr_data, weights = _case(seed, ray_types=ray_types)
    channels = list(tt_all)
    n_groups = max(ch_to_group.values()) + 1
    ref, ref_max = grouped_multiray_numba(corr_data, tt_all, channels, ch_to_group, n_groups, pair_weights=weights)
    out, out_max = backend.grouped(corr_data, tt_all, channels, ch_to_group, n_groups, pair_weights=weights)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)
    assert abs(out_max - ref_max) <= 1e-12


@pytest.mark.slow
def test_grid_lookup_matches_cpu_lookup(backend, det, table_dir):
    """Grouped and per-pair maps with the travel times looked up on the device equal those of the CPU lookup to rounding.

    Grids inside the tables, grids reaching past their edges (rho to 1600 m, z above the surface) and a grid
    outside every table, for which both return the (1,) map of no usable travel time.
    """
    from conftest import STATION, reference_config
    from synthetic import VPOL_CHANNELS
    from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
    cfg = reference_config(table_dir, multi_ray_types=True, table_scheme='solution_ordered',
                           multiray_combo_mode='grouped')
    reco = InterferometricReco3D()
    reco.begin(STATION, cfg, det)
    channels = VPOL_CHANNELS
    rng = np.random.default_rng(11)
    corr_data = [(np.convolve(rng.normal(0, 0.3, 4001), np.ones(25) / 25, mode='same'), 0.1, -200.0)
                 for _ in itertools.combinations(channels, 2)]
    weights = list(rng.uniform(0.2, 1.5, len(corr_data)))
    grids = [(np.linspace(150, 170, 9), np.radians(np.linspace(30, 40, 11)), np.linspace(-120, -100, 13)),
             (np.linspace(1, 1600, 23), np.radians(np.linspace(0, 358, 37)), np.linspace(-1500, 300, 41)),
             (np.linspace(1400, 1600, 7), np.radians(np.linspace(200, 230, 5)), np.linspace(-200, 300, 17)),
             (np.linspace(1400, 1600, 7), np.radians(np.linspace(200, 230, 5)), np.linspace(-10, 300, 17))]
    for (rho, phi, z), perpair in itertools.product(grids, (False, True)):
        reco.multiray_backend = None
        ref, ref_max = reco._multiray_grid(rho, phi, z, corr_data, channels, weights, force_perpair=perpair)
        reco.multiray_backend = backend
        out, out_max = reco._multiray_grid(rho, phi, z, corr_data, channels, weights, force_perpair=perpair)
        assert out.shape == ref.shape
        if ref.shape == (1,):
            assert np.isnan(out_max) and np.isnan(ref_max)
            continue
        assert ref.shape == (len(rho), len(phi), len(z))
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-9)
        assert abs(out_max - ref_max) <= 1e-9
