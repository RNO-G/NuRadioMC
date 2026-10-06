"""Compiled paths against the implementations they replace, bit for bit, on synthetic inputs.

The coarse maps on the channel-major travel-time stack and the refine and polish grids from
channel-major travel times must equal the point-major kernels (one sum per point in pair order);
the numba peak extraction and map SNR must equal their numpy code, including NaN, -inf and tied
maps; the numpy summation order of np.std is reproduced exactly; the batched pair-series transforms
must equal the per-pair loop; the point-major grouped multiray map must equal the pair-major
grouped kernel for 1 to 276 pairs, missing ray types, NaN and -inf travel times and delays outside
the correlations.
"""

import itertools
import warnings

import numpy as np
import pytest
from scipy import fft as sp_fft
from scipy.signal import hilbert, windows

import NuRadioReco.modules.interferometricDirectionReconstruction3D as module
from NuRadioReco.utilities import reco3d_kernels as kernels

pytestmark = pytest.mark.skipif(not kernels.USE_NUMBA, reason="needs numba")


def random_maps(rng, n):
    """Random 3D maps with NaN, -inf, ties and non-contiguous views."""
    maps = []
    for trial in range(n):
        shape = tuple(int(v) for v in rng.integers(1, (30, 90, 60)))
        m = rng.standard_normal(shape) * 0.1
        if trial % 3 == 0:
            m[rng.random(shape) < 0.3] = np.nan
        if trial % 5 == 0:
            m = np.round(m, 1)
        if trial % 11 == 0:
            m[...] = np.nan
        if trial % 13 == 0:
            m[rng.random(shape) < 0.5] = -np.inf
        if trial % 4 == 1:
            m = m[:, :, ::2]
        maps.append(m)
    return maps


def test_peaks_and_map_snr_match_numpy(monkeypatch):
    """_extract_top_n_peaks and _compute_map_snr give the numpy results and types exactly."""
    rng = np.random.default_rng(3)
    reco = module.InterferometricReco3D()
    for m in random_maps(rng, 120):
        rho = np.geomspace(1, 250, m.shape[0])
        phi = np.arange(m.shape[1]) * 3.0
        z = np.linspace(-100, 0, m.shape[2])
        sep = [int(rng.choice([10, 50])), int(rng.choice([5, 15])), int(rng.choice([10, 50]))]
        peaks = [(i % m.shape[0], i % m.shape[1], i % m.shape[2]) for i in range(3)]
        monkeypatch.setattr(module, 'USE_NUMBA', True)
        fast = reco._extract_top_n_peaks(m, rho, phi, z, 7, sep)
        fast_snr = [reco._compute_map_snr(m, p) for p in peaks]
        monkeypatch.setattr(module, 'USE_NUMBA', False)
        ref = reco._extract_top_n_peaks(m, rho, phi, z, 7, sep)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref_snr = [reco._compute_map_snr(m, p) for p in peaks]
        assert fast == ref
        assert [type(v) for p in fast for v in p] == [type(v) for p in ref for v in p]
        assert np.array_equal(fast_snr, ref_snr, equal_nan=True)


@pytest.mark.parametrize('n', [1, 7, 8, 9, 127, 128, 129, 136, 1000, 8192, 100_003])
def test_numpy_std_order(n):
    """_numpy_std equals np.std (pairwise summation of numpy) bit for bit."""
    rng = np.random.default_rng(n)
    for _ in range(5):
        a = rng.standard_normal(n) * rng.uniform(0.01, 100) + rng.uniform(-5, 5)
        assert kernels._numpy_std(a) == np.std(a)
        assert 0.0 + kernels._pairwise_sum(a, 0, n) == np.sum(a)


def synthetic_geometry(rng, n_ch, n_tables=None):
    """Random channel geometry and table stack with NaN cells, in the kernels' argument order."""
    n_tables = n_ch if n_tables is None else n_tables
    nr, nz = 260, 140
    values = 50.0 + rng.uniform(0, 2000, (n_tables, nr, nz)).cumsum(axis=1) * 0.01
    values[:, :, :3] = np.nan
    values[rng.random(values.shape) < 0.002] = np.nan
    ok = np.isfinite(values)
    slot = rng.permutation(n_tables)[:n_ch].astype(np.int64)
    return (5.0, -3.0, rng.uniform(-30, 30, (n_ch, 2)), values, ok, slot,
            np.full(n_ch, 1.0), np.full(n_ch, 1.0), np.full(n_ch, nr, dtype=np.int64),
            np.full(n_ch, -120.0), np.full(n_ch, 1.0), np.full(n_ch, nz, dtype=np.int64))


def synthetic_corr(rng, n_ch, k):
    """K correlation sets on one lag geometry, weights and normalisation, in the kernels' order."""
    pair_ch1, pair_ch2 = (np.array(v, dtype=np.int64) for v in zip(*itertools.combinations(range(n_ch), 2)))
    n_pairs = len(pair_ch1)
    m = 3001
    corr = rng.standard_normal((k, n_pairs, m))
    lengths = np.full(n_pairs, m, dtype=np.int64)
    lengths[::3] -= 7
    offsets = -150.0 + rng.uniform(-20, 20, n_pairs)
    dts = np.full(n_pairs, 0.1)
    weights = rng.uniform(0.2, 3.0, n_pairs)
    return (corr, lengths, 1.0 / dts, offsets, pair_ch1, pair_ch2, weights, float(weights.sum()))


@pytest.mark.parametrize('k', [1, 3])
@pytest.mark.parametrize('valid_norm', [False, True])
def test_channel_major_maps_match_point_kernels(k, valid_norm):
    """The block kernel on channel-major travel times equals the point-major stack and grid kernels."""
    rng = np.random.default_rng(10 * k + valid_norm)
    n_ch = 9
    geom = synthetic_geometry(rng, n_ch)
    corr = synthetic_corr(rng, n_ch, k) + (valid_norm, 0.6)
    rho = np.geomspace(1.0, 220.0, 17)
    phi = np.radians(np.arange(0.0, 360.0, 7.5))
    z = np.linspace(-110.0, 15.0, 41)
    for tolerant in (False, True):
        grid = kernels._singleray_grid_numba(rho, phi, z, *geom, tolerant, *corr)
        ttsT, validT = kernels._singleray_grid_tts_numba(rho, phi, z, *geom, tolerant)
        assert validT.any() and not validT.all()
        for block in (1, 37, 256, 4096, ttsT.shape[1]):
            got = kernels._singleray_stackT_corr_numba(ttsT, validT, block, *corr)
            assert np.array_equal(got, grid, equal_nan=True)
        point = kernels._singleray_stack_corr_numba(np.ascontiguousarray(ttsT.T), np.ascontiguousarray(validT.T), *corr)
        assert np.array_equal(point, grid, equal_nan=True)


def reference_pair_series(times, volt_arrays, modes, apply_hann_window, norm_mode):
    """Per-pair transform loop of _pair_series before the batching, kept as the reference."""
    n_ch = len(volt_arrays)
    channel_pairs = list(itertools.combinations(range(n_ch), 2))
    n = len(volt_arrays[0])
    m = 2 * n - 1
    nfft = sp_fft.next_fast_len(m, real=True)
    series = {}
    for envelope_traces, group in ((False, [x for x in (None, 'correlation') if x in modes]),
                                   (True, [x for x in ('traces',) if x in modes])):
        if not group:
            continue
        spec, spec_rev, energy = [], [], []
        for v in volt_arrays:
            if envelope_traces:
                v = np.abs(hilbert(v))
            vn = v - v.mean()
            if norm_mode == 'pearson':
                std = vn.std()
                if std > 0:
                    vn = vn / std
            energy.append(np.sum(vn ** 2))
            spec.append(sp_fft.rfft(vn, nfft))
            spec_rev.append(sp_fft.rfft(vn[::-1], nfft))
        out = {x: np.zeros((len(channel_pairs), m)) for x in group}
        for pidx, (c1, c2) in enumerate(channel_pairs):
            product = spec[c1] * spec_rev[c2]
            rows = []
            for x in group:
                corr = out[x][pidx]
                if x == 'correlation':
                    product[1:(nfft + 1) // 2] *= 2.0
                    corr[:] = np.abs(sp_fft.ifft(product, nfft)[:m])
                else:
                    corr[:] = sp_fft.irfft(product, nfft)[:m]
                rows.append(corr)
            for corr in rows:
                if norm_mode == 'energy':
                    energy_norm = np.sqrt(energy[c1] * energy[c2])
                    if energy_norm > 0:
                        corr /= energy_norm
                else:
                    corr /= np.concatenate([np.arange(1, n + 1), np.arange(n - 1, 0, -1)], dtype=np.float64)
                if apply_hann_window:
                    corr *= windows.hann(m)
        series.update(out)
    return series


@pytest.mark.parametrize('norm_mode', ['energy', 'pearson', 'overlap_only'])
def test_batched_pair_series_match_loop(norm_mode):
    """Batched transforms give the per-pair series bit for bit for every envelope mode."""
    rng = np.random.default_rng(len(norm_mode))
    for n_ch, n in ((11, 2048), (4, 999)):
        volts = [rng.standard_normal(n) * rng.uniform(0.1, 3) for _ in range(n_ch)]
        volts[2][:] = 0.0
        times = [np.arange(n) * 0.1 + rng.uniform(-50, 50) for _ in range(n_ch)]
        modes = (None, 'correlation', 'traces')
        for hann in (False, True):
            got = module.InterferometricReco3D._pair_series(times, volts, modes, hann, norm_mode)
            ref = reference_pair_series(times, volts, modes, hann, norm_mode)
            for x in modes:
                assert np.array_equal(got[x].corr, ref[x])


def random_grouped_case(rng, channels, shape):
    """Grouped multiray inputs: travel times with missing ray types, NaN and -inf, and pair correlations.

    Some delays fall outside the correlations, whose lengths, sample spacings and offsets vary.
    """
    tt_all = {}
    for ch in channels:
        tt_all[ch] = {}
        for rt in ('direct', 'refracted', 'reflected'):
            if rng.random() < 0.25:
                continue
            t = rng.uniform(200.0, 600.0, shape)
            t[rng.random(shape) < 0.05] = np.nan
            t[rng.random(shape) < 0.05] = -np.inf
            tt_all[ch][rt] = t
    corr_data = []
    for _ in itertools.combinations(channels, 2):
        m = int(rng.integers(50, 3000))
        dt = float(rng.uniform(0.1, 0.5))
        corr_data.append((rng.standard_normal(m), dt, -0.5 * m * dt * float(rng.uniform(0.2, 1.2))))
    return tt_all, corr_data


@pytest.mark.skipif(not kernels.USE_NUMBA_GROUPED, reason="needs fast_grouped_multiray")
@pytest.mark.parametrize('n_ch', list(range(2, 25)))
def test_grouped_points_kernel_matches_pairmajor_kernel(n_ch):
    """grouped_multiray_points equals grouped_multiray_numba bit for bit (map and maximum), up to five groups."""
    rng = np.random.default_rng(100 + n_ch)
    channels = sorted(int(c) for c in rng.choice(24, n_ch, replace=False))
    n_groups = min(n_ch, 5)
    ch_to_group = {ch: int(g) for ch, g in zip(channels, rng.permutation(np.arange(n_ch) % n_groups))}
    shape = (int(rng.integers(3, 9)), int(rng.integers(3, 9)), int(rng.integers(3, 20)))
    tt_all, corr_data = random_grouped_case(rng, channels, shape)
    weights = [None, rng.uniform(0.05, 3.0, len(corr_data))]
    for pw in weights:
        want, want_max = kernels.grouped_multiray_numba(corr_data, tt_all, channels, ch_to_group, n_groups, pw)
        got, got_max = kernels.grouped_multiray_points(corr_data, tt_all, channels, ch_to_group, n_groups, pw)
        assert got.shape == want.shape and got.tobytes() == want.tobytes()
        assert np.float64(got_max).tobytes() == np.float64(want_max).tobytes()


@pytest.mark.skipif(not kernels.USE_NUMBA_GROUPED, reason="needs fast_grouped_multiray")
def test_grouped_points_kernel_on_refine_sized_grid():
    """The same on a refine-sized grid of the selection configuration (11 channels, 6 groups) and without ray types."""
    rng = np.random.default_rng(7)
    channels = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
    ch_to_group, _ = module.InterferometricReco3D._build_channel_groups(channels)
    n_groups = max(ch_to_group.values()) + 1
    tt_all, corr_data = random_grouped_case(rng, channels, (25, 17, 25))
    pw = rng.uniform(0.05, 3.0, len(corr_data))
    want, _ = kernels.grouped_multiray_numba(corr_data, tt_all, channels, ch_to_group, n_groups, pw)
    got, _ = kernels.grouped_multiray_points(corr_data, tt_all, channels, ch_to_group, n_groups, pw)
    assert got.tobytes() == want.tobytes()
    empty = {ch: {} for ch in channels}
    want = kernels.grouped_multiray_numba(corr_data, empty, channels, ch_to_group, n_groups, pw)
    got = kernels.grouped_multiray_points(corr_data, empty, channels, ch_to_group, n_groups, pw)
    assert np.array_equal(got[0], want[0]) and np.isnan(got[1]) and np.isnan(want[1])
