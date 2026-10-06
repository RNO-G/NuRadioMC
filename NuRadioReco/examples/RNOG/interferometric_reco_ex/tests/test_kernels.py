"""Kernel group: packed correlation preparation, fused grid kernels, compass optimizer, valid-weight normalisation.

The correlation builder transforms every channel once and every pair once and hands
one packed set to every stage; it is checked pair by pair against the per-pair
scipy.signal.correlate builder it replaced. The correlation envelope now comes from
the padded pair spectrum instead of a Hilbert transform of the cropped correlation;
the two differ by the one-sample periodic extension and the difference is gated at
its measured level. The caches built from the tables are emptied when tables are loaded
again, and the coarse caches follow the channel order, so one instance that runs the
same channel set in another order gives a fresh instance's result. In multiray grouped
mode the result does not depend on use_fused_correlator: the fused multiray refine
kernel (per-pair best ray-type combination) serves the per_pair mode only, and the
point-major grouped kernel the flag selects equals the pair-major one end to end.
"""

import itertools

import numpy as np
import pytest
from scipy.signal import correlate, hilbert, windows

from conftest import STATION, reference_config, rng_sources
from synthetic import VPOL_CHANNELS, angular_separation, cylindrical_to_enu, make_event, make_noise_event, same_value
from test_reco_known_answer import SUMMARY_SOURCES
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

SNR = 20.0
PAIR_TOL = 1e-12
ENVELOPE_TOL = 1e-6


def reference_corr_funcs(times, volt_arrays, hilbert_envelope_mode=None,
                         apply_hann_window=False, correlation_normalization='normalized'):
    """Per-pair correlation builder of commit e6a0cfe4c, kept as the reference."""
    v_array_pairs = list(itertools.combinations(volt_arrays, 2))
    channel_pairs = list(itertools.combinations(range(len(times)), 2))
    dts = np.array([t[1] - t[0] if len(t) > 1 else 1.0 for t in times])
    len_trace = len(v_array_pairs[0][0])
    overlap_norm = np.concatenate([np.arange(1, len_trace + 1),
                                   np.arange(len_trace - 1, 0, -1)], dtype=np.float64)
    hilbert_cache = {}
    if hilbert_envelope_mode == 'traces':
        for pidx in range(len(v_array_pairs)):
            cidx1, cidx2 = channel_pairs[pidx]
            if cidx1 not in hilbert_cache:
                hilbert_cache[cidx1] = np.abs(hilbert(v_array_pairs[pidx][0]))
            if cidx2 not in hilbert_cache:
                hilbert_cache[cidx2] = np.abs(hilbert(v_array_pairs[pidx][1]))
    corr_data = []
    for pidx in range(len(v_array_pairs)):
        v1, v2 = v_array_pairs[pidx]
        cidx1, cidx2 = channel_pairs[pidx]
        t1, t2 = times[cidx1], times[cidx2]
        dt = min(dts[cidx1], dts[cidx2])
        if hilbert_envelope_mode == 'traces':
            v1 = hilbert_cache[cidx1]
            v2 = hilbert_cache[cidx2]
        norm_mode = correlation_normalization
        if norm_mode == 'normalized':
            norm_mode = 'pearson'
        v1n = v1 - v1.mean()
        v2n = v2 - v2.mean()
        if norm_mode == 'pearson':
            std1, std2 = v1n.std(), v2n.std()
            if std1 > 0 and std2 > 0:
                v1n = v1n / std1
                v2n = v2n / std2
        corr = correlate(v1n, v2n, mode='full', method='auto')
        if hilbert_envelope_mode == 'correlation':
            corr = np.abs(hilbert(corr))
        if norm_mode == 'energy':
            energy_norm = np.sqrt(np.sum(v1n**2) * np.sum(v2n**2))
            if energy_norm > 0:
                corr /= energy_norm
        else:
            corr /= overlap_norm
        if apply_hann_window:
            corr *= windows.hann(len(corr))
        M = len(corr)
        offset = -(M // 2) * dt + (t1[0] - t2[0])
        corr_data.append((corr.astype(np.float64), float(dt), float(offset)))
    return corr_data


def _traces(stn):
    """Time and voltage arrays of the VPol channels of a station."""
    times = [stn.get_channel(ch).get_times() for ch in VPOL_CHANNELS]
    volts = [stn.get_channel(ch).get_trace() for ch in VPOL_CHANNELS]
    return times, volts


@pytest.fixture(scope='module')
def events(det, tables, pa):
    """Three seeded synthetic VPol events at SNR 20."""
    out = []
    for i, src in enumerate(rng_sources(3, 20260930)):
        evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS,
                                 tables, snr=SNR, seed=11 + i)
        out.append((src, evt, stn))
    return out


@pytest.mark.slow
@pytest.mark.parametrize('mode,hann,norm', [
    (None, True, 'energy'), (None, False, 'pearson'), (None, True, 'overlap_only'),
    ('traces', True, 'energy'), ('traces', False, 'pearson')])
def test_packed_correlations_match_reference(reco, events, mode, hann, norm):
    """Every pair of the packed builder equals the per-pair scipy builder to 1e-12."""
    for _, _, stn in events:
        times, volts = _traces(stn)
        corr_data, packed = reco._prepare_corr_funcs(times, volts, mode, hann, norm)
        ref = reference_corr_funcs(times, volts, mode, hann, norm)
        assert len(corr_data) == len(ref) == packed.corr.shape[0]
        for pidx, ((c, dt, off), (rc, rdt, roff)) in enumerate(zip(corr_data, ref)):
            assert dt == rdt and off == roff
            assert c.shape == rc.shape
            assert np.max(np.abs(c - rc)) < PAIR_TOL
            assert np.shares_memory(c, packed.corr[pidx])
            assert packed.lengths[pidx] == len(rc)
        assert np.all(packed.inv_dts * packed.dts == 1.0)


@pytest.mark.slow
def test_correlation_envelope_from_pair_spectrum(reco, events):
    """The envelope from the padded pair spectrum agrees with the Hilbert envelope of the cropped correlation."""
    worst = 0.0
    for _, _, stn in events:
        times, volts = _traces(stn)
        corr_data, _ = reco._prepare_corr_funcs(times, volts, 'correlation', True, 'energy')
        ref = reference_corr_funcs(times, volts, 'correlation', True, 'energy')
        for (c, _, _), (rc, _, _) in zip(corr_data, ref):
            worst = max(worst, float(np.max(np.abs(c - rc))))
    assert worst < ENVELOPE_TOL, worst


MAP_TOL = 1e-9
COARSE_AXES = (np.geomspace(1.0, 250.0, 30), np.arange(0.0, 360.0, 3.0) * (np.pi / 180.0),
               np.linspace(-100.0, 0.0, 100))


def _refine_axes(src):
    """Refine-level axes of the reference configuration around a source (15 m, 3 deg, 15 m at 1 m, 0.3 deg, 1 m)."""
    rho, phi, z = src
    rho_vec = np.arange(max(rho - 15.0, 1.0), min(rho + 15.0, 250.0) + 1.0, 1.0)
    phi_vec = np.arange(phi - 3.0, phi + 3.0 + 0.3, 0.3) * (np.pi / 180.0)
    z_vec = np.arange(max(z - 15.0, -100.0), min(z + 15.0, 0.0) + 1.0, 1.0)
    return rho_vec, phi_vec, z_vec


def _reference_map(reco, corr_data, packed, axes, weights):
    """Map from the per-pair delay matrices and the all-pairs kernel of the record code."""
    src_enu = reco._build_source_enu_matrix(*axes)
    delays = reco._compute_delay_matrices(src_enu, VPOL_CHANNELS)
    return reco._correlator_lean(corr_data, delays, pair_weights=weights, packed=packed)[0]


def _weighted_corr(reco, stn, mode=None):
    """Packed correlations of an event and its SNR pair weights."""
    times, volts = _traces(stn)
    corr_data, packed = reco._prepare_corr_funcs(times, volts, mode, True, 'energy')
    weights, _ = reco._compute_snr_pair_weights(volts, VPOL_CHANNELS)
    return corr_data, packed, weights


@pytest.mark.slow
def test_fused_refine_maps_match_delay_matrix_path(reco, events):
    """The fused grid kernel reproduces the refine maps of the delay-matrix path to 1e-9."""
    assert reco._singleray_kernel_active()
    worst = 0.0
    for src, _, stn in events:
        corr_data, packed, weights = _weighted_corr(reco, stn)
        axes = _refine_axes(src)
        fused = reco._singleray_grid_maps(*axes, VPOL_CHANNELS, [packed], weights)[0]
        ref = _reference_map(reco, corr_data, packed, axes, weights)
        assert fused.shape == ref.shape
        worst = max(worst, float(np.max(np.abs(fused - ref))))
        assert np.unravel_index(np.argmax(fused), fused.shape) == np.unravel_index(np.argmax(ref), ref.shape)
    assert worst < MAP_TOL, worst


@pytest.mark.slow
def test_coarse_stack_map_matches_delay_matrix_path(reco, events):
    """The cached travel-time stack gives the coarse map of the delay-matrix path to 1e-9."""
    stack = reco._singleray_tt_stack(('test', 'coarse'), *COARSE_AXES, VPOL_CHANNELS)
    assert stack[0].shape == (30 * 120 * 100, len(VPOL_CHANNELS))
    assert stack[1].dtype == np.bool_
    for _, _, stn in events:
        corr_data, packed, weights = _weighted_corr(reco, stn)
        fused = reco._singleray_stack_maps(stack, VPOL_CHANNELS, [packed], weights)[0].reshape(30, 120, 100)
        ref = _reference_map(reco, corr_data, packed, COARSE_AXES, weights)
        assert np.max(np.abs(fused - ref)) < MAP_TOL
        assert np.all(fused[:, :, -1] == 0.0)


@pytest.mark.slow
def test_k_stack_equals_single_maps(reco, events):
    """K correlation sets in one geometry pass give bit for bit the maps of K single passes."""
    src, _, stn = events[0]
    _, raw, weights = _weighted_corr(reco, stn)
    _, env, _ = _weighted_corr(reco, stn, 'traces')
    axes = _refine_axes(src)
    both = reco._singleray_grid_maps(*axes, VPOL_CHANNELS, [raw, env], weights)
    assert np.array_equal(both[0], reco._singleray_grid_maps(*axes, VPOL_CHANNELS, [raw], weights)[0])
    assert np.array_equal(both[1], reco._singleray_grid_maps(*axes, VPOL_CHANNELS, [env], weights)[0])
    stack = reco._singleray_tt_stack(('test', 'coarse'), *COARSE_AXES, VPOL_CHANNELS)
    both = reco._singleray_stack_maps(stack, VPOL_CHANNELS, [raw, env], weights)
    assert np.array_equal(both[0], reco._singleray_stack_maps(stack, VPOL_CHANNELS, [raw], weights)[0])
    assert np.array_equal(both[1], reco._singleray_stack_maps(stack, VPOL_CHANNELS, [env], weights)[0])


@pytest.mark.slow
def test_tolerant_edge_hook_validates_the_surface_slice(reco, events):
    """With tolerant_table_edge the z = 0 slice carries the scalar objective; by default it is 0."""
    src, _, stn = events[0]
    _, packed, weights = _weighted_corr(reco, stn)
    axes = (np.array([src[0]]), np.array([np.radians(src[1])]), np.array([-1.0, 0.0]))
    strict = reco._singleray_grid_maps(*axes, VPOL_CHANNELS, [packed], weights)[0]
    assert strict[0, 0, 1] == 0.0
    reco._tolerant_table_edge = True
    try:
        tolerant = reco._singleray_grid_maps(*axes, VPOL_CHANNELS, [packed], weights)[0]
    finally:
        reco._tolerant_table_edge = False
    assert tolerant[0, 0, 0] == strict[0, 0, 0]
    cache = reco._build_optimizer_cache(VPOL_CHANNELS, weights, packed=packed)
    at_surface = -reco._correlation_at_point([src[0], src[1], 0.0], None, VPOL_CHANNELS, weights, _cache=cache)
    assert at_surface != 0.0
    assert abs(tolerant[0, 0, 1] - at_surface) < MAP_TOL


def _toy_scalar_corr(td_values, td_ok):
    """Scalar singleray objective on three co-located channels with 4 x 4 tables and unit correlations.

    The query (rho 1.5 m, z -1.5 m) sits inside every table, all three pair delays are 0 and fall
    inside the correlation arrays, so the record objective is -1 when every channel is valid and
    -1/3 when one channel drops out.

    Returns:
        The kernel's negative weighted mean correlation.
    """
    from NuRadioReco.utilities.reco3d_kernels import _scalar_singleray_corr_numba
    n_ch = 3
    ones = np.ones(n_ch, dtype=np.float64)
    return _scalar_singleray_corr_numba(
        1.5, 0.0, -1.5, 0.0, 0.0, np.zeros((n_ch, 2)), td_values, td_ok,
        np.arange(n_ch, dtype=np.int64), np.zeros(n_ch), ones, np.full(n_ch, 4, dtype=np.int64),
        np.full(n_ch, -3.0), ones, np.full(n_ch, 4, dtype=np.int64),
        np.ones((n_ch, 16)), np.full(n_ch, 16, dtype=np.int64), ones, np.full(n_ch, -8.0),
        np.array([0, 0, 1], dtype=np.int64), np.array([1, 2, 2], dtype=np.int64), ones, 3.0,
        False, 0.6)


def test_scalar_kernel_takes_validity_from_the_mask():
    """The scalar objective rejects a lookup whose corners the finiteness mask flags, whatever the values hold."""
    pytest.importorskip('numba')
    values = np.ones((3, 4, 4))
    ok = np.ones((3, 4, 4), dtype=np.bool_)
    assert _toy_scalar_corr(values, ok) == -1.0
    ok[0, 1, 1] = False
    assert abs(_toy_scalar_corr(values, ok) + 1.0 / 3.0) < 1e-15
    values[0, 1, 1] = np.nan
    assert abs(_toy_scalar_corr(values, np.isfinite(values)) + 1.0 / 3.0) < 1e-15


def test_table_reload_clears_the_travel_time_stack_cache(table_dir):
    """Loading tables again empties every cache built from the previous tables."""
    r = InterferometricReco3D()
    r._cpu_delay_T_cache = {}
    caches = (r._tt_stack_cache, r._delay_matrix_cache, r._gpu_delay_stack_cache, r._opt_geom_cache,
              r._packed_multiray_tables, r._cpu_delay_T_cache, r._table_mask_cache)
    for cache in caches:
        cache[('stale',)] = None
    r._preload_tables(STATION, reference_config(table_dir))
    assert not any(caches)


@pytest.mark.slow
@pytest.mark.parametrize('path', ['fused', 'delay_matrix', 'multiray'])
def test_coarse_caches_follow_the_channel_order(det, table_dir, tables, pa, path):
    """The same channel set in another order on a used instance gives a fresh instance's result.

    The pair order of the coarse delay matrices follows the configured channel order, so
    a coarse cache keyed by the sorted channels would hand the reversed order the first
    order's matrices. The fused travel-time stack and the multiray travel times (a dict
    keyed by channel) are checked the same way.
    """
    options = {'fused': {}, 'delay_matrix': {'use_fused_correlator': False},
               'multiray': {'multi_ray_types': True, 'table_scheme': 'solution_ordered'}}[path]
    first = reference_config(table_dir, **options)
    second = reference_config(table_dir, channels=VPOL_CHANNELS[::-1], **options)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(80.0, 120.0, -40.0, pa), VPOL_CHANNELS, tables,
                             snr=SNR, seed=5)
    shared = InterferometricReco3D()
    shared.begin(STATION, first, det)
    shared.run(evt, stn, det, first)
    fresh = InterferometricReco3D()
    fresh.begin(STATION, second, det)
    got = shared.run(evt, stn, det, second)
    want = fresh.run(evt, stn, det, second)
    assert set(got) == set(want)
    differ = [k for k in want if 'time' not in k and not same_value(want[k], got[k])]
    assert not differ, [(k, want[k], got[k]) for k in differ]


GROUPED_SOURCES = [(80.0, 120.0, -40.0), (180.0, 250.0, -90.0)]


@pytest.mark.slow
def test_grouped_multiray_result_does_not_depend_on_the_fused_flag(det, table_dir, tables, pa, monkeypatch):
    """With multiray_combo_mode grouped the result equals the delay-matrix path's bit for bit.

    The fused multiray refine kernel takes each pair's own best ray-type combination (the
    per_pair mode). Grouped mode refines through the grouped correlator whether
    use_fused_correlator is on (point-major ``grouped_multiray_points``) or off (pair-major
    ``grouped_multiray_numba``); per_pair still refines on the fused kernel.
    """
    calls = []
    fused_refine = InterferometricReco3D._fused_multiray_refine

    def spy(self, *args, **kwargs):
        """Record the combo mode of every fused multiray refine call."""
        calls.append(self._multiray_combo_mode)
        return fused_refine(self, *args, **kwargs)

    monkeypatch.setattr(InterferometricReco3D, '_fused_multiray_refine', spy)
    events = [make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables, snr=SNR, seed=5)[:2]
              for src in GROUPED_SOURCES]
    results = {}
    for mode, fused in (('grouped', True), ('grouped', False), ('per_pair', True)):
        cfg = reference_config(table_dir, multi_ray_types=True, multiray_combo_mode=mode, use_fused_correlator=fused)
        r = InterferometricReco3D()
        r.begin(STATION, cfg, det)
        results[mode, fused] = [r.run(evt, stn, det, cfg) for evt, stn in events]
    assert calls and set(calls) == {'per_pair'}, calls
    for src, got, want in zip(GROUPED_SOURCES, results['grouped', True], results['grouped', False]):
        assert set(got) == set(want)
        differ = [k for k in want if 'time' not in k and not same_value(want[k], got[k])]
        assert not differ, (src, [(k, want[k], got[k]) for k in differ])


DOMINANCE_TOL = 1e-9
COMPASS_MEDIAN_TOL_DEG = 0.01
COMPASS_DOMINANCE = pytest.mark.xfail(
    strict=True,
    reason="the compass search does not dominate L-BFGS-B event by event: both stop at vertices of the "
           "piecewise-linear objective (differences of 1e-7 to 1e-4 either way, below L-BFGS-B's own 1e-6 "
           "reproducibility between starting points) and on about 8 percent of synthetic events one of them "
           "climbs through successive correlation lobes to a peak the other does not reach")


def _run_both(reco, det, tables, pa, sources, base, compass):
    """Reconstruct each source with two configurations and return (sep_base, sep_compass, corr_base, corr_compass, src) rows."""
    rows = []
    for src in sources:
        evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables,
                                 snr=SNR, seed=5)
        res_l = reco.run(evt, stn, det, base)
        res_c = reco.run(evt, stn, det, compass)
        rows.append((angular_separation((res_l['rho'], res_l['phi'], res_l['z']), src, pa),
                     angular_separation((res_c['rho'], res_c['phi'], res_c['z']), src, pa),
                     res_l['max_corr'], res_c['max_corr'], src))
    return rows


@COMPASS_DOMINANCE
@pytest.mark.slow
def test_compass_search_dominates_lbfgsb(reco, det, tables, pa, base_config):
    """The compass search never returns a lower correlation than L-BFGS-B on the 16-source set."""
    compass_config = dict(base_config, optimizer_method='compass')
    for sep_l, sep_c, corr_l, corr_c, src in _run_both(
            reco, det, tables, pa, SUMMARY_SOURCES + rng_sources(8, 20260916), base_config, compass_config):
        assert corr_c >= corr_l - DOMINANCE_TOL, (src, corr_l, corr_c)


@COMPASS_DOMINANCE
@pytest.mark.slow
def test_compass_dominates_in_candidate_mode(reco, det, tables, pa, base_config):
    """Candidate polishing with the compass search never ranks below the L-BFGS-B polish."""
    cand = dict(base_config, candidate_search=['envelope', 'raw'])
    for sep_l, sep_c, corr_l, corr_c, src in _run_both(
            reco, det, tables, pa, SUMMARY_SOURCES, cand, dict(cand, optimizer_method='compass')):
        assert corr_c >= corr_l - DOMINANCE_TOL, (src, corr_l, corr_c)


@pytest.mark.slow
def test_compass_keeps_candidate_mode_accuracy(reco, det, tables, pa, base_config):
    """In candidate mode the compass search keeps the median accuracy of L-BFGS-B within 0.01 degree."""
    cand = dict(base_config, candidate_search=['envelope', 'raw'])
    rows = _run_both(reco, det, tables, pa, SUMMARY_SOURCES + rng_sources(8, 20260916), cand,
                     dict(cand, optimizer_method='compass'))
    seps = np.array([(r[0], r[1]) for r in rows])
    assert abs(np.median(seps[:, 0]) - np.median(seps[:, 1])) < COMPASS_MEDIAN_TOL_DEG, seps


def test_compass_options_are_validated():
    """Malformed compass steps or budgets and unknown optimizer methods raise ValueError."""
    from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
    reco = InterferometricReco3D()
    for bad in ({'compass_step': [1.0, 0.2]}, {'compass_step': 1.0},
                {'compass_step_min': [1e-8, 0.0, 1e-8]}, {'compass_max_evals': 0},
                {'compass_max_evals': 2.5}, {'optimizer_method': 'Powell'}):
        with pytest.raises(ValueError):
            reco._validate_config(bad)
    assert InterferometricReco3D._compass_options(
        {'compass_step': [2.0, 0.5, 2.0], 'compass_max_evals': 200}) == {
        'step': [2.0, 0.5, 2.0], 'step_min': None, 'max_evals': 200, 'phi_scan': False}


@pytest.mark.slow
def test_single_seed_compass_reads_the_config(reco, events):
    """The single-seed compass path honours the compass keys: a one-evaluation budget returns the seed."""
    src, _, stn = events[0]
    _, packed, weights = _weighted_corr(reco, stn)
    cache = reco._build_optimizer_cache(VPOL_CHANNELS, weights, packed=packed)
    bounds = [(1.0, 250.0), (0.0, 360.0), (-100.0, 0.0)]
    seed = (max(src[0] - 2.0, 2.0), (src[1] + 0.5) % 360.0, min(src[2] + 2.0, -1.0))
    held = reco._optimize_from_seed(seed, None, VPOL_CHANNELS, bounds, weights, method='compass',
                                    _cache=cache, config={'compass_max_evals': 1})
    assert held[:3] == seed
    moved = reco._optimize_from_seed(seed, None, VPOL_CHANNELS, bounds, weights, method='compass',
                                     _cache=cache, config={})
    assert moved[3] > held[3]


def test_compass_falls_back_without_singleray_cache():
    """Without the singleray numba cache the compass request resolves to L-BFGS-B."""
    from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
    assert not InterferometricReco3D._compass_available(None)
    assert not InterferometricReco3D._compass_available({'corr_packed': None})


FAR_SOURCES = [(160.0, 40.0, -30.0), (185.0, 130.0, -8.0), (210.0, 220.0, -60.0),
               (235.0, 310.0, -20.0), (175.0, 95.0, -85.0), (225.0, 5.0, -4.0)]
N_NOISE = 12
VALID_NORM_GATE = pytest.mark.xfail(
    strict=True,
    reason="the valid-weight objective fails its acceptance gate on the synthetic families: a mean over "
           "fewer valid pairs has a larger noise spread, so pure noise piles at the table edge "
           "(median reco rho 45 m to 154 m, fraction beyond 200 m 0.02 to 0.32 at floor 0.6; floors 0.5 "
           "and 0.8 fail too) and far sources recovered by the record objective are lost")


def _reco_pair(det, table_dir):
    """Two reconstruction objects on the reference configuration, record and valid-weight objectives."""
    from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
    from conftest import reference_config
    pair = []
    for overrides in ({}, {'objective_normalisation': 'valid'}):
        cfg = reference_config(table_dir, **overrides)
        r = InterferometricReco3D()
        r.begin(STATION, cfg, det)
        pair.append((r, cfg))
    return pair


def test_objective_normalisation_config_is_validated():
    """Unknown normalisations, floors outside (0, 1] and unsupported paths raise ValueError."""
    from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
    reco = InterferometricReco3D()
    for bad in ({'objective_normalisation': 'bogus'},
                {'objective_normalisation': 'valid', 'valid_weight_floor': 0.0},
                {'objective_normalisation': 'valid', 'valid_weight_floor': 1.5},
                {'objective_normalisation': 'valid', 'multi_ray_types': True},
                {'objective_normalisation': 'valid', 'use_fused_correlator': False}):
        with pytest.raises(ValueError):
            reco._validate_config(bad)
    reco._validate_config({'objective_normalisation': 'valid', 'valid_weight_floor': 0.5})
    reco._validate_config({'objective_normalisation': 'total'})


@pytest.mark.slow
def test_valid_norm_equals_total_norm_where_every_pair_contributes(reco, events):
    """With every pair valid and in range (the whole refine grid here) the valid-weight objective equals the record objective."""
    src, _, stn = events[0]
    _, packed, weights = _weighted_corr(reco, stn)
    cache = reco._build_optimizer_cache(VPOL_CHANNELS, weights, packed=packed)
    axes = _refine_axes(src)
    total = -reco._correlation_at_point(list(src), None, VPOL_CHANNELS, weights, _cache=cache)
    grid_total = reco._singleray_grid_maps(*axes, VPOL_CHANNELS, [packed], weights)[0]
    reco._valid_norm = True
    try:
        valid = -reco._correlation_at_point(list(src), None, VPOL_CHANNELS, weights, _cache=cache)
        grid_valid = reco._singleray_grid_maps(*axes, VPOL_CHANNELS, [packed], weights)[0]
    finally:
        reco._valid_norm = False
    assert abs(valid - total) < 1e-12
    idx = np.unravel_index(np.argmax(grid_total), grid_total.shape)
    assert np.unravel_index(np.argmax(grid_valid), grid_valid.shape) == idx
    assert np.max(np.abs(grid_valid - grid_total)) < 1e-12


@pytest.mark.slow
def test_valid_norm_reports_objective_version(det, tables, pa, table_dir):
    """Results carry objective_version 0 under the record objective and 1 under the valid-weight objective."""
    (reco_t, cfg_t), (reco_v, cfg_v) = _reco_pair(det, table_dir)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*FAR_SOURCES[0], pa), VPOL_CHANNELS, tables,
                             snr=SNR, seed=7)
    assert reco_t.run(evt, stn, det, cfg_t)['objective_version'] == 0
    assert reco_v.run(evt, stn, det, cfg_v)['objective_version'] == 1
    reco_t.end()
    reco_v.end()


@VALID_NORM_GATE
@pytest.mark.slow
def test_valid_norm_keeps_far_sources(det, tables, pa, table_dir):
    """Far sources (rho 160 to 235 m) recovered by the record objective stay recovered under the valid-weight objective."""
    (reco_t, cfg_t), (reco_v, cfg_v) = _reco_pair(det, table_dir)
    for src in FAR_SOURCES:
        evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables,
                                 snr=SNR, seed=7)
        res_t = reco_t.run(evt, stn, det, cfg_t)
        res_v = reco_v.run(evt, stn, det, cfg_v)
        sep_t = angular_separation((res_t['rho'], res_t['phi'], res_t['z']), src, pa)
        sep_v = angular_separation((res_v['rho'], res_v['phi'], res_v['z']), src, pa)
        if sep_t < 1.0:
            assert sep_v < 1.0, (src, sep_t, sep_v)
    reco_t.end()
    reco_v.end()


@VALID_NORM_GATE
@pytest.mark.slow
def test_valid_norm_does_not_pile_noise_at_the_edge(det, table_dir):
    """Pure-noise reconstructions do not move toward the rho or z edges under the valid-weight objective."""
    (reco_t, cfg_t), (reco_v, cfg_v) = _reco_pair(det, table_dir)
    edge_t = 0
    edge_v = 0
    for i in range(N_NOISE):
        evt, stn = make_noise_event(STATION, VPOL_CHANNELS, seed=900 + i)
        res_t = reco_t.run(evt, stn, det, cfg_t)
        res_v = reco_v.run(evt, stn, det, cfg_v)
        edge_t += int(res_t['rho'] > 200.0 or res_t['z'] > -10.0)
        edge_v += int(res_v['rho'] > 200.0 or res_v['z'] > -10.0)
    assert edge_v <= edge_t + 2, (edge_t, edge_v)
    reco_t.end()
    reco_v.end()
