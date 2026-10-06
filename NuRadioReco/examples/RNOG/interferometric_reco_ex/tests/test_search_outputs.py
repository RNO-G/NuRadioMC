"""Search options: candidate-pool outputs and versioned quality metrics.

`candidate_fill_saved_peaks` takes the saved peaks after the primary from the candidate
pool (the ranked candidates plus the refined peaks no chain optimized, graded with the raw
correlation where they stand, in raw order) so `n_peaks_save` distinct positions are saved
whenever the pool has them; the primary keeps the polished ranking and `n_filled_peaks`
counts the unpolished saved peaks by provenance, wherever they rank (checked by forcing the
unpolished entries above the polished ones). With `candidate_include_refined` the pool is the
ranked candidates alone. In candidate mode the legacy quality metrics
(`peak_isolation_ratio`, `peak_{i}_map_snr`) keep their record definition on the raw
chain's coarse map (at the raw chain's own peaks, so they equal the default search's), and
`peak_{i}_map_snr_v2`, `peak_isolation_ratio_v2`, `map_snr_v2` and `candidate_n_pool`
describe the candidate result and the ranked pool. `candidate_diagnostics` records what
the harness needs to split a failure into an unvisited basin and a misranked one,
including the best pre-polish raw value per chain. The search geometry options are checked the same way:
`tolerant_table_edge` makes the batch table lookup accept the table top (z = 0) and the rho
edge of a table, in the singleray lookup, the multiray lookup and the fused multiray refine,
and recovers sources 0.3 m below the surface within 1 m in z (the strict rule loses one of
them to a wrong basin), `refine_window_mode: adaptive` does at least as well as the fixed
windows over the far sources (the characterisation set at 120 m and beyond plus a far grid:
gated on the fraction within 3 degrees, the median, no new failure and the count of improved
against worsened sources, all at the level measured at SNR 20), and `subbin_coarse_seeds`
is applied to the envelope chains only (the raw default search is unchanged by it) and
keeps the 16-source summary gates on the candidate search. With every key absent the result
carries none of the new fields and the golden master covers the numbers.
"""

import itertools

import numpy as np
import pytest

from conftest import STATION, reference_config, rng_sources
from synthetic import VPOL_CHANNELS, angular_separation, cylindrical_to_enu, make_event
from test_candidate_search import SUMMARY_GATES as CANDIDATE_GATES
from test_reco_known_answer import EXACT_SOURCES, SUMMARY_GATES, SUMMARY_SOURCES
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

SNR = 20.0
DEDUP_TOL = (10.0, 5.0, 10.0)
POOL_SAVE = 8
SOURCES = [p.values[0] for p in EXACT_SOURCES]
SOURCE_IDS = [p.id for p in EXACT_SOURCES]
NEW_KEYS = ('peak_isolation_ratio_v2', 'map_snr_v2', 'candidate_n_pool', 'n_filled_peaks',
            'candidate_prepolish_max_corr', 'candidate_n_basins')
SURFACE_SOURCES = [(80.0, 30.0, -0.3), (80.0, 270.0, -0.3)]
SURFACE_TOL_DEG = 1.0
SURFACE_TOL_Z = 1.0
FAR_SOURCES = [s for s in SUMMARY_SOURCES + rng_sources(8, 20260916) if s[0] >= 120.0] + [
    (float(r), float(p), float(z)) for r in (130, 160, 200, 240) for p in (50, 170, 290) for z in (-10, -50, -90)]
FAR_CHANGE_DEG = 0.05
FILL_BOOST = 10.0


@pytest.fixture(scope='module')
def fill_config(table_dir):
    """Candidate search with the saved peaks filled from the pool."""
    return reference_config(table_dir, candidate_search=['envelope', 'raw'],
                            candidate_fill_saved_peaks=True)


@pytest.fixture(scope='module')
def diag_config(table_dir):
    """Candidate search with the validation metrics and the diagnostics."""
    return reference_config(table_dir, candidate_search=['envelope', 'raw'], validation=True,
                            candidate_diagnostics=True)


def _event(det, src, tables, pa, seed=5):
    """Build a synthetic VPol event for a (rho, phi, z) source relative to the PA."""
    return make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables, snr=SNR, seed=seed)


def _distinct(a, b):
    """True when two (rho, phi, z) positions differ beyond the deduplication tolerances."""
    dphi = abs(a[1] - b[1])
    dphi = min(dphi, 360.0 - dphi)
    return abs(a[0] - b[0]) >= DEDUP_TOL[0] or dphi >= DEDUP_TOL[1] or abs(a[2] - b[2]) >= DEDUP_TOL[2]


def _saved(res):
    """Saved peaks of a result as (rho, phi, z, corr) tuples."""
    return [(res[f'peak_{i}_rho'], res[f'peak_{i}_phi'], res[f'peak_{i}_z'], res[f'peak_{i}_corr'])
            for i in range(res['n_saved_peaks'])]


def _raw_coarse_map(reco, stn, config):
    """Raw-correlation coarse map of the reference grid, recomputed from the module's pieces."""
    channels = config['channels']
    volt = [stn.get_channel(ch).get_trace() for ch in channels]
    times = [stn.get_channel(ch).get_times() for ch in channels]
    pw, _ = reco._compute_snr_pair_weights(volt, channels)
    raw, _ = reco._prepare_corr_funcs(times, volt, hilbert_envelope_mode=None,
                                      apply_hann_window=config['apply_hann_window'],
                                      correlation_normalization=config['correlation_normalization'])
    lim = config['coarse_limits']
    rho_vec = np.geomspace(max(lim[0], 1.0), lim[1], config['coarse_n_rho'])
    phi_vec = np.arange(lim[2], lim[3], config['coarse_step_sizes'][1]) * (np.pi / 180.0)
    z_vec = np.linspace(lim[4], lim[5], config['coarse_n_z'])
    src = reco._build_source_enu_matrix(rho_vec, phi_vec, z_vec)
    mean_corr, _ = reco._correlator_lean(raw, reco._compute_delay_matrices(src, channels), pair_weights=pw)
    return mean_corr, rho_vec, phi_vec * (180.0 / np.pi), z_vec


def test_invalid_wave1_keys_are_rejected():
    """The fill and diagnostics keys must be booleans."""
    reco = InterferometricReco3D()
    for bad in ({'candidate_fill_saved_peaks': 1}, {'candidate_diagnostics': 'yes'},
                {'candidate_search': ['raw'], 'candidate_fill_saved_peaks': None}):
        with pytest.raises(ValueError):
            reco._validate_config(bad)
    reco._validate_config({'candidate_fill_saved_peaks': True, 'candidate_diagnostics': True})
    reco._validate_config({'candidate_search': ['envelope', 'raw'], 'candidate_fill_saved_peaks': False})


@pytest.mark.slow
@pytest.mark.parametrize('src', SOURCES, ids=SOURCE_IDS)
def test_fill_saved_peaks_gives_n_peaks_save_distinct(reco, det, fill_config, tables, pa, src):
    """With the fill the exact-recovery sources save n_peaks_save distinct peaks and peak_0 is the primary."""
    evt, stn, _ = _event(det, src, tables, pa)
    res = reco.run(evt, stn, det, fill_config)
    saved = _saved(res)
    assert res['n_saved_peaks'] == fill_config['n_peaks_save'] == len(saved), res['n_saved_peaks']
    assert saved[0][:3] == (res['rho'], res['phi'], res['z']) and saved[0][3] == res['max_corr']
    assert angular_separation(saved[0][:3], src, pa) < 0.3
    for a, b in itertools.combinations(saved, 2):
        assert _distinct(a, b), (a, b)
    assert all(np.isfinite(p[3]) for p in saved)
    assert res['candidate_n_pool'] >= res['n_saved_peaks']
    assert max(0, res['n_saved_peaks'] - res['n_candidates']) <= res['n_filled_peaks'] <= res['n_saved_peaks'] - 1
    for i in range(res['n_saved_peaks']):
        assert res[f'candidate_origin_{i}'] in (0, 1)
        assert np.isfinite(res[f'peak_{i}_map_snr_v2'])


@pytest.mark.slow
def test_n_filled_peaks_counts_unpolished_entries_wherever_they_rank(reco, det, table_dir, tables, pa, monkeypatch):
    """n_filled_peaks counts the saved peaks graded in place even when they outrank the polished candidates."""
    graded_in_place = InterferometricReco3D._grade_at_position

    def boosted(self, *args, **kwargs):
        """Grade in place as the module does, then raise the value and the pre-polish value by FILL_BOOST."""
        return [e[:3] + (e[3] + FILL_BOOST, e[4], e[5] + FILL_BOOST) + e[6:]
                for e in graded_in_place(self, *args, **kwargs)]

    monkeypatch.setattr(InterferometricReco3D, '_grade_at_position', boosted)
    cfg = reference_config(table_dir, candidate_search=['envelope', 'raw'], candidate_fill_saved_peaks=True,
                           candidate_diagnostics=True)
    evt, stn, _ = _event(det, (80.0, 120.0, -40.0), tables, pa)
    res = reco.run(evt, stn, det, cfg)
    n_saved = res['n_saved_peaks']
    assert n_saved == cfg['n_peaks_save']
    assert res['peak_0_corr'] == res['max_corr'] < FILL_BOOST
    filled = [i for i in range(1, n_saved) if res[f'peak_{i}_corr'] > FILL_BOOST]
    assert filled == list(range(1, 1 + len(filled))) and len(filled) >= 1, filled
    assert res['n_filled_peaks'] == len(filled)
    for i in filled:
        assert res[f'candidate_prepolish_corr_{i}'] == res[f'peak_{i}_corr']
    for i in range(1 + len(filled), n_saved):
        assert res[f'candidate_prepolish_corr_{i}'] < res[f'peak_{i}_corr'] < FILL_BOOST


@pytest.mark.slow
def test_include_refined_pool_is_the_ranked_candidates(reco, det, table_dir, tables, pa):
    """With candidate_include_refined every refined peak is a candidate, so the pool adds nothing and no peak is filled."""
    cfg = reference_config(table_dir, candidate_search=['envelope', 'raw'], candidate_include_refined=True,
                           candidate_fill_saved_peaks=True)
    evt, stn, _ = _event(det, (80.0, 120.0, -40.0), tables, pa)
    res = reco.run(evt, stn, det, cfg)
    assert res['candidate_n_pool'] == res['n_candidates']
    assert res['n_filled_peaks'] == 0
    assert res['n_saved_peaks'] == min(cfg['n_peaks_save'], res['n_candidates'])


@pytest.mark.slow
def test_fill_saved_peaks_in_default_search(reco, det, table_dir, tables, pa):
    """The fill also completes the saved peaks of the default search from its unseeded refined peaks."""
    cfg = reference_config(table_dir, candidate_fill_saved_peaks=True)
    evt, stn, _ = _event(det, (80.0, 120.0, -40.0), tables, pa)
    res = reco.run(evt, stn, det, cfg)
    saved = _saved(res)
    assert len(saved) == cfg['n_peaks_save']
    assert saved[0][:3] == (res['rho'], res['phi'], res['z']) and saved[0][3] == res['max_corr']
    for a, b in itertools.combinations(saved, 2):
        assert _distinct(a, b), (a, b)
    assert res['n_filled_peaks'] >= 0
    assert not [k for k in res if k.startswith('candidate_') or k.endswith('_v2')]


@pytest.mark.slow
def test_v2_metrics_present_and_legacy_keys_keep_raw_definition(reco, det, diag_config, base_config, tables, pa):
    """The v2 metrics are finite and the legacy metrics equal a recomputation from the raw coarse map."""
    src = (80.0, 120.0, -40.0)
    evt, stn, _ = _event(det, src, tables, pa)
    res = reco.run(evt, stn, det, diag_config)
    ref = reco.run(evt, stn, det, dict(base_config, validation=True))
    for key in ('peak_isolation_ratio_v2', 'map_snr_v2', 'peak_isolation_ratio'):
        assert np.isfinite(res[key]), (key, res[key])
    assert res['candidate_n_pool'] >= res['n_candidates'] >= 1
    assert res['candidate_map_snr_chain'] == 0
    assert res['peak_isolation_ratio'] == ref['peak_isolation_ratio']
    assert res['surf_corr_z'] == ref['surf_corr_z'] and res['surf_corr_zen'] == ref['surf_corr_zen']
    assert res['coarse_peaks'] == ref['coarse_peaks']

    mean_corr, rho_vec, phi_vec_deg, z_vec = _raw_coarse_map(reco, stn, diag_config)
    peaks = reco._extract_top_n_peaks(mean_corr, rho_vec, phi_vec_deg, z_vec,
                                      diag_config['coarse_n_peaks'], diag_config['coarse_peak_separation'])
    corrs = sorted([p[3] for p in peaks], reverse=True)
    assert res['peak_isolation_ratio'] == pytest.approx(corrs[0] / np.mean(corrs[:5]), rel=1e-9)
    assert res['map_snr_v2'] == pytest.approx(res['max_corr'] / np.std(mean_corr[np.isfinite(mean_corr)]), rel=1e-9)
    pidx = reco._find_peak_bin(res['rho'], res['phi'], res['z'], rho_vec, phi_vec_deg, z_vec)
    assert res['peak_0_map_snr_v2'] == pytest.approx(reco._compute_map_snr(mean_corr, pidx), rel=1e-9)
    for i in range(res['n_saved_peaks']):
        if i < ref['n_saved_peaks']:
            assert res[f'peak_{i}_map_snr'] == ref[f'peak_{i}_map_snr'], i
        else:
            assert np.isnan(res[f'peak_{i}_map_snr']), i
    pool = [res[f'candidate_pool_{i}_corr'] for i in range(res['candidate_n_pool'])]
    assert res['peak_isolation_ratio_v2'] == pytest.approx(
        res['max_corr'] / np.mean(sorted(pool, reverse=True)[:5]), rel=1e-9)


@pytest.mark.slow
def test_candidate_diagnostics_keys(reco, det, diag_config, tables, pa):
    """The diagnostics describe the pool consistently with the primary and the ranked candidates."""
    src = (80.0, 120.0, -40.0)
    evt, stn, _ = _event(det, src, tables, pa)
    res = reco.run(evt, stn, det, diag_config)
    assert res['candidate_prepolish_max_corr'] <= res['max_corr'] + 1e-12
    assert 1 <= res['n_candidates'] <= res['candidate_n_basins']
    n_pool = res['candidate_n_pool']
    assert n_pool <= POOL_SAVE or all(f'candidate_pool_{i}_rho' in res for i in range(POOL_SAVE))
    assert (res['candidate_pool_0_rho'], res['candidate_pool_0_phi'], res['candidate_pool_0_z']) == \
        (res['rho'], res['phi'], res['z'])
    assert res['candidate_pool_0_corr'] == res['max_corr']
    for i in range(POOL_SAVE):
        vals = [res[f'candidate_pool_{i}_{k}'] for k in ('rho', 'phi', 'z', 'corr')]
        if i < n_pool:
            assert all(np.isfinite(v) for v in vals), (i, vals)
            assert res[f'candidate_pool_{i}_origin'] in (0, 1)
        else:
            assert all(np.isnan(v) for v in vals), (i, vals)
            assert res[f'candidate_pool_{i}_origin'] == -1
    pool_corrs = [res[f'candidate_pool_{i}_corr'] for i in range(1, min(n_pool, POOL_SAVE))]
    assert pool_corrs == sorted(pool_corrs, reverse=True)
    for i in range(res['n_saved_peaks']):
        assert res[f'candidate_prepolish_corr_{i}'] <= res[f'peak_{i}_corr'] + 1e-12
    by_chain = {k: res[f'candidate_prepolish_max_corr_{k}'] for k in ('raw', 'traces', 'correlation')}
    assert np.isnan(by_chain['correlation'])
    assert np.isfinite(by_chain['raw']) and np.isfinite(by_chain['traces'])
    assert max(by_chain['raw'], by_chain['traces']) == res['candidate_prepolish_max_corr']


@pytest.mark.slow
def test_default_config_has_no_wave1_keys(reco, det, base_config, tables, pa):
    """Without the new keys the result carries none of the new fields."""
    evt, stn, _ = _event(det, (80.0, 120.0, -40.0), tables, pa)
    res = reco.run(evt, stn, det, base_config)
    assert not [k for k in res if k in NEW_KEYS or k.startswith('candidate_')]


@pytest.fixture(scope='module')
def tolerant_reco(det, table_dir):
    """Reconstruction object with the tolerant table edge (a begin-time option)."""
    r = InterferometricReco3D()
    r.begin(STATION, reference_config(table_dir, tolerant_table_edge=True), det)
    return r


def _sep(res, src, pa):
    """Angular separation in degrees between a result and the source."""
    return angular_separation((res['rho'], res['phi'], res['z']), src, pa)


def _events(det, sources, tables, pa):
    """Synthetic events for the sources that have a table solution on every channel."""
    events = []
    for src in sources:
        try:
            evt, stn, _ = _event(det, src, tables, pa)
        except ValueError:
            continue
        events.append((src, evt, stn))
    return events


def test_invalid_search_geometry_keys_are_rejected():
    """The geometry keys take booleans and refine_window_mode one of fixed or adaptive."""
    reco = InterferometricReco3D()
    for bad in ({'tolerant_table_edge': 1}, {'subbin_coarse_seeds': 'yes'}, {'refine_window_mode': 'bogus'},
                {'refine_window_mode': None}):
        with pytest.raises(ValueError):
            reco._validate_config(bad)
    reco._validate_config({'tolerant_table_edge': True, 'subbin_coarse_seeds': True, 'refine_window_mode': 'adaptive'})
    reco._validate_config({'refine_window_mode': 'fixed'})


def test_subbin_peak_estimate_on_synthetic_map():
    """The parabola estimate moves a bin-centred peak toward the true sub-bin position and guards the edges."""
    reco = InterferometricReco3D()
    rho_vec = np.geomspace(1.0, 250.0, 30)
    phi_vec = np.arange(0.0, 360.0, 3.0)
    z_vec = np.linspace(-100.0, 0.0, 100)
    ii, jj, kk = np.meshgrid(np.arange(30), np.arange(120), np.arange(100), indexing='ij')

    def gaussian_map(ir, ip, iz, sigma=1.5):
        """Return a Gaussian peak in index space centred at (ir, ip, iz), wrapped in phi."""
        dp = np.minimum(np.abs(jj - ip), 120 - np.abs(jj - ip))
        return np.exp(-((ii - ir) ** 2 + dp ** 2 + (kk - iz) ** 2) / (2 * sigma ** 2))

    corr_map = gaussian_map(20.3, 10.4, 50.25)
    est = reco._subbin_peak(corr_map, rho_vec, phi_vec, z_vec, (rho_vec[20], phi_vec[10], z_vec[50], 1.0))
    assert est[0] == pytest.approx(np.exp(np.interp(20.3, np.arange(30), np.log(rho_vec))), rel=0.02)
    assert est[1] == pytest.approx(10.4 * 3.0, abs=0.1)
    assert est[2] == pytest.approx(np.interp(50.25, np.arange(100), z_vec), abs=0.1)
    assert est[3] == 1.0
    wrapped = gaussian_map(20.0, -0.3, 50.0)
    est = reco._subbin_peak(wrapped, rho_vec, phi_vec, z_vec, (rho_vec[20], phi_vec[0], z_vec[50], 1.0))
    assert est[1] == pytest.approx(-0.3 * 3.0, abs=0.1) and est[0] == pytest.approx(rho_vec[20], rel=0.02)
    edge = gaussian_map(0.4, 10.0, 99.0)
    est = reco._subbin_peak(edge, rho_vec, phi_vec, z_vec, (rho_vec[0], phi_vec[10], z_vec[99], 1.0))
    assert est[0] == rho_vec[0] and est[2] == z_vec[99]
    holed = gaussian_map(20.3, 10.4, 50.25)
    holed[21, 10, 50] = np.nan
    est = reco._subbin_peak(holed, rho_vec, phi_vec, z_vec, (rho_vec[20], phi_vec[10], z_vec[50], 1.0))
    assert est[0] == rho_vec[20]
    flat = np.ones((30, 120, 100))
    est = reco._subbin_peak(flat, rho_vec, phi_vec, z_vec, (rho_vec[20], phi_vec[10], z_vec[50], 1.0))
    assert est[:3] == (rho_vec[20], phi_vec[10], z_vec[50])


@pytest.fixture(scope='module')
def multiray_recos(det, table_dir):
    """Strict and tolerant reconstruction objects with the two-table multiray scheme and the fused refine."""
    recos = []
    for tolerant in (False, True):
        r = InterferometricReco3D()
        r.begin(STATION, reference_config(table_dir, multi_ray_types=True, table_scheme='solution_ordered',
                                          use_fused_correlator=True, tolerant_table_edge=tolerant), det)
        recos.append(r)
    return tuple(recos)


@pytest.mark.slow
def test_tolerant_table_edge_accepts_the_table_top(reco, tolerant_reco):
    """At z = 0 exactly the strict batch lookup is invalid and the tolerant one returns the top row."""
    channels = [0, 1]
    at_top = reco._build_source_enu_matrix(np.array([50.0]), np.array([0.0]), np.array([0.0]))
    below = reco._build_source_enu_matrix(np.array([50.0]), np.array([0.0]), np.array([-1e-6]))
    assert not np.isfinite(reco._compute_delay_matrices(at_top, channels)[0]).any()
    tolerant = tolerant_reco._compute_delay_matrices(at_top, channels)[0]
    strict_below = reco._compute_delay_matrices(below, channels)[0]
    assert np.isfinite(tolerant).all()
    assert tolerant == pytest.approx(strict_below, abs=1e-3)
    assert tolerant_reco._compute_delay_matrices(below, channels)[0] == pytest.approx(strict_below, abs=1e-12)


@pytest.mark.slow
def test_tolerant_table_edge_accepts_the_table_rho_edge(reco, tolerant_reco):
    """A query on the last rho column of a table is invalid under the strict rule and valid under the tolerant one."""
    td = reco._interpolators[0]
    r_max = td.r_min + (td.nr - 1) / td.dr_inv
    finite = np.isfinite(td.values[td.nr - 2:, :]).all(axis=0)
    z = td.z_min + int(np.flatnonzero(finite[:-1] & finite[1:])[0]) / td.dz_inv
    at_edge = np.array([[r_max, z]])
    inside = np.array([[r_max - 1e-6, z]])
    assert not np.isfinite(reco._table_lookup_batch(td, at_edge)).any()
    tolerant = tolerant_reco._table_lookup_batch(td, at_edge)
    assert np.isfinite(tolerant).all()
    assert tolerant == pytest.approx(reco._table_lookup_batch(td, inside), abs=1e-3)


@pytest.mark.slow
def test_tolerant_table_edge_in_multiray_lookup_and_fused_refine(multiray_recos, det, tables, pa):
    """At z = 0 the multiray lookup and the fused refine are invalid under the strict rule and match z = -1e-6 under the tolerant one."""
    strict, tolerant = multiray_recos
    channels = list(VPOL_CHANNELS)
    rho_vec = np.array([48.0, 50.0, 52.0])
    phi_vec = np.array([30.0]) * (np.pi / 180.0)
    grids = {z: (strict._build_source_enu_matrix(rho_vec, phi_vec, np.array([z])), rho_vec, phi_vec, np.array([z]))
             for z in (0.0, -1e-6)}
    assert all(not tt for tt in strict._compute_tt_multiray(grids[0.0][0], channels).values())
    tt_top = tolerant._compute_tt_multiray(grids[0.0][0], channels)
    tt_below = strict._compute_tt_multiray(grids[-1e-6][0], channels)
    assert sum(len(tt) for tt in tt_below.values()) >= len(channels)
    for ch in channels:
        assert tt_top[ch].keys() == tt_below[ch].keys(), ch
        for rt, grid in tt_below[ch].items():
            assert tt_top[ch][rt] == pytest.approx(grid, abs=1e-3), (ch, rt)

    evt, stn, _ = _event(det, (50.0, 30.0, -1e-6), tables, pa)
    volt = [stn.get_channel(ch).get_trace() for ch in channels]
    times = [stn.get_channel(ch).get_times() for ch in channels]
    corr, _ = strict._prepare_corr_funcs(times, volt, hilbert_envelope_mode=None,
                                         apply_hann_window=True, correlation_normalization='energy')
    sep = [10.0, 5.0, 10.0]
    strict_top = strict._fused_multiray_refine([grids[0.0]], corr, channels, None, 1, sep)
    assert strict_top and not any(p[3] > 0.0 for p in strict_top), strict_top
    tolerant_top = tolerant._fused_multiray_refine([grids[0.0]], corr, channels, None, 1, sep)
    strict_below = strict._fused_multiray_refine([grids[-1e-6]], corr, channels, None, 1, sep)
    assert tolerant_top[0][3] > 0.1 and tolerant_top[0][2] == 0.0
    assert tolerant_top[0][:2] == strict_below[0][:2]
    assert tolerant_top[0][3] == pytest.approx(strict_below[0][3], abs=1e-3)
    assert tolerant._fused_multiray_refine([grids[-1e-6]], corr, channels, None, 1, sep) == strict_below


@pytest.mark.slow
@pytest.mark.parametrize('src', SURFACE_SOURCES, ids=[f'rho{s[0]:g}_phi{s[1]:g}' for s in SURFACE_SOURCES])
def test_tolerant_table_edge_recovers_surface_source(tolerant_reco, det, table_dir, tables, pa, src):
    """A source 0.3 m below the surface is reconstructed within 1 m in z with the default search."""
    cfg = reference_config(table_dir, tolerant_table_edge=True)
    evt, stn, _ = _event(det, src, tables, pa)
    res = tolerant_reco.run(evt, stn, det, cfg)
    assert abs(res['z'] - src[2]) < SURFACE_TOL_Z, (src, res['z'])
    assert _sep(res, src, pa) < SURFACE_TOL_DEG, (src, _sep(res, src, pa))


@pytest.mark.slow
def test_adaptive_windows_recover_far_sources_at_least_as_well(reco, det, table_dir, base_config, tables, pa):
    """Over the far sources the adaptive windows match or beat the fixed windows on every summary."""
    cfg = reference_config(table_dir, refine_window_mode='adaptive')
    fixed, adaptive = [], []
    for src, evt, stn in _events(det, FAR_SOURCES, tables, pa):
        fixed.append(_sep(reco.run(evt, stn, det, base_config), src, pa))
        adaptive.append(_sep(reco.run(evt, stn, det, cfg), src, pa))
    fixed, adaptive = np.array(fixed), np.array(adaptive)
    print('far sources fixed:', np.round(fixed, 3).tolist())
    print('far sources adaptive:', np.round(adaptive, 3).tolist())
    assert len(fixed) >= 20
    assert np.mean(adaptive < 3.0) >= np.mean(fixed < 3.0)
    assert np.median(adaptive) <= np.median(fixed)
    assert not np.any((fixed < 3.0) & (adaptive >= 3.0)), 'a source recovered with fixed windows is lost'
    assert np.sum(adaptive < fixed - FAR_CHANGE_DEG) >= np.sum(adaptive > fixed + FAR_CHANGE_DEG)


@pytest.mark.slow
def test_subbin_seeds_apply_to_envelope_chains_only(reco, det, table_dir, tables, pa, monkeypatch):
    """The parabola is taken on envelope coarse maps and never on a raw coarse map."""
    calls = []
    search_chain = InterferometricReco3D._search_chain

    def recording(self, *args, **kwargs):
        """Record the chain's ``subbin_seeds`` flag and run the original search chain."""
        calls.append(kwargs.get('subbin_seeds', False))
        return search_chain(self, *args, **kwargs)

    monkeypatch.setattr(InterferometricReco3D, '_search_chain', recording)
    evt, stn, _ = _event(det, (80.0, 120.0, -40.0), tables, pa)
    cases = [(dict(subbin_coarse_seeds=True, candidate_search=['envelope', 'raw']), [True, False]),
             (dict(subbin_coarse_seeds=True, candidate_search=['raw', 'envelope:correlation']), [False, True]),
             (dict(subbin_coarse_seeds=True), [False]),
             (dict(subbin_coarse_seeds=True, hilbert_envelope_mode='traces'), [True]),
             (dict(candidate_search=['envelope', 'raw']), [False, False])]
    for overrides, expected in cases:
        calls.clear()
        reco.run(evt, stn, det, reference_config(table_dir, **overrides))
        assert calls == expected, (overrides, calls)


@pytest.mark.slow
@pytest.mark.parametrize('src', SOURCES[:2], ids=SOURCE_IDS[:2])
def test_subbin_seeds_leave_the_raw_default_search_unchanged(reco, det, table_dir, base_config, tables, pa, src):
    """On the raw default search the key changes nothing: the result equals the fixed seeds exactly."""
    cfg = reference_config(table_dir, subbin_coarse_seeds=True)
    evt, stn, _ = _event(det, src, tables, pa)
    res = reco.run(evt, stn, det, cfg)
    ref = reco.run(evt, stn, det, base_config)
    assert (res['rho'], res['phi'], res['z'], res['max_corr']) == (ref['rho'], ref['phi'], ref['z'], ref['max_corr'])
    assert _saved(res) == _saved(ref)


@pytest.mark.slow
def test_subbin_seeds_keep_summary_gates(reco, det, table_dir, tables, pa):
    """With the sub-bin seeds on the envelope chain the candidate search keeps the 16-source gates.

    The parabola is taken on the traces-envelope coarse maxima; the raw coarse map and the raw
    polish grids are never shifted. Gated at the known-answer summary gates of the default search
    and at the candidate search's own fraction within 3 degrees.
    """
    cfg = reference_config(table_dir, candidate_search=['envelope', 'raw'], subbin_coarse_seeds=True)
    sources = SUMMARY_SOURCES + rng_sources(8, 20260916)
    seps = np.array([_sep(reco.run(*_event(det, src, tables, pa)[:2], det, cfg), src, pa) for src in sources])
    summary = dict(median_deg=float(np.median(seps)), p68_deg=float(np.percentile(seps, 68)),
                   frac_lt_3deg=float(np.mean(seps < 3)), frac_lt_1deg=float(np.mean(seps < 1)))
    print('sub-bin seeds (candidate search, traces envelope) summary at SNR 20:', summary)
    assert summary['median_deg'] <= SUMMARY_GATES['median_deg'], summary
    assert summary['p68_deg'] <= SUMMARY_GATES['p68_deg'], summary
    assert summary['frac_lt_3deg'] >= max(SUMMARY_GATES['frac_lt_3deg'], CANDIDATE_GATES['traces']), summary
    assert summary['frac_lt_1deg'] >= SUMMARY_GATES['frac_lt_1deg'], summary
