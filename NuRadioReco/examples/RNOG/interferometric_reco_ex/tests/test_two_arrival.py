"""Consistent two-arrival polish objective on the solution-ordered tables.

A shallow source beyond the total-internal-reflection radius sends the deep
antennas a full-amplitude surface-reflected pulse after the direct one; the
single-arrival objective treats it as clutter. With `polish_objective:
two_arrival_consistent` the candidate polish evaluates every pair at the same
solution index in both channels (solution_0 and solution_1 of the solution-ordered
tables, no cross terms) and adds the solution_1 term weighted by a critical-angle
mask (1 where the solution_1 ray leaves the source steeper than arcsin(1/n(z)) from
the vertical, 0 where the reflection is sub-critical). Without `two_arrival_margin`
the ranking stays raw and the two-arrival value is a diagnostic; with the margin a
position is ranked by its two-arrival value where that exceeds its raw correlation
by more than the margin. The tests check the mask at reference points and against
the analytic tracer's launch angle, that a zero second weight and the absent margin
reproduce the raw polish, the result keys, that the default path carries no new
keys, that the tie band composes with the margin through the raw gain of the ranked best,
and the recovery of a synthetic doublet family (rho 120 to 200 m, z -1 to
-20 m, second pulse at the solution_1 time with amplitude 1.0 where the analytic
tracer's launch angle is super-critical and 0.12 where it is not) against the
single-arrival objectives on the same events, once with same-polarity second
pulses and once with the reflection phase rotating per channel.
"""

import logging

import numpy as np
import pytest

from conftest import STATION, reference_config
from synthetic import (VPOL_CHANNELS, TravelTimeTables, angular_separation, antenna_locations,
                       cylindrical_to_enu, make_doublet_event, make_event, make_noise_event)
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

PA_CHANNELS = [0, 1, 2, 3]
SNR = 20.0
TOL_DEG = 1.0
TOL_Z_M = 3.0
TWIN_DZ_M = -15.0
WITHIN_FLOOR = 0.6
ROTATED_WITHIN_FLOOR = 0.6
SUB_CRITICAL_AMPLITUDE = 0.12
TRACER_TOL_DEG = 0.5
FAMILY_RHO = (120.0, 140.0, 160.0, 180.0, 200.0)
FAMILY_Z = (-1.0, -2.0, -5.0, -10.0, -20.0)
FAMILY_PHI = (30.0, 200.0)
MIN_CHANNELS = 6
SOLUTION_PATTERN = 'st{station_id}_ch{ch}_rz_table_solution_{slot}.npz'
ICE = (1.78, 0.51, 37.25)


@pytest.fixture(scope='module')
def two_arrival_config(table_dir):
    """Candidate search ranked by the two-arrival objective (margin 0) with the default mask weighting."""
    return reference_config(table_dir, candidate_search=['envelope', 'raw'],
                            polish_objective='two_arrival_consistent', two_arrival_margin=0.0)


@pytest.fixture(scope='module')
def diagnostic_config(table_dir):
    """Two-arrival objective without a margin: raw ranking plus the diagnostic keys."""
    return reference_config(table_dir, candidate_search=['envelope', 'raw'],
                            polish_objective='two_arrival_consistent')


@pytest.fixture(scope='module')
def raw_polish_config(table_dir):
    """Candidate search with the default raw polish objective."""
    return reference_config(table_dir, candidate_search=['envelope', 'raw'])


@pytest.fixture(scope='module')
def reco2(det, two_arrival_config):
    """Reconstruction object with the solution-ordered tables loaded for the two-arrival polish."""
    r = InterferometricReco3D()
    r.begin(STATION, two_arrival_config, det)
    return r


@pytest.fixture(scope='module')
def solution_tables(table_dir):
    """(solution_0, solution_1) travel-time tables of the VPol channels."""
    return tuple(TravelTimeTables(table_dir, STATION, VPOL_CHANNELS,
                                  pattern=SOLUTION_PATTERN.replace('{slot}', str(slot)))
                 for slot in (0, 1))


def _tracer():
    """Return the analytic ray tracer module and the ice model, or skip the test."""
    try:
        from NuRadioMC.SignalProp import analyticraytracing
        from NuRadioMC.utilities import medium
    except ImportError:
        pytest.skip('analytic ray tracer not importable')
    logging.getLogger('NuRadioMC.analytic_ray_tracing').setLevel(logging.ERROR)
    return analyticraytracing, medium.greenland_simple()


def _second_solution(analyticraytracing, ice, start, end):
    """Launch zenith (deg) and travel time (ns) of the second-arriving ray, or None with fewer than two."""
    tracer = analyticraytracing.ray_tracing(ice)
    tracer.set_start_and_end_point(np.asarray(start, dtype=float), np.asarray(end, dtype=float))
    tracer.find_solutions()
    if tracer.get_number_of_solutions() < 2:
        return None
    times = [tracer.get_travel_time(i) for i in range(tracer.get_number_of_solutions())]
    second = int(np.argsort(times)[1])
    launch = tracer.get_launch_vector(second)
    return float(np.degrees(np.arccos(launch[2] / np.linalg.norm(launch)))), float(times[second])


def _critical_deg(z):
    """Critical angle from the vertical at depth z for the exponential ice profile."""
    n_ice, delta_n, z_0 = ICE
    return float(np.degrees(np.arcsin(1.0 / (n_ice - delta_n * np.exp(z / z_0)))))


@pytest.fixture(scope='module')
def doublet_family(det, solution_tables, pa):
    """Doublet events on the rho, z, phi grid whose source lights at least MIN_CHANNELS channels.

    The second pulse of every channel takes amplitude 1.0 where the analytic
    tracer's second solution leaves the source steeper than the critical angle
    and SUB_CRITICAL_AMPLITUDE elsewhere (none where the tracer finds a single
    solution). Each source is built twice with the same noise: with same-polarity
    second pulses, and with the reflection phase rotated per channel from 0 at the
    critical angle to pi at grazing incidence (linear in the launch angle) and a
    polarity flip of the sub-critical reflection.
    """
    analyticraytracing, ice = _tracer()
    ant_locs = antenna_locations(det, STATION)
    family = []
    seed = 100
    for rho in FAMILY_RHO:
        for z in FAMILY_Z:
            for phi in FAMILY_PHI:
                src = (rho, phi, z)
                src_enu = cylindrical_to_enu(*src, pa)
                critical = _critical_deg(z)
                amplitudes, phases = {}, {}
                for ch in VPOL_CHANNELS:
                    second = _second_solution(analyticraytracing, ice, src_enu, ant_locs[ch])
                    if second is None:
                        amplitudes[ch] = 0.0
                        continue
                    launch = second[0]
                    if launch > critical:
                        amplitudes[ch] = 1.0
                        phases[ch] = np.pi * min((launch - critical) / (90.0 - critical), 1.0)
                    else:
                        amplitudes[ch] = SUB_CRITICAL_AMPLITUDE
                        phases[ch] = np.pi
                try:
                    plain = make_doublet_event(det, STATION, src_enu, VPOL_CHANNELS, *solution_tables,
                                               amplitudes, snr=SNR, seed=seed, min_channels=MIN_CHANNELS)
                except ValueError:
                    continue
                rotated = make_doublet_event(det, STATION, src_enu, VPOL_CHANNELS, *solution_tables,
                                             amplitudes, snr=SNR, seed=seed, min_channels=MIN_CHANNELS,
                                             second_phases=phases)
                seed += 1
                family.append((src, plain[:2], rotated[:2]))
    return family


def _sep(res, src, pa):
    """Angular separation in degrees between a reconstruction result and the source."""
    return angular_separation((res['rho'], res['phi'], res['z']), src, pa)


def _rank_keys(res):
    """Ranking keys of the saved peaks under the margin rule, from the saved two-arrival and raw values."""
    keys = []
    for i in range(res['n_saved_peaks']):
        two, raw = res[f'peak_{i}_corr_two_arrival'], res[f'peak_{i}_raw_corr_single']
        keys.append(two if two - raw > 0.0 else raw)
    return keys


def test_two_arrival_config_validation():
    """Unknown objectives, weight modes, sources, negative weights or margins and a missing candidate search raise."""
    reco = InterferometricReco3D()
    for bad in ({'polish_objective': 'two_arrival'},
                {'polish_objective': 'two_arrival_consistent'},
                {'polish_objective': 'two_arrival_consistent', 'candidate_search': ['raw'],
                 'two_arrival_weight_mode': 'none'},
                {'polish_objective': 'two_arrival_consistent', 'candidate_search': ['raw'],
                 'two_arrival_second_weight': -0.5},
                {'polish_objective': 'two_arrival_consistent', 'candidate_search': ['raw'],
                 'two_arrival_second_weight': True},
                {'polish_objective': 'two_arrival_consistent', 'candidate_search': ['raw'],
                 'two_arrival_margin': -0.1},
                {'polish_objective': 'two_arrival_consistent', 'candidate_search': ['raw'],
                 'two_arrival_margin': '0.1'},
                {'polish_objective': 'two_arrival_consistent', 'candidate_search': ['raw'],
                 'two_arrival_margin': False},
                {'polish_objective': 'two_arrival_consistent', 'candidate_search': ['raw'],
                 'max_corr_source': 'both'},
                {'polish_objective': 'two_arrival_consistent', 'candidate_search': ['raw'],
                 'multi_ray_types': True},
                {'polish_objective': 'two_arrival_consistent', 'candidate_search': ['raw'],
                 'objective_normalisation': 'valid'}):
        with pytest.raises(ValueError):
            reco._validate_config(bad)
    with pytest.raises(ValueError, match='objective_normalisation: total'):
        reco._two_arrival_settings({'polish_objective': 'two_arrival_consistent', 'candidate_search': ['raw'],
                                    'objective_normalisation': 'valid'})
    reco._validate_config({})
    assert reco._two_arrival_settings({'polish_objective': 'raw'}) is None
    settings = reco._two_arrival_settings({'polish_objective': 'two_arrival_consistent',
                                           'candidate_search': ['envelope', 'raw']})
    assert settings == {'weight_mode': 'mask', 'second_weight': 1.0, 'margin': None,
                        'max_corr_source': 'raw'}
    settings = reco._two_arrival_settings({'polish_objective': 'two_arrival_consistent',
                                           'candidate_search': ['envelope', 'raw'],
                                           'two_arrival_weight_mode': 'fixed',
                                           'two_arrival_second_weight': 0.5,
                                           'two_arrival_margin': 0.05,
                                           'max_corr_source': 'two_arrival'})
    assert settings == {'weight_mode': 'fixed', 'second_weight': 0.5, 'margin': 0.05,
                        'max_corr_source': 'two_arrival'}


@pytest.mark.slow
@pytest.mark.parametrize('phi', [0.0, 90.0, 225.0])
def test_critical_angle_mask_at_reference_points(reco2, phi):
    """The mask is 1 at (150 m, -5 m) and 0 at (30 m, -5 m) for every phased-array channel."""
    for ch in PA_CHANNELS:
        mask, launch, critical = reco2.two_arrival_mask(ch, 150.0, phi, -5.0)
        assert mask == 1.0 and launch > critical, (ch, phi, mask, launch, critical)
        mask, launch, critical = reco2.two_arrival_mask(ch, 30.0, phi, -5.0)
        assert mask == 0.0 and launch < critical, (ch, phi, mask, launch, critical)
        assert abs(critical - _critical_deg(-5.0)) < 1e-9


@pytest.mark.slow
def test_launch_angle_matches_analytic_tracer(reco2, ant_locs):
    """The table-gradient launch angle of the channel-0 solution_1 ray agrees with the analytic tracer."""
    analyticraytracing, ice = _tracer()
    td1 = reco2._two_arrival_interpolators[0]['solution_1']
    z_ant = float(ant_locs[0][2])
    for r, z in ((150.0, -5.0), (30.0, -5.0), (100.0, -5.0), (120.0, -20.0)):
        launch_tracer, t_tracer = _second_solution(analyticraytracing, ice, (r, 0.0, z), (0.0, 0.0, z_ant))
        tt1, launch_table, critical, mask = reco2._two_arrival_launch(td1, r, z, reco2._two_arrival_ice)
        assert abs(launch_table - launch_tracer) < TRACER_TOL_DEG, (r, z, launch_table, launch_tracer)
        assert abs(tt1 - t_tracer) < 1.0, (r, z, tt1, t_tracer)
        assert mask == float(launch_tracer > critical), (r, z, mask, launch_tracer, critical)


@pytest.mark.slow
def test_numba_point_matches_numpy_fallback(reco2, det, tables, pa):
    """The numba two-arrival point value equals the numpy fallback at a few points, with and without the mask."""
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(80.0, 120.0, -40.0, pa), VPOL_CHANNELS, tables,
                             snr=SNR, seed=5)
    volt = [stn.get_channel(ch).get_trace() for ch in VPOL_CHANNELS]
    times = [stn.get_channel(ch).get_times() for ch in VPOL_CHANNELS]
    weights, _ = reco2._compute_snr_pair_weights(volt, VPOL_CHANNELS)
    corr_data, packed = reco2._prepare_corr_funcs(times, volt, hilbert_envelope_mode=None,
                                                  apply_hann_window=True, correlation_normalization='energy')
    cache = reco2._build_optimizer_cache(VPOL_CHANNELS, weights, corr_data, packed=packed)
    points = [(150.0, 30.0, -5.0), (120.0, 200.0, -2.0), (90.0, 75.0, -20.0), (60.0, 300.0, -50.0),
              (180.0, 140.0, -1.0)]
    base = {'margin': None, 'max_corr_source': 'raw'}
    second_term = False
    for settings in (dict(base, weight_mode='mask', second_weight=1.0),
                     dict(base, weight_mode='fixed', second_weight=0.5)):
        for rho, phi, z in points:
            fast = -reco2._two_arrival_at_point([rho, phi, z], VPOL_CHANNELS, cache, settings)
            phi_rad = np.radians(phi)
            x = np.array([rho * np.cos(phi_rad) + reco2._pa_center[0]])
            y = np.array([rho * np.sin(phi_rad) + reco2._pa_center[1]])
            slow = reco2._two_arrival_numpy(x, y, np.array([z]), VPOL_CHANNELS, cache, settings)[0]
            assert abs(fast - slow) < 1e-9, (settings, rho, phi, z, fast, slow)
            single = -reco2._two_arrival_at_point(
                [rho, phi, z], VPOL_CHANNELS, cache, dict(settings, second_weight=0.0))
            second_term |= abs(fast - single) > 1e-6
    assert second_term, 'no test point carries a solution_1 term'


@pytest.mark.slow
def test_fixed_zero_weight_matches_raw_polish(reco2, det, table_dir, raw_polish_config, tables, pa):
    """A fixed second weight of zero reproduces the raw polish on a single-pulse event, margin set."""
    cfg = reference_config(table_dir, candidate_search=['envelope', 'raw'],
                           polish_objective='two_arrival_consistent', two_arrival_margin=0.0,
                           two_arrival_weight_mode='fixed', two_arrival_second_weight=0.0)
    src = (80.0, 120.0, -40.0)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables,
                             snr=SNR, seed=5)
    res = reco2.run(evt, stn, det, cfg)
    ref = reco2.run(evt, stn, det, raw_polish_config)
    for key in ('rho', 'phi', 'z', 'max_corr'):
        assert abs(res[key] - ref[key]) < 1e-6, (key, res[key], ref[key])
    assert abs(res['corr_two_arrival'] - res['raw_corr_single']) < 1e-9
    assert res['raw_corr_single'] == res['max_corr']


@pytest.mark.slow
def test_without_margin_matches_raw_polish(reco2, det, diagnostic_config, raw_polish_config, tables, pa):
    """Without a margin the two-arrival mode returns the raw polish plus the diagnostic keys."""
    src = (80.0, 120.0, -40.0)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables,
                             snr=SNR, seed=5)
    res = reco2.run(evt, stn, det, diagnostic_config)
    ref = reco2.run(evt, stn, det, raw_polish_config)
    assert res['n_saved_peaks'] == ref['n_saved_peaks']
    for i in range(res['n_saved_peaks']):
        for key in ('rho', 'phi', 'z', 'corr'):
            assert res[f'peak_{i}_{key}'] == ref[f'peak_{i}_{key}'], (i, key)
        assert res[f'peak_{i}_raw_corr_single'] == ref[f'peak_{i}_corr']
        assert np.isfinite(res[f'peak_{i}_corr_two_arrival'])
    assert res['max_corr'] == ref['max_corr'] == res['raw_corr_single']
    assert res['corr_two_arrival'] == res['peak_0_corr_two_arrival']


@pytest.mark.slow
def test_two_arrival_result_keys(reco2, det, two_arrival_config, tables, pa):
    """The peaks are ordered by the margin rule; max_corr follows max_corr_source."""
    src = (80.0, 120.0, -40.0)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables,
                             snr=SNR, seed=5)
    res = reco2.run(evt, stn, det, two_arrival_config)
    assert _sep(res, src, pa) < TOL_DEG, res
    assert res['peak_0_corr'] == res['max_corr'] == res['raw_corr_single']
    assert res['peak_0_corr_two_arrival'] == res['corr_two_arrival']
    assert res['peak_0_raw_corr_single'] == res['raw_corr_single']
    n_saved = res['n_saved_peaks']
    assert all(f'peak_{i}_corr_two_arrival' in res for i in range(n_saved))
    keys = _rank_keys(res)
    assert keys == sorted(keys, reverse=True), keys
    cfg = dict(two_arrival_config, max_corr_source='two_arrival')
    res2 = reco2.run(evt, stn, det, cfg)
    assert res2['max_corr'] == res2['corr_two_arrival'] == res2['peak_0_corr']
    assert res2['raw_corr_single'] == res['raw_corr_single']
    assert [res2[f'peak_{i}_rho'] for i in range(n_saved)] == [res[f'peak_{i}_rho'] for i in range(n_saved)]


@pytest.mark.slow
def test_tie_band_with_two_arrival_margin(reco2, det, two_arrival_config, raw_polish_config, tables, pa):
    """With a margin the tie band tests the raw gain of the ranked best, with the saved peaks filled.

    A band of 1 sends every event to the raw chain's answer, which must be the raw polish
    configuration's answer under the same band. A band of 0 keeps the ranked best exactly
    when its raw correlation is at least the raw chain's (the gain then equals that
    difference) and otherwise returns the raw chain's answer with a negative gain.
    """
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(80.0, 120.0, -40.0, pa), VPOL_CHANNELS, tables,
                             snr=SNR, seed=5)
    events = [(evt, stn), make_noise_event(STATION, VPOL_CHANNELS, seed=3)]
    for evt, stn in events:
        runs = {}
        for band in (1.0, 0.0):
            for name, cfg in (('two', two_arrival_config), ('raw', raw_polish_config)):
                runs[name, band] = reco2.run(evt, stn, det, dict(cfg, candidate_tie_band=band,
                                                                 candidate_fill_saved_peaks=True))
        fallback, ref = runs['two', 1.0], runs['raw', 1.0]
        r0 = ref['candidate_raw_chain_corr']
        assert np.isfinite(r0)
        assert fallback['candidate_fallback'] == ref['candidate_fallback'] == 1
        assert fallback['candidate_raw_chain_corr'] == r0
        assert [fallback[k] for k in ('rho', 'phi', 'z')] == [ref[k] for k in ('rho', 'phi', 'z')]
        assert fallback['max_corr'] == fallback['raw_corr_single'] == ref['max_corr'] == r0
        res = runs['two', 0.0]
        assert res['candidate_raw_chain_corr'] == r0
        if res['candidate_fallback']:
            assert res['candidate_gain'] < 0.0
            assert [res[k] for k in ('rho', 'phi', 'z')] == [ref[k] for k in ('rho', 'phi', 'z')]
            assert res['raw_corr_single'] == r0
        else:
            assert res['candidate_gain'] >= 0.0
            assert res['candidate_gain'] == res['raw_corr_single'] - r0


@pytest.mark.slow
def test_default_path_has_no_two_arrival_keys(reco, det, base_config, tables, pa):
    """Without the new keys the result dict carries no two-arrival fields."""
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(80.0, 120.0, -40.0, pa), VPOL_CHANNELS,
                             tables, snr=SNR, seed=5)
    res = reco.run(evt, stn, det, base_config)
    assert not [k for k in res if 'two_arrival' in k or 'raw_corr_single' in k]


def _family_summary(reco2, det, family, variant, configs, pa, label):
    """Reconstruct one family variant with every config and summarise the per-config errors.

    Args:
        reco2: Reconstruction object with the solution-ordered tables loaded.
        det: Detector description.
        family: Output of the `doublet_family` fixture.
        variant: 1 for the same-polarity events, 2 for the rotated ones.
        configs: (name, config) pairs, reconstructed in order.
        pa: Phased-array reference point.
        label: Printed with the summary.

    Returns:
        Dict of statistic -> {config name: value} with the twin rate, median and
        68th percentile of |dz| and the fraction within TOL_DEG and TOL_Z_M.
    """
    rows = []
    for src, *variants in family:
        evt, stn = variants[variant - 1]
        row = list(src)
        for _, config in configs:
            res = reco2.run(evt, stn, det, config)
            row += [_sep(res, src, pa), res['z'] - src[2]]
        rows.append(row)
    rows = np.array(rows)
    columns = [(name, 3 + 2 * i) for i, (name, _) in enumerate(configs)]
    stats = {
        'twin': {name: float(np.mean(rows[:, col + 1] < TWIN_DZ_M)) for name, col in columns},
        'dz_median': {name: float(np.median(np.abs(rows[:, col + 1]))) for name, col in columns},
        'dz_p68': {name: float(np.percentile(np.abs(rows[:, col + 1]), 68)) for name, col in columns},
        'within': {name: float(np.mean((rows[:, col] < TOL_DEG) & (np.abs(rows[:, col + 1]) < TOL_Z_M)))
                   for name, col in columns},
    }
    print(f'{label}: {len(rows)} events; twin rate (z more than {-TWIN_DZ_M:g} m deeper) {stats["twin"]}; '
          f'median |dz| {stats["dz_median"]}; p68 |dz| {stats["dz_p68"]}; '
          f'within {TOL_DEG:g} deg and {TOL_Z_M:g} m {stats["within"]}')
    fmt = ' | '.join(f'{name} sep %6.3f dz %7.2f' for name, _ in configs)
    for row in rows:
        print(('  src rho %5.1f phi %5.1f z %5.1f | ' + fmt) % tuple(row))
    return stats


@pytest.mark.slow
def test_doublet_family_two_arrival_beats_single_arrival(reco2, det, two_arrival_config, raw_polish_config,
                                                         base_config, doublet_family, pa):
    """On the same-polarity doublet family the two-arrival ranking lowers the twin rate and the depth error.

    Compared with the raw polish and the record chain on the same events, the
    twin rate and the median and 68th percentile of |dz| must not be larger, and
    the fraction within TOL_DEG and TOL_Z_M must not be smaller than the raw
    polish's nor below WITHIN_FLOOR (measured 0.66 against 0.63 and 0.32).
    """
    assert len(doublet_family) >= 20, len(doublet_family)
    stats = _family_summary(reco2, det, doublet_family, 1,
                            (('two_arrival', two_arrival_config), ('raw_polish', raw_polish_config),
                             ('record_chain', base_config)), pa, 'doublet family, same polarity')
    for other in ('raw_polish', 'record_chain'):
        for name in ('twin', 'dz_median', 'dz_p68'):
            assert stats[name]['two_arrival'] <= stats[name][other], (name, other, stats[name])
        assert stats['within']['two_arrival'] >= stats['within'][other], (other, stats['within'])
    assert stats['within']['two_arrival'] >= WITHIN_FLOOR, stats['within']


@pytest.mark.slow
def test_doublet_family_rotated_second_pulse(reco2, det, two_arrival_config, raw_polish_config,
                                             base_config, doublet_family, pa):
    """With the reflection phase rotating per channel the two-arrival ranking still beats the raw polish.

    The per-channel rotation decorrelates the solution_1 terms between channels,
    which is the physical risk of the summed objective. The twin rate must not
    exceed the raw polish's or the record chain's and the fraction within
    TOL_DEG and TOL_Z_M must not fall below ROTATED_WITHIN_FLOOR (measured 0.71
    against 0.63 for the raw polish and 0.37 for the record chain).
    """
    stats = _family_summary(reco2, det, doublet_family, 2,
                            (('two_arrival', two_arrival_config), ('raw_polish', raw_polish_config),
                             ('record_chain', base_config)), pa, 'doublet family, rotated second pulse')
    for other in ('raw_polish', 'record_chain'):
        assert stats['twin']['two_arrival'] <= stats['twin'][other], (other, stats['twin'])
    assert stats['within']['two_arrival'] >= ROTATED_WITHIN_FLOOR, stats['within']
