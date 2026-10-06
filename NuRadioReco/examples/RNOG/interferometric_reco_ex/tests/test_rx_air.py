"""The rx pass 2 of the driver for sources above the surface.

Three driver options, each inert when absent: ``pass2_volume: pass1`` re-searches the
pass-1 configuration (tables, ``allow_above_surface``, split z grid, search keys) in a
window around the pass-1 answer clamped to the pass-1 limits, where the default in-ice
template refuses a window above the surface; ``rx_arrival_mode: first_arrival`` takes
each channel's arrival direction from the first-arriving ray of the ``air_ice``
propagator around the phased-array axis, and the straight air path for the surface LPDAs,
checked here against the ray invariant of the layered greenland_simple profile and the
straight line; ``cross_type_sign_mode: abs`` scores pairs of two antenna types by the
absolute correlation, which recovers a source whose pulse is inverted on the channels of
one type. ``clear_grid_caches`` empties the per-window caches without changing a result.
"""

import os

import numpy as np
import pytest

from conftest import STATION, reference_config
from synthetic import VPOL_CHANNELS, angular_separation, antenna_locations, cylindrical_to_enu, make_event, pa_center
from test_split_z_grid import ABOVE, BELOW, split_config
import reco_pass2 as drv
from NuRadioMC.utilities.medium import greenland_simple
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

AIR_SRC = (150.0, 200.0, 50.0)
PASS1_GUESS = {'rho': 175.0, 'phi': 206.0, 'z': 62.0}
WINDOW = [50, 20, 50]
SNR = 20.0
RECOVER_DEG = 1.0
LPDA = 13
ZENITH_TOL_DEG = 0.01
SIGN_SRC = (80.0, 120.0, -40.0)
FAKE_LPDA = [22, 23]


def legacy_template(config, station_id, channels, p2_steps):
    """The in-ice pass-2 template as the driver built it inline before the pass-2 options."""
    p2_hierarchical = config.get('pass2_hierarchical', False)
    t = {
        'station_id': station_id, 'channels': channels, 'coord_system': 'cylindrical',
        'hierarchical': p2_hierarchical,
        'multi_ray_types': config.get('multi_ray_types', False),
        'multiray_combo_mode': config.get('multiray_combo_mode', 'grouped'),
        'time_delay_tables': config['time_delay_tables'],
        'interp_method': config.get('interp_method', 'linear'),
        'snr_pair_weighting': config.get('snr_pair_weighting', True),
        'hilbert_envelope_mode': config.get('hilbert_envelope_mode', 'traces'),
        'correlation_normalization': config.get('correlation_normalization', 'normalized'),
        'apply_hann_window': config.get('apply_hann_window', False),
        'limits': [1, 100, 0, 360, -100, 0], 'step_sizes': p2_steps, 'n_rho': 0,
        'n_z': config.get('pass2_n_z', config.get('n_z', 0)),
        'z_spacing': config.get('z_spacing', 'linear'),
        'z_surface_offset': config.get('z_surface_offset', 0.1),
    }
    if p2_hierarchical:
        t.update({'coarse_limits': [1, 100, 0, 360, -100, 0], 'coarse_n_rho': 0,
                  'coarse_n_z': config.get('pass2_coarse_n_z', 0),
                  'coarse_step_sizes': config.get('pass2_coarse_step_sizes', [2, 1, 2]),
                  'coarse_n_peaks': config.get('pass2_coarse_n_peaks', 3),
                  'coarse_peak_separation': config.get('pass2_coarse_peak_separation', [5, 3, 5]),
                  'refine_window': config.get('pass2_refine_window', [5, 3, 5]),
                  'refine_step_sizes': p2_steps})
    if 'table_name_pattern' in config:
        t['table_name_pattern'] = config['table_name_pattern']
    if 'multiray_table_name_pattern' in config:
        t['multiray_table_name_pattern'] = config['multiray_table_name_pattern']
    return t


def invariant_zenith(D, h, z_ch, n_of_z):
    """Receive zenith (deg) at depth z_ch of a source at horizontal distance D and height h > 0.

    Uses the ray invariant p = n(z) sin(theta) of a horizontally layered medium with an air
    leg of n = 1, bisecting on the horizontal reach, independently of the NuRadioMC tracers.
    """
    zz = np.linspace(z_ch, 0.0, 20000)
    n = n_of_z(zz)

    def reach(p):
        s = p / n
        f = s / np.sqrt(1 - s ** 2)
        return np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(zz)) + h * p / np.sqrt(1 - p ** 2)

    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if reach(mid) < D else (lo, mid)
    return np.degrees(np.arcsin(0.5 * (lo + hi) / n[0]))


@pytest.fixture(scope='module')
def airice_dir():
    path = os.environ.get('RECO3D_TEST_AIRICE_TABLES', '')
    needed = [os.path.join(path, f'station{STATION}', f'st{STATION}_ch{ch}_rz_table.npz') for ch in VPOL_CHANNELS]
    if not all(os.path.isfile(p) for p in needed):
        pytest.skip(f'air-ice tables for every VPol channel not found under {path}')
    return path


def test_options_default_and_validation():
    """The three options default to the previous behaviour and reject unknown or conflicting values."""
    assert drv.pass2_options({}) == ('template', 'direct', 'signed')
    assert drv.pass2_options({'pass2_volume': 'pass1', 'rx_arrival_mode': 'first_arrival',
                              'cross_type_sign_mode': 'abs'}) == ('pass1', 'first_arrival', 'abs')
    for bad in ({'pass2_volume': 'air'}, {'rx_arrival_mode': 'direct_ray'}, {'cross_type_sign_mode': 'joint'},
                {'pass2_volume': 'pass1', 'z_profile_step': 5},
                {'cross_type_sign_mode': 'abs', 'polarization_groups': {'vpol': [0, 1]}},
                {'cross_type_sign_mode': 'abs', 'pair_signs': [1]}):
        with pytest.raises(ValueError):
            drv.pass2_options(bad)


def test_driver_keys_registered_with_the_module():
    """begin() knows the driver keys, so it logs no unknown-key warning for them."""
    assert {'pass2_volume', 'rx_arrival_mode', 'cross_type_sign_mode'} <= InterferometricReco3D._KNOWN_CONFIG_KEYS


def test_ray_tracer_backend_names_the_resolved_backend():
    """The recorded backend is the one NuRadioMC's analytic tracer resolves to by default."""
    from NuRadioMC.SignalProp.AnalyticRayTracingImpl.single_layer_analytic_raytracer import cpp_available
    assert drv.ray_tracer_backend() == ('cpp' if cpp_available else 'python')


@pytest.mark.parametrize('hierarchical', [False, True])
def test_template_path_unchanged(table_dir, hierarchical):
    """Without pass2_volume the template and a window inside the search volume are those of the previous inline code."""
    cfg = reference_config(table_dir, pass2_hierarchical=hierarchical)
    steps = [1, 0.3, 1]
    want = legacy_template(cfg, STATION, cfg['channels'], steps)
    template = drv.pass2_template(cfg, STATION, cfg['channels'], steps)
    assert template == want
    r1 = {'rho': 30.0, 'phi': 350.0, 'z': -50.0}
    got = drv.pass2_search_config(cfg, template, r1, WINDOW)
    limits = [1, 80.0, 330.0, 370.0, -100.0, 0.0]
    assert got['limits'] == limits
    assert ('coarse_limits' in got) == hierarchical
    if hierarchical:
        assert got['coarse_limits'] == limits
    assert {k: v for k, v in got.items() if k not in ('limits', 'coarse_limits')} == \
        {k: v for k, v in want.items() if k not in ('limits', 'coarse_limits')}


@pytest.mark.slow
@pytest.mark.parametrize('hierarchical', [False, True])
def test_template_window_stops_at_the_surface(det, table_dir, tables, pa, hierarchical):
    """A pass-1 answer within the z half-window of the surface: the template window stops at z = 0.

    Before the clamp the window of a pass-1 answer at z = -20 m reached z = +30 m and the
    in-ice template refused it, so the driver's rx and rxtx modes stopped at such events.
    Now pass 2 searches the in-ice block up to the surface and recovers a near-surface
    source; a pass-1 answer above the surface is refused with a pointer to pass2_volume: pass1.
    """
    cfg = reference_config(table_dir, pass2_hierarchical=hierarchical)
    template = drv.pass2_template(cfg, STATION, cfg['channels'], [1, 0.3, 1])
    got = drv.pass2_search_config(cfg, template, {'rho': 80.0, 'phi': 120.0, 'z': -20.0}, WINDOW)
    assert got['limits'] == [30.0, 130.0, 100.0, 140.0, -70.0, 0.0]
    if hierarchical:
        assert got['coarse_limits'] == got['limits']
    src = (80.0, 120.0, -12.0)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables, snr=SNR, seed=6)
    reco2 = InterferometricReco3D()
    reco2.begin(STATION, template, det)
    res = reco2.run(evt, stn, det, got)
    assert -70.0 <= res['z'] <= 0.0 and 30.0 <= res['rho'] <= 130.0, res
    assert angular_separation((res['rho'], res['phi'], res['z']), src, pa) < RECOVER_DEG, res
    with pytest.raises(ValueError, match='pass2_volume: pass1'):
        drv.pass2_search_config(cfg, template, {'rho': 80.0, 'phi': 120.0, 'z': 5.0}, WINDOW)


@pytest.mark.slow
def test_template_window_stays_in_the_search_volume(det, table_dir, tables, pa):
    """A template window crossing the volume's rho_max or z_min is clamped to the pass-1 limits.

    Before the clamp the window of a pass-1 answer at rho 230 m, z -80 m reached rho 280 m
    and z -130 m outside the volume rho 1 to 250 m, z -100 to 0 m, and pass 2 could
    return a position there. A window inside the volume is unchanged.
    """
    cfg = reference_config(table_dir)
    template = drv.pass2_template(cfg, STATION, cfg['channels'], [1, 0.3, 1])
    got = drv.pass2_search_config(cfg, template, {'rho': 230.0, 'phi': 60.0, 'z': -80.0}, WINDOW)
    assert got['limits'] == [180.0, 250, 40.0, 80.0, -100, -30.0]
    deep = dict(cfg, limits=[1, 250, 0, 360, -200, 0])
    inside = drv.pass2_search_config(deep, template, {'rho': 120.0, 'phi': 60.0, 'z': -60.0}, WINDOW)
    assert inside['limits'] == [70.0, 170.0, 40.0, 80.0, -110.0, -10.0]
    src = (245.0, 60.0, -97.0)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables, snr=SNR, seed=8)
    reco2 = InterferometricReco3D()
    reco2.begin(STATION, template, det)
    res = reco2.run(evt, stn, det, got)
    assert 180.0 <= res['rho'] <= 250.0 and -100.0 <= res['z'] <= -30.0, res


def test_pass1_window_keeps_volume_keys_and_clamps(table_dir):
    """pass2_volume: pass1 copies the pass-1 config and sets both limits to the clamped window."""
    cfg = split_config(table_dir, rho_max=1600, pass2_volume='pass1',
                       candidate_search=['envelope:traces', 'envelope:correlation', 'raw'])
    before = {k: (list(v) if isinstance(v, list) else v) for k, v in cfg.items()}
    got = drv.pass2_search_config(cfg, None, {'rho': 166.0, 'phi': 350.0, 'z': 41.0}, WINDOW)
    assert got['limits'] == got['coarse_limits'] == [116.0, 216.0, 330.0, 370.0, -9.0, 91.0]
    for key in ('allow_above_surface', 'z_grid_below', 'z_grid_above', 'time_delay_tables',
                'candidate_search', 'channels', 'hierarchical'):
        assert got[key] == cfg[key], key
    assert got['z_grid_below'] == BELOW and got['z_grid_above'] == ABOVE
    assert cfg == before
    edge = drv.pass2_search_config(cfg, None, {'rho': 20.0, 'phi': 10.0, 'z': 280.0}, WINDOW)
    assert edge['limits'] == [1.0, 70.0, -10.0, 30.0, 230.0, 300.0]
    low = drv.pass2_search_config(cfg, None, {'rho': 1590.0, 'phi': 0.0, 'z': -80.0}, WINDOW)
    assert low['limits'] == [1540.0, 1600, -20.0, 20.0, -100, -30.0]


@pytest.mark.airice
@pytest.mark.slow
def test_pass2_window_above_surface_recovers_source(det, airice_dir, table_dir, pa):
    """The pass-1 window searches above the surface and recovers an air source; the template refuses it.

    Clearing the per-window caches after the search leaves them empty and a repeated search
    gives the same answer.
    """
    from synthetic import TravelTimeTables
    cfg = split_config(airice_dir, rho_max=300, pass2_volume='pass1')
    tables = TravelTimeTables(airice_dir, STATION, VPOL_CHANNELS)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*AIR_SRC, pa), VPOL_CHANNELS, tables, snr=SNR, seed=4)
    cfg_p2 = drv.pass2_search_config(cfg, None, PASS1_GUESS, WINDOW)
    reco2 = InterferometricReco3D()
    reco2.begin(STATION, cfg, det)
    res = reco2.run(evt, stn, det, cfg_p2)
    got = (res['rho'], res['phi'], res['z'])
    assert angular_separation(got, AIR_SRC, pa) < RECOVER_DEG, got
    lim = cfg_p2['limits']
    assert lim[0] <= res['rho'] <= lim[1] and lim[4] <= res['z'] <= lim[5], (got, lim)
    assert res['z'] > 0, got
    assert reco2._tt_stack_cache or reco2._delay_matrix_cache
    reco2.clear_grid_caches()
    assert not reco2._tt_stack_cache and not reco2._delay_matrix_cache
    again = reco2.run(evt, stn, det, cfg_p2)
    assert all(again[k] == res[k] for k in ('rho', 'phi', 'z', 'max_corr'))

    ice_cfg = reference_config(table_dir)
    template = drv.pass2_template(ice_cfg, STATION, VPOL_CHANNELS, [1, 0.3, 1])
    legacy = InterferometricReco3D()
    legacy.begin(STATION, template, det)
    with pytest.raises(ValueError):
        legacy.run(evt, stn, det, drv.pass2_search_config(ice_cfg, template, PASS1_GUESS, WINDOW))


@pytest.fixture(scope='module')
def medium():
    return greenland_simple()


@pytest.mark.airice
def test_first_arrival_angles_above_surface(det, ant_locs, pa, medium):
    """Deep channels: the air_ice receive direction matches the ray invariant; the LPDA takes the straight line."""
    rho, phi, z = 300.0, 40.0, 80.0
    channels = VPOL_CHANNELS + [LPDA]
    angles = drv.compute_first_arrival_angles(rho, phi, z, STATION, det, channels, {LPDA})
    assert set(angles) == set(channels)
    src = cylindrical_to_enu(rho, phi, z, pa)
    n_of_z = np.vectorize(lambda zz: medium.get_index_of_refraction(np.array([0.0, 0.0, zz])))
    for ch in VPOL_CHANNELS:
        d = src - ant_locs[ch]
        zen, az = np.degrees(angles[ch])
        assert az == pytest.approx(np.degrees(np.arctan2(d[1], d[0])), abs=1e-6), ch
        want = invariant_zenith(np.hypot(d[0], d[1]), z, ant_locs[ch][2], n_of_z)
        assert abs(zen - want) < ZENITH_TOL_DEG, (ch, zen, want)
    d = src - ant_locs[LPDA]
    zen, az = angles[LPDA]
    assert zen == pytest.approx(np.arccos(d[2] / np.linalg.norm(d)), abs=1e-12)
    assert az == pytest.approx(np.arctan2(d[1], d[0]), abs=1e-12)
    assert np.degrees(angles[1][1]) == pytest.approx(phi, abs=0.05)


@pytest.mark.airice
def test_first_arrival_angles_in_ice_take_the_earliest_solution(det, ant_locs, pa):
    """In the ice every channel, the LPDA included, takes the solution with the smallest travel time."""
    from NuRadioMC.SignalProp import propagation
    rho, phi, z = 40.0, 100.0, -15.0
    channels = [1, 5, 9, LPDA]
    angles = drv.compute_first_arrival_angles(rho, phi, z, STATION, det, channels, {LPDA})
    assert set(angles) == set(channels)
    stn_abs = np.array(det.get_absolute_position(STATION))
    src = cylindrical_to_enu(rho, phi, z, pa)
    src_abs = np.array([stn_abs[0] + src[0], stn_abs[1] + src[1], z])
    rt = propagation.get_propagation_module('air_ice')(greenland_simple())
    n_two = 0
    for ch in channels:
        rt.set_start_and_end_point(src_abs, stn_abs + np.array(det.get_relative_position(STATION, ch)))
        rt.find_solutions()
        times = [rt.get_travel_time(i) for i in range(rt.get_number_of_solutions())]
        n_two += len(times) > 1
        rv = rt.get_receive_vector(int(np.argmin(times)))
        assert angles[ch][0] == pytest.approx(np.arccos(rv[2] / np.linalg.norm(rv)), abs=1e-12), ch
        assert angles[ch][1] == pytest.approx(np.arctan2(rv[1], rv[0]), abs=1e-12), ch
    assert n_two > 0


def test_antenna_types_and_cross_type_signs(det):
    """LPDAs, HPols and VPols are told apart by model name; only pairs of two types are scored by |corr|."""
    types = {ch: drv.antenna_type(det, STATION, ch) for ch in range(24)}
    assert [ch for ch in range(24) if types[ch] == 'lpda'] == list(range(12, 21))
    assert [ch for ch in range(24) if types[ch] == 'hpol'] == [4, 8, 11, 21]
    channels = [0, 1, 13, 16]
    assert drv.cross_type_pair_signs(channels, types) == [1, 'abs', 'abs', 'abs', 'abs', 1]
    assert drv.cross_type_pair_signs(VPOL_CHANNELS, types) == [1] * 55


@pytest.mark.slow
def test_cross_type_abs_recovers_inverted_type(reco, det, base_config, tables, pa):
    """Channels 22 and 23 standing in for a second antenna type, with their pulses inverted.

    The absolute correlation on the cross-type pairs recovers the source and scores it
    higher than the signed objective does; without the inversion it recovers it as well.
    """
    types = {ch: ('lpda' if ch in FAKE_LPDA else 'vpol') for ch in VPOL_CHANNELS}
    signs = drv.cross_type_pair_signs(VPOL_CHANNELS, types)
    assert signs.count('abs') == 2 * 9
    src_enu = cylindrical_to_enu(*SIGN_SRC, pa)
    for invert in (FAKE_LPDA, ()):
        evt, stn, _ = make_event(det, STATION, src_enu, VPOL_CHANNELS, tables, snr=SNR, seed=5, invert=invert)
        res_abs = reco.run(evt, stn, det, dict(base_config, pair_signs=signs))
        got = (res_abs['rho'], res_abs['phi'], res_abs['z'])
        assert angular_separation(got, SIGN_SRC, pa) < RECOVER_DEG, (invert, got)
        if invert:
            res_signed = reco.run(evt, stn, det, base_config)
            assert res_abs['max_corr'] > res_signed['max_corr'], (res_abs['max_corr'], res_signed['max_corr'])
