"""Far-field hypothesis: the best plane-wave direction over the sky on the same pair series.

The plane-wave arrival times (``plane_wave_times``) use the conserved horizontal slowness
sin(zen) / c and the vertical delay integral of sqrt(n(z)^2 - sin(zen)^2) / c from the
antenna depth to the surface. They are checked against the closed form at zenith 0 and an
adaptive quadrature at zeniths from 2 to 89.5 deg for deep and near-surface (LPDA-like)
depths and above the surface, and against the air-ice tables: the pair delays of
point sources on a ray approach the plane-wave delays as the source recedes (residual about
inversely proportional to the distance; pairs with an unreadable table cell are left out). On synthetic plane-wave events the hypothesis finds
the direction within 0.1 degree, its correlation exceeds the in-ice point-source search's,
every other result field is unchanged, the far-field lag windows hold the delays of random
sky directions, and the search on stored, cut pairs equals the batch search. Near the
vertical (zenith 5 deg is tested) the plane wave is defined where the LPDA tables and the
air-ice tracer have no solution. With the lobe guard (``far_field_lobe_guard_ns``) the far
field still finds a broadband pulse and a narrow-band burst within 0.1 degree and changes no
other field; the guarded answer moves no weighted pair delay by more than the guard from the
envelope peak, which a vanishing guard returns; invalid guard values are refused. The n(z) of
the plane wave is that of the tables' ice model (recorded in the tables, else the config key
``ice_model``, else greenland_simple): a layered model (greenland_3exp_layered) integrates
layer by layer to an adaptive quadrature, a single layer equals the exponential form, the
lag windows hold its sky delays, and a model the tables contradict or an unknown name raises;
the exact far-field gradient (``optimizer_gradient: exact``) refuses a layered model. Settings that differ
only in the point-source volume share one far-field search, with the values of a fresh search.
"""

import itertools
import os

import numpy as np
import pytest
from scipy.integrate import quad

from conftest import STATION, reference_config
from synthetic import (N_NATIVE, NATIVE_RATE, SAMPLING_RATE, VPOL_CHANNELS, antenna_locations,
                       filtered_trace, same_value)
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.interferometricDirectionReconstruction3D import (
    InterferometricReco3D, _C_M_PER_NS, far_field_profile, plane_wave_times)
from NuRadioReco.utilities import units
from pair_store import cut_pairs

ICE = (1.78, 0.51, 37.25)
FIELDS = ('zen', 'az', 'corr_raw', 'corr_env_traces', 'corr_env_correlation', 'map_snr', 'origin')
DIRECTIONS = [(5.0, 120.0), (30.0, 40.0), (60.0, 200.0), (80.0, 300.0)]
BURST_PERIOD_NS = 8.6
BURST_WIDTH_NS = 12.0


def plane_wave_event(det, zen_deg, az_deg, snr=20.0, seed=0):
    """VPol event whose pulses arrive at the plane-wave times of a sky direction."""
    locs = antenna_locations(det, STATION)
    ant = np.array([locs[ch] for ch in VPOL_CHANNELS])
    times = plane_wave_times(np.radians(zen_deg), np.radians(az_deg), ant, ICE)
    rng = np.random.default_rng(seed)
    evt = Event(0, seed)
    stn = Station(STATION)
    for ch, t in zip(VPOL_CHANNELS, times):
        channel = Channel(ch)
        channel.set_trace(filtered_trace(150.0 * units.ns + t - times.min(), 1.0, 1.0 / snr, rng), NATIVE_RATE)
        channel.resample(SAMPLING_RATE)
        stn.add_channel(channel)
    evt.set_station(stn)
    return evt, stn


def burst_event(det, zen_deg, az_deg, snr=20.0, seed=0):
    """VPol event of a narrow-band burst (carrier period BURST_PERIOD_NS, Gaussian envelope of
    standard deviation BURST_WIDTH_NS) arriving at the plane-wave times of a sky direction."""
    locs = antenna_locations(det, STATION)
    ant = np.array([locs[ch] for ch in VPOL_CHANNELS])
    times = plane_wave_times(np.radians(zen_deg), np.radians(az_deg), ant, ICE)
    rng = np.random.default_rng(seed)
    t = np.arange(N_NATIVE) / NATIVE_RATE
    evt = Event(0, seed)
    stn = Station(STATION)
    for ch, t_arrival in zip(VPOL_CHANNELS, times):
        dt = t - (150.0 * units.ns + t_arrival - times.min())
        burst = np.exp(-0.5 * (dt / BURST_WIDTH_NS) ** 2) * np.cos(2 * np.pi * dt / BURST_PERIOD_NS)
        channel = Channel(ch)
        channel.set_trace(burst + filtered_trace(0.0, 0.0, 1.0 / snr, rng), NATIVE_RATE)
        channel.resample(SAMPLING_RATE)
        stn.add_channel(channel)
    evt.set_station(stn)
    return evt, stn


def sky_separation(zen_a, az_a, zen_b, az_b):
    """Angle in degrees between two sky directions."""
    za, aa, zb, ab = np.radians([zen_a, az_a, zen_b, az_b])
    va = np.array([np.sin(za) * np.cos(aa), np.sin(za) * np.sin(aa), np.cos(za)])
    vb = np.array([np.sin(zb) * np.cos(ab), np.sin(zb) * np.sin(ab), np.cos(zb)])
    return float(np.degrees(np.arccos(np.clip(va @ vb, -1.0, 1.0))))


def test_model_against_closed_form_and_quadrature():
    """The vertical delay equals its closed form at zenith 0 and an adaptive quadrature elsewhere."""
    n_ice, delta_n, z_0 = ICE
    ant = np.array([[0.0, 0.0, -97.0], [0.0, 0.0, -38.0], [0.0, 0.0, -3.3], [0.0, 0.0, -0.5],
                    [0.0, 0.0, 2.0]])
    t0 = plane_wave_times(0.0, 0.0, ant, ICE)
    closed = (n_ice * -ant[:4, 2] - delta_n * z_0 * (1.0 - np.exp(ant[:4, 2] / z_0))) / _C_M_PER_NS
    assert np.allclose(t0[:4], closed, rtol=0, atol=1e-9)
    assert t0[4] == pytest.approx(-2.0 / _C_M_PER_NS)
    for zen in np.radians([2.0, 5.0, 8.0, 20.0, 70.0, 89.5]):
        t = plane_wave_times(zen, 1.3, ant, ICE)
        for i, z in enumerate(ant[:4, 2]):
            ref = quad(lambda zz: np.sqrt((n_ice - delta_n * np.exp(zz / z_0)) ** 2 - np.sin(zen) ** 2),
                       z, 0.0, epsabs=1e-12, epsrel=1e-13)[0] / _C_M_PER_NS
            assert t[i] == pytest.approx(ref, abs=1e-9)
        assert t[4] == pytest.approx(-2.0 * np.cos(zen) / _C_M_PER_NS)
    shifted = plane_wave_times(np.radians(50.0), np.radians(30.0), ant + [10.0, 0.0, 0.0], ICE)
    base = plane_wave_times(np.radians(50.0), np.radians(30.0), ant, ICE)
    assert np.allclose(shifted - base, -10.0 * np.sin(np.radians(50.0)) * np.cos(np.radians(30.0)) / _C_M_PER_NS)


@pytest.mark.slow
def test_far_field_windows_hold_sky_delays(reco):
    """Plane-wave pair delays of random sky directions lie inside the far-field windows."""
    assert_windows_hold_sky_delays(reco)


@pytest.mark.slow
def test_far_field_windows_hold_layered_sky_delays(det, table_dir):
    """The same with the three-layer exponential profile named by the config."""
    r = InterferometricReco3D()
    r.begin(STATION, reference_config(table_dir, ice_model='greenland_3exp_layered'), det)
    assert len(r._far_field_geometry(VPOL_CHANNELS)[1]) == 3
    assert_windows_hold_sky_delays(r)


def assert_windows_hold_sky_delays(reco):
    """Assert that the far-field windows of a reconstruction hold the delays of random sky directions."""
    pairs = list(itertools.combinations(VPOL_CHANNELS, 2))
    windows = reco.far_field_lag_windows(pairs)
    ant, ice = reco._far_field_geometry(VPOL_CHANNELS)
    rng = np.random.default_rng(5)
    zen = np.concatenate([np.arccos(rng.uniform(0.0, 1.0, 20000)), [0.0, 0.5 * np.pi]])
    az = rng.uniform(0.0, 2 * np.pi, len(zen))
    times = plane_wave_times(zen, az, ant, ice)
    for k, (a, b) in enumerate(itertools.combinations(range(len(VPOL_CHANNELS)), 2)):
        delay = times[:, a] - times[:, b]
        assert np.all((delay >= windows[k, 0]) & (delay <= windows[k, 1])), pairs[k]


@pytest.mark.slow
@pytest.mark.airice
def test_tables_converge_to_the_plane_wave(det):
    """Air-ice table pair delays approach the plane-wave delays as the source recedes along a direction."""
    table_dir = os.environ.get('RECO3D_TEST_AIRICE_TABLES', '')
    if not os.path.isfile(os.path.join(table_dir, f'station{STATION}', f'st{STATION}_ch0_rz_table.npz')):
        pytest.skip('air-ice tables not found')
    config = reference_config(table_dir, coarse_limits=[1, 1600, 0, 360, -100, 300],
                              limits=[1, 1600, 0, 360, -100, 300], allow_above_surface=True)
    reco = InterferometricReco3D()
    reco.begin(STATION, config, det)
    ant, ice = reco._far_field_geometry(VPOL_CHANNELS)
    zen, az = np.radians(85.0), np.radians(70.0)
    pw = plane_wave_times(zen, az, ant, ice)
    rms = {}
    for dist in (150.0, 300.0, 600.0, 1200.0):
        tts = reco._compute_travel_times_single_point(dist * np.sin(zen), np.degrees(az), dist * np.cos(zen),
                                                      VPOL_CHANNELS)
        t = np.array([tts[ch] for ch in VPOL_CHANNELS])
        res = [(t[a] - t[b]) - (pw[a] - pw[b]) for a, b in itertools.combinations(range(len(t)), 2)
               if np.isfinite(t[a]) and np.isfinite(t[b])]
        assert len(res) >= 28, (dist, len(res))
        rms[dist] = float(np.sqrt(np.mean(np.square(res))))
    assert rms[300.0] < 0.6 * rms[150.0] and rms[600.0] < 0.6 * rms[300.0] and rms[1200.0] < 0.6 * rms[600.0], rms
    assert rms[1200.0] < 1.0, rms


@pytest.mark.slow
@pytest.mark.parametrize('direction', DIRECTIONS, ids=lambda d: f'zen{d[0]:g}_az{d[1]:g}')
def test_plane_wave_direction_recovered(reco, det, base_config, direction):
    """The far-field hypothesis recovers a plane wave and leaves every other field unchanged."""
    evt, stn = plane_wave_event(det, *direction, seed=int(direction[1]))
    off = reco.run(evt, stn, det, base_config)
    on = reco.run(evt, stn, det, dict(base_config, far_field_hypothesis=True))
    assert set(on) - set(off) == {f'far_{f}_v1' for f in FIELDS}
    for key, value in off.items():
        if '_time' not in key:
            assert same_value(value, on[key]), key
    sep = sky_separation(on['far_zen_v1'], on['far_az_v1'], *direction)
    assert sep < 0.1, (sep, on['far_zen_v1'], on['far_az_v1'], direction)
    assert on['far_corr_raw_v1'] > on['max_corr'], (on['far_corr_raw_v1'], on['max_corr'])
    assert on['far_map_snr_v1'] > 3.0 and on['far_origin_v1'] in (0, 1, 2)

    config = dict(base_config, far_field_hypothesis=True)
    pairs = reco.compute_pairs(stn, config, store=True)
    windows = reco.pair_lag_windows(pairs.pairs, config) + [-1.0, 1.0]
    stored = reco.reconstruct_from_pairs(cut_pairs(pairs, windows, np.float64), config)
    for key, value in on.items():
        if '_time' not in key:
            assert same_value(value, stored[key]), key


@pytest.mark.slow
def test_far_field_shared_across_volumes(reco, det, base_config):
    """Settings that differ only in the point-source volume search the far field once and get the values a
    fresh search gives; different pair weights search again."""
    evt, stn = plane_wave_event(det, 30.0, 40.0, seed=3)
    config = dict(base_config, far_field_hypothesis=True)
    shallow = dict(config, coarse_limits=[1, 250, 0, 360, -60, 0], limits=[1, 250, 0, 360, -60, 0])
    pairs = reco.compute_pairs(stn, config)
    far = lambda r: {k: v for k, v in r.items() if k.startswith('far_')}
    reco.reset_work()
    first = reco.reconstruct_from_pairs(pairs, config)
    shared = reco.reconstruct_from_pairs(pairs, shallow)
    work = reco.work_counts()
    assert work['far_searches'] == 1 and work['far_shared'] == 1, work
    reco._far_field_memo.clear()
    fresh = reco.reconstruct_from_pairs(pairs, shallow)
    assert far(shared) == far(fresh) == far(first)
    uniform = {p: 1.0 for p in pairs.pairs}
    reco.reset_work()
    reco.reconstruct_from_pairs(pairs, shallow, pair_weights=uniform)
    assert reco.work_counts()['far_searches'] == 1 and 'far_shared' not in reco.work_counts()


@pytest.mark.slow
@pytest.mark.parametrize('make_event', [plane_wave_event, burst_event], ids=['pulse', 'burst'])
def test_lobe_guard_recovers_plane_wave(reco, det, base_config, make_event):
    """With the lobe guard the far field finds a pulse and a burst within 0.1 deg and changes no other field."""
    direction = (60.0, 200.0)
    evt, stn = make_event(det, *direction, seed=7)
    config = dict(base_config, far_field_hypothesis=True)
    off = reco.run(evt, stn, det, config)
    on = reco.run(evt, stn, det, dict(config, far_field_lobe_guard_ns=0.5 * BURST_PERIOD_NS))
    assert set(on) == set(off)
    for key, value in off.items():
        if not key.startswith('far_') and '_time' not in key:
            assert same_value(value, on[key]), key
    sep = sky_separation(on['far_zen_v1'], on['far_az_v1'], *direction)
    assert sep < 0.1, (sep, on['far_zen_v1'], on['far_az_v1'])
    assert on['far_origin_v1'] == 2


@pytest.mark.slow
def test_lobe_guard_bounds_pair_delays(reco, det, base_config):
    """The guarded direction keeps each pair delay within the guard of the envelope peak (a vanishing guard)."""
    evt, stn = burst_event(det, 70.0, 100.0, snr=5.0, seed=3)
    config = dict(base_config, far_field_hypothesis=True)
    peak = reco.run(evt, stn, det, dict(config, far_field_lobe_guard_ns=1e-9))
    guard = 0.5 * BURST_PERIOD_NS
    on = reco.run(evt, stn, det, dict(config, far_field_lobe_guard_ns=guard))
    ant, ice = reco._far_field_geometry(VPOL_CHANNELS)
    t_peak, t_on = (plane_wave_times(np.radians(r['far_zen_v1']), np.radians(r['far_az_v1']), ant, ice)
                    for r in (peak, on))
    for a, b in itertools.combinations(range(len(VPOL_CHANNELS)), 2):
        assert abs((t_on[a] - t_on[b]) - (t_peak[a] - t_peak[b])) <= guard + 1e-9, (a, b)
    assert on['far_corr_raw_v1'] >= peak['far_corr_raw_v1']


@pytest.mark.parametrize('value', [0.0, -1.0, 'half', True])
def test_lobe_guard_validation(det, table_dir, value):
    """far_field_lobe_guard_ns must be a positive number or null."""
    with pytest.raises(ValueError, match='far_field_lobe_guard_ns'):
        InterferometricReco3D().begin(STATION, reference_config(table_dir, far_field_lobe_guard_ns=value), det)


def test_layered_profile_against_quadrature():
    """A layered profile integrates to an adaptive quadrature of its n(z); one layer equals the exponential form."""
    from NuRadioMC.utilities import medium
    model = medium.greenland_3exp_layered()
    profile = far_field_profile(model)
    assert len(profile) == 3
    ant = np.array([[0.0, 0.0, -97.0], [0.0, 0.0, -38.0], [0.0, 0.0, -14.9], [0.0, 0.0, -3.3], [0.0, 0.0, 2.0]])
    for zen in np.radians([0.0, 20.0, 50.0, 89.5]):
        t = plane_wave_times(zen, 0.7, ant, profile)
        for i, z in enumerate(ant[:4, 2]):
            ref = quad(lambda zz: np.sqrt(model.get_index_of_refraction(np.array([0.0, 0.0, zz])) ** 2
                                          - np.sin(zen) ** 2),
                       z, 0.0, points=[b for b in (-80.5, -14.9) if z < b] or None, epsabs=1e-12, epsrel=1e-13,
                       limit=200)[0] / _C_M_PER_NS
            assert t[i] == pytest.approx(ref, abs=1e-9), (np.degrees(zen), z)
        assert t[4] == pytest.approx(-2.0 * np.cos(zen) / _C_M_PER_NS)
    one = ((-np.inf, 0.0) + ICE,)
    for zen in np.radians([0.0, 30.0, 80.0]):
        assert np.allclose(plane_wave_times(zen, 0.2, ant, one), plane_wave_times(zen, 0.2, ant, ICE), rtol=0, atol=1e-9)


def test_far_field_ice_model_follows_config(reco, det, table_dir):
    """The record tables record no model: greenland_simple by default, the config's model when named, unknown names raise."""
    assert reco._far_field_geometry(VPOL_CHANNELS)[1] == ICE
    with pytest.raises(ValueError, match='ice_model'):
        InterferometricReco3D().begin(STATION, reference_config(table_dir, ice_model='no_such_model'), det)


def test_exact_gradient_refuses_a_layered_far_field(det, table_dir):
    """optimizer_gradient 'exact' with the far field and a layered ice model is refused (its kernel takes one exponential)."""
    with pytest.raises(ValueError, match="optimizer_gradient 'exact'"):
        InterferometricReco3D().begin(STATION, reference_config(
            table_dir, far_field_hypothesis=True, optimizer_gradient='exact', ice_model='greenland_3exp_layered'), det)


@pytest.mark.airice
def test_recorded_ice_model_wins_and_must_match(det):
    """Tables that record their ice model set the far field's n(z); a config naming another model raises."""
    path = os.environ.get('RECO3D_TEST_AIRICE_TABLES', '')
    if not all(os.path.isfile(os.path.join(path, f'station{STATION}', f'st{STATION}_ch{ch}_rz_table.npz'))
               for ch in VPOL_CHANNELS):
        pytest.skip(f'air-ice tables for every VPol channel not found under {path}')
    r = InterferometricReco3D()
    r.begin(STATION, reference_config(path, ice_model='greenland_simple'), det)
    assert r._far_field_geometry(VPOL_CHANNELS)[1] == ICE
    with pytest.raises(ValueError, match='ice_model'):
        InterferometricReco3D().begin(STATION, reference_config(path, ice_model='greenland_3exp_layered'), det)


@pytest.mark.parametrize('model', ['greenland_simple', 'greenland_simple_layered'])
def test_fused_fd_points_and_times_bit_equal(model):
    """The numba forward-difference points and plane-wave times of the far-field L-BFGS-B equal the numpy ones bit for bit."""
    from NuRadioMC.utilities import medium
    from NuRadioReco.modules.interferometricDirectionReconstruction3D import (
        _LBFGSB_ABS_STEP, _LBFGSB_REL_STEP, _lbfgsb_fd_steps, _plane_wave_arrays)
    from NuRadioReco.utilities.reco3d_kernels import _far_fd_points_numba, _plane_wave_times_numba
    ice = far_field_profile(medium.get_ice_model(model))
    rng = np.random.default_rng(5)
    ant = np.column_stack([rng.uniform(-40, 40, 12), rng.uniform(-40, 40, 12), rng.uniform(-100, 3, 12)])
    ant[0, 2] = 0.0
    arrays = _plane_wave_arrays(ant, ice)
    for _ in range(3000):
        start = rng.uniform(-10, 370)
        lb, ub = np.array([0.0, start - 3.0]), np.array([90.0, start + 3.0])
        zen = rng.choice([rng.uniform(0, 90), 0.0, 90.0, 90.0 - 1e-9, 1e-12])
        x = np.array([zen, start + rng.uniform(-3, 3)])
        h = _lbfgsb_fd_steps(x, lb, ub)
        ref_points = np.repeat(x[None, :], 3, axis=0) + np.vstack([np.zeros(2), np.diag(h)])
        ref_tts = plane_wave_times(np.radians(ref_points[:, 0]), np.radians(ref_points[:, 1]), ant, ice)
        points, zen3, az3 = _far_fd_points_numba(x, lb, ub, _LBFGSB_ABS_STEP, _LBFGSB_REL_STEP)
        tts = _plane_wave_times_numba(np.sin(zen3), np.cos(zen3), np.cos(az3), np.sin(az3), *arrays, _C_M_PER_NS)
        assert points.tobytes() == ref_points.tobytes()
        assert tts.tobytes() == ref_tts.tobytes()
