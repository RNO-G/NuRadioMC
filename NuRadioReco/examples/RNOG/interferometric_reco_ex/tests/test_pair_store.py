"""The two-stage reconstruction: pair series (compute_pairs) and search on them (reconstruct_from_pairs).

`run` of a hierarchical configuration is the search on the series of `compute_pairs`, so
the search on the pair series of every channel computed at once (as the pair store keeps
them) must equal `run` bit for bit, for the record and the recommended candidate
configurations, with polarization groups and under the HPol sign modes. The store keeps
each series only over the lags the tables can reach in the configured volume: that cut
leaves every result unchanged (float64 store), the float32 store reproduces the search
the driver runs on the rounded series exactly, and the lag windows hold every delay of
sources drawn anywhere in the volume. A channel mask equals the configuration without
those channels, a polarity of -1 equals negating the channel's trace, and a delay shift
equals moving the channel's trace start time by minus the shift (the delay-corrections
convention), within float rounding of the lag offsets.
"""

import itertools

import numpy as np
import pytest

from conftest import STATION, reference_config
from synthetic import (HPOL_CHANNELS, VPOL_CHANNELS, TravelTimeTables, angular_separation,
                       cylindrical_to_enu, make_event, make_noise_event, same_value)
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from pair_store import PairStore, PairStoreWriter, config_pair_weights, cut_pairs

CHANNELS = VPOL_CHANNELS + HPOL_CHANNELS
GROUPS = {'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS}
SOURCES = [(60.0, 40.0, -30.0), (80.0, 120.0, -40.0)]
CANDIDATE_KEYS = dict(candidate_search=['envelope:traces', 'envelope:correlation', 'raw'],
                      candidate_tie_band=0.0054, candidate_tie_band_max_raw_corr=0.035,
                      candidate_fill_saved_peaks=True, candidate_diagnostics=True)


def _configs(table_dir):
    """Record and recommended configurations on the VPol channels and with both polarization groups."""
    groups = dict(channels=CHANNELS, polarization_groups=GROUPS)
    return {
        'record_vpol': reference_config(table_dir, validation=True),
        'recommended_vpol': reference_config(table_dir, validation=True, **CANDIDATE_KEYS),
        'record_groups': reference_config(table_dir, **groups),
        'recommended_groups': reference_config(table_dir, **groups, **CANDIDATE_KEYS),
        'hpol_joint': reference_config(table_dir, **groups, hpol_sign_mode='joint_sign',
                                       snr_window_ns=1.5),
        'recommended_groups_regions': reference_config(table_dir, **groups, **CANDIDATE_KEYS,
                                                       region_hypotheses=True),
    }


@pytest.fixture(scope='module')
def configs(table_dir):
    """Configurations compared in this module."""
    return _configs(table_dir)


@pytest.fixture(scope='module')
def all_reco(det, table_dir):
    """Reconstruction object initialised with the 15-channel configuration."""
    reco = InterferometricReco3D()
    reco.begin(STATION, reference_config(table_dir, channels=CHANNELS), det)
    return reco


@pytest.fixture(scope='module')
def all_tables(table_dir):
    """Combined travel-time tables for the VPol and HPol channels."""
    return TravelTimeTables(table_dir, STATION, CHANNELS)


@pytest.fixture(scope='module')
def events(det, all_tables, pa):
    """Two 15-channel source events and one noise event."""
    out = [make_event(det, STATION, cylindrical_to_enu(*src, pa), CHANNELS, all_tables, snr=12.0,
                      seed=11 + i)[:2] for i, src in enumerate(SOURCES)]
    out.append(make_noise_event(STATION, CHANNELS, seed=5))
    return out


def _assert_same(a, b, context):
    """Assert two result dicts carry the same keys and equal values, timings (any key with '_time') excepted."""
    keys = {k for k in a if '_time' not in k}
    assert keys == {k for k in b if '_time' not in k}, (context, keys ^ set(b))
    for k in keys:
        assert same_value(a[k], b[k]), (context, k, a[k], b[k])


def _store_config(config):
    """The configuration the pair store is made with: every channel, every pair."""
    return dict(config, channels=CHANNELS)


@pytest.mark.slow
@pytest.mark.parametrize('name', ['record_vpol', 'recommended_vpol', 'record_groups',
                                  'recommended_groups', 'hpol_joint', 'recommended_groups_regions'])
def test_search_on_all_pairs_equals_run(all_reco, det, configs, events, name):
    """The search on the store's series of every pair equals run() bit for bit."""
    config = configs[name]
    for i, (evt, stn) in enumerate(events):
        expected = all_reco.run(evt, stn, det, config)
        pairs = all_reco.compute_pairs(stn, _store_config(config), store=True)
        _assert_same(expected, all_reco.reconstruct_from_pairs(pairs, config), (name, i))


@pytest.mark.slow
@pytest.mark.parametrize('name', ['record_groups', 'recommended_groups'])
def test_store_round_trip_and_cut_invariance(all_reco, det, configs, events, tmp_path, name):
    """A float64 store reproduces the uncut search; a float32 store reproduces the search on the rounded cut series."""
    config = configs[name]
    store_config = dict(_store_config(config), station_id=STATION)
    windows = all_reco.pair_lag_windows(list(itertools.combinations(CHANNELS, 2)), config) + [-5.0, 5.0]
    for dtype in ('float64', 'float32'):
        path = tmp_path / f'{name}_{dtype}.h5'
        writer = PairStoreWriter(str(path), store_config, windows, 5.0, dtype)
        expected = []
        for evt, stn in events:
            full = all_reco.compute_pairs(stn, store_config, store=True)
            cut = cut_pairs(full, windows, dtype)
            res_cut = all_reco.reconstruct_from_pairs(cut, config)
            if dtype == 'float64':
                _assert_same(all_reco.reconstruct_from_pairs(full, config), res_cut, (name, 'cut'))
            expected.append(res_cut)
            writer.append(cut, evt.get_run_number(), evt.get_id(), 'synthetic', STATION, np.nan,
                          config_pair_weights(all_reco, cut, config))
        writer.close({'mode': 'hw'})
        with PairStore(str(path)) as store:
            assert len(store) == len(events)
            assert store.config['channels'] == CHANNELS
            for i, res in enumerate(expected):
                _assert_same(res, all_reco.reconstruct_from_pairs(store.event(i), config),
                             (name, dtype, i))
                weights = store.pair_weights(i)
                assert np.isfinite(weights[(0, 1)]) and np.isnan(weights[(0, 4)])


@pytest.mark.slow
def test_lag_windows_hold_every_reachable_delay(all_reco, configs):
    """Pair delays at random sources in the volume lie inside the lag windows, which are narrow for close pairs."""
    config = configs['record_vpol']
    pairs = list(itertools.combinations(CHANNELS, 2))
    windows = all_reco.pair_lag_windows(pairs, config)
    rng = np.random.default_rng(3)
    lo, hi = config['coarse_limits'][0], config['coarse_limits'][1]
    rho = np.concatenate([rng.uniform(lo, hi, 3000), np.full(200, float(lo)), np.full(200, float(hi))])
    phi = rng.uniform(0.0, 360.0, len(rho))
    z = np.concatenate([rng.uniform(-100.0, 0.0, len(rho) - 400), rng.choice([-100.0, 0.0], 400)])
    n_checked = 0
    for r, p, zz in zip(rho, phi, z):
        tts = all_reco._compute_travel_times_single_point(r, p, zz, CHANNELS)
        for k, (a, b) in enumerate(pairs):
            if np.isfinite(tts[a]) and np.isfinite(tts[b]) and tts[a] > 0 and tts[b] > 0:
                delay = tts[a] - tts[b]
                assert windows[k, 0] <= delay <= windows[k, 1], ((a, b), (r, p, zz), delay, windows[k])
                n_checked += 1
    assert n_checked > 100000
    widths = windows[:, 1] - windows[:, 0]
    print('lag window widths (ns): PA pair 0-1 %.1f, median %.1f, max %.1f' % (
        widths[pairs.index((0, 1))], np.median(widths), widths.max()))
    assert widths[pairs.index((0, 1))] < 60.0, windows[pairs.index((0, 1))]


@pytest.mark.slow
def test_cut_beyond_the_margin_is_refused(all_reco, configs, events):
    """A delay shift larger than the stored margin is refused instead of reading cut-away lags."""
    config = configs['record_vpol']
    pairs_full = all_reco.compute_pairs(events[0][1], config, store=True)
    windows = all_reco.pair_lag_windows(pairs_full.pairs, config) + [-2.0, 2.0]
    cut = cut_pairs(pairs_full, windows, 'float64')
    all_reco.reconstruct_from_pairs(cut, config, channel_delay_shift={5: 1.5})
    with pytest.raises(ValueError, match='lag windows'):
        all_reco.reconstruct_from_pairs(cut, config, channel_delay_shift={5: 3.0})


@pytest.mark.slow
def test_channel_mask_equals_config_without_the_channels(all_reco, det, configs, events):
    """Masking channels equals running the configuration without them."""
    for name in ('recommended_vpol', 'record_groups'):
        config = configs[name]
        evt, stn = events[0]
        pairs = all_reco.compute_pairs(stn, _store_config(config), store=True)
        masked = dict(config, channels=[ch for ch in config['channels'] if ch not in (6, 7, 21)])
        _assert_same(all_reco.run(evt, stn, det, masked),
                     all_reco.reconstruct_from_pairs(pairs, config, channel_mask=[6, 7, 21]), name)


@pytest.mark.slow
def test_polarity_equals_negated_trace(all_reco, det, configs, events, pa):
    """A polarity of -1 on a channel equals negating its trace; on inverted pulses it restores the source."""
    config = configs['recommended_vpol']
    evt, stn = events[0]
    pairs = all_reco.compute_pairs(stn, config, store=True)
    for ch in (6, 7):
        channel = stn.get_channel(ch)
        channel.set_trace(-channel.get_trace(), channel.get_sampling_rate())
    flipped = all_reco.reconstruct_from_pairs(all_reco.compute_pairs(stn, config, store=True),
                                              config)
    for ch in (6, 7):
        channel = stn.get_channel(ch)
        channel.set_trace(-channel.get_trace(), channel.get_sampling_rate())
    _assert_same(flipped, all_reco.reconstruct_from_pairs(
        pairs, config, channel_polarity={6: -1, 7: -1}), 'polarity')

    src = (90.0, 300.0, -60.0)
    tables = TravelTimeTables(config['time_delay_tables'], STATION, VPOL_CHANNELS)
    _, stn_inv, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables,
                               snr=20.0, seed=4, invert=(6, 7, 9, 10))
    pairs_inv = all_reco.compute_pairs(stn_inv, config, store=True)
    plain = all_reco.reconstruct_from_pairs(pairs_inv, config)
    fixed = all_reco.reconstruct_from_pairs(pairs_inv, config,
                                            channel_polarity={6: -1, 7: -1, 9: -1, 10: -1})
    sep = angular_separation((fixed['rho'], fixed['phi'], fixed['z']), src, pa)
    assert fixed['max_corr'] > plain['max_corr'] + 0.05, (plain['max_corr'], fixed['max_corr'])
    assert sep < 1.0, (sep, fixed)


@pytest.mark.slow
def test_delay_shift_equals_shifted_trace_start(all_reco, det, configs, events):
    """A delay shift s equals moving the channel's trace start time by -s, within offset rounding.

    The lag offsets agree to float rounding (about 1e-13 ns), which moves the
    finite-difference optimizer's end point by a few 1e-7 in correlation.
    """
    config = configs['recommended_vpol']
    evt, stn = events[0]
    reference = all_reco.reconstruct_from_pairs(all_reco.compute_pairs(stn, config), config)
    shifts = {5: 2.5, 9: -1.25}
    starts = {ch: stn.get_channel(ch).get_trace_start_time() for ch in shifts}
    for ch, s in shifts.items():
        stn.get_channel(ch).add_trace_start_time(s)
    late = all_reco.compute_pairs(stn, config)
    for ch, start in starts.items():
        stn.get_channel(ch).set_trace_start_time(start)
    corrected = all_reco.reconstruct_from_pairs(late, config, channel_delay_shift=shifts)
    wrong = all_reco.reconstruct_from_pairs(late, config, channel_delay_shift={5: -2.5, 9: 1.25})
    for k in ('rho', 'phi', 'z'):
        assert abs(corrected[k] - reference[k]) < 1e-2, (k, reference[k], corrected[k])
    assert abs(corrected['max_corr'] - reference['max_corr']) < 1e-5
    assert wrong['max_corr'] < reference['max_corr'] - 1e-3


@pytest.mark.slow
def test_mismatched_series_settings_are_refused(all_reco, configs, events):
    """The search refuses series made with another Hann or normalisation setting or lacking an envelope mode."""
    config = configs['record_vpol']
    pairs = all_reco.compute_pairs(events[0][1], config)
    with pytest.raises(ValueError, match='settings differ'):
        all_reco.reconstruct_from_pairs(pairs, dict(config, apply_hann_window=False))
    with pytest.raises(ValueError, match='lacks'):
        all_reco.reconstruct_from_pairs(pairs, configs['recommended_vpol'])


@pytest.mark.slow
def test_polarity_leaves_abs_pairs_unchanged(all_reco, det, table_dir, events):
    """A channel polarity does not touch pairs scored by their absolute value, with or without polarization groups.

    All pairs of channel 21 (helper C HPol) cross strings, so under ``abs_cross_string`` the
    HPol group scores every one of them by its absolute value and a polarity of -1 on
    channel 21 leaves both groups' results unchanged; on a single group with every pair
    ``abs`` no polarity changes anything.
    """
    evt, stn = events[0]
    groups = reference_config(table_dir, channels=CHANNELS, polarization_groups=GROUPS,
                              hpol_sign_mode='abs_cross_string')
    pairs = all_reco.compute_pairs(stn, groups, store=True)
    _assert_same(all_reco.reconstruct_from_pairs(pairs, groups),
                 all_reco.reconstruct_from_pairs(pairs, groups, channel_polarity={21: -1}), 'groups')
    n_pairs = len(VPOL_CHANNELS) * (len(VPOL_CHANNELS) - 1) // 2
    single = reference_config(table_dir, pair_signs=['abs'] * n_pairs, **CANDIDATE_KEYS)
    pairs = all_reco.compute_pairs(stn, single, store=True)
    _assert_same(all_reco.reconstruct_from_pairs(pairs, single),
                 all_reco.reconstruct_from_pairs(pairs, single, channel_polarity={6: -1, 9: -1}), 'single')
