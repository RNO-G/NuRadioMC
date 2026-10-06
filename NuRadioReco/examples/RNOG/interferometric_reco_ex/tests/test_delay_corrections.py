"""The per-channel delay-corrections step of channelPreprocessor.

A corrections file adds `delta_ns` to the database cable delay of a channel; the step
shifts that channel's trace start time by minus delta right after the cable-delay
subtraction, negates the traces of the channels its optional polarity block marks -1, and
touches nothing else. The pair-store trial of one file against another is their difference. The file must carry provenance and an
uncertainty per corrected channel, the event must fall inside the validity window
(narrowed per station by a station's own dates in derived_from),
unknown preprocessor keys raise instead of being dropped, and the simulation data
provider drops the keys because simulated events share the tables' description.
"""

import datetime
import hashlib

import numpy as np
import pytest
import yaml

from conftest import STATION
from synthetic import RECORD_PREPROCESSOR, VPOL_CHANNELS, filtered_trace
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.RNO_G.channelPreprocessor import channelPreprocessor, load_delay_corrections
from NuRadioReco.utilities import units

EVENT_TIME = datetime.datetime(2022, 7, 1, 12, 0, 0)
NATIVE_RATE = 3.2 * units.GHz


def _doc(corrections, uncertainty=None, **overrides):
    """A valid corrections document with the given blocks."""
    doc = dict(method='test', date='2026-09-30', derived_from={'21': 'run476'}, valid_from='2022-01-01',
               valid_to='2022-12-31', corrections=corrections,
               uncertainty_ns=uncertainty if uncertainty is not None else
               {s: {ch: 0.3 for ch in chs} for s, chs in corrections.items()})
    doc.update(overrides)
    return doc


def _write(tmp_path, doc, name='corr.yaml'):
    """Write a document and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc))
    return str(path)


def _station(channels, seed=0, station_time=EVENT_TIME):
    """Native-rate noise event on the given channels with a station time."""
    rng = np.random.default_rng(seed)
    evt = Event(0, seed)
    stn = Station(STATION)
    stn.set_station_time(station_time)
    for ch in channels:
        c = Channel(ch)
        c.set_trace(filtered_trace(200.0, 1.0, 0.2, rng), NATIVE_RATE, trace_start_time=5.0 * ch)
        stn.add_channel(c)
    evt.set_station(stn)
    return evt, stn


def _snapshot(stn):
    """Traces and start times of every channel."""
    return {ch: (stn.get_channel(ch).get_trace().copy(), stn.get_channel(ch).get_trace_start_time())
            for ch in stn.get_channel_ids()}


def _preprocess(det, evt, stn, **config):
    """Run the preprocessor with block offsets off so that only the timing steps act."""
    pre = channelPreprocessor()
    pre.begin(dict(apply_block_offset_removal=False, **config))
    pre.run(evt, stn, det)
    return pre


def test_load_validates_and_hashes(tmp_path):
    """A valid file loads with int keys, float values, dates and the SHA-256 of its bytes."""
    path = _write(tmp_path, _doc({21: {9: -4.4, 10: -4.3}, 23: {5: -2.8}}))
    dc = load_delay_corrections(path)
    assert dc.corrections == {21: {9: -4.4, 10: -4.3}, 23: {5: -2.8}}
    assert dc.uncertainty_ns[21][9] == 0.3
    assert dc.valid_from == datetime.date(2022, 1, 1) and dc.valid_to == datetime.date(2022, 12, 31)
    assert dc.sha256 == hashlib.sha256(open(path, 'rb').read()).hexdigest()
    assert dc.provenance['method'] == 'test' and dc.path == path


def test_load_rejects_incomplete_files(tmp_path):
    """Missing corrections, provenance or uncertainties raise."""
    with pytest.raises(ValueError, match='corrections'):
        load_delay_corrections(_write(tmp_path, {'method': 'x'}, 'a.yaml'))
    doc = _doc({21: {9: -4.4}})
    del doc['valid_to']
    with pytest.raises(ValueError, match='provenance'):
        load_delay_corrections(_write(tmp_path, doc, 'b.yaml'))
    with pytest.raises(ValueError, match='uncertainty'):
        load_delay_corrections(_write(tmp_path, _doc({21: {9: -4.4, 10: -4.3}}, {21: {9: 0.3}}), 'c.yaml'))
    with pytest.raises(ValueError):
        load_delay_corrections(_write(tmp_path, _doc({21: [9, 10]}, {}), 'd.yaml'))


def test_unknown_preprocessor_keys_raise():
    """A misspelt or foreign key is an error, not a silent no-op; the notch keys are known."""
    channelPreprocessor().begin({'apply_notch': True, 'notch_bands': [[0.399, 0.407]]})
    with pytest.raises(ValueError, match='apply_bandpas'):
        channelPreprocessor().begin({'apply_bandpas': True})


def test_apply_needs_a_file_and_the_cable_delay_step(tmp_path):
    """Corrections without a file, or without the cable-delay step they correct, raise."""
    with pytest.raises(ValueError, match='delay_corrections_file'):
        channelPreprocessor().begin({'apply_delay_corrections': True})
    path = _write(tmp_path, _doc({23: {5: -2.8}}))
    with pytest.raises(ValueError, match='apply_cable_delay'):
        channelPreprocessor().begin({'apply_delay_corrections': True, 'delay_corrections_file': path,
                                     'apply_cable_delay': False})


def test_step_shifts_only_the_listed_channels(det, tmp_path):
    """delta_ns = -2.8 on channel 5 moves its start time by +2.8 ns and nothing else."""
    path = _write(tmp_path, _doc({23: {5: -2.8, 9: 1.5}}))
    evt_a, stn_a = _station(VPOL_CHANNELS)
    evt_b, stn_b = _station(VPOL_CHANNELS)
    _preprocess(det, evt_a, stn_a)
    pre = _preprocess(det, evt_b, stn_b, apply_delay_corrections=True, delay_corrections_file=path)
    assert pre.delay_corrections is not None and pre.delay_corrections.corrections[23] == {5: -2.8, 9: 1.5}
    before, after = _snapshot(stn_a), _snapshot(stn_b)
    for ch in VPOL_CHANNELS:
        assert np.array_equal(before[ch][0], after[ch][0])
        expected = {5: 2.8, 9: -1.5}.get(ch, 0.0)
        assert abs((after[ch][1] - before[ch][1]) - expected) < 1e-9, (ch, before[ch][1], after[ch][1])


def test_step_matches_cable_delay_sign(det, tmp_path):
    """A correction equal to minus the database delay undoes the cable-delay subtraction exactly."""
    delay = float(det.get_cable_delay(STATION, 9))
    path = _write(tmp_path, _doc({23: {9: -delay}}))
    evt, stn = _station([9])
    start = stn.get_channel(9).get_trace_start_time()
    _preprocess(det, evt, stn, apply_delay_corrections=True, delay_corrections_file=path)
    assert abs(stn.get_channel(9).get_trace_start_time() - start) < 1e-9


def test_stations_absent_from_the_file_are_untouched(det, tmp_path):
    """A file that lists another station leaves this station's timing alone."""
    path = _write(tmp_path, _doc({21: {9: -4.4}}))
    evt_a, stn_a = _station([0, 9])
    evt_b, stn_b = _station([0, 9])
    _preprocess(det, evt_a, stn_a)
    _preprocess(det, evt_b, stn_b, apply_delay_corrections=True, delay_corrections_file=path)
    assert _snapshot(stn_a)[9][1] == _snapshot(stn_b)[9][1]


def test_events_outside_the_validity_window_raise(det, tmp_path):
    """An event dated outside valid_from to valid_to is refused."""
    path = _write(tmp_path, _doc({23: {5: -2.8}}))
    evt, stn = _station([5], station_time=datetime.datetime(2023, 3, 1))
    with pytest.raises(ValueError, match='valid'):
        _preprocess(det, evt, stn, apply_delay_corrections=True, delay_corrections_file=path)
    evt, stn = _station([5], station_time=None)
    with pytest.raises(ValueError, match='station time'):
        _preprocess(det, evt, stn, apply_delay_corrections=True, delay_corrections_file=path)


def test_station_windows_in_derived_from_narrow_the_file_window(det, tmp_path):
    """A station's own valid_from and valid_to in derived_from narrow the file window for that station alone."""
    derived = {STATION: {'run': 999, 'valid_from': '2022-07-04', 'valid_to': '2022-12-31'}, '21': 'run476'}
    path = _write(tmp_path, _doc({STATION: {5: -2.8}, 21: {9: -4.4}}, derived_from=derived))
    dc = load_delay_corrections(path)
    assert dc.station_windows == {STATION: (datetime.date(2022, 7, 4), datetime.date(2022, 12, 31)),
                                  21: (datetime.date(2022, 1, 1), datetime.date(2022, 12, 31))}
    evt, stn = _station([5])
    with pytest.raises(ValueError, match=f'station {STATION}'):
        _preprocess(det, evt, stn, apply_delay_corrections=True, delay_corrections_file=path)
    evt_a, stn_a = _station([5], station_time=datetime.datetime(2022, 7, 4, 0, 0, 1))
    evt_b, stn_b = _station([5], station_time=datetime.datetime(2022, 7, 4, 0, 0, 1))
    _preprocess(det, evt_a, stn_a)
    _preprocess(det, evt_b, stn_b, apply_delay_corrections=True, delay_corrections_file=path)
    assert abs(stn_b.get_channel(5).get_trace_start_time() - stn_a.get_channel(5).get_trace_start_time() - 2.8) < 1e-9
    wide = _doc({STATION: {5: -2.8}}, derived_from={STATION: {'valid_from': '2021-01-01', 'valid_to': '2023-12-31'}})
    assert load_delay_corrections(_write(tmp_path, wide, 'wide.yaml')).station_windows[STATION] == (
        datetime.date(2022, 1, 1), datetime.date(2022, 12, 31))
    disjoint = _doc({STATION: {5: -2.8}}, derived_from={STATION: {'valid_from': '2023-01-01'}})
    with pytest.raises(ValueError, match='overlap'):
        load_delay_corrections(_write(tmp_path, disjoint, 'disjoint.yaml'))


def test_simulation_provider_ignores_the_corrections(det, tmp_path):
    """dataProviderNuRadio drops the keys, so a simulated event is not shifted."""
    from NuRadioReco.modules.io.eventWriter import eventWriter
    from NuRadioReco.modules.RNO_G.dataProviderNuRadio import dataProviderNuRadio
    path = _write(tmp_path, _doc({23: {5: -2.8}}))
    evt, stn = _station([0, 5], station_time=datetime.datetime(2022, 10, 1))
    start = stn.get_channel(5).get_trace_start_time()
    writer = eventWriter()
    writer.begin(str(tmp_path / 'sim.nur'))
    writer.run(evt)
    writer.end()
    config = dict(RECORD_PREPROCESSOR, apply_upsampling=False, apply_delay_corrections=True,
                  delay_corrections_file=path)
    dp = dataProviderNuRadio()
    dp.begin(str(tmp_path / 'sim.nur'), det, preprocessor_config=config)
    assert dp.preprocessor.delay_corrections is None
    assert dp.preprocessor._config['apply_delay_corrections'] is False
    out = dp.get_event(0, 0)
    delay = float(det.get_cable_delay(STATION, 5))
    assert abs(out.get_station(STATION).get_channel(5).get_trace_start_time() - (start - delay)) < 1e-9
    dp.end()


def test_polarity_block_loads_and_validates(tmp_path):
    """The optional polarity block loads as ints, gives polarity-only stations a window and rejects other values."""
    dc = load_delay_corrections(_write(tmp_path, _doc({23: {5: -2.8}}, polarity={23: {6: -1, 7: -1}, 13: {6: 1}})))
    assert dc.polarity == {23: {6: -1, 7: -1}, 13: {6: 1}}
    assert set(dc.station_windows) == {13, 23}
    assert load_delay_corrections(_write(tmp_path, _doc({23: {5: -2.8}}), 'b.yaml')).polarity == {}
    with pytest.raises(ValueError, match='polarity'):
        load_delay_corrections(_write(tmp_path, _doc({23: {5: -2.8}}, polarity={23: {6: 0.5}}), 'c.yaml'))


def test_step_negates_inverted_channels(det, tmp_path):
    """Polarity -1 negates the channel's trace exactly, keeps its timing and leaves the other channels alone."""
    path = _write(tmp_path, _doc({STATION: {5: -2.8}}, polarity={STATION: {6: -1, 7: 1}}))
    evt, stn = _station(VPOL_CHANNELS)
    _preprocess(det, evt, stn, apply_cable_delay=True, apply_delay_corrections=True, delay_corrections_file=path)
    evt_ref, stn_ref = _station(VPOL_CHANNELS)
    _preprocess(det, evt_ref, stn_ref, apply_cable_delay=True)
    after, plain = _snapshot(stn), _snapshot(stn_ref)
    assert np.array_equal(after[6][0], -plain[6][0]) and after[6][1] == plain[6][1]
    assert np.array_equal(after[7][0], plain[7][0])
    assert after[5][1] == plain[5][1] + 2.8 and np.array_equal(after[5][0], plain[5][0])
    for ch in VPOL_CHANNELS:
        if ch not in (5, 6):
            assert np.array_equal(after[ch][0], plain[ch][0]) and after[ch][1] == plain[ch][1]


def test_calibration_trial_is_the_file_difference(tmp_path):
    """The trial shifts are the correction differences and the polarities the products, per station."""
    from pair_store import calibration_trial
    applied = load_delay_corrections(_write(tmp_path, _doc({23: {5: -2.8, 9: 1.0}}, polarity={23: {6: -1}}), 'a.yaml'))
    trial = load_delay_corrections(_write(tmp_path, _doc({23: {5: -2.0, 22: 9.0}}, polarity={23: {6: -1, 7: -1}}), 'b.yaml'))
    shift, sign = calibration_trial(applied, trial, 23)
    assert shift == pytest.approx({5: 0.8, 9: -1.0, 22: 9.0}) and sign == {7: -1}
    assert calibration_trial(None, trial, 23) == ({5: -2.0, 22: 9.0}, {6: -1, 7: -1})
    assert calibration_trial(applied, None, 23) == ({5: 2.8, 9: -1.0}, {6: -1})
    assert calibration_trial(applied, trial, 13) == ({}, {})


def test_position_shift_block_loads_and_validates(tmp_path):
    """The optional position_shift block loads as (dx, dy) in m, gives its stations a window and rejects other values."""
    doc = _doc({23: {5: -2.8}}, position_shift={23: {22: [1.25, -0.5]}, 13: {9: [1, 0]}})
    dc = load_delay_corrections(_write(tmp_path, doc))
    assert dc.position_shift == {23: {22: (1.25, -0.5)}, 13: {9: (1.0, 0.0)}}
    assert set(dc.station_windows) == {13, 23}
    assert load_delay_corrections(_write(tmp_path, _doc({23: {5: -2.8}}), 'b.yaml')).position_shift == {}
    for k, bad in enumerate(([1.0], [1.0, 2.0, 0.5], ['a', 1.0], [float('nan'), 0.0], 1.0)):
        with pytest.raises(ValueError, match='position_shift'):
            load_delay_corrections(_write(tmp_path, _doc({23: {5: -2.8}}, position_shift={23: {22: bad}}), f'c{k}.yaml'))
