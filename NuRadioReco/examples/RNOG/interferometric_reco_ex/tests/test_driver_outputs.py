"""Output files of the driver, run from its command line on a NUR file of synthetic events.

With `save_coherent_waveforms` and polarization groups the driver finishes with and
without `--save-nur`: the results file is complete and carries every group's waveforms,
and the NUR file holds the waveforms of the primary result as channels 100 and above.
Without stored waveforms `--save-nur` still writes its file, which the event reader opens
and which holds no event; the event writer does this on request only.
"""

import sys

import h5py
import numpy as np
import pytest
import yaml

import interferometric_reco_3d_advanced as driver
from conftest import DETECTOR_DATE, STATION, load_station_detector, reference_config, station_detector_file
from reco_output import COHERENT_CHANNEL_BASE, COHERENT_GROUP
from synthetic import (HPOL_CHANNELS, VPOL_CHANNELS, TravelTimeTables, antenna_locations, cylindrical_to_enu,
                       make_event, pa_center)
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.utilities import units

SOURCE = (80.0, 120.0, -40.0)
CHANNELS = VPOL_CHANNELS + HPOL_CHANNELS
SEEDS = (51, 52)


@pytest.fixture(scope='module')
def inputs(table_dir, tmp_path_factory):
    """NUR file with two synthetic 15-channel events and the driver config that reads them unchanged."""
    det = load_station_detector(STATION)
    tables = TravelTimeTables(table_dir, STATION, CHANNELS)
    src = cylindrical_to_enu(*SOURCE, pa_center(antenna_locations(det, STATION)))
    path = str(tmp_path_factory.mktemp('driver') / 'events.nur')
    writer = eventWriter()
    writer.begin(path)
    for seed in SEEDS:
        evt, stn, _ = make_event(det, STATION, src, CHANNELS, tables, snr=30.0, seed=seed)
        stn.set_station_time(DETECTOR_DATE)
        writer.run(evt)
    writer.end()
    config = reference_config(
        table_dir, channels=CHANNELS, polarization_groups={'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS},
        station_id=STATION, detector_file=station_detector_file(STATION), n_coherent_waveforms=3,
        preprocessor=dict(apply_block_offset_removal=False, apply_cable_delay=False))
    return path, config


def _drive(monkeypatch, tmp_path, nur_input, config, save_nur):
    """Run the driver's `main` on the NUR file; returns the paths of its results file and its `--save-nur` file."""
    config_path, out, nur = (tmp_path / name for name in ('config.yaml', 'reco.h5', 'coherent.nur'))
    config_path.write_text(yaml.safe_dump(config))
    argv = ['driver', '--config', str(config_path), '--input', nur_input, '--outputfile', str(out),
            '--mode', 'hw', '--validation']
    monkeypatch.setattr(sys, 'argv', argv + (['--save-nur', str(nur)] if save_nur else []))
    driver.main()
    return out, nur


def _nur_events(path):
    """Events of a NUR file."""
    reader = eventReader()
    reader.begin(str(path))
    return list(reader.run())


@pytest.mark.slow
@pytest.mark.parametrize('save_nur', [True, False])
def test_stored_waveforms_with_polarization_groups(inputs, monkeypatch, tmp_path, save_nur):
    """The driver finishes with and without `--save-nur` and writes every group's waveforms."""
    nur_input, config = inputs
    out, nur = _drive(monkeypatch, tmp_path, nur_input, dict(config, save_coherent_waveforms=True), save_nur)
    with h5py.File(out) as f:
        assert f.attrs['n_events'] == 2 and list(f['results']['event_number'][:]) == list(SEEDS)
        assert f['results']['rho_hpol'].shape == (2,) and not any(k.startswith('coherent') for k in f['results'].keys())
        g = f[COHERENT_GROUP]
        assert {'times', 'times_vpol', 'times_hpol', 'peak_0', 'peak_0_vpol', 'peak_0_hpol'} <= set(g.keys())
        assert g['peak_0'].shape == (2, g['times'].shape[0]) and g['peak_0_hpol'].shape == (2, g['times_hpol'].shape[0])
        primary = {name: g[name][:] for name in g if name[len('peak_'):].isdigit()}
    assert nur.exists() == save_nur
    if save_nur:
        events = _nur_events(nur)
        assert [evt.get_id() for evt in events] == list(SEEDS)
        for row, evt in enumerate(events):
            channels = {ch.get_id(): ch for ch in evt.get_station(STATION).iter_channels()}
            expected = {COHERENT_CHANNEL_BASE + int(name[len('peak_'):]): waveforms[row]
                        for name, waveforms in primary.items() if np.any(waveforms[row])}
            assert COHERENT_CHANNEL_BASE in expected and sorted(channels) == sorted(expected)
            for channel_id, waveform in expected.items():
                assert np.array_equal(channels[channel_id].get_trace(), waveform)
                assert abs(channels[channel_id].get_sampling_rate() / units.GHz - 10.0) < 1e-9


@pytest.mark.slow
def test_save_nur_without_stored_waveforms_writes_a_file_without_events(inputs, monkeypatch, tmp_path):
    """Without `save_coherent_waveforms` the `--save-nur` file exists and the event reader finds no event in it."""
    nur_input, config = inputs
    out, nur = _drive(monkeypatch, tmp_path, nur_input, config, True)
    with h5py.File(out) as f:
        assert f.attrs['n_events'] == 2 and COHERENT_GROUP not in f
    assert nur.exists() and _nur_events(nur) == []


def test_event_writer_writes_a_file_without_events_on_request(tmp_path):
    """After no event, `end(write_empty_file=True)` leaves a file the reader opens without events; `end()` leaves none."""
    default, requested = tmp_path / 'default.nur', tmp_path / 'requested.nur'
    writer = eventWriter()
    writer.begin(str(default))
    assert writer.end() == 0 and not default.exists()
    writer = eventWriter()
    writer.begin(str(requested))
    assert writer.end(write_empty_file=True) == 0
    assert requested.exists() and _nur_events(requested) == []


def test_event_writer_with_events_is_unchanged_by_the_request(tmp_path):
    """With an event written, the file is the same bytes with and without the request."""
    paths = [tmp_path / 'default.nur', tmp_path / 'requested.nur']
    for path, write_empty_file in zip(paths, (False, True)):
        evt = Event(3, 7)
        evt.set_station(Station(STATION))
        writer = eventWriter()
        writer.begin(str(path))
        writer.run(evt)
        assert writer.end(write_empty_file=write_empty_file) == 1
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert [(evt.get_run_number(), evt.get_id()) for evt in _nur_events(paths[1])] == [(3, 7)]
