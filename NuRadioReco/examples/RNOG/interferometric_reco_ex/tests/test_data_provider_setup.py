"""The shared set-up functions of the RNO-G scripts: detector description and data provider.

`init_detector` reads an exported detector description for the station of the config, or
queries the database without one, and sets the detector to the config's date.
`select_data_provider` opens a `.nur` file with `dataProviderNuRadio` and anything else
with `dataProviderRNOG`, the caller's reader options merged over the mattak defaults. The
reconstruction driver calls both and keeps the names other code imports from it.
"""

import datetime
import logging

import pytest

import interferometric_reco_3d_advanced as driver
import reco_validation
from conftest import STATION, station_detector_file
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.RNO_G import dataProviderSetup
from NuRadioReco.modules.RNO_G.dataProviderNuRadio import dataProviderNuRadio
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from NuRadioReco.modules.RNO_G.dataProviderSetup import init_detector, select_data_provider

MATTAK_DEFAULTS = {'read_daq_status': False, 'backend': 'uproot'}


class _Detector:
    """Stand-in for a detector class: keeps its constructor arguments and the time it is updated to."""

    def __init__(self, **kwargs):
        """Keep the keyword arguments."""
        self.kwargs = kwargs
        self.time = None

    def update(self, time):
        """Keep the time."""
        self.time = time


def _record_begin(monkeypatch):
    """Replace `begin` of both providers by a recorder; returns the list of (class name, arguments) it fills."""
    calls = []

    def begin(self, files, det, reader_kwargs=None, preprocessor_config=None):
        """Note the provider class and the arguments instead of opening the input."""
        calls.append((type(self).__name__, files, det, reader_kwargs, preprocessor_config))

    monkeypatch.setattr(dataProviderRNOG, 'begin', begin)
    monkeypatch.setattr(dataProviderNuRadio, 'begin', begin)
    return calls


def test_init_detector_without_a_file_queries_the_database(monkeypatch):
    """Without `detector_file` the database detector is built and updated to the config's date, 2022-10-01 by default."""
    monkeypatch.setattr(dataProviderSetup.detector, 'Detector', _Detector)
    det = init_detector({'station_id': 13})
    assert det.kwargs == {'source': 'rnog_mongo'} and det.time == datetime.datetime(2022, 10, 1)
    assert init_detector({'station_id': 13, 'detector_date': '2023-07-15'}).time == datetime.datetime(2023, 7, 15)
    assert init_detector({'station_id': 13, 'detector_file': None}).kwargs == {'source': 'rnog_mongo'}


def test_init_detector_with_a_file_reads_it_for_the_station(monkeypatch):
    """With `detector_file` the RNO-G detector is built from the file for the station of the config."""
    monkeypatch.setattr(dataProviderSetup.rnog_detector, 'Detector', _Detector)
    det = init_detector({'station_id': 23, 'detector_file': 'export.json.xz', 'detector_date': '2022-08-01'})
    assert det.kwargs == {'detector_file': 'export.json.xz', 'log_level': logging.WARNING, 'select_stations': 23}
    assert det.time == datetime.datetime(2022, 8, 1)


def test_init_detector_on_an_exported_description():
    """The detector of a real export holds the station and gives its cable delays at the date."""
    path = station_detector_file(STATION)
    if path is None:
        pytest.skip(f'no detector export for station {STATION}')
    det = init_detector({'station_id': STATION, 'detector_file': path})
    direct = rnog_detector.Detector(detector_file=path, select_stations=STATION, log_level=logging.WARNING)
    direct.update(datetime.datetime(2022, 10, 1))
    assert type(det) is rnog_detector.Detector
    assert det.get_cable_delay(STATION, 0) == direct.get_cable_delay(STATION, 0) > 0
    assert list(det.get_relative_position(STATION, 0)) == list(direct.get_relative_position(STATION, 0))


@pytest.mark.parametrize('input_file', ['/data/station13/run219', '/data/station13/run219/combined.root'])
def test_run_folders_and_root_files_get_the_rnog_provider(monkeypatch, input_file):
    """Anything but a `.nur` file is opened by dataProviderRNOG with the mattak defaults."""
    calls = _record_begin(monkeypatch)
    det, preprocessor = object(), {'apply_bandpass': True}
    provider = select_data_provider(input_file, det, preprocessor_config=preprocessor)
    assert type(provider) is dataProviderRNOG
    assert calls == [('dataProviderRNOG', input_file, det, {'mattak_kwargs': MATTAK_DEFAULTS}, preprocessor)]


def test_reader_kwargs_are_merged_over_the_mattak_defaults(monkeypatch):
    """The caller's reader options are passed on, its mattak keys one by one over the defaults, its dict untouched."""
    calls = _record_begin(monkeypatch)
    reader_kwargs = {'select_triggers': 'FORCE', 'mattak_kwargs': {'read_run_info': False, 'backend': 'pyroot'}}
    select_data_provider('/data/station13/run219', None, reader_kwargs=reader_kwargs)
    select_data_provider('/data/station13/run219', None, reader_kwargs={'select_triggers': 'FORCE', 'mattak_kwargs': None})
    select_data_provider('/data/station13/run219', None, reader_kwargs={})
    assert [call[3] for call in calls] == [
        {'select_triggers': 'FORCE', 'mattak_kwargs': {'read_daq_status': False, 'read_run_info': False, 'backend': 'pyroot'}},
        {'select_triggers': 'FORCE', 'mattak_kwargs': MATTAK_DEFAULTS},
        {'mattak_kwargs': MATTAK_DEFAULTS}]
    assert reader_kwargs == {'select_triggers': 'FORCE', 'mattak_kwargs': {'read_run_info': False, 'backend': 'pyroot'}}


def test_nur_files_get_the_nur_provider(monkeypatch):
    """A `.nur` file is opened by dataProviderNuRadio, which gets the preprocessor block and no reader options."""
    calls = _record_begin(monkeypatch)
    det, preprocessor = object(), {'apply_cable_delay': False}
    provider = select_data_provider('/sim/events.nur', det, reader_kwargs={'select_triggers': 'FORCE'},
                                    preprocessor_config=preprocessor)
    assert type(provider) is dataProviderNuRadio
    assert calls == [('dataProviderNuRadio', '/sim/events.nur', det, None, preprocessor)]


def test_driver_keeps_the_names_other_code_imports():
    """`init_detector` and `preprocessing_channels` stay importable from the driver and are the shared functions."""
    assert driver.init_detector is init_detector
    assert driver.preprocessing_channels is reco_validation.preprocessing_channels
    assert driver.select_data_provider is select_data_provider
