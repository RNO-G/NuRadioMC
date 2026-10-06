"""Files the RNO-G reader needs in a run folder.

A run folder is read when it holds combined.root, or waveforms.root with headers.root and
daqstatus.root. daqstatus.root is not needed when the caller does not read it
(`read_daq_status: False` in the mattak keyword arguments, as the reconstruction driver
sets it): hand-carried runs can lack the file. A caller that reads it still gets the
folder refused, with the missing file named.
"""

import logging

import pytest

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import _all_files_in_directory, readRNOGData


def _folder(path, *names):
    """Make a folder holding empty files of the given names; returns its path as a string."""
    path.mkdir()
    for name in names:
        (path / name).touch()
    return str(path)


class _Dataset:
    """Stand-in for the mattak dataset of a run folder: three events of station 13 run 7."""

    backend = 'stub'
    run = 7
    station = 13

    def N(self):
        """Number of events."""
        return 3


def test_daqstatus_is_needed_only_when_it_is_read(tmp_path):
    """A folder without daqstatus.root passes the check exactly when the caller does not read the file."""
    full = _folder(tmp_path / 'full', 'waveforms.root', 'headers.root', 'daqstatus.root')
    bare = _folder(tmp_path / 'bare', 'waveforms.root', 'headers.root')
    assert _all_files_in_directory(full) and _all_files_in_directory(full, read_daq_status=False)
    assert not _all_files_in_directory(bare) and not _all_files_in_directory(bare, read_daq_status=True)
    assert _all_files_in_directory(bare, read_daq_status=False)


def test_the_other_files_stay_needed(tmp_path):
    """headers.root and a waveform file are needed either way; combined.root alone is enough."""
    for read_daq_status in (True, False):
        base = tmp_path / str(read_daq_status)
        base.mkdir()
        assert not _all_files_in_directory(_folder(base / 'no_headers', 'waveforms.root', 'daqstatus.root'), read_daq_status)
        assert not _all_files_in_directory(_folder(base / 'no_waveforms', 'headers.root', 'daqstatus.root'), read_daq_status)
        assert not _all_files_in_directory(_folder(base / 'empty'), read_daq_status)
        assert _all_files_in_directory(_folder(base / 'combined', 'combined.root'), read_daq_status)


@pytest.mark.parametrize('mattak_kwargs', [{}, {'backend': 'uproot'}, {'read_daq_status': True, 'backend': 'uproot'}])
def test_reader_refuses_a_folder_without_daqstatus_when_it_reads_it(tmp_path, caplog, mattak_kwargs):
    """With daqstatus.root to be read and absent, `begin` stops with FileNotFoundError and the log names the file."""
    bare = _folder(tmp_path / 'bare', 'waveforms.root', 'headers.root')
    with caplog.at_level(logging.ERROR), pytest.raises(FileNotFoundError, match='no valid datasets'):
        readRNOGData().begin(bare, mattak_kwargs=dict(mattak_kwargs))
    assert f'File daqstatus.root could not be found in {bare}' in caplog.text


def test_reader_opens_a_folder_without_daqstatus_when_it_does_not_read_it(tmp_path, monkeypatch):
    """With `read_daq_status: False` the folder passes the check and is opened with the caller's mattak arguments."""
    bare = _folder(tmp_path / 'bare', 'waveforms.root', 'headers.root')
    opened = []

    def open_dataset(self, path):
        """Note the folder and the mattak arguments instead of opening the files."""
        opened.append((path, dict(self._mattak_kwargs)))
        return _Dataset()

    monkeypatch.setattr(readRNOGData, '_readRNOGData__get_dataset', open_dataset)
    reader = readRNOGData()
    reader.begin(bare, mattak_kwargs={'read_daq_status': False, 'backend': 'uproot'})
    assert opened == [(bare, {'read_daq_status': False, 'backend': 'uproot'})]
    assert list(reader.get_run_numbers()) == [7]
