"""Detector-versus-table consistency check and the cable-delay hash.

`check_detector_consistency` compares the absolute antenna z of every channel at the
detector's current epoch with the `antenna_z_abs` a table records, fails above 0.05 m,
skips tables without metadata (the in-ice tables of record), and reports the per-channel
cable delays with a hash so that a results file states which delays it used. The stub
detector shows that the hash follows the delays and nothing else.
"""

import datetime
import os

import numpy as np
import pytest

from conftest import STATION, reference_config
from synthetic import VPOL_CHANNELS
from NuRadioReco.modules.interferometricDirectionReconstruction3D import (
    DETECTOR_Z_TOLERANCE_M, check_detector_consistency, detector_delay_vector, table_files_from_config,
    table_metadata)

HEX16 = 16


def _table(path, z_abs, det_file=''):
    """Write a minimal table carrying the provenance metadata."""
    np.savez(path, r_range_vals=np.arange(3.0), z_range_vals=np.arange(-2.0, 1.0), data=np.ones((3, 3)),
             antenna_z_abs=float(z_abs), det_date='2022-10-01', det_source='rnog_file', det_file=det_file)
    return path


class StubDetector:
    """Minimal detector: positions, cable delays and an epoch."""

    def __init__(self, delays, epoch=datetime.datetime(2022, 10, 1)):
        """Store per-channel delays in ns."""
        self.delays = dict(delays)
        self.epoch = epoch

    def get_absolute_position(self, station_id):
        """Station at the origin, 3 m below the surface."""
        return np.array([0.0, 0.0, -3.0])

    def get_relative_position(self, station_id, channel_id):
        """Channels 1 m apart in depth."""
        return np.array([0.0, 0.0, -90.0 - channel_id])

    def get_cable_delay(self, station_id, channel_id):
        """Stored delay."""
        return self.delays[channel_id]

    def get_detector_time(self):
        """Stored epoch."""
        return self.epoch


def test_matching_tables_pass(det, ant_locs, tmp_path):
    """Tables at the live antenna z pass, every channel is checked and the report is complete."""
    tables = {ch: [_table(str(tmp_path / f'ch{ch}.npz'), ant_locs[ch][2])] for ch in VPOL_CHANNELS}
    report = check_detector_consistency(det, STATION, VPOL_CHANNELS, tables)
    assert sorted(report['z_checked']) == sorted(VPOL_CHANNELS)
    assert report['z_max_diff_m'] < 1e-9
    assert len(report['delay_hash']) == HEX16 and report['epoch'].startswith('2022-10-01')
    assert set(report['cable_delays_ns']) == set(VPOL_CHANNELS)
    assert all(v > 0 for v in report['cable_delays_ns'].values())


def test_offset_above_tolerance_fails(det, ant_locs, tmp_path):
    """A 0.1 m table-versus-detector z difference on one channel raises and names it."""
    tables = {ch: [_table(str(tmp_path / f'ch{ch}.npz'), ant_locs[ch][2])] for ch in VPOL_CHANNELS}
    tables[9] = [_table(str(tmp_path / 'ch9_off.npz'), ant_locs[9][2] + 2 * DETECTOR_Z_TOLERANCE_M)]
    with pytest.raises(ValueError, match='ch9'):
        check_detector_consistency(det, STATION, VPOL_CHANNELS, tables)
    tables[9] = [_table(str(tmp_path / 'ch9_ok.npz'), ant_locs[9][2] + 0.5 * DETECTOR_Z_TOLERANCE_M)]
    assert check_detector_consistency(det, STATION, VPOL_CHANNELS, tables)['z_max_diff_m'] < DETECTOR_Z_TOLERANCE_M


def test_tables_without_metadata_are_skipped(det, table_dir):
    """The in-ice tables of record carry no metadata, so only the delay vector is recorded."""
    tables = table_files_from_config(STATION, reference_config(table_dir))
    assert all(os.path.isfile(p) for paths in tables.values() for p in paths)
    assert table_metadata(tables[0][0]) == {}
    report = check_detector_consistency(det, STATION, VPOL_CHANNELS, tables)
    assert report['z_checked'] == [] and report['xy_checked'] == []


def test_hash_follows_the_delays_only():
    """The hash changes with a 1 ns delay change and with nothing else."""
    delays = {ch: 950.0 + ch for ch in VPOL_CHANNELS}
    _, digest, epoch = detector_delay_vector(StubDetector(delays), 23, VPOL_CHANNELS)
    assert epoch == '2022-10-01T00:00:00'
    same, digest_same, _ = detector_delay_vector(StubDetector(dict(delays), datetime.datetime(2022, 7, 1)), 23, VPOL_CHANNELS)
    assert digest_same == digest and same == delays
    shifted = dict(delays)
    shifted[22] += 1.0
    assert detector_delay_vector(StubDetector(shifted), 23, VPOL_CHANNELS)[1] != digest
    assert detector_delay_vector(StubDetector(delays), 23, list(reversed(VPOL_CHANNELS)))[1] != digest


def test_table_files_from_config_follows_the_scheme():
    """One combined table per channel, or one per ray type with multi_ray_types."""
    single = table_files_from_config(23, dict(time_delay_tables='/t', channels=[0, 9]))
    assert single == {0: ['/t/station23/st23_ch0_rz_table.npz'], 9: ['/t/station23/st23_ch9_rz_table.npz']}
    multi = table_files_from_config(23, dict(time_delay_tables='/t', channels=[0], multi_ray_types=True))
    assert [os.path.basename(p) for p in multi[0]] == [f'st23_ch0_rz_table_{t}.npz' for t in ('direct', 'refracted', 'reflected')]
    ordered = table_files_from_config(23, dict(time_delay_tables='/t', channels=[0], multi_ray_types=True,
                                               table_scheme='solution_ordered'))
    assert [os.path.basename(p) for p in ordered[0]] == ['st23_ch0_rz_table_solution_0.npz', 'st23_ch0_rz_table_solution_1.npz']


def test_begin_records_the_report(reco, det):
    """The reconstruction object keeps the report of its begin-time check."""
    delays, digest, epoch = detector_delay_vector(det, STATION, VPOL_CHANNELS)
    assert reco.detector_report['delay_hash'] == digest
    assert reco.detector_report['epoch'] == epoch
    assert reco.detector_report['cable_delays_ns'] == delays
