"""The pulser truth frame: device positions share the channel frame and get the station offset.

The detector description stores device positions relative to the station reference point,
like the channel positions; the reconstruction works in absolute z. `pulser_truth` adds the
station z. The invariant that proves the frame is the 1 m spacing between each in-ice pulser
and the HPol channel of the string that hosts it, which only holds when both are in the
same frame. The station-23 values are pinned to the corrected residual tables of
2026-09-30 (helper-C pulser at rho 34.33 m, phi 119.91 deg, z -95.61 m).
"""

import numpy as np
import pytest

from conftest import load_station_detector
from reco_validation import (PULSER_DEVICE_STRING, STRING_HPOL_CHANNEL, deep_pulser_devices,
                             pa_reference_point, pulser_host_string, pulser_truth)

EXPORTED_STATIONS = [13, 21, 22, 23]
PULSER_HPOL_SPACING_M = 1.0
SPACING_TOL_M = 0.05
XY_TOL_M = 0.01
REFERENCE_ST23 = {0: (34.32791, 119.91018, -95.61114), 1: (34.54213, 180.20314, -94.79814)}


@pytest.mark.parametrize('station', EXPORTED_STATIONS)
def test_device_frame_equals_channel_frame(station):
    """Every in-ice pulser sits 1 m above the HPol channel of its string, at the same (x, y)."""
    det = load_station_detector(station)
    devices = deep_pulser_devices(det, station)
    assert set(devices) == {0, 1}, devices
    for device_id in devices:
        string = pulser_host_string(det, station, device_id)
        assert PULSER_DEVICE_STRING[device_id] == string, (station, device_id, string)
        dev = np.asarray(det.get_relative_position_device(station, device_id), dtype=float)
        hpol = np.asarray(det.get_relative_position(station, STRING_HPOL_CHANNEL[string]), dtype=float)
        assert np.hypot(dev[0] - hpol[0], dev[1] - hpol[1]) < XY_TOL_M, (station, device_id, dev, hpol)
        assert abs((dev[2] - hpol[2]) - PULSER_HPOL_SPACING_M) < SPACING_TOL_M, (station, device_id, dev[2], hpol[2])


@pytest.mark.parametrize('station', EXPORTED_STATIONS)
def test_pulser_truth_adds_station_offset(station):
    """z_abs is the device z plus the station z, and rho, phi refer to the PA reference point."""
    det = load_station_detector(station)
    station_z = float(det.get_absolute_position(station)[2])
    pa = pa_reference_point(det, station)
    for device_id in deep_pulser_devices(det, station):
        rho, phi, z = pulser_truth(det, station, device_id)
        dev = np.asarray(det.get_relative_position_device(station, device_id), dtype=float)
        assert abs(z - (dev[2] + station_z)) < 1e-9
        assert abs(rho - np.hypot(dev[0] - pa[0], dev[1] - pa[1])) < 1e-9
        assert 0.0 <= phi < 360.0
        assert abs(z - dev[2] - station_z) < 1e-9 and station_z < 0


def test_station23_truth_matches_corrected_reference():
    """The station-23 pulser truth equals the corrected residual tables' positions."""
    det = load_station_detector(23)
    for device_id, expected in REFERENCE_ST23.items():
        got = pulser_truth(det, 23, device_id)
        assert np.allclose(got, expected, atol=1e-3), (device_id, got, expected)
