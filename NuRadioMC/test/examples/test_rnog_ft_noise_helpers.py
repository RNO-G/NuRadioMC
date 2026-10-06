"""Checks of the forced-trigger noise helpers of examples/08_RNO_G_trigger_simulation/simulate.py."""
import os
import sys

import numpy as np

import NuRadioReco.framework.channel
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.trigger
from NuRadioReco.utilities import units

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "08_RNO_G_trigger_simulation"))
from simulate import (
    RNO_G_HighLow_Thresh, TILE_OVERLAP, tile_noise_overlap_add, upsample_trace,
    zero_padded_readout_window_cutter)


class ReadoutDetector:
    """Detector stub with the RADIANT readout: 2048 samples at 3.2 GHz."""

    def get_sampling_frequency(self, station_id, channel_id):
        """Return the readout sampling rate."""
        return 3.2 * units.GHz

    def get_number_of_samples(self, station_id, channel_id):
        """Return the readout trace length."""
        return 2048


def test_upsample_trace():
    """A sine with a whole number of periods keeps its values on the finer grid."""
    n, n_up = 2048, 3200
    trace = np.sin(2 * np.pi * 37 * np.arange(n) / n)
    expected = np.sin(2 * np.pi * 37 * np.arange(n_up) / n_up)
    np.testing.assert_allclose(upsample_trace(trace, n_up), expected, atol=1e-10)


def test_tile_crossfade_is_equal_power():
    """The crossfade weights of two neighbouring tiles add up to one in power."""
    n_tile = 3200
    ones, zeros = np.ones(n_tile), np.zeros(n_tile)
    target = 2 * n_tile - TILE_OVERLAP
    tail = tile_noise_overlap_add([ones, zeros], target)
    head = tile_noise_overlap_add([zeros, ones], target)
    assert len(tail) == target and len(head) == target
    np.testing.assert_allclose(tail ** 2 + head ** 2, 1, atol=1e-12)
    # the outer edges have no neighbour and stay at full amplitude
    assert tail[0] == 1 and head[-1] == 1

    rng = np.random.default_rng(1)
    noise = np.array([tile_noise_overlap_add(list(rng.normal(size=(2, n_tile))), target) for _ in range(2000)])
    np.testing.assert_allclose(noise.std(axis=0), 1, atol=0.1)


def test_tile_length():
    """The stitched trace is cut to the requested length and an empty tile list gives zeros."""
    tiles = [np.ones(3200) for _ in range(5)]
    assert len(tile_noise_overlap_add(tiles, 12000)) == 12000
    assert not np.any(tile_noise_overlap_add([], 100))


def test_threshold():
    """A noise trigger rate of 1 Hz corresponds to a threshold of 3.76 sigma."""
    assert abs(RNO_G_HighLow_Thresh(0) - 3.759) < 1e-3


def cut(trigger_time):
    """Cut a 6000-sample ramp at 5 GHz to the readout window of a trigger at trigger_time."""
    station = NuRadioReco.framework.station.Station(11)
    channel = NuRadioReco.framework.channel.Channel(0)
    channel.set_trace(np.arange(1., 6001.), 5 * units.GHz)
    channel.set_trace_start_time(0)
    station.add_channel(channel)
    trigger = NuRadioReco.framework.trigger.Trigger("test", pre_trigger_times=200 * units.ns)
    trigger.set_triggered(True)
    trigger.set_trigger_time(trigger_time)
    station.set_trigger(trigger)
    event = NuRadioReco.framework.event.Event(0, 0)
    event.set_station(station)
    zero_padded_readout_window_cutter(event, station, ReadoutDetector())
    return channel.get_trace(), channel.get_trace_start_time()


def test_zero_padded_cutter():
    """Samples of the readout window outside the trace are zeros, not a cyclic copy."""
    ramp = np.arange(1., 6001.)

    trace, start = cut(400 * units.ns)
    assert start == 200 * units.ns
    np.testing.assert_array_equal(trace, ramp[1000:4200])

    trace, start = cut(1000 * units.ns)
    assert start == 800 * units.ns
    np.testing.assert_array_equal(trace[:2000], ramp[4000:])
    assert not np.any(trace[2000:])

    trace, start = cut(100 * units.ns)
    assert start == -100 * units.ns
    assert not np.any(trace[:500])
    np.testing.assert_array_equal(trace[500:], ramp[:2700])


if __name__ == "__main__":
    test_upsample_trace()
    test_tile_crossfade_is_equal_power()
    test_tile_length()
    test_threshold()
    test_zero_padded_cutter()
    print("All forced-trigger noise helper checks passed.")
