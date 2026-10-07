"""Synthetic RNO-G events for the feature extraction tests."""
import astropy.time
import numpy as np

import NuRadioReco.framework.channel
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.trigger
from NuRadioReco.utilities import units

SAMPLING_RATE = 3.2 * units.GHz
N_SAMPLES = 2048
DEEP_CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23)


def make_event(seed, missing=(), pulse_amplitude=8.0, event_id=0):
    """
    Build an event of station 13 with 24 channels of unit Gaussian noise.

    Parameters
    ----------
    seed : int
        Seed of the noise and the pulse shape
    missing : tuple of int
        Channels left out of the station
    pulse_amplitude : float
        Standard deviation of the samples of a 24-sample pulse added to the deep channels,
        each channel two samples later than the one before. 0 gives noise only.
    event_id : int
        Id of the event

    Returns
    -------
    event : `NuRadioReco.framework.event.Event`
        With one station that has an "LT" trigger and a station time
    """
    rng = np.random.default_rng(seed)
    pulse = rng.normal(0, 1, 24) * pulse_amplitude
    station = NuRadioReco.framework.station.Station(13)
    station.set_station_time(astropy.time.Time("2022-08-01T00:00:00") + seed * astropy.time.TimeDelta(1, format="sec"))
    for ch in range(24):
        trace = rng.normal(0, 1, N_SAMPLES)
        if ch in DEEP_CHANNELS:
            start = 900 + 2 * DEEP_CHANNELS.index(ch)
            trace[start:start + 24] += pulse
        if ch in missing:
            continue
        channel = NuRadioReco.framework.channel.Channel(ch)
        channel.set_trace(trace, SAMPLING_RATE)
        station.add_channel(channel)
    trigger = NuRadioReco.framework.trigger.Trigger("LT")
    trigger.set_triggered(True)
    trigger.set_trigger_time(0)
    station.set_trigger(trigger)
    event = NuRadioReco.framework.event.Event(1, event_id)
    event.set_station(station)
    return event
