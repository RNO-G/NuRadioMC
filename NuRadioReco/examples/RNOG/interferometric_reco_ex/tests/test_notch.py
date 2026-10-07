"""The notch step of channelPreprocessor.

With `apply_notch` the spectrum bins of every channel inside each band of `notch_bands`
are set to zero, after the CW subtraction and before the bandpass. A line inside the band
disappears, an impulse keeps every bin outside the band and its shape, the step acts on
the preprocessed channels only, and without the key nothing changes.
"""

import numpy as np

from conftest import STATION
from synthetic import NATIVE_RATE, filtered_trace
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.RNO_G.channelPreprocessor import channelPreprocessor
from NuRadioReco.utilities import fft, units

BAND = (0.399 * units.GHz, 0.407 * units.GHz)
LINE = 403 * units.MHz
# 3,200 samples at 3.2 GHz put the line on a bin, so that it has no leakage outside the band.
N_LINE = 3200


def _event(traces):
    """Event with one native-rate channel per trace, channel ids counting from 0."""
    evt = Event(0, 0)
    stn = Station(STATION)
    for ch, trace in enumerate(traces):
        c = Channel(ch)
        c.set_trace(trace, NATIVE_RATE)
        stn.add_channel(c)
    evt.set_station(stn)
    return evt, stn


def _step(evt, stn, **config):
    """Run the preprocessor on the event with only the given steps on."""
    pre = channelPreprocessor()
    pre.begin(dict(apply_block_offset_removal=False, apply_cable_delay=False, **config))
    pre.run(evt, stn, None)


def _line(amplitude=1.0, phase=0.3):
    """A 403 MHz sine over N_LINE native samples."""
    return amplitude * np.sin(2 * np.pi * LINE * np.arange(N_LINE) / NATIVE_RATE + phase)


def test_line_in_the_band_goes_to_zero():
    """A 403 MHz line is removed, by the default band and by a band given in the config."""
    for config in ({}, {'notch_bands': [[0.399, 0.407]]}):
        evt, stn = _event([_line()])
        _step(evt, stn, apply_notch=True, **config)
        assert np.max(np.abs(stn.get_channel(0).get_trace())) < 1e-9
    evt, stn = _event([_line()])
    _step(evt, stn, apply_notch=True, notch_bands=[[0.2, 0.21], [0.5, 0.51]])
    assert np.max(np.abs(stn.get_channel(0).get_trace() - _line())) < 1e-9


def test_impulse_keeps_its_shape_outside_the_band():
    """Bins outside the band are untouched bit for bit, bins inside are zero, and the pulse changes by about 1%."""
    pulse = filtered_trace(200.0, 1.0, 0.0, np.random.default_rng(0))
    evt, stn = _event([pulse])
    _step(evt, stn, apply_notch=True)
    channel = stn.get_channel(0)
    freqs = channel.get_frequencies()
    inside = (freqs >= BAND[0]) & (freqs <= BAND[1])
    before = fft.time2freq(pulse, NATIVE_RATE)
    after = channel.get_frequency_spectrum()
    assert inside.sum() == 5 and np.all(np.abs(before[inside]) > 0)
    assert np.array_equal(after[~inside], before[~inside]) and not np.any(after[inside])
    notched = channel.get_trace()
    assert np.argmax(np.abs(notched)) == np.argmax(np.abs(pulse))
    assert abs(np.max(np.abs(notched)) / np.max(np.abs(pulse)) - 1.0) < 0.02
    assert np.dot(notched, pulse) / np.sqrt(np.dot(notched, notched) * np.dot(pulse, pulse)) > 0.99


def test_default_off_changes_nothing():
    """Without `apply_notch` the line stays and the trace is the input bit for bit, whatever `notch_bands` holds."""
    for config in ({}, {'apply_notch': False}, {'notch_bands': [[0.399, 0.407]]}):
        evt, stn = _event([_line()])
        _step(evt, stn, **config)
        assert np.array_equal(stn.get_channel(0).get_trace(), _line()), config


def test_notch_respects_the_channel_restriction():
    """With `channels` set only the listed channels are notched."""
    evt, stn = _event([_line(), _line()])
    _step(evt, stn, apply_notch=True, channels=[0])
    assert np.max(np.abs(stn.get_channel(0).get_trace())) < 1e-9
    assert np.array_equal(stn.get_channel(1).get_trace(), _line())


def test_notch_runs_after_cw_removal_and_before_the_bandpass():
    """The three steps in one run equal CW removal, notch and bandpass run one after the other on the same event."""
    steps = [dict(apply_cw_removal=True), dict(apply_notch=True),
             dict(apply_bandpass=True, bandpass_band=[0.1, 0.7])]

    def trace(order):
        """Noise with a pulse and a strong 250 MHz line after the given sequence of preprocessor runs."""
        rng = np.random.default_rng(5)
        raw = filtered_trace(200.0, 5.0, 1.0, rng)
        raw += 3.0 * np.sin(2 * np.pi * 0.25 * np.arange(len(raw)) / NATIVE_RATE)
        evt, stn = _event([raw])
        for config in order:
            _step(evt, stn, **config)
        return stn.get_channel(0).get_trace()

    together = trace([dict(steps[0], **steps[1], **steps[2])])
    assert np.array_equal(together, trace(steps))
    assert not np.array_equal(together, trace([steps[1], steps[0], steps[2]]))
