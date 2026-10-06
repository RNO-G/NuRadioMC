import datetime
import hashlib
import math
import os
from collections import namedtuple

import yaml

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units

import NuRadioReco.modules.RNO_G.channelBlockOffsetFitter
import NuRadioReco.modules.RNO_G.channelGlitchDetector
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelSinewaveSubtraction
import NuRadioReco.modules.channelResampler

import logging
logger = logging.getLogger('NuRadioReco.RNO_G.channelPreprocessor')

DelayCorrections = namedtuple(
    'DelayCorrections',
    'corrections uncertainty_ns provenance valid_from valid_to station_windows sha256 path polarity '
    'position_shift',
    defaults=({}, {}))
_PROVENANCE_KEYS = ('derived_from', 'method', 'date', 'valid_from', 'valid_to')


def _as_date(value, key):
    """Return a YAML date or ISO string as a date.

    Parameters
    ----------
    value : datetime.date, datetime.datetime or str
        The value read from the file.
    key : str
        The key the value belongs to, for the error message.

    Returns
    -------
    datetime.date

    Raises
    ------
    ValueError
        When the value is none of the accepted types.
    """
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        return datetime.date.fromisoformat(value[:10])
    raise ValueError(f"delay corrections: {key} must be an ISO date, got {value!r}")


def _channel_map(block, name):
    """Validate a ``{station_id: {channel_id: float}}`` block and return it with int keys.

    Parameters
    ----------
    block : dict
        The block read from the file.
    name : str
        The block's name, for the error message.

    Returns
    -------
    dict
        ``{int station_id: {int channel_id: float}}``.

    Raises
    ------
    ValueError
        When the block or one of its station entries is not a mapping.
    """
    if not isinstance(block, dict):
        raise ValueError(f"delay corrections: {name} must map station ids to channel maps")
    out = {}
    for station_id, channels in block.items():
        if not isinstance(channels, dict):
            raise ValueError(f"delay corrections: {name}[{station_id}] must map channel ids to values")
        out[int(station_id)] = {int(ch): float(val) for ch, val in channels.items()}
    return out


def _station_windows(derived_from, corrections, valid_from, valid_to):
    """Return the validity window of every corrected station.

    Every station starts from the file-level window. A ``derived_from`` entry of
    the station (keyed by its id) that is a mapping with ``valid_from`` or
    ``valid_to`` narrows it to the overlap of the two windows.

    Parameters
    ----------
    derived_from : object
        The ``derived_from`` value of the file; only a mapping is read.
    corrections : iterable
        The station ids of the ``corrections``, ``polarity`` and ``position_shift`` blocks.
    valid_from, valid_to : datetime.date
        The file-level window.

    Returns
    -------
    dict
        ``{station_id: (first valid date, last valid date)}`` for every corrected station.

    Raises
    ------
    ValueError
        When a station's window does not overlap the file-level window.
    """
    entries = {}
    if isinstance(derived_from, dict):
        entries = {int(k): v for k, v in derived_from.items() if str(k).isdigit()}
    windows = {}
    for station_id in corrections:
        start, stop = valid_from, valid_to
        entry = entries.get(station_id)
        if isinstance(entry, dict):
            if 'valid_from' in entry:
                start = max(start, _as_date(entry['valid_from'], f'derived_from[{station_id}].valid_from'))
            if 'valid_to' in entry:
                stop = min(stop, _as_date(entry['valid_to'], f'derived_from[{station_id}].valid_to'))
        if start > stop:
            raise ValueError(f"delay corrections: the window of station {station_id} does not overlap "
                             f"the file window {valid_from} to {valid_to}")
        windows[station_id] = (start, stop)
    return windows


def load_delay_corrections(path):
    """Load and validate a per-channel delay-corrections file.

    The file is YAML with a ``corrections`` block ``{station_id: {channel_id: delta_ns}}``,
    where ``delta_ns`` is the amount to add to the database cable delay of that channel,
    an ``uncertainty_ns`` block with the same nesting and one entry per corrected channel,
    and the provenance keys ``derived_from``, ``method``, ``date``, ``valid_from`` and
    ``valid_to`` (ISO dates bounding the data epoch the corrections apply to). When
    ``derived_from`` maps a station id to a block with its own ``valid_from`` or
    ``valid_to``, that station's window is the overlap with the file-level window.
    An optional ``polarity`` block ``{station_id: {channel_id: +1 or -1}}`` names
    channels whose response is inverted; their traces are negated by the
    preprocessor, under the same validity windows. An optional ``position_shift``
    block ``{station_id: {channel_id: [dx, dy]}}`` (m) moves channels horizontally
    from their database positions; the preprocessor does not read it (positions
    enter the reconstruction, ``channel_position_shift``).

    Parameters
    ----------
    path : str
        Path of the YAML file.

    Returns
    -------
    DelayCorrections
        The two maps, the provenance dict, the file-level validity dates, the
        per-station windows, the SHA-256 hex digest of the file bytes, the
        absolute path, the polarity map (``{station_id: {channel_id: +1 or
        -1}}``, empty without the block) and the position shifts
        (``{station_id: {channel_id: (dx, dy)}}``, empty without the block).

    Raises
    ------
    ValueError
        On a missing block or provenance key, a wrong nesting, a corrected
        channel without an uncertainty, a polarity other than +1 or -1, a
        position shift that is not two finite numbers or a station window
        outside the file window.
    """
    with open(path, 'rb') as f:
        raw = f.read()
    doc = yaml.safe_load(raw)
    if not isinstance(doc, dict) or 'corrections' not in doc:
        raise ValueError(f"delay corrections file {path} has no 'corrections' block")
    missing = [k for k in _PROVENANCE_KEYS if k not in doc]
    if missing:
        raise ValueError(f"delay corrections file {path} lacks provenance keys {missing}")
    corrections = _channel_map(doc['corrections'], 'corrections')
    uncertainty = _channel_map(doc.get('uncertainty_ns', {}), 'uncertainty_ns')
    for station_id, channels in corrections.items():
        without = [ch for ch in channels if ch not in uncertainty.get(station_id, {})]
        if without:
            raise ValueError(f"delay corrections file {path}: station {station_id} channels {without} "
                             "have no uncertainty_ns entry")
    polarity = _channel_map(doc.get('polarity', {}), 'polarity')
    invalid = {(s, ch): v for s, channels in polarity.items() for ch, v in channels.items()
               if v not in (1.0, -1.0)}
    if invalid:
        raise ValueError(f"delay corrections file {path}: polarity entries must be +1 or -1, got {invalid}")
    polarity = {s: {ch: int(v) for ch, v in channels.items()} for s, channels in polarity.items()}
    position_shift = {}
    block = doc.get('position_shift', {})
    if not isinstance(block, dict) or not all(isinstance(v, dict) for v in block.values()):
        raise ValueError(f"delay corrections file {path}: position_shift must map station ids to channel maps")
    for station_id, channels in block.items():
        for ch, value in channels.items():
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                               for v in value)):
                raise ValueError(f"delay corrections file {path}: position_shift[{station_id}][{ch}] must be "
                                 f"[dx, dy] in m, got {value!r}")
            position_shift.setdefault(int(station_id), {})[int(ch)] = (float(value[0]), float(value[1]))
    provenance = {k: doc[k] for k in _PROVENANCE_KEYS}
    valid_from = _as_date(doc['valid_from'], 'valid_from')
    valid_to = _as_date(doc['valid_to'], 'valid_to')
    return DelayCorrections(corrections, uncertainty, provenance, valid_from, valid_to,
                            _station_windows(doc['derived_from'],
                                             set(corrections) | set(polarity) | set(position_shift),
                                             valid_from, valid_to),
                            hashlib.sha256(raw).hexdigest(), os.path.abspath(path), polarity, position_shift)


class channelPreprocessor:
    """
    RNO-G waveform preprocessing chain as a single composable module.

    Wraps the standard sequence of per-event processing steps so that the
    chain lives in one place and can be reused across different readers
    (``dataProviderRNOG`` for ROOT, ``dataProviderNuRadio`` for NUR, or
    ad-hoc pipelines). Each step is gated by a flag in the config passed
    to ``begin`` so callers can opt in or out without reimplementing the
    wiring.

    Steps, in order:

    1. ``channelBlockOffsetFitter`` (fit + subtract LAB4D block offsets)
    2. ``channelGlitchDetector`` (flag scrambled readout blocks; does not
       fix them, only sets ``channelParameter.glitch``)
    3. ``channelAddCableDelay`` (subtract cable delays)
    4. per-channel delay corrections from a ``delay_corrections_file``
       (``apply_delay_corrections``): each listed channel's trace start
       time is shifted by minus the correction, so the effective cable
       delay is the database value plus the correction, and the trace of
       each channel with polarity -1 in the file is negated. Meant for real
       data only; ``dataProviderNuRadio`` drops the keys because
       simulations are generated with the description the tables use.
    5. ``hardwareResponseIncorporator`` (invert hardware phase response;
       angle-independent, unlike antenna dedispersion which is reco-only)
    6. ``channelResampler`` (upsample to a target rate, typically 5 GHz)
    7. ``channelSinewaveSubtraction`` (CW peak removal)
    8. ``channelBandPassFilter`` (apply analysis passband)

    Block-offset removal is on by default. Glitch detection and steps
    4-8 are off by default. Unknown configuration keys raise.

    ``channels`` (default ``None``) restricts every step to the listed
    channel ids: the other channels are detached from the station while
    the steps run and re-attached afterwards in their original order, so
    they stay in the event with their traces and parameters untouched.
    Each step acts on channels independently, so the processed channels
    are identical to a run over the full station.

    See Also
    --------
    NuRadioReco.modules.RNO_G.dataProviderRNOG
    NuRadioReco.modules.RNO_G.channelBlockOffsetFitter
    NuRadioReco.modules.RNO_G.channelGlitchDetector
    NuRadioReco.modules.channelAddCableDelay
    NuRadioReco.modules.channelResampler
    NuRadioReco.modules.channelSinewaveSubtraction
    NuRadioReco.modules.channelBandPassFilter
    """

    _DEFAULT_CONFIG = {
        "apply_block_offset_removal": True,
        "apply_glitch_detection": False,
        "apply_cable_delay": True,
        "cable_delay_mode": "subtract",
        "apply_delay_corrections": False,
        "delay_corrections_file": None,
        "apply_hw_phase_removal": False,
        "hw_phase_mode": "phase_only",
        "hw_phase_sim_to_data": False,
        "apply_upsampling": False,
        "target_sampling_rate": 5.0 * units.GHz,
        "apply_cw_removal": False,
        "cw_peak_prominence": 4.0,
        "cw_freq_band": (0.1, 0.6),
        "cw_algorithm": "sliding",
        "apply_bandpass": False,
        "bandpass_band": (0.1 * units.GHz, 0.6 * units.GHz),
        "bandpass_filter_type": "butter",
        "bandpass_order": 10,
        "glitch_cut_value": 0.0,
        "channels": None,
    }

    def __init__(self):
        self._block_offset = NuRadioReco.modules.RNO_G.channelBlockOffsetFitter.channelBlockOffsets()
        self._glitch_detector = None
        self._cable_delay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
        self._hw_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
        self._resampler = NuRadioReco.modules.channelResampler.channelResampler()
        self._cw_filter = NuRadioReco.modules.channelSinewaveSubtraction.channelSinewaveSubtraction()
        self._bandpass = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
        self._config = dict(self._DEFAULT_CONFIG)
        self.delay_corrections = None

    def begin(self, config=None):
        """Initialize submodules with merged defaults + user config.

        Parameters
        ----------
        config : dict, optional
            Per-step flags and parameters. Keys override the class
            defaults (``_DEFAULT_CONFIG``).

        Raises
        ------
        ValueError
            On a key that is not in ``_DEFAULT_CONFIG``, or when
            ``apply_delay_corrections`` is set without a file or without
            the cable-delay step it corrects.
        """
        if config:
            unknown = sorted(set(config) - set(self._DEFAULT_CONFIG))
            if unknown:
                raise ValueError(f"unknown channelPreprocessor config keys: {unknown}")
            self._config.update(config)
        cfg = self._config

        if cfg["apply_delay_corrections"]:
            if not cfg["delay_corrections_file"]:
                raise ValueError("apply_delay_corrections needs delay_corrections_file")
            if not cfg["apply_cable_delay"]:
                raise ValueError("apply_delay_corrections corrects the cable delays, "
                                 "which apply_cable_delay: false does not apply")
            self.delay_corrections = load_delay_corrections(cfg["delay_corrections_file"])
            logger.info("delay corrections from %s (sha256 %s) for stations %s",
                        self.delay_corrections.path, self.delay_corrections.sha256[:16],
                        sorted(self.delay_corrections.corrections))

        self._glitch_detector = NuRadioReco.modules.RNO_G.channelGlitchDetector.channelGlitchDetector(
            cut_value=cfg["glitch_cut_value"]
        )

        self._block_offset.begin()
        self._glitch_detector.begin()
        self._cable_delay.begin()
        self._hw_response.begin()
        self._resampler.begin()
        self._bandpass.begin()
        self._cw_filter.begin(
            save_filtered_freqs=False,
            freq_band=tuple(cfg["cw_freq_band"]),
        )

    def end(self):
        """Call end on submodules that maintain per-run state."""
        self._block_offset.end()
        if self._glitch_detector is not None:
            self._glitch_detector.end()
        self._resampler.end()

    @register_run()
    def run(self, event, station, det):
        """Apply the enabled preprocessing steps in order.

        With ``channels`` set in the config, only those channels are
        processed; the others are detached from the station for the
        duration of the steps and re-attached in the original order.

        Parameters
        ----------
        event : NuRadioReco.framework.event.Event
        station : NuRadioReco.framework.station.Station
        det : Detector
        """
        if self._config["channels"] is None:
            self._run_steps(event, station, det)
            return

        all_channels = [station.get_channel(ch_id) for ch_id in station.get_channel_ids()]
        selected = set(self._config["channels"])
        for channel in all_channels:
            if channel.get_id() not in selected:
                station.remove_channel(channel)
        try:
            self._run_steps(event, station, det)
        finally:
            for channel in all_channels:
                if station.has_channel(channel.get_id()):
                    station.remove_channel(channel)
                station.add_channel(channel)

    def _run_steps(self, event, station, det):
        """Run the enabled steps on every channel currently in the station."""
        cfg = self._config

        if cfg["apply_block_offset_removal"]:
            self._block_offset.run(event, station, det)

        if cfg["apply_glitch_detection"]:
            self._glitch_detector.run(event, station, det)

        if cfg["apply_cable_delay"] and det is not None:
            self._cable_delay.run(event, station, det, mode=cfg["cable_delay_mode"])

        if cfg["apply_delay_corrections"] and det is not None:
            self._apply_delay_corrections(station)

        if cfg["apply_hw_phase_removal"]:
            self._hw_response.run(
                event, station, det,
                sim_to_data=cfg["hw_phase_sim_to_data"],
                mode=cfg["hw_phase_mode"],
            )

        if cfg["apply_upsampling"]:
            self._resampler.run(event, station, det,
                                sampling_rate=cfg["target_sampling_rate"])

        if cfg["apply_cw_removal"]:
            self._cw_filter.run(event, station, det,
                                algorithm=cfg["cw_algorithm"],
                                peak_prominence=cfg["cw_peak_prominence"])

        if cfg["apply_bandpass"]:
            self._bandpass.run(
                event, station, det,
                passband=tuple(cfg["bandpass_band"]),
                filter_type=cfg["bandpass_filter_type"],
                order=cfg["bandpass_order"],
            )

    def _apply_delay_corrections(self, station):
        """Shift the listed channels of the station by minus their delay correction and negate the inverted ones.

        Parameters
        ----------
        station : NuRadioReco.framework.station.Station
            The station whose channels are shifted in place.

        Raises
        ------
        ValueError
            When the station time is missing or outside the station's validity window.
        """
        dc = self.delay_corrections
        deltas = dc.corrections.get(station.get_id(), {})
        polarity = dc.polarity.get(station.get_id(), {})
        if not deltas and not polarity:
            return
        station_time = station.get_station_time()
        if station_time is None:
            raise ValueError("delay corrections need the station time to check their validity")
        day = station_time.to_datetime().date()
        valid_from, valid_to = dc.station_windows[station.get_id()]
        if not valid_from <= day <= valid_to:
            raise ValueError(f"delay corrections {dc.path} are valid for station {station.get_id()} "
                             f"from {valid_from} to {valid_to}; event at {day}")
        for channel_id, delta in deltas.items():
            if station.has_channel(channel_id):
                station.get_channel(channel_id).add_trace_start_time(-delta)
        for channel_id, sign in polarity.items():
            if sign == -1 and station.has_channel(channel_id):
                channel = station.get_channel(channel_id)
                channel.set_trace(-channel.get_trace(), 'same')
