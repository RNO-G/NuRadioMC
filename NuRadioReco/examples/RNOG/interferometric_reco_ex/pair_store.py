"""Pair store of the 3D reconstruction: the pass-1 pair correlation series of every event in one HDF5 file.

The search stage of the reconstruction (``InterferometricReco3D.reconstruct_from_pairs``)
reads nothing but the pair correlation series, their lag axes and the channel SNRs, so a
file of those reproduces the search for any channel subset, pair weighting, per-channel
delay shift or polarity without the waveforms. Each series is kept only over the lags the
travel-time tables can reach anywhere in the configured volume (``pair_lag_windows``) plus
a margin for delay-shift trials and a two-sample interpolation guard, so the cut never
changes a result.

Layout (``n`` events, ``P`` pairs, ``C`` channels):

- file attributes: ``format``, ``format_version``, ``created``, ``code_commit``,
  ``reco_version``, ``config_json`` (the driver config), ``config_hash``, ``station_id``,
  ``channels``, ``dtype``, ``margin_ns``, ``apply_hann_window``,
  ``correlation_normalization``, ``snr_window_ns`` (NaN when unset), ``n_events``.
- ``pairs/ch_a``, ``pairs/ch_b`` (P): channel ids, lag = t_a - t_b; ``pairs/window_ns``
  (P, 2): lag window the series were cut to (NaN: the volume reaches no table cell).
- ``events/run_number``, ``event_number``, ``source_file``, ``station_id`` (n);
  ``events/time`` (n): station time in unix seconds (NaN when the event has none);
  ``events/snr`` (n, C): record SNR; ``events/snr_windowed`` (n, C) with ``snr_window_ns``;
  ``events/pair_weight`` (n, P): weight of each pair in the driver config's search (NaN
  for a pair no group searches); ``events/lag_offset``, ``lag_dt`` (n, P): lag of sample 0
  of the complete series and the sample spacing in ns; ``events/length`` (n, P): length of
  the complete series; ``events/k0``, ``n``, ``start`` (n, P): first kept sample, number of
  kept samples and position of the first one in the value arrays.
- ``series/raw``, ``series/envelope_traces``, ``series/envelope_correlation``: kept
  samples of every event and pair, concatenated in event then pair order.
- ``results_attrs`` group: the attributes of the driver's results file.
"""

import datetime
import hashlib
import itertools
import json
import os
import subprocess

import h5py
import numpy as np

from NuRadioReco.modules.RNO_G.channelPreprocessor import load_delay_corrections
from NuRadioReco.modules.interferometricDirectionReconstruction3D import (
    CorrPacked, PairSet, RECO_VERSION)

FORMAT = 'reco3d_pair_store'
FORMAT_VERSION = 1
MODE_NAMES = {None: 'raw', 'traces': 'envelope_traces', 'correlation': 'envelope_correlation'}
DEFAULT_MARGIN_NS = 20.0
_CHUNK = 1 << 16


def config_hash(config):
    """SHA-256 hex digest of a config dict printed as JSON with sorted keys."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def code_commit(path=__file__):
    """Commit of the git checkout holding ``path``, '+dirty' appended for a modified tree.

    Outside a git checkout (a ``git archive`` snapshot) the value of the environment
    variable ``RECO3D_CODE_COMMIT``, or ''.
    """
    root = os.path.dirname(os.path.abspath(path))
    head = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode != 0:
        return os.environ.get('RECO3D_CODE_COMMIT', '')
    dirty = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    return head.stdout.strip() + ('+dirty' if dirty else '')


def cut_indices(windows, packed):
    """First kept sample and number of kept samples of every pair for lag windows.

    A delay x inside a window is read by the kernels at samples floor((x - offset) / dt) and
    the next one; one sample below and two above that range are kept as a rounding guard.

    Args:
        windows: (P, 2) lag windows in ns (NaN rows keep nothing).
        packed: CorrPacked whose offsets, spacings and lengths define the lag axes.

    Returns:
        (k0, n) int64 arrays of length P.
    """
    with np.errstate(invalid='ignore'):
        lo = np.floor((windows[:, 0] - packed.offsets) / packed.dts) - 1
        hi = np.floor((windows[:, 1] - packed.offsets) / packed.dts) + 3
    keep = np.isfinite(lo) & np.isfinite(hi)
    k0 = np.where(keep, np.clip(lo, 0, packed.lengths), 0).astype(np.int64)
    k1 = np.where(keep, np.clip(hi, k0, packed.lengths), k0).astype(np.int64)
    return k0, k1 - k0


def cut_pairs(pairs, windows, dtype=np.float32):
    """Cut the series of a PairSet to lag windows and round them through the store dtype.

    Args:
        pairs: PairSet with complete series (``compute_pairs(..., store=True)``).
        windows: (P, 2) lag windows in ns, one per pair of ``pairs``.
        dtype: Storage dtype the kept samples are rounded through.

    Returns:
        PairSet whose series are complete-length float64 arrays holding the rounded kept
        samples and zeros elsewhere, with ``windows`` set. The search on it equals the
        search on the same event read back from the store bit for bit.
    """
    k0, n = cut_indices(windows, pairs.series[None])
    series = {}
    for mode, packed in pairs.series.items():
        corr = np.zeros_like(packed.corr)
        for p in range(len(k0)):
            kept = slice(k0[p], k0[p] + n[p])
            corr[p, kept] = packed.corr[p, kept].astype(dtype)
        series[mode] = packed._replace(corr=corr)
    return pairs._replace(series=series, windows=np.asarray(windows, dtype=np.float64))


def calibration_trial(applied, trial, station_id):
    """Delay shifts and polarities that turn pair series made with one calibration into those of another.

    Args:
        applied: DelayCorrections the traces were preprocessed with, or None.
        trial: DelayCorrections to evaluate, or None for no calibration.
        station_id: Station of the events.

    Returns:
        (channel_delay_shift, channel_polarity) for ``reconstruct_from_pairs``: the
        per-channel correction difference (trial minus applied, zero entries left out)
        and the product of the two polarities (+1 entries left out).
    """
    def station_part(dc):
        """Corrections and polarities of the station in one calibration."""
        if dc is None:
            return {}, {}
        return dc.corrections.get(station_id, {}), dc.polarity.get(station_id, {})

    applied_shift, applied_sign = station_part(applied)
    trial_shift, trial_sign = station_part(trial)
    shift = {ch: trial_shift.get(ch, 0.0) - applied_shift.get(ch, 0.0)
             for ch in set(applied_shift) | set(trial_shift)}
    sign = {ch: trial_sign.get(ch, 1) * applied_sign.get(ch, 1)
            for ch in set(applied_sign) | set(trial_sign)}
    return ({ch: v for ch, v in shift.items() if v != 0.0},
            {ch: v for ch, v in sign.items() if v != 1})


def config_pair_weights(reco, pairs, config):
    """Weight of every pair of a PairSet in the search of a config (NaN where no group searches the pair).

    Args:
        reco: InterferometricReco3D (its weight rules).
        pairs: PairSet of the event.
        config: Reconstruction config dict; its polarization groups define the searched pairs.

    Returns:
        (P,) float64 array; all NaN without ``snr_pair_weighting``.
    """
    index = {p: i for i, p in enumerate(pairs.pairs)}
    out = np.full(len(pairs.pairs), np.nan)
    groups = config.get('polarization_groups', None) or {'all': config['channels']}
    for members in groups.values():
        active = [ch for ch in config['channels'] if ch in members]
        if len(active) < 2:
            continue
        weights = reco._group_pair_weights(pairs.snr, pairs.snr_windowed, active, config)
        if weights is None:
            continue
        for p, w in zip(itertools.combinations(active, 2), weights):
            out[index[p]] = w
    return out


class PairStoreWriter:
    """Append the cut pair series of one event at a time to a new pair-store file.

    Side effects:
        Creates (overwrites) the file at construction and writes to it on every ``append``.
    """

    def __init__(self, path, config, windows, margin_ns, dtype='float32'):
        """Create the file and write the configuration and pair list.

        Args:
            path: Output HDF5 path.
            config: Driver config dict (its channels define the pairs).
            windows: (P, 2) lag windows in ns over ``combinations(config['channels'], 2)``,
                margin included.
            margin_ns: Margin added on both sides of the reachable lags, for the record.
            dtype: Storage dtype of the series ('float32' or 'float64').
        """
        self.dtype = np.dtype(dtype)
        self.windows = np.asarray(windows, dtype=np.float64)
        self.channels = list(config['channels'])
        self.pairs = list(itertools.combinations(self.channels, 2))
        self._n_events = 0
        self._n_values = 0
        self.file = h5py.File(path, 'w')
        norm = config.get('correlation_normalization', 'normalized')
        attrs = {
            'format': FORMAT, 'format_version': FORMAT_VERSION,
            'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'code_commit': code_commit(), 'reco_version': RECO_VERSION,
            'config_json': json.dumps(config, sort_keys=True, default=str),
            'config_hash': config_hash(config), 'station_id': int(config['station_id']),
            'channels': np.array(self.channels, dtype=np.int64), 'dtype': self.dtype.name,
            'margin_ns': float(margin_ns),
            'apply_hann_window': bool(config.get('apply_hann_window', False)),
            'correlation_normalization': 'pearson' if norm == 'normalized' else norm,
            'snr_window_ns': float(config.get('snr_window_ns', None) or np.nan),
        }
        self.file.attrs.update(attrs)
        grp = self.file.create_group('pairs')
        grp.create_dataset('ch_a', data=np.array([a for a, _ in self.pairs], dtype=np.int64))
        grp.create_dataset('ch_b', data=np.array([b for _, b in self.pairs], dtype=np.int64))
        grp.create_dataset('window_ns', data=self.windows)
        self._events = self.file.create_group('events')
        self._series = self.file.create_group('series')
        n_pairs, n_ch = len(self.pairs), len(self.channels)
        for name, dt, width in (('run_number', np.int64, None), ('event_number', np.int64, None),
                                ('station_id', np.int64, None), ('time', np.float64, None),
                                ('snr', np.float64, n_ch),
                                ('pair_weight', np.float64, n_pairs),
                                ('lag_offset', np.float64, n_pairs), ('lag_dt', np.float64, n_pairs),
                                ('length', np.int32, n_pairs), ('k0', np.int32, n_pairs),
                                ('n', np.int32, n_pairs), ('start', np.int64, n_pairs)):
            shape = (0,) if width is None else (0, width)
            self._events.create_dataset(name, shape=shape, maxshape=(None,) + shape[1:],
                                        dtype=dt, chunks=(256,) + shape[1:])
        self._events.create_dataset('source_file', shape=(0,), maxshape=(None,), chunks=(256,),
                                    dtype=h5py.string_dtype())
        if config.get('snr_window_ns', None) is not None:
            self._events.create_dataset('snr_windowed', shape=(0, n_ch), maxshape=(None, n_ch),
                                        dtype=np.float64, chunks=(256, n_ch))
        for name in MODE_NAMES.values():
            self._series.create_dataset(name, shape=(0,), maxshape=(None,), dtype=self.dtype,
                                        chunks=(_CHUNK,))

    def append(self, pairs, run_number, event_number, source_file, station_id, time, pair_weight):
        """Write the kept samples of one event.

        Args:
            pairs: PairSet from ``cut_pairs`` (every envelope mode, the writer's pairs and windows).
            run_number, event_number: Event keys.
            source_file: Input file name of the event ('' when not applicable).
            station_id: Station of the event.
            time: Station time in unix seconds, NaN when the event has none.
            pair_weight: (P,) weights of the pairs in the driver config's search.
        """
        ref = pairs.series[None]
        k0, n = cut_indices(self.windows, ref)
        start = self._n_values + np.concatenate(([0], np.cumsum(n)[:-1]))
        total = int(n.sum())
        i = self._n_events
        for name, value in (('run_number', run_number), ('event_number', event_number),
                            ('station_id', station_id), ('time', time),
                            ('source_file', source_file),
                            ('snr', [pairs.snr[ch] for ch in self.channels]),
                            ('pair_weight', pair_weight), ('lag_offset', ref.offsets),
                            ('lag_dt', ref.dts), ('length', ref.lengths), ('k0', k0), ('n', n),
                            ('start', start)):
            ds = self._events[name]
            ds.resize(i + 1, axis=0)
            ds[i] = value
        if 'snr_windowed' in self._events:
            self._events['snr_windowed'].resize(i + 1, axis=0)
            self._events['snr_windowed'][i] = [pairs.snr_windowed[ch] for ch in self.channels]
        for mode, name in MODE_NAMES.items():
            corr = pairs.series[mode].corr
            values = np.concatenate([corr[p, k0[p]:k0[p] + n[p]] for p in range(len(n))])
            ds = self._series[name]
            ds.resize(self._n_values + total, axis=0)
            ds[self._n_values:] = values.astype(self.dtype)
        self._n_values += total
        self._n_events += 1

    def close(self, results_attrs=None):
        """Write the event count and the driver's results-file attributes, and close the file."""
        self.file.attrs['n_events'] = self._n_events
        grp = self.file.create_group('results_attrs')
        for key, value in (results_attrs or {}).items():
            grp.attrs[key] = value
        self.file.close()


class PairStore:
    """Read access to a pair-store file: configuration, event keys and the PairSet of each event."""

    def __init__(self, path):
        """Open the file and read its header.

        Raises:
            ValueError: If the file is not a pair store of a supported format version.
        """
        self.path = path
        self.file = h5py.File(path, 'r')
        attrs = self.file.attrs
        if attrs.get('format') != FORMAT or int(attrs.get('format_version', -1)) != FORMAT_VERSION:
            raise ValueError(f"{path} is not a {FORMAT} file of version {FORMAT_VERSION}")
        self.config = json.loads(attrs['config_json'])
        self.channels = tuple(int(ch) for ch in attrs['channels'])
        self.pairs = tuple(zip(self.file['pairs/ch_a'][()].tolist(), self.file['pairs/ch_b'][()].tolist()))
        self.windows = self.file['pairs/window_ns'][()]
        window_ns = float(attrs['snr_window_ns'])
        self.settings = {'apply_hann_window': bool(attrs['apply_hann_window']),
                         'correlation_normalization': str(attrs['correlation_normalization']),
                         'snr_window_ns': None if np.isnan(window_ns) else window_ns}
        self.results_attrs = dict(self.file['results_attrs'].attrs) if 'results_attrs' in self.file else {}
        events = self.file['events']
        self.run_number = events['run_number'][()]
        self.event_number = events['event_number'][()]
        self.source_file = np.array([s.decode() if isinstance(s, bytes) else s
                                     for s in events['source_file'][()]], dtype=object)
        self.station_id = events['station_id'][()]
        self.time = events['time'][()]

    def __len__(self):
        """Number of events in the file."""
        return len(self.run_number)

    def __enter__(self):
        """Return the open store."""
        return self

    def __exit__(self, *exc):
        """Close the file."""
        self.close()

    def close(self):
        """Close the file."""
        self.file.close()

    def index(self, run_number, event_number, source_file=None):
        """Position of an event in the file, matched by run and event number and, when given, source file.

        Raises:
            KeyError: If no event or more than one event matches.
        """
        match = (self.run_number == run_number) & (self.event_number == event_number)
        if source_file is not None:
            match &= self.source_file == source_file
        found = np.flatnonzero(match)
        if len(found) != 1:
            raise KeyError(f"{len(found)} events match run {run_number}, event {event_number}, "
                           f"source file {source_file}")
        return int(found[0])

    def event(self, i):
        """PairSet of event ``i``: complete-length float64 series, zeros outside the stored windows."""
        events = self.file['events']
        k0, n, start = events['k0'][i], events['n'][i], events['start'][i]
        lengths = events['length'][i].astype(np.int64)
        dts = events['lag_dt'][i]
        offsets = events['lag_offset'][i]
        lo = int(start[0])
        hi = int(start[-1] + n[-1])
        series = {}
        for mode, name in MODE_NAMES.items():
            values = self.file['series'][name][lo:hi].astype(np.float64)
            corr = np.zeros((len(n), int(lengths.max())), dtype=np.float64)
            for p in range(len(n)):
                corr[p, k0[p]:k0[p] + n[p]] = values[start[p] - lo:start[p] - lo + n[p]]
            series[mode] = CorrPacked(corr, lengths, dts, offsets, 1.0 / dts)
        snr = dict(zip(self.channels, events['snr'][i]))
        windowed = (dict(zip(self.channels, events['snr_windowed'][i]))
                    if 'snr_windowed' in events else {})
        return PairSet(self.channels, self.pairs, series, snr, windowed, dict(self.settings),
                       self.windows)

    def applied_calibration(self):
        """DelayCorrections the stored traces were preprocessed with, or None.

        Raises:
            ValueError: If the file recorded at write time has changed since (hash mismatch).
        """
        path = self.results_attrs.get('delay_corrections_file', '')
        if not path:
            return None
        dc = load_delay_corrections(path)
        if dc.sha256 != self.results_attrs['delay_corrections_hash']:
            raise ValueError(f"{path} changed since the store was written (sha256 {dc.sha256[:16]} "
                             f"against {self.results_attrs['delay_corrections_hash'][:16]})")
        return dc

    def pair_weights(self, i):
        """Dict (ch_a, ch_b) -> weight the driver config's search gave each pair of event ``i`` (NaN: unsearched)."""
        return dict(zip(self.pairs, self.file['events/pair_weight'][i].tolist()))
