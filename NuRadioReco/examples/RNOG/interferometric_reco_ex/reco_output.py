"""Result writers of the 3D reconstruction driver and the output contract they keep.

Every dataset name, dtype and definition listed here is consumed downstream (classifier
features, coherent-sum alignment, cuts, combined event tables); a change to any of them
is a versioned change. New per-event quantities are added under a name that ends in a
version suffix (`_v<N>`) so that consumers can tell them apart from the contract keys.
"""

import re

import h5py
import numpy as np

from NuRadioReco.framework.channel import Channel
from NuRadioReco.modules.interferometricDirectionReconstruction3D import RECO_VERSION

BASE_KEYS = ['rho', 'phi', 'z', 'max_corr']
VALIDATION_SUMMARY_KEYS = [
    'surf_corr_z', 'surf_corr_zen', 'peak_isolation_ratio',
    'pa_avg_snr', 'pa_max_snr',
    'helper_b_max_snr', 'helper_b_min_snr',
    'helper_c_max_snr', 'helper_c_min_snr',
]
VALIDATION_COUNT_KEYS = [
    ('n_helpers_above', int, 0),
    ('n_channels_above', int, 0),
    ('has_helper_signal', bool, False),
]
OPTIONAL_KEYS = [
    'pass1_rho', 'pass1_phi', 'pass1_z', 'pass1_corr',
    'grid_time', 'opt_time', 'coarse_time', 'refine_time',
    'preproc_time', 'post_time', 'raw_refine_time',
    'peak_time',
    'p1_preproc_time', 'p1_total_time',
    'p1_coarse_time', 'p1_refine_time', 'p1_opt_time',
    'p2_dedisp_time', 'p2_reco_time',
    'p2_coarse_time', 'p2_refine_time', 'p2_opt_time',
    'p2_grid_time',
    'plane_wave_fallback', 'n_saved_peaks',
]
PEAK_FIELDS = ['rho', 'phi', 'z', 'corr', 'map_snr']
IDENTITY_KEYS = ['run_number', 'event_number', 'source_file']
COHERENT_GROUP = 'coherent_waveforms'
COHERENT_CHANNEL_BASE = 100
SKIPPED_RESULT_KEYS = {'run_number', 'event_number', 'source_file', 'coarse_peaks', 'coherent_times'}
FILE_ATTRS = ['mode', 'n_events', 'validation', 'reco_version', 'detector_delay_hash',
              'detector_epoch', 'delay_corrections_hash', 'delay_corrections_file']
VERSIONED_KEY = re.compile(r'_v\d+$')


def validation_keys(channels):
    """Per-event validation datasets written with `--validation` for a channel list."""
    return [f'ch{ch}_snr' for ch in channels] + list(VALIDATION_SUMMARY_KEYS)


def peak_keys(n_peaks):
    """Saved-peak datasets for `n_peaks` peaks."""
    return [f'peak_{i}_{f}' for i in range(n_peaks) for f in PEAK_FIELDS]


def contract_keys(channels, n_peaks=3, validation=True):
    """Datasets a results file of the reference configuration must carry."""
    keys = BASE_KEYS + IDENTITY_KEYS + peak_keys(n_peaks) + [
        'coarse_time', 'refine_time', 'opt_time', 'post_time', 'raw_refine_time', 'peak_time',
        'n_coarse_peaks', 'n_refined_peaks', 'n_saved_peaks', 'objective_version']
    if validation:
        keys += validation_keys(channels) + [k for k, _, _ in VALIDATION_COUNT_KEYS]
    return keys


def coherent_keys(n_waveforms):
    """Datasets of the `coherent_waveforms` group for `n_waveforms` saved peaks."""
    return ['times'] + [f'peak_{i}' for i in range(n_waveforms)]


def is_versioned_key(key):
    """True when a dataset name carries the version suffix new quantities must use."""
    return VERSIONED_KEY.search(key) is not None


def first_values(results):
    """Every key of a chunk's result dicts with the value of the first event that carries it."""
    first = {}
    for r in reversed(results):
        first.update(r)
    return first


def numeric_result_keys(results, channels, validation):
    """Datasets to write for a list of per-event result dicts, in the driver's historical order.

    The base keys always come first, then the validation keys, then every optional key
    present in a result, then every key of a result that starts with `peak_`, ends in
    `_vpol` or `_hpol`, or holds a scalar number. A key counts when any event of the
    chunk carries it, so an event with fewer saved peaks than the others does not drop
    their columns. A key that holds an array is left out (the summed waveforms of the
    polarization groups go to the `coherent_waveforms` group), except the coarse peaks
    of a polarization group, `coarse_peaks_<group>`.
    """
    numeric_keys = list(BASE_KEYS)
    if validation:
        numeric_keys.extend(validation_keys(channels))
    optional_keys = list(OPTIONAL_KEYS)
    existing = set(numeric_keys) | set(optional_keys)
    first = first_values(results)
    for k in sorted(first):
        if k in existing or k in SKIPPED_RESULT_KEYS:
            continue
        if np.ndim(first[k]) and not k.startswith('coarse_peaks_'):
            continue
        if k.startswith('peak_') or k.endswith(('_vpol', '_hpol')):
            optional_keys.append(k)
        elif isinstance(first[k], (int, float, np.floating)):
            optional_keys.append(k)
    for k in optional_keys:
        if k in first and k not in numeric_keys:
            numeric_keys.append(k)
    return numeric_keys


def stacked_peaks(peak_lists):
    """Stack the peak lists of the events of a chunk into one array.

    Args:
        peak_lists: Per event, a list of (rho, phi, z, corr) peaks; the lists may differ in length.

    Returns:
        Array of shape (n_events, n_peaks, 4), n_peaks being the longest list, with NaN rows
        where an event has fewer peaks.
    """
    peaks = [np.asarray(p, dtype=float).reshape(-1, 4) for p in peak_lists]
    out = np.full((len(peaks), max(len(p) for p in peaks), 4), np.nan)
    for row, p in zip(out, peaks):
        row[:len(p)] = p
    return out


def coherent_datasets(results):
    """Datasets of the `coherent_waveforms` group for a list of per-event result dicts.

    `coherent_times` becomes `times` and `coherent_wf_<i>` becomes `peak_<i>` with one row
    per event. With polarization groups the module returns both under each group's suffix
    as well; they become `times_<group>` and `peak_<i>_<group>`. A time axis is that of the
    first event that carries it, and an event without a waveform has a row of zeros.

    Returns:
        Dict dataset name -> array, empty when no event carries a waveform.
    """
    first = first_values(results)
    datasets = {}
    for key in sorted(first):
        if key.startswith('coherent_times'):
            datasets[key[len('coherent_'):]] = first[key]
        elif key.startswith('coherent_wf_'):
            datasets['peak_' + key[len('coherent_wf_'):]] = np.array(
                [r.get(key, np.zeros_like(first[key])) for r in results])
    return datasets


def coherent_channels(result):
    """Channels with the summed waveforms of one event, for the driver's `--save-nur` file.

    The waveform of saved peak `i` of the primary result (`coherent_wf_<i>`) becomes
    channel `COHERENT_CHANNEL_BASE + i`, at the sampling rate of `coherent_times`. The
    copies per polarization group (`coherent_wf_<i>_<group>`) are written to the results
    file only.

    Returns:
        List of Channel, empty when the reconstruction stored no waveform for the event.
    """
    channels = []
    for key in sorted(k for k in result if k.startswith('coherent_wf_')):
        index = key[len('coherent_wf_'):]
        if index.isdigit():
            times = result['coherent_times']
            channel = Channel(COHERENT_CHANNEL_BASE + int(index))
            channel.set_trace(result[key], 1.0 / (times[1] - times[0]))
            channels.append(channel)
    return channels


def write_results_h5(path, results, channels, mode, validation, attrs=None):
    """Write per-event reconstruction results to an HDF5 file.

    Args:
        path: Output file.
        results: List of per-event result dicts; each carries `run_number`,
            `event_number` and `source_file` besides the reconstruction keys.
        channels: Reconstructed channel list (names the `ch{N}_snr` datasets).
        mode: Driver mode string (`hw`, `rx` or `rxtx`).
        validation: Whether the validation datasets were requested.
        attrs: Extra file attributes (provenance) written next to `mode`,
            `n_events`, `validation` and `reco_version`.

    An event that lacks a key of the chunk has NaN in that dataset. With polarization
    groups the coarse peaks of each group are written as `coarse_peaks_<group>` of shape
    (n_events, n_peaks, 4), see `stacked_peaks`.

    With `save_coherent_waveforms` the results carry `coherent_times` and
    `coherent_wf_<i>`, written as the `times` and `peak_<i>` datasets of the
    `coherent_waveforms` group (one row per event), see `coherent_datasets`.

    Every array is built before the file is opened, so results that cannot be stacked
    raise without leaving a file.

    Side effects:
        Overwrites `path`.
    """
    columns = {}
    for key in numeric_result_keys(results, channels, validation):
        if key.startswith('coarse_peaks_'):
            columns[key] = stacked_peaks([r.get(key, []) for r in results])
        else:
            columns[key] = np.array([r.get(key, np.nan) for r in results])
    waveforms = coherent_datasets(results)
    with h5py.File(path, 'w') as f:
        grp = f.create_group('results')
        for key, data in columns.items():
            grp.create_dataset(key, data=data)
        grp.create_dataset('run_number', data=np.array([r['run_number'] for r in results], dtype=int))
        grp.create_dataset('event_number', data=np.array([r['event_number'] for r in results], dtype=int))
        filenames = [r.get('source_file', '') for r in results]
        grp.create_dataset('source_file', data=filenames, dtype=h5py.special_dtype(vlen=str))

        if waveforms:
            wf_grp = f.create_group(COHERENT_GROUP)
            for name, data in waveforms.items():
                wf_grp.create_dataset(name, data=data)

        if validation:
            for val_key, val_dtype, val_default in VALIDATION_COUNT_KEYS:
                if val_key not in grp:
                    grp.create_dataset(
                        val_key, data=np.array([r.get(val_key, val_default) for r in results], dtype=val_dtype))

        f.attrs['mode'] = mode
        f.attrs['n_events'] = len(results)
        f.attrs['validation'] = validation
        f.attrs['reco_version'] = RECO_VERSION
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
