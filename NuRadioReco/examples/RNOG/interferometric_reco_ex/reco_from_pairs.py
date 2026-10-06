#!/usr/bin/env python3
"""Run the search stage of the 3D reconstruction on a pair store and write a results file.

With the store's own configuration and no option the results equal the pass-1 results of
the driver run that wrote the store bit for bit: its results file for ``--mode hw``, its
``pass1_*`` fields for ``rx`` and ``rxtx`` (pass 2 dedisperses the waveforms and cannot be
run from pairs). ``--config``, ``--mask``, ``--polarity`` and ``--delay-shift`` run the same
search on the same series under that variation (``reconstruct_from_pairs``).
``--calibration FILE`` evaluates a delay-corrections file (with its optional polarity
block) in place of the one the store was made with: the difference of the two files
becomes the delay shifts and polarities, added to any given on the command line, and every
event must lie in the new file's validity window for its station. The file's
``position_shift`` block (none: no shift) replaces the store configuration's
``channel_position_shift``; positions do not enter the stored series.
"""

import argparse
import datetime
import logging
import os
import time

import numpy as np
import yaml

from NuRadioReco.modules.RNO_G.channelPreprocessor import load_delay_corrections
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

from interferometric_reco_3d_advanced import init_detector, objective_attrs
from pair_store import PairStore, calibration_trial
from reco_output import write_results_h5


def channel_map(items, cast):
    """Parse 'channel:value' arguments into a dict channel -> cast(value)."""
    return {int(ch): cast(value) for ch, value in (item.split(':') for item in items or [])}


def check_validity(calibration, station_id, unix_time):
    """Raise when an event lies outside a calibration's window for its station.

    A station the calibration does not list is left unchecked: the file changes nothing there.

    Raises:
        ValueError: If the event has no time or its date lies outside the station's window.
    """
    if station_id not in calibration.station_windows:
        return
    if not np.isfinite(unix_time):
        raise ValueError(f"{calibration.path} needs the event time to check its validity")
    day = datetime.datetime.fromtimestamp(unix_time, datetime.timezone.utc).date()
    valid_from, valid_to = calibration.station_windows[station_id]
    if not valid_from <= day <= valid_to:
        raise ValueError(f"{calibration.path} is valid for station {station_id} from {valid_from} "
                         f"to {valid_to}; event at {day}")


def main():
    """Reconstruct every event of a pair store and write the results file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", required=True, help="Pair-store HDF5 file")
    parser.add_argument("-o", "--outputfile", required=True)
    parser.add_argument("--config", default=None,
                        help="Reconstruction config YAML (default: the config stored in the file)")
    parser.add_argument("--mask", type=int, nargs="+", default=None,
                        help="Channels removed from the search")
    parser.add_argument("--polarity", nargs="+", default=None,
                        help="Channel polarities as channel:sign, e.g. 6:-1 7:-1")
    parser.add_argument("--delay-shift", nargs="+", default=None,
                        help="Delay shifts as channel:ns, added to the cable delay")
    parser.add_argument("--calibration", default=None,
                        help="Delay-corrections file to evaluate instead of the one the store was made with")
    parser.add_argument("--max_events", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    store = PairStore(args.pairs)
    if args.config is None:
        config = store.config
    else:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for block in (config, config.get('preprocessor') or {}):
            for key, val in block.items():
                if isinstance(val, str) and '$' in val:
                    block[key] = os.path.expandvars(val)
    polarity = channel_map(args.polarity, int)
    shift = channel_map(args.delay_shift, float)
    trial = applied = None
    if args.calibration:
        trial = load_delay_corrections(args.calibration)
        applied = store.applied_calibration()

    reco = InterferometricReco3D()
    reco.begin(config['station_id'], config, init_detector(config))
    results = []
    n_events = len(store) if args.max_events is None else min(len(store), args.max_events)
    for i in range(n_events):
        t0 = time.time()
        pairs = store.event(i)
        t_read = time.time() - t0
        event_shift, event_polarity, event_position = shift, polarity, None
        if trial is not None:
            station_id = int(store.station_id[i])
            check_validity(trial, station_id, store.time[i])
            delta, signs = calibration_trial(applied, trial, station_id)
            event_shift = {ch: shift.get(ch, 0.0) + delta.get(ch, 0.0) for ch in set(shift) | set(delta)}
            event_polarity = {ch: polarity.get(ch, 1) * signs.get(ch, 1)
                              for ch in set(polarity) | set(signs)}
            event_position = trial.position_shift.get(station_id, {})
        result = reco.reconstruct_from_pairs(pairs, config, channel_mask=args.mask,
                                             channel_delay_shift=event_shift,
                                             channel_polarity=event_polarity,
                                             channel_position_shift=event_position)
        result['preproc_time'] = t_read
        result['run_number'] = int(store.run_number[i])
        result['event_number'] = int(store.event_number[i])
        result['source_file'] = store.source_file[i]
        results.append(result)
    reco.end()

    attrs = dict(store.results_attrs)
    attrs.update(objective_attrs(config))
    attrs.update({'pair_store_file': os.path.abspath(args.pairs),
                  'pair_store_config_hash': store.file.attrs['config_hash']})
    if trial is not None:
        attrs.update({'delay_corrections_hash': trial.sha256, 'delay_corrections_file': trial.path})
    store.close()
    if results:
        write_results_h5(args.outputfile, results, config['channels'], 'hw',
                         bool(config.get('validation', False)), attrs)
    print(f"Reconstructed {len(results)} events from {args.pairs}")


if __name__ == "__main__":
    main()
