#!/usr/bin/env python3
"""
Write the feature table of RNO-G events: one row per event with the variables of
:class:`NuRadioReco.modules.RNO_G.stationFeatureExtractor.stationFeatureExtractor`.

See README.md in this folder for the config keys, the columns and an example call.
"""
import argparse
import json
import logging
import os
import re
import sys

import h5py
import pandas as pd
import yaml

from NuRadioReco.modules.RNO_G.dataProviderNuRadio import dataProviderNuRadio
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from NuRadioReco.modules.RNO_G.stationFeatureExtractor import stationFeatureExtractor
from NuRadioReco.utilities.io_utilities import parse_event_ids

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "interferometric_reco_ex"))
from interferometric_reco_3d_advanced import init_detector  # noqa: E402

logger = logging.getLogger("NuRadioReco.examples.RNOG.feature_extraction")

CONFIG_KEYS = ("station_id", "year", "experiment_id", "detector_file", "detector_date", "preprocessor",
               "reader_kwargs", "output_root_dir", "features")


def is_selected(event_filter, file_basename, run_number, event_number):
    """
    Check an event against the filter of ``--events``.

    Parameters
    ----------
    event_filter : dict
        Result of :func:`NuRadioReco.utilities.io_utilities.parse_event_ids`
    file_basename : str
        Name of the input file or run folder the event is from
    run_number, event_number : int
        Identify the event

    Returns
    -------
    selected : bool
    """
    if "by_file" in event_filter:
        return (run_number, event_number) in event_filter["by_file"].get(file_basename, ())
    if "by_run" in event_filter:
        return event_number in event_filter["by_run"].get(run_number, ())
    return event_number in event_filter["by_event"]


def iter_events(provider, event_filter, file_basename):
    """
    Read the events of one input, preprocessed by the provider.

    With a filter only the selected events are read and preprocessed.

    Parameters
    ----------
    provider : dataProviderRNOG or dataProviderNuRadio
        The provider, after ``begin``
    event_filter : dict or None
        Result of :func:`NuRadioReco.utilities.io_utilities.parse_event_ids`; None reads every event
    file_basename : str
        Name of the input file or run folder

    Yields
    ------
    event : `NuRadioReco.framework.event.Event`
    """
    if event_filter is None:
        yield from provider.run()
        return
    for run_number, event_number in provider.get_event_ids():
        if is_selected(event_filter, file_basename, int(run_number), int(event_number)):
            yield provider.get_event(int(run_number), int(event_number))


def extract_features(config, input_files, event_filter=None):
    """
    Calculate the feature rows of all selected events of the input files.

    Parameters
    ----------
    config : dict
        The config file's content, see README.md. A key that is not in `CONFIG_KEYS` raises.
    input_files : list of str
        RNO-G run folders or ``.root`` files, or ``.nur`` files
    event_filter : dict, optional
        Result of :func:`NuRadioReco.utilities.io_utilities.parse_event_ids`

    Returns
    -------
    table : pandas.DataFrame
        One row per event: the extractor's variables, ``run_number``, ``event_number``, ``source_file``,
        ``trigger_time`` (unix time of the station) and, for ``.nur`` files named ``..lgE_<value>..``,
        ``log10_energy``
    """
    unknown = sorted(set(config) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}; the driver reads {list(CONFIG_KEYS)}")
    det = init_detector(config)
    extractor = stationFeatureExtractor()
    extractor.begin(config.get("features"))

    rows = []
    # one input at a time, so that every row can name the file it is from
    for path in input_files:
        file_basename = os.path.basename(path)
        if event_filter is not None and "by_file" in event_filter and file_basename not in event_filter["by_file"]:
            continue
        if path.endswith(".nur"):
            provider = dataProviderNuRadio()
            provider.begin(path, det, preprocessor_config=config.get("preprocessor"))
        else:
            provider = dataProviderRNOG()
            reader_kwargs = dict(config.get("reader_kwargs") or {})
            reader_kwargs["mattak_kwargs"] = {"read_daq_status": False, "backend": "uproot",
                                              **reader_kwargs.get("mattak_kwargs", {})}
            provider.begin(path, det, reader_kwargs=reader_kwargs, preprocessor_config=config.get("preprocessor"))
        energy = re.search(r"lgE_?([0-9.]+)", file_basename) if path.endswith(".nur") else None
        for event in iter_events(provider, event_filter, file_basename):
            station = event.get_station()
            row = extractor.run(event, station, det)
            row["run_number"] = event.get_run_number()
            row["event_number"] = event.get_id()
            row["source_file"] = path
            row["trigger_time"] = float(station.get_station_time().unix)
            if energy:
                row["log10_energy"] = float(energy.group(1))
            rows.append(row)
        provider.end()

    extractor.end()
    return pd.DataFrame(rows)


def output_path(config, run_chunk, is_simulation):
    """
    Build the path of the table of one chunk.

    Parameters
    ----------
    config : dict
        Uses ``output_root_dir`` (default: the environment variable ``FEATURE_OUTPUT_ROOT``, then
        ``./feature_extraction``), ``experiment_id``, ``station_id`` and ``year``
    run_chunk : str
        Name of the chunk
    is_simulation : bool
        Simulated events go under ``sim_data``, measured ones under ``real_data``

    Returns
    -------
    path : str
        ``<root>/results/<sim_data|real_data>/<experiment>/station<id>/<year>/`` and in it
        ``station<id>_features_df_chunk<chunk>_<experiment>.h5``
    """
    root = config.get("output_root_dir",
                      os.environ.get("FEATURE_OUTPUT_ROOT", os.path.join(os.getcwd(), "feature_extraction")))
    experiment = config.get("experiment_id", "default")
    station = f"station{config.get('station_id', 0)}"
    return os.path.join(root, "results", "sim_data" if is_simulation else "real_data", experiment, station,
                        str(config.get("year", 0)), f"{station}_features_df_chunk{run_chunk}_{experiment}.h5")


def write_table(table, path, config):
    """
    Write the feature table and the config it was made with.

    Parameters
    ----------
    table : pandas.DataFrame
        Result of `extract_features`
    path : str
        Output file. The table goes to the key ``data`` sorted by run and event number, the config to the
        attributes of the group ``config`` (lists and dictionaries as JSON).
    config : dict
        The config file's content
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = table.sort_values(["run_number", "event_number"])
    table.to_hdf(path, key="data", mode="w", format="table", complevel=5)
    with h5py.File(path, "a") as f:
        group = f.require_group("config")
        for key, value in config.items():
            group.attrs[key] = value if isinstance(value, (str, int, float)) else json.dumps(value, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNO-G feature extraction: one table row per event")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("-i", "--input", type=str, nargs="+", required=True,
                        help="RNO-G run folders or .root files, or .nur files")
    parser.add_argument("--station_id", type=int, default=None, help="replaces station_id of the config")
    parser.add_argument("--year", type=int, default=None, help="replaces year of the config")
    parser.add_argument("--experiment_id", type=str, default=None, help="replaces experiment_id of the config")
    parser.add_argument("--run_chunk", type=str, default="0", help="name of the chunk in the output file name")
    parser.add_argument("--events", type=str, nargs="+", default=None,
                        help="event numbers, or a JSON file {run: [events]} or {file name: [[run, event], ...]}; "
                             "see NuRadioReco.utilities.io_utilities.parse_event_ids")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    for key in ("station_id", "year", "experiment_id"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key, value in config.items():
        if isinstance(value, str):
            config[key] = os.path.expandvars(value)

    table = extract_features(config, args.input, parse_event_ids(args.events) if args.events else None)
    if table.empty:
        logger.warning("no events, no table written")
    else:
        path = output_path(config, args.run_chunk, args.input[0].endswith(".nur"))
        write_table(table, path, config)
        logger.info(f"wrote {len(table)} events to {path}")
