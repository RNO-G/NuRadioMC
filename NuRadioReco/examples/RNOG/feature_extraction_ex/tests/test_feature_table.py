"""The driver around `stationFeatureExtractor`: event selection, output path and the written table."""
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from NuRadioReco.modules.RNO_G.stationFeatureExtractor import stationFeatureExtractor
from NuRadioReco.utilities.io_utilities import parse_event_ids

from feature_extraction import extract_features, is_selected, iter_events, output_path, write_table
from synthetic import make_event


class ListProvider:
    """A data provider that serves synthetic events of run 1 and counts how many it handed out."""

    def __init__(self, event_ids):
        self.events = {event_id: make_event(event_id, event_id=event_id) for event_id in event_ids}
        self.n_read = 0

    def run(self):
        """Yield every event."""
        for event_id in self.events:
            yield self.get_event(1, event_id)

    def get_event_ids(self):
        """Return the (run, event) pairs as the providers do."""
        return np.array([(1, event_id) for event_id in self.events])

    def get_event(self, run_number, event_number):
        """Return one event."""
        self.n_read += 1
        return self.events[event_number]


def test_is_selected_for_the_three_filter_forms(tmp_path):
    by_event = parse_event_ids(["3", "5"])
    assert is_selected(by_event, "run7", 7, 3) and not is_selected(by_event, "run7", 7, 4)

    path = tmp_path / "by_run.json"
    path.write_text(json.dumps({"7": [3, 5], "8": [1]}))
    by_run = parse_event_ids([str(path)])
    assert is_selected(by_run, "run7", 7, 5) and is_selected(by_run, "run8", 8, 1)
    assert not is_selected(by_run, "run8", 8, 3) and not is_selected(by_run, "run9", 9, 3)

    path = tmp_path / "by_file.json"
    path.write_text(json.dumps({"a.nur": [[0, 3]], "b.nur": [[0, 4]]}))
    by_file = parse_event_ids([str(path)])
    assert is_selected(by_file, "a.nur", 0, 3) and is_selected(by_file, "b.nur", 0, 4)
    assert not is_selected(by_file, "a.nur", 0, 4) and not is_selected(by_file, "c.nur", 0, 3)


def test_iter_events_reads_only_the_selected_events():
    provider = ListProvider([2, 3, 5, 8])
    assert [event.get_id() for event in iter_events(provider, None, "run1")] == [2, 3, 5, 8]

    provider = ListProvider([2, 3, 5, 8])
    selected = list(iter_events(provider, parse_event_ids(["5", "2", "9"]), "run1"))
    assert [event.get_id() for event in selected] == [2, 5]
    assert provider.n_read == 2


def test_config_key_outside_the_driver_keys_raises():
    with pytest.raises(ValueError, match="hit_filter"):
        extract_features({"station_id": 13, "hit_filter": {"enabled": True}}, [])


def test_output_path(monkeypatch, tmp_path):
    config = {"station_id": 13, "year": 2022, "experiment_id": "exp", "output_root_dir": "/root"}
    assert output_path(config, "4_1000", False) == (
        "/root/results/real_data/exp/station13/2022/station13_features_df_chunk4_1000_exp.h5")
    assert "/results/sim_data/exp/" in output_path(config, "0", True)

    del config["output_root_dir"]
    monkeypatch.setenv("FEATURE_OUTPUT_ROOT", str(tmp_path))
    assert output_path(config, "0", True).startswith(str(tmp_path / "results"))


def test_written_table_reads_back(tmp_path):
    extractor = stationFeatureExtractor()
    extractor.begin({"hit_filter": True})
    rows = []
    for event_id in (5, 2, 3):
        event = make_event(event_id, event_id=event_id)
        row = extractor.run(event, event.get_station(), None)
        row.update(run_number=event.get_run_number(), event_number=event.get_id(), source_file="run1",
                   trigger_time=float(event.get_station().get_station_time().unix))
        rows.append(row)
    table = pd.DataFrame(rows)
    config = {"station_id": 13, "experiment_id": "exp", "preprocessor": {"apply_bandpass": True},
              "features": {"hit_filter": True}, "detector_file": None}

    path = str(tmp_path / "sub" / "features.h5")
    write_table(table, path, config)

    read = pd.read_hdf(path, key="data")
    assert list(read["event_number"]) == [2, 3, 5]
    assert len(read.columns) == 665 + 4
    pd.testing.assert_frame_equal(read.reset_index(drop=True),
                                  table.sort_values("event_number").reset_index(drop=True)[list(read.columns)])
    assert read["trigger_time"].iloc[1] - read["trigger_time"].iloc[0] == 1.0
    with h5py.File(path, "r") as f:
        attrs = dict(f["config"].attrs)
    assert attrs["station_id"] == 13 and attrs["experiment_id"] == "exp"
    assert json.loads(attrs["preprocessor"]) == {"apply_bandpass": True}
    assert json.loads(attrs["features"]) == {"hit_filter": True} and json.loads(attrs["detector_file"]) is None
