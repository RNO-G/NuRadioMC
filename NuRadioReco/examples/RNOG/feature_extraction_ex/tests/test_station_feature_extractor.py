"""The rows of `stationFeatureExtractor` on synthetic events: columns, values, missing channels and config."""
import numpy as np
import pytest

from NuRadioReco.modules.RNO_G.stationFeatureExtractor import stationFeatureExtractor
from NuRadioReco.utilities import trace_utilities, units

from synthetic import DEEP_CHANNELS, SAMPLING_RATE, make_event

GROUPS = {"pa": (0, 1, 2, 3), "vpol": (0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23), "hpol": (4, 8, 11, 21),
          "deep": DEEP_CHANNELS}
CHANNEL_VARIABLES = 31
MEAN_VARIABLES = 25
SUMMED_VARIABLES = 22
POLARIZATION_VARIABLES = 7


def run(event, config=None):
    """Return the row of one event for a config."""
    extractor = stationFeatureExtractor()
    extractor.begin(config)
    row = extractor.run(event, event.get_station(), None)
    extractor.end()
    return row


def test_columns_of_the_default_config():
    row = run(make_event(1))
    assert len(row) == (CHANNEL_VARIABLES * len(DEEP_CHANNELS) + (MEAN_VARIABLES + SUMMED_VARIABLES) * len(GROUPS)
                        + POLARIZATION_VARIABLES)
    assert all(isinstance(value, float) and np.isfinite(value) for value in row.values())
    for ch in DEEP_CHANNELS:
        assert f"ch{ch}_snr" in row and f"ch{ch}_impulse_corr_sinc" in row
    assert "ch12_snr" not in row
    for group in GROUPS:
        assert f"snr_avg_{group}" in row and f"coherent_spectral_slope_{group}" in row
        for name in ("noise_rms", "root_power_ratio", "impulsivity_slope"):
            assert f"{name}_avg_{group}" not in row
        assert f"coherent_noise_rms_{group}" not in row and f"coherent_band_snr_{group}" not in row
    assert not any("hit" in name for name in row)


def test_values_are_those_of_the_trace_utilities():
    event = make_event(2)
    row = run(event)
    traces = {ch.get_id(): ch.get_trace() for ch in event.get_station().iter_channels()}

    trace = traces[5]
    noise_rms = trace_utilities.get_split_trace_noise_RMS(trace)
    assert row["ch5_noise_rms"] == noise_rms
    assert row["ch5_snr"] == trace_utilities.get_signal_to_noise_ratio(trace, noise_rms)
    assert row["ch5_root_power_ratio"] == trace_utilities.get_root_power_ratio(
        trace, np.arange(len(trace)) / SAMPLING_RATE, noise_rms)
    assert row["ch5_max_amplitude_envelope"] == np.max(trace_utilities.get_hilbert_envelope(trace))
    assert row["ch5_impulsivity"] == trace_utilities.get_impulsivity(trace)
    assert row["ch5_kurtosis"] == trace_utilities.get_kurtosis(trace)
    assert row["ch5_entropy"] == trace_utilities.get_entropy(trace)
    spectral = trace_utilities.get_spectral_features(trace, SAMPLING_RATE, fmin=0.08 * units.GHz, fmax=0.6 * units.GHz,
                                                     low_band_boundary=0.1 * units.GHz)
    assert row["ch5_spectral_centroid"] == spectral["spectral_centroid"]
    band = trace_utilities.get_band_features(trace, SAMPLING_RATE, 0.1 * units.GHz, 0.3 * units.GHz,
                                             fmin=0.08 * units.GHz, fmax=0.6 * units.GHz)
    assert row["ch5_band_snr"] == band["band_snr"]
    assert row["ch5_impulse_corr_bipolar"] == trace_utilities.get_impulse_template_correlations(
        trace, SAMPLING_RATE)["bipolar"]

    for group, channels in GROUPS.items():
        assert row[f"entropy_avg_{group}"] == np.mean([row[f"ch{ch}_entropy"] for ch in channels])
        summed = trace_utilities.get_coherent_sum([traces[ch] for ch in channels[1:]], traces[channels[0]])
        assert row[f"coherent_impulsivity_{group}"] == trace_utilities.get_impulsivity(summed)
        assert row[f"coherent_snr_{group}"] == trace_utilities.get_signal_to_noise_ratio(
            summed, trace_utilities.get_split_trace_noise_RMS(summed))

    assert row["hpol_vpol_band_snr_ratio"] == row["band_snr_avg_hpol"] / row["band_snr_avg_vpol"]
    v = np.mean([row["ch9_band_snr"], row["ch10_band_snr"]])
    assert row["pol_fraction_b"] == v / (v + row["ch11_band_snr"])
    corr, lag = trace_utilities.get_normalized_cross_correlation(traces[21], traces[22])
    assert row["hv_xcorr_c"] == corr and row["hv_dt_c"] == lag
    # the same pulse two samples later on the next channel
    assert abs(row["hv_dt_b"]) <= 6 and row["hv_xcorr_b"] > 0.25


def test_pulse_stands_out_from_noise():
    signal, noise = run(make_event(3)), run(make_event(3, pulse_amplitude=0.0))
    for group in GROUPS:
        assert signal[f"snr_avg_{group}"] > 2 * noise[f"snr_avg_{group}"]
        assert signal[f"impulsivity_avg_{group}"] > noise[f"impulsivity_avg_{group}"] + 0.2
        assert signal[f"coherent_snr_{group}"] > signal[f"snr_avg_{group}"]
    assert signal["hv_xcorr_b"] > 3 * noise["hv_xcorr_b"]


def test_missing_channels():
    row = run(make_event(4, missing=(4, 8, 11, 21, 9)))
    assert "ch9_snr" not in row and "ch11_snr" not in row
    assert np.isnan(row["snr_avg_hpol"]) and np.isnan(row["hpol_vpol_band_snr_ratio"])
    assert "coherent_snr_hpol" not in row and "coherent_snr_deep" in row
    assert row["snr_avg_vpol"] == np.mean([row[f"ch{ch}_snr"] for ch in GROUPS["vpol"] if ch != 9])
    for pair in ("b", "c"):
        assert np.isnan(row[f"pol_fraction_{pair}"]) and np.isnan(row[f"hv_xcorr_{pair}"])
        assert np.isnan(row[f"hv_dt_{pair}"])

    # the second vertically polarised antenna of the pair takes over
    row = run(make_event(4, missing=(9,)))
    assert row["pol_fraction_b"] == row["ch10_band_snr"] / (row["ch10_band_snr"] + row["ch11_band_snr"])
    assert np.isfinite(row["hv_xcorr_b"])


def test_feature_groups_and_band_edges():
    event = make_event(5)
    full = run(event)
    band = run(event, {"feature_groups": ["band"]})
    assert len(band) == 5 * len(DEEP_CHANNELS) + 5 * len(GROUPS) + POLARIZATION_VARIABLES
    assert all(full[name] == value for name, value in band.items())

    wide = run(event, {"feature_groups": ["band"], "band_lo": 0.15 * units.GHz, "band_hi": 0.6 * units.GHz})
    assert set(wide) == set(band)
    assert wide["ch0_band_power"] != band["ch0_band_power"]
    assert wide["ch0_peak_frequency"] == band["ch0_peak_frequency"]

    slope = run(event, {"feature_groups": ["spectral"], "spectral_fmin": 0.15 * units.GHz,
                        "spectral_fmax": 0.35 * units.GHz})
    assert slope["ch0_spectral_slope"] != full["ch0_spectral_slope"]
    assert "coherent_spectral_slope_pa" in slope and "ch0_snr" not in slope

    without_sums = run(event, {"build_coherent_sums": False})
    assert set(without_sums) == {name for name in full if not name.startswith("coherent_")}

    rpr = run(event, {"feature_groups": ["rpr"], "build_coherent_sums": False})
    assert set(rpr) == {f"ch{ch}_{name}" for ch in DEEP_CHANNELS for name in ("noise_rms", "root_power_ratio")}


def test_other_channel_groups():
    row = run(make_event(6), {"channel_groups": {"surface": [13, 16, 19], "pa": [0, 1, 2, 3]}, "hv_pairs": {}})
    assert {name.split("_")[0] for name in row if name.startswith("ch")} == {"ch0", "ch1", "ch2", "ch3", "ch13",
                                                                              "ch16", "ch19"}
    assert row["kurtosis_avg_surface"] == np.mean([row[f"ch{ch}_kurtosis"] for ch in (13, 16, 19)])
    assert "coherent_snr_surface" in row
    assert not any(name.startswith(("hpol_vpol", "pol_fraction", "hv_")) for name in row)


def test_config_errors():
    extractor = stationFeatureExtractor()
    with pytest.raises(ValueError, match="band_low"):
        extractor.begin({"band_low": 0.1})
    with pytest.raises(ValueError, match="spectrum"):
        extractor.begin({"feature_groups": ["snr", "spectrum"]})


def test_summed_trace_features():
    event = make_event(7)
    extractor = stationFeatureExtractor()
    extractor.begin()
    row = extractor.run(event, event.get_station(), None)
    assert extractor.get_channel_groups() == GROUPS

    station = event.get_station()
    summed = trace_utilities.get_coherent_sum([station.get_channel(ch).get_trace() for ch in (1, 2, 3)],
                                              station.get_channel(0).get_trace())
    features = extractor.get_summed_trace_features(summed, SAMPLING_RATE)
    assert len(features) == SUMMED_VARIABLES and "noise_rms" not in features
    assert all(row[f"coherent_{name}_pa"] == value for name, value in features.items())


def test_hit_filter_columns():
    columns = ["passed_hit_filter", "n_coincident_pairs_pa", "n_high_hits_pa", "n_coincident_pairs_deep",
               "n_high_hits_deep"]
    extractor = stationFeatureExtractor()
    extractor.begin({"hit_filter": True})

    event = make_event(8, pulse_amplitude=12.0)
    row = extractor.run(event, event.get_station(), None)
    assert list(row)[-5:] == columns
    assert [row[name] for name in columns] == [1, 6, 4, 9, 15]

    event = make_event(8, pulse_amplitude=0.0)
    row = extractor.run(event, event.get_station(), None)
    assert row["passed_hit_filter"] == 0 and row["n_high_hits_pa"] == 0 and row["n_high_hits_deep"] == 0
    extractor.end()

    # the filter leaves the traces as they are
    event = make_event(8)
    assert {name: value for name, value in extractor.run(event, event.get_station(), None).items()
            if name not in columns} == run(make_event(8))
