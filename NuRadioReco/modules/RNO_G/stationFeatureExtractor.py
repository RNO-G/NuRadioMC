from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.RNO_G.stationHitFilter import stationHitFilter
from NuRadioReco.utilities import trace_utilities, units

import numpy as np


class stationFeatureExtractor:
    """
    Calculates the variables of one RNO-G event that describe how signal-like its waveforms are.

    The module reads the channel traces as they are: preprocessing (filters, CW removal, resampling)
    is the job of the modules that run before it, for example
    :class:`NuRadioReco.modules.RNO_G.channelPreprocessor.channelPreprocessor` through one of the data providers.

    For every channel of the configured channel groups the variables of the enabled feature groups are calculated
    with the functions of :mod:`NuRadioReco.utilities.trace_utilities`:

    - ``snr``: noise RMS of the quietest trace segments and the peak-to-peak signal-to-noise ratio
    - ``rpr``: root power ratio
    - ``max_amplitude``: largest peak-to-peak amplitude of the trace in units of its standard deviation,
      and the maximum of the Hilbert envelope
    - ``impulsivity``: impulsivity, and the slope, intercept, R squared and Kolmogorov-Smirnov distance
      that say how linear its cumulative distribution is
    - ``kurtosis_entropy``: kurtosis and entropy of the sample values
    - ``spectral``: centroid, bandwidth, skewness, kurtosis, entropy, slope, 90 % roll-off, flatness
      and low-band fraction of the power spectrum
    - ``band``: power, noise-referenced signal-to-noise ratio, power fraction, slope and peak frequency in one band
    - ``impulse_correlations``: largest correlation with five idealised impulse shapes

    From these the station-level variables are built: the mean over the channels of each channel group, the same
    variables on the sum of each group's traces aligned by cross correlation, the comparison of horizontally and
    vertically polarised antennas, and the counts of the station hit filter. The result of
    :meth:`run` is one flat dictionary per event, ready to become a table row. Its keys:

    - ``ch{id}_{variable}`` for each channel
    - ``{variable}_avg_{group}``: mean over the channels of a group that are in the station. The noise RMS,
      the root power ratio and the four linearity variables of the impulsivity have no mean.
    - ``coherent_{variable}_{group}``: variables of the summed trace of a group (feature groups ``snr``,
      ``impulsivity``, ``kurtosis_entropy``, ``spectral``, ``impulse_correlations``)
    - with the feature group ``band``: ``hpol_vpol_band_snr_ratio`` (ratio of the group means of the groups
      named ``hpol`` and ``vpol``, if both are configured), and for each pair of ``hv_pairs``
      ``pol_fraction_{pair}`` (V / (V + H) of the band signal-to-noise ratio), ``hv_xcorr_{pair}`` and
      ``hv_dt_{pair}`` (largest normalised cross correlation of the two traces and its lag in samples)
    - with ``hit_filter``: ``passed_hit_filter``, ``n_coincident_pairs_pa``, ``n_high_hits_pa``,
      ``n_coincident_pairs_deep``, ``n_high_hits_deep``

    A variable that cannot be calculated because its channels are missing is NaN.
    """

    FEATURE_GROUPS = ("snr", "rpr", "max_amplitude", "impulsivity", "kurtosis_entropy",
                      "spectral", "band", "impulse_correlations")

    _SUMMED_TRACE_GROUPS = ("snr", "impulsivity", "kurtosis_entropy", "spectral", "impulse_correlations")

    _NO_GROUP_MEAN = ("noise_rms", "root_power_ratio", "impulsivity_slope", "impulsivity_intercept",
                      "impulsivity_r_squared", "impulsivity_ks_statistic")

    _DEFAULT_CONFIG = {
        "channel_groups": {
            "pa": [0, 1, 2, 3],
            "vpol": [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
            "hpol": [4, 8, 11, 21],
            "deep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23],
        },
        "hv_pairs": {
            "b": {"hpol": 11, "vpol": [9, 10]},
            "c": {"hpol": 21, "vpol": [22, 23]},
        },
        "feature_groups": list(FEATURE_GROUPS),
        "build_coherent_sums": True,
        "hit_filter": False,
        "spectral_fmin": 0.08 * units.GHz,
        "spectral_fmax": 0.6 * units.GHz,
        "spectral_low_band_boundary": 0.1 * units.GHz,
        "band_lo": 0.1 * units.GHz,
        "band_hi": 0.3 * units.GHz,
    }

    def __init__(self):
        self.begin()

    def begin(self, config=None):
        """
        Set the configuration.

        Parameters
        ----------
        config : dict, optional
            Settings that replace the defaults. An unknown key raises. Keys:

            channel_groups : dict
                Group name to list of channel ids. Variables are calculated for every channel of every group.
                Default: the RNO-G groups ``pa`` (phased array), ``vpol``, ``hpol`` and ``deep`` (all antennas
                on the strings).
            hv_pairs : dict
                Pair name to ``{"hpol": id, "vpol": [ids]}``: a horizontally polarised antenna and the vertically
                polarised antennas next to it, the first one being the partner of the cross correlation.
                Default: the pairs on the two RNO-G helper strings.
            feature_groups : list of str
                The feature groups to calculate, from `FEATURE_GROUPS`. Default: all.
            build_coherent_sums : bool
                Calculate the variables of each group's summed trace. Default: True
            hit_filter : bool
                Run the hit filter of :mod:`NuRadioReco.modules.RNO_G.stationHitFilter` and add its result
                and counts. Default: False
            spectral_fmin, spectral_fmax : float
                Frequency range of the ``spectral`` variables and of the peak frequency and power fraction of
                ``band``. Default: 0.08 GHz, 0.6 GHz
            spectral_low_band_boundary : float
                Upper edge of the low band of ``low_band_fraction``. Default: 0.1 GHz
            band_lo, band_hi : float
                Edges of the band of the ``band`` variables. Default: 0.1 GHz, 0.3 GHz
        """
        config = config or {}
        unknown = sorted(set(config) - set(self._DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"unknown stationFeatureExtractor config keys: {unknown}")
        self._config = {**self._DEFAULT_CONFIG, **config}

        unknown = sorted(set(self._config["feature_groups"]) - set(self.FEATURE_GROUPS))
        if unknown:
            raise ValueError(f"unknown feature groups: {unknown}; available: {list(self.FEATURE_GROUPS)}")
        self._feature_groups = set(self._config["feature_groups"])
        self._channel_groups = {name: tuple(ids) for name, ids in self._config["channel_groups"].items()}
        self._channel_ids = sorted({ch for ids in self._channel_groups.values() for ch in ids})

        self._hit_filter = None
        if self._config["hit_filter"]:
            # the counts need the checks of all pairs and channels, not only the pass decision
            self._hit_filter = stationHitFilter(complete_time_check=True, complete_hit_check=True)
            self._hit_filter.begin()

    @register_run()
    def run(self, event, station, det=None):
        """
        Calculate the variables of one event.

        Parameters
        ----------
        event : `NuRadioReco.framework.event.Event`
            The event (the hit filter reads its trigger)
        station : `NuRadioReco.framework.station.Station`
            The station with preprocessed channel traces
        det : Detector, optional
            Not used; kept for the common signature of the modules

        Returns
        -------
        row : dict
            Variable name to value, see the class description
        """
        hit_filter_row = {}
        if self._hit_filter is not None:
            passed = self._hit_filter.run(event, station, det)
            # one list of pair flags per channel group of the filter: the phased array first, then three
            # pairs of antennas on the strings; one threshold flag per channel, the phased array first
            in_time_window = self._hit_filter.is_in_time_window()
            over_threshold = self._hit_filter.is_over_hit_threshold()
            n_pairs_pa = int(sum(in_time_window[0]))
            hit_filter_row = {
                "passed_hit_filter": int(passed),
                "n_coincident_pairs_pa": n_pairs_pa,
                "n_high_hits_pa": int(sum(over_threshold[:4])),
                "n_coincident_pairs_deep": n_pairs_pa + int(sum(pairs[0] for pairs in in_time_window[1:])),
                "n_high_hits_deep": int(sum(over_threshold)),
            }

        traces, sampling_rates, per_channel = {}, {}, {}
        for channel in station.iter_channels(use_channels=self._channel_ids):
            ch = channel.get_id()
            traces[ch] = np.asarray(channel.get_trace())
            sampling_rates[ch] = channel.get_sampling_rate()
            per_channel[ch] = self._trace_features(traces[ch], sampling_rates[ch], self._feature_groups)

        row = {f"ch{ch}_{name}": value for ch, features in per_channel.items() for name, value in features.items()}

        names = [name for name in next(iter(per_channel.values()), {}) if name not in self._NO_GROUP_MEAN]
        for name in names:
            for group, ids in self._channel_groups.items():
                values = [per_channel[ch][name] for ch in ids if ch in per_channel]
                row[f"{name}_avg_{group}"] = float(np.mean(values)) if values else np.nan

        if "band" in self._feature_groups:
            row.update(self._polarization_features(row, per_channel, traces))

        if self._config["build_coherent_sums"]:
            for group, ids in self._channel_groups.items():
                available = [ch for ch in ids if ch in traces]
                if not available:
                    continue
                # the first channel of the group is the reference the others are aligned to
                summed = traces[available[0]]
                if len(available) > 1:
                    summed = trace_utilities.get_coherent_sum([traces[ch] for ch in available[1:]], summed)
                features = self.get_summed_trace_features(summed, sampling_rates[available[0]])
                row.update({f"coherent_{name}_{group}": value for name, value in features.items()})

        row.update(hit_filter_row)
        return row

    def end(self):
        """Log the summary of the hit filter, if it ran."""
        if self._hit_filter is not None:
            self._hit_filter.end()

    def get_channel_groups(self):
        """
        Returns
        -------
        channel_groups : dict
            Group name to tuple of channel ids, as configured
        """
        return self._channel_groups

    def get_summed_trace_features(self, trace, sampling_rate):
        """
        Calculate the variables of a trace that is the sum of several channels.

        These are the variables of the enabled feature groups among ``snr``, ``impulsivity``, ``kurtosis_entropy``,
        ``spectral`` and ``impulse_correlations``, without the noise RMS. :meth:`run` calls this for the sum
        aligned by cross correlation; callers that align the traces themselves, for example with the travel times
        of a reconstructed position, get the same variables for their sum.

        Parameters
        ----------
        trace : array of floats
            The summed trace
        sampling_rate : float
            Its sampling rate

        Returns
        -------
        features : dict
            Variable name (without prefix) to value
        """
        features = self._trace_features(
            trace, sampling_rate, self._feature_groups.intersection(self._SUMMED_TRACE_GROUPS))
        features.pop("noise_rms", None)
        return features

    def _trace_features(self, trace, sampling_rate, feature_groups):
        """
        Calculate the variables of the given feature groups for one trace.

        Parameters
        ----------
        trace : array of floats
            The trace
        sampling_rate : float
            Its sampling rate
        feature_groups : set of str
            Names from `FEATURE_GROUPS`

        Returns
        -------
        features : dict
            Variable name to float
        """
        cfg = self._config
        features = {}

        if "snr" in feature_groups or "rpr" in feature_groups:
            noise_rms = float(trace_utilities.get_split_trace_noise_RMS(trace))
            features["noise_rms"] = noise_rms
        if "snr" in feature_groups:
            features["snr"] = float(trace_utilities.get_signal_to_noise_ratio(trace, noise_rms))
        if "rpr" in feature_groups:
            times = np.arange(len(trace)) / sampling_rate
            features["root_power_ratio"] = float(trace_utilities.get_root_power_ratio(trace, times, noise_rms))

        if "max_amplitude" in feature_groups or "impulsivity" in feature_groups:
            envelope = trace_utilities.get_hilbert_envelope(trace)
        if "max_amplitude" in feature_groups:
            std = np.std(trace)
            normalized = trace / std if std > 0 else trace
            features["max_amplitude_norm"] = float(np.amax(
                trace_utilities.get_maximum_peak_to_peak_amplitude(normalized)))
            features["max_amplitude_envelope"] = float(np.amax(envelope))
        if "impulsivity" in feature_groups:
            features.update(trace_utilities.get_impulsivity(trace, envelope=envelope, return_diagnostics=True))

        if "kurtosis_entropy" in feature_groups:
            features["kurtosis"] = float(trace_utilities.get_kurtosis(trace))
            features["entropy"] = float(trace_utilities.get_entropy(trace))

        if "spectral" in feature_groups:
            spectral = trace_utilities.get_spectral_features(
                trace, sampling_rate, fmin=cfg["spectral_fmin"], fmax=cfg["spectral_fmax"],
                low_band_boundary=cfg["spectral_low_band_boundary"])
            features.update({name: float(value) for name, value in spectral.items()})

        if "band" in feature_groups:
            band = trace_utilities.get_band_features(
                trace, sampling_rate, band_lo=cfg["band_lo"], band_hi=cfg["band_hi"],
                fmin=cfg["spectral_fmin"], fmax=cfg["spectral_fmax"])
            features.update({name: float(value) for name, value in band.items()})

        if "impulse_correlations" in feature_groups:
            correlations = trace_utilities.get_impulse_template_correlations(trace, sampling_rate)
            features.update({f"impulse_corr_{name}": value for name, value in correlations.items()})

        return features

    def _polarization_features(self, row, per_channel, traces):
        """
        Compare horizontally and vertically polarised antennas.

        The band signal-to-noise ratio is referenced to the noise of the same channel in the same band, so the
        gain of a channel cancels; the cross correlation is normalised by both traces. A polarised signal shows
        in both, thermal noise in neither.

        Parameters
        ----------
        row : dict
            The row so far, with the group means of ``band_snr``
        per_channel : dict
            Channel id to its variables
        traces : dict
            Channel id to trace

        Returns
        -------
        features : dict
            ``hpol_vpol_band_snr_ratio`` and, per pair, ``pol_fraction_``, ``hv_xcorr_`` and ``hv_dt_``
        """
        features = {}
        if "hpol" in self._channel_groups and "vpol" in self._channel_groups:
            h, v = row["band_snr_avg_hpol"], row["band_snr_avg_vpol"]
            features["hpol_vpol_band_snr_ratio"] = h / v if np.isfinite(h) and np.isfinite(v) and v != 0 else np.nan

        for pair, channels in self._config["hv_pairs"].items():
            h_snr = per_channel[channels["hpol"]]["band_snr"] if channels["hpol"] in per_channel else np.nan
            v_snrs = [per_channel[ch]["band_snr"] for ch in channels["vpol"] if ch in per_channel]
            v_snrs = [snr for snr in v_snrs if np.isfinite(snr)]
            v_snr = float(np.mean(v_snrs)) if v_snrs else np.nan
            total = v_snr + h_snr
            features[f"pol_fraction_{pair}"] = float(v_snr / total) if np.isfinite(total) and total > 0 else np.nan

            v_partner = next((ch for ch in channels["vpol"] if ch in traces), None)
            if channels["hpol"] in traces and v_partner is not None:
                corr, lag = trace_utilities.get_normalized_cross_correlation(
                    traces[channels["hpol"]], traces[v_partner])
                features[f"hv_xcorr_{pair}"] = corr
                features[f"hv_dt_{pair}"] = float(lag)
            else:
                features[f"hv_xcorr_{pair}"] = np.nan
                features[f"hv_dt_{pair}"] = np.nan
        return features
