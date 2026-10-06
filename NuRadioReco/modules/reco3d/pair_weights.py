"""Pair weights and channel SNR of the 3D reconstruction."""

import numpy as np
import itertools
import numbers


# Bound to the class by the module that defines it, once the class exists.
InterferometricReco3D = None


class PairWeightsMixin:
    """Methods of InterferometricReco3D for pair weights and channel SNR."""

    RECORD_SNR_WINDOW_SAMPLES = 3
    HELPER_SNR_THRESHOLD_WINDOWED = 19.5
    PAIR_WEIGHT_MODES = ('record', 'information')
    PAIR_WEIGHT_K_NS = 12.65
    PAIR_WEIGHT_FLOOR_NS = 2.0

    @classmethod
    def _validate_snr_config(cls, config):
        """Check the SNR window, helper threshold and pair weight keys.

        Raises:
            ValueError: If ``snr_window_ns`` is not null or a positive number,
                ``helper_snr_threshold_windowed`` is not a number,
                ``pair_weight_mode`` is unknown, ``information`` is asked for
                without ``snr_window_ns`` or without ``snr_pair_weighting``,
                ``pair_weight_k_ns`` is not positive or ``pair_weight_floor_ns``
                is negative.
        """
        window = config.get('snr_window_ns', None)
        if window is not None and not (isinstance(window, numbers.Real) and window > 0):
            raise ValueError(
                f"snr_window_ns must be null or a positive number of ns, got {window!r}")
        threshold = config.get('helper_snr_threshold_windowed', cls.HELPER_SNR_THRESHOLD_WINDOWED)
        if not isinstance(threshold, numbers.Real):
            raise ValueError(
                f"helper_snr_threshold_windowed must be a number, got {threshold!r}")
        mode = config.get('pair_weight_mode', 'record')
        if mode not in cls.PAIR_WEIGHT_MODES:
            raise ValueError(
                f"pair_weight_mode must be one of {cls.PAIR_WEIGHT_MODES}, got {mode!r}")
        if mode == 'information' and window is None:
            raise ValueError("pair_weight_mode 'information' needs snr_window_ns")
        if mode == 'information' and not config.get('snr_pair_weighting', False):
            raise ValueError("pair_weight_mode 'information' needs snr_pair_weighting: true")
        k_ns = config.get('pair_weight_k_ns', cls.PAIR_WEIGHT_K_NS)
        if not (isinstance(k_ns, numbers.Real) and k_ns > 0):
            raise ValueError(f"pair_weight_k_ns must be a positive number, got {k_ns!r}")
        floor_ns = config.get('pair_weight_floor_ns', cls.PAIR_WEIGHT_FLOOR_NS)
        if not (isinstance(floor_ns, numbers.Real) and floor_ns >= 0):
            raise ValueError(
                f"pair_weight_floor_ns must be a non-negative number, got {floor_ns!r}")
        cls._validate_sign_config(config)

    @staticmethod
    def _compute_snr_pair_weights(volt_arrays, channels):
        """Compute the record per-pair weights from the 3-sample channel SNRs.

        Each pair is weighted by the geometric mean of the two channels' SNRs,
        normalized so the maximum is 1. All pairs contribute; low-SNR channels
        are naturally downweighted.

        Args:
            volt_arrays: Voltage traces, one per channel (same order as channels).
            channels: Channel IDs.

        Returns:
            (weights, channel_snrs): per-pair weights in ``combinations`` order and
            the SNR of every channel keyed by channel id.
        """
        snrs = InterferometricReco3D._channel_snrs(
            volt_arrays, InterferometricReco3D.RECORD_SNR_WINDOW_SAMPLES)
        return InterferometricReco3D._record_pair_weights(snrs), dict(zip(channels, snrs))

    @staticmethod
    def _record_pair_weights(snrs):
        """Geometric-mean SNR pair weights in ``combinations`` order, normalised to a maximum of 1."""
        weights = [np.sqrt(snrs[i] * snrs[j])
                   for i, j in itertools.combinations(range(len(snrs)), 2)]
        max_w = max(weights) if weights else 1.0
        if max_w > 0:
            weights = [w / max_w for w in weights]
        return weights

    @staticmethod
    def _channel_snrs(volt_arrays, window_size):
        """Peak-to-peak SNR of every trace over a window of ``window_size`` samples.

        The largest peak-to-peak amplitude found within the window is divided by
        twice the split-trace noise RMS; a trace with zero noise RMS reads 0.

        Args:
            volt_arrays: Voltage traces.
            window_size: Window length in samples (at least 2).

        Returns:
            List of SNR values in the order of ``volt_arrays``.
        """
        from NuRadioReco.utilities.trace_utilities import (
            get_split_trace_noise_RMS, get_signal_to_noise_ratio)

        snrs = []
        for v in volt_arrays:
            noise_rms = get_split_trace_noise_RMS(v)
            snrs.append(get_signal_to_noise_ratio(v, noise_rms, window_size)
                        if noise_rms > 0 else 0.0)
        return snrs

    @staticmethod
    def _information_pair_weights(snrs, k_ns, floor_ns):
        """Inverse timing-variance pair weights, normalised to a maximum of 1.

        The timing variance of a pair is ``floor^2 + (k / SNR_i)^2 + (k / SNR_j)^2``:
        a bandwidth-limited timing error ``k / SNR`` per channel plus a floor
        for the pulse-shape mismatch that does not fall with SNR, so a pair with
        a noise channel keeps a small weight without a hard gate and the loudest
        channels saturate at ``1 / floor^2``. A channel with SNR 0 zeroes its
        pairs.

        Args:
            snrs: Per-channel SNR values in channel order.
            k_ns: Timing error at unit SNR in ns.
            floor_ns: Timing error floor in ns.

        Returns:
            List of per-pair weights in ``combinations`` order.
        """
        var = [(k_ns / s) ** 2 if s > 0 else np.inf for s in snrs]
        weights = []
        for i, j in itertools.combinations(range(len(snrs)), 2):
            total = floor_ns ** 2 + var[i] + var[j]
            weights.append(0.0 if not np.isfinite(total) or total <= 0 else 1.0 / total)
        max_w = max(weights) if weights else 1.0
        if max_w > 0:
            weights = [w / max_w for w in weights]
        return weights

    @staticmethod
    def _snr_window_samples(window_ns, dt):
        """Number of samples spanning ``window_ns`` at sample spacing ``dt`` ns, at least 2."""
        return max(2, int(round(window_ns / dt)))

    @staticmethod
    def _snr_suffix(window_ns):
        """Column suffix of the SNR measured over ``window_ns``: ``w15`` for 1.5 ns."""
        return 'w' + f'{window_ns:g}'.replace('.', '')

    def _event_snrs(self, volt_arrays, time_arrays, channels, config, record=False):
        """Per-channel SNRs of one event.

        The record SNR (3-sample window on the traces as given) is measured when
        ``snr_pair_weighting`` or ``validation`` is on, or with ``record``. With
        ``snr_window_ns`` set the SNR is also measured over that time window
        (``round(window / dt)`` samples per channel, the split-trace noise RMS
        unchanged).

        Args:
            volt_arrays: Voltage traces, one per channel.
            time_arrays: Time arrays, one per channel.
            channels: Channel IDs in the order of the traces.
            config: Reconstruction config dict.
            record: Measure the record SNR whatever the config says.

        Returns:
            (snr, windowed): the record SNR per channel id (empty when not
            measured) and the windowed SNR per channel id (empty without
            ``snr_window_ns``).
        """
        snr = {}
        if record or config.get('snr_pair_weighting', False) or config.get('validation', False):
            snr = dict(zip(channels, self._channel_snrs(volt_arrays, self.RECORD_SNR_WINDOW_SAMPLES)))
        windowed = {}
        window_ns = config.get('snr_window_ns', None)
        if window_ns is not None:
            for ch, v, t in zip(channels, volt_arrays, time_arrays):
                dt = t[1] - t[0] if len(t) > 1 else 1.0
                windowed[ch] = self._channel_snrs([v], self._snr_window_samples(window_ns, dt))[0]
        return snr, windowed

    def _group_pair_weights(self, snr, windowed, channels, config):
        """Per-pair weights of one channel group from its channel SNRs.

        With ``snr_pair_weighting`` and ``pair_weight_mode: record`` the
        geometric-mean weights of the record SNR; with ``pair_weight_mode:
        information`` the inverse timing-variance weights of the windowed SNR with
        ``pair_weight_k_ns`` and ``pair_weight_floor_ns``.

        Args:
            snr: Record SNR per channel id.
            windowed: Windowed SNR per channel id.
            channels: Channel IDs of the group, in pair order.
            config: Reconstruction config dict.

        Returns:
            Weights in ``combinations`` order, or None without ``snr_pair_weighting``.
        """
        if not config.get('snr_pair_weighting', False):
            return None
        if config.get('pair_weight_mode', 'record') == 'information':
            return self._information_pair_weights(
                [windowed[ch] for ch in channels],
                config.get('pair_weight_k_ns', self.PAIR_WEIGHT_K_NS),
                config.get('pair_weight_floor_ns', self.PAIR_WEIGHT_FLOOR_NS))
        return self._record_pair_weights([snr[ch] for ch in channels])
