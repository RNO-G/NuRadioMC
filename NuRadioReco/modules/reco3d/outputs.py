"""Polarization, pair signs, validation and above-surface outputs of the 3D reconstruction."""

import numpy as np
import itertools
import numbers

from NuRadioReco.utilities.reco3d_kernels import USE_NUMBA

from NuRadioReco.modules.reco3d.shared import logger

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import _bilinear_scalar_numba


class OutputsMixin:
    """Methods of InterferometricReco3D for polarization, pair signs, validation and above-surface outputs."""

    # Channels sharing a string; pairs within one string keep their sign under
    # every hpol_sign_mode. Channels not listed form a string of their own.
    STRING_GROUPS = {
        'power': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'helper_b': [9, 10, 11],
        'helper_c': [21, 22, 23],
    }
    HPOL_SIGN_MODES = ('signed', 'abs_cross_string', 'joint_sign')
    HPOL_SIGN_GROUP = 'hpol'

    @classmethod
    def _validate_sign_config(cls, config):
        """Check ``hpol_sign_mode`` and ``pair_signs``.

        Raises:
            ValueError: If the mode is unknown, a mode other than ``signed`` is
                set without a ``polarization_groups`` entry named ``hpol``, or
                ``pair_signs`` is not a list of 1, -1 and ``'abs'`` entries, is
                set together with ``polarization_groups`` or does not hold one
                entry per pair of ``channels``.
        """
        mode = config.get('hpol_sign_mode', 'signed')
        if mode not in cls.HPOL_SIGN_MODES:
            raise ValueError(
                f"hpol_sign_mode must be one of {cls.HPOL_SIGN_MODES}, got {mode!r}")
        groups = config.get('polarization_groups', None) or {}
        if mode != 'signed' and cls.HPOL_SIGN_GROUP not in groups:
            raise ValueError(
                f"hpol_sign_mode {mode!r} needs polarization_groups with a "
                f"{cls.HPOL_SIGN_GROUP!r} group")
        signs = config.get('pair_signs', None)
        if signs is not None and not (isinstance(signs, (list, tuple)) and all(
                s == 'abs' or (isinstance(s, numbers.Real) and s in (1, -1)) for s in signs)):
            raise ValueError(
                f"pair_signs must be a list of 1, -1 or 'abs' per pair, got {signs!r}")
        if signs is not None and groups:
            raise ValueError(
                "pair_signs applies to a single channel group; with polarization_groups "
                "use hpol_sign_mode")
        channels = config.get('channels', None)
        if signs is not None and channels is not None:
            n_pairs = len(channels) * (len(channels) - 1) // 2
            if len(signs) != n_pairs:
                raise ValueError(
                    f"pair_signs needs one entry per channel pair ({n_pairs} for "
                    f"{len(channels)} channels), got {len(signs)}")

    @classmethod
    def _string_of(cls, ch):
        """Name of the string holding channel ``ch``, or the channel id when unlisted."""
        for name, chs in cls.STRING_GROUPS.items():
            if ch in chs:
                return name
        return ch

    @classmethod
    def _cross_string_abs_signs(cls, channels):
        """Per-pair signs for ``abs_cross_string``: ``'abs'`` across strings, 1 within one."""
        return ['abs' if cls._string_of(a) != cls._string_of(b) else 1
                for a, b in itertools.combinations(channels, 2)]

    @classmethod
    def _sign_assignments(cls, channels):
        """Relative sign assignments of the strings among ``channels`` for ``joint_sign``.

        The first string in channel order keeps sign +1 and every other string
        takes +1 or -1, enumerated as ``itertools.product((1, -1))`` over those
        strings, so n strings give 2^(n - 1) assignments and assignment 0 is
        the signed objective.

        Args:
            channels: Channel IDs of the group.

        Returns:
            (strings, assignments): the string names in channel order and one
            per-pair list of +1 or -1 in ``combinations`` order per assignment.
        """
        strings = list(dict.fromkeys(cls._string_of(ch) for ch in channels))
        pairs = list(itertools.combinations(channels, 2))
        assignments = []
        for flips in itertools.product((1, -1), repeat=len(strings) - 1):
            sign = dict(zip(strings, (1,) + flips))
            assignments.append([sign[cls._string_of(a)] * sign[cls._string_of(b)]
                                for a, b in pairs])
        return strings, assignments

    def _compute_travel_times_single_point(self, rho, phi_deg, z, channels):
        """Compute per-channel travel times for a single source position.

        Args:
            rho: Horizontal distance from PA center (meters).
            phi_deg: Azimuth in degrees.
            z: Depth (meters, negative below surface).
            channels: List of channel IDs.

        Returns:
            Dict mapping channel ID to travel time (ns); NaN where the table holds
            no ray solution, minus infinity for a position outside the table.
        """
        phi_rad = phi_deg * (np.pi / 180.0)
        x = rho * np.cos(phi_rad) + self._pa_center[0]
        y = rho * np.sin(phi_rad) + self._pa_center[1]

        travel_times = {}
        for ch in channels:
            pos = self.ant_locs[ch]
            dx = x - pos[0]
            dy = y - pos[1]
            r = max(np.sqrt(dx * dx + dy * dy), 1.0)
            td = self._interpolators[ch]
            if USE_NUMBA:
                tt = _bilinear_scalar_numba(
                    td.values, td.r_min, td.dr_inv, td.nr,
                    td.z_min, td.dz_inv, td.nz, r, z)
            else:
                tt = td.interp(np.array([[r, z]]))[0]
            travel_times[ch] = tt
        return travel_times

    def _compute_coherent_waveform(self, rho, phi_deg, z,
                                   volt_arrays, time_arrays, channels):
        """Form coherent delay-and-stack waveform at a source direction.

        Shifts each channel trace by the relative travel time delay and
        sums. Uses linear interpolation for sub-sample shifting.

        Args:
            rho, phi_deg, z: Source position in cylindrical coordinates.
            volt_arrays: List of voltage traces per channel.
            time_arrays: List of time arrays per channel.
            channels: List of channel IDs.

        Returns:
            (times, coherent_trace) where times is the output time array
            and coherent_trace is the delay-and-stack sum, normalized by
            the number of contributing channels. Returns (None, None) if
            travel times are unavailable.
        """
        travel_times = self._compute_travel_times_single_point(
            rho, phi_deg, z, channels)

        valid_tt = {ch: tt for ch, tt in travel_times.items()
                    if np.isfinite(tt) and tt > 0}
        if len(valid_tt) < 2:
            return None, None

        t_ref = min(valid_tt.values())
        delays = {ch: tt - t_ref for ch, tt in valid_tt.items()}

        ref_idx = channels.index(list(valid_tt.keys())[0])
        out_times = time_arrays[ref_idx].copy()
        n_out = len(out_times)
        coherent = np.zeros(n_out, dtype=np.float64)
        n_contributing = 0

        for ci, ch in enumerate(channels):
            if ch not in valid_tt:
                continue
            trace = volt_arrays[ci]
            times = time_arrays[ci]
            delay = delays[ch]

            shifted_times = out_times + delay
            shifted_trace = np.interp(shifted_times, times, trace,
                                      left=0.0, right=0.0)
            coherent += shifted_trace
            n_contributing += 1

        if n_contributing > 0:
            coherent /= n_contributing

        return out_times, coherent

    @staticmethod
    def _chain_blocks(chain, key):
        """(air, list) per z block of a search chain for one of its result lists.

        ``air`` is True for the air block of the split z grid, False for its in-ice
        block and None without the split grid.
        """
        if 'blocks' in chain:
            return [(st['z_above'] is not None, st.get(key, [])) for st in chain['blocks']]
        return [(None, chain.get(key, []))]

    def _region_hypotheses(self, entries, channels, pair_weights, series, snr_map, coarse_axes):
        """Best position of the search below and above the ice surface, with its correlations.

        The region of a position is its z block under the split z grid (in-ice block
        below, air block above, so a position at z = 0 belongs to the block that
        produced it) and the sign of z otherwise (z > 0 above). The best position of a
        region is the entry with the highest search value (the value the search ranks
        by: the polished raw correlation in candidate mode, the chain's objective in the
        default search). At it the raw and both envelope objectives are evaluated and
        the map SNR is read on the coarse map of the saved peaks' map SNR.

        Args:
            entries: (air, entry) pairs for every optimizer or polish output and every
                graded refined peak of the search, entry being (rho, phi_deg, z, value,
                origin, ...) and air as in ``_chain_blocks``.
            channels: Channel IDs of the group.
            pair_weights: Per-pair weights of the search.
            series: Series accessor of ``_group_inputs``.
            snr_map: Coarse map the saved peaks' map SNR is read on.
            coarse_axes: (rho_vec, phi_vec_deg, z_vec) of the coarse grid.

        Returns:
            Dict with, for region in (below, above), ``{region}_rho_v1``, ``_phi_v1``,
            ``_z_v1``, ``_corr_raw_v1``, ``_corr_env_traces_v1``,
            ``_corr_env_correlation_v1``, ``_map_snr_v1`` (NaN without an entry in
            the region) and ``_origin_v1`` (chain code, -1 without an entry).
        """
        best = {}
        for air, entry in entries:
            region = 'above' if (entry[2] > 0 if air is None else air) else 'below'
            if region not in best or entry[3] > best[region][3]:
                best[region] = entry
        caches = {}
        out = {}
        for region in ('below', 'above'):
            entry = best.get(region)
            if entry is None:
                for key in ('rho', 'phi', 'z', 'corr_raw', 'corr_env_traces',
                            'corr_env_correlation', 'map_snr'):
                    out[f'{region}_{key}_v1'] = np.nan
                out[f'{region}_origin_v1'] = -1
                continue
            rho, phi, z = float(entry[0]), float(entry[1]) % 360.0, float(entry[2])
            out[f'{region}_rho_v1'] = rho
            out[f'{region}_phi_v1'] = phi
            out[f'{region}_z_v1'] = z
            for mode, name in ((None, 'raw'), ('traces', 'env_traces'),
                               ('correlation', 'env_correlation')):
                corr_data, packed = series(mode)
                if mode not in caches:
                    caches[mode] = self._build_optimizer_cache(
                        channels, pair_weights, corr_data, packed=packed)
                out[f'{region}_corr_{name}_v1'] = -self._correlation_at_point(
                    [rho, phi, z], corr_data, channels, pair_weights, _cache=caches[mode])
            out[f'{region}_map_snr_v1'] = self._compute_map_snr(
                snr_map, self._find_peak_bin(rho, phi, z, *coarse_axes))
            out[f'{region}_origin_v1'] = int(entry[4])
        return out

    def _snr_summaries(self, channel_snrs, threshold, suffix=''):
        """Phased-array and helper SNR summaries and the counts above ``threshold``.

        Args:
            channel_snrs: Dict mapping channel id to SNR.
            threshold: SNR above which a channel counts as lit.
            suffix: Appended to every key (``_w15`` for the windowed SNR).

        Returns:
            Dict with ``pa_avg_snr``, ``pa_max_snr``, the helper-string maxima and
            minima, ``n_helpers_above``, ``n_channels_above`` and
            ``has_helper_signal``, each key carrying the suffix.
        """
        pa = self.DEFAULT_DEPTH_GROUPS['pa']
        hb = self.DEFAULT_DEPTH_GROUPS['helper_b']
        hc = self.DEFAULT_DEPTH_GROUPS['helper_c']
        helpers = hb + hc
        all_chs = pa + hb + hc

        def _safe_stat(chs, func):
            """Apply ``func`` to the SNRs of ``chs``, a missing channel counting as 0.

            Returns:
                The statistic as a float, 0.0 for an empty channel list.
            """
            vals = [channel_snrs.get(ch, 0.0) for ch in chs]
            return float(func(vals)) if vals else 0.0

        result = {}
        result[f'pa_avg_snr{suffix}'] = _safe_stat(pa, np.mean)
        result[f'pa_max_snr{suffix}'] = _safe_stat(pa, np.max)
        result[f'helper_b_max_snr{suffix}'] = _safe_stat(hb, np.max)
        result[f'helper_b_min_snr{suffix}'] = _safe_stat(hb, np.min)
        result[f'helper_c_max_snr{suffix}'] = _safe_stat(hc, np.max)
        result[f'helper_c_min_snr{suffix}'] = _safe_stat(hc, np.min)
        helper_snrs = [channel_snrs.get(ch, 0.0) for ch in helpers]
        all_snrs = [channel_snrs.get(ch, 0.0) for ch in all_chs]
        result[f'n_helpers_above{suffix}'] = int(sum(1 for s in helper_snrs if s > threshold))
        result[f'n_channels_above{suffix}'] = int(sum(1 for s in all_snrs if s > threshold))
        result[f'has_helper_signal{suffix}'] = result[f'n_helpers_above{suffix}'] > 0
        return result

    def _compute_validation_metrics(self, mean_corr, rho_vec, phi_vec, z_vec,
                                    channel_snrs, coarse_peaks, config,
                                    windowed_snrs=None):
        """Compute per-channel SNR summaries, surface correlation, and peak isolation.

        The record SNR columns (``ch{N}_snr`` and the summaries gated by
        ``helper_snr_threshold``) are always written. With ``windowed_snrs`` the
        same columns are added under the suffix of ``snr_window_ns``
        (``ch{N}_snr_w15`` for 1.5 ns), gated by ``helper_snr_threshold_windowed``.

        Args:
            mean_corr: 3D coarse correlation array.
            rho_vec: Coarse rho grid (meters).
            phi_vec: Coarse phi grid (radians).
            z_vec: Coarse z grid (meters).
            channel_snrs: Dict mapping channel_id to the record SNR.
            coarse_peaks: List of (rho, phi, z, corr) tuples.
            config: Reco config dict.
            windowed_snrs: Dict mapping channel_id to the time-window SNR, or None.

        Returns:
            Dict of validation metrics.
        """
        result = {}

        for ch, snr in channel_snrs.items():
            result[f'ch{ch}_snr'] = snr
        result.update(self._snr_summaries(
            channel_snrs, config.get('helper_snr_threshold', 5.0)))
        if windowed_snrs:
            suffix = '_' + self._snr_suffix(config['snr_window_ns'])
            for ch, snr in windowed_snrs.items():
                result[f'ch{ch}_snr{suffix}'] = snr
            result.update(self._snr_summaries(
                windowed_snrs,
                config.get('helper_snr_threshold_windowed',
                           self.HELPER_SNR_THRESHOLD_WINDOWED),
                suffix))

        # Surface correlation
        pa_center_z = self._pa_center[2]
        surf_z_max = config.get('surf_corr_z_max', -10.0)
        z_mask = z_vec >= surf_z_max
        if not isinstance(mean_corr, np.ndarray):
            result['surf_corr_z'] = mean_corr.masked_max(z_mask=z_mask) if z_mask.any() else np.nan
        else:
            result['surf_corr_z'] = (float(np.nanmax(mean_corr[:, :, z_mask]))
                                     if z_mask.any() else np.nan)
        surf_zen_max = config.get('surf_corr_zen_max', 65.0)
        mask_key = (rho_vec.tobytes(), len(phi_vec), z_vec.tobytes(),
                    float(surf_zen_max), float(pa_center_z))
        if not hasattr(self, '_zen_mask_cache'):
            self._zen_mask_cache = {}
        zen_mask = self._zen_mask_cache.get(mask_key)
        if zen_mask is None:
            rho_g, _, z_g = np.meshgrid(rho_vec, phi_vec, z_vec, indexing='ij')
            zen_grid = np.degrees(np.arctan2(rho_g, z_g - pa_center_z))
            zen_mask = zen_grid <= surf_zen_max
            self._zen_mask_cache[mask_key] = zen_mask
        if not isinstance(mean_corr, np.ndarray):
            result['surf_corr_zen'] = mean_corr.masked_max(mask=zen_mask) if zen_mask.any() else np.nan
        else:
            result['surf_corr_zen'] = (float(np.nanmax(mean_corr[zen_mask]))
                                       if zen_mask.any() else np.nan)

        # Peak isolation
        if len(coarse_peaks) >= 2:
            corrs = sorted([p[3] for p in coarse_peaks], reverse=True)
            top_mean = np.mean(corrs[:5])
            result['peak_isolation_ratio'] = (
                float(corrs[0] / top_mean) if top_mean > 0 else np.nan)
        else:
            result['peak_isolation_ratio'] = np.nan

        return result

    def _run_per_polarization(self, config, run_group, station=None):
        """Run independent reconstruction per polarization group.

        Each polarization gets its own correlation map, peak finding, and
        optimizer. No cross-pol pairs are ever formed. The first group
        (typically VPOL) provides the primary result; subsequent groups
        add supplementary fields with a group-name suffix.

        Parameters
        ----------
        config : dict
            Must contain 'polarization_groups' mapping group names to
            channel lists.
        run_group : callable
            Reconstructs one group: takes the group's config (channels of the
            group, no 'polarization_groups') and returns its result dict.
        station : Station or None
            Receives the result parameters of a joint_sign HPol group.

        Returns
        -------
        dict
            Primary result from first group, plus per-group results
            keyed as '{field}_{group_name}'.
        """
        pol_groups = config['polarization_groups']
        all_channels = config['channels']
        results = {}

        # Choose the primary group deterministically. Not "first in yaml
        # order", which silently changes when configs round-trip through
        # yaml.safe_dump (alphabetical sort). Rules:
        #   1. If config.primary_polarization is set, use that group.
        #   2. Otherwise, pick the group with the most active channels,
        #      which is the most-constrained reco. Ties broken
        #      alphabetically for reproducibility.
        active_groups = [
            (name, [ch for ch in all_channels if ch in chs])
            for name, chs in pol_groups.items()
        ]
        active_groups = [(n, chs) for n, chs in active_groups if len(chs) >= 2]
        if not active_groups:
            logger.warning("No polarization group had >= 2 channels")
            return {'rho': np.nan, 'phi': np.nan, 'z': np.nan,
                    'max_corr': np.nan}

        primary_pol_override = config.get('primary_polarization', None)
        if primary_pol_override is not None:
            if primary_pol_override not in dict(active_groups):
                raise ValueError(
                    f"primary_polarization='{primary_pol_override}' not in "
                    f"active polarization groups "
                    f"{[n for n, _ in active_groups]}")
            primary_name = primary_pol_override
        else:
            # Largest group wins; alphabetical tiebreak.
            primary_name = sorted(
                active_groups,
                key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
        logger.info("Primary polarization group: %s (channels: %s)",
                    primary_name,
                    dict(active_groups).get(primary_name))

        # Run primary first so its result lands at the top level, then
        # the remaining groups in their original yaml order.
        group_items = ([(n, chs) for n, chs in active_groups if n == primary_name]
                       + [(n, chs) for n, chs in active_groups if n != primary_name])

        sign_mode = config.get('hpol_sign_mode', 'signed')
        for group_name, active in group_items:
            group_config = dict(config)
            group_config['channels'] = active
            group_config.pop('polarization_groups', None)
            group_config.pop('hpol_weight_scale', None)
            group_config.pop('hpol_sign_mode', None)

            logger.info("Running %s reco: %d channels %s",
                        group_name, len(active), active)

            if group_name == self.HPOL_SIGN_GROUP and sign_mode != 'signed':
                grp_result = self._run_hpol_signed(group_config, sign_mode, run_group, station)
            else:
                grp_result = run_group(group_config)

            for key, val in grp_result.items():
                results[f'{key}_{group_name}'] = val

            if group_name == primary_name:
                results.update(grp_result)

        return results

    def _run_hpol_signed(self, config, mode, run_group, station=None):
        """Run the HPol group under a polarity-aware sign mode.

        The horizontal Askaryan field flips sign across the vertical plane
        through the shower axis, so strings on opposite sides of it receive
        HPol pulses of opposite polarity and a signed correlation scores the
        true delay of such a cross-string pair at minus its peak.
        ``abs_cross_string`` scores every cross-string pair by the absolute
        value of its raw correlation and keeps same-string pairs signed.
        ``joint_sign`` runs the group once per relative sign assignment of its
        strings (``_sign_assignments``) and keeps the run with the largest
        correlation, which is the maximum of the signed objective over the
        assignments; ``sign_assignment`` is the winning index and
        ``sign_corr_{k}`` every assignment's correlation. ``sign_mode`` (1
        abs_cross_string, 2 joint_sign) marks the result and the station
        parameters hold the kept run.

        Args:
            config: Group config dict (channels of the HPol group).
            mode: ``abs_cross_string`` or ``joint_sign``.
            run_group: Reconstructs the group from a config (see ``_run_per_polarization``).
            station: Station that receives the parameters of the kept run, or None.

        Returns:
            Result dict of the group with the sign fields added.
        """
        channels = config['channels']
        if mode == 'abs_cross_string':
            result = dict(run_group(dict(config, pair_signs=self._cross_string_abs_signs(channels))))
        else:
            _, assignments = self._sign_assignments(channels)
            runs = [run_group(dict(config, pair_signs=signs)) for signs in assignments]
            corrs = [r.get('max_corr', np.nan) for r in runs]
            best = int(np.nanargmax(corrs)) if np.any(np.isfinite(corrs)) else 0
            result = dict(runs[best])
            result['sign_assignment'] = best
            for k, corr in enumerate(corrs):
                result[f'sign_corr_{k}'] = corr
            self._set_station_parameters(
                station, result['rho'], result['phi'], result['z'], result['max_corr'])
        result['sign_mode'] = self.HPOL_SIGN_MODES.index(mode)
        return result
