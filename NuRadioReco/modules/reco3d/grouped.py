"""Grouped ray types of the 3D reconstruction."""

import numpy as np
import itertools

from NuRadioReco.modules.reco3d.shared import logger


# Bound to the class by the module that defines it, once the class exists.
InterferometricReco3D = None


class GroupedMixin:
    """Methods of InterferometricReco3D for grouped ray types."""

    # Depth groups for grouped combo selection. Channels in the same group
    # are assumed to see the same ray type. Channels not listed get their
    # own individual group.
    DEFAULT_DEPTH_GROUPS = {
        'pa': [0, 1, 2, 3],       # ~94-97m, 1m spacing
        'pa_hpol': [4, 8],        # ~92-93m HPOL on power string
        'helper_b': [9, 10],       # ~96-97m on string B
        'helper_b_hpol': [11],    # ~95m HPOL on string B
        'helper_c': [22, 23],      # ~96-97m on string C
        'helper_c_hpol': [21],    # ~95m HPOL on string C
    }

    @staticmethod
    def _build_channel_groups(channels, depth_groups=None):
        """Assign each channel to a depth group.

        Channels in the same group share a ray-type assignment.
        Ungrouped channels get their own individual group.

        Parameters
        ----------
        channels : list
            Channel IDs used in reconstruction.
        depth_groups : dict or None
            Maps group name to list of channel IDs. Defaults to
            DEFAULT_DEPTH_GROUPS.

        Returns
        -------
        dict
            Maps channel ID to group index.
        list
            List of group names (for logging).
        """
        if depth_groups is None:
            depth_groups = InterferometricReco3D.DEFAULT_DEPTH_GROUPS

        ch_to_group = {}
        group_names = []
        gidx = 0
        for name, members in depth_groups.items():
            active = [ch for ch in members if ch in channels]
            if active:
                for ch in active:
                    ch_to_group[ch] = gidx
                group_names.append(name)
                gidx += 1
        for ch in channels:
            if ch not in ch_to_group:
                ch_to_group[ch] = gidx
                group_names.append(f'ch{ch}')
                gidx += 1
        return ch_to_group, group_names

    def _correlator_grouped_multiray(self, corr_data, tt_all, channels,
                                     pair_weights=None):
        """Grouped combo multiray correlator.

        Instead of taking per-pair max across 9 ray combos, enumerate all
        depth-group ray-type assignments and take the max of the weighted
        mean correlation across assignments. Channels in the same depth
        group share a ray type.

        Parameters
        ----------
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        tt_all : dict
            Maps ch -> {ray_type -> grid}.
        channels : list
            Channel IDs.
        pair_weights : list or None
            Per-pair weights.

        Returns
        -------
        tuple
            (mean_corr_map, max_corr)
        """
        ch_pairs = list(itertools.combinations(channels, 2))
        n_pairs = len(ch_pairs)

        grid_shape = None
        for ch in channels:
            for rt in tt_all.get(ch, {}):
                grid_shape = tt_all[ch][rt].shape
                break
            if grid_shape is not None:
                break
        if grid_shape is None:
            return np.zeros(1), np.nan

        if pair_weights is not None:
            w = np.asarray(pair_weights, dtype=np.float64)
            w_sum = float(w.sum())
        else:
            w = np.ones(n_pairs, dtype=np.float64)
            w_sum = float(n_pairs)

        ch_to_group, group_names = self._build_channel_groups(channels)
        n_groups = len(group_names)

        # Available ray types per group. Start with intersection; fall back
        # to union if the intersection is empty (e.g. shallow channels that
        # lack some ray types at certain source depths).
        group_ray_types = []
        for gidx in range(n_groups):
            group_chs = [ch for ch in channels if ch_to_group[ch] == gidx]
            rts = set(self._active_ray_types)
            for ch in group_chs:
                rts &= set(tt_all.get(ch, {}).keys())
            if not rts:
                for ch in group_chs:
                    rts |= set(tt_all.get(ch, {}).keys())
            if not rts:
                rts = {self._active_ray_types[0]}
            group_ray_types.append(sorted(rts))

        combos = list(itertools.product(*group_ray_types))
        logger.debug("Grouped multiray: %d groups (%s), %d combos",
                      n_groups, group_names, len(combos))

        best_mean_corr = np.full(grid_shape, -np.inf, dtype=np.float64)
        combo_corr = np.zeros(grid_shape, dtype=np.float64)

        for combo in combos:
            ch_rt = {ch: combo[ch_to_group[ch]] for ch in channels}

            combo_corr[:] = 0.0
            for pidx, (c1, c2) in enumerate(ch_pairs):
                rt1 = ch_rt[c1]
                rt2 = ch_rt[c2]

                tt1 = tt_all.get(c1, {}).get(rt1)
                tt2 = tt_all.get(c2, {}).get(rt2)
                if tt1 is None or tt2 is None:
                    continue

                delay = tt1 - tt2
                valid = np.isfinite(delay)
                if not np.any(valid):
                    continue

                flat_delays = delay[valid].ravel().astype(np.float64)
                vals = self._interp_delays(corr_data[pidx][0],
                                           corr_data[pidx][1],
                                           corr_data[pidx][2],
                                           flat_delays)
                np.nan_to_num(vals, copy=False, nan=0.0)
                combo_corr[valid] += vals * w[pidx]

            if w_sum > 0:
                combo_corr /= w_sum

            np.maximum(best_mean_corr, combo_corr, out=best_mean_corr)

        neg_inf = best_mean_corr == -np.inf
        best_mean_corr[neg_inf] = 0.0

        max_corr = float(np.max(best_mean_corr)) if best_mean_corr.size > 0 else np.nan
        return best_mean_corr, max_corr

    def _correlation_at_point_grouped(self, ch_tt, corr_data, channels,
                                      ch_pairs, pair_weights, w_total,
                                      _cache=None):
        """Grouped-combo scalar correlation for L-BFGS-B optimizer.

        Enumerates depth-group ray-type assignments and returns the negative
        of the best weighted mean correlation.

        Parameters
        ----------
        ch_tt : dict
            Maps ch -> {ray_type: travel_time} at this point.
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        channels : list
            Channel IDs.
        ch_pairs : list of tuple
            Channel pairs.
        pair_weights : array-like
            Per-pair weights.
        w_total : float
            Sum of pair weights.
        _cache : dict or None
            Pre-computed combo_rt from _build_optimizer_cache.

        Returns
        -------
        float
            Negative mean correlation (for minimization).
        """
        if _cache is not None and 'combo_rt' in _cache:
            combo_rt_list = _cache['combo_rt']
        else:
            ch_to_group, _ = self._build_channel_groups(channels)
            n_groups = max(ch_to_group.values()) + 1
            group_ray_types = []
            for gidx in range(n_groups):
                group_chs = [ch for ch in channels
                             if ch_to_group[ch] == gidx]
                rts = set(self._active_ray_types)
                for ch in group_chs:
                    rts &= set(ch_tt.get(ch, {}).keys())
                if not rts:
                    for ch in group_chs:
                        rts |= set(ch_tt.get(ch, {}).keys())
                if not rts:
                    rts = {self._active_ray_types[0]}
                group_ray_types.append(sorted(rts))
            combos = list(itertools.product(*group_ray_types))
            combo_rt_list = []
            for combo in combos:
                combo_rt_list.append(
                    [combo[ch_to_group[ch]] for ch in channels])

        best_total = -np.inf
        ch_idx = {ch: i for i, ch in enumerate(channels)}
        for combo_rt in combo_rt_list:
            total = 0.0
            for pidx, (c1, c2) in enumerate(ch_pairs):
                rt1 = combo_rt[ch_idx[c1]]
                rt2 = combo_rt[ch_idx[c2]]
                tt1 = ch_tt.get(c1, {}).get(rt1)
                tt2 = ch_tt.get(c2, {}).get(rt2)
                if tt1 is None or tt2 is None:
                    continue
                delay = tt1 - tt2
                corr_arr, dt, offset = corr_data[pidx]
                val = self._interp_corr_scalar(corr_arr, dt, offset, delay)
                w = pair_weights[pidx] if not isinstance(pair_weights, (int, float)) else pair_weights
                total += w * val
            if total > best_total:
                best_total = total

        if best_total == -np.inf:
            return 0.0
        return -best_total / w_total if w_total > 0 else 0.0
