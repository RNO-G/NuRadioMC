"""Several ray types per pair and the two-arrival objective of the 3D reconstruction."""

import numpy as np
import itertools

from NuRadioReco.utilities.reco3d_kernels import (
    USE_NUMBA,
    USE_CUPY,
    USE_NUMBA_GROUPED,
    RAY_TYPES,
    SOLUTION_TYPES,
)

from NuRadioReco.modules.reco3d.shared import _C_M_PER_NS

try:
    from NuRadioReco.utilities.reco3d_kernels import _FUSED_MULTIRAY_CORR_KERNEL
except ImportError:
    _FUSED_MULTIRAY_CORR_KERNEL = None

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import (
        _bilinear_scalar_numba,
        _fused_multiray_grid_numba,
        _two_arrival_point_numba,
        _fused_two_arrival_grid_numba,
    )

if USE_NUMBA_GROUPED:
    from NuRadioReco.utilities.reco3d_kernels import grouped_multiray_numba, perpair_multiray_numba
    if USE_NUMBA:
        from NuRadioReco.utilities.reco3d_kernels import grouped_multiray_points


# Bound to the class by the module that defines it, once the class exists.
InterferometricReco3D = None


class MultirayMixin:
    """Methods of InterferometricReco3D for several ray types per pair and the two-arrival objective."""

    def _compute_tt_multiray(self, src_enu, channels):
        """Compute per-channel, per-ray-type travel time grids.

        Returns only the travel times (not delay matrices), keeping memory at
        O(n_channels * 3 * grid) instead of O(n_pairs * 9 * grid).

        Parameters
        ----------
        src_enu : np.ndarray
            Source ENU matrix, shape (n_rho, n_phi, n_z, 3).
        channels : list
            Channel IDs.

        Returns
        -------
        dict
            Maps channel_id -> {ray_type_name -> travel_time_grid}.
            Only ray types with valid data are included.
        """
        grid_shape = src_enu.shape[:3]
        coords_per_ch = self._compute_rho_and_coords(src_enu, channels)

        tt_all = {}
        for ch in channels:
            tt_all[ch] = {}
            for rt in self._active_ray_types:
                tt = self._table_lookup_batch(
                    self._multiray_interpolators[ch][rt],
                    coords_per_ch[ch]).reshape(grid_shape)
                if np.any(np.isfinite(tt) & (tt > 0)):
                    tt_all[ch][rt] = tt

        return tt_all

    def _pack_multiray_tables(self, channels):
        """Pack multiray TT tables into arrays for the fused kernel.

        Returns:
            Tuple of (td_values, td_ok, td_r_min, td_dr_inv, td_nr,
                      td_z_min, td_dz_inv, td_nz, ant_xy, n_rt): td_values of
            shape (n_ch * n_rt, nr_max, nz_max) with slot ci * n_rt + ri, td_ok
            its finiteness mask and the other td_* of shape (n_ch, n_rt).
        """
        cached = self._packed_multiray_tables.get(tuple(channels))
        if cached is not None:
            return cached
        n_ch = len(channels)
        rts = self._active_ray_types
        n_rt = len(rts)
        nr_max = 0
        nz_max = 0
        for ch in channels:
            for rt in rts:
                td = self._multiray_interpolators[ch][rt]
                nr_max = max(nr_max, td.nr)
                nz_max = max(nz_max, td.nz)
        td_values = np.full((n_ch, n_rt, nr_max, nz_max), np.nan, dtype=np.float64)
        td_r_min = np.empty((n_ch, n_rt), dtype=np.float64)
        td_dr_inv = np.empty((n_ch, n_rt), dtype=np.float64)
        td_nr = np.empty((n_ch, n_rt), dtype=np.int64)
        td_z_min = np.empty((n_ch, n_rt), dtype=np.float64)
        td_dz_inv = np.empty((n_ch, n_rt), dtype=np.float64)
        td_nz = np.empty((n_ch, n_rt), dtype=np.int64)
        ant_xy = np.empty((n_ch, 2), dtype=np.float64)
        for ci, ch in enumerate(channels):
            ant_xy[ci] = [self.ant_locs[ch][0], self.ant_locs[ch][1]]
            for ri, rt in enumerate(rts):
                td = self._multiray_interpolators[ch][rt]
                td_values[ci, ri, :td.nr, :td.nz] = td.values
                td_r_min[ci, ri] = td.r_min
                td_dr_inv[ci, ri] = td.dr_inv
                td_nr[ci, ri] = td.nr
                td_z_min[ci, ri] = td.z_min
                td_dz_inv[ci, ri] = td.dz_inv
                td_nz[ci, ri] = td.nz
        td_values = td_values.reshape(n_ch * n_rt, nr_max, nz_max)
        packed = (td_values, np.isfinite(td_values), td_r_min, td_dr_inv, td_nr,
                  td_z_min, td_dz_inv, td_nz, ant_xy, n_rt)
        self._packed_multiray_tables[tuple(channels)] = packed
        return packed

    def _fused_multiray_refine(self, peak_grids, corr_data, channels,
                               pair_weights, n_extract, level_sep):
        """Run all refine peaks through the fused multiray kernel (per_pair combo mode).

        Processes each peak's grid in one fused Numba call (inline TT
        lookup + per-pair combo evaluation), avoiding per-channel per-rt
        Numba launch overhead. The grouped combo mode does not come here.
        """
        (td_values, td_ok, td_r_min, td_dr_inv, td_nr,
         td_z_min, td_dz_inv, td_nz, ant_xy, n_rt) = \
            self._pack_multiray_tables(channels)
        n_ch = len(channels)
        ch_pairs = list(itertools.combinations(range(n_ch), 2))
        n_pairs = len(ch_pairs)

        corr_lens = [c[0].shape[0] for c in corr_data]
        M_max = max(corr_lens)
        corr_packed = np.zeros((n_pairs, M_max), dtype=np.float64)
        corr_lengths = np.empty(n_pairs, dtype=np.int64)
        corr_dts = np.empty(n_pairs, dtype=np.float64)
        corr_offsets = np.empty(n_pairs, dtype=np.float64)
        pair_ch1 = np.empty(n_pairs, dtype=np.int64)
        pair_ch2 = np.empty(n_pairs, dtype=np.int64)
        pw = np.ones(n_pairs, dtype=np.float64)
        for pidx, (c1i, c2i) in enumerate(ch_pairs):
            corr_packed[pidx, :corr_lens[pidx]] = corr_data[pidx][0]
            corr_lengths[pidx] = corr_lens[pidx]
            corr_dts[pidx] = corr_data[pidx][1]
            corr_offsets[pidx] = corr_data[pidx][2]
            pair_ch1[pidx] = c1i
            pair_ch2[pidx] = c2i
        if pair_weights is not None:
            pw = np.asarray(pair_weights, dtype=np.float64)
        w_total = float(pw.sum())

        pa_x, pa_y = float(self._pa_center[0]), float(self._pa_center[1])

        level_peaks = []
        for src_enu_r, rho_vec_r, phi_vec_r, z_vec_r in peak_grids:
            corr_flat = _fused_multiray_grid_numba(
                rho_vec_r, phi_vec_r, z_vec_r,
                pa_x, pa_y, ant_xy, n_ch,
                td_values, td_ok, td_r_min, td_dr_inv, td_nr,
                td_z_min, td_dz_inv, td_nz, n_rt,
                corr_packed, corr_lengths, corr_dts, corr_offsets,
                pair_ch1, pair_ch2, pw, w_total, self._tolerant_table_edge)
            local_shape = (len(rho_vec_r), len(phi_vec_r), len(z_vec_r))
            local_corr = corr_flat.reshape(local_shape)
            phi_vec_deg_r = phi_vec_r * (180.0 / np.pi)
            local_peaks = self._extract_top_n_peaks(
                local_corr, rho_vec_r, phi_vec_deg_r, z_vec_r,
                n_extract, level_sep)
            level_peaks.extend(local_peaks)
        return level_peaks

    def _correlator_lean_multiray(self, corr_data, tt_all, channels,
                                  pair_weights=None):
        """Multi-ray-type correlator: take max across ray combinations per pair.

        Computes delays inline from per-channel travel times to avoid
        materializing all 9*n_pairs delay matrices simultaneously. Memory
        stays at O(grid) instead of O(n_pairs * 9 * grid).

        Parameters
        ----------
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        tt_all : dict
            From ``_compute_tt_multiray``. Maps ch -> {ray_type -> grid}.
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

        mean_corr = np.zeros(grid_shape, dtype=np.float64)
        best_corr = np.empty(grid_shape, dtype=np.float64)

        for pidx, (c1, c2) in enumerate(ch_pairs):
            corr_arr, dt, offset = corr_data[pidx]
            tt1_dict = tt_all.get(c1, {})
            tt2_dict = tt_all.get(c2, {})

            if not tt1_dict or not tt2_dict:
                continue

            best_corr[:] = -np.inf

            for rt1, tt1 in tt1_dict.items():
                for rt2, tt2 in tt2_dict.items():
                    delay = tt1 - tt2
                    valid = np.isfinite(delay)
                    if not np.any(valid):
                        continue

                    flat_delays = delay[valid].ravel().astype(np.float64)
                    vals = self._interp_delays(corr_arr, dt, offset,
                                               flat_delays)
                    np.nan_to_num(vals, copy=False, nan=0.0)

                    current = best_corr[valid]
                    np.maximum(current, vals, out=current)
                    best_corr[valid] = current

            no_combo = best_corr == -np.inf
            best_corr[no_combo] = 0.0

            mean_corr += best_corr * w[pidx]

        if w_sum > 0:
            mean_corr /= w_sum

        max_corr = float(np.max(mean_corr)) if mean_corr.size > 0 else np.nan
        return mean_corr, max_corr

    def _multiray_grid(self, rho_vec, phi_vec_rad, z_vec, corr_data, channels, pair_weights,
                       src_enu=None, force_perpair=False):
        """Multi-ray map of ``corr_data`` on a (rho, phi, z) grid: ``_multiray_correlate`` of its travel times.

        With a device backend (``_multiray_on_device``) the backend looks the travel times up on its device
        (``grouped_grid``, ``perpair_grid``); otherwise they are looked up on the CPU
        (``_compute_tt_multiray``).

        Args:
            rho_vec, phi_vec_rad, z_vec: Grid axes (m, rad, m).
            corr_data: Raw correlation functions.
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.
            src_enu: The grid's ``_build_source_enu_matrix`` when already built.
            force_perpair: Per-pair map whatever the combination mode (the coarse grid).

        Returns:
            (mean_corr_map, max_corr) of the grid shape.
        """
        n_points = len(rho_vec) * len(phi_vec_rad) * len(z_vec)
        n_rt = (len(RAY_TYPES) if set(self._active_ray_types) & set(RAY_TYPES) else len(SOLUTION_TYPES)) \
            if self._multiray_on_device() else self._n_ray_slots
        ch_to_group, _ = self._build_channel_groups(channels)
        same = sum(ch_to_group[a] == ch_to_group[b] for a, b in itertools.combinations(channels, 2))
        n_pairs = len(channels) * (len(channels) - 1) // 2
        self.work['map_points'] += n_points
        self.work['tt_lookups'] += n_points * len(channels) * n_rt
        if self._multiray_combo_mode == 'grouped' and not force_perpair:
            self.work['mr_pair_terms'] += n_points * (same * n_rt + (n_pairs - same) * n_rt * n_rt)
        else:
            self.work['mr_pair_terms'] += n_points * n_pairs * n_rt * n_rt
        if self._multiray_on_device():
            names = RAY_TYPES if set(self._active_ray_types) & set(RAY_TYPES) else SOLUTION_TYPES
            slot_tables = [self._multiray_interpolators[ch].get(rt) if rt in self._active_ray_types else None
                           for ch in channels for rt in names]
            ant_xy = [self.ant_locs[ch] for ch in channels]
            if self._multiray_combo_mode == 'grouped' and not force_perpair:
                ch_to_group, _ = self._build_channel_groups(channels)
                return self.multiray_backend.grouped_grid(
                    rho_vec, phi_vec_rad, z_vec, self._pa_center, ant_xy, slot_tables, len(names), corr_data,
                    channels, ch_to_group, max(ch_to_group.values()) + 1, pair_weights=pair_weights)
            return self.multiray_backend.perpair_grid(
                rho_vec, phi_vec_rad, z_vec, self._pa_center, ant_xy, slot_tables, len(names), corr_data, channels,
                pair_weights=pair_weights)
        if src_enu is None:
            src_enu = self._build_source_enu_matrix(rho_vec, phi_vec_rad, z_vec)
        tt_data = self._compute_tt_multiray(src_enu, channels)
        return self._multiray_correlate(corr_data, tt_data, channels, pair_weights=pair_weights,
                                        force_perpair=force_perpair)

    def _multiray_on_device(self):
        """Whether multi-ray maps look their travel times up on the ``multiray_backend`` device (strict table edge)."""
        return (self.multiray_backend is not None and USE_NUMBA_GROUPED and not self._tolerant_table_edge)

    def _multiray_correlate(self, corr_data, tt_all, channels,
                            pair_weights=None, force_perpair=False):
        """Dispatch to per-pair or grouped multiray correlator.

        Grouped mode runs ``grouped_multiray_points`` (one point-major call per grid),
        equal bit for bit to ``grouped_multiray_numba``, which
        ``use_fused_correlator: false`` selects.

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
        force_perpair : bool
            Force per-pair mode regardless of combo_mode setting.
            Used for the coarse grid where grouped is too expensive.

        With ``multiray_backend`` set (``reco3d_batch_gpu.GpuMultiray``) the per-pair and
        grouped maps of the numba kernels are computed by that backend instead.

        Returns
        -------
        tuple
            (mean_corr_map, max_corr)
        """
        use_grouped = (self._multiray_combo_mode == 'grouped'
                       and not force_perpair)
        if self.multiray_backend is not None and USE_NUMBA_GROUPED:
            if use_grouped:
                ch_to_group, _ = self._build_channel_groups(channels)
                return self.multiray_backend.grouped(
                    corr_data, tt_all, channels, ch_to_group, max(ch_to_group.values()) + 1,
                    pair_weights=pair_weights)
            return self.multiray_backend.perpair(corr_data, tt_all, channels, pair_weights=pair_weights)
        if use_grouped and USE_NUMBA_GROUPED:
            ch_to_group, _ = self._build_channel_groups(channels)
            n_groups = max(ch_to_group.values()) + 1
            grouped = (grouped_multiray_points if USE_NUMBA and self._use_fused_correlator
                       else grouped_multiray_numba)
            return grouped(corr_data, tt_all, channels, ch_to_group, n_groups,
                           pair_weights=pair_weights)
        if use_grouped:
            return self._correlator_grouped_multiray(
                corr_data, tt_all, channels, pair_weights=pair_weights
            )
        if (self._use_gpu and USE_CUPY
                and _FUSED_MULTIRAY_CORR_KERNEL is not None
                and not use_grouped):
            rts = self._active_ray_types
            n_rt = len(rts)
            ch_list = list(channels)
            n_ch = len(ch_list)
            grid_shape = None
            for ch in ch_list:
                for rt in rts:
                    if rt in tt_all.get(ch, {}):
                        grid_shape = tt_all[ch][rt].shape
                        break
                if grid_shape is not None:
                    break
            if grid_shape is not None:
                n_pts = int(np.prod(grid_shape))
                if n_pts >= self._gpu_min_grid_cells:
                    tt_np = np.full((n_ch, n_rt, n_pts), np.nan, dtype=np.float64)
                    for ci, ch in enumerate(ch_list):
                        for ri, rt in enumerate(rts):
                            if rt in tt_all.get(ch, {}):
                                tt_np[ci, ri] = tt_all[ch][rt].ravel()
                    mc, mx = self._multiray_correlate_gpu(
                        corr_data, tt_np, ch_list, pair_weights=pair_weights)
                    return mc.reshape(grid_shape), mx
        if USE_NUMBA_GROUPED:
            return perpair_multiray_numba(
                corr_data, tt_all, channels, pair_weights=pair_weights
            )
        return self._correlator_lean_multiray(
            corr_data, tt_all, channels, pair_weights=pair_weights
        )

    def _two_arrival_pack(self, channels):
        """Pack the solution-ordered tables of a channel set for the two-arrival kernels.

        Returns:
            Dict with 'ant_xy' (n_ch, 2) and, for slot in (0, 1), 'td{slot}_values'
            (n_ch, nr_max, nz_max), its finiteness mask 'td{slot}_ok' and the grid
            parameter arrays
            'td{slot}_r_min', 'td{slot}_dr_inv', 'td{slot}_nr', 'td{slot}_z_min',
            'td{slot}_dz_inv', 'td{slot}_nz'; cached per channel tuple.
        """
        key = tuple(channels)
        if key in self._two_arrival_packed:
            return self._two_arrival_packed[key]
        n_ch = len(channels)
        packed = {'ant_xy': np.array([self.ant_locs[ch][:2] for ch in channels],
                                     dtype=np.float64)}
        for slot, rt in enumerate(SOLUTION_TYPES):
            tds = [self._two_arrival_interpolators[ch][rt] for ch in channels]
            nr_max = max(td.nr for td in tds)
            nz_max = max(td.nz for td in tds)
            values = np.full((n_ch, nr_max, nz_max), np.nan, dtype=np.float64)
            for ci, td in enumerate(tds):
                values[ci, :td.nr, :td.nz] = td.values
            packed[f'td{slot}_values'] = values
            packed[f'td{slot}_ok'] = np.isfinite(values)
            packed[f'td{slot}_r_min'] = np.array([td.r_min for td in tds], dtype=np.float64)
            packed[f'td{slot}_dr_inv'] = np.array([td.dr_inv for td in tds], dtype=np.float64)
            packed[f'td{slot}_nr'] = np.array([td.nr for td in tds], dtype=np.int64)
            packed[f'td{slot}_z_min'] = np.array([td.z_min for td in tds], dtype=np.float64)
            packed[f'td{slot}_dz_inv'] = np.array([td.dz_inv for td in tds], dtype=np.float64)
            packed[f'td{slot}_nz'] = np.array([td.nz for td in tds], dtype=np.int64)
        self._two_arrival_packed[key] = packed
        return packed

    @staticmethod
    def _table_lookup(td, r, z):
        """Bilinear travel-time lookup on one table; -inf out of bounds, NaN at a NaN corner."""
        if USE_NUMBA:
            return float(_bilinear_scalar_numba(
                td.values, td.r_min, td.dr_inv, td.nr, td.z_min, td.dz_inv, td.nz, r, z))
        return float(td.interp(np.array([[r, z]]))[0])

    @staticmethod
    def _two_arrival_launch(td1, r, z, ice):
        """Launch angle and critical-angle mask of the solution_1 ray at one source cell.

        The horizontal slowness of the ray at the source is the table gradient
        p = dT1/dR (central difference over 1 m, one sided at a table boundary).
        With n(z) = n_ice - delta_n exp(z / z_0) the launch zenith follows from
        sin(theta) = c p / n(z); the ray is totally reflected at the surface when
        theta exceeds arcsin(1 / n(z)), which is the condition c p > 1. A
        solution_1 ray that turns below the surface (refracted) has c p >= n(0) and
        therefore mask 1 as well, matching its full amplitude. The approximation is
        the bilinear table gradient, exact to the table's interpolation error.

        Args:
            td1: solution_1 TableData of the channel.
            r: Horizontal distance from the source to the antenna (m, at least 1).
            z: Absolute source depth (m).
            ice: (n_ice, delta_n, z_0) of the exponential profile.

        Returns:
            (tt1, launch_deg, critical_deg, mask): the solution_1 travel time in ns,
            the launch zenith and the critical angle in degrees, and the mask as
            0.0 or 1.0. launch_deg is NaN where the solution or its gradient is
            undefined.
        """
        lookup = InterferometricReco3D._table_lookup
        tt1 = lookup(td1, r, z)
        n_ice, delta_n, z_0 = ice
        n_src = n_ice - delta_n * np.exp(z / z_0) if z <= 0 else 1.0
        critical_deg = float(np.degrees(np.arcsin(1.0 / n_src)))
        if not (np.isfinite(tt1) and tt1 > 0):
            return tt1, np.nan, critical_deg, 0.0
        t_plus = lookup(td1, r + 0.5, z)
        t_minus = lookup(td1, r - 0.5, z)
        if np.isfinite(t_plus) and np.isfinite(t_minus):
            p = t_plus - t_minus
        elif np.isfinite(t_plus):
            p = 2.0 * (t_plus - tt1)
        elif np.isfinite(t_minus):
            p = 2.0 * (tt1 - t_minus)
        else:
            return tt1, np.nan, critical_deg, 0.0
        sin_launch = p * _C_M_PER_NS / n_src
        launch_deg = float(np.degrees(np.arcsin(min(sin_launch, 1.0))))
        return tt1, launch_deg, critical_deg, float(sin_launch > 1.0 / n_src)

    def two_arrival_mask(self, ch, rho, phi_deg, z):
        """Critical-angle mask of one channel for a source given in the search frame.

        Args:
            ch: Channel ID (its solution-ordered tables must be loaded).
            rho, phi_deg, z: Source position relative to the PA centre (m, deg, m).

        Returns:
            (mask, launch_deg, critical_deg) as described in ``_two_arrival_launch``.
        """
        phi = np.radians(phi_deg)
        x = rho * np.cos(phi) + self._pa_center[0]
        y = rho * np.sin(phi) + self._pa_center[1]
        r = max(np.hypot(x - self.ant_locs[ch][0], y - self.ant_locs[ch][1]), 1.0)
        _, launch_deg, critical_deg, mask = self._two_arrival_launch(
            self._two_arrival_interpolators[ch]['solution_1'], r, z, self._two_arrival_ice)
        return mask, launch_deg, critical_deg

    @staticmethod
    def _packed_lookup(corr_arr, dt, offset, delays):
        """Linear correlation lookup with the Numba kernels' bin rule.

        A delay whose lower bin index k satisfies 0 <= k < len(corr_arr) - 1 is
        interpolated between bins k and k + 1; every other delay (including the
        last bin and non-finite delays) gives 0, as in the kernels.

        Args:
            corr_arr: Correlation samples of one pair.
            dt, offset: Lag step and lag of the first sample (ns).
            delays: Array of delays (ns).

        Returns:
            Array of correlation values, one per delay.
        """
        kf = (delays - offset) / dt
        kf = np.where(np.isfinite(kf), kf, -1.0)
        k = np.floor(kf).astype(np.int64)
        inside = (k >= 0) & (k < len(corr_arr) - 1)
        k = np.clip(k, 0, max(len(corr_arr) - 2, 0))
        alpha = kf - k
        return np.where(inside, corr_arr[k] + (corr_arr[k + 1] - corr_arr[k]) * alpha, 0.0)

    def _two_arrival_numpy(self, x, y, z, channels, cache, settings):
        """Consistent two-arrival correlation at source points without the Numba kernels.

        Args:
            x, y, z: Flat arrays of absolute source coordinates (m).
            channels: Channel IDs.
            cache: Optimizer cache holding the correlation data and pair weights.
            settings: From ``_two_arrival_settings``.

        Returns:
            Array of the weighted mean pair values, one per point.
        """
        n_pts = len(x)
        tt = {}
        for ch in channels:
            tds = self._two_arrival_interpolators[ch]
            r = np.maximum(np.hypot(x - self.ant_locs[ch][0], y - self.ant_locs[ch][1]), 1.0)
            tt0 = tds['solution_0'].interp(np.column_stack((r, z)))
            tt1 = np.empty(n_pts)
            mask = np.empty(n_pts)
            for i in range(n_pts):
                tt1[i], _, _, mask[i] = self._two_arrival_launch(
                    tds['solution_1'], r[i], z[i], self._two_arrival_ice)
            tt[ch] = (tt0, np.isfinite(tt0) & (tt0 > 0), tt1,
                      np.isfinite(tt1) & (tt1 > 0), mask)
        pw = cache['pw']
        total = np.zeros(n_pts)
        for pidx, (c1, c2) in enumerate(cache['ch_pairs']):
            corr_arr, dt, offset = cache['corr_data'][pidx]
            t0a, v0a, t1a, v1a, ma = tt[c1]
            t0b, v0b, t1b, v1b, mb = tt[c2]
            valid = v0a & v0b
            value = np.where(valid, self._packed_lookup(corr_arr, dt, offset, t0a - t0b), 0.0)
            w2 = np.full(n_pts, settings['second_weight'])
            if settings['weight_mode'] == 'mask':
                w2 *= ma * mb
            both = valid & v1a & v1b & (w2 != 0)
            value += np.where(both, w2 * self._packed_lookup(corr_arr, dt, offset, t1a - t1b), 0.0)
            total += pw[pidx] * value
        return total / cache['w_total'] if cache['w_total'] > 0 else total

    def _two_arrival_at_point(self, params, channels, cache, settings):
        """Negative consistent two-arrival correlation at one (rho, phi_deg, z) point.

        Args:
            params: [rho, phi_deg, z] in m, deg, m.
            channels: Channel IDs.
            cache: Optimizer cache built from the raw correlation data.
            settings: From ``_two_arrival_settings``.

        Returns:
            Negative two-arrival value (for minimization).
        """
        rho, phi_deg, z = params
        phi_rad = phi_deg * (np.pi / 180.0)
        x = rho * np.cos(phi_rad) + self._pa_center[0]
        y = rho * np.sin(phi_rad) + self._pa_center[1]
        if USE_NUMBA and 'corr_packed' in cache:
            t = self._two_arrival_pack(channels)
            n_ice, delta_n, z_0 = self._two_arrival_ice
            return -float(_two_arrival_point_numba(
                float(x), float(y), float(z), t['ant_xy'],
                t['td0_values'], t['td0_ok'], t['td0_r_min'], t['td0_dr_inv'], t['td0_nr'],
                t['td0_z_min'], t['td0_dz_inv'], t['td0_nz'],
                t['td1_values'], t['td1_ok'], t['td1_r_min'], t['td1_dr_inv'], t['td1_nr'],
                t['td1_z_min'], t['td1_dz_inv'], t['td1_nz'],
                n_ice, delta_n, z_0, _C_M_PER_NS,
                settings['second_weight'], settings['weight_mode'] == 'mask',
                cache['corr_packed'], cache['corr_lengths'],
                cache['corr_dts'], cache['corr_offsets'],
                cache['pair_ch1'], cache['pair_ch2'], cache['pw'], cache['w_total']))
        return -float(self._two_arrival_numpy(
            np.array([x]), np.array([y]), np.array([float(z)]), channels, cache,
            settings)[0])

    def _two_arrival_grid(self, rho_vec, phi_vec_rad, z_vec, channels, cache, settings):
        """Consistent two-arrival correlation map on a (rho, phi, z) grid.

        Args:
            rho_vec, phi_vec_rad, z_vec: Grid axes (m, rad, m).
            channels: Channel IDs.
            cache: Optimizer cache built from the raw correlation data.
            settings: From ``_two_arrival_settings``.

        Returns:
            Array of shape (n_rho, n_phi, n_z).
        """
        shape = (len(rho_vec), len(phi_vec_rad), len(z_vec))
        if USE_NUMBA and 'corr_packed' in cache:
            t = self._two_arrival_pack(channels)
            n_ice, delta_n, z_0 = self._two_arrival_ice
            flat = _fused_two_arrival_grid_numba(
                np.ascontiguousarray(rho_vec, dtype=np.float64),
                np.ascontiguousarray(phi_vec_rad, dtype=np.float64),
                np.ascontiguousarray(z_vec, dtype=np.float64),
                float(self._pa_center[0]), float(self._pa_center[1]), t['ant_xy'],
                t['td0_values'], t['td0_ok'], t['td0_r_min'], t['td0_dr_inv'], t['td0_nr'],
                t['td0_z_min'], t['td0_dz_inv'], t['td0_nz'],
                t['td1_values'], t['td1_ok'], t['td1_r_min'], t['td1_dr_inv'], t['td1_nr'],
                t['td1_z_min'], t['td1_dz_inv'], t['td1_nz'],
                n_ice, delta_n, z_0, _C_M_PER_NS,
                settings['second_weight'], settings['weight_mode'] == 'mask',
                cache['corr_packed'], cache['corr_lengths'],
                cache['corr_dts'], cache['corr_offsets'],
                cache['pair_ch1'], cache['pair_ch2'], cache['pw'], cache['w_total'])
            return flat.reshape(shape)
        src = self._build_source_enu_matrix(rho_vec, phi_vec_rad, z_vec).reshape(-1, 3)
        return self._two_arrival_numpy(
            src[:, 0], src[:, 1], src[:, 2], channels, cache, settings).reshape(shape)

    def _with_two_arrival(self, entry, channels, cache, settings):
        """Append the two-arrival value and the raw correlation to a raw-graded pool entry.

        Args:
            entry: (rho, phi_deg, z, raw_corr, origin, prepolish_corr, unpolished).
            channels: Channel IDs.
            cache: Optimizer cache built from the raw correlation data.
            settings: From ``_two_arrival_settings``.

        Returns:
            The entry extended by (corr_two_arrival, raw_corr_single), its fourth
            field the value ``max_corr_source`` names.
        """
        two = -self._two_arrival_at_point(list(entry[:3]), channels, cache, settings)
        shown = entry[3] if settings['max_corr_source'] == 'raw' else two
        return tuple(entry[:3]) + (shown,) + tuple(entry[4:7]) + (two, entry[3])
