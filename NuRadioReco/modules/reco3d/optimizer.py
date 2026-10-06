"""Peak extraction, the point objective and the optimizer of the 3D reconstruction."""

import numpy as np
import itertools
import numbers
from collections import namedtuple

from scipy.optimize import minimize

from NuRadioReco.utilities.reco3d_kernels import USE_NUMBA

from NuRadioReco.modules.reco3d.shared import logger, _LBFGSB_ABS_STEP, _LBFGSB_REL_STEP

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import (
        _scalar_grouped_corr_numba,
        _scalar_singleray_corr_numba,
        _scalar_singleray_corr_grad_numba,
        _lbfgsb_singleray_value_grad,
        _lbfgsb_fd_points,
        _grouped_fd_value_grad,
        _bilinear_scalar_numba,
        _multiray_point_tts_numba,
        _compass_search_numba,
        _map_snr_numba,
        _top_peaks_numba,
    )


def _lbfgsb_fd_steps(x, lb, ub):
    """Forward-difference steps scipy's L-BFGS-B takes without a gradient, for x inside [lb, ub].

    ``approx_derivative`` with method '2-point' and the absolute step ``eps`` = 1e-8
    (a relative step where x + 1e-8 rounds to x), each step reversed when it would
    leave the bounds and fits on the other side, else replaced by the distance to
    the farther bound.

    Returns:
        (n,) signed steps.
    """
    sign = (x >= 0).astype(float) * 2 - 1
    h = np.where(((x + _LBFGSB_ABS_STEP) - x) == 0,
                 _LBFGSB_REL_STEP * sign * np.maximum(1.0, np.abs(x)), _LBFGSB_ABS_STEP)
    if np.all((lb == -np.inf) & (ub == np.inf)):
        return h
    lower, upper = x - lb, ub - x
    stepped = x + h
    fitting = np.abs(h) <= np.maximum(lower, upper)
    h = np.where(((stepped < lb) | (stepped > ub)) & fitting, -h, h)
    h = np.where((upper >= lower) & ~fitting, upper, h)
    return np.where((upper < lower) & ~fitting, -lower, h)


_LbfgsbResult = namedtuple('_LbfgsbResult', ['x', 'fun', 'nit', 'nfev'])


def _minimize_lbfgsb(fun, x0, bounds, maxiter, ftol, args=()):
    """``minimize(fun, x0, args, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter', 'ftol'})``
    without scipy's function wrappers.

    The loop of scipy's ``_minimize_lbfgsb`` (scipy 1.15 or later, the C ``setulb``) with its defaults
    (maxcor 10, gtol 1e-5, maxls 20, maxfun 15000): the same ``setulb`` calls on the same arrays, the
    function evaluated at a copy of the requested point unless that point equals the previous one (as
    ``ScalarFunction`` caches it), the same stopping rules. The iterates, ``x`` and ``fun`` equal
    scipy's bit for bit; only the per-call wrapper overhead is gone.

    Args:
        fun: Callable ``fun(x, *args) -> (value, gradient)``.
        x0: Start point.
        bounds: Sequence of (low, high) per variable (None for an open side).
        maxiter: Largest number of iterations.
        ftol: Relative reduction of the value that stops the iteration.
        args: Extra positional arguments of ``fun``.

    Returns:
        Named tuple (x, fun, nit, nfev).
    """
    from scipy.optimize import _lbfgsb
    m, pgtol, maxls, maxfun = 10, 1e-5, 20, 15000
    factr = ftol / np.finfo(float).eps
    x0 = np.atleast_1d(np.asarray(x0))
    if x0.dtype.kind in np.typecodes['AllInteger']:
        x0 = np.asarray(x0, dtype=float)
    n = x0.size
    low = np.array([-np.inf if b[0] is None else float(b[0]) for b in bounds], dtype=np.float64)
    high = np.array([np.inf if b[1] is None else float(b[1]) for b in bounds], dtype=np.float64)
    x = np.array(np.clip(x0, low, high), dtype=np.float64)
    nbd = np.zeros(n, np.int32)
    low_bnd = np.zeros(n, np.float64)
    upper_bnd = np.zeros(n, np.float64)
    for i in range(n):
        has_low, has_high = not np.isinf(low[i]), not np.isinf(high[i])
        if has_low:
            low_bnd[i] = low[i]
        if has_high:
            upper_bnd[i] = high[i]
        nbd[i] = {(False, False): 0, (True, False): 1, (True, True): 2, (False, True): 3}[has_low, has_high]
    f = np.array(0.0, dtype=np.float64)
    g = np.zeros((n,), dtype=np.float64)
    wa = np.zeros(2 * m * n + 5 * n + 11 * m * m + 8 * m, np.float64)
    iwa = np.zeros(3 * n, dtype=np.int32)
    task = np.zeros(2, dtype=np.int32)
    ln_task = np.zeros(2, dtype=np.int32)
    lsave = np.zeros(4, dtype=np.int32)
    isave = np.zeros(44, dtype=np.int32)
    dsave = np.zeros(29, dtype=np.float64)

    def evaluate(point):
        """Value and gradient at a copy of the point, the value as a scalar (scipy's wrapper rules)."""
        value, grad = fun(np.copy(point), *args)
        if not np.isscalar(value):
            value = np.asarray(value).item()
        return value, np.atleast_1d(grad)

    last_x = x.astype(np.float64)
    last = evaluate(last_x)
    nfev = 1
    n_iterations = 0
    while True:
        g = g.astype(np.float64)
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr, pgtol, wa, iwa, task, lsave, isave, dsave,
                       maxls, ln_task)
        if task[0] == 3:
            if not np.array_equal(x, last_x):
                last_x = x.astype(np.float64)
                last = evaluate(last_x)
                nfev += 1
            f, g = last
        elif task[0] == 1:
            n_iterations += 1
            if n_iterations >= maxiter:
                task[0] = 5
                task[1] = 504
            elif nfev > maxfun:
                task[0] = 5
                task[1] = 502
        else:
            break
    return _LbfgsbResult(x, f, n_iterations, nfev)


class OptimizerMixin:
    """Methods of InterferometricReco3D for peak extraction, the point objective and the optimizer."""

    def _interp_corr_scalar(self, corr_arr, dt, offset, delay):
        """Interpolate correlation at a single delay value.

        Parameters
        ----------
        corr_arr : np.ndarray
            1D correlation array.
        dt : float
            Sample spacing.
        offset : float
            Time offset of first sample.
        delay : float
            Delay to interpolate at.

        Returns
        -------
        float
            Interpolated correlation value.
        """
        M = len(corr_arr)
        kf = (delay - offset) / dt
        k = int(np.floor(kf))
        if k < 0 or k >= M - 1:
            return 0.0
        alpha = kf - k
        return corr_arr[k] + (corr_arr[k + 1] - corr_arr[k]) * alpha

    def _tt_scalar(self, ch, rt, r, z):
        """Look up travel time for a single (R, Z) point.

        Uses Numba scalar bilinear interpolation when available, falling back
        to SciPy RegularGridInterpolator.

        Args:
            ch: Channel ID.
            rt: Ray type string ('direct', 'refracted', 'reflected').
            r: Horizontal distance in meters.
            z: Depth in meters.

        Returns:
            Travel time in ns, or -inf if out of bounds.
        """
        td = self._multiray_interpolators[ch][rt]
        if USE_NUMBA:
            return _bilinear_scalar_numba(
                td.values, td.r_min, td.dr_inv, td.nr,
                td.z_min, td.dz_inv, td.nz, r, z)
        return td.interp(np.array([[r, z]]))[0]

    def _build_optimizer_cache(self, channels, pair_weights, corr_data=None,
                               packed=None):
        """Pre-compute invariant data for the optimizer objective.

        Event-invariant geometry (channel positions, pair indices, TT tables)
        is stored in ``self._opt_geom_cache`` keyed by the channel tuple so
        repeated calls across events reuse it. Only the per-event pieces
        (pair weights, packed correlation arrays) are set each call; the
        singleray tables are the shared per-process stack indexed by slot.

        Args:
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.
            corr_data: Pre-computed correlation data (for Numba packing).
            packed: CorrPacked of corr_data; when given its arrays are used
                directly instead of packing corr_data again.

        Returns:
            Dict with cached arrays for fast objective evaluation.
        """
        geom_key = (tuple(channels), bool(self._multi_ray_types),
                    self._multiray_combo_mode)
        geom = self._opt_geom_cache.get(geom_key)
        if geom is not None:
            cache = dict(geom)
            cache['corr_data'] = corr_data
            n_pairs = cache['n_pairs']
            pw = np.ones(n_pairs, dtype=np.float64)
            if pair_weights is not None:
                pw = np.asarray(pair_weights, dtype=np.float64)
            cache['pw'] = pw
            cache['w_total'] = float(pw.sum())
            if (corr_data is not None or packed is not None) and USE_NUMBA:
                self._set_cache_corr(cache, corr_data, packed)
            return cache

        ch_pairs = list(itertools.combinations(channels, 2))
        n_pairs = len(ch_pairs)
        pw = np.ones(n_pairs, dtype=np.float64)
        if pair_weights is not None:
            pw = np.asarray(pair_weights, dtype=np.float64)
        w_total = float(pw.sum())

        pa_center = self._pa_center

        ant_pos = np.array([self.ant_locs[ch][:2] for ch in channels])

        ch_idx = list(range(len(channels)))
        pair_ch1 = np.array([p[0] for p in
                             itertools.combinations(ch_idx, 2)],
                            dtype=np.int64)
        pair_ch2 = np.array([p[1] for p in
                             itertools.combinations(ch_idx, 2)],
                            dtype=np.int64)

        # Pack table data for fast scalar TT lookup
        n_ch = len(channels)
        td_list = []
        if self._multi_ray_types:
            for ch in channels:
                ch_tds = []
                for rt in self._active_ray_types:
                    ch_tds.append(self._multiray_interpolators[ch][rt])
                td_list.append(ch_tds)
        else:
            for ch in channels:
                td_list.append([self._interpolators[ch]])

        cache = {
            'ch_pairs': ch_pairs,
            'n_pairs': n_pairs,
            'pw': pw,
            'w_total': w_total,
            'corr_data': corr_data,
            'pa_center': pa_center,
            'ant_pos': ant_pos,
            'channels': channels,
            'pair_ch1': pair_ch1,
            'pair_ch2': pair_ch2,
            'td_list': td_list,
            'n_ch': n_ch,
        }

        if self._multi_ray_types and self._multiray_combo_mode == 'grouped':
            ch_to_group, _ = self._build_channel_groups(channels)
            n_groups = max(ch_to_group.values()) + 1
            rt_map = {rt: i for i, rt in enumerate(self._active_ray_types)}
            group_rts_all = []
            for gidx in range(n_groups):
                group_chs = [ch for ch in channels
                             if ch_to_group[ch] == gidx]
                rts = set(self._active_ray_types)
                for ch in group_chs:
                    avail = set(self._multiray_interpolators[ch].keys())
                    rts &= avail
                if not rts:
                    for ch in group_chs:
                        rts |= set(self._multiray_interpolators[ch].keys())
                if not rts:
                    rts = {self._active_ray_types[0]}
                group_rts_all.append(sorted(rts))

            combos = list(itertools.product(*group_rts_all))
            n_combos = len(combos)
            ch_group_indices = [ch_to_group[ch] for ch in channels]
            combo_table = np.empty((n_combos, n_ch), dtype=np.int64)
            for ci, combo in enumerate(combos):
                for chi in range(n_ch):
                    combo_table[ci, chi] = rt_map[combo[ch_group_indices[chi]]]
            cache['combo_table'] = combo_table
            cache['n_combos'] = n_combos
            cache['ch_group'] = np.array(ch_group_indices, dtype=np.int64)
            group_rts = np.full((n_groups, max(len(r) for r in group_rts_all)), -1, dtype=np.int64)
            for gidx, rts in enumerate(group_rts_all):
                group_rts[gidx, :len(rts)] = [rt_map[rt] for rt in rts]
            cache['group_rts'] = group_rts
            cache['group_nrt'] = np.array([len(r) for r in group_rts_all], dtype=np.int64)

            # Also keep string-based combo_rt for fallback
            combo_rt = []
            for combo in combos:
                combo_rt.append([combo[ch_to_group[ch]] for ch in channels])
            cache['combo_rt'] = combo_rt

        if (corr_data is not None or packed is not None) and USE_NUMBA:
            self._set_cache_corr(cache, corr_data, packed)

        if USE_NUMBA and self._multi_ray_types:
            from numba import typed
            tds = [td for ch_tds in td_list for td in ch_tds]
            cache['mr_tables'] = (
                typed.List([np.ascontiguousarray(td.values, dtype=np.float64) for td in tds]),
                np.array([td.r_min for td in tds], dtype=np.float64),
                np.array([td.dr_inv for td in tds], dtype=np.float64),
                np.array([td.nr for td in tds], dtype=np.int64),
                np.array([td.z_min for td in tds], dtype=np.float64),
                np.array([td.dz_inv for td in tds], dtype=np.float64),
                np.array([td.nz for td in tds], dtype=np.int64))

        if USE_NUMBA and not self._multi_ray_types:
            geom_tables = self._pack_singleray_tables(channels)
            cache['td_values_packed'] = geom_tables['td_values']
            cache['td_ok'] = geom_tables['td_ok']
            cache['td_slot'] = geom_tables['td_slot']
            for k in ('td_nr', 'td_nz', 'td_r_min', 'td_dr_inv', 'td_z_min',
                      'td_dz_inv', 'pa_x', 'pa_y'):
                cache[k] = geom_tables[k]

        # Store event-invariant geometry for reuse across events. The
        # per-event parts (pw, w_total, corr_packed, corr_dts, corr_offsets)
        # will be overwritten on subsequent calls.
        geom = {k: v for k, v in cache.items()
                if k not in ('pw', 'w_total', 'corr_data', 'corr_packed', 'corr_lengths',
                             'corr_dts', 'corr_offsets', 'corr_inv_dts', 'packed')}
        self._opt_geom_cache[geom_key] = geom

        return cache

    def _set_cache_corr(self, cache, corr_data, packed):
        """Attach the packed correlation arrays of one event to an optimizer cache."""
        if packed is None:
            packed = self._pack_corr_data(corr_data)
        cache['packed'] = packed
        cache['corr_packed'] = packed.corr
        cache['corr_lengths'] = packed.lengths
        cache['corr_dts'] = packed.dts
        cache['corr_offsets'] = packed.offsets
        cache['corr_inv_dts'] = packed.inv_dts

    def _correlation_at_point(self, params, corr_data, channels,
                              pair_weights=None, _cache=None):
        """Evaluate negative mean correlation at a single (rho, phi_deg, z) point.

        When multi_ray_types is enabled, tests all 9 ray type combinations per
        pair and takes the maximum.

        Parameters
        ----------
        params : array-like
            [rho, phi_deg, z] in meters and degrees.
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        channels : list
            Channel IDs.
        pair_weights : list or None
            Per-pair weights.
        _cache : dict or None
            Pre-computed invariants from _build_optimizer_cache.

        Returns
        -------
        float
            Negative mean correlation (for minimization).
        """
        rho, phi_deg, z = params
        phi_rad = phi_deg * (np.pi / 180.0)
        self.work['point_evals'] += 1
        self.work['tt_lookups'] += len(channels) * (self._n_ray_slots if self._multi_ray_types else 1)

        if _cache is not None:
            pa_center = _cache['pa_center']
            ch_pairs = _cache['ch_pairs']
            w_total = _cache['w_total']
            pw = _cache['pw']
            ant_pos = _cache['ant_pos']
        else:
            pa_center = self._pa_center
            ch_pairs = list(itertools.combinations(channels, 2))
            if pair_weights is not None:
                pw = np.array(pair_weights, dtype=np.float64)
            else:
                pw = np.ones(len(ch_pairs))
            w_total = float(pw.sum())
            ant_pos = None

        # Fast path: fused singleray Numba kernel
        if (not self._multi_ray_types and USE_NUMBA and _cache is not None
                and 'td_values_packed' in _cache
                and 'corr_packed' in _cache):
            return _scalar_singleray_corr_numba(
                float(rho), float(phi_rad), float(z),
                _cache['pa_x'], _cache['pa_y'],
                _cache['ant_pos'],
                _cache['td_values_packed'], _cache['td_ok'], _cache['td_slot'],
                _cache['td_r_min'], _cache['td_dr_inv'], _cache['td_nr'],
                _cache['td_z_min'], _cache['td_dz_inv'], _cache['td_nz'],
                _cache['corr_packed'], _cache['corr_lengths'],
                _cache['corr_dts'], _cache['corr_offsets'],
                _cache['pair_ch1'], _cache['pair_ch2'], pw, w_total,
                self._valid_norm, self._valid_floor)

        x = rho * np.cos(phi_rad) + pa_center[0]
        y = rho * np.sin(phi_rad) + pa_center[1]

        if self._multi_ray_types and _cache is not None and 'mr_tables' in _cache:
            tt_vals, tt_valid = _multiray_point_tts_numba(
                float(x), float(y), float(z), ant_pos, *_cache['mr_tables'], self._n_ray_slots)
            if self._multiray_combo_mode == 'grouped' and 'corr_packed' in _cache:
                return _scalar_grouped_corr_numba(
                    tt_vals, tt_valid,
                    _cache['corr_packed'], _cache['corr_lengths'],
                    _cache['corr_dts'], _cache['corr_offsets'],
                    _cache['pair_ch1'], _cache['pair_ch2'], pw,
                    _cache['ch_group'], _cache['group_rts'], _cache['group_nrt'],
                    _cache['n_pairs'], w_total, self._mr_walk)
        if self._multi_ray_types:
            n_ch = _cache['n_ch'] if _cache else len(channels)

            n_rt = self._n_ray_slots
            tt_vals = np.full((n_ch, n_rt), -np.inf, dtype=np.float64)
            tt_valid = np.zeros((n_ch, n_rt), dtype=np.bool_)
            for ci in range(n_ch):
                ch = channels[ci]
                if ant_pos is not None:
                    dx = x - ant_pos[ci, 0]
                    dy = y - ant_pos[ci, 1]
                else:
                    pos = self.ant_locs[ch]
                    dx = x - pos[0]
                    dy = y - pos[1]
                r = max(np.sqrt(dx * dx + dy * dy), 1.0)
                if _cache is not None:
                    td_ch = _cache['td_list'][ci]
                    for rti in range(n_rt):
                        td = td_ch[rti]
                        if USE_NUMBA:
                            tt = _bilinear_scalar_numba(
                                td.values, td.r_min, td.dr_inv, td.nr,
                                td.z_min, td.dz_inv, td.nz, r, z)
                        else:
                            tt = td.interp(np.array([[r, z]]))[0]
                        if np.isfinite(tt) and tt > 0:
                            tt_vals[ci, rti] = tt
                            tt_valid[ci, rti] = True
                else:
                    for rti, rt in enumerate(self._active_ray_types):
                        tt = self._tt_scalar(ch, rt, r, z)
                        if np.isfinite(tt) and tt > 0:
                            tt_vals[ci, rti] = tt
                            tt_valid[ci, rti] = True

            # Fast Numba path for grouped mode
            if (self._multiray_combo_mode == 'grouped' and USE_NUMBA
                    and _cache is not None and 'corr_packed' in _cache):
                return _scalar_grouped_corr_numba(
                    tt_vals, tt_valid,
                    _cache['corr_packed'], _cache['corr_lengths'],
                    _cache['corr_dts'], _cache['corr_offsets'],
                    _cache['pair_ch1'], _cache['pair_ch2'], pw,
                    _cache['ch_group'], _cache['group_rts'], _cache['group_nrt'],
                    _cache['n_pairs'], w_total, self._mr_walk)

            # Python fallback for grouped mode
            if self._multiray_combo_mode == 'grouped':
                ch_tt = {}
                for ci, ch in enumerate(channels):
                    ch_tt_ch = {}
                    for rti, rt in enumerate(self._active_ray_types):
                        if tt_valid[ci, rti]:
                            ch_tt_ch[rt] = tt_vals[ci, rti]
                    ch_tt[ch] = ch_tt_ch
                return self._correlation_at_point_grouped(
                    ch_tt, corr_data, channels, ch_pairs, pw, w_total,
                    _cache=_cache
                )

            # Per-pair multiray mode
            total = 0.0
            for pidx, (c1, c2) in enumerate(ch_pairs):
                best_val = 0.0
                corr_arr, dt, offset = corr_data[pidx]
                ci1 = channels.index(c1)
                ci2 = channels.index(c2)
                for rti1 in range(n_rt):
                    if not tt_valid[ci1, rti1]:
                        continue
                    for rti2 in range(n_rt):
                        if not tt_valid[ci2, rti2]:
                            continue
                        delay = tt_vals[ci1, rti1] - tt_vals[ci2, rti2]
                        val = self._interp_corr_scalar(
                            corr_arr, dt, offset, delay)
                        if val > best_val:
                            best_val = val
                total += pw[pidx] * best_val
            return -total / w_total if w_total > 0 else 0.0

        # Single-table mode
        travel_times = {}
        for ci, ch in enumerate(channels):
            if ant_pos is not None:
                dx = x - ant_pos[ci, 0]
                dy = y - ant_pos[ci, 1]
            else:
                pos = self.ant_locs[ch]
                dx = x - pos[0]
                dy = y - pos[1]
            r = max(np.sqrt(dx * dx + dy * dy), 1.0)
            td = self._interpolators[ch]
            if USE_NUMBA:
                travel_times[ch] = _bilinear_scalar_numba(
                    td.values, td.r_min, td.dr_inv, td.nr,
                    td.z_min, td.dz_inv, td.nz, r, z)
            else:
                travel_times[ch] = td.interp(np.array([[r, z]]))[0]

        total = 0.0
        for pidx, (c1, c2) in enumerate(ch_pairs):
            t1 = travel_times[c1]
            t2 = travel_times[c2]
            if not np.isfinite(t1) or not np.isfinite(t2):
                continue
            delay = t1 - t2
            corr_arr, dt, offset = corr_data[pidx]
            val = self._interp_corr_scalar(corr_arr, dt, offset, delay)
            total += pw[pidx] * val

        return -total / w_total if w_total > 0 else 0.0

    def _subbin_peak(self, corr_map, rho_vec, phi_vec_deg, z_vec, peak):
        """Estimate the sub-bin position of a coarse peak from a parabola per axis.

        Along each axis the peak bin and its two neighbours are fitted with a
        parabola in index space and the vertex offset (clipped to half a bin)
        is mapped back to the grid: log-linearly in rho, linearly in z and by
        the uniform step in phi. An axis keeps the bin centre when the peak is
        on the grid edge (phi wraps only when the grid covers the full circle),
        a neighbour is not finite, or the three values do not form a maximum.

        Args:
            corr_map: 3D coarse map (n_rho, n_phi, n_z).
            rho_vec: Coarse rho grid (m).
            phi_vec_deg: Coarse phi grid (deg).
            z_vec: Coarse z grid (m).
            peak: (rho, phi_deg, z, corr) at a grid point.

        Returns:
            (rho, phi_deg, z, corr) with the shifted position and the same corr.
        """
        ir, ip, iz = self._find_peak_bin(peak[0], peak[1], peak[2], rho_vec,
                                         phi_vec_deg, z_vec)
        n_rho, n_phi, n_z = corr_map.shape
        full_circle = n_phi > 2 and np.isclose(
            (phi_vec_deg[-1] - phi_vec_deg[0]) + (phi_vec_deg[1] - phi_vec_deg[0]),
            360.0)

        def offset(values):
            """Return the parabola vertex offset of three neighbouring values in bins.

            The offset is clipped to half a bin and is 0 when a neighbour is not
            finite or the values do not form a maximum.
            """
            y_m, y_0, y_p = values
            denom = y_m - 2.0 * y_0 + y_p
            if not (np.isfinite(y_m) and np.isfinite(y_p)) or denom >= 0.0:
                return 0.0
            return float(np.clip(0.5 * (y_m - y_p) / denom, -0.5, 0.5))

        rho, phi, z = peak[0], peak[1], peak[2]
        if 0 < ir < n_rho - 1:
            d = offset(corr_map[ir - 1:ir + 2, ip, iz])
            rho = float(np.exp(np.interp(ir + d, np.arange(n_rho), np.log(rho_vec))))
        if 0 < iz < n_z - 1:
            d = offset(corr_map[ir, ip, iz - 1:iz + 2])
            z = float(np.interp(iz + d, np.arange(n_z), z_vec))
        if 0 < ip < n_phi - 1 or full_circle:
            d = offset(corr_map[ir, [(ip - 1) % n_phi, ip, (ip + 1) % n_phi], iz])
            phi = float(phi + d * (phi_vec_deg[1] - phi_vec_deg[0]))
        return (rho, phi, z, peak[3])

    def _extract_top_n_peaks(self, corr_map, rho_vec, phi_vec_deg, z_vec, n,
                             separation):
        """Find top-N peaks in the 3D correlation map with minimum separation.

        Parameters
        ----------
        corr_map : np.ndarray
            3D correlation map (n_rho, n_phi, n_z).
        rho_vec : array
            Rho values in meters.
        phi_vec_deg : array
            Phi values in degrees.
        z_vec : array
            Z values in meters.
        n : int
            Number of peaks to find.
        separation : list
            [d_rho, d_phi, d_z] minimum separation.

        Returns
        -------
        list of tuple
            Each element is (rho, phi_deg, z, corr_value).
        """
        if not isinstance(corr_map, np.ndarray):
            return corr_map.top_n_peaks(rho_vec, phi_vec_deg, z_vec, n, separation)
        d_rho, d_phi, d_z = separation
        if USE_NUMBA:
            idx, val = _top_peaks_numba(
                np.ascontiguousarray(corr_map, dtype=np.float64), np.ascontiguousarray(rho_vec, dtype=np.float64),
                np.ascontiguousarray(phi_vec_deg, dtype=np.float64), np.ascontiguousarray(z_vec, dtype=np.float64),
                int(n), float(d_rho), float(d_phi), float(d_z))
            return [(rho_vec[i], phi_vec_deg[j], z_vec[k], float(v))
                    for (i, j, k), v in zip(idx.tolist(), val.tolist())]
        work = corr_map.copy()
        peaks = []

        for _ in range(n):
            if np.all(np.isnan(work)):
                break
            idx = np.unravel_index(np.nanargmax(work), work.shape)
            val = work[idx]
            if np.isnan(val):
                break

            rho_peak = rho_vec[idx[0]]
            phi_peak = phi_vec_deg[idx[1]]
            z_peak = z_vec[idx[2]]
            peaks.append((rho_peak, phi_peak, z_peak, float(val)))

            rho_mask = np.abs(rho_vec - rho_peak) < d_rho
            z_mask = np.abs(z_vec - z_peak) < d_z
            phi_diff = np.abs(phi_vec_deg - phi_peak)
            phi_diff = np.minimum(phi_diff, 360.0 - phi_diff)
            phi_mask = phi_diff < d_phi

            work[np.ix_(rho_mask, phi_mask, z_mask)] = np.nan

        return peaks

    def _compute_map_snr(self, corr_map, peak_idx, exclusion_bins=3):
        """Compute map SNR: peak correlation / RMS of map away from peak.

        Args:
            corr_map: 3D correlation map (n_rho, n_phi, n_z).
            peak_idx: Tuple (i_rho, i_phi, i_z) of the peak bin.
            exclusion_bins: Number of bins to exclude around peak in each dim.

        Returns:
            Map SNR (float), or NaN if map RMS is zero.
        """
        if not isinstance(corr_map, np.ndarray):
            return corr_map.map_snr(peak_idx, exclusion_bins)
        if USE_NUMBA:
            ir, ip, iz = peak_idx
            return float(_map_snr_numba(np.ascontiguousarray(corr_map, dtype=np.float64),
                                        int(ir), int(ip), int(iz), int(exclusion_bins)))
        mask = np.ones(corr_map.shape, dtype=bool)
        ir, ip, iz = peak_idx
        nr, nphi, nz = corr_map.shape
        r_lo = max(0, ir - exclusion_bins)
        r_hi = min(nr, ir + exclusion_bins + 1)
        p_lo = max(0, ip - exclusion_bins)
        p_hi = min(nphi, ip + exclusion_bins + 1)
        z_lo = max(0, iz - exclusion_bins)
        z_hi = min(nz, iz + exclusion_bins + 1)
        mask[r_lo:r_hi, p_lo:p_hi, z_lo:z_hi] = False

        away = corr_map[mask]
        away = away[np.isfinite(away)]
        if len(away) == 0:
            return np.nan
        rms = np.std(away)
        if rms < 1e-12:
            return np.nan
        return float(corr_map[peak_idx]) / rms

    def _find_peak_bin(self, rho, phi_deg, z, rho_vec, phi_vec_deg, z_vec):
        """Find the nearest bin index in the coarse grid for a peak position.

        Args:
            rho, phi_deg, z: Peak position.
            rho_vec, phi_vec_deg, z_vec: Coarse grid vectors.

        Returns:
            Tuple (i_rho, i_phi, i_z).
        """
        ir = int(np.argmin(np.abs(rho_vec - rho)))
        phi_diff = np.abs(phi_vec_deg - phi_deg)
        phi_diff = np.minimum(phi_diff, 360.0 - phi_diff)
        ip = int(np.argmin(phi_diff))
        iz = int(np.argmin(np.abs(z_vec - z)))
        return (ir, ip, iz)

    def _deduplicate_peaks(self, peaks, d_rho=10, d_phi=5, d_z=10):
        """Remove duplicate peaks that are within separation thresholds.

        Keeps the highest-correlation peak from each cluster.

        Args:
            peaks: List of (rho, phi_deg, z, corr, ...) tuples, sorted by corr desc.
            d_rho, d_phi, d_z: Minimum separation thresholds.

        Returns:
            Filtered list of peaks.
        """
        kept = []
        for peak in peaks:
            rho_p, phi_p, z_p = peak[:3]
            is_dup = False
            for rho_k, phi_k, z_k in (k[:3] for k in kept):
                if abs(rho_p - rho_k) < d_rho and abs(z_p - z_k) < d_z:
                    dphi = abs(phi_p - phi_k)
                    dphi = min(dphi, 360 - dphi)
                    if dphi < d_phi:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(peak)
        return kept

    def _optimize_from_seed(self, seed, corr_data, channels, bounds,
                            pair_weights=None, method='L-BFGS-B',
                            maxiter=30, _cache=None, config=None, objective_fn=None,
                            gradient='finite_difference'):
        """Run local optimization from a single seed point.

        Parameters
        ----------
        seed : tuple
            (rho, phi_deg, z) starting point.
        corr_data : list of tuple
            Pre-computed correlation data.
        channels : list
            Channel IDs.
        bounds : list of tuple
            [(rho_min, rho_max), (phi_min, phi_max), (z_min, z_max)].
        pair_weights : list or None
            Per-pair weights.
        method : str
            'L-BFGS-B', 'Nelder-Mead' (scipy) or 'compass' (numba compass search;
            scipy L-BFGS-B when the singleray numba cache is not available).
        maxiter : int
            Maximum optimizer iterations.
        _cache : dict or None
            Pre-computed invariants from _build_optimizer_cache.
        config : dict or None
            Reconstruction config; its compass_* keys set the compass search
            (defaults when None).
        objective_fn : callable or None
            Function of [rho, phi_deg, z] returning the value to minimize; None
            uses the negative raw correlation ``_correlation_at_point``.
        gradient : str
            'finite_difference': L-BFGS-B receives, with the objective, the
            forward differences it would take itself (``_lbfgsb_singleray_value_grad``,
            one compiled call per point), so its iterates are those of scipy's own
            finite differences bit for bit; 'exact': the exact gradient of the
            objective (``_scalar_singleray_corr_grad_numba``). Both need the
            singleray numba cache and the raw objective. Grouped multi-ray with
            the raw objective and the 'mr_tables' cache always takes the
            compiled forward differences (``_grouped_fd_value_grad``);
            otherwise, and for Nelder-Mead, scipy differences
            ``_correlation_at_point`` itself.

        Returns
        -------
        tuple
            (rho, phi_deg, z, corr_value) at optimum.
        """
        rho0, phi0, z0 = seed
        if method == 'compass' and objective_fn is None and self._compass_available(_cache):
            return self._compass_seeds(
                [seed], bounds, _cache, **self._compass_options(config or {}))[0]

        phi_shift = phi0 - 180.0
        phi_start = 180.0

        def objective(params):
            rho, phi_shifted, z = params
            phi_actual = (phi_shifted + phi_shift) % 360.0
            if objective_fn is not None:
                return objective_fn([rho, phi_actual, z])
            return self._correlation_at_point(
                [rho, phi_actual, z], corr_data, channels, pair_weights,
                _cache=_cache
            )

        x0 = np.array([rho0, phi_start, z0])
        shifted_bounds = [
            bounds[0],
            (0.0, 360.0),
            bounds[2],
        ]

        lb = np.array([float(b[0]) for b in shifted_bounds])
        ub = np.array([float(b[1]) for b in shifted_bounds])
        # scipy drops a variable with equal bounds before differencing, which the compiled differences do not
        compiled = (method != 'Nelder-Mead' and objective_fn is None and not self._multi_ray_types
                    and self._compass_available(_cache)
                    and (gradient == 'exact' or bool(np.all(lb < ub))))
        grouped = (method != 'Nelder-Mead' and objective_fn is None and self._multi_ray_types
                   and self._multiray_combo_mode == 'grouped' and _cache is not None
                   and 'mr_tables' in _cache and 'corr_packed' in _cache and bool(np.all(lb < ub)))
        if compiled and gradient == 'exact':
            args = self._gradient_args(_cache)

            def objective_and_gradient(params):
                """Negative raw correlation (the record kernel's value) and its exact gradient."""
                rho, phi_shifted, z = params
                phi_actual = (phi_shifted + phi_shift) % 360.0
                _, g_rho, g_phi, g_z = _scalar_singleray_corr_grad_numba(rho, phi_actual, z, *args)
                f = _scalar_singleray_corr_numba(rho, phi_actual * (np.pi / 180.0), z, *args)
                return f, np.array((g_rho, g_phi, g_z))

            result = _minimize_lbfgsb(objective_and_gradient, x0, shifted_bounds, maxiter, 1e-10)
        elif compiled:
            args = (float(phi_shift), lb, ub) + self._gradient_args(_cache)

            def objective_and_differences(params):
                """Negative raw correlation and scipy's forward differences of it, in one compiled call."""
                return _lbfgsb_singleray_value_grad(params, *args)

            result = _minimize_lbfgsb(objective_and_differences, x0, shifted_bounds, maxiter, 1e-10)
        elif grouped:
            c = _cache
            args = ((float(c['pa_center'][0]), float(c['pa_center'][1]), c['ant_pos']) + c['mr_tables']
                    + (self._n_ray_slots, c['corr_packed'], c['corr_lengths'], c['corr_dts'], c['corr_offsets'],
                       c['pair_ch1'], c['pair_ch2'], c['pw'], c['ch_group'], c['group_rts'], c['group_nrt'], c['n_pairs'],
                       c['w_total'], self._mr_walk))
            shift = float(phi_shift)

            def grouped_and_differences(params):
                """Negative grouped multi-ray correlation and scipy's forward differences of it."""
                points, phi_rad = _lbfgsb_fd_points(params, shift, lb, ub)
                return _grouped_fd_value_grad(points, np.cos(phi_rad), np.sin(phi_rad), *args)

            result = _minimize_lbfgsb(grouped_and_differences, x0, shifted_bounds, maxiter, 1e-10)
        elif method == 'Nelder-Mead':
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 0.1, 'fatol': 1e-8}
            )
        else:
            result = minimize(
                objective, x0, method='L-BFGS-B', bounds=shifted_bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10}
            )

        fd_points = len(x0) + 1 if (compiled and gradient != 'exact') or grouped else 1
        self.work['optimizer_runs'] += 1
        self.work['optimizer_nit'] += int(result.nit)
        self.work['optimizer_nfev'] += int(result.nfev)
        if compiled or grouped:
            self.work['optimizer_points'] += int(result.nfev) * fd_points
            self.work['tt_lookups'] += (int(result.nfev) * fd_points * len(channels)
                                        * (self._n_ray_slots if grouped else 1))
        rho_opt, phi_shifted_opt, z_opt = result.x
        if method == 'Nelder-Mead':
            rho_opt = np.clip(rho_opt, bounds[0][0], bounds[0][1])
            z_opt = np.clip(z_opt, bounds[2][0], bounds[2][1])
        phi_opt = (phi_shifted_opt + phi_shift) % 360.0
        corr_opt = -result.fun

        return rho_opt, phi_opt, z_opt, corr_opt

    def _optimize_seeds(self, seeds, corr_data, channels, bounds, pair_weights,
                        config, _cache, objective_fn=None):
        """Optimize every seed of one stage and return the (rho, phi_deg, z, corr) results.

        With ``optimizer_method: compass`` all seeds run in one call of the numba
        compass search on the fused scalar objective; the scipy methods run
        ``_optimize_from_seed`` per seed. The compass search needs the singleray
        numba cache; multiray configurations and a custom ``objective_fn`` fall
        back to L-BFGS-B.

        Args:
            seeds: List of (rho, phi_deg, z, ...) tuples; only the first three are used.
            corr_data: Correlation functions.
            channels: Channel IDs.
            bounds: [(rho_min, rho_max), (phi_min, phi_max), (z_min, z_max)].
            pair_weights: Per-pair weights or None.
            config: Reconstruction config dict.
            _cache: Optimizer cache from _build_optimizer_cache.
            objective_fn: Function of [rho, phi_deg, z] to minimize instead of the
                negative raw correlation, or None.

        Returns:
            List of (rho, phi_deg, z, corr) in seed order, corr being minus the
            minimized value.
        """
        method = config.get('optimizer_method', 'L-BFGS-B')
        maxiter = config.get('optimizer_maxiter', 30)
        if method == 'compass' and (objective_fn is not None
                                    or not self._compass_available(_cache)):
            logger.debug("compass search needs the singleray numba cache and the raw "
                         "objective; using L-BFGS-B")
            method = 'L-BFGS-B'
        if method == 'compass':
            return self._compass_seeds(seeds, bounds, _cache, **self._compass_options(config))
        gradient = config.get('optimizer_gradient', 'finite_difference')
        return [self._optimize_from_seed(
                    (s[0], s[1] % 360.0, s[2]), corr_data, channels, bounds,
                    pair_weights, method=method, maxiter=maxiter, _cache=_cache,
                    objective_fn=objective_fn, gradient=gradient)
                for s in seeds]

    @staticmethod
    def _compass_options(config):
        """Return the compass search settings of a config as keyword arguments of ``_compass_seeds``.

        Raises:
            ValueError: If ``compass_step`` or ``compass_step_min`` is not three
                positive numbers (m, deg, m) or ``compass_max_evals`` is not a
                positive integer.
        """
        opts = {}
        for key, name in (('compass_step', 'step'), ('compass_step_min', 'step_min')):
            val = config.get(key, None)
            if val is not None and not (
                    isinstance(val, (list, tuple, np.ndarray)) and len(val) == 3
                    and all(isinstance(v, numbers.Real) and v > 0 for v in val)):
                raise ValueError(f"{key} must be three positive numbers (m, deg, m), got {val!r}")
            opts[name] = val
        max_evals = config.get('compass_max_evals', 1500)
        if isinstance(max_evals, bool) or not isinstance(max_evals, numbers.Integral) or max_evals < 1:
            raise ValueError(f"compass_max_evals must be a positive integer, got {max_evals!r}")
        opts['max_evals'] = int(max_evals)
        opts['phi_scan'] = bool(config.get('compass_phi_scan', False))
        return opts

    @staticmethod
    def _compass_available(cache):
        """Whether the compass search can run on this optimizer cache."""
        return (USE_NUMBA and cache is not None and 'td_values_packed' in cache
                and 'corr_packed' in cache)

    def _gradient_args(self, cache):
        """Arguments of ``_scalar_singleray_corr_grad_numba`` after the coordinates, from a singleray optimizer cache."""
        return (cache['pa_x'], cache['pa_y'], cache['ant_pos'],
                cache['td_values_packed'], cache['td_ok'], cache['td_slot'],
                cache['td_r_min'], cache['td_dr_inv'], cache['td_nr'],
                cache['td_z_min'], cache['td_dz_inv'], cache['td_nz'],
                cache['corr_packed'], cache['corr_lengths'],
                cache['corr_dts'], cache['corr_offsets'],
                cache['pair_ch1'], cache['pair_ch2'], cache['pw'], cache['w_total'],
                self._valid_norm, self._valid_floor)

    def _compass_seeds(self, seeds, bounds, cache, step=None, step_min=None,
                       max_evals=1500, phi_scan=False):
        """Run the numba gradient-and-compass search from every seed on the fused scalar objective.

        Args:
            seeds: List of (rho, phi_deg, z, ...) tuples.
            bounds: [(rho_min, rho_max), (phi_min, phi_max), (z_min, z_max)].
            cache: Singleray optimizer cache with the packed correlations.
            step: Initial steps (m, deg, m); default [1, 0.2, 1].
            step_min: Stopping steps (m, deg, m); default [1e-8, 2e-9, 1e-8].
            max_evals: Objective evaluations per seed.
            phi_scan: Precede the search with the +/- 1.5 deg azimuth line scan.

        Returns:
            List of (rho, phi_deg, z, corr) in seed order.
        """
        seed_arr = np.array([[s[0], s[1], s[2]] for s in seeds], dtype=np.float64)
        out = _compass_search_numba(
            seed_arr, float(bounds[0][0]), float(bounds[0][1]),
            float(bounds[2][0]), float(bounds[2][1]),
            np.array(step if step is not None else [1.0, 0.2, 1.0], dtype=np.float64),
            np.array(step_min if step_min is not None else [1e-8, 2e-9, 1e-8],
                     dtype=np.float64),
            int(max_evals), bool(phi_scan),
            cache['pa_x'], cache['pa_y'], cache['ant_pos'],
            cache['td_values_packed'], cache['td_ok'], cache['td_slot'],
            cache['td_r_min'], cache['td_dr_inv'], cache['td_nr'],
            cache['td_z_min'], cache['td_dz_inv'], cache['td_nz'],
            cache['corr_packed'], cache['corr_lengths'],
            cache['corr_dts'], cache['corr_offsets'],
            cache['pair_ch1'], cache['pair_ch2'], cache['pw'], cache['w_total'],
            self._valid_norm, self._valid_floor)
        return [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in out]
