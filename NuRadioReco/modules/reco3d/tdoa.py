"""The time-difference solver of the 3D reconstruction."""

import numpy as np
import itertools
import time

from NuRadioReco.utilities.reco3d_kernels import USE_NUMBA

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import _bilinear_scalar_numba


class TdoaMixin:
    """Methods of InterferometricReco3D for the time-difference solver."""

    def _extract_peak_delays(self, corr_data):
        """Extract peak delay from each cross-correlation function.

        Args:
            corr_data: List of (corr_array, dt, offset) per pair.

        Returns:
            1D array of peak delay values (ns) for each pair.
        """
        delays = np.empty(len(corr_data), dtype=np.float64)
        for pidx, (corr_arr, dt, offset) in enumerate(corr_data):
            peak_idx = np.argmax(corr_arr)
            delays[pidx] = offset + peak_idx * dt
        return delays

    def _chan_ho_initialize(self, corr_data, channels, pair_weights=None):
        """Estimate source position via Chan-Ho TDOA linearization.

        Assumes straight-line propagation at average ice velocity for an
        initial estimate. The bias from ray curvature is acceptable as
        a starting point for iterative refinement.

        Args:
            corr_data: Pre-computed correlation data per pair.
            channels: Channel IDs.
            pair_weights: Per-pair weights (used to select strongest pairs).

        Returns:
            (rho, phi_deg, z) initial estimate, or None if solve fails.
        """
        n_ch = len(channels)
        if n_ch < 4:
            return None

        c_ice = 0.3 / 1.55  # average in-ice velocity ~0.1935 m/ns

        ant_pos_3d = np.array([self.ant_locs[ch] for ch in channels])
        pa_center = self._pa_center

        # Extract per-channel delays relative to first channel using
        # pairwise correlations. Build a per-channel TDOA vector.
        pair_delays = self._extract_peak_delays(corr_data)
        ch_pairs = list(itertools.combinations(range(n_ch), 2))

        # Use weighted average of pairwise delays to get per-channel TDOAs
        # relative to channel 0
        tdoa = np.zeros(n_ch)
        tdoa_count = np.zeros(n_ch)

        if pair_weights is not None:
            pw = np.asarray(pair_weights, dtype=np.float64)
        else:
            pw = np.ones(len(ch_pairs))

        for pidx, (ci, cj) in enumerate(ch_pairs):
            w = pw[pidx]
            # delay = t_ci - t_cj (positive means ci signal arrives later)
            delay = pair_delays[pidx]
            if ci == 0:
                tdoa[cj] += -delay * w
                tdoa_count[cj] += w
            elif cj == 0:
                tdoa[ci] += delay * w
                tdoa_count[ci] += w

        # For channels without direct pair to ch0, use transitive delays
        for ci in range(1, n_ch):
            if tdoa_count[ci] == 0:
                for pidx, (ca, cb) in enumerate(ch_pairs):
                    w = pw[pidx]
                    if ca == ci and tdoa_count[cb] > 0:
                        tdoa[ci] += (pair_delays[pidx] +
                                     tdoa[cb] / tdoa_count[cb]) * w
                        tdoa_count[ci] += w
                    elif cb == ci and tdoa_count[ca] > 0:
                        tdoa[ci] += (-pair_delays[pidx] +
                                     tdoa[ca] / tdoa_count[ca]) * w
                        tdoa_count[ci] += w

        for ci in range(1, n_ch):
            if tdoa_count[ci] > 0:
                tdoa[ci] /= tdoa_count[ci]

        # Chan-Ho linear system: A * [x, y, z, d0] = b
        # Reference receiver is channel 0
        r0 = ant_pos_3d[0]
        r0_sq = np.dot(r0, r0)

        n_eq = n_ch - 1
        A = np.zeros((n_eq, 4))
        b = np.zeros(n_eq)

        for i in range(n_eq):
            ri = ant_pos_3d[i + 1]
            di0 = c_ice * tdoa[i + 1]  # range difference in meters

            A[i, :3] = 2.0 * (ri - r0)
            A[i, 3] = 2.0 * di0
            b[i] = r0_sq - np.dot(ri, ri) - di0 * di0

        # Weighted least squares
        try:
            result, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        x, y, z, d0 = result

        # Convert to cylindrical relative to PA center
        dx = x - pa_center[0]
        dy = y - pa_center[1]
        rho = max(np.sqrt(dx**2 + dy**2), 1.0)
        phi_deg = np.degrees(np.arctan2(dy, dx)) % 360.0

        rho = np.clip(rho, 1.0, 1500.0)
        z = np.clip(z, -1500.0, 0.0)

        return rho, phi_deg, z

    def _tdoa_solve(self, corr_data, channels, pair_weights=None,
                    initial_guess=None, cache=None):
        """Solve for source position using TDOA least-squares.

        Extracts observed time delays from cross-correlation peaks, then
        finds the position that minimizes weighted delay residuals.

        Args:
            corr_data: Pre-computed correlation data per pair.
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.
            initial_guess: (rho, phi_deg, z) starting point, or None.
            cache: Pre-computed cache from _build_optimizer_cache.

        Returns:
            (rho, phi_deg, z, residual) or None if solve fails.
        """
        from scipy.optimize import least_squares

        observed_delays = self._extract_peak_delays(corr_data)
        n_pairs = len(corr_data)

        if cache is not None:
            pa_center = cache['pa_center']
            ant_pos = cache['ant_pos']
            pair_ch1 = cache['pair_ch1']
            pair_ch2 = cache['pair_ch2']
            td_list = cache['td_list']
            n_ch = cache['n_ch']
            pw = cache['pw']
        else:
            pa_center = self._pa_center
            ant_pos = np.array([self.ant_locs[ch][:2] for ch in channels])
            ch_idx = list(range(len(channels)))
            pair_ch1 = np.array([p[0] for p in
                                 itertools.combinations(ch_idx, 2)],
                                dtype=np.int64)
            pair_ch2 = np.array([p[1] for p in
                                 itertools.combinations(ch_idx, 2)],
                                dtype=np.int64)
            n_ch = len(channels)
            td_list = []
            for ch in channels:
                ch_tds = [self._multiray_interpolators[ch][rt]
                          for rt in self._active_ray_types]
                td_list.append(ch_tds)
            pw = np.ones(n_pairs, dtype=np.float64)
            if pair_weights is not None:
                pw = np.asarray(pair_weights, dtype=np.float64)

        sqrt_weights = np.sqrt(pw)

        if initial_guess is None:
            ch_init = self._chan_ho_initialize(
                corr_data, channels, pair_weights)
            if ch_init is not None:
                initial_guess = ch_init
            else:
                initial_guess = (500.0, 180.0, -500.0)
        rho0, phi0, z0 = initial_guess

        best_result = None
        best_cost = np.inf

        if cache is not None and 'combo_table' in cache:
            combo_table = cache['combo_table']
            n_combos = cache['n_combos']
        else:
            combo_table = np.zeros((1, n_ch), dtype=np.int64)
            n_combos = 1

        for ci in range(n_combos):
            rt_indices = combo_table[ci]

            def residuals(params):
                rho, phi_deg, z = params
                phi_rad = phi_deg * (np.pi / 180.0)
                x = rho * np.cos(phi_rad) + pa_center[0]
                y = rho * np.sin(phi_rad) + pa_center[1]

                tt = np.full(n_ch, np.nan)
                for chi in range(n_ch):
                    dx = x - ant_pos[chi, 0]
                    dy = y - ant_pos[chi, 1]
                    r = max(np.sqrt(dx * dx + dy * dy), 1.0)
                    rti = int(rt_indices[chi])
                    td = td_list[chi][rti]
                    if USE_NUMBA:
                        val = _bilinear_scalar_numba(
                            td.values, td.r_min, td.dr_inv, td.nr,
                            td.z_min, td.dz_inv, td.nz, r, z)
                    else:
                        val = td.interp(np.array([[r, z]]))[0]
                    if np.isfinite(val) and val > 0:
                        tt[chi] = val

                resid = np.zeros(n_pairs)
                for pidx in range(n_pairs):
                    c1, c2 = int(pair_ch1[pidx]), int(pair_ch2[pidx])
                    if np.isfinite(tt[c1]) and np.isfinite(tt[c2]):
                        predicted = tt[c1] - tt[c2]
                        resid[pidx] = (observed_delays[pidx] - predicted) \
                            * sqrt_weights[pidx]
                    else:
                        resid[pidx] = 0.0
                return resid

            try:
                result = least_squares(
                    residuals, [rho0, phi0, z0],
                    bounds=([1.0, -180.0, -1500.0],
                            [1500.0, 540.0, 0.0]),
                    method='trf', max_nfev=50,
                    ftol=1e-6, xtol=1e-4,
                )
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result.x
            except Exception:
                continue

        if best_result is None:
            return None

        rho_sol, phi_sol, z_sol = best_result
        phi_sol = phi_sol % 360.0

        # Evaluate correlation at TDOA solution to get comparable metric
        corr_val = -self._correlation_at_point(
            [rho_sol, phi_sol, z_sol], corr_data, channels, pair_weights,
            _cache=cache
        )

        return rho_sol, phi_sol, z_sol, corr_val

    def run_tdoa(self, evt, station, det, config):
        """TDOA-based 3D reco: Chan-Ho initialization + iterative refinement + optimizer.

        Bypasses the grid search entirely. Uses cross-correlation peak delays
        to estimate source position via TDOA least-squares, then refines with
        L-BFGS-B on the full correlation objective.

        Args:
            evt: NuRadioReco Event object.
            station: Station object containing channel data.
            det: Detector description.
            config: Configuration dictionary.

        Returns:
            Reconstruction results dict, or None on failure.
        """
        channels = config['channels']
        hilbert_mode = config.get('hilbert_envelope_mode', None)
        apply_hann = config.get('apply_hann_window', False)
        corr_norm = config.get('correlation_normalization', 'normalized')

        volt_arrays = []
        time_arrays = []
        for ch in channels:
            channel = station.get_channel(ch)
            volt_arrays.append(channel.get_trace())
            time_arrays.append(channel.get_times())

        pair_weights = None
        if config.get('snr_pair_weighting', False):
            pair_weights, _ = self._compute_snr_pair_weights(
                volt_arrays, channels, config.get('noise_rms_summation', 'sequential')
            )

        corr_data, _ = self._prepare_corr_funcs(
            time_arrays, volt_arrays,
            hilbert_envelope_mode=hilbert_mode,
            apply_hann_window=apply_hann,
            correlation_normalization=corr_norm,
        )

        full_limits = config.get('limits', [1, 1500, 0, 360, -1500, 0])
        opt_method = config.get('optimizer_method', 'L-BFGS-B')
        opt_maxiter = config.get('optimizer_maxiter', 30)

        t0 = time.time()
        opt_cache = self._build_optimizer_cache(channels, pair_weights,
                                                corr_data)

        # TDOA solve (Chan-Ho init + iterative refinement)
        tdoa_result = self._tdoa_solve(
            corr_data, channels, pair_weights, cache=opt_cache
        )
        t_tdoa = time.time() - t0

        if tdoa_result is None:
            return None

        rho_tdoa, phi_tdoa, z_tdoa, corr_tdoa = tdoa_result

        t0_opt = time.time()
        bounds = [
            (max(full_limits[0], 1.0), full_limits[1]),
            (full_limits[2], full_limits[3]),
            (full_limits[4], full_limits[5]),
        ]
        rho_best, phi_best, z_best, corr_best = self._optimize_from_seed(
            (rho_tdoa, phi_tdoa, z_tdoa), corr_data, channels, bounds,
            pair_weights, method=opt_method, maxiter=opt_maxiter,
            _cache=opt_cache, config=config
        )
        t_opt = time.time() - t0_opt

        phi_best = phi_best % 360.0

        self._set_station_parameters(
            station, rho_best, phi_best, z_best, corr_best)

        return {
            'rho': rho_best,
            'phi': phi_best,
            'z': z_best,
            'max_corr': corr_best,
            'tdoa_time': t_tdoa,
            'opt_time': t_opt,
            'tdoa_rho': rho_tdoa,
            'tdoa_phi': phi_tdoa,
            'tdoa_z': z_tdoa,
            'tdoa_corr': corr_tdoa,
        }
