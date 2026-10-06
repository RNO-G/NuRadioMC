"""The hierarchical search of the 3D reconstruction."""

import numpy as np
import time

from NuRadioReco.utilities.reco3d_kernels import (
    USE_NUMBA,
    USE_CUPY,
    _FUSED_CORR_KERNEL,
    _build_split_z_window,
)

from NuRadioReco.modules.reco3d.shared import logger, _CANDIDATE_ORIGIN_CODES

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import _numpy_std


_CANDIDATE_POOL_SAVE = 8
_ADAPTIVE_WINDOW_CELL_FRACTION = 0.6


def _adjacent_cell_span(vec, value):
    """Return the width of the wider of the two grid cells adjacent to a position.

    Args:
        vec: Ascending grid vector.
        value: Position on the axis (on or between grid points).

    Returns:
        The larger of the cell ending at or containing ``value`` and the next
        cell up (the lower cell alone at the top edge), as a float.
    """
    i = int(np.clip(np.searchsorted(vec, value), 1, len(vec) - 1))
    upper = vec[min(i + 1, len(vec) - 1)] - vec[i]
    return float(max(vec[i] - vec[i - 1], upper))


class HierarchicalMixin:
    """Methods of InterferometricReco3D for the hierarchical search."""

    def _build_adaptive_refine_levels(self, config, n_refinements):
        """Build refine_levels dict from n_refinements + factor + bins.

        Each level is an adaptive dict that the refine loop evaluates
        per-peak using the local coarse grid spacing. Returned levels
        have 'adaptive': True, 'factor', 'window_bins', 'n_peaks'.
        The actual window and step are computed per-peak at refine time.

        Args:
            config: Dict with refinement_factor, refinement_window_bins.
            n_refinements: Int, number of refinement levels.

        Returns:
            List of dicts, one per level.
        """
        factor = config.get('refinement_factor', 4)
        window_bins = config.get('refinement_window_bins', 2)
        n_peaks_per_level = config.get('n_optimizer_seeds', 3)
        levels = []
        for k in range(n_refinements):
            levels.append({
                'adaptive': True,
                'factor': factor,
                'window_bins': window_bins,
                'n_peaks': n_peaks_per_level,
            })
        return levels

    def _local_coarse_spacing(self, rho_vec_c, phi_vec_c, z_vec_c,
                              rho_p, z_p, n_refinements_so_far=0, factor=4):
        """Compute local coarse grid spacing at (rho_p, z_p).

        For log-spaced axes, the spacing at a given location is the
        difference between the nearest two grid points. For uniform
        spacing it's just a constant. For subsequent refine levels,
        the "coarse" step shrinks by factor^n_refinements_so_far
        (since each previous level divided the step by factor).

        Args:
            rho_vec_c, phi_vec_c, z_vec_c: Coarse grid arrays (rho/z
                absolute, phi in radians).
            rho_p, z_p: Peak coordinates.
            n_refinements_so_far: Refinement level index (0-based). The
                "effective coarse spacing" at level k is divided by
                factor^k since previous levels already narrowed in.
            factor: Refinement factor.

        Returns:
            (drho, dphi_deg, dz) tuple, each a positive scalar.
        """
        i_rho = int(np.clip(np.searchsorted(rho_vec_c, rho_p),
                            1, len(rho_vec_c) - 1))
        drho = float(rho_vec_c[i_rho] - rho_vec_c[i_rho - 1])
        # z_vec_c is sorted ascending (negative to 0 with log spacing)
        i_z = int(np.clip(np.searchsorted(z_vec_c, z_p),
                          1, len(z_vec_c) - 1))
        dz = float(z_vec_c[i_z] - z_vec_c[i_z - 1])
        dphi_rad = float(phi_vec_c[1] - phi_vec_c[0])
        dphi_deg = dphi_rad * (180.0 / np.pi)
        # Account for previous refinements already narrowing the step
        scale = 1.0 / (factor ** n_refinements_so_far)
        return drho * scale, dphi_deg * scale, dz * scale

    def _search_chain(self, corr_data, channels, pair_weights, config,
                      coarse_grid, full_limits, packed=None, coarse_map=None,
                      subbin_seeds=False):
        """Run the coarse map, peak extraction, refine levels and optimizer for one set of correlation functions.

        With ``subbin_seeds`` the refine grids are centred on the sub-bin
        estimate of each coarse peak (``_subbin_peak``); the caller sets it
        from ``subbin_coarse_seeds`` for a chain whose objective is an
        envelope and never for a raw chain, whose coarse map has lobes
        narrower than a bin. With ``refine_window_mode`` "adaptive" the first
        refine level's half-window in rho and z is at least
        ``_ADAPTIVE_WINDOW_CELL_FRACTION`` times the wider of the two coarse
        cells adjacent to the peak, so the grid covers at least half of
        either neighbouring cell, while the phi half-window and the steps
        stay as configured.

        Args:
            corr_data: Correlation functions from _prepare_corr_funcs.
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.
            config: Reconstruction config dict.
            coarse_grid: (rho_vec, phi_vec_rad, z_vec, delay_data, cache_key) of the coarse
                grid; delay_data is the cached travel-time stack on the fused singleray
                path, the per-pair delay matrices otherwise.
            full_limits: [rho_min, rho_max, phi_min, phi_max, z_min, z_max] search limits.
            packed: CorrPacked of corr_data for the fused kernels, or None to pack per call.
            coarse_map: Coarse map already computed for corr_data (shared geometry pass), or None.
            subbin_seeds: Centre the refine grids on the sub-bin estimates of the coarse peaks.

        Returns:
            Dict with 'mean_corr_c' (coarse map), 'coarse_peaks' and 't_coarse'. When
            coarse peaks exist it also holds 'refined_peaks' (the optimizer seeds),
            'unseeded_peaks' (refined peaks beyond n_optimizer_seeds), 'all_optimized'
            (optimizer outputs sorted by correlation and deduplicated), 'opt_cache' and
            'bounds' (None when the optimizer is skipped), 't_refine' and 't_opt'. Under
            the split z grid the stages after the coarse map run once per z block
            (_z_blocks) on that block's slice of the coarse map, the peak lists are
            merged by correlation (all_optimized deduplicated again), 'bounds' are the
            full limits and 'blocks' holds the per-block results with their 'limits'
            and 'z_above'.
        """
        rho_vec_c, phi_vec_c, z_vec_c, delay_data_c, coarse_cache_key = coarse_grid

        t0 = time.time()
        if coarse_map is not None:
            mean_corr_c = coarse_map
        elif self._multi_ray_types and delay_data_c is None:
            mean_corr_c, _ = self._multiray_grid(
                rho_vec_c, phi_vec_c, z_vec_c, corr_data, channels, pair_weights, force_perpair=True)
        elif self._multi_ray_types:
            mean_corr_c, _ = self._multiray_correlate(
                corr_data, delay_data_c, channels,
                pair_weights=pair_weights, force_perpair=True
            )
        elif self._singleray_kernel_active():
            if packed is None:
                packed = self._pack_corr_data(corr_data)
            mean_corr_c = self._singleray_stack_maps(
                delay_data_c, channels, [packed], pair_weights)[0].reshape(
                len(rho_vec_c), len(phi_vec_c), len(z_vec_c))
        else:
            mean_corr_c, _ = self._correlator_lean(
                corr_data, delay_data_c, pair_weights=pair_weights,
                delay_cache_key=coarse_cache_key, packed=packed,
            )
        t_coarse = time.time() - t0
        if coarse_map is None:
            self.work['coarse_maps'] += 1
            self.work['coarse_points'] += len(rho_vec_c) * len(phi_vec_c) * len(z_vec_c)

        blocks = self._z_blocks(config, z_vec_c, full_limits)
        if blocks is None:
            stages = self._refine_and_optimize(
                mean_corr_c, rho_vec_c, phi_vec_c, z_vec_c, corr_data, channels,
                pair_weights, config, full_limits, packed=packed,
                subbin_seeds=subbin_seeds)
            stages['mean_corr_c'] = mean_corr_c
            stages['t_coarse'] = t_coarse
            return stages

        per_block = []
        for z_slice, limits, z_above in blocks:
            stages = self._refine_and_optimize(
                mean_corr_c[:, :, z_slice], rho_vec_c, phi_vec_c, z_vec_c[z_slice],
                corr_data, channels, pair_weights, config, limits, packed=packed,
                subbin_seeds=subbin_seeds, z_above=z_above)
            stages['limits'] = limits
            stages['z_above'] = z_above
            per_block.append(stages)

        def merged(key):
            """Return the peaks of every block under key, sorted by correlation."""
            peaks = [p for stages in per_block for p in stages.get(key, [])]
            peaks.sort(key=lambda x: x[3], reverse=True)
            return peaks

        result = {'mean_corr_c': mean_corr_c, 'coarse_peaks': merged('coarse_peaks'),
                  't_coarse': t_coarse, 'blocks': per_block}
        searched = [stages for stages in per_block if 'refined_peaks' in stages]
        if not searched:
            return result
        all_optimized = merged('all_optimized')
        result.update({
            'refined_peaks': merged('refined_peaks'),
            'unseeded_peaks': merged('unseeded_peaks'),
            'all_optimized': self._deduplicate_peaks(all_optimized) if all_optimized else [],
            'opt_cache': next((st['opt_cache'] for st in searched if st['opt_cache'] is not None), None),
            'bounds': None if searched[0]['bounds'] is None else [
                (max(full_limits[0], 1.0), full_limits[1]),
                (full_limits[2], full_limits[3]),
                (full_limits[4], full_limits[5]),
            ],
            't_refine': sum(st['t_refine'] for st in searched),
            't_opt': sum(st['t_opt'] for st in searched),
        })
        return result

    def _z_blocks(self, config, z_vec_c, full_limits):
        """Return the z blocks searched separately under the split z grid, or None without it.

        With ``z_grid_below`` and ``z_grid_above`` the coarse grid holds an in-ice block
        (z <= 0) and an air block (z > 0). Each block runs its own peak extraction,
        refine levels and optimizer within its own limits, so the in-ice block makes the
        same choices as an in-ice search volume ending at z = 0 and the air block cannot
        take its peak or seed slots; the results are merged by correlation afterwards.

        Args:
            config: Reconstruction config dict.
            z_vec_c: Coarse z vector, ascending.
            full_limits: [rho_min, rho_max, phi_min, phi_max, z_min, z_max] search limits.

        Returns:
            None, or a list of (z_slice, limits, z_above) for the non-empty blocks, where
            z_slice selects the block's coarse z nodes, limits are the search limits with
            z clamped to the block (z_max at most 0 for the ice, z_min at least 0 for the
            air) and z_above is None for the ice and the air block dict for the air.
        """
        split = self._split_z_grid(config)
        if split is None:
            return None
        n_ice = int(np.count_nonzero(z_vec_c <= 0))
        blocks = []
        if n_ice > 0:
            ice = list(full_limits)
            ice[5] = min(full_limits[5], 0.0)
            blocks.append((slice(0, n_ice), ice, None))
        if n_ice < len(z_vec_c):
            air = list(full_limits)
            air[4] = max(full_limits[4], 0.0)
            blocks.append((slice(n_ice, None), air, split[1]))
        return blocks

    def _refine_and_optimize(self, mean_corr_c, rho_vec_c, phi_vec_c, z_vec_c, corr_data,
                             channels, pair_weights, config, full_limits, packed=None,
                             subbin_seeds=False, z_above=None):
        """Extract coarse peaks from a coarse map, refine them and run the optimizer.

        Args:
            mean_corr_c: Coarse correlation map (n_rho, n_phi, n_z).
            rho_vec_c: Coarse rho values in m.
            phi_vec_c: Coarse phi values in rad.
            z_vec_c: Coarse z values in m.
            corr_data: Correlation functions from _prepare_corr_funcs.
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.
            config: Reconstruction config dict.
            full_limits: Search limits that clamp the refine windows and bound the optimizer.
            packed: CorrPacked of corr_data for the fused kernels, or None.
            subbin_seeds: Centre the refine grids on the sub-bin estimates of the coarse peaks.
            z_above: Air block of the split z grid, which places the z vector of the
                refine windows (_build_split_z_window), or None for plain linear windows.

        Returns:
            Dict with 'coarse_peaks' and, when coarse peaks exist, 'refined_peaks',
            'unseeded_peaks', 'all_optimized', 'opt_cache', 'bounds', 't_refine' and
            't_opt' as described in _search_chain.
        """
        n_coarse_peaks = config.get('coarse_n_peaks', 3)
        coarse_sep = config.get('coarse_peak_separation', [60, 15, 60])
        phi_vec_deg_c = phi_vec_c * (180.0 / np.pi)

        coarse_peaks = self._extract_top_n_peaks(
            mean_corr_c, rho_vec_c, phi_vec_deg_c, z_vec_c,
            n_coarse_peaks, coarse_sep
        )

        if not coarse_peaks:
            return {'coarse_peaks': []}

        # Adaptive refinement: if n_refinements is configured, build
        # refine_levels programmatically from the coarse grid spacing.
        # Each level reduces step by refinement_factor; window covers
        # refinement_window_bins previous-level bins on each side.
        n_refinements = config.get('n_refinements', None)
        refine_levels = config.get('refine_levels', None)

        if refine_levels is None and n_refinements is not None:
            refine_levels = self._build_adaptive_refine_levels(
                config, n_refinements)

        if refine_levels is None:
            refine_levels = [{
                'window': config.get('refine_window', [150, 20, 150]),
                'steps': config.get('refine_step_sizes', [5, 1, 5]),
                'n_peaks': config.get('coarse_n_peaks', 3),
            }]

        # Convergence early-stop (optional)
        conv_db = config.get('refinement_convergence_db', None)
        n_refinements_max = config.get('n_refinements_max', len(refine_levels))

        t0_ref = time.time()
        current_peaks = coarse_peaks
        if subbin_seeds:
            current_peaks = [
                self._subbin_peak(mean_corr_c, rho_vec_c, phi_vec_deg_c,
                                  z_vec_c, p) for p in coarse_peaks]
        adaptive_window = (config.get('refine_window_mode', 'fixed')
                           == 'adaptive')

        for level_idx, level in enumerate(refine_levels):
            level_window = level.get('window', None)
            level_steps = level.get('steps', None)
            level_adaptive = level.get('adaptive', False)
            level_factor = level.get('factor', 4)
            level_bins = level.get('window_bins', 2)
            level_n_peaks = level.get('n_peaks', 3)
            level_sep = level.get('peak_separation',
                                  config.get('peak_separation_threshold',
                                             [10, 5, 10]))

            peak_grids = []
            prev_level_peaks_by_input = list(current_peaks)
            for rho_p, phi_p, z_p, corr_p in current_peaks:
                if level_adaptive:
                    local_drho, local_dphi, local_dz = self._local_coarse_spacing(
                        rho_vec_c, phi_vec_c, z_vec_c, rho_p, z_p,
                        n_refinements_so_far=level_idx,
                        factor=level_factor)
                    win_rho = level_bins * local_drho
                    win_phi_deg = level_bins * local_dphi
                    win_z = level_bins * local_dz
                    step_rho = local_drho / level_factor
                    step_phi_deg = local_dphi / level_factor
                    step_z = local_dz / level_factor
                else:
                    win_rho, win_phi_deg, win_z = level_window
                    step_rho, step_phi_deg, step_z = level_steps
                    if adaptive_window and level_idx == 0:
                        win_rho = max(win_rho, _ADAPTIVE_WINDOW_CELL_FRACTION
                                      * _adjacent_cell_span(rho_vec_c, rho_p))
                        win_z = max(win_z, _ADAPTIVE_WINDOW_CELL_FRACTION
                                    * _adjacent_cell_span(z_vec_c, z_p))

                rho_lo = max(max(full_limits[0], 1.0), rho_p - win_rho)
                rho_hi = min(full_limits[1], rho_p + win_rho)
                phi_lo = phi_p - win_phi_deg
                phi_hi = phi_p + win_phi_deg
                z_lo = max(full_limits[4], z_p - win_z)
                z_hi = min(full_limits[5], z_p + win_z)

                rho_vec_r = np.arange(max(rho_lo, 1.0),
                                      rho_hi + step_rho, step_rho)
                phi_vec_r = np.arange(phi_lo, phi_hi + step_phi_deg,
                                      step_phi_deg) * (np.pi / 180.0)
                if z_above is None:
                    z_vec_r = np.arange(z_lo, z_hi + step_z, step_z)
                else:
                    z_vec_r = _build_split_z_window(z_lo, z_hi, step_z, z_above)

                if (len(rho_vec_r) == 0 or len(phi_vec_r) == 0
                        or len(z_vec_r) == 0):
                    continue

                # the fused singleray kernel computes the source positions itself
                src_enu_r = None if self._singleray_kernel_active() else self._build_source_enu_matrix(
                    rho_vec_r, phi_vec_r, z_vec_r)
                peak_grids.append((src_enu_r, rho_vec_r, phi_vec_r, z_vec_r,
                                   step_rho, step_phi_deg, step_z))

            if not peak_grids:
                current_peaks = current_peaks[:level_n_peaks]
                continue

            # Downstream dispatchers expect 4-tuples; strip the step info.
            peak_grids_4 = [(pg[0], pg[1], pg[2], pg[3])
                            for pg in peak_grids]
            self.work['refine_grids'] += len(peak_grids_4)
            self.work['refine_points'] += sum(len(g[1]) * len(g[2]) * len(g[3]) for g in peak_grids_4)

            n_extract = config.get('n_optimizer_seeds', 3)
            use_batched_gpu = (
                self._use_gpu and USE_CUPY
                and not self._multi_ray_types
                and len(peak_grids_4) > 1
                and _FUSED_CORR_KERNEL is not None
            )
            # The fused kernel takes each pair's own best ray-type combination
            # (per_pair); grouped refines through _multiray_correlate.
            use_fused_multiray = (
                self._multi_ray_types
                and self._multiray_combo_mode != 'grouped'
                and USE_NUMBA
                and hasattr(self, '_multiray_interpolators')
                and self._use_fused_correlator
            )

            if use_batched_gpu:
                level_peaks = self._refine_batched_gpu(
                    peak_grids_4, corr_data, channels,
                    pair_weights, n_extract, level_sep)
            elif use_fused_multiray:
                level_peaks = self._fused_multiray_refine(
                    peak_grids_4, corr_data, channels,
                    pair_weights, n_extract, level_sep)
            else:
                level_peaks = []
                for src_enu_r, rho_vec_r, phi_vec_r, z_vec_r in peak_grids_4:
                    if self._multi_ray_types:
                        mean_corr_r, _ = self._multiray_grid(
                            rho_vec_r, phi_vec_r, z_vec_r, corr_data, channels,
                            pair_weights, src_enu=src_enu_r)
                    elif self._singleray_kernel_active():
                        mean_corr_r = self._singleray_grid_maps(
                            rho_vec_r, phi_vec_r, z_vec_r, channels,
                            [packed], pair_weights)[0]
                    else:
                        delay_data_r = self._compute_delay_matrices(
                            src_enu_r, channels)
                        mean_corr_r, _ = self._correlator_lean(
                            corr_data, delay_data_r,
                            pair_weights=pair_weights, packed=packed)
                    phi_vec_deg_r = phi_vec_r * (180.0 / np.pi)
                    local_peaks = self._extract_top_n_peaks(
                        mean_corr_r, rho_vec_r, phi_vec_deg_r, z_vec_r,
                        n_extract, level_sep)
                    level_peaks.extend(local_peaks)

            if not level_peaks:
                level_peaks = list(current_peaks)

            level_peaks.sort(key=lambda x: x[3], reverse=True)
            new_peaks = level_peaks[:level_n_peaks]

            # Position-convergence early-stop (optional, dB-based).
            # Compute delta/step ratio per axis for the best peak at this
            # level vs its best parent from the previous level. Stop if
            # all three axes are below the threshold.
            if (conv_db is not None and len(new_peaks) > 0
                    and len(prev_level_peaks_by_input) > 0
                    and level_adaptive):
                new_best = new_peaks[0]
                # Nearest parent by position (proxy for "which level-prev
                # peak did this refine peak come from")
                parent = min(
                    prev_level_peaks_by_input,
                    key=lambda p: ((new_best[0] - p[0]) ** 2
                                   + (new_best[2] - p[2]) ** 2))
                # Step sizes from first peak_grids entry (all peaks share
                # the same factor/bins structure so step scales are ~same)
                _, _, _, _, s_rho, s_phi, s_z = peak_grids[0]
                d_rho = abs(new_best[0] - parent[0])
                d_phi = abs(new_best[1] - parent[1])
                d_phi = min(d_phi, 360.0 - d_phi)  # wrap
                d_z = abs(new_best[2] - parent[2])
                eps = 1e-12
                rho_db = 20.0 * np.log10(max(d_rho / max(s_rho, eps), eps))
                phi_db = 20.0 * np.log10(max(d_phi / max(s_phi, eps), eps))
                z_db = 20.0 * np.log10(max(d_z / max(s_z, eps), eps))
                if (rho_db < conv_db and phi_db < conv_db
                        and z_db < conv_db):
                    current_peaks = new_peaks
                    break  # converged, stop refinement

            current_peaks = new_peaks

            # Hard cap enforcement
            if level_idx + 1 >= n_refinements_max:
                break

        t_refine = time.time() - t0_ref
        refined_peaks = current_peaks

        n_seeds = config.get('n_optimizer_seeds', 3)
        refined_peaks.sort(key=lambda x: x[3], reverse=True)
        unseeded_peaks = refined_peaks[n_seeds:]
        refined_peaks = refined_peaks[:n_seeds]

        # Stage 3: optimizer (optional)
        skip_optimizer = config.get('skip_optimizer', False)
        use_tdoa = config.get('use_tdoa_seed', False)
        t0_opt = time.time()

        opt_cache = None
        bounds = None
        if skip_optimizer:
            all_optimized = [(p[0], p[1] % 360.0, p[2], p[3])
                             for p in refined_peaks]
        else:
            bounds = [
                (max(full_limits[0], 1.0), full_limits[1]),
                (full_limits[2], full_limits[3]),
                (full_limits[4], full_limits[5]),
            ]
            opt_cache = self._build_optimizer_cache(channels, pair_weights,
                                                       corr_data, packed=packed)

            # Add TDOA seed from top coarse peak
            if use_tdoa and self._multi_ray_types:
                top_peak = refined_peaks[0]
                tdoa_result = self._tdoa_solve(
                    corr_data, channels, pair_weights,
                    initial_guess=(top_peak[0], top_peak[1], top_peak[2]),
                    cache=opt_cache,
                )
                if tdoa_result is not None:
                    refined_peaks.append(tdoa_result)

            # Add rho-perturbed seeds to explore rho space
            rho_offsets = config.get('optimizer_rho_offsets', None)
            if rho_offsets:
                top = refined_peaks[0]
                for frac in rho_offsets:
                    rho_seed = max(1.0, top[0] * (1 + frac))
                    refined_peaks.append(
                        (rho_seed, top[1], top[2], top[3] * 0.99)
                    )

            all_optimized = [
                (rho_opt, phi_opt % 360.0, z_opt, corr_opt)
                for rho_opt, phi_opt, z_opt, corr_opt in self._optimize_seeds(
                    refined_peaks, corr_data, channels, bounds, pair_weights,
                    config, opt_cache)]

            all_optimized.sort(key=lambda x: x[3], reverse=True)
            all_optimized = self._deduplicate_peaks(all_optimized)

        t_opt = time.time() - t0_opt

        return {
            'coarse_peaks': coarse_peaks,
            'refined_peaks': refined_peaks,
            'unseeded_peaks': unseeded_peaks,
            'all_optimized': all_optimized,
            'opt_cache': opt_cache,
            'bounds': bounds,
            't_refine': t_refine,
            't_opt': t_opt,
        }

    def run_hierarchical(self, evt, station, det, config):
        """Hierarchical 3D reco: coarse log-grid + refined linear grid + optimizer.

        Stage 1 uses a logarithmic rho grid (finer resolution at close range)
        with linear phi and z. Stage 2 refines around the top coarse peaks with
        a local linear grid. Stage 3 runs L-BFGS-B from the refined peaks.

        The split z grid (``_coarse_z_grid``, ``_z_blocks``), the candidate
        search with its polish, tie band and saved-peak fill
        (``_candidate_chains``, ``_polish_candidates``, ``_candidate_tie_band``)
        and the two-arrival polish objective (``_two_arrival_settings``) are
        described, with every result field they add, in the Coarse z grid,
        Candidate search and Two-arrival polish objective sections of
        ``INTERFEROMETRIC_RECONSTRUCTION_README.md``.

        Parameters
        ----------
        evt : Event
            NuRadioReco Event object.
        station : Station
            Station object containing channel data.
        det : Detector
            Detector description.
        config : dict
            Configuration dictionary (already loaded).

        Returns
        -------
        dict
            Reconstruction results.
        """
        return self._search_hierarchical(self.compute_pairs(station, config), config,
                                         station=station)

    def _search_hierarchical(self, pairs, config, station=None, pair_weights=None,
                             channel_delay_shift=None, channel_polarity=None):
        """Search stage of ``run_hierarchical`` on the pair series of one channel group.

        Args:
            pairs: PairSet holding every pair of ``config['channels']``.
            config: Reconstruction config dict of the group.
            station: Station that receives the result parameters and supplies the
                traces of ``save_coherent_waveforms``, or None.
            pair_weights: Optional override of the pair weights (see
                ``reconstruct_from_pairs``).
            channel_delay_shift: Optional per-channel delay shifts in ns.
            channel_polarity: Optional per-channel polarities (+1 or -1).

        Returns:
            Result dict as described in ``run_hierarchical``.
        """
        station_id = self._station_id
        channels = config['channels']
        hilbert_mode = config.get('hilbert_envelope_mode', None)
        pair_weights, channel_snrs, windowed_snrs, series = self._group_inputs(
            pairs, channels, config, pair_weights, channel_delay_shift, channel_polarity)
        self.work['searches'] += 1
        self.work['pairs'] += len(pair_weights)

        candidate_search = self._candidate_chains(config)
        tie_band = self._candidate_tie_band(config)
        two_arrival = self._two_arrival_settings(config)
        if two_arrival is not None and not self._two_arrival_interpolators:
            raise RuntimeError(
                "polish_objective two_arrival_consistent needs the solution-ordered "
                "tables, which are loaded in begin() when the key is set there")
        tie_ceiling = self._candidate_tie_band_max_raw_corr(config)
        region_on = config.get('region_hypotheses', False)
        corr_data = None
        packed = None
        if not candidate_search:
            corr_data, packed = series(hilbert_mode)

        # Stage 1: Coarse scan with log rho grid
        coarse_limits = config.get('coarse_limits', [1, 1500, 0, 360, -1500, 0])
        coarse_steps = config.get('coarse_step_sizes', [30, 5, 30])
        n_rho_coarse = config.get('coarse_n_rho', 50)

        rho_min_c = max(coarse_limits[0], 1.0)
        rho_max_c = coarse_limits[1]
        phi_min_c, phi_max_c = coarse_limits[2], coarse_limits[3]
        z_min_c, z_max_c = coarse_limits[4], coarse_limits[5]

        if n_rho_coarse > 0:
            rho_vec_c = np.geomspace(rho_min_c, rho_max_c, n_rho_coarse)
        else:
            rho_vec_c = np.arange(rho_min_c, rho_max_c + coarse_steps[0],
                                  coarse_steps[0])
        phi_vec_c = np.arange(phi_min_c, phi_max_c, coarse_steps[1]) * (np.pi / 180.0)

        z_vec_c, z_key = self._coarse_z_grid(config, z_min_c, z_max_c, coarse_steps[2])

        coarse_cache_key = (
            'coarse', station_id, tuple(channels),
            n_rho_coarse, rho_min_c, rho_max_c,
            coarse_steps[1], phi_min_c, phi_max_c,
        ) + z_key

        if self._singleray_kernel_active():
            delay_data_c = self._singleray_tt_stack(
                coarse_cache_key, rho_vec_c, phi_vec_c, z_vec_c, channels)
        elif self._multi_ray_types and self._multiray_on_device():
            delay_data_c = None
        elif coarse_cache_key in self._delay_matrix_cache:
            delay_data_c = self._delay_matrix_cache[coarse_cache_key]
        else:
            src_enu_c = self._build_source_enu_matrix(rho_vec_c, phi_vec_c, z_vec_c)
            if self._multi_ray_types:
                delay_data_c = self._compute_tt_multiray(
                    src_enu_c, channels
                )
            else:
                delay_data_c = self._compute_delay_matrices(src_enu_c, channels)
            self._delay_matrix_cache[coarse_cache_key] = delay_data_c

        coarse_grid = (rho_vec_c, phi_vec_c, z_vec_c, delay_data_c,
                       coarse_cache_key)
        full_limits = config.get('limits', coarse_limits)
        skip_optimizer = config.get('skip_optimizer', False)
        opt_method = config.get('optimizer_method', 'L-BFGS-B')
        opt_maxiter = config.get('optimizer_maxiter', 30)
        n_candidates = 0
        snr_map_chain = None
        raw_chain_corr = np.nan
        candidate_gain = np.nan
        fallback = 0
        t_search = 0.0
        t_polish = 0.0
        subbin = config.get('subbin_coarse_seeds', False)

        if candidate_search:
            t0_search = time.time()
            chains = {}
            corr_by_objective = {}
            for name in candidate_search:
                corr_by_objective[name] = series(name.partition(':')[2] or None)
            coarse_maps = {}
            if self._singleray_kernel_active() and len(candidate_search) > 1:
                t0 = time.time()
                maps = self._singleray_stack_maps(
                    delay_data_c, channels,
                    [corr_by_objective[n][1] for n in candidate_search],
                    pair_weights)
                shape = (len(rho_vec_c), len(phi_vec_c), len(z_vec_c))
                coarse_maps = {n: maps[i].reshape(shape)
                               for i, n in enumerate(candidate_search)}
                t_coarse_shared = time.time() - t0
            for name in candidate_search:
                chains[name] = self._search_chain(
                    corr_by_objective[name][0], channels, pair_weights, config,
                    coarse_grid, full_limits, packed=corr_by_objective[name][1],
                    coarse_map=coarse_maps.get(name),
                    subbin_seeds=subbin and name != 'raw')
            if coarse_maps:
                chains[candidate_search[0]]['t_coarse'] += t_coarse_shared
            raw_corr_data, raw_packed = series(None)
            t_search = time.time() - t0_search

            include_refined = config.get('candidate_include_refined', False)
            first = chains[candidate_search[0]]
            if 'blocks' in first:
                block_specs = [(b, st['limits'], st['z_above'])
                               for b, st in enumerate(first['blocks'])]
            else:
                block_specs = [(None, full_limits, None)]
            block_candidates = []
            for b, _, _ in block_specs:
                candidates = []
                for name in candidate_search:
                    code = _CANDIDATE_ORIGIN_CODES[name]
                    chain = chains[name] if b is None else chains[name]['blocks'][b]
                    for rho_p, phi_p, z_p, corr_p in chain.get('all_optimized', []):
                        candidates.append((rho_p, phi_p, z_p, corr_p, code))
                    if include_refined:
                        for rho_p, phi_p, z_p, corr_p in chain.get('unseeded_peaks', []):
                            candidates.append((rho_p, phi_p, z_p, corr_p, code))
                block_candidates.append(candidates)

            if not any(block_candidates):
                logger.warning("No candidates found")
                self._set_station_parameters(
                    station, np.nan, np.nan, np.nan, np.nan)
                return {'rho': np.nan, 'phi': np.nan, 'z': np.nan,
                        'max_corr': np.nan}

            opt_cache = chains.get('raw', {}).get('opt_cache')
            if opt_cache is None:
                opt_cache = self._build_optimizer_cache(
                    channels, pair_weights, raw_corr_data, packed=raw_packed)
            region_entries = []
            t0_polish = time.time()
            candidate_diag = {}
            ranked = []
            for (b, limits, z_above), candidates in zip(block_specs, block_candidates):
                if not candidates:
                    continue
                block_diag = {}
                polished = self._polish_candidates(
                    candidates, raw_corr_data, opt_cache, channels, pair_weights,
                    config, limits, diagnostics=block_diag,
                    two_arrival=two_arrival, z_above=z_above)
                ranked.extend(polished)
                region_entries += [(None if b is None else z_above is not None, c) for c in polished]
                if not candidate_diag:
                    candidate_diag.update(block_diag)
                    continue
                candidate_diag['prepolish_max_corr'] = max(
                    candidate_diag['prepolish_max_corr'], block_diag['prepolish_max_corr'])
                by_origin = candidate_diag['prepolish_max_corr_by_origin']
                for code, value in block_diag['prepolish_max_corr_by_origin'].items():
                    by_origin[code] = max(by_origin.get(code, value), value)
                candidate_diag['n_basins'] += block_diag['n_basins']
            if len(block_specs) > 1:
                ranked.sort(key=lambda c: c[3], reverse=True)
                ranked = self._deduplicate_peaks(ranked)
            n_candidates = len(ranked)
            if two_arrival is not None:
                shown = 8 if two_arrival['max_corr_source'] == 'raw' else 7
                ranked = [c[:3] + (c[shown],) + c[4:] for c in ranked]
            raw_best = chains.get('raw', {}).get('all_optimized')
            if raw_best:
                raw_best = raw_best[0]
                raw_chain_corr = raw_best[3]
                best_raw = ranked[0][8] if two_arrival is not None else ranked[0][3]
                candidate_gain = best_raw - raw_chain_corr
                if tie_band is not None and candidate_gain < tie_band and (
                        tie_ceiling is None or raw_chain_corr < tie_ceiling):
                    fallback = 1
                    entry = tuple(raw_best) + (_CANDIDATE_ORIGIN_CODES['raw'], raw_best[3], False)
                    if two_arrival is not None:
                        entry = self._with_two_arrival(entry, channels, opt_cache, two_arrival)
                    ranked = self._deduplicate_peaks([entry] + ranked)
            fill = []
            if not include_refined:
                for name in candidate_search:
                    fill += self._grade_at_position(
                        chains[name].get('unseeded_peaks', []),
                        _CANDIDATE_ORIGIN_CODES[name], raw_corr_data,
                        opt_cache, channels, pair_weights)
            if two_arrival is not None:
                fill = [self._with_two_arrival(e, channels, opt_cache, two_arrival)
                        for e in fill]
            if region_on and not include_refined:
                for name in candidate_search:
                    for air, peaks in self._chain_blocks(chains[name], 'unseeded_peaks'):
                        region_entries += [(air, e) for e in self._grade_at_position(
                            peaks, _CANDIDATE_ORIGIN_CODES[name], raw_corr_data, opt_cache,
                            channels, pair_weights)]
            pool = self._deduplicate_peaks(ranked + fill)
            candidate_pool = [pool[0]] + sorted(
                pool[1:], key=lambda c: c[3], reverse=True)
            t_polish = time.time() - t0_polish

            all_optimized = [c[:4] for c in ranked]
            saved_pool = (candidate_pool
                          if config.get('candidate_fill_saved_peaks', False)
                          else ranked)
            rho_best, phi_best, z_best, corr_best = all_optimized[0]
            corr_data = raw_corr_data
            bounds = [
                (max(full_limits[0], 1.0), full_limits[1]),
                (full_limits[2], full_limits[3]),
                (full_limits[4], full_limits[5]),
            ]

            quality_chain = 'raw' if 'raw' in chains else candidate_search[0]
            quality = chains[quality_chain]
            mean_corr_c = quality['mean_corr_c']
            coarse_peaks = quality['coarse_peaks']
            refined_peaks = quality.get('refined_peaks', [])
            snr_map = mean_corr_c
            snr_map_chain = _CANDIDATE_ORIGIN_CODES[quality_chain]
            raw_map = chains['raw']['mean_corr_c'] if 'raw' in chains else None
            record_peaks = quality.get('all_optimized', [])
            t_coarse = sum(c['t_coarse'] for c in chains.values())
            t_refine = sum(c.get('t_refine', 0.0) for c in chains.values())
            t_opt = sum(c.get('t_opt', 0.0) for c in chains.values())
        else:
            chain = self._search_chain(
                corr_data, channels, pair_weights, config, coarse_grid,
                full_limits, packed=packed,
                subbin_seeds=subbin and hilbert_mode is not None)
            mean_corr_c = chain['mean_corr_c']
            coarse_peaks = chain['coarse_peaks']
            t_coarse = chain['t_coarse']

            if not coarse_peaks:
                logger.warning("No coarse peaks found")
                self._set_station_parameters(
                    station, np.nan, np.nan, np.nan, np.nan)
                return {'rho': np.nan, 'phi': np.nan, 'z': np.nan,
                        'max_corr': np.nan}

            refined_peaks = chain['refined_peaks']
            all_optimized = chain['all_optimized']
            opt_cache = chain['opt_cache']
            bounds = chain['bounds']
            if region_on:
                region_entries = [(air, tuple(p) + (0,))
                                  for air, peaks in self._chain_blocks(chain, 'all_optimized')
                                  for p in peaks]
                region_cache = opt_cache if opt_cache is not None else self._build_optimizer_cache(
                    channels, pair_weights, corr_data, packed=packed)
                for air, peaks in self._chain_blocks(chain, 'unseeded_peaks'):
                    region_entries += [(air, e) for e in self._grade_at_position(
                        peaks, 0, corr_data, region_cache, channels, pair_weights)]
            t_refine = chain['t_refine']
            t_opt = chain['t_opt']
            rho_best, phi_best, z_best, corr_best = all_optimized[0]
            snr_map = mean_corr_c
            saved_pool = all_optimized
            if config.get('candidate_fill_saved_peaks', False):
                fill_cache = opt_cache
                if fill_cache is None:
                    fill_cache = self._build_optimizer_cache(
                        channels, pair_weights, corr_data, packed=packed)
                fill = self._grade_at_position(
                    chain['unseeded_peaks'], 0, corr_data, fill_cache,
                    channels, pair_weights)
                saved_pool = self._deduplicate_peaks(
                    [p + (0, p[3], False) for p in all_optimized] + fill)

        # Stage 4: Post-optimizer refinement
        t0_post = time.time()
        post_mode = config.get('post_optimizer_mode', None)
        if post_mode and two_arrival is not None:
            logger.debug("post_optimizer_mode %s skipped in two-arrival mode", post_mode)
            post_mode = None

        if post_mode == 'rho_scan' and not skip_optimizer:
            rho_step = config.get('rho_scan_step', 5.0)
            rho_scan = np.arange(
                max(full_limits[0], 1.0), full_limits[1] + rho_step, rho_step)
            best_scan_corr = corr_best
            best_scan_rho = rho_best
            for rho_s in rho_scan:
                neg_corr = self._correlation_at_point(
                    [rho_s, phi_best, z_best], corr_data, channels,
                    pair_weights, _cache=opt_cache)
                if -neg_corr > best_scan_corr:
                    best_scan_corr = -neg_corr
                    best_scan_rho = rho_s
            if abs(best_scan_rho - rho_best) > rho_step:
                r2, p2, z2, c2 = self._optimize_from_seed(
                    (best_scan_rho, phi_best, z_best),
                    corr_data, channels, bounds, pair_weights,
                    method=opt_method, maxiter=opt_maxiter, _cache=opt_cache,
                    config=config)
                if c2 > corr_best:
                    rho_best, phi_best, z_best, corr_best = r2, p2, z2, c2

        elif post_mode == 'differential_evolution' and not skip_optimizer:
            from scipy.optimize import differential_evolution
            de_window = config.get('de_window', [200, 10, 100])
            de_bounds = [
                (max(full_limits[0], max(1.0, rho_best - de_window[0])),
                 min(full_limits[1], rho_best + de_window[0])),
                (phi_best - de_window[1], phi_best + de_window[1]),
                (max(full_limits[4], z_best - de_window[2]),
                 min(full_limits[5], z_best + de_window[2])),
            ]

            def de_obj(params):
                rho, phi_deg, z = params
                phi_actual = phi_deg % 360.0
                return self._correlation_at_point(
                    [rho, phi_actual, z], corr_data, channels,
                    pair_weights, _cache=opt_cache)

            de_result = differential_evolution(
                de_obj, de_bounds,
                maxiter=config.get('de_maxiter', 50),
                popsize=config.get('de_popsize', 10),
                tol=1e-6, seed=42, polish=True,
                init='sobol',
            )
            de_corr = -de_result.fun
            if de_corr > corr_best:
                rho_best = de_result.x[0]
                phi_best = de_result.x[1] % 360.0
                z_best = de_result.x[2]
                corr_best = de_corr

        elif post_mode == 'basinhopping' and not skip_optimizer:
            from scipy.optimize import basinhopping
            bh_window = config.get('bh_window', [200, 10, 100])
            bh_bounds = [
                (max(full_limits[0], max(1.0, rho_best - bh_window[0])),
                 min(full_limits[1], rho_best + bh_window[0])),
                (phi_best - bh_window[1], phi_best + bh_window[1]),
                (max(full_limits[4], z_best - bh_window[2]),
                 min(full_limits[5], z_best + bh_window[2])),
            ]

            def bh_obj(params):
                rho, phi_deg, z = params
                phi_actual = phi_deg % 360.0
                return self._correlation_at_point(
                    [rho, phi_actual, z], corr_data, channels,
                    pair_weights, _cache=opt_cache)

            bh_kwargs = {'method': 'L-BFGS-B', 'bounds': bh_bounds,
                         'options': {'maxiter': 20}}
            bh_result = basinhopping(
                bh_obj, [rho_best, phi_best, z_best],
                minimizer_kwargs=bh_kwargs,
                niter=config.get('bh_niter', 20),
                stepsize=config.get('bh_stepsize', 50.0),
                seed=42,
            )
            bh_corr = -bh_result.fun
            if bh_corr > corr_best:
                rho_best = bh_result.x[0]
                phi_best = bh_result.x[1] % 360.0
                z_best = bh_result.x[2]
                corr_best = bh_corr

        t_post = time.time() - t0_post

        # Stage 5: Raw-correlation refinement (hybrid envelope approach)
        # Rebuild correlation functions without envelope and re-optimize
        # from the current best seed for sharper peak localization.
        t0_raw = time.time()
        refine_envelope = config.get('refinement_envelope_mode', 'UNSET')
        if refine_envelope != 'UNSET' and refine_envelope != hilbert_mode \
                and not skip_optimizer and not candidate_search:
            raw_corr_data, raw_packed = series(refine_envelope)
            raw_cache = self._build_optimizer_cache(
                channels, pair_weights, raw_corr_data, packed=raw_packed)

            raw_window = config.get('refinement_window', [30, 3, 30])
            raw_bounds = [
                (max(full_limits[0], max(1.0, rho_best - raw_window[0])),
                 min(full_limits[1], rho_best + raw_window[0])),
                (phi_best - raw_window[1], phi_best + raw_window[1]),
                (max(full_limits[4], z_best - raw_window[2]),
                 min(full_limits[5], z_best + raw_window[2])),
            ]
            raw_maxiter = config.get('refinement_maxiter', 30)

            r_raw, p_raw, z_raw, c_raw = self._optimize_from_seed(
                (rho_best, phi_best, z_best),
                raw_corr_data, channels, raw_bounds, pair_weights,
                method=opt_method, maxiter=raw_maxiter, _cache=raw_cache,
                config=config,
            )
            # Use raw-refined position but keep Hilbert correlation as
            # quality metric (raw correlation has different normalization)
            corr_hilbert = corr_best
            rho_best, phi_best, z_best = r_raw, p_raw, z_raw
            corr_best = corr_hilbert

        t_raw_refine = time.time() - t0_raw
        phi_best = phi_best % 360.0

        # Stage 6: Multi-peak quality metrics and coherent waveforms
        t0_peaks = time.time()
        n_peaks_save = config.get('n_peaks_save', 1)
        save_coh_wf = config.get('save_coherent_waveforms', False)
        n_coh_wf = config.get('n_coherent_waveforms', 1)

        phi_vec_deg_c = phi_vec_c * (180.0 / np.pi)
        saved_entries = saved_pool[:n_peaks_save]
        saved_peaks = [tuple(e[:4]) for e in saved_entries]

        peak_map_snrs = []
        for rho_p, phi_p, z_p, corr_p in saved_peaks:
            pidx = self._find_peak_bin(
                rho_p, phi_p, z_p, rho_vec_c, phi_vec_deg_c, z_vec_c)
            peak_map_snrs.append(self._compute_map_snr(snr_map, pidx))
        if candidate_search:
            record_map_snrs = []
            for i in range(len(saved_peaks)):
                if i < len(record_peaks):
                    rho_p, phi_p, z_p = record_peaks[i][:3]
                    pidx = self._find_peak_bin(
                        rho_p, phi_p, z_p, rho_vec_c, phi_vec_deg_c, z_vec_c)
                    record_map_snrs.append(self._compute_map_snr(snr_map, pidx))
                else:
                    record_map_snrs.append(np.nan)

        coherent_waveforms = []
        coherent_times = None
        if save_coh_wf and not self._multi_ray_types:
            volt_arrays = [station.get_channel(ch).get_trace() for ch in channels]
            time_arrays = [station.get_channel(ch).get_times() for ch in channels]
            for i, (rho_p, phi_p, z_p, corr_p) in enumerate(saved_peaks):
                if i >= n_coh_wf:
                    break
                t_coh, v_coh = self._compute_coherent_waveform(
                    rho_p, phi_p, z_p, volt_arrays, time_arrays, channels)
                if t_coh is not None:
                    if coherent_times is None:
                        coherent_times = t_coh
                    coherent_waveforms.append(v_coh)

        t_peaks = time.time() - t0_peaks

        logger.debug(
            "3D hierarchical: coarse=%.3fs, refine=%.3fs, opt=%.3fs, "
            "post=%.3fs, raw=%.3fs, peaks=%.3fs, rho=%.1f phi=%.1f z=%.1f "
            "corr=%.4f, n_saved=%d",
            t_coarse, t_refine, t_opt, t_post, t_raw_refine, t_peaks,
            rho_best, phi_best, z_best, corr_best, len(saved_peaks)
        )

        self._set_station_parameters(
            station, rho_best, phi_best, z_best, corr_best)

        result = {
            'rho': rho_best,
            'phi': phi_best,
            'z': z_best,
            'max_corr': corr_best,
            'objective_version': int(self._valid_norm),
            'coarse_time': t_coarse,
            'refine_time': t_refine,
            'opt_time': t_opt,
            'post_time': t_post if post_mode else 0.0,
            'raw_refine_time': t_raw_refine if refine_envelope != 'UNSET' else 0.0,
            'peak_time': t_peaks,
            'n_coarse_peaks': len(coarse_peaks),
            'n_refined_peaks': len(refined_peaks),
            'n_saved_peaks': len(saved_peaks),
            'coarse_peaks': coarse_peaks,
        }

        for i, (rho_p, phi_p, z_p, corr_p) in enumerate(saved_peaks):
            result[f'peak_{i}_rho'] = rho_p
            result[f'peak_{i}_phi'] = phi_p
            result[f'peak_{i}_z'] = z_p
            result[f'peak_{i}_corr'] = corr_p
            if candidate_search:
                result[f'peak_{i}_map_snr'] = record_map_snrs[i]
                result[f'peak_{i}_map_snr_v2'] = peak_map_snrs[i]
            else:
                result[f'peak_{i}_map_snr'] = peak_map_snrs[i]

        for i, wf in enumerate(coherent_waveforms):
            result[f'coherent_wf_{i}'] = wf
        if coherent_times is not None:
            result['coherent_times'] = coherent_times

        if config.get('candidate_fill_saved_peaks', False):
            result['n_filled_peaks'] = sum(int(e[6]) for e in saved_entries)
        if candidate_search:
            result['n_candidates'] = n_candidates
            result['candidate_n_pool'] = len(candidate_pool)
            result['candidate_search_time'] = t_search
            result['candidate_polish_time'] = t_polish
            result['candidate_map_snr_chain'] = snr_map_chain
            result['candidate_raw_chain_corr'] = raw_chain_corr
            result['candidate_gain'] = candidate_gain
            if tie_band is not None:
                result['candidate_fallback'] = fallback
            for i, entry in enumerate(saved_entries):
                result[f'candidate_origin_{i}'] = entry[4]
            pool_primary = candidate_pool[0][3]
            top_pool = sorted((c[3] for c in candidate_pool), reverse=True)[:5]
            pool_mean = float(np.mean(top_pool))
            result['peak_isolation_ratio_v2'] = (
                float(pool_primary / pool_mean)
                if len(candidate_pool) >= 2 and pool_mean > 0 else np.nan)
            result['map_snr_v2'] = np.nan
            if raw_map is not None:
                if not isinstance(raw_map, np.ndarray):
                    raw_std = raw_map.finite_std()
                elif USE_NUMBA:
                    finite = np.ascontiguousarray(raw_map[np.isfinite(raw_map)], dtype=np.float64)
                    raw_std = float(_numpy_std(finite)) if finite.size else np.nan
                else:
                    raw_std = float(np.std(raw_map[np.isfinite(raw_map)]))
                if raw_std > 1e-12:
                    result['map_snr_v2'] = float(pool_primary / raw_std)
            if config.get('candidate_diagnostics', False):
                result['candidate_prepolish_max_corr'] = candidate_diag['prepolish_max_corr']
                by_origin = candidate_diag['prepolish_max_corr_by_origin']
                for name, code in _CANDIDATE_ORIGIN_CODES.items():
                    suffix = name.partition(':')[2] or name
                    result[f'candidate_prepolish_max_corr_{suffix}'] = by_origin.get(
                        code, np.nan)
                result['candidate_n_basins'] = candidate_diag['n_basins']
                for i, entry in enumerate(saved_entries):
                    result[f'candidate_prepolish_corr_{i}'] = entry[5]
                for i in range(_CANDIDATE_POOL_SAVE):
                    entry = (candidate_pool[i] if i < len(candidate_pool)
                             else (np.nan, np.nan, np.nan, np.nan, -1))
                    result[f'candidate_pool_{i}_rho'] = entry[0]
                    result[f'candidate_pool_{i}_phi'] = entry[1]
                    result[f'candidate_pool_{i}_z'] = entry[2]
                    result[f'candidate_pool_{i}_corr'] = entry[3]
                    result[f'candidate_pool_{i}_origin'] = entry[4]
        if two_arrival is not None:
            result['corr_two_arrival'] = saved_entries[0][7]
            result['raw_corr_single'] = saved_entries[0][8]
            for i, entry in enumerate(saved_entries):
                result[f'peak_{i}_corr_two_arrival'] = entry[7]
                result[f'peak_{i}_raw_corr_single'] = entry[8]
        if region_on:
            result.update(self._region_hypotheses(
                region_entries, channels, pair_weights, series, snr_map,
                (rho_vec_c, phi_vec_deg_c, z_vec_c)))
        if config.get('far_field_hypothesis', False):
            result.update(self._far_field_hypothesis(
                channels, pair_weights, series, config.get('far_field_lobe_guard_ns'),
                config.get('optimizer_gradient', 'finite_difference')))
        if config.get('validation', False):
            val = self._compute_validation_metrics(
                mean_corr_c, rho_vec_c, phi_vec_c, z_vec_c,
                channel_snrs, coarse_peaks, config, windowed_snrs)
            result.update(val)

        return result
