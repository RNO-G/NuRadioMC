"""The candidate and polish stage of the hierarchical search of the 3D reconstruction."""

import numpy as np
import numbers

from NuRadioReco.utilities.reco3d_kernels import _build_split_z_window

from NuRadioReco.modules.reco3d.shared import _CANDIDATE_ORIGIN_CODES


# Bound to the class by the module that defines it, once the class exists.
InterferometricReco3D = None


_CANDIDATE_ENVELOPE_MODES = ('traces', 'correlation')
_POLISH_OBJECTIVES = ('raw', 'two_arrival_consistent')
_TWO_ARRIVAL_WEIGHT_MODES = ('mask', 'fixed')
_MAX_CORR_SOURCES = ('raw', 'two_arrival')


class CandidatesMixin:
    """Methods of InterferometricReco3D for the candidate and polish stage of the hierarchical search."""

    @staticmethod
    def _candidate_chains(config):
        """Return the candidate search chains named by ``candidate_search``.

        Each entry is "raw", "envelope:<mode>" with mode "traces" or
        "correlation", or plain "envelope", which stands for
        "envelope:<candidate_envelope_mode>". An absent or empty key gives [].

        Raises:
            ValueError: If the key is not a list of strings, an entry has none
                of these forms, ``candidate_envelope_mode`` is unknown, or two
                entries resolve to the same chain.
        """
        entries = config.get('candidate_search', None) or []
        if not isinstance(entries, (list, tuple)) or not all(
                isinstance(c, str) for c in entries):
            raise ValueError(
                f"candidate_search must be a list of strings, got {entries!r}")
        env_mode = config.get('candidate_envelope_mode', 'traces')
        if entries and env_mode not in _CANDIDATE_ENVELOPE_MODES:
            raise ValueError(
                "candidate_envelope_mode must be one of "
                f"{_CANDIDATE_ENVELOPE_MODES}, got {env_mode!r}")
        chains = []
        for entry in entries:
            name = f'envelope:{env_mode}' if entry == 'envelope' else entry
            if name not in _CANDIDATE_ORIGIN_CODES:
                raise ValueError(
                    "candidate_search entries must be 'raw', 'envelope' or "
                    "'envelope:<mode>' with mode in "
                    f"{_CANDIDATE_ENVELOPE_MODES}, got {entry!r}")
            if name in chains:
                raise ValueError(
                    f"candidate_search entry {entry!r} resolves to chain "
                    f"{name!r} more than once")
            chains.append(name)
        return chains

    @staticmethod
    def _candidate_tie_band(config):
        """Return ``candidate_tie_band`` as a float, or None when the key is absent.

        Returns:
            float or None: The band, or None when the key is absent.

        Raises:
            ValueError: If the band is not a non-negative real number or
                ``candidate_search`` does not include the "raw" chain.
        """
        band = config.get('candidate_tie_band', None)
        if band is None:
            return None
        if isinstance(band, bool) or not isinstance(band, numbers.Real) \
                or not band >= 0:
            raise ValueError(
                f"candidate_tie_band must be a non-negative number, got {band!r}")
        if 'raw' not in InterferometricReco3D._candidate_chains(config):
            raise ValueError(
                "candidate_tie_band requires 'raw' among the candidate_search chains")
        return float(band)

    @staticmethod
    def _two_arrival_settings(config):
        """Return the two-arrival polish settings, or None when ``polish_objective`` is raw.

        ``polish_objective`` is "raw" (default) or "two_arrival_consistent";
        ``two_arrival_weight_mode`` is "mask" (default: the solution_1 term of a
        pair is weighted by ``two_arrival_second_weight`` times the product of the
        two channels' critical-angle masks) or "fixed" (by
        ``two_arrival_second_weight`` everywhere); ``two_arrival_margin`` is absent
        (default: the candidates are polished and ranked by the raw correlation and
        the two-arrival value is saved as a diagnostic) or a non-negative number
        (the candidates are also polished by the two-arrival objective and a
        position is ranked by its two-arrival value when that exceeds its raw
        correlation by more than the margin, otherwise by its raw correlation);
        ``max_corr_source`` is "raw" (default: ``max_corr`` and ``peak_{i}_corr``
        keep the single-arrival raw correlation) or "two_arrival" (they carry the
        two-arrival value). The two-arrival value is always divided by the total
        pair weight and reads the tables with the tolerant edge rule (last row
        and column accepted), whatever ``tolerant_table_edge`` says.

        Returns:
            Dict with 'weight_mode', 'second_weight', 'margin' (None or float) and
            'max_corr_source', or None.

        Raises:
            ValueError: On an unknown objective, weight mode or max_corr source, a
                negative or non-numeric second weight or margin, or the two-arrival
                objective without ``candidate_search`` (the objective runs only in
                the candidate polish stage), with ``multi_ray_types`` (the raw
                polish it is ranked against must be the single-table objective) or
                with ``objective_normalisation`` other than "total" (the
                two-arrival value is always divided by the total pair weight).
        """
        objective = config.get('polish_objective', 'raw')
        if objective not in _POLISH_OBJECTIVES:
            raise ValueError(
                f"polish_objective must be one of {_POLISH_OBJECTIVES}, got {objective!r}")
        weight_mode = config.get('two_arrival_weight_mode', 'mask')
        if weight_mode not in _TWO_ARRIVAL_WEIGHT_MODES:
            raise ValueError(
                "two_arrival_weight_mode must be one of "
                f"{_TWO_ARRIVAL_WEIGHT_MODES}, got {weight_mode!r}")
        second_weight = config.get('two_arrival_second_weight', 1.0)
        if not isinstance(second_weight, numbers.Real) or isinstance(second_weight, bool) \
                or second_weight < 0:
            raise ValueError(
                f"two_arrival_second_weight must be a non-negative number, got {second_weight!r}")
        margin = config.get('two_arrival_margin', None)
        if margin is not None and (not isinstance(margin, numbers.Real)
                                   or isinstance(margin, bool) or margin < 0):
            raise ValueError(
                f"two_arrival_margin must be absent or a non-negative number, got {margin!r}")
        max_corr_source = config.get('max_corr_source', 'raw')
        if max_corr_source not in _MAX_CORR_SOURCES:
            raise ValueError(
                f"max_corr_source must be one of {_MAX_CORR_SOURCES}, got {max_corr_source!r}")
        if objective == 'raw':
            return None
        if not InterferometricReco3D._candidate_chains(config):
            raise ValueError(
                "polish_objective two_arrival_consistent needs candidate_search; "
                "the objective runs in the candidate polish stage only")
        if config.get('multi_ray_types', False):
            raise ValueError(
                "polish_objective two_arrival_consistent needs multi_ray_types: false; "
                "the raw polish it is ranked against is the single-table objective")
        if config.get('objective_normalisation', 'total') != 'total':
            raise ValueError(
                "polish_objective two_arrival_consistent needs objective_normalisation: total; "
                "the two-arrival value is always normalised by the total pair weight")
        return {'weight_mode': weight_mode, 'second_weight': float(second_weight),
                'margin': None if margin is None else float(margin),
                'max_corr_source': max_corr_source}

    @staticmethod
    def _candidate_tie_band_max_raw_corr(config):
        """Return the tie-band ceiling on the raw chain's correlation, or None when absent.

        The tie-band fallback applies only to events whose raw chain answer
        ``candidate_raw_chain_corr`` is below this ceiling.

        Returns:
            float or None: The ceiling, or None when the key is absent.

        Raises:
            ValueError: If the ceiling is not a non-negative real number or
                ``candidate_tie_band`` is not set.
        """
        ceiling = config.get('candidate_tie_band_max_raw_corr', None)
        if ceiling is None:
            return None
        if isinstance(ceiling, bool) or not isinstance(ceiling, numbers.Real) \
                or not ceiling >= 0:
            raise ValueError(
                "candidate_tie_band_max_raw_corr must be a non-negative number, "
                f"got {ceiling!r}")
        if InterferometricReco3D._candidate_tie_band(config) is None:
            raise ValueError(
                "candidate_tie_band_max_raw_corr requires candidate_tie_band")
        return float(ceiling)

    @staticmethod
    def _polish_levels(config):
        """Return the candidate polish grid levels as a list of (window, steps) pairs.

        ``candidate_polish_window`` and ``candidate_polish_steps`` are each one
        triple (m, deg, m) or a list of triples, one per level. The levels run in
        order, each centred on the best point found so far.

        Raises:
            ValueError: If either key is not three positive numbers per level or
                the two keys have different numbers of levels.
        """
        seq = (list, tuple, np.ndarray)
        levels = {}
        for key, default in (('candidate_polish_window', [3.0, 1.0, 3.0]),
                             ('candidate_polish_steps', [0.5, 0.1, 0.5])):
            val = config.get(key, None)
            if val is None:
                val = default
            nested = isinstance(val, seq) and len(val) > 0 and all(
                isinstance(t, seq) for t in val)
            tiers = list(val) if nested else [val]
            if not all(isinstance(t, seq) and len(t) == 3
                       and all(isinstance(v, numbers.Real) and v > 0 for v in t)
                       for t in tiers):
                raise ValueError(
                    f"{key} must be three positive numbers (m, deg, m) or a "
                    f"list of such triples, got {val!r}")
            levels[key] = [[float(v) for v in t] for t in tiers]
        if len(levels['candidate_polish_window']) != len(levels['candidate_polish_steps']):
            raise ValueError(
                "candidate_polish_window and candidate_polish_steps must have "
                "the same number of levels")
        return list(zip(levels['candidate_polish_window'],
                        levels['candidate_polish_steps']))

    def _polish_grid_max(self, rho_p, phi_p, z_p, corr_data, packed, channels,
                         pair_weights, window, steps, full_limits, evaluate=None,
                         z_above=None):
        """Evaluate a dense local grid around one candidate and return its maximum.

        The grid is clamped to the search limits in rho (never below 1 m) and z;
        phi is not clamped and is wrapped in the result. With ``z_above`` the z
        vector is built by _build_split_z_window.

        Args:
            rho_p: Candidate rho in m.
            phi_p: Candidate phi in deg.
            z_p: Candidate z in m.
            corr_data: Correlation functions to evaluate.
            packed: CorrPacked of corr_data for the fused kernel, or None.
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.
            window: [rho, phi_deg, z] half-widths of the grid.
            steps: [rho, phi_deg, z] grid steps.
            full_limits: Search limits.
            evaluate: Optional function of (rho_vec, phi_vec_rad, z_vec) returning
                the map to maximize; None evaluates the raw correlation.
            z_above: Air block of the split z grid (offset and refine spacing), or None.

        Returns:
            (rho, phi_deg, z, corr) at the grid maximum, or None when the grid is
            empty or holds no finite value.
        """
        rho_lo = max(max(full_limits[0], 1.0), rho_p - window[0])
        rho_hi = min(full_limits[1], rho_p + window[0])
        z_lo = max(full_limits[4], z_p - window[2])
        z_hi = min(full_limits[5], z_p + window[2])
        rho_vec = np.arange(max(rho_lo, 1.0), rho_hi + steps[0], steps[0])
        rho_vec = rho_vec[rho_vec <= rho_hi + 1e-9]
        phi_vec_deg = np.arange(phi_p - window[1], phi_p + window[1] + steps[1],
                                steps[1])
        if z_above is None:
            z_vec = np.arange(z_lo, z_hi + steps[2], steps[2])
        else:
            z_vec = _build_split_z_window(z_lo, z_hi, steps[2], z_above)
        z_vec = z_vec[z_vec <= z_hi + 1e-9]
        if len(rho_vec) == 0 or len(phi_vec_deg) == 0 or len(z_vec) == 0:
            return None
        self.work['polish_grids'] += 1
        self.work['polish_points'] += len(rho_vec) * len(phi_vec_deg) * len(z_vec)

        phi_vec_rad = phi_vec_deg * (np.pi / 180.0)
        if evaluate is not None:
            mean_corr = evaluate(rho_vec, phi_vec_rad, z_vec)
        elif self._multi_ray_types:
            mean_corr, _ = self._multiray_grid(
                rho_vec, phi_vec_rad, z_vec, corr_data, channels, pair_weights)
        elif self._singleray_kernel_active():
            if packed is None:
                packed = self._pack_corr_data(corr_data)
            mean_corr = self._singleray_grid_maps(
                rho_vec, phi_vec_rad, z_vec, channels, [packed], pair_weights)[0]
        else:
            src_enu = self._build_source_enu_matrix(rho_vec, phi_vec_rad, z_vec)
            delay_data = self._compute_delay_matrices(src_enu, channels)
            mean_corr, _ = self._correlator_lean(
                corr_data, delay_data, pair_weights=pair_weights, packed=packed)
        if not np.any(np.isfinite(mean_corr)):
            return None
        idx = np.unravel_index(np.nanargmax(mean_corr), mean_corr.shape)
        return (float(rho_vec[idx[0]]), float(phi_vec_deg[idx[1]] % 360.0),
                float(z_vec[idx[2]]), float(mean_corr[idx]))

    def _grade_at_position(self, peaks, origin, corr_data, opt_cache, channels,
                           pair_weights):
        """Evaluate the objective of ``corr_data`` at each peak position without moving it.

        Args:
            peaks: List of (rho, phi_deg, z, corr, ...) tuples.
            origin: Chain code stored with every graded entry.
            corr_data: Correlation functions defining the objective.
            opt_cache: Optimizer cache built from corr_data.
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.

        Returns:
            List of (rho, phi_deg, z, corr, origin, corr, True) with phi wrapped
            to [0, 360), corr the objective at the position (repeated as the
            pre-polish value) and True marking the entry as unpolished, sorted
            by corr.
        """
        graded = []
        for rho_p, phi_p, z_p in (p[:3] for p in peaks):
            phi_p = phi_p % 360.0
            corr_at = -self._correlation_at_point(
                [rho_p, phi_p, z_p], corr_data, channels, pair_weights,
                _cache=opt_cache)
            graded.append((float(rho_p), float(phi_p), float(z_p),
                           float(corr_at), origin, float(corr_at), True))
        graded.sort(key=lambda c: c[3], reverse=True)
        return graded

    def _polish_with_objective(self, candidates, corr_data, opt_cache, channels,
                               pair_weights, config, full_limits, value_at,
                               evaluate=None, objective_fn=None, prepolish_at=None,
                               diagnostics=None, z_above=None):
        """Grade, deduplicate and optimize candidates with one objective.

        Every candidate is graded on one or more local grids of the objective,
        each level centred on the best point so far (the better of the grid
        maximum and that point is kept), the graded list is deduplicated by
        position keeping the highest value, and the survivors are optimized from
        their graded position and deduplicated again. Grading all candidates
        before deduplicating makes the value the tie-breaker between chains;
        deduplicating before the optimizer keeps the optimizer count at the
        number of distinct positions.

        Args:
            candidates: List of (rho, phi_deg, z, corr, origin) tuples.
            corr_data: Raw correlation functions.
            opt_cache: Optimizer cache built from corr_data.
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.
            config: Reconstruction config dict.
            full_limits: Search limits that clamp the grids and bound the optimizer.
            value_at: Function of [rho, phi_deg, z] returning the objective value.
            evaluate: Grid map function for ``_polish_grid_max`` (None: raw correlation).
            objective_fn: Minimization function for the optimizer (None: raw).
            prepolish_at: Function of [rho, phi_deg, z] giving the pre-polish
                value recorded with each candidate (None: ``value_at``).
            diagnostics: Optional dict that receives 'prepolish_max_corr' (the
                highest pre-polish value at any candidate position),
                'prepolish_max_corr_by_origin' (the same per origin code) and
                'n_basins' (distinct positions after the grading deduplication,
                before the optimizer).
            z_above: Air block of the split z grid, which places the z vector of the
                polish grids (_build_split_z_window), or None for plain linear grids.

        Returns:
            List of (rho, phi_deg, z, value, origin, prepolish, False) sorted by
            value and deduplicated with the optimizer-output tolerances;
            prepolish is the pre-polish value at the candidate's position and
            False marks the entry as polished.
        """
        levels = self._polish_levels(config)
        packed = opt_cache.get('packed')

        graded = []
        for rho_p, phi_p, z_p, _, origin in candidates:
            phi_p = phi_p % 360.0
            start = [rho_p, phi_p, z_p]
            value = float(value_at(start))
            prepolish = value if prepolish_at is None else float(prepolish_at(start))
            best = (float(rho_p), float(phi_p), float(z_p), value)
            for window, steps in levels:
                grid_max = self._polish_grid_max(
                    best[0], best[1], best[2], corr_data, packed, channels,
                    pair_weights, window, steps, full_limits, evaluate=evaluate,
                    z_above=z_above)
                if grid_max is not None and grid_max[3] > best[3]:
                    best = grid_max
            graded.append(best + (origin, prepolish))
        if diagnostics is not None:
            diagnostics['prepolish_max_corr'] = max(c[5] for c in graded)
            diagnostics['prepolish_max_corr_by_origin'] = {
                code: max(c[5] for c in graded if c[4] == code)
                for code in {c[4] for c in graded}}
        graded.sort(key=lambda c: c[3], reverse=True)
        graded = self._deduplicate_peaks(graded)
        if diagnostics is not None:
            diagnostics['n_basins'] = len(graded)

        if config.get('skip_optimizer', False):
            return [c[:6] + (False,) for c in graded]

        bounds = [
            (max(full_limits[0], 1.0), full_limits[1]),
            (full_limits[2], full_limits[3]),
            (full_limits[4], full_limits[5]),
        ]
        polished = []
        optimized = self._optimize_seeds(
            graded, corr_data, channels, bounds, pair_weights, config, opt_cache,
            objective_fn=objective_fn)
        for (rho_p, phi_p, z_p, corr_p, origin, corr_pre), (rho_o, phi_o, z_o, corr_o) in zip(
                graded, optimized):
            if corr_o > corr_p:
                polished.append((float(rho_o), float(phi_o % 360.0),
                                 float(z_o), float(corr_o), origin, corr_pre,
                                 False))
            else:
                polished.append((rho_p, phi_p, z_p, corr_p, origin, corr_pre,
                                 False))
        polished.sort(key=lambda c: c[3], reverse=True)
        return self._deduplicate_peaks(polished)

    def _polish_candidates(self, candidates, corr_data, opt_cache, channels,
                           pair_weights, config, full_limits, diagnostics=None,
                           two_arrival=None, z_above=None):
        """Polish candidate positions and rank them.

        Without ``two_arrival`` the candidates are polished and ranked by the raw
        correlation (``_polish_with_objective``). With ``two_arrival`` every
        raw-polished position also gets its consistent two-arrival correlation on
        the solution-ordered tables (same solution index in both channels of a
        pair, the solution_1 term weighted by the critical-angle masks or a fixed
        weight). When the settings carry a margin the candidates are polished a
        second time by the two-arrival objective, and every position from either
        polish is ranked by its two-arrival value when that exceeds its raw
        correlation by more than the margin and by its raw correlation otherwise;
        without a margin the ranking is the raw one and the two-arrival value is
        a diagnostic.

        Args:
            candidates: List of (rho, phi_deg, z, corr, origin) tuples.
            corr_data: Raw correlation functions.
            opt_cache: Optimizer cache built from corr_data.
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.
            config: Reconstruction config dict.
            full_limits: Search limits that clamp the grids and bound the optimizer.
            diagnostics: Optional dict filled by the raw polish (see
                ``_polish_with_objective``).
            two_arrival: Settings from ``_two_arrival_settings`` or None.
            z_above: Air block of the split z grid, which places the z vector of the
                polish grids (_build_split_z_window), or None for plain linear grids.

        Returns:
            List of (rho, phi_deg, z, corr, origin, prepolish_corr, False) sorted
            by raw correlation and deduplicated with the optimizer-output
            tolerances, prepolish_corr being the raw correlation at the
            candidate's position before polishing; with ``two_arrival`` the
            tuples carry corr_two_arrival and raw_corr_single after those seven
            fields, the fourth field is the ranking value and the list is sorted
            by it.
        """
        def raw_at(params):
            """Raw single-arrival correlation at [rho, phi_deg, z]."""
            return -self._correlation_at_point(
                params, corr_data, channels, pair_weights, _cache=opt_cache)

        polished = self._polish_with_objective(
            candidates, corr_data, opt_cache, channels, pair_weights, config,
            full_limits, raw_at, diagnostics=diagnostics, z_above=z_above)
        if two_arrival is None:
            return polished

        def two_arrival_at(params):
            """Consistent two-arrival correlation at [rho, phi_deg, z]."""
            return -self._two_arrival_at_point(params, channels, opt_cache, two_arrival)

        pool = [c[:7] + (two_arrival_at(list(c[:3])), c[3]) for c in polished]
        margin = two_arrival['margin']
        if margin is not None:
            def evaluate(rho_vec, phi_vec_rad, z_vec):
                """Two-arrival correlation map on a polish grid."""
                return self._two_arrival_grid(
                    rho_vec, phi_vec_rad, z_vec, channels, opt_cache, two_arrival)

            def objective_fn(params):
                """Negative two-arrival correlation for the optimizer."""
                return -two_arrival_at(params)

            for c in self._polish_with_objective(
                    candidates, corr_data, opt_cache, channels, pair_weights,
                    config, full_limits, two_arrival_at, evaluate=evaluate,
                    objective_fn=objective_fn, prepolish_at=raw_at, z_above=z_above):
                pool.append(c[:7] + (c[3], raw_at(list(c[:3]))))
        ranked = []
        for entry in pool:
            two, raw = float(entry[7]), float(entry[8])
            key = two if margin is not None and two - raw > margin else raw
            ranked.append(entry[:3] + (float(key),) + entry[4:7] + (two, raw))
        ranked.sort(key=lambda c: c[3], reverse=True)
        return self._deduplicate_peaks(ranked)
