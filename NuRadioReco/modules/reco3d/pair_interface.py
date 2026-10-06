"""The pair store interface of the 3D reconstruction."""

import numpy as np
import itertools
import yaml
from collections import namedtuple

from NuRadioReco.modules import reco3d_batch

from NuRadioReco.modules.reco3d.correlation import CorrPacked, SERIES_MODES


class PairSet(namedtuple('PairSet', ['channels', 'pairs', 'series', 'snr', 'snr_windowed',
                                     'settings', 'windows'])):
    """Pair correlation series of one event: the input of the search stage of the reconstruction.

    Fields:
        channels: Tuple of channel ids in the order the traces were read.

        pairs: Tuple of (ch_a, ch_b) channel pairs, ``itertools.combinations`` of
            ``channels``; lag = t_a - t_b.

        series: Dict envelope mode (None raw, 'traces', 'correlation') -> CorrPacked
            over ``pairs`` at full length (the sample lags of every mode coincide).

        snr: Dict channel -> record SNR (3-sample window), empty when not computed.

        snr_windowed: Dict channel -> SNR over ``settings['snr_window_ns']``, or empty.

        settings: Dict of the preprocessing keys the series depend on
            (``apply_hann_window``, ``correlation_normalization``, ``snr_window_ns``).

        windows: None for complete series, else (n_pairs, 2) lag windows in ns
            outside which the series were cut (zeros there).
    """

    __slots__ = ()


class PairInterfaceMixin:
    """Methods of InterferometricReco3D for the pair store interface."""

    def compute_pairs(self, station, config, store=False):
        """Compute the pair correlation series of one event: the first stage of the reconstruction.

        Reads the preprocessed traces of ``config['channels']`` and returns the
        pair series and channel SNRs that the search stage
        (``reconstruct_from_pairs``) reads. Without ``store`` only the envelope
        modes the config searches (``_series_modes``) and the SNRs it uses are
        computed, which is what ``run`` does; with ``store`` every envelope mode
        and the record SNR are computed, as the pair store keeps them.

        Args:
            station: Station holding the preprocessed traces.
            config: Reconstruction config dict.
            store: Compute every envelope mode and the record SNR.

        Returns:
            PairSet over every pair of ``config['channels']`` with complete series.
        """
        self._set_position_shift(config.get('channel_position_shift'))
        channels = list(config['channels'])
        volt_arrays = []
        time_arrays = []
        for ch in channels:
            channel = station.get_channel(ch)
            volt_arrays.append(channel.get_trace())
            time_arrays.append(channel.get_times())
        snr, windowed = self._event_snrs(volt_arrays, time_arrays, channels, config, record=store)
        series = self._pair_series(
            time_arrays, volt_arrays, SERIES_MODES if store else self._series_modes(config),
            config.get('apply_hann_window', False),
            config.get('correlation_normalization', 'normalized'))
        return PairSet(tuple(channels), tuple(itertools.combinations(channels, 2)), series,
                       snr, windowed, self._series_settings(config), None)

    def reconstruct_from_pairs(self, pairs, config, channel_mask=None, pair_weights=None,
                               channel_delay_shift=None, channel_polarity=None, station=None,
                               channel_position_shift=None):
        """Run the search stage of the reconstruction on the pair series of one event.

        The search ``run`` performs on an event's traces (coarse grid, candidate
        chains, raw ranking, tie band, noise ceiling, polish, refine levels and
        optimizer, per polarization group and HPol sign mode), on the pair series
        of ``compute_pairs`` or of a pair store. ``run`` of a hierarchical
        configuration is this function on ``compute_pairs(station, config)``, so
        the two agree bit for bit on the same series. ``begin`` must have loaded
        the tables of every channel searched.

        Args:
            pairs: PairSet from ``compute_pairs`` or ``pair_store.PairStore.event``.

            config: Reconstruction config dict or YAML path. Its channels must be
                listed in the order of ``pairs.channels``; its
                ``apply_hann_window`` and ``correlation_normalization`` (and
                ``snr_window_ns`` when set) must be those the series were made with.

            channel_mask: Channels removed from the channel list (and so from every
                polarization group) before the search.

            pair_weights: Dict (ch_a, ch_b) -> weight that replaces the SNR pair
                weights of every searched pair; either channel order is accepted.

            channel_delay_shift: Dict channel -> ns added to the channel's cable
                delay, the convention of the delay-corrections files: the lag axis
                of every series of a pair (a, b) moves by shift_b - shift_a, as if
                the traces had been preprocessed with that extra correction.

            channel_polarity: Dict channel -> +1 or -1. The raw series of a pair is
                multiplied by the product of its channels' polarities, which equals
                negating the traces bit for bit; envelopes carry no polarity.

            station: Station that receives the result parameters and supplies the
                traces of ``save_coherent_waveforms``, or None.

            channel_position_shift: Dict channel -> (dx, dy) in m added to the channel's
                horizontal position in every place the search reads positions (grid,
                refine and optimizer kernels, lag windows, region and far-field
                hypotheses); the reconstruction frame (the phased-array centre of the
                database) stays. None reads the config key of the same name. Each
                table is axially symmetric about its antenna, so no table changes;
                vertical shifts would need new tables and are not accepted.

        Returns:
            Result dict, as ``run`` returns it.

        Raises:
            ValueError: For a configuration this stage cannot run (not hierarchical,
                ``tdoa_mode``, coherent waveforms without a station), series
                settings or envelope modes the pair set lacks, channels missing from
                it, or cut series whose lag windows do not hold every delay the
                configuration and shifts can read.
        """
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.safe_load(f)
        if channel_position_shift is not None:
            config = dict(config, channel_position_shift=channel_position_shift)
        self._set_position_shift(config.get('channel_position_shift'))
        if channel_mask:
            config = self._mask_channels(config, channel_mask)
        if config.get('polarization_groups', None) is not None:
            return self._run_per_polarization(
                config, lambda group_config: self.reconstruct_from_pairs(
                    pairs, group_config, None, pair_weights, channel_delay_shift,
                    channel_polarity, station, channel_position_shift),
                station)
        if config.get('tdoa_mode', False) or not config.get('hierarchical', False):
            raise ValueError("reconstruct_from_pairs runs hierarchical configurations only")
        if config.get('save_coherent_waveforms', False) and station is None:
            raise ValueError("save_coherent_waveforms needs the traces: pass the station")
        settings = self._series_settings(config)
        mismatch = [k for k in ('apply_hann_window', 'correlation_normalization')
                    if settings[k] != pairs.settings[k]]
        if settings['snr_window_ns'] not in (None, pairs.settings['snr_window_ns']):
            mismatch.append('snr_window_ns')
        if mismatch:
            raise ValueError(f"series settings differ in {mismatch}: the pair set was made with "
                             f"{pairs.settings}")
        missing = [m for m in self._series_modes(config) if m not in pairs.series]
        if missing:
            raise ValueError(f"the pair set lacks the envelope modes {missing} the config searches")
        if pairs.windows is not None:
            self._check_lag_windows(pairs, config, channel_delay_shift)
        return self._search_hierarchical(pairs, config, station, pair_weights,
                                         channel_delay_shift, channel_polarity)

    def reconstruct_from_pairs_batch(self, pairs, settings, coarse_backend=None, stats=None,
                                     batch_grids=False):
        """Run the search of many settings on the pair series of one event; equals separate calls.

        Every setting gets exactly the result of ``reconstruct_from_pairs(pairs, **setting)``:
        its own complete coarse maps, peaks, refine levels, optimizer, polish and hypotheses.
        Only the coarse maps of the settings are computed together: the per-pair
        contributions on the coarse grid are shared by every setting that reads the same
        series at the same lag offset, and each setting sums its own pairs with its own
        weights and signs in its own order (``reco3d_batch``).

        Args:
            pairs: PairSet of one event.

            settings: List of dicts of ``reconstruct_from_pairs`` keyword arguments (config,
                channel_mask, pair_weights, channel_delay_shift, channel_polarity,
                channel_position_shift).

            coarse_backend: Optional coarse-map backend (e.g. GPU); None uses the CPU kernel.

            stats: Optional dict that receives the batch timing.

            batch_grids: Also batch the refine and polish grids of the settings.

        Returns:
            List of result dicts in the order of ``settings``.
        """
        return reco3d_batch.reconstruct_batch(self, [(pairs, s) for s in settings],
                                              coarse_backend=coarse_backend, stats=stats,
                                              batch_grids=batch_grids)

    @classmethod
    def _series_modes(cls, config):
        """Envelope modes of the pair series the search of a config reads (None: raw correlation)."""
        if config.get('region_hypotheses', False) or config.get('far_field_hypothesis', False):
            return SERIES_MODES
        chains = cls._candidate_chains(config)
        if chains:
            modes = [name.partition(':')[2] or None for name in chains] + [None]
        else:
            modes = [config.get('hilbert_envelope_mode', None)]
            refine = config.get('refinement_envelope_mode', 'UNSET')
            if refine != 'UNSET' and not config.get('skip_optimizer', False):
                modes.append(refine)
        return tuple(dict.fromkeys(modes))

    @staticmethod
    def _series_settings(config):
        """Preprocessing keys of a config that the pair series and SNRs depend on."""
        norm = config.get('correlation_normalization', 'normalized')
        return {'apply_hann_window': bool(config.get('apply_hann_window', False)),
                'correlation_normalization': 'pearson' if norm == 'normalized' else norm,
                'snr_window_ns': config.get('snr_window_ns', None)}

    @staticmethod
    def _mask_channels(config, channel_mask):
        """Return a copy of a config without the masked channels in its channel list and pair signs."""
        masked = set(channel_mask)
        channels = [ch for ch in config['channels'] if ch not in masked]
        out = dict(config, channels=channels)
        if config.get('pair_signs', None) is not None:
            signs = dict(zip(itertools.combinations(config['channels'], 2), config['pair_signs']))
            out['pair_signs'] = [signs[p] for p in itertools.combinations(channels, 2)]
        return out

    def _group_inputs(self, pairs, channels, config, pair_weights=None,
                      channel_delay_shift=None, channel_polarity=None):
        """Pair weights, channel SNRs and the series of one channel group.

        The group's pairs are ``itertools.combinations(channels, 2)``, taken from
        ``pairs`` as they are when they are all of its pairs in order and as row
        copies otherwise. The config's ``pair_signs`` and the channel polarities
        act on copies of the raw series; the delay shifts replace the lag offsets.

        Args:
            pairs: PairSet holding every pair of ``channels``.
            channels: Channel IDs of the group, in the order of ``pairs.channels``.
            config: Reconstruction config dict of the group.
            pair_weights: Optional dict (ch_a, ch_b) -> weight replacing the SNR weights.
            channel_delay_shift: Optional dict channel -> ns (delay-correction convention).
            channel_polarity: Optional dict channel -> +1 or -1.

        Returns:
            (pair_weights, channel_snrs, windowed_snrs, series) where series(mode)
            returns (corr_data, packed) of the group for an envelope mode (computed
            once per mode).

        Raises:
            ValueError: If a pair of the group is not in ``pairs``.
        """
        index = {p: i for i, p in enumerate(pairs.pairs)}
        group_pairs = list(itertools.combinations(channels, 2))
        missing = [p for p in group_pairs if p not in index]
        if missing:
            raise ValueError(f"pairs {missing[:3]} are not in the pair set of channels "
                             f"{list(pairs.channels)}; list the channels in that order")
        rows = np.array([index[p] for p in group_pairs], dtype=np.int64)
        all_rows = len(rows) == len(pairs.pairs) and bool(np.all(rows == np.arange(len(rows))))

        weighting = config.get('snr_pair_weighting', False)
        channel_snrs = ({ch: pairs.snr[ch] for ch in channels}
                        if weighting or config.get('validation', False) else {})
        windowed = ({ch: pairs.snr_windowed[ch] for ch in channels}
                    if config.get('snr_window_ns', None) is not None else {})
        if pair_weights is not None:
            weights = [pair_weights[p] if p in pair_weights else pair_weights[p[::-1]]
                       for p in group_pairs]
        else:
            weights = self._group_pair_weights(pairs.snr, pairs.snr_windowed, channels, config)

        signs = config.get('pair_signs', None)
        if channel_polarity:
            products = [channel_polarity.get(a, 1) * channel_polarity.get(b, 1)
                        for a, b in group_pairs]
            signs = products if signs is None else [
                s if isinstance(s, str) else s * q for s, q in zip(signs, products)]
        shift = None
        if channel_delay_shift:
            shift = np.array([channel_delay_shift.get(b, 0.0) - channel_delay_shift.get(a, 0.0)
                              for a, b in group_pairs], dtype=np.float64)
        cache = {}

        def series(mode):
            """Return (corr_data, packed) of the group's pairs for one envelope mode."""
            if mode not in cache:
                packed = pairs.series[mode]
                if not all_rows:
                    packed = CorrPacked(*(a[rows] for a in packed))
                if signs is not None and mode is None:
                    if all_rows:
                        packed = packed._replace(corr=packed.corr.copy())
                    self._apply_pair_signs(packed, signs)
                if shift is not None:
                    packed = packed._replace(offsets=packed.offsets + shift)
                cache[mode] = (self._corr_data(packed), packed)
                executor = reco3d_batch.current_executor()
                if executor is not None:
                    executor.register_series(packed, pairs.series[mode], rows,
                                             signs if mode is None else None)
            return cache[mode]

        return weights, channel_snrs, windowed, series

    def pair_lag_windows(self, pairs, config, far_field=None):
        """Lowest and highest delay the search of a config can read for each channel pair.

        The search reads pair (a, b) at the delay T_a - T_b of the loaded
        travel-time tables for sources inside the union of ``coarse_limits`` and
        ``limits`` (every refine, polish and optimizer step is clamped to them; the
        Nelder-Mead optimizer is not, and is not covered). The kernels read a
        table only where the query lies in the table and the four corners of its
        cell are finite (a query on the last row or column reads that cell's edge),
        and a bilinear value lies between the corners of its cell. The horizontal
        distances of a source to the two antennas differ by at most the antennas'
        horizontal separation D. So the delay at any source whose distance to a
        lies in r cell i, at a depth in z cell j, lies between the corner extremes
        of cell (i, j) of table a minus those of the cells (i', j) of table b with
        ``|i' - i| <= ceil(D / dr) + 1``. Cells with a non-finite corner or outside a
        table never enter (they bound nothing the kernels read); at a table's top
        node the top row alone stands for the cell above it, since a source there
        reads that row while a taller table reads its next cell. The bound runs
        over every cell the volume reaches in r (whatever the azimuth) and z, with
        one cell of slack on each side, and over every table loaded for the two
        channels. Tables may cover different r and z ranges on one aligned grid.

        With the far field (``far_field_hypothesis`` or ``far_field=True``) the
        windows also hold every plane-wave delay of the sky (``far_field_lag_windows``).

        Args:
            pairs: Sequence of (ch_a, ch_b) channel pairs.
            config: Reconstruction config dict (volume keys).
            far_field: Include the plane-wave delays; None follows ``far_field_hypothesis``.

        Returns:
            (n_pairs, 2) float64 array of [lowest, highest] delay in ns, NaN for a
            pair with no valid table cell in the volume (and no far field). Cached
            per pair list, volume and far-field flag until ``end``.

        Raises:
            ValueError: If the tables do not share one r and z spacing or their
                nodes do not fall on one common grid.
        """
        from scipy.ndimage import maximum_filter1d, minimum_filter1d

        self._set_position_shift(config.get('channel_position_shift'))

        coarse = config.get('coarse_limits', [1, 1500, 0, 360, -1500, 0])
        limits = config.get('limits', coarse)
        volume = (max(min(coarse[0], limits[0]), 1.0), float(max(coarse[1], limits[1])),
                  float(min(coarse[4], limits[4])), float(max(coarse[5], limits[5])))
        if far_field is None:
            far_field = config.get('far_field_hypothesis', False)
        key = (tuple(tuple(p) for p in pairs), volume, bool(far_field))
        if key in self._lag_window_cache:
            return self._lag_window_cache[key]
        rho_lo, rho_hi, z_lo, z_hi = volume

        channels = sorted({ch for p in pairs for ch in p})
        tables = {ch: self._channel_tables(ch) for ch in channels}
        all_tables = [td for ch in channels for td in tables[ch]]
        dr_inv, dz_inv = all_tables[0].dr_inv, all_tables[0].dz_inv
        if any(td.dr_inv != dr_inv or td.dz_inv != dz_inv for td in all_tables):
            raise ValueError(f"the travel-time tables of channels {channels} do not share one "
                             "r and z spacing")
        r_min = min(td.r_min for td in all_tables)
        z_min = min(td.z_min for td in all_tables)
        origin = {}
        for td in all_tables:
            oi, oj = (td.r_min - r_min) * dr_inv, (td.z_min - z_min) * dz_inv
            if abs(oi - round(oi)) > 1e-6 or abs(oj - round(oj)) > 1e-6:
                raise ValueError(f"the travel-time tables of channels {channels} do not lie on "
                                 "one common grid")
            origin[id(td)] = (int(round(oi)), int(round(oj)))
        n_r = max(origin[id(td)][0] + td.nr for td in all_tables)
        n_z = max(origin[id(td)][1] + td.nz for td in all_tables)

        def cell(x, x_min, inv, n_cells):
            """Index of the grid cell holding x, clipped to [0, n_cells - 1]."""
            return int(np.clip(np.floor((x - x_min) * inv), 0, n_cells - 1))

        pa_xy = self._pa_center[:2]
        r_cells = {}
        for ch in channels:
            e = float(np.hypot(*(self.ant_locs[ch][:2] - pa_xy)))
            r_cells[ch] = (max(cell(max(1.0, rho_lo - e, e - rho_hi), r_min, dr_inv, n_r - 1) - 1, 0),
                           min(cell(max(1.0, rho_hi + e), r_min, dr_inv, n_r - 1) + 1, n_r - 2))
        j0 = max(cell(z_lo, z_min, dz_inv, n_z) - 1, 0)
        j1 = min(cell(z_hi, z_min, dz_inv, n_z) + 1, n_z - 1)
        half = {tuple(p): int(np.ceil(np.hypot(*(self.ant_locs[p[0]][:2] - self.ant_locs[p[1]][:2]))
                                      * dr_inv)) + 1 for p in pairs}
        i0 = max(min(lo for lo, _ in r_cells.values()) - max(half.values()), 0)
        i1 = min(max(hi for _, hi in r_cells.values()) + max(half.values()), n_r - 2)
        rows = np.arange(i0, i1 + 1)

        def envelope(td, reach):
            """Corner minimum and maximum of a table's cells on rows i0..i1 and columns j0..j1 of the grid.

            Column j is the table's z cell j, or its top row alone at the table's top
            node; cells outside the table or the channel's reach, or with a
            non-finite corner, hold +inf and -inf.
            """
            lo = np.full((len(rows), j1 - j0 + 1), np.inf)
            hi = np.full((len(rows), j1 - j0 + 1), -np.inf)
            oi, oj = origin[id(td)]
            a0, a1 = max(i0 - oi, 0), min(i1 - oi, td.nr - 2)
            b0, b1 = max(j0 - oj, 0), min(j1 - oj, td.nz - 1)
            if a0 > a1 or b0 > b1:
                return lo, hi
            v = td.values[a0:a1 + 2, b0:b1 + 2]
            if b1 + 2 > td.nz:
                v = np.concatenate([v, v[:, -1:]], axis=1)
            ok = np.isfinite(v)
            valid = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
            valid &= reach[a0 + oi - i0:a1 + oi - i0 + 1, None]
            corners = (v[:-1, :-1], v[1:, :-1], v[:-1, 1:], v[1:, 1:])
            block = (slice(a0 + oi - i0, a1 + oi - i0 + 1), slice(b0 + oj - j0, b1 + oj - j0 + 1))
            lo[block] = np.where(valid, np.minimum.reduce(corners), np.inf)
            hi[block] = np.where(valid, np.maximum.reduce(corners), -np.inf)
            return lo, hi

        envelopes = {ch: [envelope(td, (rows >= r_cells[ch][0]) & (rows <= r_cells[ch][1]))
                          for td in tables[ch]] for ch in channels}

        out = np.full((len(pairs), 2), np.nan)
        for k, (a, b) in enumerate(pairs):
            size = 2 * half[(a, b)] + 1
            lo, hi = np.inf, -np.inf
            for b_min, b_max in envelopes[b]:
                b_max = maximum_filter1d(b_max, size, axis=0, mode='constant', cval=-np.inf)
                b_min = minimum_filter1d(b_min, size, axis=0, mode='constant', cval=np.inf)
                for a_min, a_max in envelopes[a]:
                    lo = min(lo, float(np.min(a_min - b_max)))
                    hi = max(hi, float(np.max(a_max - b_min)))
            if lo <= hi:
                out[k] = (lo, hi)
        if far_field:
            far = self.far_field_lag_windows(pairs)
            out = np.column_stack([np.fmin(out[:, 0], far[:, 0]), np.fmax(out[:, 1], far[:, 1])])
        self._lag_window_cache[key] = out
        return out

    def _check_lag_windows(self, pairs, config, channel_delay_shift):
        """Raise when cut series cannot hold every delay the search of a config can read.

        Raises:
            ValueError: If for a pair of ``config['channels']`` the reachable delays
                (``pair_lag_windows``) minus the pair's delay shift leave the window
                its series were cut to.
        """
        group_pairs = list(itertools.combinations(config['channels'], 2))
        need = self.pair_lag_windows(group_pairs, config)
        index = {p: i for i, p in enumerate(pairs.pairs)}
        have = pairs.windows[[index[p] for p in group_pairs]]
        shifts = channel_delay_shift or {}
        shift = np.array([shifts.get(b, 0.0) - shifts.get(a, 0.0) for a, b in group_pairs])
        inside = (need[:, 0] - shift >= have[:, 0]) & (need[:, 1] - shift <= have[:, 1])
        bad = np.isfinite(need[:, 0]) & ~inside
        if bad.any():
            worst = [(group_pairs[i], need[i].tolist(), have[i].tolist(), float(shift[i]))
                     for i in np.flatnonzero(bad)[:3]]
            raise ValueError(f"{int(bad.sum())} pairs need delays outside their stored lag "
                             f"windows (pair, needed, stored, shift): {worst}")
