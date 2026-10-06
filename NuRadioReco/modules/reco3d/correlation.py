"""Pair correlations and the correlator back ends of the 3D reconstruction."""

import numpy as np
import itertools
import os
from collections import namedtuple
from functools import lru_cache

from scipy import fft as sp_fft
from scipy.signal import hilbert, windows

from NuRadioReco.modules import reco3d_batch
from NuRadioReco.utilities.reco3d_kernels import USE_NUMBA, USE_CUPY

from NuRadioReco.modules.reco3d.shared import _STACK_BLOCK_POINTS

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import (
        _interp_uniform_numba,
        _all_pairs_corr_numba,
        _bilinear_ok_batch_numba,
        _singleray_stackT_corr_numba,
        _singleray_grid_tts_numba,
        get_num_threads,
    )


CorrPacked = namedtuple('CorrPacked', ['corr', 'lengths', 'dts', 'offsets', 'inv_dts'])

SERIES_MODES = (None, 'traces', 'correlation')


@lru_cache(maxsize=None)
def _pair_indices(n_ch):
    """(pair_ch1, pair_ch2) int64 channel indices of itertools.combinations(range(n_ch), 2), shared: never write to them."""
    idx = list(itertools.combinations(range(n_ch), 2))
    pair_ch1 = np.array([p[0] for p in idx], dtype=np.int64)
    pair_ch2 = np.array([p[1] for p in idx], dtype=np.int64)
    return pair_ch1, pair_ch2


class TTStack(tuple):
    """Point-major (tts, valid) of a cached travel-time stack, carrying its channel-major arrays.

    Indexing and unpacking give the (n_points, n_ch) views ``tts`` and ``valid``;
    ``ttsT`` and ``validT`` are the contiguous (n_ch, n_points) arrays they view.
    """

    def __new__(cls, ttsT, validT):
        stack = super().__new__(cls, (ttsT.T, validT.T))
        stack.ttsT = ttsT
        stack.validT = validT
        return stack


class PairSeriesBatch:
    """Batched transforms of ``InterferometricReco3D._pair_series`` for channels of one trace length."""

    @staticmethod
    def fill(packed, group, normed, energy, channel_pairs, nfft, norm_mode, apply_hann_window,
             overlap, hann):
        """Fill the packed series of one transform group, as the per-pair loop of _pair_series does.

        The channel spectra and the pair inverse transforms run as batches along the last
        axis (pocketfft transforms each row exactly as a single call, threads split the
        rows); the pair spectra are formed one pair at a time as before, and the overlap
        or energy normalisation and the Hann taper are the same element-wise operations.

        Args:
            packed: Dict mode -> (n_pairs, M_max) zero array, filled in place.
            group: Modes of the group: None and "correlation" (trace correlation) or "traces".
            normed: Mean-subtracted (and in pearson mode standardised) traces, one length n.
            energy: (n_ch,) sum of squares of each normed trace.
            channel_pairs: itertools.combinations of the channel indices.
            nfft: Padded transform length.
            norm_mode: "pearson", "energy" or "overlap_only".
            apply_hann_window: Multiply by the Hann taper.
            overlap, hann: Per-call caches of the overlap counts and tapers of _pair_series.
        """
        workers = max(1, len(os.sched_getaffinity(0))) if hasattr(os, 'sched_getaffinity') else 1
        n = normed[0].shape[0]
        M = 2 * n - 1
        traces = np.stack(normed)
        spec = sp_fft.rfft(traces, nfft, axis=-1, workers=workers)
        spec_rev = sp_fft.rfft(traces[:, ::-1], nfft, axis=-1, workers=workers)
        product = np.empty((len(channel_pairs), spec.shape[1]), dtype=spec.dtype)
        for pidx, (cidx1, cidx2) in enumerate(channel_pairs):
            np.multiply(spec[cidx1], spec_rev[cidx2], out=product[pidx])
        rows = []
        for mode in group:
            corr = packed[mode][:, :M]
            if mode == "correlation":
                product[:, 1:(nfft + 1) // 2] *= 2.0
                corr[:] = np.abs(sp_fft.ifft(product, nfft, axis=-1, workers=workers)[:, :M])
            else:
                corr[:] = sp_fft.irfft(product, nfft, axis=-1, workers=workers)[:, :M]
            rows.append(corr)
        idx1 = np.array([p[0] for p in channel_pairs], dtype=np.int64)
        idx2 = np.array([p[1] for p in channel_pairs], dtype=np.int64)
        for corr in rows:
            if norm_mode == "energy":
                energy_norm = np.sqrt(energy[idx1] * energy[idx2])
                scaled = energy_norm > 0
                corr[scaled] /= energy_norm[scaled][:, None]
            else:
                if (n, n) not in overlap:
                    overlap[(n, n)] = np.concatenate(
                        [np.arange(1, n + 1), np.full(0, n), np.arange(n - 1, 0, -1)], dtype=np.float64)
                corr /= overlap[(n, n)]
            if apply_hann_window:
                if M not in hann:
                    hann[M] = windows.hann(M)
                corr *= hann[M]


class CorrelationMixin:
    """Methods of InterferometricReco3D for pair correlations and the correlator back ends."""

    def _correlator(self, corr_data, delay_matrices, pair_weights=None):
        """Compute correlation map over the 3D grid.

        Parameters
        ----------
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair from
            ``_prepare_corr_funcs``.
        delay_matrices : list of array
            3D delay matrices, one per pair.
        pair_weights : list or None
            Per-pair weights.

        Returns
        -------
        tuple
            (mean_corr_map, max_corr, pair_corr_maps)
        """
        n_pairs = len(corr_data)
        grid_shape = delay_matrices[0].shape

        pair_corr = np.full((n_pairs, *grid_shape), np.nan, dtype=np.float64)

        for pidx in range(n_pairs):
            corr_arr, dt, offset = corr_data[pidx]
            delays = delay_matrices[pidx]
            valid = np.isfinite(delays)

            if np.any(valid):
                flat_delays = delays[valid].ravel().astype(np.float64)
                if USE_NUMBA:
                    vals = _interp_uniform_numba(corr_arr, dt, offset,
                                                 flat_delays)
                else:
                    M = len(corr_arr)
                    time_lags = np.arange(M) * dt + offset
                    vals = np.interp(flat_delays, time_lags, corr_arr)

                pair_corr[pidx][valid] = vals

        if pair_weights is not None:
            w = np.asarray(pair_weights, dtype=np.float64).reshape(-1, 1, 1, 1)
            w_sum = np.nansum(w)
            mean_corr = np.nansum(w * pair_corr, axis=0) / w_sum if w_sum > 0 else np.nanmean(pair_corr, axis=0)
        else:
            mean_corr = np.nansum(pair_corr, axis=0) / n_pairs

        max_corr = float(np.nanmax(mean_corr)) if not np.all(np.isnan(mean_corr)) else np.nan

        return mean_corr, max_corr, pair_corr

    @staticmethod
    def _pack_corr_data(corr_data):
        """Pack a list of (corr_array, dt, offset) into a zero-padded CorrPacked."""
        n_pairs = len(corr_data)
        lengths = np.array([c[0].shape[0] for c in corr_data], dtype=np.int64)
        corr_packed = np.zeros((n_pairs, int(lengths.max())), dtype=np.float64)
        dts = np.empty(n_pairs, dtype=np.float64)
        offsets = np.empty(n_pairs, dtype=np.float64)
        for pidx, (corr_arr, dt, offset) in enumerate(corr_data):
            corr_packed[pidx, :corr_arr.shape[0]] = corr_arr
            dts[pidx] = dt
            offsets[pidx] = offset
        return CorrPacked(corr_packed, lengths, dts, offsets, 1.0 / dts)

    def _interp_delays(self, corr_arr, dt, offset, delays):
        """Interpolate a correlation function at given delay values.

        Parameters
        ----------
        corr_arr : np.ndarray
            1D correlation array.
        dt : float
            Sample spacing.
        offset : float
            Time offset of first sample.
        delays : np.ndarray
            Flat array of delay values to interpolate at.

        Returns
        -------
        np.ndarray
            Interpolated correlation values.
        """
        if USE_NUMBA:
            return _interp_uniform_numba(corr_arr, dt, offset, delays)
        M = len(corr_arr)
        time_lags = np.arange(M) * dt + offset
        return np.interp(delays, time_lags, corr_arr)

    def _correlator_lean(self, corr_data, delay_matrices, pair_weights=None,
                         delay_cache_key=None, packed=None):
        """Memory-efficient correlator that accumulates in place.

        Unlike ``_correlator``, does not allocate the full (n_pairs, *grid)
        array. Returns only the weighted mean correlation map.

        Parameters
        ----------
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        delay_matrices : list of array
            3D delay matrices, one per pair.
        pair_weights : list or None
            Per-pair weights.
        delay_cache_key : hashable or None
            If provided and GPU path is active, cache the stacked delay
            matrices on the GPU under this key. The caller is responsible
            for ensuring the key uniquely identifies the geometry of
            ``delay_matrices``. Pass None to skip caching (safe default).
        packed : CorrPacked or None
            Packed form of ``corr_data`` for the fused CPU kernel; None packs
            on every call.

        Returns
        -------
        tuple
            (mean_corr_map, max_corr)
        """
        if self._use_gpu and USE_CUPY:
            # Small grids (refine stages) are dominated by kernel launch
            # overhead; run them on CPU. Threshold chosen so the coarse
            # scan (>=100k cells) always goes to GPU.
            grid_size = int(np.prod(delay_matrices[0].shape))
            if grid_size >= self._gpu_min_grid_cells:
                return self._correlator_lean_gpu(
                    corr_data, delay_matrices, pair_weights=pair_weights,
                    delay_cache_key=delay_cache_key)

        n_pairs = len(corr_data)
        grid_shape = delay_matrices[0].shape

        if pair_weights is not None:
            w = np.asarray(pair_weights, dtype=np.float64)
            w_sum = float(w.sum())
        else:
            w = np.ones(n_pairs, dtype=np.float64)
            w_sum = float(n_pairs)

        if USE_NUMBA and self._use_fused_correlator:
            # Fused all-pairs kernel: one launch, parallel over cells.
            # Points-major layout so the inner pair loop is cache-friendly.
            n_points = int(np.prod(grid_shape))
            # Cache the stacked delay_T by cache key to avoid rebuilding
            # every call for stable coarse grids. Key None means skip cache.
            delay_T = None
            if delay_cache_key is not None:
                if not hasattr(self, '_cpu_delay_T_cache'):
                    self._cpu_delay_T_cache = {}
                cached = self._cpu_delay_T_cache.get(delay_cache_key)
                if cached is not None and cached.shape[0] == n_points and cached.shape[1] == n_pairs:
                    delay_T = cached
            if delay_T is None:
                delay_T = np.empty((n_points, n_pairs), dtype=np.float64)
                for pidx in range(n_pairs):
                    delay_T[:, pidx] = delay_matrices[pidx].reshape(-1)
                if delay_cache_key is not None:
                    self._cpu_delay_T_cache[delay_cache_key] = delay_T

            if packed is None:
                packed = self._pack_corr_data(corr_data)
            flat = _all_pairs_corr_numba(
                delay_T, packed.corr, packed.lengths, packed.dts,
                packed.offsets, w)
            mean_corr = flat.reshape(grid_shape)
        else:
            mean_corr = np.zeros(grid_shape, dtype=np.float64)
            for pidx in range(n_pairs):
                corr_arr, dt, offset = corr_data[pidx]
                delays = delay_matrices[pidx]
                valid = np.isfinite(delays)

                if not np.any(valid):
                    continue

                flat_delays = delays[valid].ravel().astype(np.float64)
                vals = self._interp_delays(corr_arr, dt, offset, flat_delays)
                np.nan_to_num(vals, copy=False, nan=0.0)
                mean_corr[valid] += vals * w[pidx]

            if w_sum > 0:
                mean_corr /= w_sum

        max_corr = float(np.max(mean_corr)) if mean_corr.size > 0 else np.nan
        return mean_corr, max_corr

    def _singleray_kernel_active(self):
        """Whether the fused singleray CPU kernels serve the grids (numba, fused, no GPU, single table)."""
        return (USE_NUMBA and self._use_fused_correlator and not self._use_gpu
                and not self._multi_ray_types)

    def _singleray_geom(self, channels):
        """Geometry arguments of the fused singleray kernels for a channel group, in kernel order."""
        g = self._pack_singleray_tables(channels)
        return (g['pa_x'], g['pa_y'], g['ant_xy'], g['td_values'], g['td_ok'],
                g['td_slot'], g['td_r_min'], g['td_dr_inv'], g['td_nr'],
                g['td_z_min'], g['td_dz_inv'], g['td_nz'], self._tolerant_table_edge)

    def _singleray_corr_args(self, channels, packed_list, pair_weights):
        """Correlation arguments of the fused singleray kernels, in kernel order.

        Args:
            channels: Channel IDs of the group.
            packed_list: List of CorrPacked sets sharing one lag geometry (K sets);
                a single set is passed as a view without copying.
            pair_weights: Per-pair weights or None.

        Returns:
            Tuple (corr_stack, lengths, inv_dts, offsets, pair_ch1, pair_ch2,
            weights, w_total, valid_norm, valid_floor).
        """
        pair_ch1, pair_ch2 = _pair_indices(len(channels))
        if pair_weights is not None:
            w = np.asarray(pair_weights, dtype=np.float64)
        else:
            w = np.ones(len(pair_ch1), dtype=np.float64)
        if len(packed_list) == 1:
            corr_stack = packed_list[0].corr[None]
        else:
            corr_stack = np.stack([p.corr for p in packed_list])
        first = packed_list[0]
        return (corr_stack, first.lengths, first.inv_dts, first.offsets,
                pair_ch1, pair_ch2, w, float(w.sum()), self._valid_norm, self._valid_floor)

    def _singleray_grid_maps(self, rho_vec, phi_vec_rad, z_vec, channels,
                             packed_list, pair_weights):
        """Correlation maps of K packed sets over a product grid with the fused kernel.

        Inside a batch with ``batch_grids`` the maps come from the batch, which evaluates the
        grids of every waiting search together (equal bit for bit).

        Returns:
            (K, n_rho, n_phi, n_z) float64 array.
        """
        n_points = len(rho_vec) * len(phi_vec_rad) * len(z_vec)
        self.work['map_points'] += n_points * len(packed_list)
        self.work['tt_lookups'] += n_points * len(channels)
        executor = reco3d_batch.current_executor()
        if executor is not None and executor.batch_grids:
            return executor.request('grid_maps', (rho_vec, phi_vec_rad, z_vec, channels, packed_list,
                                                  pair_weights))
        ttsT, validT = _singleray_grid_tts_numba(
            *self._grid_axes(rho_vec, phi_vec_rad, z_vec), *self._singleray_geom(channels))
        # one block per thread: blocks only partition the points, every point's sum is unchanged
        block = max(256, -(-ttsT.shape[1] // get_num_threads()))
        flat = _singleray_stackT_corr_numba(
            ttsT, validT, block, *self._singleray_corr_args(channels, packed_list, pair_weights))
        return flat.reshape(len(packed_list), len(rho_vec), len(phi_vec_rad), len(z_vec))

    def _singleray_tt_stack(self, cache_key, rho_vec, phi_vec_rad, z_vec, channels):
        """Per-channel travel-time stack of a product grid, cached under ``cache_key``.

        The query coordinates come from the numpy geometry of the delay-matrix path
        so the cached stack reproduces its travel times exactly; the lookup runs on
        the masked bilinear kernel with the configured edge rule.

        The stack is held channel-major (``TTStack.ttsT``, ``TTStack.validT``), the
        layout of the map kernel; the point-major (tts, valid) are views of it. Inside a
        batch (``reconstruct_from_pairs_batch``) the stack of every table channel is computed
        once per grid by the batch instead, and a ``reco3d_batch.GridRef`` is returned for
        ``_singleray_stack_maps``.

        Returns:
            TTStack (tts, valid) of shapes (n_points, n_ch), points in C order of the grid,
            or a GridRef.
        """
        if reco3d_batch.current_executor() is not None and cache_key[0] == 'coarse':
            return reco3d_batch.GridRef(cache_key[:2] + cache_key[3:], tuple(channels),
                                        (rho_vec, phi_vec_rad, z_vec))
        stack = self._tt_stack_cache.get(cache_key)
        if stack is None:
            g = self._pack_singleray_tables(channels)
            coords = self._compute_rho_and_coords(
                self._build_source_enu_matrix(rho_vec, phi_vec_rad, z_vec), channels)
            n_points = coords[channels[0]].shape[0]
            self.work['tt_lookups'] += n_points * len(channels)
            ttsT = np.empty((len(channels), n_points), dtype=np.float64)
            validT = np.empty((len(channels), n_points), dtype=np.bool_)
            for ci, ch in enumerate(channels):
                ttsT[ci], validT[ci] = _bilinear_ok_batch_numba(
                    g['td_values'], g['td_ok'], g['td_slot'][ci], g['td_r_min'][ci],
                    g['td_dr_inv'][ci], g['td_nr'][ci], g['td_z_min'][ci],
                    g['td_dz_inv'][ci], g['td_nz'][ci], coords[ch][:, 0], coords[ch][:, 1],
                    self._tolerant_table_edge)
            stack = TTStack(ttsT, validT)
            self._tt_stack_cache[cache_key] = stack
        return stack

    def _singleray_stack_maps(self, stack, channels, packed_list, pair_weights):
        """Correlation maps of K packed sets over a cached travel-time stack.

        Inside a batch the stack is a ``reco3d_batch.GridRef`` and the maps come from the
        batch, which computes the maps of every waiting search together (equal bit for bit).

        Args:
            stack: TTStack, a point-major (tts, valid) pair, or a GridRef.

        Returns:
            (K, n_points) float64 array.
        """
        if isinstance(stack, reco3d_batch.GridRef):
            self.work['map_points'] += len(packed_list) * int(np.prod([len(a) for a in stack.axes]))
            return reco3d_batch.current_executor().request(
                'stack_maps', (stack, channels, packed_list, pair_weights))
        if not isinstance(stack, TTStack):
            stack = TTStack(np.ascontiguousarray(stack[0].T), np.ascontiguousarray(stack[1].T))
        self.work['map_points'] += len(packed_list) * stack.ttsT.shape[1]
        return _singleray_stackT_corr_numba(
            stack.ttsT, stack.validT, _STACK_BLOCK_POINTS,
            *self._singleray_corr_args(channels, packed_list, pair_weights))

    def _prepare_corr_funcs(self, times, volt_arrays, hilbert_envelope_mode=None,
                            apply_hann_window=False,
                            correlation_normalization="normalized",
                            pair_signs=None):
        """Pre-compute the cross-correlation of every channel pair in one packed set.

        The series of one envelope mode from ``_pair_series``, with ``pair_signs``
        applied to the raw correlation.

        Args:
            times: Time arrays for each channel.
            volt_arrays: Voltage traces for each channel, in the pair order of
                itertools.combinations.
            hilbert_envelope_mode: None, "traces" (envelope of each trace before
                correlating) or "correlation" (envelope of each correlation).
            apply_hann_window: Multiply each correlation by a Hann taper over lag.
            correlation_normalization: See ``_pair_series``.
            pair_signs: Optional per-pair polarity handling, one entry per pair in
                itertools.combinations order: 1 keeps the signed correlation, -1
                negates it and "abs" takes its absolute value. Applied in place to
                the raw correlation only (an envelope carries no polarity); the
                result equals applying it before the (non-negative) Hann taper
                bit for bit.

        Returns:
            (corr_data, packed) where corr_data is a list of (corr_array, dt, offset)
            per pair whose arrays are rows of packed.corr, and packed is a CorrPacked
            of the padded (n_pairs, M_max) array, the per-pair lengths, sample
            spacings, offsets and inverse spacings.
        """
        n_ch = len(volt_arrays)
        n_pairs = n_ch * (n_ch - 1) // 2
        if pair_signs is not None and len(pair_signs) != n_pairs:
            raise ValueError(
                f"pair_signs needs one entry per channel pair ({n_pairs}), got {len(pair_signs)}")
        packed = self._pair_series(times, volt_arrays, (hilbert_envelope_mode,),
                                   apply_hann_window, correlation_normalization)[hilbert_envelope_mode]
        if pair_signs is not None and hilbert_envelope_mode is None:
            self._apply_pair_signs(packed, pair_signs)
        return self._corr_data(packed), packed

    @staticmethod
    def _pair_series(times, volt_arrays, modes, apply_hann_window=False,
                     correlation_normalization="normalized"):
        """Cross-correlation series of every channel pair for one or more envelope modes.

        Each channel is mean-subtracted (and standardised in pearson mode) once,
        transformed once forward and once reversed at the padded FFT length that
        scipy.signal.correlate uses, and every pair takes one inverse transform of
        its spectrum product, which reproduces scipy.signal.correlate(v1, v2, 'full')
        bit for bit. The correlation envelope comes from the same pair spectrum
        (one-sided spectrum doubled and inverted at the padded length), so the raw
        and correlation-envelope modes share the channel transforms and the pair
        products; the traces-envelope mode transforms the trace envelopes. The Hann
        taper and the overlap normalisation are built once per event. Lag ``k`` of
        pair (a, b) is ``offset + k dt`` with the convention lag = t_a - t_b.

        Args:
            times: Time arrays for each channel.
            volt_arrays: Voltage traces for each channel; the pairs are
                itertools.combinations of their indices.
            modes: Envelope modes to compute: None (raw correlation), "traces"
                (envelope of each trace before correlating) and "correlation"
                (envelope of each correlation).
            apply_hann_window: Multiply each correlation by a Hann taper over lag.
            correlation_normalization: "pearson" (legacy alias "normalized"):
                standardise and divide by the overlap count, values in [-1, 1];
                "energy": divide by sqrt(E1 E2), keeps relative amplitude across
                pairs; "overlap_only": divide by the overlap count only.

        Returns:
            Dict mode -> CorrPacked of the padded (n_pairs, M_max) array, the
            per-pair lengths, sample spacings, offsets and inverse spacings.
        """
        n_ch = len(volt_arrays)
        channel_pairs = list(itertools.combinations(range(n_ch), 2))
        n_pairs = len(channel_pairs)
        dts = np.array([t[1] - t[0] if len(t) > 1 else 1.0 for t in times])
        norm_mode = correlation_normalization
        if norm_mode == "normalized":
            norm_mode = "pearson"

        lengths_ch = np.array([len(v) for v in volt_arrays])
        M_max = int(2 * lengths_ch.max() - 1)
        nfft = sp_fft.next_fast_len(M_max, real=True)

        lengths = np.empty(n_pairs, dtype=np.int64)
        pair_dts = np.empty(n_pairs, dtype=np.float64)
        offsets = np.empty(n_pairs, dtype=np.float64)
        for pidx, (cidx1, cidx2) in enumerate(channel_pairs):
            M = int(lengths_ch[cidx1] + lengths_ch[cidx2] - 1)
            dt = min(dts[cidx1], dts[cidx2])
            lengths[pidx] = M
            pair_dts[pidx] = dt
            offsets[pidx] = -(M // 2) * dt + (times[cidx1][0] - times[cidx2][0])
        inv_dts = 1.0 / pair_dts

        hann = {}
        overlap = {}
        series = {}
        for envelope_traces, group in ((False, [m for m in (None, "correlation") if m in modes]),
                                       (True, [m for m in ("traces",) if m in modes])):
            # the raw series must be inverted before the correlation envelope doubles the product in place
            if not group:
                continue
            normed = []
            energy = np.empty(n_ch, dtype=np.float64)
            for ci, v in enumerate(volt_arrays):
                if envelope_traces:
                    v = np.abs(hilbert(v))
                vn = v - v.mean()
                if norm_mode == "pearson":
                    std = vn.std()
                    if std > 0:
                        vn = vn / std
                energy[ci] = np.sum(vn**2)
                normed.append(vn)

            packed = {m: np.zeros((n_pairs, M_max), dtype=np.float64) for m in group}
            if np.all(lengths_ch == lengths_ch[0]):
                # one length: every transform runs batched along the last axis, row for row
                # the same as the single transforms
                PairSeriesBatch.fill(packed, group, normed, energy, channel_pairs, nfft, norm_mode,
                                     apply_hann_window, overlap, hann)
                for mode in group:
                    series[mode] = CorrPacked(packed[mode], lengths, pair_dts, offsets, inv_dts)
                continue
            spec = [sp_fft.rfft(vn, nfft) for vn in normed]
            spec_rev = [sp_fft.rfft(vn[::-1], nfft) for vn in normed]
            for pidx, (cidx1, cidx2) in enumerate(channel_pairs):
                n1, n2 = int(lengths_ch[cidx1]), int(lengths_ch[cidx2])
                M = n1 + n2 - 1
                product = spec[cidx1] * spec_rev[cidx2]
                rows = []
                for mode in group:
                    corr = packed[mode][pidx, :M]
                    if mode == "correlation":
                        product[1:(nfft + 1) // 2] *= 2.0
                        corr[:] = np.abs(sp_fft.ifft(product, nfft)[:M])
                    else:
                        corr[:] = sp_fft.irfft(product, nfft)[:M]
                    rows.append(corr)
                for corr in rows:
                    if norm_mode == "energy":
                        energy_norm = np.sqrt(energy[cidx1] * energy[cidx2])
                        if energy_norm > 0:
                            corr /= energy_norm
                    else:
                        if (n1, n2) not in overlap:
                            overlap[(n1, n2)] = np.concatenate(
                                [np.arange(1, min(n1, n2) + 1),
                                 np.full(abs(n1 - n2), min(n1, n2)),
                                 np.arange(min(n1, n2) - 1, 0, -1)], dtype=np.float64)
                        corr /= overlap[(n1, n2)]
                    if apply_hann_window:
                        if M not in hann:
                            hann[M] = windows.hann(M)
                        corr *= hann[M]
            for mode in group:
                series[mode] = CorrPacked(packed[mode], lengths, pair_dts, offsets, inv_dts)
        return series

    @staticmethod
    def _corr_data(packed):
        """Per-pair (corr_array, dt, offset) list whose arrays are views of the rows of a CorrPacked."""
        return [(packed.corr[p, :packed.lengths[p]], float(packed.dts[p]), float(packed.offsets[p]))
                for p in range(len(packed.lengths))]

    @staticmethod
    def _apply_pair_signs(packed, signs):
        """Apply per-pair signs (1, -1 or 'abs') in place to the rows of a CorrPacked."""
        for p, sign in enumerate(signs):
            corr = packed.corr[p, :packed.lengths[p]]
            if isinstance(sign, str):
                np.abs(corr, out=corr)
            elif sign == -1:
                np.negative(corr, out=corr)
