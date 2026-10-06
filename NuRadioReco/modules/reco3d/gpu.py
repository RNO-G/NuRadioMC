"""The per-event GPU back end of the 3D reconstruction."""

import numpy as np
import itertools

from NuRadioReco.utilities.reco3d_kernels import USE_CUPY, _FUSED_CORR_KERNEL

from NuRadioReco.modules.reco3d.shared import logger

try:
    from NuRadioReco.utilities.reco3d_kernels import _FUSED_MULTIRAY_CORR_KERNEL
except ImportError:
    _FUSED_MULTIRAY_CORR_KERNEL = None

if USE_CUPY:
    import cupy as cp


class GpuMixin:
    """Methods of InterferometricReco3D for the per-event GPU back end."""

    def _warmup_gpu_kernels(self):
        """Launch the fused CUDA RawKernel on tiny dummy data.

        Triggers nvrtc JIT compilation and cache-warming for the fused
        correlator. The compiled kernel is then reused for real events
        without the ~200-500 ms first-launch compile cost.
        """
        n_pairs = 3
        n_points = 16
        delay = cp.zeros((n_pairs, n_points), dtype=cp.float64)
        corr_packed = cp.ones((n_pairs, 32), dtype=cp.float64)
        corr_lens = cp.full(n_pairs, 32, dtype=cp.int64)
        dts = cp.ones(n_pairs, dtype=cp.float64)
        offsets = cp.zeros(n_pairs, dtype=cp.float64)
        weights = cp.ones(n_pairs, dtype=cp.float64)
        out = cp.empty(n_points, dtype=cp.float64)
        threads = 256
        blocks = (n_points + threads - 1) // threads
        _FUSED_CORR_KERNEL(
            (blocks,), (threads,),
            (delay, corr_packed, corr_lens, dts, offsets, weights,
             cp.float64(3.0), np.int32(n_pairs), np.int64(n_points),
             np.int64(corr_packed.strides[0] // corr_packed.itemsize),
             out)
        )
        cp.cuda.Device().synchronize()
        logger.debug("GPU kernels warmed up")

    def _stack_delay_matrices_gpu(self, delay_matrices, cache_key=None):
        """Stack per-pair delay matrices into one GPU array.

        When cache_key is provided, stores the stacked GPU tensor under that
        key and returns the cached value on subsequent calls, skipping the
        host→device transfer. When cache_key is None, transfers fresh every
        call. The caller owns the key: it must be stable across calls that
        share the same geometry (same grid, same pair ordering) and unique
        otherwise.

        Args:
            delay_matrices: List of numpy delay matrices, one per pair.
            cache_key: Optional hashable cache key.

        Returns:
            cupy array of shape (n_pairs, *grid_shape), dtype float64.
        """
        if cache_key is not None:
            cached = self._gpu_delay_stack_cache.get(cache_key)
            if cached is not None and cached.shape[0] == len(delay_matrices):
                return cached
        stacked = np.stack(delay_matrices, axis=0).astype(np.float64)
        stacked_gpu = cp.asarray(stacked)
        if cache_key is not None:
            self._gpu_delay_stack_cache[cache_key] = stacked_gpu
        return stacked_gpu

    def _correlator_lean_gpu(self, corr_data, delay_matrices, pair_weights=None,
                             delay_cache_key=None):
        """Batched GPU correlator: one fused CUDA kernel over all pairs and cells.

        Uses a custom CUDA RawKernel when available (much less HBM traffic
        than the cupy elementwise path), otherwise falls back to vectorized
        cupy. The delay-matrix stack is cached on the GPU across calls so
        only the (small) per-event correlation arrays transfer each time.

        Args:
            corr_data: List of (corr_array, dt, offset) per pair.
            delay_matrices: List of 3D delay arrays, one per pair.
            pair_weights: Per-pair weights or None.

        Returns:
            (mean_corr_map, max_corr) with mean_corr_map as a numpy array.
        """
        n_pairs = len(corr_data)
        grid_shape = delay_matrices[0].shape

        if pair_weights is not None:
            w_np = np.asarray(pair_weights, dtype=np.float64)
        else:
            w_np = np.ones(n_pairs, dtype=np.float64)
        w_sum = float(w_np.sum())

        # Stack delay matrices (optionally cached on GPU across calls).
        delay_gpu = self._stack_delay_matrices_gpu(
            delay_matrices, cache_key=delay_cache_key)

        # Stack corr arrays (varies per event). Pad to max length in case
        # pair trace lengths differ.
        corr_lens = [c[0].shape[0] for c in corr_data]
        M_max = max(corr_lens)
        corr_stack = np.zeros((n_pairs, M_max), dtype=np.float64)
        dts = np.empty(n_pairs, dtype=np.float64)
        offsets = np.empty(n_pairs, dtype=np.float64)
        M_per_pair = np.empty(n_pairs, dtype=np.int64)
        for p, (corr_arr, dt, offset) in enumerate(corr_data):
            corr_stack[p, :corr_arr.shape[0]] = corr_arr
            dts[p] = dt
            offsets[p] = offset
            M_per_pair[p] = corr_arr.shape[0]

        corr_stack_gpu = cp.asarray(corr_stack)
        dts_gpu = cp.asarray(dts)
        offsets_gpu = cp.asarray(offsets)
        M_gpu = cp.asarray(M_per_pair)
        w_gpu = cp.asarray(w_np)

        n_points = int(np.prod(grid_shape))

        if _FUSED_CORR_KERNEL is not None:
            # Ensure contiguous layout required by the RawKernel: delay_gpu
            # must be (n_pairs, n_points) C-contiguous. Reshape the cached
            # stack once.
            delay_flat = delay_gpu.reshape(n_pairs, n_points)
            if not delay_flat.flags.c_contiguous:
                delay_flat = cp.ascontiguousarray(delay_flat)
            mean_corr_gpu = cp.empty(n_points, dtype=cp.float64)
            corr_stride = corr_stack_gpu.strides[0] // corr_stack_gpu.itemsize

            threads = 256
            blocks = (n_points + threads - 1) // threads
            _FUSED_CORR_KERNEL(
                (blocks,), (threads,),
                (delay_flat, corr_stack_gpu, M_gpu,
                 dts_gpu, offsets_gpu, w_gpu,
                 cp.float64(w_sum),
                 np.int32(n_pairs), np.int64(n_points),
                 np.int64(corr_stride),
                 mean_corr_gpu)
            )
            mean_corr_gpu = mean_corr_gpu.reshape(grid_shape)
        else:
            # Fallback: elementwise cupy path (many kernel launches).
            dts_gpu = dts_gpu.reshape((n_pairs,) + (1,) * len(grid_shape))
            offsets_gpu = offsets_gpu.reshape(
                (n_pairs,) + (1,) * len(grid_shape))
            M_gpu = M_gpu.reshape((n_pairs,) + (1,) * len(grid_shape))
            w_gpu = w_gpu.reshape((n_pairs,) + (1,) * len(grid_shape))
            kf = (delay_gpu - offsets_gpu) / dts_gpu
            k = cp.floor(kf).astype(cp.int64)
            alpha = kf - k
            in_bounds = (k >= 0) & (k < (M_gpu - 1)) & cp.isfinite(delay_gpu)
            k_safe = cp.where(in_bounds, k, 0)
            pair_idx = cp.arange(n_pairs).reshape(
                (n_pairs,) + (1,) * len(grid_shape))
            y0 = corr_stack_gpu[pair_idx, k_safe]
            y1 = corr_stack_gpu[pair_idx, cp.minimum(k_safe + 1, M_gpu - 1)]
            vals = y0 + (y1 - y0) * alpha
            vals = cp.where(in_bounds, vals, 0.0)
            weighted = vals * w_gpu
            mean_corr_gpu = weighted.sum(axis=0)
            if w_sum > 0:
                mean_corr_gpu /= w_sum

        mean_corr = cp.asnumpy(mean_corr_gpu)
        max_corr = float(np.max(mean_corr)) if mean_corr.size > 0 else np.nan
        return mean_corr, max_corr

    def _multiray_correlate_gpu(self, corr_data, tt_packed_np, channels,
                                pair_weights=None):
        """GPU multiray per-pair correlator using CUDA RawKernel.

        Args:
            corr_data: List of (corr_array, dt, offset) per pair.
            tt_packed_np: numpy array (n_ch, n_rt, n_points).
            channels: Channel list.
            pair_weights: Per-pair weights or None.

        Returns:
            (mean_corr_map_flat, max_corr)
        """
        n_ch, n_rt, n_points = tt_packed_np.shape
        ch_pairs = list(itertools.combinations(range(n_ch), 2))
        n_pairs = len(ch_pairs)

        if pair_weights is not None:
            w_np = np.asarray(pair_weights, dtype=np.float64)
        else:
            w_np = np.ones(n_pairs, dtype=np.float64)
        w_sum = float(w_np.sum())

        corr_lens = [c[0].shape[0] for c in corr_data]
        M_max = max(corr_lens)
        corr_stack = np.zeros((n_pairs, M_max), dtype=np.float64)
        dts = np.empty(n_pairs, dtype=np.float64)
        offsets = np.empty(n_pairs, dtype=np.float64)
        M_per = np.empty(n_pairs, dtype=np.int64)
        pair_ch1 = np.empty(n_pairs, dtype=np.int32)
        pair_ch2 = np.empty(n_pairs, dtype=np.int32)
        for pidx, (c1i, c2i) in enumerate(ch_pairs):
            corr_stack[pidx, :corr_lens[pidx]] = corr_data[pidx][0]
            dts[pidx] = corr_data[pidx][1]
            offsets[pidx] = corr_data[pidx][2]
            M_per[pidx] = corr_lens[pidx]
            pair_ch1[pidx] = c1i
            pair_ch2[pidx] = c2i

        tt_gpu = cp.asarray(tt_packed_np.reshape(n_ch * n_rt, n_points))
        corr_gpu = cp.asarray(corr_stack)
        dts_gpu = cp.asarray(dts)
        off_gpu = cp.asarray(offsets)
        M_gpu = cp.asarray(M_per)
        w_gpu = cp.asarray(w_np)
        ch1_gpu = cp.asarray(pair_ch1)
        ch2_gpu = cp.asarray(pair_ch2)
        out_gpu = cp.empty(n_points, dtype=cp.float64)
        corr_stride = np.int64(M_max)

        threads = 256
        blocks = (n_points + threads - 1) // threads
        _FUSED_MULTIRAY_CORR_KERNEL(
            (blocks,), (threads,),
            (tt_gpu, corr_gpu, M_gpu, dts_gpu, off_gpu, w_gpu,
             cp.float64(w_sum), ch1_gpu, ch2_gpu,
             np.int32(n_pairs), np.int32(n_ch), np.int32(n_rt),
             np.int64(n_points), np.int64(corr_stride),
             out_gpu)
        )
        mean_corr = cp.asnumpy(out_gpu)
        max_corr = float(np.max(mean_corr)) if mean_corr.size > 0 else np.nan
        return mean_corr, max_corr

    def _refine_batched_gpu(self, peak_grids, corr_data, channels,
                            pair_weights, n_extract, level_sep):
        """Batch all refine peaks into one GPU kernel call.

        Concatenates delay matrices from all local grids along the
        point dimension, runs one fused RawKernel, then splits the
        result back per peak for independent peak extraction.

        Args:
            peak_grids: List of (src_enu, rho_vec, phi_vec_rad, z_vec).
            corr_data: Pre-computed correlation data.
            channels: Channel list.
            pair_weights: Per-pair weights or None.
            n_extract: Peaks to extract per local grid.
            level_sep: Peak separation threshold.

        Returns:
            List of (rho, phi_deg, z, corr) peak tuples.
        """
        per_peak_delays = []
        per_peak_npts = []
        for src_enu_r, rho_vec_r, phi_vec_r, z_vec_r in peak_grids:
            delay_data_r = self._compute_delay_matrices(
                src_enu_r, channels)
            per_peak_delays.append(delay_data_r)
            per_peak_npts.append(int(np.prod(delay_data_r[0].shape)))

        n_pairs = len(corr_data)
        total_points = sum(per_peak_npts)

        combined_delays = []
        for p in range(n_pairs):
            combined_delays.append(
                np.concatenate([d[p].ravel() for d in per_peak_delays]))

        combined_delay_list = [d.reshape(1, -1)[0] for d in combined_delays]
        dummy_shape = (total_points,)
        combined_delay_matrices = [d.reshape(dummy_shape)
                                   for d in combined_delay_list]

        mean_corr_flat, _ = self._correlator_lean_gpu(
            corr_data, combined_delay_matrices,
            pair_weights=pair_weights)

        level_peaks = []
        offset = 0
        for i, (src_enu_r, rho_vec_r, phi_vec_r, z_vec_r) in enumerate(
                peak_grids):
            n_pts = per_peak_npts[i]
            local_shape = (len(rho_vec_r), len(phi_vec_r), len(z_vec_r))
            local_corr = mean_corr_flat[offset:offset + n_pts].reshape(
                local_shape)
            phi_vec_deg_r = phi_vec_r * (180.0 / np.pi)
            local_peaks = self._extract_top_n_peaks(
                local_corr, rho_vec_r, phi_vec_deg_r, z_vec_r,
                n_extract, level_sep)
            level_peaks.extend(local_peaks)
            offset += n_pts

        return level_peaks
