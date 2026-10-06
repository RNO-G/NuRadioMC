"""
Compute kernels for 3D interferometric direction reconstruction.

Pure compute functions with no class dependency; the sole intended
consumer is ``NuRadioReco.modules.interferometricDirectionReconstruction3D``.
Function signatures and available symbols may change without notice to
support performance work on that module.
"""

import itertools
import logging

import numpy as np

logger = logging.getLogger("NuRadioReco.utilities.reco3d_kernels")

USE_NUMBA = False
try:
    from numba import get_num_threads, njit, prange, types
    from numba.extending import intrinsic
    USE_NUMBA = True
except ImportError:
    pass

USE_CUPY = False
_FUSED_CORR_KERNEL = None
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        USE_CUPY = True
        logger.info("CuPy GPU backend available (device count=%d)",
                    cp.cuda.runtime.getDeviceCount())

        _FUSED_CORR_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void fused_correlator(
    const double* __restrict__ delay_stack,
    const double* __restrict__ corr_packed,
    const long long* __restrict__ corr_lens,
    const double* __restrict__ dts,
    const double* __restrict__ offsets,
    const double* __restrict__ pair_weights,
    const double w_sum,
    const int n_pairs,
    const long long n_points,
    const long long corr_stride,
    double* __restrict__ out
) {
    long long pt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;

    double acc = 0.0;
    for (int p = 0; p < n_pairs; ++p) {
        double d = delay_stack[(long long)p * n_points + pt];
        if (isnan(d)) continue;
        double kf = (d - offsets[p]) / dts[p];
        long long k = (long long)floor(kf);
        long long clen = corr_lens[p];
        if (k < 0 || k >= clen - 1) continue;
        double alpha = kf - (double)k;
        double y0 = corr_packed[(long long)p * corr_stride + k];
        double y1 = corr_packed[(long long)p * corr_stride + k + 1];
        double v = y0 + (y1 - y0) * alpha;
        acc += v * pair_weights[p];
    }
    out[pt] = (w_sum > 0.0) ? (acc / w_sum) : 0.0;
}
''', 'fused_correlator')

        _FUSED_MULTIRAY_CORR_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void fused_multiray_correlator(
    const double* __restrict__ tt_packed,
    const double* __restrict__ corr_packed,
    const long long* __restrict__ corr_lens,
    const double* __restrict__ dts,
    const double* __restrict__ offsets,
    const double* __restrict__ pair_weights,
    const double w_sum,
    const int* __restrict__ pair_ch1,
    const int* __restrict__ pair_ch2,
    const int n_pairs,
    const int n_ch,
    const int n_rt,
    const long long n_points,
    const long long corr_stride,
    double* __restrict__ out
) {
    long long pt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;

    double acc = 0.0;
    for (int pidx = 0; pidx < n_pairs; ++pidx) {
        int c1 = pair_ch1[pidx];
        int c2 = pair_ch2[pidx];
        double best_val = -1e30;

        for (int rt1 = 0; rt1 < n_rt; ++rt1) {
            double tt1 = tt_packed[((long long)c1 * n_rt + rt1) * n_points + pt];
            if (isnan(tt1) || tt1 <= 0.0) continue;
            for (int rt2 = 0; rt2 < n_rt; ++rt2) {
                double tt2 = tt_packed[((long long)c2 * n_rt + rt2) * n_points + pt];
                if (isnan(tt2) || tt2 <= 0.0) continue;
                double d = tt1 - tt2;
                double kf = (d - offsets[pidx]) / dts[pidx];
                long long k = (long long)floor(kf);
                long long clen = corr_lens[pidx];
                if (k < 0 || k >= clen - 1) continue;
                double alpha = kf - (double)k;
                double y0 = corr_packed[(long long)pidx * corr_stride + k];
                double y1 = corr_packed[(long long)pidx * corr_stride + k + 1];
                double v = y0 + (y1 - y0) * alpha;
                if (v > best_val) best_val = v;
            }
        }
        if (best_val > -1e29) {
            acc += best_val * pair_weights[pidx];
        }
    }
    out[pt] = (w_sum > 0.0) ? (acc / w_sum) : 0.0;
}
''', 'fused_multiray_correlator')
except Exception:
    cp = None
    _FUSED_MULTIRAY_CORR_KERNEL = None

USE_NUMBA_GROUPED = False
try:
    from fast_grouped_multiray import (
        grouped_multiray_numba, perpair_multiray_numba,
        pack_tt_grids, pack_corr_data, build_combo_table,
    )
    if USE_NUMBA:
        USE_NUMBA_GROUPED = True
        logger.info("Numba grouped multiray kernels loaded")
except ImportError:
    pass

RAY_TYPES = ['direct', 'refracted', 'reflected']
SOLUTION_TYPES = ['solution_0', 'solution_1']
_GROUPED_BLOCK_POINTS = 64


if USE_NUMBA:
    @njit(fastmath=False, cache=True)
    def _scalar_grouped_corr_numba(
            tt_vals, tt_valid, corr_packed, corr_lengths,
            corr_dts, corr_offsets, pair_ch1, pair_ch2, pair_weights,
            ch_group, group_rts, group_nrt, n_pairs, w_sum, walk):
        """Scalar grouped correlation at a single grid point.

        The best, over the depth-group ray-type combinations, of the weighted pair sum. A pair's term
        depends on the combination only through the ray types of its channels' groups, so the terms
        are summed once into group blocks (lower group, upper group, their ray types; within a group
        both channels take the group's ray type) and the combinations are walked depth first over the
        groups, each level adding the blocks that end at its group to the running sum of the levels
        above. A branch is left when its running sum plus the largest values the blocks of the deeper
        levels can add stays below the best sum found by more than rounding; the sums of the leaves
        walked are formed as in the full walk, so the best sum is the full walk's, bit for bit. An
        unusable time adds nothing. Compiled without fastmath, so the value does not depend on the
        context the function is compiled into.

        Args:
            tt_vals: float64 array (n_ch, n_rt).
            tt_valid: bool array (n_ch, n_rt).
            corr_packed: float64 array (n_pairs, max_corr_len).
            corr_lengths: int64 array (n_pairs,).
            corr_dts: float64 array (n_pairs,).
            corr_offsets: float64 array (n_pairs,).
            pair_ch1: int64 array (n_pairs,).
            pair_ch2: int64 array (n_pairs,).
            pair_weights: float64 array (n_pairs,).
            ch_group: int64 array (n_ch,), depth group of each channel.
            group_rts: int64 array (n_groups, max options), ray-type indices each group may take.
            group_nrt: int64 array (n_groups,), number of options of each group.
            n_pairs: int.
            w_sum: float64.
            walk: int64 array (2,) of work counters: walk[0] gains the nodes evaluated, walk[1] the
                branches left by the bound (no effect on the value).

        Returns:
            Negative best weighted mean correlation (for minimization).
        """
        n_rt = tt_vals.shape[1]
        n_g = group_nrt.shape[0]
        blocks = np.zeros((n_g, n_g, n_rt, n_rt), dtype=np.float64)
        for pidx in range(n_pairs):
            c1 = pair_ch1[pidx]
            c2 = pair_ch2[pidx]
            g1 = ch_group[c1]
            g2 = ch_group[c2]
            dt = corr_dts[pidx]
            offset = corr_offsets[pidx]
            clen = corr_lengths[pidx]
            for rt1 in range(n_rt):
                for rt2 in range(n_rt):
                    if g1 == g2 and rt1 != rt2:
                        continue
                    if not tt_valid[c1, rt1] or not tt_valid[c2, rt2]:
                        continue
                    delay = tt_vals[c1, rt1] - tt_vals[c2, rt2]
                    kf = (delay - offset) / dt
                    k = int(np.floor(kf))
                    if k < 0 or k >= clen - 1:
                        continue
                    alpha = kf - k
                    val = (corr_packed[pidx, k]
                           + (corr_packed[pidx, k + 1]
                              - corr_packed[pidx, k]) * alpha)
                    if g1 <= g2:
                        blocks[g1, g2, rt1, rt2] += val * pair_weights[pidx]
                    else:
                        blocks[g2, g1, rt2, rt1] += val * pair_weights[pidx]

        # rest[d]: the most the blocks added at levels d and below can add, for pruning
        rest = np.zeros(n_g + 1, dtype=np.float64)
        for gb in range(n_g - 1, -1, -1):
            level = 0.0
            for ga in range(gb + 1):
                most = -np.inf
                for ia in range(group_nrt[ga]):
                    for ib in range(group_nrt[gb]):
                        if ga == gb and ia != ib:
                            continue
                        v = blocks[ga, gb, group_rts[ga, ia], group_rts[gb, ib]]
                        if v > most:
                            most = v
                level += most
            rest[gb] = rest[gb + 1] + level

        best_total = -np.inf
        idx = np.full(n_g, -1, dtype=np.int64)
        partial = np.zeros(n_g + 1, dtype=np.float64)
        d = 0
        while d >= 0:
            idx[d] += 1
            if idx[d] >= group_nrt[d]:
                idx[d] = -1
                d -= 1
                continue
            r = group_rts[d, idx[d]]
            total = partial[d] + blocks[d, d, r, r]
            for ga in range(d):
                total += blocks[ga, d, group_rts[ga, idx[ga]], r]
            walk[0] += 1
            if d == n_g - 1:
                if total > best_total:
                    best_total = total
            elif total + rest[d + 1] >= best_total - 1e-9 * (1.0 + abs(best_total)):
                partial[d + 1] = total
                d += 1
            else:
                walk[1] += 1

        if best_total == -np.inf:
            return 0.0
        return -best_total / w_sum if w_sum > 0.0 else 0.0

    @njit(parallel=True, fastmath=True, cache=True)
    def _interp_uniform_numba(y, dt, offset, x):
        """Fast uniform-grid linear interpolation."""
        M = y.shape[0]
        n = x.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            kf = (x[i] - offset) / dt
            k = int(np.floor(kf))
            if k < 0 or k >= M - 1:
                out[i] = np.nan
            else:
                alpha = kf - k
                out[i] = y[k] + (y[k + 1] - y[k]) * alpha
        return out

    @intrinsic
    def _fma(typingctx, a, b, c):
        """Fused multiply-add a * b + c with a single rounding (llvm.fma, never split into a product and a sum)."""
        sig = types.float64(types.float64, types.float64, types.float64)

        def codegen(context, builder, signature, args):
            """Emit the llvm.fma call."""
            return builder.fma(*args)
        return sig, codegen

    @njit(fastmath=False, cache=True)
    def _coverage_ramp(w_valid, w_total, valid_floor):
        """Coverage factor f of the valid-weight normalisation, within one ulp of its exact value.

        With cov = w_valid / w_total and P = valid_floor * w_total, f is 1 at
        cov >= valid_floor (w_valid >= P), 0 at cov <= valid_floor / 2
        (2 w_valid <= P), and on the ramp between
        f = (cov - valid_floor / 2) / (valid_floor / 2) = (2 w_valid - P) / P.
        Near the lower knee the numerator cancels, so it is formed without
        rounding: p = fl(valid_floor * w_total) and e = fma(valid_floor, w_total, -p)
        give P = p + e exactly (error-free product); d = 2 w_valid - p is exact on
        the ramp (Sterbenz, p <= 2 w_valid <= 2 p) and N = d - e is kept as
        n + n_err with TwoSum. The knees are decided exactly by comparing w_valid
        and 2 w_valid with p and, on a tie, by the sign of e. The quotient
        q = fl(n / p) is corrected with its remainder n - q p, which is a float and
        so exact as fma(-q, p, n): f = q + (n - q p + n_err - q e) / p, with an error
        below one ulp of f. TwoSum and the fma TwoProduct: Ogita, Rump and Oishi,
        SIAM J. Sci. Comput. 26, 1955 (2005); the exact division remainder:
        Bohlender, Walter, Kornerup and Matula, Proc. 10th IEEE Symposium on
        Computer Arithmetic (1991).

        Compiled with ``fastmath=False`` set explicitly: numba compiles a callee
        whose fastmath option is unset with the flags of the caller that triggers
        its compilation, and fast-math reassociation or contraction would break
        the error-free steps differently in each kernel that inlines it.

        Returns:
            f in [0, 1]; 0 when w_valid <= 0 or w_total <= 0.
        """
        if w_valid <= 0.0 or w_total <= 0.0:
            return 0.0
        p = valid_floor * w_total
        e = _fma(valid_floor, w_total, -p)
        if w_valid > p or (w_valid == p and e <= 0.0):
            return 1.0
        two_w = 2.0 * w_valid
        if two_w < p or (two_w == p and e >= 0.0):
            return 0.0
        d = two_w - p
        n = d - e
        z = n - d
        n_err = (d - (n - z)) - (e + z)
        q = n / p
        return q + (_fma(-q, p, n) + n_err - q * e) / p

    @njit(fastmath=False, cache=True)
    def _valid_mean(acc, w_valid, f):
        """Valid-weight normalised sum acc / w_valid times the coverage factor f, in strict IEEE arithmetic.

        Every kernel applies the valid-weight normalisation through this function
        and ``_coverage_ramp`` (both compiled with ``fastmath=False``), so a map
        value depends only on acc, w_valid, w_total and the floor, not on the
        kernel that accumulated them.

        Returns:
            The normalised value, 0 when f is 0.
        """
        if f == 0.0:
            return 0.0
        return acc / w_valid * f

    @njit(fastmath=True, cache=True)
    def _total_factor(w_total):
        """Multiplier 1 / w_total of the record normalisation (0 when w_total <= 0)."""
        if w_total > 0.0:
            return 1.0 / w_total
        return 0.0

    @njit(fastmath=True, cache=True)
    def _scalar_singleray_corr_numba(
            rho, phi_rad, z, pa_x, pa_y,
            ant_xy, td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Fused single-point singleray correlation for the optimizer.

        Args:
            rho, phi_rad, z: Scalar source coordinates (m, rad, m).
            pa_x, pa_y: PA center absolute coordinates (m).
            ant_xy: (n_ch, 2) float64, channel absolute (x, y).
            td_values: (n_tables, nr_max, nz_max) float64, packed TT tables.
            td_ok: Boolean finiteness mask of td_values; a lookup is valid only
                when its four corners are finite and the value is positive.
            td_slot: (n_ch,) int64, table slot of each channel in td_values.
            td_r_min, td_dr_inv: (n_ch,) float64.
            td_nr: (n_ch,) int64.
            td_z_min, td_dz_inv: same in z.
            td_nz: (n_ch,) int64.
            corr_packed: (n_pairs, max_corr_len) padded.
            corr_lengths, corr_dts, corr_offsets: (n_pairs,).
            pair_ch1, pair_ch2: (n_pairs,) int64.
            pair_weights: (n_pairs,) float64.
            w_total: sum(pair_weights).
            valid_norm: Divide by the weight of the contributing pairs times the
                coverage factor instead of by w_total.
            valid_floor: Coverage floor of the valid-weight normalisation.

        Returns:
            Negative weighted mean correlation (for minimization).
        """
        n_ch = ant_xy.shape[0]
        x_src = rho * np.cos(phi_rad) + pa_x
        y_src = rho * np.sin(phi_rad) + pa_y

        tts = np.empty(n_ch, dtype=np.float64)
        valid = np.zeros(n_ch, dtype=np.bool_)
        for ci in range(n_ch):
            dx = x_src - ant_xy[ci, 0]
            dy = y_src - ant_xy[ci, 1]
            r = np.sqrt(dx * dx + dy * dy)
            if r < 1.0:
                r = 1.0
            ri = (r - td_r_min[ci]) * td_dr_inv[ci]
            zi = (z - td_z_min[ci]) * td_dz_inv[ci]
            i0 = int(np.floor(ri))
            j0 = int(np.floor(zi))
            nr_ch = td_nr[ci]
            nz_ch = td_nz[ci]
            if i0 < 0 or j0 < 0:
                continue
            if i0 >= nr_ch - 1:
                if ri <= nr_ch - 1 + 1e-9:
                    i0 = nr_ch - 2
                    fx = 1.0
                else:
                    continue
            else:
                fx = ri - i0
            if j0 >= nz_ch - 1:
                if zi <= nz_ch - 1 + 1e-9:
                    j0 = nz_ch - 2
                    fy = 1.0
                else:
                    continue
            else:
                fy = zi - j0
            ti = td_slot[ci]
            if not (td_ok[ti, i0, j0] and td_ok[ti, i0 + 1, j0]
                    and td_ok[ti, i0, j0 + 1] and td_ok[ti, i0 + 1, j0 + 1]):
                continue
            v = ((1.0 - fx) * (1.0 - fy) * td_values[ti, i0, j0]
                 + fx * (1.0 - fy) * td_values[ti, i0 + 1, j0]
                 + (1.0 - fx) * fy * td_values[ti, i0, j0 + 1]
                 + fx * fy * td_values[ti, i0 + 1, j0 + 1])
            if v > 0.0:
                tts[ci] = v
                valid[ci] = True

        n_pairs = pair_ch1.shape[0]
        total = 0.0
        w_valid = 0.0
        for pidx in range(n_pairs):
            c1 = pair_ch1[pidx]
            c2 = pair_ch2[pidx]
            if not valid[c1] or not valid[c2]:
                continue
            delay = tts[c1] - tts[c2]
            dt = corr_dts[pidx]
            offset = corr_offsets[pidx]
            clen = corr_lengths[pidx]
            kf = (delay - offset) / dt
            k = int(np.floor(kf))
            if k < 0 or k >= clen - 1:
                continue
            alpha = kf - k
            v = (corr_packed[pidx, k]
                 + (corr_packed[pidx, k + 1] - corr_packed[pidx, k]) * alpha)
            total += v * pair_weights[pidx]
            w_valid += pair_weights[pidx]

        if valid_norm:
            return -_valid_mean(total, w_valid, _coverage_ramp(w_valid, w_total, valid_floor))
        if w_total > 0.0:
            return -total / w_total
        return 0.0

    @njit(fastmath=True, cache=True)
    def _scalar_singleray_corr_grad_numba(
            rho, phi_deg, z, pa_x, pa_y,
            ant_xy, td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Objective of ``_scalar_singleray_corr_numba`` and its exact gradient in (rho, phi_deg, z).

        The value equals ``_scalar_singleray_corr_numba`` at phi_rad = phi_deg * pi / 180
        to float rounding (fastmath may sum the pairs in another order), so callers that
        need the record value take it from that kernel. The gradient is the chain rule through the
        objective, a weighted sum over pairs of the linearly interpolated correlation
        series at the delay t_a - t_b: d corr / d delay is the slope of the series cell
        (c[k + 1] - c[k]) / dt; each travel time is the bilinear table value at
        (r, z) with r the horizontal distance to the antenna (clamped at 1 m), so
        d t / d r = dr_inv ((1 - fy)(v10 - v00) + fy (v11 - v01)) and
        d t / d z = dz_inv ((1 - fx)(v01 - v00) + fx (v11 - v10)); and
        d r / d rho = (dx cos phi + dy sin phi) / r, d r / d phi = rho (dy cos phi - dx sin phi) / r.
        Where the value does not change with a coordinate (r clamped at 1 m, a query
        on the last table row or column, a pair or channel that is not read) that
        part of the gradient is zero; on a cell boundary of the table or of a series
        the gradient is the one-sided derivative of the cell the value is read from.
        Under ``valid_norm`` the coverage factor is constant between changes of the
        set of pairs read, so it scales the gradient like the value.

        Args:
            rho, phi_deg, z: Source coordinates (m, deg, m).
            Remaining arguments: as ``_scalar_singleray_corr_numba``.

        Returns:
            (value, d value / d rho, d value / d phi_deg, d value / d z), value being
            the negative weighted mean correlation.
        """
        phi_rad = phi_deg * (np.pi / 180.0)
        n_ch = ant_xy.shape[0]
        cos_phi = np.cos(phi_rad)
        sin_phi = np.sin(phi_rad)
        x_src = rho * cos_phi + pa_x
        y_src = rho * sin_phi + pa_y

        tts = np.empty(n_ch, dtype=np.float64)
        d_rho = np.zeros(n_ch, dtype=np.float64)
        d_phi = np.zeros(n_ch, dtype=np.float64)
        d_z = np.zeros(n_ch, dtype=np.float64)
        valid = np.zeros(n_ch, dtype=np.bool_)
        for ci in range(n_ch):
            dx = x_src - ant_xy[ci, 0]
            dy = y_src - ant_xy[ci, 1]
            r = np.sqrt(dx * dx + dy * dy)
            r_free = True
            if r < 1.0:
                r = 1.0
                r_free = False
            ri = (r - td_r_min[ci]) * td_dr_inv[ci]
            zi = (z - td_z_min[ci]) * td_dz_inv[ci]
            i0 = int(np.floor(ri))
            j0 = int(np.floor(zi))
            nr_ch = td_nr[ci]
            nz_ch = td_nz[ci]
            if i0 < 0 or j0 < 0:
                continue
            if i0 >= nr_ch - 1:
                if ri <= nr_ch - 1 + 1e-9:
                    i0 = nr_ch - 2
                    fx = 1.0
                    r_free = False
                else:
                    continue
            else:
                fx = ri - i0
            z_free = True
            if j0 >= nz_ch - 1:
                if zi <= nz_ch - 1 + 1e-9:
                    j0 = nz_ch - 2
                    fy = 1.0
                    z_free = False
                else:
                    continue
            else:
                fy = zi - j0
            ti = td_slot[ci]
            if not (td_ok[ti, i0, j0] and td_ok[ti, i0 + 1, j0]
                    and td_ok[ti, i0, j0 + 1] and td_ok[ti, i0 + 1, j0 + 1]):
                continue
            v00 = td_values[ti, i0, j0]
            v10 = td_values[ti, i0 + 1, j0]
            v01 = td_values[ti, i0, j0 + 1]
            v11 = td_values[ti, i0 + 1, j0 + 1]
            v = ((1.0 - fx) * (1.0 - fy) * v00
                 + fx * (1.0 - fy) * v10
                 + (1.0 - fx) * fy * v01
                 + fx * fy * v11)
            if v > 0.0:
                tts[ci] = v
                valid[ci] = True
                if r_free:
                    dv_dr = ((1.0 - fy) * (v10 - v00) + fy * (v11 - v01)) * td_dr_inv[ci]
                    d_rho[ci] = dv_dr * (dx * cos_phi + dy * sin_phi) / r
                    d_phi[ci] = dv_dr * rho * (dy * cos_phi - dx * sin_phi) / r
                if z_free:
                    d_z[ci] = ((1.0 - fx) * (v01 - v00) + fx * (v11 - v10)) * td_dz_inv[ci]

        n_pairs = pair_ch1.shape[0]
        total = 0.0
        w_valid = 0.0
        g_rho = 0.0
        g_phi = 0.0
        g_z = 0.0
        for pidx in range(n_pairs):
            c1 = pair_ch1[pidx]
            c2 = pair_ch2[pidx]
            if not valid[c1] or not valid[c2]:
                continue
            delay = tts[c1] - tts[c2]
            dt = corr_dts[pidx]
            offset = corr_offsets[pidx]
            clen = corr_lengths[pidx]
            kf = (delay - offset) / dt
            k = int(np.floor(kf))
            if k < 0 or k >= clen - 1:
                continue
            alpha = kf - k
            v = (corr_packed[pidx, k]
                 + (corr_packed[pidx, k + 1] - corr_packed[pidx, k]) * alpha)
            total += v * pair_weights[pidx]
            w_valid += pair_weights[pidx]
            slope = (corr_packed[pidx, k + 1] - corr_packed[pidx, k]) / dt * pair_weights[pidx]
            g_rho += slope * (d_rho[c1] - d_rho[c2])
            g_phi += slope * (d_phi[c1] - d_phi[c2])
            g_z += slope * (d_z[c1] - d_z[c2])

        deg = np.pi / 180.0
        if valid_norm:
            f = _coverage_ramp(w_valid, w_total, valid_floor)
            return (-_valid_mean(total, w_valid, f), -_valid_mean(g_rho, w_valid, f),
                    -_valid_mean(g_phi * deg, w_valid, f), -_valid_mean(g_z, w_valid, f))
        if w_total > 0.0:
            return -total / w_total, -g_rho / w_total, -g_phi * deg / w_total, -g_z / w_total
        return 0.0, 0.0, 0.0, 0.0

    @njit(cache=True)
    def _lbfgsb_fd_step(xi, lbi, ubi, bounded):
        """Forward-difference step scipy's L-BFGS-B takes in one coordinate without a gradient.

        ``approx_derivative`` with method '2-point' and the absolute step ``eps`` = 1e-8 (a relative
        step where xi + 1e-8 rounds to xi); with ``bounded`` (any bound finite) the step is reversed
        when it would leave [lbi, ubi] and fits on the other side, else replaced by the distance to
        the farther bound.

        Returns:
            Signed step.
        """
        h = 1e-8
        if (xi + h) - xi == 0.0:
            sign = 1.0 if xi >= 0.0 else -1.0
            h = 1.4901161193847656e-08 * sign * max(1.0, abs(xi))
        if bounded:
            lower = xi - lbi
            upper = ubi - xi
            xh = xi + h
            if abs(h) <= max(lower, upper):
                if xh < lbi or xh > ubi:
                    h = -h
            elif upper >= lower:
                h = upper
            else:
                h = -lower
        return h

    @njit(cache=True)
    def _shifted_singleray_corr(
            rho, phi_shifted, z, phi_shift, pa_x, pa_y, ant_xy, td_values, td_ok, td_slot,
            td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Optimizer objective at (rho, phi_shifted + phi_shift wrapped to [0, 360) deg, z).

        Compiled without fastmath so the azimuth wrap and the degree-to-radian
        conversion round as the numpy scalar operations of the Python objective do.

        Returns:
            Negative weighted mean correlation of ``_scalar_singleray_corr_numba``.
        """
        phi_deg = (phi_shifted + phi_shift) % 360.0
        return _scalar_singleray_corr_numba(
            rho, phi_deg * (np.pi / 180.0), z, pa_x, pa_y, ant_xy,
            td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)

    @njit(cache=True)
    def _lbfgsb_singleray_value_grad(
            x, phi_shift, lb, ub, pa_x, pa_y, ant_xy, td_values, td_ok, td_slot,
            td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Optimizer objective and the gradient scipy's L-BFGS-B would estimate for it.

        Reproduces the forward differences ``scipy.optimize.minimize`` takes for
        L-BFGS-B without a Jacobian (``approx_derivative``, method '2-point',
        absolute step ``eps`` = 1e-8, steps adjusted to the bounds), so passing
        this function with ``jac=True`` gives the same objective and gradient
        values, hence the same iterates, in one compiled call per point instead
        of four Python calls. Compiled without fastmath, as numpy evaluates the
        differences.

        Args:
            x: (3,) point (rho, phi_shifted, z) of the optimizer.
            phi_shift: Added to x[1] before the wrap to [0, 360) deg.
            lb, ub: (3,) lower and upper bounds of the optimizer.
            Remaining arguments: the ``_scalar_singleray_corr_numba`` geometry
                and correlation arguments.

        Returns:
            (f, g): objective value and (3,) gradient estimate.
        """
        n = x.shape[0]
        f0 = _shifted_singleray_corr(
            x[0], x[1], x[2], phi_shift, pa_x, pa_y, ant_xy, td_values, td_ok, td_slot,
            td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)
        bounded = False
        for i in range(n):
            if not (lb[i] == -np.inf and ub[i] == np.inf):
                bounded = True
        g = np.empty(n, dtype=np.float64)
        xs = x.copy()
        for i in range(n):
            h = _lbfgsb_fd_step(x[i], lb[i], ub[i], bounded)
            xs[i] = x[i] + h
            fi = _shifted_singleray_corr(
                xs[0], xs[1], xs[2], phi_shift, pa_x, pa_y, ant_xy, td_values, td_ok, td_slot,
                td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
                corr_packed, corr_lengths, corr_dts, corr_offsets,
                pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)
            xs[i] = x[i]
            g[i] = (fi - f0) / ((x[i] + h) - x[i])
        return f0, g

    @njit(parallel=True, fastmath=True, cache=True)
    def _all_pairs_corr_numba(delay_T, corr_packed, corr_lengths,
                              corr_dts, corr_offsets, pair_weights):
        """Fused all-pairs weighted correlation sum.

        Parallelizes over grid points (outer prange). Uses points-major
        delay layout ``delay_T[pt, pidx]`` for cache-friendly access.

        Args:
            delay_T: (n_points, n_pairs) float64. NaN = skip.
            corr_packed: (n_pairs, M_max) float64, zero-padded.
            corr_lengths: (n_pairs,) int64.
            corr_dts: (n_pairs,) float64.
            corr_offsets: (n_pairs,) float64.
            pair_weights: (n_pairs,) float64.

        Returns:
            (n_points,) float64 weighted-mean correlation.
        """
        n_points, n_pairs = delay_T.shape
        w_sum = 0.0
        for p in range(n_pairs):
            w_sum += pair_weights[p]

        out = np.empty(n_points, dtype=np.float64)
        inv_w_sum = 1.0 / w_sum if w_sum > 0.0 else 0.0

        for pt in prange(n_points):
            acc = 0.0
            for pidx in range(n_pairs):
                d = delay_T[pt, pidx]
                if np.isnan(d):
                    continue
                dt = corr_dts[pidx]
                offset = corr_offsets[pidx]
                clen = corr_lengths[pidx]
                kf = (d - offset) / dt
                k = int(np.floor(kf))
                if k < 0 or k >= clen - 1:
                    continue
                alpha = kf - k
                val = (corr_packed[pidx, k]
                       + (corr_packed[pidx, k + 1]
                          - corr_packed[pidx, k]) * alpha)
                acc += val * pair_weights[pidx]
            out[pt] = acc * inv_w_sum
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _bilinear_batch_numba(values, r_min, dr_inv, nr, z_min, dz_inv, nz,
                              r_coords, z_coords):
        """Batch 2D bilinear interpolation on a uniform grid.

        Args:
            values: 2D array (nr, nz).
            r_min, dr_inv, nr, z_min, dz_inv, nz: Grid parameters.
            r_coords, z_coords: 1D query arrays.

        Returns:
            1D array of interpolated values. Out-of-bounds returns -inf.
        """
        n = r_coords.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            ri = (r_coords[i] - r_min) * dr_inv
            zi = (z_coords[i] - z_min) * dz_inv
            i0 = int(np.floor(ri))
            j0 = int(np.floor(zi))
            if i0 < 0 or i0 >= nr - 1 or j0 < 0 or j0 >= nz - 1:
                out[i] = -np.inf
            else:
                fx = ri - i0
                fy = zi - j0
                out[i] = ((1 - fx) * (1 - fy) * values[i0, j0]
                          + fx * (1 - fy) * values[i0 + 1, j0]
                          + (1 - fx) * fy * values[i0, j0 + 1]
                          + fx * fy * values[i0 + 1, j0 + 1])
        return out

    @njit(fastmath=True, cache=True)
    def _bilinear_scalar_numba(values, r_min, dr_inv, nr, z_min, dz_inv, nz,
                               r_val, z_val):
        """Single-point 2D bilinear interpolation on a uniform grid.

        Args:
            values: 2D array (nr, nz).
            r_min, dr_inv, nr, z_min, dz_inv, nz: Grid parameters.
            r_val, z_val: Query coordinates.

        Returns:
            Interpolated value, or -inf if out of bounds.
        """
        ri = (r_val - r_min) * dr_inv
        zi = (z_val - z_min) * dz_inv
        i0 = int(np.floor(ri))
        j0 = int(np.floor(zi))
        if i0 < 0 or j0 < 0:
            return -np.inf
        if i0 >= nr - 1:
            if ri <= nr - 1 + 1e-9:
                i0 = nr - 2
                fx = 1.0
            else:
                return -np.inf
        else:
            fx = ri - i0
        if j0 >= nz - 1:
            if zi <= nz - 1 + 1e-9:
                j0 = nz - 2
                fy = 1.0
            else:
                return -np.inf
        else:
            fy = zi - j0
        return ((1 - fx) * (1 - fy) * values[i0, j0]
                + fx * (1 - fy) * values[i0 + 1, j0]
                + (1 - fx) * fy * values[i0, j0 + 1]
                + fx * fy * values[i0 + 1, j0 + 1])


if USE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _fused_multiray_grid_numba(
            rho_vec, phi_vec_rad, z_vec,
            pa_x, pa_y,
            ant_xy, n_ch,
            td_values, td_ok, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz,
            n_rt,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, tolerant_edge):
        """Fused multiray grid correlator: TT lookup + combo evaluation in one kernel.

        For each grid point, computes per-channel per-ray-type travel times
        via inline bilinear table lookup, then takes for each pair the best of its
        ray-type combinations and returns the weighted mean of these per-pair maxima
        (the per_pair combo mode, not grouped). The lookups go through
        ``_bilinear_ok_numba``: validity comes from the finiteness mask, and with
        ``tolerant_edge`` a lookup on the last table row or column takes that row
        or column instead of being invalid.

        Args:
            rho_vec: (n_rho,) float64.
            phi_vec_rad: (n_phi,) float64, in radians.
            z_vec: (n_z,) float64.
            pa_x, pa_y: PA center absolute coordinates.
            ant_xy: (n_ch, 2) float64, antenna positions.
            n_ch: int.
            td_values: (n_ch * n_rt, nr_max, nz_max) float64, packed TT tables,
                slot ci * n_rt + ri for channel ci and ray type ri.
            td_ok: Boolean finiteness mask of td_values.
            td_r_min, td_dr_inv: (n_ch, n_rt) float64.
            td_nr: (n_ch, n_rt) int64.
            td_z_min, td_dz_inv: (n_ch, n_rt) float64.
            td_nz: (n_ch, n_rt) int64.
            n_rt: int.
            corr_packed: (n_pairs, max_corr_len) float64.
            corr_lengths: (n_pairs,) int64.
            corr_dts, corr_offsets: (n_pairs,) float64.
            pair_ch1, pair_ch2: (n_pairs,) int64.
            pair_weights: (n_pairs,) float64.
            w_total: float64.
            tolerant_edge: bool, accept queries on the last table row or column.

        Returns:
            (n_rho * n_phi * n_z,) float64 per-pair-max correlation at each point.
        """
        n_rho = rho_vec.shape[0]
        n_phi = phi_vec_rad.shape[0]
        n_z = z_vec.shape[0]
        n_points = n_rho * n_phi * n_z
        n_pairs = pair_ch1.shape[0]
        inv_w = 1.0 / w_total if w_total > 0.0 else 0.0

        out = np.empty(n_points, dtype=np.float64)

        for pt in prange(n_points):
            ir = pt // (n_phi * n_z)
            rem = pt % (n_phi * n_z)
            ip = rem // n_z
            iz = rem % n_z

            rho = rho_vec[ir]
            phi = phi_vec_rad[ip]
            z = z_vec[iz]
            x_src = rho * np.cos(phi) + pa_x
            y_src = rho * np.sin(phi) + pa_y

            tts = np.empty((n_ch, n_rt), dtype=np.float64)
            tt_valid = np.zeros((n_ch, n_rt), dtype=np.bool_)

            for ci in range(n_ch):
                dx = x_src - ant_xy[ci, 0]
                dy = y_src - ant_xy[ci, 1]
                r = np.sqrt(dx * dx + dy * dy)
                if r < 1.0:
                    r = 1.0

                for ri in range(n_rt):
                    tts[ci, ri], tt_valid[ci, ri] = _bilinear_ok_numba(
                        td_values, td_ok, ci * n_rt + ri,
                        (r - td_r_min[ci, ri]) * td_dr_inv[ci, ri],
                        (z - td_z_min[ci, ri]) * td_dz_inv[ci, ri],
                        td_nr[ci, ri], td_nz[ci, ri], tolerant_edge)

            # Per-pair mode: for each pair, try all n_rt^2 ray combos
            # and keep the max correlation. Then sum across pairs.
            total = 0.0
            for pidx in range(n_pairs):
                c1 = pair_ch1[pidx]
                c2 = pair_ch2[pidx]
                best_pair = -np.inf
                for rt1 in range(n_rt):
                    if not tt_valid[c1, rt1]:
                        continue
                    for rt2 in range(n_rt):
                        if not tt_valid[c2, rt2]:
                            continue
                        delay = tts[c1, rt1] - tts[c2, rt2]
                        dt = corr_dts[pidx]
                        offset = corr_offsets[pidx]
                        clen = corr_lengths[pidx]
                        kf = (delay - offset) / dt
                        k = int(np.floor(kf))
                        if k < 0 or k >= clen - 1:
                            continue
                        alpha = kf - k
                        val = (corr_packed[pidx, k]
                               + (corr_packed[pidx, k + 1]
                                  - corr_packed[pidx, k]) * alpha)
                        if val > best_pair:
                            best_pair = val
                if best_pair > -np.inf:
                    total += best_pair * pair_weights[pidx]

            out[pt] = total * inv_w
        return out

    @njit(fastmath=True, cache=True)
    def _grouped_weight_sum(pair_weights):
        """Sum of the pair weights as ``_grouped_multiray_kernel_pairmajor`` forms it.

        The same loop compiled with fast math, so the vectorised reduction adds the
        weights in the same order and gives the same value.
        """
        w_sum = 0.0
        for p in range(pair_weights.shape[0]):
            w_sum += pair_weights[p]
        return w_sum

    @njit(parallel=True, fastmath=False, cache=True)
    def _grouped_multiray_points_numba(tt, corr_packed, corr_lengths, corr_inv_dts, corr_offsets,
                                       row_pair, row_rt1, row_rt2, combo_rows, pair_ch1, pair_ch2,
                                       pair_weights, w_sum):
        """Grouped multiray map, point-major: per point the best over combos of the weighted mean correlation.

        Equals ``_grouped_multiray_kernel_pairmajor`` bit for bit. That kernel is
        compiled with fast math, which turns its pair term into
        val = fma(alpha, c[k + 1] - c[k], c[k]) at kf = (tt1 - (offset + tt2)) * (1 / dt),
        k = floor(kf), alpha = kf - k (val = 0 outside 0 <= k < len - 1 or for a
        non-finite delay), accumulates acc = fma(val, w, acc) in pair order and scales
        by 1 / w_sum; this kernel writes the same operations out explicitly and is
        compiled without fast math, so its result does not depend on vectorisation or
        on the caller. The pair term depends only on the pair and the ray types of its
        two channels, so each distinct (pair, ray type, ray type) row is evaluated once
        per point and every combo sums its rows.

        Args:
            tt: (n_ch, n_rt, n_points) travel times (NaN where a ray type is missing).
            corr_packed, corr_lengths, corr_offsets: Padded pair correlations, their
                lengths and lag offsets (``pack_corr_data``).
            corr_inv_dts: (n_pairs,) 1 / dt of each pair.
            row_pair, row_rt1, row_rt2: (n_rows,) pair and ray types of each row.
            combo_rows: (n_combos, n_pairs) row of each pair under each combo.
            pair_ch1, pair_ch2: (n_pairs,) channel indices of each pair.
            pair_weights: (n_pairs,) pair weights.
            w_sum: ``_grouped_weight_sum(pair_weights)``.

        Returns:
            (n_points,) best combo value per point (-inf where no combo value exceeds -inf).
        """
        n_points = tt.shape[2]
        n_rows = row_pair.shape[0]
        n_combos, n_pairs = combo_rows.shape
        inv_w = 1.0 / w_sum if w_sum > 0.0 else 1.0
        block = _GROUPED_BLOCK_POINTS
        out = np.empty(n_points, dtype=np.float64)
        for b in prange((n_points + block - 1) // block):
            p0 = b * block
            nb = min(block, n_points - p0)
            vals = np.empty((n_rows, block), dtype=np.float64)
            acc = np.empty(block, dtype=np.float64)
            best = np.full(block, -np.inf)
            for r in range(n_rows):
                p = row_pair[r]
                c1 = pair_ch1[p]
                c2 = pair_ch2[p]
                rt1 = row_rt1[r]
                rt2 = row_rt2[r]
                offset = corr_offsets[p]
                inv_dt = corr_inv_dts[p]
                last = corr_lengths[p] - 1
                for j in range(nb):
                    kf = (tt[c1, rt1, p0 + j] - (offset + tt[c2, rt2, p0 + j])) * inv_dt
                    v = 0.0
                    if kf >= 0.0 and kf < last:
                        k = int(kf)
                        c0 = corr_packed[p, k]
                        v = _fma(kf - k, corr_packed[p, k + 1] - c0, c0)
                    vals[r, j] = v
            for ci in range(n_combos):
                for j in range(nb):
                    acc[j] = 0.0
                for p in range(n_pairs):
                    r = combo_rows[ci, p]
                    w = pair_weights[p]
                    for j in range(nb):
                        acc[j] = _fma(vals[r, j], w, acc[j])
                for j in range(nb):
                    a = acc[j] * inv_w if w_sum > 0.0 else acc[j]
                    if a > best[j]:
                        best[j] = a
            for j in range(nb):
                out[p0 + j] = best[j]
        return out


if USE_NUMBA:
    @njit(fastmath=True, cache=True)
    def _bilinear_ok_numba(values, ok, ti, ri, zi, nr_ch, nz_ch, tolerant_edge):
        """Bilinear table value at fractional indices with an explicit validity flag.

        Validity requires the query inside the table (the last row and column are
        accepted only with ``tolerant_edge``), all four corners finite according
        to ``ok`` and a positive result; no NaN test is used.

        Args:
            values: (n_tables, nr_max, nz_max) float64 table stack.
            ok: Boolean finiteness mask of ``values``.
            ti: Table slot.
            ri, zi: Fractional row and column indices.
            nr_ch, nz_ch: Table size of this slot.
            tolerant_edge: Accept ri == nr_ch - 1 and zi == nz_ch - 1 (within 1e-9).

        Returns:
            (value, valid) with value 0.0 when invalid.
        """
        i0 = int(np.floor(ri))
        j0 = int(np.floor(zi))
        if i0 < 0 or j0 < 0:
            return 0.0, False
        if i0 >= nr_ch - 1:
            if tolerant_edge and ri <= nr_ch - 1 + 1e-9:
                i0 = nr_ch - 2
                fx = 1.0
            else:
                return 0.0, False
        else:
            fx = ri - i0
        if j0 >= nz_ch - 1:
            if tolerant_edge and zi <= nz_ch - 1 + 1e-9:
                j0 = nz_ch - 2
                fy = 1.0
            else:
                return 0.0, False
        else:
            fy = zi - j0
        if not (ok[ti, i0, j0] and ok[ti, i0 + 1, j0]
                and ok[ti, i0, j0 + 1] and ok[ti, i0 + 1, j0 + 1]):
            return 0.0, False
        v = ((1.0 - fx) * (1.0 - fy) * values[ti, i0, j0]
             + fx * (1.0 - fy) * values[ti, i0 + 1, j0]
             + (1.0 - fx) * fy * values[ti, i0, j0 + 1]
             + fx * fy * values[ti, i0 + 1, j0 + 1])
        if v > 0.0:
            return v, True
        return 0.0, False

    @njit(fastmath=True, cache=True)
    def _pairs_corr_numba(tts, valid, corr_stack, k, corr_lengths, corr_inv_dts,
                          corr_offsets, pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Weighted mean of the pair correlations at one point from per-channel travel times.

        Args:
            tts: (n_ch,) travel times.
            valid: (n_ch,) validity flags.
            corr_stack: (K, n_pairs, M_max) packed correlations.
            k: Correlation set to evaluate.
            corr_lengths, corr_inv_dts, corr_offsets: (n_pairs,) lag geometry.
            pair_ch1, pair_ch2: (n_pairs,) channel indices of each pair.
            pair_weights: (n_pairs,) weights.
            w_total: Sum of the weights.
            valid_norm, valid_floor: Valid-weight normalisation switch and floor.

        Returns:
            Weighted mean correlation (0 when no pair contributes).
        """
        acc = 0.0
        w_valid = 0.0
        for pidx in range(pair_ch1.shape[0]):
            c1 = pair_ch1[pidx]
            c2 = pair_ch2[pidx]
            if not valid[c1] or not valid[c2]:
                continue
            kf = (tts[c1] - tts[c2] - corr_offsets[pidx]) * corr_inv_dts[pidx]
            kk = int(np.floor(kf))
            if kk < 0 or kk >= corr_lengths[pidx] - 1:
                continue
            alpha = kf - kk
            y0 = corr_stack[k, pidx, kk]
            acc += (y0 + (corr_stack[k, pidx, kk + 1] - y0) * alpha) * pair_weights[pidx]
            w_valid += pair_weights[pidx]
        if valid_norm:
            return _valid_mean(acc, w_valid, _coverage_ramp(w_valid, w_total, valid_floor))
        return acc * _total_factor(w_total)

    @njit(fastmath=True, cache=True)
    def _plane_wave_corr_grad_numba(tts, zen, az, ant_xyz, n2_nodes, gl_weights, c_m_per_ns,
                                    corr_stack, k, corr_lengths, corr_inv_dts, corr_offsets,
                                    pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Gradient in (zen_deg, az_deg) of ``_pairs_corr_numba`` at plane-wave arrival times.

        The arrival time at antenna (x, y, z) is that of ``plane_wave_times``:
        -sin(zen) (x cos az + y sin az) / c plus, below the surface, the Gauss-Legendre
        sum 0.5 |z| sum_i w_i sqrt(n_i^2 - sin^2 zen) / c over the nodes between z and 0,
        and above it -z cos(zen) / c. Its derivatives are
        d t / d zen = -cos(zen) (x cos az + y sin az) / c
        - 0.5 |z| sum_i w_i sin(zen) cos(zen) / sqrt(n_i^2 - sin^2 zen) / c (below) or
        + z sin(zen) / c (above), and d t / d az = -sin(zen) (y cos az - x sin az) / c.
        Each pair contributes its weight times the slope of the series cell its delay
        falls in, (c[k + 1] - c[k]) / dt, times the delay's derivative; the coverage
        factor of the valid-weight normalisation is constant between changes of the set
        of pairs read and scales the sum like the value.

        Args:
            tts: (n_ch,) arrival times of ``plane_wave_times`` at the direction (ns); the
                cells are taken from them, so the gradient belongs to the value computed
                from the same times.
            zen, az: Direction in rad.
            ant_xyz: (n_ch, 3) antenna positions, z relative to the surface (m).
            n2_nodes: (n_ch, n_nodes) squared refractive index at each antenna's quadrature nodes.
            gl_weights: (n_nodes,) Gauss-Legendre weights.
            c_m_per_ns: Speed of light in m/ns.
            Remaining arguments: as ``_pairs_corr_numba``.

        Returns:
            (d value / d zen_deg, d value / d az_deg).
        """
        n_ch = ant_xyz.shape[0]
        sin_z = np.sin(zen)
        cos_z = np.cos(zen)
        sin_a = np.sin(az)
        cos_a = np.cos(az)
        s2 = sin_z * sin_z
        d_zen = np.empty(n_ch, dtype=np.float64)
        d_az = np.empty(n_ch, dtype=np.float64)
        for ci in range(n_ch):
            x = ant_xyz[ci, 0]
            y = ant_xyz[ci, 1]
            z = ant_xyz[ci, 2]
            dz = -cos_z * (x * cos_a + y * sin_a)
            if z < 0.0:
                acc = 0.0
                for i in range(gl_weights.shape[0]):
                    acc += gl_weights[i] / np.sqrt(n2_nodes[ci, i] - s2)
                dz -= 0.5 * -z * sin_z * cos_z * acc
            elif z > 0.0:
                dz += z * sin_z
            d_zen[ci] = dz / c_m_per_ns
            d_az[ci] = -sin_z * (y * cos_a - x * sin_a) / c_m_per_ns
        g_zen = 0.0
        g_az = 0.0
        w_valid = 0.0
        for pidx in range(pair_ch1.shape[0]):
            c1 = pair_ch1[pidx]
            c2 = pair_ch2[pidx]
            kf = (tts[c1] - tts[c2] - corr_offsets[pidx]) * corr_inv_dts[pidx]
            kk = int(np.floor(kf))
            if kk < 0 or kk >= corr_lengths[pidx] - 1:
                continue
            slope = (corr_stack[k, pidx, kk + 1] - corr_stack[k, pidx, kk]) * corr_inv_dts[pidx] * pair_weights[pidx]
            g_zen += slope * (d_zen[c1] - d_zen[c2])
            g_az += slope * (d_az[c1] - d_az[c2])
            w_valid += pair_weights[pidx]
        if valid_norm:
            scale = _valid_mean(1.0, w_valid, _coverage_ramp(w_valid, w_total, valid_floor))
        else:
            scale = _total_factor(w_total)
        scale *= np.pi / 180.0
        return g_zen * scale, g_az * scale

    @njit(parallel=True, fastmath=True, cache=True)
    def _singleray_grid_numba(
            rho_vec, phi_vec_rad, z_vec, pa_x, pa_y, ant_xy,
            td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz, tolerant_edge,
            corr_stack, corr_lengths, corr_inv_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Fused singleray grid correlator with inline table lookup.

        Rows of the (rho, phi) plane run in parallel; along each row the z
        loop is innermost so the table gathers of one channel stay local. Every
        channel's travel time is looked up inline, the pair delays are formed in
        registers and each of the K correlation sets is evaluated at the same
        geometry.

        Args:
            rho_vec, phi_vec_rad, z_vec: Grid axes (m, rad, m).
            pa_x, pa_y: PA centre absolute coordinates (m).
            ant_xy: (n_ch, 2) channel absolute (x, y).
            td_values, td_ok: Table stack and its finiteness mask.
            td_slot: (n_ch,) table slot per channel.
            td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz: (n_ch,) grid parameters.
            tolerant_edge: Accept queries on the last table row or column.
            corr_stack: (K, n_pairs, M_max) packed correlations.
            corr_lengths, corr_inv_dts, corr_offsets: (n_pairs,) lag geometry.
            pair_ch1, pair_ch2: (n_pairs,) channel indices of each pair.
            pair_weights: (n_pairs,) weights; w_total their sum.
            valid_norm, valid_floor: Valid-weight normalisation switch and floor.

        Returns:
            (K, n_rho * n_phi * n_z) float64 maps in C order of the grid.
        """
        n_rho = rho_vec.shape[0]
        n_phi = phi_vec_rad.shape[0]
        n_z = z_vec.shape[0]
        n_ch = ant_xy.shape[0]
        n_k = corr_stack.shape[0]
        n_rows = n_rho * n_phi
        out = np.empty((n_k, n_rows * n_z), dtype=np.float64)
        r_idx_rows = np.empty((n_rows, n_ch), dtype=np.float64)
        tts_rows = np.empty((n_rows, n_ch), dtype=np.float64)
        valid_rows = np.zeros((n_rows, n_ch), dtype=np.bool_)
        for row in prange(n_rows):
            rho = rho_vec[row // n_phi]
            phi = phi_vec_rad[row % n_phi]
            x_src = rho * np.cos(phi) + pa_x
            y_src = rho * np.sin(phi) + pa_y
            r_idx = r_idx_rows[row]
            tts = tts_rows[row]
            valid = valid_rows[row]
            for ci in range(n_ch):
                dx = x_src - ant_xy[ci, 0]
                dy = y_src - ant_xy[ci, 1]
                r = np.sqrt(dx * dx + dy * dy)
                if r < 1.0:
                    r = 1.0
                r_idx[ci] = (r - td_r_min[ci]) * td_dr_inv[ci]
            for iz in range(n_z):
                z = z_vec[iz]
                for ci in range(n_ch):
                    tts[ci], valid[ci] = _bilinear_ok_numba(
                        td_values, td_ok, td_slot[ci], r_idx[ci],
                        (z - td_z_min[ci]) * td_dz_inv[ci],
                        td_nr[ci], td_nz[ci], tolerant_edge)
                pt = row * n_z + iz
                for k in range(n_k):
                    out[k, pt] = _pairs_corr_numba(
                        tts, valid, corr_stack, k, corr_lengths, corr_inv_dts,
                        corr_offsets, pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _bilinear_ok_batch_numba(values, ok, ti, r_min, dr_inv, nr, z_min, dz_inv, nz,
                                 r_coords, z_coords, tolerant_edge):
        """Batch bilinear lookup of one table with explicit validity flags.

        Args:
            values, ok: Table stack and its finiteness mask.
            ti: Table slot.
            r_min, dr_inv, nr, z_min, dz_inv, nz: Grid parameters of the slot.
            r_coords, z_coords: (n,) query coordinates.
            tolerant_edge: Accept queries on the last table row or column.

        Returns:
            (tt, valid) of shape (n,), tt 0.0 where invalid.
        """
        n = r_coords.shape[0]
        tt = np.empty(n, dtype=np.float64)
        valid = np.empty(n, dtype=np.bool_)
        for i in prange(n):
            tt[i], valid[i] = _bilinear_ok_numba(
                values, ok, ti, (r_coords[i] - r_min) * dr_inv,
                (z_coords[i] - z_min) * dz_inv, nr, nz, tolerant_edge)
        return tt, valid

    @njit(parallel=True, fastmath=True, cache=True)
    def _singleray_stack_corr_numba(
            tts, valid, corr_stack, corr_lengths, corr_inv_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Correlation maps over a cached per-channel travel-time stack.

        Args:
            tts, valid: (n_points, n_ch) travel times and validity flags from
                ``_bilinear_ok_batch_numba``.
            corr_stack: (K, n_pairs, M_max) packed correlations.
            corr_lengths, corr_inv_dts, corr_offsets: (n_pairs,) lag geometry.
            pair_ch1, pair_ch2: (n_pairs,) channel indices of each pair.
            pair_weights: (n_pairs,) weights; w_total their sum.
            valid_norm, valid_floor: Valid-weight normalisation switch and floor.

        Returns:
            (K, n_points) float64 maps.
        """
        n_points = tts.shape[0]
        n_k = corr_stack.shape[0]
        out = np.empty((n_k, n_points), dtype=np.float64)
        for pt in prange(n_points):
            for k in range(n_k):
                out[k, pt] = _pairs_corr_numba(
                    tts[pt], valid[pt], corr_stack, k, corr_lengths, corr_inv_dts,
                    corr_offsets, pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _singleray_stackT_corr_numba(
            ttsT, validT, block, corr_stack, corr_lengths, corr_inv_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Correlation maps over a cached channel-major travel-time stack.

        The points run in blocks of ``block`` consecutive points, blocks in
        parallel. Within a block the pairs are the outer loop and the points the
        inner one, so each pair reads two contiguous travel-time rows and a narrow
        stretch of its series; the lag index of a pair at a point is computed once
        and serves all K correlation sets. Every point accumulates its pairs in
        pair order with the arithmetic of ``_pairs_corr_numba``, so the maps equal
        ``_singleray_stack_corr_numba`` on the point-major stack bit for bit.

        Args:
            ttsT, validT: (n_ch, n_points) travel times and validity flags.
            block: Points per block.
            corr_stack: (K, n_pairs, M_max) packed correlations.
            corr_lengths, corr_inv_dts, corr_offsets: (n_pairs,) lag geometry.
            pair_ch1, pair_ch2: (n_pairs,) channel indices of each pair.
            pair_weights: (n_pairs,) weights; w_total their sum.
            valid_norm, valid_floor: Valid-weight normalisation switch and floor.

        Returns:
            (K, n_points) float64 maps.
        """
        n_points = ttsT.shape[1]
        n_blocks = (n_points + block - 1) // block
        n_k = corr_stack.shape[0]
        n_pairs = pair_ch1.shape[0]
        out = np.empty((n_k, n_points), dtype=np.float64)
        for b in prange(n_blocks):
            base = b * block
            nb = min(block, n_points - base)
            acc = np.zeros((n_k, nb), dtype=np.float64)
            w_valid = np.zeros(nb, dtype=np.float64)
            for pidx in range(n_pairs):
                c1 = pair_ch1[pidx]
                c2 = pair_ch2[pidx]
                off = corr_offsets[pidx]
                inv = corr_inv_dts[pidx]
                last = corr_lengths[pidx] - 1
                w = pair_weights[pidx]
                t1 = ttsT[c1, base:base + nb]
                t2 = ttsT[c2, base:base + nb]
                v1 = validT[c1, base:base + nb]
                v2 = validT[c2, base:base + nb]
                for i in range(nb):
                    if not v1[i] or not v2[i]:
                        continue
                    kf = (t1[i] - t2[i] - off) * inv
                    kk = int(np.floor(kf))
                    if kk < 0 or kk >= last:
                        continue
                    alpha = kf - kk
                    for k in range(n_k):
                        y0 = corr_stack[k, pidx, kk]
                        acc[k, i] += (y0 + (corr_stack[k, pidx, kk + 1] - y0) * alpha) * w
                    w_valid[i] += w
            for i in range(nb):
                if valid_norm:
                    f = _coverage_ramp(w_valid[i], w_total, valid_floor)
                    for k in range(n_k):
                        out[k, base + i] = _valid_mean(acc[k, i], w_valid[i], f)
                else:
                    f = _total_factor(w_total)
                    for k in range(n_k):
                        out[k, base + i] = acc[k, i] * f
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _singleray_grid_tts_numba(
            rho_vec, phi_vec_rad, z_vec, pa_x, pa_y, ant_xy,
            td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz, tolerant_edge):
        """Channel-major travel times of a product grid with the lookups of ``_singleray_grid_numba``.

        The source position, horizontal distances and table lookups are the
        grid kernel's, row by row, so ``_singleray_stackT_corr_numba`` on the
        result gives that kernel's maps bit for bit.

        Returns:
            (ttsT, validT) of shape (n_ch, n_rho * n_phi * n_z), points in C order.
        """
        n_rho = rho_vec.shape[0]
        n_phi = phi_vec_rad.shape[0]
        n_z = z_vec.shape[0]
        n_ch = ant_xy.shape[0]
        n_rows = n_rho * n_phi
        ttsT = np.empty((n_ch, n_rows * n_z), dtype=np.float64)
        validT = np.empty((n_ch, n_rows * n_z), dtype=np.bool_)
        for row in prange(n_rows):
            rho = rho_vec[row // n_phi]
            phi = phi_vec_rad[row % n_phi]
            x_src = rho * np.cos(phi) + pa_x
            y_src = rho * np.sin(phi) + pa_y
            r_idx = np.empty(n_ch, dtype=np.float64)
            for ci in range(n_ch):
                dx = x_src - ant_xy[ci, 0]
                dy = y_src - ant_xy[ci, 1]
                r = np.sqrt(dx * dx + dy * dy)
                if r < 1.0:
                    r = 1.0
                r_idx[ci] = (r - td_r_min[ci]) * td_dr_inv[ci]
            for iz in range(n_z):
                z = z_vec[iz]
                pt = row * n_z + iz
                for ci in range(n_ch):
                    ttsT[ci, pt], validT[ci, pt] = _bilinear_ok_numba(
                        td_values, td_ok, td_slot[ci], r_idx[ci],
                        (z - td_z_min[ci]) * td_dz_inv[ci],
                        td_nr[ci], td_nz[ci], tolerant_edge)
        return ttsT, validT

    _ORDERED_FASTMATH = {'contract', 'nnan', 'ninf', 'nsz', 'arcp', 'afn'}

    @njit(fastmath=_ORDERED_FASTMATH, cache=True)
    def _accumulate_column(acc, wv, cval, cok, j, w, wa, nb):
        """Add one column times its weight to the per-point sums of a block, without reassociation.

        Each point's sum keeps the order of the columns (the setting's pair order);
        the points are independent, so the loop may still be vectorized across them.
        """
        for b in range(nb):
            if cok[j, b]:
                acc[b] += cval[j, b] * w
                wv[b] += wa

    @njit(fastmath=_ORDERED_FASTMATH, cache=True)
    def _ordered_map_sum(cval, cok, map_ptr, map_col, map_w, map_wabs, m):
        """Weighted sum of one map's columns at one point in the map's entry order, without reassociation.

        Returns:
            (sum of value times weight, sum of the weights of the readable columns).
        """
        acc = 0.0
        wv = 0.0
        for e in range(map_ptr[m], map_ptr[m + 1]):
            j = map_col[e]
            if cok[j]:
                acc += cval[j] * map_w[e]
                wv += map_wabs[e]
        return acc, wv

    @njit(parallel=True, fastmath=True, cache=True)
    def _batched_stack_maps_numba(
            tts, valid, col_ch1, col_ch2, col_row, col_lengths, col_inv_dts, col_offsets,
            series, map_ptr, map_col, map_w, map_wabs, map_wtotal, valid_norm, valid_floor,
            block):
        """Correlation maps of many settings over one cached travel-time stack.

        The per-pair contributions C[x, j] (interpolated series ``col_row[j]`` at the
        delay of column j's channel pair at point x) do not depend on the setting, so
        each block of points evaluates them once per column and every map then sums
        its own columns with its own weights in its own pair order. Each map value
        equals ``_pairs_corr_numba`` of that setting bit for bit: the same
        interpolation, a pair skipped under the same conditions, the same
        accumulation order (the sums run in a helper compiled without
        reassociation, so the compiler cannot reorder them), and a sign of the
        series folded into the weight (negating a series negates the
        interpolated value exactly).

        Args:
            tts, valid: (n_ch, n_points) travel times and validity of the channels.
            col_ch1, col_ch2: (n_cols,) union channel indices of each column's pair.
            col_row: (n_cols,) row of ``series`` each column reads.
            col_lengths, col_inv_dts, col_offsets: (n_cols,) lag geometry of each column.
            series: (n_rows, M_max) correlation series.
            map_ptr: (n_maps + 1,) start of each map's entries in ``map_col``.
            map_col: Column of each entry, in the map's pair order.
            map_w: Signed weight of each entry (pair weight times series sign).
            map_wabs: Pair weight of each entry (for the valid-weight normalisation).
            map_wtotal: (n_maps,) sum of each map's pair weights.
            valid_norm, valid_floor: Valid-weight normalisation switch and floor.
            block: Points per block.

        Returns:
            (n_maps, n_points) float64 maps.
        """
        n_points = tts.shape[1]
        n_cols = col_ch1.shape[0]
        n_maps = map_wtotal.shape[0]
        out = np.empty((n_maps, n_points), dtype=np.float64)
        n_blocks = (n_points + block - 1) // block
        for bi in prange(n_blocks):
            p0 = bi * block
            nb = min(block, n_points - p0)
            cval = np.zeros((n_cols, block), dtype=np.float64)
            cok = np.zeros((n_cols, block), dtype=np.bool_)
            acc = np.empty(block, dtype=np.float64)
            wv = np.empty(block, dtype=np.float64)
            for j in range(n_cols):
                c1 = col_ch1[j]
                c2 = col_ch2[j]
                row = col_row[j]
                off = col_offsets[j]
                inv_dt = col_inv_dts[j]
                last = col_lengths[j] - 1
                for b in range(nb):
                    pt = p0 + b
                    if not valid[c1, pt] or not valid[c2, pt]:
                        continue
                    kf = (tts[c1, pt] - tts[c2, pt] - off) * inv_dt
                    kk = int(np.floor(kf))
                    if kk < 0 or kk >= last:
                        continue
                    alpha = kf - kk
                    y0 = series[row, kk]
                    cval[j, b] = y0 + (series[row, kk + 1] - y0) * alpha
                    cok[j, b] = True
            for m in range(n_maps):
                for b in range(nb):
                    acc[b] = 0.0
                    wv[b] = 0.0
                for e in range(map_ptr[m], map_ptr[m + 1]):
                    _accumulate_column(acc, wv, cval, cok, map_col[e], map_w[e], map_wabs[e], nb)
                if valid_norm:
                    for b in range(nb):
                        out[m, p0 + b] = _valid_mean(acc[b], wv[b], _coverage_ramp(wv[b], map_wtotal[m], valid_floor))
                else:
                    for b in range(nb):
                        out[m, p0 + b] = acc[b] * _total_factor(map_wtotal[m])
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _batched_grid_maps_numba(
            rho_vec, phi_vec_rad, z_vec, pa_x, pa_y, ant_xy,
            td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz, tolerant_edge,
            col_ch1, col_ch2, col_row, col_lengths, col_inv_dts, col_offsets, series,
            map_ptr, map_col, map_w, map_wabs, map_wtotal, valid_norm, valid_floor):
        """Correlation maps of many settings over one product grid with inline table lookup.

        The grid geometry and travel times are those of ``_singleray_grid_numba`` (same
        arithmetic per channel); per point every distinct column is interpolated once
        and every map sums its own entries in its own pair order in a helper
        compiled without reassociation (with it, the compiler vectorizes the sum
        over the entries and changes the last bits), so each map equals the
        separate grid map bit for bit.

        Args:
            rho_vec, phi_vec_rad, z_vec: Grid axes (m, rad, m).
            pa_x, pa_y: PA centre absolute coordinates (m).
            ant_xy: (n_ch, 2) channel absolute (x, y) of the union channels.
            td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz:
                Table stack, finiteness mask and per-channel grid parameters.
            tolerant_edge: Accept queries on the last table row or column.
            col_ch1, col_ch2, col_row, col_lengths, col_inv_dts, col_offsets: (n_cols,) columns.
            series: (n_rows, M_max) correlation series.
            map_ptr, map_col, map_w, map_wabs, map_wtotal: Map entries (see
                ``_batched_stack_maps_numba``).
            valid_norm, valid_floor: Valid-weight normalisation switch and floor.

        Returns:
            (n_maps, n_rho * n_phi * n_z) float64 maps in C order of the grid.
        """
        n_rho = rho_vec.shape[0]
        n_phi = phi_vec_rad.shape[0]
        n_z = z_vec.shape[0]
        n_ch = ant_xy.shape[0]
        n_cols = col_ch1.shape[0]
        n_maps = map_wtotal.shape[0]
        n_rows = n_rho * n_phi
        out = np.empty((n_maps, n_rows * n_z), dtype=np.float64)
        for row in prange(n_rows):
            r_idx = np.empty(n_ch, dtype=np.float64)
            tts = np.empty(n_ch, dtype=np.float64)
            valid = np.zeros(n_ch, dtype=np.bool_)
            cval = np.zeros(n_cols, dtype=np.float64)
            cok = np.zeros(n_cols, dtype=np.bool_)
            rho = rho_vec[row // n_phi]
            phi = phi_vec_rad[row % n_phi]
            x_src = rho * np.cos(phi) + pa_x
            y_src = rho * np.sin(phi) + pa_y
            for ci in range(n_ch):
                dx = x_src - ant_xy[ci, 0]
                dy = y_src - ant_xy[ci, 1]
                r = np.sqrt(dx * dx + dy * dy)
                if r < 1.0:
                    r = 1.0
                r_idx[ci] = (r - td_r_min[ci]) * td_dr_inv[ci]
            for iz in range(n_z):
                z = z_vec[iz]
                for ci in range(n_ch):
                    tts[ci], valid[ci] = _bilinear_ok_numba(
                        td_values, td_ok, td_slot[ci], r_idx[ci],
                        (z - td_z_min[ci]) * td_dz_inv[ci],
                        td_nr[ci], td_nz[ci], tolerant_edge)
                for j in range(n_cols):
                    cok[j] = False
                    c1 = col_ch1[j]
                    c2 = col_ch2[j]
                    if not valid[c1] or not valid[c2]:
                        continue
                    kf = (tts[c1] - tts[c2] - col_offsets[j]) * col_inv_dts[j]
                    kk = int(np.floor(kf))
                    if kk < 0 or kk >= col_lengths[j] - 1:
                        continue
                    alpha = kf - kk
                    row_j = col_row[j]
                    y0 = series[row_j, kk]
                    cval[j] = y0 + (series[row_j, kk + 1] - y0) * alpha
                    cok[j] = True
                pt = row * n_z + iz
                for m in range(n_maps):
                    acc, wv = _ordered_map_sum(cval, cok, map_ptr, map_col, map_w, map_wabs, m)
                    if valid_norm:
                        out[m, pt] = _valid_mean(acc, wv, _coverage_ramp(wv, map_wtotal[m], valid_floor))
                    else:
                        out[m, pt] = acc * _total_factor(map_wtotal[m])
        return out

    @njit(fastmath=True, cache=True)
    def _compass_point_corr(rho, phi_deg, z, pa_x, pa_y, ant_xy, td_values, td_ok, td_slot,
                            td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
                            corr_packed, corr_lengths, corr_dts, corr_offsets,
                            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Weighted mean correlation at one (rho, phi_deg, z) point (the compass objective)."""
        return -_scalar_singleray_corr_numba(
            rho, phi_deg * (np.pi / 180.0), z, pa_x, pa_y, ant_xy,
            td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)

    @njit(parallel=True, fastmath=True, cache=True)
    def _compass_search_numba(
            seeds, rho_lo, rho_hi, z_lo, z_hi, step0, step_min, max_evals, phi_scan,
            pa_x, pa_y, ant_xy, td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor):
        """Bounded gradient-and-compass search from every seed on the scalar singleray objective.

        Each iteration takes the forward-difference gradient (1e-7 m, deg, m),
        scales it by the current steps and scans along it at 1/4, 1/2, 1, 2, 4,
        8, 16, 32 and 64 steps, taking the best point; scanning fixed multiples
        instead of extending while the objective rises lets the search cross the
        dips between correlation lobes as the L-BFGS-B line search does. When no
        point on that line improves, the six axis moves and the eight diagonal
        moves at the current steps are tried, the best improving one is accepted
        and the same multiples are scanned along its displacement. When nothing
        improves the steps are halved; the search stops once every step is below
        its minimum and an iteration has improved nothing, or the evaluation
        budget is spent. Once the steps are below 1e-3 of the initial steps the
        scans stop at 2 steps, since the long multiples only serve to cross
        lobes. The objective is piecewise linear in position (linear
        interpolation of the correlations and bilinear travel times), so its
        maximum is a vertex and the stopping steps set the final precision: at
        a slope of 0.05 per metre, 1e-8 m gives 5e-10 in correlation. phi is
        wrapped, rho and z are clamped to the bounds. With ``phi_scan`` an
        azimuth line scan of +/- 1.5 deg at 0.02 deg at the seed's rho and z
        precedes the search. Seeds run in parallel.

        Args:
            seeds: (n_seeds, 3) float64 of (rho, phi_deg, z).
            rho_lo, rho_hi, z_lo, z_hi: Bounds in m.
            step0: (3,) initial steps (m, deg, m).
            step_min: (3,) stopping steps (m, deg, m).
            max_evals: Evaluation budget per seed.
            phi_scan: Run the azimuth line scan first.
            Remaining arguments: the ``_scalar_singleray_corr_numba`` geometry
                and correlation arguments.

        Returns:
            (n_seeds, 5) float64 of (rho, phi_deg, z, correlation, evaluations).
        """
        eps = 1e-7
        n_seeds = seeds.shape[0]
        out = np.empty((n_seeds, 5), dtype=np.float64)
        moves = np.zeros((14, 3), dtype=np.float64)
        for axis in range(3):
            moves[2 * axis, axis] = -1.0
            moves[2 * axis + 1, axis] = 1.0
        for corner in range(8):
            for axis in range(3):
                moves[6 + corner, axis] = 1.0 if (corner >> axis) & 1 else -1.0
        scan = np.array([1.0, 0.5, 0.25, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
        scratch = np.empty((5, n_seeds, 3), dtype=np.float64)
        for s in prange(n_seeds):
            pos = scratch[0, s]
            cand = scratch[1, s]
            best_pos = scratch[2, s]
            delta = scratch[3, s]
            steps = scratch[4, s]
            pos[0] = min(max(seeds[s, 0], rho_lo), rho_hi)
            pos[1] = seeds[s, 1] % 360.0
            pos[2] = min(max(seeds[s, 2], z_lo), z_hi)
            f = _compass_point_corr(
                pos[0], pos[1], pos[2], pa_x, pa_y, ant_xy, td_values, td_ok, td_slot,
                td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
                corr_packed, corr_lengths, corr_dts, corr_offsets,
                pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)
            evals = 1
            if phi_scan:
                phi0 = pos[1]
                for j in range(-75, 76):
                    phi_j = (phi0 + 0.02 * j) % 360.0
                    fj = _compass_point_corr(
                        pos[0], phi_j, pos[2], pa_x, pa_y, ant_xy, td_values, td_ok, td_slot,
                        td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
                        corr_packed, corr_lengths, corr_dts, corr_offsets,
                        pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)
                    evals += 1
                    if fj > f:
                        f = fj
                        pos[1] = phi_j
            for a in range(3):
                steps[a] = step0[a]
            while evals < max_evals:
                norm = 0.0
                for a in range(3):
                    cand[0] = pos[0]
                    cand[1] = pos[1]
                    cand[2] = pos[2]
                    cand[a] = pos[a] + eps
                    sign = 1.0
                    if (a == 0 and cand[0] > rho_hi) or (a == 2 and cand[2] > z_hi):
                        cand[a] = pos[a] - eps
                        sign = -1.0
                    fa = _compass_point_corr(
                        cand[0], cand[1] % 360.0, cand[2], pa_x, pa_y, ant_xy, td_values, td_ok,
                        td_slot, td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
                        corr_packed, corr_lengths, corr_dts, corr_offsets,
                        pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)
                    evals += 1
                    delta[a] = sign * (fa - f) / eps * steps[a]
                    norm += delta[a] * delta[a]
                best_f = f
                best_t = 0.0
                n_scan = scan.shape[0] if steps[0] >= 1e-3 * step0[0] else 4
                if norm > 0.0:
                    norm = np.sqrt(norm)
                    for a in range(3):
                        delta[a] = delta[a] / norm * steps[a]
                    for k in range(n_scan):
                        if evals >= max_evals:
                            break
                        t = scan[k]
                        cand[0] = min(max(pos[0] + t * delta[0], rho_lo), rho_hi)
                        cand[1] = (pos[1] + t * delta[1]) % 360.0
                        cand[2] = min(max(pos[2] + t * delta[2], z_lo), z_hi)
                        if cand[0] == pos[0] and cand[1] == pos[1] and cand[2] == pos[2]:
                            continue
                        fc = _compass_point_corr(
                            cand[0], cand[1], cand[2], pa_x, pa_y, ant_xy, td_values, td_ok,
                            td_slot, td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
                            corr_packed, corr_lengths, corr_dts, corr_offsets,
                            pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)
                        evals += 1
                        if fc > best_f:
                            best_f = fc
                            best_t = t
                            for a in range(3):
                                best_pos[a] = cand[a]
                if best_t > 0.0:
                    for a in range(3):
                        pos[a] = best_pos[a]
                    f = best_f
                    continue
                best_m = -1
                for m in range(14):
                    if evals >= max_evals:
                        break
                    cand[0] = min(max(pos[0] + moves[m, 0] * steps[0], rho_lo), rho_hi)
                    cand[1] = (pos[1] + moves[m, 1] * steps[1]) % 360.0
                    cand[2] = min(max(pos[2] + moves[m, 2] * steps[2], z_lo), z_hi)
                    if cand[0] == pos[0] and cand[1] == pos[1] and cand[2] == pos[2]:
                        continue
                    fc = _compass_point_corr(
                        cand[0], cand[1], cand[2], pa_x, pa_y, ant_xy, td_values, td_ok, td_slot,
                        td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
                        corr_packed, corr_lengths, corr_dts, corr_offsets,
                        pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)
                    evals += 1
                    if fc > best_f:
                        best_f = fc
                        best_m = m
                        for a in range(3):
                            best_pos[a] = cand[a]
                if best_m < 0:
                    if (steps[0] < step_min[0] and steps[1] < step_min[1]
                            and steps[2] < step_min[2]):
                        break
                    for a in range(3):
                        steps[a] *= 0.5
                    continue
                delta[0] = best_pos[0] - pos[0]
                delta[1] = (best_pos[1] - pos[1] + 180.0) % 360.0 - 180.0
                delta[2] = best_pos[2] - pos[2]
                for a in range(3):
                    cand[a] = pos[a]
                    pos[a] = best_pos[a]
                f = best_f
                for k in range(3, n_scan):
                    if evals >= max_evals:
                        break
                    mult = scan[k]
                    rho_c = min(max(cand[0] + mult * delta[0], rho_lo), rho_hi)
                    phi_c = (cand[1] + mult * delta[1]) % 360.0
                    z_c = min(max(cand[2] + mult * delta[2], z_lo), z_hi)
                    if rho_c == pos[0] and phi_c == pos[1] and z_c == pos[2]:
                        continue
                    fc = _compass_point_corr(
                        rho_c, phi_c, z_c, pa_x, pa_y, ant_xy, td_values, td_ok, td_slot,
                        td_r_min, td_dr_inv, td_nr, td_z_min, td_dz_inv, td_nz,
                        corr_packed, corr_lengths, corr_dts, corr_offsets,
                        pair_ch1, pair_ch2, pair_weights, w_total, valid_norm, valid_floor)
                    evals += 1
                    if fc > f:
                        pos[0] = rho_c
                        pos[1] = phi_c
                        pos[2] = z_c
                        f = fc
            out[s, 0] = pos[0]
            out[s, 1] = pos[1]
            out[s, 2] = pos[2]
            out[s, 3] = f
            out[s, 4] = evals
        return out


if USE_NUMBA:
    @njit(fastmath=True, cache=True)
    def _two_arrival_channel_numba(ci, td0_values, td0_ok, td0_r_min, td0_dr_inv, td0_nr,
                                   td0_z_min, td0_dz_inv, td0_nz,
                                   td1_values, td1_ok, td1_r_min, td1_dr_inv, td1_nr,
                                   td1_z_min, td1_dz_inv, td1_nz,
                                   r, z, n_ice, delta_n, z_0, c_m_per_ns):
        """Look up both solution-ordered arrivals of one channel and the critical-angle mask.

        The mask is 1 when the solution_1 ray leaves the source steeper than the
        critical angle arcsin(1 / n(z)) from the vertical. Its launch angle comes
        from the horizontal slowness of the solution_1 table, p = dT/dR at the
        source (central difference over 1 m, one sided at a table boundary), through
        sin(theta) = c p / n(z) with n(z) the exponential ice profile. Every lookup
        goes through ``_bilinear_ok_numba`` with the scalar kernel's edge rule (the
        last table row and column accepted), so validity comes from the tables'
        finiteness masks and a positive value, never from a NaN test under fastmath.

        Args:
            ci: Channel slot in the packed tables.
            td0_values, td0_ok, td0_r_min, td0_dr_inv, td0_nr, td0_z_min, td0_dz_inv, td0_nz:
                Packed solution_0 tables, their finiteness mask and grid parameters.
            td1_values, td1_ok, td1_r_min, td1_dr_inv, td1_nr, td1_z_min, td1_dz_inv, td1_nz:
                The same for solution_1.
            r, z: Horizontal distance to the antenna and absolute depth of the source (m).
            n_ice, delta_n, z_0: Ice profile n(z) = n_ice - delta_n exp(z / z_0).
            c_m_per_ns: Speed of light in m/ns.

        Returns:
            (tt0, valid0, tt1, valid1, mask) with travel times in ns, validity flags
            and the mask as 0.0 or 1.0.
        """
        tt0, valid0 = _bilinear_ok_numba(
            td0_values, td0_ok, ci, (r - td0_r_min[ci]) * td0_dr_inv[ci],
            (z - td0_z_min[ci]) * td0_dz_inv[ci], td0_nr[ci], td0_nz[ci], True)
        zi1 = (z - td1_z_min[ci]) * td1_dz_inv[ci]
        tt1, valid1 = _bilinear_ok_numba(
            td1_values, td1_ok, ci, (r - td1_r_min[ci]) * td1_dr_inv[ci], zi1,
            td1_nr[ci], td1_nz[ci], True)
        mask = 0.0
        if valid1:
            t_plus, ok_plus = _bilinear_ok_numba(
                td1_values, td1_ok, ci, (r + 0.5 - td1_r_min[ci]) * td1_dr_inv[ci], zi1,
                td1_nr[ci], td1_nz[ci], True)
            t_minus, ok_minus = _bilinear_ok_numba(
                td1_values, td1_ok, ci, (r - 0.5 - td1_r_min[ci]) * td1_dr_inv[ci], zi1,
                td1_nr[ci], td1_nz[ci], True)
            has_p = True
            p = 0.0
            if ok_plus and ok_minus:
                p = t_plus - t_minus
            elif ok_plus:
                p = 2.0 * (t_plus - tt1)
            elif ok_minus:
                p = 2.0 * (tt1 - t_minus)
            else:
                has_p = False
            if has_p:
                n_src = 1.0
                if z <= 0.0:
                    n_src = n_ice - delta_n * np.exp(z / z_0)
                if p * c_m_per_ns / n_src > 1.0 / n_src:
                    mask = 1.0
        return tt0, valid0, tt1, valid1, mask

    @njit(fastmath=True, cache=True)
    def _two_arrival_point_numba(x_src, y_src, z, ant_xy,
                                 td0_values, td0_ok, td0_r_min, td0_dr_inv, td0_nr,
                                 td0_z_min, td0_dz_inv, td0_nz,
                                 td1_values, td1_ok, td1_r_min, td1_dr_inv, td1_nr,
                                 td1_z_min, td1_dz_inv, td1_nz,
                                 n_ice, delta_n, z_0, c_m_per_ns,
                                 second_weight, use_mask,
                                 corr_packed, corr_lengths, corr_dts, corr_offsets,
                                 pair_ch1, pair_ch2, pair_weights, w_total):
        """Consistent two-arrival correlation at one source point.

        Every pair is evaluated at the same solution index in both channels: the
        solution_0 delay always, plus the solution_1 delay weighted by
        ``second_weight`` times the product of the two channels' critical-angle
        masks (``use_mask``) or by ``second_weight`` alone. A pair whose
        solution_1 is undefined in either channel contributes its solution_0
        value only; a pair whose solution_0 is undefined contributes nothing.

        Args:
            x_src, y_src, z: Absolute source coordinates (m).
            ant_xy: (n_ch, 2) float64 antenna positions.
            td0_*: (n_ch, ...) packed solution_0 tables, their finiteness mask
                (td0_ok) and grid parameters.
            td1_*: The same for solution_1.
            n_ice, delta_n, z_0, c_m_per_ns: Ice profile and speed of light (m/ns).
            second_weight: Weight of the solution_1 term.
            use_mask: Multiply the weight by the critical-angle masks.
            corr_packed, corr_lengths, corr_dts, corr_offsets: Packed correlations.
            pair_ch1, pair_ch2: (n_pairs,) channel indices of each pair.
            pair_weights: (n_pairs,) float64.
            w_total: Sum of the pair weights.

        Returns:
            Weighted mean of the pair values (positive, larger is better).
        """
        n_ch = ant_xy.shape[0]
        tt0 = np.empty(n_ch, dtype=np.float64)
        tt1 = np.empty(n_ch, dtype=np.float64)
        valid0 = np.zeros(n_ch, dtype=np.bool_)
        valid1 = np.zeros(n_ch, dtype=np.bool_)
        mask = np.zeros(n_ch, dtype=np.float64)
        for ci in range(n_ch):
            dx = x_src - ant_xy[ci, 0]
            dy = y_src - ant_xy[ci, 1]
            r = np.sqrt(dx * dx + dy * dy)
            if r < 1.0:
                r = 1.0
            t0, v0, t1, v1, m = _two_arrival_channel_numba(
                ci, td0_values, td0_ok, td0_r_min, td0_dr_inv, td0_nr,
                td0_z_min, td0_dz_inv, td0_nz,
                td1_values, td1_ok, td1_r_min, td1_dr_inv, td1_nr,
                td1_z_min, td1_dz_inv, td1_nz,
                r, z, n_ice, delta_n, z_0, c_m_per_ns)
            tt0[ci] = t0
            valid0[ci] = v0
            tt1[ci] = t1
            valid1[ci] = v1
            mask[ci] = m

        total = 0.0
        for pidx in range(pair_ch1.shape[0]):
            c1 = pair_ch1[pidx]
            c2 = pair_ch2[pidx]
            if not valid0[c1] or not valid0[c2]:
                continue
            dt = corr_dts[pidx]
            offset = corr_offsets[pidx]
            clen = corr_lengths[pidx]
            value = 0.0
            kf = (tt0[c1] - tt0[c2] - offset) / dt
            k = int(np.floor(kf))
            if k >= 0 and k < clen - 1:
                alpha = kf - k
                value += (corr_packed[pidx, k]
                          + (corr_packed[pidx, k + 1] - corr_packed[pidx, k]) * alpha)
            if valid1[c1] and valid1[c2]:
                w2 = second_weight
                if use_mask:
                    w2 = w2 * mask[c1] * mask[c2]
                if w2 != 0.0:
                    kf = (tt1[c1] - tt1[c2] - offset) / dt
                    k = int(np.floor(kf))
                    if k >= 0 and k < clen - 1:
                        alpha = kf - k
                        value += w2 * (corr_packed[pidx, k]
                                       + (corr_packed[pidx, k + 1]
                                          - corr_packed[pidx, k]) * alpha)
            total += value * pair_weights[pidx]
        if w_total > 0.0:
            return total / w_total
        return 0.0

    @njit(parallel=True, fastmath=True, cache=True)
    def _fused_two_arrival_grid_numba(rho_vec, phi_vec_rad, z_vec, pa_x, pa_y, ant_xy,
                                      td0_values, td0_ok, td0_r_min, td0_dr_inv, td0_nr,
                                      td0_z_min, td0_dz_inv, td0_nz,
                                      td1_values, td1_ok, td1_r_min, td1_dr_inv, td1_nr,
                                      td1_z_min, td1_dz_inv, td1_nz,
                                      n_ice, delta_n, z_0, c_m_per_ns,
                                      second_weight, use_mask,
                                      corr_packed, corr_lengths, corr_dts, corr_offsets,
                                      pair_ch1, pair_ch2, pair_weights, w_total):
        """Consistent two-arrival correlation on a (rho, phi, z) grid.

        Args:
            rho_vec, phi_vec_rad, z_vec: Grid axes (m, rad, m).
            pa_x, pa_y: Grid centre.
            Remaining arguments: as for ``_two_arrival_point_numba``.

        Returns:
            (n_rho * n_phi * n_z,) float64 map in C order.
        """
        n_phi = phi_vec_rad.shape[0]
        n_z = z_vec.shape[0]
        n_points = rho_vec.shape[0] * n_phi * n_z
        out = np.empty(n_points, dtype=np.float64)
        for pt in prange(n_points):
            ir = pt // (n_phi * n_z)
            rem = pt % (n_phi * n_z)
            rho = rho_vec[ir]
            phi = phi_vec_rad[rem // n_z]
            z = z_vec[rem % n_z]
            out[pt] = _two_arrival_point_numba(
                rho * np.cos(phi) + pa_x, rho * np.sin(phi) + pa_y, z, ant_xy,
                td0_values, td0_ok, td0_r_min, td0_dr_inv, td0_nr,
                td0_z_min, td0_dz_inv, td0_nz,
                td1_values, td1_ok, td1_r_min, td1_dr_inv, td1_nr,
                td1_z_min, td1_dz_inv, td1_nz,
                n_ice, delta_n, z_0, c_m_per_ns, second_weight, use_mask,
                corr_packed, corr_lengths, corr_dts, corr_offsets,
                pair_ch1, pair_ch2, pair_weights, w_total)
        return out


if USE_NUMBA:
    @njit(cache=True)
    def _pairwise_leaf(a, lo, n):
        """numpy's pairwise sum of at most 128 items a[lo:lo + n] (8 accumulators from 8 items)."""
        if n < 8:
            res = 0.0
            for i in range(n):
                res += a[lo + i]
            return res
        r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
        r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
        i = 8
        stop = n - n % 8
        while i < stop:
            r0 += a[lo + i]
            r1 += a[lo + i + 1]
            r2 += a[lo + i + 2]
            r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]
            r5 += a[lo + i + 5]
            r6 += a[lo + i + 6]
            r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res

    @njit(cache=True)
    def _pairwise_sum(a, lo, n):
        """numpy's pairwise sum of a[lo:lo + n]: halves (cut to a multiple of 8) above 128 items.

        Evaluates numpy's recursion with an explicit stack (numba's cache does not
        reload recursive functions reliably): the same leaves, added in the same tree.
        """
        if n <= 128:
            return _pairwise_leaf(a, lo, n)
        f_lo = np.empty(64, dtype=np.int64)
        f_n = np.empty(64, dtype=np.int64)
        f_stage = np.empty(64, dtype=np.int64)
        values = np.empty(64, dtype=np.float64)
        top = 0
        n_values = 0
        f_lo[0], f_n[0], f_stage[0] = lo, n, 0
        while top >= 0:
            m = f_n[top]
            if m <= 128:
                values[n_values] = _pairwise_leaf(a, f_lo[top], m)
                n_values += 1
                top -= 1
                continue
            half = m // 2
            half -= half % 8
            if f_stage[top] == 0:
                f_stage[top] = 1
                f_lo[top + 1], f_n[top + 1], f_stage[top + 1] = f_lo[top], half, 0
                top += 1
            elif f_stage[top] == 1:
                f_stage[top] = 2
                f_lo[top + 1], f_n[top + 1], f_stage[top + 1] = f_lo[top] + half, m - half, 0
                top += 1
            else:
                values[n_values - 2] = values[n_values - 2] + values[n_values - 1]
                n_values -= 1
                top -= 1
        return values[0]

    @njit(cache=True)
    def _numpy_std(a):
        """np.std of a non-empty contiguous float64 vector, with numpy's summation order.

        np.sum of a contiguous float64 vector is 0 plus the pairwise sum of all of
        it; np.std is the square root of the sum of squared deviations from that
        mean divided by the count.
        """
        n = a.shape[0]
        mean = (0.0 + _pairwise_sum(a, 0, n)) / n
        dev = np.empty(n, dtype=np.float64)
        for i in range(n):
            d = a[i] - mean
            dev[i] = d * d
        return np.sqrt((0.0 + _pairwise_sum(dev, 0, n)) / n)

    @njit(cache=True)
    def _map_snr_numba(corr_map, ir, ip, iz, exclusion):
        """Map value at (ir, ip, iz) over the standard deviation of the finite map values outside a box.

        The box spans ``exclusion`` bins on each side of the peak in each axis
        (clipped to the map); a negative ``exclusion`` keeps every value. The
        values are taken in C order, as boolean indexing does, and their
        standard deviation is ``_numpy_std``.

        Returns:
            The ratio, or NaN when no value remains or the deviation is below 1e-12.
        """
        nr, nphi, nz = corr_map.shape
        r_lo, r_hi = max(0, ir - exclusion), min(nr, ir + exclusion + 1)
        p_lo, p_hi = max(0, ip - exclusion), min(nphi, ip + exclusion + 1)
        z_lo, z_hi = max(0, iz - exclusion), min(nz, iz + exclusion + 1)
        buf = np.empty(nr * nphi * nz, dtype=np.float64)
        m = 0
        for i in range(nr):
            in_r = exclusion >= 0 and r_lo <= i < r_hi
            for j in range(nphi):
                in_rp = in_r and p_lo <= j < p_hi
                for k in range(nz):
                    if in_rp and z_lo <= k < z_hi:
                        continue
                    v = corr_map[i, j, k]
                    if np.isfinite(v):
                        buf[m] = v
                        m += 1
        if m == 0:
            return np.nan
        rms = _numpy_std(buf[:m])
        if rms < 1e-12:
            return np.nan
        return corr_map[ir, ip, iz] / rms

    @njit(cache=True)
    def _top_peaks_numba(corr_map, rho_vec, phi_vec_deg, z_vec, n, d_rho, d_phi, d_z):
        """Indices and values of the top-n peaks of a 3D map with minimum separation.

        Each round takes the first maximum in C order with NaN ignored (np.nanargmax)
        and blanks every node closer than (d_rho, d_phi, d_z) in all three axes, phi
        wrapped; it stops when every value is NaN or the maximum is NaN.

        Returns:
            (idx, val): (k, 3) int64 indices and (k,) values of the k <= n peaks.
        """
        work = corr_map.copy()
        nr, nphi, nz = work.shape
        flat = work.reshape(nr * nphi * nz)
        idx = np.empty((n, 3), dtype=np.int64)
        val = np.empty(n, dtype=np.float64)
        rho_mask = np.empty(nr, dtype=np.bool_)
        phi_mask = np.empty(nphi, dtype=np.bool_)
        z_mask = np.empty(nz, dtype=np.bool_)
        count = 0
        for _ in range(n):
            best = -np.inf
            best_i = 0
            all_nan = True
            for i in range(flat.shape[0]):
                v = flat[i]
                if np.isnan(v):
                    continue
                all_nan = False
                if v > best:
                    best = v
                    best_i = i
            if all_nan:
                break
            v = flat[best_i]
            if np.isnan(v):
                break
            i0 = best_i // (nphi * nz)
            i1 = (best_i // nz) % nphi
            i2 = best_i % nz
            idx[count, 0] = i0
            idx[count, 1] = i1
            idx[count, 2] = i2
            val[count] = v
            count += 1
            for i in range(nr):
                rho_mask[i] = abs(rho_vec[i] - rho_vec[i0]) < d_rho
            for j in range(nphi):
                diff = abs(phi_vec_deg[j] - phi_vec_deg[i1])
                phi_mask[j] = min(diff, 360.0 - diff) < d_phi
            for k in range(nz):
                z_mask[k] = abs(z_vec[k] - z_vec[i2]) < d_z
            for i in range(nr):
                if not rho_mask[i]:
                    continue
                for j in range(nphi):
                    if not phi_mask[j]:
                        continue
                    for k in range(nz):
                        if z_mask[k]:
                            work[i, j, k] = np.nan
        return idx[:count], val[:count]


def _build_z_vec(z_min, z_max, n_z, spacing='linear', surface_offset=0.1):
    """Build a z-axis vector with linear or log spacing.

    Log spacing concentrates grid density near the ice surface (z=0).

    Args:
        z_min: Minimum z (meters, negative).
        z_max: Maximum z (meters, should be >= 0 for log mode).
        n_z: Number of grid points.
        spacing: 'linear' or 'log'.
        surface_offset: Minimum |z| for the shallowest log bin (meters).

    Returns:
        np.ndarray of length n_z, sorted ascending.
    """
    if spacing == 'log':
        if z_max > surface_offset and z_min < 0:
            n_above = max(2, int(round(n_z * np.log(z_max / surface_offset)
                                       / (np.log(z_max / surface_offset) + np.log(-z_min / surface_offset)))))
            n_below = max(2, n_z - n_above)
            below = -np.geomspace(surface_offset, -z_min, n_below)[::-1]
            above = np.geomspace(surface_offset, z_max, n_above)
            return np.concatenate((below, above))
        if z_max >= 0 and z_min < 0:
            z_depths = np.geomspace(surface_offset, -z_min, n_z)
            return -z_depths[::-1]
        logger.warning(
            "z_spacing='log' requires z_min < 0 and z_max >= 0; "
            "got [%s, %s]. Falling back to linear.", z_min, z_max)
    return np.linspace(z_min, z_max, n_z)


def _build_split_z_vec(z_min, z_max, below, above):
    """Build a z vector from an in-ice block and an air block that meet at the surface.

    The in-ice block covers [z_min, min(z_max, 0)] and is built by _build_z_vec with the
    count, spacing and offset of ``below``, so it is the grid an in-ice search volume
    ending at z = 0 builds; it is left out when z_min is not below the surface. The air
    block holds ``above['n']`` points from max(``above['offset']``, z_min) to z_max, log
    or linear as ``above['spacing']`` says, and is left out when z_max does not reach its
    start.

    Args:
        z_min: Lower edge of the volume in m.
        z_max: Upper edge of the volume in m.
        below: Dict with keys n, spacing and offset for the in-ice block.
        above: Dict with keys n, spacing and offset for the air block.

    Returns:
        np.ndarray sorted ascending.

    Raises:
        ValueError: If neither block holds a point.
    """
    parts = []
    if z_min < 0:
        parts.append(_build_z_vec(z_min, min(z_max, 0.0), below['n'],
                                  below['spacing'], below['offset']))
    air_lo = max(above['offset'], z_min)
    if z_max > air_lo:
        if above['spacing'] == 'log':
            parts.append(np.geomspace(air_lo, z_max, above['n']))
        else:
            parts.append(np.linspace(air_lo, z_max, above['n']))
    if not parts:
        raise ValueError(
            "the split z grid holds no point: z_grid_below needs z_min < 0 and "
            "z_grid_above needs z_max > max(offset, z_min); got z in [%s, %s] with "
            "offset %s" % (z_min, z_max, above['offset']))
    return np.concatenate(parts)


def _build_split_z_window(z_lo, z_hi, step, above):
    """Build the z vector of a local window that may cross the ice surface.

    The in-ice part runs from z_lo to min(z_hi, 0) linearly at ``step``, as a window
    clamped at the surface does in an in-ice search volume. The air part runs from
    max(above['offset'], z_lo) to z_hi with as many points as the linear step gives
    there, placed log or linear as ``above['refine_spacing']`` says; a window entirely
    in the air therefore starts at z_lo.

    Args:
        z_lo: Lower edge of the window in m.
        z_hi: Upper edge of the window in m.
        step: Linear step in m.
        above: Dict with keys offset and refine_spacing for the air part.

    Returns:
        np.ndarray sorted ascending, empty when the window holds no point.
    """
    parts = []
    if z_lo <= 0:
        ice = np.arange(z_lo, min(z_hi, 0.0) + step, step)
        parts.append(ice[ice <= 1e-9])
    air_lo = max(above['offset'], z_lo)
    if z_hi >= air_lo:
        air = np.arange(air_lo, z_hi + step, step)
        air = air[air <= z_hi + 1e-9]
        if above['refine_spacing'] == 'log' and len(air) > 1:
            air = np.geomspace(air_lo, z_hi, len(air))
        parts.append(air)
    if not parts:
        return np.empty(0)
    return np.concatenate(parts)


if USE_NUMBA:
    @njit(fastmath=False, cache=True)
    def _numpy_pairwise_sum(a):
        """``np.sum`` of a contiguous 1D float64 row, in numpy's pairwise order (8 to 128 elements)."""
        n = a.shape[0]
        r0, r1, r2, r3 = a[0], a[1], a[2], a[3]
        r4, r5, r6, r7 = a[4], a[5], a[6], a[7]
        i = 8
        stop = n - n % 8
        while i < stop:
            r0 += a[i]
            r1 += a[i + 1]
            r2 += a[i + 2]
            r3 += a[i + 3]
            r4 += a[i + 4]
            r5 += a[i + 5]
            r6 += a[i + 6]
            r7 += a[i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[i]
            i += 1
        return 0.0 + res

    @njit(fastmath=False, cache=True)
    def _far_fd_points_numba(x, lb, ub, abs_step, rel_step):
        """The point and its two forward-difference points of the far-field L-BFGS-B, in deg and rad.

        ``_lbfgsb_fd_steps`` and the points ``negative_and_gradient`` builds from them, operation for
        operation (no fast math), and ``np.radians`` of each coordinate.

        Returns:
            (points (3, 2) deg, zen (3,) rad, az (3,) rad).
        """
        n = x.shape[0]
        h = np.empty(n)
        open_bounds = True
        for i in range(n):
            sign = 1.0 if x[i] >= 0 else -1.0
            if (x[i] + abs_step) - x[i] == 0:
                h[i] = rel_step * sign * max(1.0, abs(x[i]))
            else:
                h[i] = abs_step
            if not (lb[i] == -np.inf and ub[i] == np.inf):
                open_bounds = False
        if not open_bounds:
            for i in range(n):
                lower = x[i] - lb[i]
                upper = ub[i] - x[i]
                stepped = x[i] + h[i]
                farther = lower if (lower >= upper or np.isnan(lower)) else upper
                fitting = abs(h[i]) <= farther
                if (stepped < lb[i] or stepped > ub[i]) and fitting:
                    h[i] = -h[i]
                if upper >= lower and not fitting:
                    h[i] = upper
                if upper < lower and not fitting:
                    h[i] = -lower
        points = np.empty((3, n))
        for j in range(3):
            for i in range(n):
                points[j, i] = x[i] + (h[i] if j == i + 1 else 0.0)
        zen = np.empty(3)
        az = np.empty(3)
        for j in range(3):
            zen[j] = points[j, 0] * (np.pi / 180.0)
            az[j] = points[j, 1] * (np.pi / 180.0)
        return points, zen, az

    @njit(fastmath=False, cache=True)
    def _plane_wave_times_numba(sin_zen, cos_zen, cos_az, sin_az, x, y, z, gl_weights, lengths, n2, layered,
                                c_m_per_ns):
        """``plane_wave_times`` from the sines and cosines of the directions, operation for operation.

        The in-ice delay of every direction is computed as ``plane_wave_times`` computes it for that
        zenith (numpy's pairwise sum over the Gauss-Legendre nodes, no fast math), so with sines and
        cosines from numpy the arrival times equal it bit for bit.

        Args:
            sin_zen, cos_zen, cos_az, sin_az: (n_dir,) np.sin / np.cos of the direction angles (rad).
            x, y, z: (n_ch,) antenna coordinates (z relative to the surface).
            gl_weights: (nodes,) Gauss-Legendre weights.
            lengths: (n_layers, n_ch) path length per layer (layered), or (1, n_ch) of
                0.5 * -min(z, 0) for a single exponential.
            n2: (n_layers, n_ch, nodes) squared refractive index at the nodes.
            layered: Layered profile (sum of layers, n^2 - sin^2 floored at 0).
            c_m_per_ns: Speed of light in m/ns.

        Returns:
            (n_dir, n_ch) arrival times in ns.
        """
        n_dir = sin_zen.shape[0]
        n_ch = x.shape[0]
        nodes = gl_weights.shape[0]
        out = np.empty((n_dir, n_ch))
        term = np.empty(nodes)
        for d in range(n_dir):
            sin2 = sin_zen[d] ** 2
            for c in range(n_ch):
                if layered:
                    in_ice = 0.0
                    for l in range(n2.shape[0]):
                        for j in range(nodes):
                            v = n2[l, c, j] - sin2
                            if not (v >= 0.0 or np.isnan(v)):
                                v = 0.0
                            term[j] = gl_weights[j] * np.sqrt(v)
                        in_ice = in_ice + (0.5 * lengths[l, c]) * _numpy_pairwise_sum(term)
                else:
                    for j in range(nodes):
                        term[j] = gl_weights[j] * np.sqrt(n2[0, c, j] - sin2)
                    in_ice = lengths[0, c] * _numpy_pairwise_sum(term)
                above = z[c] if (z[c] >= 0.0 or np.isnan(z[c])) else 0.0
                vertical = (in_ice - above * cos_zen[d]) / c_m_per_ns
                horizontal = -sin_zen[d] * (cos_az[d] * x[c] + sin_az[d] * y[c])
                out[d, c] = horizontal / c_m_per_ns + vertical
        return out

    @njit(fastmath=False, cache=True)
    def _multiray_point_tts_numba(x, y, z, ant_pos, tables, r_min, dr_inv, nr, z_min, dz_inv, nz, n_rt):
        """Travel time of every channel and ray type at one source point and whether it is usable.

        The multi-ray branch of ``_correlation_at_point`` step for step: the horizontal distance
        floored at 1 m and ``_bilinear_scalar_numba`` on each table (slot ci * n_rt + rti); a time is
        usable when finite and positive, else it stays -inf.

        Args:
            x, y, z: Source position (m).
            ant_pos: (n_ch, 2) antenna x, y.
            tables: Typed list of the (nr, nz) table values per slot.
            r_min, dr_inv, nr, z_min, dz_inv, nz: Per-slot grid parameters.
            n_rt: Ray-type slots per channel.

        Returns:
            (tt_vals, tt_valid) of shape (n_ch, n_rt).
        """
        n_ch = ant_pos.shape[0]
        tt_vals = np.full((n_ch, n_rt), -np.inf)
        tt_valid = np.zeros((n_ch, n_rt), dtype=np.bool_)
        for ci in range(n_ch):
            dx = x - ant_pos[ci, 0]
            dy = y - ant_pos[ci, 1]
            r = np.sqrt(dx * dx + dy * dy)
            if 1.0 > r:
                r = 1.0
            for rti in range(n_rt):
                s = ci * n_rt + rti
                tt = _bilinear_scalar_numba(tables[s], r_min[s], dr_inv[s], nr[s], z_min[s], dz_inv[s], nz[s],
                                            r, z)
                if np.isfinite(tt) and tt > 0:
                    tt_vals[ci, rti] = tt
                    tt_valid[ci, rti] = True
        return tt_vals, tt_valid

    @njit(fastmath=False, cache=True)
    def _lbfgsb_fd_points(x, phi_shift, lb, ub):
        """Points at which scipy's L-BFGS-B evaluates an objective of (rho, phi_shifted, z) without a gradient.

        Row 0 is x and row i + 1 is x with coordinate i moved by ``_lbfgsb_fd_step``. The azimuth of a
        row is (row[1] + phi_shift) wrapped to [0, 360) deg and converted to radians, rounded as the
        numpy scalar operations of the Python objective round it; its cosine and sine are left to numpy,
        whose libm results the compiled ones do not always equal.

        Returns:
            (points, phi_rad) of shapes (4, 3) and (4,).
        """
        n = x.shape[0]
        bounded = False
        for i in range(n):
            if not (lb[i] == -np.inf and ub[i] == np.inf):
                bounded = True
        points = np.empty((n + 1, n), dtype=np.float64)
        for k in range(n + 1):
            for i in range(n):
                points[k, i] = x[i]
        for i in range(n):
            points[i + 1, i] = x[i] + _lbfgsb_fd_step(x[i], lb[i], ub[i], bounded)
        phi_rad = np.empty(n + 1, dtype=np.float64)
        for k in range(n + 1):
            phi_rad[k] = ((points[k, 1] + phi_shift) % 360.0) * (np.pi / 180.0)
        return points, phi_rad

    @njit(fastmath=False, cache=True)
    def _grouped_fd_value_grad(
            points, cos_phi, sin_phi, pa_x, pa_y, ant_pos, tables, r_min, dr_inv, nr, z_min, dz_inv, nz, n_rt,
            corr_packed, corr_lengths, corr_dts, corr_offsets, pair_ch1, pair_ch2, pair_weights,
            ch_group, group_rts, group_nrt, n_pairs, w_sum, walk):
        """Grouped multi-ray optimizer objective at the points of ``_lbfgsb_fd_points`` and its forward differences.

        Each value is the 'mr_tables' grouped branch of ``_correlation_at_point``: the source position
        rho * cos + PA centre, ``_multiray_point_tts_numba`` and ``_scalar_grouped_corr_numba``. Passing
        the result to ``_minimize_lbfgsb`` gives the values and gradients scipy's L-BFGS-B computes by
        differencing the Python objective, with one compiled call for the four values instead of four
        Python calls.

        Args:
            points, cos_phi, sin_phi: Rows of ``_lbfgsb_fd_points`` and the cosine and sine of their azimuths.
            Remaining arguments: PA centre x, y, the ``_multiray_point_tts_numba`` tables and the
                ``_scalar_grouped_corr_numba`` correlation arguments and work counters.

        Returns:
            (f, g): objective value at row 0 and the (3,) forward-difference gradient.
        """
        n = points.shape[1]
        f = np.empty(n + 1, dtype=np.float64)
        for k in range(n + 1):
            x = points[k, 0] * cos_phi[k] + pa_x
            y = points[k, 0] * sin_phi[k] + pa_y
            tt_vals, tt_valid = _multiray_point_tts_numba(x, y, points[k, 2], ant_pos, tables, r_min, dr_inv, nr,
                                                          z_min, dz_inv, nz, n_rt)
            f[k] = _scalar_grouped_corr_numba(tt_vals, tt_valid, corr_packed, corr_lengths, corr_dts, corr_offsets,
                                              pair_ch1, pair_ch2, pair_weights, ch_group, group_rts, group_nrt,
                                              n_pairs, w_sum, walk)
        g = np.empty(n, dtype=np.float64)
        for i in range(n):
            g[i] = (f[i + 1] - f[0]) / (points[i + 1, i] - points[0, i])
        return f[0], g


if USE_NUMBA and USE_NUMBA_GROUPED:
    def grouped_multiray_points(corr_data, tt_all, channels, ch_to_group, n_groups, pair_weights=None):
        """Grouped multiray correlator on ``_grouped_multiray_points_numba``, equal to ``grouped_multiray_numba``.

        Same inputs, packing and combo table as ``grouped_multiray_numba``; the map is
        computed in one point-major call that evaluates each distinct pair term once per
        point.

        Args:
            corr_data: (corr_array, dt, offset) per pair.
            tt_all: Maps channel -> {ray type -> travel-time grid}.
            channels: Channel IDs.
            ch_to_group: Maps channel ID to depth group index.
            n_groups: Number of depth groups.
            pair_weights: Per-pair weights or None (all 1).

        Returns:
            (mean_corr_map, max_corr), the map in the grid shape (``np.zeros(1), nan``
            when no channel has a ray type).
        """
        grid_shape = next((tt_all[ch][rt].shape for ch in channels for rt in tt_all.get(ch, {})), None)
        if grid_shape is None:
            return np.zeros(1), np.nan
        n_ch = len(channels)
        n_pairs = n_ch * (n_ch - 1) // 2
        tt_packed, ch_available_rts = pack_tt_grids(tt_all, channels, grid_shape)
        n_rt = tt_packed.shape[1]
        corr_packed, corr_lengths, dts, offsets = pack_corr_data(corr_data, n_pairs)
        combo_table = build_combo_table(channels, ch_to_group, n_groups, ch_available_rts, n_rt=n_rt)
        pair_ch1 = np.array([a for a, _ in itertools.combinations(range(n_ch), 2)], dtype=np.int64)
        pair_ch2 = np.array([b for _, b in itertools.combinations(range(n_ch), 2)], dtype=np.int64)
        pw = (np.asarray(pair_weights, dtype=np.float64) if pair_weights is not None
              else np.ones(n_pairs, dtype=np.float64))
        keys = (np.arange(n_pairs)[None, :] * n_rt + combo_table[:, pair_ch1]) * n_rt + combo_table[:, pair_ch2]
        rows, inverse = np.unique(keys, return_inverse=True)
        result = _grouped_multiray_points_numba(
            tt_packed, corr_packed, corr_lengths, 1.0 / dts, offsets,
            rows // (n_rt * n_rt), rows // n_rt % n_rt, rows % n_rt,
            np.ascontiguousarray(inverse.reshape(keys.shape), dtype=np.int64), pair_ch1, pair_ch2,
            np.ascontiguousarray(pw), _grouped_weight_sum(pw))
        result[result == -np.inf] = 0.0
        result = result.reshape(grid_shape)
        return result, float(np.max(result)) if result.size > 0 else np.nan
