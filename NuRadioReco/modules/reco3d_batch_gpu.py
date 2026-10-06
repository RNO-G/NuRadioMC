"""GPU backend of the batched coarse maps (``reco3d_batch``): per-pair contributions and per-setting sums on a CUDA device.

The coarse travel times of a grid stay on the device for the life of the backend; the base series of a
lockstep go up once (``device_series``), per batch only the small column and weight tables go up (packed into a
few arrays), and the maps come back in float64 through pinned memory. Each map sums its own pairs in its own order (dense weight rows with zeros
for the pairs it does not use, or its own entry list), so a GPU map equals the CPU map up to the rounding of
the device arithmetic (contraction to fused multiply-adds may differ), not bit for bit.

Needs CuPy and a CUDA device.
"""

import itertools
import time

import numpy as np

import cupy as cp
import cupyx

_SRC = r'''
#define MC 16
#define NEG_INF (-__longlong_as_double(0x7ff0000000000000LL))

__device__ __forceinline__ double total_factor(double wt) {
    if (wt > 0.0) return 1.0 / wt;
    return 0.0;
}

// reco3d_kernels._coverage_ramp step for step; the __*_rn intrinsics are never contracted into fused
// multiply-adds, so the factor equals the CPU one bit for bit.
__device__ __forceinline__ double coverage_ramp(double wv, double wt, double vfloor) {
    if (wv <= 0.0 || wt <= 0.0) return 0.0;
    double p = __dmul_rn(vfloor, wt);
    double e = __fma_rn(vfloor, wt, -p);
    if (wv > p || (wv == p && e <= 0.0)) return 1.0;
    double two_w = __dmul_rn(2.0, wv);
    if (two_w < p || (two_w == p && e >= 0.0)) return 0.0;
    double d = __dsub_rn(two_w, p);
    double n = __dsub_rn(d, e);
    double z = __dsub_rn(n, d);
    double n_err = __dsub_rn(__dsub_rn(d, __dsub_rn(n, z)), __dadd_rn(e, z));
    double q = __ddiv_rn(n, p);
    double r = __dsub_rn(__dadd_rn(__fma_rn(-q, p, n), n_err), __dmul_rn(q, e));
    return __dadd_rn(q, __ddiv_rn(r, p));
}

__device__ __forceinline__ double valid_mean(double acc, double wv, double f) {
    if (f == 0.0) return 0.0;
    return __dmul_rn(__ddiv_rn(acc, wv), f);
}

__device__ __forceinline__ double normalised(double acc, double wv, double wt, int valid_norm, double vfloor) {
    if (valid_norm) return valid_mean(acc, wv, coverage_ramp(wv, wt, vfloor));
    return acc * total_factor(wt);
}

__device__ __forceinline__ bool contribution(
        const double* __restrict__ tts, const unsigned char* __restrict__ valid, long long n_points,
        long long pt, int c1, int c2, double off, double inv_dt, long long last,
        const double* __restrict__ series, long long start, long long lo, long long hi, double* c) {
    if (!valid[c1 * n_points + pt] || !valid[c2 * n_points + pt]) return false;
    double kf = (tts[c1 * n_points + pt] - tts[c2 * n_points + pt] - off) * inv_dt;
    double fk = floor(kf);
    long long kk = (long long)fk;
    if (kk < 0 || kk >= last) return false;
    double alpha = kf - (double)kk;
    double y0 = (kk >= lo && kk < hi) ? series[start + kk - lo] : 0.0;
    double y1 = (kk + 1 >= lo && kk + 1 < hi) ? series[start + kk + 1 - lo] : 0.0;
    *c = y0 + (y1 - y0) * alpha;
    return true;
}

extern "C" __global__ void maps_dense(
        const double* __restrict__ tts, const unsigned char* __restrict__ valid, long long n_points,
        const int* __restrict__ col_ch1, const int* __restrict__ col_ch2, const long long* __restrict__ col_last,
        const double* __restrict__ col_inv_dt, const double* __restrict__ col_off,
        const long long* __restrict__ col_start, const long long* __restrict__ col_lo,
        const long long* __restrict__ col_hi, int n_cols,
        const double* __restrict__ series,
        const double* __restrict__ W, const double* __restrict__ Wabs, const double* __restrict__ wtotal,
        int n_maps, int valid_norm, double vfloor, double* __restrict__ out) {
    long long pt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;
    int m0 = blockIdx.y * MC;
    int nm = n_maps - m0;
    if (nm > MC) nm = MC;
    double acc[MC];
    double wv[MC];
    #pragma unroll
    for (int m = 0; m < MC; ++m) { acc[m] = 0.0; wv[m] = 0.0; }
    for (int j = 0; j < n_cols; ++j) {
        double c;
        if (!contribution(tts, valid, n_points, pt, col_ch1[j], col_ch2[j], col_off[j], col_inv_dt[j],
                          col_last[j], series, col_start[j], col_lo[j], col_hi[j], &c)) continue;
        #pragma unroll
        for (int m = 0; m < MC; ++m) {
            if (m < nm) {
                long long e = (long long)(m0 + m) * n_cols + j;
                acc[m] += c * W[e];
                wv[m] += Wabs[e];
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MC; ++m) {
        if (m < nm) out[(long long)(m0 + m) * n_points + pt] =
            normalised(acc[m], wv[m], wtotal[m0 + m], valid_norm, vfloor);
    }
}

extern "C" __global__ void maps_csr(
        const double* __restrict__ tts, const unsigned char* __restrict__ valid, long long n_points,
        const int* __restrict__ col_ch1, const int* __restrict__ col_ch2, const long long* __restrict__ col_last,
        const double* __restrict__ col_inv_dt, const double* __restrict__ col_off,
        const long long* __restrict__ col_start, const long long* __restrict__ col_lo,
        const long long* __restrict__ col_hi,
        const double* __restrict__ series,
        const long long* __restrict__ ptr, const int* __restrict__ ecol, const double* __restrict__ ew,
        const double* __restrict__ ewabs, const double* __restrict__ wtotal,
        int valid_norm, double vfloor, double* __restrict__ out) {
    long long pt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;
    int m = blockIdx.y;
    double acc = 0.0, wv = 0.0;
    for (long long e = ptr[m]; e < ptr[m + 1]; ++e) {
        int j = ecol[e];
        double c;
        if (!contribution(tts, valid, n_points, pt, col_ch1[j], col_ch2[j], col_off[j], col_inv_dt[j],
                          col_last[j], series, col_start[j], col_lo[j], col_hi[j], &c)) continue;
        acc += c * ew[e];
        wv += ewabs[e];
    }
    out[(long long)m * n_points + pt] = normalised(acc, wv, wtotal[m], valid_norm, vfloor);
}

__device__ __forceinline__ bool bilinear_ok(
        const double* __restrict__ values, const unsigned char* __restrict__ ok, long long nr_max,
        long long nz_max, long long ti, double ri, double zi, long long nr, long long nz, int tolerant,
        double* v_out) {
    double fri = floor(ri), fzi = floor(zi);
    long long i0 = (long long)fri, j0 = (long long)fzi;
    double fx, fy;
    if (i0 < 0 || j0 < 0) return false;
    if (i0 >= nr - 1) {
        if (tolerant && ri <= nr - 1 + 1e-9) { i0 = nr - 2; fx = 1.0; } else return false;
    } else fx = ri - (double)i0;
    if (j0 >= nz - 1) {
        if (tolerant && zi <= nz - 1 + 1e-9) { j0 = nz - 2; fy = 1.0; } else return false;
    } else fy = zi - (double)j0;
    long long b = (ti * nr_max + i0) * nz_max + j0;
    if (!(ok[b] && ok[b + nz_max] && ok[b + 1] && ok[b + nz_max + 1])) return false;
    double v = (1.0 - fx) * (1.0 - fy) * values[b] + fx * (1.0 - fy) * values[b + nz_max]
             + (1.0 - fx) * fy * values[b + 1] + fx * fy * values[b + nz_max + 1];
    if (v > 0.0) { *v_out = v; return true; }
    return false;
}

#define NCH_MAX 32

extern "C" __global__ void grid_maps(
        const double* __restrict__ td_values, const unsigned char* __restrict__ td_ok,
        long long nr_max, long long nz_max, int tolerant,
        const long long* __restrict__ ch_slot, const double* __restrict__ ch_r_min,
        const double* __restrict__ ch_dr_inv, const long long* __restrict__ ch_nr,
        const double* __restrict__ ch_z_min, const double* __restrict__ ch_dz_inv,
        const long long* __restrict__ ch_nz, const double* __restrict__ ch_x, const double* __restrict__ ch_y,
        double pa_x, double pa_y,
        const double* __restrict__ axes, const long long* __restrict__ g_rho, const long long* __restrict__ g_phi,
        const long long* __restrict__ g_z, const long long* __restrict__ g_n,
        const long long* __restrict__ m_out, const long long* __restrict__ m_grid,
        const long long* __restrict__ m_ptr, const double* __restrict__ m_wtotal, int n_maps,
        const int* __restrict__ e_c1, const int* __restrict__ e_c2, const long long* __restrict__ e_last,
        const double* __restrict__ e_inv_dt, const double* __restrict__ e_off,
        const long long* __restrict__ e_start, const long long* __restrict__ e_lo, const long long* __restrict__ e_hi,
        const double* __restrict__ e_w, const double* __restrict__ e_wabs,
        const double* __restrict__ series, long long n_total, int valid_norm, double vfloor,
        double* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_total) return;
    int lo_m = 0, hi_m = n_maps - 1;
    while (lo_m < hi_m) {
        int mid = (lo_m + hi_m + 1) / 2;
        if (m_out[mid] <= t) lo_m = mid; else hi_m = mid - 1;
    }
    int m = lo_m;
    long long pt = t - m_out[m];
    long long g = m_grid[m];
    long long n_phi = g_n[3 * g + 1], n_z = g_n[3 * g + 2];
    long long iz = pt % n_z;
    long long row = pt / n_z;
    long long ip = row % n_phi;
    long long ir = row / n_phi;
    double rho = axes[g_rho[g] + ir], phi = axes[g_phi[g] + ip], z = axes[g_z[g] + iz];
    double x_src = rho * cos(phi) + pa_x;
    double y_src = rho * sin(phi) + pa_y;
    double tt[NCH_MAX];
    unsigned int done = 0, okm = 0;
    double acc = 0.0, wv = 0.0;
    for (long long e = m_ptr[m]; e < m_ptr[m + 1]; ++e) {
        int cs[2] = {e_c1[e], e_c2[e]};
        for (int q = 0; q < 2; ++q) {
            int c = cs[q];
            if (!(done & (1u << c))) {
                done |= (1u << c);
                double dx = x_src - ch_x[c], dy = y_src - ch_y[c];
                double r = sqrt(dx * dx + dy * dy);
                if (r < 1.0) r = 1.0;
                double v;
                if (bilinear_ok(td_values, td_ok, nr_max, nz_max, ch_slot[c], (r - ch_r_min[c]) * ch_dr_inv[c],
                                (z - ch_z_min[c]) * ch_dz_inv[c], ch_nr[c], ch_nz[c], tolerant, &v)) {
                    tt[c] = v;
                    okm |= (1u << c);
                }
            }
        }
        if (!(okm & (1u << cs[0])) || !(okm & (1u << cs[1]))) continue;
        double kf = (tt[cs[0]] - tt[cs[1]] - e_off[e]) * e_inv_dt[e];
        double fk = floor(kf);
        long long kk = (long long)fk;
        if (kk < 0 || kk >= e_last[e]) continue;
        double alpha = kf - (double)kk;
        long long lo = e_lo[e], hi = e_hi[e], st = e_start[e];
        double y0 = (kk >= lo && kk < hi) ? series[st + kk - lo] : 0.0;
        double y1 = (kk + 1 >= lo && kk + 1 < hi) ? series[st + kk + 1 - lo] : 0.0;
        acc += (y0 + (y1 - y0) * alpha) * e_w[e];
        wv += e_wabs[e];
    }
    out[t] = normalised(acc, wv, m_wtotal[m], valid_norm, vfloor);
}

extern "C" __global__ void row_argmax(const double* __restrict__ W, long long size, const int* __restrict__ rows,
                                      double* __restrict__ pv, long long* __restrict__ pi) {
    __shared__ double sv[256];
    __shared__ long long si[256];
    const double* w = W + (long long)rows[blockIdx.y] * size;
    double best = NEG_INF;
    long long bi = size;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += (long long)gridDim.x * blockDim.x) {
        double v = w[i];
        if (v > best) { best = v; bi = i; }
    }
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) {
            double ov = sv[threadIdx.x + h];
            long long oi = si[threadIdx.x + h];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pv[(long long)blockIdx.y * gridDim.x + blockIdx.x] = sv[0];
        pi[(long long)blockIdx.y * gridDim.x + blockIdx.x] = si[0];
    }
}

extern "C" __global__ void mask_boxes(double* __restrict__ W, long long size, long long nr, long long nphi,
                                      long long nz, const int* __restrict__ rows,
                                      const unsigned char* __restrict__ rm, const unsigned char* __restrict__ pm,
                                      const unsigned char* __restrict__ zm) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (t >= size) return;
    long long iz = t % nz;
    long long row = t / nz;
    long long ip = row % nphi;
    long long ir = row / nphi;
    if (rm[k * nr + ir] && pm[k * nphi + ip] && zm[k * nz + iz]) W[(long long)rows[k] * size + t] = NEG_INF;
}

extern "C" __global__ void masked_moments(const double* __restrict__ d, long long nr, long long nphi,
                                          long long nz_full, long long z0, long long nzv, long long r_lo,
                                          long long r_hi, long long p_lo, long long p_hi, long long z_lo,
                                          long long z_hi, int pass, double mean, double* __restrict__ partial) {
    __shared__ double ss[256];
    __shared__ double sc[256];
    long long total = nr * nphi * nzv;
    double s = 0.0, c = 0.0;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        long long iz = t % nzv;
        long long row = t / nzv;
        long long ip = row % nphi;
        long long ir = row / nphi;
        if (ir >= r_lo && ir < r_hi && ip >= p_lo && ip < p_hi && iz >= z_lo && iz < z_hi) continue;
        double v = d[(ir * nphi + ip) * nz_full + z0 + iz];
        if (!isfinite(v)) continue;
        if (pass == 1) { s += v; c += 1.0; } else { double x = v - mean; s += x * x; }
    }
    ss[threadIdx.x] = s;
    sc[threadIdx.x] = c;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) { ss[threadIdx.x] += ss[threadIdx.x + h]; sc[threadIdx.x] += sc[threadIdx.x + h]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { partial[2 * blockIdx.x] = ss[0]; partial[2 * blockIdx.x + 1] = sc[0]; }
}
'''

_MC = 16
_THREADS = 128
_RED_THREADS = 256
_RED_BLOCKS = 160


def _upload_packed(arrays, dtype):
    """Upload 1-D host arrays of one dtype as one device array; returns the device views in order."""
    host = [np.asarray(a, dtype=dtype).ravel() for a in arrays]
    dev = cp.asarray(np.concatenate(host))
    out, pos = [], 0
    for h in host:
        out.append(dev[pos:pos + len(h)])
        pos += len(h)
    return out


class DeviceMap:
    """Coarse correlation map kept on the GPU: an (n_rho, n_phi, n_z) view of a device array.

    The search reads a coarse map only through peak extraction (``_extract_top_n_peaks``), the map
    SNR (``_compute_map_snr``), the standard deviation of its finite values (``map_snr_v2``), two maxima
    of the validation metrics and, for the sub-bin estimate, a few single values; a DeviceMap does each
    on the device and downloads only the result. The map values are the float64 values the GPU kernel
    computed (equal to the CPU map up to the last bits); peaks and maxima are exact functions of them,
    the standard deviations differ from numpy's in the last bits (summation order).
    """

    def __init__(self, backend, data, z0=0, nz=None):
        """Wrap a device array (flat, or 3D with an optional z window starting at ``z0``)."""
        self.backend = backend
        self.data = data
        self.z0 = z0
        self.nz = (data.shape[2] - z0 if nz is None else nz) if data.ndim == 3 else None

    @property
    def shape(self):
        """Shape of the view."""
        if self.data.ndim != 3:
            return self.data.shape
        return (self.data.shape[0], self.data.shape[1], self.nz)

    def reshape(self, *shape):
        """3D DeviceMap of a flat map (the search reshapes the flat coarse map to the grid)."""
        shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return DeviceMap(self.backend, self.data.reshape(shape))

    def view(self):
        """The device array of the view."""
        return self.data[:, :, self.z0:self.z0 + self.nz]

    def __getitem__(self, key):
        """A z window ``[:, :, a:b]`` stays on the device; any other index downloads the values."""
        if (isinstance(key, tuple) and len(key) == 3 and key[0] == slice(None) and key[1] == slice(None)
                and isinstance(key[2], slice) and key[2].step in (None, 1)):
            start, stop, _ = key[2].indices(self.nz)
            return DeviceMap(self.backend, self.data, self.z0 + start, max(0, stop - start))
        return cp.asnumpy(self.view()[key])

    def top_n_peaks(self, rho_vec, phi_vec_deg, z_vec, n, separation):
        """``_extract_top_n_peaks`` on the device (batched with the other searches inside a lockstep)."""
        from NuRadioReco.modules import reco3d_batch
        args = (self, rho_vec, phi_vec_deg, z_vec, n, separation)
        executor = reco3d_batch.current_executor()
        if executor is not None:
            return executor.request('peaks', args)
        return self.backend.batch_peaks(None, [(0, args)])[0]

    def map_snr(self, peak_idx, exclusion_bins=3):
        """``_compute_map_snr`` on the device."""
        return self.backend.map_snr(self, peak_idx, exclusion_bins)

    def finite_std(self):
        """Standard deviation of the finite values (``np.std(m[np.isfinite(m)])``) on the device."""
        return self.backend.map_std(self)

    def masked_max(self, z_mask=None, mask=None):
        """``np.nanmax`` of ``m[:, :, z_mask]`` or of ``m[mask]`` on the device."""
        return self.backend.masked_max(self, z_mask, mask)


class GpuCoarseBackend:
    """Coarse maps of a lockstep batch on a CUDA device (``coarse_backend`` of ``reco3d_batch``).

    Attributes:
        timing: Accumulated seconds of host preparation, host-to-device copies, kernels and
            device-to-host copies, and counts of batches and maps.
    """

    def __init__(self, reco, dense=True, device_maps=True):
        """Compile the kernels and bind the reconstruction object.

        Args:
            reco: InterferometricReco3D after ``begin``.

            dense: Use the dense weight kernel (16 maps per thread) when every map's pairs follow the
                column order; False always uses the per-map entry kernel.

            device_maps: Keep the coarse maps on the device (``DeviceMap``) and download only peaks and
                statistics; False downloads every map in float64.
        """
        self.reco = reco
        self.dense = dense
        module = cp.RawModule(code=_SRC)
        self._dense_kernel = module.get_function('maps_dense')
        self._csr_kernel = module.get_function('maps_csr')
        self._grid_kernel = module.get_function('grid_maps')
        self._argmax_kernel = module.get_function('row_argmax')
        self._mask_kernel = module.get_function('mask_boxes')
        self._moments_kernel = module.get_function('masked_moments')
        self._mask_cache = {}
        self.device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode()
        self.device_maps = device_maps
        self.map_timing = {'peaks_s': 0.0, 'n_peak_rounds': 0, 'n_peak_maps': 0, 'snr_s': 0.0, 'n_snr': 0}
        self._tables = None
        self.grid_timing = {'prep_s': 0.0, 'h2d_s': 0.0, 'kernel_s': 0.0, 'd2h_s': 0.0, 'n_rounds': 0,
                            'n_maps': 0, 'n_points': 0}
        self._tt = {}
        self.timing = {'prep_s': 0.0, 'h2d_s': 0.0, 'kernel_s': 0.0, 'd2h_s': 0.0, 'n_batches': 0,
                       'n_maps': 0, 'h2d_bytes': 0, 'd2h_bytes': 0}
        self.keep_device_maps = False
        self.last_device_maps = None
        self._pool = None

    def device_tt(self, lockstep, ref):
        """Travel times and validity of a grid on the device (uploaded once per grid and channel positions)."""
        key = (ref.grid_key, self.reco._position_key)
        entry = self._tt.get(key)
        if entry is None:
            tts, valid, index = lockstep.grid_tt(ref)
            t0 = time.perf_counter()
            entry = (cp.asarray(tts), cp.asarray(valid.view(np.uint8)), index, tts.shape[1])
            cp.cuda.Device().synchronize()
            self.timing['h2d_s'] += time.perf_counter() - t0
            self.timing['h2d_bytes'] += tts.nbytes + valid.nbytes
            self._tt[key] = entry
        return entry

    def device_series(self, lockstep, row_list):
        """Device buffer of the base series a plan reads, with each plan row's start and kept range.

        Every base series (the packed correlations of a pair set, raw or absolute) goes up once per lockstep,
        whole, with the nonzero extent [lo, hi) of each of its rows. A row's start is the position of its
        element lo in the buffer; the kernels read a row only inside [lo, hi) and take zero outside it.
        The pool holds a reference to every base array, so no id is reused while the pool lives.

        Returns:
            (device float64 buffer, start, lo, hi), the last three with one entry per row of ``row_list``.
        """
        pool = self._pool
        if pool is None or pool['lock'] is not lockstep:
            pool = self._pool = {'lock': lockstep, 'bases': {}, 'pending': [], 'size': 0, 'buf': None}
        bases = pool['bases']
        n = len(row_list)
        start = np.empty(n, dtype=np.int64)
        lo = np.empty(n, dtype=np.int64)
        hi = np.empty(n, dtype=np.int64)
        for r, (base, row, is_abs) in enumerate(row_list):
            key = (id(base.corr), is_abs)
            entry = bases.get(key)
            if entry is None:
                corr = np.asarray(base.corr, dtype=np.float64)
                nonzero = corr != 0
                found = nonzero.any(axis=1)
                width = corr.shape[1]
                a = np.where(found, nonzero.argmax(axis=1), 0)
                b = np.where(found, width - nonzero[:, ::-1].argmax(axis=1), 0)
                entry = bases[key] = (pool['size'], width, a, b, base.corr)
                pool['pending'].append(np.abs(corr).ravel() if is_abs else corr.ravel())
                pool['size'] += corr.size
            off, width, a, b, _ = entry
            lo[r] = a[row]
            hi[r] = b[row]
            start[r] = off + row * width + a[row]
        if pool['pending']:
            t0 = time.perf_counter()
            new = cp.asarray(np.concatenate(pool['pending']))
            pool['buf'] = new if pool['buf'] is None else cp.concatenate([pool['buf'], new])
            self.timing['h2d_bytes'] += new.nbytes
            self.timing['h2d_s'] += time.perf_counter() - t0
            pool['pending'] = []
        return pool['buf'], start, lo, hi

    def compute(self, lockstep, items, host=False):
        """Coarse maps of a list of requests on one or more grids (point-source grids or sky grids).

        Args:
            lockstep: The batch's ``Lockstep``.
            items: List of (job, (GridRef or SkyRef, channels, packed_list, pair_weights)).
            host: Download every map (else device maps when ``device_maps``).

        Returns:
            (maps, owners): maps a host float64 array (n_maps, n_points) per grid in a dict keyed by grid
            key, owners a list of (grid key, job, first map, number of maps).
        """
        t_prep = time.perf_counter()
        by_grid = {}
        for job, args in items:
            by_grid.setdefault(args[0].grid_key, []).append((job, args))
        results = {}
        owners = []
        prep = 0.0
        for grid_key, group in by_grid.items():
            tts_d, valid_d, index, n_points = self.device_tt(lockstep, group[0][1][0])
            t0 = time.perf_counter()
            plan = lockstep.plan(group, index)
            series_d, start, lo, hi = self.device_series(lockstep, plan['rows'])
            n_maps = len(plan['wtotal'])
            col_base = np.array([id(plan['rows'][r][0].corr) for r in plan['row']], dtype=np.int64)
            map_cols = [plan['col'][plan['ptr'][m]:plan['ptr'][m + 1]] for m in range(n_maps)]
            map_base = np.array([col_base[c[0]] if len(c) else 0 for c in map_cols], dtype=np.int64)
            prep += time.perf_counter() - t0
            t0 = time.perf_counter()
            cols_d = {
                'ch1': cp.asarray(plan['ch1'].astype(np.int32)), 'ch2': cp.asarray(plan['ch2'].astype(np.int32)),
                'last': cp.asarray(plan['lengths'] - 1), 'inv_dt': cp.asarray(plan['inv_dts']),
                'off': cp.asarray(plan['offsets']), 'start': cp.asarray(start[plan['row']]),
                'lo': cp.asarray(lo[plan['row']]), 'hi': cp.asarray(hi[plan['row']]),
            }
            self.timing['h2d_bytes'] += 60 * len(plan['row'])
            out_d = cp.empty((n_maps, n_points), dtype=cp.float64)
            self.timing['h2d_s'] += time.perf_counter() - t0
            blocks_x = (n_points + _THREADS - 1) // _THREADS
            for base in np.unique(map_base):
                t0 = time.perf_counter()
                maps = np.flatnonzero(map_base == base)
                cols = np.flatnonzero(col_base == base)
                order = np.lexsort((plan['offsets'][cols], plan['row'][cols], plan['ch2'][cols], plan['ch1'][cols]))
                cols = cols[order]
                rank = np.full(len(plan['row']), -1, dtype=np.int64)
                rank[cols] = np.arange(len(cols))
                ranked = [rank[map_cols[m]] for m in maps]
                dense = self.dense and all(len(r) == 0 or np.all(np.diff(r) > 0) for r in ranked)
                wtot = plan['wtotal'][maps]
                if dense:
                    Wm = np.zeros((len(maps), len(cols)))
                    Wa = np.zeros((len(maps), len(cols)))
                    for k, m in enumerate(maps):
                        sl = slice(plan['ptr'][m], plan['ptr'][m + 1])
                        Wm[k, ranked[k]] = plan['w'][sl]
                        Wa[k, ranked[k]] = plan['wabs'][sl]
                prep += time.perf_counter() - t0
                t0 = time.perf_counter()
                sub = {k: v[cp.asarray(cols)] for k, v in cols_d.items()}
                sub_out = cp.empty((len(maps), n_points), dtype=cp.float64)
                if dense:
                    args = (tts_d, valid_d, np.int64(n_points), sub['ch1'], sub['ch2'], sub['last'],
                            sub['inv_dt'], sub['off'], sub['start'], sub['lo'], sub['hi'], np.int32(len(cols)),
                            series_d, cp.asarray(Wm), cp.asarray(Wa), cp.asarray(wtot), np.int32(len(maps)),
                            np.int32(self.reco._valid_norm), np.float64(self.reco._valid_floor), sub_out)
                    self._dense_kernel((blocks_x, (len(maps) + _MC - 1) // _MC), (_THREADS,), args)
                else:
                    ptr = np.zeros(len(maps) + 1, dtype=np.int64)
                    ptr[1:] = np.cumsum([len(r) for r in ranked])
                    ecol = np.concatenate(ranked).astype(np.int32)
                    ew = np.concatenate([plan['w'][plan['ptr'][m]:plan['ptr'][m + 1]] for m in maps])
                    ewa = np.concatenate([plan['wabs'][plan['ptr'][m]:plan['ptr'][m + 1]] for m in maps])
                    args = (tts_d, valid_d, np.int64(n_points), sub['ch1'], sub['ch2'], sub['last'],
                            sub['inv_dt'], sub['off'], sub['start'], sub['lo'], sub['hi'], series_d,
                            cp.asarray(ptr), cp.asarray(ecol), cp.asarray(ew), cp.asarray(ewa), cp.asarray(wtot),
                            np.int32(self.reco._valid_norm), np.float64(self.reco._valid_floor), sub_out)
                    self._csr_kernel((blocks_x, len(maps)), (_THREADS,), args)
                out_d[cp.asarray(maps)] = sub_out
                cp.cuda.Device().synchronize()
                self.timing['kernel_s'] += time.perf_counter() - t0
            if self.device_maps and not host:
                results[grid_key] = [DeviceMap(self, out_d[m]) for m in range(n_maps)]
            else:
                t0 = time.perf_counter()
                host = cupyx.empty_pinned((n_maps, n_points), dtype=np.float64)
                out_d.get(out=host)
                self.timing['d2h_s'] += time.perf_counter() - t0
                self.timing['d2h_bytes'] += host.nbytes
                results[grid_key] = host
            if self.keep_device_maps:
                self.last_device_maps = out_d
            m = 0
            for job, k in plan['owners']:
                owners.append((grid_key, job, m, k))
                m += k
            self.timing['n_maps'] += n_maps
        self.timing['n_batches'] += 1
        self.timing['prep_s'] += prep + 0.0 * (time.perf_counter() - t_prep)
        return results, owners

    def device_tables(self):
        """Table stack, finiteness mask and per-channel table and position arrays on the device.

        Uploaded once, and again when the channel positions change (a position shift).
        """
        if self._tables is None or self._tables['position_key'] != self.reco._position_key:
            reco = self.reco
            channels = list(reco._table_channels)
            if len(channels) > 32:
                raise ValueError("the GPU grid kernel handles at most 32 table channels")
            g = reco._pack_singleray_tables(channels)
            t0 = time.perf_counter()
            self._tables = {
                'values': cp.asarray(g['td_values']), 'ok': cp.asarray(g['td_ok'].view(np.uint8)),
                'shape': g['td_values'].shape,
                'slot': cp.asarray(g['td_slot']), 'r_min': cp.asarray(g['td_r_min']),
                'dr_inv': cp.asarray(g['td_dr_inv']), 'nr': cp.asarray(g['td_nr']),
                'z_min': cp.asarray(g['td_z_min']), 'dz_inv': cp.asarray(g['td_dz_inv']),
                'nz': cp.asarray(g['td_nz']), 'x': cp.asarray(np.ascontiguousarray(g['ant_xy'][:, 0])),
                'y': cp.asarray(np.ascontiguousarray(g['ant_xy'][:, 1])), 'pa': (g['pa_x'], g['pa_y']),
                'index': {ch: i for i, ch in enumerate(channels)}, 'position_key': reco._position_key,
            }
            cp.cuda.Device().synchronize()
            self.grid_timing['h2d_s'] += time.perf_counter() - t0
        return self._tables

    def grid_maps(self, lockstep, items):
        """Refine and polish grid maps of every waiting request in one GPU launch.

        Each map is evaluated by its own threads (one per grid point) with the travel times looked up
        on the device tables, in the grid arithmetic of ``_singleray_grid_numba``.

        Returns:
            {job: (K, n_rho, n_phi, n_z) maps}.
        """
        t0 = time.perf_counter()
        reco = self.reco
        tab = self.device_tables()
        grids = {}
        axes_list = []
        reqs = []
        for job, (rho_vec, phi_vec_rad, z_vec, channels, packed_list, pair_weights) in items:
            axes = reco._grid_axes(rho_vec, phi_vec_rad, z_vec)
            key = tuple(a.tobytes() for a in axes)
            if key not in grids:
                grids[key] = len(axes_list)
                axes_list.append(axes)
            reqs.append((job, (grids[key], channels, packed_list, pair_weights)))
        plan = lockstep.plan(reqs, tab['index'])
        series_d, start, lo, hi = self.device_series(lockstep, plan['rows'])
        offs = [0]
        g_rho, g_phi, g_z, g_n = [], [], [], []
        for axes in axes_list:
            for vec, lst in zip(axes, (g_rho, g_phi, g_z)):
                lst.append(offs[-1])
                offs.append(offs[-1] + len(vec))
            g_n.extend(len(a) for a in axes)
        all_axes = np.concatenate([a for axes in axes_list for a in axes])
        m_grid, m_out, n_tot = [], [], 0
        shapes = []
        for job, (gid, channels, packed_list, _) in reqs:
            n = int(np.prod(g_n[3 * gid:3 * gid + 3]))
            for _ in packed_list:
                m_grid.append(gid)
                m_out.append(n_tot)
                n_tot += n
            shapes.append((job, len(packed_list), tuple(g_n[3 * gid:3 * gid + 3])))
        col = plan['col']
        row = plan['row'][col]
        self.grid_timing['prep_s'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        g_rho_d, g_phi_d, g_z_d, g_n_d, m_out_d, m_grid_d, ptr_d, last_d, start_d, lo_d, hi_d = _upload_packed(
            (g_rho, g_phi, g_z, g_n, m_out, m_grid, plan['ptr'], plan['lengths'][col] - 1, start[row], lo[row],
             hi[row]), np.int64)
        ch1_d, ch2_d = _upload_packed((plan['ch1'][col], plan['ch2'][col]), np.int32)
        axes_d, wtotal_d, inv_dt_d, off_d, w_d, wabs_d = _upload_packed(
            (all_axes, plan['wtotal'], plan['inv_dts'][col], plan['offsets'][col], plan['w'], plan['wabs']),
            np.float64)
        args = (tab['values'], tab['ok'], np.int64(tab['shape'][1]), np.int64(tab['shape'][2]),
                np.int32(reco._tolerant_table_edge), tab['slot'], tab['r_min'], tab['dr_inv'], tab['nr'],
                tab['z_min'], tab['dz_inv'], tab['nz'], tab['x'], tab['y'], np.float64(tab['pa'][0]),
                np.float64(tab['pa'][1]), axes_d, g_rho_d, g_phi_d, g_z_d, g_n_d, m_out_d, m_grid_d, ptr_d,
                wtotal_d, np.int32(len(m_out)), ch1_d, ch2_d, last_d, inv_dt_d, off_d, start_d, lo_d, hi_d, w_d,
                wabs_d, series_d, np.int64(n_tot), np.int32(reco._valid_norm), np.float64(reco._valid_floor))
        out_d = cp.empty(n_tot, dtype=cp.float64)
        self.grid_timing['h2d_s'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        self._grid_kernel(((n_tot + _THREADS - 1) // _THREADS,), (_THREADS,), args + (out_d,))
        cp.cuda.Device().synchronize()
        self.grid_timing['kernel_s'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        host = out_d.get()
        self.grid_timing['d2h_s'] += time.perf_counter() - t0
        self.grid_timing['n_rounds'] += 1
        self.grid_timing['n_maps'] += len(m_out)
        self.grid_timing['n_points'] += n_tot
        replies = {}
        pos = 0
        for job, k, shape in shapes:
            n = int(np.prod(shape))
            replies[job] = host[pos:pos + k * n].reshape((k,) + shape)
            pos += k * n
        return replies

    def batch_peaks(self, lockstep, items):
        """``_extract_top_n_peaks`` of many device maps: per round one argmax and one masking launch for all.

        The work copy holds the map with NaN as minus infinity; each round takes the first index of
        the largest value per map (the ``nanargmax`` rule), builds the separation masks on the host from
        the grid vectors exactly as the CPU code does and sets the masked cells to minus infinity on
        the device. A map whose cells are all masked stops, as the CPU loop does when all are NaN.

        Returns:
            {job: list of (rho, phi_deg, z, corr)}.
        """
        t0 = time.perf_counter()
        replies = {}
        groups = {}
        for job, args in items:
            groups.setdefault(args[0].shape, []).append((job, args))
        for shape, group in groups.items():
            nr, nphi, nz = shape
            size = nr * nphi * nz
            m = len(group)
            work = cp.empty((m, size), dtype=cp.float64)
            for r, (_, args) in enumerate(group):
                work[r].reshape(shape)[...] = args[0].view()
            work[cp.isnan(work)] = -cp.inf
            peaks = [[] for _ in group]
            active = [True] * m
            for it in range(max(args[4] for _, args in group)):
                rows = [r for r in range(m) if active[r] and it < group[r][1][4]]
                if not rows:
                    break
                rows_d = cp.asarray(np.array(rows, dtype=np.int32))
                pv = cp.empty((len(rows), _RED_BLOCKS), dtype=cp.float64)
                pi = cp.empty((len(rows), _RED_BLOCKS), dtype=cp.int64)
                self._argmax_kernel((_RED_BLOCKS, len(rows)), (_RED_THREADS,),
                                    (work, np.int64(size), rows_d, pv, pi))
                vals = cp.asnumpy(pv)
                idxs = cp.asnumpy(pi)
                best = vals.max(axis=1)
                first = np.where(vals == best[:, None], idxs, size).min(axis=1)
                hit, rm, pm, zm = [], [], [], []
                for k, r in enumerate(rows):
                    if best[k] == -np.inf:
                        active[r] = False
                        continue
                    _, rho_vec, phi_vec_deg, z_vec, _, separation = group[r][1]
                    d_rho, d_phi, d_z = separation
                    idx = np.unravel_index(int(first[k]), shape)
                    rho_peak = rho_vec[idx[0]]
                    phi_peak = phi_vec_deg[idx[1]]
                    z_peak = z_vec[idx[2]]
                    peaks[r].append((rho_peak, phi_peak, z_peak, float(best[k])))
                    rho_mask = np.abs(rho_vec - rho_peak) < d_rho
                    z_mask = np.abs(z_vec - z_peak) < d_z
                    phi_diff = np.abs(phi_vec_deg - phi_peak)
                    phi_diff = np.minimum(phi_diff, 360.0 - phi_diff)
                    phi_mask = phi_diff < d_phi
                    hit.append(r)
                    rm.append(rho_mask)
                    pm.append(phi_mask)
                    zm.append(z_mask)
                if hit:
                    blocks = ((size + _THREADS - 1) // _THREADS, len(hit))
                    self._mask_kernel(blocks, (_THREADS,), (
                        work, np.int64(size), np.int64(nr), np.int64(nphi), np.int64(nz),
                        cp.asarray(np.array(hit, dtype=np.int32)),
                        cp.asarray(np.array(rm, dtype=np.uint8)), cp.asarray(np.array(pm, dtype=np.uint8)),
                        cp.asarray(np.array(zm, dtype=np.uint8))))
            for r, (job, _) in enumerate(group):
                replies[job] = peaks[r]
            self.map_timing['n_peak_maps'] += m
        self.map_timing['n_peak_rounds'] += 1
        self.map_timing['peaks_s'] += time.perf_counter() - t0
        return replies

    def _moments(self, dmap, box):
        """Count, mean and population variance of the finite cells of a device map outside a box.

        Two passes (sum and count, then the sum of squared deviations from the mean), each a fixed
        block reduction, so the result is deterministic; it differs from numpy's pairwise sums in the
        last bits.
        """
        view = dmap.view()
        base = dmap.data
        nr, nphi, nzv = dmap.shape
        geom = (np.int64(nr), np.int64(nphi), np.int64(base.shape[2]), np.int64(dmap.z0), np.int64(nzv)) + tuple(
            np.int64(b) for b in box)
        partial = cp.empty((_RED_BLOCKS, 2), dtype=cp.float64)
        self._moments_kernel((_RED_BLOCKS,), (_RED_THREADS,), (base, *geom, np.int32(1), np.float64(0.0), partial))
        s, c = (float(v) for v in cp.asnumpy(partial.sum(axis=0)))
        if c == 0:
            return 0, np.nan, np.nan
        mean = s / c
        self._moments_kernel((_RED_BLOCKS,), (_RED_THREADS,), (base, *geom, np.int32(2), np.float64(mean), partial))
        ss = float(cp.asnumpy(partial[:, 0].sum()))
        del view
        return c, mean, ss / c

    def map_snr(self, dmap, peak_idx, exclusion_bins=3):
        """``_compute_map_snr`` of a device map: peak value over the RMS of the map outside the box."""
        t0 = time.perf_counter()
        ir, ip, iz = peak_idx
        nr, nphi, nz = dmap.shape
        box = (max(0, ir - exclusion_bins), min(nr, ir + exclusion_bins + 1),
               max(0, ip - exclusion_bins), min(nphi, ip + exclusion_bins + 1),
               max(0, iz - exclusion_bins), min(nz, iz + exclusion_bins + 1))
        c, _, var = self._moments(dmap, box)
        self.map_timing['snr_s'] += time.perf_counter() - t0
        self.map_timing['n_snr'] += 1
        if c == 0:
            return np.nan
        rms = np.sqrt(np.float64(var))
        if rms < 1e-12:
            return np.nan
        return float(dmap.view()[peak_idx]) / rms

    def map_std(self, dmap):
        """Standard deviation of the finite cells of a device map (NaN when there are none)."""
        c, _, var = self._moments(dmap, (0, 0, 0, 0, 0, 0))
        return float(np.sqrt(var)) if c else np.nan

    def masked_max(self, dmap, z_mask=None, mask=None):
        """Largest non-NaN value of ``m[:, :, z_mask]`` or ``m[mask]`` of a device map."""
        view = dmap.view()
        if z_mask is not None:
            return float(cp.nanmax(view[:, :, cp.asarray(np.flatnonzero(z_mask))]))
        key = id(mask)
        entry = self._mask_cache.get(key)
        if entry is None or entry[0] is not mask:
            entry = self._mask_cache[key] = (mask, cp.asarray(mask))
        return float(cp.nanmax(view[entry[1]]))

    def stack_maps(self, lockstep, items):
        """Coarse maps of the waiting requests of a lockstep batch; returns {job: (K, n_points) maps}."""
        results, owners = self.compute(lockstep, items)
        return {job: results[g][m:m + k] for g, job, m, k in owners}

    def batch_sky_maps(self, lockstep, items):
        """Far-field coarse sky maps of the waiting requests (host float64); returns {job: (K, n_directions)}."""
        results, owners = self.compute(lockstep, items, host=True)
        return {job: results[g][m:m + k] for g, job, m, k in owners}


_MR_SRC = r'''
#define MR_NEG_INF (-__longlong_as_double(0x7ff0000000000000LL))

__device__ __forceinline__ double mr_value(const double* __restrict__ corr, long long len, double delay,
                                           double dt, double off) {
    double kf = (delay - off) / dt;
    double fk = floor(kf);
    if (fk < 0.0 || fk >= (double)(len - 1)) return 0.0;
    long long k = (long long)fk;
    double alpha = kf - (double)k;
    return corr[k] + (corr[k + 1] - corr[k]) * alpha;
}

extern "C" __global__ void perpair_multiray(
        const double* __restrict__ tt, int n_rt, long long n_points, const unsigned char* __restrict__ rt_ok,
        const int* __restrict__ ch1, const int* __restrict__ ch2, const double* __restrict__ corr,
        long long stride, const long long* __restrict__ lengths, const double* __restrict__ dts,
        const double* __restrict__ offs, const double* __restrict__ w, int n_pairs, double w_sum,
        double* __restrict__ out) {
    long long pt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;
    double total = 0.0;
    for (int p = 0; p < n_pairs; ++p) {
        int c1 = ch1[p], c2 = ch2[p];
        double best = 0.0;
        for (int r1 = 0; r1 < n_rt; ++r1) {
            if (!rt_ok[c1 * n_rt + r1]) continue;
            double t1 = tt[((long long)c1 * n_rt + r1) * n_points + pt];
            if (!isfinite(t1)) continue;
            for (int r2 = 0; r2 < n_rt; ++r2) {
                if (!rt_ok[c2 * n_rt + r2]) continue;
                double delay = t1 - tt[((long long)c2 * n_rt + r2) * n_points + pt];
                if (!isfinite(delay)) continue;
                double v = mr_value(corr + (long long)p * stride, lengths[p], delay, dts[p], offs[p]);
                if (v > best) best = v;
            }
        }
        total += best * w[p];
    }
    if (w_sum > 0.0) total /= w_sum;
    out[pt] = total;
}

extern "C" __global__ void grouped_multiray(
        const double* __restrict__ tt, int n_rt, long long n_points, const long long* __restrict__ combos,
        int n_combos, int n_ch, const int* __restrict__ ch1, const int* __restrict__ ch2,
        const double* __restrict__ corr, long long stride, const long long* __restrict__ lengths,
        const double* __restrict__ dts, const double* __restrict__ offs, const double* __restrict__ w,
        int n_pairs, double w_sum, double* __restrict__ out) {
    long long pt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;
    double best = MR_NEG_INF;
    for (int c = 0; c < n_combos; ++c) {
        const long long* rts = combos + (long long)c * n_ch;
        double acc = 0.0;
        for (int p = 0; p < n_pairs; ++p) {
            int c1 = ch1[p], c2 = ch2[p];
            double delay = tt[((long long)c1 * n_rt + rts[c1]) * n_points + pt]
                         - tt[((long long)c2 * n_rt + rts[c2]) * n_points + pt];
            if (!isfinite(delay)) continue;
            acc += mr_value(corr + (long long)p * stride, lengths[p], delay, dts[p], offs[p]) * w[p];
        }
        if (w_sum > 0.0) acc /= w_sum;
        if (acc > best) best = acc;
    }
    out[pt] = (best == MR_NEG_INF) ? 0.0 : best;
}

#define MR_NG_MAX 10
#define MR_NB_MAX (MR_NG_MAX * (MR_NG_MAX + 1) / 2 * 9)

extern "C" __global__ void grouped_blocks(
        const double* __restrict__ tt, int n_rt, long long n_points, const int* __restrict__ ch_group, int n_g,
        const int* __restrict__ g_rts, int max_opt, const int* __restrict__ g_nrt,
        const int* __restrict__ ch1, const int* __restrict__ ch2, const double* __restrict__ corr, long long stride,
        const long long* __restrict__ lengths, const double* __restrict__ dts, const double* __restrict__ offs,
        const double* __restrict__ w, int n_pairs, double w_sum, double* __restrict__ out) {
    long long pt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;
    double blocks[MR_NB_MAX];
    int nb = n_g * (n_g + 1) / 2 * n_rt * n_rt;
    for (int i = 0; i < nb; ++i) blocks[i] = 0.0;
    for (int p = 0; p < n_pairs; ++p) {
        int c1 = ch1[p], c2 = ch2[p], g1 = ch_group[c1], g2 = ch_group[c2];
        int lo = min(g1, g2), hi = max(g1, g2);
        int tri = lo * n_g - lo * (lo - 1) / 2 + (hi - lo);
        for (int r1 = 0; r1 < n_rt; ++r1) {
            double t1 = tt[((long long)c1 * n_rt + r1) * n_points + pt];
            for (int r2 = 0; r2 < n_rt; ++r2) {
                if (g1 == g2 && r1 != r2) continue;
                double delay = t1 - tt[((long long)c2 * n_rt + r2) * n_points + pt];
                if (!isfinite(delay)) continue;
                double v = mr_value(corr + (long long)p * stride, lengths[p], delay, dts[p], offs[p]) * w[p];
                if (g1 <= g2) blocks[(tri * n_rt + r1) * n_rt + r2] += v;
                else blocks[(tri * n_rt + r2) * n_rt + r1] += v;
            }
        }
    }
    int idx[MR_NG_MAX];
    double partial[MR_NG_MAX + 1];
    for (int g = 0; g < n_g; ++g) idx[g] = -1;
    partial[0] = 0.0;
    double best = MR_NEG_INF;
    int d = 0;
    while (d >= 0) {
        idx[d] += 1;
        if (idx[d] >= g_nrt[d]) { idx[d] = -1; d -= 1; continue; }
        int r = g_rts[d * max_opt + idx[d]];
        int tri_d = d * n_g - d * (d - 1) / 2;
        double total = partial[d] + blocks[(tri_d * n_rt + r) * n_rt + r];
        for (int ga = 0; ga < d; ++ga) {
            int tri = ga * n_g - ga * (ga - 1) / 2 + (d - ga);
            total += blocks[(tri * n_rt + g_rts[ga * max_opt + idx[ga]]) * n_rt + r];
        }
        if (d == n_g - 1) { if (total > best) best = total; }
        else { partial[d + 1] = total; d += 1; }
    }
    if (best == MR_NEG_INF) { out[pt] = 0.0; return; }
    out[pt] = (w_sum > 0.0) ? best / w_sum : best;
}

extern "C" __global__ void mr_lookup(
        const unsigned long long* __restrict__ s_tab, const double* __restrict__ s_rmin, const double* __restrict__ s_drinv, const long long* __restrict__ s_nr,
        const double* __restrict__ s_zmin, const double* __restrict__ s_dzinv, const long long* __restrict__ s_nz,
        const double* __restrict__ ch_x, const double* __restrict__ ch_y, int n_ch, int n_rt,
        const double* __restrict__ rho, const double* __restrict__ cphi, const double* __restrict__ sphi,
        const double* __restrict__ zv, long long n_phi, long long n_z, long long n_points, double pa_x, double pa_y,
        double* __restrict__ tt, unsigned char* __restrict__ avail) {
    // Positions, distances and table coordinates without contraction, as numpy rounds them on the CPU
    long long pt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;
    long long iz = pt % n_z, row = pt / n_z;
    long long ip = row % n_phi, ir = row / n_phi;
    double x = __dadd_rn(__dmul_rn(rho[ir], cphi[ip]), pa_x), y = __dadd_rn(__dmul_rn(rho[ir], sphi[ip]), pa_y);
    double z = zv[iz];
    for (int c = 0; c < n_ch; ++c) {
        double dx = __dsub_rn(x, ch_x[c]), dy = __dsub_rn(y, ch_y[c]);
        double r = sqrt(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
        if (r < 1.0) r = 1.0;
        for (int q = 0; q < n_rt; ++q) {
            long long s = (long long)c * n_rt + q;
            const double* tab = (const double*)s_tab[s];
            double v;
            if (tab == 0) {
                v = __longlong_as_double(0x7ff8000000000000LL);
            } else {
                double ri = __dmul_rn(__dsub_rn(r, s_rmin[s]), s_drinv[s]);
                double zi = __dmul_rn(__dsub_rn(z, s_zmin[s]), s_dzinv[s]);
                long long i0 = (long long)floor(ri), j0 = (long long)floor(zi);
                if (i0 < 0 || i0 >= s_nr[s] - 1 || j0 < 0 || j0 >= s_nz[s] - 1) {
                    v = MR_NEG_INF;
                } else {
                    double fx = ri - (double)i0, fy = zi - (double)j0;
                    long long nz = s_nz[s], b = i0 * nz + j0;
                    v = (1.0 - fx) * (1.0 - fy) * tab[b] + fx * (1.0 - fy) * tab[b + nz]
                        + (1.0 - fx) * fy * tab[b + 1] + fx * fy * tab[b + nz + 1];
                    if (isfinite(v) && v > 0.0) avail[s] = 1;
                }
            }
            tt[s * n_points + pt] = v;
        }
    }
}
'''


class GpuMultiray:
    """Multi-ray correlation maps on a CUDA device: the per-pair map (coarse grid) and the grouped map (refine grids).

    The arithmetic of ``fast_grouped_multiray.perpair_multiray_numba`` and ``grouped_multiray_numba`` point for
    point: per pair the best ray-type combination (floored at 0) or, per depth-group combination, the weighted
    mean over the pairs with the best combination kept; a delay outside a series reads 0 and a non-finite delay
    is skipped. Those kernels compile with fastmath, so the maps agree to the rounding of either side, not bit for
    bit. The travel times of a coarse grid stay on the device while the reconstruction caches them.

    Attributes:
        timing: Seconds and counts of the per-pair and grouped calls.
    """

    def __init__(self, max_cached=6):
        """Compile the kernels.

        Args:
            max_cached: Largest number of coarse travel-time grids kept on the device.
        """
        module = cp.RawModule(code=_MR_SRC)
        self._perpair = module.get_function('perpair_multiray')
        self._grouped = module.get_function('grouped_multiray')
        self._lookup = module.get_function('mr_lookup')
        self._blocks = module.get_function('grouped_blocks')
        self._tt_cache = {}
        self._corr_cache = {}
        self._table_cache = {}
        self._table_dev = {}
        self._table_bytes = 0
        self._max_cached = max_cached
        self.timing = {'perpair_s': 0.0, 'n_perpair': 0, 'grouped_s': 0.0, 'n_grouped': 0, 'perpair_points': 0,
                       'grouped_points': 0, 'walk_nodes': 0, 'walk_leaves': 0}

    @staticmethod
    def _grid_shape(tt_all, channels):
        """Shape of the travel-time grids of a multi-ray lookup, or None without any."""
        for ch in channels:
            for grid in tt_all.get(ch, {}).values():
                return grid.shape
        return None

    def _device_tt(self, tt_all, channels, grid_shape, keep):
        """Packed (n_ch, n_rt, n_points) travel times on the device and the per-channel available ray types.

        With ``keep`` the upload is cached under the lookup dict (a reference to it is held, so its id
        stays unique) and the channels.
        """
        from fast_grouped_multiray import pack_tt_grids
        key = (id(tt_all), tuple(channels))
        entry = self._tt_cache.get(key)
        if entry is not None and entry[0] is tt_all:
            return entry[1], entry[2]
        packed, avail = pack_tt_grids(tt_all, channels, grid_shape)
        dev = cp.asarray(packed)
        if keep:
            if len(self._tt_cache) >= self._max_cached:
                self._tt_cache.pop(next(iter(self._tt_cache)))
            self._tt_cache[key] = (tt_all, dev, avail)
        return dev, avail

    def _pairs(self, corr_data, channels, pair_weights):
        """Device arrays of the pair channels, packed series, lengths, dts, offsets and weights, and w_sum.

        The packed series of a ``corr_data`` list go up once and are reused while that list is among the
        last few seen (a reference to it is held, so its id stays unique).
        """
        from fast_grouped_multiray import pack_corr_data
        n_ch = len(channels)
        entry = self._corr_cache.get(id(corr_data))
        if entry is None or entry[0] is not corr_data or entry[1] != n_ch:
            ch_pairs = list(itertools.combinations(range(n_ch), 2))
            corr, lengths, dts, offs = pack_corr_data(corr_data, len(ch_pairs))
            ch1_d, ch2_d = _upload_packed(([a for a, _ in ch_pairs], [b for _, b in ch_pairs]), np.int32)
            dts_d, offs_d = _upload_packed((dts, offs), np.float64)
            entry = (corr_data, n_ch, ch1_d, ch2_d, cp.asarray(corr), np.int64(corr.shape[1]), cp.asarray(lengths),
                     dts_d, offs_d, np.int32(len(ch_pairs)))
            if len(self._corr_cache) >= 8:
                self._corr_cache.pop(next(iter(self._corr_cache)))
            self._corr_cache[id(corr_data)] = entry
        n_pairs = int(entry[9])
        w = (np.ones(n_pairs, dtype=np.float64) if pair_weights is None
             else np.asarray(pair_weights, dtype=np.float64))
        w_sum = 0.0
        for v in w:
            w_sum += v
        return entry[2:9] + (cp.asarray(w), entry[9], np.float64(w_sum))

    def perpair(self, corr_data, tt_all, channels, pair_weights=None, keep=True):
        """``perpair_multiray_numba`` on the device; returns (map of the grid shape, max)."""
        t0 = time.perf_counter()
        grid_shape = self._grid_shape(tt_all, channels)
        if grid_shape is None:
            return np.zeros(1), np.nan
        tt_d, avail = self._device_tt(tt_all, channels, grid_shape, keep)
        ok = np.zeros((len(channels), int(tt_d.shape[1])), dtype=bool)
        for ci, rts in enumerate(avail):
            ok[ci, rts] = True
        return self._perpair_map(tt_d, ok, grid_shape, corr_data, channels, pair_weights, t0)

    def _perpair_map(self, tt_d, ok, grid_shape, corr_data, channels, pair_weights, t0):
        """Run the per-pair kernel on device travel times (n_ch, n_rt, n_points); returns (map, max)."""
        n_points = int(np.prod(grid_shape))
        n_rt = int(tt_d.shape[1])
        ch1_d, ch2_d, corr_d, stride, len_d, dts_d, offs_d, w_d, n_pairs, w_sum = self._pairs(
            corr_data, channels, pair_weights)
        out = cp.empty(n_points, dtype=cp.float64)
        self._perpair(((n_points + _THREADS - 1) // _THREADS,), (_THREADS,),
                      (tt_d, np.int32(n_rt), np.int64(n_points), cp.asarray(ok.astype(np.uint8)), ch1_d, ch2_d, corr_d,
                       stride, len_d, dts_d, offs_d, w_d, n_pairs, w_sum, out))
        result = out.get().reshape(grid_shape)
        self.timing['perpair_s'] += time.perf_counter() - t0
        self.timing['n_perpair'] += 1
        self.timing['perpair_points'] += n_points
        return result, float(np.max(result)) if result.size > 0 else np.nan

    def grouped(self, corr_data, tt_all, channels, ch_to_group, n_groups, pair_weights=None):
        """``grouped_multiray_numba`` on the device; returns (map of the grid shape, max)."""
        t0 = time.perf_counter()
        grid_shape = self._grid_shape(tt_all, channels)
        if grid_shape is None:
            return np.zeros(1), np.nan
        tt_d, avail = self._device_tt(tt_all, channels, grid_shape, False)
        return self._grouped_map(tt_d, avail, grid_shape, corr_data, channels, ch_to_group, n_groups, pair_weights,
                                 t0)

    def _device_tables(self, slot_tables):
        """Device pointers and grid parameters of the tables of the (channel, ray type) slots.

        Each table goes up once as its own array and stays while the tables held stay under 3 GB (least
        recently used dropped first); the per-slot arrays are cached under the tables (references held, so
        their ids stay unique).

        Returns:
            (slot table pointer (0 without a table), r_min, dr_inv, nr, z_min, dz_inv, nz) on the device.
        """
        tables = list({id(td): td for td in slot_tables if td is not None}.values())
        for td in tables:
            entry = self._table_dev.pop(id(td), None)
            if entry is None or entry[0] is not td:
                entry = (td, cp.asarray(np.ascontiguousarray(td.values, dtype=np.float64)))
                self._table_bytes += entry[1].nbytes
            self._table_dev[id(td)] = entry
        needed = {id(td) for td in tables}
        for key in list(self._table_dev):
            if self._table_bytes <= 3 << 30:
                break
            if key not in needed:
                self._table_bytes -= self._table_dev.pop(key)[1].nbytes
                self._table_cache = {k: v for k, v in self._table_cache.items() if key not in k}
        key = tuple(id(td) for td in slot_tables)
        entry = self._table_cache.get(key)
        if entry is not None and all(a is b for a, b in zip(entry[0], slot_tables)):
            return entry[1]
        ptr = np.array([0 if td is None else self._table_dev[id(td)][1].data.ptr for td in slot_tables],
                       dtype=np.uint64)

        def per_slot(attr):
            return [0 if td is None else getattr(td, attr) for td in slot_tables]

        r_min, dr_inv, z_min, dz_inv = _upload_packed(
            (per_slot('r_min'), per_slot('dr_inv'), per_slot('z_min'), per_slot('dz_inv')), np.float64)
        nr, nz = _upload_packed((per_slot('nr'), per_slot('nz')), np.int64)
        device = (cp.asarray(ptr), r_min, dr_inv, nr, z_min, dz_inv, nz)
        self._table_cache[key] = (list(slot_tables), device)
        return device

    def grouped_grid(self, rho_vec, phi_vec_rad, z_vec, pa_center, ant_xy, slot_tables, n_rt, corr_data, channels,
                     ch_to_group, n_groups, pair_weights=None):
        """Grouped map on a (rho, phi, z) grid with the travel times looked up on the device.

        The lookup of ``_compute_tt_multiray`` with the strict bilinear edge rule (``_bilinear_batch_numba``):
        source x, y = rho cos(phi), rho sin(phi) plus the PA centre, horizontal distance floored at 1 m, -inf
        outside a table; a ray type is available to a channel when any of its times on the grid is finite and
        positive, and the times of unavailable ray types are NaN, as ``pack_tt_grids`` leaves them. The cosines
        and sines come from numpy and the positions, distances and table coordinates round as numpy's do, so a
        query falls in the same table cell as on the CPU; the interpolated times and the map equal the CPU's
        to rounding.

        Args:
            rho_vec, phi_vec_rad, z_vec: Grid axes (m, rad, m).

            pa_center: PA centre (x, y).

            ant_xy: Per channel (x, y).

            slot_tables: Table per (channel, ray type slot), channel-major with n_rt slots per channel, None
                for a slot without one.

            n_rt: Ray type slots per channel.

            corr_data, channels, ch_to_group, n_groups, pair_weights: As for ``grouped``.

        Returns:
            (map of shape (n_rho, n_phi, n_z), max).
        """
        t0 = time.perf_counter()
        grid_shape = (len(rho_vec), len(phi_vec_rad), len(z_vec))
        found = self._lookup_grid(rho_vec, phi_vec_rad, z_vec, pa_center, ant_xy, slot_tables, n_rt)
        if found is None:
            return np.zeros(1), np.nan
        tt_d, ok = found
        avail = [list(np.flatnonzero(ok[c])) for c in range(len(channels))]
        return self._grouped_map(tt_d, avail, grid_shape, corr_data, channels, ch_to_group, n_groups, pair_weights,
                                 t0)

    def perpair_grid(self, rho_vec, phi_vec_rad, z_vec, pa_center, ant_xy, slot_tables, n_rt, corr_data, channels,
                     pair_weights=None):
        """Per-pair map on a (rho, phi, z) grid with the travel times looked up on the device.

        The lookup of ``grouped_grid``; arguments as there.

        Returns:
            (map of shape (n_rho, n_phi, n_z), max), or ((1,) zeros, NaN) without a usable time.
        """
        t0 = time.perf_counter()
        grid_shape = (len(rho_vec), len(phi_vec_rad), len(z_vec))
        found = self._lookup_grid(rho_vec, phi_vec_rad, z_vec, pa_center, ant_xy, slot_tables, n_rt)
        if found is None:
            return np.zeros(1), np.nan
        tt_d, ok = found
        return self._perpair_map(tt_d, ok, grid_shape, corr_data, channels, pair_weights, t0)

    def _lookup_grid(self, rho_vec, phi_vec_rad, z_vec, pa_center, ant_xy, slot_tables, n_rt):
        """Look a grid's travel times up on the device (see ``grouped_grid``).

        Returns:
            (travel times (n_ch, n_rt, n_points), NaN for unavailable ray types, and the (n_ch, n_rt) availability),
            or None when no time is usable.
        """
        n_points = len(rho_vec) * len(phi_vec_rad) * len(z_vec)
        n_ch = len(ant_xy)
        slot, r_min, dr_inv, nr, z_min, dz_inv, nz = self._device_tables(slot_tables)
        ch_x, ch_y = _upload_packed(([float(p[0]) for p in ant_xy], [float(p[1]) for p in ant_xy]), np.float64)
        rho_d, cphi_d, sphi_d, z_d = _upload_packed((rho_vec, np.cos(phi_vec_rad), np.sin(phi_vec_rad), z_vec),
                                                    np.float64)
        tt_d = cp.empty((n_ch, n_rt, n_points), dtype=cp.float64)
        avail_d = cp.zeros(n_ch * n_rt, dtype=cp.uint8)
        self._lookup(((n_points + _THREADS - 1) // _THREADS,), (_THREADS,),
                     (slot, r_min, dr_inv, nr, z_min, dz_inv, nz, ch_x, ch_y, np.int32(n_ch),
                      np.int32(n_rt), rho_d, cphi_d, sphi_d, z_d, np.int64(len(phi_vec_rad)), np.int64(len(z_vec)),
                      np.int64(n_points), np.float64(pa_center[0]), np.float64(pa_center[1]), tt_d, avail_d))
        ok = avail_d.get().reshape(n_ch, n_rt).astype(bool)
        if not ok.any():
            return None
        if not ok.all():
            tt_d[cp.asarray(~ok)] = np.nan
        return tt_d, ok

    def _grouped_map(self, tt_d, avail, grid_shape, corr_data, channels, ch_to_group, n_groups, pair_weights, t0):
        """Grouped map of device travel times (n_ch, n_rt, n_points); returns (map, max).

        Each group's ray types are those ``build_combo_table`` enumerates. Up to 10 groups and 3 ray types the
        ``grouped_blocks`` kernel sums the pair terms into group blocks and walks the combinations depth first;
        otherwise ``grouped_multiray`` sums every combination over the pairs. The best combination's mean is
        the same either way, to rounding.
        """
        from fast_grouped_multiray import build_combo_table
        n_points = int(np.prod(grid_shape))
        n_rt = int(tt_d.shape[1])
        ch1_d, ch2_d, corr_d, stride, len_d, dts_d, offs_d, w_d, n_pairs, w_sum = self._pairs(
            corr_data, channels, pair_weights)
        out = cp.empty(n_points, dtype=cp.float64)
        grid = ((n_points + _THREADS - 1) // _THREADS,)
        if n_groups <= 10 and n_rt <= 3:
            options = []
            for g in range(n_groups):
                members = [ci for ci, ch in enumerate(channels) if ch_to_group[ch] == g]
                rts = set(range(n_rt))
                for ci in members:
                    rts &= set(avail[ci])
                if not rts:
                    for ci in members:
                        rts |= set(avail[ci])
                options.append(sorted(rts) if rts else [0])
            max_opt = max(len(o) for o in options)
            g_rts = np.zeros((n_groups, max_opt), dtype=np.int32)
            for g, o in enumerate(options):
                g_rts[g, :len(o)] = o
            group_d, g_rts_d, g_nrt_d = _upload_packed(
                ([ch_to_group[ch] for ch in channels], g_rts.ravel(), [len(o) for o in options]), np.int32)
            n_opt = [len(o) for o in options]
            self.timing['walk_nodes'] += n_points * sum(int(np.prod(n_opt[:d + 1])) for d in range(n_groups))
            self.timing['walk_leaves'] += n_points * int(np.prod(n_opt))
            self._blocks(grid, (_THREADS,),
                         (tt_d, np.int32(n_rt), np.int64(n_points), group_d, np.int32(n_groups), g_rts_d,
                          np.int32(max_opt), g_nrt_d, ch1_d, ch2_d, corr_d, stride, len_d, dts_d, offs_d, w_d, n_pairs,
                          w_sum, out))
        else:
            combos = build_combo_table(channels, ch_to_group, n_groups, avail, n_rt=n_rt)
            self._grouped(grid, (_THREADS,),
                          (tt_d, np.int32(n_rt), np.int64(n_points), cp.asarray(np.ascontiguousarray(combos)),
                           np.int32(combos.shape[0]), np.int32(len(channels)), ch1_d, ch2_d, corr_d, stride, len_d,
                           dts_d, offs_d, w_d, n_pairs, w_sum, out))
        result = out.get().reshape(grid_shape)
        self.timing['grouped_s'] += time.perf_counter() - t0
        self.timing['n_grouped'] += 1
        self.timing['grouped_points'] += n_points
        return result, float(np.max(result)) if result.size > 0 else np.nan
