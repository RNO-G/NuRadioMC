"""Batched search stage of the 3D interferometric reconstruction: many settings of one or more events at once.

``reconstruct_batch(reco, jobs)`` returns for every job (pair set, keyword arguments of
``InterferometricReco3D.reconstruct_from_pairs``) exactly the result dict that call returns. Each job runs
the unchanged search code as its own job (a greenlet, or a thread without the greenlet package); the jobs run one
at a time (cooperative lockstep), and when every job is waiting on a coarse correlation map, the maps of all of
them are computed in one kernel call.

The objective at a point x is sum_p w_p s_p C_p(tau_p(x)) / sum_p w_p over the setting's pairs, with C_p the
pair's correlation series (raw, or its absolute value for an ``abs`` pair sign), s_p = +-1 its sign (pair
sign times channel polarities) and tau_p the pair delay. The per-pair contributions C_p(tau_p(x)) on the coarse
grid depend on the event, the series transform and the pair's lag offset (its delay shift) but not on the
channel subset, the weights or the signs, so they are evaluated once per distinct (series row, transform,
lag offset, channel pair) and every setting sums its own pairs in its own order (``_batched_stack_maps_numba``).
Each map equals the map of a separate call bit for bit; the peaks, refine levels, optimizer, polish and
hypotheses of every setting then run as in a separate call on its own complete coarse map.

With ``batch_grids`` the far-field coarse sky maps of the waiting searches are computed together the same way
(``SkyRef``: plane-wave arrival times of every table channel on the sky grid), each equal to its separate map
bit for bit on the CPU.

Jobs with different channel position shifts run in separate lockstep groups, since the positions are state of
the reconstruction object.
"""

import itertools
import threading
from collections import namedtuple

import numpy as np

from NuRadioReco.utilities.reco3d_kernels import USE_NUMBA

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import (
        _batched_grid_maps_numba, _batched_stack_maps_numba, _bilinear_ok_batch_numba)

try:
    import greenlet
except ImportError:
    greenlet = None

_state = threading.local()
_greenlet_jobs = {}
_BLOCK = 128

GridRef = namedtuple('GridRef', ['grid_key', 'channels', 'axes'])
GridRef.__doc__ = """Coarse travel-time stack of a channel group in batch mode: the grid key (cache key
without the channels), the group's channels and the grid axes (rho, phi_rad, z)."""

SkyRef = namedtuple('SkyRef', ['grid_key', 'zen_deg', 'az_deg'])
SkyRef.__doc__ = """Sky grid of the far-field coarse maps in batch mode: the cache key and the zenith and
azimuth vectors (deg) of the plane-wave directions."""


def current_executor():
    """Executor of the batch the calling job belongs to, or None outside a batch."""
    if greenlet is not None:
        job = _greenlet_jobs.get(greenlet.getcurrent())
        if job is not None:
            return job[0]
    return getattr(_state, 'executor', None)


class Lockstep:
    """Run jobs one at a time and serve their map requests in batches.

    A job runs until it requests a map (``request``) or finishes; then the next job runs. When every job
    is waiting or done, all pending requests are computed together and the waiting jobs resume in job
    order. Only one job runs at any moment, so the reconstruction object is never used concurrently and
    the order of all work is deterministic. Jobs are greenlets on the calling thread when the ``greenlet``
    package is available (a switch costs about a microsecond and every compiled kernel is launched from
    the one thread), threads handing over through one semaphore each otherwise.
    """

    def __init__(self, reco, coarse_backend=None, batch_grids=False):
        """Bind the executor to a reconstruction object.

        Args:
            reco: InterferometricReco3D after ``begin``.

            coarse_backend: Optional object with ``stack_maps(lockstep, items)`` computing the coarse maps
                (for example on a GPU); None uses the CPU kernel.

            batch_grids: Also batch the refine and polish grid maps (``_singleray_grid_maps``): the
                settings waiting on the same grid share its per-pair contributions.
        """
        self.reco = reco
        self.backend = coarse_backend
        self.batch_grids = batch_grids
        self._series_cache = {}
        self._requests = {}
        self._replies = {}
        self._provenance = {}
        self._main = None
        self._sems = None
        self._main_sem = None
        self.timing = {'coarse_batch_s': 0.0, 'n_coarse_batches': 0, 'n_coarse_maps': 0,
                       'grid_batch_s': 0.0, 'n_grid_rounds': 0, 'n_grid_requests': 0, 'n_grid_unique': 0}

    def register_series(self, packed, base, rows, signs):
        """Record where the rows of a group's packed series come from.

        Args:
            packed: CorrPacked the search will read.
            base: CorrPacked of the pair set the rows were taken from.
            rows: (n_pairs,) row of ``base`` for each row of ``packed``.
            signs: Per-row sign (1, -1 or 'abs') applied to the raw series, or None.
        """
        self._provenance[id(packed.corr)] = (packed, base, np.asarray(rows, dtype=np.int64), signs)

    def request(self, kind, args):
        """Hand a request to the scheduler and wait for its result (called from a job)."""
        if self._main is not None:
            reply = self._main.switch((kind, args))
        else:
            i = _state.index
            self._requests[i] = (kind, args)
            self._main_sem.release()
            self._sems[i].acquire()
            reply = self._replies.pop(i)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def run_jobs(self, fns):
        """Run zero-argument callables in lockstep and return their results in order.

        Raises:
            The first exception any job raised (after every job has stopped).
        """
        n = len(fns)
        results = [None] * n
        errors = [None] * n
        try:
            if greenlet is not None:
                self._run_greenlets(fns, results, errors)
            else:
                self._run_threads(fns, results, errors)
        finally:
            self._provenance.clear()
            self._series_cache.clear()
        for exc in errors:
            if exc is not None:
                raise exc
        return results

    def _run_greenlets(self, fns, results, errors):
        """Scheduler of the greenlet jobs."""
        self._main = greenlet.getcurrent()

        def body(i, fn):
            """Run one job and record its result or exception."""
            try:
                results[i] = fn()
            except BaseException as exc:  # handed to the caller
                errors[i] = exc
            return None

        lets = [greenlet.greenlet(lambda i=i, fn=fn: body(i, fn), parent=self._main) for i, fn in enumerate(fns)]
        for i, g in enumerate(lets):
            _greenlet_jobs[g] = (self, i)
        try:
            replies = {}
            runnable = list(range(len(fns)))
            while True:
                for i in runnable:
                    req = lets[i].switch(replies.pop(i)) if i in replies else lets[i].switch()
                    if req is not None:
                        self._requests[i] = req
                if not self._requests:
                    break
                pending = self._requests
                self._requests = {}
                replies = self._serve(pending)
                runnable = sorted(pending)
        finally:
            for g in lets:
                _greenlet_jobs.pop(g, None)
            self._main = None

    def _run_threads(self, fns, results, errors):
        """Scheduler of the thread jobs (without greenlet): one semaphore per job and one for the scheduler."""
        n = len(fns)
        self._sems = [threading.Semaphore(0) for _ in range(n)]
        self._main_sem = threading.Semaphore(0)

        def worker(i, fn):
            """Thread body: wait for the first turn, run the job, record its result or exception."""
            _state.executor = self
            _state.index = i
            self._sems[i].acquire()
            try:
                results[i] = fn()
            except BaseException as exc:  # handed to the caller
                errors[i] = exc
            finally:
                _state.executor = None
                self._main_sem.release()

        threads = [threading.Thread(target=worker, args=(i, fn), daemon=True) for i, fn in enumerate(fns)]
        for t in threads:
            t.start()
        runnable = list(range(n))
        while True:
            for i in runnable:
                self._sems[i].release()
                self._main_sem.acquire()
            if not self._requests:
                break
            pending = self._requests
            self._requests = {}
            self._replies.update(self._serve(pending))
            runnable = sorted(pending)
        for t in threads:
            t.join()

    def _serve(self, pending):
        """Compute every pending request; returns {job index: result or exception}."""
        import time
        replies = {}
        stack_maps = [(i, args) for i, (kind, args) in pending.items() if kind == 'stack_maps']
        grid_maps = [(i, args) for i, (kind, args) in pending.items() if kind == 'grid_maps']
        if stack_maps:
            t0 = time.time()
            try:
                if self.backend is not None:
                    replies.update(self.backend.stack_maps(self, stack_maps))
                else:
                    replies.update(self.stack_maps_cpu(stack_maps))
            except Exception as exc:
                for i, _ in stack_maps:
                    replies[i] = exc
            self.timing['coarse_batch_s'] += time.time() - t0
            self.timing['n_coarse_batches'] += 1
            self.timing['n_coarse_maps'] += sum(len(a[2]) for _, a in stack_maps)
        if grid_maps:
            t0 = time.time()
            try:
                if self.backend is not None and hasattr(self.backend, 'grid_maps'):
                    replies.update(self.backend.grid_maps(self, grid_maps))
                else:
                    replies.update(self.grid_maps_cpu(grid_maps))
            except Exception as exc:
                for i, _ in grid_maps:
                    replies[i] = exc
            self.timing['grid_batch_s'] += time.time() - t0
            self.timing['n_grid_rounds'] += 1
            self.timing['n_grid_requests'] += len(grid_maps)
        other = {}
        for i, (kind, args) in pending.items():
            if kind not in ('stack_maps', 'grid_maps'):
                other.setdefault(kind, []).append((i, args))
        for kind, group in other.items():
            backend_serve = getattr(self.backend, 'batch_' + kind, None)
            cpu_serve = getattr(self, kind + '_cpu', None)
            if backend_serve is None and cpu_serve is None:
                for i, _ in group:
                    replies[i] = ValueError(f"unknown batch request {kind}")
                continue
            try:
                replies.update(backend_serve(self, group) if backend_serve is not None else cpu_serve(group))
            except Exception as exc:
                for i, _ in group:
                    replies[i] = exc
        return replies

    def grid_tt(self, ref):
        """Channel-major travel times and validity of every table channel on a coarse grid (cached).

        The values are those of ``_singleray_tt_stack``: the same query coordinates per channel and the
        same masked bilinear lookup, each channel independent of the others.

        Returns:
            (tts, valid, index): (n_ch, n_points) arrays and dict channel -> row.
        """
        if isinstance(ref, SkyRef):
            return self.sky_tt(ref)
        reco = self.reco
        cache = reco._batch_grid_cache
        entry = cache.get(ref.grid_key)
        if entry is not None:
            return entry
        channels = list(reco._table_channels)
        g = reco._pack_singleray_tables(channels)
        src_enu = reco._build_source_enu_matrix(*ref.axes)
        n_points = int(np.prod(src_enu.shape[:3]))
        reco.work['tt_lookups'] += n_points * len(channels)
        tts = np.empty((len(channels), n_points), dtype=np.float64)
        valid = np.empty((len(channels), n_points), dtype=np.bool_)
        for ci, ch in enumerate(channels):
            coords = reco._compute_rho_and_coords(src_enu, [ch])
            tts[ci], valid[ci] = _bilinear_ok_batch_numba(
                g['td_values'], g['td_ok'], g['td_slot'][ci], g['td_r_min'][ci],
                g['td_dr_inv'][ci], g['td_nr'][ci], g['td_z_min'][ci],
                g['td_dz_inv'][ci], g['td_nz'][ci], coords[ch][:, 0], coords[ch][:, 1],
                reco._tolerant_table_edge)
            del coords
        entry = (tts, valid, {ch: ci for ci, ch in enumerate(channels)})
        cache[ref.grid_key] = entry
        return entry

    def sky_tt(self, ref):
        """Channel-major plane-wave arrival times of every table channel on a sky grid (cached).

        The values are those the far-field maps of ``_far_field_hypothesis`` use: ``plane_wave_times``
        of each channel, which does not depend on the other channels; every direction is valid. The
        cache is cleared with the other batch grids when the channel positions change.

        Returns:
            (tts, valid, index) as ``grid_tt`` returns them.
        """
        from NuRadioReco.modules.interferometricDirectionReconstruction3D import plane_wave_times
        reco = self.reco
        cache = reco._batch_grid_cache
        entry = cache.get(ref.grid_key)
        if entry is None:
            channels = list(reco._table_channels)
            ant_xyz, ice = reco._far_field_geometry(channels)
            zz, aa = np.meshgrid(np.radians(ref.zen_deg), np.radians(ref.az_deg), indexing='ij')
            tts = np.ascontiguousarray(plane_wave_times(zz, aa, ant_xyz, ice).reshape(-1, len(channels)).T)
            entry = cache[ref.grid_key] = (tts, np.ones(tts.shape, dtype=np.bool_),
                                           {ch: ci for ci, ch in enumerate(channels)})
        return entry

    def sky_maps_cpu(self, items):
        """Far-field coarse sky maps of every request with the batched CPU kernel (as ``stack_maps_cpu``).

        Returns:
            {job: (K, n_directions) maps}.
        """
        return self.stack_maps_cpu(items)

    def plan(self, items, index):
        """Columns, series rows and per-map entries of a set of coarse-map requests on one grid.

        Args:
            items: List of (job index, (GridRef, channels, packed_list, pair_weights)).
            index: Dict channel -> row of the grid travel times.

        Returns:
            Dict with the column arrays (ch1, ch2, row, lengths, inv_dts, offsets), the series rows
            (list of (base CorrPacked, row, abs flag)), the CSR map arrays (ptr, col, w, wabs, wtotal) and
            ``owners`` (job index and number of maps per request, in order).
        """
        cols = {}
        col_list = []
        rows = {}
        row_list = []
        ptr = [0]
        m_col, m_w, m_wabs, m_wtot = [], [], [], []
        owners = []
        for job, (ref, channels, packed_list, pair_weights) in items:
            n_pairs = len(channels) * (len(channels) - 1) // 2
            w = (np.ones(n_pairs, dtype=np.float64) if pair_weights is None
                 else np.asarray(pair_weights, dtype=np.float64))
            w_total = float(w.sum())
            idx = list(itertools.combinations(range(len(channels)), 2))
            for packed in packed_list:
                prov = self._provenance.get(id(packed.corr))
                if prov is None:
                    raise RuntimeError("batched coarse map: series of unknown origin")
                _, base, base_rows, signs = prov
                for p, (a, b) in enumerate(idx):
                    sign = 1 if signs is None else signs[p]
                    is_abs = isinstance(sign, str)
                    s = -1.0 if (not is_abs and sign == -1) else 1.0
                    rkey = (id(base.corr), int(base_rows[p]), is_abs)
                    r = rows.get(rkey)
                    if r is None:
                        r = rows[rkey] = len(row_list)
                        row_list.append((base, int(base_rows[p]), is_abs))
                    ckey = (r, index[channels[a]], index[channels[b]], float(packed.offsets[p]),
                            int(packed.lengths[p]), float(packed.inv_dts[p]))
                    c = cols.get(ckey)
                    if c is None:
                        c = cols[ckey] = len(col_list)
                        col_list.append(ckey)
                    m_col.append(c)
                    m_w.append(w[p] * s)
                    m_wabs.append(w[p])
                ptr.append(len(m_col))
                m_wtot.append(w_total)
            owners.append((job, len(packed_list)))
        col_arr = list(zip(*col_list))
        return {
            'ch1': np.array(col_arr[1], dtype=np.int64), 'ch2': np.array(col_arr[2], dtype=np.int64),
            'row': np.array(col_arr[0], dtype=np.int64), 'lengths': np.array(col_arr[4], dtype=np.int64),
            'offsets': np.array(col_arr[3], dtype=np.float64), 'inv_dts': np.array(col_arr[5], dtype=np.float64),
            'rows': row_list, 'ptr': np.array(ptr, dtype=np.int64), 'col': np.array(m_col, dtype=np.int64),
            'w': np.array(m_w, dtype=np.float64), 'wabs': np.array(m_wabs, dtype=np.float64),
            'wtotal': np.array(m_wtot, dtype=np.float64), 'owners': owners,
        }

    def cached_series_rows(self, row_list):
        """``series_rows`` of a row list, reused while the same rows are requested again (grid rounds)."""
        key = tuple((id(base.corr), row, is_abs) for base, row, is_abs in row_list)
        series = self._series_cache.get(key)
        if series is None:
            if len(self._series_cache) > 16:
                self._series_cache.clear()
            series = self._series_cache[key] = self.series_rows(row_list)
        return series

    def grid_maps_cpu(self, items):
        """Refine or polish grid maps of every request, one kernel call per distinct grid.

        Returns:
            {job: (K, n_rho, n_phi, n_z) maps}, as ``_singleray_grid_maps`` returns them.
        """
        reco = self.reco
        replies = {}
        groups = {}
        for job, (rho_vec, phi_vec_rad, z_vec, channels, packed_list, pair_weights) in items:
            axes = reco._grid_axes(rho_vec, phi_vec_rad, z_vec)
            key = tuple(a.tobytes() for a in axes)
            groups.setdefault(key, (axes, []))[1].append((job, (None, channels, packed_list, pair_weights)))
        self.timing['n_grid_unique'] += len(groups)
        order = {ch: i for i, ch in enumerate(reco._table_channels)}
        for axes, group in groups.values():
            chans = sorted({ch for _, a in group for ch in a[1]}, key=order.__getitem__)
            index = {ch: i for i, ch in enumerate(chans)}
            plan = self.plan(group, index)
            g = reco._pack_singleray_tables(chans)
            out = _batched_grid_maps_numba(
                *axes, g['pa_x'], g['pa_y'], g['ant_xy'], g['td_values'], g['td_ok'], g['td_slot'],
                g['td_r_min'], g['td_dr_inv'], g['td_nr'], g['td_z_min'], g['td_dz_inv'], g['td_nz'],
                reco._tolerant_table_edge, plan['ch1'], plan['ch2'], plan['row'], plan['lengths'],
                plan['inv_dts'], plan['offsets'], self.cached_series_rows(plan['rows']), plan['ptr'],
                plan['col'], plan['w'], plan['wabs'], plan['wtotal'], reco._valid_norm, reco._valid_floor)
            shape = tuple(len(a) for a in axes)
            m = 0
            for job, k in plan['owners']:
                replies[job] = out[m:m + k].reshape((k,) + shape)
                m += k
        return replies

    @staticmethod
    def series_rows(row_list):
        """Stack the series rows a plan reads (absolute value for an ``abs`` row)."""
        width = max(base.corr.shape[1] for base, _, _ in row_list)
        series = np.zeros((len(row_list), width), dtype=np.float64)
        for r, (base, row, is_abs) in enumerate(row_list):
            src = base.corr[row]
            series[r, :src.shape[0]] = np.abs(src) if is_abs else src
        return series

    def stack_maps_cpu(self, items):
        """Coarse maps of every request with the batched CPU kernel; returns {job: (K, n_points) maps}."""
        replies = {}
        by_grid = {}
        for job, args in items:
            by_grid.setdefault(args[0].grid_key, []).append((job, args))
        for group in by_grid.values():
            tts, valid, index = self.grid_tt(group[0][1][0])
            plan = self.plan(group, index)
            out = _batched_stack_maps_numba(
                tts, valid, plan['ch1'], plan['ch2'], plan['row'], plan['lengths'], plan['inv_dts'],
                plan['offsets'], self.series_rows(plan['rows']), plan['ptr'], plan['col'], plan['w'],
                plan['wabs'], plan['wtotal'], self.reco._valid_norm, self.reco._valid_floor, _BLOCK)
            m = 0
            for job, k in plan['owners']:
                replies[job] = out[m:m + k]
                m += k
        return replies


def backend_info(coarse_backend=None):
    """Backend of a batch's coarse maps, for the results: code (0 CPU, 1 GPU), name and device type.

    Returns:
        Dict with ``reco_backend`` (0 or 1, the per-row column ``reconstruct_batch`` adds),
        ``reco_backend_name`` ('cpu' or 'gpu') and ``reco_device`` (GPU name or CPU model), for
        results-file attributes.
    """
    if coarse_backend is not None:
        return {'reco_backend': 1, 'reco_backend_name': 'gpu',
                'reco_device': getattr(coarse_backend, 'device_name', 'gpu')}
    model = ''
    try:
        with open('/proc/cpuinfo') as f:
            model = next((line.split(':', 1)[1].strip() for line in f if line.startswith('model name')), '')
    except OSError:
        pass
    import platform
    return {'reco_backend': 0, 'reco_backend_name': 'cpu', 'reco_device': model or platform.processor() or 'cpu'}


def reconstruct_batch(reco, jobs, coarse_backend=None, stats=None, batch_grids=False):
    """Run many searches on stored pairs, batching their coarse maps; equals separate calls.

    Args:
        reco: InterferometricReco3D after ``begin`` (tables of every searched channel loaded).

        jobs: List of (PairSet, kwargs) with kwargs the keyword arguments of ``reconstruct_from_pairs``
            (config, channel_mask, pair_weights, channel_delay_shift, channel_polarity,
            channel_position_shift).

        coarse_backend: Optional coarse-map backend (see ``Lockstep``); None uses the CPU kernel.

        stats: Optional dict that receives the summed batch timing of the lockstep groups.

        batch_grids: Also batch the refine and polish grid maps (see ``Lockstep``).

    Returns:
        List of result dicts, one per job, each equal to ``reco.reconstruct_from_pairs(pairs, **kwargs)``
        plus the column ``reco_backend`` (0: CPU maps, equal bit for bit; 1: GPU maps, equal to the
        last bits of the map values, see ``backend_info``).

    Raises:
        RuntimeError: Without numba or with a configuration whose coarse maps the fused singleray
            kernel does not compute (multi-ray tables, the GPU path or the unfused correlator).
    """
    if not (USE_NUMBA and reco._use_fused_correlator and not reco._use_gpu and not reco._multi_ray_types):
        raise RuntimeError("reconstruct_batch needs the fused singleray numba kernels")
    groups = {}
    for j, (pairs, kwargs) in enumerate(jobs):
        config = kwargs.get('config')
        shift = kwargs.get('channel_position_shift')
        if shift is None and isinstance(config, dict):
            shift = config.get('channel_position_shift')
        groups.setdefault(reco._position_shift_key(shift), []).append(j)
    results = [None] * len(jobs)
    for idx in groups.values():
        lock = Lockstep(reco, coarse_backend, batch_grids)
        out = lock.run_jobs([(lambda p=jobs[j][0], kw=jobs[j][1]: reco.reconstruct_from_pairs(p, **kw))
                             for j in idx])
        for j, r in zip(idx, out):
            results[j] = r
        if stats is not None:
            for key, val in lock.timing.items():
                stats[key] = stats.get(key, 0) + val
    code = 1 if coarse_backend is not None else 0
    for r in results:
        r['reco_backend'] = code
    return results
