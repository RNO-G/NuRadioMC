"""Batched search over many settings (reconstruct_from_pairs_batch) against separate reconstruct_from_pairs calls.

The batch shares the per-pair contributions of the coarse map (and, with ``batch_grids``, of identical
refine and polish grids) between settings and sums them per setting in its own pair order, so every
result must equal the separate call bit for bit. Checked on the kernels directly (random subsets,
weights, signs, delay shifts and unreadable channels against the separate kernels) and end to end on
synthetic events for the record and recommended configurations, polarization groups, the HPol sign
search and region hypotheses, with masks, uniform weights, polarities, delay shifts and position shifts
in one batch. The cooperative scheduler is checked for order, results and exceptions with and without
greenlets.
"""

import itertools

import numpy as np
import pytest

from conftest import STATION, reference_config
from synthetic import HPOL_CHANNELS, VPOL_CHANNELS, TravelTimeTables, cylindrical_to_enu, make_event, make_noise_event
from NuRadioReco.modules import reco3d_batch
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

CHANNELS = VPOL_CHANNELS + HPOL_CHANNELS
GROUPS = {'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS}
CANDIDATE_KEYS = dict(candidate_search=['envelope:traces', 'envelope:correlation', 'raw'],
                      candidate_tie_band=0.0054, candidate_tie_band_max_raw_corr=0.035,
                      candidate_fill_saved_peaks=True, candidate_diagnostics=True)


def bit_equal(a, b):
    """Bitwise equality of two result values (float bits, NaN payloads and signed zeros included)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(bit_equal(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in 'fc':
        return a.tobytes() == b.tobytes()
    return bool(np.array_equal(a, b))


def assert_bit_equal(a, b, context):
    """Assert two result dicts have the same keys and bit-equal values.

    Wall-clock times are excepted, and the batch's ``reco_backend`` column (0 for CPU maps) is
    checked and then ignored.
    """
    for r in (a, b):
        assert r.get('reco_backend', 0) == 0, (context, r.get('reco_backend'))
    keys = {k for k in a if '_time' not in k and k != 'reco_backend'}
    assert keys == {k for k in b if '_time' not in k and k != 'reco_backend'}, (context, keys ^ set(b))
    bad = [k for k in sorted(keys) if not bit_equal(a[k], b[k])]
    assert not bad, (context, bad[:10])


class _Echo:
    """Coarse-map backend that answers every request with the job's request count."""

    def __init__(self):
        """Start with no requests."""
        self.rounds = []

    def stack_maps(self, lock, items):
        """Record the jobs of one round and answer each with its own args."""
        self.rounds.append([job for job, _ in items])
        return {job: args[:2] for job, args in items}


@pytest.mark.parametrize('use_greenlet', [True, False])
def test_scheduler_rounds_results_and_exceptions(monkeypatch, use_greenlet):
    """Jobs run in order, wait for one round per request, get their own replies; exceptions reach the caller."""
    if not use_greenlet:
        monkeypatch.setattr(reco3d_batch, 'greenlet', None)
    elif reco3d_batch.greenlet is None:
        pytest.skip('greenlet not installed')
    echo = _Echo()
    lock = reco3d_batch.Lockstep(None, echo)

    def job(i, n):
        """Make n requests and return the replies."""
        assert reco3d_batch.current_executor() is lock
        return [lock.request('stack_maps', (i, r, [None])) for r in range(n)]

    out = lock.run([lambda i=i: job(i, i % 3) for i in range(7)])
    assert out == [[(i, r) for r in range(i % 3)] for i in range(7)]
    assert echo.rounds == [[1, 2, 4, 5], [2, 5]]
    assert reco3d_batch.current_executor() is None

    lock = reco3d_batch.Lockstep(None, _Echo())

    def bad():
        """Fail after one request."""
        lock.request('stack_maps', (0, 0, [None]))
        raise KeyError('job failed')

    with pytest.raises(KeyError, match='job failed'):
        lock.run([lambda: job(0, 2), bad, lambda: job(2, 1)])


def _random_case(rng, n_ch=9, n_settings=12, m=3000):
    """Random pair series, weights, signs, shifts and channel subsets for the kernel checks."""
    pairs = list(itertools.combinations(range(n_ch), 2))
    base = rng.normal(size=(len(pairs), m))
    lengths = rng.integers(m - 100, m, len(pairs)).astype(np.int64)
    dts = rng.uniform(0.08, 0.12, len(pairs))
    offs = rng.uniform(-300.0, -200.0, len(pairs))
    settings = []
    for s in range(n_settings):
        keep = sorted(rng.choice(n_ch, size=int(rng.integers(3, n_ch + 1)), replace=False).tolist())
        sub = [p for p in pairs if p[0] in keep and p[1] in keep]
        settings.append(dict(keep=keep, sub=sub, rows=[pairs.index(p) for p in sub],
                             w=rng.uniform(0.1, 3.0, len(sub)), sign=rng.choice([1, -1], len(sub)),
                             shift=rng.uniform(-3.0, 3.0, len(sub)) * (s % 2)))
    return pairs, base, lengths, dts, offs, settings


def _plan(settings, offs, lengths, dts):
    """Columns and CSR map entries of the random settings (the layout reco3d_batch builds)."""
    cols, ch1, ch2, row, clen, cinv, coff = {}, [], [], [], [], [], []
    ptr, mcol, mw, mwa, wt = [0], [], [], [], []
    for st in settings:
        for (a, b), r, w, s, sh in zip(st['sub'], st['rows'], st['w'], st['sign'], st['shift']):
            key = (r, float(offs[r] + sh))
            if key not in cols:
                cols[key] = len(ch1)
                ch1.append(a)
                ch2.append(b)
                row.append(r)
                clen.append(lengths[r])
                cinv.append(1.0 / dts[r])
                coff.append(offs[r] + sh)
            mcol.append(cols[key])
            mw.append(w * s)
            mwa.append(w)
        ptr.append(len(mcol))
        wt.append(float(st['w'].sum()))
    return (np.array(ch1), np.array(ch2), np.array(row), np.array(clen, dtype=np.int64), np.array(cinv),
            np.array(coff)), (np.array(ptr), np.array(mcol), np.array(mw), np.array(mwa), np.array(wt))


def _separate_args(st, base, lengths, dts, offs, valid_norm):
    """Correlation arguments of the separate kernels for one random setting, its channels numbered from 0."""
    loc = {c: i for i, c in enumerate(st['keep'])}
    rows = st['rows']
    return ((base[rows] * st['sign'][:, None])[None], lengths[rows], 1.0 / dts[rows], offs[rows] + st['shift'],
            np.array([loc[a] for a, _ in st['sub']]), np.array([loc[b] for _, b in st['sub']]), st['w'],
            float(st['w'].sum()), valid_norm, 0.6)


def _grid_case(rng, n_ch=9, nr=300, nz=400):
    """Travel-time tables (increasing in r, finite) and per-channel geometry for the grid kernel checks."""
    values = np.cumsum(rng.uniform(0.5, 1.5, (n_ch, nr, nz)), axis=1) + 100.0
    geom = dict(slot=np.arange(n_ch, dtype=np.int64), r_min=np.zeros(n_ch), dr_inv=np.full(n_ch, 0.5),
                nr=np.full(n_ch, nr, dtype=np.int64), z_min=np.full(n_ch, -200.0), dz_inv=np.full(n_ch, 1.0),
                nz=np.full(n_ch, nz, dtype=np.int64), xy=rng.uniform(-20.0, 20.0, (n_ch, 2)))
    return values, geom


def _grid_tables(values, g, keep=slice(None)):
    """Table arguments of the grid kernels (absolute xy through tables), for the channels ``keep``."""
    return (np.ascontiguousarray(g['xy'][keep]), values, np.isfinite(values), g['slot'][keep], g['r_min'][keep],
            g['dr_inv'][keep], g['nr'][keep], g['z_min'][keep], g['dz_inv'][keep], g['nz'][keep], False)


def _unit_maps(base, maps):
    """Series of ones and unsigned weights: a map then reads the contributing weight over the total (the coverage)."""
    ptr, col, _, wabs, wtotal = maps
    return np.ones_like(base), (ptr, col, wabs, wabs, wtotal)


@pytest.mark.slow
@pytest.mark.parametrize('valid_norm', [False, True])
def test_batched_stack_kernel_equals_separate_kernel(valid_norm):
    """Every map of the batched coarse kernel equals the separate stack kernel of its setting bit for bit."""
    from NuRadioReco.utilities.reco3d_kernels import _batched_stack_maps_numba, _singleray_stack_corr_numba
    rng = np.random.default_rng(7)
    pairs, base, lengths, dts, offs, settings = _random_case(rng)
    n_points = 20000
    tts = rng.uniform(100.0, 400.0, (n_points, 9))
    valid = rng.uniform(size=(n_points, 9)) > 0.05
    cols, maps = _plan(settings, offs, lengths, dts)
    out = _batched_stack_maps_numba(np.ascontiguousarray(tts.T), np.ascontiguousarray(valid.T), *cols, base,
                                    *maps, valid_norm, 0.6, 128)
    for m, st in enumerate(settings):
        ref = _singleray_stack_corr_numba(
            np.ascontiguousarray(tts[:, st['keep']]), np.ascontiguousarray(valid[:, st['keep']]),
            *_separate_args(st, base, lengths, dts, offs, valid_norm))[0]
        assert out[m].tobytes() == ref.tobytes(), m


@pytest.mark.slow
def test_batched_grid_kernel_equals_separate_kernel():
    """Every map of the batched grid kernel equals the separate grid kernel of its setting bit for bit."""
    from NuRadioReco.utilities.reco3d_kernels import _batched_grid_maps_numba, _singleray_grid_numba
    rng = np.random.default_rng(3)
    pairs, base, lengths, dts, offs, settings = _random_case(rng, n_settings=8)
    values, g = _grid_case(rng)
    values[2, :, 300:] = np.nan
    rho = np.arange(40.0, 71.0, 1.0)
    phi = np.radians(np.arange(30.0, 36.3, 0.3))
    z = np.arange(-60.0, 120.0, 3.0)
    cols, maps = _plan(settings, offs, lengths, dts)
    out = _batched_grid_maps_numba(rho, phi, z, 1.0, 2.0, *_grid_tables(values, g), *cols, base, *maps, False, 0.6)
    for m, st in enumerate(settings):
        ref = _singleray_grid_numba(rho, phi, z, 1.0, 2.0, *_grid_tables(values, g, st['keep']),
                                    *_separate_args(st, base, lengths, dts, offs, False))[0]
        assert out[m].tobytes() == ref.tobytes(), m


@pytest.mark.slow
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_batched_maps_equal_separate_in_the_coverage_ramp(seed):
    """Under the valid-weight normalisation the batched kernels equal the separate ones bit for bit on the ramp.

    Random pair weights, signs and delay shifts as above. In the stack check every point drops each
    channel with its own probability (0 to 0.6); in the grid check each of seven channels has its tables
    unreadable over its own 60 m band in z, so one to three channels are missing at each depth. The
    contributing weight of a setting then falls anywhere from none to all of its total, and the coverage
    read back through unit series must put at least a tenth of the map values on the ramp (between half
    the floor and the floor), where the factor is evaluated, besides the flat and zero regions.
    """
    from NuRadioReco.utilities.reco3d_kernels import (
        _batched_grid_maps_numba, _batched_stack_maps_numba, _singleray_grid_numba, _singleray_stack_corr_numba,
        _singleray_stackT_corr_numba)
    rng = np.random.default_rng(seed)
    pairs, base, lengths, dts, offs, settings = _random_case(rng)
    n_points = 20000
    tts = rng.uniform(100.0, 400.0, (n_points, 9))
    valid = rng.uniform(size=(n_points, 9)) > rng.uniform(0.0, 0.6, (n_points, 1))
    ttsT, validT = np.ascontiguousarray(tts.T), np.ascontiguousarray(valid.T)
    cols, maps = _plan(settings, offs, lengths, dts)
    out = _batched_stack_maps_numba(ttsT, validT, *cols, base, *maps, True, 0.6, 128)
    ones, unit = _unit_maps(base, maps)
    cov = _batched_stack_maps_numba(ttsT, validT, *cols, ones, *unit, False, 0.6, 128)
    assert np.mean((cov > 0.3) & (cov < 0.6)) > 0.1
    for m, st in enumerate(settings):
        args = _separate_args(st, base, lengths, dts, offs, True)
        sub_t = np.ascontiguousarray(tts[:, st['keep']])
        sub_v = np.ascontiguousarray(valid[:, st['keep']])
        ref = _singleray_stack_corr_numba(sub_t, sub_v, *args)[0]
        ref_t = _singleray_stackT_corr_numba(np.ascontiguousarray(sub_t.T), np.ascontiguousarray(sub_v.T), 128,
                                             *args)[0]
        assert out[m].tobytes() == ref.tobytes(), ('stack', m)
        assert out[m].tobytes() == ref_t.tobytes(), ('stackT', m)

    pairs, base, lengths, dts, offs, settings = _random_case(rng, n_settings=8)
    values, g = _grid_case(rng)
    for c in range(7):
        values[c, :, 140 + 20 * c:200 + 20 * c] = np.nan
    rho = np.arange(40.0, 71.0, 1.0)
    phi = np.radians(np.arange(30.0, 36.3, 0.3))
    z = np.arange(-60.0, 120.0, 3.0)
    cols, maps = _plan(settings, offs, lengths, dts)
    out = _batched_grid_maps_numba(rho, phi, z, 1.0, 2.0, *_grid_tables(values, g), *cols, base, *maps, True, 0.6)
    ones, unit = _unit_maps(base, maps)
    cov = _batched_grid_maps_numba(rho, phi, z, 1.0, 2.0, *_grid_tables(values, g), *cols, ones, *unit, False, 0.6)
    assert np.mean((cov > 0.3) & (cov < 0.6)) > 0.1
    for m, st in enumerate(settings):
        ref = _singleray_grid_numba(rho, phi, z, 1.0, 2.0, *_grid_tables(values, g, st['keep']),
                                    *_separate_args(st, base, lengths, dts, offs, True))[0]
        assert out[m].tobytes() == ref.tobytes(), ('grid', m)


@pytest.fixture(scope='module')
def batch_reco(det, table_dir):
    """Reconstruction object initialised with the 15-channel configuration."""
    reco = InterferometricReco3D()
    reco.begin(STATION, reference_config(table_dir, channels=CHANNELS), det)
    return reco


@pytest.fixture(scope='module')
def batch_pairs(batch_reco, det, table_dir, pa):
    """Pair sets of every 15-channel pair for a source event and a noise event."""
    tables = TravelTimeTables(table_dir, STATION, CHANNELS)
    config = reference_config(table_dir, channels=CHANNELS)
    evt = make_event(det, STATION, cylindrical_to_enu(70.0, 60.0, -35.0, pa), CHANNELS, tables, snr=12.0, seed=21)
    noise = make_noise_event(STATION, CHANNELS, seed=8)
    return [batch_reco.compute_pairs(stn, config, store=True) for _, stn in (evt[:2], noise)]


def _settings(table_dir):
    """A batch mixing configurations, masks, weights, polarities, delay and position shifts."""
    groups = dict(channels=CHANNELS, polarization_groups=GROUPS)
    rec_vpol = reference_config(table_dir, channels=CHANNELS, validation=True, **CANDIDATE_KEYS)
    record = reference_config(table_dir, channels=CHANNELS)
    uniform = {p: 1.0 for p in itertools.combinations(CHANNELS, 2)}
    rng = np.random.default_rng(2)
    random_w = {p: float(w) for p, w in zip(itertools.combinations(CHANNELS, 2), rng.uniform(0.2, 2.0, 105))}
    vmask = HPOL_CHANNELS
    return [
        dict(config=rec_vpol, channel_mask=vmask),
        dict(config=rec_vpol, channel_mask=vmask, pair_weights=uniform),
        dict(config=rec_vpol, channel_mask=vmask + [6, 7]),
        dict(config=rec_vpol, channel_mask=vmask, channel_polarity={6: -1, 7: -1}),
        dict(config=rec_vpol, channel_mask=vmask, channel_delay_shift={22: 2.5, 23: 2.0}),
        dict(config=rec_vpol, channel_mask=vmask, pair_weights=random_w, channel_polarity={9: -1}),
        dict(config=record, channel_mask=vmask),
        dict(config=dict(rec_vpol, region_hypotheses=True), channel_mask=vmask),
        dict(config=reference_config(table_dir, **groups, **CANDIDATE_KEYS)),
        dict(config=reference_config(table_dir, **groups, hpol_sign_mode='joint_sign')),
        dict(config=reference_config(table_dir, **groups, hpol_sign_mode='abs_cross_string'),
             channel_polarity={21: -1}),
        dict(config=rec_vpol, channel_mask=vmask, channel_position_shift={22: (1.5, -0.5), 23: (1.5, -0.5)}),
        dict(config=rec_vpol, channel_mask=vmask),
    ]


@pytest.mark.slow
@pytest.mark.parametrize('batch_grids', [False, True])
def test_batch_equals_separate_calls(batch_reco, batch_pairs, table_dir, batch_grids):
    """Every setting of a batch equals its separate reconstruct_from_pairs call bit for bit."""
    settings = _settings(table_dir)
    for e, pairs in enumerate(batch_pairs):
        separate = [batch_reco.reconstruct_from_pairs(pairs, **s) for s in settings]
        stats = {}
        batched = batch_reco.reconstruct_from_pairs_batch(pairs, settings, stats=stats, batch_grids=batch_grids)
        for j, (a, b) in enumerate(zip(separate, batched)):
            assert_bit_equal(a, b, (e, j, batch_grids))
        assert stats['n_coarse_batches'] >= 2
        if batch_grids:
            assert stats['n_grid_requests'] > stats['n_grid_unique']


@pytest.mark.slow
def test_batch_over_events_equals_per_event(batch_reco, batch_pairs, table_dir):
    """One batch spanning two events equals the batch of each event alone."""
    settings = _settings(table_dir)[:6]
    jobs = [(pairs, s) for pairs in batch_pairs for s in settings]
    together = reco3d_batch.reconstruct_batch(batch_reco, jobs)
    alone = [r for pairs in batch_pairs for r in batch_reco.reconstruct_from_pairs_batch(pairs, settings)]
    for j, (a, b) in enumerate(zip(alone, together)):
        assert_bit_equal(a, b, j)


@pytest.mark.slow
def test_batch_far_field_equals_separate_calls(batch_reco, batch_pairs, table_dir):
    """Far-field settings: the batched coarse sky maps give every setting its separate result bit for bit."""
    settings = [dict(s, config=dict(s['config'], far_field_hypothesis=True)) for s in _settings(table_dir)[:6]]
    for e, pairs in enumerate(batch_pairs):
        separate = [batch_reco.reconstruct_from_pairs(pairs, **s) for s in settings]
        batched = batch_reco.reconstruct_from_pairs_batch(pairs, settings, batch_grids=True)
        for j, (a, b) in enumerate(zip(separate, batched)):
            assert 'far_zen_v1' in b
            assert_bit_equal(a, b, (e, j))


def test_gpu_kernel_source_compiles():
    """The CUDA source of the GPU backend compiles with NVRTC for the V100 architecture (no device needed)."""
    pytest.importorskip('cupy')
    from cupy_backends.cuda.libs import nvrtc
    from NuRadioReco.modules import reco3d_batch_gpu

    try:
        program = nvrtc.createProgram(reco3d_batch_gpu._SRC, 'reco3d_batch_gpu.cu', (), ())
    except Exception as err:
        pytest.skip(f'NVRTC unavailable: {err}')
    try:
        nvrtc.compileProgram(program, ('--gpu-architecture=compute_70',))
    except nvrtc.NVRTCError:
        pytest.fail(nvrtc.getProgramLog(program))


@pytest.mark.slow
def test_gpu_batch_matches_cpu_batch(batch_reco, batch_pairs, table_dir):
    """The GPU backend (device-resident series, packed uploads) gives the CPU batch's primaries and correlations.

    The GPU maps equal the CPU maps to the rounding of the device arithmetic (the grid kernel interpolates the
    travel times with contracted multiply-adds), so the primaries agree to 1e-6 and the polish-grid correlation
    to 1e-6, and so do the far-field directions (coarse sky maps on the GPU); two grid batches on one backend
    also check that the series pool is rebuilt per lockstep.
    """
    cp = pytest.importorskip('cupy')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as err:
        pytest.skip(f'no CUDA device: {err}')
    from NuRadioReco.modules.reco3d_batch_gpu import GpuCoarseBackend

    settings = _settings(table_dir)[:8]
    settings += [dict(s, config=dict(s['config'], far_field_hypothesis=True)) for s in settings[:3]]
    jobs = [(pairs, s) for pairs in batch_pairs for s in settings]
    cpu = reco3d_batch.reconstruct_batch(batch_reco, jobs, batch_grids=True)
    backend = GpuCoarseBackend(batch_reco)
    for _ in range(2):
        gpu = reco3d_batch.reconstruct_batch(batch_reco, jobs, coarse_backend=backend, batch_grids=True)
        for j, (a, b) in enumerate(zip(cpu, gpu)):
            assert b['reco_backend'] == 1
            for k in ('rho', 'phi', 'z'):
                assert abs(a[k] - b[k]) <= 1e-6 * max(1.0, abs(a[k])), (j, k, a[k], b[k])
            assert abs(a['max_corr'] - b['max_corr']) <= 1e-6, (j, a['max_corr'], b['max_corr'])
            for k in ('far_zen_v1', 'far_az_v1', 'far_corr_raw_v1'):
                if k in a:
                    assert abs(a[k] - b[k]) <= 1e-6, (j, k, a[k], b[k])
