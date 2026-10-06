"""The far-field (plane wave) hypothesis of the 3D reconstruction."""

import numpy as np
import hashlib
from functools import lru_cache

from NuRadioReco.modules import reco3d_batch
from NuRadioReco.utilities.reco3d_kernels import USE_NUMBA

from NuRadioReco.modules.reco3d.shared import (
    _CANDIDATE_ORIGIN_CODES,
    _C_M_PER_NS,
    _LBFGSB_ABS_STEP,
    _LBFGSB_REL_STEP,
)
from NuRadioReco.modules.reco3d.correlation import SERIES_MODES
from NuRadioReco.modules.reco3d.optimizer import _lbfgsb_fd_steps, _minimize_lbfgsb

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import (
        _singleray_stack_corr_numba,
        _pairs_corr_numba,
        _plane_wave_corr_grad_numba,
        _far_fd_points_numba,
        _plane_wave_times_numba,
    )


_FAR_COARSE_STEP_DEG = 1.0
_FAR_N_SEEDS = 3
_FAR_SEED_SEPARATION_DEG = 5.0
_FAR_REFINE_HALF_DEG = 1.5
_FAR_REFINE_STEP_DEG = 0.1
_FAR_QUADRATURE_NODES = 64
_FAR_WINDOW_ZENITH_SAMPLES = 9001
_FAR_GUARD_HALF_DEG = 5.0
_FAR_MEMO_SIZE = 256


@lru_cache(maxsize=64)
def _plane_wave_profile(depth_bytes, ice):
    """Gauss-Legendre weights and n(z)^2 at the quadrature depths of each antenna, for plane_wave_times.

    Args:
        depth_bytes: float64 bytes of the antenna depths min(z, 0) in m.
        ice: (n_ice, delta_n, z_0) of the exponential profile.

    Returns:
        (weights, n2): the (_FAR_QUADRATURE_NODES,) weights and the (n_ch, nodes) squared
        refractive index between each antenna's depth and the surface (read-only arrays).
    """
    n_ice, delta_n, z_0 = ice
    depth = np.frombuffer(depth_bytes, dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(_FAR_QUADRATURE_NODES)
    z_nodes = 0.5 * depth[:, None] * (1.0 - nodes[None, :])
    n2 = (n_ice - delta_n * np.exp(z_nodes / z_0)) ** 2
    weights.setflags(write=False)
    n2.setflags(write=False)
    return weights, n2


@lru_cache(maxsize=64)
def _plane_wave_layers(depth_bytes, ice):
    """Gauss-Legendre weights and, per in-ice layer, each antenna's path length and n(z)^2 at its nodes, for plane_wave_times.

    Args:
        depth_bytes: float64 bytes of the antenna depths min(z, 0) in m.
        ice: Tuple of in-ice layers (z_min, z_max, n_ice, delta_n, z_0) (``far_field_profile``).

    Returns:
        (weights, layers): the (_FAR_QUADRATURE_NODES,) weights and per layer (length, n2), the
        (n_ch,) length in m of the part of the layer between each antenna's depth and the surface
        and the (n_ch, nodes) squared refractive index at the quadrature depths of that part
        (read-only arrays).
    """
    depth = np.frombuffer(depth_bytes, dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(_FAR_QUADRATURE_NODES)
    layers = []
    for z_min, z_max, n_ice, delta_n, z_0 in ice:
        lower = np.maximum(depth, z_min)
        length = np.maximum(min(z_max, 0.0) - lower, 0.0)
        z_nodes = lower[:, None] + 0.5 * length[:, None] * (1.0 + nodes[None, :])
        n2 = (n_ice - delta_n * np.exp(z_nodes / z_0)) ** 2
        length.setflags(write=False)
        n2.setflags(write=False)
        layers.append((length, n2))
    weights.setflags(write=False)
    return weights, tuple(layers)


def plane_wave_times(zen, az, ant_xyz, ice):
    """Arrival times of a downgoing plane wave at antennas in a horizontally stratified medium.

    The wave comes from sky zenith ``zen`` and azimuth ``az`` (the direction toward the
    source). Its horizontal slowness sin(zen) / c is conserved through the layers (n = 1 in
    air), so, up to one common constant, the arrival time at (x, y, z) is
    -sin(zen) (x cos(az) + y sin(az)) / c plus a vertical delay: below the surface the
    integral from z to 0 of sqrt(n(z')^2 - sin(zen)^2) / c over z', with the profile n(z) of
    the travel-time tables (``far_field_profile``): one exponential n_ice - delta_n
    exp(z / z_0), or exponential layers integrated layer by layer (Gauss-Legendre
    quadrature, exact to float rounding for these smooth integrands), and above it
    -z cos(zen) / c. This is the limit of the point-source travel time minus its value at a
    reference point as the source recedes along the direction.

    Args:
        zen: Sky zenith angles in rad (0 to pi / 2), any shape.

        az: Azimuths in rad, broadcastable with ``zen``.

        ant_xyz: (n_ch, 3) antenna positions in m, z relative to the ice surface.

        ice: (n_ice, delta_n, z_0) of an exponential profile, or a tuple of in-ice layers
            (z_min, z_max, n_ice, delta_n, z_0) (``far_field_profile``).

    Returns:
        Array of shape broadcast(zen, az).shape + (n_ch,), in ns.
    """
    zen, az = np.broadcast_arrays(np.asarray(zen, dtype=np.float64), np.asarray(az, dtype=np.float64))
    x, y, z = (np.asarray(ant_xyz, dtype=np.float64)[:, k] for k in range(3))
    unique_zen, inverse = np.unique(zen, return_inverse=True)
    sin2 = np.sin(unique_zen)[:, None, None] ** 2
    depth = np.minimum(z, 0.0)
    if np.ndim(ice[0]) == 0:
        weights, n2 = _plane_wave_profile(depth.tobytes(), tuple(float(v) for v in ice))
        in_ice = 0.5 * -depth[None, :] * np.sum(weights * np.sqrt(n2[None] - sin2), axis=-1)
    else:
        weights, layers = _plane_wave_layers(depth.tobytes(), tuple(tuple(float(v) for v in layer) for layer in ice))
        in_ice = 0.0
        for length, n2 in layers:
            in_ice = in_ice + 0.5 * length[None, :] * np.sum(
                weights * np.sqrt(np.maximum(n2[None] - sin2, 0.0)), axis=-1)
    vertical = (in_ice - np.maximum(z, 0.0)[None, :] * np.cos(unique_zen)[:, None]) / _C_M_PER_NS
    horizontal = -np.sin(zen)[..., None] * (np.cos(az)[..., None] * x + np.sin(az)[..., None] * y)
    return horizontal / _C_M_PER_NS + vertical[inverse.reshape(zen.shape)]


def _plane_wave_arrays(ant_xyz, ice):
    """Per-antenna arrays of ``plane_wave_times`` in the form ``_plane_wave_times_numba`` reads.

    Returns:
        (x, y, z, Gauss-Legendre weights, lengths, n2, layered): lengths (n_layers, n_ch) and n2
        (n_layers, n_ch, nodes) per layer of a layered profile, or 0.5 * -min(z, 0) and the
        single profile's n2 as one layer.
    """
    x, y, z = (np.ascontiguousarray(ant_xyz[:, k], dtype=np.float64) for k in range(3))
    depth = np.minimum(z, 0.0)
    if np.ndim(ice[0]) == 0:
        weights, n2 = _plane_wave_profile(depth.tobytes(), tuple(float(v) for v in ice))
        return x, y, z, weights, (0.5 * -depth)[None, :], n2[None], False
    weights, layers = _plane_wave_layers(depth.tobytes(), tuple(tuple(float(v) for v in layer) for layer in ice))
    return (x, y, z, weights, np.stack([length for length, _ in layers]), np.stack([n2 for _, n2 in layers]),
            True)


def far_field_profile(ice_model):
    """Index-of-refraction profile of a NuRadioMC ice model in the form ``plane_wave_times`` reads.

    Args:
        ice_model: NuRadioMC medium (``IceModelSimple`` or ``IceModelExpLayers``, e.g.
            greenland_simple or greenland_3exp_layered).

    Returns:
        (n_ice, delta_n, z_0) of a single exponential, or the tuple of in-ice layers
        (z_min, z_max, n_ice, delta_n, z_0) of a layered model.

    Raises:
        ValueError: For any other medium class.
    """
    from NuRadioMC.utilities import medium_base
    if isinstance(ice_model, medium_base.IceModelSimple):
        return float(ice_model.n_ice), float(ice_model.delta_n), float(ice_model.z_0)
    if isinstance(ice_model, medium_base.IceModelExpLayers):
        return tuple((float(layer['z_min']), float(layer['z_max']), float(layer['n_ice']),
                      float(layer['delta_n']), float(layer['z_0']))
                     for layer in ice_model.layers if layer['z_min'] < 0.0)
    raise ValueError(f"the far field needs an exponential or exponential-layer ice model, got "
                     f"{type(ice_model).__name__}")


def _surface_index(ice):
    """Index of refraction just below the surface of a ``far_field_profile`` profile."""
    if np.ndim(ice[0]) == 0:
        return ice[0] - ice[1]
    z_min, z_max, n_ice, delta_n, z_0 = max(ice, key=lambda layer: layer[1])
    return n_ice - delta_n


class FarFieldMixin:
    """Methods of InterferometricReco3D for the far-field (plane wave) hypothesis."""

    def _far_field_geometry(self, channels):
        """Antenna positions (z relative to the surface) and the ice profile of the plane-wave model.

        The profile is that of the tables' ice model (``_tables_ice_model``).
        """
        if self._far_field_ice is None:
            from NuRadioMC.utilities import medium
            self._far_field_ice = far_field_profile(medium.get_ice_model(self._ice_model))
        return np.array([self.ant_locs[ch] for ch in channels], dtype=np.float64), self._far_field_ice

    def _far_field_hypothesis(self, channels, pair_weights, series, lobe_guard_ns=None,
                              gradient='finite_difference'):
        """Best plane-wave direction over the sky, on the same pair series as the point-source search.

        The per-channel arrival times are ``plane_wave_times``. The raw and both envelope
        correlations are mapped on a sky grid of ``_FAR_COARSE_STEP_DEG`` (zenith 0 to 90
        deg, azimuth 0 to 360 deg); the ``_FAR_N_SEEDS`` highest maxima of each map, at
        least ``_FAR_SEED_SEPARATION_DEG`` apart, are refined on the raw correlation over
        +/- ``_FAR_REFINE_HALF_DEG`` in steps of ``_FAR_REFINE_STEP_DEG`` and then by
        L-BFGS-B on the raw correlation; the direction with the highest raw correlation is
        kept. Pair weights and normalisation are those of the point-source search. Inside a
        batch with ``batch_grids`` the coarse sky maps come from the batch, which maps every
        waiting search together (``reco3d_batch.SkyRef``; on the CPU equal bit for bit).

        With ``lobe_guard_ns`` (config ``far_field_lobe_guard_ns``) the direction is chosen
        on the envelope of the correlation, which has no carrier lobes: the seeds of its
        map are refined on it as above. The raw correlation then only refines that
        direction, over +/- ``_FAR_GUARD_HALF_DEG`` in steps of ``_FAR_REFINE_STEP_DEG``
        and by L-BFGS-B, among the directions where every weighted pair's plane-wave delay
        stays within ``lobe_guard_ns`` of its delay at the envelope peak. Set to half the
        carrier period of a narrow-band source, this keeps the raw correlation from moving
        a pair onto a neighbouring lobe.

        Args:
            channels: Channel IDs of the group.
            pair_weights: Per-pair weights of the search.
            series: Series accessor of ``_group_inputs``.
            lobe_guard_ns: Largest change of any pair delay (ns) the raw refinement may make
                from the envelope peak; None searches the raw and both envelope maps alike.
            gradient: 'finite_difference': L-BFGS-B receives with the objective the forward
                differences it would take itself, the three points evaluated in one
                ``plane_wave_times`` call (with numba ``_far_fd_points_numba`` and
                ``_plane_wave_times_numba``, equal bit for bit; iterates bit for bit those of
                scipy's own differences); 'exact': the exact gradient (``_plane_wave_corr_grad_numba``,
                single-exponential ice models only; ``begin`` refuses a layered one).

        Returns:
            Dict with ``far_zen_v1`` (sky zenith, deg), ``far_az_v1`` (azimuth toward the
            source, deg, the convention of ``phi``), ``far_corr_raw_v1``,
            ``far_corr_env_traces_v1``, ``far_corr_env_correlation_v1`` (the three
            objectives at that direction), ``far_map_snr_v1`` (the raw sky map at the nearest
            grid direction over its standard deviation away from it) and ``far_origin_v1``
            (the chain whose map seeded it, coded as ``candidate_origin``; with the guard,
            the envelope of the correlation).

        The result depends only on the channels, their positions, the ice profile, the pair weights, the
        three pair series, the options, the objective normalisation and where the coarse sky maps are
        computed, so it is kept under a hash of those (the last ``_FAR_MEMO_SIZE``): settings that differ
        only in what the plane-wave search does not read (the point-source volume, for example) compute it
        once, and every one gets the same values.
        """
        ant_xyz, ice = self._far_field_geometry(channels)
        packs = [series(mode)[1] for mode in SERIES_MODES]
        args = self._singleray_corr_args(channels, packs, pair_weights)
        n_ch = len(channels)
        executor = reco3d_batch.current_executor()
        batched = executor is not None and executor.batch_grids
        h = hashlib.sha1(repr((tuple(channels), ice, lobe_guard_ns, gradient, args[7:],
                               type(executor.backend).__name__ if batched else None)).encode())
        for a in (ant_xyz,) + args[:7]:
            h.update(np.ascontiguousarray(a).tobytes())
        key = h.hexdigest()
        if key in self._far_field_memo:
            self._far_field_memo.move_to_end(key)
            self.work['far_shared'] += 1
            return dict(self._far_field_memo[key])
        self.work['far_searches'] += 1

        def sky_maps(zen_deg, az_deg):
            """(3, n_zen, n_az) objective maps of the raw and envelope series on a sky grid."""
            self.work['far_sky_points'] += len(zen_deg) * len(az_deg)
            zz, aa = np.meshgrid(np.radians(zen_deg), np.radians(az_deg), indexing='ij')
            tts = plane_wave_times(zz, aa, ant_xyz, ice).reshape(-1, n_ch)
            maps = _singleray_stack_corr_numba(tts, np.ones(tts.shape, dtype=np.bool_), *args)
            return maps.reshape(len(packs), len(zen_deg), len(az_deg))

        def objective(direction, k=0):
            """Objective of series k at (zen_deg, az_deg)."""
            tts = plane_wave_times(np.radians(direction[0]), np.radians(direction[1]), ant_xyz, ice)
            return _pairs_corr_numba(tts, np.ones(n_ch, dtype=np.bool_), args[0], k, *args[1:])

        if gradient == 'exact':
            gl_weights, n2_nodes = _plane_wave_profile(np.minimum(ant_xyz[:, 2], 0.0).tobytes(),
                                                       tuple(float(v) for v in ice))
        ones = np.ones(n_ch, dtype=np.bool_)
        fd_arrays = _plane_wave_arrays(ant_xyz, ice) if USE_NUMBA else None

        def negative_and_gradient(direction, k, lb, ub):
            """Minus the objective of series k at (zen_deg, az_deg) and the gradient L-BFGS-B receives."""
            if gradient == 'exact':
                zen, az = np.radians(direction[0]), np.radians(direction[1])
                tts = plane_wave_times(zen, az, ant_xyz, ice)
                value = _pairs_corr_numba(tts, ones, args[0], k, *args[1:])
                g_zen, g_az = _plane_wave_corr_grad_numba(
                    tts, float(zen), float(az), ant_xyz, n2_nodes, gl_weights, _C_M_PER_NS,
                    args[0], k, *args[1:])
                return -value, np.array((-g_zen, -g_az))
            if fd_arrays is not None:
                points, zen3, az3 = _far_fd_points_numba(direction, lb, ub, _LBFGSB_ABS_STEP, _LBFGSB_REL_STEP)
                tts = _plane_wave_times_numba(np.sin(zen3), np.cos(zen3), np.cos(az3), np.sin(az3), *fd_arrays,
                                              _C_M_PER_NS)
            else:
                h = _lbfgsb_fd_steps(direction, lb, ub)
                points = np.repeat(direction[None, :], 3, axis=0) + np.vstack([np.zeros(2), np.diag(h)])
                tts = plane_wave_times(np.radians(points[:, 0]), np.radians(points[:, 1]), ant_xyz, ice)
            f = [-_pairs_corr_numba(t, ones, args[0], k, *args[1:]) for t in tts]
            return f[0], np.array([(f[i + 1] - f[0]) / (points[i + 1, i] - direction[i]) for i in range(2)])

        def minimize_far(k, start, bounds):
            """L-BFGS-B on minus the objective of series k from start within bounds."""
            res = _minimize_lbfgsb(negative_and_gradient, start, bounds, 30, 1e-10,
                                   args=(k, np.array([b[0] for b in bounds], dtype=np.float64),
                                         np.array([b[1] for b in bounds], dtype=np.float64)))
            self.work['far_optimizer_runs'] += 1
            self.work['far_optimizer_nit'] += int(res.nit)
            self.work['far_optimizer_nfev'] += int(res.nfev)
            return res

        def refine(zen_p, az_p, k, k_map):
            """Best (zen, az, objective k) near a seed: grid on map k_map, then L-BFGS-B on k."""
            zen_r = np.arange(max(zen_p - _FAR_REFINE_HALF_DEG, 0.0),
                              min(zen_p + _FAR_REFINE_HALF_DEG, 90.0) + 1e-9, _FAR_REFINE_STEP_DEG)
            az_r = np.arange(az_p - _FAR_REFINE_HALF_DEG,
                             az_p + _FAR_REFINE_HALF_DEG + 1e-9, _FAR_REFINE_STEP_DEG)
            refined = sky_maps(zen_r, az_r)[k_map]
            iz, ia = np.unravel_index(np.argmax(refined), refined.shape)
            start = [zen_r[iz], az_r[ia]]
            opt = minimize_far(k, start, [(0.0, 90.0), (start[1] - 2 * _FAR_REFINE_HALF_DEG,
                                                         start[1] + 2 * _FAR_REFINE_HALF_DEG)])
            return ((opt.x[0], opt.x[1], -opt.fun) if -opt.fun > refined[iz, ia]
                    else (start[0], start[1], refined[iz, ia]))

        zen_c = np.arange(0.0, 90.0 + 0.5 * _FAR_COARSE_STEP_DEG, _FAR_COARSE_STEP_DEG)
        az_c = np.arange(0.0, 360.0, _FAR_COARSE_STEP_DEG)
        if batched:
            ref = reco3d_batch.SkyRef(('sky', zen_c.tobytes(), az_c.tobytes()), zen_c, az_c)
            self.work['far_sky_points'] += len(zen_c) * len(az_c)
            coarse = executor.request('sky_maps', (ref, channels, packs, pair_weights)).reshape(
                len(packs), len(zen_c), len(az_c))
        else:
            coarse = sky_maps(zen_c, az_c)
        sep = [_FAR_SEED_SEPARATION_DEG, _FAR_SEED_SEPARATION_DEG, 1.0]
        best = None
        if lobe_guard_ns is None:
            for k, mode in enumerate(SERIES_MODES):
                origin = _CANDIDATE_ORIGIN_CODES['raw' if mode is None else f'envelope:{mode}']
                for zen_p, az_p, _, _ in self._extract_top_n_peaks(
                        coarse[k][:, :, None], zen_c, az_c, np.zeros(1), _FAR_N_SEEDS, sep):
                    found = refine(zen_p, az_p, 0, 0)
                    if best is None or found[2] > best[2]:
                        best = found + (origin,)
        else:
            k_env = SERIES_MODES.index('correlation')
            env = None
            for zen_p, az_p, _, _ in self._extract_top_n_peaks(
                    coarse[k_env][:, :, None], zen_c, az_c, np.zeros(1), _FAR_N_SEEDS, sep):
                found = refine(zen_p, az_p, k_env, k_env)
                if env is None or found[2] > env[2]:
                    env = found
            best = (self._far_field_guarded_raw(env[0], env[1], lobe_guard_ns, sky_maps, objective,
                                                minimize_far, ant_xyz, ice, args)
                    + (_CANDIDATE_ORIGIN_CODES['envelope:correlation'],))
        zen, az = float(best[0]), float(best[1]) % 360.0
        az_diff = np.abs(az_c - az)
        nearest = (int(np.argmin(np.abs(zen_c - zen))), int(np.argmin(np.minimum(az_diff, 360.0 - az_diff))), 0)
        out = {'far_zen_v1': zen, 'far_az_v1': az}
        for k, name in enumerate(('raw', 'env_traces', 'env_correlation')):
            out[f'far_corr_{name}_v1'] = float(objective([zen, az], k))
        out['far_map_snr_v1'] = self._compute_map_snr(coarse[0][:, :, None], nearest)
        out['far_origin_v1'] = int(best[3])
        self._far_field_memo[key] = dict(out)
        if len(self._far_field_memo) > _FAR_MEMO_SIZE:
            self._far_field_memo.popitem(last=False)
        return out

    @staticmethod
    def _far_field_guarded_raw(zen_e, az_e, guard_ns, sky_maps, objective, minimize_far,
                               ant_xyz, ice, args):
        """Best raw correlation among directions whose weighted pair delays stay within guard_ns of the envelope peak.

        Args:
            zen_e, az_e: Envelope peak (deg).
            guard_ns: Largest allowed change of any weighted pair delay (ns).
            sky_maps, objective: Map and point objective closures of ``_far_field_hypothesis``.
            minimize_far: Closure of ``_far_field_hypothesis`` running L-BFGS-B on minus the
                objective of a series from a start within bounds.
            ant_xyz, ice: Plane-wave geometry of ``_far_field_geometry``.
            args: Correlation arguments of ``_singleray_corr_args`` (pair indices and weights).

        Returns:
            (zen, az, raw correlation); the envelope peak itself when no other allowed
            direction correlates better.
        """
        used = args[6] > 0
        ch1, ch2 = args[4][used], args[5][used]
        t_e = plane_wave_times(np.radians(zen_e), np.radians(az_e), ant_xyz, ice)

        def allowed(tts):
            """Whether each row of arrival times keeps every weighted pair delay within the guard."""
            shift = (tts[..., ch2] - tts[..., ch1]) - (t_e[ch2] - t_e[ch1])
            return np.all(np.abs(shift) <= guard_ns, axis=-1)

        zen_g = np.arange(max(zen_e - _FAR_GUARD_HALF_DEG, 0.0),
                          min(zen_e + _FAR_GUARD_HALF_DEG, 90.0) + 1e-9, _FAR_REFINE_STEP_DEG)
        az_g = np.arange(az_e - _FAR_GUARD_HALF_DEG, az_e + _FAR_GUARD_HALF_DEG + 1e-9, _FAR_REFINE_STEP_DEG)
        zz, aa = np.meshgrid(np.radians(zen_g), np.radians(az_g), indexing='ij')
        ok = allowed(plane_wave_times(zz, aa, ant_xyz, ice).reshape(len(zen_g), len(az_g), -1))
        raw = np.where(ok, sky_maps(zen_g, az_g)[0], -np.inf)
        best = (zen_e, az_e, objective([zen_e, az_e]))
        iz, ia = np.unravel_index(np.argmax(raw), raw.shape)
        if raw[iz, ia] > best[2]:
            best = (zen_g[iz], az_g[ia], raw[iz, ia])
        step = _FAR_REFINE_STEP_DEG
        opt = minimize_far(0, best[:2], [(max(best[0] - step, 0.0), min(best[0] + step, 90.0)),
                                         (best[1] - step, best[1] + step)])
        polished = plane_wave_times(np.radians(opt.x[0]), np.radians(opt.x[1]), ant_xyz, ice)
        if -opt.fun > best[2] and allowed(polished):
            best = (opt.x[0], opt.x[1], -opt.fun)
        return best

    def far_field_lag_windows(self, pairs):
        """Lowest and highest delay of each pair over every plane-wave direction of the sky.

        Over the azimuth the horizontal term of ``plane_wave_times`` spans +/- sin(zen) D / c
        (D the antennas' horizontal separation); the vertical terms depend on the zenith
        only and are sampled on ``_FAR_WINDOW_ZENITH_SAMPLES`` zeniths from 0 to 90 deg, the
        sampling error bounded by the largest zenith derivative (``|z| / (c sqrt(n(0)^2 - 1))``
        per antenna below the surface, ``|z| / c`` above it, plus D / c) times half a step.

        Args:
            pairs: Sequence of (ch_a, ch_b) channel pairs.

        Returns:
            (n_pairs, 2) float64 array of [lowest, highest] delay in ns.
        """
        channels = sorted({ch for p in pairs for ch in p})
        ant_xyz, ice = self._far_field_geometry(channels)
        zen = np.linspace(0.0, 0.5 * np.pi, _FAR_WINDOW_ZENITH_SAMPLES)
        vertical = plane_wave_times(zen, 0.0, ant_xyz * [0.0, 0.0, 1.0], ice)
        n_surface = _surface_index(ice)
        slope = np.where(ant_xyz[:, 2] < 0, -ant_xyz[:, 2] / np.sqrt(n_surface ** 2 - 1.0),
                         ant_xyz[:, 2]) / _C_M_PER_NS
        index = {ch: i for i, ch in enumerate(channels)}
        out = np.empty((len(pairs), 2))
        for k, (a, b) in enumerate(pairs):
            ia, ib = index[a], index[b]
            reach = np.sin(zen) * np.hypot(*(ant_xyz[ia, :2] - ant_xyz[ib, :2])) / _C_M_PER_NS
            delta = vertical[:, ia] - vertical[:, ib]
            margin = 0.5 * (zen[1] - zen[0]) * (
                slope[ia] + slope[ib] + np.hypot(*(ant_xyz[ia, :2] - ant_xyz[ib, :2])) / _C_M_PER_NS)
            out[k] = (np.min(delta - reach) - margin, np.max(delta + reach) + margin)
        return out
