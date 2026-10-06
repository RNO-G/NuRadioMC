"""Travel-time tables and source geometry of the 3D reconstruction."""

import numpy as np
import datetime
import hashlib
import itertools
import os
import logging
from collections import namedtuple
from functools import lru_cache

from scipy.interpolate import RegularGridInterpolator

from NuRadioReco.utilities.reco3d_kernels import (
    USE_NUMBA,
    _build_z_vec,
    _build_split_z_vec,
    RAY_TYPES,
    SOLUTION_TYPES,
)

from NuRadioReco.modules.reco3d.shared import logger

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import _bilinear_batch_numba, _bilinear_ok_batch_numba


# Bound to the class by the module that defines it, once the class exists.
InterferometricReco3D = None


TableData = namedtuple('TableData', [
    'interp', 'values', 'r_min', 'z_min', 'dr_inv', 'dz_inv', 'nr', 'nz',
])

DETECTOR_Z_TOLERANCE_M = 0.05
_TABLE_METADATA_KEYS = ('antenna_z_abs', 'det_date', 'det_source', 'det_file', 'ice_model')
_SNAPSHOT_DETECTORS = {}


def table_files_from_config(station_id, config):
    """Return channel -> list of travel-time table files named by a configuration.

    Mirrors the file selection of `InterferometricReco3D._preload_tables`: one combined
    table per channel, or one table per active ray type with `multi_ray_types`.
    """
    base = config['time_delay_tables']
    if config.get('multi_ray_types', False):
        ray_types = SOLUTION_TYPES if config.get('table_scheme', 'ray_type') == 'solution_ordered' else RAY_TYPES
        pattern = config.get('multiray_table_name_pattern', 'st{station_id}_ch{ch}_rz_table_{ray_type}.npz')
        return {ch: [os.path.join(base, f"station{station_id}",
                                  pattern.format(station_id=station_id, ch=ch, ray_type=rt))
                     for rt in ray_types] for ch in config['channels']}
    pattern = config.get('table_name_pattern', 'st{station_id}_ch{ch}_rz_table.npz')
    return {ch: [os.path.join(base, f"station{station_id}", pattern.format(station_id=station_id, ch=ch))]
            for ch in config['channels']}


def snapshot_search_dirs(config):
    """Return the directories searched for the detector snapshot a table was generated from."""
    dirs = []
    if config.get('detector_file'):
        dirs.append(os.path.dirname(os.path.abspath(config['detector_file'])))
    if os.environ.get('NURADIO_DETECTOR_DIR'):
        dirs.append(os.environ['NURADIO_DETECTOR_DIR'])
    return dirs


def table_metadata(path):
    """Return the provenance metadata stored in a travel-time table (empty for tables without it)."""
    with np.load(path) as d:
        return {k: d[k].item() for k in _TABLE_METADATA_KEYS if k in d.files}


def detector_delay_vector(det, station_id, channels):
    """Return the per-channel cable delays at the detector's current epoch, their hash and the epoch.

    Returns:
        Tuple (delays, digest, epoch): delays maps channel -> delay in ns, digest is the
        first 16 hex characters of the SHA-256 of the delays printed to 1 fs, and epoch
        is the ISO string of the detector time.
    """
    delays = {int(ch): float(det.get_cable_delay(station_id, ch)) for ch in channels}
    digest = hashlib.sha256(','.join(f'{ch}:{delays[ch]:.6f}' for ch in delays).encode()).hexdigest()[:16]
    return delays, digest, det.get_detector_time().isoformat()


def _snapshot_detector(path, det_date, station_id):
    """Load (once per process) the detector snapshot a table was generated from."""
    key = (path, det_date, int(station_id))
    if key not in _SNAPSHOT_DETECTORS:
        from NuRadioReco.detector.RNO_G import rnog_detector
        snap = rnog_detector.Detector(detector_file=path, select_stations=int(station_id),
                                      log_level=logging.WARNING)
        snap.update(datetime.datetime.fromisoformat(det_date))
        _SNAPSHOT_DETECTORS[key] = snap
    return _SNAPSHOT_DETECTORS[key]


def check_detector_consistency(det, station_id, channels, tables, snapshot_dirs=(),
                               z_tolerance=DETECTOR_Z_TOLERANCE_M):
    """Check the detector description at its current epoch against the travel-time tables.

    For every channel whose table carries `antenna_z_abs` the absolute antenna z of the
    live description (relative z plus station z) must agree within `z_tolerance`; a
    larger difference means the tables were generated for another description and the
    reconstruction would be inconsistent, so it raises. When the table names the
    snapshot it was generated from (`det_file`, `det_date`) and that file is found in
    `snapshot_dirs`, the horizontal channel positions are compared as well and a
    difference above the tolerance is logged as a warning. The per-channel cable delays
    at the current epoch are recorded with their hash so that a results file can state
    which delays it was reconstructed with.

    Args:
        det: Detector description, updated to the epoch to check.
        station_id: Station number.
        channels: Channels to check.
        tables: Mapping channel -> list of table files (`table_files_from_config`), or None.
        snapshot_dirs: Directories searched for the snapshot named in the table metadata.
        z_tolerance: Largest accepted antenna z difference in metres.

    Returns:
        Report dict with `station_id`, `epoch`, `cable_delays_ns`, `delay_hash`,
        `z_checked` (channels with a table z), `z_max_diff_m`, `xy_checked` and
        `xy_max_diff_m`.

    Raises:
        ValueError: if any checked antenna z differs from its table by more than the tolerance.
    """
    station_id = int(station_id)
    station_z = float(det.get_absolute_position(station_id)[2])
    delays, digest, epoch = detector_delay_vector(det, station_id, channels)
    report = {'station_id': station_id, 'epoch': epoch, 'cable_delays_ns': delays,
              'delay_hash': digest, 'z_checked': [], 'z_max_diff_m': 0.0,
              'xy_checked': [], 'xy_max_diff_m': 0.0}
    failures = []
    for ch in channels:
        rel = np.asarray(det.get_relative_position(station_id, ch), dtype=float)
        z_live = rel[2] + station_z
        for path in (tables or {}).get(ch, []):
            meta = table_metadata(path)
            if 'antenna_z_abs' not in meta:
                continue
            diff = abs(z_live - float(meta['antenna_z_abs']))
            report['z_checked'].append(int(ch))
            report['z_max_diff_m'] = max(report['z_max_diff_m'], diff)
            if diff > z_tolerance:
                failures.append(f"ch{ch}: table {os.path.basename(path)} z {meta['antenna_z_abs']:.3f} m, "
                                f"detector {z_live:.3f} m at {epoch}")
            snapshot = next((os.path.join(d, meta['det_file']) for d in snapshot_dirs
                             if meta.get('det_file') and meta.get('det_date')
                             and os.path.isfile(os.path.join(d, meta['det_file']))), None)
            if snapshot is None:
                continue
            snap_rel = np.asarray(_snapshot_detector(snapshot, meta['det_date'], station_id)
                                  .get_relative_position(station_id, ch), dtype=float)
            diff_xy = float(np.hypot(rel[0] - snap_rel[0], rel[1] - snap_rel[1]))
            report['xy_checked'].append(int(ch))
            report['xy_max_diff_m'] = max(report['xy_max_diff_m'], diff_xy)
            if diff_xy > z_tolerance:
                logger.warning("ch%d horizontal position differs from the table snapshot %s by %.3f m",
                               ch, os.path.basename(snapshot), diff_xy)
    if failures:
        raise ValueError("antenna z of the detector description differs from the travel-time tables: "
                         + '; '.join(failures))
    logger.info("detector consistency at %s: delay hash %s, %d channels checked in z (max %.3f m)",
                epoch, digest, len(report['z_checked']), report['z_max_diff_m'])
    return report


class TablesMixin:
    """Methods of InterferometricReco3D for travel-time tables and source geometry."""

    @staticmethod
    def _split_z_grid(config):
        """Return the (below, above) blocks of the split z grid, or None when the keys are absent.

        ``z_grid_below`` describes the in-ice block and ``z_grid_above`` the air block,
        each a mapping with ``n`` (points), ``spacing`` (``linear`` or ``log``) and
        ``offset`` (m); ``z_grid_above`` also takes ``refine_spacing``, the placement
        of the air part of refine and polish windows. Both keys must be present
        together and replace ``coarse_n_z``, ``n_z`` and ``z_spacing``. Missing
        spacings default to linear below and log above, the air refine spacing to
        linear, the in-ice offset to ``z_surface_offset`` (0.1 m, used by log spacing
        only) and the air offset to 1.0 m, the first tabulated air row.

        Returns:
            None, or a (below, above) pair of dicts with keys n, spacing and offset,
            plus refine_spacing for the air block.

        Raises:
            ValueError: If only one key is present, ``coarse_n_z``, ``n_z`` or
                ``z_spacing`` is given as well, a block is not a mapping or holds an
                unknown key, ``n`` is not an integer of at least 2, a spacing is
                unknown or an offset is not positive.
        """
        present = [k for k in ('z_grid_below', 'z_grid_above') if k in config]
        if not present:
            return None
        if len(present) == 1:
            raise ValueError(
                "z_grid_below and z_grid_above must be given together, "
                f"got only {present[0]}")
        replaced = [k for k in ('coarse_n_z', 'n_z') if config.get(k, 0)]
        if 'z_spacing' in config:
            replaced.append('z_spacing')
        if replaced:
            raise ValueError(
                "z_grid_below and z_grid_above replace coarse_n_z, n_z and "
                f"z_spacing; remove {replaced}")
        specs = {
            'z_grid_below': (('n', 'spacing', 'offset'), 'linear',
                             config.get('z_surface_offset', 0.1)),
            'z_grid_above': (('n', 'spacing', 'offset', 'refine_spacing'), 'log', 1.0),
        }
        blocks = []
        for key, (allowed, spacing_default, offset_default) in specs.items():
            block = config[key]
            if not isinstance(block, dict):
                raise ValueError(f"{key} must be a mapping, got {block!r}")
            unknown = sorted(set(block) - set(allowed))
            if unknown:
                raise ValueError(f"{key} has unknown keys {unknown}; allowed: {list(allowed)}")
            n = block.get('n', None)
            if isinstance(n, bool) or not isinstance(n, int) or n < 2:
                raise ValueError(f"{key}.n must be an integer of at least 2, got {n!r}")
            spacing = block.get('spacing', spacing_default)
            if spacing not in ('linear', 'log'):
                raise ValueError(
                    f"{key}.spacing must be 'linear' or 'log', got {spacing!r}")
            offset = block.get('offset', offset_default)
            if not isinstance(offset, (int, float)) or isinstance(offset, bool) or offset <= 0:
                raise ValueError(f"{key}.offset must be a positive number, got {offset!r}")
            parsed = {'n': int(n), 'spacing': spacing, 'offset': float(offset)}
            if key == 'z_grid_above':
                refine_spacing = block.get('refine_spacing', 'linear')
                if refine_spacing not in ('linear', 'log'):
                    raise ValueError(
                        f"{key}.refine_spacing must be 'linear' or 'log', got {refine_spacing!r}")
                parsed['refine_spacing'] = refine_spacing
            blocks.append(parsed)
        return tuple(blocks)

    def _coarse_z_grid(self, config, z_min, z_max, d_z):
        """Build the coarse z vector and the part of the delay cache key that describes it.

        With ``z_grid_below`` and ``z_grid_above`` the vector is the split grid of
        _build_split_z_vec and the key carries the two blocks; otherwise
        ``coarse_n_z`` points from _build_z_vec with ``z_spacing`` and
        ``z_surface_offset``, or steps of ``d_z`` when ``coarse_n_z`` is 0, with the
        key of the single z vector.

        Args:
            config: Reconstruction config dict.
            z_min: Lower edge of the coarse volume in m.
            z_max: Upper edge of the coarse volume in m.
            d_z: Coarse z step in m, used when neither point count is given.

        Returns:
            (z_vec, key) with z_vec sorted ascending and key a hashable tuple.
        """
        n_z = config.get('coarse_n_z', 0)
        z_spacing = config.get('z_spacing', 'linear')
        z_surf_offset = float(config.get('z_surface_offset', 0.1))
        split = self._split_z_grid(config)
        if split is not None:
            z_vec = _build_split_z_vec(z_min, z_max, *split)
            key = (z_min, z_max, 'split') + tuple(
                tuple(sorted(block.items())) for block in split)
        elif n_z > 0:
            z_vec = _build_z_vec(z_min, z_max, n_z, z_spacing, z_surf_offset)
            key = (d_z, z_min, z_max, n_z, z_spacing, z_surf_offset)
        else:
            z_vec = np.arange(z_min, z_max + d_z, d_z)
            key = (d_z, z_min, z_max, n_z, z_spacing, z_surf_offset)
        return z_vec, key

    @staticmethod
    def _read_rz_table(table_filename):
        """Read an R-Z travel-time table and fill its NaN rows.

        Interior NaN rows are filled by averaging the adjacent z slices and the
        top row by linear extrapolation, which handles the ice surface where ray
        tracers produce NaN between valid rows.

        Args:
            table_filename: Path to a .npz file with keys 'r_range_vals',
                'z_range_vals' and 'data'.

        Returns:
            (values, r_range, z_range) with values of shape (nr, nz).
        """
        f = np.load(table_filename)
        travel_time_table = f['data'].copy()
        r_range = f['r_range_vals']
        z_range = f['z_range_vals']

        nr, nz = travel_time_table.shape
        # Interior: average of neighbors on both sides
        for j in range(1, nz - 1):
            nan_mask = np.isnan(travel_time_table[:, j])
            if not nan_mask.any():
                continue
            below = travel_time_table[:, j - 1]
            above = travel_time_table[:, j + 1]
            fillable = nan_mask & np.isfinite(below) & np.isfinite(above)
            if fillable.any():
                travel_time_table[fillable, j] = (
                    below[fillable] + above[fillable]) / 2.0
        # Top boundary: linear extrapolation from j-2, j-1
        if nz >= 3:
            j = nz - 1
            nan_mask = np.isnan(travel_time_table[:, j])
            if nan_mask.any():
                v1 = travel_time_table[:, j - 2]
                v2 = travel_time_table[:, j - 1]
                fillable = nan_mask & np.isfinite(v1) & np.isfinite(v2)
                if fillable.any():
                    travel_time_table[fillable, j] = (
                        2.0 * v2[fillable] - v1[fillable])
        return np.ascontiguousarray(travel_time_table, dtype=np.float64), r_range, z_range

    @staticmethod
    def _table_data(values, r_range, z_range, interpolation_method):
        """Wrap a filled table in a TableData with its SciPy interpolator and grid parameters."""
        interp = RegularGridInterpolator(
            (r_range, z_range), values,
            method=interpolation_method, bounds_error=False, fill_value=-np.inf
        )
        return TableData(
            interp=interp,
            values=values,
            r_min=float(r_range[0]),
            z_min=float(z_range[0]),
            dr_inv=1.0 / (r_range[1] - r_range[0]),
            dz_inv=1.0 / (z_range[1] - z_range[0]),
            nr=len(r_range),
            nz=len(z_range),
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_rz_interpolator(table_filename, interpolation_method):
        """Load one R-Z travel time table as a TableData, cached per file.

        Args:
            table_filename: Path to a .npz file with keys 'r_range_vals',
                'z_range_vals' and 'data'.
            interpolation_method: Interpolation method for RegularGridInterpolator.

        Returns:
            TableData wrapping the SciPy interpolator and the raw grid arrays for
            the Numba bilinear kernels.
        """
        values, r_range, z_range = InterferometricReco3D._read_rz_table(table_filename)
        return InterferometricReco3D._table_data(values, r_range, z_range, interpolation_method)

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_table_stack(table_files, interpolation_method):
        """Load several R-Z tables into one contiguous stack, cached per file tuple.

        The kernels index the stack by channel slot; each TableData's ``values`` is
        a view into the stack when the tables share one shape, so the tables are held
        once per process for a given channel set.

        Args:
            table_files: Tuple of .npz paths, one per channel.
            interpolation_method: Interpolation method for RegularGridInterpolator.

        Returns:
            (stack, ok, tables) with stack (n, nr_max, nz_max) float64 padded with
            NaN, ok the boolean finiteness mask of the stack and tables a list of
            TableData in the order of ``table_files``.
        """
        loaded = [InterferometricReco3D._read_rz_table(f) for f in table_files]
        nr_max = max(v.shape[0] for v, _, _ in loaded)
        nz_max = max(v.shape[1] for v, _, _ in loaded)
        stack = np.full((len(loaded), nr_max, nz_max), np.nan, dtype=np.float64)
        tables = []
        for i, (values, r_range, z_range) in enumerate(loaded):
            nr, nz = values.shape
            stack[i, :nr, :nz] = values
            view = stack[i, :nr, :nz]
            if not view.flags.c_contiguous:
                view = values
            tables.append(InterferometricReco3D._table_data(
                view, r_range, z_range, interpolation_method))
        return stack, np.isfinite(stack), tables

    def _preload_tables(self, station_id, config):
        """Load travel time interpolators for all channels.

        Loads per-ray-type tables (direct, refracted, reflected) when
        multi_ray_types is enabled. Falls back to single combined table.
        Empties every cache built from the previous tables (``end``), so a
        second ``begin`` with other tables never reuses their travel times.

        Parameters
        ----------
        station_id : int
            Station ID.
        config : dict
            Must contain 'channels', 'time_delay_tables', and optionally
            'table_name_pattern' and 'interp_method'.
        """
        interp_method = config.get('interp_method', 'linear')
        table_base = config['time_delay_tables']
        self._multi_ray_types = config.get('multi_ray_types', False)
        self._multiray_combo_mode = config.get('multiray_combo_mode', 'per_pair')

        table_scheme = config.get('table_scheme', 'ray_type')
        if table_scheme == 'solution_ordered':
            self._active_ray_types = SOLUTION_TYPES
        else:
            self._active_ray_types = RAY_TYPES
        self._n_ray_slots = len(self._active_ray_types)

        self._interpolators = {}
        self._multiray_interpolators = {}
        self._table_files = table_files_from_config(station_id, config)

        if self._multi_ray_types:
            pattern = config.get(
                'multiray_table_name_pattern',
                'st{station_id}_ch{ch}_rz_table_{ray_type}.npz'
            )
            for ch in config['channels']:
                self._multiray_interpolators[ch] = {}
                for rt in self._active_ray_types:
                    fname = pattern.format(
                        station_id=station_id, ch=ch, ray_type=rt
                    )
                    table_file = os.path.join(
                        table_base, f"station{station_id}", fname
                    )
                    self._multiray_interpolators[ch][rt] = \
                        self._load_rz_interpolator(table_file, interp_method)
            logger.info("Loaded %s tables (%d types) for %d channels",
                        table_scheme, self._n_ray_slots,
                        len(config['channels']))
        else:
            pattern = config.get('table_name_pattern',
                                 'st{station_id}_ch{ch}_rz_table.npz')
            channels = list(config['channels'])
            files = tuple(
                os.path.join(table_base, f"station{station_id}",
                             pattern.format(station_id=station_id, ch=ch))
                for ch in channels)
            stack, ok, tables = self._load_table_stack(files, interp_method)
            self._interpolators = dict(zip(channels, tables))
            self._table_stack = stack
            self._table_ok = ok
            self._table_channels = tuple(channels)
            self._singleray_table_cache = {}
        self.end()

        self._two_arrival_interpolators = {}
        self._two_arrival_packed = {}
        if self._two_arrival_settings(config) is not None:
            from NuRadioMC.utilities import medium
            ice = medium.greenland_simple()
            self._two_arrival_ice = (float(ice.n_ice), float(ice.delta_n),
                                     float(ice.z_0))
            pattern = config.get(
                'multiray_table_name_pattern',
                'st{station_id}_ch{ch}_rz_table_{ray_type}.npz'
            )
            for ch in config['channels']:
                self._two_arrival_interpolators[ch] = {}
                for rt in SOLUTION_TYPES:
                    fname = pattern.format(station_id=station_id, ch=ch, ray_type=rt)
                    table_file = os.path.join(table_base, f"station{station_id}", fname)
                    self._two_arrival_interpolators[ch][rt] = \
                        self._load_rz_interpolator(table_file, interp_method)
            logger.info("Loaded solution-ordered tables for the two-arrival polish "
                        "(%d channels)", len(config['channels']))

    def _position_shift_key(self, shift):
        """Validated, hashable form of a channel_position_shift (zero entries dropped).

        Args:
            shift: Dict channel -> (dx, dy) in m, or None.

        Returns:
            Sorted tuple of (channel, dx, dy) with a nonzero shift.

        Raises:
            ValueError: For a channel without a position or an entry that is not two
                finite numbers (a vertical shift would need new tables).
        """
        if not shift:
            return ()
        if not isinstance(shift, dict):
            raise ValueError("channel_position_shift must map channels to (dx, dy) in m")
        key = []
        for ch, value in shift.items():
            if int(ch) not in self.ant_locs:
                raise ValueError(f"channel_position_shift: channel {ch} has no position")
            try:
                dxy = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                dxy = np.full(1, np.nan)
            if dxy.shape != (2,) or not np.all(np.isfinite(dxy)):
                raise ValueError(f"channel_position_shift[{ch}] must be two finite numbers (dx, dy) in m, "
                                 f"got {value!r}")
            if dxy.any():
                key.append((int(ch), float(dxy[0]), float(dxy[1])))
        return tuple(sorted(key))

    def _set_position_shift(self, shift):
        """Move the channels' horizontal positions to the database values plus ``shift``.

        Does nothing when the shift equals the current one; otherwise every cache built
        from the positions is emptied. The phased-array centre that defines the
        reconstruction frame is not moved. ``run``, ``compute_pairs``,
        ``reconstruct_from_pairs`` and ``pair_lag_windows`` set the shift of their
        config; the positions keep it until the next of these calls.

        Args:
            shift: Dict channel -> (dx, dy) in m, or None for the database positions.
        """
        key = self._position_shift_key(shift)
        if key == self._position_key:
            return
        self.ant_locs = {ch: pos.copy() for ch, pos in self._ant_locs_db.items()}
        for ch, dx, dy in key:
            self.ant_locs[ch][:2] += (dx, dy)
        for name in ('_delay_matrix_cache', '_gpu_delay_stack_cache', '_tt_stack_cache', '_batch_grid_cache',
                     '_opt_geom_cache',
                     '_packed_multiray_tables', '_table_mask_cache', '_lag_window_cache', '_cpu_delay_T_cache',
                     '_singleray_table_cache', '_two_arrival_packed', '_zen_mask_cache'):
            cache = getattr(self, name, None)
            if cache is not None:
                cache.clear()
        self._position_key = key

    @staticmethod
    def _get_ant_locs(station_id, det):
        """Get antenna positions with absolute Z.

        Parameters
        ----------
        station_id : int
            Station ID.
        det : Detector
            Detector description.

        Returns
        -------
        dict
            Channel ID -> [x_rel, y_rel, z_abs] array.
        """
        station_abs_pos = det.get_absolute_position(int(station_id))
        station_abs_z = station_abs_pos[2]
        locs = {}
        for ch in range(24):
            rel = np.array(det.get_relative_position(int(station_id), int(ch)))
            rel[2] += station_abs_z
            locs[ch] = rel
        return locs

    _MAX_GRID_POINTS = 50_000_000

    def _generate_coord_arrays(self, config):
        """Generate 1D coordinate arrays from 6-element limits.

        Parameters
        ----------
        config : dict
            Must contain 'limits' (6 elements) and 'step_sizes' (3 elements).

        Returns
        -------
        tuple
            (rho_vec, phi_vec, z_vec) in physical units (m, rad, m).
        """
        rho_min, rho_max, phi_min, phi_max, z_min, z_max = config['limits']
        d_rho, d_phi, d_z = config['step_sizes']

        if z_max > 0 and not config.get('allow_above_surface', False):
            raise ValueError(
                f"z_max={z_max} is above the ice surface. Travel time "
                f"tables only cover in-ice positions (z <= 0) unless "
                f"above-surface tables are used; set allow_above_surface: "
                f"true with such tables, or use negative z values, e.g. "
                f"limits: [{rho_min}, {rho_max}, {phi_min}, {phi_max}, "
                f"-{abs(z_max)}, {z_min}]"
            )

        rho_min = max(rho_min, 1.0)  # avoid R=0 table edge effects

        z_spacing = config.get('z_spacing', 'linear')
        n_z_cfg = config.get('n_z', 0)
        z_split = self._split_z_grid(config)

        n_rho = len(np.arange(rho_min, rho_max + d_rho, d_rho))
        n_phi = len(np.arange(phi_min, phi_max, d_phi))
        if z_split is not None:
            z_vec = _build_split_z_vec(z_min, z_max, *z_split)
        elif n_z_cfg > 0:
            z_vec = _build_z_vec(
                z_min, z_max, n_z_cfg, z_spacing, config.get('z_surface_offset', 0.1))
        else:
            z_vec = np.arange(z_min, z_max + d_z, d_z)
        n_z = len(z_vec)
        n_total = n_rho * n_phi * n_z

        if n_total > self._MAX_GRID_POINTS:
            raise ValueError(
                f"Grid has {n_total:,} points "
                f"({n_rho} rho x {n_phi} phi x {n_z} z), which exceeds "
                f"the {self._MAX_GRID_POINTS:,} point limit. This would "
                f"require excessive memory. Use 'coarse_limits' and "
                f"'coarse_step_sizes' with the hierarchical search "
                f"(set hierarchical: true) for large search volumes, "
                f"or increase step_sizes."
            )

        if not config.get('hierarchical', False):
            logger.warning(
                "Running flat grid scan with %s points. Consider using "
                "hierarchical: true with coarse_limits/coarse_step_sizes "
                "for better performance.", f"{n_total:,}"
            )

        rho_vec = np.arange(rho_min, rho_max + d_rho, d_rho)
        phi_vec = np.arange(phi_min, phi_max, d_phi) * (np.pi / 180.0)
        return rho_vec, phi_vec, z_vec

    def _build_source_enu_matrix(self, rho_vec, phi_vec, z_vec):
        """Build 3D matrix of source ENU positions.

        Parameters
        ----------
        rho_vec : array
            Radial distances in meters.
        phi_vec : array
            Azimuths in radians.
        z_vec : array
            Depths in meters (absolute).

        Returns
        -------
        np.ndarray
            Shape (n_rho, n_phi, n_z, 3) with [x, y, z] at each point.
        """
        rho_g, phi_g, z_g = np.meshgrid(rho_vec, phi_vec, z_vec, indexing='ij')
        x = rho_g * np.cos(phi_g) + self._pa_center[0]
        y = rho_g * np.sin(phi_g) + self._pa_center[1]

        return np.stack((x, y, z_g), axis=-1)

    def _compute_rho_and_coords(self, src_enu, channels):
        """Compute per-channel (R, Z) coordinate buffers for table lookups.

        Parameters
        ----------
        src_enu : np.ndarray
            Source ENU matrix, shape (n_rho, n_phi, n_z, 3).
        channels : list
            Channel IDs.

        Returns
        -------
        dict
            Channel ID -> (flat_size, 2) array of [R, Z] coordinates.
        """
        grid_shape = src_enu.shape[:3]
        flat_size = int(np.prod(grid_shape))
        xy_positions = src_enu[..., :2]
        z_grid = src_enu[..., 2]

        dx = np.empty(grid_shape, dtype=np.float64)
        dy = np.empty(grid_shape, dtype=np.float64)

        coords_per_ch = {}
        for ch in channels:
            pos = self.ant_locs[ch]
            np.subtract(xy_positions[..., 0], pos[0], out=dx)
            np.subtract(xy_positions[..., 1], pos[1], out=dy)
            np.multiply(dx, dx, out=dx)
            np.multiply(dy, dy, out=dy)
            np.add(dx, dy, out=dx)
            np.sqrt(dx, out=dx)
            np.maximum(dx, 1.0, out=dx)

            buf = np.empty((flat_size, 2), dtype=np.float64)
            buf[:, 0] = dx.ravel()
            buf[:, 1] = z_grid.ravel()
            coords_per_ch[ch] = buf

        return coords_per_ch

    def _compute_delay_matrices(self, src_enu, channels):
        """Compute pairwise time delay matrices over a 3D grid.

        Parameters
        ----------
        src_enu : np.ndarray
            Source ENU matrix, shape (n_rho, n_phi, n_z, 3).
        channels : list
            Channel IDs.

        Returns
        -------
        list
            One 3D delay matrix per channel pair.
        """
        grid_shape = src_enu.shape[:3]
        coords_per_ch = self._compute_rho_and_coords(src_enu, channels)

        travel_times = {
            ch: self._table_lookup_batch(
                self._interpolators[ch], coords_per_ch[ch]).reshape(grid_shape)
            for ch in channels}

        ch_pairs = list(itertools.combinations(channels, 2))
        return [travel_times[c1] - travel_times[c2] for c1, c2 in ch_pairs]

    def _table_lookup_batch(self, td, coords):
        """Look up travel times for a batch of (R, Z) points in one table.

        Uses the strict Numba batch kernel; with ``tolerant_table_edge`` the
        masked lookup ``_bilinear_ok_batch_numba`` of the fused kernels, whose
        edge rule accepts a query on the last table row or column, with the
        table's finiteness mask; the SciPy interpolator (whose bounds are
        inclusive) without Numba.

        Args:
            td: TableData of one channel and ray type.
            coords: (n, 2) array of [R, Z] query points.

        Returns:
            (n,) array of travel times, -inf out of bounds.
        """
        if not USE_NUMBA:
            return td.interp(coords)
        if not self._tolerant_table_edge:
            return _bilinear_batch_numba(td.values, td.r_min, td.dr_inv, td.nr,
                                         td.z_min, td.dz_inv, td.nz,
                                         coords[:, 0], coords[:, 1])
        values, ok = self._table_slot_and_mask(td)
        tt, valid = _bilinear_ok_batch_numba(
            values, ok, 0, td.r_min, td.dr_inv, td.nr, td.z_min, td.dz_inv, td.nz,
            np.ascontiguousarray(coords[:, 0]), np.ascontiguousarray(coords[:, 1]), True)
        return np.where(valid, tt, -np.inf)

    def _table_slot_and_mask(self, td):
        """Return one table as a single-slot stack and its finiteness mask, cached per table.

        Args:
            td: TableData of one channel and ray type.

        Returns:
            (values, ok) of shape (1, nr, nz) for ``_bilinear_ok_batch_numba``.
        """
        entry = self._table_mask_cache.get(id(td))
        if entry is None or entry[0] is not td:
            values = td.values[None]
            entry = (td, values, np.isfinite(values))
            self._table_mask_cache[id(td)] = entry
        return entry[1], entry[2]

    def _get_t_delay_matrices(self, station_id, config, src_enu, channels, z_vec):
        """Compute pairwise time delay matrices, cached by grid.

        Parameters
        ----------
        station_id : int
            Station ID.
        config : dict
            Configuration dictionary with 'limits' and 'step_sizes'.
        src_enu : np.ndarray
            Source ENU matrix, shape (n_rho, n_phi, n_z, 3).
        channels : list
            Channel IDs; the pair order of the matrices follows their order.
        z_vec : np.ndarray
            The z axis of ``src_enu``; part of the cache key, since equal limits
            and steps give different z vectors under ``n_z``, ``z_spacing`` or
            the split z grid keys.

        Returns
        -------
        list
            One 3D delay matrix per channel pair.
        """
        cache_key = (
            station_id,
            tuple(channels),
            tuple(config['limits']),
            tuple(config['step_sizes']),
            z_vec.tobytes(),
        )
        if cache_key in self._delay_matrix_cache:
            return self._delay_matrix_cache[cache_key]

        delay_matrices = self._compute_delay_matrices(src_enu, channels)
        self._delay_matrix_cache[cache_key] = delay_matrices
        return delay_matrices

    def _pack_singleray_tables(self, channels):
        """Return the singleray table geometry of a channel group for the fused kernels.

        The travel-time stack and its finiteness mask are shared by every group;
        ``td_slot`` maps the group's channel index to its slot in the stack. The
        per-group arrays are cached by channel tuple.

        Args:
            channels: Channel IDs of the group, in kernel order.

        Returns:
            Dict with td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz, ant_xy, pa_x and pa_y.
        """
        key = tuple(channels)
        geom = self._singleray_table_cache.get(key)
        if geom is not None:
            return geom
        tds = [self._interpolators[ch] for ch in channels]
        geom = {
            'td_values': self._table_stack,
            'td_ok': self._table_ok,
            'td_slot': np.array([self._table_channels.index(ch) for ch in channels],
                                dtype=np.int64),
            'td_r_min': np.array([td.r_min for td in tds], dtype=np.float64),
            'td_dr_inv': np.array([td.dr_inv for td in tds], dtype=np.float64),
            'td_nr': np.array([td.nr for td in tds], dtype=np.int64),
            'td_z_min': np.array([td.z_min for td in tds], dtype=np.float64),
            'td_dz_inv': np.array([td.dz_inv for td in tds], dtype=np.float64),
            'td_nz': np.array([td.nz for td in tds], dtype=np.int64),
            'ant_xy': np.array([self.ant_locs[ch][:2] for ch in channels], dtype=np.float64),
            'pa_x': float(self._pa_center[0]),
            'pa_y': float(self._pa_center[1]),
        }
        self._singleray_table_cache[key] = geom
        return geom

    @staticmethod
    def _grid_axes(rho_vec, phi_vec_rad, z_vec):
        """Return the three grid axes as contiguous float64 arrays."""
        return (np.ascontiguousarray(rho_vec, dtype=np.float64),
                np.ascontiguousarray(phi_vec_rad, dtype=np.float64),
                np.ascontiguousarray(z_vec, dtype=np.float64))

    def _channel_tables(self, ch):
        """Every travel-time table loaded for a channel (single, per ray type, solution ordered)."""
        tables = [self._interpolators[ch]] if ch in self._interpolators else []
        tables += list(self._multiray_interpolators.get(ch, {}).values())
        tables += list(self._two_arrival_interpolators.get(ch, {}).values())
        return tables

    def _tables_ice_model(self, config):
        """NuRadioMC ice model of the loaded tables, whose n(z) the far field uses.

        The ``ice_model`` the tables record (the air-ice generator writes it), else the
        config key ``ice_model`` (trusted for tables without metadata, such as the record
        in-ice tables), else greenland_simple, the profile of the record tables.

        Raises:
            ValueError: If the tables record different models, the config names another
                model than the tables, or the name is not a NuRadioMC ice model.
        """
        from NuRadioMC.utilities import medium
        recorded = {table_metadata(path).get('ice_model') for files in self._table_files.values()
                    for path in files if os.path.isfile(path)} - {None}
        key = config.get('ice_model')
        if len(recorded) > 1:
            raise ValueError(f"the tables record different ice models {sorted(recorded)}")
        if recorded and key is not None and key != next(iter(recorded)):
            raise ValueError(f"ice_model {key!r} differs from the ice model {next(iter(recorded))!r} the tables record")
        name = next(iter(recorded)) if recorded else (key or 'greenland_simple')
        if not isinstance(getattr(medium, name, None), type):
            raise ValueError(f"ice_model {name!r} is not a NuRadioMC ice model (NuRadioMC.utilities.medium)")
        return name
