"""
3D interferometric direction reconstruction for RNO-G.

Searches all three cylindrical coordinates (rho, phi, z) simultaneously using
a coarse 3D grid scan followed by L-BFGS-B optimizer refinement from the top-N
coarse peaks. Compute kernels (Numba, CUDA) live in ``NuRadioReco/utilities/reco3d_kernels.py``.
"""

import numpy as np
import numbers
import yaml
import time
import sys
import types
from collections import Counter, OrderedDict

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp

from NuRadioReco.utilities.reco3d_kernels import USE_NUMBA, USE_CUPY, _FUSED_CORR_KERNEL, RAY_TYPES

from NuRadioReco.modules.reco3d.shared import logger, _STACK_BLOCK_POINTS
from NuRadioReco.modules.reco3d.tables import (
    TablesMixin,
    snapshot_search_dirs,
    check_detector_consistency,
)
from NuRadioReco.modules.reco3d.correlation import CorrelationMixin
from NuRadioReco.modules.reco3d.pair_interface import PairInterfaceMixin
from NuRadioReco.modules.reco3d.pair_weights import PairWeightsMixin
from NuRadioReco.modules.reco3d.optimizer import OptimizerMixin
from NuRadioReco.modules.reco3d.hierarchical import HierarchicalMixin
from NuRadioReco.modules.reco3d.far_field import FarFieldMixin, far_field_profile
from NuRadioReco.modules.reco3d.multiray import MultirayMixin
from NuRadioReco.modules.reco3d.grouped import GroupedMixin
from NuRadioReco.modules.reco3d.outputs import OutputsMixin
from NuRadioReco.modules.reco3d.tdoa import TdoaMixin
from NuRadioReco.modules.reco3d.gpu import GpuMixin
from NuRadioReco.modules.reco3d.candidates import CandidatesMixin

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import (
        _scalar_singleray_corr_numba,
        _bilinear_scalar_numba,
        _singleray_grid_numba,
        _bilinear_ok_batch_numba,
        _singleray_stack_corr_numba,
        _map_snr_numba,
        _top_peaks_numba,
        _singleray_stackT_corr_numba,
        _singleray_grid_tts_numba,
    )

# Names kept importable from this module for existing users.
import datetime  # noqa: F401
import hashlib  # noqa: F401
import itertools  # noqa: F401
import os  # noqa: F401
import logging  # noqa: F401
from collections import namedtuple  # noqa: F401
from functools import lru_cache  # noqa: F401

from scipy import fft as sp_fft  # noqa: F401
from scipy.constants import c as _SPEED_OF_LIGHT  # noqa: F401
from scipy.signal import hilbert, windows  # noqa: F401
from scipy.interpolate import RegularGridInterpolator  # noqa: F401
from scipy.optimize import minimize  # noqa: F401

from NuRadioReco.modules import reco3d_batch  # noqa: F401
from NuRadioReco.utilities.reco3d_kernels import (  # noqa: F401
    USE_NUMBA_GROUPED,
    _build_z_vec,
    _build_split_z_vec,
    _build_split_z_window,
    SOLUTION_TYPES,
)

from NuRadioReco.modules.reco3d.shared import (  # noqa: F401
    _CANDIDATE_ORIGIN_CODES,
    _C_M_PER_NS,
    _LBFGSB_ABS_STEP,
    _LBFGSB_REL_STEP,
)
from NuRadioReco.modules.reco3d.tables import (  # noqa: F401
    TableData,
    DETECTOR_Z_TOLERANCE_M,
    _TABLE_METADATA_KEYS,
    _SNAPSHOT_DETECTORS,
    table_files_from_config,
    table_metadata,
    detector_delay_vector,
    _snapshot_detector,
)
from NuRadioReco.modules.reco3d.correlation import (  # noqa: F401
    CorrPacked,
    SERIES_MODES,
    _pair_indices,
    TTStack,
    PairSeriesBatch,
)
from NuRadioReco.modules.reco3d.pair_interface import PairSet  # noqa: F401
from NuRadioReco.modules.reco3d.optimizer import (  # noqa: F401
    _lbfgsb_fd_steps,
    _LbfgsbResult,
    _minimize_lbfgsb,
)
from NuRadioReco.modules.reco3d.hierarchical import (  # noqa: F401
    _CANDIDATE_POOL_SAVE,
    _ADAPTIVE_WINDOW_CELL_FRACTION,
    _adjacent_cell_span,
)
from NuRadioReco.modules.reco3d.far_field import (  # noqa: F401
    _FAR_COARSE_STEP_DEG,
    _FAR_N_SEEDS,
    _FAR_SEED_SEPARATION_DEG,
    _FAR_REFINE_HALF_DEG,
    _FAR_REFINE_STEP_DEG,
    _FAR_QUADRATURE_NODES,
    _FAR_WINDOW_ZENITH_SAMPLES,
    _FAR_GUARD_HALF_DEG,
    _FAR_MEMO_SIZE,
    _plane_wave_profile,
    _plane_wave_layers,
    plane_wave_times,
    _plane_wave_arrays,
    _surface_index,
)
from NuRadioReco.modules.reco3d.candidates import (  # noqa: F401
    _CANDIDATE_ENVELOPE_MODES,
    _POLISH_OBJECTIVES,
    _TWO_ARRIVAL_WEIGHT_MODES,
    _MAX_CORR_SOURCES,
)

try:
    from NuRadioReco.utilities.reco3d_kernels import _FUSED_MULTIRAY_CORR_KERNEL  # noqa: F401
except ImportError:
    _FUSED_MULTIRAY_CORR_KERNEL = None

if USE_NUMBA:
    from NuRadioReco.utilities.reco3d_kernels import (  # noqa: F401
        _scalar_grouped_corr_numba,
        _interp_uniform_numba,
        _scalar_singleray_corr_grad_numba,
        _lbfgsb_singleray_value_grad,
        _lbfgsb_fd_points,
        _grouped_fd_value_grad,
        _all_pairs_corr_numba,
        _bilinear_batch_numba,
        _fused_multiray_grid_numba,
        _pairs_corr_numba,
        _plane_wave_corr_grad_numba,
        _far_fd_points_numba,
        _plane_wave_times_numba,
        _multiray_point_tts_numba,
        _compass_search_numba,
        _two_arrival_point_numba,
        _fused_two_arrival_grid_numba,
        _numpy_std,
        get_num_threads,
    )

if USE_CUPY:
    import cupy as cp  # noqa: F401

if USE_NUMBA_GROUPED:
    from NuRadioReco.utilities.reco3d_kernels import (  # noqa: F401
        grouped_multiray_numba,
        perpair_multiray_numba,
    )
    if USE_NUMBA:
        from NuRadioReco.utilities.reco3d_kernels import grouped_multiray_points  # noqa: F401

RECO_VERSION = '1.1.0'

_REFINE_WINDOW_MODES = ('fixed', 'adaptive')
_OPTIMIZER_GRADIENTS = ('finite_difference', 'exact')


class InterferometricReco3D(
        TablesMixin, CorrelationMixin, PairInterfaceMixin, PairWeightsMixin, OptimizerMixin,
        HierarchicalMixin, FarFieldMixin, MultirayMixin, GroupedMixin, OutputsMixin, TdoaMixin,
        GpuMixin, CandidatesMixin):
    """3D interferometric reconstruction: coarse grid + L-BFGS-B refinement."""

    def __init__(self):
        """Initialize with empty caches and default settings."""
        self._delay_matrix_cache = {}
        self._gpu_delay_stack_cache = {}
        self._tt_stack_cache = {}
        self._batch_grid_cache = {}
        self._opt_geom_cache = {}
        self._packed_multiray_tables = {}
        self._table_mask_cache = {}
        self._tolerant_table_edge = False
        self._valid_norm = False
        self._valid_floor = 0.6
        self._interpolators = {}
        self._multiray_interpolators = {}
        self._table_files = {}
        self.detector_report = None
        self.multiray_backend = None
        self._two_arrival_interpolators = {}
        self._two_arrival_ice = None
        self._two_arrival_packed = {}
        self.ant_locs = None
        self._multi_ray_types = False
        self._multiray_combo_mode = 'per_pair'
        self._active_ray_types = RAY_TYPES
        self._n_ray_slots = len(RAY_TYPES)
        self._use_gpu = False
        # Grids smaller than this fall back to CPU path when GPU is active,
        # since kernel-launch overhead exceeds compute time for tiny grids.
        self._gpu_min_grid_cells = 10000
        # Use the fused all-pairs Numba kernel by default. Can be disabled
        # for debugging via config 'use_fused_correlator: false'.
        self._use_fused_correlator = True
        self._station_id = None
        self._lag_window_cache = {}
        self._far_field_ice = None
        self.work = Counter()
        self._mr_walk = np.zeros(2, dtype=np.int64)
        self._far_field_memo = OrderedDict()

    def work_counts(self):
        """Work counters of the searches since the last ``reset_work`` (counting changes no result).

        Keys (each summed over the searches; a stage absent from a search adds nothing):
        searches and pairs (pairs of each searched channel group); coarse_maps and coarse_points (each
        coarse correlation map computed, one per candidate chain, and its grid points); refine_grids and
        refine_points (the refine-level grids around the peaks); polish_grids and polish_points (the polish
        grids around the candidates); map_points (points of every correlation map, a single-ray map of K
        series counted K times, coarse, refine, polish and region grids together); tt_lookups (channel-point
        travel-time table lookups: each grid lookup, a coarse stack once per cached grid, and every
        optimizer or single-point evaluation; times the ray types for multi-ray); optimizer_runs,
        optimizer_nit, optimizer_nfev (L-BFGS-B runs, iterations and calls of the value-and-gradient
        function) and optimizer_points (objective evaluations inside the optimizer: with forward
        differences the point and one step per coordinate per call); point_evals (single-point objective
        evaluations outside the compiled optimizer); far_searches, far_sky_points (coarse and refine sky
        map directions), far_optimizer_runs, far_optimizer_nit, far_optimizer_nfev (the far-field
        hypothesis) and far_shared (far-field hypotheses taken from an earlier search with the same inputs); multi-ray: mr_pair_terms (pair ray-type terms of the per-pair and grouped maps, a
        pair within one depth group taking one ray type per point), mr_walk_nodes and mr_walk_pruned (nodes
        of the depth-first ray-type walk of the CPU grouped objective evaluated, and branches it left by
        the bound; the GPU grouped maps walk every node and count in ``GpuMultiray.timing``).

        Returns:
            Dict of int.
        """
        out = {k: int(v) for k, v in self.work.items()}
        if self._mr_walk.any():
            out['mr_walk_nodes'], out['mr_walk_pruned'] = (int(v) for v in self._mr_walk)
        return out

    def reset_work(self):
        """Zero the work counters."""
        self.work.clear()
        self._mr_walk[:] = 0

    def _set_station_parameters(self, station, rho, phi, z, corr):
        """Set reconstruction parameters on station, if supported.

        Args:
            station: Station object.
            rho, phi, z, corr: Reconstructed position and correlation.
        """
        if not hasattr(station, 'set_parameter'):
            return
        station.set_parameter(stnp.rec_max_correlation, corr)
        station.set_parameter(stnp.rec_coord_0, rho * units.m)
        station.set_parameter(stnp.rec_coord_1, phi * units.deg)
        station.set_parameter(stnp.rec_coord_2, z * units.m)

    _KNOWN_CONFIG_KEYS = {
        'time_delay_tables', 'station_id', 'channels', 'limits', 'step_sizes',
        'coord_system', 'rec_type', 'fixed_coord',
        'coarse_limits', 'coarse_step_sizes', 'coarse_n_rho', 'coarse_n_z',
        'coarse_n_peaks', 'coarse_peak_separation',
        'n_z', 'z_spacing', 'z_surface_offset', 'z_grid_below', 'z_grid_above',
        'refine_step_sizes', 'refine_window', 'refine_n_peaks', 'refine_radius',
        'refine_levels',
        'n_refinements', 'refinement_factor', 'refinement_window_bins',
        'refinement_convergence_db', 'n_refinements_max', 'rho_spacing',
        'pass2_step_sizes', 'pass2_coarse_step_sizes', 'pass2_n_rho',
        'pass2_n_z', 'pass2_coarse_n_z',
        'pass2_window', 'pass2_hierarchical',
        'pass2_coarse_n_peaks', 'pass2_coarse_peak_separation',
        'pass2_refine_window', 'z_profile_step',
        'hilbert_envelope_mode', 'use_hilbert_envelope',
        'apply_hann_window', 'correlation_normalization', 'interp_method',
        'apply_upsampling', 'apply_cw_removal', 'apply_cable_delays',
        'apply_bandpass', 'apply_cable_delay', 'apply_hw_phase_removal',
        'apply_dedispersion',
        'bandpass_band', 'bandpass_order', 'bandpass_filter_type',
        'cw_peak_prominence', 'cw_freq_band',
        'peak_separation_threshold',
        'helper_snr_threshold', 'surf_corr_z_max', 'surf_corr_zen_max',
        'mode', 'hierarchical', 'tdoa_mode',
        'multi_ray_types', 'multiray_combo_mode',
        'multiray_table_name_pattern', 'table_name_pattern', 'table_scheme',
        'allow_above_surface',
        'optimizer_method', 'optimizer_maxiter', 'n_optimizer_seeds', 'optimizer_gradient',
        'optimizer_rho_offsets',
        'skip_optimizer', 'use_tdoa_seed',
        'snr_pair_weighting', 'pair_weights',
        'save_results_to', 'detector_file', 'detector_date',
        'interpolation_method', 'table_type',
        'n_peaks_save', 'save_coherent_waveforms', 'n_coherent_waveforms',
        'polarization_groups', 'hpol_weight_scale',
        'validation', 'use_gpu', 'gpu_min_grid_cells', 'use_fused_correlator',
        'warmup_numba', 'warmup_gpu', 'primary_polarization',
        'post_optimizer_mode', 'rho_scan_step',
        'refinement_envelope_mode', 'refinement_window', 'refinement_maxiter',
        'de_window', 'de_maxiter', 'de_popsize',
        'bh_window', 'bh_niter', 'bh_stepsize',
        'plane_wave_fallback', 'plane_wave_snr_threshold',
        'candidate_search', 'candidate_envelope_mode',
        'candidate_polish_window', 'candidate_polish_steps',
        'candidate_include_refined', 'tolerant_table_edge',
        'compass_step', 'compass_step_min', 'compass_max_evals', 'compass_phi_scan',
        'objective_normalisation', 'valid_weight_floor',
        'snr_window_ns', 'helper_snr_threshold_windowed',
        'pair_weight_mode', 'pair_weight_k_ns', 'pair_weight_floor_ns',
        'hpol_sign_mode', 'pair_signs',
        'candidate_fill_saved_peaks', 'candidate_diagnostics',
        'refine_window_mode', 'subbin_coarse_seeds',
        'candidate_tie_band',
        'polish_objective', 'two_arrival_weight_mode',
        'two_arrival_second_weight', 'two_arrival_margin', 'max_corr_source',
        'candidate_tie_band_max_raw_corr', 'preprocessor',
        'pass2_volume', 'rx_arrival_mode', 'cross_type_sign_mode',
        'save_pair_store', 'pair_store_margin_ns', 'pair_store_dtype',
        'region_hypotheses', 'far_field_hypothesis', 'far_field_lobe_guard_ns',
        'channel_position_shift', 'ice_model',
    }

    def begin(self, station_id, config, det):
        """Initialize interpolators and antenna positions.

        Parameters
        ----------
        station_id : int
            Station ID.
        config : dict or str
            Configuration dictionary or path to YAML file.
        det : Detector
            Detector description object.
        """
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.safe_load(f)

        self._validate_config(config)
        self._station_id = station_id
        self._preload_tables(station_id, config)
        self._ice_model = self._tables_ice_model(config)
        self._far_field_ice = None
        if config.get('far_field_hypothesis', False) and config.get('optimizer_gradient') == 'exact':
            from NuRadioMC.utilities import medium
            if np.ndim(far_field_profile(medium.get_ice_model(self._ice_model))[0]) != 0:
                raise ValueError(f"optimizer_gradient 'exact' covers the far field of a single-exponential ice "
                                 f"model only, not the layered {self._ice_model!r}; use 'finite_difference'")
        self.ant_locs = self._get_ant_locs(station_id, det)
        self._ant_locs_db = {ch: pos.copy() for ch, pos in self.ant_locs.items()}
        self._position_key = ()
        self._position_shift_key(config.get('channel_position_shift'))
        self.detector_report = check_detector_consistency(
            det, station_id, config['channels'], self._table_files,
            snapshot_dirs=snapshot_search_dirs(config))
        if 1 in self.ant_locs and 2 in self.ant_locs:
            self._pa_center = (self.ant_locs[1] + self.ant_locs[2]) / 2.0
        else:
            logger.warning(
                "Channels 1 and 2 not in ant_locs (available: %s). "
                "PA center defaulting to origin.", sorted(self.ant_locs.keys()))
            self._pa_center = np.zeros(3)

        want_gpu = bool(config.get('use_gpu', False))
        if want_gpu and not USE_CUPY:
            logger.warning(
                "use_gpu=True requested but CuPy or CUDA device is not "
                "available. Falling back to CPU path.")
            self._use_gpu = False
        else:
            self._use_gpu = want_gpu
        self._gpu_min_grid_cells = int(
            config.get('gpu_min_grid_cells', self._gpu_min_grid_cells))
        self._use_fused_correlator = bool(
            config.get('use_fused_correlator', self._use_fused_correlator))
        tolerant_edge = bool(config.get('tolerant_table_edge', False))
        if tolerant_edge != self._tolerant_table_edge:
            self.end()
        self._tolerant_table_edge = tolerant_edge
        self._valid_norm = config.get('objective_normalisation', 'total') == 'valid'
        self._valid_floor = float(config.get('valid_weight_floor', 0.6))
        if self._use_gpu:
            logger.info(
                "InterferometricReco3D running on GPU (CuPy). "
                "Grids < %d cells fall back to CPU.",
                self._gpu_min_grid_cells)

        # Trigger Numba JIT compilation on tiny dummy data so the first
        # real event doesn't pay the compile cost. Kernels with
        # ``cache=True`` only compile once per (type signature, host)
        # anyway; this just moves the work into begin() rather than
        # the first run() call.
        if USE_NUMBA and config.get('warmup_numba', True):
            try:
                self._warmup_numba_kernels()
            except Exception as exc:
                logger.debug("Numba warmup skipped: %s", exc)

        # Same idea for the CuPy RawKernel: compile on dummy data so the
        # first real reco doesn't pay CUDA nvrtc compile cost.
        if self._use_gpu and USE_CUPY and _FUSED_CORR_KERNEL is not None \
                and config.get('warmup_gpu', True):
            try:
                self._warmup_gpu_kernels()
            except Exception as exc:
                logger.debug("GPU warmup skipped: %s", exc)

    def _warmup_numba_kernels(self):
        """Compile the Numba kernels of the singleray CPU path with tiny dummy inputs.

        Eliminates first-event JIT compile latency. The compiled kernels
        are cached (cache=True on the decorators) so the cost is paid
        once per host across runs.
        """
        values = np.ones((4, 4), dtype=np.float64)
        _bilinear_scalar_numba(values, 0.0, 1.0, 4, 0.0, 1.0, 4, 1.5, 1.5)

        n_pairs = 3
        corr_packed = np.ones((n_pairs, 16), dtype=np.float64)
        corr_lens = np.full(n_pairs, 16, dtype=np.int64)
        dts = np.ones(n_pairs, dtype=np.float64)
        offsets = np.zeros(n_pairs, dtype=np.float64)
        weights = np.ones(n_pairs, dtype=np.float64)

        n_ch = 3
        ant_xy = np.zeros((n_ch, 2), dtype=np.float64)
        td_values = np.ones((n_ch, 4, 4), dtype=np.float64)
        td_ok = np.ones((n_ch, 4, 4), dtype=np.bool_)
        td_r_min = np.zeros(n_ch, dtype=np.float64)
        td_dr_inv = np.ones(n_ch, dtype=np.float64)
        td_nr = np.full(n_ch, 4, dtype=np.int64)
        td_z_min = np.zeros(n_ch, dtype=np.float64)
        td_dz_inv = np.ones(n_ch, dtype=np.float64)
        td_nz = np.full(n_ch, 4, dtype=np.int64)
        pair_ch1 = np.array([0, 0, 1], dtype=np.int64)
        pair_ch2 = np.array([1, 2, 2], dtype=np.int64)
        td_slot = np.arange(n_ch, dtype=np.int64)
        _scalar_singleray_corr_numba(
            1.0, 0.0, -1.0, 0.0, 0.0, ant_xy,
            td_values, td_ok, td_slot, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz,
            corr_packed, corr_lens, dts, offsets,
            pair_ch1, pair_ch2, weights, 3.0, False, 0.6)

        axis = np.array([1.0, 1.5], dtype=np.float64)
        geom = (0.0, 0.0, ant_xy, td_values, td_ok, td_slot, td_r_min, td_dr_inv,
                td_nr, td_z_min, td_dz_inv, td_nz, False)
        corr_args = (corr_packed[None], corr_lens, 1.0 / dts, offsets,
                     pair_ch1, pair_ch2, weights, 3.0, False, 0.6)
        _singleray_grid_numba(axis, axis, axis, *geom, *corr_args)
        tt, ok = _bilinear_ok_batch_numba(td_values, td_ok, 0, 0.0, 1.0, 4, 0.0, 1.0, 4,
                                          axis, axis, False)
        stack = (np.repeat(tt[:, None], n_ch, axis=1), np.repeat(ok[:, None], n_ch, axis=1))
        _singleray_stack_corr_numba(stack[0], stack[1], *corr_args)
        _singleray_stackT_corr_numba(np.ascontiguousarray(stack[0].T), np.ascontiguousarray(stack[1].T),
                                     _STACK_BLOCK_POINTS, *corr_args)
        _singleray_grid_tts_numba(axis, axis, axis, *geom)
        _top_peaks_numba(np.zeros((2, 2, 2)), axis, axis, axis, 1, 1.0, 1.0, 1.0)
        _map_snr_numba(np.ones((2, 2, 2)), 0, 0, 0, 0)

        logger.debug("Numba kernels warmed up")

    def _validate_config(self, config):
        """Check a config for mistakes: warn about unknown keys, raise on invalid values.

        Args:
            config: Reconstruction config dict.

        Raises:
            ValueError: On an invalid candidate search or polish grid
                (``_candidate_chains``, ``_polish_levels``), a non-boolean
                ``candidate_include_refined``, ``candidate_fill_saved_peaks``,
                ``candidate_diagnostics``, ``tolerant_table_edge``,
                ``subbin_coarse_seeds``, ``region_hypotheses`` or
                ``far_field_hypothesis``, an unknown
                ``refine_window_mode``, an
                invalid tie band or noise ceiling (``_candidate_tie_band``,
                ``_candidate_tie_band_max_raw_corr``), two-arrival settings
                (``_two_arrival_settings``), split z grid (``_split_z_grid``), SNR,
                pair-weight or sign keys (``_validate_snr_config``), an unknown
                ``optimizer_method`` or ``optimizer_gradient``, invalid compass settings (``_compass_options``),
                an unknown ``objective_normalisation``, a ``valid_weight_floor``
                outside (0, 1], or the valid normalisation without the fused
                singleray numba kernels.
        """
        unknown = set(config.keys()) - self._KNOWN_CONFIG_KEYS
        if unknown:
            logger.warning(
                "Unknown config keys (ignored): %s. "
                "This module uses cylindrical coordinates (rho, phi, z). "
                "Keys like 'coord_system', 'rec_type', and 'fixed_coord' "
                "have no effect.", sorted(unknown)
            )

        if 'coord_system' in config:
            cs = config['coord_system']
            if cs != 'cylindrical':
                logger.warning(
                    "coord_system='%s' has no effect. This module always "
                    "uses cylindrical coordinates (rho, phi, z). The "
                    "'limits' key is interpreted as "
                    "[rho_min, rho_max, phi_min, phi_max, z_min, z_max]. "
                    "If you intended spherical coordinates, this config "
                    "will produce wrong results.", cs
                )

        if self._candidate_chains(config):
            self._polish_levels(config)
        for key in ('candidate_include_refined', 'candidate_fill_saved_peaks',
                    'candidate_diagnostics', 'tolerant_table_edge',
                    'subbin_coarse_seeds', 'region_hypotheses', 'far_field_hypothesis'):
            if not isinstance(config.get(key, False), bool):
                raise ValueError(f"{key} must be a boolean")
        guard = config.get('far_field_lobe_guard_ns')
        if guard is not None and (isinstance(guard, bool) or not isinstance(guard, (int, float)) or not guard > 0):
            raise ValueError(f"far_field_lobe_guard_ns must be a positive number of ns or null, got {guard!r}")
        if config.get('refine_window_mode', 'fixed') not in _REFINE_WINDOW_MODES:
            raise ValueError(
                f"refine_window_mode must be one of {_REFINE_WINDOW_MODES}, "
                f"got {config.get('refine_window_mode')!r}")
        self._candidate_tie_band(config)
        self._candidate_tie_band_max_raw_corr(config)
        self._two_arrival_settings(config)
        self._split_z_grid(config)
        self._validate_snr_config(config)

        method = config.get('optimizer_method', 'L-BFGS-B')
        if method not in ('L-BFGS-B', 'Nelder-Mead', 'compass'):
            raise ValueError(
                f"optimizer_method must be 'L-BFGS-B', 'Nelder-Mead' or 'compass', got {method!r}")
        self._compass_options(config)
        if config.get('optimizer_gradient', 'finite_difference') not in _OPTIMIZER_GRADIENTS:
            raise ValueError(f"optimizer_gradient must be one of {_OPTIMIZER_GRADIENTS}, "
                             f"got {config.get('optimizer_gradient')!r}")

        norm = config.get('objective_normalisation', 'total')
        if norm not in ('total', 'valid'):
            raise ValueError(
                f"objective_normalisation must be 'total' or 'valid', got {norm!r}")
        floor = config.get('valid_weight_floor', 0.6)
        if not isinstance(floor, numbers.Real) or not 0.0 < floor <= 1.0:
            raise ValueError(f"valid_weight_floor must be in (0, 1], got {floor!r}")
        if norm == 'valid' and (config.get('multi_ray_types', False)
                                or config.get('use_gpu', False)
                                or not config.get('use_fused_correlator', True)
                                or not USE_NUMBA):
            raise ValueError(
                "objective_normalisation: valid needs the fused singleray numba "
                "kernels (no multi_ray_types, use_gpu or use_fused_correlator: false)")

    @classmethod
    def objective_version(cls, config):
        """Name the search objective a configuration evaluates.

        ``record`` is the geometric-mean SNR pair weighting of the signed
        correlations in every group, divided by the total pair weight. Any other
        value means ``max_corr`` and the peak correlations (of the primary group
        under a weight or normalisation change, of the HPol group under a sign
        mode) are on a different scale and names the keys that changed it, so
        consumers of a results file can tell the two apart. The per-event
        result field ``objective_version`` (0 total, 1 valid) records the
        normalisation alone.

        Args:
            config: Reconstruction config dict.

        Returns:
            ``'record'`` or a string naming the objective keys and their values.
        """
        parts = []
        if config.get('pair_weight_mode', 'record') != 'record':
            parts.append(
                f"pair_weight_mode={config['pair_weight_mode']}"
                f"(k_ns={config.get('pair_weight_k_ns', cls.PAIR_WEIGHT_K_NS)},"
                f"floor_ns={config.get('pair_weight_floor_ns', cls.PAIR_WEIGHT_FLOOR_NS)},"
                f"snr_window_ns={config.get('snr_window_ns')})")
        if config.get('hpol_sign_mode', 'signed') != 'signed':
            parts.append(f"hpol_sign_mode={config['hpol_sign_mode']}")
        if config.get('objective_normalisation', 'total') != 'total':
            parts.append(
                f"objective_normalisation={config['objective_normalisation']}"
                f"(valid_weight_floor={config.get('valid_weight_floor', 0.6)})")
        return ';'.join(parts) if parts else 'record'

    @register_run()
    def run(self, evt, station, det, config):
        """Run 3D interferometric reconstruction on one event.

        Performs a coarse 3D grid scan, extracts the top-N peaks, then refines
        each with L-BFGS-B. Sets station parameters with the best result.

        Parameters
        ----------
        evt : Event
            NuRadioReco Event object.
        station : Station
            Station object containing channel data.
        det : Detector
            Detector description.
        config : dict or str
            Configuration dictionary or path to YAML file.

        Returns
        -------
        dict
            Reconstruction results with keys 'rho', 'phi', 'z', 'max_corr'.
        """
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.safe_load(f)
        self._set_position_shift(config.get('channel_position_shift'))

        if config.get('polarization_groups', None) is not None:
            return self._run_per_polarization(
                config, lambda group_config: self.run(evt, station, det, group_config), station)

        if config.get('tdoa_mode', False):
            return self.run_tdoa(evt, station, det, config)

        if config.get('hierarchical', False):
            return self.run_hierarchical(evt, station, det, config)

        station_id = station.get_id()
        channels = config['channels']
        hilbert_mode = config.get('hilbert_envelope_mode', None)
        apply_hann = config.get('apply_hann_window', False)
        corr_norm = config.get('correlation_normalization', 'normalized')
        pair_weights = config.get('pair_weights', None)

        rho_vec, phi_vec, z_vec = self._generate_coord_arrays(config)
        phi_vec_deg = phi_vec * (180.0 / np.pi)

        volt_arrays = []
        time_arrays = []
        for ch in channels:
            channel = station.get_channel(ch)
            volt_arrays.append(channel.get_trace())
            time_arrays.append(channel.get_times())

        if pair_weights is None:
            pair_weights = self._group_pair_weights(
                *self._event_snrs(volt_arrays, time_arrays, channels, config), channels, config)

        corr_data, packed = self._prepare_corr_funcs(
            time_arrays, volt_arrays,
            hilbert_envelope_mode=hilbert_mode,
            apply_hann_window=apply_hann,
            correlation_normalization=corr_norm,
            pair_signs=config.get('pair_signs', None),
        )

        t0 = time.time()
        if self._multi_ray_types:
            src_enu = self._build_source_enu_matrix(rho_vec, phi_vec, z_vec)
            tt_data = self._compute_tt_multiray(src_enu, channels)
            mean_corr, _ = self._multiray_correlate(
                corr_data, tt_data, channels, pair_weights=pair_weights,
            )
        elif self._singleray_kernel_active():
            stack = self._singleray_tt_stack(
                ('flat', station_id, tuple(channels), tuple(config['limits']),
                 tuple(config['step_sizes']), z_vec.tobytes()),
                rho_vec, phi_vec, z_vec, channels)
            mean_corr = self._singleray_stack_maps(
                stack, channels, [packed], pair_weights)[0].reshape(
                len(rho_vec), len(phi_vec), len(z_vec))
        else:
            src_enu = self._build_source_enu_matrix(rho_vec, phi_vec, z_vec)
            delay_matrices = self._get_t_delay_matrices(
                station_id, config, src_enu, channels, z_vec
            )
            mean_corr, _, _ = self._correlator(
                corr_data, delay_matrices, pair_weights=pair_weights,
            )
        t_grid = time.time() - t0

        n_seeds = config.get('n_optimizer_seeds', 3)
        sep = config.get('peak_separation_threshold', [10, 10, 10])
        peaks = self._extract_top_n_peaks(
            mean_corr, rho_vec, phi_vec_deg, z_vec, n_seeds, sep
        )

        if not peaks:
            logger.warning("No peaks found in coarse grid")
            self._set_station_parameters(
                station, np.nan, np.nan, np.nan, np.nan)
            return {'rho': np.nan, 'phi': np.nan, 'z': np.nan,
                    'max_corr': np.nan}

        bounds = [
            (max(config['limits'][0], 1.0), config['limits'][1]),
            (config['limits'][2], config['limits'][3]),
            (config['limits'][4], config['limits'][5]),
        ]

        t0_opt = time.time()
        best = None
        for rho_p, phi_p, z_p, corr_p in peaks:
            rho_opt, phi_opt, z_opt, corr_opt = self._optimize_from_seed(
                (rho_p, phi_p, z_p), corr_data, channels, bounds, pair_weights
            )
            if best is None or corr_opt > best[3]:
                best = (rho_opt, phi_opt, z_opt, corr_opt)
        t_opt = time.time() - t0_opt

        rho_best, phi_best, z_best, corr_best = best

        logger.debug(
            "3D reco: grid=%.3fs, opt=%.3fs, "
            "rho=%.1f phi=%.1f z=%.1f corr=%.4f",
            t_grid, t_opt, rho_best, phi_best, z_best, corr_best
        )

        self._set_station_parameters(
            station, rho_best, phi_best, z_best, corr_best)

        return {
            'rho': rho_best,
            'phi': phi_best,
            'z': z_best,
            'max_corr': corr_best,
            'objective_version': int(self._valid_norm),
            'grid_time': t_grid,
            'opt_time': t_opt,
            'n_peaks_found': len(peaks),
            'coarse_peaks': peaks,
        }

    def clear_grid_caches(self):
        """Empty the travel-time and delay caches keyed by a search grid.

        A search whose limits change from event to event (the local pass 2 of the
        driver's rx and rxtx modes) adds one entry per event to these caches; clearing
        them after each such search bounds the memory without changing any result.
        The table geometry and optimizer caches, keyed by channel set, are kept.
        """
        self._tt_stack_cache.clear()
        self._delay_matrix_cache.clear()
        self._gpu_delay_stack_cache.clear()
        if hasattr(self, '_cpu_delay_T_cache'):
            self._cpu_delay_T_cache.clear()

    def end(self):
        """Empty the caches built from the travel-time tables and the station geometry."""
        self._delay_matrix_cache.clear()
        self._gpu_delay_stack_cache.clear()
        self._tt_stack_cache.clear()
        self._opt_geom_cache.clear()
        self._packed_multiray_tables.clear()
        self._table_mask_cache.clear()
        self._lag_window_cache.clear()
        if hasattr(self, '_cpu_delay_T_cache'):
            self._cpu_delay_T_cache.clear()


class _ClassModule(types.ModuleType):
    """Module type that passes the rebinding of a module-level name on to the mixin files.

    The methods of the class live in the mixin files and read module-level names (``USE_NUMBA``,
    kernels, constants) from their own file. Setting or deleting such a name on this module, as a
    test that patches ``USE_NUMBA`` does, reaches every one of those files.
    """

    def __setattr__(self, name, value):
        """Set a name here and in every mixin file."""
        super().__setattr__(name, value)
        if not name.startswith('__'):
            for module in _MIXIN_MODULES:
                setattr(module, name, value)

    def __delattr__(self, name):
        """Delete a name here and in every mixin file that has it."""
        super().__delattr__(name)
        if not name.startswith('__'):
            for module in _MIXIN_MODULES:
                if name in vars(module):
                    delattr(module, name)


_MIXIN_MODULES = tuple(sys.modules[base.__module__] for base in InterferometricReco3D.__bases__)
sys.modules[__name__].__class__ = _ClassModule
# Staticmethods in the mixin files reach the class by its name: bind it there now that it exists.
sys.modules[__name__].InterferometricReco3D = InterferometricReco3D
