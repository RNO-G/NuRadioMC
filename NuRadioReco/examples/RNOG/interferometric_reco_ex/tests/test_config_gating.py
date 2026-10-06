"""Every configuration key added since the reference code is inert when absent.

The reference key sets are those of the code the golden master was generated on. The
keys that `_KNOWN_CONFIG_KEYS` and `channelPreprocessor._DEFAULT_CONFIG` have gained
since then must each be registered here with the value that reproduces the absent
behaviour, and setting a key to that value must leave the reconstruction and the
preprocessed traces bit-identical. The golden master (`test_golden_master.py`) is the
proof that the absent configuration itself is unchanged. The split z grid keys have no
such value (given, they replace `coarse_n_z` and the coarse z grid), so they are listed
in MODULE_KEYS_WITHOUT_INERT_VALUE instead; `test_split_z_grid.py` checks that the
reference configuration carries neither and that a split grid without an air block
repeats the default search. The `preprocessor` block is read by the driver and never by
the module; it is registered so that begin() does not warn about it, and listed in
DRIVER_OWNED_KEYS. The driver's pass-2 options (``pass2_volume``, ``rx_arrival_mode``,
``cross_type_sign_mode``) are registered for the same reason with the values that keep the
previous driver behaviour in DRIVER_INERT_VALUES; the module ignores them, and
test_rx_air.py checks the driver side. No shipped config carries a key the module does not know.
"""

import datetime
import glob
import os

import numpy as np
import pytest
import yaml

from conftest import STATION, rng_sources
from synthetic import RECORD_PREPROCESSOR, VPOL_CHANNELS, cylindrical_to_enu, filtered_trace, make_event
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.RNO_G.channelPreprocessor import channelPreprocessor
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.utilities import units
from reco_config import misplaced_preprocessor_keys

MODULE_KEYS_REFERENCE = frozenset({
    'time_delay_tables', 'station_id', 'channels', 'limits', 'step_sizes',
    'coord_system', 'rec_type', 'fixed_coord',
    'coarse_limits', 'coarse_step_sizes', 'coarse_n_rho', 'coarse_n_z',
    'coarse_n_peaks', 'coarse_peak_separation',
    'n_z', 'z_spacing', 'z_surface_offset',
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
    'optimizer_method', 'optimizer_maxiter', 'n_optimizer_seeds',
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
})
PREPROCESSOR_KEYS_REFERENCE = frozenset({
    'apply_block_offset_removal', 'apply_glitch_detection', 'apply_cable_delay', 'cable_delay_mode',
    'apply_hw_phase_removal', 'hw_phase_mode', 'hw_phase_sim_to_data', 'apply_upsampling',
    'target_sampling_rate', 'apply_cw_removal', 'cw_peak_prominence', 'cw_freq_band', 'cw_algorithm',
    'apply_bandpass', 'bandpass_band', 'bandpass_filter_type', 'bandpass_order', 'glitch_cut_value',
})
MODULE_INERT_VALUES = {
    'candidate_search': [],
    'candidate_envelope_mode': 'traces',
    'candidate_polish_window': [3.0, 1.0, 3.0],
    'candidate_polish_steps': [0.5, 0.1, 0.5],
    'candidate_include_refined': False,
    'tolerant_table_edge': False,
    'compass_step': [1.0, 0.2, 1.0],
    'compass_step_min': [1e-8, 2e-9, 1e-8],
    'compass_max_evals': 1500,
    'compass_phi_scan': False,
    'objective_normalisation': 'total',
    'valid_weight_floor': 0.6,
    'snr_window_ns': None,
    'helper_snr_threshold_windowed': 19.5,
    'pair_weight_mode': 'record',
    'pair_weight_k_ns': 12.65,
    'pair_weight_floor_ns': 2.0,
    'hpol_sign_mode': 'signed',
    'pair_signs': None,
    'candidate_fill_saved_peaks': False,
    'candidate_diagnostics': False,
    'refine_window_mode': 'fixed',
    'subbin_coarse_seeds': False,
    'candidate_tie_band': None,
    'candidate_tie_band_max_raw_corr': None,
    'polish_objective': 'raw',
    'two_arrival_weight_mode': 'mask',
    'two_arrival_second_weight': 1.0,
    'two_arrival_margin': None,
    'max_corr_source': 'raw',
    'region_hypotheses': False,
    'far_field_hypothesis': False,
    'far_field_lobe_guard_ns': None,
    'channel_position_shift': None,
    'optimizer_gradient': 'finite_difference',
    'ice_model': None,
}
MODULE_KEYS_WITHOUT_INERT_VALUE = frozenset({'z_grid_below', 'z_grid_above'})
DRIVER_OWNED_KEYS = frozenset({'preprocessor'})
DRIVER_INERT_VALUES = {
    'pass2_volume': 'template',
    'rx_arrival_mode': 'direct',
    'cross_type_sign_mode': 'signed',
    'save_pair_store': False,
    'pair_store_margin_ns': 20.0,
    'pair_store_dtype': 'float32',
}
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
PREPROCESSOR_INERT_VALUES = {
    'apply_delay_corrections': False,
    'delay_corrections_file': None,
    'channels': None,
    'apply_notch': False,
    'notch_bands': ((0.399 * units.GHz, 0.407 * units.GHz),),
}
COMPARED_KEYS = ['rho', 'phi', 'z', 'max_corr'] + [f'peak_{i}_{f}' for i in range(3) for f in ('rho', 'phi', 'z', 'corr', 'map_snr')]


def test_new_module_keys_are_registered():
    """Every key added to _KNOWN_CONFIG_KEYS since the golden-master reference code has a registered inert value or is listed without one."""
    new = set(InterferometricReco3D._KNOWN_CONFIG_KEYS) - MODULE_KEYS_REFERENCE
    registered = (set(MODULE_INERT_VALUES) | MODULE_KEYS_WITHOUT_INERT_VALUE | DRIVER_OWNED_KEYS
                  | set(DRIVER_INERT_VALUES))
    assert new == registered, (sorted(new), sorted(registered))
    assert not MODULE_KEYS_WITHOUT_INERT_VALUE & set(MODULE_INERT_VALUES)
    assert not DRIVER_OWNED_KEYS & (set(MODULE_INERT_VALUES) | MODULE_KEYS_WITHOUT_INERT_VALUE)
    assert not set(DRIVER_INERT_VALUES) & (set(MODULE_INERT_VALUES) | MODULE_KEYS_WITHOUT_INERT_VALUE
                                           | DRIVER_OWNED_KEYS)


def test_shipped_configs_have_no_unknown_keys():
    """Every config in configs/ uses only keys the module knows, so begin() logs no unknown-key warning."""
    paths = sorted(glob.glob(os.path.join(CONFIG_DIR, '*.yaml')))
    assert paths
    for path in paths:
        with open(path) as f:
            config = yaml.safe_load(f)
        unknown = set(config) - InterferometricReco3D._KNOWN_CONFIG_KEYS
        assert not unknown, (os.path.basename(path), sorted(unknown))


def test_shipped_configs_keep_preprocessing_keys_in_the_block():
    """No shipped config puts a preprocessing key at the top level, where the driver ignores it."""
    paths = sorted(glob.glob(os.path.join(CONFIG_DIR, '*.yaml')))
    assert paths
    for path in paths:
        with open(path) as f:
            config = yaml.safe_load(f)
        assert not misplaced_preprocessor_keys(config), (os.path.basename(path), misplaced_preprocessor_keys(config))
    assert misplaced_preprocessor_keys({'apply_hw_phase_removal': True, 'apply_upsampling': True,
                                        'preprocessor': {'apply_bandpass': True}}) == ['apply_hw_phase_removal']


def test_new_preprocessor_keys_are_registered():
    """Every key added to channelPreprocessor._DEFAULT_CONFIG since the golden-master reference code has a registered inert value."""
    new = set(channelPreprocessor._DEFAULT_CONFIG) - PREPROCESSOR_KEYS_REFERENCE
    assert new == set(PREPROCESSOR_INERT_VALUES), (sorted(new), sorted(PREPROCESSOR_INERT_VALUES))
    for key, value in PREPROCESSOR_INERT_VALUES.items():
        assert channelPreprocessor._DEFAULT_CONFIG[key] == value, key


@pytest.mark.slow
def test_each_new_module_key_at_its_inert_value_is_bit_identical(reco, det, base_config, tables, pa):
    """Setting a new key to its inert value reproduces the absent configuration exactly."""
    src = rng_sources(1, 20260930)[0]
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables, snr=20.0, seed=7)
    reference = reco.run(evt, stn, det, base_config)
    inert = dict(MODULE_INERT_VALUES, **DRIVER_INERT_VALUES)
    for key, value in inert.items():
        res = reco.run(evt, stn, det, dict(base_config, **{key: value}))
        for k in COMPARED_KEYS:
            assert res[k] == reference[k], (key, k, reference[k], res[k])
    res = reco.run(evt, stn, det, dict(base_config, **inert))
    assert all(res[k] == reference[k] for k in COMPARED_KEYS)


def _native_event(seed):
    """A native-rate event on the VPol channels with a station time, as the preprocessor sees it."""
    rng = np.random.default_rng(seed)
    evt = Event(0, seed)
    stn = Station(STATION)
    stn.set_station_time(datetime.datetime(2022, 10, 1))
    for ch in VPOL_CHANNELS:
        c = Channel(ch)
        c.set_trace(filtered_trace(200.0 + 3.0 * ch, 1.0, 0.3, rng), 3.2 * units.GHz, trace_start_time=2.0 * ch)
        stn.add_channel(c)
    evt.set_station(stn)
    return evt, stn


def _preprocessed(det, config):
    """Traces and start times after the record preprocessor chain with the given overrides."""
    evt, stn = _native_event(1)
    pre = channelPreprocessor()
    pre.begin(dict(RECORD_PREPROCESSOR, apply_upsampling=False, **config))
    pre.run(evt, stn, det)
    return {ch: (stn.get_channel(ch).get_trace(), stn.get_channel(ch).get_trace_start_time()) for ch in VPOL_CHANNELS}


@pytest.mark.slow
def test_each_new_preprocessor_key_at_its_inert_value_is_bit_identical(det):
    """The record preprocessor chain is unchanged by the new keys at their inert values."""
    reference = _preprocessed(det, {})
    for key, value in PREPROCESSOR_INERT_VALUES.items():
        out = _preprocessed(det, {key: value})
        for ch in VPOL_CHANNELS:
            assert np.array_equal(out[ch][0], reference[ch][0]) and out[ch][1] == reference[ch][1], (key, ch)
    out = _preprocessed(det, PREPROCESSOR_INERT_VALUES)
    assert all(np.array_equal(out[ch][0], reference[ch][0]) for ch in VPOL_CHANNELS)
