"""Output contract of the reconstruction: keys, dtypes and definitions consumed downstream.

Under the reference configuration the module returns exactly the contract keys (any
other key must carry a `_v<N>` suffix), the definitions that downstream code relies on
hold (peak 0 is the primary result, the SNR summaries derive from the per-channel SNRs,
the isolation ratio from the coarse peaks, `n_channels_above` counts the PA and helper
channels), and the results file written by the driver's writer carries the contract
datasets with their dtypes and the provenance attributes, including `reco_version`; a
peak slot the search did not fill is NaN in the file; with `save_coherent_waveforms` the
file carries the `coherent_waveforms` group with its `times` and `peak_<i>` datasets.
"""

import re

import h5py
import numpy as np
import pytest

from conftest import STATION, reference_config
from reco_output import (COHERENT_GROUP, FILE_ATTRS, IDENTITY_KEYS, PEAK_FIELDS, VALIDATION_COUNT_KEYS,
                         coherent_keys, contract_keys, is_versioned_key, peak_keys, write_results_h5)
from reco_validation import HELPER_CHANNELS, PA_CHANNELS
from synthetic import HPOL_CHANNELS, VPOL_CHANNELS, TravelTimeTables, cylindrical_to_enu, make_event
from NuRadioReco.modules.interferometricDirectionReconstruction3D import RECO_VERSION, InterferometricReco3D

SOURCE = (80.0, 120.0, -40.0)
VERSION_RE = re.compile(r'^\d+\.\d+\.\d+$')
FLOAT_KEYS = ['rho', 'phi', 'z', 'max_corr', 'surf_corr_z', 'surf_corr_zen', 'peak_isolation_ratio']
INT_KEYS = ['n_coarse_peaks', 'n_refined_peaks', 'n_saved_peaks', 'n_helpers_above', 'n_channels_above']


def _result(reco, det, config, tables, pa, seed):
    """Reconstruct one synthetic event with validation on."""
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*SOURCE, pa), config['channels'], tables,
                             snr=30.0, seed=seed)
    return reco.run(evt, stn, det, config)


def test_versioned_key_rule():
    """New quantities are recognised by their version suffix."""
    assert is_versioned_key('closure_rms_ns_v1') and is_versioned_key('max_corr_tau_v12')
    assert not is_versioned_key('max_corr') and not is_versioned_key('peak_0_corr_v')


@pytest.mark.slow
def test_module_keys_and_definitions(reco, det, base_config, tables, pa):
    """The reference configuration yields the contract keys with their documented meaning."""
    config = dict(base_config, validation=True)
    res = _result(reco, det, config, tables, pa, seed=11)
    expected = set(contract_keys(VPOL_CHANNELS, 3, True)) - set(IDENTITY_KEYS) | {'coarse_peaks'}
    missing = expected - set(res)
    assert not missing, missing
    extra = set(res) - expected
    assert all(is_versioned_key(k) for k in extra), sorted(extra)
    for k in FLOAT_KEYS:
        assert isinstance(res[k], (float, np.floating)), k
    for k in INT_KEYS:
        assert isinstance(res[k], (int, np.integer)) and not isinstance(res[k], bool), k
    assert isinstance(res['has_helper_signal'], (bool, np.bool_))
    assert 0.0 <= res['phi'] < 360.0 and -100.0 <= res['z'] <= 0.0 and res['rho'] >= 1.0
    assert res['peak_0_rho'] == res['rho'] and res['peak_0_phi'] == res['phi']
    assert res['peak_0_z'] == res['z'] and res['peak_0_corr'] == res['max_corr']
    assert res['n_saved_peaks'] == 3 and res['n_coarse_peaks'] == len(res['coarse_peaks'])
    corrs = [res[f'peak_{i}_corr'] for i in range(3)]
    assert corrs == sorted(corrs, reverse=True)
    pa_snrs = [res[f'ch{ch}_snr'] for ch in PA_CHANNELS]
    assert res['pa_max_snr'] == max(pa_snrs) and abs(res['pa_avg_snr'] - np.mean(pa_snrs)) < 1e-12
    assert res['helper_b_max_snr'] == max(res['ch9_snr'], res['ch10_snr'])
    assert res['helper_c_min_snr'] == min(res['ch22_snr'], res['ch23_snr'])
    assert res['n_helpers_above'] == sum(res[f'ch{ch}_snr'] > 5.0 for ch in HELPER_CHANNELS)
    assert res['n_channels_above'] == sum(res[f'ch{ch}_snr'] > 5.0 for ch in PA_CHANNELS + HELPER_CHANNELS)
    assert res['has_helper_signal'] == (res['n_helpers_above'] > 0)
    top = sorted([p[3] for p in res['coarse_peaks']], reverse=True)
    assert abs(res['peak_isolation_ratio'] - top[0] / np.mean(top[:5])) < 1e-12
    assert res['surf_corr_z'] <= 1.0 and res['surf_corr_zen'] <= 1.0


@pytest.mark.slow
def test_polarization_groups_suffix_every_key(det, table_dir, pa):
    """With polarization groups every module key appears with the `_vpol` and `_hpol` suffixes.

    Per-channel SNR keys carry a group's suffix for that group's channels only, and peak
    slots for the peaks the group's search filled.
    """
    channels = VPOL_CHANNELS + HPOL_CHANNELS
    cfg = reference_config(table_dir, channels=channels, validation=True,
                           polarization_groups={'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS})
    reco = InterferometricReco3D()
    reco.begin(STATION, cfg, det)
    all_tables = TravelTimeTables(table_dir, STATION, channels)
    res = _result(reco, det, cfg, all_tables, pa, seed=3)
    bare = set(contract_keys(VPOL_CHANNELS, 3, True)) - set(IDENTITY_KEYS) | {'coarse_peaks'}
    per_channel = {k for k in bare if k.startswith('ch') and k.endswith('_snr')}
    for k in bare - per_channel - set(peak_keys(3)):
        assert f'{k}_vpol' in res and f'{k}_hpol' in res, k
    for name, group in (('vpol', VPOL_CHANNELS), ('hpol', HPOL_CHANNELS)):
        assert all(f'ch{ch}_snr_{name}' in res for ch in group), name
        assert not any(f'ch{ch}_snr_{name}' in res for ch in channels if ch not in group), name
        for i in range(res[f'n_saved_peaks_{name}']):
            assert all(f'peak_{i}_{f}_{name}' in res for f in PEAK_FIELDS), (name, i)
    assert res['rho'] == res['rho_vpol'] and res['max_corr'] == res['max_corr_vpol']
    assert all(f'ch{ch}_snr' in res for ch in VPOL_CHANNELS) and 'ch4_snr' not in res


@pytest.mark.slow
def test_results_file_contract(reco, det, base_config, tables, pa, tmp_path):
    """The writer produces the contract datasets, dtypes and attributes."""
    config = dict(base_config, validation=True)
    results = []
    for i, seed in enumerate((21, 22)):
        res = _result(reco, det, config, tables, pa, seed)
        res.update(run_number=100 + i, event_number=i, source_file='fixture.nur', preproc_time=0.1)
        results.append(res)
    path = str(tmp_path / 'reco.h5')
    write_results_h5(path, results, VPOL_CHANNELS, 'hw', True,
                     dict(detector_delay_hash='abc', detector_epoch='2022-10-01T00:00:00',
                          delay_corrections_hash='', delay_corrections_file=''))
    with h5py.File(path) as f:
        g = f['results']
        keys = set(g.keys())
        expected = set(contract_keys(VPOL_CHANNELS, 3, True)) | {'preproc_time'}
        assert expected <= keys, expected - keys
        assert all(is_versioned_key(k) for k in keys - expected), sorted(keys - expected)
        for k in FLOAT_KEYS + ['peak_1_rho', 'ch0_snr', 'pa_max_snr', 'coarse_time']:
            assert g[k].dtype == np.float64, k
        for k in ['run_number', 'event_number'] + [k for k, t, _ in VALIDATION_COUNT_KEYS if t is int]:
            assert g[k].dtype == np.int64, k
        assert g['has_helper_signal'].dtype == np.bool_
        assert g['source_file'].dtype.kind == 'O' and g['source_file'][0] in (b'fixture.nur', 'fixture.nur')
        assert list(g['run_number'][:]) == [100, 101] and g['rho'].shape == (2,)
        assert g['rho'][0] == results[0]['rho'] and g['peak_0_corr'][1] == results[1]['peak_0_corr']
        for i, res in enumerate(results):
            for key in peak_keys(3):
                assert g[key][i] == res[key] if key in res else np.isnan(g[key][i]), (i, key)
        for attr in FILE_ATTRS:
            assert attr in f.attrs, attr
        assert f.attrs['mode'] == 'hw' and f.attrs['n_events'] == 2 and bool(f.attrs['validation']) is True
        assert f.attrs['reco_version'] == RECO_VERSION and VERSION_RE.match(RECO_VERSION)
        assert f.attrs['detector_delay_hash'] == 'abc'
        assert COHERENT_GROUP not in f


@pytest.mark.slow
def test_results_file_coherent_waveforms(reco, det, base_config, tables, pa, tmp_path):
    """With `save_coherent_waveforms` the file carries the group with one row per event and peak."""
    config = dict(base_config, validation=True, save_coherent_waveforms=True, n_coherent_waveforms=2)
    results = []
    for i, seed in enumerate((31, 32)):
        res = _result(reco, det, config, tables, pa, seed)
        assert res['coherent_times'].ndim == 1 and len(res['coherent_times']) > 1
        assert all(res[f'coherent_wf_{k}'].shape == res['coherent_times'].shape for k in range(2))
        res.update(run_number=200, event_number=i, source_file='fixture.nur')
        results.append(res)
    path = str(tmp_path / 'reco_coherent.h5')
    write_results_h5(path, results, VPOL_CHANNELS, 'hw', True)
    with h5py.File(path) as f:
        g = f[COHERENT_GROUP]
        assert set(g.keys()) == set(coherent_keys(2))
        n_times = len(results[0]['coherent_times'])
        assert g['times'].shape == (n_times,) and g['times'].dtype == np.float64
        assert np.array_equal(g['times'][:], results[0]['coherent_times'])
        for k in range(2):
            assert g[f'peak_{k}'].shape == (2, n_times) and g[f'peak_{k}'].dtype == np.float64, k
            for i, res in enumerate(results):
                assert np.array_equal(g[f'peak_{k}'][i], res[f'coherent_wf_{k}']), (k, i)
        assert not any(k.startswith('coherent') for k in f['results'].keys())
        assert set(contract_keys(VPOL_CHANNELS, 3, True)) <= set(f['results'].keys())
