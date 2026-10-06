"""Output contract of the reconstruction: keys, dtypes and definitions consumed downstream.

Under the reference configuration the module returns exactly the contract keys (any
other key must carry a `_v<N>` suffix), the definitions that downstream code relies on
hold (peak 0 is the primary result, the SNR summaries derive from the per-channel SNRs,
the isolation ratio from the coarse peaks, `n_channels_above` counts the PA and helper
channels), and the results file written by the driver's writer carries the contract
datasets with their dtypes and the provenance attributes, including `reco_version`; a
peak slot the search did not fill is NaN in the file; with `save_coherent_waveforms` the
file carries the `coherent_waveforms` group with its `times` and `peak_<i>` datasets, with
polarization groups also `times_<group>` and `peak_<i>_<group>`, and the waveforms of the
primary result become channels 100 and above at their sampling rate. The columns and
waveforms of the file are the union over its events, whichever event comes first, and the
coarse peaks of a polarization group are stacked with NaN rows for events with fewer.
"""

import re

import h5py
import numpy as np
import pytest

from conftest import STATION, reference_config
from reco_output import (COHERENT_CHANNEL_BASE, COHERENT_GROUP, FILE_ATTRS, IDENTITY_KEYS, PEAK_FIELDS,
                         VALIDATION_COUNT_KEYS, coherent_channels, coherent_keys, contract_keys, is_versioned_key,
                         numeric_result_keys, peak_keys, stacked_peaks, write_results_h5)
from reco_validation import HELPER_CHANNELS, PA_CHANNELS
from synthetic import HPOL_CHANNELS, VPOL_CHANNELS, TravelTimeTables, cylindrical_to_enu, make_event
from NuRadioReco.modules.interferometricDirectionReconstruction3D import RECO_VERSION, InterferometricReco3D
from NuRadioReco.utilities import units

SOURCE = (80.0, 120.0, -40.0)
VERSION_RE = re.compile(r'^\d+\.\d+\.\d+$')
FLOAT_KEYS = ['rho', 'phi', 'z', 'max_corr', 'surf_corr_z', 'surf_corr_zen', 'peak_isolation_ratio']
INT_KEYS = ['n_coarse_peaks', 'n_refined_peaks', 'n_saved_peaks', 'n_helpers_above', 'n_channels_above']


def _result(reco, det, config, tables, pa, seed):
    """Reconstruct one synthetic event with validation on."""
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*SOURCE, pa), config['channels'], tables,
                             snr=30.0, seed=seed)
    return reco.run(evt, stn, det, config)


def _group_result(n_peaks, seed):
    """Fields of one polarization group as the hierarchical search returns them, with `n_peaks` saved and coarse peaks."""
    rng = np.random.default_rng(seed)
    res = dict(rho=float(rng.uniform(1, 250)), phi=float(rng.uniform(0, 360)), z=float(rng.uniform(-200, 0)),
               max_corr=float(rng.uniform()), n_saved_peaks=n_peaks, n_coarse_peaks=n_peaks,
               coarse_peaks=[tuple(float(v) for v in rng.uniform(size=4)) for _ in range(n_peaks)])
    for i in range(n_peaks):
        res.update({f'peak_{i}_{f}': float(rng.uniform()) for f in PEAK_FIELDS})
    return res


def _grouped_result(event, vpol, hpol):
    """Result dict of an event under polarization groups: every group suffixed, the VPol group also bare."""
    res = dict(vpol, run_number=7, event_number=event, source_file='synthetic')
    for name, group in (('vpol', vpol), ('hpol', hpol)):
        res.update({f'{k}_{name}': v for k, v in group.items()})
    return res


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


def test_stacked_peaks_pads_the_shorter_lists():
    """Peak lists of equal length stack as numpy does; shorter lists get NaN rows."""
    two, three = [(1.0, 2.0, 3.0, 0.5), (4.0, 5.0, 6.0, 0.4)], [(7.0, 8.0, 9.0, 0.3)] * 3
    assert np.array_equal(stacked_peaks([two, two]), np.array([two, two]))
    out = stacked_peaks([two, three, []])
    assert out.shape == (3, 3, 4) and out.dtype == np.float64
    assert np.array_equal(out[0, :2], two) and np.all(np.isnan(out[0, 2]))
    assert np.array_equal(out[1], three) and np.all(np.isnan(out[2]))


@pytest.mark.parametrize('order', [(0, 1), (1, 0)])
def test_results_file_columns_are_the_union_over_events(order, tmp_path):
    """An event with two peaks beside one with three: every column is written, NaN where an event lacks the value."""
    events = [_grouped_result(0, _group_result(2, 1), _group_result(3, 2)),
              _grouped_result(1, _group_result(3, 3), _group_result(2, 4))]
    results = [events[i] for i in order]
    path = str(tmp_path / 'reco_union.h5')
    write_results_h5(path, results, VPOL_CHANNELS + HPOL_CHANNELS, 'hw', False)
    with h5py.File(path) as f:
        g = f['results']
        for suffix in ('', '_vpol', '_hpol'):
            assert set(peak_keys(3)) <= {k[:len(k) - len(suffix)] for k in g if k.endswith(suffix)}, suffix
        for row, res in enumerate(results):
            for key in (k for k in g if k.startswith('peak_')):
                assert g[key][row] == res[key] if key in res else np.isnan(g[key][row]), (row, key)
            for name in ('vpol', 'hpol'):
                stored, peaks = g[f'coarse_peaks_{name}'][row], res[f'coarse_peaks_{name}']
                assert stored.shape == (3, 4) and np.array_equal(stored[:len(peaks)], peaks)
                assert np.all(np.isnan(stored[len(peaks):]))
        assert g['n_saved_peaks'].dtype == np.int64 and sorted(g['n_saved_peaks_hpol'][:]) == [2, 3]
        assert f.attrs['n_events'] == 2
    assert set(numeric_result_keys(results, [], False)) == set(numeric_result_keys(results[::-1], [], False))
    full = [events[1], events[1]]
    assert numeric_result_keys(full, [], False) == numeric_result_keys(full[:1], [], False)


def _assert_coherent_channels(result, n_waveforms):
    """The waveforms of the primary result as channels 100 and above, sampled at 10 GHz in NuRadioReco units."""
    channels = coherent_channels(result)
    assert [c.get_id() for c in channels] == [COHERENT_CHANNEL_BASE + k for k in range(n_waveforms)]
    for k, channel in enumerate(channels):
        assert abs(channel.get_sampling_rate() / units.GHz - 10.0) < 1e-9
        assert np.array_equal(channel.get_trace(), result[f'coherent_wf_{k}'])


def test_coherent_waveforms_are_the_union_over_events(tmp_path):
    """An event without waveforms gets rows of zeros in either event order; unequal lengths leave no file."""
    times = 3.0 + 0.1 * np.arange(8)
    with_two = dict(_group_result(2, 1), run_number=1, event_number=0, source_file='synthetic',
                    coherent_times=times, coherent_wf_0=np.arange(8.0), coherent_wf_1=np.arange(8.0) - 3.0)
    without = dict(_group_result(2, 2), run_number=1, event_number=1, source_file='synthetic')
    for results in ([with_two, without], [without, with_two]):
        path = str(tmp_path / f'reco_{results.index(with_two)}.h5')
        write_results_h5(path, results, [], 'hw', False)
        with h5py.File(path) as f:
            g = f[COHERENT_GROUP]
            row = results.index(with_two)
            assert set(g.keys()) == set(coherent_keys(2)) and np.array_equal(g['times'][:], times)
            for k in range(2):
                assert np.array_equal(g[f'peak_{k}'][row], with_two[f'coherent_wf_{k}'])
                assert not np.any(g[f'peak_{k}'][1 - row])
    _assert_coherent_channels(with_two, 2)
    assert coherent_channels(without) == []
    path = tmp_path / 'reco_unequal.h5'
    with pytest.raises(ValueError):
        write_results_h5(str(path), [with_two, dict(with_two, coherent_wf_0=np.arange(5.0))], [], 'hw', False)
    assert not path.exists()


@pytest.mark.slow
def test_results_file_coherent_waveforms_with_polarization_groups(det, table_dir, pa, tmp_path):
    """With polarization groups the file carries every group's waveforms and time axis, and no waveform in `results`."""
    channels = VPOL_CHANNELS + HPOL_CHANNELS
    cfg = reference_config(table_dir, channels=channels, validation=True, save_coherent_waveforms=True,
                           n_coherent_waveforms=3, polarization_groups={'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS})
    reco = InterferometricReco3D()
    reco.begin(STATION, cfg, det)
    all_tables = TravelTimeTables(table_dir, STATION, channels)
    results = []
    for i, seed in enumerate((41, 42)):
        res = _result(reco, det, cfg, all_tables, pa, seed)
        res.update(run_number=300, event_number=i, source_file='fixture.nur')
        results.append(res)
    path = str(tmp_path / 'reco_groups.h5')
    write_results_h5(path, results, channels, 'hw', True)
    with h5py.File(path) as f:
        g = f[COHERENT_GROUP]
        assert not any(k.startswith('coherent') for k in f['results'].keys())
        assert f['results']['coarse_peaks_hpol'].ndim == 3 and f['results']['rho_hpol'].shape == (2,)
        for suffix in ('', '_vpol', '_hpol'):
            times = results[0]['coherent_times' + suffix]
            assert np.array_equal(g['times' + suffix][:], times)
            n_waveforms = max(sum(f'coherent_wf_{k}{suffix}' in res for k in range(3)) for res in results)
            assert n_waveforms >= 1 and f'peak_{n_waveforms}{suffix}' not in g
            for k in range(n_waveforms):
                assert g[f'peak_{k}{suffix}'].shape == (2, len(times)), (k, suffix)
                for i, res in enumerate(results):
                    expected = res.get(f'coherent_wf_{k}{suffix}', np.zeros(len(times)))
                    assert np.array_equal(g[f'peak_{k}{suffix}'][i], expected), (k, suffix, i)
        assert np.array_equal(g['peak_0'][:], g['peak_0_vpol'][:])
        assert f.attrs['n_events'] == 2
    for res in results:
        _assert_coherent_channels(res, sum(f'coherent_wf_{k}' in res for k in range(3)))
