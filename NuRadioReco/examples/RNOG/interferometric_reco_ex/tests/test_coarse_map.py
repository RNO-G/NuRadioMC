"""The `save_coarse_map` option: the coarse correlation map and its axes in the result.

With the option the result of a hierarchical search carries `coarse_map_v1`, the map its
search chain returned (the raw chain's in candidate mode), on the axes
`coarse_map_rho_v1`, `coarse_map_phi_v1` and `coarse_map_z_v1`; the map SNR of the
saved peaks is read on exactly this map. Polarization groups carry one map each. Without
the option, or with it off, the result is what it was, and the driver's results file
never holds a map.
"""

import h5py
import numpy as np
import pytest

from conftest import STATION, reference_config
from reco_output import write_results_h5
from synthetic import HPOL_CHANNELS, VPOL_CHANNELS, TravelTimeTables, cylindrical_to_enu, make_event, same_value
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

SOURCE = (80.0, 120.0, -40.0)
MAP_KEYS = {'coarse_map_v1', 'coarse_map_rho_v1', 'coarse_map_phi_v1', 'coarse_map_z_v1'}
SHAPE = (30, 120, 100)


def _chain_maps(monkeypatch):
    """Collect the coarse map every `_search_chain` call returns for the rest of the test.

    The method is replaced on the class: a replacement on the shared `reco` object would stay on it as an instance
    attribute after the test.
    """
    maps = []
    chain = InterferometricReco3D._search_chain

    def keep(self, *args, **kwargs):
        """Original chain search, its coarse map noted."""
        out = chain(self, *args, **kwargs)
        maps.append(out['mean_corr_c'])
        return out

    monkeypatch.setattr(InterferometricReco3D, '_search_chain', keep)
    return maps


def _event(det, tables, pa, channels=VPOL_CHANNELS, seed=61):
    """Synthetic event with a pulse from SOURCE on the given channels."""
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*SOURCE, pa), channels, tables, snr=30.0, seed=seed)
    return evt, stn


@pytest.mark.slow
def test_map_is_the_search_chain_map_on_the_coarse_axes(reco, det, base_config, tables, pa, monkeypatch):
    """The stored map is the chain's map bit for bit, on the coarse grid, and reproduces the saved peaks' map SNR."""
    maps = _chain_maps(monkeypatch)
    evt, stn = _event(det, tables, pa)
    res = reco.run(evt, stn, det, dict(base_config, save_coarse_map=True))
    cmap, rho, phi, z = (res[k] for k in ('coarse_map_v1', 'coarse_map_rho_v1', 'coarse_map_phi_v1', 'coarse_map_z_v1'))
    assert len(maps) == 1 and cmap.shape == SHAPE and cmap.dtype == np.float64
    assert np.array_equal(cmap, maps[0], equal_nan=True)
    assert np.array_equal(rho, np.geomspace(1.0, 250.0, 30)) and np.allclose(phi, np.arange(0.0, 360.0, 3.0), atol=1e-9)
    assert len(z) == 100 and z[0] == -100.0 and z[-1] == 0.0 and np.all(np.diff(z) > 0)
    assert np.nanmax(cmap) == max(p[3] for p in res['coarse_peaks'])
    i_rho, i_phi, i_z = np.unravel_index(np.nanargmax(cmap), SHAPE)
    assert abs(phi[i_phi] - SOURCE[1]) <= 3.0 and abs(z[i_z] - SOURCE[2]) <= 15.0
    for i in range(res['n_saved_peaks']):
        peak = [res[f'peak_{i}_{f}'] for f in ('rho', 'phi', 'z')]
        assert reco._compute_map_snr(cmap, reco._find_peak_bin(*peak, rho, phi, z)) == res[f'peak_{i}_map_snr']


@pytest.mark.slow
def test_candidate_mode_stores_the_raw_chain_map(reco, det, base_config, tables, pa, monkeypatch):
    """With a candidate search the stored map is that of the raw chain, not the envelope chain."""
    maps = _chain_maps(monkeypatch)
    evt, stn = _event(det, tables, pa)
    res = reco.run(evt, stn, det, dict(base_config, candidate_search=['envelope', 'raw'], save_coarse_map=True))
    assert len(maps) == 2 and not np.array_equal(maps[0], maps[1], equal_nan=True)
    assert np.array_equal(res['coarse_map_v1'], maps[1], equal_nan=True)


@pytest.mark.slow
def test_option_off_leaves_the_result_unchanged(reco, det, base_config, tables, pa):
    """Absent and false give the same keys and values; on adds the four map keys and changes nothing else."""
    results = {}
    for name, extra in (('absent', {}), ('off', {'save_coarse_map': False}), ('on', {'save_coarse_map': True})):
        evt, stn = _event(det, tables, pa)
        results[name] = reco.run(evt, stn, det, dict(base_config, validation=True, **extra))
    absent = results['absent']
    assert set(results['off']) == set(absent) and not any('coarse_map' in k for k in absent)
    assert set(results['on']) == set(absent) | MAP_KEYS
    for name in ('off', 'on'):
        for key in (k for k in absent if not k.endswith('_time')):
            assert same_value(results[name][key], absent[key]), (name, key)
    with pytest.raises(ValueError, match='save_coarse_map'):
        reco._validate_config(dict(base_config, save_coarse_map='yes'))


@pytest.mark.slow
def test_polarization_groups_carry_one_map_each(det, table_dir, pa, monkeypatch):
    """Every group's map is stored under its suffix, the primary group's also without one."""
    channels = VPOL_CHANNELS + HPOL_CHANNELS
    cfg = reference_config(table_dir, channels=channels, save_coarse_map=True,
                           polarization_groups={'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS})
    reco = InterferometricReco3D()
    reco.begin(STATION, cfg, det)
    maps = _chain_maps(monkeypatch)
    evt, stn = _event(det, TravelTimeTables(table_dir, STATION, channels), pa, channels)
    res = reco.run(evt, stn, det, cfg)
    assert len(maps) == 2 and not np.array_equal(maps[0], maps[1], equal_nan=True)
    for name, chain_map in (('vpol', maps[0]), ('hpol', maps[1])):
        assert res[f'coarse_map_v1_{name}'].shape == SHAPE
        assert np.array_equal(res[f'coarse_map_v1_{name}'], chain_map, equal_nan=True)
        for axis in ('rho', 'phi', 'z'):
            assert np.array_equal(res[f'coarse_map_{axis}_v1_{name}'], res[f'coarse_map_{axis}_v1'])
    assert np.array_equal(res['coarse_map_v1'], res['coarse_map_v1_vpol'], equal_nan=True)


def test_results_file_holds_no_map(tmp_path):
    """The driver's writer leaves the maps and their axes out of the results file."""
    rng = np.random.default_rng(0)
    results = []
    for event in range(2):
        res = dict(rho=50.0, phi=10.0, z=-30.0, max_corr=0.5, run_number=1, event_number=event, source_file='synthetic')
        for suffix in ('', '_vpol', '_hpol'):
            res.update({f'coarse_map_v1{suffix}': rng.uniform(size=(3, 4, 5)), f'coarse_map_rho_v1{suffix}': np.arange(3.0),
                        f'coarse_map_phi_v1{suffix}': np.arange(4.0), f'coarse_map_z_v1{suffix}': np.arange(5.0),
                        f'max_corr{suffix}': 0.5})
        results.append(res)
    path = str(tmp_path / 'reco_maps.h5')
    write_results_h5(path, results, [], 'hw', False)
    with h5py.File(path) as f:
        assert f.attrs['n_events'] == 2 and 'max_corr_hpol' in f['results']
        assert not any('coarse_map' in k for k in f['results'].keys())
