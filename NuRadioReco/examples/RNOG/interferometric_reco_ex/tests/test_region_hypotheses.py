"""Region hypotheses: the best position of the same search below and above the ice surface.

With ``region_hypotheses: true`` every result also reports, for the region below the surface
and the region above it, the best position the search produced there (optimizer and polish
outputs and graded refined peaks), the raw and both envelope correlations at it, its map SNR
and the chain it came from. The key adds fields only: every other result field equals the
search without it. The region holding the winner reports a search value at least the
winner's and the other region a lower one; on an in-ice volume no above-surface position
exists. With the recommended candidate search, which recovers these sources, the best
position of the source's region is the source within 1 degree, on the split z grid for an
above-surface and an in-ice source.
"""

import pytest

from conftest import STATION, reference_config
from synthetic import (VPOL_CHANNELS, angular_separation, cylindrical_to_enu, make_event,
                       make_noise_event, same_value)
from test_split_z_grid import RECOMMENDED_CANDIDATE, airice_dir, airice_tables, split_config, split_reco  # noqa: F401

REGIONS = ('below', 'above')
FIELDS = ('rho', 'phi', 'z', 'corr_raw', 'corr_env_traces', 'corr_env_correlation', 'map_snr',
          'origin')
AIR_SOURCE = (100.0, 200.0, 20.0)
ICE_SOURCE = (60.0, 40.0, -30.0)


def _region_keys():
    """Result keys the region hypotheses add."""
    return {f'{r}_{f}_v1' for r in REGIONS for f in FIELDS}


def _assert_unchanged(off, on):
    """Every field of the search without the key is unchanged with it, and only the region keys are added."""
    assert set(on) - set(off) == _region_keys(), sorted(set(on) ^ set(off))
    for k, v in off.items():
        if not k.endswith('_time'):
            assert same_value(v, on[k]), (k, v, on[k])


@pytest.mark.slow
@pytest.mark.parametrize('candidate', [False, True], ids=['record', 'recommended'])
def test_in_ice_volume(reco, det, base_config, tables, pa, candidate):
    """On an in-ice volume the winner is unchanged, the best in-ice position is the winner and none lies above."""
    config = dict(base_config, **(RECOMMENDED_CANDIDATE if candidate else {}))
    events = [make_event(det, STATION, cylindrical_to_enu(*ICE_SOURCE, pa), VPOL_CHANNELS, tables,
                         snr=15.0, seed=21)[:2], make_noise_event(STATION, VPOL_CHANNELS, seed=8)]
    for i, (evt, stn) in enumerate(events):
        off = reco.run(evt, stn, det, config)
        on = reco.run(evt, stn, det, dict(config, region_hypotheses=True))
        _assert_unchanged(off, on)
        assert on['above_origin_v1'] == -1 and on['above_rho_v1'] != on['above_rho_v1']
        assert on['below_corr_raw_v1'] >= on['max_corr'] - 1e-12, (i, on['below_corr_raw_v1'], on['max_corr'])
        if i == 0 and candidate:
            assert angular_separation((on['below_rho_v1'], on['below_phi_v1'], on['below_z_v1']),
                                      ICE_SOURCE, pa) < 1.0


@pytest.mark.slow
@pytest.mark.airice
@pytest.mark.parametrize('candidate', [False, True], ids=['record', 'recommended'])
@pytest.mark.parametrize('source', [AIR_SOURCE, ICE_SOURCE], ids=['air', 'ice'])
def test_split_grid_regions(split_reco, det, airice_dir, airice_tables, pa, candidate, source):
    """On the split grid the source's region holds the winner and the other region a lower correlation."""
    config = split_config(airice_dir, **(RECOMMENDED_CANDIDATE if candidate else {}))
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*source, pa), VPOL_CHANNELS,
                             airice_tables, snr=20.0, seed=31)
    off = split_reco.run(evt, stn, det, config)
    on = split_reco.run(evt, stn, det, dict(config, region_hypotheses=True))
    _assert_unchanged(off, on)
    near, far = ('above', 'below') if on['z'] > 0 else ('below', 'above')
    if candidate:
        assert near == ('above' if source[2] > 0 else 'below')
        position = (on[f'{near}_rho_v1'], on[f'{near}_phi_v1'], on[f'{near}_z_v1'])
        assert angular_separation(position, source, pa) < 1.0, (position, source)
    assert on[f'{near}_corr_raw_v1'] >= on['max_corr'] - 1e-12
    assert on[f'{far}_origin_v1'] >= 0 and on[f'{far}_corr_raw_v1'] < on[f'{near}_corr_raw_v1']
    assert (on['above_z_v1'] >= 0.0) and (on['below_z_v1'] <= 0.0)
    for region in REGIONS:
        assert -1.0 <= on[f'{region}_corr_env_traces_v1'] <= 1.0
        assert -1.0 <= on[f'{region}_corr_env_correlation_v1'] <= 1.0
        assert on[f'{region}_map_snr_v1'] == on[f'{region}_map_snr_v1']
    if candidate:
        assert on[f'{near}_corr_env_traces_v1'] > on[f'{far}_corr_env_traces_v1']


def test_region_key_validation(det, table_dir):
    """region_hypotheses must be a boolean."""
    from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
    with pytest.raises(ValueError, match='region_hypotheses'):
        InterferometricReco3D().begin(STATION, reference_config(table_dir, region_hypotheses='yes'), det)
