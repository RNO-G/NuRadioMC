"""Candidate search: envelope and raw search chains polished and ranked by the raw correlation.

With `candidate_search` the reconstruction runs the search chain once per objective,
polishes every optimizer output with the raw correlation and ranks the candidates by
that value. Because the raw chain's candidates are among them, the primary raw
correlation can never fall below the default configuration's on the same event; that
invariant is checked for both envelope modes. The envelope chain supplies the basin
whenever the raw chain lands in a sidelobe: with the default (traces) envelope the
synthetic sources are recovered to better than 0.02 degree (gated at 0.3 degree and 2 m)
and the 16-source set is gated at 0.9 within 3 degrees (measured 1.0). With the
correlation envelope the candidates sit 0.5 to 3 degrees and several metres from the raw
peak on some geometries, so that mode is gated at its measured level over the same set.
A `candidate_search` entry names its chain directly (`raw`, `envelope:traces`,
`envelope:correlation`) or as plain `envelope` for the configured mode, so both envelope
chains can run beside the raw chain and the raw ranking chooses between them.
"""

import numpy as np
import pytest

from conftest import STATION, reference_config, rng_sources
from synthetic import (HPOL_CHANNELS, VPOL_CHANNELS, TravelTimeTables, angular_separation,
                       cylindrical_to_enu, make_event, make_noise_event)
from test_reco_known_answer import EXACT_SOURCES, SUMMARY_SOURCES
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

SNR = 20.0
TOL_DEG = 0.3
TOL_M = 2.0
SUMMARY_GATES = {'traces': 0.9, 'correlation': 0.8}
NOISE_CORR_MAX = 0.2
RECOVERY_SOURCES = list(dict.fromkeys([p.values[0] for p in EXACT_SOURCES] + SUMMARY_SOURCES))
SOURCE_IDS = [f'rho{s[0]:g}_phi{s[1]:g}_z{s[2]:g}' for s in RECOVERY_SOURCES]


@pytest.fixture(scope='module')
def cand_config(table_dir):
    """Reference configuration with the candidate search and the default (traces) envelope."""
    return reference_config(table_dir, candidate_search=['envelope', 'raw'])


@pytest.fixture(scope='module')
def corr_config(table_dir):
    """Reference configuration with the candidate search and the correlation envelope."""
    return reference_config(table_dir, candidate_search=['envelope', 'raw'],
                            candidate_envelope_mode='correlation')


def _event(det, src, tables, pa, snr=SNR, seed=5):
    """Build a synthetic VPol event for a (rho, phi, z) source relative to the PA."""
    return make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables, snr=snr, seed=seed)


def _sep(res, src, pa):
    """Angular separation in degrees between a reconstruction result and the source."""
    return angular_separation((res['rho'], res['phi'], res['z']), src, pa)


def test_invalid_candidate_keys_are_rejected():
    """Unknown chain names, duplicates, bad envelope modes and malformed windows raise ValueError."""
    reco = InterferometricReco3D()
    for bad in ({'candidate_search': ['bogus']},
                {'candidate_search': ['raw', 'raw']},
                {'candidate_search': 'raw'},
                {'candidate_search': [['raw']]},
                {'candidate_search': ['envelope:hilbert']},
                {'candidate_search': ['envelope', 'envelope:traces']},
                {'candidate_search': ['envelope:correlation', 'envelope:correlation']},
                {'candidate_search': ['envelope'], 'candidate_envelope_mode': 'hilbert'},
                {'candidate_search': ['envelope'], 'candidate_polish_window': [3.0, 1.0]},
                {'candidate_search': ['envelope'], 'candidate_polish_window': 3.0},
                {'candidate_search': ['envelope'], 'candidate_polish_steps': 0.5},
                {'candidate_search': ['envelope'], 'candidate_polish_steps': [0.5, 0.0, 0.5]},
                {'candidate_search': ['envelope'], 'candidate_polish_window': [[15, 3, 15], [3, 1, 3]]},
                {'candidate_search': ['envelope'], 'candidate_include_refined': 1}):
        with pytest.raises(ValueError):
            reco._validate_config(bad)
    reco._validate_config({'candidate_search': ['envelope', 'raw'], 'candidate_envelope_mode': 'correlation'})
    reco._validate_config({'candidate_search': ['envelope', 'envelope:correlation', 'raw']})
    assert reco._candidate_chains({'candidate_search': ['envelope', 'raw']}) == ['envelope:traces', 'raw']
    assert reco._candidate_chains({'candidate_search': ['envelope:traces', 'envelope:correlation', 'raw']}) \
        == ['envelope:traces', 'envelope:correlation', 'raw']
    reco._validate_config({'candidate_search': ['raw'], 'candidate_polish_window': [[15, 3, 15], [3, 1, 3]],
                           'candidate_polish_steps': [[1, 0.3, 1], [0.5, 0.1, 0.5]]})
    reco._validate_config({'candidate_search': None})


@pytest.mark.slow
@pytest.mark.parametrize('src', RECOVERY_SOURCES, ids=SOURCE_IDS)
def test_candidate_search_recovers_source(reco, det, cand_config, base_config, tables, pa, src):
    """Each exact-recovery and characterisation source is recovered at a raw correlation no lower than the default's."""
    evt, stn, _ = _event(det, src, tables, pa)
    res = reco.run(evt, stn, det, cand_config)
    ref = reco.run(evt, stn, det, base_config)
    sep = _sep(res, src, pa)
    assert sep < TOL_DEG, (src, sep, {k: res[k] for k in ('rho', 'phi', 'z', 'max_corr')})
    assert abs(res['rho'] - src[0]) < TOL_M and abs(res['z'] - src[2]) < TOL_M, res
    assert res['max_corr'] >= ref['max_corr'], (res['max_corr'], ref['max_corr'])


@pytest.mark.slow
@pytest.mark.parametrize('src', RECOVERY_SOURCES, ids=SOURCE_IDS)
def test_correlation_envelope_never_below_default(reco, det, corr_config, base_config, tables, pa, src):
    """With the correlation envelope the ranked raw correlation is still at least the default chain's."""
    evt, stn, _ = _event(det, src, tables, pa)
    res = reco.run(evt, stn, det, corr_config)
    ref = reco.run(evt, stn, det, base_config)
    assert res['max_corr'] >= ref['max_corr'], (src, res['max_corr'], ref['max_corr'])


@pytest.mark.slow
@pytest.mark.parametrize('mode', list(SUMMARY_GATES))
def test_candidate_search_accuracy_summary(request, reco, det, tables, pa, mode):
    """The fraction of the 16-source characterisation set within 3 degrees stays at the measured level."""
    config = request.getfixturevalue('cand_config' if mode == 'traces' else 'corr_config')
    sources = SUMMARY_SOURCES + rng_sources(8, 20260916)
    seps = np.array([_sep(reco.run(*_event(det, src, tables, pa)[:2], det, config), src, pa) for src in sources])
    print(f'candidate search ({mode} envelope) separations at SNR 20:', np.round(seps, 3).tolist())
    assert np.mean(seps < 3.0) >= SUMMARY_GATES[mode], seps


@pytest.mark.slow
def test_candidate_result_keys(reco, det, cand_config, tables, pa):
    """peak_0 is the primary result, saved peaks are ranked and each carries its chain of origin."""
    evt, stn, _ = _event(det, (80.0, 120.0, -40.0), tables, pa)
    res = reco.run(evt, stn, det, cand_config)
    assert res['peak_0_rho'] == res['rho'] and res['peak_0_phi'] == res['phi']
    assert res['peak_0_z'] == res['z'] and res['peak_0_corr'] == res['max_corr']
    n_saved = sum(1 for k in res if k.startswith('peak_') and k.endswith('_corr'))
    assert n_saved == res['n_saved_peaks'] <= cand_config['n_peaks_save']
    assert n_saved <= res['n_candidates']
    corrs = [res[f'peak_{i}_corr'] for i in range(n_saved)]
    assert corrs == sorted(corrs, reverse=True)
    for i in range(n_saved):
        assert res[f'candidate_origin_{i}'] in (0, 1)
        assert np.isfinite(res[f'peak_{i}_map_snr'])
    assert res['candidate_map_snr_chain'] == 0
    assert res['candidate_search_time'] > 0 and res['candidate_polish_time'] > 0
    assert 'coarse_peaks' in res and res['n_coarse_peaks'] == len(res['coarse_peaks'])


@pytest.mark.slow
def test_three_chain_search_recovers_source(reco, det, table_dir, base_config, tables, pa):
    """Both envelope chains beside the raw chain recover the source and carry their own origin codes."""
    cfg = reference_config(table_dir, candidate_search=['envelope:traces', 'envelope:correlation', 'raw'])
    src = (80.0, 120.0, -40.0)
    evt, stn, _ = _event(det, src, tables, pa)
    res = reco.run(evt, stn, det, cfg)
    ref = reco.run(evt, stn, det, base_config)
    assert _sep(res, src, pa) < TOL_DEG, res
    assert res['max_corr'] >= ref['max_corr'], (res['max_corr'], ref['max_corr'])
    assert res['candidate_map_snr_chain'] == 0
    origins = {res[f'candidate_origin_{i}'] for i in range(res['n_saved_peaks'])}
    assert origins <= {0, 1, 2}, origins


@pytest.mark.slow
def test_two_level_polish_recovers_source(reco, det, table_dir, tables, pa):
    """A coarser polish level ahead of the dense grid still recovers the source."""
    cfg = reference_config(table_dir, candidate_search=['envelope', 'raw'],
                           candidate_polish_window=[[15.0, 3.0, 15.0], [3.0, 1.0, 3.0]],
                           candidate_polish_steps=[[1.0, 0.3, 1.0], [0.5, 0.1, 0.5]])
    src = (80.0, 120.0, -40.0)
    evt, stn, _ = _event(det, src, tables, pa)
    res = reco.run(evt, stn, det, cfg)
    assert _sep(res, src, pa) < TOL_DEG, res
    assert res['peak_0_corr'] == res['max_corr']


@pytest.mark.slow
def test_default_config_has_no_candidate_keys(reco, det, base_config, tables, pa):
    """Without the new keys the result dict carries no candidate fields."""
    evt, stn, _ = _event(det, (80.0, 120.0, -40.0), tables, pa)
    res = reco.run(evt, stn, det, base_config)
    assert not [k for k in res if k.startswith('candidate_') or k == 'n_candidates']


@pytest.mark.slow
def test_noise_only_event_has_low_correlation(reco, det, cand_config):
    """Pure noise must not produce a confident peak under the candidate search either."""
    evt, stn = make_noise_event(STATION, VPOL_CHANNELS, seed=5)
    res = reco.run(evt, stn, det, cand_config)
    assert res['max_corr'] < NOISE_CORR_MAX, res


@pytest.mark.slow
def test_polarization_groups_run_their_own_chains(det, table_dir, pa):
    """With polarization groups each group runs its own chains and reports its own candidates."""
    channels = VPOL_CHANNELS + HPOL_CHANNELS
    cfg = reference_config(table_dir, channels=channels,
                           polarization_groups={'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS},
                           candidate_search=['envelope', 'raw'])
    all_tables = TravelTimeTables(table_dir, STATION, channels)
    reco = InterferometricReco3D()
    reco.begin(STATION, cfg, det)
    src = (80.0, 120.0, -40.0)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), channels, all_tables, snr=SNR, seed=3)
    res = reco.run(evt, stn, det, cfg)
    assert _sep(res, src, pa) < TOL_DEG, res
    for group in ('vpol', 'hpol'):
        assert res[f'candidate_origin_0_{group}'] in (0, 1)
        assert res[f'n_candidates_{group}'] >= 1
        assert res[f'peak_0_corr_{group}'] == res[f'max_corr_{group}']
    assert angular_separation((res['rho_hpol'], res['phi_hpol'], res['z_hpol']), src, pa) < 5.0
