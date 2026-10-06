"""Polarity-aware HPol objective on synthetic events with one string's HPol pulse inverted.

The horizontal Askaryan field flips sign across the vertical plane through the shower
axis, so the HPol channels of strings on opposite sides of that plane see pulses of
opposite polarity and the signed correlation of such a cross-string pair scores the true
delay at minus its peak. The events here invert the HPol pulse of string C (channel 21),
so the three pairs with channel 21 anti-correlate at the truth: the signed HPol
reconstruction does not recover the source, `abs_cross_string` (absolute correlation on
the five cross-string pairs, signed on the same-string pair 4-8) does, and `joint_sign`
(the signed objective maximised over the relative sign assignments of the three strings)
does and reports the assignment with string C flipped. The VPol group is untouched by
every mode. Recovery is judged at 1 degree and 10 m: four HPol channels fix the direction
to about half a degree on these events but the range and depth only to a few metres.
"""

import numpy as np
import pytest

from conftest import STATION, reference_config
from synthetic import (HPOL_CHANNELS, VPOL_CHANNELS, TravelTimeTables, angular_separation,
                       cylindrical_to_enu, make_event)
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

CHANNELS = VPOL_CHANNELS + HPOL_CHANNELS
SRC = (80.0, 120.0, -40.0)
SNR = 20.0
RECOVER_DEG = 1.0
RECOVER_M = 10.0
MODES = InterferometricReco3D.HPOL_SIGN_MODES
PRIMARY_KEYS = ('rho', 'phi', 'z', 'max_corr', 'rho_vpol', 'phi_vpol', 'z_vpol', 'max_corr_vpol')


def _config(table_dir, mode):
    """Reference configuration with both polarization groups and the given HPol sign mode."""
    return reference_config(table_dir, channels=CHANNELS,
                            polarization_groups={'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS},
                            hpol_sign_mode=mode)


@pytest.fixture(scope='module')
def pol_reco(det, table_dir):
    """Reconstruction object initialised with the 15-channel configuration."""
    reco = InterferometricReco3D()
    reco.begin(STATION, _config(table_dir, 'signed'), det)
    return reco


@pytest.fixture(scope='module')
def pol_tables(table_dir):
    """Combined travel-time tables for the VPol and HPol channels."""
    return TravelTimeTables(table_dir, STATION, CHANNELS)


def _event(det, pol_tables, pa, invert=(), seed=3):
    """Build the 15-channel event at SRC with the pulses of the ``invert`` channels inverted.

    Returns:
        (event, station, travel_times) from ``make_event``.
    """
    return make_event(det, STATION, cylindrical_to_enu(*SRC, pa), CHANNELS, pol_tables, snr=SNR,
                      seed=seed, invert=invert)


def _hpol(res):
    """Return the HPol group's (rho, phi, z) from a result dict."""
    return (res['rho_hpol'], res['phi_hpol'], res['z_hpol'])


def _recovered(res, pa):
    """Return whether the HPol position is within RECOVER_DEG of SRC and within RECOVER_M in rho and z."""
    pos = _hpol(res)
    return (angular_separation(pos, SRC, pa) < RECOVER_DEG and abs(pos[0] - SRC[0]) < RECOVER_M
            and abs(pos[2] - SRC[2]) < RECOVER_M)


@pytest.mark.slow
def test_inverted_string_recovered_by_polarity_aware_modes(pol_reco, det, table_dir, pol_tables, pa):
    """With string C's HPol pulse inverted only the polarity-aware modes recover the source."""
    evt, stn, _ = _event(det, pol_tables, pa, invert=[21])
    res = {mode: pol_reco.run(evt, stn, det, _config(table_dir, mode)) for mode in MODES}
    report = {m: (round(angular_separation(_hpol(r), SRC, pa), 3), tuple(round(v, 2) for v in _hpol(r)),
                  round(r['max_corr_hpol'], 4)) for m, r in res.items()}
    print('inverted string C, HPol (sep deg, position, corr) by mode:', report)
    assert _recovered(res['abs_cross_string'], pa), report
    assert _recovered(res['joint_sign'], pa), report
    assert not _recovered(res['signed'], pa), report
    assert res['signed']['max_corr_hpol'] < 0.75 * res['abs_cross_string']['max_corr_hpol'], report
    assert res['signed']['max_corr_hpol'] < 0.75 * res['joint_sign']['max_corr_hpol'], report
    assert res['joint_sign']['sign_assignment_hpol'] == 1
    assert res['joint_sign']['sign_corr_1_hpol'] == res['joint_sign']['max_corr_hpol']
    assert res['joint_sign']['sign_corr_0_hpol'] == res['signed']['max_corr_hpol']
    assert res['abs_cross_string']['sign_mode_hpol'] == 1
    assert res['joint_sign']['sign_mode_hpol'] == 2
    assert 'sign_mode_hpol' not in res['signed'] and 'sign_mode' not in res['signed']
    for key in PRIMARY_KEYS:
        assert res['signed'][key] == res['abs_cross_string'][key] == res['joint_sign'][key], key


@pytest.mark.slow
def test_modes_agree_without_inversion(pol_reco, det, table_dir, pol_tables, pa):
    """Without a polarity flip every mode recovers the source and joint_sign keeps the signed run."""
    evt, stn, _ = _event(det, pol_tables, pa)
    res = {mode: pol_reco.run(evt, stn, det, _config(table_dir, mode)) for mode in MODES}
    report = {m: (round(angular_separation(_hpol(r), SRC, pa), 3), round(r['max_corr_hpol'], 4))
              for m, r in res.items()}
    print('no inversion, HPol (sep deg, corr) by mode:', report)
    for mode in MODES:
        assert _recovered(res[mode], pa), (mode, report)
    assert res['joint_sign']['sign_assignment_hpol'] == 0
    for key in ('rho_hpol', 'phi_hpol', 'z_hpol', 'max_corr_hpol'):
        assert res['joint_sign'][key] == res['signed'][key], key
    for key in PRIMARY_KEYS:
        assert res['signed'][key] == res['abs_cross_string'][key] == res['joint_sign'][key], key


def test_sign_helpers_and_invalid_keys(table_dir):
    """String membership, the per-pair sign tables and the key validation."""
    reco = InterferometricReco3D
    assert [reco._string_of(ch) for ch in (4, 8, 11, 21, 99)] == ['power', 'power', 'helper_b', 'helper_c', 99]
    assert reco._cross_string_abs_signs(HPOL_CHANNELS) == [1, 'abs', 'abs', 'abs', 'abs', 'abs']
    strings, assignments = reco._sign_assignments(HPOL_CHANNELS)
    assert strings == ['power', 'helper_b', 'helper_c']
    assert assignments == [[1, 1, 1, 1, 1, 1], [1, 1, -1, 1, -1, -1], [1, -1, 1, -1, 1, -1],
                           [1, -1, -1, -1, -1, 1]]
    assert reco.objective_version(_config(table_dir, 'joint_sign')) == 'hpol_sign_mode=joint_sign'
    assert reco.objective_version(_config(table_dir, 'signed')) == 'record'
    for bad in ({'hpol_sign_mode': 'abs'},
                {'hpol_sign_mode': 'abs_cross_string'},
                {'hpol_sign_mode': 'joint_sign', 'polarization_groups': {'vpol': [0, 1]}},
                {'pair_signs': [1, 2]},
                {'pair_signs': 'abs'},
                {'channels': [0, 1, 2], 'pair_signs': [1, -1]},
                {'pair_signs': [1], 'polarization_groups': {'vpol': [0, 1]}}):
        with pytest.raises(ValueError):
            reco()._validate_config(bad)
    reco()._validate_config(_config(table_dir, 'abs_cross_string'))
    reco()._validate_config({'pair_signs': [1, -1, 'abs']})
    reco()._validate_config({'channels': [0, 1, 2], 'pair_signs': [1, -1, 'abs']})
    with pytest.raises(ValueError):
        reco()._prepare_corr_funcs([np.arange(8.0)] * 3, [np.ones(8)] * 3, pair_signs=[1, -1])
