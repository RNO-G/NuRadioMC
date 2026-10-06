"""Horizontal channel position shifts (``channel_position_shift``) in the search.

A zero shift leaves every result bit-identical to no key, in ``run`` and in
``reconstruct_from_pairs``. A shifted channel gives the same results, to float rounding,
as a detector description with that channel's position edited (the tables are axially
symmetric about their antenna, so the position is all that changes). A source seen by a
string moved from its database position is recovered with the shift and not without it.
Cut series whose lag windows cannot hold the shifted delays are refused, and vertical
shifts and unknown channels raise.
"""

import json
import logging
import lzma
import os

import numpy as np
import pytest

from conftest import DETECTOR_DATE, STATION, reference_config
from synthetic import VPOL_CHANNELS, angular_separation, cylindrical_to_enu, make_event, same_value
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from pair_store import DEFAULT_MARGIN_NS, cut_pairs

SOURCE = (120.0, 70.0, -60.0)
STRING_C = (22, 23)


def edited_detector(tmp_path, shifts):
    """Station-23 detector read from RECO3D_TEST_DETECTOR_FILE with channel positions moved by shifts {ch: (dx, dy)}."""
    path = os.environ.get('RECO3D_TEST_DETECTOR_FILE')
    if not path or not os.path.isfile(path):
        pytest.skip('needs RECO3D_TEST_DETECTOR_FILE to edit channel positions')
    doc = json.loads(lzma.open(path).read())
    for ch, (dx, dy) in shifts.items():
        pos = doc['data'][str(STATION)]['channels'][str(ch)]['channel_position']['position']
        pos[0] += dx
        pos[1] += dy
    out = tmp_path / 'detector_edited.json.xz'
    with lzma.open(out, 'w') as f:
        f.write(json.dumps(doc).encode('utf-8'))
    d = rnog_detector.Detector(detector_file=str(out), select_stations=STATION, log_level=logging.WARNING)
    d.update(DETECTOR_DATE)
    return d


@pytest.fixture(scope='module')
def own_reco(det, table_dir):
    """Reconstruction object of this module, so that shifted positions never reach the shared fixture."""
    r = InterferometricReco3D()
    r.begin(STATION, reference_config(table_dir), det)
    return r


def assert_close(a, b):
    """Every non-timing field equal to float rounding."""
    assert set(a) == set(b)
    for key, value in a.items():
        if '_time' in key:
            continue
        if isinstance(value, (float, np.floating)) and np.isfinite(value):
            assert np.isclose(value, b[key], rtol=1e-9, atol=1e-6), (key, value, b[key])
        else:
            assert same_value(value, b[key]), key


@pytest.mark.slow
def test_zero_shift_is_bit_identical(reco, det, base_config, tables, pa):
    """A zero shift equals no key, in run and in reconstruct_from_pairs."""
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*SOURCE, pa), VPOL_CHANNELS, tables, snr=10.0, seed=4)
    off = reco.run(evt, stn, det, base_config)
    zero = reco.run(evt, stn, det, dict(base_config, channel_position_shift={22: (0.0, 0.0), 9: [0, 0]}))
    pairs = reco.compute_pairs(stn, base_config)
    from_pairs = reco.reconstruct_from_pairs(pairs, base_config, channel_position_shift={22: (0.0, 0.0)})
    for key, value in off.items():
        if '_time' not in key:
            assert same_value(value, zero[key]), key
            assert same_value(value, from_pairs[key]), key


@pytest.mark.slow
def test_shift_equals_edited_detector(det, base_config, tables, pa, table_dir, tmp_path):
    """Shifting channels equals reconstructing with their positions edited in the detector description."""
    shift = {22: (1.3, -0.7), 23: (1.3, -0.7), 10: (-0.4, 0.9)}
    config = dict(base_config, region_hypotheses=True, far_field_hypothesis=True)
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*SOURCE, pa), VPOL_CHANNELS, tables, snr=10.0, seed=5)
    shifted = InterferometricReco3D()
    shifted.begin(STATION, reference_config(table_dir), det)
    by_shift = shifted.run(evt, stn, det, dict(config, channel_position_shift=shift))
    det_edit = edited_detector(tmp_path, shift)
    edited = InterferometricReco3D()
    edited.begin(STATION, reference_config(table_dir), det_edit)
    by_edit = edited.run(evt, stn, det_edit, config)
    assert_close(by_shift, by_edit)
    pairs = shifted.compute_pairs(stn, config)
    assert_close(shifted.reconstruct_from_pairs(pairs, config, channel_position_shift=shift), by_edit)


@pytest.mark.slow
def test_moved_string_recovered_only_with_the_shift(own_reco, det, base_config, tables, pa, tmp_path):
    """Pulses timed for a string moved 8.6 m: with the shift the source is found as well as
    when nothing moved, without it the direction is off by more than 5 degrees."""
    shift = {ch: (5.0, 7.0) for ch in STRING_C}
    moved = edited_detector(tmp_path, shift)
    evt, stn, _ = make_event(moved, STATION, cylindrical_to_enu(*SOURCE, pa), VPOL_CHANNELS, tables, snr=20.0, seed=6)
    evt0, stn0, _ = make_event(det, STATION, cylindrical_to_enu(*SOURCE, pa), VPOL_CHANNELS, tables, snr=20.0, seed=6)
    without = own_reco.run(evt, stn, det, base_config)
    with_shift = own_reco.run(evt, stn, det, dict(base_config, channel_position_shift=shift))
    unmoved = own_reco.run(evt0, stn0, det, base_config)
    sep = {name: angular_separation((r['rho'], r['phi'], r['z']), SOURCE, pa)
           for name, r in (('without', without), ('with', with_shift), ('unmoved', unmoved))}
    assert sep['with'] < sep['unmoved'] + 0.2, sep
    assert sep['without'] > 5.0, sep
    assert with_shift['max_corr'] > without['max_corr']


@pytest.mark.slow
def test_stored_windows_refuse_a_shift_they_cannot_hold(own_reco, det, base_config, tables, pa):
    """Series cut to the database geometry with the store margin hold a 10 cm shift and refuse a 10 m one."""
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*SOURCE, pa), VPOL_CHANNELS, tables, snr=10.0, seed=7)
    pairs = own_reco.compute_pairs(stn, base_config, store=True)
    windows = own_reco.pair_lag_windows(pairs.pairs, base_config) + [-DEFAULT_MARGIN_NS, DEFAULT_MARGIN_NS]
    cut = cut_pairs(pairs, windows, np.float64)
    own_reco.reconstruct_from_pairs(cut, base_config, channel_position_shift={22: (0.1, 0.0)})
    with pytest.raises(ValueError, match='lag windows'):
        own_reco.reconstruct_from_pairs(cut, base_config, channel_position_shift={22: (10.0, 0.0)})


@pytest.mark.parametrize('shift', [{22: (1.0, 0.0, 0.5)}, {22: (1.0,)}, {22: ('a', 0.0)}, {22: (np.nan, 0.0)},
                                   {99: (1.0, 0.0)}, [(22, 1.0, 0.0)]])
def test_invalid_shifts_raise(det, table_dir, shift):
    """Vertical shifts, malformed entries and channels without a position are refused at begin."""
    with pytest.raises(ValueError, match='channel_position_shift'):
        InterferometricReco3D().begin(STATION, reference_config(table_dir, channel_position_shift=shift), det)
