"""Golden master on real calibration-pulser events through the driver's preprocessing chain.

The fixtures are ten-event voltage-only `.nur` cuts of station 23 run 1000, station 22
run 2090 and station 21 run 476 (`make_real_pulser_fixtures.py`: raw mattak voltages,
reconstruction channels only, the first ten events with PA SNR at least 5). The test
runs the record preprocessing (block offsets, cable delays, hardware phase, 0.1 to 0.7 GHz
bandpass, CW subtraction, 10 GHz resampling) on them inside the test, so a change to the
cable delays, the detector description or any preprocessing step is visible, which a
preprocessed fixture could not show. The expected (rho, phi, z, max_corr) per event
under the record configuration are compared to 1e-6 and the per-pair residual medians at
the pulser truth to 0.15 ns (the lag grid is 0.1 ns) for the pairs whose reference peak
correlation is at least 0.3 (on a noise pair the largest correlation in the window is not
a stable quantity).

The fixtures, their manifests and the stored reference are not part of the repository. The
tests read them from the folder named by RECO3D_TEST_PULSER_DATA and skip when it is not set
or a file is absent. Each file is first compared with its SHA-256 value in
`golden/real_pulser_data.sha256`; a file that is present with another checksum fails the test.
Set RECO3D_REGEN_PULSER_GOLDEN=1 to write the reference into that folder instead of comparing
(the checksums are then not compared); the README gives the steps.
"""

import hashlib
import json
import os

import numpy as np
import pytest

from conftest import load_station_detector, reference_config
from pulser_pair_residuals import pair_residuals, residual_medians
from reco_validation import pulser_truth
from synthetic import RECORD_PREPROCESSOR, VPOL_CHANNELS, write_json_rows
from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.modules.RNO_G.dataProviderNuRadio import dataProviderNuRadio
from NuRadioReco.utilities import units

DATA_ENV = 'RECO3D_TEST_PULSER_DATA'
CHECKSUMS = os.path.join(os.path.dirname(__file__), 'golden', 'real_pulser_data.sha256')
DATA_POINTER = ('The tests need the seven files listed in golden/real_pulser_data.sha256: ten calibration-pulser events '
                'of each of the stations 21, 22 and 23 as voltage files, their manifests and the stored reconstruction '
                'result. They are kept outside the repository, on the RNO-G Chicago server under '
                f'/data/reconstruction/test_data/reco3d/v1. Set {DATA_ENV} to a copy of that folder.')
FIXTURES = [pytest.param(23, 1000, 0, id='st23_run1000'), pytest.param(22, 2090, 1, id='st22_run2090'),
            pytest.param(21, 476, 0, id='st21_run476')]
TIGHT = 1e-6
RESIDUAL_TOL_NS = 0.15
SIGNAL_PAIR_CORR = 0.3
DELAY_SHIFT_NS = 1.0
SHIFTED_CHANNEL = 9
STRONG_CHANNELS = {0, 1, 2, 3, 9, 10}


def pulser_data_dir():
    """Folder of the real-pulser test data, with its files checked against the stored SHA-256 values.

    Skips the calling test when RECO3D_TEST_PULSER_DATA is not set, names no folder or the folder
    lacks a file, and fails it when a file is present with another checksum than the one in
    `golden/real_pulser_data.sha256`. With RECO3D_REGEN_PULSER_GOLDEN=1 the files are not looked
    at, because the reference and the list are being rewritten.

    Returns:
        The folder named by RECO3D_TEST_PULSER_DATA.
    """
    directory = os.environ.get(DATA_ENV, '')
    if not os.path.isdir(directory):
        state = f'{DATA_ENV}={directory} is not a folder' if directory else f'{DATA_ENV} is not set'
        pytest.skip(f'real-pulser test data not available: {state}. {DATA_POINTER}')
    if os.environ.get('RECO3D_REGEN_PULSER_GOLDEN') == '1':
        return directory
    with open(CHECKSUMS) as f:
        expected = dict(line.split()[::-1] for line in f)
    missing, wrong = [], []
    for name, digest in expected.items():
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            missing.append(name)
            continue
        with open(path, 'rb') as f:
            if hashlib.sha256(f.read()).hexdigest() != digest:
                wrong.append(name)
    if wrong:
        pytest.fail(f'real-pulser test data in {directory} do not match golden/real_pulser_data.sha256: the SHA-256 '
                    f'value differs for {", ".join(wrong)}')
    if missing:
        pytest.skip(f'real-pulser test data not available: {directory} has no {", ".join(missing)}. {DATA_POINTER}')
    return directory


def _preprocessed_events(data_dir, station, run, det):
    """Yield (run, event, station object) after the driver's preprocessing chain."""
    path = os.path.join(data_dir, f'st{station}_run{run}_voltage_reference.nur')
    dp = dataProviderNuRadio()
    dp.begin(path, det, preprocessor_config=dict(RECORD_PREPROCESSOR, apply_upsampling=False))
    resampler = channelResampler()
    resampler.begin()
    for run_nr, evt_nr in dp.get_event_ids():
        evt = dp.get_event(int(run_nr), int(evt_nr))
        stn = evt.get_station(station)
        resampler.run(evt, stn, det, sampling_rate=10 * units.GHz)
        yield int(run_nr), int(evt_nr), evt, stn
    dp.end()


def _reconstruct(data_dir, station, run, device, table_dir):
    """Reconstruct the fixture and measure the pair residuals at the pulser truth."""
    det = load_station_detector(station)
    config = reference_config(table_dir)
    reco = InterferometricReco3D()
    reco.begin(station, config, det)
    truth = pulser_truth(det, station, device)
    events, records = [], []
    for run_nr, evt_nr, evt, stn in _preprocessed_events(data_dir, station, run, det):
        res = reco.run(evt, stn, det, config)
        events.append(dict(run=run_nr, event=evt_nr, rho=float(res['rho']), phi=float(res['phi']),
                           z=float(res['z']), max_corr=float(res['max_corr'])))
        volt = [stn.get_channel(ch).get_trace() for ch in VPOL_CHANNELS]
        times = [stn.get_channel(ch).get_times() for ch in VPOL_CHANNELS]
        records.append(pair_residuals(reco, VPOL_CHANNELS, volt, times, {'truth': truth}))
    assert len(events) == 10, len(events)
    return dict(station=station, run=run, device=device, truth=list(truth), events=events,
                residual_medians_ns=residual_medians(records)['truth'],
                peak_corr_medians=residual_medians(records, 'c_max')['truth'])


@pytest.mark.slow
@pytest.mark.parametrize('station,run,device', FIXTURES)
def test_real_pulser_golden(table_dir, station, run, device):
    """Reproduce the stored reconstruction and pair residuals of the real pulser fixtures."""
    data_dir = pulser_data_dir()
    golden = os.path.join(data_dir, 'real_pulser_golden.json')
    got = _reconstruct(data_dir, station, run, device, table_dir)
    key = f'st{station}_run{run}'
    if os.environ.get('RECO3D_REGEN_PULSER_GOLDEN') == '1':
        ref = json.load(open(golden)) if os.path.isfile(golden) else {}
        ref[key] = got
        write_json_rows(golden, ref)
        pytest.skip(f'golden reference for {key} written to {golden}')
    ref = json.load(open(golden))[key]
    assert ref['device'] == device and np.allclose(ref['truth'], got['truth'], atol=1e-9)
    drift = []
    for exp, row in zip(ref['events'], got['events']):
        assert (row['run'], row['event']) == (exp['run'], exp['event'])
        for k in ('rho', 'phi', 'z', 'max_corr'):
            if abs(row[k] - exp[k]) > TIGHT:
                drift.append((row['event'], k, exp[k], row[k]))
    assert not drift, f'numerical drift from the golden reference (event, key, reference, now): {drift}'
    signal_pairs = [pair for pair, c in ref['peak_corr_medians'].items() if abs(c) >= SIGNAL_PAIR_CORR]
    assert len(signal_pairs) >= 6, signal_pairs
    off = {pair: (ref['residual_medians_ns'][pair], got['residual_medians_ns'][pair]) for pair in signal_pairs
           if abs(got['residual_medians_ns'][pair] - ref['residual_medians_ns'][pair]) > RESIDUAL_TOL_NS}
    assert not off, f'pair residual medians moved (pair: reference, now): {off}'


@pytest.mark.slow
def test_golden_detects_a_cable_delay_change(table_dir):
    """A 1 ns shift of channel 9 moves its pairs with the other strong channels by 1 ns at station 23."""
    data_dir = pulser_data_dir()
    det = load_station_detector(23)
    config = reference_config(table_dir)
    reco = InterferometricReco3D()
    reco.begin(23, config, det)
    truth = pulser_truth(det, 23, 0)
    shifted, plain = [], []
    for _, _, evt, stn in _preprocessed_events(data_dir, 23, 1000, det):
        volt = [stn.get_channel(ch).get_trace() for ch in VPOL_CHANNELS]
        times = [stn.get_channel(ch).get_times() for ch in VPOL_CHANNELS]
        plain.append(pair_residuals(reco, VPOL_CHANNELS, volt, times, {'truth': truth}))
        stn.get_channel(SHIFTED_CHANNEL).add_trace_start_time(DELAY_SHIFT_NS)
        times = [stn.get_channel(ch).get_times() for ch in VPOL_CHANNELS]
        shifted.append(pair_residuals(reco, VPOL_CHANNELS, volt, times, {'truth': truth}))
    before, after = residual_medians(plain)['truth'], residual_medians(shifted)['truth']
    checked = 0
    for pair in before:
        a, b = (int(v) for v in pair.split('-'))
        if SHIFTED_CHANNEL not in (a, b):
            assert after[pair] == before[pair], pair
            continue
        if not {a, b} <= STRONG_CHANNELS:
            continue
        expected = DELAY_SHIFT_NS if a == SHIFTED_CHANNEL else -DELAY_SHIFT_NS
        assert abs((after[pair] - before[pair]) - expected) <= RESIDUAL_TOL_NS, (pair, before[pair], after[pair])
        checked += 1
    assert checked == len(STRONG_CHANNELS) - 1
