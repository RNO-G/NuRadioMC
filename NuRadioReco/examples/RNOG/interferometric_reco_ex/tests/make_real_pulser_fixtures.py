"""Cut ten-event voltage-only `.nur` fixtures from real calibration-pulser runs.

Events are selected through the driver's data provider with the record preprocessing
(the first `--n-events` events in file order whose PA SNR is at least `--min-pa-snr`),
then re-read raw (mattak linear voltage conversion, no baseline correction, no cable
delays) and written with the reconstruction channels only, so that the test suite can
run the preprocessing chain itself and see a cable-delay or detector change. A manifest
JSON next to the fixture records the selection, the per-event SNRs, the run comment and
the truth device. Both are written to `--out-dir` as `st{station}_run{run}_voltage_reference.nur`
and `_manifest.json`, the names `test_real_pulser_golden.py` reads from the folder named by
RECO3D_TEST_PULSER_DATA. The fixtures are test data and are not committed, so an output folder
inside the repository is refused. Needs mattak, so run it as a SLURM job in the rnog_py311
environment.
"""

import argparse
import datetime
import logging
import os
import subprocess

from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from NuRadioReco.utilities import units

from synthetic import RECORD_PREPROCESSOR, VPOL_CHANNELS, write_json_rows

MATTAK_KWARGS = {'mattak_kwargs': {'read_daq_status': False, 'backend': 'uproot'}}
WRITE_MODE = {'Channels': True, 'ElectricFields': False, 'SimChannels': False, 'SimElectricFields': False}


def select_events(input_path, det, station, channels, n_events, min_pa_snr):
    """First `n_events` (run, event) ids with PA SNR at least `min_pa_snr`, with their SNRs."""
    dp = dataProviderRNOG()
    dp.begin(input_path, det, reader_kwargs=dict(MATTAK_KWARGS),
             preprocessor_config=dict(RECORD_PREPROCESSOR, apply_upsampling=False))
    resampler = channelResampler()
    resampler.begin()
    selected = []
    for run_nr, evt_nr in dp.get_event_ids():
        evt = dp.get_event(int(run_nr), int(evt_nr))
        stn = evt.get_station(station)
        resampler.run(evt, stn, det, sampling_rate=10 * units.GHz)
        volt = [stn.get_channel(ch).get_trace() for ch in channels]
        _, snrs = InterferometricReco3D._compute_snr_pair_weights(volt, channels)
        pa = max(snrs[ch] for ch in (0, 1, 2, 3))
        if pa < min_pa_snr:
            continue
        selected.append(dict(run=int(run_nr), event=int(evt_nr), pa_snr=float(pa),
                             helper_b_snr=float(max(snrs[9], snrs[10])), helper_c_snr=float(max(snrs[22], snrs[23])),
                             snr={str(ch): float(snrs[ch]) for ch in channels}))
        if len(selected) == n_events:
            break
    dp.end()
    return selected


def write_fixture(input_path, station, channels, selected, out_path):
    """Write the selected events raw, restricted to `channels`."""
    reader = readRNOGData()
    reader.begin([input_path], apply_baseline_correction=None, **MATTAK_KWARGS)
    writer = eventWriter()
    writer.begin(out_path)
    for sel in selected:
        evt = reader.get_event(sel['run'], sel['event'])
        stn = evt.get_station(station)
        for ch in list(stn.get_channel_ids()):
            if ch not in channels:
                stn.remove_channel(ch)
        writer.run(evt, mode=WRITE_MODE)
    writer.end()
    reader.end()


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--station', type=int, required=True)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--input', required=True, help='combined.root or run directory')
    ap.add_argument('--detector-file', required=True)
    ap.add_argument('--out-dir', required=True, help='existing folder outside the repository for the fixture and manifest')
    ap.add_argument('--n-events', type=int, default=10)
    ap.add_argument('--min-pa-snr', type=float, default=5.0)
    ap.add_argument('--device', type=int, required=True, help='truth pulser device id (0 helper C, 1 helper B)')
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=here, capture_output=True, text=True).stdout.strip()
    if repo and os.path.commonpath([os.path.realpath(args.out_dir), os.path.realpath(repo)]) == os.path.realpath(repo):
        ap.error(f'--out-dir {args.out_dir} is inside the repository {repo}; test data are kept outside of it')
    out = os.path.join(args.out_dir, f'st{args.station}_run{args.run}_voltage_reference.nur')
    det = rnog_detector.Detector(detector_file=args.detector_file, select_stations=args.station,
                                 log_level=logging.WARNING)
    det.update(datetime.datetime(2022, 10, 1))
    selected = select_events(args.input, det, args.station, VPOL_CHANNELS, args.n_events, args.min_pa_snr)
    write_fixture(args.input, args.station, VPOL_CHANNELS, selected, out)
    run_dir = os.path.dirname(args.input) if args.input.endswith('.root') else args.input
    comment_path = os.path.join(run_dir, 'aux', 'comment.txt')
    comment = open(comment_path).read().strip() if os.path.isfile(comment_path) else None
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=here, capture_output=True, text=True).stdout.strip()
    manifest = dict(station=args.station, run=args.run, input=os.path.join(*os.path.normpath(args.input).split(os.sep)[-3:]),
                    detector_file=os.path.basename(args.detector_file),
                    detector_date='2022-10-01', channels=VPOL_CHANNELS, n_events=len(selected),
                    selection=(f'first {args.n_events} events in file order with PA SNR >= {args.min_pa_snr} '
                               'after the record preprocessing'),
                    selection_preprocessor=RECORD_PREPROCESSOR, run_comment=comment, truth_device=args.device,
                    content=('raw mattak voltages (linear ADC conversion), no baseline correction, no cable delays, '
                             'reconstruction channels only'),
                    code_commit=commit, created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                    file_bytes=os.path.getsize(out), events=selected)
    write_json_rows(out[:-4] + '_manifest.json', manifest)
    print(f"wrote {out}: {len(selected)} events, {os.path.getsize(out) / 1e6:.2f} MB")
    for sel in selected:
        print(f"  run {sel['run']} event {sel['event']}: PA {sel['pa_snr']:.1f} B {sel['helper_b_snr']:.1f} "
              f"C {sel['helper_c_snr']:.1f}")


if __name__ == '__main__':
    main()
