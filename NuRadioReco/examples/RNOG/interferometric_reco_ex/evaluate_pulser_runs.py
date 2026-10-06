"""Score reconstructions of calibration-pulser runs against the pulser device positions.

Truth is the in-ice pulser position from the detector description in the absolute
frame of the reconstruction (`reco_validation.pulser_truth`). The device that fired is
taken from the run's `aux/comment.txt` (`fiber0` is the helper-C pulser, `fiber1` the
helper-B pulser) and checked against the helper SNR pattern (the pulser's own string is
the loudest); it is never the device closest to a reconstruction. Signal events are those
with PA SNR and host-string SNR above thresholds (from the validation columns of the
results file). Reports residuals in rho, phi, z, the 3D distance and the angular
separation from the phased-array reference point, per run.
"""

import argparse
import datetime
import glob
import json
import logging
import os

import h5py
import yaml

from reco_scoring import (DEFAULT_MIN_HOST_SNR, DEFAULT_MIN_PA_SNR, pa_reference_absolute,
                          pulser_residual_rows, run_number_from_path, select_signal_events,
                          summarise_pulser_rows, truth_device)
from reco_validation import PULSER_DEVICE_STRING, deep_pulser_devices, pulser_truth


def load_rows(path):
    """Per-event rows of a reconstruction results file as dicts of scalar columns."""
    with h5py.File(path) as f:
        g = f['results']
        cols = {k: g[k][:] for k in g.keys() if g[k].ndim == 1 and g[k].dtype.kind in 'fiub'}
    n = len(cols['rho'])
    return [{k: v[i] for k, v in cols.items()} for i in range(n)]


def run_comment(run_root, run_number):
    """Contents of `<run_root>/run<N>/aux/comment.txt`, or None when absent."""
    if run_root is None or run_number is None:
        return None
    path = os.path.join(run_root, f'run{run_number}', 'aux', 'comment.txt')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read().strip()


def score_run(rows, det, station_id, pa, comment, device=None, min_pa_snr=DEFAULT_MIN_PA_SNR,
              min_host_snr=DEFAULT_MIN_HOST_SNR):
    """Score one run's rows against the pulser that fired.

    Returns:
        Summary dict with the device, how it was chosen, the selection counts and the
        residual summary of the selected events at the top level (`n` is the selected
        count), plus the summary of every event with PA signal under `pa_signal`.
    """
    device_id, source = truth_device(comment, rows, override=device)
    host = PULSER_DEVICE_STRING[device_id]
    truth = pulser_truth(det, station_id, device_id)
    selected = select_signal_events(rows, host, min_pa_snr, min_host_snr)
    pa_only = select_signal_events(rows, host, min_pa_snr, 0.0)
    return dict(device=device_id, device_name=deep_pulser_devices(det, station_id).get(device_id, ''),
                device_source=source, host_string=host, truth=list(truth),
                n_total=len(rows), n_pa_signal=len(pa_only), min_pa_snr=min_pa_snr, min_host_snr=min_host_snr,
                **summarise_pulser_rows(pulser_residual_rows(selected, truth, pa[2])),
                pa_signal=summarise_pulser_rows(pulser_residual_rows(pa_only, truth, pa[2])))


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--result-dir', required=True, help='directory of reco_<run>.h5 files')
    ap.add_argument('--config', required=True, help='reco config (detector file, station, date)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--run-root', default=None, help='directory holding run<N>/aux/comment.txt')
    ap.add_argument('--device', type=int, default=None, help='pulser device id when no comment or SNR pattern decides')
    ap.add_argument('--min-pa-snr', type=float, default=DEFAULT_MIN_PA_SNR)
    ap.add_argument('--min-host-snr', type=float, default=DEFAULT_MIN_HOST_SNR)
    args = ap.parse_args()
    from NuRadioReco.detector.RNO_G.rnog_detector import Detector
    cfg = yaml.safe_load(open(args.config))
    station_id = int(cfg['station_id'])
    det = Detector(detector_file=cfg['detector_file'], select_stations=station_id, log_level=logging.WARNING)
    det.update(datetime.datetime.fromisoformat(cfg.get('detector_date', '2022-10-01')))
    pa = pa_reference_absolute(det, station_id)
    summary = {}
    for path in sorted(glob.glob(os.path.join(args.result_dir, 'reco_*.h5'))):
        label = os.path.basename(path)[5:-3]
        comment = run_comment(args.run_root, run_number_from_path(path))
        rows = load_rows(path)
        if not rows:
            continue
        try:
            summary[label] = score_run(rows, det, station_id, pa, comment, args.device,
                                       args.min_pa_snr, args.min_host_snr)
        except ValueError as err:
            summary[label] = dict(n_total=len(rows), error=str(err))
            print(f'{label}: not scored ({err})')
            continue
        s = summary[label]
        print(f"{label}: device {s['device']} ({s['device_source']}), {s['n']} of {s['n_total']} "
              f"events selected ({s['n_pa_signal']} with PA signal)")
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
