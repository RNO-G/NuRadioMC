"""Per-pair timing residuals of real pulser events at given source positions.

For every channel pair the raw energy-normalised correlation is evaluated at the delay
the tables predict for a source position and the lag of the largest absolute correlation
within a window around it is recorded, so that per-string timing offsets and polarity
flips show up. Positions default to the in-ice pulser devices of the station in the
absolute frame (`reco_validation.pulser_truth`). The command line reads a run through
the same data provider and preprocessing chain as the reconstruction driver; the
functions take preprocessed traces so that tests can call them on fixtures.
"""

import argparse
import datetime
import itertools
import json
import logging

import numpy as np
import yaml

from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.utilities import units

from reco_validation import deep_pulser_devices, pulser_truth

STRING = {0: 'PA', 1: 'PA', 2: 'PA', 3: 'PA', 5: 'pow5', 6: 'pow6', 7: 'pow7', 9: 'B', 10: 'B', 22: 'C', 23: 'C'}
DEFAULT_WINDOW_NS = 60.0
LAG_STEP_NS = 0.1


def pulser_positions(det, station_id):
    """{device name with underscores: [rho, phi_deg, z_abs]} of the station's in-ice pulsers."""
    return {name.replace(' ', '_'): list(pulser_truth(det, station_id, device_id))
            for device_id, name in deep_pulser_devices(det, station_id).items()}


def pair_residuals(reco, channels, volt, times, positions, window_ns=DEFAULT_WINDOW_NS):
    """Per-pair residuals of one event at each position.

    Args:
        reco: `InterferometricReco3D` after `begin` (tables and antenna positions).
        channels: Channel list matching `volt` and `times`.
        volt: Upsampled voltage traces per channel.
        times: Time arrays per channel.
        positions: {name: (rho, phi_deg, z_abs)}.
        window_ns: Half-width of the lag window searched around the predicted delay.

    Returns:
        {name: {'a-b': dict(d_pred, c_pred, resid, c_max)}} with the predicted pair delay,
        the correlation at it, the lag of the largest absolute correlation relative to it
        and that correlation.
    """
    pairs = list(itertools.combinations(channels, 2))
    lags = np.arange(-window_ns, window_ns + LAG_STEP_NS / 2, LAG_STEP_NS)
    corr_data, _ = reco._prepare_corr_funcs(times, volt, hilbert_envelope_mode=None,
                                            apply_hann_window=False, correlation_normalization='energy')
    out = {}
    for name, (rho, phi, z) in positions.items():
        tts = reco._compute_travel_times_single_point(rho, phi, z, channels)
        out[name] = {}
        for pidx, (a, b) in enumerate(pairs):
            d_pred = float(tts[a] - tts[b])
            carr, dt, off = corr_data[pidx]
            vals = np.array([reco._interp_corr_scalar(carr, dt, off, d_pred + lag) for lag in lags])
            k = int(np.argmax(np.abs(vals)))
            out[name][f'{a}-{b}'] = dict(d_pred=d_pred, c_pred=float(reco._interp_corr_scalar(carr, dt, off, d_pred)),
                                         resid=float(lags[k]), c_max=float(vals[k]))
    return out


def residual_medians(records, key='resid'):
    """Median of one per-pair quantity (`resid` or `c_max`) per position over `pair_residuals` outputs."""
    return {name: {pair: float(np.median([rec[name][pair][key] for rec in records])) for pair in records[0][name]}
            for name in records[0]}


def print_table(name, position, rows_by_pair):
    """Print the per-pair table of one position."""
    print(f'=== position {name} {position}')
    print(f"{'pair':8s} {'strings':10s} {'d_pred':>8s} {'c_pred':>7s} {'resid_med':>9s} {'resid_std':>9s} {'c_max_med':>9s} {'sign-':>5s}")
    for pair, rows in rows_by_pair.items():
        a, b = (int(v) for v in pair.split('-'))
        r = np.array([x['resid'] for x in rows])
        cp = np.array([x['c_pred'] for x in rows])
        cm = np.array([x['c_max'] for x in rows])
        print(f"{a:2d}-{b:<5d} {STRING.get(a, '?'):>4s}-{STRING.get(b, '?'):<5s} {np.median([x['d_pred'] for x in rows]):8.1f} "
              f"{np.median(cp):7.3f} {np.median(r):9.2f} {np.std(r):9.2f} {np.median(cm):9.3f} {np.mean(cm < 0):5.2f}")


def main():
    """Command-line entry point: residuals of a real run at the pulser positions."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--positions', default=None, help='JSON {name: [rho, phi_deg, z_abs]}; default: the pulser devices')
    ap.add_argument('--max-events', type=int, default=25)
    ap.add_argument('--min-pa-snr', type=float, default=5.0)
    ap.add_argument('--window-ns', type=float, default=DEFAULT_WINDOW_NS)
    args = ap.parse_args()
    from NuRadioReco.detector.RNO_G import rnog_detector
    from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
    cfg = yaml.safe_load(open(args.config))
    station_id = int(cfg['station_id'])
    det = rnog_detector.Detector(detector_file=cfg['detector_file'], select_stations=station_id, log_level=logging.WARNING)
    det.update(datetime.datetime.fromisoformat(cfg['detector_date']))
    channels = list(cfg['channels'])
    reco = InterferometricReco3D()
    reco.begin(station_id, cfg, det)
    preproc = dict(cfg.get('preprocessor', {}))
    preproc['apply_upsampling'] = False
    dp = dataProviderRNOG()
    dp.begin(args.input, det, reader_kwargs={'mattak_kwargs': {'read_daq_status': False, 'backend': 'uproot'}},
             preprocessor_config=preproc)
    resampler = channelResampler()
    resampler.begin()
    positions = json.load(open(args.positions)) if args.positions else pulser_positions(det, station_id)
    records = []
    events = []
    for eid in dp.get_event_ids()[:args.max_events]:
        evt = dp.get_event(int(eid[0]), int(eid[1]))
        stn = evt.get_station(station_id)
        resampler.run(evt, stn, det, sampling_rate=10 * units.GHz)
        volt = [stn.get_channel(ch).get_trace() for ch in channels]
        times = [stn.get_channel(ch).get_times() for ch in channels]
        _, snrs = reco._compute_snr_pair_weights(volt, channels)
        if max(snrs.get(ch, 0.0) for ch in (0, 1, 2, 3)) < args.min_pa_snr:
            continue
        rec = pair_residuals(reco, channels, volt, times, positions, args.window_ns)
        for name in rec:
            for pair in rec[name]:
                a, b = (int(v) for v in pair.split('-'))
                rec[name][pair].update(event=int(eid[1]), snr_a=float(snrs.get(a, 0)), snr_b=float(snrs.get(b, 0)))
        records.append(rec)
        events.append(int(eid[1]))
    out = {name: {pair: [rec[name][pair] for rec in records] for pair in records[0][name]} for name in positions} if records else {}
    with open(args.out, 'w') as f:
        json.dump(dict(n_used=len(records), channels=channels, positions=positions, events=events, results=out), f)
    print(f'{len(records)} events with PA SNR >= {args.min_pa_snr}')
    for name in out:
        print_table(name, positions[name], out[name])


if __name__ == '__main__':
    main()
