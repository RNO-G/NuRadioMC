"""Scoring helpers shared by the evaluation scripts.

One PA reference, one angular-separation definition, one truth join keyed by
(source file, run number), the shower-axis truth points of the cosmic-ray tier and the
signal-event selection of the calibration-pulser runs live here so that every evaluator
scores the same way and the arithmetic is testable without data.
"""

import datetime
import logging
import os
import re

import numpy as np

from reco_validation import (PA_CHANNELS, FIBER_DEVICE, PULSER_DEVICE_STRING, STRING_CHANNELS,
                             pa_reference_point)

DEFAULT_MIN_PA_SNR = 5.0
DEFAULT_MIN_HOST_SNR = 20.0
DEFAULT_SNR_VOTE_MARGIN = 2.0
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
FIBER_RE = re.compile(r'fiber\s*(\d)', re.IGNORECASE)
RESULT_FILE_RE = re.compile(r'^reco_(?:run)?(\d+)\.h5$')


def pa_reference(detector_file, station_id, det_date='2022-10-01'):
    """Absolute PA reference point [x, y, z] from a detector snapshot file."""
    from NuRadioReco.detector.RNO_G.rnog_detector import Detector
    det = Detector(detector_file=detector_file, select_stations=station_id, log_level=logging.WARNING)
    det.update(datetime.datetime.fromisoformat(det_date))
    return pa_reference_absolute(det, station_id)


def pa_reference_absolute(det, station_id):
    """Absolute PA reference point [x, y, z] (station position plus the ch1/ch2 midpoint)."""
    station = np.asarray(det.get_absolute_position(station_id), dtype=float)
    pa = pa_reference_point(det, station_id)
    return np.array([station[0] + pa[0], station[1] + pa[1], pa[2]])


def angular_separation(reco, truth, pa_z):
    """Great-circle angle in degrees between two (rho, phi_deg, z_abs) directions from the PA."""
    vecs = []
    for rho, phi, z in (reco, truth):
        p = np.radians(phi)
        vecs.append(np.array([rho * np.cos(p), rho * np.sin(p), z - pa_z]))
    a, b = vecs
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def wrap_dphi(dphi):
    """Wrap an azimuth difference in degrees into (-180, 180]."""
    return (dphi + 180.0) % 360.0 - 180.0


def run_number_from_path(path):
    """Run number of a per-run results file named `reco_run<N>.h5` or `reco_<N>.h5`, or None."""
    m = RESULT_FILE_RE.match(os.path.basename(path))
    return int(m.group(1)) if m else None


def cluster_bootstrap(values, clusters, statistic, n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES, seed=0, level=0.95):
    """Percentile interval of a statistic under resampling of whole clusters with replacement.

    Events of one cluster (a pulser run, an emitter position, an energy chunk, a synthetic
    source) share noise or geometry, so the resampling unit is the cluster, not the event.

    Args:
        values: Array whose first axis runs over events; a second axis may hold paired
            columns (reference, candidate) for a statistic of paired differences.
        clusters: Cluster label per event.
        statistic: Callable mapping a resampled `values` array to one number.
        n_resamples: Number of cluster resamples.
        seed: Seed of the resampling generator.
        level: Two-sided interval coverage.

    Returns:
        (point, low, high): the statistic of the full sample and the percentile interval.
    """
    values = np.asarray(values)
    index = np.unique(np.asarray(clusters), return_inverse=True)[1]
    members = [np.flatnonzero(index == i) for i in range(index.max() + 1)]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        pick = rng.integers(0, len(members), len(members))
        boots[b] = statistic(values[np.concatenate([members[i] for i in pick])])
    tail = 100.0 * (1.0 - level) / 2.0
    return float(statistic(values)), float(np.percentile(boots, tail)), float(np.percentile(boots, 100.0 - tail))


def truth_key(source_file, run_number):
    """Join key of a reconstruction row with simulation truth: (file basename, run number).

    The run number of a NuRadioMC event is its event group id, which is reused across
    files (different shower realisations of one vertex), so the file name is part of
    the key.
    """
    return os.path.basename(str(source_file)), int(run_number)


def join_truth(rows, truth):
    """Pair reconstruction rows with truth entries by `truth_key`.

    Args:
        rows: Iterable of dicts with `source_file` and `run_number`.
        truth: Mapping from `truth_key` tuples to truth values.

    Returns:
        (matched, n_unmatched) where matched is a list of (row, truth value).
    """
    matched = []
    n_unmatched = 0
    for row in rows:
        key = truth_key(row['source_file'], row['run_number'])
        if key in truth:
            matched.append((row, truth[key]))
        else:
            n_unmatched += 1
    return matched, n_unmatched


def shower_axis(zenith, azimuth):
    """Unit vector along the shower propagation for NuRadioMC (zenith, azimuth) in radians.

    NuRadioMC stores the direction the neutrino comes from; the shower propagates the
    opposite way.
    """
    return -np.array([np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)])


def axis_point(vertex, axis, distance):
    """Point `distance` metres down the shower axis from the vertex."""
    return np.asarray(vertex, dtype=float) + float(distance) * np.asarray(axis, dtype=float)


def bounded_blend_point(vertex, axis, d_had, d_em, e_had, e_em):
    """Bounded-blend Xmax point: HAD_Xmax + t (EM_Xmax - HAD_Xmax), t = E_EM / (E_HAD + E_EM) in [0, 1].

    With no electromagnetic energy (NC events) the point is the hadronic Xmax; the blend
    never leaves the segment between the two shower maxima.
    """
    had = axis_point(vertex, axis, d_had)
    em = axis_point(vertex, axis, d_em)
    total = float(e_had) + float(e_em)
    t = 0.0 if total <= 0 else float(np.clip(float(e_em) / total, 0.0, 1.0))
    return had + t * (em - had)


def enu_to_cylindrical(point, pa):
    """(rho, phi_deg in [0, 360), z_abs) of an absolute point seen from the PA reference."""
    dx, dy = point[0] - pa[0], point[1] - pa[1]
    return float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dy, dx)) % 360.0), float(point[2])


def string_snr(row, string):
    """Largest per-channel SNR of a helper string in a reconstruction row (0 when absent)."""
    return max(float(row.get(f'ch{ch}_snr', 0.0)) for ch in STRING_CHANNELS[string])


def pa_snr(row):
    """Largest phased-array channel SNR in a reconstruction row (0 when absent)."""
    return max(float(row.get(f'ch{ch}_snr', 0.0)) for ch in PA_CHANNELS)


def select_signal_events(rows, host_string, min_pa_snr=DEFAULT_MIN_PA_SNR, min_host_snr=DEFAULT_MIN_HOST_SNR):
    """Rows of a pulser run in which the pulser fired: PA SNR and host-string SNR above thresholds."""
    return [r for r in rows if pa_snr(r) >= min_pa_snr and string_snr(r, host_string) >= min_host_snr]


def device_from_comment(text):
    """Pulser device id named by a run comment (`fiber0` or `fiber1`), or None."""
    if not text:
        return None
    m = FIBER_RE.search(text)
    if m is None:
        return None
    return FIBER_DEVICE.get(f'fiber{m.group(1)}')


def loudest_helper_string(rows, min_pa_snr=DEFAULT_MIN_PA_SNR, margin=DEFAULT_SNR_VOTE_MARGIN):
    """Helper string that is louder than the other by at least `margin` in median SNR, or None.

    Only rows with PA SNR at or above `min_pa_snr` vote. The pulser's own string is the
    loudest for a fixed in-ice pulser; the vote is inconclusive when the two strings are
    within the margin, as at strongly attenuated runs.
    """
    voters = [r for r in rows if pa_snr(r) >= min_pa_snr]
    if not voters:
        return None
    b = float(np.median([string_snr(r, 'B') for r in voters]))
    c = float(np.median([string_snr(r, 'C') for r in voters]))
    if b - c >= margin:
        return 'B'
    if c - b >= margin:
        return 'C'
    return None


def truth_device(comment, rows, override=None):
    """Decide which pulser fired in a run from the run comment and the helper SNR pattern.

    The run comment names the fibre (`fiber0` is the helper-C pulser, `fiber1` the
    helper-B pulser); the SNR vote of `loudest_helper_string` must not contradict it.
    Without a comment the vote decides; when neither is conclusive the caller must pass
    `override`. The closest device to a reconstruction is never used.

    Returns:
        (device_id, source) with source one of 'override', 'comment', 'snr'.

    Raises:
        ValueError: when the comment and a conclusive SNR vote disagree, or when nothing
            names the device.
    """
    if override is not None:
        return int(override), 'override'
    from_comment = device_from_comment(comment)
    voted = loudest_helper_string(rows)
    if from_comment is not None:
        if voted is not None and PULSER_DEVICE_STRING[from_comment] != voted:
            raise ValueError(f"run comment names device {from_comment} (string "
                             f"{PULSER_DEVICE_STRING[from_comment]}) but string {voted} is loudest")
        return from_comment, 'comment'
    if voted is not None:
        return {v: k for k, v in PULSER_DEVICE_STRING.items()}[voted], 'snr'
    raise ValueError("no run comment and an inconclusive helper SNR pattern; pass the device explicitly")


def pulser_residual_rows(rows, truth, pa_z):
    """Residuals of reconstruction rows against one pulser truth (rho, phi_deg, z_abs)."""
    tx, ty = truth[0] * np.cos(np.radians(truth[1])), truth[0] * np.sin(np.radians(truth[1]))
    out = []
    for r in rows:
        rho, phi, z = float(r['rho']), float(r['phi']), float(r['z'])
        x, y = rho * np.cos(np.radians(phi)), rho * np.sin(np.radians(phi))
        out.append(dict(event=int(r['event_number']), rho=rho, phi=phi, z=z, corr=float(r['max_corr']),
                        drho=rho - truth[0], dphi=wrap_dphi(phi - truth[1]), dz=z - truth[2],
                        d3=float(np.hypot(np.hypot(x - tx, y - ty), z - truth[2])),
                        sep=angular_separation((rho, phi, z), truth, pa_z)))
    return out


def summarise_pulser_rows(res, close_deg=1.0, close_m=3.0):
    """Medians, 68th percentiles and close fractions of pulser residual rows."""
    if not res:
        return dict(n=0)
    med = lambda k: float(np.median([r[k] for r in res]))
    return dict(n=len(res), median_drho_m=med('drho'), median_dphi_deg=med('dphi'), median_dz_m=med('dz'),
                median_d3_m=med('d3'), median_sep_deg=med('sep'),
                p68_sep_deg=float(np.percentile(np.abs([r['sep'] for r in res]), 68)),
                frac_sep_lt_1deg=float(np.mean([r['sep'] < close_deg for r in res])),
                frac_d3_lt_3m=float(np.mean([r['d3'] < close_m for r in res])), median_corr=med('corr'))
