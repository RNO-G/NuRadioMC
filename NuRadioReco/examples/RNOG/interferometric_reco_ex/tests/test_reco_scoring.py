"""Scoring arithmetic of the evaluation scripts on synthetic tables.

The truth join is keyed by (source file, run number) because NuRadioMC reuses event
group ids across files; the PA reference is the ch1/ch2 midpoint in absolute
coordinates; the bounded blend interpolates between the hadronic and electromagnetic
shower maxima with a weight clipped to the segment; the pulser device comes from the
run comment and the helper SNR pattern, never from the closest device; the run comment is
found from the harness file naming; the cluster bootstrap resamples whole clusters.
"""

import os

import numpy as np
import pytest

from evaluate_pulser_runs import run_comment
from reco_scoring import (angular_separation, axis_point, bounded_blend_point, cluster_bootstrap,
                          device_from_comment, enu_to_cylindrical, join_truth, loudest_helper_string,
                          pa_reference_absolute, pulser_residual_rows, run_number_from_path,
                          select_signal_events, shower_axis, string_snr, summarise_pulser_rows,
                          truth_device, truth_key, wrap_dphi)


class StubDetector:
    """Station at (100, 200, -3) with channels 1 and 2 at -93 and -92 m relative."""

    def get_absolute_position(self, station_id):
        """Absolute station position."""
        return np.array([100.0, 200.0, -3.0])

    def get_relative_position(self, station_id, channel_id):
        """Relative channel position."""
        return np.array([0.0, 0.0, {1: -93.0, 2: -92.0}[channel_id]])


def _row(**kw):
    """A reconstruction row with per-channel SNR defaults of 1."""
    row = {f'ch{ch}_snr': 1.0 for ch in (0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23)}
    row.update(rho=30.0, phi=100.0, z=-95.0, max_corr=0.5, event_number=0)
    row.update(kw)
    return row


def test_truth_key_strips_directories_and_join_keeps_files_apart():
    """The same run number in two files joins to two different truths; unmatched rows are counted."""
    assert truth_key('/a/b/lgE17.0_c0001.nur', np.int64(5)) == ('lgE17.0_c0001.nur', 5)
    truth = {('f1.nur', 7): 'A', ('f2.nur', 7): 'B'}
    rows = [dict(source_file='/x/f1.nur', run_number=7), dict(source_file='f2.nur', run_number=7),
            dict(source_file='f3.nur', run_number=7)]
    matched, n_unmatched = join_truth(rows, truth)
    assert [t for _, t in matched] == ['A', 'B'] and n_unmatched == 1


def test_pa_reference_and_angular_separation():
    """The PA reference is the absolute ch1/ch2 midpoint; separations follow the geometry."""
    pa = pa_reference_absolute(StubDetector(), 23)
    assert np.allclose(pa, [100.0, 200.0, -95.5])
    assert angular_separation((30.0, 45.0, -80.0), (30.0, 45.0, -80.0), pa[2]) == 0.0
    assert abs(angular_separation((30.0, 0.0, pa[2]), (30.0, 90.0, pa[2]), pa[2]) - 90.0) < 1e-9
    assert abs(angular_separation((30.0, 0.0, pa[2]), (30.0, 180.0, pa[2]), pa[2]) - 180.0) < 1e-9
    assert abs(angular_separation((10.0, 0.0, pa[2] + 10.0), (10.0, 0.0, pa[2]), pa[2]) - 45.0) < 1e-9
    assert wrap_dphi(359.0) == -1.0 and wrap_dphi(-181.0) == 179.0
    assert np.allclose(enu_to_cylindrical([103.0, 204.0, -50.0], pa), (5.0, np.degrees(np.arctan2(4, 3)), -50.0))


def test_bounded_blend_stays_on_the_segment():
    """t = E_EM / (E_HAD + E_EM) clipped to [0, 1] interpolates between the two maxima."""
    vertex = np.array([0.0, 0.0, -1.0])
    axis = np.array([0.0, 0.6, -0.8])
    had, em = axis_point(vertex, axis, 10.0), axis_point(vertex, axis, 30.0)
    assert np.allclose(had, [0.0, 6.0, -9.0]) and np.allclose(em, [0.0, 18.0, -25.0])
    assert np.allclose(bounded_blend_point(vertex, axis, 10.0, 30.0, 1e17, 0.0), had)
    assert np.allclose(bounded_blend_point(vertex, axis, 10.0, 30.0, 0.0, 1e17), em)
    assert np.allclose(bounded_blend_point(vertex, axis, 10.0, 30.0, 1e17, 1e17), axis_point(vertex, axis, 20.0))
    assert np.allclose(bounded_blend_point(vertex, axis, 10.0, 30.0, 1e17, 3e17), axis_point(vertex, axis, 25.0))
    assert np.allclose(bounded_blend_point(vertex, axis, 10.0, 30.0, 0.0, 0.0), had)
    assert np.allclose(shower_axis(0.0, 0.0), [0.0, 0.0, -1.0])
    assert np.allclose(shower_axis(np.pi / 2, 0.0), [-1.0, 0.0, 0.0])


def test_signal_selection_uses_pa_and_host_string_snr():
    """PA SNR and the host string's largest channel SNR gate the selection."""
    rows = [_row(ch0_snr=15.0, ch22_snr=25.0), _row(ch0_snr=15.0, ch23_snr=10.0), _row(ch1_snr=3.0, ch22_snr=40.0)]
    assert string_snr(rows[0], 'C') == 25.0 and string_snr(rows[0], 'B') == 1.0
    assert select_signal_events(rows, 'C') == [rows[0]]
    assert select_signal_events(rows, 'C', min_host_snr=8.0) == rows[:2]
    assert select_signal_events(rows, 'B') == []


def test_truth_device_from_comment_and_snr_pattern():
    """fiber0 is the helper-C pulser, fiber1 the helper-B pulser; the SNR vote must agree."""
    assert device_from_comment('Calibration pulsing (PA trigger, 0dB attenuation, fiber0, pulser)') == 0
    assert device_from_comment('Calibration run (PA trigger, 5dB attenuation, fiber1, pulser)') == 1
    assert device_from_comment('forced triggers') is None and device_from_comment(None) is None
    loud_c = [_row(ch0_snr=12.0, ch9_snr=12.0, ch22_snr=30.0) for _ in range(3)]
    loud_b = [_row(ch0_snr=12.0, ch9_snr=30.0, ch22_snr=12.0) for _ in range(3)]
    even = [_row(ch0_snr=12.0, ch9_snr=11.0, ch22_snr=12.0) for _ in range(3)]
    assert loudest_helper_string(loud_c) == 'C' and loudest_helper_string(loud_b) == 'B'
    assert loudest_helper_string(even) is None and loudest_helper_string([_row(ch0_snr=2.0)]) is None
    assert truth_device('fiber0', loud_c) == (0, 'comment')
    assert truth_device('fiber1', even) == (1, 'comment')
    assert truth_device(None, loud_c) == (0, 'snr')
    assert truth_device(None, even, override=1) == (1, 'override')
    with pytest.raises(ValueError):
        truth_device('fiber0', loud_b)
    with pytest.raises(ValueError):
        truth_device(None, even)


def test_pulser_residual_rows_and_summary():
    """A row at the truth has zero residuals; the summary reports medians and close fractions."""
    truth = (34.0, 120.0, -95.6)
    rows = [_row(rho=34.0, phi=120.0, z=-95.6, max_corr=0.7, event_number=5),
            _row(rho=34.0, phi=121.0, z=-95.6, max_corr=0.6, event_number=6)]
    res = pulser_residual_rows(rows, truth, -95.0)
    assert res[0]['drho'] == 0.0 and res[0]['dphi'] == 0.0 and res[0]['dz'] == 0.0 and res[0]['d3'] == 0.0
    assert abs(res[0]['sep']) < 1e-5 and res[1]['dphi'] == 1.0 and res[1]['sep'] > 0.9
    summary = summarise_pulser_rows(res)
    assert summary['n'] == 2 and summary['median_dphi_deg'] == 0.5 and summary['frac_d3_lt_3m'] == 1.0
    assert abs(summary['median_corr'] - 0.65) < 1e-12 and summarise_pulser_rows([]) == dict(n=0)


def test_run_comment_follows_the_harness_file_naming(tmp_path):
    """`reco_run1000.h5` and `reco_1000.h5` both lead to `run1000/aux/comment.txt`."""
    assert run_number_from_path('/r/pulser/reco_run1000.h5') == 1000
    assert run_number_from_path('reco_2090.h5') == 2090
    assert run_number_from_path('reco_chunk0.h5') is None and run_number_from_path('reco_run1000.h5.bak') is None
    aux = tmp_path / 'run1000' / 'aux'
    os.makedirs(aux)
    (aux / 'comment.txt').write_text('Calibration pulsing (PA trigger, 0dB attenuation, fiber0, pulser)\n')
    for name in ('reco_run1000.h5', 'reco_1000.h5'):
        text = run_comment(str(tmp_path), run_number_from_path(name))
        assert text and device_from_comment(text) == 0, name
    assert run_comment(str(tmp_path), run_number_from_path('reco_run2090.h5')) is None
    assert run_comment(str(tmp_path), None) is None and run_comment(None, 1000) is None


def test_cluster_bootstrap_resamples_clusters():
    """Singleton clusters give the plain bootstrap; whole-cluster resampling widens correlated data."""
    rng = np.random.default_rng(1)
    values = rng.normal(0.0, 1.0, 50)
    point, low, high = cluster_bootstrap(values, np.arange(50), np.median, n_resamples=500, seed=3)
    assert point == np.median(values) and low <= point <= high
    assert cluster_bootstrap(values, np.arange(50), np.median, n_resamples=500, seed=3) == (point, low, high)
    check = np.random.default_rng(3)
    plain = [np.median(values[check.integers(0, 50, 50)]) for _ in range(500)]
    np.testing.assert_allclose((low, high), (np.percentile(plain, 2.5), np.percentile(plain, 97.5)), rtol=1e-12)
    clustered = np.repeat(np.arange(10, dtype=float), 20)
    labels = np.repeat(np.arange(10), 20)
    _, lo_c, hi_c = cluster_bootstrap(clustered, labels, np.mean, n_resamples=500, seed=0)
    _, lo_e, hi_e = cluster_bootstrap(clustered, np.arange(200), np.mean, n_resamples=500, seed=0)
    assert hi_c - lo_c > 2.0 * (hi_e - lo_e)
    paired = np.column_stack([values, values + 1.0])
    diff = lambda v: np.median(v[:, 1]) - np.median(v[:, 0])
    np.testing.assert_allclose(cluster_bootstrap(paired, labels[:50], diff, n_resamples=100), (1.0, 1.0, 1.0), rtol=1e-12)
