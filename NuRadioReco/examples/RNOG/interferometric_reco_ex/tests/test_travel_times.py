"""The public travel-time lookup of the reconstruction class.

`travel_times` is the lookup that callers outside the class made through three private
methods (source matrix, per-channel table coordinates, table interpolator). It returns
their numbers bit for bit, values in ns from a position about the phased-array centre,
agrees with the single-point method of the summed waveform to rounding, and keeps NaN in
the shadow zone of a channel and minus infinity above the surface.
"""

import numpy as np

from conftest import rng_sources
from synthetic import VPOL_CHANNELS, cylindrical_to_enu, same_value

C_M_PER_NS = 0.299792458
POSITIONS = rng_sources(60, 20261006) + [(250.0, 0.0, 0.0), (1.0, 359.9, -100.0), (0.2, 45.0, -50.0)]
# 147 m out and 9 m deep: no ray reaches channel 6 in the in-ice tables
SHADOWED = (147.3, 124.3, -8.8)


def _three_private_calls(reco, rho, phi_deg, z, channels):
    """The lookup as callers wrote it before the public method."""
    src = reco._build_source_enu_matrix(np.array([rho]), np.array([np.radians(phi_deg)]), np.array([z]))
    coords = reco._compute_rho_and_coords(src, channels)
    return {ch: float(reco._interpolators[ch].interp(coords[ch])[0]) for ch in channels}


def test_travel_times_equal_the_private_calls_bit_for_bit(reco):
    """Every value equals the three private calls exactly, one float per requested channel."""
    n_finite = 0
    for rho, phi, z in POSITIONS:
        new = reco.travel_times(rho, phi, z, VPOL_CHANNELS)
        old = _three_private_calls(reco, rho, phi, z, VPOL_CHANNELS)
        assert list(new) == VPOL_CHANNELS and all(type(v) is float for v in new.values())
        assert all(same_value(new[ch], old[ch]) for ch in VPOL_CHANNELS), (rho, phi, z)
        n_finite += sum(np.isfinite(v) for v in new.values())
    assert n_finite > 0.9 * len(POSITIONS) * len(VPOL_CHANNELS)
    assert list(reco.travel_times(80.0, 120.0, -40.0, [9, 0])) == [9, 0]


def test_travel_times_are_ns_from_a_position_about_the_phased_array(reco, tables, ant_locs, pa):
    """The values are the table values at the position's ENU point, between the straight-line time in vacuum and twice it."""
    for rho, phi, z in POSITIONS[:20]:
        src = cylindrical_to_enu(rho, phi, z, pa)
        times = reco.travel_times(rho, phi, z, VPOL_CHANNELS)
        for ch in (ch for ch in VPOL_CHANNELS if np.isfinite(times[ch])):
            assert abs(times[ch] - tables.travel_time(ch, src, ant_locs[ch])) < 1e-9, (ch, rho, phi, z)
            vacuum = np.linalg.norm(src - ant_locs[ch]) / C_M_PER_NS
            assert vacuum < times[ch] < 2.0 * vacuum, (ch, rho, phi, z)


def test_single_point_method_agrees_to_rounding(reco):
    """The summed waveform's own lookup gives the same times within 1e-9 ns."""
    for rho, phi, z in POSITIONS:
        new = reco.travel_times(rho, phi, z, VPOL_CHANNELS)
        single = reco._compute_travel_times_single_point(rho, phi, z, VPOL_CHANNELS)
        for ch in VPOL_CHANNELS:
            assert abs(new[ch] - single[ch]) < 1e-9 or (np.isnan(new[ch]) and np.isnan(single[ch])), (ch, rho, phi, z)


def test_values_without_a_table_solution(reco):
    """NaN in the shadow zone of a channel; minus infinity for every channel above the range of the in-ice tables."""
    times = reco.travel_times(*SHADOWED, VPOL_CHANNELS)
    assert np.isnan(times[6]) and all(np.isfinite(times[ch]) for ch in VPOL_CHANNELS if ch != 6)
    assert all(v == -np.inf for v in reco.travel_times(50.0, 10.0, 5.0, VPOL_CHANNELS).values())
