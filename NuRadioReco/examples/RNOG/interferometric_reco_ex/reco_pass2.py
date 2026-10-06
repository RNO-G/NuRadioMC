"""Pass-2 configuration of the 3D reconstruction driver: re-search volume, arrival angles and pair signs.

The driver's ``rx`` and ``rxtx`` modes re-search a window around the pass-1 answer after
removing the antenna phase responses at the arrival angles of the pass-1 position. These
helpers build that re-search configuration, the arrival angles of the first-arriving ray
for sources in the ice or above it, and the per-pair signs of the cross-type correlation.
Each option keeps the previous driver behaviour when its key is absent.
"""

import copy
import itertools
import logging

import numpy as np

from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities.medium import greenland_simple

ice = greenland_simple()

PASS2_VOLUMES = ('template', 'pass1')
RX_ARRIVAL_MODES = ('direct', 'first_arrival')
CROSS_TYPE_SIGN_MODES = ('signed', 'abs')


def ray_tracer_backend():
    """Backend of the analytic ray tracer that the driver's tracers resolve to.

    NuRadioMC uses the C++ tracer when its extension imports (or compiles on first
    import) and otherwise falls back to the python version without raising. The two
    agree only to numerical precision, which shows in the rx/rxtx arrival angles, so
    the driver records the backend of every run.

    Returns:
        'cpp' or 'python'.
    """
    tracer = propagation.get_propagation_module('analytic')(ice, log_level=logging.WARNING)
    return 'cpp' if tracer.use_cpp else 'python'


def pass2_options(config):
    """Return the driver's pass-2 options, checked.

    ``pass2_volume``: ``template`` (default) re-searches the in-ice template volume
    of ``pass2_template``; ``pass1`` re-searches the pass-1 configuration itself
    (its tables, ``allow_above_surface``, split z grid and search keys) with the
    limits set to the ``pass2_window`` around the pass-1 answer, clamped to the
    pass-1 limits (``pass2_search_config``).
    ``rx_arrival_mode``: ``direct`` (default) takes the arrival angles of the
    driver's ``compute_arrival_angles`` (the direct ray from a position placed around
    the station origin); ``first_arrival`` those of ``compute_first_arrival_angles``.
    ``cross_type_sign_mode``: ``signed`` (default) correlates every pair with its
    sign; ``abs`` scores every pair of two antenna types (VPol, HPol, LPDA) by the
    absolute value of its raw correlation (``cross_type_pair_signs``), in both passes.

    Args:
        config: Reconstruction config dict.

    Returns:
        (pass2_volume, rx_arrival_mode, cross_type_sign_mode).

    Raises:
        ValueError: On an unknown value, ``pass2_volume: pass1`` together with
            ``z_profile_step``, or ``cross_type_sign_mode: abs`` together with
            ``polarization_groups`` or ``pair_signs``.
    """
    volume = config.get('pass2_volume', 'template')
    rx_mode = config.get('rx_arrival_mode', 'direct')
    sign_mode = config.get('cross_type_sign_mode', 'signed')
    for key, value, allowed in (('pass2_volume', volume, PASS2_VOLUMES),
                                ('rx_arrival_mode', rx_mode, RX_ARRIVAL_MODES),
                                ('cross_type_sign_mode', sign_mode, CROSS_TYPE_SIGN_MODES)):
        if value not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
    if volume == 'pass1' and config.get('z_profile_step') is not None:
        raise ValueError("pass2_volume: pass1 does not support z_profile_step")
    if sign_mode != 'signed' and (config.get('polarization_groups') or config.get('pair_signs') is not None):
        raise ValueError("cross_type_sign_mode sets pair_signs itself; it cannot be combined "
                         "with polarization_groups or pair_signs")
    return volume, rx_mode, sign_mode


def antenna_type(det, station_id, ch):
    """Antenna type of a channel from its antenna model name: ``lpda``, ``hpol`` or ``vpol``."""
    name = det.get_antenna_model(station_id, ch).lower()
    if 'lpda' in name:
        return 'lpda'
    if 'hpol' in name:
        return 'hpol'
    return 'vpol'


def cross_type_pair_signs(channels, types):
    """Per-pair signs for ``cross_type_sign_mode: abs``, in ``itertools.combinations`` order.

    Args:
        channels: Channel IDs in reconstruction order.
        types: Channel ID -> antenna type.

    Returns:
        List with ``'abs'`` for a pair of two antenna types and 1 for a pair of one type.
    """
    return ['abs' if types[a] != types[b] else 1 for a, b in itertools.combinations(channels, 2)]


def compute_first_arrival_angles(rho, phi_deg, z, station_id, det, channels, surface_channels,
                                 tracer=None):
    """Per-channel arrival angles of the first-arriving ray from a pass-1 position.

    The position is placed as the reconstruction defines it: (rho, phi) around the
    phased-array axis (midpoint of channels 1 and 2), z absolute. Each channel takes
    the receive vector of the solution with the smallest travel time of the
    ``air_ice`` propagator (greenland_simple), which traces a straight air leg and
    refracts at a flat surface when the position is above the ice, and delegates to
    the analytic tracer otherwise. A channel in ``surface_channels`` (the surface
    LPDAs) takes the straight line from the antenna to a position above the surface
    instead. A channel without a solution is left out.

    Args:
        rho: Horizontal distance from the phased-array axis in m.
        phi_deg: Azimuth in deg, counter-clockwise from east.
        z: Absolute z in m.
        station_id: Station ID.
        det: Detector description.
        channels: Channel IDs.
        surface_channels: Channel IDs given the straight air path.
        tracer: ``air_ice`` propagator instance to reuse, or None to build one.

    Returns:
        Dict channel ID -> (zenith_rad, azimuth_rad) of the direction the signal
        arrives from.
    """
    stn_abs = np.array(det.get_absolute_position(station_id))
    pa_rel = 0.5 * (np.array(det.get_relative_position(station_id, 1))
                    + np.array(det.get_relative_position(station_id, 2)))
    phi_rad = np.radians(phi_deg)
    source_abs = np.array([stn_abs[0] + pa_rel[0] + rho * np.cos(phi_rad),
                           stn_abs[1] + pa_rel[1] + rho * np.sin(phi_rad), z])
    if tracer is None:
        tracer = propagation.get_propagation_module('air_ice')(ice, log_level=logging.WARNING)
    angles = {}
    for ch_id in channels:
        ch_abs = stn_abs + np.array(det.get_relative_position(station_id, ch_id))
        if ch_id in surface_channels and z > 0:
            rv = source_abs - ch_abs
        else:
            tracer.set_start_and_end_point(source_abs, ch_abs)
            tracer.find_solutions()
            n_sol = tracer.get_number_of_solutions()
            if n_sol == 0:
                continue
            first = min(range(n_sol), key=tracer.get_travel_time)
            rv = np.asarray(tracer.get_receive_vector(first), dtype=float)
        angles[ch_id] = (np.arccos(rv[2] / np.linalg.norm(rv)), np.arctan2(rv[1], rv[0]))
    return angles


def pass2_template(config, station_id, channels, p2_steps):
    """In-ice template configuration of the pass-2 re-search (``pass2_volume: template``).

    With ``cross_type_sign_mode: abs`` the template also carries the ``pair_signs``
    the driver set on ``config``, and it carries ``channel_position_shift`` when set.
    """
    p2_hierarchical = config.get('pass2_hierarchical', False)
    template = {
        'station_id': station_id, 'channels': channels,
        'coord_system': 'cylindrical',
        'hierarchical': p2_hierarchical,
        'multi_ray_types': config.get('multi_ray_types', False),
        'multiray_combo_mode': config.get('multiray_combo_mode', 'grouped'),
        'time_delay_tables': config['time_delay_tables'],
        'interp_method': config.get('interp_method', 'linear'),
        'snr_pair_weighting': config.get('snr_pair_weighting', True),
        'hilbert_envelope_mode': config.get('hilbert_envelope_mode', 'traces'),
        'correlation_normalization': config.get('correlation_normalization', 'normalized'),
        'apply_hann_window': config.get('apply_hann_window', False),
        'limits': [1, 100, 0, 360, -100, 0],
        'step_sizes': p2_steps,
        'n_rho': 0,
        'n_z': config.get('pass2_n_z', config.get('n_z', 0)),
        'z_spacing': config.get('z_spacing', 'linear'),
        'z_surface_offset': config.get('z_surface_offset', 0.1),
    }
    if p2_hierarchical:
        template.update({
            'coarse_limits': [1, 100, 0, 360, -100, 0],
            'coarse_n_rho': 0,
            'coarse_n_z': config.get('pass2_coarse_n_z', 0),
            'coarse_step_sizes': config.get(
                'pass2_coarse_step_sizes', [2, 1, 2]),
            'coarse_n_peaks': config.get('pass2_coarse_n_peaks', 3),
            'coarse_peak_separation': config.get(
                'pass2_coarse_peak_separation', [5, 3, 5]),
            'refine_window': config.get(
                'pass2_refine_window', [5, 3, 5]),
            'refine_step_sizes': p2_steps,
        })
    if 'table_name_pattern' in config:
        template['table_name_pattern'] = config['table_name_pattern']
    if 'multiray_table_name_pattern' in config:
        template['multiray_table_name_pattern'] = config['multiray_table_name_pattern']
    if config.get('cross_type_sign_mode', 'signed') != 'signed':
        template['pair_signs'] = config['pair_signs']
    if config.get('channel_position_shift'):
        template['channel_position_shift'] = config['channel_position_shift']
    return template


def pass2_window_limits(limits, r1, p2_window):
    """Limits of the pass-2 window around the pass-1 answer, clamped to ``limits``.

    rho and z are clamped to the search limits (rho never below 1 m); phi is not
    clamped, since it wraps.
    """
    return [max(limits[0], 1.0, r1['rho'] - p2_window[0]),
            min(limits[1], r1['rho'] + p2_window[0]),
            r1['phi'] - p2_window[1],
            r1['phi'] + p2_window[1],
            max(limits[4], r1['z'] - p2_window[2]),
            min(limits[5], r1['z'] + p2_window[2])]


def pass2_search_config(config, template, r1, p2_window):
    """Configuration of the pass-2 re-search around the pass-1 answer ``r1``.

    With ``pass2_volume: pass1`` it is a copy of the pass-1 configuration whose
    ``limits`` and ``coarse_limits`` are the clamped window of
    ``pass2_window_limits``, so pass 2 keeps the pass-1 tables, volume keys
    (``allow_above_surface``, ``z_grid_below``/``z_grid_above``) and search keys;
    a window that straddles z = 0 is searched in both blocks, as in pass 1.
    Otherwise it is the in-ice template, whose volume ends at the surface (no
    ``allow_above_surface``): the window is the pass-2 window around the pass-1
    answer clamped to the pass-1 search volume (``limits``, rho never below 1 m)
    and at z = 0, so pass 2 searches only where pass 1 may and belongs to the
    in-ice block.

    Raises:
        ValueError: With the template, for a pass-1 answer above the surface,
            which only ``pass2_volume: pass1`` can re-search.
    """
    if config.get('pass2_volume', 'template') == 'pass1':
        cfg = copy.deepcopy(config)
        limits = pass2_window_limits(config['limits'], r1, p2_window)
        cfg['limits'] = limits
        cfg['coarse_limits'] = list(limits)
        return cfg
    if r1['z'] > 0:
        raise ValueError(f"the pass-1 answer lies above the surface (z = {r1['z']:.2f} m), which the "
                         "in-ice pass-2 template cannot re-search; set pass2_volume: pass1")
    cfg = dict(template)
    p2_limits = pass2_window_limits(config.get('limits', config.get('coarse_limits')), r1, p2_window)
    p2_limits[5] = min(p2_limits[5], 0.0)
    cfg['limits'] = p2_limits
    if template['hierarchical']:
        cfg['coarse_limits'] = p2_limits
    return cfg
