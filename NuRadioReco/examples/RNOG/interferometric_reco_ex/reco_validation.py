"""Channel groups, calibration-pulser truth and per-channel SNR for the 3D interferometric reco driver."""

import numpy as np

PA_CHANNELS = [0, 1, 2, 3]
HELPER_B_CHANNELS = [9, 10]
HELPER_C_CHANNELS = [22, 23]
HELPER_CHANNELS = [9, 10, 22, 23]
SHALLOW_CHANNELS = [5, 6, 7]

STRING_CHANNELS = {'B': HELPER_B_CHANNELS, 'C': HELPER_C_CHANNELS}
STRING_HPOL_CHANNEL = {'B': 11, 'C': 21}
PULSER_DEVICE_STRING = {0: 'C', 1: 'B'}
FIBER_DEVICE = {'fiber0': 0, 'fiber1': 1}
DEEP_PULSER_Z_MAX = -50.0


def pa_reference_point(det, station_id):
    """Return the phased-array reference point as [x_rel, y_rel, z_abs].

    The reconstruction centres its grid on the midpoint of channels 1 and 2, with the
    horizontal coordinates relative to the station reference point and the absolute depth.
    """
    station_z = float(det.get_absolute_position(station_id)[2])
    pa = 0.5 * (np.asarray(det.get_relative_position(station_id, 1), dtype=float)
                + np.asarray(det.get_relative_position(station_id, 2), dtype=float))
    pa[2] += station_z
    return pa


def pulser_truth(det, station_id, device_id):
    """Return the (rho, phi_deg, z_abs) of a calibration pulser in the reconstruction frame.

    The detector description stores device positions relative to the station reference
    point, like the channel positions, so the absolute depth is the device z plus the
    station z. rho and phi are measured from the phased-array reference point of
    `pa_reference_point`, the same reference the reconstruction uses.

    Args:
        det: Detector description, updated to the epoch of interest.
        station_id: Station number.
        device_id: Device id in the detector description (0 is the helper-C pulser and
            1 the helper-B pulser at the 2022 stations).

    Returns:
        Tuple (rho in m, phi in degrees in [0, 360), z absolute in m).
    """
    station_z = float(det.get_absolute_position(station_id)[2])
    pa = pa_reference_point(det, station_id)
    dev = np.asarray(det.get_relative_position_device(station_id, device_id), dtype=float)
    dx, dy = dev[0] - pa[0], dev[1] - pa[1]
    return (float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dy, dx)) % 360.0),
            float(dev[2] + station_z))


def deep_pulser_devices(det, station_id, z_max=DEEP_PULSER_Z_MAX):
    """Return {device_id: device_name} of the in-ice pulsers of a station.

    A device counts as an in-ice pulser when its name contains "pulser" and its
    station-relative z lies below `z_max`.
    """
    devices = {}
    for device_id, name in det.get_devices(station_id).items():
        if 'pulser' not in name.lower():
            continue
        if float(det.get_relative_position_device(station_id, device_id)[2]) < z_max:
            devices[int(device_id)] = name
    return devices


def pulser_host_string(det, station_id, device_id, tolerance_m=0.5):
    """Return the helper string ('B' or 'C') whose channels share the pulser's hole.

    The string is identified by the horizontal position of its HPol channel; the
    device name is not used.

    Raises:
        ValueError: if no helper string sits within `tolerance_m` of the device.
    """
    dev = np.asarray(det.get_relative_position_device(station_id, device_id), dtype=float)
    for string, hpol in STRING_HPOL_CHANNEL.items():
        pos = np.asarray(det.get_relative_position(station_id, hpol), dtype=float)
        if np.hypot(dev[0] - pos[0], dev[1] - pos[1]) < tolerance_m:
            return string
    raise ValueError(f'device {device_id} of station {station_id} is not on a helper string')


def preprocessing_channels(config):
    """Return the sorted union of every channel the driver and the reconstruction read.

    Covers config['channels'], every polarization group and, when the
    plane-wave fallback is enabled, the helper channels it tests and the
    channels of the fallback configuration.

    Args:
        config: Reco config dict.

    Returns:
        Sorted list of channel ids.
    """
    used = set(config['channels'])
    for group in (config.get('polarization_groups') or {}).values():
        used.update(group)
    if config.get('plane_wave_fallback', False):
        used.update(HELPER_CHANNELS)
        used.update(PA_CHANNELS + SHALLOW_CHANNELS)
    return sorted(used)


def compute_channel_snrs(volt_arrays, channels):
    """Per-channel SNR from preprocessed voltage traces.

    Uses split-trace noise RMS (lowest segments) to avoid including
    signal in the noise estimate, and peak-to-peak amplitude within a
    coincidence window for the signal estimate. Matches the definition
    in feature_extraction/gather_variables.py.

    Args:
        volt_arrays: List of voltage trace arrays, one per channel.
        channels: List of channel IDs (same order as volt_arrays).

    Returns:
        Dict mapping channel_id to SNR value.
    """
    from NuRadioReco.utilities.trace_utilities import (
        get_split_trace_noise_RMS, get_signal_to_noise_ratio)

    snrs = {}
    for v, ch in zip(volt_arrays, channels):
        noise_rms = get_split_trace_noise_RMS(v)
        snrs[ch] = float(get_signal_to_noise_ratio(v, noise_rms)
                         if noise_rms > 0 else 0.0)
    return snrs

