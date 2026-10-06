"""The dedispersion module evaluates every station antenna where the antenna responds."""

import numpy as np
import pytest

from NuRadioReco.detector.antennapattern import AntennaPatternProvider
from NuRadioReco.modules.channelAntennaDedispersion import channelAntennaDedispersion

from conftest import STATION

MIN_VEL_M = 0.005


@pytest.fixture(scope='module')
def module():
    return channelAntennaDedispersion()


@pytest.fixture(scope='module')
def provider():
    return AntennaPatternProvider()


@pytest.mark.parametrize('ch', list(range(24)))
def test_sensitive_direction_is_not_a_null(det, module, provider, ch):
    """The direction used for each channel has a clearly non-zero effective length in 150-600 MHz."""
    name = det.get_antenna_model(STATION, ch)
    match = next(k for k in module.antennas_most_sensitive_directions if k.lower() in name.lower())
    zen_ori, az_ori, zen_rot, az_rot = det.get_antenna_orientation(STATION, ch)
    zen = zen_ori + module.antennas_most_sensitive_directions[match][0]
    az = az_ori + module.antennas_most_sensitive_directions[match][1]
    ff = np.fft.rfftfreq(2048, 1 / 3.2)
    band = (ff > 0.15) & (ff < 0.6)
    vel = provider.load_antenna_pattern(name).get_antenna_response_vectorized(
        ff, zen, az, zen_ori, az_ori, zen_rot, az_rot)
    magnitude = np.mean(np.abs(vel['theta'][band]) + np.abs(vel['phi'][band]))
    assert magnitude > MIN_VEL_M, (ch, name, magnitude)
