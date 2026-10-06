"""Detector description and data provider of an RNO-G processing script, built from its config values."""

import datetime
import logging

import NuRadioReco.detector.detector as detector
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.RNO_G.dataProviderNuRadio import dataProviderNuRadio
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG


def init_detector(config):
    """Build the detector description of a station and set it to the date of the config.

    With ``detector_file`` (an exported RNO-G detector description) the file is read for the
    station alone; without it the RNO-G database is queried.

    Args:
        config: Dict with ``station_id`` and optionally ``detector_file`` and ``detector_date`` (ISO, default 2022-10-01).

    Returns:
        The detector, updated to ``detector_date``.
    """
    det_file = config.get('detector_file', None)
    det_date_str = config.get('detector_date', '2022-10-01')
    det_date = datetime.datetime.fromisoformat(det_date_str)
    station_id = config['station_id']

    if det_file:
        det = rnog_detector.Detector(
            detector_file=det_file,
            log_level=logging.WARNING,
            select_stations=station_id,
        )
    else:
        det = detector.Detector(source="rnog_mongo")

    det.update(det_date)
    return det


def select_data_provider(input_file, det, reader_kwargs=None, preprocessor_config=None):
    """Open an input with the data provider that reads its format.

    A ``.nur`` file is read by ``dataProviderNuRadio``. Anything else (an RNO-G run folder
    or ROOT file) is read by ``dataProviderRNOG``, with ``reader_kwargs`` passed to
    ``readRNOGData.begin`` over the mattak defaults ``read_daq_status: False`` and
    ``backend: uproot``; the ``mattak_kwargs`` entry of the caller is merged into these
    key by key, so a caller changes only the keys it names.

    Args:
        input_file: Path of a ``.nur`` file, an RNO-G run folder or a ROOT file.
        det: Detector description, for example from ``init_detector``.
        reader_kwargs: Keyword arguments of ``readRNOGData.begin`` or None; not used for a ``.nur`` file.
        preprocessor_config: Overrides of the ``channelPreprocessor`` defaults, or None.

    Returns:
        The provider, with ``begin`` called.
    """
    if input_file.endswith('.nur'):
        provider = dataProviderNuRadio()
        provider.begin(input_file, det, preprocessor_config=preprocessor_config)
        return provider
    user = reader_kwargs or {}
    mattak_kwargs = {'read_daq_status': False, 'backend': 'uproot', **(user.get('mattak_kwargs') or {})}
    provider = dataProviderRNOG()
    provider.begin(input_file, det, reader_kwargs={**user, 'mattak_kwargs': mattak_kwargs},
                   preprocessor_config=preprocessor_config)
    return provider
