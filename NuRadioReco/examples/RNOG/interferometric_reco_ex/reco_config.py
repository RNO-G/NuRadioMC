"""Config layout checks of the 3D reconstruction driver that need no data reader."""

from NuRadioReco.modules.RNO_G.channelPreprocessor import channelPreprocessor

TOP_LEVEL_KEYS_SHARED_WITH_PREPROCESSOR = ('apply_upsampling', 'channels')
MATTAK_DEFAULTS = {'read_daq_status': False, 'backend': 'uproot'}


def misplaced_preprocessor_keys(config):
    """Preprocessing keys at the top level of a config, where the driver does not read them.

    The driver passes only the ``preprocessor`` block to channelPreprocessor. At the top
    level it reads ``apply_upsampling`` itself, and ``channels`` is the reconstruction's
    channel list; any other channelPreprocessor key at the top level has no effect.

    Returns:
        Sorted list of the misplaced keys.
    """
    return sorted(set(config) & (set(channelPreprocessor._DEFAULT_CONFIG)
                                 - set(TOP_LEVEL_KEYS_SHARED_WITH_PREPROCESSOR)))


def reader_options(config):
    """Reader options of the driver for run folders: its defaults with the config's ``reader_kwargs`` merged in.

    ``reader_kwargs`` holds keyword arguments of ``readRNOGData.begin``, for example
    ``select_triggers``. Its ``mattak_kwargs`` entry is merged key by key into the
    driver's defaults (``MATTAK_DEFAULTS``), so a config changes only the keys it names.

    Returns:
        Dict for the ``reader_kwargs`` argument of ``dataProviderRNOG.begin``.
    """
    user = config.get('reader_kwargs') or {}
    return {**user, 'mattak_kwargs': {**MATTAK_DEFAULTS, **(user.get('mattak_kwargs') or {})}}
