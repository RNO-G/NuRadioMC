"""Config layout checks of the 3D reconstruction driver that need no data reader."""

from NuRadioReco.modules.RNO_G.channelPreprocessor import channelPreprocessor

TOP_LEVEL_KEYS_SHARED_WITH_PREPROCESSOR = ('apply_upsampling', 'channels')


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
