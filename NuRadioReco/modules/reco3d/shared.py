"""Module-level names read by several files of the 3D reconstruction."""

import numpy as np
import logging

from scipy.constants import c as _SPEED_OF_LIGHT


logger = logging.getLogger("reco3d.interferometric_reco_3d")

_CANDIDATE_ORIGIN_CODES = {'raw': 0, 'envelope:traces': 1, 'envelope:correlation': 2}
_C_M_PER_NS = _SPEED_OF_LIGHT * 1e-9
_LBFGSB_ABS_STEP = 1e-8
_LBFGSB_REL_STEP = np.finfo(np.float64).eps ** 0.5
_STACK_BLOCK_POINTS = 8192
