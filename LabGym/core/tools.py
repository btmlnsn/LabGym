'''
Copyright (C)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)#fulltext.

For license issues, please contact:

Dr. Bing Ye
Life Sciences Institute
University of Michigan
210 Washtenaw Avenue, Room 5403
Ann Arbor, MI 48109-2216
USA

Email: bingye@umich.edu
'''

"""
LabGym.core.tools - DEPRECATED SHIM

Video / image helpers now live in LabGym.io.video
Filesystem helpers now live in LabGym.io.filesystem

Plotting hlpers now live in LabGym.workflows.analysis.behavior_plot 
Distance metric helpers now live in LabGym.workflows.analysis.distance_metrics
"""
# Standard library imports.
import warnings
import logging
import datetime
import logging
import math
import os

# Log the load of this module (by the module loader, on first import).
# Intentionally positioning these statements before other imports, against the
# guidance of PEP-8, to log the load before other imports log messages.
logger =  logging.getLogger(__name__)  # pylint: disable=wrong-import-position
logger.debug('loading %s', __file__)  # pylint: disable=wrong-import-position

# Related third party imports.
import cv2
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap,Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

# Local application/library specific imports.
# RE-EXPORTING LEGACY HELPER FUNCTIONS THAT WERE MOVED 
from LabGym.io.video import *
from LabGym.io.filesystem import *
from LabGym.workflows.analysis.behavior_plot import *
from LabGym.workflows.analysis.distance_metrics import *




# DEPRECATION NOTICE
warnings.warn(
	"LabGym.core.tools is deprecated; "
	"import helpers from LabGym.io.video, LabGym.io.filesystem, or "
	"LabGym.workflows.analysis*.",
	DeprecationWarning,
	stacklevel=2,
)
