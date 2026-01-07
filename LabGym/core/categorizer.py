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
LabGym.core.categorizer - DEPRECATED SHIM

Categorizer training function now lives in LabGym.workflows.training.detector.train
Categorizer evaluation, loading, and inference functions now live in LabGym.workflows.evaluation.categorizer
"""

# Standard library imports
import warnings

# Local application imports
# RE-EXPORTING LEGACY FUNCTIONS THAT WERE MOVED
from LabGym.workflows.training.categorizer.train import *
from LabGym.workflows.evaluation.categorizer import *


# DEPRECATION NOTICE
warnings.warn(
	"LabGym.core.detector is deprecated; import from"
	"LabGym.workflows.training.categorizer.train or"
	"LabGym.workflows.evaluation.categorizer instead.",
	DeprecationWarning,
	stacklevel=2,
)
