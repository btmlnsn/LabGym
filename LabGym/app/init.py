"""
LabGym.app.init

Public facade for UI / CLI - everything outside LabGym.app should import from here.
"""

# Local application imports
from .context import ProgressCallback, noop_progress

# data helpers
from .data.generate_images import run as make_detector_images
from .data.generate_examples import run as make_categorizer_examples

# training
from .train.detector import run as train_detector
from .train.categorizer import run as train_categorizer

# evaluation
from .evaluate.detector import run as evaluate_detector
from .evaluate.categorizer import run as evaluate_categorizer

# analysis
from .analyze.behaviors import (
    run_background_subtraction,
    run_detector as analyze_with_detector,
)
from .analyze.mine_results import run as mine_results



__all__ = [
    "ProgressCallback",
    "noop_progress",
    # data
    "make_detector_images",
    "make_categorizer_examples",
    # training
    "train_detector",
    "train_categorizer",
    # evaluation
    "evaluate_detector",
    "evaluate_categorizer",
    # analysis
    "analyze_with_background_subtraction",
    "analyze_with_detector",
    "mine_results",
]
