"""
LabGym.subsystems.detection.api
"""

from .train import DetectorTrainer as train
from .eval import Detector as evaluate


__all__ =["train", "evaluate"]
