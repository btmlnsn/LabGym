"""
LabGym.subsystems.shared.events
"""

from LabGym.subsystems.detection.api import evaluate as evaluate_detector
from LabGym.subsystems.categorization.predict import AnalyzeAnimal, AnalyzeAnimalDetector




__all__ = ["evaluate_detector", "AnalyzeAnimal", "AnalyzeAnimalDetector"]
