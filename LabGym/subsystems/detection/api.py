"""
LabGym.subsystems.detection.api
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path

# Detection subsystem imports
from .train import DetectorTrainer
from .eval import Detector


def train (
    *,
    annotation_path: str | Path,
    images_path: str | Path,
    output_path: str | Path,
    iterations: int = 5000,
    inference_size: int = 640,
) -> None:
    """Train a new Detectron2 detector model."""
    trainer = DetectorTrainer()
    trainer.train(
        annotation_path,
        images_path,
        output_path,
        iterations,
        inference_size,
)


def evaluate (
    *,
    annotation_path: str | Path,
    images_path: str | Path,
    detector_path: str | Path,
    output_path: str | Path,
) -> None:
    """Evaluate a detector against ground-truth annotations."""
    detector = Detector()
    detector.test(annotation_path, images_path, detector_path, output_path)


def load(
    detector_path: str | Path,
    animal_kinds: list[str]
) -> Detector:
    """Load a detector for inference."""

    detector = Detector()
    detector.load(detector_path, animal_kinds)

    return detector


def infer(detector: Detector, inputs):
    return detector.inference(inputs)


__all__ = [
    "train",
    "evaluate",
    "load",
    "infer",
    # Also exposing classes for advanced users
    "DetectorTrainer",
    "Detector",
]