"""
LabGym.app.train.detector
"""

from __future__ import annotations

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.subsystems.detection.api import train as DetectorTrainer
from LabGym.domain.options import TrainDetectorOptions


def run(
    *,
    path_to_annotation: str,
    path_to_trainingimages: str,
    path_to_detector: str,
    iteration_num: int,
    inference_size : int,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Train a Detectron2 detector."""
    
    on_progress = on_progress or noop_progress()
    on_progress(0, "Starting detector training")

    DetectorTrainer().train(
        path_to_annotation,
        path_to_trainingimages,
        path_to_detector,
        iteration_num,
        inference_size,
    )

    on_progress(100, "Detector training")


def run_options(opts: TrainDetectorOptions) -> None:
    """Entry point that accepts a TrainDetectorOptions dataclass."""
    run(
        path_to_annotation = str(opts.path_to_annotation),
        path_to_trainingimages = str(opts.path_to_trainingimages),
        path_to_detector = str(opts.path_to_detector),
        iteration_num = opts.iteration_num,
        inference_size = opts.inference_size,
    )

