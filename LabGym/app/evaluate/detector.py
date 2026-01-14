"""
LabGym.app.evaluate.detector
"""

from __future__ import annotations

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.subsystems.detection.api import evaluate as Detector


def run(
    *,
    annotation_json: str,
    test_images_dir: str,
    trained_detector_dir: str,
    out_dir: str,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Compute validation metrics for a trained Detectron2 detector."""

    on_progress = on_progress or noop_progress()
    on_progress(0, "Starting detector testing")

    Detector().test(
        annotation_json,
        test_images_dir,
        trained_detector_dir,
        out_dir,
    )    

    on_progress(100, "Detector testing complete")
