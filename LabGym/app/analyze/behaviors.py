"""
LabGym.app.analyze.behaviors
"""

from __future__ import annotations

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.workflows.analysis.analyze_behaviors import AnalyzeAnimal, AnalyzeAnimalDetector


def run_background_subtraction(
    video: str,
    out_dir: str,
    *,
    animal_number: int,
    on_progress: ProgressCallback | None = None,
    **opts,
) -> None:
    """Analyze behaviors using background subtraction pipeline"""

    on_progress = on_progress or noop_progress()
    on_progress(0, "Analyzing video (background subtraction)")

    AA = AnalyzeAnimal()
    AA.prepare_analysis(video, out_dir, animal_number, **opts)
    AA.generate_data(**opts)
    
    on_progress(100, "Analysis complete")


def run_detector(
    video: str,
    out_dir: str,
    *,
    animal_kinds: list[str],
    detector_path: str,
    on_progress: ProgressCallback | None = None,
    **opts,
) -> None:
    """Analyze behaviors using a trained Detectron2 detector"""

    on_progress = on_progress or noop_progress()
    on_progress(0, "Analyzing video (detector)")

    AAD = AnalyzeAnimalDetector()
    AAD.prepare_analysis(video, out_dir, animal_kinds, detector_path, **opts)
    AAD.generate_data(**opts)

    on_progress(100, "Analysis complete")
    
    