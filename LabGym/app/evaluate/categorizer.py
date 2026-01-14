"""
LabGym.app.evaluate.categorizer
"""

from __future__ import annotations

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.subsystems.categorization.api import evaluate as Categorizer


def run(
    *,
    groundtruth_dir: str,
    trained_model_dir: str,
    results_dir: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Compute validation metrics for a trained Categorizer."""
    
    on_progress = on_progress or noop_progress()
    on_progress(0, "Starting categorizer testing")

    Categorizer().test(
        groundtruth_dir,
        trained_model_dir,
        results_dir,
    )

    on_progress(100, "Categorizer testing complete")

    