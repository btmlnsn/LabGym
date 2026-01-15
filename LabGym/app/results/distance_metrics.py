"""
LabGym.app.results.distance_metrics
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import Sequence

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.subsystems.results import calculate_distances


def run(
    *,
    results_path: str | Path,
    behaviors: Sequence[str],
    video_folder_name: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    on_progress = on_progress or noop_progress()
    on_progress(0, "Calculating distance metrics")

    calculate_distances(
        path_to_folder = str(results_path),
        filename = video_folder_name or Path(results_path).name,
        behavior_to_include = list(behaviors),
        out_path = str(results_path),
    )

    on_progress(100, "Distance metrics complete")

