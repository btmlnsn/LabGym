"""
LabGym.app.results.behavior_plot
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import Sequence

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.subsystems.results.api import behavior_plot


def run(
    *,
    results_path: str | Path,
    behaviors: Sequence[str],
    width: int = 0,
    height: int = 0,
    on_progress: ProgressCallback | None = None,
) -> None:
    on_progress = on_progress or noop_progress()
    on_progress(0, "Rendering raster plot")

    behavior_plot(
        result_path = str(results_path),
        event_probability = None,
        time_points = None,
        names_and_colors = None,
        behavior_to_include = behaviors,
        width = width,
        height = height,
    )

    on_progress(100, "Plot saved")

