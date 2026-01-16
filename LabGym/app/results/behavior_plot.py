"""
LabGym.app.results.behavior_plot
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import Sequence

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.domain.options import ResultsOptions
from LabGym.subsystems.results.api import plot_behavior_events


def run(
    opts: ResultsOptions,
    on_progress: ProgressCallback | None = None,
) -> None:
    on_progress = on_progress or noop_progress()
    on_progress(0, "Rendering raster plot")

    plot_behavior_events(**opts.as_plot_kwargs())

    on_progress(100, "Plot saved")

