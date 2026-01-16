"""
LabGym.app.results.distance_metrics
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import Sequence

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.domain.options import ResultsOptions
from LabGym.subsystems.results.api import compute_distance_metrics


def run(
    opts: ResultsOptions,
    on_progress: ProgressCallback | None = None,
) -> None:
    on_progress = on_progress or noop_progress()
    on_progress(0, "Calculating distance metrics")

    compute_distance_metrics(**opts.as_stat_kwargs())

    on_progress(100, "Distance metrics complete")

