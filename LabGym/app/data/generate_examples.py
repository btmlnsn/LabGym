"""
LabGym.app.data.generate_examples
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import Iterable

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.subsystems.data_generation.api import generate_examples


def run(
    *,
    videos: Iterable[str | Path],
    out_dir: str | Path,
    behavior_mode: int,
    use_detector: bool = False,
    detector_path: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
    **legacy_kwargs,
) -> None:
    "Generate behavior example pairs for Categorizer training."

    on_progress = on_progress or noop_progress()
    on_progress(0, "Generating behavior examples")

    generate_examples(
        videos = list(videos),
        out_dir = out_dir,
        behavior_mode = behavior_mode,
        use_detector = use_detector,
        detector_path = detector_path,
        **legacy_kwargs,
    )

    on_progress(100, "Behavior example generation complete")
