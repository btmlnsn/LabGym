"""
LabGym.app.data.generate_images
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import Iterable

#  Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.subsystems.data_generation.api import generate_images 


def run(
    *,
    videos: Iterable[str | Path],
    out_dir: str | Path,
    framewidth: int | None = None,
    start_t: float = 0,
    duration: float = 0,
    skip: int = 1_000,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Generate frame images for Detector training."""
    
    on_progress = on_progress or noop_progress()
    on_progress(0, "Generating detector images")
    
    videos = list(videos)
    out_dir = Path(out_dir)

    generate_images(
        videos = videos,
        out_dir = out_dir,
        framewidth = framewidth,
        start_t = start_t,
        duration = duration,
        skip = skip,
        progress_cb = lambda pct, msg: on_progress(int(pct * 100), msg)
    )

    on_progress(100, "Images generation complete")
