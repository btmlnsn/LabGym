"""
LabGym.subsystems.data_generation.generate_images

Generate images for detector training
Safe to import from any layer; no GUI dependencies.
"""

from __future__ import annotations

# Standard library imports
import logging
from pathlib import Path
from typing import Iterable, Callable


logger = logging.getLogger(__name__)

# Local application imports
from LabGym.io.video import extract_frames

_SUPPORTED_VIDEO_EXTENSIONS = (
    ".avi", ".mpg", ".mpeg", ".wmv", ".mp4", ".mkv", ".m4v", ".mov"
)


# pure generate function append
def generate_images(*,
    videos: Iterable[str | Path],
    out_dir: str | Path,
    framewidth: int | None = None,
    start_t: float = 0,
    duration: float = 0,
    skip: int = 1000,
    progress_cb: Callable[[float, str], None] | None = None, 
) -> None:
    videos = list(map(Path, videos))
    out_dir.mkdir(parents= True, exist_ok=True)
    if not videos:
        raise ValueError("videos collection is empty")
    
    total = len(videos)

    for i, vid in enumerate(videos, 1):
        extract_frames(
            str(vid),
            str(out_dir),
            framewidth = framewidth,
            start_t = start_t,
            duration = duration,
            skip_redundant = skip,
        )

    if progress_cb:
        progress_cb(i / total, f"{i}/{total}")

    
    logger.info("generated %d images set(s) -> %s", total, out_dir)



__all__ = "generate_images"
