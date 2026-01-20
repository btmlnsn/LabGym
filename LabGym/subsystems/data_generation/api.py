"""
LabGym.subsystems.data_generation.api
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import Callable, Iterable

# Data generation subsystem imports
from .generate_images import generate_images as _generate_images
from .generate_examples import generate_examples as _generate_examples


def generate_images(
    *,
    videos: Iterable[str | Path],
    out_dir: str | Path,
    framewidth: int | None = None,
    start_t: float = 0,
    duration: float = 0,
    skip: int = 1000,
    progress_cb: Callable[[float, str], None] | None = None,
) -> None:
    """Extract frames from videos for detector training."""
    _generate_images(
        videos = videos,
        out_dir = out_dir,
        framewidth = framewidth,
        start_t = start_t,
        duration = duration,
        skip = skip,
        progress_cb = progress_cb,
    )


def generate_examples(
    *,
    videos: Iterable[str | Path],
    out_dir: str | Path,
    behavior_mode: int,
    animal_number: int | None = None,

    # Detector Options
    use_detector: bool = False,
    detector_path: str | Path | None = None,
    animal_kinds: list[str] | None = None,
    detection_threshold: float = 0.0,

    # Video/Frame Options
    framewidth: int | None = None,
    delta: int = 10_000,
    t: float = 0,
    duration: float = 0,
    ex_start: float = 0,
    ex_end: float | None = None,
    length: int = 15,

    # Background Options
    autofind_t: bool = False,
    stable_illumination: bool = False,
    background_path: str | Path | None = None,
    background_free: bool = True,
    black_background: bool = True,

    # Analysis Options
    include_bodyparts: bool = True,
    std: int = 0,
    skip_redundant: int = 1,
    social_distance: int = 0,
    decode_animalnumber: bool | int = False,
) -> None:
    """Generate behavior examples for categorizer training."""
    _generate_examples(
        videos=videos,
        out_dir=out_dir,
        behavior_mode=behavior_mode,
        use_detector=use_detector,
        detector_path=detector_path,
        animal_kinds=animal_kinds,
        framewidth=framewidth,
        delta=delta,
        decode_animalnumber=decode_animalnumber,
        animal_number=animal_number,
        autofind_t=autofind_t,
        t=t,
        duration=duration,
        ex_start=ex_start,
        ex_end=ex_end,
        length=length,
        include_bodyparts=include_bodyparts,
        std=std,
        background_free=background_free,
        black_background=black_background,
        skip_redundant=skip_redundant,
        social_distance=social_distance,
        stable_illumination=stable_illumination,
        background_path=background_path,
        detection_threshold=detection_threshold,
    )
    
__all__ = ["generate_images", "generate_examples"]
