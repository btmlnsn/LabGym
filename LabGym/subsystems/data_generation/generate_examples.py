"""
LabGym.subsystems.data_generation.generate_examples

Generate behavior examples for categorizer training
Safe to import from any layer; no GUI dependencies.
"""

from __future__ import annotations

# Standard library imports
import logging
from pathlib import Path
from typing import Iterable, Sequence


# Local application imports
# heavy workflows for analysis
from LabGym.subsystems.categorization.predict import (
    AnalyzeAnimal,
    AnalyzeAnimalDetector,
)

logger = logging.getLogger(__name__)

# path helper?
def _as_path_seq(x: Iterable[str | Path]) -> Sequence[Path]:
    paths = [Path(p) for p in x]
    if not paths:
        raise ValueError("videos collection is empty")
    return paths


# ADDEDFRM gui/categorizer.py generate_data()
# Long generate-data loop that used to live inside gui/categorizer Panel class
def generate_examples(
    *,
    videos: Iterable[str | Path],
    out_dir: str | Path,
    behavior_mode: int,
    use_detector: bool = False,
    detector_path: str | Path | None = None,
    animal_kinds:list[str] | None = None,
    # kwargs mirrored from GUI state (gui/categorizer.py)
    framewidth: int | None = None,
    delta: int = 10_000,
    decode_animalnumber: bool | int = False,
    animal_number = None,
    autofind_t: bool = False,
    t: float = 0,
    duration: float = 0,
    ex_start: float = 0,
    ex_end: float | None = None,
    length: int = 15,
    include_bodyparts: bool = True,
    std: int = 0,
    background_free : bool = True,
    black_background: bool = True,
    skip_redundant: int = 1,
    social_distance: int = 0,
    stable_illumination: bool = False,
    background_path: str | Path | None = None,
    detection_threshold: float = 0.0,
) -> None:
    """
    Run the long generate-data loop that used to live inside of gui.categorizer.PanelLv2_GenerateExamples.generate_data().
    """

    videos = _as_path_seq(videos)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating behavior examples for %d video(s)…", len(videos))

    for vid in videos:
        logger.debug("processing %s", vid)

        if use_detector is False:
            AA = AnalyzeAnimal()
            AA.prepare_analysis(
                vid,
                out_dir,
                animal_number,
                delta = delta,
                framewidth = framewidth,
                stable_illumination = stable_illumination,
                channel = 3,
                include_bodyparts = include_bodyparts,
                std = std,
                categorize_behavior = False,
                animation_analyzer = False,
                path_background = background_path,
                autofind_t = autofind_t,
                t = t,
                duration = duration,
                ex_start = ex_start,
                ex_end = ex_end,
                length = length,
                animal_vs_bg = social_distance,
            )

            if behavior_mode == 0:
                AA.generate_data(
                    background_free = background_free,
                    black_background = black_background,
                    skip_redundant = skip_redundant,
                )

            else:
                AA.generate_data_interact_basic(
                    background_free = background_free,
                    black_background = black_background,
                    skip_redundant = skip_redundant,
                )
        

        else:
            if detector_path is None:
                raise ValueError("detector_path must be given when use_detector =True")

            AAD = AnalyzeAnimalDetector()
            AAD.prepare_analysis(
                detector_path,
                vid,
                out_dir,
                animal_number,
                animal_kinds or [],
                behavior_mode,
                framewidth = framewidth,
                channel = 3,
                include_bodyparts = include_bodyparts,
                std = std,
                categorize_behavior = False,
                animation_analyzer = False,
                t = t,
                duration = duration,
                length = length,
                social_distance = social_distance,
            )


            if behavior_mode == 0:
                AAD.generate_data(
                    background_free = background_free,
                    black_background = black_background,
                    skip_redundant = skip_redundant,
                )

            elif behavior_mode == 1:
                AAD.generate_data_interact_basic(
                    background_free = background_free,
                    black_background = black_background,
                    skip_redundant = skip_redundant,
                )

            else:
                AAD.generate_data_interact_advance(
                    background_free = background_free,
                    black_background = black_background,
                    skip_redundant = skip_redundant,
                )


    logger.info("Behavior example generate complete -> %s", out_dir)        
    


__all__ = ["generate_examples"]
