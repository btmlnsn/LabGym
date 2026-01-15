"""
LabGym.ui.bindings.gui
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import List

# Local application imports
from LabGym.domain.options import AnalyzeOptions


def _maybe_path(p) -> Path | None:
    """Convert str/Path/None to Path/None """
    return Path(p) if p else None


def analyze_from_panel(panel) -> AnalyzeOptions:
    """Translate a PanelLv2_AnalyzeBehaviors instance to domain.AnalyzeOptions"""
    videos: List[Path] = []
    if panel.path_to_videos:
        if isinstance(panel.path_to_videos, list):
            videos = [Path(v) for v in panel.path_to_videos]
        else:
            videos = [Path(panel.path_to_videos)]
    
    return AnalyzeOptions(
        videos = videos,
        output_dir = _maybe_path(panel.result_path) or Path.cwd(),
        categorizer_path = _maybe_path(panel.path_to_categorizer),
        detector_path = _maybe_path(panel.path_to_detector),
        animal_kinds = panel.animal_kinds,
        use_detector = panel.use_detector,
        behavior_mode = panel.behavior_mode,
        animal_number = panel.animal_number or 1,
        start_t = panel.t,
        duration = panel.duration,
        min_length = panel.min_length,
        uncertain = panel.uncertain,
        length = panel.length,
        delta = panel.delta,
        background_path = _maybe_path(panel.background_path),
        behaviornames_and_colors = panel.behaviornames_and_colors,
        parameter_to_analyze = panel.parameter_to_analyze,
    )


