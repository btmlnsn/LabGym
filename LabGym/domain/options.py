"""
LabGym.domain.options
"""

from __future__ import annotations

# Standard library imports
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from pandas.core.arrays.timedeltas import parse_timedelta_unit

Color = Tuple[int, int, int]


# ANALYSIS
@dataclass
class AnalyzeOptions:
    # Required
    videos: List[Path]
    output_dir: Path

    # Optional - default values match current GUI defaults
    categorizer_path: Optional[Path] = None
    detector_path: Optional[Path] = None
    animal_kinds: List[str] = field(default_factory=list)
    use_detector : bool = False
    behavior_mode: int = 0
    animal_number: int = 1
    start_t: float = 0.0  # seconds
    duration: float = 0.0  # 0 = "to end"
    min_length: Optional[int] = None
    uncertain: float = 0.0  # probability diff 0-1
    length: int = 15
    delta: float = 10_000.0  # illumination change
    background_path: Optional[Path] = None
    behaviornames_and_colors: Dict[str, Color] = field(default_factory=dict)
    parameter_to_analyze: List[str] = field(default_factory=list)

    # Convenience helpers
    def as_prepare_kwargs(self) -> dict:       # ← rename + improve
        """Return only the kwargs accepted by AnalyzeAnimal.prepare_analysis()."""
        return dict(
            path_to_video=[str(p) for p in self.videos],
            results_path=str(self.output_dir),
            animal_number=self.animal_number,
            delta=self.delta,
            names_and_colors=self.behaviornames_and_colors,
            framewidth=None,
            categorize_behavior=bool(self.categorizer_path),
            path_background=str(self.background_path) if self.background_path else None,
            t=self.start_t,
            duration=self.duration,
            length=self.length,
        )


# DETECTOR TRAINING (to be fleshed out later)
@dataclass
class TrainDetectorOptions:
    data_root: Path
    output_dir: Path
    epochs: int = 300
    learning_rate: float = 1e-4


# RESULTS 
@dataclass 
class ResultsOptions:
    results_dir: Path
    behaviors: list[str] = field(default_factory = list)
    
    # statistical settings
    paired: bool = False
    p_value: float = 0.05
    control_frame: Any | None = None  # Optional pandas DP
    file_names: list[str] = field(default_factory = list)

    # extra outputs
    make_plot: bool = False

    
    # helpers
    def as_stat_kwargs(self) -> dict:
        return dict(
            data_frames = [],
            control_frame = self.control_frame,
            paired = self.paired,
            out_dir = self.results_dir,
            p_value = self.p_value,
            file_names = self.file_names,
        )


    def as_plot_kwargs(self) -> dict:
        return dict(
            results_path = self.results_dir,
            behaviors = self.behaviors,
            width = 0,
            height = 0,
        )
    
    
