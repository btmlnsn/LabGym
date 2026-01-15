"""
LabGym.app.results.mine_results
"""

from __future__ import annotations

# Standard library i mports
from pathlib import Path
from typing import Sequence, Any

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.subsystems.results.api import mine_results


def run(
    *,
    data_frames: Sequence[Any],  # deliberately opaque type to avoid importing pandas
    control_frame: Any | None = None,
    paired: bool = False,
    out_dir: str | Path,
    p_value: float = 0.05,
    file_names: Sequence[str] | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    on_progress = on_progress or noop_progress()
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    
    on_progress(0, "Mining statistical results")

    miner = mine_results(
        data_in = list(data_frames),
        control_in = control_frame,
        paired_in = paired,
        result_path_in = str(out_dir),
        pval_in = p_value,
        file_names_in = list(file_names or []),
    )

    if len(data_frames) == 1 and control_frame is not None:
        miner.two_groups()
    else:
        miner.multiple_groups()

    
    on_progress(100, "Statistical mining complete")

    