"""
LabGym.app.results.mine_results
"""

from __future__ import annotations

# Standard library i mports
from pathlib import Path
from typing import Sequence, Any

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.domain.options import ResultsOptions
from LabGym.subsystems.results.api import create_stats_miner


def run(
    opts: ResultsOptions,
    on_progress: ProgressCallback | None = None,
) -> None:
    on_progress = on_progress or noop_progress()
    
    on_progress(0, "Mining statistical results")

    miner = create_stats_miner(**opts.as_stat_kwargs())

    miner.statistical_analysis()
    
    on_progress(100, "Statistical mining complete")

    