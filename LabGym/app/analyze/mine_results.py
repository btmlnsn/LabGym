"""
LabGym.app.analyze.mine_results
"""

from __future__ import annotations

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.workflows.analysis.mine_results import data_mining


def run(
    *,
    data_in,
    control_in = None,
    paired_in = False,
    result_path_in: str,
    pval_in: float = 0.05,
    file_names_in = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Perform Excel-exported statistical mining."""

    on_progress = on_progress or noop_progress()
    on_progress(0, "Mining statistical results")

    miner = data_mining(
        data_in,
        control_in = control_in,
        paired_in = paired_in,
        result_path_in = result_path_in,
        pval_in = pval_in,
        file_names_in = file_names_in
    )

    miner.two_groups() # or .multiple_groups()

    on_progress(100, "Mining complete")

