"""
LabGym.ui.bindings.cli
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import List, Dict

# Local application imports
from LabGym.domain.options import AnalyzeOptions


def analyze_from_cli(args: Dict) -> AnalyzeOptions:
    """Convert the dict returned by LabGym.utils.cli.parse_args() into AnalyzeOptions."""

    # Current CLI does not yet expose analysis-specific flags.
    # Extend utils/cli.py to add them maybe - just mapping the basics now

    return AnalyzeOptions(
        videos = [Path(p) for p in args.get("videos", [])],
        output_dir = Path(args.get("output", ".")),

        # more flags to add if CLI surface grows
    )