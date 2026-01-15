"""
LabGym.ui.cli.__main__
"""

from __future__ import annotations

# Standard library imports
import sys

# Local application imports
from LabGym.domain import cli_flags
from LabGym.ui.bindings import cli as cli_bindings
from LabGym.app import app_analyze, app_train, app_evaluate, app_results


def _usage() -> None:
    print(
        "Usage: labgym <command> [options]\n"
        "Commands: analyze | train-detector | eval-categorizer\n"
        "Use --help with a command for its options."
    )
    sys.exit(1)

def main() -> None:
    # Parse the legacy global flags first
    global_cfg = cli_flags.parse_args()

    if len(sys.argv) <= 1:
        _usage()

    command = sys.argv[1]
    rest = sys.argv[2:]

    if command == "analyze":
        opts = cli_bindings.analyze_from_argv(rest)
        app_analyze.behaviors.run_options(opts)           # << app layer
    elif command == "train-detector":
        opts = cli_bindings.train_detector_from_argv(rest)
        app_train.detector.run_options(opts)
    elif command == "eval-categorizer":
        opts = cli_bindings.eval_categorizer_from_argv(rest)
        app_evaluate.categorizer.run_options(opts)
    else:
        _usage()

if __name__ == "__main__":
    main()

