"""
LabGym.app.context
"""

# Standard library imports
from typing import Protocol, Callable


class ProgressCallback(Protocol):
    """
    Generic progress reporter injected by UI / CLI.
    The caller guarantees:
        - 0 <= percent <= 100
        - final call uses percent == 100
    """

    def __call__(self, percent: int, message: str) -> None: ...


def noop_progress() -> ProgressCallback:
    """Handy default that does nothing."""
    return lambda _percent, _msg: None

    