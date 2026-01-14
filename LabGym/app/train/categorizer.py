"""
LabGym.app.train.categorizer
"""

from __future__ import annotations

# Local application imports
from LabGym.app.context import ProgressCallback, noop_progress
from LabGym.workflows.training.categorizer.train import CategorizerTrainer


def run(
    *,
    data_path: str,
    model_path: str,
    network_type: int = 2,
    on_progress: ProgressCallback | None = None,
    **legacy_kwargs,
) -> None:
    """Train a Categorizer."""
    # network type: 0=pattern, 1=animation, 2=comb (default)
    
    on_progress = on_progress or noop_progress()
    on_progress(0, "Starting categorizer training")

    trainer = CategorizerTrainer()

    if network_type == 0:
        trainer.train_pattern_recognizer(data_path, model_path, **legacy_kwargs)

    elif network_type == 1:
        trainer.train_animation_analyzer(data_path, model_path, **legacy_kwargs)

    else: 
        trainer.train_combnet(data_path, model_path, **legacy_kwargs)

    on_progress(100, "Categorizer training complete")

