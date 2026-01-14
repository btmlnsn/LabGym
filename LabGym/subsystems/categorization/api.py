"""
LabGym.subsystems.categorization.api
Thin façade: exposes three callables while hiding internal structure.
The categorization subsystem is OPTIONAL; heavy dependencies are loaded
only when these functions are invoked.  Importing this module itself
is safe even if TensorFlow/Keras are absent.
"""
from __future__ import annotations

# Standard library imports
from pathlib import Path

# Categorization subsystem imports
from .train import CategorizerTrainer
from .eval import Categorizer


def train(
    *,
    data_path: str | Path,
    model_path: str | Path,
    network: str = "comb",               # "pattern", "animation", "comb"
    **kw,
) -> None:
    trainer = CategorizerTrainer()
    method = {
        "pattern":   "train_pattern_recognizer",
        "animation": "train_animation_analyzer",
        "comb":      "train_combnet",
    }.get(network.lower())
    if method is None:
        raise ValueError(f"unknown network type '{network}'")
    getattr(trainer, method)(data_path, model_path, **kw)


def evaluate(
    *,
    groundtruth_path: str | Path,
    model_path: str | Path,
    out_path: str | Path | None = None,
    **kw,
) -> None:
    Categorizer().test_categorizer(
        groundtruth_path, model_path, result_path=out_path, **kw
    )


def predict(
    *,
    model_path: str | Path,
    animations,
    pattern_images,
    batch_size: int = 32,
    **kw,
):
    cat = Categorizer()
    cat.load(model_path)
    return cat.inference(
        animations, pattern_images, batch_size=batch_size, **kw
    )


__all__ = ["train", "evaluate", "predict"]
