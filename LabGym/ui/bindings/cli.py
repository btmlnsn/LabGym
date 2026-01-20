"""
LabGym.ui.bindings.cli
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from LabGym.domain.options import (
    AnalyzeOptions,
    TrainDetectorOptions,
    TrainCategorizerOptions,
    EvalCategorizerOptions,
)


# analyze
def analyze_from_argv(argv: List[str]) -> AnalyzeOptions:
    p = argparse.ArgumentParser(prog="labgym analyze")
    p.add_argument("--videos", nargs="+", required=True, metavar="FILE")
    p.add_argument("--output-dir", required=True, metavar="DIR")
    p.add_argument("--animal-number", type=int, default=1)
    p.add_argument("--start-t", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--categorizer-path")
    p.add_argument("--detector-path")
    ns = p.parse_args(argv)

    return AnalyzeOptions(
        videos=[Path(v) for v in ns.videos],
        output_dir=Path(ns.output_dir),
        animal_number=ns.animal_number,
        start_t=ns.start_t,
        duration=ns.duration,
        categorizer_path=Path(ns.categorizer_path) if ns.categorizer_path else None,
        detector_path=Path(ns.detector_path) if ns.detector_path else None,
    )


# train detector
def train_detector_from_argv(argv: List[str]) -> TrainDetectorOptions:
    p = argparse.ArgumentParser(prog="labgym train-detector")
    p.add_argument("--ann", required=True, metavar="ANNOTATIONS_JSON")
    p.add_argument("--images", required=True, metavar="IMAGES_DIR")
    p.add_argument("--out", required=True, metavar="DETECTOR_DIR")
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--inf-size", type=int, default=800)
    ns = p.parse_args(argv)

    return TrainDetectorOptions(
        path_to_annotation = Path(ns.ann),
        path_to_trainingimages = Path(ns.images),
        path_to_detector = Path(ns.out),
        iteration_num = ns.iterations,
        inference_size = ns.inf_size,
    )


# train categorizer
def train_categorizer_from_argv(argv: List[str]) -> TrainCategorizerOptions:
    p = argparse.ArgumentParser(prog="labgym train-categorizer")
    p.add_argument("--data", required=True, metavar="DATA_DIR")
    p.add_argument("--out", required=True, metavar="MODEL_DIR")
    p.add_argument("--network", choices=["pattern", "animation", "comb"], default="comb")
    ns = p.parse_args(argv)

    return TrainCategorizerOptions(
        groundtruth_dir = Path(ns.gt),
        trained_model_dir = Path(ns.model),
        network_type = {"pattern": 0, "animation": 1, "comb": 2}[ns.network],
    )


# eval categorizer
def eval_categorizer_from_argv(argv: List[str]) -> EvalCategorizerOptions:
    p = argparse.ArgumentParser(prog="labgym eval-categorizer")
    p.add_argument("--gt", required=True, metavar="GROUND_TRUTH_DIR")
    p.add_argument("--model", required=True, metavar="MODEL_PATH")
    p.add_argument("--out", metavar="REPORT_DIR")
    ns = p.parse_args(argv)

    return EvalCategorizerOptions(
        groundtruth_dir = Path(ns.gt),
        trained_model_dir =Path(ns.model),
        results_dir = Path(ns.out) if ns.out else None,
    )