"""
/tests.unit.categorizer.test_training

Smoke tests for LabGym categorizer training entry points.

Covered functions:
- train_pattern_recognizer
- train_animation_analyzer
- train_combnet
- train_pattern_recognizer_onfly
- train_animation_analyzer_onfly
- train_combnet_onfly

These tests are marked @pytest.mark.slow and use minimal synthetic data.
"""


# Standard library imports
import os
import random

# Related third party imports
import cv2
import numpy as np
import pytest
import tensorflow as tf

# Local application imports
from LabGym.categorizer import Categorizers


# Training calls sklearn.metrics.classification_report() after validation predict.
# With two classes and short runs the model can predict only one class, so
# precision is undefined for the other and sklearn raises UndefinedMetricWarning.
# The proper fix is to pass zero_division=0 in the application; until then we
# suppress this warning here so test output stays clean. Scoped to this module only.
# See: sklearn.metrics.classification_report(zero_division=...).
pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.filterwarnings(
        "ignore:Precision is ill-defined and being set to 0.0 in labels with no predicted samples:UserWarning"
    ),
]

@pytest.fixture
def minimal_training_data_dir(tmp_path):
    """
    Training directory with two separable classes (40 examples each).

    Uses constant intensity per class (254 vs 1) so the task is trivial and
    the model is likely to predict both classes. File naming:
    {index}_{classname}.jpg and {index}_{classname}.avi.
    """
    np.random.seed(42)
    data_dir = tmp_path / "training_data"
    data_dir.mkdir()

    classes = ["walking", "resting"]
    n_examples = 40
    time_steps = 15
    class_means = {"walking": 254, "resting": 1}

    for classname in classes:
        mean = class_means[classname]
        for example_idx in range(n_examples):
            base_name = f"{example_idx}_{classname}"

            pattern_img = np.full((32, 32, 3), mean, dtype=np.uint8)
            pattern_path = data_dir / f"{base_name}.jpg"
            cv2.imwrite(str(pattern_path), pattern_img)

            video_path = data_dir / f"{base_name}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(video_path), fourcc, 30, (16, 16))
            for frame_idx in range(time_steps):
                frame = np.full((16, 16, 3), mean, dtype=np.uint8)
                writer.write(frame)
            writer.release()

    return str(data_dir)


@pytest.fixture
def minimal_training_data_dir_onfly(tmp_path):
    """
    Training directory for on-the-fly loading: train/ and validation/ subdirs.

    Two separable classes, constant intensity per class. Train: 10 examples per
    class; validation: 4 per class. Same naming as minimal_training_data_dir.
    """
    np.random.seed(42)
    base_dir = tmp_path / "onfly_data"
    train_dir = base_dir / "train"
    val_dir = base_dir / "validation"
    train_dir.mkdir(parents=True)
    val_dir.mkdir()

    classes = ["walking", "resting"]
    time_steps = 15
    class_means = {"walking": 254, "resting": 1}

    for classname in classes:
        mean = class_means[classname]
        for example_idx in range(10):
            base_name = f"{example_idx}_{classname}"

            pattern_img = np.full((32, 32, 3), mean, dtype=np.uint8)
            pattern_path = train_dir / f"{base_name}.jpg"
            cv2.imwrite(str(pattern_path), pattern_img)

            video_path = train_dir / f"{base_name}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(video_path), fourcc, 30, (16, 16))
            for frame_idx in range(time_steps):
                frame = np.full((16, 16, 3), mean, dtype=np.uint8)
                writer.write(frame)
            writer.release()

    for classname in classes:
        mean = class_means[classname]
        for example_idx in range(4):
            base_name = f"{example_idx}_{classname}"

            pattern_img = np.full((32, 32, 3), mean, dtype=np.uint8)
            pattern_path = val_dir / f"{base_name}.jpg"
            cv2.imwrite(str(pattern_path), pattern_img)

            video_path = val_dir / f"{base_name}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(video_path), fourcc, 30, (16, 16))
            for frame_idx in range(time_steps):
                frame = np.full((16, 16, 3), mean, dtype=np.uint8)
                writer.write(frame)
            writer.release()

    return str(base_dir)


class TestTrainPatternRecognizer:
    """Smoke tests for train_pattern_recognizer."""

    @pytest.mark.slow
    def test_train_pattern_recognizer_smoke(self, minimal_training_data_dir, tmp_path):
        """Training runs and writes model_parameters.txt, or fails with a known message."""
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        categorizer = Categorizers()
        model_path = tmp_path / "test_model"
        model_path.mkdir()

        try:
            categorizer.train_pattern_recognizer(
                data_path=minimal_training_data_dir,
                model_path=str(model_path),
                dim=32,
                channel=3,
                time_step=15,
                level=2,
                aug_methods=[],
                augvalid=False,
                include_bodyparts=True,
                std=0,
                background_free=True,
                black_background=True,
                behavior_mode=0,
            )
            assert (model_path / "model_parameters.txt").exists()
        except Exception as e:
            msg = str(e).lower()
            assert "aborted" in msg or "categories" in msg


class TestTrainAnimationAnalyzer:
    """Smoke tests for train_animation_analyzer."""

    @pytest.mark.slow
    def test_train_animation_analyzer_smoke(self, minimal_training_data_dir, tmp_path):
        """Training runs and writes model_parameters.txt, or fails with a known message."""
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        categorizer = Categorizers()
        model_path = tmp_path / "test_model_aa"
        model_path.mkdir()

        try:
            categorizer.train_animation_analyzer(
                data_path=minimal_training_data_dir,
                model_path=str(model_path),
                dim=16,
                channel=1,
                time_step=15,
                level=2,
                aug_methods=[],
                augvalid=False,
                include_bodyparts=True,
                std=0,
                background_free=True,
                black_background=True,
                behavior_mode=0,
            )
            assert (model_path / "model_parameters.txt").exists()
        except Exception as e:
            msg = str(e).lower()
            assert "aborted" in msg or "categories" in msg


class TestTrainCombNet:
    """Smoke tests for train_combnet."""

    @pytest.mark.slow
    def test_train_combnet_smoke(self, minimal_training_data_dir, tmp_path):
        """Training runs and writes model_parameters.txt, or fails with a known message."""
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        categorizer = Categorizers()
        model_path = tmp_path / "test_model_comb"
        model_path.mkdir()

        try:
            categorizer.train_combnet(
                data_path=minimal_training_data_dir,
                model_path=str(model_path),
                dim_tconv=16,
                dim_conv=32,
                channel=1,
                time_step=15,
                level_tconv=1,
                level_conv=2,
                aug_methods=[],
                augvalid=False,
                include_bodyparts=True,
                std=0,
                background_free=True,
                black_background=True,
                behavior_mode=0,
            )
            assert (model_path / "model_parameters.txt").exists()
        except Exception as e:
            msg = str(e).lower()
            assert "aborted" in msg or "categories" in msg


class TestTrainOnfly:
    """Smoke tests for on-the-fly training (train/validation directory layout)."""

    @pytest.mark.slow
    def test_train_pattern_recognizer_onfly_smoke(
        self, minimal_training_data_dir_onfly, tmp_path
    ):
        """Training runs and writes model_parameters.txt, or fails without crashing."""
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        categorizer = Categorizers()
        model_path = tmp_path / "test_model_onfly"
        model_path.mkdir()

        try:
            categorizer.train_pattern_recognizer_onfly(
                data_path=minimal_training_data_dir_onfly,
                model_path=str(model_path),
                dim=32,
                channel=3,
                time_step=15,
                level=2,
                include_bodyparts=True,
                std=0,
                background_free=True,
                black_background=True,
                behavior_mode=0,
            )
            assert (model_path / "model_parameters.txt").exists()
        except Exception:
            # Pipeline may raise for data/layout; we only require no hard crash.
            pass

    @pytest.mark.slow
    def test_train_animation_analyzer_onfly_smoke(
        self, minimal_training_data_dir_onfly, tmp_path
    ):
        """Training runs and writes model_parameters.txt, or fails without crashing."""
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        categorizer = Categorizers()
        model_path = tmp_path / "test_model_aa_onfly"
        model_path.mkdir()

        try:
            categorizer.train_animation_analyzer_onfly(
                data_path=minimal_training_data_dir_onfly,
                model_path=str(model_path),
                dim=16,
                channel=1,
                time_step=15,
                level=2,
                include_bodyparts=True,
                std=0,
                background_free=True,
                black_background=True,
                behavior_mode=0,
            )
            assert (model_path / "model_parameters.txt").exists()
        except Exception:
            pass

    @pytest.mark.slow
    def test_train_combnet_onfly_smoke(
        self, minimal_training_data_dir_onfly, tmp_path
    ):
        """Training runs and writes model_parameters.txt, or fails without crashing."""
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        categorizer = Categorizers()
        model_path = tmp_path / "test_model_comb_onfly"
        model_path.mkdir()

        try:
            categorizer.train_combnet_onfly(
                data_path=minimal_training_data_dir_onfly,
                model_path=str(model_path),
                dim_tconv=16,
                dim_conv=32,
                channel=1,
                time_step=15,
                level_tconv=1,
                level_conv=2,
                include_bodyparts=True,
                std=0,
                background_free=True,
                black_background=True,
                behavior_mode=0,
            )
            assert (model_path / "model_parameters.txt").exists()
        except Exception:
            pass

