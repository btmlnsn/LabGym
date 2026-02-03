"""
/tests.unit.categorizer.test_inference

Tests for Categorizers.test_categorizer() inference method.
Uses mock model and synthetic groundtruth/model dirs; no real trained model.
"""

# Standard library imports
import os

# Related third party imports
import cv2
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Local application imports
from LabGym.categorizer import Categorizers


def _write_model_parameters_txt(model_dir, network=0, classnames=None, **kwargs):
    """Write model_parameters.txt in the format produced by training code."""
    if classnames is None:
        classnames = ['walking', 'resting']
    defaults = {
        'classnames': classnames,
        'dim_conv': 32,
        'dim_tconv': 16,
        'channel': 3,
        'time_step': 15,
        'network': network,
        'level_conv': 2,
        'level_tconv': 1,
        'inner_code': 1,
        'std': 0,
        'background_free': 1,
        'black_background': 0,
        'behavior_kind': 0,
        'social_distance': 0,
    }
    defaults.update(kwargs)
    if network == 0:
        params = {k: v for k, v in defaults.items() if k in [
            'classnames', 'dim_conv', 'channel', 'time_step', 'network',
            'level_conv', 'inner_code', 'std', 'background_free',
            'black_background', 'behavior_kind', 'social_distance'
        ]}
    elif network == 1:
        params = {k: v for k, v in defaults.items() if k in [
            'classnames', 'dim_tconv', 'channel', 'time_step', 'network',
            'level_tconv', 'inner_code', 'std', 'background_free',
            'black_background', 'behavior_kind', 'social_distance'
        ]}
    else:
        params = defaults
    df = pd.DataFrame.from_dict(params)
    df.to_csv(os.path.join(model_dir, 'model_parameters.txt'), index=False)


@pytest.fixture
def groundtruth_dir_pattern_only(tmp_path):
    """Groundtruth dir for network=0: one subdir per class, .jpg only."""
    gt = tmp_path / "groundtruth"
    gt.mkdir()
    for c in ['walking', 'resting']:
        (gt / c).mkdir()
        for i in range(2):
            path = gt / c / f"ex_{i}.jpg"
            cv2.imwrite(str(path), np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    return str(gt)


@pytest.fixture
def groundtruth_dir_with_animation(tmp_path):
    """Groundtruth dir for network=1 or 2: subdirs with .avi and .jpg (same base name)."""
    gt = tmp_path / "groundtruth"
    gt.mkdir()
    for c in ['walking', 'resting']:
        (gt / c).mkdir()
        for i in range(2):
            base = gt / c / f"ex_{i}"
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            w = cv2.VideoWriter(str(base) + '.avi', fourcc, 30, (16, 16))
            for _ in range(15):
                w.write(np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8))
            w.release()
            cv2.imwrite(str(base) + '.jpg', np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    return str(gt)


@pytest.fixture
def model_dir_pattern_only(tmp_path):
    """Model dir for network=0 with model_parameters.txt only."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_model_parameters_txt(str(model_dir), network=0)
    return str(model_dir)


@pytest.fixture
def model_dir_animation_only(tmp_path):
    """Model dir for network=1."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_model_parameters_txt(str(model_dir), network=1)
    return str(model_dir)


@pytest.fixture
def model_dir_combined(tmp_path):
    """Model dir for network=2."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_model_parameters_txt(str(model_dir), network=2)
    return str(model_dir)


@pytest.fixture
def mock_categorizer_model_binary():
    """Mock Keras model: predict returns (n, 1) for binary classification."""
    m = MagicMock()
    m.predict.return_value = np.array([[0.2], [0.8], [0.3], [0.7]], dtype=np.float32)
    return m


class TestTestCategorizerPatternOnly:
    """Tests for test_categorizer() with network=0 (pattern recognizer only)."""

    @patch('LabGym.categorizer.load_model')
    def test_runs_and_writes_report(
        self,
        mock_load_model,
        groundtruth_dir_pattern_only,
        model_dir_pattern_only,
        mock_categorizer_model_binary,
        tmp_path,
    ):
        """Should run inference and write testing_reports.xlsx when result_path is set."""
        # Arrange
        mock_load_model.return_value = mock_categorizer_model_binary
        categorizer = Categorizers()
        result_path = str(tmp_path / "results")
        os.makedirs(result_path, exist_ok=True)

        # Act
        categorizer.test_categorizer(
            groundtruth_dir_pattern_only,
            model_dir_pattern_only,
            result_path=result_path,
        )

        # Assert
        mock_load_model.assert_called_once_with(model_dir_pattern_only)
        assert os.path.isfile(os.path.join(result_path, 'testing_reports.xlsx'))

    @patch('LabGym.categorizer.load_model')
    def test_predict_called_with_pattern_images_only(
        self,
        mock_load_model,
        groundtruth_dir_pattern_only,
        model_dir_pattern_only,
        mock_categorizer_model_binary,
    ):
        """Should call model.predict(pattern_images) for network=0."""
        mock_load_model.return_value = mock_categorizer_model_binary
        categorizer = Categorizers()

        categorizer.test_categorizer(
            groundtruth_dir_pattern_only,
            model_dir_pattern_only,
            result_path=None,
        )

        mock_load_model.return_value.predict.assert_called_once()
        call_args = mock_load_model.return_value.predict.call_args
        assert len(call_args[0][0].shape) == 4  # (n, h, w, c)


class TestTestCategorizerAnimationOnly:
    """Tests for test_categorizer() with network=1 (animation analyzer only)."""

    @patch('LabGym.categorizer.load_model')
    def test_runs_without_result_path(
        self,
        mock_load_model,
        groundtruth_dir_with_animation,
        model_dir_animation_only,
        mock_categorizer_model_binary,
    ):
        """Should run inference and complete when result_path is None."""
        mock_load_model.return_value = mock_categorizer_model_binary
        categorizer = Categorizers()

        categorizer.test_categorizer(
            groundtruth_dir_with_animation,
            model_dir_animation_only,
            result_path=None,
        )

        mock_load_model.assert_called_once_with(model_dir_animation_only)
        call_args = mock_load_model.return_value.predict.call_args
        assert len(call_args[0][0].shape) == 5  # (n, time, h, w, c)


class TestTestCategorizerCombined:
    """Tests for test_categorizer() with network=2 (animation + pattern)."""

    @patch('LabGym.categorizer.load_model')
    def test_runs_and_calls_predict_with_two_inputs(
        self,
        mock_load_model,
        groundtruth_dir_with_animation,
        model_dir_combined,
        mock_categorizer_model_binary,
    ):
        """Should call model.predict([animations, pattern_images]) for network=2."""
        mock_load_model.return_value = mock_categorizer_model_binary
        categorizer = Categorizers()

        categorizer.test_categorizer(
            groundtruth_dir_with_animation,
            model_dir_combined,
            result_path=None,
        )

        mock_load_model.return_value.predict.assert_called_once()
        call_args = mock_load_model.return_value.predict.call_args
        inputs = call_args[0][0]
        assert isinstance(inputs, list)
        assert len(inputs) == 2
        assert inputs[0].ndim == 5 and inputs[1].ndim == 4


class TestTestCategorizerMismatchedClasses:
    """When groundtruth subdirs don't match model classnames, no model load or predict."""

    @patch('LabGym.categorizer.load_model')
    def test_does_not_call_load_model_when_behaviors_mismatch(
        self,
        mock_load_model,
        model_dir_pattern_only,
        tmp_path,
    ):
        """Should not load model when groundtruth classes don't match model classnames."""
        gt = tmp_path / "groundtruth"
        gt.mkdir()
        (gt / "other").mkdir()
        cv2.imwrite(
            str(gt / "other" / "ex0.jpg"),
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
        )

        categorizer = Categorizers()
        categorizer.test_categorizer(str(gt), model_dir_pattern_only, result_path=None)

        mock_load_model.assert_not_called()

        