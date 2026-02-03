"""
/tests.unit.categorizer.test_data_loading

Tests for LabGym categorizer data loading classes:
- DatasetFromPath_AA (Animation Analyzer data loader)
- DatasetFromPath (Static Analyzer data loader)
"""

# Standard library imports
# (none needed)

# Related third party imports
import cv2
import numpy as np
import pytest

# Local application imports
from LabGym.categorizer import DatasetFromPath_AA, DatasetFromPath


@pytest.fixture
def synthetic_pattern_image_dir_aa(tmp_path):
    """
    Create a directory with pattern images (.jpg) and corresponding videos (.avi)
    for testing DatasetFromPath_AA.
    Format: {index}_{classname}.jpg and {index}_{classname}.avi
    """
    data_dir = tmp_path / "aa_data"
    data_dir.mkdir()

    classes = ['walking', 'resting']
    n_examples = 5
    time_steps = 15

    for classname in classes:
        for example_idx in range(n_examples):
            base_name = f"{example_idx}_{classname}"

            pattern_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            pattern_path = data_dir / f"{base_name}.jpg"
            cv2.imwrite(str(pattern_path), pattern_img)

            video_path = data_dir / f"{base_name}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(str(video_path), fourcc, 30, (16, 16))
            for frame_idx in range(time_steps):
                frame = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

    return str(data_dir)


@pytest.fixture
def synthetic_pattern_image_dir_static(tmp_path):
    """
    Create a directory with pattern images (.jpg) only for testing DatasetFromPath.
    Format: {index}_{classname}.jpg
    """
    data_dir = tmp_path / "static_data"
    data_dir.mkdir()

    classes = ['walking', 'resting']
    n_examples = 5

    for classname in classes:
        for example_idx in range(n_examples):
            base_name = f"{example_idx}_{classname}"
            pattern_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            pattern_path = data_dir / f"{base_name}.jpg"
            cv2.imwrite(str(pattern_path), pattern_img)

    return str(data_dir)


class TestDatasetFromPathAA:
    """Tests for DatasetFromPath_AA (Animation Analyzer data loader)."""

    def test_initializes_with_defaults(self, synthetic_pattern_image_dir_aa):
        """Should initialize with default parameters."""
        dataset = DatasetFromPath_AA(synthetic_pattern_image_dir_aa)
        assert dataset.length == 15
        assert dataset.batch_size == 32
        assert dataset.dim_tconv == 16
        assert dataset.dim_conv == 32
        assert dataset.channel == 1
        assert len(dataset.pattern_image_paths) == 10

    def test_initializes_with_custom_params(self, synthetic_pattern_image_dir_aa):
        """Should initialize with custom parameters."""
        dataset = DatasetFromPath_AA(
            synthetic_pattern_image_dir_aa,
            length=20,
            batch_size=4,
            dim_tconv=32,
            dim_conv=64,
            channel=3
        )
        assert dataset.length == 20
        assert dataset.batch_size == 4
        assert dataset.dim_tconv == 32
        assert dataset.dim_conv == 64
        assert dataset.channel == 3

    def test_load_info_creates_classmapping(self, synthetic_pattern_image_dir_aa):
        """Should create classmapping dictionary from pattern image filenames."""
        dataset = DatasetFromPath_AA(synthetic_pattern_image_dir_aa)
        assert 'walking' in dataset.classmapping
        assert 'resting' in dataset.classmapping
        assert len(dataset.classmapping) == 2
        for classname, label in dataset.classmapping.items():
            assert isinstance(label, list)
            # With 2 classes LabelBinarizer outputs 1 column (binary), not 2 (one-hot)
            assert len(label) == 1
            assert label[0] in (0, 1)

    def test_len_returns_correct_batch_count(self, synthetic_pattern_image_dir_aa):
        """Should return correct number of batches."""
        dataset = DatasetFromPath_AA(synthetic_pattern_image_dir_aa, batch_size=4)
        assert len(dataset) == 2

    def test_getitem_returns_correct_shapes(self, synthetic_pattern_image_dir_aa):
        """Should return data with correct shapes."""
        dataset = DatasetFromPath_AA(
            synthetic_pattern_image_dir_aa,
            batch_size=2,
            dim_tconv=16,
            dim_conv=32,
            channel=1,
            length=15
        )
        batch_data, labels = dataset[0]
        animations, pattern_images = batch_data
        assert animations.shape == (2, 15, 16, 16, 1)
        assert pattern_images.shape == (2, 32, 32, 3)
        # With 2 classes LabelBinarizer gives shape (batch, 1)
        assert labels.shape == (2, 1)
        assert animations.dtype == np.float32
        assert pattern_images.dtype == np.float32
        assert 0.0 <= animations.min() and animations.max() <= 1.0
        assert 0.0 <= pattern_images.min() and pattern_images.max() <= 1.0

    def test_getitem_with_grayscale_channel(self, synthetic_pattern_image_dir_aa):
        """Should convert animations to grayscale when channel=1."""
        dataset = DatasetFromPath_AA(
            synthetic_pattern_image_dir_aa,
            batch_size=2,
            channel=1,
            dim_tconv=16
        )
        batch_data, labels = dataset[0]
        animations, _ = batch_data
        assert animations.shape[-1] == 1

    def test_getitem_with_rgb_channel(self, synthetic_pattern_image_dir_aa):
        """Should keep RGB when channel=3."""
        dataset = DatasetFromPath_AA(
            synthetic_pattern_image_dir_aa,
            batch_size=2,
            channel=3,
            dim_tconv=16
        )
        batch_data, labels = dataset[0]
        animations, _ = batch_data
        assert animations.shape[-1] == 3


class TestDatasetFromPath:
    """Tests for DatasetFromPath (Static Analyzer data loader)."""

    def test_initializes_with_defaults(self, synthetic_pattern_image_dir_static):
        """Should initialize with default parameters."""
        dataset = DatasetFromPath(synthetic_pattern_image_dir_static)
        assert dataset.batch_size == 32
        assert dataset.dim_conv == 32
        assert dataset.channel == 3
        assert len(dataset.pattern_image_paths) == 10

    def test_initializes_with_custom_params(self, synthetic_pattern_image_dir_static):
        """Should initialize with custom parameters."""
        dataset = DatasetFromPath(
            synthetic_pattern_image_dir_static,
            batch_size=4,
            dim_conv=64,
            channel=1
        )
        assert dataset.batch_size == 4
        assert dataset.dim_conv == 64
        assert dataset.channel == 1

    def test_load_info_creates_classmapping(self, synthetic_pattern_image_dir_static):
        """Should create classmapping dictionary from pattern image filenames."""
        dataset = DatasetFromPath(synthetic_pattern_image_dir_static)
        assert 'walking' in dataset.classmapping
        assert 'resting' in dataset.classmapping
        assert len(dataset.classmapping) == 2

    def test_len_returns_correct_batch_count(self, synthetic_pattern_image_dir_static):
        """Should return correct number of batches."""
        dataset = DatasetFromPath(synthetic_pattern_image_dir_static, batch_size=4)
        assert len(dataset) == 2

    def test_getitem_returns_correct_shapes(self, synthetic_pattern_image_dir_static):
        """Should return data with correct shapes."""
        dataset = DatasetFromPath(
            synthetic_pattern_image_dir_static,
            batch_size=2,
            dim_conv=32,
            channel=3
        )
        pattern_images, labels = dataset[0]
        assert pattern_images.shape == (2, 32, 32, 3)
        # With 2 classes LabelBinarizer gives shape (batch, 1)
        assert labels.shape == (2, 1)
        assert pattern_images.dtype == np.float32
        assert 0.0 <= pattern_images.min() and pattern_images.max() <= 1.0

    def test_getitem_with_grayscale_channel(self, synthetic_pattern_image_dir_static):
        """Should convert pattern images to grayscale when channel=1."""
        dataset = DatasetFromPath(
            synthetic_pattern_image_dir_static,
            batch_size=2,
            channel=1,
            dim_conv=32
        )
        pattern_images, labels = dataset[0]
        assert pattern_images.shape[-1] == 1

    def test_getitem_with_rgb_channel(self, synthetic_pattern_image_dir_static):
        """Should keep RGB when channel=3."""
        dataset = DatasetFromPath(
            synthetic_pattern_image_dir_static,
            batch_size=2,
            channel=3,
            dim_conv=32
        )
        pattern_images, labels = dataset[0]
        assert pattern_images.shape[-1] == 3

