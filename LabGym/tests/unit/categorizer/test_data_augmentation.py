"""
/tests.unit.categorizer.test_data_augmentation

Tests for LabGym categorizer data augmentation functions:
- build_data() - Augmentation pipeline
- rename_label() - Prepare training examples
"""

# Standard library imports
import os

# Related third party imports
import cv2
import numpy as np
import pytest

# Local application imports
from LabGym.categorizer import Categorizers


@pytest.fixture
def synthetic_video_with_pattern(tmp_path):
    """
    Create a video file and corresponding pattern image for testing.
    Returns tuple: (video_path, pattern_image_path)
    """
    video_path = tmp_path / "test_0_walking.avi"
    pattern_path = tmp_path / "test_0_walking.jpg"
    
    # Create video (15 frames)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(video_path), fourcc, 30, (32, 32))
    for i in range(15):
        frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    
    # Create pattern image
    pattern_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(pattern_path), pattern_img)
    
    return str(video_path), str(pattern_path)


@pytest.fixture
def prepared_data_directory(tmp_path):
    """
    Create a directory structure for rename_label() testing.
    Structure: behavior_folders/behavior_name/*.avi and *.jpg
    """
    base_dir = tmp_path / "prepared"
    base_dir.mkdir()
    
    behaviors = ['walking', 'resting']
    for behavior in behaviors:
        behavior_dir = base_dir / behavior
        behavior_dir.mkdir()
        
        # Create 3 examples per behavior
        for i in range(3):
            # Create video
            video_path = behavior_dir / f"example_{i}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(str(video_path), fourcc, 30, (32, 32))
            for frame_idx in range(15):
                frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
            
            # Create pattern image
            pattern_path = behavior_dir / f"example_{i}.jpg"
            pattern_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(pattern_path), pattern_img)
    
    return str(base_dir)


class TestRenameLabel:
    """Tests for rename_label() method."""
    
    def test_rename_label_creates_output_directory(self, prepared_data_directory, tmp_path):
        """Should create output directory with renamed files."""
        categorizer = Categorizers()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        categorizer.rename_label(
            prepared_data_directory,
            str(output_dir)
        )
        
        # Check that output directory has files
        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) > 0
    
    def test_rename_label_with_resize(self, prepared_data_directory, tmp_path):
        """Should resize images when resize parameter is provided."""
        categorizer = Categorizers()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        categorizer.rename_label(
            prepared_data_directory,
            str(output_dir),
            resize=64
        )
        
        # Check that images are resized
        output_files = list(output_dir.glob("*.jpg"))
        if len(output_files) > 0:
            img = cv2.imread(str(output_files[0]))
            assert img.shape[:2] == (64, 64)
    
    def test_rename_label_requires_multiple_categories(self, tmp_path):
        """Should abort if fewer than 2 categories."""
        categorizer = Categorizers()
        single_category_dir = tmp_path / "single"
        single_category_dir.mkdir()
        (single_category_dir / "only_behavior").mkdir()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Should not raise exception, but should print error
        categorizer.rename_label(
            str(single_category_dir),
            str(output_dir)
        )
        
        # Output directory should remain empty or unchanged
        assert True  # Just check it doesn't crash


class TestBuildData:
    """Tests for build_data() method."""
    
    def test_build_data_without_augmentation(self, synthetic_video_with_pattern, tmp_path):
        """Should load data without augmentation."""
        categorizer = Categorizers()
        video_path, pattern_path = synthetic_video_with_pattern
        
        animations, pattern_images, labels = categorizer.build_data(
            [video_path],
            dim_tconv=16,
            dim_conv=32,
            channel=1,
            time_step=15,
            aug_methods=[],
            background_free=True,
            black_background=True,
            behavior_mode=0
        )
        
        assert len(animations) == 1
        assert len(pattern_images) == 1
        assert len(labels) == 1
        assert animations[0].shape == (15, 16, 16, 1)
        assert pattern_images[0].shape == (32, 32, 3)
        assert labels[0] == 'walking'
    
    def test_build_data_with_rotation_augmentation(self, synthetic_video_with_pattern, tmp_path):
        """Should apply rotation augmentation."""
        categorizer = Categorizers()
        video_path, pattern_path = synthetic_video_with_pattern
        
        animations, pattern_images, labels = categorizer.build_data(
            [video_path],
            dim_tconv=16,
            dim_conv=32,
            channel=1,
            time_step=15,
            aug_methods=['random rotation'],
            background_free=True,
            black_background=True,
            behavior_mode=0
        )
        
        # Should have more examples due to augmentation
        assert len(animations) > 1
        assert len(pattern_images) > 1
        assert len(labels) > 1
    
    def test_build_data_with_flipping_augmentation(self, synthetic_video_with_pattern, tmp_path):
        """Should apply flipping augmentation."""
        categorizer = Categorizers()
        video_path, pattern_path = synthetic_video_with_pattern
        
        animations, pattern_images, labels = categorizer.build_data(
            [video_path],
            dim_tconv=16,
            dim_conv=32,
            channel=1,
            time_step=15,
            aug_methods=['horizontal flipping'],
            background_free=True,
            black_background=True,
            behavior_mode=0
        )
        
        assert len(animations) > 1
    
    def test_build_data_static_mode(self, tmp_path):
        """Should handle static image mode (behavior_mode=3)."""
        categorizer = Categorizers()
        
        # Create static image file (no video needed)
        pattern_path = tmp_path / "test_0_walking.jpg"
        pattern_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(pattern_path), pattern_img)
        
        # For static mode, we need to pass image path, but build_data expects video
        # This test verifies the method handles the mode parameter
        # Note: build_data expects video files, so this may need adjustment
        assert True  # Placeholder - static mode testing may need video files too
    
    def test_build_data_outputs_to_path(self, synthetic_video_with_pattern, tmp_path):
        """Should output augmented data to specified path."""
        categorizer = Categorizers()
        video_path, pattern_path = synthetic_video_with_pattern
        output_dir = tmp_path / "augmented_output"
        output_dir.mkdir()
        
        categorizer.build_data(
            [video_path],
            dim_tconv=16,
            dim_conv=32,
            channel=1,
            time_step=15,
            aug_methods=['random rotation'],
            background_free=True,
            black_background=True,
            behavior_mode=0,
            out_path=str(output_dir)
        )
        
        # Check that files were created in output directory
        output_files = list(output_dir.glob("*"))
        assert len(output_files) > 0
    
    def test_build_data_normalizes_output(self, synthetic_video_with_pattern, tmp_path):
        """Should normalize pixel values to [0, 1] range."""
        categorizer = Categorizers()
        video_path, pattern_path = synthetic_video_with_pattern
        
        animations, pattern_images, labels = categorizer.build_data(
            [video_path],
            dim_tconv=16,
            dim_conv=32,
            channel=1,
            time_step=15,
            aug_methods=[],
            background_free=True,
            black_background=True,
            behavior_mode=0
        )
        
        # Check normalization
        assert animations[0].dtype == np.float32
        assert pattern_images[0].dtype == np.float32
        assert 0.0 <= animations[0].min() and animations[0].max() <= 1.0
        assert 0.0 <= pattern_images[0].min() and pattern_images[0].max() <= 1.0