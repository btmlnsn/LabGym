"""
/tests.unit.frame_analysis.test_pattern_images

Tests for LabGym.tools pattern image generation functions:
- generate_patternimage()
- generate_patternimage_all()
- generate_patternimage_interact()
"""

# Related third party imports
import numpy as np
import pytest
import cv2

# Local application imports
from LabGym.tools import (
    generate_patternimage,
    generate_patternimage_all,
    generate_patternimage_interact,
    crop_frame,
)


@pytest.fixture
def sample_outline_sequence():
    """Sequence of 15 contours representing movement over time."""
    outlines = []
    for i in range(15):
        # Circle moving from left to right
        x_center = 30 + i * 3
        angles = np.linspace(0, 2* np.pi, 30, endpoint = False)
        contour = np.array([
            [[int(x_center + 10 * np.cos(a)), int(50 + 10 * np.sin(a))]]
            for a in angles
        ], dtype = np.int32)
        
        outlines.append(contour)
        
    return outlines



class TestGeneratePatternimage:
    """Tests for generate_patternimage() function."""

    def test_returns_array(self, sample_background_frame, sample_outline_sequence):
        """Should return a numpy array."""
        result = generate_patternimage(
            sample_background_frame,
            sample_outline_sequence,
            inners=None,
            std=0
        )

        assert isinstance(result, np.ndarray)


    def test_output_is_color(self, sample_background_frame, sample_outline_sequence):
        """Output should be a color image (3 channels)."""
        result = generate_patternimage(
            sample_background_frame,
            sample_outline_sequence,
            inners=None,
            std=0
        )

        assert len(result.shape) == 3
        assert result.shape[2] == 3


    def test_without_inners(self, sample_background_frame, sample_outline_sequence):
        """Should work without inner contours."""
        result = generate_patternimage(
            sample_background_frame,
            sample_outline_sequence,
            inners=None,
            std=0
        )

        assert result is not None


    def test_with_inners(self, sample_background_frame, sample_outline_sequence):
        """Should work with inner contours provided."""
        # Create dummy inners (same as outlines for simplicity)
        inners = [[outline] for outline in sample_outline_sequence]

        result = generate_patternimage(
            sample_background_frame,
            sample_outline_sequence,
            inners=inners,
            std=0
        )
        assert result is not None

    def test_std_parameter(self, sample_background_frame, sample_outline_sequence):
        """std parameter should filter inner contours."""
        inners = [[outline] for outline in sample_outline_sequence]

        result_std0 = generate_patternimage(
            sample_background_frame,
            sample_outline_sequence,
            inners=inners,
            std=0
        )
        result_std100 = generate_patternimage(
            sample_background_frame,
            sample_outline_sequence,
            inners=inners,
            std=100
        )
        
        # Both should produce valid output
        assert result_std0 is not None
        assert result_std100 is not None


    def test_single_outline(self, sample_background_frame, sample_circular_contour):
        """Should work with single outline."""
        result = generate_patternimage(
            sample_background_frame,
            [sample_circular_contour],
            inners=None,
            std=0
        )

        assert result is not None


    def test_output_not_all_zeros(self, sample_background_frame, sample_outline_sequence):
        """Output should contain drawn contours (not all black)."""
        result = generate_patternimage(
            sample_background_frame,
            sample_outline_sequence,
            inners=None,
            std=0
        )

        assert np.any(result > 0), "Pattern image should have non-zero pixels"



class TestGeneratePatternimageAll:
    """Tests for generate_patternimage_all() function."""

    def test_returns_array(self, sample_background_frame, sample_outline_sequence):
        """Should return a numpy array."""
        # outlines_list is a list of lists of contours (for multiple animals per frame)
        outlines_list = [[outline] for outline in sample_outline_sequence]

        y_bt, y_tp, x_lf, x_rt = 20, 80, 20, 80

        result = generate_patternimage_all(
            sample_background_frame,
            y_bt, y_tp, x_lf, x_rt,
            outlines_list,
            inners_list=None,
            std=0
        )

        assert isinstance(result, np.ndarray)


    def test_without_inners(self, sample_background_frame, sample_outline_sequence):
        """Should work without inner contours."""
        outlines_list = [[outline] for outline in sample_outline_sequence]
        y_bt, y_tp, x_lf, x_rt = 20, 80, 20, 80

        result = generate_patternimage_all(
            sample_background_frame,
            y_bt, y_tp, x_lf, x_rt,
            outlines_list,
            inners_list=None,
            std=0
        )

        assert result is not None


    def test_output_shape_matches_crop(self, sample_background_frame, sample_outline_sequence):
        """Output shape should match the specified crop region."""
        outlines_list = [[outline] for outline in sample_outline_sequence]
        y_bt, y_tp, x_lf, x_rt = 20, 80, 10, 90

        result = generate_patternimage_all(
            sample_background_frame,
            y_bt, y_tp, x_lf, x_rt,
            outlines_list,
            inners_list=None,
            std=0
        )
        expected_height = y_tp - y_bt
        expected_width = x_rt - x_lf

        assert result.shape[0] == expected_height
        assert result.shape[1] == expected_width



class TestGeneratePatternimageInteract:
    """Tests for generate_patternimage_interact() function."""

    def test_returns_array(self, sample_background_frame, sample_outline_sequence):
        """Should return a numpy array."""
        # For interact, we need outlines for "self" and "other" animal
        other_outlines = []
        for i in range(15):
            x_center = 70 - i * 2  # Other animal moving opposite direction
            angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
            contour = np.array([
                [[int(x_center + 8 * np.cos(a)), int(50 + 8 * np.sin(a))]]
                for a in angles
            ], dtype=np.int32)
            other_outlines.append([contour])

        result = generate_patternimage_interact(
            sample_background_frame,
            sample_outline_sequence,
            other_outlines,
            inners=None,
            other_inners=None,
            std=0
        )

        assert isinstance(result, np.ndarray)


    def test_two_animals_different_colors(self, sample_background_frame, sample_outline_sequence):
        """Two animals should be drawn (pattern should have content)."""
        other_outlines = []
        for i in range(15):
            x_center = 70 - i * 2
            angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
            contour = np.array([
                [[int(x_center + 8 * np.cos(a)), int(50 + 8 * np.sin(a))]]
                for a in angles
            ], dtype=np.int32)
            other_outlines.append([contour])

        result = generate_patternimage_interact(
            sample_background_frame,
            sample_outline_sequence,
            other_outlines,
            inners=None,
            other_inners=None,
            std=0
        )

        assert np.any(result > 0), "Pattern should contain drawn outlines"


