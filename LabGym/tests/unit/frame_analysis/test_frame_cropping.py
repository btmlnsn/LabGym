"""
tests/unit/frame_analysis/test_frame_cropping

Tests for LabGym.tools frame cropping and blob extraction functions:
- crop_frame()
- extract_blob_background()
- extract_blob_all()
"""

# Related third party imports
import numpy as np
import pytest
import cv2

# Local application imports
from LabGym.tools import crop_frame, extract_blob_background, extract_blob_all



class TestCropFrame:
    """Tests for crop_frame() function"""

    def test_single_contour_returns_valid_bounds(self, sample_circular_contour):
        """Single contour should return valid y_bt, y_tp, x_lf, x_rt"""
        frame = np.zeros((100, 100, 3), dtype = np.uint8)
        y_bt, y_tp, x_lf, x_rt = crop_frame(frame, [sample_circular_contour])

        assert y_bt < y_tp, "y_bt should be less than y_tp"
        assert x_lf < x_rt, "x_lf should be less than x_rt"
        assert y_bt >= 0, "y_bt should be non-negative"
        assert x_lf >= 0, "x_lf should be non-negative"

    
    def test_bounds_contain_contour(self, sample_circular_contour):
        """Returned bounds should contain the entire contour"""
        frame = np.zeros((100, 100, 3), dtype = np.uint8)
        y_bt, y_tp, x_lf, x_rt = crop_frame(frame, [sample_circular_contour])

        # Contour is centered at (50, 50) with radius 15
        # Thus, it spans roughly x: 35-65, y: 35-65
        assert x_lf <= 35, f"x_lf ({x_lf}) should be <= 35"
        assert x_rt >= 65, f"x_rt ({x_rt}) should be >= 65"
        assert y_bt <= 35, f"y_bt ({y_bt}) should be <= 35"
        assert y_tp >= 65, f"y_tp ({y_tp}) should be >= 65"


    def test_multiple_contours(self, sample_circular_contour):
        """Multiple contours should expand bounds to contain them all."""
        frame = np.zeros((100, 100, 3), dtype = np.uint8)

        # Create a second contour offset to the right
        contour2 = sample_circular_contour.copy()
        contour2[:, :, 0] += 20  # Shift x by 20

        y_bt, y_tp, x_lf, x_rt = crop_frame(frame, [sample_circular_contour, contour2])

        # Combined bounds should be wider
        assert x_rt - x_lf > 30, "Combined width should be larger"


    def test_rectangular_contour(self, sample_rectangular_contour):
        """Rectangular contour should work correctly"""
        frame = np.zeros((100, 100, 3), dtype = np.uint8)
        y_bt, y_tp, x_lf, x_rt = crop_frame(frame, [sample_rectangular_contour])

        assert y_bt < y_tp, "y_bt should be less than y_tp"
        assert x_lf < x_rt, "x_lf should be less than x_rt"

    
    def test_wide_aspect_ratio_pads_height(self):
        """When width > height, function should pad height to make sure it is square-ish"""
        frame = np.zeros((100, 100, 3), dtype = np.uint8)

        # Wide contour: 60 pixels wide, 20 pixels tall
        contour = np.array([
            [[10, 40]], [[70, 40]], [[70, 60]], [[10, 60]]
        ], dtype = np.int32)

        y_bt, y_tp, x_lf, x_rt = crop_frame(frame, [contour])

        height = y_tp - y_bt
        width = x_rt - x_lf
        
        # Should have added padding to height
        assert height >= 20, "Height should be padded"


    def test_tall_aspect_ratio_pads_width(self):
        """When height > width, function should pad width to make sure it's square-ish"""
        frame = np.zeros((100, 100, 3), dtype = np.uint8)

        # Tall contour: 20 pixels wide, 60 pixels tall
        contour = np.array([
            [[40, 10]], [[60, 10]], [[60, 70]], [[40, 70]]
        ], dtype = np.int32)

        y_bt, y_tp, x_lf, x_rt = crop_frame(frame, [contour])
        
        height = y_tp - y_bt
        width = x_rt - x_lf

        # Should have added padding to width
        assert width >= 20, "Width should be padded"



class TestExtractBlobBackground:
    """Tests for extract_blob_background() function."""

    def test_returns_array(self, sample_bgr_frame, sample_circular_contour):
        """Should return a numpy array."""
        result = extract_blob_background(
            sample_bgr_frame,
            [sample_circular_contour],
            contour=sample_circular_contour,
            channel=1
        )

        assert isinstance(result, np.ndarray)


    def test_grayscale_output(self, sample_bgr_frame, sample_circular_contour):
        """channel=1 should return grayscale (single channel)."""
        result = extract_blob_background(
            sample_bgr_frame,
            [sample_circular_contour],
            contour=sample_circular_contour,
            channel=1
        )

        # img_to_array adds a channel dimension, so shape is (H, W, 1)
        assert result.shape[-1] == 1, f"Expected 1 channel, got shape {result.shape}"


    def test_rgb_output(self, sample_bgr_frame, sample_circular_contour):
        """channel=3 should return RGB (3 channels)."""
        result = extract_blob_background(
            sample_bgr_frame,
            [sample_circular_contour],
            contour=sample_circular_contour,
            channel=3
        )
        assert result.shape[-1] == 3, f"Expected 3 channels, got shape {result.shape}"


    def test_background_free_with_black_bg(self, sample_bgr_frame, sample_circular_contour):
        """background_free=True, black_background=True should mask background to black."""
        result = extract_blob_background(
            sample_bgr_frame,
            [sample_circular_contour],
            contour=sample_circular_contour,
            channel=3,
            background_free=True,
            black_background=True
        )

        assert result is not None


    def test_background_free_with_white_bg(self, sample_bgr_frame, sample_circular_contour):
        """background_free=True, black_background=False should mask background to white."""
        result = extract_blob_background(
            sample_bgr_frame,
            [sample_circular_contour],
            contour=sample_circular_contour,
            channel=3,
            background_free=True,
            black_background=False
        )

        assert result is not None



class TestExtractBlobAll:
    """Tests for extract_blob_all() function."""

    def test_returns_array(self, sample_bgr_frame, sample_circular_contour):
        """Should return a numpy array."""
        y_bt, y_tp, x_lf, x_rt = crop_frame(sample_bgr_frame, [sample_circular_contour])
        result = extract_blob_all(
            sample_bgr_frame,
            y_bt, y_tp, x_lf, x_rt,
            contours=[sample_circular_contour],
            channel=1
        )

        assert isinstance(result, np.ndarray)

    def test_grayscale_output(self, sample_bgr_frame, sample_circular_contour):
        """channel=1 should return grayscale."""
        y_bt, y_tp, x_lf, x_rt = crop_frame(sample_bgr_frame, [sample_circular_contour])
        result = extract_blob_all(
            sample_bgr_frame,
            y_bt, y_tp, x_lf, x_rt,
            contours=[sample_circular_contour],
            channel=1
        )

        assert result.shape[-1] == 1


    def test_rgb_output(self, sample_bgr_frame, sample_circular_contour):
        """channel=3 should return RGB."""
        y_bt, y_tp, x_lf, x_rt = crop_frame(sample_bgr_frame, [sample_circular_contour])
        result = extract_blob_all(
            sample_bgr_frame,
            y_bt, y_tp, x_lf, x_rt,
            contours=[sample_circular_contour],
            channel=3
        )

        assert result.shape[-1] == 3


    def test_with_background_masking(self, sample_bgr_frame, sample_circular_contour):
        """background_free=True should apply contour masking."""
        y_bt, y_tp, x_lf, x_rt = crop_frame(sample_bgr_frame, [sample_circular_contour])
        result = extract_blob_all(
            sample_bgr_frame,
            y_bt, y_tp, x_lf, x_rt,
            contours=[sample_circular_contour],
            channel=3,
            background_free=True
        )
        assert result is not None


