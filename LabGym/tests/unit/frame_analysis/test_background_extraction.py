"""
/tests.unit.frame_analysis.test_background_extraction

Tests for extract_background()
Function extracts a static background from a list of video frames.
"""

# Related third party imports
import numpy as np
import pytest

# Local application imports
from LabGym.tools import extract_background



class TestExtractBackgroundEdgeCases:
    """Test edge cases and boundary conditions for extract_background()"""

    def test_empty_list_returns_none(self):
        """Empty frame list should return None"""
        result = extract_background([])
        assert result is None

    def test_single_frame_returns_none(self):
        """Single frame shoud return None (need >3 frames)"""
        frame = np.full((100, 100, 3), 128, dtype = np.uint8)
        result = extract_background([frame])
        assert result is None

    def test_two_Frames_returns_none(self):
        """Two frames should return None"""
        frame = np.full((100, 100, 3), 128, dtype = np.uint8)
        result = extract_background([frame, frame])
        assert result is None

    def test_three_frames_returns_none(self):
        """Three frames should return  None (need >3 frames)"""
        frame = np.full((100, 100, 3), 128, dtype = np.uint8)
        result = extract_background([frame, frame, frame])
        assert result is None

    def test_four_frames_returns_background(self):
        """Four frames should return a valid background."""
        frame = np.full((100, 100, 3), 128, dtype = np.uint8)
        result = extract_background([frame, frame, frame, frame])
        assert result is not None
        assert result.shape == frame.shape

    

class TestExtractBackgroundOutput:
    """Test output properties."""

    def test_output_shape_matches_input(self, sample_video_frames_bgr):
        """Output should match input frame shape."""
        result = extract_background(sample_video_frames_bgr)
        assert result.shape == sample_video_frames_bgr[0].shape

    def test_output_dtype_is_uint8(self, sample_video_frames_bgr):
        """Output dtype should be uint8"""
        result = extract_background(sample_video_frames_bgr)
        assert result.dtype == np.uint8

    

class TestExtractBackgroundAnimalVsBg:
    """Test animal_vs_bg parameter variations"""
    
    def test_animal_brighter_than_bg(self, sample_video_frames_bgr):
        """animal_vs_bg=0: animals brighter than background"""
        result = extract_background(
            sample_video_frames_bgr,
            stable_illumination=True,
            animal_vs_bg=0,
        )
        
        assert result is not None

    def test_animal_darker_than_bg(self, sample_video_frames_bgr):
        """animal_vs_bg=1: animals darker than background"""
        result = extract_background(
            sample_video_frames_bgr,
            stable_illumination = True,
            animal_vs_bg = 1
        )

        assert result is not None

    def test_animal_unclear(self, sample_video_frames_bgr):
        """animal_vs_bg=2: hard to tell (uses median)"""
        result = extract_background(
            sample_video_frames_bgr,
            stable_illumination = True,
            animal_vs_bg = 2
        )

        assert result is not None



class TestExtractBackgroundIllumination:
    """Test stable_illumination parameter."""

    def test_stable_illumination_true(self, sample_video_frames_bgr):
        """stable_illumination=True uses simpler algorithm"""
        result = extract_background(
            sample_video_frames_bgr,
            stable_illumination = True,
            animal_vs_bg = 0
        )

        assert result is not None


    def test_stable_illumination_false(self, sample_video_frames_bgr):
        """stable_illumination=False handles varying lighting"""
        result = extract_background(
            sample_video_frames_bgr,
            stable_illumination = False,
            animal_vs_bg = 0
        )

        assert result is not None



class TestExtractBackgroundManyFrames:
    """Test with >101 frames (triggers diffent code path)"""

    def test_many_frames_animal_brighter(self):
        """Over 101 frames with animal_vs_bg=0."""
        frames = [
            np.full((50, 50, 3), 128, dtype = np.uint8)
            for _ in range(120)
        ]

        result = extract_background(frames, animal_vs_bg = 0)

        assert result is not None

    
    def test_many_frames_animal_unclear(self):
        """Over 101 frames with animal_vs_bg=2 (complex algorithm)"""
        frames = [
            np.full((50, 50, 3), 128, dtype = np.uint8)
            for _ in range(120)
        ]
        
        result = extract_background(frames, animal_vs_bg = 2)

        assert result is not None


