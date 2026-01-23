"""
/tests/conftest.py

Shared pytest fixtures for LabGym tests
"""

# Standard library imports
import sys

# Related third party imports
import pytest

@pytest.fixture(scope="session")
def wx_app():
    from LabGym import mywx
    import wx
    app = wx.App()
    yield app


@pytest.fixture
def mock_argv(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['LabGym'])


@pytest.fixture
def sample_bgr_frame():
    """100x100 BGR frame with gray background (200) and darker circle (50)"""
    import numpy as np
    import cv2
    frame = np.full((100,100, 3), 200, dtype = np.uint8)
    cv2.circle(frame, (50, 50), 15, (50, 50, 50), -1)
    return frame


@pytest.fixture
def sample_background_frame():
    """100x100 BGR frame - uniform gray background, no animal or circle."""
    import numpy as np
    return np.full((100, 100, 3), 200, dtype = np.uint8)


@pytest.fixture
def sample_video_frames_bgr():
    """60 BGR frames witha circle moving left-to-right (simulates animal)"""
    import numpy as np
    import cv2
    frames = []
    for i in range(60):
        frame = np.full((100, 100, 3), 200, dtype = np.uint8)
        x_pos = 20 + (i % 60)
        cv2.circle(frame, (x_pos, 50), 15, (50, 50, 50), -1)
        frames.append(frame)
    return frames


@pytest.fixture
def sample_circular_contour():
    """"A circular contour centered at (50, 50) with radius 15"""
    import numpy as np
    angles = np.linspace(0, 2 * np.pi, 50, endpoint = False)
    contour = np.array([
        [[int(50 + 15 * np.cos(a)), int(50 + 15 * np.sin(a))]]
        for a in angles],
        dtype = np.int32) 

    return contour


@pytest.fixture
def sample_rectangular_contour():
    """A rectangular contour from (30, 30) to (70, 70)"""
    import numpy as np
    contour = np.array([
        [[30, 30]],
        [[70, 30]],
        [[70, 70]],
        [[30, 70]]], dtype = np.int32)
    
    return contour


