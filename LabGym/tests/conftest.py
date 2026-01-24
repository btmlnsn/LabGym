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


@pytest.fixture
def normal_series_large():
    """
    50-sample Series drawn from normal distribution.
    Large enough sample to reliably pass Shapiro-Wilk test.
    """
    import numpy as np
    import pandas as pd
    np.random.seed(42)
    
    return pd.Series(np.random.normal(loc = 100, scale = 10, size = 50))


@pytest.fixture
def normal_series_small():
    """
    10-sample Series drawn from normal distrubtion.
    Smaller sample, should still pass Shaprio-Wilk test.
    """
    import numpy as np
    import pandas as pd
    np.random.seed(42)

    return pd.Series(np.random.normal(loc = 100, scale = 10, size = 10))


@pytest.fixture
def non_normal_series():
    """
    50-sample Series drawn from exponential distribution.
    Should fail Shapiro-Wilk normality test.
    """
    import numpy as np
    import pandas as pd
    np.random.seed(42)

    return pd.Series(np.random.exponential(scale = 10, size = 50))


@pytest.fixture
def identical_values_series():
    """
    Series with all identical values (edge case for normality test)
    """
    import pandas as pd

    return pd.Series([5.0] * 20)


@pytest.fixture
def series_with_nans():
    """
    Series containing NaN values to test dropna() handling
    """
    import numpy as np
    import pandas as pd
    np.random.seed(42)

    data = np.random.normal(100, 10, 20)
    data[5] = np.nan
    data[10] = np.nan
    data[15] = np.nan

    return pd.Series(data)


@pytest.fixture
def two_groups_normal_same():
    """
    Two groups from same normal distribution (no significant difference expected).
    Format: [{behavior: {parameter: Series}}, {behavior: {parameter: Series}}]
    """
    import numpy as np
    import pandas as pd
    np.random.seed(42)

    group_a = {
        'walking': {
            'speed': pd.Series(np.random.normal(10, 2, 30)),
            'duration': pd.Series(np.random.normal(5, 1, 30)),
        }
    }
    # Same distribution parameters
    np.random.seed(43)
    group_b = {
        'walking': {
            'speed': pd.Series(np.random.normal(10, 2, 30)),
            'duration': pd.Series(np.random.normal(5, 1, 30)),
        }
    }
    
    return [group_a, group_b]


@pytest.fixture
def two_groups_normal_different():
    """
    Two groups from different normal distributions (significant difference expected).
    Group B has higher mean for 'speed'.
    """
    import numpy as np
    import pandas as pd
    np.random.seed(42)
    
    group_a = {
        'walking': {
            'speed': pd.Series(np.random.normal(10, 1, 30)),  # mean=10
            'duration': pd.Series(np.random.normal(5, 1, 30)),
        }
    }
    np.random.seed(43)
    group_b = {
        'walking': {
            'speed': pd.Series(np.random.normal(20, 1, 30)),  # mean=20, clearly different
            'duration': pd.Series(np.random.normal(5, 1, 30)),
        }
    }
    
    return [group_a, group_b]


@pytest.fixture
def two_groups_non_normal():
    """
    Two groups from exponential distributions (non-normal).
    """
    import numpy as np
    import pandas as pd
    np.random.seed(42)

    group_a = {
        'walking': {
            'speed': pd.Series(np.random.exponential(10, 30)),
            'duration': pd.Series(np.random.exponential(5, 30)),
        }
    }
    np.random.seed(43)
    group_b = {
        'walking': {
            'speed': pd.Series(np.random.exponential(25, 30)),  # Different scale
            'duration': pd.Series(np.random.exponential(5, 30)),
        }
    }

    return [group_a, group_b]


@pytest.fixture
def multi_groups_normal(request):
    """
    4 groups from normal distributions with increasing means.
    Parametrizable via indirect fixture if needed.
    """
    import numpy as np
    import pandas as pd
    np.random.seed(42)

    groups = []
    for i in range(4):
        np.random.seed(42 + i)
        group = {
            'walking': {
                'speed': pd.Series(np.random.normal(10 + i * 5, 2, 30)),
                'duration': pd.Series(np.random.normal(5 + i, 1, 30)),
            }
        }
        groups.append(group)
   
    return groups


@pytest.fixture
def multi_groups_non_normal():
    """
    4 groups from exponential distributions (non-normal).
    """
    import numpy as np
    import pandas as pd
    np.random.seed(42)

    groups = []
    for i in range(4):
        np.random.seed(42+i)
        group = {
            'walking': {
                'speed': pd.Series(np.random.exponential(10 + i * 3, 30)),
                'duration': pd.Series(np.random.exponential(5 + i, 30)),
            }
        }
        groups.append(group)
    
    return groups


@pytest.fixture
def control_group_normal():
    """
    A single control group (nromal distribution) for use with control_in parameter.
    """
    import numpy as np
    import pandas as pd
    np.random.seed(99)

    return {
        'walking': {
            'speed': pd.Series(np.random.normal(10, 2, 30)),
            'duration': pd.Series(np.random.normal(5, 1, 30)),
        }
    }


@pytest.fixture
def sample_file_names_two():
    """File names for two-group comparisons"""
    
    return ['Group_A', 'Group_B']


@pytest.fixture
def sample_file_names_four():
    """File names for four-group comparisons"""
    
    return ['Control', 'Treatment_1', 'Treatment_2', 'Treatment_3']
    

