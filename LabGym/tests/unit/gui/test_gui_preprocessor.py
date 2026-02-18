"""
/tests.unit.gui.test_gui_preprocessor

Tests for gui_preprocessor panels
"""

import pytest

from LabGym import mywx
import wx

from LabGym.gui_preprocessor import (
    PanelLv2_ProcessVideos,
    PanelLv2_DrawMarkers,
)

pytestmark = pytest.mark.gui


def test_panel_process_videos_instantiation(gui_panel_setup):
    """PanelLv2_ProcessVideos instantiates and shows default path labels."""
    panel = PanelLv2_ProcessVideos(gui_panel_setup)
    assert panel is not None
    assert panel.text_inputvideos.GetLabel() == "None."
    assert panel.text_outputfolder.GetLabel() == "None."


def test_panel_draw_markers_instantiation(gui_panel_setup):
    """PanelLv2_DrawMarkers instantiates."""
    panel = PanelLv2_DrawMarkers(gui_panel_setup)
    assert panel is not None

