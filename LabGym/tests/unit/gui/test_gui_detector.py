"""
/tests.unit.gui.test_gui_detector

Tests for gui_detector panels
"""

import pytest

from LabGym import mywx
import wx

from LabGym.gui_detector import (
    PanelLv2_GenerateImages,
    PanelLv2_TrainDetectors,
    PanelLv2_TestDetectors,
)

pytestmark = pytest.mark.gui


def test_panel_generate_images_instantiation(gui_panel_setup):
    """PanelLv2_GenerateImages instantiates and shows default labels."""
    panel = PanelLv2_GenerateImages(gui_panel_setup)
    assert panel is not None
    assert panel.text_inputvideos.GetLabel() == "None."
    assert panel.text_outputfolder.GetLabel() == "None."


def test_panel_train_detectors_instantiation(gui_panel_setup):
    """PanelLv2_TrainDetectors instantiates."""
    panel = PanelLv2_TrainDetectors(gui_panel_setup)
    assert panel is not None


def test_panel_test_detectors_instantiation(gui_panel_setup):
    """PanelLv2_TestDetectors instantiates."""
    panel = PanelLv2_TestDetectors(gui_panel_setup)
    assert panel is not None

