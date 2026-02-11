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


def test_panel_generate_images_instantiation(wx_app, wx_notebook, mock_config):
    """PanelLv2_GenerateImages instantiates and shows default labels."""
    panel = PanelLv2_GenerateImages(wx_notebook)
    assert panel is not None
    assert panel.text_inputvideos.GetLabel() == "None."
    assert panel.text_outputfolder.GetLabel() == "None."


def test_panel_train_detectors_instantiation(wx_app, wx_notebook, mock_config):
    """PanelLv2_TrainDetectors instantiates."""
    panel = PanelLv2_TrainDetectors(wx_notebook)
    assert panel is not None


def test_panel_test_detectors_instantiation(wx_app, wx_notebook, mock_config):
    """PanelLv2_TestDetectors instantiates."""
    panel = PanelLv2_TestDetectors(wx_notebook)
    assert panel is not None

