"""
/tests.unit.gui.test_gui_categorizer

Tests for gui_categorizer panels
"""

import pytest

from LabGym import mywx
import wx

from LabGym.gui_categorizer import (
    PanelLv2_GenerateExamples,
    PanelLv2_TrainCategorizers,
    PanelLv2_SortBehaviors,
    PanelLv2_TestCategorizers,
)

pytestmark = pytest.mark.gui


def test_panel_generate_examples_instantiation(wx_app, wx_notebook, mock_config):
    """PanelLv2_GenerateExamples instantiates and shows default path label."""
    panel = PanelLv2_GenerateExamples(wx_notebook)
    assert panel is not None
    assert panel.text_inputvideos.GetLabel() == "None."


def test_panel_train_categorizers_instantiation(wx_app, wx_notebook, mock_config):
    """PanelLv2_TrainCategorizers instantiates."""
    panel = PanelLv2_TrainCategorizers(wx_notebook)
    assert panel is not None


def test_panel_sort_behaviors_instantiation(wx_app, wx_notebook, mock_config):
    """PanelLv2_SortBehaviors instantiates."""
    panel = PanelLv2_SortBehaviors(wx_notebook)
    assert panel is not None


def test_panel_test_categorizers_instantiation(wx_app, wx_notebook, mock_config):
    """PanelLv2_TestCategorizers instantiates."""
    panel = PanelLv2_TestCategorizers(wx_notebook)
    assert panel is not None

