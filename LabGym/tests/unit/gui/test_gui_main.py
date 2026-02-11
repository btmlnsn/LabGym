"""
/tests.unit.gui.test_gui_main

Tests for gui_main panels and MainFrame
"""

# Related third party imports
import pytest
from LabGym import mywx
import wx
import wx.aui

# Local application imports
from LabGym.gui_main import(
    InitialPanel,
    PanelLv1_ProcessModule,
    PanelLv1_TrainingModule,
    PanelLv1_AnalysisModule,
    MainFrame,
)

pytestmark = pytest.mark.gui


def test_initial_panel_instantiation(wx_app, wx_notebook):
    """InitialPanel can be created and has welcome text and module buttons."""
    panel = InitialPanel(wx_notebook)
    assert panel is not None
    assert hasattr(panel, "text_welcome")
    assert "Welcome" in panel.text_welcome.GetLabel()
    buttons = [c for c in panel.GetChildren() if isinstance(c, wx.Button)]
    labels = [b.GetLabel() for b in buttons]
    assert "Preprocessing Module" in labels
    assert "Training Module" in labels
    assert "Analysis Module" in labels


def test_panel_lv1_process_module_instantiation(wx_app, wx_notebook):
    """PanelLv1_ProcessModule instantiates and has expected buttons."""
    panel = PanelLv1_ProcessModule(wx_notebook)
    assert panel is not None
    buttons = [c for c in panel.GetChildren() if isinstance(c, wx.Button)]
    labels = [b.GetLabel() for b in buttons]
    assert "Preprocess Videos" in labels
    assert "Draw Markers" in labels


def test_panel_lv1_training_module_instantiation(wx_app, wx_notebook):
    """PanelLv1_TrainingModule instantiates and has expected buttons."""
    panel = PanelLv1_TrainingModule(wx_notebook)
    assert panel is not None
    buttons = [c for c in panel.GetChildren() if isinstance(c, wx.Button)]
    labels = [b.GetLabel() for b in buttons]
    assert "Generate Image Examples" in labels
    assert "Train Detectors" in labels
    assert "Train Categorizers" in labels


def test_panel_lv1_analysis_module_instantiation(wx_app, wx_notebook):
    """PanelLv1_AnalysisModule instantiates and has expected buttons."""
    panel = PanelLv1_AnalysisModule(wx_notebook)
    assert panel is not None
    buttons = [c for c in panel.GetChildren() if isinstance(c, wx.Button)]
    labels = [b.GetLabel() for b in buttons]
    assert "Analyze Behaviors" in labels
    assert "Mine Results" in labels


def test_main_frame_instantiation(wx_app):
    """MainFrame can be created and has notebook with Home tab."""
    frame = MainFrame()
    assert frame is not None
    assert frame.notebook.GetPageCount() >= 1
    assert frame.notebook.GetPageText(0) == "Home"
    frame.Destroy()

