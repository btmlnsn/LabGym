"""
/tests.unit.gui.test_gui_analyzer

Tests for gui_analyzer panels
"""

import pytest

from LabGym import mywx
import wx


from LabGym.gui_analyzer import (
    PanelLv2_AnalyzeBehaviors,
    PanelLv2_MineResults,
    PanelLv2_PlotBehaviors,
    PanelLv2_CalculateDistances,
)

pytestmark = pytest.mark.gui


def test_panel_analyze_behaviors_instantiation(gui_panel_setup):
    """PanelLv2_AnalyzeBehaviors instantiates."""
    panel = PanelLv2_AnalyzeBehaviors(gui_panel_setup)
    assert panel is not None


def test_panel_mine_results_instantiation(gui_panel_setup):
    """PanelLv2_MineResults instantiates."""
    panel = PanelLv2_MineResults(gui_panel_setup)
    assert panel is not None


def test_panel_plot_behaviors_instantiation(gui_panel_setup):
    """PanelLv2_PlotBehaviors instantiates."""
    panel = PanelLv2_PlotBehaviors(gui_panel_setup)
    assert panel is not None


def test_panel_calculate_distances_instantiation(gui_panel_setup):
    """PanelLv2_CalculateDistances instantiates."""
    panel = PanelLv2_CalculateDistances(gui_panel_setup)
    assert panel is not None

    