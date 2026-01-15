from .main import main_window

from .analyzer import (
    PanelLv2_AnalyzeBehaviors,
    PanelLv2_MineResults,
    PanelLv2_PlotBehaviors,
    PanelLv2_CalculateDistances,
)

from .categorizer import (
    PanelLv2_GenerateExamples,
    PanelLv2_TrainCategorizers,
    PanelLv2_SortBehaviors,
    PanelLv2_TestCategorizers
)

from .detector import (
    PanelLv2_GenerateImages,
    PanelLv2_TrainDetectors,
    PanelLv2_TestDetectors
)

from .preprocessor import (
    PanelLv2_ProcessVideos,
    PanelLv2_DrawMarkers
)

from .utils import add_or_select_notebook_page
from .app_icon import set_frame_icon, setup_application_icons

__all__ = [
    'main_window',
    'PanelLv2_AnalyzeBehaviors',
    'PanelLv2_MineResults',
    'PanelLv2_PlotBehaviors',
    'PanelLv2_CalculateDistances',
    'PanelLv2_GenerateExamples',
    'PanelLv2_TrainCategorizers',
    'PanelLv2_SortBehaviors',
    'PanelLv2_TestCategorizers',
    'PanelLv2_GenerateImages',
    'PanelLv2_TrainDetectors',
    'PanelLv2_TestDetectors',
    'PanelLv2_ProcessVideos',
    'PanelLv2_DrawMarkers',
    'add_or_select_notebook_page',
    'set_frame_icon',
    'setup_application_icons',
    'gui_main',
    'gui_analyzer',
    'gui_categorizer',
    'gui_detector',
    'gui_preprocessor',
    'gui_utils',
    'gui_app_icon',
]
