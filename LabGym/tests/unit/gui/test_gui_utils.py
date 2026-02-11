"""
/tests.unit.gui.test_gui_utils

Tests for gui_utils (add_or_select_notebook_page)
"""

# Related third party imports
import pytest

# Local application imports
from LabGym.gui_utils import add_or_select_notebook_page


@pytest.mark.gui
def test_add_or_select_notebook_page_home_select_only():
    """If title is 'Home', only select existing Home tab; do not add page."""
    class MockNotebook:
        def __init__(self):
            self.pages = [("Home", None)]
            self.selected = -1
            self.added = []

        def GetPageCount(self):
            return len(self.pages)

        def GetPageText(self, i):
            return self.pages[i][0]

        def SetSelection(self, i):
            self.selected = i

        def AddPage(self, panel, title, select=True):
            self.added.append((panel, title, select))

    nb = MockNotebook()
    add_or_select_notebook_page(nb, lambda: None, "Home")
    assert nb.selected == 0
    assert nb.added == []


@pytest.mark.gui
def test_add_or_select_notebook_page_select_existing():
    """If page with title exists, select it and do not call factory."""
    class MockNotebook:
        def __init__(self):
            self.pages = [("A", None), ("Preprocess Videos", None)]
            self.selected = -1
            self.added = []
            self.factory_called = False

        def GetPageCount(self):
            return len(self.pages)

        def GetPageText(self, i):
            return self.pages[i][0]

        def SetSelection(self, i):
            self.selected = i

        def AddPage(self, panel, title, select=True):
            self.added.append((panel, title, select))

    def factory():
        nb.factory_called = True
        return "panel"

    nb = MockNotebook()
    add_or_select_notebook_page(nb, factory, "Preprocess Videos")
    assert nb.selected == 1
    assert not nb.factory_called
    assert nb.added == []


@pytest.mark.gui
def test_add_or_select_notebook_page_add_new():
    """If page with title does not exist, call factory and AddPage."""
    class MockNotebook:
        def __init__(self):
            self.pages = [("Home", None)]
            self.selected = -1
            self.added = []

        def GetPageCount(self):
            return len(self.pages)

        def GetPageText(self, i):
            return self.pages[i][0]

        def SetSelection(self, i):
            self.selected = i

        def AddPage(self, panel, title, select=True):
            self.pages.append((title, panel))
            self.added.append((panel, title, select))

    created = []

    def factory():
        p = "new_panel"
        created.append(p)
        return p

    nb = MockNotebook()
    add_or_select_notebook_page(nb, factory, "Preprocess Videos")
    assert len(created) == 1
    assert created[0] == "new_panel"
    assert nb.added == [("new_panel", "Preprocess Videos", True)]

    