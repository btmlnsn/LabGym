"""
LabGym.ui.gui.app_icon
Centralized helpers for setting application-window / task-bar / dock icon

All disk access is delegated to 'LabGym.ui.assets.loader', so this file is the only
GUI module that knows the package-data path.
"""

from __future__ import annotations

# Standard library imports
import sys
from contextlib import contextmanager
from typing import Iterator

# Related third-party imports
import wx

# Cross-platform asset access
from LabGym.ui.assets import loader as _res


@contextmanager
def wx_app_icon(name: str = "labgym.ico") -> Iterator[None]:
	"""
	Yields a wx.Icon instance ready to be used with `frame.SetIcon()`.

	When Labgym is frozen by PyInstaller one-file, the underlying file is a 
	temporary extract - hence the context manager to keep it alive.
	"""
	with _res.resource_tmp(name) as icon_path:
		yield wx.Icon(str(icon_path))

	
# OS-specifc niceties
def _set_windows_app_id(app_id: str = "umyelab.LabGym") -> None:
	"""Attach a fixed AppUserModelID so the task-bar groups windows correctly."""
	import ctypes
	ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)


def _set_macos_dock_icon(name: str = "labgym.icns") -> None:
	"""Set the Dock icon at runtime (requires optional PyObjC)."""

	try:
		from AppKit import NSApplication, NSImage  # type: ignore

	except Exception:
		return

	with _res.resource_tmp(name) as icns:
		img = NSImage. alloc().initWithContentsOfFile_(str(icns))
		if img:
			NSApplication.sharedApplication().setApplicationIconImage_(img)


def setup_application_icons() -> None:
	"""
	Call once, early in GUI start-up. Sets:
		- main window icon (developers still call wx_app_icon() per-frame)
		- task-bar / dock icon on Windows / macOS
	"""
	if sys.platform.startswith("win"):
		_set_windows_app_id()
	elif sys.platform == "darwin":
		_set_macos_dock_icon()
