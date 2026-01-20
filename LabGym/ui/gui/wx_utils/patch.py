"""Monkeypatch wx.App to be a strict-singleton.

Import this module before the first import of wx, to patch wx.App to be
a strict-singleton before an unpatched instance of wx.App can possibly
be created.
"""

# Allow use of newer syntax Python 3.10 type hints in Python 3.9.
from __future__ import annotations

# Standard library imports.
import logging
import sys
import textwrap
import wx

# Log the load of this module (by the module loader, on first import).
# Intentionally positioning these statements before other imports, against the
# guidance of PEP 8, to log the load before other imports log messages.
logger = logging.getLogger(__name__)
logger.debug('%s', f'loading {__name__}')

# Related third party imports.
# Idempotent guard
_marker = "_labgym_patch_applied"
patched: bool = getattr(wx, _marker, False)

if not patched:
	setattr(wx, _marker, True)

	# Local application/library specific imports.
	# None

	class StrictSingleton(wx.App):
		_instance = None  # Class variable to hold the single instance

		def __new__(cls, *args, **kwargs):
			logger.debug('patched __new__ -- entered')
			if cls._instance is None:
				logger.debug('patched __new__ -- instantiating')
				cls._instance = super().__new__(cls)
			else:
				raise AssertionError('wx.App() is called once at most.')
			logger.debug(f'patched __new__ -- returning {cls._instance}')
			wx.mywx_AppCount += 1
			return cls._instance

		def __init__(self, *args, **kwargs):
			"""Initialize the app only once, ensuring wx recognizes it."""
			# Only initialize if not already initialized
			if not hasattr(self, '_initialized'):
				logger.debug('patched __init__ -- initializing')
				super().__init__(*args, **kwargs)
				self._initialized = True

	# monkeypatch wx.App
	wx.mywx_AppCount = 0
	wx.App = StrictSingleton
	_original_GetApp = wx.GetApp
	def patched_GetApp():
		# "Return the StrictSingleton instance if it exists; otherwise, use the original"
		if StrictSingleton._instance is not None:
			return StrictSingleton._instance
		return _original_GetApp()
	wx.GetApp = patched_GetApp
