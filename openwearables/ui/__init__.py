"""
OpenWearables UI Module

Web-based user interface for the OpenWearables platform.
"""

# Prevent circular import issues when running as main module
import sys
import os

# Add the parent directory to the path if needed
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

__version__ = "1.0.0"
__author__ = "OpenWearables Team"

# Only import Flask app when not running as main module to avoid conflicts
if __name__ != '__main__' and 'openwearables.ui.app' not in sys.modules:
    try:
        from .app import app, socketio
        __all__ = ['app', 'socketio']
    except ImportError:
        # Handle case where dependencies aren't available
        __all__ = []
else:
    __all__ = [] 