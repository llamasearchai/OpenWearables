"""
OpenWearables: AI-Powered Wearable Health Monitoring Platform

A comprehensive platform for real-time wearable health monitoring and analysis
with hardware-optimized AI models for Apple Silicon, NVIDIA GPUs, and CPU.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

__version__ = "1.0.0"
__author__ = "Nik Jois <nikjois@llamasearch.ai>"
__email__ = "dev@openwearables.org"
__license__ = "MIT"

# Core imports
from openwearables.core.architecture import OpenWearablesCore
from openwearables.core.sensor_manager import SensorManager, SensorInterface
from openwearables.core.data_processor import DataProcessor
from openwearables.core.health_analyzer import HealthAnalyzer
from openwearables.core.privacy import PrivacyManager

# Version information
VERSION = __version__

# Package metadata
__all__ = [
    "__version__",
    "VERSION",
    "OpenWearablesCore",
    "SensorManager", 
    "SensorInterface",
    "DataProcessor",
    "HealthAnalyzer",
    "PrivacyManager",
]

# Configuration and logging setup
import logging
from pathlib import Path

# Default configuration
DEFAULT_CONFIG_PATH = os.path.join(Path.home(), ".openwearables", "config.json")
DEFAULT_DATA_PATH = os.path.join(Path.home(), ".openwearables", "data")
DEFAULT_LOG_PATH = os.path.join(Path.home(), ".openwearables", "logs")

# Ensure directories exist
for path in [DEFAULT_DATA_PATH, DEFAULT_LOG_PATH, os.path.dirname(DEFAULT_CONFIG_PATH)]:
    os.makedirs(path, exist_ok=True)

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DEFAULT_LOG_PATH, "openwearables.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"OpenWearables v{__version__} initialized")

def get_version() -> str:
    """Return the current version of OpenWearables."""
    return __version__

def get_config_path() -> str:
    """Return the default configuration path."""
    return DEFAULT_CONFIG_PATH

def get_data_path() -> str:
    """Return the default data path."""
    return DEFAULT_DATA_PATH

# Only import Flask app when not running as main module to avoid conflicts 