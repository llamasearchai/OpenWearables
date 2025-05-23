"""
OpenWearables Device Implementations

This module contains implementations of various wearable devices including
smart glasses, smart headphones, and smart watches.
"""

from .smart_glasses import SmartGlassesDevice
from .smart_headphones import SmartHeadphonesDevice
from .smart_watch import SmartWatchDevice

__all__ = [
    "SmartGlassesDevice",
    "SmartHeadphonesDevice",
    "SmartWatchDevice"
] 