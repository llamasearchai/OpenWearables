"""
OpenWearables Core Package

Comprehensive wearable health monitoring platform with advanced analytics,
real-time processing, and intelligent health insights.
"""

from .sensor_manager import SensorManager, SensorInterface
from .data_processor import DataProcessor, ProcessedData
from .health_analyzer import HealthAnalyzer, HealthInsight
from .privacy import PrivacyManager, EncryptedData
from .advanced_analytics import AdvancedAnalyticsEngine, CircadianRhythmAnalyzer, StressPatternAnalyzer, HealthPattern, PredictiveInsight
from .performance_optimizer import PerformanceManager, SystemMonitor, PerformanceOptimizer

# Import device types
from .devices import SmartGlassesDevice, SmartHeadphonesDevice, SmartWatchDevice

# Import OpenWearables core
from .architecture import OpenWearablesCore

__all__ = [
    # Core management
    "OpenWearablesCore",
    
    # Sensor management
    "SensorManager",
    "SensorInterface",
    
    # Data processing
    "DataProcessor", 
    "ProcessedData",
    
    # Health analysis
    "HealthAnalyzer",
    "HealthInsight",
    
    # Privacy and security
    "PrivacyManager",
    "EncryptedData",
    
    # Advanced analytics
    "AdvancedAnalyticsEngine",
    "CircadianRhythmAnalyzer", 
    "StressPatternAnalyzer",
    "HealthPattern",
    "PredictiveInsight",
    
    # Performance optimization
    "PerformanceManager",
    "SystemMonitor",
    "PerformanceOptimizer",
    
    # Device implementations
    "SmartGlassesDevice",
    "SmartHeadphonesDevice", 
    "SmartWatchDevice"
] 