"""
Swift MLX Bridge Package

Provides comprehensive Swift and MLX integration for OpenWearables platform
with real-time data streaming, model processing, and native Apple ecosystem support.
"""

from .swift_bridge import SwiftMLXBridge, SwiftSensorData, SwiftHealthInsight, SwiftDeviceConfig
from .model_bridge import MLXModelBridge, ModelPrediction, HealthInsight, HealthAnomalyDetector, HeartRateAnalyzer, SleepQualityAnalyzer, ActivityClassifier
from .data_bridge import DataBridge, RealTimeDataStream, SensorDataPacket, DataStreamConfig, StreamMetrics

__all__ = [
    # Swift Bridge Components
    "SwiftMLXBridge",
    "SwiftSensorData", 
    "SwiftHealthInsight",
    "SwiftDeviceConfig",
    
    # Model Bridge Components
    "MLXModelBridge",
    "ModelPrediction",
    "HealthInsight",
    "HealthAnomalyDetector",
    "HeartRateAnalyzer", 
    "SleepQualityAnalyzer",
    "ActivityClassifier",
    
    # Data Bridge Components
    "DataBridge",
    "RealTimeDataStream",
    "SensorDataPacket",
    "DataStreamConfig",
    "StreamMetrics"
] 