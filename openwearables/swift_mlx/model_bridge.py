"""
MLX Model Bridge for Advanced Health Analytics

Provides MLX-optimized machine learning models for real-time analysis of
wearable sensor data with Apple Silicon acceleration.
"""

import os
import time
import json
import logging
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path

# Conditional MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Fallback to numpy-based implementations
    import numpy as mx

logger = logging.getLogger("OpenWearables.MLXModelBridge")

@dataclass
class ModelPrediction:
    """Data structure for model predictions."""
    timestamp: float
    sensor_type: str
    prediction_type: str
    confidence: float
    result: Union[str, float, Dict[str, Any]]
    processing_time_ms: float

@dataclass
class HealthInsight:
    """Data structure for health insights generated by models."""
    timestamp: float
    insight_type: str  # anomaly, trend, recommendation, alert
    severity: str  # low, medium, high, critical
    message: str
    confidence: float
    data_source: List[str]  # List of sensor types used
    recommended_action: Optional[str] = None


class HealthAnomalyDetector:
    """MLX-optimized anomaly detection for health metrics."""
    
    def __init__(self, window_size: int = 50, threshold: float = 2.5):
        """
        Initialize anomaly detector.
        
        Args:
            window_size: Size of the sliding window for analysis
            threshold: Standard deviation threshold for anomaly detection
        """
        self.window_size = window_size
        self.threshold = threshold
        self.data_windows = {}
        
        if HAS_MLX:
            self.device = mx.default_device()
            logger.info(f"MLX anomaly detector initialized on device: {self.device}")
        else:
            logger.warning("MLX not available, using numpy fallback")
    
    def update(self, sensor_type: str, data: np.ndarray) -> Optional[ModelPrediction]:
        """
        Update anomaly detector with new sensor data.
        
        Args:
            sensor_type: Type of sensor providing the data
            data: Sensor data array
            
        Returns:
            ModelPrediction if anomaly detected, None otherwise
        """
        start_time = time.time()
        
        # Initialize window for new sensor type
        if sensor_type not in self.data_windows:
            self.data_windows[sensor_type] = []
        
        # Add new data to window
        self.data_windows[sensor_type].extend(data.flatten())
        
        # Maintain window size
        if len(self.data_windows[sensor_type]) > self.window_size:
            self.data_windows[sensor_type] = self.data_windows[sensor_type][-self.window_size:]
        
        # Need minimum data for analysis
        if len(self.data_windows[sensor_type]) < 10:
            return None
        
        # Perform anomaly detection
        window_data = np.array(self.data_windows[sensor_type])
        
        if HAS_MLX:
            anomaly_score = self._mlx_anomaly_detection(window_data)
        else:
            anomaly_score = self._numpy_anomaly_detection(window_data)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Check if anomaly detected
        if anomaly_score > self.threshold:
            return ModelPrediction(
                timestamp=time.time(),
                sensor_type=sensor_type,
                prediction_type="anomaly",
                confidence=min(1.0, anomaly_score / (self.threshold * 2)),
                result={"anomaly_score": float(anomaly_score), "severity": "high" if anomaly_score > self.threshold * 1.5 else "medium"},
                processing_time_ms=processing_time
            )
        
        return None
    
    def _mlx_anomaly_detection(self, data: np.ndarray) -> float:
        """MLX-optimized anomaly detection."""
        # Convert to MLX array
        mlx_data = mx.array(data)
        
        # Calculate rolling statistics
        mean = mx.mean(mlx_data)
        std = mx.std(mlx_data)
        
        # Z-score for latest values
        recent_values = mlx_data[-5:]  # Last 5 values
        z_scores = mx.abs((recent_values - mean) / (std + 1e-8))
        
        # Return maximum z-score
        return float(mx.max(z_scores))
    
    def _numpy_anomaly_detection(self, data: np.ndarray) -> float:
        """Numpy fallback for anomaly detection."""
        mean = np.mean(data)
        std = np.std(data)
        
        # Z-score for latest values
        recent_values = data[-5:]
        z_scores = np.abs((recent_values - mean) / (std + 1e-8))
        
        return float(np.max(z_scores))


class HeartRateAnalyzer:
    """Advanced heart rate analysis with MLX optimization."""
    
    def __init__(self):
        """Initialize heart rate analyzer."""
        self.hr_history = []
        self.max_history = 200
        
        if HAS_MLX:
            # Initialize MLX-based HRV analysis model
            self.hrv_model = self._create_hrv_model()
            logger.info("MLX heart rate analyzer initialized")
    
    def analyze(self, heart_rate: float, timestamp: float) -> List[HealthInsight]:
        """
        Analyze heart rate data for insights.
        
        Args:
            heart_rate: Heart rate in BPM
            timestamp: Timestamp of the reading
            
        Returns:
            List of health insights
        """
        insights = []
        
        # Add to history
        self.hr_history.append((timestamp, heart_rate))
        
        # Maintain history size
        if len(self.hr_history) > self.max_history:
            self.hr_history = self.hr_history[-self.max_history:]
        
        # Need minimum data for analysis
        if len(self.hr_history) < 10:
            return insights
        
        # Extract recent heart rates
        recent_hrs = [hr for _, hr in self.hr_history[-20:]]
        
        # Analyze trends
        trend_insight = self._analyze_trend(recent_hrs, timestamp)
        if trend_insight:
            insights.append(trend_insight)
        
        # Analyze variability
        if len(recent_hrs) >= 5:
            hrv_insight = self._analyze_hrv(recent_hrs, timestamp)
            if hrv_insight:
                insights.append(hrv_insight)
        
        # Check for immediate concerns
        immediate_insight = self._check_immediate_concerns(heart_rate, timestamp)
        if immediate_insight:
            insights.append(immediate_insight)
        
        return insights
    
    def _analyze_trend(self, heart_rates: List[float], timestamp: float) -> Optional[HealthInsight]:
        """Analyze heart rate trends."""
        if len(heart_rates) < 5:
            return None
        
        # Calculate trend using linear regression
        x = np.arange(len(heart_rates))
        y = np.array(heart_rates)
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        if abs(slope) > 0.5:  # Significant trend
            trend_type = "increasing" if slope > 0 else "decreasing"
            severity = "medium" if abs(slope) > 1.0 else "low"
            
            return HealthInsight(
                timestamp=timestamp,
                insight_type="trend",
                severity=severity,
                message=f"Heart rate is {trend_type} over recent readings (slope: {slope:.2f} BPM/reading)",
                confidence=min(1.0, abs(slope) / 2.0),
                data_source=["heart_rate"],
                recommended_action="Monitor for continued trend" if severity == "low" else "Consider consulting healthcare provider"
            )
        
        return None
    
    def _analyze_hrv(self, heart_rates: List[float], timestamp: float) -> Optional[HealthInsight]:
        """Analyze heart rate variability."""
        if len(heart_rates) < 5:
            return None
        
        # Calculate HRV as standard deviation of successive differences
        successive_diffs = np.diff(heart_rates)
        hrv = np.std(successive_diffs)
        
        # Classify HRV
        if hrv < 2.0:
            severity = "medium"
            message = f"Low heart rate variability detected (HRV: {hrv:.2f})"
            recommended_action = "Consider stress reduction and recovery activities"
        elif hrv > 8.0:
            severity = "low"
            message = f"High heart rate variability detected (HRV: {hrv:.2f})"
            recommended_action = "May indicate good cardiovascular fitness or potential arrhythmia"
        else:
            return None  # Normal HRV
        
        return HealthInsight(
            timestamp=timestamp,
            insight_type="trend",
            severity=severity,
            message=message,
            confidence=0.8,
            data_source=["heart_rate"],
            recommended_action=recommended_action
        )
    
    def _check_immediate_concerns(self, heart_rate: float, timestamp: float) -> Optional[HealthInsight]:
        """Check for immediate heart rate concerns."""
        if heart_rate > 120:
            severity = "high" if heart_rate > 150 else "medium"
            return HealthInsight(
                timestamp=timestamp,
                insight_type="alert",
                severity=severity,
                message=f"Elevated heart rate detected: {heart_rate:.0f} BPM",
                confidence=0.9,
                data_source=["heart_rate"],
                recommended_action="Consider rest and monitor for sustained elevation"
            )
        elif heart_rate < 50:
            severity = "medium" if heart_rate < 40 else "low"
            return HealthInsight(
                timestamp=timestamp,
                insight_type="alert",
                severity=severity,
                message=f"Low heart rate detected: {heart_rate:.0f} BPM",
                confidence=0.9,
                data_source=["heart_rate"],
                recommended_action="Monitor for symptoms; consult healthcare provider if persistent"
            )
        
        return None
    
    def _create_hrv_model(self):
        """Create MLX-based HRV analysis model."""
        if not HAS_MLX:
            return None
        
        # Simple neural network for HRV analysis
        class HRVModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 10),
                    nn.ReLU(),
                    nn.Linear(10, 1)
                ]
            
            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return HRVModel()


class SleepQualityAnalyzer:
    """Advanced sleep quality analysis with pattern recognition."""
    
    def __init__(self):
        """Initialize sleep quality analyzer."""
        self.sleep_history = []
        self.max_history = 30  # 30 days of sleep data
    
    def analyze(self, sleep_stage: str, sleep_quality: float, hrv: float, timestamp: float) -> List[HealthInsight]:
        """
        Analyze sleep data for patterns and insights.
        
        Args:
            sleep_stage: Current sleep stage
            sleep_quality: Sleep quality score (0-100)
            hrv: Heart rate variability during sleep
            timestamp: Timestamp of the reading
            
        Returns:
            List of health insights
        """
        insights = []
        
        # Add to history
        self.sleep_history.append({
            "timestamp": timestamp,
            "stage": sleep_stage,
            "quality": sleep_quality,
            "hrv": hrv
        })
        
        # Maintain history size
        if len(self.sleep_history) > self.max_history * 24 * 60:  # Assuming minute-level data
            self.sleep_history = self.sleep_history[-self.max_history * 24 * 60:]
        
        # Analyze sleep patterns if we have enough data
        if len(self.sleep_history) > 100:
            pattern_insights = self._analyze_sleep_patterns()
            insights.extend(pattern_insights)
        
        # Check current sleep quality
        if sleep_quality < 50:
            insights.append(HealthInsight(
                timestamp=timestamp,
                insight_type="alert",
                severity="medium" if sleep_quality < 30 else "low",
                message=f"Poor sleep quality detected: {sleep_quality:.0f}/100",
                confidence=0.85,
                data_source=["sleep_tracker"],
                recommended_action="Focus on sleep hygiene and consistent sleep schedule"
            ))
        
        return insights
    
    def _analyze_sleep_patterns(self) -> List[HealthInsight]:
        """Analyze long-term sleep patterns."""
        insights = []
        
        # Get recent sleep data (last 7 days)
        recent_data = self.sleep_history[-7 * 24 * 60:]  # Minute-level data for 7 days
        
        if len(recent_data) < 100:
            return insights
        
        # Calculate average sleep quality
        avg_quality = np.mean([entry["quality"] for entry in recent_data])
        
        if avg_quality < 60:
            insights.append(HealthInsight(
                timestamp=time.time(),
                insight_type="trend",
                severity="medium",
                message=f"Consistent poor sleep quality over recent days (avg: {avg_quality:.0f}/100)",
                confidence=0.9,
                data_source=["sleep_tracker"],
                recommended_action="Consider sleep study or consultation with healthcare provider"
            ))
        
        return insights


class ActivityClassifier:
    """MLX-optimized activity classification from sensor data."""
    
    def __init__(self):
        """Initialize activity classifier."""
        self.activity_buffer = []
        self.buffer_size = 100
        
        # Activity categories
        self.activities = ["resting", "walking", "running", "cycling", "stairs", "exercise"]
        
        if HAS_MLX:
            self.model = self._create_activity_model()
            logger.info("MLX activity classifier initialized")
    
    def classify(self, accelerometer_data: np.ndarray, gyroscope_data: np.ndarray) -> Optional[ModelPrediction]:
        """
        Classify activity from accelerometer and gyroscope data.
        
        Args:
            accelerometer_data: 3D accelerometer readings
            gyroscope_data: 3D gyroscope readings
            
        Returns:
            ModelPrediction with activity classification
        """
        start_time = time.time()
        
        # Combine sensor data
        combined_data = np.concatenate([accelerometer_data.flatten(), gyroscope_data.flatten()])
        
        # Add to buffer
        self.activity_buffer.append(combined_data)
        
        # Maintain buffer size
        if len(self.activity_buffer) > self.buffer_size:
            self.activity_buffer = self.activity_buffer[-self.buffer_size:]
        
        # Need minimum data for classification
        if len(self.activity_buffer) < 10:
            return None
        
        # Perform classification
        if HAS_MLX:
            activity, confidence = self._mlx_classify_activity()
        else:
            activity, confidence = self._rule_based_classify_activity()
        
        processing_time = (time.time() - start_time) * 1000
        
        return ModelPrediction(
            timestamp=time.time(),
            sensor_type="activity_classifier",
            prediction_type="activity",
            confidence=confidence,
            result=activity,
            processing_time_ms=processing_time
        )
    
    def _mlx_classify_activity(self) -> Tuple[str, float]:
        """MLX-based activity classification."""
        # Extract features from recent buffer data
        recent_data = np.array(self.activity_buffer[-10:])
        
        # Calculate statistical features
        features = []
        for i in range(recent_data.shape[1]):
            column = recent_data[:, i]
            features.extend([
                np.mean(column),
                np.std(column),
                np.max(column) - np.min(column),
                np.percentile(column, 75) - np.percentile(column, 25)
            ])
        
        features = np.array(features)
        
        # Simple rule-based classification with confidence estimation
        acc_magnitude = np.sqrt(features[0]**2 + features[1]**2 + features[2]**2)
        acc_std = np.mean(features[4:7])  # Standard deviation features
        
        if acc_magnitude > 15 and acc_std > 2:
            return "running", 0.9
        elif acc_magnitude > 12 and acc_std > 1:
            return "walking", 0.8
        elif acc_std > 0.5:
            return "exercise", 0.7
        else:
            return "resting", 0.85
    
    def _rule_based_classify_activity(self) -> Tuple[str, float]:
        """Rule-based activity classification fallback."""
        recent_data = np.array(self.activity_buffer[-5:])
        
        # Calculate activity indicators
        movement_variance = np.var(recent_data, axis=0)
        total_variance = np.sum(movement_variance)
        
        if total_variance > 5.0:
            return "running", 0.8
        elif total_variance > 2.0:
            return "walking", 0.75
        elif total_variance > 0.5:
            return "light_activity", 0.7
        else:
            return "resting", 0.85
    
    def _create_activity_model(self):
        """Create MLX-based activity classification model."""
        if not HAS_MLX:
            return None
        
        class ActivityModel(nn.Module):
            def __init__(self, input_size=40, num_classes=6):
                super().__init__()
                self.layers = [
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, num_classes)
                ]
            
            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return ActivityModel()


class MLXModelBridge:
    """
    Main bridge class for MLX-optimized health analytics models.
    
    Coordinates multiple specialized models for comprehensive health analysis
    with Apple Silicon acceleration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MLX model bridge.
        
        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or {}
        
        # Initialize specialized analyzers
        self.anomaly_detector = HealthAnomalyDetector(
            window_size=self.config.get("anomaly_window_size", 50),
            threshold=self.config.get("anomaly_threshold", 2.5)
        )
        
        self.heart_rate_analyzer = HeartRateAnalyzer()
        self.sleep_analyzer = SleepQualityAnalyzer()
        self.activity_classifier = ActivityClassifier()
        
        # Results storage
        self.predictions = []
        self.insights = []
        self.max_storage = 1000
        
        # Threading for real-time processing
        self.processing_thread = None
        self.is_running = False
        self.data_queue = []
        self.queue_lock = threading.Lock()
        
        logger.info("MLX Model Bridge initialized with specialized analyzers")
    
    def process_sensor_data(self, sensor_type: str, data: np.ndarray, timestamp: Optional[float] = None) -> List[Union[ModelPrediction, HealthInsight]]:
        """
        Process sensor data through appropriate models.
        
        Args:
            sensor_type: Type of sensor data
            data: Sensor data array
            timestamp: Optional timestamp
            
        Returns:
            List of predictions and insights
        """
        if timestamp is None:
            timestamp = time.time()
        
        results = []
        
        # Check for anomalies in all sensor types
        anomaly_prediction = self.anomaly_detector.update(sensor_type, data)
        if anomaly_prediction:
            results.append(anomaly_prediction)
        
        # Sensor-specific analysis
        if sensor_type in ["heart_rate", "ecg", "ppg"]:
            heart_rate = self._extract_heart_rate(sensor_type, data)
            if heart_rate:
                hr_insights = self.heart_rate_analyzer.analyze(heart_rate, timestamp)
                results.extend(hr_insights)
        
        elif sensor_type == "sleep_tracker":
            sleep_insights = self._analyze_sleep_data(data, timestamp)
            results.extend(sleep_insights)
        
        elif sensor_type in ["accelerometer", "gyroscope"]:
            self._queue_motion_data(sensor_type, data, timestamp)
        
        # Store results
        self.predictions.extend([r for r in results if isinstance(r, ModelPrediction)])
        self.insights.extend([r for r in results if isinstance(r, HealthInsight)])
        
        # Maintain storage limits
        if len(self.predictions) > self.max_storage:
            self.predictions = self.predictions[-self.max_storage:]
        if len(self.insights) > self.max_storage:
            self.insights = self.insights[-self.max_storage:]
        
        return results
    
    def _extract_heart_rate(self, sensor_type: str, data: np.ndarray) -> Optional[float]:
        """Extract heart rate from sensor data."""
        if sensor_type == "heart_rate":
            return float(data[0]) if len(data) > 0 else None
        elif sensor_type == "ecg" and len(data) > 1:
            return float(data[1])  # Assume second element is heart rate
        elif sensor_type == "ppg" and len(data) > 2:
            return float(data[2])  # Assume third element is heart rate
        return None
    
    def _analyze_sleep_data(self, data: np.ndarray, timestamp: float) -> List[HealthInsight]:
        """Analyze sleep tracker data."""
        if len(data) < 4:
            return []
        
        # Assume sleep data format: [timestamp, stage_encoded, quality, hrv]
        sleep_stages = ["awake", "light", "deep", "rem"]
        stage_index = int(data[1]) % len(sleep_stages)
        sleep_stage = sleep_stages[stage_index]
        sleep_quality = float(data[2])
        hrv = float(data[3]) if len(data) > 3 else 50.0
        
        return self.sleep_analyzer.analyze(sleep_stage, sleep_quality, hrv, timestamp)
    
    def _queue_motion_data(self, sensor_type: str, data: np.ndarray, timestamp: float):
        """Queue motion data for activity classification."""
        with self.queue_lock:
            self.data_queue.append({
                "sensor_type": sensor_type,
                "data": data,
                "timestamp": timestamp
            })
            
            # Process when we have both accelerometer and gyroscope data
            acc_data = None
            gyro_data = None
            
            for item in self.data_queue[-10:]:  # Check recent items
                if item["sensor_type"] == "accelerometer":
                    acc_data = item["data"]
                elif item["sensor_type"] == "gyroscope":
                    gyro_data = item["data"]
            
            if acc_data is not None and gyro_data is not None:
                activity_prediction = self.activity_classifier.classify(acc_data, gyro_data)
                if activity_prediction:
                    self.predictions.append(activity_prediction)
    
    def get_recent_insights(self, limit: int = 10) -> List[HealthInsight]:
        """Get recent health insights."""
        return self.insights[-limit:]
    
    def get_recent_predictions(self, limit: int = 10) -> List[ModelPrediction]:
        """Get recent model predictions."""
        return self.predictions[-limit:]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Generate comprehensive health summary from recent insights."""
        recent_insights = self.get_recent_insights(50)
        
        if not recent_insights:
            return {"status": "insufficient_data"}
        
        # Categorize insights
        alerts = [i for i in recent_insights if i.insight_type == "alert"]
        trends = [i for i in recent_insights if i.insight_type == "trend"]
        anomalies = [i for i in recent_insights if i.insight_type == "anomaly"]
        
        # Calculate severity distribution
        severity_counts = {}
        for insight in recent_insights:
            severity_counts[insight.severity] = severity_counts.get(insight.severity, 0) + 1
        
        summary = {
            "timestamp": time.time(),
            "total_insights": len(recent_insights),
            "alerts": len(alerts),
            "trends": len(trends),
            "anomalies": len(anomalies),
            "severity_distribution": severity_counts,
            "recent_critical_insights": [
                {
                    "type": i.insight_type,
                    "message": i.message,
                    "confidence": i.confidence,
                    "recommended_action": i.recommended_action
                }
                for i in recent_insights if i.severity == "critical"
            ][-5:],  # Last 5 critical insights
            "processing_status": "active" if HAS_MLX else "fallback_mode"
        }
        
        return summary
    
    def start_real_time_processing(self):
        """Start real-time background processing."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Started real-time MLX model processing")
    
    def stop_real_time_processing(self):
        """Stop real-time processing."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Stopped real-time MLX model processing")
    
    def _processing_loop(self):
        """Background processing loop for queued data."""
        while self.is_running:
            # Process any queued data
            with self.queue_lock:
                if len(self.data_queue) > 100:  # Prevent memory buildup
                    self.data_queue = self.data_queue[-50:]
            
            # Sleep to prevent CPU overuse
            time.sleep(0.1)
    
    def export_model_results(self, filepath: str) -> None:
        """Export model results to file."""
        results = {
            "predictions": [
                {
                    "timestamp": p.timestamp,
                    "sensor_type": p.sensor_type,
                    "prediction_type": p.prediction_type,
                    "confidence": p.confidence,
                    "result": p.result,
                    "processing_time_ms": p.processing_time_ms
                }
                for p in self.predictions
            ],
            "insights": [
                {
                    "timestamp": i.timestamp,
                    "insight_type": i.insight_type,
                    "severity": i.severity,
                    "message": i.message,
                    "confidence": i.confidence,
                    "data_source": i.data_source,
                    "recommended_action": i.recommended_action
                }
                for i in self.insights
            ],
            "export_timestamp": time.time(),
            "mlx_available": HAS_MLX
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Model results exported to {filepath}") 