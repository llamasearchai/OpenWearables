"""
MLX Models for Apple Silicon Hardware Acceleration

This module provides MLX-optimized models for health data analysis on Apple Silicon.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

logger = logging.getLogger("OpenWearables.MLXModels")

# Try to import MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
    logger.info("MLX is available for Apple Silicon acceleration")
except ImportError:
    HAS_MLX = False
    logger.warning("MLX not available, falling back to CPU processing")

class MLXHealthModel(nn.Module):
    """Base class for MLX health analysis models."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_classes: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Define layers
        self.layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ArrhythmiaDetectionModel(MLXHealthModel):
    """MLX-optimized arrhythmia detection model."""
    
    def __init__(self, ecg_length: int = 1000):
        # Input: ECG signal (1000 samples at 250Hz = 4 seconds)
        # Output: Normal (0) or Arrhythmia (1)
        super().__init__(input_size=ecg_length, hidden_size=256, num_classes=2)
        self.ecg_length = ecg_length
        
        # Add convolutional layers for ECG signal processing
        self.conv_layers = [
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        ]
        
        # Calculate the size after convolutions
        conv_output_size = ecg_length // 4 * 128  # After 2 stride-2 convolutions
        
        # Update fully connected layers
        self.fc_layers = [
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes)
        ]
    
    def __call__(self, x):
        # x shape: (batch_size, ecg_length)
        # Reshape for conv1d: (batch_size, 1, ecg_length)
        x = mx.expand_dims(x, axis=1)
        
        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten for fully connected layers
        x = mx.reshape(x, (x.shape[0], -1))
        
        # Apply fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        return x

class StressAnalysisModel(MLXHealthModel):
    """MLX-optimized stress level analysis model."""
    
    def __init__(self):
        # Input: HRV features, heart rate, activity level
        # Output: Stress level (0=Low, 1=Medium, 2=High)
        super().__init__(input_size=10, hidden_size=64, num_classes=3)
        
        # Stress-specific architecture
        self.stress_layers = [
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.BatchNorm(32),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        ]
    
    def __call__(self, x):
        # Apply stress-specific layers
        for layer in self.stress_layers:
            x = layer(x)
        return x
    
    def prepare_features(self, hrv_data: Dict[str, float], heart_rate: float, activity: str) -> mx.array:
        """Prepare input features for stress analysis."""
        # Convert activity to numeric
        activity_map = {"resting": 0, "walking": 1, "running": 2, "other": 3}
        activity_numeric = activity_map.get(activity.lower(), 3)
        
        # Create feature vector
        features = [
            hrv_data.get("SDNN", 0) / 100,  # Normalized SDNN
            hrv_data.get("RMSSD", 0) / 100,  # Normalized RMSSD
            hrv_data.get("pNN50", 0) / 100,  # Normalized pNN50
            heart_rate / 100,  # Normalized heart rate
            activity_numeric / 3,  # Normalized activity level
            hrv_data.get("triangular_index", 0) / 50,  # Normalized triangular index
            hrv_data.get("LF", 0) / 1000,  # Normalized LF power
            hrv_data.get("HF", 0) / 1000,  # Normalized HF power
            hrv_data.get("LF_HF_ratio", 1),  # LF/HF ratio
            1.0  # Bias term
        ]
        
        return mx.array(features, dtype=mx.float32)

class ActivityRecognitionModel(MLXHealthModel):
    """MLX-optimized activity recognition model."""
    
    def __init__(self, window_size: int = 100):
        # Input: Accelerometer + Gyroscope data (6 channels)
        # Output: Activity type (0=Rest, 1=Walk, 2=Run, 3=Other)
        self.window_size = window_size
        super().__init__(input_size=window_size * 6, hidden_size=128, num_classes=4)
        
        # Activity-specific LSTM-like architecture using MLX
        self.activity_layers = [
            nn.Linear(window_size * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        ]
    
    def __call__(self, x):
        # Flatten the input
        x = mx.reshape(x, (x.shape[0], -1))
        
        # Apply activity-specific layers
        for layer in self.activity_layers:
            x = layer(x)
        return x
    
    def prepare_features(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> mx.array:
        """Prepare input features from accelerometer and gyroscope data."""
        # Combine accelerometer and gyroscope data
        # accel_data shape: (window_size, 3)
        # gyro_data shape: (window_size, 3)
        
        # Ensure we have the right window size
        if len(accel_data) > self.window_size:
            accel_data = accel_data[-self.window_size:]
            gyro_data = gyro_data[-self.window_size:]
        elif len(accel_data) < self.window_size:
            # Pad with zeros if needed
            pad_size = self.window_size - len(accel_data)
            accel_data = np.pad(accel_data, ((pad_size, 0), (0, 0)), mode='constant')
            gyro_data = np.pad(gyro_data, ((pad_size, 0), (0, 0)), mode='constant')
        
        # Combine data: (window_size, 6)
        combined_data = np.concatenate([accel_data, gyro_data], axis=1)
        
        # Convert to MLX array
        return mx.array(combined_data, dtype=mx.float32)

def load_model(model_path: str, model_type: str) -> Optional[MLXHealthModel]:
    """Load a pre-trained MLX model."""
    if not HAS_MLX:
        logger.error("MLX not available, cannot load MLX models")
        return None
    
    try:
        if model_type == "arrhythmia_detection":
            model = ArrhythmiaDetectionModel()
        elif model_type == "stress_analysis":
            model = StressAnalysisModel()
        elif model_type == "activity_recognition":
            model = ActivityRecognitionModel()
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        # In a real implementation, we would load weights from model_path
        # For demo purposes, we'll return the initialized model
        logger.info(f"Loaded MLX model: {model_type}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading MLX model {model_type}: {str(e)}")
        return None

def predict_arrhythmia(model: ArrhythmiaDetectionModel, ecg_data: np.ndarray) -> Dict[str, Any]:
    """Predict arrhythmia from ECG data using MLX model."""
    if not HAS_MLX or model is None:
        return {"error": "MLX model not available"}
    
    try:
        # Prepare input
        if len(ecg_data) != model.ecg_length:
            # Resample or pad to required length
            if len(ecg_data) > model.ecg_length:
                ecg_data = ecg_data[-model.ecg_length:]
            else:
                ecg_data = np.pad(ecg_data, (model.ecg_length - len(ecg_data), 0), mode='constant')
        
        # Convert to MLX array and add batch dimension
        x = mx.array(ecg_data, dtype=mx.float32)
        x = mx.expand_dims(x, axis=0)
        
        # Make prediction
        logits = model(x)
        probabilities = mx.softmax(logits, axis=1)
        prediction = mx.argmax(probabilities, axis=1)
        
        # Convert back to numpy for compatibility
        prob_normal = float(probabilities[0, 0])
        prob_arrhythmia = float(probabilities[0, 1])
        predicted_class = int(prediction[0])
        
        return {
            "prediction": "arrhythmia" if predicted_class == 1 else "normal",
            "confidence": max(prob_normal, prob_arrhythmia),
            "probabilities": {
                "normal": prob_normal,
                "arrhythmia": prob_arrhythmia
            }
        }
        
    except Exception as e:
        logger.error(f"Error in arrhythmia prediction: {str(e)}")
        return {"error": str(e)}

def predict_stress_level(model: StressAnalysisModel, hrv_data: Dict[str, float], 
                        heart_rate: float, activity: str) -> Dict[str, Any]:
    """Predict stress level using MLX model."""
    if not HAS_MLX or model is None:
        return {"error": "MLX model not available"}
    
    try:
        # Prepare features
        features = model.prepare_features(hrv_data, heart_rate, activity)
        features = mx.expand_dims(features, axis=0)  # Add batch dimension
        
        # Make prediction
        logits = model(features)
        probabilities = mx.softmax(logits, axis=1)
        prediction = mx.argmax(probabilities, axis=1)
        
        # Convert results
        stress_levels = ["low", "medium", "high"]
        predicted_level = stress_levels[int(prediction[0])]
        confidence = float(mx.max(probabilities))
        
        return {
            "stress_level": predicted_level,
            "confidence": confidence,
            "probabilities": {
                "low": float(probabilities[0, 0]),
                "medium": float(probabilities[0, 1]),
                "high": float(probabilities[0, 2])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in stress level prediction: {str(e)}")
        return {"error": str(e)}

def predict_activity(model: ActivityRecognitionModel, accel_data: np.ndarray, 
                    gyro_data: np.ndarray) -> Dict[str, Any]:
    """Predict activity type using MLX model."""
    if not HAS_MLX or model is None:
        return {"error": "MLX model not available"}
    
    try:
        # Prepare features
        features = model.prepare_features(accel_data, gyro_data)
        features = mx.expand_dims(features, axis=0)  # Add batch dimension
        
        # Make prediction
        logits = model(features)
        probabilities = mx.softmax(logits, axis=1)
        prediction = mx.argmax(probabilities, axis=1)
        
        # Convert results
        activities = ["resting", "walking", "running", "other"]
        predicted_activity = activities[int(prediction[0])]
        confidence = float(mx.max(probabilities))
        
        return {
            "activity": predicted_activity,
            "confidence": confidence,
            "probabilities": {
                "resting": float(probabilities[0, 0]),
                "walking": float(probabilities[0, 1]),
                "running": float(probabilities[0, 2]),
                "other": float(probabilities[0, 3])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in activity prediction: {str(e)}")
        return {"error": str(e)}

# Export main functions
__all__ = [
    "ArrhythmiaDetectionModel",
    "StressAnalysisModel", 
    "ActivityRecognitionModel",
    "load_model",
    "predict_arrhythmia",
    "predict_stress_level",
    "predict_activity",
    "HAS_MLX"
]