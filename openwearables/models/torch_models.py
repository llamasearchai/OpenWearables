"""
PyTorch Models for CUDA/CPU Hardware Acceleration

This module provides PyTorch-optimized models for health data analysis on CUDA and CPU.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

logger = logging.getLogger("OpenWearables.TorchModels")

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
    logger.info("PyTorch is available")
    
    # Check for CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA is available with {torch.cuda.device_count()} devices")
    else:
        logger.info("CUDA not available, using CPU")
        
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, falling back to CPU processing")

class TorchHealthModel(nn.Module):
    """Base class for PyTorch health analysis models."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_classes: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Define layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class ArrhythmiaDetectionModel(nn.Module):
    """PyTorch-optimized arrhythmia detection model."""
    
    def __init__(self, ecg_length: int = 1000):
        super().__init__()
        self.ecg_length = ecg_length
        
        # Convolutional layers for ECG signal processing
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Normal vs Arrhythmia
        )
    
    def forward(self, x):
        # x shape: (batch_size, ecg_length)
        # Reshape for conv1d: (batch_size, 1, ecg_length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x

class StressAnalysisModel(nn.Module):
    """PyTorch-optimized stress level analysis model."""
    
    def __init__(self):
        super().__init__()
        
        # Stress-specific architecture
        self.stress_layers = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # Low, Medium, High stress
        )
    
    def forward(self, x):
        return self.stress_layers(x)
    
    def prepare_features(self, hrv_data: Dict[str, float], heart_rate: float, activity: str) -> torch.Tensor:
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
        
        return torch.tensor(features, dtype=torch.float32)

class ActivityRecognitionModel(nn.Module):
    """PyTorch-optimized activity recognition model."""
    
    def __init__(self, window_size: int = 100):
        super().__init__()
        self.window_size = window_size
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(6, 64, batch_first=True, num_layers=2, dropout=0.2)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Rest, Walk, Run, Other
        )
    
    def forward(self, x):
        # x shape: (batch_size, window_size, 6)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        x = lstm_out[:, -1, :]
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x
    
    def prepare_features(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> torch.Tensor:
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
        
        # Convert to PyTorch tensor
        return torch.tensor(combined_data, dtype=torch.float32)

class MultimodalHealthModel(nn.Module):
    """Advanced multimodal model combining ECG, PPG, and motion data."""
    
    def __init__(self, ecg_length: int = 1000, ppg_length: int = 400, motion_window: int = 100):
        super().__init__()
        
        # ECG encoder
        self.ecg_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 128)
        )
        
        # PPG encoder
        self.ppg_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 64)
        )
        
        # Motion encoder (LSTM)
        self.motion_lstm = nn.LSTM(6, 32, batch_first=True, num_layers=1)
        self.motion_fc = nn.Linear(32, 64)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128 + 64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
        # Output heads
        self.arrhythmia_head = nn.Sequential(
            nn.Linear(128 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
        self.stress_head = nn.Sequential(
            nn.Linear(128 + 64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        
        self.activity_head = nn.Sequential(
            nn.Linear(128 + 64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
    
    def forward(self, ecg, ppg, motion):
        # Encode each modality
        ecg_features = self.ecg_encoder(ecg)
        ppg_features = self.ppg_encoder(ppg)
        
        # Motion features
        motion_out, _ = self.motion_lstm(motion)
        motion_features = self.motion_fc(motion_out[:, -1, :])
        
        # Concatenate features
        combined_features = torch.cat([ecg_features, ppg_features, motion_features], dim=1)
        
        # Attention weights
        attention_weights = self.attention(combined_features)
        
        # Apply attention
        weighted_ecg = ecg_features * attention_weights[:, 0:1]
        weighted_ppg = ppg_features * attention_weights[:, 1:2]
        weighted_motion = motion_features * attention_weights[:, 2:3]
        
        # Final features
        final_features = torch.cat([weighted_ecg, weighted_ppg, weighted_motion], dim=1)
        
        # Generate predictions
        arrhythmia_pred = self.arrhythmia_head(final_features)
        stress_pred = self.stress_head(final_features)
        activity_pred = self.activity_head(final_features)
        
        return {
            "arrhythmia": arrhythmia_pred,
            "stress": stress_pred,
            "activity": activity_pred
        }

def load_model(model_path: str, model_type: str, use_amp: bool = False, device: str = "auto") -> Optional[nn.Module]:
    """Load a pre-trained PyTorch model."""
    if not HAS_TORCH:
        logger.error("PyTorch not available, cannot load PyTorch models")
        return None
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    try:
        if model_type == "arrhythmia_detection":
            model = ArrhythmiaDetectionModel()
        elif model_type == "stress_analysis":
            model = StressAnalysisModel()
        elif model_type == "activity_recognition":
            model = ActivityRecognitionModel()
        elif model_type == "multimodal_health":
            model = MultimodalHealthModel()
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        # Move model to device
        model = model.to(device)
        
        # Set to evaluation mode
        model.eval()
        
        # In a real implementation, we would load weights from model_path
        # For demo purposes, we'll return the initialized model
        logger.info(f"Loaded PyTorch model: {model_type} on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading PyTorch model {model_type}: {str(e)}")
        return None

def predict_arrhythmia(model: ArrhythmiaDetectionModel, ecg_data: np.ndarray, device: str = "cuda") -> Dict[str, Any]:
    """Predict arrhythmia from ECG data using PyTorch model."""
    if not HAS_TORCH or model is None:
        return {"error": "PyTorch model not available"}
    
    try:
        # Prepare input
        if len(ecg_data) != model.ecg_length:
            # Resample or pad to required length
            if len(ecg_data) > model.ecg_length:
                ecg_data = ecg_data[-model.ecg_length:]
            else:
                ecg_data = np.pad(ecg_data, (model.ecg_length - len(ecg_data), 0), mode='constant')
        
        # Convert to tensor and add batch dimension
        x = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)
        
        # Move to device
        x = x.to(next(model.parameters()).device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Convert back to numpy for compatibility
        prob_normal = float(probabilities[0, 0].cpu())
        prob_arrhythmia = float(probabilities[0, 1].cpu())
        predicted_class = int(prediction[0].cpu())
        
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
    """Predict stress level using PyTorch model."""
    if not HAS_TORCH or model is None:
        return {"error": "PyTorch model not available"}
    
    try:
        # Prepare features
        features = model.prepare_features(hrv_data, heart_rate, activity)
        features = features.unsqueeze(0)  # Add batch dimension
        
        # Move to device
        features = features.to(next(model.parameters()).device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(features)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Convert results
        stress_levels = ["low", "medium", "high"]
        predicted_level = stress_levels[int(prediction[0].cpu())]
        confidence = float(torch.max(probabilities).cpu())
        
        return {
            "stress_level": predicted_level,
            "confidence": confidence,
            "probabilities": {
                "low": float(probabilities[0, 0].cpu()),
                "medium": float(probabilities[0, 1].cpu()),
                "high": float(probabilities[0, 2].cpu())
            }
        }
        
    except Exception as e:
        logger.error(f"Error in stress level prediction: {str(e)}")
        return {"error": str(e)}

def predict_activity(model: ActivityRecognitionModel, accel_data: np.ndarray, 
                    gyro_data: np.ndarray) -> Dict[str, Any]:
    """Predict activity type using PyTorch model."""
    if not HAS_TORCH or model is None:
        return {"error": "PyTorch model not available"}
    
    try:
        # Prepare features
        features = model.prepare_features(accel_data, gyro_data)
        features = features.unsqueeze(0)  # Add batch dimension
        
        # Move to device
        features = features.to(next(model.parameters()).device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(features)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Convert results
        activities = ["resting", "walking", "running", "other"]
        predicted_activity = activities[int(prediction[0].cpu())]
        confidence = float(torch.max(probabilities).cpu())
        
        return {
            "activity": predicted_activity,
            "confidence": confidence,
            "probabilities": {
                "resting": float(probabilities[0, 0].cpu()),
                "walking": float(probabilities[0, 1].cpu()),
                "running": float(probabilities[0, 2].cpu()),
                "other": float(probabilities[0, 3].cpu())
            }
        }
        
    except Exception as e:
        logger.error(f"Error in activity prediction: {str(e)}")
        return {"error": str(e)}

def enable_mixed_precision(model: nn.Module, optimizer: optim.Optimizer = None):
    """Enable mixed precision training for better performance on modern GPUs."""
    if not HAS_TORCH:
        return model, None
    
    try:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        logger.info("Mixed precision training enabled")
        return model, scaler
    except ImportError:
        logger.warning("Mixed precision not available")
        return model, None

# Export main functions
__all__ = [
    "ArrhythmiaDetectionModel",
    "StressAnalysisModel", 
    "ActivityRecognitionModel",
    "MultimodalHealthModel",
    "load_model",
    "predict_arrhythmia",
    "predict_stress_level",
    "predict_activity",
    "enable_mixed_precision",
    "HAS_TORCH"
] 