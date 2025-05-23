import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import pywt  # PyWavelets for wavelet transformations
from scipy import signal
from scipy.stats import skew, kurtosis

logger = logging.getLogger("OpenWearables.DataProcessor")

class DataProcessor:
    """
    Processes raw sensor data for health analysis.
    
    This class handles signal processing, feature extraction, and data 
    preparation for machine learning models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or {}
        self.window_size = self.config.get("window_size", 10)  # seconds
        self.overlap = self.config.get("overlap", 0.5)  # 50% overlap
        self.features = self.config.get("features", ["time_domain", "frequency_domain"])
        
        # Initialize preprocessing parameters
        self._initialize_filters()
    
    def _initialize_filters(self) -> None:
        """Initialize signal processing filters."""
        # ECG filters
        self.ecg_bandpass = signal.butter(
            3,  # Order
            [0.5, 40],  # Cutoff frequencies (Hz)
            btype='bandpass',
            output='sos',
            fs=250  # Typical ECG sampling rate
        )
        
        # PPG filters
        self.ppg_lowpass = signal.butter(
            3,  # Order
            5,  # Cutoff frequency (Hz)
            btype='lowpass',
            output='sos',
            fs=100  # Typical PPG sampling rate
        )
        
        # Motion filters (for accelerometer and gyroscope)
        self.motion_lowpass = signal.butter(
            2,  # Order
            10,  # Cutoff frequency (Hz)
            btype='lowpass',
            output='sos',
            fs=50  # Typical motion sampling rate
        )
    
    def preprocess_ecg(self, ecg_data: List[Tuple[float, np.ndarray]]) -> np.ndarray:
        """
        Preprocess ECG data.
        
        Args:
            ecg_data: List of (timestamp, reading) tuples
            
        Returns:
            Preprocessed ECG signal
        """
        if not ecg_data:
            return np.array([])
        
        # Extract timestamps and readings
        timestamps, readings = zip(*ecg_data)
        
        # Convert to continuous signal
        signal_array = np.array([r[0] for r in readings])
        
        # Apply bandpass filter
        filtered_signal = signal.sosfilt(self.ecg_bandpass, signal_array)
        
        # Normalize
        normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
        
        return normalized_signal
    
    def preprocess_ppg(self, ppg_data: List[Tuple[float, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess PPG data.
        
        Args:
            ppg_data: List of (timestamp, reading) tuples
            
        Returns:
            Tuple of (preprocessed PPG signal, SpO2 values)
        """
        if not ppg_data:
            return np.array([]), np.array([])
        
        # Extract timestamps and readings
        timestamps, readings = zip(*ppg_data)
        
        # Convert to continuous signals
        ppg_signal = np.array([r[0] for r in readings])
        spo2_values = np.array([r[1] for r in readings])
        
        # Apply lowpass filter to PPG
        filtered_ppg = signal.sosfilt(self.ppg_lowpass, ppg_signal)
        
        # Normalize
        normalized_ppg = (filtered_ppg - np.mean(filtered_ppg)) / np.std(filtered_ppg)
        
        return normalized_ppg, spo2_values
    
    def preprocess_motion(self, motion_data: List[Tuple[float, np.ndarray]]) -> np.ndarray:
        """
        Preprocess motion data (accelerometer or gyroscope).
        
        Args:
            motion_data: List of (timestamp, reading) tuples
            
        Returns:
            Preprocessed motion signal
        """
        if not motion_data:
            return np.array([])
        
        # Extract timestamps and readings
        timestamps, readings = zip(*motion_data)
        
        # Convert to continuous signal
        signal_array = np.array(readings)
        
        # Apply lowpass filter to each axis
        filtered_signal = np.zeros_like(signal_array)
        for i in range(signal_array.shape[1]):
            filtered_signal[:, i] = signal.sosfilt(self.motion_lowpass, signal_array[:, i])
        
        return filtered_signal
    
    def extract_heart_rate(self, ecg_signal: np.ndarray, sampling_rate: float = 250) -> float:
        """
        Extract heart rate from ECG signal.
        
        Args:
            ecg_signal: Preprocessed ECG signal
            sampling_rate: Sampling rate of the ECG signal in Hz
            
        Returns:
            Heart rate in beats per minute
        """
        if len(ecg_signal) < sampling_rate * 2:  # Need at least 2 seconds
            return 0.0
        
        try:
            # Find R-peaks (QRS complex)
            # Using simple peak detection for demonstration
            # In a real system, would use more robust algorithms like Pan-Tompkins
            
            # First, find local maxima that exceed a threshold
            peak_threshold = np.mean(ecg_signal) + 1.5 * np.std(ecg_signal)
            potential_peaks = []
            
            for i in range(1, len(ecg_signal) - 1):
                if (ecg_signal[i] > ecg_signal[i-1] and 
                    ecg_signal[i] > ecg_signal[i+1] and 
                    ecg_signal[i] > peak_threshold):
                    potential_peaks.append(i)
            
            # Calculate RR intervals and heart rate
            if len(potential_peaks) >= 2:
                rr_intervals = np.diff(potential_peaks) / sampling_rate  # in seconds
                heart_rate = 60 / np.mean(rr_intervals)  # in bpm
                return heart_rate
            else:
                return 0.0
        
        except Exception as e:
            logger.error(f"Error extracting heart rate: {str(e)}")
            return 0.0
    
    def extract_hrv(self, ecg_signal: np.ndarray, sampling_rate: float = 250) -> Dict[str, float]:
        """
        Extract heart rate variability metrics from ECG signal.
        
        Args:
            ecg_signal: Preprocessed ECG signal
            sampling_rate: Sampling rate of the ECG signal in Hz
            
        Returns:
            Dictionary of HRV metrics
        """
        if len(ecg_signal) < sampling_rate * 10:  # Need at least 10 seconds
            return {"SDNN": 0.0, "RMSSD": 0.0, "pNN50": 0.0}
        
        try:
            # Find R-peaks as in extract_heart_rate
            peak_threshold = np.mean(ecg_signal) + 1.5 * np.std(ecg_signal)
            potential_peaks = []
            
            for i in range(1, len(ecg_signal) - 1):
                if (ecg_signal[i] > ecg_signal[i-1] and 
                    ecg_signal[i] > ecg_signal[i+1] and 
                    ecg_signal[i] > peak_threshold):
                    potential_peaks.append(i)
            
            # Calculate RR intervals
            if len(potential_peaks) >= 3:
                rr_intervals = np.diff(potential_peaks) / sampling_rate * 1000  # in ms
                
                # Calculate HRV metrics
                sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals
                
                # RMSSD: Root Mean Square of Successive Differences
                rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
                
                # pNN50: Percentage of successive RR intervals that differ by more than 50 ms
                nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
                pnn50 = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0
                
                return {
                    "SDNN": sdnn,
                    "RMSSD": rmssd,
                    "pNN50": pnn50
                }
            else:
                return {"SDNN": 0.0, "RMSSD": 0.0, "pNN50": 0.0}
        
        except Exception as e:
            logger.error(f"Error extracting HRV: {str(e)}")
            return {"SDNN": 0.0, "RMSSD": 0.0, "pNN50": 0.0}
    
    def extract_spo2(self, ppg_signal: np.ndarray, spo2_values: np.ndarray) -> float:
        """
        Extract SpO2 from PPG signal and direct SpO2 readings.
        
        Args:
            ppg_signal: Preprocessed PPG signal
            spo2_values: SpO2 values from sensor
            
        Returns:
            Average SpO2 value
        """
        # In a real system, this would implement SpO2 calculation from
        # red and infrared PPG signals. For simplicity, we'll use the
        # direct SpO2 readings from our simulated sensor.
        
        if len(spo2_values) == 0:
            return 0.0
        
        # Filter outliers (values outside physiological range)
        valid_spo2 = spo2_values[(spo2_values >= 70) & (spo2_values <= 100)]
        
        if len(valid_spo2) == 0:
            return 0.0
        
        return float(np.mean(valid_spo2))
    
    def extract_activity(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> str:
        """
        Extract activity type from accelerometer and gyroscope data.
        
        Args:
            accel_data: Preprocessed accelerometer data
            gyro_data: Preprocessed gyroscope data
            
        Returns:
            Activity classification
        """
        if len(accel_data) == 0 or len(gyro_data) == 0:
            return "unknown"
        
        # Calculate activity intensity from accelerometer
        # Use magnitude of acceleration
        accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
        
        # Remove gravity component (9.8 m/s²)
        accel_magnitude = np.abs(accel_magnitude - 9.8)
        
        # Calculate rotation intensity from gyroscope
        gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
        
        # Simple threshold-based classification
        mean_accel = np.mean(accel_magnitude)
        mean_gyro = np.mean(gyro_magnitude)
        
        if mean_accel < 0.2 and mean_gyro < 0.2:
            return "resting"
        elif mean_accel < 1.0 and mean_gyro < 0.5:
            return "walking"
        elif mean_accel >= 1.0 or mean_gyro >= 0.5:
            return "running"
        else:
            return "unknown"
    
    def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract time domain features from a signal.
        
        Args:
            signal: Input signal
            
        Returns:
            Dictionary of time domain features
        """
        if len(signal.shape) == 1:
            # Single channel signal
            mean = np.mean(signal)
            std = np.std(signal)
            minimum = np.min(signal)
            maximum = np.max(signal)
            median = np.median(signal)
            skewness = skew(signal)
            kurt = kurtosis(signal)
            
            return {
                "mean": mean,
                "std": std,
                "min": minimum,
                "max": maximum,
                "median": median,
                "skewness": skewness,
                "kurtosis": kurt
            }
        else:
            # Multi-channel signal (e.g., accelerometer)
            features = {}
            
            for i in range(signal.shape[1]):
                channel_features = self.extract_time_domain_features(signal[:, i])
                for key, value in channel_features.items():
                    features[f"channel_{i}_{key}"] = value
            
            return features
    
    def extract_frequency_domain_features(self, signal: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """
        Extract frequency domain features from a signal.
        
        Args:
            signal: Input signal
            sampling_rate: Sampling rate of the signal in Hz
            
        Returns:
            Dictionary of frequency domain features
        """
        if len(signal) < 10:
            return {"dominant_frequency": 0.0, "power_0_1Hz": 0.0, "power_1_5Hz": 0.0, "power_5_15Hz": 0.0}
        
        try:
            if len(signal.shape) == 1:
                # Compute power spectral density
                f, psd = signal.welch(signal, fs=sampling_rate, nperseg=min(256, len(signal)))
                
                # Find dominant frequency
                dominant_idx = np.argmax(psd)
                dominant_freq = f[dominant_idx]
                
                # Calculate power in different frequency bands
                # Define bands for physiological signals
                bands = {
                    "power_0_1Hz": (0, 1),    # Very low frequency
                    "power_1_5Hz": (1, 5),    # Low to medium (includes heart rate)
                    "power_5_15Hz": (5, 15)   # Higher frequencies
                }
                
                features = {"dominant_frequency": dominant_freq}
                
                # Calculate power in each band
                for band_name, (low, high) in bands.items():
                    band_power = np.sum(psd[(f >= low) & (f < high)])
                    features[band_name] = band_power
                
                return features
            else:
                # Multi-channel signal
                features = {}
                
                for i in range(signal.shape[1]):
                    channel_features = self.extract_frequency_domain_features(signal[:, i], sampling_rate)
                    for key, value in channel_features.items():
                        features[f"channel_{i}_{key}"] = value
                
                return features
        
        except Exception as e:
            logger.error(f"Error extracting frequency domain features: {str(e)}")
            return {"dominant_frequency": 0.0, "power_0_1Hz": 0.0, "power_1_5Hz": 0.0, "power_5_15Hz": 0.0}
    
    def extract_wavelet_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet transform features from a signal.
        
        Args:
            signal: Input signal
            
        Returns:
            Dictionary of wavelet features
        """
        if len(signal) < 16:
            return {"wavelet_energy": 0.0, "wavelet_entropy": 0.0}
        
        try:
            if len(signal.shape) == 1:
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(signal, 'db4', level=3)
                
                # Calculate energy in each sub-band
                energies = [np.sum(np.square(c)) for c in coeffs]
                total_energy = np.sum(energies)
                
                # Normalize energies
                if total_energy > 0:
                    normalized_energies = [e / total_energy for e in energies]
                else:
                    normalized_energies = [0.0] * len(energies)
                
                # Calculate wavelet entropy
                entropy = -np.sum([e * np.log2(e) if e > 0 else 0 for e in normalized_energies])
                
                return {
                    "wavelet_energy": total_energy,
                    "wavelet_entropy": entropy
                }
            else:
                # Multi-channel signal
                features = {}
                
                for i in range(signal.shape[1]):
                    channel_features = self.extract_wavelet_features(signal[:, i])
                    for key, value in channel_features.items():
                        features[f"channel_{i}_{key}"] = value
                
                return features
        
        except Exception as e:
            logger.error(f"Error extracting wavelet features: {str(e)}")
            return {"wavelet_energy": 0.0, "wavelet_entropy": 0.0}
    
    def process_batch(self, sensor_data: Dict[str, List[Tuple[float, np.ndarray]]]) -> Dict[str, Any]:
        """
        Process a batch of sensor data from multiple sensors.
        
        Args:
            sensor_data: Dictionary mapping sensor names to lists of (timestamp, reading) tuples
            
        Returns:
            Dictionary of processed health metrics and features
        """
        processed_data = {}
        
        # Process ECG data if available
        if "ecg" in sensor_data and sensor_data["ecg"]:
            ecg_signal = self.preprocess_ecg(sensor_data["ecg"])
            
            if len(ecg_signal) > 0:
                # Extract heart rate and HRV
                heart_rate = self.extract_heart_rate(ecg_signal, sampling_rate=250)
                hrv_metrics = self.extract_hrv(ecg_signal, sampling_rate=250)
                
                # Extract signal features
                time_features = self.extract_time_domain_features(ecg_signal)
                freq_features = self.extract_frequency_domain_features(ecg_signal, sampling_rate=250)
                wavelet_features = self.extract_wavelet_features(ecg_signal)
                
                # Store processed ECG data
                processed_data["ecg"] = {
                    "heart_rate": heart_rate,
                    "hrv": hrv_metrics,
                    "features": {
                        "time_domain": time_features,
                        "frequency_domain": freq_features,
                        "wavelet": wavelet_features
                    }
                }
        
        # Process PPG data if available
        if "ppg" in sensor_data and sensor_data["ppg"]:
            ppg_signal, spo2_values = self.preprocess_ppg(sensor_data["ppg"])
            
            if len(ppg_signal) > 0:
                # Extract SpO2
                spo2 = self.extract_spo2(ppg_signal, spo2_values)
                
                # Extract signal features
                time_features = self.extract_time_domain_features(ppg_signal)
                freq_features = self.extract_frequency_domain_features(ppg_signal, sampling_rate=100)
                
                # Store processed PPG data
                processed_data["ppg"] = {
                    "spo2": spo2,
                    "features": {
                        "time_domain": time_features,
                        "frequency_domain": freq_features
                    }
                }
        
        # Process accelerometer data if available
        if "accelerometer" in sensor_data and sensor_data["accelerometer"]:
            accel_data = self.preprocess_motion(sensor_data["accelerometer"])
            
            if len(accel_data) > 0:
                # Extract features
                time_features = self.extract_time_domain_features(accel_data)
                freq_features = self.extract_frequency_domain_features(accel_data, sampling_rate=50)
                
                # Store processed accelerometer data
                processed_data["accelerometer"] = {
                    "features": {
                        "time_domain": time_features,
                        "frequency_domain": freq_features
                    }
                }
        
        # Process gyroscope data if available
        if "gyroscope" in sensor_data and sensor_data["gyroscope"]:
            gyro_data = self.preprocess_motion(sensor_data["gyroscope"])
            
            if len(gyro_data) > 0:
                # Extract features
                time_features = self.extract_time_domain_features(gyro_data)
                freq_features = self.extract_frequency_domain_features(gyro_data, sampling_rate=50)
                
                # Store processed gyroscope data
                processed_data["gyroscope"] = {
                    "features": {
                        "time_domain": time_features,
                        "frequency_domain": freq_features
                    }
                }
        
        # Combine accelerometer and gyroscope for activity recognition
        if "accelerometer" in processed_data and "gyroscope" in processed_data:
            accel_data = self.preprocess_motion(sensor_data["accelerometer"])
            gyro_data = self.preprocess_motion(sensor_data["gyroscope"])
            
            if len(accel_data) > 0 and len(gyro_data) > 0:
                activity = self.extract_activity(accel_data, gyro_data)
                processed_data["activity"] = activity
        
        # Process temperature data if available
        if "temperature" in sensor_data and sensor_data["temperature"]:
            temp_data = [reading for _, reading in sensor_data["temperature"]]
            
            if temp_data:
                # Calculate average temperature
                temp_values = np.array([t[0] for t in temp_data])
                avg_temp = float(np.mean(temp_values))
                
                # Store processed temperature data
                processed_data["temperature"] = avg_temp
        
        return processed_data


class ProcessedData:
    """
    Container class for processed health data from multiple sensors.
    
    This class provides a structured way to store and access processed
    sensor data, health metrics, and extracted features.
    """
    
    def __init__(self, data: Dict[str, Any] = None):
        """
        Initialize ProcessedData container.
        
        Args:
            data: Dictionary of processed sensor data and metrics
        """
        self.data = data or {}
        self.timestamp = pd.Timestamp.now()
    
    def get_heart_rate(self) -> Optional[float]:
        """Get heart rate from ECG data."""
        return self.data.get("ecg", {}).get("heart_rate")
    
    def get_spo2(self) -> Optional[float]:
        """Get SpO2 from PPG data."""
        return self.data.get("ppg", {}).get("spo2")
    
    def get_temperature(self) -> Optional[float]:
        """Get temperature reading."""
        return self.data.get("temperature")
    
    def get_activity(self) -> Optional[str]:
        """Get detected activity."""
        return self.data.get("activity")
    
    def get_hrv_metrics(self) -> Dict[str, float]:
        """Get heart rate variability metrics."""
        return self.data.get("ecg", {}).get("hrv", {})
    
    def get_sensor_features(self, sensor_name: str) -> Dict[str, Any]:
        """
        Get extracted features for a specific sensor.
        
        Args:
            sensor_name: Name of the sensor (e.g., 'ecg', 'ppg', 'accelerometer')
            
        Returns:
            Dictionary of features for the specified sensor
        """
        return self.data.get(sensor_name, {}).get("features", {})
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all processed data and metrics."""
        return self.data.copy()
    
    def has_sensor_data(self, sensor_name: str) -> bool:
        """Check if data exists for a specific sensor."""
        return sensor_name in self.data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert processed data to pandas DataFrame.
        
        Returns:
            DataFrame with flattened metrics as columns
        """
        flat_data = {}
        
        # Flatten nested dictionaries
        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> None:
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key
                
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                elif isinstance(value, (int, float, str, bool)):
                    flat_data[new_key] = value
        
        flatten_dict(self.data)
        flat_data["timestamp"] = self.timestamp
        
        return pd.DataFrame([flat_data])
    
    def export_json(self, filepath: str) -> None:
        """
        Export processed data to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def __repr__(self) -> str:
        """String representation of ProcessedData."""
        sensors = list(self.data.keys())
        return f"ProcessedData(sensors={sensors}, timestamp={self.timestamp})"
    
    def __len__(self) -> int:
        """Return number of sensors with data."""
        return len(self.data)