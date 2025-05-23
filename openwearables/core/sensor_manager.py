import time
import threading
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger("OpenWearables.Sensors")

class SensorInterface(ABC):
    """Abstract base class for all sensor interfaces."""
    
    def __init__(self, sensor_id: int, name: str, sampling_rate: float = 50):
        """
        Initialize a sensor interface.
        
        Args:
            sensor_id: Unique identifier for the sensor
            name: Name of the sensor
            sampling_rate: Sampling rate in Hz
        """
        self.sensor_id = sensor_id
        self.name = name
        self.sampling_rate = sampling_rate
        self.is_running = False
        self.thread = None
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.last_reading = None
    
    @abstractmethod
    def read(self) -> np.ndarray:
        """
        Read sensor data.
        
        Returns:
            Numpy array containing sensor readings
        """
        pass
    
    def start(self) -> None:
        """Start sensor data collection in a separate thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._collection_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started sensor: {self.name}")
    
    def stop(self) -> None:
        """Stop sensor data collection."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info(f"Stopped sensor: {self.name}")
    
    def _collection_loop(self) -> None:
        """Main sensor data collection loop."""
        interval = 1.0 / self.sampling_rate
        last_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            elapsed = current_time - last_time
            
            if elapsed >= interval:
                try:
                    reading = self.read()
                    self.last_reading = reading
                    
                    with self.buffer_lock:
                        self.buffer.append((current_time, reading))
                    
                    last_time = current_time
                except Exception as e:
                    logger.error(f"Error reading from sensor {self.name}: {str(e)}")
            
            # Sleep for a short time to avoid CPU hogging
            time.sleep(max(0.001, interval / 10))
    
    def get_buffer(self, clear: bool = True) -> List[Tuple[float, np.ndarray]]:
        """
        Get all readings from the buffer and optionally clear it.
        
        Args:
            clear: Whether to clear the buffer after reading
            
        Returns:
            List of tuples containing (timestamp, reading)
        """
        with self.buffer_lock:
            data = list(self.buffer)
            if clear:
                self.buffer.clear()
        return data


class ECGSensor(SensorInterface):
    """ECG (Electrocardiogram) sensor implementation."""
    
    def __init__(self, sensor_id: int, sampling_rate: float = 250):
        """
        Initialize ECG sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            sampling_rate: Sampling rate in Hz (typical ECG: 250-500 Hz)
        """
        super().__init__(sensor_id, "ecg", sampling_rate)
        
        # Normal ECG parameters (for simulation)
        self.heart_rate = 70  # bpm
        self.hrv = 50  # ms
        self.p_duration = 0.08  # seconds
        self.pr_interval = 0.16  # seconds
        self.qrs_duration = 0.08  # seconds
        self.qt_interval = 0.36  # seconds
        self.st_segment = 0.12  # seconds
        
        # Create synthetic ECG template
        self._create_ecg_template()
    
    def _create_ecg_template(self) -> None:
        """Create a synthetic ECG waveform template."""
        # Time points for one cardiac cycle at 60 bpm (1 second)
        t = np.linspace(0, 1, int(self.sampling_rate))
        
        # Initialize ECG with baseline
        ecg = np.zeros_like(t)
        
        # P wave (atrial depolarization)
        p_center = self.pr_interval - self.p_duration/2
        p_wave = 0.15 * np.exp(-((t - p_center) ** 2) / (2 * (self.p_duration/5) ** 2))
        
        # QRS complex (ventricular depolarization)
        qrs_start = self.pr_interval
        q_wave = -0.1 * np.exp(-((t - (qrs_start + 0.02)) ** 2) / (2 * 0.01 ** 2))
        r_wave = 1.0 * np.exp(-((t - (qrs_start + 0.04)) ** 2) / (2 * 0.01 ** 2))
        s_wave = -0.3 * np.exp(-((t - (qrs_start + 0.06)) ** 2) / (2 * 0.01 ** 2))
        
        # T wave (ventricular repolarization)
        t_center = self.pr_interval + self.qrs_duration + self.st_segment + 0.1
        t_wave = 0.3 * np.exp(-((t - t_center) ** 2) / (2 * 0.07 ** 2))
        
        # Combine components
        ecg = p_wave + q_wave + r_wave + s_wave + t_wave
        
        # Store template
        self.template = ecg
    
    def read(self) -> np.ndarray:
        """
        Read ECG data (simulated).
        
        Returns:
            Numpy array with ECG signal
        """
        # Calculate current phase in the cardiac cycle
        cycle_duration = 60.0 / self.heart_rate  # seconds
        phase = (time.time() % cycle_duration) / cycle_duration
        
        # Get index in template
        idx = int(phase * len(self.template))
        
        # Add noise and variability
        noise_level = 0.03
        noise = np.random.normal(0, noise_level)
        variability = np.random.normal(0, 0.05)
        
        # Return single sample with noise
        return np.array([self.template[idx] + noise + variability])


class PPGSensor(SensorInterface):
    """PPG (Photoplethysmogram) sensor implementation."""
    
    def __init__(self, sensor_id: int, sampling_rate: float = 100):
        """
        Initialize PPG sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            sampling_rate: Sampling rate in Hz (typical PPG: 100 Hz)
        """
        super().__init__(sensor_id, "ppg", sampling_rate)
        
        # Normal PPG parameters
        self.heart_rate = 70  # bpm
        self.amplitude = 1.0
        self.baseline = 0.5
        
        # Create synthetic PPG template
        self._create_ppg_template()
    
    def _create_ppg_template(self) -> None:
        """Create a synthetic PPG waveform template."""
        # Time points for one cardiac cycle at 60 bpm (1 second)
        t = np.linspace(0, 1, int(self.sampling_rate))
        
        # Create PPG waveform (simplified model)
        # Rising edge (systolic upstroke)
        systolic_time = 0.15  # portion of cycle
        diastolic_time = 0.85  # portion of cycle
        
        # Initialize with baseline
        ppg = np.zeros_like(t) + self.baseline
        
        # Create PPG shape
        for i, time in enumerate(t):
            if time < systolic_time:
                # Systolic upstroke (fast rise)
                ppg[i] += self.amplitude * (time/systolic_time) ** 2
            else:
                # Diastolic decay (exponential decay)
                decay_factor = 5.0
                normalized_time = (time - systolic_time) / diastolic_time
                ppg[i] += self.amplitude * np.exp(-decay_factor * normalized_time)
        
        # Store template
        self.template = ppg
    
    def read(self) -> np.ndarray:
        """
        Read PPG data (simulated).
        
        Returns:
            Numpy array with PPG signal and derived SpO2 value
        """
        # Calculate current phase in the cardiac cycle
        cycle_duration = 60.0 / self.heart_rate  # seconds
        phase = (time.time() % cycle_duration) / cycle_duration
        
        # Get index in template
        idx = int(phase * len(self.template))
        
        # Add noise and variability
        noise_level = 0.02
        noise = np.random.normal(0, noise_level)
        
        # Simulate SpO2 (normally 95-100%)
        spo2 = 97 + np.random.normal(0, 0.5)
        spo2 = min(100, max(70, spo2))  # Clamp to reasonable values
        
        # Return PPG signal and SpO2
        return np.array([self.template[idx] + noise, spo2])


class AccelerometerSensor(SensorInterface):
    """Accelerometer sensor implementation."""
    
    def __init__(self, sensor_id: int, sampling_rate: float = 50):
        """
        Initialize accelerometer sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            sampling_rate: Sampling rate in Hz
        """
        super().__init__(sensor_id, "accelerometer", sampling_rate)
        
        # Initial state
        self.current_activity = "resting"  # resting, walking, running
        self.transition_prob = 0.001  # probability of changing activity state
        self.activity_patterns = {
            "resting": {
                "mean": np.array([0, 0, 9.8]),  # gravity on z-axis
                "std": np.array([0.05, 0.05, 0.05])
            },
            "walking": {
                "mean": np.array([0.2, 0.2, 9.8]),
                "std": np.array([0.5, 0.5, 0.3]),
                "frequency": 2.0  # Hz
            },
            "running": {
                "mean": np.array([0.5, 0.5, 9.8]),
                "std": np.array([1.5, 1.5, 0.8]),
                "frequency": 3.0  # Hz
            }
        }
        
        # Time counter for periodic signals
        self.time_counter = 0
    
    def read(self) -> np.ndarray:
        """
        Read accelerometer data (simulated).
        
        Returns:
            Numpy array with accelerometer readings (x, y, z)
        """
        # Randomly transition between activities
        if np.random.random() < self.transition_prob:
            activities = list(self.activity_patterns.keys())
            activities.remove(self.current_activity)
            self.current_activity = np.random.choice(activities)
        
        # Get activity pattern
        pattern = self.activity_patterns[self.current_activity]
        
        # For dynamic activities, add periodic signals
        if self.current_activity in ["walking", "running"]:
            frequency = pattern["frequency"]
            self.time_counter += 1.0 / self.sampling_rate
            
            # Add sinusoidal motion for x and y
            x_motion = pattern["std"][0] * np.sin(2 * np.pi * frequency * self.time_counter)
            y_motion = pattern["std"][1] * np.sin(2 * np.pi * frequency * self.time_counter + np.pi/2)
            
            # Create reading with periodic motion
            reading = pattern["mean"] + np.array([x_motion, y_motion, 0])
            
            # Add random noise
            noise = np.random.normal(0, pattern["std"] * 0.2)
            reading += noise
        else:
            # For resting, just add random noise to baseline
            reading = pattern["mean"] + np.random.normal(0, pattern["std"])
        
        return reading


class GyroscopeSensor(SensorInterface):
    """Gyroscope sensor implementation."""
    
    def __init__(self, sensor_id: int, sampling_rate: float = 50):
        """
        Initialize gyroscope sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            sampling_rate: Sampling rate in Hz
        """
        super().__init__(sensor_id, "gyroscope", sampling_rate)
        
        # Coordinated with accelerometer for activity patterns
        self.current_activity = "resting"
        self.transition_prob = 0.001
        self.activity_patterns = {
            "resting": {
                "mean": np.array([0, 0, 0]),
                "std": np.array([0.02, 0.02, 0.02])
            },
            "walking": {
                "mean": np.array([0, 0, 0]),
                "std": np.array([0.2, 0.3, 0.1]),
                "frequency": 2.0  # Hz
            },
            "running": {
                "mean": np.array([0, 0, 0]),
                "std": np.array([0.5, 0.8, 0.3]),
                "frequency": 3.0  # Hz
            }
        }
        
        # Time counter for periodic signals
        self.time_counter = 0
    
    def read(self) -> np.ndarray:
        """
        Read gyroscope data (simulated).
        
        Returns:
            Numpy array with gyroscope readings (x, y, z rotation rates in rad/s)
        """
        # Randomly transition between activities
        if np.random.random() < self.transition_prob:
            activities = list(self.activity_patterns.keys())
            activities.remove(self.current_activity)
            self.current_activity = np.random.choice(activities)
        
        # Get activity pattern
        pattern = self.activity_patterns[self.current_activity]
        
        # For dynamic activities, add periodic signals
        if self.current_activity in ["walking", "running"]:
            frequency = pattern["frequency"]
            self.time_counter += 1.0 / self.sampling_rate
            
            # Add sinusoidal rotation for x and y
            x_rotation = pattern["std"][0] * np.sin(2 * np.pi * frequency * self.time_counter + np.pi/4)
            y_rotation = pattern["std"][1] * np.sin(2 * np.pi * frequency * self.time_counter + np.pi/2)
            z_rotation = pattern["std"][2] * np.sin(2 * np.pi * frequency * self.time_counter)
            
            # Create reading with periodic motion
            reading = pattern["mean"] + np.array([x_rotation, y_rotation, z_rotation])
            
            # Add random noise
            noise = np.random.normal(0, pattern["std"] * 0.2)
            reading += noise
        else:
            # For resting, just add random noise to baseline
            reading = pattern["mean"] + np.random.normal(0, pattern["std"])
        
        return reading


class TemperatureSensor(SensorInterface):
    """Body temperature sensor implementation."""
    
    def __init__(self, sensor_id: int, sampling_rate: float = 1):
        """
        Initialize temperature sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            sampling_rate: Sampling rate in Hz (typically low for temperature)
        """
        super().__init__(sensor_id, "temperature", sampling_rate)
        
        # Normal body temperature parameters
        self.base_temp = 36.8  # degrees Celsius
        self.daily_variation = 0.5  # degrees Celsius
        self.noise_level = 0.05  # degrees Celsius
    
    def read(self) -> np.ndarray:
        """
        Read temperature data (simulated).
        
        Returns:
            Numpy array with temperature in degrees Celsius
        """
        # Simulate daily temperature variation (lowest at ~4am, highest at ~6pm)
        hour_of_day = (time.time() % 86400) / 3600  # 0-24
        daily_cycle = np.sin(2 * np.pi * (hour_of_day - 4) / 24)
        
        # Calculate temperature with variation and noise
        temperature = self.base_temp + self.daily_variation * daily_cycle * 0.5
        noise = np.random.normal(0, self.noise_level)
        
        return np.array([temperature + noise])


class SensorManager:
    """Manages multiple sensors and their data collection."""
    
    def __init__(self, sensor_types: List[str], sampling_rates: Dict[str, float] = None, config: Dict[str, Any] = None):
        """
        Initialize the sensor manager.
        
        Args:
            sensor_types: List of sensor types to initialize
            sampling_rates: Dictionary mapping sensor types to sampling rates
            config: Configuration dictionary
        """
        self.sensors = {}
        self.config = config or {}
        self.sampling_rates = sampling_rates or {}
        
        # Default sampling rates
        default_rates = {
            "ecg": 250,
            "ppg": 100,
            "accelerometer": 50,
            "gyroscope": 50,
            "temperature": 1
        }
        
        # Initialize sensors
        sensor_id = 1
        for sensor_type in sensor_types:
            rate = self.sampling_rates.get(sensor_type, default_rates.get(sensor_type, 50))
            
            if sensor_type == "ecg":
                self.sensors[sensor_type] = ECGSensor(sensor_id, rate)
            elif sensor_type == "ppg":
                self.sensors[sensor_type] = PPGSensor(sensor_id, rate)
            elif sensor_type == "accelerometer":
                self.sensors[sensor_type] = AccelerometerSensor(sensor_id, rate)
            elif sensor_type == "gyroscope":
                self.sensors[sensor_type] = GyroscopeSensor(sensor_id, rate)
            elif sensor_type == "temperature":
                self.sensors[sensor_type] = TemperatureSensor(sensor_id, rate)
            else:
                logger.warning(f"Unknown sensor type: {sensor_type}")
                continue
            
            sensor_id += 1
        
        logger.info(f"Initialized {len(self.sensors)} sensors: {list(self.sensors.keys())}")
    
    def start(self) -> None:
        """Start all sensors."""
        for sensor in self.sensors.values():
            sensor.start()
    
    def stop(self) -> None:
        """Stop all sensors."""
        for sensor in self.sensors.values():
            sensor.stop()
    
    def is_running(self) -> bool:
        """Check if any sensors are running."""
        return any(sensor.is_running for sensor in self.sensors.values())
    
    def get_latest_readings(self) -> Dict[str, np.ndarray]:
        """
        Get the latest reading from each sensor.
        
        Returns:
            Dictionary mapping sensor names to their latest readings
        """
        readings = {}
        for name, sensor in self.sensors.items():
            if sensor.last_reading is not None:
                readings[name] = sensor.last_reading
        return readings
    
    def get_buffered_data(self, clear: bool = True) -> Dict[str, List[Tuple[float, np.ndarray]]]:
        """
        Get buffered data from all sensors.
        
        Args:
            clear: Whether to clear buffers after reading
            
        Returns:
            Dictionary mapping sensor names to their buffered data
        """
        data = {}
        for name, sensor in self.sensors.items():
            data[name] = sensor.get_buffer(clear)
        return data