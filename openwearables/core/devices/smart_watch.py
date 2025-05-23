"""
Smart Watch Device Implementation

Apple Watch inspired smart watch with comprehensive health monitoring,
activity tracking, ECG analysis, blood oxygen monitoring, and advanced biometrics.
"""

import time
import logging
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..sensor_manager import SensorInterface

logger = logging.getLogger("OpenWearables.SmartWatch")

@dataclass
class ECGReading:
    """Data structure for ECG measurements."""
    timestamp: float
    voltage_samples: List[float]  # mV
    heart_rate: float  # bpm
    rhythm_classification: str  # normal, afib, inconclusive
    signal_quality: float  # 0-1 scale

@dataclass
class BloodOxygenReading:
    """Data structure for blood oxygen measurements."""
    timestamp: float
    spo2_percentage: float  # %
    perfusion_index: float  # 0-20 scale
    measurement_confidence: float  # 0-1 scale
    pulse_amplitude: float  # mV

@dataclass
class ActivityMetrics:
    """Data structure for activity and fitness metrics."""
    timestamp: float
    steps: int
    distance_meters: float
    calories_burned: float
    active_energy: float  # kcal
    exercise_minutes: int
    stand_hours: int
    heart_rate_zones: Dict[str, int]  # zone -> minutes

@dataclass
class SleepData:
    """Data structure for sleep analysis."""
    timestamp: float
    sleep_stage: str  # awake, light, deep, rem
    sleep_quality_score: float  # 0-100
    movement_intensity: float  # 0-1 scale
    heart_rate_variability: float  # ms

@dataclass
class FallDetectionData:
    """Data structure for fall detection analysis."""
    timestamp: float
    impact_magnitude: float  # g-force
    fall_detected: bool
    confidence_level: float  # 0-1 scale
    motion_pattern: str  # normal, stumble, fall, hard_fall


class AdvancedECGSensor:
    """Advanced ECG sensor with rhythm analysis."""
    
    def __init__(self, sampling_rate: float = 512):
        """
        Initialize advanced ECG sensor.
        
        Args:
            sampling_rate: ECG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.recording_duration = 30  # seconds for ECG strip
        
        # Heart rhythm parameters
        self.base_heart_rate = 72  # bpm
        self.hrv_mean = 42  # ms
        self.noise_level = 0.02  # mV
        
        # ECG waveform parameters
        self.p_wave_duration = 0.08  # seconds
        self.pr_interval = 0.16
        self.qrs_duration = 0.08
        self.qt_interval = 0.36
        self.t_wave_duration = 0.16
        
        # Rhythm states
        self.rhythm_states = ["normal", "afib", "bradycardia", "tachycardia"]
        self.current_rhythm = "normal"
        self.rhythm_transition_prob = 0.001
        
    def read(self) -> ECGReading:
        """
        Read ECG data with rhythm analysis.
        
        Returns:
            ECGReading with comprehensive cardiac analysis
        """
        current_time = time.time()
        
        # Simulate rhythm transitions
        if np.random.random() < self.rhythm_transition_prob:
            self.current_rhythm = np.random.choice(self.rhythm_states)
        
        # Generate heart rate based on rhythm
        if self.current_rhythm == "bradycardia":
            heart_rate = np.random.normal(50, 5)
        elif self.current_rhythm == "tachycardia":
            heart_rate = np.random.normal(120, 10)
        elif self.current_rhythm == "afib":
            heart_rate = np.random.normal(85, 20)  # Irregular
        else:  # normal
            heart_rate = np.random.normal(self.base_heart_rate, 8)
        
        heart_rate = max(40, min(180, heart_rate))  # Physiological limits
        
        # Generate ECG waveform samples
        samples_per_beat = int(self.sampling_rate * 60 / heart_rate)
        num_samples = min(100, samples_per_beat)  # Limit for performance
        
        voltage_samples = self._generate_ecg_waveform(num_samples, heart_rate)
        
        # Calculate signal quality based on noise
        signal_quality = max(0.1, 1.0 - np.random.exponential(0.1))
        
        return ECGReading(
            timestamp=current_time,
            voltage_samples=voltage_samples.tolist(),
            heart_rate=float(heart_rate),
            rhythm_classification=self.current_rhythm,
            signal_quality=float(signal_quality)
        )
    
    def _generate_ecg_waveform(self, num_samples: int, heart_rate: float) -> np.ndarray:
        """Generate realistic ECG waveform."""
        # Time array for one cardiac cycle
        cycle_duration = 60.0 / heart_rate
        t = np.linspace(0, cycle_duration, num_samples)
        
        # Initialize ECG signal
        ecg = np.zeros_like(t)
        
        # P wave (atrial depolarization)
        p_center = self.pr_interval - self.p_wave_duration/2
        p_wave = 0.15 * np.exp(-((t - p_center) ** 2) / (2 * (self.p_wave_duration/5) ** 2))
        
        # QRS complex (ventricular depolarization)
        qrs_start = self.pr_interval
        q_wave = -0.1 * np.exp(-((t - (qrs_start + 0.02)) ** 2) / (2 * 0.008 ** 2))
        r_wave = 1.0 * np.exp(-((t - (qrs_start + 0.04)) ** 2) / (2 * 0.008 ** 2))
        s_wave = -0.3 * np.exp(-((t - (qrs_start + 0.06)) ** 2) / (2 * 0.008 ** 2))
        
        # T wave (ventricular repolarization)
        t_center = self.pr_interval + self.qrs_duration + 0.12
        t_wave = 0.25 * np.exp(-((t - t_center) ** 2) / (2 * (self.t_wave_duration/4) ** 2))
        
        # Combine components
        ecg = p_wave + q_wave + r_wave + s_wave + t_wave
        
        # Add realistic noise
        noise = np.random.normal(0, self.noise_level, len(ecg))
        
        # Add rhythm-specific variations
        if self.current_rhythm == "afib":
            # Remove P waves and add irregularity
            ecg = q_wave + r_wave + s_wave + t_wave
            irregularity = np.random.normal(0, 0.05, len(ecg))
            ecg += irregularity
        
        return ecg + noise


class BloodOxygenSensor:
    """Advanced blood oxygen sensor with perfusion monitoring."""
    
    def __init__(self):
        """Initialize blood oxygen sensor."""
        self.baseline_spo2 = 98.5  # %
        self.measurement_interval = 10  # seconds
        self.led_wavelengths = [660, 940]  # nm (red, infrared)
        
    def read(self) -> BloodOxygenReading:
        """
        Read blood oxygen saturation with perfusion analysis.
        
        Returns:
            BloodOxygenReading with SpO2 and perfusion data
        """
        current_time = time.time()
        
        # Simulate SpO2 with natural variation
        spo2 = np.random.normal(self.baseline_spo2, 0.8)
        spo2 = max(85, min(100, spo2))  # Physiological limits
        
        # Perfusion index (signal strength indicator)
        perfusion_index = np.random.normal(2.5, 0.8)
        perfusion_index = max(0.1, min(20, perfusion_index))
        
        # Measurement confidence based on perfusion and motion
        motion_artifact = np.random.exponential(0.1)
        confidence = max(0.1, 1.0 - motion_artifact) * (perfusion_index / 5.0)
        confidence = min(1.0, confidence)
        
        # Pulse amplitude (related to perfusion)
        pulse_amplitude = perfusion_index * np.random.normal(1.0, 0.2)
        pulse_amplitude = max(0.1, pulse_amplitude)
        
        return BloodOxygenReading(
            timestamp=current_time,
            spo2_percentage=float(spo2),
            perfusion_index=float(perfusion_index),
            measurement_confidence=float(confidence),
            pulse_amplitude=float(pulse_amplitude)
        )


class ActivityTracker:
    """Comprehensive activity and fitness tracking."""
    
    def __init__(self):
        """Initialize activity tracker."""
        self.daily_step_count = 0
        self.daily_distance = 0.0  # meters
        self.daily_calories = 0.0
        self.daily_active_energy = 0.0  # kcal
        self.daily_exercise_minutes = 0
        self.daily_stand_hours = 0
        self.last_reset = time.time()
        
        # Activity state simulation
        self.current_activity = "sedentary"
        self.activity_states = ["sedentary", "light", "moderate", "vigorous"]
        self.activity_transition_prob = 0.002
        
        # Heart rate zones (age-adjusted)
        self._max_heart_rate = 190  # 220 - age estimate
        self._calculate_hr_zones()
        
        self.zone_time_minutes = {zone: 0 for zone in self.hr_zones.keys()}
        
    @property
    def max_heart_rate(self):
        """Get maximum heart rate."""
        return self._max_heart_rate
    
    @max_heart_rate.setter
    def max_heart_rate(self, value):
        """Set maximum heart rate and recalculate zones."""
        self._max_heart_rate = value
        self._calculate_hr_zones()
        self.zone_time_minutes = {zone: 0 for zone in self.hr_zones.keys()}
    
    def _calculate_hr_zones(self):
        """Calculate heart rate zones based on max heart rate."""
        self.hr_zones = {
            "zone1": (int(0.5 * self._max_heart_rate), int(0.6 * self._max_heart_rate)),  # Recovery
            "zone2": (int(0.6 * self._max_heart_rate), int(0.7 * self._max_heart_rate)),  # Aerobic base
            "zone3": (int(0.7 * self._max_heart_rate), int(0.8 * self._max_heart_rate)),  # Aerobic
            "zone4": (int(0.8 * self._max_heart_rate), int(0.9 * self._max_heart_rate)),  # Lactate threshold
            "zone5": (int(0.9 * self._max_heart_rate), int(1.0 * self._max_heart_rate))   # Neuromuscular
        }
    
    def read(self) -> ActivityMetrics:
        """
        Read comprehensive activity metrics.
        
        Returns:
            ActivityMetrics with fitness and activity data
        """
        current_time = time.time()
        
        # Reset daily counters if new day
        if current_time - self.last_reset > 86400:  # 24 hours
            self._reset_daily_counters()
            self.last_reset = current_time
        
        # Simulate activity transitions
        if np.random.random() < self.activity_transition_prob:
            self.current_activity = np.random.choice(self.activity_states)
        
        # Update metrics based on current activity
        time_delta = 1.0 / 60  # Assume 1 minute increment
        
        if self.current_activity == "light":
            steps_increment = np.random.poisson(15)
            calories_increment = 3.5
            active_energy_increment = 2.5
        elif self.current_activity == "moderate":
            steps_increment = np.random.poisson(80)
            calories_increment = 8.0
            active_energy_increment = 6.0
            self.daily_exercise_minutes += 1
        elif self.current_activity == "vigorous":
            steps_increment = np.random.poisson(150)
            calories_increment = 15.0
            active_energy_increment = 12.0
            self.daily_exercise_minutes += 1
        else:  # sedentary
            steps_increment = np.random.poisson(2)
            calories_increment = 1.2
            active_energy_increment = 0.5
        
        # Update daily totals
        self.daily_step_count += steps_increment
        self.daily_distance += steps_increment * 0.0008  # ~0.8m per step
        self.daily_calories += calories_increment * time_delta
        self.daily_active_energy += active_energy_increment * time_delta
        
        # Simulate stand hours (standing for at least 1 minute per hour)
        hour_of_day = int((current_time % 86400) / 3600)
        if self.current_activity != "sedentary" and hour_of_day < self.daily_stand_hours + 1:
            if np.random.random() < 0.1:  # 10% chance per minute
                self.daily_stand_hours = min(12, self.daily_stand_hours + 1)
        
        return ActivityMetrics(
            timestamp=current_time,
            steps=int(self.daily_step_count),
            distance_meters=float(self.daily_distance),
            calories_burned=float(self.daily_calories),
            active_energy=float(self.daily_active_energy),
            exercise_minutes=int(self.daily_exercise_minutes),
            stand_hours=int(self.daily_stand_hours),
            heart_rate_zones=dict(self.zone_time_minutes)
        )
    
    def _reset_daily_counters(self):
        """Reset daily activity counters."""
        self.daily_step_count = 0
        self.daily_distance = 0.0
        self.daily_calories = 0.0
        self.daily_active_energy = 0.0
        self.daily_exercise_minutes = 0
        self.daily_stand_hours = 0
        self.zone_time_minutes = {zone: 0 for zone in self.hr_zones.keys()}


class SleepTracker:
    """Advanced sleep tracking and analysis."""
    
    def __init__(self):
        """Initialize sleep tracker."""
        self.sleep_stages = ["awake", "light", "deep", "rem"]
        self.current_stage = "awake"
        self.stage_transition_prob = 0.01
        
        # Sleep cycle parameters (90-minute cycles)
        self.cycle_duration = 90 * 60  # seconds
        self.stage_durations = {
            "light": 0.45,  # 45% of cycle
            "deep": 0.25,   # 25% of cycle
            "rem": 0.25,    # 25% of cycle
            "awake": 0.05   # 5% of cycle
        }
        
        # Time tracking
        self.sleep_start_time = None
        self.total_sleep_time = 0
        
        # Testing flag to prevent automatic stage transitions
        self._manual_stage_override = False
    
    def read(self) -> SleepData:
        """
        Read sleep analysis data.
        
        Returns:
            SleepData with sleep stage and quality metrics
        """
        current_time = time.time()
        hour_of_day = (current_time % 86400) / 3600
        
        # Only do automatic stage transitions if not in manual override mode
        if not self._manual_stage_override:
            # Determine if it's typical sleep hours (10 PM - 8 AM)
            is_sleep_time = hour_of_day >= 22 or hour_of_day <= 8
            
            if is_sleep_time:
                # Simulate sleep stage transitions
                if np.random.random() < self.stage_transition_prob:
                    # Weight transitions based on natural sleep patterns
                    if self.current_stage == "awake":
                        self.current_stage = np.random.choice(["light", "awake"], p=[0.8, 0.2])
                    elif self.current_stage == "light":
                        self.current_stage = np.random.choice(["light", "deep", "rem"], p=[0.5, 0.3, 0.2])
                    elif self.current_stage == "deep":
                        self.current_stage = np.random.choice(["deep", "light", "rem"], p=[0.4, 0.4, 0.2])
                    elif self.current_stage == "rem":
                        self.current_stage = np.random.choice(["rem", "light", "awake"], p=[0.5, 0.4, 0.1])
            else:
                self.current_stage = "awake"
        
        # Calculate sleep quality based on stage distribution
        sleep_quality = self._calculate_sleep_quality()
        
        # Movement intensity based on sleep stage (more deterministic)
        movement_base = {
            "awake": 0.8,
            "light": 0.25,  # Reduced to ensure it's higher than deep
            "deep": 0.08,   # Reduced to ensure it's lowest
            "rem": 0.35
        }[self.current_stage]
        
        # Add small controlled variation that maintains ordering
        if self.current_stage == "deep":
            # Deep sleep: very low movement with minimal variation
            movement_variation = np.random.uniform(-0.02, 0.02)
            movement_intensity = max(0.01, min(0.15, movement_base + movement_variation))
        elif self.current_stage == "light":
            # Light sleep: moderate movement, always higher than deep
            movement_variation = np.random.uniform(-0.05, 0.05)
            movement_intensity = max(0.16, min(0.4, movement_base + movement_variation))
        elif self.current_stage == "rem":
            # REM sleep: moderate to high movement
            movement_variation = np.random.uniform(-0.05, 0.05)
            movement_intensity = max(0.25, min(0.5, movement_base + movement_variation))
        else:  # awake
            # Awake: highest movement
            movement_variation = np.random.uniform(-0.1, 0.1)
            movement_intensity = max(0.6, min(1.0, movement_base + movement_variation))
        
        # Heart rate variability (higher in deep sleep)
        hrv_base = {
            "awake": 35,
            "light": 42,
            "deep": 65,
            "rem": 48
        }[self.current_stage]
        
        hrv = np.random.normal(hrv_base, 8)
        hrv = max(20, min(100, hrv))
        
        return SleepData(
            timestamp=current_time,
            sleep_stage=self.current_stage,
            sleep_quality_score=float(sleep_quality),
            movement_intensity=float(movement_intensity),
            heart_rate_variability=float(hrv)
        )
    
    def _calculate_sleep_quality(self) -> float:
        """Calculate sleep quality score based on various factors."""
        # Simplified sleep quality calculation
        stage_quality_weights = {
            "awake": 0.0,
            "light": 0.6,
            "deep": 1.0,
            "rem": 0.8
        }
        
        base_quality = stage_quality_weights[self.current_stage]
        
        # Add some controlled randomness
        quality_variation = np.random.normal(0, 0.05)  # Reduced variance for more stability
        
        quality = base_quality + quality_variation
        return max(0, min(100, quality * 100))


class FallDetectionSystem:
    """Advanced fall detection with machine learning analysis."""
    
    def __init__(self):
        """Initialize fall detection system."""
        self.sensitivity_threshold = 2.5  # g-force threshold
        self.fall_patterns = ["normal", "stumble", "fall", "hard_fall"]
        self.current_pattern = "normal"
        
        # Accelerometer state
        self.baseline_acceleration = np.array([0, 0, 9.8])  # gravity
        self.motion_state = "stationary"
        
    def read(self) -> FallDetectionData:
        """
        Read fall detection analysis.
        
        Returns:
            FallDetectionData with fall analysis
        """
        current_time = time.time()
        
        # Simulate normal movement with occasional events
        impact_magnitude = np.random.exponential(0.2)  # Most values near 0
        
        # Occasionally simulate significant movements
        if np.random.random() < 0.001:  # 0.1% chance
            event_type = np.random.choice(["stumble", "fall", "hard_fall"], p=[0.7, 0.25, 0.05])
            
            if event_type == "stumble":
                impact_magnitude = np.random.normal(1.8, 0.3)
            elif event_type == "fall":
                impact_magnitude = np.random.normal(3.2, 0.5)
            elif event_type == "hard_fall":
                impact_magnitude = np.random.normal(5.5, 0.8)
            
            self.current_pattern = event_type
        else:
            self.current_pattern = "normal"
        
        impact_magnitude = max(0, impact_magnitude)
        
        # Determine if fall is detected
        fall_detected = impact_magnitude > self.sensitivity_threshold
        
        # Calculate confidence based on impact magnitude and pattern recognition
        if fall_detected:
            confidence = min(1.0, (impact_magnitude - self.sensitivity_threshold) / 3.0)
            confidence = max(0.1, confidence + np.random.normal(0, 0.1))
        else:
            confidence = max(0.0, 1.0 - impact_magnitude / self.sensitivity_threshold)
            confidence = min(0.99, confidence)
        
        confidence = max(0, min(1, confidence))
        
        return FallDetectionData(
            timestamp=current_time,
            impact_magnitude=float(impact_magnitude),
            fall_detected=fall_detected,
            confidence_level=float(confidence),
            motion_pattern=self.current_pattern
        )


class SmartWatchDevice(SensorInterface):
    """
    Comprehensive Apple Watch inspired smart watch implementation.
    
    Features:
    - Advanced ECG monitoring with rhythm analysis
    - Blood oxygen saturation monitoring
    - Comprehensive activity and fitness tracking
    - Sleep stage analysis and quality assessment
    - Fall detection with emergency response
    - Heart rate variability analysis
    - Digital Crown and haptic feedback simulation
    """
    
    def __init__(self, sensor_id: int, sampling_rate: float = 1):
        """
        Initialize smart watch device.
        
        Args:
            sensor_id: Unique identifier for the device
            sampling_rate: Main sampling rate in Hz
        """
        super().__init__(sensor_id, "smart_watch", sampling_rate)
        
        # Initialize subsensors
        self.ecg_sensor = AdvancedECGSensor()
        self.blood_oxygen_sensor = BloodOxygenSensor()
        self.activity_tracker = ActivityTracker()
        self.sleep_tracker = SleepTracker()
        self.fall_detector = FallDetectionSystem()
        
        # Device state
        self.battery_level = 0.92
        self.is_on_wrist = True
        self.water_resistance_active = True
        
        # Health settings
        self.user_profile = {
            "age": 32,
            "weight_kg": 70,
            "height_cm": 175,
            "sex": "M",
            "activity_level": "moderate"
        }
        
        # Emergency settings
        self.emergency_contacts = ["Emergency Contact 1", "Emergency Contact 2"]
        self.fall_detection_enabled = True
        self.emergency_sos_enabled = True
        
        logger.info(f"Smart watch device initialized with ID {sensor_id}")
    
    def read(self) -> np.ndarray:
        """
        Read comprehensive smart watch data.
        
        Returns:
            Combined sensor data array with all health metrics
        """
        current_time = time.time()
        
        # Get data from all subsensors
        ecg_data = self.ecg_sensor.read()
        blood_oxygen_data = self.blood_oxygen_sensor.read()
        activity_data = self.activity_tracker.read()
        sleep_data = self.sleep_tracker.read()
        fall_data = self.fall_detector.read()
        
        # Calculate additional derived metrics
        heart_rate_variability = sleep_data.heart_rate_variability
        overall_health_score = self._calculate_overall_health_score(
            ecg_data, blood_oxygen_data, activity_data, sleep_data
        )
        
        # Combine data into structured array
        # Format: [timestamp, heart_rate, spo2, steps, distance_meters,
        #          sleep_quality, fall_detected, hrv, health_score,
        #          exercise_minutes, calories_burned, battery_level]
        combined_data = np.array([
            current_time,
            ecg_data.heart_rate,
            blood_oxygen_data.spo2_percentage,
            float(activity_data.steps),
            activity_data.distance_meters,
            sleep_data.sleep_quality_score,
            float(fall_data.fall_detected),
            heart_rate_variability,
            overall_health_score,
            float(activity_data.exercise_minutes),
            activity_data.calories_burned,
            self.battery_level
        ])
        
        return combined_data
    
    def _calculate_overall_health_score(self, ecg_data: ECGReading, 
                                      blood_oxygen_data: BloodOxygenReading,
                                      activity_data: ActivityMetrics,
                                      sleep_data: SleepData) -> float:
        """Calculate overall health score from all metrics."""
        
        # Heart rate score (target: 60-100 bpm)
        hr_score = 1.0
        if ecg_data.heart_rate < 60:
            hr_score = max(0.3, ecg_data.heart_rate / 60)
        elif ecg_data.heart_rate > 100:
            hr_score = max(0.3, 1.0 - (ecg_data.heart_rate - 100) / 50)
        
        # SpO2 score (target: >95%)
        spo2_score = min(1.0, blood_oxygen_data.spo2_percentage / 98)
        
        # Activity score (target: 10,000 steps, 30 min exercise)
        steps_score = min(1.0, activity_data.steps / 10000)
        exercise_score = min(1.0, activity_data.exercise_minutes / 30)
        activity_score = (steps_score + exercise_score) / 2
        
        # Sleep score
        sleep_score = sleep_data.sleep_quality_score / 100
        
        # Weighted overall score
        weights = {
            "heart_rate": 0.25,
            "spo2": 0.25,
            "activity": 0.3,
            "sleep": 0.2
        }
        
        overall_score = (
            weights["heart_rate"] * hr_score +
            weights["spo2"] * spo2_score +
            weights["activity"] * activity_score +
            weights["sleep"] * sleep_score
        )
        
        return float(overall_score * 100)  # 0-100 scale
    
    def get_ecg_reading(self) -> ECGReading:
        """Get detailed ECG reading."""
        return self.ecg_sensor.read()
    
    def get_blood_oxygen_reading(self) -> BloodOxygenReading:
        """Get detailed blood oxygen reading."""
        return self.blood_oxygen_sensor.read()
    
    def get_activity_metrics(self) -> ActivityMetrics:
        """Get detailed activity metrics."""
        return self.activity_tracker.read()
    
    def get_sleep_analysis(self) -> SleepData:
        """Get detailed sleep analysis."""
        return self.sleep_tracker.read()
    
    def get_fall_detection_status(self) -> FallDetectionData:
        """Get fall detection status."""
        return self.fall_detector.read()
    
    def trigger_emergency_sos(self) -> Dict[str, Any]:
        """Trigger emergency SOS (simulation)."""
        if not self.emergency_sos_enabled:
            return {"status": "disabled", "message": "Emergency SOS is disabled"}
        
        current_time = time.time()
        location_data = {
            "latitude": 37.7749 + np.random.normal(0, 0.001),  # San Francisco area
            "longitude": -122.4194 + np.random.normal(0, 0.001),
            "accuracy": np.random.normal(5, 2)  # meters
        }
        
        emergency_response = {
            "status": "activated",
            "timestamp": current_time,
            "user_id": self.sensor_id,
            "location": location_data,
            "contacts_notified": self.emergency_contacts,
            "medical_id_shared": True,
            "emergency_services_contacted": True
        }
        
        logger.warning(f"Emergency SOS activated for device {self.sensor_id}")
        return emergency_response
    
    def update_user_profile(self, profile_updates: Dict[str, Any]) -> None:
        """Update user profile for personalized health analysis."""
        self.user_profile.update(profile_updates)
        
        # Update heart rate zones when age changes
        if "age" in profile_updates:
            max_hr = 220 - profile_updates["age"]
            self.activity_tracker.max_heart_rate = max_hr
            self.activity_tracker.hr_zones = {
                "zone1": (int(0.5 * max_hr), int(0.6 * max_hr)),
                "zone2": (int(0.6 * max_hr), int(0.7 * max_hr)),
                "zone3": (int(0.7 * max_hr), int(0.8 * max_hr)),
                "zone4": (int(0.8 * max_hr), int(0.9 * max_hr)),
                "zone5": (int(0.9 * max_hr), int(1.0 * max_hr))
            }
        
        logger.info(f"User profile updated for device {self.sensor_id}")
    
    def get_health_insights(self) -> Dict[str, Any]:
        """Generate comprehensive health insights."""
        recent_data = self.get_buffer(clear=False)
        if len(recent_data) < 5:
            return {"status": "insufficient_data"}
        
        # Extract recent metrics
        recent_readings = [data[1] for data in recent_data[-10:]]
        
        avg_heart_rate = np.mean([reading[1] for reading in recent_readings])
        avg_spo2 = np.mean([reading[2] for reading in recent_readings])
        total_steps = recent_readings[-1][3]  # Latest step count
        avg_sleep_quality = np.mean([reading[5] for reading in recent_readings])
        avg_hrv = np.mean([reading[7] for reading in recent_readings])
        
        insights = {
            "cardiovascular_health": {
                "avg_heart_rate": float(avg_heart_rate),
                "heart_rate_status": "normal" if 60 <= avg_heart_rate <= 100 else "attention_needed",
                "avg_hrv": float(avg_hrv),
                "hrv_status": "good" if avg_hrv > 40 else "below_average"
            },
            "respiratory_health": {
                "avg_spo2": float(avg_spo2),
                "spo2_status": "normal" if avg_spo2 >= 95 else "low"
            },
            "activity_assessment": {
                "daily_steps": int(total_steps),
                "step_goal_progress": min(100, (total_steps / 10000) * 100),
                "activity_level": "active" if total_steps > 8000 else "moderate" if total_steps > 5000 else "sedentary"
            },
            "sleep_assessment": {
                "avg_sleep_quality": float(avg_sleep_quality),
                "sleep_status": "good" if avg_sleep_quality > 70 else "fair" if avg_sleep_quality > 50 else "poor"
            },
            "recommendations": self._generate_health_recommendations(avg_heart_rate, avg_spo2, total_steps, avg_sleep_quality)
        }
        
        return insights
    
    def _generate_health_recommendations(self, heart_rate: float, spo2: float, 
                                       steps: int, sleep_quality: float) -> List[str]:
        """Generate personalized health recommendations."""
        recommendations = []
        
        if heart_rate > 100:
            recommendations.append("Consider stress reduction techniques and consult healthcare provider about elevated heart rate")
        elif heart_rate < 60:
            recommendations.append("Monitor for symptoms; consult healthcare provider about low heart rate")
        
        if spo2 < 95:
            recommendations.append("Blood oxygen levels are low; consider consulting a healthcare provider")
        
        if steps < 5000:
            recommendations.append("Increase daily physical activity; aim for at least 10,000 steps per day")
        elif steps < 8000:
            recommendations.append("Good activity level; try to increase daily steps for optimal health")
        
        if sleep_quality < 50:
            recommendations.append("Focus on sleep hygiene; maintain consistent sleep schedule and create optimal sleep environment")
        elif sleep_quality < 70:
            recommendations.append("Consider improving sleep quality through relaxation techniques and consistent bedtime routine")
        
        if not recommendations:
            recommendations.append("Excellent health metrics! Continue maintaining your healthy lifestyle")
        
        return recommendations 