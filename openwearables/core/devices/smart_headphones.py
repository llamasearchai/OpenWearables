"""
Smart Headphones Device Implementation

AirPods Pro inspired smart headphones with comprehensive audio health monitoring,
spatial audio processing, hearing protection, and biometric sensing.
"""

import time
import logging
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..sensor_manager import SensorInterface

logger = logging.getLogger("OpenWearables.SmartHeadphones")

@dataclass
class AudioHealthData:
    """Data structure for audio health measurements."""
    timestamp: float
    volume_level: float  # dB
    frequency_exposure: Dict[str, float]  # Hz bands -> exposure time
    hearing_threshold: Dict[str, float]  # Hz -> dB threshold
    tinnitus_risk_score: float  # 0-1 scale
    cumulative_exposure: float  # TWA in dB

@dataclass
class SpatialAudioData:
    """Data structure for spatial audio measurements."""
    timestamp: float
    head_rotation_x: float  # degrees
    head_rotation_y: float  # degrees
    head_rotation_z: float  # degrees
    spatial_accuracy: float  # 0-1 scale
    room_acoustics_score: float  # 0-1 scale
    localization_error: float  # degrees

@dataclass
class BiometricAudioData:
    """Data structure for audio-based biometric measurements."""
    timestamp: float
    ear_canal_temperature: float  # celsius
    heart_rate_ppg: float  # bpm via ear canal PPG
    fit_quality: float  # 0-1 scale
    earwax_moisture: float  # 0-1 scale
    speech_clarity_index: float  # 0-1 scale

@dataclass
class EnvironmentalAudioData:
    """Data structure for environmental audio measurements."""
    timestamp: float
    ambient_noise_level: float  # dB
    noise_classification: str  # traffic, music, speech, etc.
    wind_noise: float  # 0-1 scale
    noise_cancellation_effectiveness: float  # 0-1 scale

class AudioHealthMonitor:
    """Advanced audio health monitoring system."""
    
    def __init__(self, sampling_rate: float = 48000):
        self.sampling_rate = sampling_rate
        self.safe_listening_limits = {
            "85_db_limit": 8 * 3600,  # 8 hours at 85dB
            "90_db_limit": 2.5 * 3600,  # 2.5 hours at 90dB
            "100_db_limit": 15 * 60,  # 15 minutes at 100dB
            "110_db_limit": 2 * 60   # 2 minutes at 110dB
        }
        
        # Current session tracking
        self.session_exposure = 0.0  # cumulative exposure in dB*hours
        self.session_start = time.time()
        self.current_volume = 70.0  # dB
        
        # Frequency exposure tracking
        self.frequency_bands = {
            "low": (20, 250),     # 20-250 Hz
            "mid_low": (250, 500), # 250-500 Hz
            "mid": (500, 2000),   # 500-2000 Hz
            "mid_high": (2000, 4000), # 2-4 kHz
            "high": (4000, 8000), # 4-8 kHz
            "very_high": (8000, 20000) # 8-20 kHz
        }
        
        self.frequency_exposure = {band: 0.0 for band in self.frequency_bands}
        
        # Hearing threshold baseline (healthy young adult)
        self.baseline_hearing = {
            "125": 7.5,   # 125 Hz -> 7.5 dB
            "250": 7.5,   # 250 Hz -> 7.5 dB
            "500": 7.5,   # 500 Hz -> 7.5 dB
            "1000": 7.5,  # 1 kHz -> 7.5 dB
            "2000": 7.5,  # 2 kHz -> 7.5 dB
            "4000": 7.5,  # 4 kHz -> 7.5 dB
            "8000": 7.5   # 8 kHz -> 7.5 dB
        }
        
        # Progressive hearing damage simulation
        self.hearing_damage = {freq: 0.0 for freq in self.baseline_hearing}
        
    def read(self) -> AudioHealthData:
        """Generate realistic audio health monitoring data."""
        current_time = time.time()
        
        # Simulate realistic volume levels
        self._update_volume_levels()
        
        # Calculate frequency exposure
        self._update_frequency_exposure()
        
        # Calculate cumulative exposure using equivalent continuous sound level
        session_duration = (current_time - self.session_start) / 3600  # hours
        if session_duration > 0:
            # Simplified TWA calculation
            cumulative_exposure = self.current_volume + 10 * np.log10(session_duration)
        else:
            cumulative_exposure = 0
        
        # Assess tinnitus risk based on exposure patterns
        tinnitus_risk = self._calculate_tinnitus_risk()
        
        # Get current hearing thresholds (baseline + damage)
        current_thresholds = {}
        for freq, baseline in self.baseline_hearing.items():
            damage = self.hearing_damage.get(freq, 0)
            current_thresholds[freq] = baseline + damage
        
        return AudioHealthData(
            timestamp=current_time,
            volume_level=self.current_volume,
            frequency_exposure=self.frequency_exposure.copy(),
            hearing_threshold=current_thresholds,
            tinnitus_risk_score=tinnitus_risk,
            cumulative_exposure=cumulative_exposure
        )
    
    def _update_volume_levels(self):
        """Update current volume levels with realistic patterns."""
        # Volume changes based on content type and user behavior
        content_types = ["music", "podcast", "call", "silence"]
        content_probabilities = [0.4, 0.3, 0.2, 0.1]
        
        if np.random.random() < 0.1:  # 10% chance to change content
            content = np.random.choice(content_types, p=content_probabilities)
            
            if content == "music":
                self.current_volume = np.random.normal(78, 8)  # Music typically louder
            elif content == "podcast":
                self.current_volume = np.random.normal(70, 5)  # Speech content
            elif content == "call":
                self.current_volume = np.random.normal(65, 3)  # Call volume
            else:  # silence
                self.current_volume = 0
        
        # Add small random variations
        if self.current_volume > 0:
            self.current_volume += np.random.normal(0, 2)
            self.current_volume = np.clip(self.current_volume, 30, 120)
    
    def _update_frequency_exposure(self):
        """Update frequency-specific exposure tracking."""
        if self.current_volume > 0:
            # Simulate frequency distribution of current audio
            # Music has more bass, speech has more mids
            exposure_increment = 1.0 / 3600  # 1 second in hours
            
            # Different content has different frequency emphasis
            if self.current_volume > 75:  # Loud music
                self.frequency_exposure["low"] += exposure_increment * 1.5
                self.frequency_exposure["mid"] += exposure_increment * 1.2
                self.frequency_exposure["high"] += exposure_increment * 1.0
            else:  # Speech/quiet content
                self.frequency_exposure["mid_low"] += exposure_increment * 1.3
                self.frequency_exposure["mid"] += exposure_increment * 1.5
                self.frequency_exposure["mid_high"] += exposure_increment * 1.2
    
    def _calculate_tinnitus_risk(self) -> float:
        """Calculate tinnitus risk score based on exposure patterns."""
        risk_score = 0.0
        
        # High frequency exposure increases tinnitus risk
        high_freq_exposure = (self.frequency_exposure["high"] + 
                             self.frequency_exposure["very_high"])
        
        if high_freq_exposure > 4:  # More than 4 hours
            risk_score += 0.3
        elif high_freq_exposure > 2:  # More than 2 hours
            risk_score += 0.1
        
        # High volume exposure
        if self.current_volume > 100:
            risk_score += 0.4
        elif self.current_volume > 85:
            risk_score += 0.2
        
        # Session duration factor
        session_hours = (time.time() - self.session_start) / 3600
        if session_hours > 8:
            risk_score += 0.3
        
        return min(risk_score, 1.0)

class SpatialAudioProcessor:
    """Spatial audio processing and head tracking."""
    
    def __init__(self, sampling_rate: float = 1000):
        self.sampling_rate = sampling_rate
        
        # Head orientation state
        self.head_rotation_x = 0.0  # pitch
        self.head_rotation_y = 0.0  # yaw
        self.head_rotation_z = 0.0  # roll
        
        # Movement simulation
        self.movement_pattern = "stationary"  # stationary, walking, active
        
        # Spatial audio quality metrics
        self.localization_accuracy = 0.95
        self.room_characteristics = "medium_room"
        
    def read(self) -> SpatialAudioData:
        """Generate spatial audio processing data."""
        current_time = time.time()
        
        # Update head movement patterns
        self._simulate_head_movement()
        
        # Calculate spatial audio accuracy
        spatial_accuracy = self._calculate_spatial_accuracy()
        
        # Assess room acoustics
        room_score = self._assess_room_acoustics()
        
        # Calculate localization error
        localization_error = self._calculate_localization_error()
        
        return SpatialAudioData(
            timestamp=current_time,
            head_rotation_x=self.head_rotation_x,
            head_rotation_y=self.head_rotation_y,
            head_rotation_z=self.head_rotation_z,
            spatial_accuracy=spatial_accuracy,
            room_acoustics_score=room_score,
            localization_error=localization_error
        )
    
    def _simulate_head_movement(self):
        """Simulate realistic head movement patterns."""
        # Change movement pattern occasionally
        if np.random.random() < 0.01:  # 1% chance
            self.movement_pattern = np.random.choice([
                "stationary", "walking", "active"
            ], p=[0.6, 0.3, 0.1])
        
        if self.movement_pattern == "stationary":
            # Small random movements
            self.head_rotation_x += np.random.normal(0, 0.5)
            self.head_rotation_y += np.random.normal(0, 0.5)
            self.head_rotation_z += np.random.normal(0, 0.2)
        elif self.movement_pattern == "walking":
            # Walking motion with periodic head movements
            walking_frequency = 2.0  # Hz
            time_factor = time.time() * walking_frequency * 2 * np.pi
            
            self.head_rotation_x += 2 * np.sin(time_factor) + np.random.normal(0, 1)
            self.head_rotation_y += 3 * np.sin(time_factor * 0.7) + np.random.normal(0, 2)
            self.head_rotation_z += 1 * np.sin(time_factor * 1.3) + np.random.normal(0, 0.5)
        else:  # active
            # More dynamic movements
            self.head_rotation_x += np.random.normal(0, 3)
            self.head_rotation_y += np.random.normal(0, 5)
            self.head_rotation_z += np.random.normal(0, 2)
        
        # Keep rotations within reasonable bounds
        self.head_rotation_x = np.clip(self.head_rotation_x, -45, 45)
        self.head_rotation_y = np.clip(self.head_rotation_y, -180, 180)
        self.head_rotation_z = np.clip(self.head_rotation_z, -30, 30)
    
    def _calculate_spatial_accuracy(self) -> float:
        """Calculate spatial audio accuracy."""
        base_accuracy = 0.95
        
        # Accuracy decreases with rapid movement
        movement_factor = abs(self.head_rotation_x) + abs(self.head_rotation_y)
        if movement_factor > 20:
            accuracy = base_accuracy - 0.1
        elif movement_factor > 10:
            accuracy = base_accuracy - 0.05
        else:
            accuracy = base_accuracy
        
        # Add small random variation
        accuracy += np.random.normal(0, 0.02)
        return np.clip(accuracy, 0.5, 1.0)
    
    def _assess_room_acoustics(self) -> float:
        """Assess room acoustics quality."""
        # Simulate different room types
        room_types = {
            "small_room": 0.6,
            "medium_room": 0.8,
            "large_room": 0.7,
            "outdoor": 0.9
        }
        
        # Occasionally change room
        if np.random.random() < 0.001:  # 0.1% chance
            self.room_characteristics = np.random.choice(list(room_types.keys()))
        
        base_score = room_types[self.room_characteristics]
        
        # Add variation
        score = base_score + np.random.normal(0, 0.05)
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_localization_error(self) -> float:
        """Calculate sound localization error in degrees."""
        base_error = 2.0  # degrees
        
        # Error increases with movement
        movement_factor = (abs(self.head_rotation_x) + 
                         abs(self.head_rotation_y) + 
                         abs(self.head_rotation_z)) / 3
        
        error = base_error + movement_factor * 0.1
        error += np.random.normal(0, 0.5)
        
        return max(0, error)

class BiometricAudioSensor:
    """Audio-based biometric monitoring."""
    
    def __init__(self, sampling_rate: float = 25):
        self.sampling_rate = sampling_rate
        
        # Baseline values
        self.baseline_ear_temp = 35.5  # Celsius
        self.baseline_heart_rate = 70  # bpm
        
        # Fit quality simulation
        self.fit_quality = 0.85
        self.last_fit_check = time.time()
        
        # Earwax and moisture monitoring
        self.earwax_baseline = 0.3  # 0-1 scale
        
    def read(self) -> BiometricAudioData:
        """Generate biometric audio sensor data."""
        current_time = time.time()
        
        # Ear canal temperature (warmer than skin temperature)
        ear_temperature = self.baseline_ear_temp + np.random.normal(0, 0.2)
        ear_temperature += self._get_temperature_variation()
        
        # Heart rate via ear canal PPG
        heart_rate = self.baseline_heart_rate + np.random.normal(0, 5)
        heart_rate += self._get_heart_rate_variation()
        heart_rate = max(40, min(200, heart_rate))
        
        # Fit quality assessment
        self._update_fit_quality()
        
        # Earwax moisture levels
        earwax_moisture = self.earwax_baseline + np.random.normal(0, 0.1)
        earwax_moisture = np.clip(earwax_moisture, 0, 1)
        
        # Speech clarity index (for calls/voice commands)
        speech_clarity = self._assess_speech_clarity()
        
        return BiometricAudioData(
            timestamp=current_time,
            ear_canal_temperature=ear_temperature,
            heart_rate_ppg=heart_rate,
            fit_quality=self.fit_quality,
            earwax_moisture=earwax_moisture,
            speech_clarity_index=speech_clarity
        )
    
    def _get_temperature_variation(self) -> float:
        """Get temperature variation based on activity and environment."""
        # Temperature increases with physical activity
        activity_factor = np.random.choice([0, 0.5, 1.0], p=[0.7, 0.2, 0.1])
        return activity_factor * 0.5
    
    def _get_heart_rate_variation(self) -> float:
        """Get heart rate variation based on activity and stress."""
        # Heart rate varies with activity and emotional state
        activity_states = ["resting", "light_activity", "moderate_activity"]
        activity = np.random.choice(activity_states, p=[0.8, 0.15, 0.05])
        
        if activity == "resting":
            return np.random.normal(0, 3)
        elif activity == "light_activity":
            return np.random.normal(10, 5)
        else:  # moderate_activity
            return np.random.normal(25, 8)
    
    def _update_fit_quality(self):
        """Update earbud fit quality over time."""
        current_time = time.time()
        
        # Fit quality degrades slowly over time
        time_since_check = current_time - self.last_fit_check
        if time_since_check > 3600:  # Check every hour
            # Slight degradation over time
            self.fit_quality *= 0.999
            
            # Occasional readjustment
            if np.random.random() < 0.1:  # 10% chance of readjustment
                self.fit_quality = min(0.95, self.fit_quality + 0.1)
            
            self.last_fit_check = current_time
        
        # Add small random variations
        current_fit = self.fit_quality + np.random.normal(0, 0.02)
        self.fit_quality = np.clip(current_fit, 0.3, 1.0)
    
    def _assess_speech_clarity(self) -> float:
        """Assess speech clarity for calls and voice commands."""
        # Based on fit quality and ambient noise
        base_clarity = self.fit_quality
        
        # Ambient noise affects clarity
        noise_factor = np.random.uniform(0.8, 1.0)  # Simulated noise impact
        clarity = base_clarity * noise_factor
        
        # Add small variation
        clarity += np.random.normal(0, 0.05)
        return np.clip(clarity, 0.0, 1.0)

class EnvironmentalAudioSensor:
    """Environmental audio monitoring."""
    
    def __init__(self, sampling_rate: float = 10):
        self.sampling_rate = sampling_rate
        
        # Environment state
        self.current_environment = "indoor_quiet"
        
        # Noise cancellation state
        self.anc_enabled = True
        self.anc_effectiveness = 0.8
        
    def read(self) -> EnvironmentalAudioData:
        """Generate environmental audio data."""
        current_time = time.time()
        
        # Update environment occasionally
        self._update_environment()
        
        # Get ambient noise level
        ambient_noise = self._get_ambient_noise()
        
        # Classify noise type
        noise_classification = self._classify_noise()
        
        # Assess wind noise
        wind_noise = self._assess_wind_noise()
        
        # Calculate ANC effectiveness
        anc_effectiveness = self._calculate_anc_effectiveness(ambient_noise)
        
        return EnvironmentalAudioData(
            timestamp=current_time,
            ambient_noise_level=ambient_noise,
            noise_classification=noise_classification,
            wind_noise=wind_noise,
            noise_cancellation_effectiveness=anc_effectiveness
        )
    
    def _update_environment(self):
        """Update current environment type."""
        if np.random.random() < 0.01:  # 1% chance to change
            environments = [
                "indoor_quiet", "indoor_moderate", "outdoor_calm", 
                "outdoor_windy", "traffic", "public_transport"
            ]
            self.current_environment = np.random.choice(environments)
    
    def _get_ambient_noise(self) -> float:
        """Get ambient noise level based on environment."""
        noise_levels = {
            "indoor_quiet": (30, 40),
            "indoor_moderate": (45, 55),
            "outdoor_calm": (40, 50),
            "outdoor_windy": (50, 65),
            "traffic": (65, 80),
            "public_transport": (70, 85)
        }
        
        min_level, max_level = noise_levels[self.current_environment]
        ambient_noise = np.random.uniform(min_level, max_level)
        
        # Add temporal variation
        ambient_noise += np.random.normal(0, 3)
        
        return max(20, min(120, ambient_noise))
    
    def _classify_noise(self) -> str:
        """Classify the type of ambient noise."""
        environment_to_noise = {
            "indoor_quiet": "hvac",
            "indoor_moderate": "conversation",
            "outdoor_calm": "nature",
            "outdoor_windy": "wind",
            "traffic": "traffic",
            "public_transport": "mechanical"
        }
        
        base_classification = environment_to_noise[self.current_environment]
        
        # Occasionally detect other noise types
        if np.random.random() < 0.1:
            other_types = ["music", "construction", "machinery", "crowd"]
            return np.random.choice(other_types)
        
        return base_classification
    
    def _assess_wind_noise(self) -> float:
        """Assess wind noise level."""
        if self.current_environment in ["outdoor_windy", "public_transport"]:
            wind_noise = np.random.uniform(0.3, 0.8)
        else:
            wind_noise = np.random.uniform(0.0, 0.1)
        
        return wind_noise
    
    def _calculate_anc_effectiveness(self, ambient_noise: float) -> float:
        """Calculate noise cancellation effectiveness."""
        if not self.anc_enabled:
            return 0.0
        
        # ANC effectiveness depends on noise frequency and level
        base_effectiveness = self.anc_effectiveness
        
        # ANC works better on consistent, low-frequency noise
        if self.current_environment in ["traffic", "hvac"]:
            effectiveness = base_effectiveness * 1.1
        elif self.current_environment in ["conversation", "music"]:
            effectiveness = base_effectiveness * 0.7
        else:
            effectiveness = base_effectiveness
        
        # Very loud noise reduces effectiveness
        if ambient_noise > 80:
            effectiveness *= 0.8
        
        effectiveness += np.random.normal(0, 0.05)
        return np.clip(effectiveness, 0.0, 1.0)

class SmartHeadphonesDevice(SensorInterface):
    """
    Comprehensive smart headphones implementation with advanced audio health monitoring.
    
    Features:
    - Advanced audio health monitoring with hearing protection
    - Spatial audio processing with head tracking
    - In-ear biometric monitoring (PPG, temperature)
    - Environmental audio sensing and noise cancellation
    - Personalized hearing profiles and tinnitus monitoring
    """
    
    def __init__(self, sensor_id: int, sampling_rate: float = 25):
        """
        Initialize smart headphones device.
        
        Args:
            sensor_id: Unique identifier for the device
            sampling_rate: Main sampling rate in Hz
        """
        super().__init__(sensor_id, "smart_headphones", sampling_rate)
        
        # Initialize subsensors
        self.audio_health_monitor = AudioHealthMonitor()
        self.spatial_audio_processor = SpatialAudioProcessor()
        self.biometric_sensor = BiometricAudioSensor()
        self.environmental_sensor = EnvironmentalAudioSensor()
        
        # Device state
        self.is_fitted = True
        self.battery_level = 0.85
        self.anc_enabled = True
        
        # Personalization
        self.hearing_profile = self._initialize_hearing_profile()
        self.user_preferences = {
            "max_safe_volume": 85,  # dB
            "hearing_protection": True,
            "spatial_audio": True
        }
        
        logger.info(f"Smart headphones device initialized with ID {sensor_id}")
    
    def read(self) -> np.ndarray:
        """
        Read comprehensive smart headphones data.
        
        Returns:
            Combined sensor data array
        """
        current_time = time.time()
        
        # Get data from all subsensors
        audio_health = self.audio_health_monitor.read()
        spatial_audio = self.spatial_audio_processor.read()
        biometric = self.biometric_sensor.read()
        environmental = self.environmental_sensor.read()
        
        # Combine data into structured array
        # Format: [timestamp, volume_level, heart_rate, ear_temp, 
        #          spatial_accuracy, ambient_noise, fit_quality,
        #          tinnitus_risk, anc_effectiveness, hearing_safety_score]
        
        hearing_safety_score = self._calculate_hearing_safety_score(audio_health)
        
        combined_data = np.array([
            current_time,
            audio_health.volume_level,
            biometric.heart_rate_ppg,
            biometric.ear_canal_temperature,
            spatial_audio.spatial_accuracy,
            environmental.ambient_noise_level,
            biometric.fit_quality,
            audio_health.tinnitus_risk_score,
            environmental.noise_cancellation_effectiveness,
            hearing_safety_score
        ])
        
        return combined_data
    
    def _initialize_hearing_profile(self) -> Dict[str, Any]:
        """Initialize personalized hearing profile."""
        return {
            "age_group": "young_adult",  # young_adult, middle_age, senior
            "hearing_sensitivity": "normal",  # high, normal, low
            "frequency_preferences": {
                "bass_boost": 0.0,    # -10 to +10 dB
                "mid_adjustment": 0.0,
                "treble_boost": 0.0
            },
            "listening_habits": {
                "avg_daily_hours": 4.0,
                "preferred_volume": 75,  # dB
                "music_genres": ["pop", "classical", "rock"]
            }
        }
    
    def _calculate_hearing_safety_score(self, audio_health: AudioHealthData) -> float:
        """Calculate overall hearing safety score (0-1)."""
        score = 1.0
        
        # Penalize high volume
        if audio_health.volume_level > 100:
            score -= 0.4
        elif audio_health.volume_level > 85:
            score -= 0.2
        
        # Penalize high tinnitus risk
        score -= audio_health.tinnitus_risk_score * 0.3
        
        # Penalize excessive cumulative exposure
        if audio_health.cumulative_exposure > 90:
            score -= 0.3
        elif audio_health.cumulative_exposure > 85:
            score -= 0.1
        
        return max(0.0, score)
    
    def get_hearing_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive hearing health report."""
        audio_data = self.audio_health_monitor.read()
        
        # Calculate exposure summary
        total_exposure_hours = sum(audio_data.frequency_exposure.values())
        
        # Assess hearing risk
        risk_factors = []
        if audio_data.volume_level > 85:
            risk_factors.append("High volume exposure")
        if audio_data.tinnitus_risk_score > 0.5:
            risk_factors.append("Elevated tinnitus risk")
        if total_exposure_hours > 8:
            risk_factors.append("Excessive listening duration")
        
        # Generate recommendations
        recommendations = self._generate_hearing_recommendations(audio_data)
        
        return {
            "exposure_summary": {
                "current_volume": audio_data.volume_level,
                "cumulative_exposure": audio_data.cumulative_exposure,
                "frequency_exposure": audio_data.frequency_exposure,
                "total_hours": total_exposure_hours
            },
            "hearing_thresholds": audio_data.hearing_threshold,
            "risk_assessment": {
                "tinnitus_risk": audio_data.tinnitus_risk_score,
                "overall_risk": "low" if len(risk_factors) == 0 else "moderate" if len(risk_factors) <= 2 else "high",
                "risk_factors": risk_factors
            },
            "recommendations": recommendations,
            "hearing_profile": self.hearing_profile
        }
    
    def _generate_hearing_recommendations(self, audio_data: AudioHealthData) -> List[str]:
        """Generate personalized hearing health recommendations."""
        recommendations = []
        
        # Volume recommendations
        if audio_data.volume_level > 85:
            recommendations.append("Consider reducing volume to protect your hearing")
            recommendations.append("Follow the 60/60 rule: no more than 60% volume for 60 minutes")
        
        # Duration recommendations
        exposure_hours = sum(audio_data.frequency_exposure.values())
        if exposure_hours > 8:
            recommendations.append("Take regular breaks from audio to rest your ears")
        
        # Tinnitus prevention
        if audio_data.tinnitus_risk_score > 0.3:
            recommendations.append("High-frequency exposure detected - consider limiting listening time")
        
        # Frequency-specific advice
        high_freq_exposure = (audio_data.frequency_exposure.get("high", 0) + 
                             audio_data.frequency_exposure.get("very_high", 0))
        if high_freq_exposure > 2:
            recommendations.append("Reduce treble/high-frequency content to prevent hearing damage")
        
        # Environmental recommendations
        return recommendations
    
    def calibrate_spatial_audio(self) -> bool:
        """Calibrate spatial audio for user's head shape and preferences."""
        try:
            logger.info("Starting spatial audio calibration")
            
            # Simulate calibration process
            time.sleep(3)
            
            # Update spatial audio processor with calibration data
            self.spatial_audio_processor.localization_accuracy = 0.98
            
            logger.info("Spatial audio calibration completed")
            return True
            
        except Exception as e:
            logger.error(f"Spatial audio calibration failed: {str(e)}")
            return False
    
    def update_hearing_profile(self, profile_data: Dict[str, Any]) -> None:
        """Update personalized hearing profile."""
        self.hearing_profile.update(profile_data)
        
        # Adjust audio processing based on profile
        if profile_data.get("hearing_sensitivity") == "low":
            # Boost volume and clarity for hearing impaired users
            pass
        
        logger.info("Hearing profile updated")
    
    def get_fit_analysis(self) -> Dict[str, Any]:
        """Analyze earbud fit quality and provide recommendations."""
        biometric_data = self.biometric_sensor.read()
        
        fit_quality = biometric_data.fit_quality
        
        if fit_quality > 0.8:
            fit_status = "excellent"
            recommendations = ["Fit is optimal for audio quality and comfort"]
        elif fit_quality > 0.6:
            fit_status = "good"
            recommendations = ["Consider adjusting earbuds for better seal", 
                             "Try different ear tip sizes if available"]
        else:
            fit_status = "poor"
            recommendations = ["Readjust earbuds for proper fit",
                             "Poor fit affects audio quality and noise cancellation",
                             "Consider different ear tip sizes"]
        
        return {
            "fit_quality_score": fit_quality,
            "fit_status": fit_status,
            "recommendations": recommendations,
            "impact_on_audio": {
                "bass_response": fit_quality * 0.8 + 0.2,
                "noise_isolation": fit_quality * 0.9 + 0.1,
                "comfort_score": fit_quality
            }
        }
    
    def get_environmental_analysis(self) -> Dict[str, Any]:
        """Analyze environmental audio conditions."""
        env_data = self.environmental_sensor.read()
        
        # Assess listening environment
        if env_data.ambient_noise_level < 40:
            environment_quality = "quiet"
            recommendation = "Good environment for detailed listening"
        elif env_data.ambient_noise_level < 60:
            environment_quality = "moderate"
            recommendation = "Consider noise cancellation for better experience"
        else:
            environment_quality = "noisy"
            recommendation = "High ambient noise - use noise cancellation and moderate volume"
        
        return {
            "ambient_noise_level": env_data.ambient_noise_level,
            "noise_type": env_data.noise_classification,
            "environment_quality": environment_quality,
            "anc_effectiveness": env_data.noise_cancellation_effectiveness,
            "wind_noise_level": env_data.wind_noise,
            "recommendation": recommendation
        }

# Export the main class
__all__ = ["SmartHeadphonesDevice"] 