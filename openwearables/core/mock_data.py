"""
OpenWearables Mock Data Generator
Provides realistic mock data for development and testing
"""

import random
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import math

class MockDataGenerator:
    """Generates realistic mock data for OpenWearables platform"""
    
    def __init__(self, user_profile: Optional[Dict] = None):
        """Initialize mock data generator with optional user profile"""
        self.user_profile = user_profile or {
            "age": 30,
            "gender": "female",
            "height": 165,  # cm
            "weight": 65,   # kg
            "fitness_level": "moderate",
            "medical_conditions": [],
            "resting_hr": 65
        }
        
        # Base physiological parameters
        self.base_params = self._calculate_base_params()
        
        # Time-based factors
        self.start_time = datetime.now()
        self.data_points = []
        
        # Activity states
        self.activity_states = ["resting", "light", "moderate", "vigorous", "sleeping"]
        self.current_activity = "resting"
        
        # Stress factors
        self.stress_factors = ["relaxed", "mild", "moderate", "high"]
        self.current_stress = "relaxed"
        
    def _calculate_base_params(self) -> Dict[str, float]:
        """Calculate base physiological parameters based on user profile"""
        age = self.user_profile.get("age", 30)
        gender = self.user_profile.get("gender", "female")
        fitness = self.user_profile.get("fitness_level", "moderate")
        
        # Base heart rate calculations
        max_hr = 220 - age
        if fitness == "high":
            resting_hr = max(50, 70 - age * 0.2)
        elif fitness == "moderate":
            resting_hr = max(60, 75 - age * 0.1)
        else:
            resting_hr = max(70, 80 - age * 0.05)
            
        # Gender adjustments
        if gender == "male":
            resting_hr -= 3
        
        return {
            "max_hr": max_hr,
            "resting_hr": resting_hr,
            "hrv_baseline": 35 if fitness == "high" else 25 if fitness == "moderate" else 18,
            "spo2_baseline": 98.5,
            "temp_baseline": 36.6,
            "systolic_bp": 120,
            "diastolic_bp": 80
        }
    
    def generate_real_time_data(self) -> Dict[str, Any]:
        """Generate real-time sensor data"""
        now = datetime.now()
        time_factor = self._get_time_factor(now)
        activity_factor = self._get_activity_factor()
        stress_factor = self._get_stress_factor()
        
        # Generate ECG data
        ecg_data = self._generate_ecg_data(time_factor, activity_factor, stress_factor)
        
        # Generate PPG data
        ppg_data = self._generate_ppg_data(time_factor, activity_factor, stress_factor)
        
        # Generate vital signs
        vitals = self._generate_vitals(time_factor, activity_factor, stress_factor)
        
        # Generate motion data
        motion = self._generate_motion_data(activity_factor)
        
        # Generate environmental data
        environment = self._generate_environment_data()
        
        # Create the main data structure
        data = {
            "timestamp": now.isoformat(),
            "device_id": "openwearables_sim_001",
            "ecg": ecg_data,
            "ppg": ppg_data,
            "vitals": vitals,
            "motion": motion,
            "environment": environment,
            "quality_metrics": self._generate_quality_metrics(),
            "metadata": {
                "activity": self.current_activity,
                "stress_level": self.current_stress,
                "confidence": random.uniform(0.85, 0.98)
            }
        }
        
        # Add flat sensor keys for compatibility with tests and legacy code
        data["accelerometer"] = motion["accelerometer"]
        data["gyroscope"] = motion["gyroscope"]
        data["temperature"] = vitals["temperature"]
        
        return data
    
    def _generate_ecg_data(self, time_factor: float, activity_factor: float, stress_factor: float) -> Dict[str, Any]:
        """Generate realistic ECG data"""
        sampling_rate = 250  # Hz
        duration = 1.0  # seconds
        samples = int(sampling_rate * duration)
        
        # Base heart rate
        base_hr = self.base_params["resting_hr"]
        current_hr = base_hr * (1 + activity_factor * 0.7 + stress_factor * 0.3 + time_factor * 0.1)
        current_hr = max(40, min(200, current_hr))
        
        # Generate ECG waveform
        t = np.linspace(0, duration, samples)
        rr_interval = 60 / current_hr
        
        ecg_signal = []
        for i, time_point in enumerate(t):
            # P wave, QRS complex, T wave simulation
            beat_phase = (time_point % rr_interval) / rr_interval
            
            if 0.1 <= beat_phase <= 0.2:  # P wave
                amplitude = 0.1 * np.sin((beat_phase - 0.1) * 10 * np.pi)
            elif 0.3 <= beat_phase <= 0.5:  # QRS complex
                if 0.35 <= beat_phase <= 0.45:
                    amplitude = 1.0 * np.sin((beat_phase - 0.35) * 10 * np.pi)
                else:
                    amplitude = -0.3 * np.sin((beat_phase - 0.3) * 20 * np.pi)
            elif 0.6 <= beat_phase <= 0.8:  # T wave
                amplitude = 0.3 * np.sin((beat_phase - 0.6) * 5 * np.pi)
            else:
                amplitude = 0
            
            # Add noise and artifacts
            noise = random.gauss(0, 0.05)
            if random.random() < 0.01:  # Motion artifacts
                noise += random.gauss(0, 0.2)
                
            ecg_signal.append(amplitude + noise)
        
        # Calculate metrics
        rr_intervals = self._detect_rr_intervals(ecg_signal, sampling_rate)
        hrv_metrics = self._calculate_hrv_metrics(rr_intervals)
        
        return {
            "signal": ecg_signal[:50],  # Send only recent samples for real-time
            "sampling_rate": sampling_rate,
            "heart_rate": round(current_hr, 1),
            "rr_intervals": rr_intervals[-10:],  # Last 10 intervals
            "hrv_metrics": hrv_metrics,
            "signal_quality": random.uniform(0.8, 0.98),
            "lead": "Lead I"
        }
    
    def _generate_ppg_data(self, time_factor: float, activity_factor: float, stress_factor: float) -> Dict[str, Any]:
        """Generate realistic PPG data"""
        sampling_rate = 100  # Hz
        duration = 1.0
        samples = int(sampling_rate * duration)
        
        base_hr = self.base_params["resting_hr"]
        current_hr = base_hr * (1 + activity_factor * 0.7 + stress_factor * 0.3 + time_factor * 0.1)
        
        # Generate PPG waveform
        t = np.linspace(0, duration, samples)
        rr_interval = 60 / current_hr
        
        ppg_signal = []
        for time_point in t:
            beat_phase = (time_point % rr_interval) / rr_interval
            
            # Systolic peak
            if 0.2 <= beat_phase <= 0.4:
                amplitude = np.sin((beat_phase - 0.2) * 5 * np.pi)
            # Dicrotic notch
            elif 0.6 <= beat_phase <= 0.8:
                amplitude = 0.3 * np.sin((beat_phase - 0.6) * 5 * np.pi)
            else:
                amplitude = 0
            
            # Add baseline and noise
            baseline = 2.0
            noise = random.gauss(0, 0.02)
            if activity_factor > 0.5:  # Motion artifacts during activity
                noise += random.gauss(0, 0.1)
            
            ppg_signal.append(baseline + amplitude + noise)
        
        # Calculate SpO2
        spo2 = self._calculate_spo2(ppg_signal, activity_factor, stress_factor)
        
        return {
            "signal": ppg_signal[-25:],  # Recent samples
            "sampling_rate": sampling_rate,
            "heart_rate": round(current_hr, 1),
            "spo2": round(spo2, 1),
            "perfusion_index": round(random.uniform(1.5, 8.0), 2),
            "signal_quality": random.uniform(0.75, 0.95)
        }
    
    def _generate_vitals(self, time_factor: float, activity_factor: float, stress_factor: float) -> Dict[str, Any]:
        """Generate vital signs"""
        # Temperature
        base_temp = self.base_params["temp_baseline"]
        temp_variation = time_factor * 0.5 + activity_factor * 0.8 + stress_factor * 0.3
        current_temp = base_temp + temp_variation + random.gauss(0, 0.1)
        
        # Blood pressure
        systolic = self.base_params["systolic_bp"] + activity_factor * 20 + stress_factor * 15
        diastolic = self.base_params["diastolic_bp"] + activity_factor * 10 + stress_factor * 8
        
        # Respiratory rate
        base_rr = 16
        resp_rate = base_rr + activity_factor * 8 + stress_factor * 4 + random.gauss(0, 1)
        
        return {
            "temperature": round(current_temp, 1),
            "blood_pressure": {
                "systolic": round(systolic),
                "diastolic": round(diastolic)
            },
            "respiratory_rate": round(max(8, resp_rate)),
            "skin_conductance": round(random.uniform(2.0, 15.0) * (1 + stress_factor), 2)
        }
    
    def _generate_motion_data(self, activity_factor: float) -> Dict[str, Any]:
        """Generate motion sensor data"""
        # Base accelerometer values (gravity component)
        base_accel = [0.0, 0.0, 9.81]
        
        # Add movement based on activity
        if activity_factor > 0.1:
            movement_intensity = activity_factor * 5
            accel_x = random.gauss(0, movement_intensity)
            accel_y = random.gauss(0, movement_intensity)
            accel_z = 9.81 + random.gauss(0, movement_intensity * 0.5)
        else:
            accel_x = random.gauss(0, 0.1)
            accel_y = random.gauss(0, 0.1)
            accel_z = 9.81 + random.gauss(0, 0.1)
        
        # Gyroscope data
        gyro_intensity = activity_factor * 100
        gyro = [
            random.gauss(0, gyro_intensity),
            random.gauss(0, gyro_intensity),
            random.gauss(0, gyro_intensity)
        ]
        
        return {
            "accelerometer": {
                "x": round(accel_x, 3),
                "y": round(accel_y, 3),
                "z": round(accel_z, 3),
                "magnitude": round(math.sqrt(accel_x**2 + accel_y**2 + accel_z**2), 3)
            },
            "gyroscope": {
                "x": round(gyro[0], 3),
                "y": round(gyro[1], 3),
                "z": round(gyro[2], 3)
            },
            "step_count": self._generate_step_count(activity_factor),
            "activity_type": self.current_activity
        }
    
    def _generate_environment_data(self) -> Dict[str, Any]:
        """Generate environmental sensor data"""
        return {
            "ambient_temperature": round(random.uniform(18, 25), 1),
            "humidity": round(random.uniform(40, 65), 1),
            "pressure": round(random.uniform(1010, 1020), 1),
            "light_level": round(random.uniform(100, 1000), 0),
            "uv_index": round(random.uniform(0, 5), 1),
            "air_quality": {
                "pm25": round(random.uniform(5, 25), 1),
                "pm10": round(random.uniform(10, 50), 1),
                "co2": round(random.uniform(400, 800), 0)
            }
        }
    
    def _generate_quality_metrics(self) -> Dict[str, float]:
        """Generate signal quality metrics"""
        return {
            "overall_quality": round(random.uniform(0.8, 0.98), 3),
            "ecg_quality": round(random.uniform(0.85, 0.99), 3),
            "ppg_quality": round(random.uniform(0.75, 0.95), 3),
            "motion_artifacts": round(random.uniform(0.0, 0.15), 3),
            "signal_noise_ratio": round(random.uniform(15, 35), 1)
        }
    
    def _get_time_factor(self, current_time: datetime) -> float:
        """Get circadian rhythm factor"""
        hour = current_time.hour
        
        # Simulate circadian rhythm
        if 6 <= hour <= 12:  # Morning
            return 0.3
        elif 12 <= hour <= 18:  # Afternoon
            return 0.5
        elif 18 <= hour <= 22:  # Evening
            return 0.2
        else:  # Night
            return -0.3
    
    def _get_activity_factor(self) -> float:
        """Get current activity intensity factor"""
        # Simulate changing activity levels
        if random.random() < 0.05:  # 5% chance to change activity
            self.current_activity = random.choice(self.activity_states)
        
        activity_factors = {
            "sleeping": -0.3,
            "resting": 0.0,
            "light": 0.3,
            "moderate": 0.6,
            "vigorous": 1.0
        }
        
        return activity_factors.get(self.current_activity, 0.0)
    
    def _get_stress_factor(self) -> float:
        """Get current stress level factor"""
        # Simulate changing stress levels
        if random.random() < 0.02:  # 2% chance to change stress
            self.current_stress = random.choice(self.stress_factors)
        
        stress_factors = {
            "relaxed": 0.0,
            "mild": 0.2,
            "moderate": 0.5,
            "high": 0.8
        }
        
        return stress_factors.get(self.current_stress, 0.0)
    
    def _detect_rr_intervals(self, ecg_signal: List[float], sampling_rate: int) -> List[float]:
        """Detect R-R intervals from ECG signal"""
        # Simplified R-peak detection
        threshold = max(ecg_signal) * 0.6
        peaks = []
        
        for i in range(1, len(ecg_signal) - 1):
            if (ecg_signal[i] > threshold and 
                ecg_signal[i] > ecg_signal[i-1] and 
                ecg_signal[i] > ecg_signal[i+1]):
                peaks.append(i)
        
        # Calculate R-R intervals in milliseconds
        rr_intervals = []
        for i in range(1, len(peaks)):
            interval = (peaks[i] - peaks[i-1]) / sampling_rate * 1000
            if 300 <= interval <= 2000:  # Valid physiological range
                rr_intervals.append(interval)
        
        return rr_intervals[-20:]  # Return last 20 intervals
    
    def _calculate_hrv_metrics(self, rr_intervals: List[float]) -> Dict[str, float]:
        """Calculate HRV metrics from R-R intervals"""
        if len(rr_intervals) < 5:
            return {"rmssd": 0, "sdnn": 0, "pnn50": 0}
        
        # RMSSD
        successive_diffs = [abs(rr_intervals[i] - rr_intervals[i-1]) 
                           for i in range(1, len(rr_intervals))]
        rmssd = math.sqrt(sum(d**2 for d in successive_diffs) / len(successive_diffs))
        
        # SDNN
        mean_rr = sum(rr_intervals) / len(rr_intervals)
        sdnn = math.sqrt(sum((rr - mean_rr)**2 for rr in rr_intervals) / len(rr_intervals))
        
        # pNN50
        nn50_count = sum(1 for diff in successive_diffs if diff > 50)
        pnn50 = (nn50_count / len(successive_diffs)) * 100 if successive_diffs else 0
        
        return {
            "rmssd": round(rmssd, 1),
            "sdnn": round(sdnn, 1),
            "pnn50": round(pnn50, 1)
        }
    
    def _calculate_spo2(self, ppg_signal: List[float], activity_factor: float, stress_factor: float) -> float:
        """Calculate SpO2 from PPG signal"""
        base_spo2 = self.base_params["spo2_baseline"]
        
        # Activity and stress can slightly affect SpO2
        spo2_adjustment = -activity_factor * 1.5 - stress_factor * 0.5
        
        # Add some realistic variation
        spo2 = base_spo2 + spo2_adjustment + random.gauss(0, 0.3)
        
        # Clamp to realistic range
        return max(85, min(100, spo2))
    
    def _generate_step_count(self, activity_factor: float) -> int:
        """Generate step count based on activity"""
        if activity_factor <= 0:
            return 0
        
        base_steps_per_second = {
            "light": 1.5,
            "moderate": 2.5,
            "vigorous": 3.5
        }
        
        steps_rate = base_steps_per_second.get(self.current_activity, 0)
        return int(steps_rate + random.gauss(0, 0.5)) if steps_rate > 0 else 0
    
    def generate_historical_data(self, days: int = 7) -> List[Dict[str, Any]]:
        """Generate historical data for the specified number of days"""
        historical_data = []
        current_time = datetime.now() - timedelta(days=days)
        
        while current_time <= datetime.now():
            # Generate data every 5 minutes for historical data
            data_point = self.generate_real_time_data()
            data_point["timestamp"] = current_time.isoformat()
            
            historical_data.append(data_point)
            current_time += timedelta(minutes=5)
        
        return historical_data
    
    def generate_health_insights(self) -> List[Dict[str, Any]]:
        """Generate AI-powered health insights"""
        insights = [
            {
                "id": "insight_001",
                "type": "recommendation",
                "title": "Heart Rate Variability Improvement",
                "message": "Your HRV has improved by 12% this week. Consider maintaining your current exercise routine.",
                "priority": "medium",
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.87,
                "data_sources": ["ecg", "activity"],
                "actionable": True,
                "action_items": [
                    "Continue current exercise schedule",
                    "Maintain consistent sleep patterns",
                    "Consider stress management techniques"
                ]
            },
            {
                "id": "insight_002",
                "type": "alert",
                "title": "Elevated Resting Heart Rate",
                "message": "Your resting heart rate has been 8% higher than usual over the past 3 days.",
                "priority": "medium",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "confidence": 0.92,
                "data_sources": ["ecg", "ppg"],
                "actionable": True,
                "action_items": [
                    "Ensure adequate hydration",
                    "Check if you're getting enough sleep",
                    "Monitor for signs of illness"
                ]
            },
            {
                "id": "insight_003",
                "type": "pattern",
                "title": "Sleep Quality Correlation",
                "message": "Better sleep quality correlates with improved HRV the following day.",
                "priority": "low",
                "timestamp": (datetime.now() - timedelta(hours=8)).isoformat(),
                "confidence": 0.78,
                "data_sources": ["ecg", "motion", "environment"],
                "actionable": True,
                "action_items": [
                    "Maintain consistent bedtime",
                    "Optimize bedroom temperature",
                    "Limit screen time before sleep"
                ]
            }
        ]
        
        return insights
    
    def generate_device_status(self) -> Dict[str, Any]:
        """Generate device status information"""
        return {
            "device_id": "openwearables_sim_001",
            "status": "connected",
            "battery_level": random.randint(45, 95),
            "signal_strength": random.randint(70, 100),
            "firmware_version": "1.2.3",
            "last_sync": datetime.now().isoformat(),
            "sensors": {
                "ecg": {"status": "active", "last_reading": datetime.now().isoformat()},
                "ppg": {"status": "active", "last_reading": datetime.now().isoformat()},
                "accelerometer": {"status": "active", "last_reading": datetime.now().isoformat()},
                "gyroscope": {"status": "active", "last_reading": datetime.now().isoformat()},
                "temperature": {"status": "active", "last_reading": datetime.now().isoformat()}
            },
            "data_quality": {
                "overall": random.uniform(0.85, 0.98),
                "recent_artifacts": random.randint(0, 3)
            },
            "storage": {
                "used_space": f"{random.randint(1024, 8192)}MB",
                "total_space": "16GB",
                "usage_percent": random.randint(10, 85)
            }
        }

# Global instance for easy access
mock_generator = MockDataGenerator()

def get_mock_data() -> Dict[str, Any]:
    """Get current mock data"""
    return mock_generator.generate_real_time_data()

def get_mock_historical_data(days: int = 7) -> List[Dict[str, Any]]:
    """Get mock historical data"""
    return mock_generator.generate_historical_data(days)

def get_mock_insights() -> List[Dict[str, Any]]:
    """Get mock health insights"""
    return mock_generator.generate_health_insights()

def get_mock_device_status() -> Dict[str, Any]:
    """Get mock device status"""
    return mock_generator.generate_device_status() 