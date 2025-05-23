"""
Advanced Analytics Module for OpenWearables Platform

Provides sophisticated health analytics, pattern recognition, predictive modeling,
and comprehensive health insights using advanced statistical methods and machine learning.
"""

import time
import logging
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger("OpenWearables.AdvancedAnalytics")

@dataclass
class HealthPattern:
    """Data structure for identified health patterns."""
    pattern_id: str
    pattern_type: str  # circadian, weekly, seasonal, anomaly
    confidence: float
    start_time: float
    end_time: float
    description: str
    affected_metrics: List[str]
    severity: str  # low, medium, high, critical
    recommended_actions: List[str]

@dataclass
class PredictiveInsight:
    """Data structure for predictive health insights."""
    insight_id: str
    prediction_type: str
    target_metric: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_horizon_hours: int
    confidence_score: float
    risk_factors: List[str]
    preventive_measures: List[str]

@dataclass
class CorrelationAnalysis:
    """Data structure for correlation analysis results."""
    metric_pair: Tuple[str, str]
    correlation_coefficient: float
    statistical_significance: float
    relationship_type: str  # positive, negative, non_linear
    strength: str  # weak, moderate, strong, very_strong
    clinical_relevance: str
    
@dataclass
class TrendAnalysis:
    """Data structure for trend analysis results."""
    metric_name: str
    trend_direction: str  # increasing, decreasing, stable, volatile
    trend_strength: float
    trend_duration_days: int
    seasonal_component: Optional[float]
    changepoint_timestamps: List[float]
    forecast_values: List[float]
    forecast_confidence: List[Tuple[float, float]]


class CircadianRhythmAnalyzer:
    """Advanced circadian rhythm analysis and optimization."""
    
    def __init__(self):
        """Initialize circadian rhythm analyzer."""
        self.hr_data_buffer = deque(maxlen=10080)  # 7 days of minute data
        self.activity_data_buffer = deque(maxlen=10080)
        self.sleep_data_buffer = deque(maxlen=10080)
        self.temperature_data_buffer = deque(maxlen=10080)
        
        # Circadian parameters
        self.circadian_period = 24.0  # hours
        self.phase_shift_threshold = 1.0  # hours
        self.amplitude_threshold = 0.15  # normalized
        
    def add_data_point(self, timestamp: float, heart_rate: float, 
                      activity_level: float, sleep_stage: str, 
                      body_temperature: Optional[float] = None):
        """
        Add data point for circadian analysis.
        
        Args:
            timestamp: Unix timestamp
            heart_rate: Heart rate in BPM
            activity_level: Normalized activity level (0-1)
            sleep_stage: Sleep stage classification
            body_temperature: Body temperature in Celsius
        """
        # Convert sleep stage to numeric
        sleep_numeric = {
            "awake": 1.0,
            "light": 0.3,
            "deep": 0.1,
            "rem": 0.5
        }.get(sleep_stage, 0.0)
        
        self.hr_data_buffer.append((timestamp, heart_rate))
        self.activity_data_buffer.append((timestamp, activity_level))
        self.sleep_data_buffer.append((timestamp, sleep_numeric))
        
        if body_temperature:
            self.temperature_data_buffer.append((timestamp, body_temperature))
    
    def analyze_circadian_rhythm(self) -> Dict[str, Any]:
        """
        Analyze circadian rhythm patterns.
        
        Returns:
            Comprehensive circadian rhythm analysis
        """
        if len(self.hr_data_buffer) < 1440:  # Need at least 24 hours
            return {"status": "insufficient_data"}
        
        # Extract time series data
        hr_times, hr_values = zip(*list(self.hr_data_buffer))
        activity_times, activity_values = zip(*list(self.activity_data_buffer))
        sleep_times, sleep_values = zip(*list(self.sleep_data_buffer))
        
        # Convert to hours of day
        hr_hours = [(t % 86400) / 3600 for t in hr_times]
        activity_hours = [(t % 86400) / 3600 for t in activity_times]
        sleep_hours = [(t % 86400) / 3600 for t in sleep_times]
        
        # Perform circadian analysis
        hr_rhythm = self._extract_circadian_component(hr_hours, hr_values)
        activity_rhythm = self._extract_circadian_component(activity_hours, activity_values)
        sleep_rhythm = self._extract_circadian_component(sleep_hours, sleep_values)
        
        # Calculate phase coherence
        phase_coherence = self._calculate_phase_coherence([hr_rhythm, activity_rhythm, sleep_rhythm])
        
        # Detect phase shifts
        phase_shifts = self._detect_phase_shifts(hr_rhythm)
        
        # Calculate circadian strength
        circadian_strength = self._calculate_circadian_strength(hr_rhythm, activity_rhythm, sleep_rhythm)
        
        return {
            "circadian_strength": circadian_strength,
            "phase_coherence": phase_coherence,
            "peak_activity_time": self._find_peak_time(activity_rhythm),
            "lowest_hr_time": self._find_trough_time(hr_rhythm),
            "sleep_onset_consistency": self._calculate_sleep_consistency(sleep_rhythm),
            "phase_shifts_detected": len(phase_shifts),
            "rhythm_quality": self._assess_rhythm_quality(circadian_strength, phase_coherence),
            "recommendations": self._generate_circadian_recommendations(circadian_strength, phase_coherence, phase_shifts)
        }
    
    def _extract_circadian_component(self, hours: List[float], values: List[float]) -> Dict[str, float]:
        """Extract circadian rhythm components using harmonic analysis."""
        hours_array = np.array(hours)
        values_array = np.array(values)
        
        # Fit sinusoidal model: y = A*sin(2π(t-φ)/T) + C
        # Where T=24 hours (circadian period)
        omega = 2 * np.pi / 24.0  # Angular frequency for 24-hour cycle
        
        # Create design matrix for least squares fitting
        X = np.column_stack([
            np.sin(omega * hours_array),
            np.cos(omega * hours_array),
            np.ones(len(hours_array))
        ])
        
        # Solve for coefficients
        try:
            coeffs = np.linalg.lstsq(X, values_array, rcond=None)[0]
            a, b, c = coeffs
            
            # Calculate amplitude and phase
            amplitude = np.sqrt(a**2 + b**2)
            phase = np.arctan2(b, a) * 12 / np.pi  # Convert to hours
            
            return {
                "amplitude": float(amplitude),
                "phase": float(phase),
                "mesor": float(c),  # Mean level
                "acrophase": float((phase + 6) % 24)  # Peak time
            }
        except np.linalg.LinAlgError:
            return {"amplitude": 0.0, "phase": 0.0, "mesor": np.mean(values_array), "acrophase": 12.0}
    
    def _calculate_phase_coherence(self, rhythms: List[Dict[str, float]]) -> float:
        """Calculate phase coherence between different rhythms."""
        if len(rhythms) < 2:
            return 0.0
        
        phases = [r.get("phase", 0.0) for r in rhythms]
        
        # Calculate circular variance of phases
        phase_vectors = [np.exp(1j * np.pi * p / 12) for p in phases]
        mean_vector = np.mean(phase_vectors)
        coherence = abs(mean_vector)
        
        return float(coherence)
    
    def _detect_phase_shifts(self, rhythm: Dict[str, float]) -> List[float]:
        """Detect circadian phase shifts."""
        # Simplified phase shift detection
        # In practice, this would use more sophisticated change-point detection
        if len(self.hr_data_buffer) < 2880:  # Need at least 48 hours
            return []
        
        # Look for significant changes in acrophase over time
        phase_shifts = []
        current_phase = rhythm.get("acrophase", 12.0)
        
        # Check if phase is significantly different from expected
        expected_phase = 14.0  # Expected peak activity around 2 PM
        if abs(current_phase - expected_phase) > self.phase_shift_threshold:
            phase_shifts.append(time.time())
        
        return phase_shifts
    
    def _calculate_circadian_strength(self, hr_rhythm: Dict, activity_rhythm: Dict, sleep_rhythm: Dict) -> float:
        """Calculate overall circadian rhythm strength."""
        hr_strength = hr_rhythm.get("amplitude", 0.0) / max(hr_rhythm.get("mesor", 1.0), 1.0)
        activity_strength = activity_rhythm.get("amplitude", 0.0)
        sleep_strength = sleep_rhythm.get("amplitude", 0.0)
        
        # Weighted average
        weights = [0.4, 0.4, 0.2]  # HR and activity more important
        strengths = [hr_strength, activity_strength, sleep_strength]
        
        overall_strength = sum(w * s for w, s in zip(weights, strengths))
        return float(min(1.0, max(0.0, overall_strength)))
    
    def _find_peak_time(self, rhythm: Dict[str, float]) -> float:
        """Find peak time for rhythm."""
        return rhythm.get("acrophase", 12.0)
    
    def _find_trough_time(self, rhythm: Dict[str, float]) -> float:
        """Find trough time for rhythm."""
        peak_time = rhythm.get("acrophase", 12.0)
        return (peak_time + 12) % 24
    
    def _calculate_sleep_consistency(self, sleep_rhythm: Dict[str, float]) -> float:
        """Calculate sleep onset consistency."""
        # Simplified calculation based on amplitude
        amplitude = sleep_rhythm.get("amplitude", 0.0)
        return float(min(1.0, amplitude * 2))  # Higher amplitude = more consistent
    
    def _assess_rhythm_quality(self, strength: float, coherence: float) -> str:
        """Assess overall circadian rhythm quality."""
        combined_score = (strength + coherence) / 2
        
        if combined_score >= 0.8:
            return "excellent"
        elif combined_score >= 0.6:
            return "good"
        elif combined_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _generate_circadian_recommendations(self, strength: float, coherence: float, phase_shifts: List) -> List[str]:
        """Generate personalized circadian rhythm recommendations."""
        recommendations = []
        
        if strength < 0.5:
            recommendations.extend([
                "Maintain consistent sleep and wake times",
                "Increase morning light exposure",
                "Avoid screens 2 hours before bedtime",
                "Consider timed meals to strengthen circadian rhythms"
            ])
        
        if coherence < 0.6:
            recommendations.extend([
                "Synchronize daily activities with natural light cycles",
                "Establish consistent meal timing",
                "Implement regular exercise schedule"
            ])
        
        if len(phase_shifts) > 0:
            recommendations.extend([
                "Gradually adjust sleep schedule to align with natural rhythms",
                "Use light therapy for circadian realignment",
                "Avoid large meals and caffeine late in the day"
            ])
        
        if not recommendations:
            recommendations.append("Excellent circadian rhythm detected - maintain current lifestyle patterns")
        
        return recommendations


class StressPatternAnalyzer:
    """Advanced stress pattern analysis and management insights."""
    
    def __init__(self):
        """Initialize stress pattern analyzer."""
        self.stress_buffer = deque(maxlen=4320)  # 3 days of minute data
        self.hrv_buffer = deque(maxlen=4320)
        self.activity_buffer = deque(maxlen=4320)
        self.environmental_buffer = deque(maxlen=4320)
        
        # Stress analysis parameters
        self.acute_stress_threshold = 0.7
        self.chronic_stress_window = 7 * 24 * 60  # 7 days in minutes
        self.recovery_threshold = 0.3
        
    def add_stress_data(self, timestamp: float, stress_level: float, hrv: float, 
                       activity_level: float, environmental_stress: float):
        """
        Add stress-related data point.
        
        Args:
            timestamp: Unix timestamp
            stress_level: Normalized stress level (0-1)
            hrv: Heart rate variability in ms
            activity_level: Normalized activity level (0-1)
            environmental_stress: Environmental stress factors (0-1)
        """
        self.stress_buffer.append((timestamp, stress_level))
        self.hrv_buffer.append((timestamp, hrv))
        self.activity_buffer.append((timestamp, activity_level))
        self.environmental_buffer.append((timestamp, environmental_stress))
    
    def analyze_stress_patterns(self) -> Dict[str, Any]:
        """
        Perform comprehensive stress pattern analysis.
        
        Returns:
            Detailed stress analysis with patterns and recommendations
        """
        if len(self.stress_buffer) < 60:  # Need at least 1 hour of data
            return {"status": "insufficient_data"}
        
        # Extract data arrays
        stress_times, stress_values = zip(*list(self.stress_buffer))
        hrv_times, hrv_values = zip(*list(self.hrv_buffer))
        
        # Analyze stress patterns
        acute_episodes = self._detect_acute_stress_episodes(stress_times, stress_values)
        chronic_stress_level = self._calculate_chronic_stress(stress_values)
        stress_triggers = self._identify_stress_triggers()
        recovery_patterns = self._analyze_recovery_patterns(stress_times, stress_values, hrv_values)
        stress_resilience = self._calculate_stress_resilience(stress_values, hrv_values)
        
        # Generate insights
        daily_patterns = self._analyze_daily_stress_patterns(stress_times, stress_values)
        weekly_patterns = self._analyze_weekly_stress_patterns(stress_times, stress_values)
        
        return {
            "acute_stress_episodes": len(acute_episodes),
            "chronic_stress_level": chronic_stress_level,
            "stress_resilience_score": stress_resilience,
            "primary_stress_triggers": stress_triggers,
            "recovery_efficiency": recovery_patterns["efficiency"],
            "daily_stress_patterns": daily_patterns,
            "weekly_stress_patterns": weekly_patterns,
            "stress_management_score": self._calculate_overall_stress_score(chronic_stress_level, stress_resilience),
            "personalized_recommendations": self._generate_stress_management_recommendations(
                chronic_stress_level, stress_resilience, stress_triggers, recovery_patterns
            )
        }
    
    def _detect_acute_stress_episodes(self, times: List[float], stress_values: List[float]) -> List[Dict]:
        """Detect acute stress episodes."""
        episodes = []
        in_episode = False
        episode_start = None
        
        for i, (timestamp, stress) in enumerate(zip(times, stress_values)):
            if stress > self.acute_stress_threshold and not in_episode:
                in_episode = True
                episode_start = timestamp
            elif stress <= self.acute_stress_threshold and in_episode:
                in_episode = False
                if episode_start:
                    episodes.append({
                        "start_time": episode_start,
                        "end_time": timestamp,
                        "duration_minutes": (timestamp - episode_start) / 60,
                        "peak_stress": max(stress_values[max(0, i-10):i+1])
                    })
        
        return episodes
    
    def _calculate_chronic_stress(self, stress_values: List[float]) -> float:
        """Calculate chronic stress level."""
        # Use rolling average over longer periods
        if len(stress_values) < 60:
            return np.mean(stress_values)
        
        # Calculate 24-hour rolling averages
        daily_averages = []
        window_size = min(1440, len(stress_values) // 3)  # 24 hours or 1/3 of data
        
        for i in range(0, len(stress_values) - window_size + 1, window_size):
            daily_avg = np.mean(stress_values[i:i + window_size])
            daily_averages.append(daily_avg)
        
        return float(np.mean(daily_averages))
    
    def _identify_stress_triggers(self) -> List[str]:
        """Identify primary stress triggers based on patterns."""
        # Simplified trigger identification
        triggers = []
        
        if len(self.environmental_buffer) > 0:
            env_times, env_values = zip(*list(self.environmental_buffer))
            avg_env_stress = np.mean(env_values)
            
            if avg_env_stress > 0.6:
                triggers.append("environmental_factors")
        
        if len(self.activity_buffer) > 0:
            activity_times, activity_values = zip(*list(self.activity_buffer))
            
            # Check for stress during high activity
            stress_times, stress_values = zip(*list(self.stress_buffer))
            high_activity_stress = []
            
            for i, (stress_time, stress_val) in enumerate(zip(stress_times, stress_values)):
                # Find corresponding activity level
                for act_time, act_val in zip(activity_times, activity_values):
                    if abs(stress_time - act_time) < 300:  # Within 5 minutes
                        if act_val > 0.7 and stress_val > 0.6:
                            high_activity_stress.append(stress_val)
                        break
            
            if len(high_activity_stress) > 5 and np.mean(high_activity_stress) > 0.6:
                triggers.append("high_intensity_activity")
        
        # Analyze time-based patterns
        stress_times, stress_values = zip(*list(self.stress_buffer))
        hourly_stress = self._calculate_hourly_averages(stress_times, stress_values)
        
        # Check for work hours stress (9 AM - 5 PM)
        work_hours_stress = [hourly_stress[h] for h in range(9, 17) if h in hourly_stress]
        if work_hours_stress and np.mean(work_hours_stress) > 0.6:
            triggers.append("work_related_stress")
        
        # Check for evening stress
        evening_stress = [hourly_stress[h] for h in range(18, 23) if h in hourly_stress]
        if evening_stress and np.mean(evening_stress) > 0.6:
            triggers.append("evening_overstimulation")
        
        return triggers if triggers else ["general_lifestyle_factors"]
    
    def _analyze_recovery_patterns(self, stress_times: List[float], stress_values: List[float], hrv_values: List[float]) -> Dict:
        """Analyze stress recovery patterns."""
        recovery_times = []
        
        # Find stress recovery sequences
        for i in range(1, len(stress_values)):
            if stress_values[i-1] > 0.6 and stress_values[i] < 0.4:
                # Found potential recovery
                recovery_start = i
                recovery_time = 0
                
                # Calculate recovery duration
                for j in range(i, min(i + 60, len(stress_values))):
                    if stress_values[j] < self.recovery_threshold:
                        recovery_time = (stress_times[j] - stress_times[i]) / 60  # Minutes
                        break
                
                if recovery_time > 0:
                    recovery_times.append(recovery_time)
        
        if recovery_times:
            avg_recovery_time = np.mean(recovery_times)
            recovery_efficiency = max(0, 1 - (avg_recovery_time / 60))  # Normalize to 1 hour
        else:
            avg_recovery_time = 60  # Default to 1 hour
            recovery_efficiency = 0.5
        
        return {
            "average_recovery_time_minutes": float(avg_recovery_time),
            "efficiency": float(recovery_efficiency),
            "recovery_episodes": len(recovery_times)
        }
    
    def _calculate_stress_resilience(self, stress_values: List[float], hrv_values: List[float]) -> float:
        """Calculate stress resilience score."""
        # Resilience based on stress variability and HRV
        stress_variability = np.std(stress_values)
        avg_hrv = np.mean(hrv_values)
        
        # Lower stress variability and higher HRV indicate better resilience
        resilience_score = (1 - min(1, stress_variability)) * 0.6 + min(1, avg_hrv / 50) * 0.4
        
        return float(max(0, min(1, resilience_score)))
    
    def _analyze_daily_stress_patterns(self, times: List[float], stress_values: List[float]) -> Dict:
        """Analyze daily stress patterns."""
        hourly_averages = self._calculate_hourly_averages(times, stress_values)
        
        # Find peak and low stress times
        if hourly_averages:
            peak_hour = max(hourly_averages.keys(), key=lambda h: hourly_averages[h])
            low_hour = min(hourly_averages.keys(), key=lambda h: hourly_averages[h])
            
            return {
                "peak_stress_hour": peak_hour,
                "lowest_stress_hour": low_hour,
                "morning_average": np.mean([hourly_averages[h] for h in range(6, 12) if h in hourly_averages]),
                "afternoon_average": np.mean([hourly_averages[h] for h in range(12, 18) if h in hourly_averages]),
                "evening_average": np.mean([hourly_averages[h] for h in range(18, 24) if h in hourly_averages])
            }
        
        return {"status": "insufficient_data"}
    
    def _analyze_weekly_stress_patterns(self, times: List[float], stress_values: List[float]) -> Dict:
        """Analyze weekly stress patterns."""
        if len(times) < 1440:  # Need at least 24 hours
            return {"status": "insufficient_data"}
        
        # Group by day of week
        daily_averages = {}
        for timestamp, stress in zip(times, stress_values):
            day_of_week = int((timestamp // 86400) % 7)  # 0=Monday
            if day_of_week not in daily_averages:
                daily_averages[day_of_week] = []
            daily_averages[day_of_week].append(stress)
        
        # Calculate averages
        weekly_pattern = {}
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for day, values in daily_averages.items():
            weekly_pattern[day_names[day]] = float(np.mean(values))
        
        return weekly_pattern
    
    def _calculate_hourly_averages(self, times: List[float], values: List[float]) -> Dict[int, float]:
        """Calculate hourly averages for time series data."""
        hourly_data = {}
        
        for timestamp, value in zip(times, values):
            hour = int((timestamp % 86400) // 3600)
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append(value)
        
        return {hour: float(np.mean(vals)) for hour, vals in hourly_data.items()}
    
    def _calculate_overall_stress_score(self, chronic_stress: float, resilience: float) -> float:
        """Calculate overall stress management score."""
        # Lower chronic stress and higher resilience = better score
        stress_component = max(0, 1 - chronic_stress)
        
        overall_score = (stress_component * 0.6 + resilience * 0.4) * 100
        return float(overall_score)
    
    def _generate_stress_management_recommendations(self, chronic_stress: float, resilience: float, 
                                                  triggers: List[str], recovery_patterns: Dict) -> List[str]:
        """Generate personalized stress management recommendations."""
        recommendations = []
        
        # Chronic stress recommendations
        if chronic_stress > 0.6:
            recommendations.extend([
                "Implement daily stress reduction techniques such as meditation or deep breathing",
                "Consider professional stress management counseling",
                "Evaluate and modify high-stress lifestyle factors",
                "Establish better work-life boundaries"
            ])
        
        # Resilience recommendations
        if resilience < 0.5:
            recommendations.extend([
                "Practice progressive muscle relaxation techniques",
                "Engage in regular cardiovascular exercise to improve stress resilience",
                "Develop mindfulness and emotional regulation skills",
                "Ensure adequate sleep quality and duration"
            ])
        
        # Trigger-specific recommendations
        for trigger in triggers:
            if trigger == "work_related_stress":
                recommendations.append("Take regular breaks during work hours and practice desk-based relaxation exercises")
            elif trigger == "evening_overstimulation":
                recommendations.append("Establish calming evening routine with reduced screen time")
            elif trigger == "environmental_factors":
                recommendations.append("Optimize your environment to reduce external stressors")
            elif trigger == "high_intensity_activity":
                recommendations.append("Balance high-intensity activities with adequate recovery periods")
        
        # Recovery recommendations
        if recovery_patterns.get("efficiency", 0.5) < 0.4:
            recommendations.extend([
                "Practice active recovery techniques such as gentle stretching or yoga",
                "Develop personalized stress recovery protocols",
                "Consider biofeedback training to improve recovery efficiency"
            ])
        
        if not recommendations:
            recommendations.append("Excellent stress management detected - maintain current practices")
        
        return recommendations


class AdvancedAnalyticsEngine:
    """
    Main advanced analytics engine coordinating all analysis modules.
    
    Provides comprehensive health analytics, pattern recognition, and insights
    generation using sophisticated statistical and machine learning methods.
    """
    
    def __init__(self):
        """Initialize advanced analytics engine."""
        self.circadian_analyzer = CircadianRhythmAnalyzer()
        self.stress_analyzer = StressPatternAnalyzer()
        
        # Data storage for longitudinal analysis
        self.longitudinal_data = {}
        self.max_storage_days = 90
        
        # Analysis state
        self.last_analysis_time = 0
        self.analysis_interval = 3600  # 1 hour
        
        # Results storage
        self.analysis_results = deque(maxlen=100)
        self.health_patterns = deque(maxlen=50)
        self.predictive_insights = deque(maxlen=30)
        
        logger.info("Advanced Analytics Engine initialized")
    
    def process_device_data(self, device_type: str, sensor_data: Dict[str, Any], timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Process device data through advanced analytics pipeline.
        
        Args:
            device_type: Type of device (smart_watch, smart_glasses, etc.)
            sensor_data: Dictionary of sensor readings
            timestamp: Optional timestamp
            
        Returns:
            Advanced analytics results and insights
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Store longitudinal data
        self._store_longitudinal_data(device_type, sensor_data, timestamp)
        
        # Feed specialized analyzers
        if device_type == "smart_watch":
            self._process_watch_data(sensor_data, timestamp)
        elif device_type == "smart_glasses":
            self._process_glasses_data(sensor_data, timestamp)
        elif device_type == "smart_headphones":
            self._process_headphones_data(sensor_data, timestamp)
        
        # Perform periodic comprehensive analysis
        if timestamp - self.last_analysis_time > self.analysis_interval:
            analysis_results = self._perform_comprehensive_analysis(timestamp)
            self.last_analysis_time = timestamp
            return analysis_results
        
        return {"status": "data_processed", "timestamp": timestamp}
    
    def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health analytics report."""
        current_time = time.time()
        
        # Get analysis from specialized modules
        circadian_analysis = self.circadian_analyzer.analyze_circadian_rhythm()
        stress_analysis = self.stress_analyzer.analyze_stress_patterns()
        
        # Perform longitudinal trend analysis
        trend_analysis = self._analyze_longitudinal_trends()
        
        # Generate correlations
        correlation_analysis = self._analyze_metric_correlations()
        
        # Generate predictive insights
        predictive_insights = self._generate_predictive_insights()
        
        # Compile comprehensive report
        report = {
            "report_timestamp": current_time,
            "analysis_period_days": self._get_analysis_period_days(),
            "circadian_health": circadian_analysis,
            "stress_management": stress_analysis,
            "longitudinal_trends": trend_analysis,
            "metric_correlations": correlation_analysis,
            "predictive_insights": predictive_insights,
            "overall_health_score": self._calculate_overall_health_score(),
            "key_recommendations": self._generate_key_recommendations(),
            "risk_assessment": self._perform_risk_assessment(),
            "personalized_action_plan": self._generate_action_plan()
        }
        
        return report
    
    def _store_longitudinal_data(self, device_type: str, sensor_data: Dict[str, Any], timestamp: float):
        """Store data for longitudinal analysis."""
        if device_type not in self.longitudinal_data:
            self.longitudinal_data[device_type] = deque(maxlen=self.max_storage_days * 1440)  # 90 days of minute data
        
        data_point = {
            "timestamp": timestamp,
            "data": sensor_data.copy()
        }
        
        self.longitudinal_data[device_type].append(data_point)
    
    def _process_watch_data(self, sensor_data: Dict[str, Any], timestamp: float):
        """Process smart watch data for specialized analysis."""
        # Extract relevant metrics
        heart_rate = sensor_data.get("heart_rate", 70)
        activity_level = min(1.0, sensor_data.get("exercise_minutes", 0) / 60.0)
        sleep_stage = sensor_data.get("sleep_stage", "awake")
        stress_level = sensor_data.get("stress_level", 0.3)
        hrv = sensor_data.get("hrv", 40)
        
        # Feed circadian analyzer
        self.circadian_analyzer.add_data_point(
            timestamp, heart_rate, activity_level, sleep_stage
        )
        
        # Feed stress analyzer
        self.stress_analyzer.add_stress_data(
            timestamp, stress_level, hrv, activity_level, 0.2  # Default env stress
        )
    
    def _process_glasses_data(self, sensor_data: Dict[str, Any], timestamp: float):
        """Process smart glasses data for specialized analysis."""
        # Extract cognitive and environmental metrics
        stress_level = sensor_data.get("stress_level", 0.3)
        cognitive_load = sensor_data.get("cognitive_load", 0.4)
        environmental_stress = sensor_data.get("environmental_stress", 0.2)
        
        # Feed stress analyzer with environmental data
        self.stress_analyzer.add_stress_data(
            timestamp, stress_level, 40, 0.3, environmental_stress
        )
    
    def _process_headphones_data(self, sensor_data: Dict[str, Any], timestamp: float):
        """Process smart headphones data for specialized analysis."""
        # Extract audio and environmental metrics
        audio_exposure = sensor_data.get("audio_exposure_db", 60)
        environmental_noise = sensor_data.get("environmental_noise_db", 45)
        
        # Calculate environmental stress from noise levels
        env_stress = min(1.0, max(0.0, (environmental_noise - 40) / 40))
        
        # Feed relevant analyzers
        self.stress_analyzer.add_stress_data(
            timestamp, 0.3, 40, 0.3, env_stress
        )
    
    def _perform_comprehensive_analysis(self, timestamp: float) -> Dict[str, Any]:
        """Perform comprehensive periodic analysis."""
        # Run all analysis modules
        circadian_results = self.circadian_analyzer.analyze_circadian_rhythm()
        stress_results = self.stress_analyzer.analyze_stress_patterns()
        
        # Detect health patterns
        patterns = self._detect_health_patterns()
        
        # Store results
        analysis_result = {
            "timestamp": timestamp,
            "circadian_analysis": circadian_results,
            "stress_analysis": stress_results,
            "detected_patterns": patterns,
            "analysis_confidence": self._calculate_analysis_confidence()
        }
        
        self.analysis_results.append(analysis_result)
        
        return analysis_result
    
    def _analyze_longitudinal_trends(self) -> Dict[str, Any]:
        """Analyze longitudinal trends across all metrics."""
        trends = {}
        
        for device_type, data_deque in self.longitudinal_data.items():
            if len(data_deque) < 100:  # Need sufficient data
                continue
            
            # Extract time series for key metrics
            data_list = list(data_deque)
            timestamps = [d["timestamp"] for d in data_list]
            
            device_trends = {}
            
            # Analyze heart rate trends (if available)
            if device_type == "smart_watch":
                hr_values = [d["data"].get("heart_rate", 70) for d in data_list]
                device_trends["heart_rate"] = self._analyze_metric_trend(timestamps, hr_values, "heart_rate")
                
                # Analyze activity trends
                activity_values = [d["data"].get("exercise_minutes", 0) for d in data_list]
                device_trends["activity"] = self._analyze_metric_trend(timestamps, activity_values, "activity")
                
                # Analyze sleep quality trends
                sleep_values = [d["data"].get("sleep_quality", 70) for d in data_list]
                device_trends["sleep_quality"] = self._analyze_metric_trend(timestamps, sleep_values, "sleep_quality")
            
            trends[device_type] = device_trends
        
        return trends
    
    def _analyze_metric_trend(self, timestamps: List[float], values: List[float], metric_name: str) -> TrendAnalysis:
        """Analyze trend for a specific metric."""
        if len(values) < 20:
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction="insufficient_data",
                trend_strength=0.0,
                trend_duration_days=0,
                seasonal_component=None,
                changepoint_timestamps=[],
                forecast_values=[],
                forecast_confidence=[]
            )
        
        # Convert to numpy arrays
        times_array = np.array(timestamps)
        values_array = np.array(values)
        
        # Calculate linear trend
        time_normalized = (times_array - times_array[0]) / 86400  # Days since start
        
        try:
            slope, intercept = np.polyfit(time_normalized, values_array, 1)
            
            # Determine trend direction and strength
            if abs(slope) < 0.1:
                direction = "stable"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"
            
            trend_strength = min(1.0, abs(slope) / np.std(values_array))
            
            # Simple changepoint detection (look for significant changes)
            changepoints = []
            window_size = max(10, len(values) // 10)
            
            for i in range(window_size, len(values) - window_size):
                before_mean = np.mean(values_array[i-window_size:i])
                after_mean = np.mean(values_array[i:i+window_size])
                
                if abs(after_mean - before_mean) > 2 * np.std(values_array):
                    changepoints.append(timestamps[i])
            
            # Simple forecasting (linear extrapolation)
            forecast_days = 7
            future_times = np.arange(1, forecast_days + 1)
            last_day = time_normalized[-1]
            forecast_values = [slope * (last_day + day) + intercept for day in future_times]
            
            # Confidence intervals (simplified)
            std_error = np.std(values_array - (slope * time_normalized + intercept))
            forecast_confidence = [(val - 1.96 * std_error, val + 1.96 * std_error) for val in forecast_values]
            
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction=direction,
                trend_strength=float(trend_strength),
                trend_duration_days=int((timestamps[-1] - timestamps[0]) / 86400),
                seasonal_component=None,  # Could add seasonal decomposition
                changepoint_timestamps=changepoints,
                forecast_values=forecast_values,
                forecast_confidence=forecast_confidence
            )
            
        except np.linalg.LinAlgError:
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction="analysis_error",
                trend_strength=0.0,
                trend_duration_days=0,
                seasonal_component=None,
                changepoint_timestamps=[],
                forecast_values=[],
                forecast_confidence=[]
            )
    
    def _analyze_metric_correlations(self) -> List[CorrelationAnalysis]:
        """Analyze correlations between different health metrics."""
        correlations = []
        
        # Extract data from different devices
        watch_data = list(self.longitudinal_data.get("smart_watch", []))
        glasses_data = list(self.longitudinal_data.get("smart_glasses", []))
        
        if len(watch_data) < 50:  # Need sufficient data
            return correlations
        
        # Extract metric pairs for analysis
        hr_values = [d["data"].get("heart_rate", 70) for d in watch_data]
        activity_values = [d["data"].get("exercise_minutes", 0) for d in watch_data]
        sleep_values = [d["data"].get("sleep_quality", 70) for d in watch_data]
        
        # Analyze correlations
        metric_pairs = [
            ("heart_rate", "activity", hr_values, activity_values),
            ("heart_rate", "sleep_quality", hr_values, sleep_values),
            ("activity", "sleep_quality", activity_values, sleep_values)
        ]
        
        for metric1, metric2, values1, values2 in metric_pairs:
            if len(values1) == len(values2) and len(values1) > 10:
                correlation_coeff = np.corrcoef(values1, values2)[0, 1]
                
                # Determine relationship strength and type
                abs_corr = abs(correlation_coeff)
                if abs_corr >= 0.8:
                    strength = "very_strong"
                elif abs_corr >= 0.6:
                    strength = "strong"
                elif abs_corr >= 0.4:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                relationship_type = "positive" if correlation_coeff > 0 else "negative"
                
                # Clinical relevance (simplified)
                if (metric1, metric2) in [("heart_rate", "activity"), ("activity", "sleep_quality")]:
                    clinical_relevance = "high"
                else:
                    clinical_relevance = "moderate"
                
                correlations.append(CorrelationAnalysis(
                    metric_pair=(metric1, metric2),
                    correlation_coefficient=float(correlation_coeff),
                    statistical_significance=0.95 if abs_corr > 0.3 else 0.7,  # Simplified
                    relationship_type=relationship_type,
                    strength=strength,
                    clinical_relevance=clinical_relevance
                ))
        
        return correlations
    
    def _generate_predictive_insights(self) -> List[PredictiveInsight]:
        """Generate predictive health insights."""
        insights = []
        
        # Analyze recent trends to predict future values
        watch_data = list(self.longitudinal_data.get("smart_watch", []))
        
        if len(watch_data) < 100:
            return insights
        
        # Predict heart rate trends
        recent_hr = [d["data"].get("heart_rate", 70) for d in watch_data[-100:]]
        hr_trend = np.polyfit(range(len(recent_hr)), recent_hr, 1)[0]
        
        if abs(hr_trend) > 0.1:  # Significant trend
            predicted_hr = recent_hr[-1] + hr_trend * 24  # 24 hours ahead
            confidence_interval = (predicted_hr - 5, predicted_hr + 5)
            
            risk_factors = []
            preventive_measures = []
            
            if hr_trend > 0:
                risk_factors.append("increasing_resting_heart_rate")
                preventive_measures.extend([
                    "Monitor stress levels and implement relaxation techniques",
                    "Ensure adequate hydration and rest",
                    "Consider cardiovascular health assessment"
                ])
            else:
                risk_factors.append("decreasing_heart_rate_trend")
                preventive_measures.append("Monitor for symptoms and maintain regular activity")
            
            insights.append(PredictiveInsight(
                insight_id=f"hr_prediction_{int(time.time())}",
                prediction_type="heart_rate_trend",
                target_metric="heart_rate",
                predicted_value=float(predicted_hr),
                confidence_interval=confidence_interval,
                prediction_horizon_hours=24,
                confidence_score=0.8,
                risk_factors=risk_factors,
                preventive_measures=preventive_measures
            ))
        
        return insights
    
    def _detect_health_patterns(self) -> List[HealthPattern]:
        """Detect significant health patterns."""
        patterns = []
        
        # Example: Detect circadian disruption pattern
        circadian_analysis = self.circadian_analyzer.analyze_circadian_rhythm()
        
        if circadian_analysis.get("rhythm_quality") == "poor":
            patterns.append(HealthPattern(
                pattern_id=f"circadian_disruption_{int(time.time())}",
                pattern_type="circadian",
                confidence=0.8,
                start_time=time.time() - 7*24*3600,  # Last week
                end_time=time.time(),
                description="Circadian rhythm disruption detected",
                affected_metrics=["sleep_quality", "heart_rate", "activity"],
                severity="medium",
                recommended_actions=[
                    "Establish consistent sleep schedule",
                    "Increase morning light exposure",
                    "Avoid screens before bedtime"
                ]
            ))
        
        return patterns
    
    def _calculate_analysis_confidence(self) -> float:
        """Calculate confidence in analysis results."""
        # Base confidence on data availability and quality
        total_data_points = sum(len(data) for data in self.longitudinal_data.values())
        
        if total_data_points < 100:
            return 0.3
        elif total_data_points < 1000:
            return 0.6
        elif total_data_points < 5000:
            return 0.8
        else:
            return 0.95
    
    def _calculate_overall_health_score(self) -> float:
        """Calculate overall health score from all analytics."""
        scores = []
        
        # Circadian health score
        circadian_analysis = self.circadian_analyzer.analyze_circadian_rhythm()
        if circadian_analysis.get("circadian_strength"):
            scores.append(circadian_analysis["circadian_strength"] * 100)
        
        # Stress management score
        stress_analysis = self.stress_analyzer.analyze_stress_patterns()
        if stress_analysis.get("stress_management_score"):
            scores.append(stress_analysis["stress_management_score"])
        
        # Default good health if no specific scores
        if not scores:
            scores = [75.0]
        
        return float(np.mean(scores))
    
    def _generate_key_recommendations(self) -> List[str]:
        """Generate key health recommendations."""
        recommendations = []
        
        # Get recommendations from specialized analyzers
        circadian_analysis = self.circadian_analyzer.analyze_circadian_rhythm()
        stress_analysis = self.stress_analyzer.analyze_stress_patterns()
        
        if circadian_analysis.get("recommendations"):
            recommendations.extend(circadian_analysis["recommendations"][:3])  # Top 3
        
        if stress_analysis.get("personalized_recommendations"):
            recommendations.extend(stress_analysis["personalized_recommendations"][:3])  # Top 3
        
        # Add general recommendations if none specific
        if not recommendations:
            recommendations = [
                "Maintain consistent daily routines for optimal health",
                "Engage in regular physical activity appropriate for your fitness level",
                "Prioritize quality sleep and stress management"
            ]
        
        return recommendations[:5]  # Limit to top 5
    
    def _perform_risk_assessment(self) -> Dict[str, Any]:
        """Perform comprehensive health risk assessment."""
        risk_factors = []
        risk_level = "low"
        
        # Assess circadian health risks
        circadian_analysis = self.circadian_analyzer.analyze_circadian_rhythm()
        if circadian_analysis.get("rhythm_quality") in ["poor", "fair"]:
            risk_factors.append("circadian_disruption")
            risk_level = "medium"
        
        # Assess stress-related risks
        stress_analysis = self.stress_analyzer.analyze_stress_patterns()
        if stress_analysis.get("chronic_stress_level", 0) > 0.6:
            risk_factors.append("chronic_stress")
            risk_level = "medium"
        
        if stress_analysis.get("stress_resilience_score", 1) < 0.4:
            risk_factors.append("low_stress_resilience")
            if risk_level == "medium":
                risk_level = "high"
        
        return {
            "overall_risk_level": risk_level,
            "identified_risk_factors": risk_factors,
            "risk_mitigation_strategies": self._generate_risk_mitigation_strategies(risk_factors)
        }
    
    def _generate_risk_mitigation_strategies(self, risk_factors: List[str]) -> List[str]:
        """Generate strategies to mitigate identified risks."""
        strategies = []
        
        for risk in risk_factors:
            if risk == "circadian_disruption":
                strategies.extend([
                    "Implement light therapy protocols",
                    "Establish strict sleep-wake scheduling",
                    "Minimize evening blue light exposure"
                ])
            elif risk == "chronic_stress":
                strategies.extend([
                    "Develop comprehensive stress management program",
                    "Consider professional stress counseling",
                    "Implement regular relaxation practices"
                ])
            elif risk == "low_stress_resilience":
                strategies.extend([
                    "Build stress resilience through gradual exposure therapy",
                    "Practice mindfulness and emotional regulation",
                    "Strengthen social support networks"
                ])
        
        return list(set(strategies))  # Remove duplicates
    
    def _generate_action_plan(self) -> Dict[str, Any]:
        """Generate personalized action plan."""
        action_plan = {
            "immediate_actions": [],
            "weekly_goals": [],
            "monthly_objectives": [],
            "tracking_metrics": []
        }
        
        # Get current analysis results
        overall_score = self._calculate_overall_health_score()
        
        if overall_score < 60:
            action_plan["immediate_actions"].extend([
                "Schedule comprehensive health assessment",
                "Begin stress reduction protocols immediately",
                "Optimize sleep environment and routine"
            ])
        
        action_plan["weekly_goals"].extend([
            "Track daily sleep quality and duration",
            "Monitor stress levels and triggers",
            "Maintain consistent activity schedule"
        ])
        
        action_plan["monthly_objectives"].extend([
            "Improve overall health score by 10 points",
            "Establish sustainable healthy routines",
            "Review and adjust health strategies based on data"
        ])
        
        action_plan["tracking_metrics"] = [
            "circadian_rhythm_strength",
            "stress_resilience_score",
            "sleep_quality_trend",
            "heart_rate_variability"
        ]
        
        return action_plan
    
    def _get_analysis_period_days(self) -> int:
        """Get the period of analysis in days."""
        if not self.longitudinal_data:
            return 0
        
        # Find earliest data point
        earliest_time = float('inf')
        for device_data in self.longitudinal_data.values():
            if device_data:
                earliest_time = min(earliest_time, device_data[0]["timestamp"])
        
        if earliest_time == float('inf'):
            return 0
        
        return int((time.time() - earliest_time) / 86400)
    
    def export_analytics_results(self, filepath: str) -> None:
        """Export comprehensive analytics results."""
        export_data = {
            "export_timestamp": time.time(),
            "analysis_results": [asdict(result) for result in list(self.analysis_results)],
            "health_patterns": [asdict(pattern) for pattern in list(self.health_patterns)],
            "predictive_insights": [asdict(insight) for insight in list(self.predictive_insights)],
            "comprehensive_report": self.get_comprehensive_health_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Analytics results exported to {filepath}") 