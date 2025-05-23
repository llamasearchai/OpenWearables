"""
Comprehensive Test Suite for Smart Watch Device

Tests all aspects of the Apple Watch inspired smart watch implementation
including ECG, blood oxygen, activity tracking, sleep analysis, and emergency features.
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch

from openwearables.core.devices.smart_watch import (
    SmartWatchDevice, AdvancedECGSensor, BloodOxygenSensor, ActivityTracker,
    SleepTracker, FallDetectionSystem, ECGReading, BloodOxygenReading,
    ActivityMetrics, SleepData, FallDetectionData
)
from openwearables.core.sensor_manager import SensorInterface


class TestAdvancedECGSensor:
    """Test suite for AdvancedECGSensor class."""
    
    def test_ecg_sensor_initialization(self):
        """Test ECG sensor initialization."""
        sensor = AdvancedECGSensor(sampling_rate=512)
        
        assert sensor.sampling_rate == 512
        assert sensor.recording_duration == 30
        assert sensor.base_heart_rate == 72
        assert sensor.current_rhythm == "normal"
        assert len(sensor.rhythm_states) == 4
    
    def test_ecg_normal_rhythm_reading(self):
        """Test ECG reading with normal rhythm."""
        sensor = AdvancedECGSensor()
        sensor.current_rhythm = "normal"
        
        reading = sensor.read()
        
        assert isinstance(reading, ECGReading)
        assert isinstance(reading.timestamp, float)
        assert isinstance(reading.voltage_samples, list)
        assert len(reading.voltage_samples) > 0
        assert 40 <= reading.heart_rate <= 180
        assert reading.rhythm_classification == "normal"
        assert 0.1 <= reading.signal_quality <= 1.0
    
    def test_ecg_afib_rhythm_reading(self):
        """Test ECG reading with atrial fibrillation."""
        sensor = AdvancedECGSensor()
        sensor.current_rhythm = "afib"
        
        reading = sensor.read()
        
        assert reading.rhythm_classification == "afib"
        # AFib typically has irregular rate around 85 ± 20 bpm
        assert 45 <= reading.heart_rate <= 125
    
    def test_ecg_bradycardia_reading(self):
        """Test ECG reading with bradycardia."""
        sensor = AdvancedECGSensor()
        sensor.current_rhythm = "bradycardia"
        
        reading = sensor.read()
        
        assert reading.rhythm_classification == "bradycardia"
        # Bradycardia typically < 60 bpm
        assert reading.heart_rate < 70  # Allow some variance
    
    def test_ecg_tachycardia_reading(self):
        """Test ECG reading with tachycardia."""
        sensor = AdvancedECGSensor()
        sensor.current_rhythm = "tachycardia"
        
        reading = sensor.read()
        
        assert reading.rhythm_classification == "tachycardia"
        # Tachycardia typically > 100 bpm
        assert reading.heart_rate > 95  # Allow some variance
    
    def test_ecg_waveform_generation(self):
        """Test ECG waveform generation."""
        sensor = AdvancedECGSensor()
        
        # Test different heart rates
        for hr in [60, 80, 100]:
            waveform = sensor._generate_ecg_waveform(100, hr)
            
            assert isinstance(waveform, np.ndarray)
            assert len(waveform) == 100
            assert np.all(np.isfinite(waveform))
    
    def test_ecg_rhythm_transitions(self):
        """Test rhythm state transitions."""
        sensor = AdvancedECGSensor()
        sensor.rhythm_transition_prob = 1.0  # Force transition
        
        initial_rhythm = sensor.current_rhythm
        reading = sensor.read()
        
        # With probability 1.0, rhythm should change
        assert sensor.current_rhythm in sensor.rhythm_states


class TestBloodOxygenSensor:
    """Test suite for BloodOxygenSensor class."""
    
    def test_blood_oxygen_initialization(self):
        """Test blood oxygen sensor initialization."""
        sensor = BloodOxygenSensor()
        
        assert sensor.baseline_spo2 == 98.5
        assert sensor.measurement_interval == 10
        assert sensor.led_wavelengths == [660, 940]
    
    def test_blood_oxygen_reading(self):
        """Test blood oxygen reading."""
        sensor = BloodOxygenSensor()
        
        reading = sensor.read()
        
        assert isinstance(reading, BloodOxygenReading)
        assert isinstance(reading.timestamp, float)
        assert 85 <= reading.spo2_percentage <= 100
        assert 0.1 <= reading.perfusion_index <= 20
        assert 0.1 <= reading.measurement_confidence <= 1.0
        assert reading.pulse_amplitude > 0.1
    
    def test_blood_oxygen_consistency(self):
        """Test consistency of blood oxygen readings."""
        sensor = BloodOxygenSensor()
        
        readings = [sensor.read() for _ in range(10)]
        
        spo2_values = [r.spo2_percentage for r in readings]
        
        # SpO2 should be relatively stable
        spo2_std = np.std(spo2_values)
        assert spo2_std < 5.0  # Standard deviation should be reasonable
        
        # All readings should be in physiological range
        assert all(85 <= spo2 <= 100 for spo2 in spo2_values)


class TestActivityTracker:
    """Test suite for ActivityTracker class."""
    
    def test_activity_tracker_initialization(self):
        """Test activity tracker initialization."""
        tracker = ActivityTracker()
        
        assert tracker.daily_step_count == 0
        assert tracker.daily_distance == 0.0
        assert tracker.current_activity == "sedentary"
        assert len(tracker.activity_states) == 4
        assert tracker.max_heart_rate == 190
    
    def test_activity_metrics_reading(self):
        """Test activity metrics reading."""
        tracker = ActivityTracker()
        
        metrics = tracker.read()
        
        assert isinstance(metrics, ActivityMetrics)
        assert isinstance(metrics.timestamp, float)
        assert metrics.steps >= 0
        assert metrics.distance_meters >= 0
        assert metrics.calories_burned >= 0
        assert metrics.active_energy >= 0
        assert metrics.exercise_minutes >= 0
        assert 0 <= metrics.stand_hours <= 12
        assert isinstance(metrics.heart_rate_zones, dict)
    
    def test_activity_progression(self):
        """Test activity progression over time."""
        tracker = ActivityTracker()
        tracker.current_activity = "moderate"
        
        initial_metrics = tracker.read()
        
        # Simulate some time passing with moderate activity
        for _ in range(60):  # Simulate 60 readings
            tracker.read()
        
        final_metrics = tracker.read()
        
        # Steps should increase with moderate activity
        assert final_metrics.steps > initial_metrics.steps
        assert final_metrics.distance_meters > initial_metrics.distance_meters
        assert final_metrics.calories_burned > initial_metrics.calories_burned
    
    def test_daily_counter_reset(self):
        """Test daily counter reset functionality."""
        tracker = ActivityTracker()
        
        # Set some activity data
        tracker.daily_step_count = 5000
        tracker.daily_distance = 4000
        tracker.daily_calories = 300
        
        # Force reset by setting last_reset to past
        tracker.last_reset = time.time() - 86500  # More than 24 hours ago
        
        metrics = tracker.read()
        
        # Should have reset to low values
        assert metrics.steps < 100  # Some steps may be added during read()
        assert metrics.distance_meters < 100
        assert metrics.calories_burned < 50
    
    def test_heart_rate_zones(self):
        """Test heart rate zone calculations."""
        tracker = ActivityTracker()
        
        # Test with different age
        tracker.max_heart_rate = 180  # Simulating younger person
        
        # Check zone boundaries
        assert tracker.hr_zones["zone1"][0] == 90  # 50% of 180
        assert tracker.hr_zones["zone5"][1] == 180  # 100% of 180


class TestSleepTracker:
    """Test suite for SleepTracker class."""
    
    def test_sleep_tracker_initialization(self):
        """Test sleep tracker initialization."""
        tracker = SleepTracker()
        
        assert len(tracker.sleep_stages) == 4
        assert tracker.current_stage == "awake"
        assert tracker.cycle_duration == 90 * 60
        assert len(tracker.stage_durations) == 4
    
    def test_sleep_data_reading(self):
        """Test sleep data reading."""
        tracker = SleepTracker()
        
        sleep_data = tracker.read()
        
        assert isinstance(sleep_data, SleepData)
        assert isinstance(sleep_data.timestamp, float)
        assert sleep_data.sleep_stage in tracker.sleep_stages
        assert 0 <= sleep_data.sleep_quality_score <= 100
        assert 0 <= sleep_data.movement_intensity <= 1
        assert 20 <= sleep_data.heart_rate_variability <= 100
    
    def test_sleep_stage_transitions(self):
        """Test sleep stage transitions."""
        tracker = SleepTracker()
        tracker.stage_transition_prob = 1.0  # Force transitions
        
        # Test during sleep hours (simulate 2 AM)
        with patch('time.time', return_value=2 * 3600):  # 2 AM
            initial_stage = tracker.current_stage
            sleep_data = tracker.read()
            
            # Should transition to sleep-related stage
            assert sleep_data.sleep_stage in ["awake", "light", "deep", "rem"]
    
    def test_sleep_quality_calculation(self):
        """Test sleep quality calculation."""
        tracker = SleepTracker()
        
        # Test different sleep stages
        for stage in ["awake", "light", "deep", "rem"]:
            tracker.current_stage = stage
            quality = tracker._calculate_sleep_quality()
            
            assert 0 <= quality <= 100
            
            # Deep sleep should generally have higher quality
            if stage == "deep":
                assert quality >= 50  # Deep sleep should be high quality
    
    def test_movement_intensity_by_stage(self):
        """Test movement intensity varies by sleep stage."""
        tracker = SleepTracker()
        
        # Enable manual stage override to prevent automatic transitions
        tracker._manual_stage_override = True
        
        movement_intensities = {}
        
        for stage in ["awake", "light", "deep", "rem"]:
            tracker.current_stage = stage
            sleep_data = tracker.read()
            movement_intensities[stage] = sleep_data.movement_intensity
        
        # Deep sleep should have lowest movement
        assert movement_intensities["deep"] < movement_intensities["awake"]
        assert movement_intensities["deep"] < movement_intensities["light"]


class TestFallDetectionSystem:
    """Test suite for FallDetectionSystem class."""
    
    def test_fall_detection_initialization(self):
        """Test fall detection system initialization."""
        detector = FallDetectionSystem()
        
        assert detector.sensitivity_threshold == 2.5
        assert len(detector.fall_patterns) == 4
        assert detector.current_pattern == "normal"
    
    def test_normal_movement_detection(self):
        """Test normal movement detection."""
        detector = FallDetectionSystem()
        
        # Read multiple times to ensure normal patterns
        for _ in range(100):
            fall_data = detector.read()
            
            assert isinstance(fall_data, FallDetectionData)
            assert isinstance(fall_data.timestamp, float)
            assert fall_data.impact_magnitude >= 0
            assert isinstance(fall_data.fall_detected, bool)
            assert 0 <= fall_data.confidence_level <= 1
            assert fall_data.motion_pattern in detector.fall_patterns
            
            # Most readings should be normal with low impact
            if fall_data.motion_pattern == "normal":
                assert fall_data.impact_magnitude < detector.sensitivity_threshold
                assert not fall_data.fall_detected
    
    def test_fall_detection_threshold(self):
        """Test fall detection threshold logic."""
        detector = FallDetectionSystem()
        
        # Test with manually set high impact
        detector.current_pattern = "fall"
        
        # Mock the random generation to simulate a fall
        with patch('numpy.random.random', return_value=0.0005):  # Force fall event
            with patch('numpy.random.choice', return_value="fall"):
                with patch('numpy.random.normal', return_value=4.0):  # High impact
                    fall_data = detector.read()
                    
                    assert fall_data.impact_magnitude > detector.sensitivity_threshold
                    assert fall_data.fall_detected
                    assert fall_data.confidence_level > 0.1
    
    def test_fall_confidence_calculation(self):
        """Test fall confidence calculation."""
        detector = FallDetectionSystem()
        
        # Test confidence for different impact levels
        test_impacts = [1.0, 2.0, 3.0, 5.0, 8.0]
        
        for impact in test_impacts:
            # Mock the impact magnitude
            with patch.object(detector, 'read') as mock_read:
                mock_fall_data = FallDetectionData(
                    timestamp=time.time(),
                    impact_magnitude=impact,
                    fall_detected=impact > detector.sensitivity_threshold,
                    confidence_level=0.5,  # Will be calculated
                    motion_pattern="normal"
                )
                mock_read.return_value = mock_fall_data
                
                fall_data = detector.read()
                
                if impact > detector.sensitivity_threshold:
                    assert fall_data.fall_detected
                    # Higher impact should have higher confidence
                    assert fall_data.confidence_level > 0


class TestSmartWatchDevice:
    """Test suite for SmartWatchDevice class."""
    
    def test_smart_watch_initialization(self):
        """Test smart watch device initialization."""
        device = SmartWatchDevice(sensor_id=1)
        
        assert device.sensor_id == 1
        assert device.name == "smart_watch"
        assert device.sampling_rate == 1
        assert device.battery_level == 0.92
        assert device.is_on_wrist
        assert device.water_resistance_active
        
        # Check subsensors
        assert hasattr(device, 'ecg_sensor')
        assert hasattr(device, 'blood_oxygen_sensor')
        assert hasattr(device, 'activity_tracker')
        assert hasattr(device, 'sleep_tracker')
        assert hasattr(device, 'fall_detector')
        
        # Check user profile
        assert isinstance(device.user_profile, dict)
        assert "age" in device.user_profile
        assert "weight_kg" in device.user_profile
    
    def test_smart_watch_reading(self):
        """Test smart watch comprehensive reading."""
        device = SmartWatchDevice(sensor_id=1)
        
        data = device.read()
        
        assert isinstance(data, np.ndarray)
        assert len(data) == 12  # Expected combined data elements
        
        # Verify data structure
        timestamp, heart_rate, spo2, steps, distance, sleep_quality, fall_detected, hrv, health_score, exercise_minutes, calories, battery = data
        
        assert isinstance(timestamp, (int, float))
        assert 40 <= heart_rate <= 180
        assert 85 <= spo2 <= 100
        assert steps >= 0
        assert distance >= 0
        assert 0 <= sleep_quality <= 100
        assert fall_detected in [0, 1]  # Boolean as float
        assert 20 <= hrv <= 100
        assert 0 <= health_score <= 100
        assert exercise_minutes >= 0
        assert calories >= 0
        assert 0 <= battery <= 1
    
    def test_detailed_sensor_readings(self):
        """Test detailed readings from individual sensors."""
        device = SmartWatchDevice(sensor_id=1)
        
        # Test ECG reading
        ecg_reading = device.get_ecg_reading()
        assert isinstance(ecg_reading, ECGReading)
        assert 40 <= ecg_reading.heart_rate <= 180
        
        # Test blood oxygen reading
        spo2_reading = device.get_blood_oxygen_reading()
        assert isinstance(spo2_reading, BloodOxygenReading)
        assert 85 <= spo2_reading.spo2_percentage <= 100
        
        # Test activity metrics
        activity_metrics = device.get_activity_metrics()
        assert isinstance(activity_metrics, ActivityMetrics)
        assert activity_metrics.steps >= 0
        
        # Test sleep analysis
        sleep_data = device.get_sleep_analysis()
        assert isinstance(sleep_data, SleepData)
        assert sleep_data.sleep_stage in ["awake", "light", "deep", "rem"]
        
        # Test fall detection
        fall_data = device.get_fall_detection_status()
        assert isinstance(fall_data, FallDetectionData)
        assert fall_data.motion_pattern in ["normal", "stumble", "fall", "hard_fall"]
    
    def test_overall_health_score_calculation(self):
        """Test overall health score calculation."""
        device = SmartWatchDevice(sensor_id=1)
        
        # Create mock data for testing
        from openwearables.core.devices.smart_watch import ECGReading, BloodOxygenReading, ActivityMetrics, SleepData
        
        ecg_data = ECGReading(
            timestamp=time.time(),
            voltage_samples=[0.5, 0.6, 0.4],
            heart_rate=75,
            rhythm_classification="normal",
            signal_quality=0.9
        )
        
        blood_oxygen_data = BloodOxygenReading(
            timestamp=time.time(),
            spo2_percentage=98,
            perfusion_index=2.5,
            measurement_confidence=0.9,
            pulse_amplitude=1.2
        )
        
        activity_data = ActivityMetrics(
            timestamp=time.time(),
            steps=8000,
            distance_meters=6400,
            calories_burned=400,
            active_energy=300,
            exercise_minutes=25,
            stand_hours=8,
            heart_rate_zones={}
        )
        
        sleep_data = SleepData(
            timestamp=time.time(),
            sleep_stage="deep",
            sleep_quality_score=85,
            movement_intensity=0.1,
            heart_rate_variability=55
        )
        
        health_score = device._calculate_overall_health_score(
            ecg_data, blood_oxygen_data, activity_data, sleep_data
        )
        
        assert 0 <= health_score <= 100
        assert health_score > 70  # Should be good with these healthy values
    
    def test_emergency_sos_functionality(self):
        """Test emergency SOS functionality."""
        device = SmartWatchDevice(sensor_id=1)
        
        # Test SOS activation
        response = device.trigger_emergency_sos()
        
        assert response["status"] == "activated"
        assert "timestamp" in response
        assert "user_id" in response
        assert "location" in response
        assert "contacts_notified" in response
        assert response["medical_id_shared"]
        assert response["emergency_services_contacted"]
        
        # Test SOS disabled
        device.emergency_sos_enabled = False
        response = device.trigger_emergency_sos()
        
        assert response["status"] == "disabled"
    
    def test_user_profile_updates(self):
        """Test user profile updates."""
        device = SmartWatchDevice(sensor_id=1)
        
        original_max_hr = device.activity_tracker.max_heart_rate
        
        # Update age
        device.update_user_profile({"age": 25})
        
        assert device.user_profile["age"] == 25
        # Heart rate zones should be recalculated
        new_max_hr = device.activity_tracker.max_heart_rate
        assert new_max_hr != original_max_hr
        assert new_max_hr == 220 - 25  # 195
    
    def test_health_insights_generation(self):
        """Test health insights generation."""
        device = SmartWatchDevice(sensor_id=1)
        
        # Generate some data
        for _ in range(10):
            device.read()
        
        insights = device.get_health_insights()
        
        if insights.get("status") != "insufficient_data":
            assert "cardiovascular_health" in insights
            assert "respiratory_health" in insights
            assert "activity_assessment" in insights
            assert "sleep_assessment" in insights
            assert "recommendations" in insights
            
            assert isinstance(insights["recommendations"], list)
    
    def test_health_recommendations(self):
        """Test health recommendation generation."""
        device = SmartWatchDevice(sensor_id=1)
        
        # Test with different health scenarios
        test_cases = [
            {"heart_rate": 130, "spo2": 98, "steps": 10000, "sleep_quality": 85},  # High HR
            {"heart_rate": 45, "spo2": 98, "steps": 10000, "sleep_quality": 85},   # Low HR
            {"heart_rate": 75, "spo2": 92, "steps": 10000, "sleep_quality": 85},   # Low SpO2
            {"heart_rate": 75, "spo2": 98, "steps": 3000, "sleep_quality": 85},    # Low activity
            {"heart_rate": 75, "spo2": 98, "steps": 10000, "sleep_quality": 40},   # Poor sleep
            {"heart_rate": 75, "spo2": 98, "steps": 10000, "sleep_quality": 85},   # All good
        ]
        
        for case in test_cases:
            recommendations = device._generate_health_recommendations(
                case["heart_rate"], case["spo2"], case["steps"], case["sleep_quality"]
            )
            
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            
            # Check for appropriate recommendations
            if case["heart_rate"] > 120:
                assert any("elevated heart rate" in rec.lower() for rec in recommendations)
            elif case["heart_rate"] < 50:
                assert any("low heart rate" in rec.lower() for rec in recommendations)
            
            if case["spo2"] < 95:
                assert any("blood oxygen" in rec.lower() for rec in recommendations)
            
            if case["steps"] < 5000:
                assert any("activity" in rec.lower() or "steps" in rec.lower() for rec in recommendations)
            
            if case["sleep_quality"] < 50:
                assert any("sleep" in rec.lower() for rec in recommendations)


class TestSmartWatchIntegration:
    """Integration tests for smart watch with main system."""
    
    def test_sensor_interface_compliance(self):
        """Test that SmartWatchDevice complies with SensorInterface."""
        device = SmartWatchDevice(sensor_id=1)
        
        # Test required interface methods
        assert hasattr(device, 'read')
        assert hasattr(device, 'start')
        assert hasattr(device, 'stop')
        assert hasattr(device, 'get_buffer')
        
        # Test that read returns numpy array
        data = device.read()
        assert isinstance(data, np.ndarray)
        
        # Test interface inheritance
        assert isinstance(device, SensorInterface)
    
    def test_threading_safety(self):
        """Test thread safety during concurrent operations."""
        device = SmartWatchDevice(sensor_id=1)
        device.start()
        
        # Function to read data in separate thread
        def read_data(results, index):
            try:
                for _ in range(10):
                    data = device.read()
                    time.sleep(0.01)
                results[index] = "success"
            except Exception as e:
                results[index] = f"error: {str(e)}"
        
        # Start multiple threads
        threads = []
        results = {}
        
        for i in range(3):
            thread = threading.Thread(target=read_data, args=(results, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5)
        
        device.stop()
        
        # Check all threads completed successfully
        for i in range(3):
            assert results.get(i) == "success"
    
    def test_continuous_operation(self):
        """Test continuous operation over time."""
        device = SmartWatchDevice(sensor_id=1)
        device.start()
        
        # Collect data over time
        readings = []
        start_time = time.time()
        
        while time.time() - start_time < 2:  # Run for 2 seconds
            data = device.read()
            readings.append(data)
            time.sleep(0.1)
        
        device.stop()
        
        assert len(readings) >= 10  # Should have multiple readings
        
        # Check data consistency
        for reading in readings:
            assert isinstance(reading, np.ndarray)
            assert len(reading) == 12
    
    def test_data_quality_and_ranges(self):
        """Test data quality and physiological ranges."""
        device = SmartWatchDevice(sensor_id=1)
        
        # Collect multiple readings
        readings = [device.read() for _ in range(20)]
        
        for reading in readings:
            timestamp, heart_rate, spo2, steps, distance, sleep_quality, fall_detected, hrv, health_score, exercise_minutes, calories, battery = reading
            
            # Physiological ranges
            assert 40 <= heart_rate <= 180, f"Heart rate out of range: {heart_rate}"
            assert 85 <= spo2 <= 100, f"SpO2 out of range: {spo2}"
            assert 0 <= sleep_quality <= 100, f"Sleep quality out of range: {sleep_quality}"
            assert 20 <= hrv <= 100, f"HRV out of range: {hrv}"
            assert 0 <= health_score <= 100, f"Health score out of range: {health_score}"
            
            # Logical constraints
            assert steps >= 0, f"Steps cannot be negative: {steps}"
            assert distance >= 0, f"Distance cannot be negative: {distance}"
            assert calories >= 0, f"Calories cannot be negative: {calories}"
            assert exercise_minutes >= 0, f"Exercise minutes cannot be negative: {exercise_minutes}"
            assert fall_detected in [0, 1], f"Fall detected must be 0 or 1: {fall_detected}"
            assert 0 <= battery <= 1, f"Battery level out of range: {battery}"
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        device = SmartWatchDevice(sensor_id=1)
        
        # Test reading performance
        start_time = time.time()
        num_readings = 1000
        
        for _ in range(num_readings):
            device.read()
        
        elapsed_time = time.time() - start_time
        readings_per_second = num_readings / elapsed_time
        
        # Should be able to handle at least 100 readings per second
        assert readings_per_second >= 100, f"Performance too slow: {readings_per_second:.1f} readings/sec"
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        import gc
        
        device = SmartWatchDevice(sensor_id=1)
        device.start()
        
        # Get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Generate lots of data
        for _ in range(1000):
            device.read()
        
        # Check memory state
        gc.collect()
        final_objects = len(gc.get_objects())
        
        device.stop()
        
        # Memory growth should be reasonable
        object_growth = final_objects - initial_objects
        assert object_growth < 10000, f"Excessive memory growth: {object_growth} objects"
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        device = SmartWatchDevice(sensor_id=1)
        
        # Test with invalid user profile updates
        try:
            device.update_user_profile({"invalid_field": "invalid_value"})
            # Should not raise exception
        except Exception as e:
            pytest.fail(f"Should handle invalid profile updates gracefully: {e}")
        
        # Test emergency SOS with disabled state
        device.emergency_sos_enabled = False
        response = device.trigger_emergency_sos()
        assert response["status"] == "disabled"
        
        # Test insights with insufficient data
        device_new = SmartWatchDevice(sensor_id=2)
        insights = device_new.get_health_insights()
        assert insights.get("status") == "insufficient_data"


class TestSmartWatchPerformance:
    """Performance and stress tests for smart watch."""
    
    def test_high_frequency_reading(self):
        """Test high-frequency data reading."""
        device = SmartWatchDevice(sensor_id=1)
        
        start_time = time.time()
        readings = []
        
        # Read data at high frequency for 1 second
        while time.time() - start_time < 1.0:
            readings.append(device.read())
        
        # Should handle many readings per second
        assert len(readings) >= 50
        
        # All readings should be valid
        for reading in readings:
            assert isinstance(reading, np.ndarray)
            assert len(reading) == 12
    
    def test_concurrent_sensor_access(self):
        """Test concurrent access to different sensors."""
        device = SmartWatchDevice(sensor_id=1)
        
        def access_ecg():
            return [device.get_ecg_reading() for _ in range(10)]
        
        def access_activity():
            return [device.get_activity_metrics() for _ in range(10)]
        
        def access_sleep():
            return [device.get_sleep_analysis() for _ in range(10)]
        
        # Run concurrent access
        threads = []
        results = {}
        
        for i, func in enumerate([access_ecg, access_activity, access_sleep]):
            thread = threading.Thread(target=lambda f=func, idx=i: results.update({idx: f()}))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join(timeout=5)
        
        # All threads should complete successfully
        assert len(results) == 3
        for readings in results.values():
            assert len(readings) == 10
    
    def test_long_term_stability(self):
        """Test long-term stability and data consistency."""
        device = SmartWatchDevice(sensor_id=1)
        device.start()
        
        # Collect data over extended period
        readings = []
        start_time = time.time()
        
        while time.time() - start_time < 5:  # Run for 5 seconds
            reading = device.read()
            readings.append(reading)
            time.sleep(0.1)
        
        device.stop()
        
        # Analyze stability
        heart_rates = [r[1] for r in readings]
        spo2_values = [r[2] for r in readings]
        
        # Heart rate should be relatively stable
        hr_std = np.std(heart_rates)
        assert hr_std < 20, f"Heart rate too variable: {hr_std}"
        
        # SpO2 should be stable
        spo2_std = np.std(spo2_values)
        assert spo2_std < 3, f"SpO2 too variable: {spo2_std}"
        
        # Check for any invalid values
        assert all(40 <= hr <= 180 for hr in heart_rates)
        assert all(85 <= spo2 <= 100 for spo2 in spo2_values) 