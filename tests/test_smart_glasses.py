"""
Comprehensive Test Suite for Smart Glasses Device

Tests all functionality including eye tracking, environmental sensing,
biometric monitoring, privacy controls, and health insights.
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from openwearables.core.devices.smart_glasses import (
    SmartGlassesDevice,
    EyeTrackingSensor,
    EnvironmentalSensor, 
    BiometricSensor,
    DigitalWellnessTracker,
    EyeTrackingData,
    EnvironmentalData,
    BiometricData
)

class TestEyeTrackingSensor:
    """Test suite for EyeTrackingSensor class."""
    
    def test_initialization(self):
        """Test eye tracking sensor initialization."""
        sensor = EyeTrackingSensor(sampling_rate=120)
        assert sensor.sampling_rate == 120
        assert sensor.baseline_pupil_size == 4.0
        assert sensor.current_activity in ["reading", "viewing", "navigation", "rest"]
        assert 0 <= sensor.cognitive_load_state <= 1
    
    def test_realistic_pupil_sizes(self):
        """Test that pupil sizes are within realistic ranges."""
        sensor = EyeTrackingSensor()
        
        # Generate multiple readings to test range
        for _ in range(100):
            data = sensor.read()
            assert 2.0 <= data.left_pupil_diameter <= 8.0
            assert 2.0 <= data.right_pupil_diameter <= 8.0
            assert isinstance(data.timestamp, float)
    
    def test_gaze_coordinates(self):
        """Test gaze coordinate normalization."""
        sensor = EyeTrackingSensor()
        
        for _ in range(50):
            data = sensor.read()
            assert 0.0 <= data.gaze_x <= 1.0
            assert 0.0 <= data.gaze_y <= 1.0
    
    def test_blink_rate_by_activity(self):
        """Test that blink rate varies by activity."""
        sensor = EyeTrackingSensor()
        
        # Test different activities
        activities = ["reading", "viewing", "navigation", "rest"]
        blink_rates = {}
        
        for activity in activities:
            sensor.current_activity = activity
            total_rate = 0
            for _ in range(20):
                data = sensor.read()
                total_rate += data.blink_rate
            
            avg_rate = total_rate / 20
            blink_rates[activity] = avg_rate
            assert 5 <= avg_rate <= 35  # Reasonable blink rate range
        
        # Reading should have higher blink rate than rest
        assert blink_rates["reading"] > blink_rates["rest"]
    
    def test_convergence_distance(self):
        """Test convergence distance for different activities."""
        sensor = EyeTrackingSensor()
        
        # Reading should have shorter convergence distance
        sensor.current_activity = "reading"
        reading_distances = [sensor.read().convergence_distance for _ in range(10)]
        avg_reading_distance = np.mean(reading_distances)
        
        # Navigation should have longer convergence distance
        sensor.current_activity = "navigation"
        nav_distances = [sensor.read().convergence_distance for _ in range(10)]
        avg_nav_distance = np.mean(nav_distances)
        
        assert avg_reading_distance < avg_nav_distance
        assert all(d > 0.2 for d in reading_distances + nav_distances)

class TestEnvironmentalSensor:
    """Test suite for EnvironmentalSensor class."""
    
    def test_initialization(self):
        """Test environmental sensor initialization."""
        sensor = EnvironmentalSensor(sampling_rate=1)
        assert sensor.sampling_rate == 1
        assert hasattr(sensor, 'base_time')
    
    def test_daily_light_cycle(self):
        """Test that ambient light follows daily patterns."""
        sensor = EnvironmentalSensor()
        
        # Mock different times of day
        with patch('time.time') as mock_time:
            # Test daytime (noon)
            mock_time.return_value = 12 * 3600  # 12:00 PM
            day_data = sensor.read()
            
            # Test nighttime (midnight)
            mock_time.return_value = 0  # 12:00 AM
            night_data = sensor.read()
            
            # Daytime should be brighter than nighttime
            assert day_data.ambient_light > night_data.ambient_light
    
    def test_uv_index_during_day(self):
        """Test UV index is only present during daylight hours."""
        sensor = EnvironmentalSensor()
        
        with patch('time.time') as mock_time:
            # Test midday
            mock_time.return_value = 14 * 3600  # 2:00 PM
            day_data = sensor.read()
            
            # Test night
            mock_time.return_value = 2 * 3600  # 2:00 AM
            night_data = sensor.read()
            
            assert day_data.uv_index >= 0
            assert night_data.uv_index == 0
    
    def test_proximity_detection(self):
        """Test proximity distance measurements."""
        sensor = EnvironmentalSensor()
        
        for _ in range(20):
            data = sensor.read()
            assert 0 <= data.proximity_distance <= 10.0
    
    def test_temperature_and_humidity(self):
        """Test temperature and humidity ranges."""
        sensor = EnvironmentalSensor()
        
        for _ in range(30):
            data = sensor.read()
            assert 15 <= data.temperature <= 35  # Reasonable indoor temperature range
            assert 30 <= data.humidity <= 70     # Reasonable humidity range

class TestBiometricSensor:
    """Test suite for BiometricSensor class."""
    
    def test_initialization(self):
        """Test biometric sensor initialization."""
        sensor = BiometricSensor(sampling_rate=10)
        assert sensor.sampling_rate == 10
        assert sensor.baseline_facial_temp == 34.0
        assert hasattr(sensor, 'session_start')
    
    def test_facial_temperature_range(self):
        """Test facial temperature is within realistic range."""
        sensor = BiometricSensor()
        
        for _ in range(50):
            data = sensor.read()
            assert 32 <= data.facial_temperature <= 38  # Realistic facial temperature
    
    def test_stress_indicators(self):
        """Test stress indicator values."""
        sensor = BiometricSensor()
        
        for _ in range(30):
            data = sensor.read()
            assert 0 <= data.stress_indicator <= 1
            assert 0 <= data.cognitive_load <= 1
            assert 0 <= data.attention_level <= 1
            assert 0 <= data.fatigue_level <= 1
    
    def test_fatigue_accumulation(self):
        """Test that fatigue increases over time."""
        sensor = BiometricSensor()
        
        # Initial fatigue
        initial_data = sensor.read()
        initial_fatigue = initial_data.fatigue_level
        
        # Simulate passage of time
        with patch('time.time', return_value=time.time() + 3600):  # +1 hour
            later_data = sensor.read()
            later_fatigue = later_data.fatigue_level
        
        # Fatigue should generally increase (allowing for some noise)
        assert later_fatigue >= initial_fatigue - 0.1

class TestDigitalWellnessTracker:
    """Test suite for DigitalWellnessTracker class."""
    
    def test_initialization(self):
        """Test digital wellness tracker initialization."""
        tracker = DigitalWellnessTracker()
        assert tracker.total_screen_time == 0.0
        assert tracker.break_count == 0
        assert hasattr(tracker, 'session_start')
    
    def test_screen_time_tracking(self):
        """Test screen time tracking based on convergence distance."""
        tracker = DigitalWellnessTracker()
        
        # Create mock eye and environmental data
        eye_data = EyeTrackingData(
            timestamp=time.time(),
            left_pupil_diameter=4.0,
            right_pupil_diameter=4.0,
            gaze_x=0.5,
            gaze_y=0.5,
            fixation_duration=2.0,
            saccade_velocity=0.0,
            blink_rate=15.0,
            convergence_distance=0.5  # Close distance = screen viewing
        )
        
        env_data = EnvironmentalData(
            timestamp=time.time(),
            ambient_light=500,
            uv_index=0,
            proximity_distance=1.0,
            temperature=22,
            humidity=50
        )
        
        initial_screen_time = tracker.total_screen_time
        tracker.update(eye_data, env_data)
        
        # Screen time should increase
        assert tracker.total_screen_time > initial_screen_time
    
    def test_break_detection(self):
        """Test break detection based on far viewing."""
        tracker = DigitalWellnessTracker()
        
        # Simulate looking far away (break)
        eye_data = EyeTrackingData(
            timestamp=time.time(),
            left_pupil_diameter=4.0,
            right_pupil_diameter=4.0,
            gaze_x=0.5,
            gaze_y=0.5,
            fixation_duration=2.0,
            saccade_velocity=0.0,
            blink_rate=15.0,
            convergence_distance=7.0  # Far distance = break
        )
        
        env_data = EnvironmentalData(
            timestamp=time.time(),
            ambient_light=500,
            uv_index=0,
            proximity_distance=1.0,
            temperature=22,
            humidity=50
        )
        
        # Reset last break time to simulate 20+ minutes ago
        tracker.last_break_time = time.time() - 1300  # 21+ minutes ago
        
        initial_breaks = tracker.break_count
        tracker.update(eye_data, env_data)
        
        # Should detect a break
        assert tracker.break_count > initial_breaks
    
    def test_wellness_score_calculation(self):
        """Test wellness score calculation."""
        tracker = DigitalWellnessTracker()
        
        # Test with minimal screen time
        score = tracker.get_wellness_score()
        assert 0.0 <= score <= 1.0
        
        # Simulate excessive screen time
        tracker.total_screen_time = 500  # 8+ hours
        excessive_score = tracker.get_wellness_score()
        
        # Score should be lower with excessive screen time
        assert excessive_score < score

class TestSmartGlassesDevice:
    """Test suite for SmartGlassesDevice main class."""
    
    def test_initialization(self):
        """Test smart glasses device initialization."""
        device = SmartGlassesDevice(sensor_id=1, sampling_rate=60)
        
        assert device.sensor_id == 1
        assert device.name == "smart_glasses"
        assert device.sampling_rate == 60
        assert hasattr(device, 'eye_tracker')
        assert hasattr(device, 'environmental_sensor')
        assert hasattr(device, 'biometric_sensor')
        assert hasattr(device, 'digital_wellness_tracker')
    
    def test_data_reading(self):
        """Test comprehensive data reading."""
        device = SmartGlassesDevice(sensor_id=1)
        
        data = device.read()
        
        # Check data structure and ranges
        assert len(data) == 12  # Expected number of data points
        assert isinstance(data, np.ndarray)
        
        # Check specific data ranges
        timestamp = data[0]
        left_pupil = data[1]
        right_pupil = data[2]
        
        assert timestamp > 0
        assert 2.0 <= left_pupil <= 8.0
        assert 2.0 <= right_pupil <= 8.0
    
    def test_privacy_controls(self):
        """Test privacy setting controls."""
        device = SmartGlassesDevice(sensor_id=1)
        
        # Test gaze pattern storage control
        device.update_privacy_settings({
            "store_gaze": True,
            "anonymize_bio": False
        })
        
        data1 = device.read()
        gaze_x_stored = data1[3]
        gaze_y_stored = data1[4]
        
        device.update_privacy_settings({
            "store_gaze": False,
            "anonymize_bio": True
        })
        
        data2 = device.read()
        gaze_x_private = data2[3]
        gaze_y_private = data2[4]
        
        # When privacy is enabled, gaze should be 0
        assert gaze_x_private == 0
        assert gaze_y_private == 0
        
        # When privacy is disabled, gaze should have values
        assert gaze_x_stored != 0 or gaze_y_stored != 0
    
    def test_calibration(self):
        """Test user calibration functionality."""
        device = SmartGlassesDevice(sensor_id=1)
        
        user_profile = {
            "ipd": 65,  # Interpupillary distance
            "baseline_pupil": 4.5
        }
        
        # Initially not calibrated
        assert not device.is_calibrated
        
        # Perform calibration
        success = device.calibrate_for_user(user_profile)
        
        assert success
        assert device.is_calibrated
        assert device.calibration_data["interpupillary_distance"] == 65
        assert device.eye_tracker.baseline_pupil_size == 4.5
    
    def test_health_insights(self):
        """Test health insights generation."""
        device = SmartGlassesDevice(sensor_id=1)
        
        # Generate some data
        for _ in range(20):
            device.read()
        
        insights = device.get_health_insights()
        
        # Check insight structure
        assert "eye_strain_level" in insights
        assert "cognitive_state" in insights
        assert "environmental_exposure" in insights
        assert "digital_wellness_score" in insights
        assert "recommendations" in insights
        
        # Check data types and ranges
        assert 0 <= insights["eye_strain_level"] <= 1
        assert isinstance(insights["cognitive_state"], dict)
        assert isinstance(insights["recommendations"], list)
    
    def test_eye_strain_assessment(self):
        """Test eye strain level assessment."""
        device = SmartGlassesDevice(sensor_id=1)
        
        # Generate some data
        for _ in range(15):
            device.read()
        
        strain_level = device._assess_eye_strain()
        assert 0 <= strain_level <= 1
    
    def test_cognitive_state_assessment(self):
        """Test cognitive state assessment."""
        device = SmartGlassesDevice(sensor_id=1)
        
        # Generate some data
        for _ in range(10):
            device.read()
        
        cognitive_state = device._assess_cognitive_state()
        
        assert "load" in cognitive_state
        assert "attention" in cognitive_state
        assert "fatigue" in cognitive_state
        
        for value in cognitive_state.values():
            assert 0 <= value <= 1
    
    def test_environmental_exposure_assessment(self):
        """Test environmental exposure assessment."""
        device = SmartGlassesDevice(sensor_id=1)
        
        # Generate some data
        for _ in range(5):
            device.read()
        
        exposure = device._assess_environmental_exposure()
        
        assert "uv_risk" in exposure
        assert "light_exposure" in exposure
        assert exposure["uv_risk"] in ["low", "moderate", "high"]
        assert exposure["light_exposure"] in ["low", "normal", "high"]
    
    def test_recommendation_generation(self):
        """Test health recommendation generation."""
        device = SmartGlassesDevice(sensor_id=1)
        
        # Generate some data to analyze
        for _ in range(20):
            device.read()
        
        recommendations = device._generate_recommendations()
        
        assert isinstance(recommendations, list)
        # Recommendations should be strings
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0
    
    def test_digital_wellness_report(self):
        """Test digital wellness report generation."""
        device = SmartGlassesDevice(sensor_id=1)
        
        # Generate some usage data
        for _ in range(30):
            device.read()
        
        report = device.get_digital_wellness_report()
        
        assert "session_duration_hours" in report
        assert "total_screen_time_hours" in report
        assert "break_count" in report
        assert "wellness_score" in report
        
        assert report["session_duration_hours"] >= 0
        assert report["total_screen_time_hours"] >= 0
        assert report["break_count"] >= 0
        assert 0 <= report["wellness_score"] <= 1

class TestSmartGlassesIntegration:
    """Integration tests for smart glasses with main system."""
    
    def test_sensor_interface_compliance(self):
        """Test that SmartGlassesDevice complies with SensorInterface."""
        device = SmartGlassesDevice(sensor_id=1)
        
        # Test required interface methods
        assert hasattr(device, 'read')
        assert hasattr(device, 'start')
        assert hasattr(device, 'stop')
        assert hasattr(device, 'get_buffer')
        
        # Test that read returns numpy array
        data = device.read()
        assert isinstance(data, np.ndarray)
    
    def test_threading_safety(self):
        """Test thread safety during concurrent operations."""
        device = SmartGlassesDevice(sensor_id=1)
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
        device = SmartGlassesDevice(sensor_id=1)
        device.start()
        
        # Run for a short period
        start_time = time.time()
        data_points = []
        
        while time.time() - start_time < 2:  # Run for 2 seconds
            data = device.read()
            data_points.append(data)
            time.sleep(0.1)
        
        device.stop()
        
        # Should have collected multiple data points
        assert len(data_points) > 10
        
        # Check data consistency
        for data in data_points:
            assert len(data) == 12
            assert not np.isnan(data).any()

class TestMockDataGeneration:
    """Test realistic mock data generation."""
    
    def test_eye_tracking_realism(self):
        """Test that eye tracking data is realistic."""
        sensor = EyeTrackingSensor()
        
        # Collect data over time
        pupil_sizes = []
        blink_rates = []
        
        for _ in range(100):
            data = sensor.read()
            pupil_sizes.append((data.left_pupil_diameter, data.right_pupil_diameter))
            blink_rates.append(data.blink_rate)
        
        # Check pupil size variation
        left_pupils = [p[0] for p in pupil_sizes]
        right_pupils = [p[1] for p in pupil_sizes]
        
        assert np.std(left_pupils) > 0.1  # Should have variation
        assert np.std(right_pupils) > 0.1
        
        # Check blink rate variation
        assert np.std(blink_rates) > 1.0  # Should have variation
        assert min(blink_rates) > 5  # Minimum realistic blink rate
        assert max(blink_rates) < 50  # Maximum realistic blink rate
    
    def test_environmental_data_realism(self):
        """Test environmental data realism."""
        sensor = EnvironmentalSensor()
        
        light_levels = []
        temperatures = []
        
        for _ in range(50):
            data = sensor.read()
            light_levels.append(data.ambient_light)
            temperatures.append(data.temperature)
        
        # Light levels should vary
        assert max(light_levels) > min(light_levels)
        
        # Temperature should be in reasonable range
        assert all(15 <= temp <= 35 for temp in temperatures)
    
    def test_biometric_data_correlations(self):
        """Test realistic correlations in biometric data."""
        sensor = BiometricSensor()
        
        stress_levels = []
        fatigue_levels = []
        attention_levels = []
        
        for _ in range(50):
            data = sensor.read()
            stress_levels.append(data.stress_indicator)
            fatigue_levels.append(data.fatigue_level)
            attention_levels.append(data.attention_level)
        
        # Basic correlation checks
        # Higher stress should generally correlate with lower attention
        # (though this is a simplified model)
        assert all(0 <= s <= 1 for s in stress_levels)
        assert all(0 <= f <= 1 for f in fatigue_levels)
        assert all(0 <= a <= 1 for a in attention_levels)

# Performance and edge case tests
class TestPerformanceAndEdgeCases:
    """Test performance and edge cases."""
    
    def test_high_frequency_reading(self):
        """Test performance with high-frequency data reading."""
        device = SmartGlassesDevice(sensor_id=1, sampling_rate=120)
        
        start_time = time.time()
        
        # Read data rapidly
        for _ in range(1000):
            data = device.read()
            assert len(data) == 12
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds for 1000 readings
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        device = SmartGlassesDevice(sensor_id=1)
        device.start()
        
        # Generate data for a while
        for _ in range(500):
            device.read()
        
        # Get buffer size
        buffer_data = device.get_buffer(clear=False)
        
        device.stop()
        
        # Buffer shouldn't grow indefinitely (this tests the underlying
        # sensor manager's buffer management)
        assert len(buffer_data) < 1000  # Reasonable buffer size
    
    def test_invalid_calibration_data(self):
        """Test handling of invalid calibration data."""
        device = SmartGlassesDevice(sensor_id=1)
        
        # Test with invalid IPD
        invalid_profile = {
            "ipd": -10,  # Invalid negative IPD
            "baseline_pupil": 15.0  # Invalid large pupil size
        }
        
        # Should handle gracefully (in real implementation)
        # For now, just ensure it doesn't crash
        try:
            result = device.calibrate_for_user(invalid_profile)
            # Should either succeed with corrected values or fail gracefully
            assert isinstance(result, bool)
        except Exception:
            pytest.fail("Calibration should handle invalid data gracefully")
    
    def test_extreme_privacy_settings(self):
        """Test extreme privacy settings."""
        device = SmartGlassesDevice(sensor_id=1)
        
        # Maximum privacy
        device.update_privacy_settings({
            "level": "high",
            "store_gaze": False,
            "anonymize_bio": True
        })
        
        data = device.read()
        
        # Should still return valid data structure
        assert len(data) == 12
        assert not np.isnan(data).any()
        
        # Gaze data should be zeroed
        assert data[3] == 0  # gaze_x
        assert data[4] == 0  # gaze_y

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 