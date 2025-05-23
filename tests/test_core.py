"""
Tests for OpenWearables Core Architecture

This module contains comprehensive tests for the core functionality
of the OpenWearables platform.
"""

import os
import sys
import json
import time
import tempfile
import pytest
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openwearables.core.architecture import OpenWearablesCore
from openwearables.core.sensor_manager import SensorManager, SensorInterface
from openwearables.core.data_processor import DataProcessor
from openwearables.core.health_analyzer import HealthAnalyzer
from openwearables.core.privacy import PrivacyManager

class TestOpenWearablesCore:
    """Test suite for OpenWearablesCore class."""
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "database": {"path": "test_wearables.db"},
                "sensors": ["ecg", "ppg", "accelerometer"],
                "sampling_rates": {
                    "ecg": 250,
                    "ppg": 100,
                    "accelerometer": 50
                },
                "models": {
                    "arrhythmia_detection": "test/model",
                    "stress_analysis": "test/stress"
                },
                "processing": {
                    "window_size": 5,
                    "overlap": 0.5,
                    "features": ["time_domain", "frequency_domain"]
                },
                "privacy": {
                    "encryption": True,
                    "anonymization": True,
                    "data_retention": 30
                },
                "logging": {"level": "DEBUG", "file": "test.log"},
                "user_profile": {
                    "name": "Test User",
                    "age": 30,
                    "gender": "test"
                }
            }
            json.dump(config, f)
            f.flush()  # Ensure content is written to disk
            temp_filename = f.name
        
        yield temp_filename
        
        # Cleanup
        try:
            os.unlink(temp_filename)
            if os.path.exists("test_wearables.db"):
                os.unlink("test_wearables.db")
        except:
            pass
    
    def test_core_initialization(self, temp_config):
        """Test core system initialization."""
        core = OpenWearablesCore(temp_config, auto_optimize=False)
        
        assert core.config_path == temp_config
        assert isinstance(core.config, dict)
        assert core.device in ["cpu", "mlx", "cuda", "cuda_amp", "mps", "torch_cpu"]
        assert hasattr(core, 'sensor_manager')
        assert hasattr(core, 'data_processor')
        assert hasattr(core, 'health_analyzer')
        assert hasattr(core, 'privacy_manager')
        assert isinstance(core.models, dict)
    
    def test_device_detection(self, temp_config):
        """Test hardware device detection."""
        core = OpenWearablesCore(temp_config, auto_optimize=True)
        
        # Device should be detected automatically
        assert core.device is not None
        assert isinstance(core.device, str)
        
        # Test manual device setting
        core_manual = OpenWearablesCore(temp_config, auto_optimize=False)
        assert core_manual.device == "cpu"
    
    def test_config_loading(self, temp_config):
        """Test configuration loading and validation."""
        core = OpenWearablesCore(temp_config)
        
        # Check that config was loaded correctly
        assert "database" in core.config
        assert "sensors" in core.config
        assert "models" in core.config
        assert core.config["user_profile"]["name"] == "Test User"
    
    def test_database_initialization(self, temp_config):
        """Test database table creation."""
        core = OpenWearablesCore(temp_config)
        
        # Check that database file exists
        assert os.path.exists(core.db_path)
        
        # Check that tables were created
        conn = sqlite3.connect(core.db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            "sensors", "readings", "analysis_results", 
            "health_metrics", "alerts", "sessions"
        ]
        
        for table in expected_tables:
            assert table in tables
        
        conn.close()
    
    @patch('openwearables.core.architecture.SensorManager')
    def test_start_stop_system(self, mock_sensor_manager, temp_config):
        """Test starting and stopping the system."""
        # Setup mock
        mock_manager_instance = Mock()
        mock_sensor_manager.return_value = mock_manager_instance
        
        core = OpenWearablesCore(temp_config)
        
        # Test start
        assert not core.running
        core.start()
        assert core.running
        mock_manager_instance.start.assert_called_once()
        
        # Test stop
        core.stop()
        assert not core.running
        mock_manager_instance.stop.assert_called_once()
    
    def test_system_status(self, temp_config):
        """Test system status reporting."""
        core = OpenWearablesCore(temp_config)
        
        status = core.get_system_status()
        
        assert isinstance(status, dict)
        assert "running" in status
        assert "device" in status
        assert "sensors" in status
        assert "models" in status
        assert "database_path" in status
        assert "config_path" in status
        assert "version" in status
        
        assert status["version"] == "1.0.0"
        assert status["config_path"] == temp_config
    
    def test_config_update(self, temp_config):
        """Test configuration updates."""
        core = OpenWearablesCore(temp_config)
        
        # Test updating configuration
        new_config = {"test_key": "test_value"}
        result = core.update_config(new_config)
        
        assert result is True
        assert "test_key" in core.config
        assert core.config["test_key"] == "test_value"
        
        # Verify config was saved to file
        with open(temp_config, 'r') as f:
            saved_config = json.load(f)
        assert "test_key" in saved_config
    
    def test_data_export(self, temp_config):
        """Test data export functionality."""
        core = OpenWearablesCore(temp_config)
        
        # Add some test data to database
        conn = sqlite3.connect(core.db_path)
        cursor = conn.cursor()
        
        # Insert test reading
        cursor.execute(
            "INSERT INTO readings (sensor_id, timestamp, value, processed) VALUES (?, ?, ?, ?)",
            (1, time.time(), b"test_data", 0)
        )
        
        # Insert test analysis
        cursor.execute(
            "INSERT INTO analysis_results (timestamp, analysis_type, result, confidence) VALUES (?, ?, ?, ?)",
            (time.time(), "test_analysis", "test_result", 0.95)
        )
        
        conn.commit()
        conn.close()
        
        # Test JSON export
        start_time = time.time() - 3600  # 1 hour ago
        end_time = time.time()
        
        json_data = core.export_data(start_time, end_time, "json")
        assert json_data is not None
        assert isinstance(json_data, str)
        
        # Parse and validate JSON
        data = json.loads(json_data)
        assert "export_timestamp" in data
        assert "readings" in data
        assert "analyses" in data
    
    def test_health_summary(self, temp_config):
        """Test health summary generation."""
        core = OpenWearablesCore(temp_config)
        
        # Add some test health metrics
        conn = sqlite3.connect(core.db_path)
        cursor = conn.cursor()
        
        current_time = time.time()
        for i in range(5):
            cursor.execute(
                "INSERT INTO health_metrics (timestamp, metric_type, value, units) VALUES (?, ?, ?, ?)",
                (current_time - i * 3600, "heart_rate", 70 + i, "bpm")
            )
        
        conn.commit()
        conn.close()
        
        # Generate health summary
        summary = core.get_health_summary(days=1)
        
        assert isinstance(summary, dict)
        assert "period_days" in summary
        assert "metrics" in summary
        assert "analyses" in summary
        assert "alerts" in summary
        assert "generated_at" in summary
        
        # Check that heart rate metrics were aggregated
        if "heart_rate" in summary["metrics"]:
            hr_data = summary["metrics"]["heart_rate"]
            assert "average" in hr_data
            assert "minimum" in hr_data
            assert "maximum" in hr_data
            assert "count" in hr_data

class TestSensorManager:
    """Test suite for SensorManager class."""
    
    def test_sensor_manager_initialization(self):
        """Test sensor manager initialization."""
        sensor_types = ["ecg", "ppg", "accelerometer"]
        sampling_rates = {"ecg": 250, "ppg": 100, "accelerometer": 50}
        
        manager = SensorManager(
            sensor_types=sensor_types,
            sampling_rates=sampling_rates,
            config={}
        )
        
        assert len(manager.sensors) == len(sensor_types)
        for sensor_type in sensor_types:
            assert sensor_type in manager.sensors
    
    def test_sensor_interface_abstract(self):
        """Test that SensorInterface is properly abstract."""
        with pytest.raises(TypeError):
            # Should not be able to instantiate abstract class
            SensorInterface(1, "test", 50)

class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    def test_data_processor_initialization(self):
        """Test data processor initialization."""
        config = {
            "window_size": 10,
            "overlap": 0.5,
            "features": ["time_domain", "frequency_domain"]
        }
        
        processor = DataProcessor(config)
        
        assert processor.window_size == 10
        assert processor.overlap == 0.5
        assert "time_domain" in processor.features
        assert "frequency_domain" in processor.features
    
    def test_process_batch_empty_data(self):
        """Test processing empty sensor data."""
        processor = DataProcessor()
        
        empty_data = {}
        result = processor.process_batch(empty_data)
        
        assert isinstance(result, dict)

class TestHealthAnalyzer:
    """Test suite for HealthAnalyzer class."""
    
    def test_health_analyzer_initialization(self):
        """Test health analyzer initialization."""
        config = {"test": "value"}
        device = "cpu"
        
        analyzer = HealthAnalyzer(config, device)
        
        assert analyzer.config == config
        assert analyzer.device == device

class TestPrivacyManager:
    """Test suite for PrivacyManager class."""
    
    def test_privacy_manager_initialization(self):
        """Test privacy manager initialization."""
        config = {
            "encryption": True,
            "anonymization": True,
            "data_retention": 90
        }
        
        manager = PrivacyManager(config)
        
        assert manager.anonymization_enabled is True
        assert manager.data_retention_days == 90
    
    def test_sanitize_output(self):
        """Test output sanitization."""
        manager = PrivacyManager()
        
        test_data = {
            "user_id": "12345",
            "heart_rate": 72,
            "analysis": "normal",
            "device_id": "device123"
        }
        
        sanitized = manager.sanitize_output(test_data)
        
        # Check that sensitive keys are removed
        assert "user_id" not in sanitized
        assert "device_id" not in sanitized
        
        # Check that non-sensitive data remains
        assert "heart_rate" in sanitized
        assert "analysis" in sanitized
    
    def test_anonymize_identifier(self):
        """Test identifier anonymization."""
        manager = PrivacyManager({"anonymization": True})
        
        original_id = "user123"
        anonymized = manager.anonymize_identifier(original_id)
        
        # Should not be the same as original
        assert anonymized != original_id
        
        # Should be consistent (same input = same output)
        anonymized2 = manager.anonymize_identifier(original_id)
        assert anonymized == anonymized2
    
    def test_data_retention(self):
        """Test data retention policy."""
        manager = PrivacyManager({"data_retention": 30})
        
        # Test recent data (should not be expired)
        recent_timestamp = time.time() - (20 * 86400)  # 20 days ago
        assert not manager.is_data_expired(recent_timestamp)
        
        # Test old data (should be expired)
        old_timestamp = time.time() - (40 * 86400)  # 40 days ago
        assert manager.is_data_expired(old_timestamp)

class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create a complete integrated system for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "database": {"path": "integration_test.db"},
                "sensors": ["ecg", "ppg"],
                "sampling_rates": {"ecg": 250, "ppg": 100},
                "models": {},
                "processing": {"window_size": 5, "overlap": 0.5, "features": ["time_domain"]},
                "privacy": {"encryption": False, "anonymization": False, "data_retention": 30},
                "logging": {"level": "ERROR", "file": "integration_test.log"}
            }
            json.dump(config, f)
            f.flush()
            temp_filename = f.name

        core = OpenWearablesCore(temp_filename, auto_optimize=False)
        
        yield core
        
        # Cleanup
        try:
            os.unlink(temp_filename)
            if os.path.exists("integration_test.db"):
                os.unlink("integration_test.db")
            if os.path.exists("integration_test.log"):
                os.unlink("integration_test.log")
        except:
            pass
    
    @patch('openwearables.core.sensor_manager.SensorManager.start')
    @patch('openwearables.core.sensor_manager.SensorManager.stop')
    def test_system_lifecycle(self, mock_stop, mock_start, integrated_system):
        """Test complete system lifecycle."""
        core = integrated_system
        
        # Test initial state
        assert not core.is_running()
        
        # Test start
        core.start()
        assert core.running
        mock_start.assert_called_once()
        
        # Test status while running
        status = core.get_system_status()
        assert isinstance(status, dict)
        
        # Test stop
        core.stop()
        assert not core.running
        mock_stop.assert_called_once()

# Performance tests
@pytest.mark.slow
class TestPerformance:
    """Performance tests for the OpenWearables platform."""
    
    def test_core_initialization_performance(self):
        """Test that core initialization completes within reasonable time."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "database": {"path": "perf_test.db"},
                "sensors": ["ecg", "ppg", "accelerometer", "gyroscope", "temperature"],
                "sampling_rates": {"ecg": 250, "ppg": 100, "accelerometer": 50, "gyroscope": 50, "temperature": 1},
                "models": {},
                "processing": {"window_size": 10, "overlap": 0.5, "features": ["time_domain", "frequency_domain"]},
                "privacy": {"encryption": True, "anonymization": True, "data_retention": 90},
                "logging": {"level": "ERROR", "file": "perf_test.log"}
            }
            json.dump(config, f)
            f.flush()
            temp_filename = f.name

        start_time = time.time()
        core = OpenWearablesCore(temp_filename, auto_optimize=False)
        end_time = time.time()
        
        initialization_time = end_time - start_time
        
        # Should initialize within 10 seconds
        assert initialization_time < 10.0
        
        # Cleanup
        try:
            os.unlink(temp_filename)
            if os.path.exists("perf_test.db"):
                os.unlink("perf_test.db")
            if os.path.exists("perf_test.log"):
                os.unlink("perf_test.log")
        except:
            pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 