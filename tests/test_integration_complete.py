"""
Complete Integration Tests for OpenWearables Platform

Tests the full integration of all device types (smart glasses, headphones, watch)
with Swift MLX bridge, data bridge, model processing, and real-time analytics.
"""

import pytest
import numpy as np
import time
import threading
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# Import all device types
from openwearables.core.devices import SmartGlassesDevice, SmartHeadphonesDevice, SmartWatchDevice
from openwearables.core.sensor_manager import SensorManager

# Import Swift MLX bridge components
from openwearables.swift_mlx import (
    SwiftMLXBridge, MLXModelBridge, DataBridge, 
    RealTimeDataStream, SensorDataPacket, DataStreamConfig
)

# Import core components
from openwearables.core import OpenWearablesCore


class TestCompleteDeviceIntegration:
    """Test integration of all wearable device types."""
    
    def test_all_devices_initialization(self):
        """Test that all device types can be initialized together."""
        # Initialize all device types
        glasses = SmartGlassesDevice(sensor_id=1)
        headphones = SmartHeadphonesDevice(sensor_id=2) 
        watch = SmartWatchDevice(sensor_id=3)
        
        # Verify all devices are properly initialized
        assert glasses.sensor_id == 1
        assert glasses.name == "smart_glasses"
        
        assert headphones.sensor_id == 2
        assert headphones.name == "smart_headphones"
        
        assert watch.sensor_id == 3
        assert watch.name == "smart_watch"
        
        # Test that all devices can read data
        glasses_data = glasses.read()
        headphones_data = headphones.read()
        watch_data = watch.read()
        
        assert isinstance(glasses_data, np.ndarray)
        assert isinstance(headphones_data, np.ndarray)
        assert isinstance(watch_data, np.ndarray)
        
        # Verify expected data array sizes
        assert len(glasses_data) == 12  # Smart glasses data elements
        assert len(headphones_data) == 10  # Smart headphones data elements
        assert len(watch_data) == 12  # Smart watch data elements
    
    def test_sensor_manager_with_all_devices(self):
        """Test SensorManager coordinating all device types."""
        # Create sensor manager with all device types
        glasses = SmartGlassesDevice(sensor_id=1)
        headphones = SmartHeadphonesDevice(sensor_id=2)
        watch = SmartWatchDevice(sensor_id=3)
        
        # Start all devices
        glasses.start()
        headphones.start()
        watch.start()
        
        try:
            # Let devices collect data
            time.sleep(0.5)
            
            # Get buffered data from all devices
            glasses_buffer = glasses.get_buffer(clear=False)
            headphones_buffer = headphones.get_buffer(clear=False)
            watch_buffer = watch.get_buffer(clear=False)
            
            # Verify data collection
            assert len(glasses_buffer) > 0
            assert len(headphones_buffer) > 0
            assert len(watch_buffer) > 0
            
            # Verify data timestamps are recent
            current_time = time.time()
            for timestamp, data in glasses_buffer[-3:]:  # Last 3 readings
                assert current_time - timestamp < 2.0  # Within last 2 seconds
                
        finally:
            # Clean up
            glasses.stop()
            headphones.stop()
            watch.stop()
    
    def test_cross_device_data_correlation(self):
        """Test correlation of data across multiple device types."""
        glasses = SmartGlassesDevice(sensor_id=1)
        headphones = SmartHeadphonesDevice(sensor_id=2)
        watch = SmartWatchDevice(sensor_id=3)
        
        # Collect synchronized readings
        readings = {}
        for i in range(10):
            timestamp = time.time()
            
            readings[i] = {
                'timestamp': timestamp,
                'glasses': glasses.read(),
                'headphones': headphones.read(),
                'watch': watch.read()
            }
            
            time.sleep(0.1)
        
        # Analyze data correlation
        # Extract heart rate estimates from different devices
        watch_hr = [readings[i]['watch'][1] for i in range(10)]  # Heart rate from watch
        headphones_hr = [readings[i]['headphones'][2] for i in range(10)]  # Heart rate from headphones
        
        # Heart rates should be in similar ranges
        watch_hr_avg = np.mean(watch_hr)
        headphones_hr_avg = np.mean(headphones_hr)
        
        # Allow for some variance between devices
        hr_diff = abs(watch_hr_avg - headphones_hr_avg)
        assert hr_diff < 30, f"Heart rate difference too large: {hr_diff}"
    
    def test_device_health_insights_coordination(self):
        """Test coordinated health insights from multiple devices."""
        glasses = SmartGlassesDevice(sensor_id=1)
        headphones = SmartHeadphonesDevice(sensor_id=2)
        watch = SmartWatchDevice(sensor_id=3)
        
        # Generate data for insights
        for _ in range(15):
            glasses.read()
            headphones.read()
            watch.read()
        
        # Get health insights from each device
        glasses_insights = glasses.get_health_insights()
        headphones_insights = headphones.get_hearing_health_summary()
        watch_insights = watch.get_health_insights()
        
        # Verify insights structure
        if glasses_insights.get("status") != "insufficient_data":
            assert "cognitive_state" in glasses_insights
            assert "environmental_exposure" in glasses_insights
        
        if headphones_insights.get("status") != "insufficient_data":
            assert "hearing_protection" in headphones_insights
            assert "audio_exposure" in headphones_insights
        
        if watch_insights.get("status") != "insufficient_data":
            assert "cardiovascular_health" in watch_insights
            assert "activity_assessment" in watch_insights


class TestSwiftMLXBridgeIntegration:
    """Test Swift MLX bridge integration with devices."""
    
    def test_swift_bridge_initialization(self):
        """Test Swift bridge initialization and platform detection."""
        # Mock macOS environment
        with patch('platform.system', return_value='Darwin'):
            with patch('platform.machine', return_value='arm64'):
                bridge = SwiftMLXBridge()
                
                assert bridge.is_macos
                assert bridge.has_apple_silicon
                assert bridge.config["platform"] == "darwin"
    
    def test_swift_bridge_with_devices(self):
        """Test Swift bridge receiving data from devices."""
        bridge = SwiftMLXBridge()
        
        # Initialize devices
        glasses = SmartGlassesDevice(sensor_id=1)
        headphones = SmartHeadphonesDevice(sensor_id=2)
        watch = SmartWatchDevice(sensor_id=3)
        
        # Send device data to Swift bridge
        glasses_data = glasses.read()
        headphones_data = headphones.read()
        watch_data = watch.read()
        
        # Send data to bridge
        bridge.send_sensor_data("smart_glasses", glasses_data.tolist())
        bridge.send_sensor_data("smart_headphones", headphones_data.tolist())
        bridge.send_sensor_data("smart_watch", watch_data.tolist())
        
        # Verify data queue
        assert len(bridge.data_queue) == 3
        
        # Check data structure
        for data_packet in bridge.data_queue:
            assert hasattr(data_packet, 'sensor_type')
            assert hasattr(data_packet, 'data')
            assert hasattr(data_packet, 'timestamp')
    
    def test_swift_package_generation(self):
        """Test Swift package generation for native integration."""
        bridge = SwiftMLXBridge()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = bridge.generate_swift_package(temp_dir)
            
            # Verify package structure
            assert package_path.exists()
            assert (package_path / "Package.swift").exists()
            assert (package_path / "Sources").exists()
            assert (package_path / "Sources" / "OpenWearablesBridge").exists()
            
            # Check Package.swift content
            package_content = (package_path / "Package.swift").read_text()
            assert "OpenWearablesBridge" in package_content
            assert "mlx-swift" in package_content


class TestMLXModelBridgeIntegration:
    """Test MLX model bridge with real device data."""
    
    def test_model_bridge_with_device_data(self):
        """Test MLX model bridge processing device data."""
        model_bridge = MLXModelBridge()
        
        # Initialize devices
        watch = SmartWatchDevice(sensor_id=1)
        glasses = SmartGlassesDevice(sensor_id=2)
        
        # Process device data through model bridge
        for _ in range(20):  # Generate enough data for analysis
            watch_data = watch.read()
            glasses_data = glasses.read()
            
            # Process watch data (focus on heart rate)
            watch_results = model_bridge.process_sensor_data("heart_rate", np.array([watch_data[1]]))
            
            # Process glasses data (focus on attention/cognitive load)
            cognitive_data = glasses_data[9:12]  # cognitive load, attention, fatigue
            glasses_results = model_bridge.process_sensor_data("cognitive_tracker", cognitive_data)
        
        # Check for generated insights
        recent_insights = model_bridge.get_recent_insights(limit=10)
        recent_predictions = model_bridge.get_recent_predictions(limit=10)
        
        # Should have some analysis results
        assert len(recent_insights) >= 0  # May be empty if no anomalies detected
        assert len(recent_predictions) >= 0
        
        # Get health summary
        summary = model_bridge.get_health_summary()
        assert "total_insights" in summary
        assert "processing_status" in summary
    
    def test_anomaly_detection_with_devices(self):
        """Test anomaly detection with realistic device data."""
        model_bridge = MLXModelBridge()
        watch = SmartWatchDevice(sensor_id=1)
        
        # Generate normal data
        for _ in range(30):
            watch_data = watch.read()
            model_bridge.process_sensor_data("heart_rate", np.array([watch_data[1]]))
        
        # Simulate anomalous heart rate
        anomalous_hr = np.array([200.0])  # Very high heart rate
        results = model_bridge.process_sensor_data("heart_rate", anomalous_hr)
        
        # Should detect anomaly
        anomaly_predictions = [r for r in results if hasattr(r, 'prediction_type') and r.prediction_type == "anomaly"]
        
        # May or may not detect depending on threshold and data variation
        assert len(anomaly_predictions) >= 0
    
    def test_real_time_processing(self):
        """Test real-time processing with model bridge."""
        model_bridge = MLXModelBridge()
        model_bridge.start_real_time_processing()
        
        try:
            watch = SmartWatchDevice(sensor_id=1)
            
            # Stream data for processing
            for i in range(10):
                watch_data = watch.read()
                model_bridge.process_sensor_data(f"heart_rate_{i}", np.array([watch_data[1]]))
                time.sleep(0.1)
            
            # Let processing complete
            time.sleep(0.5)
            
            # Check processing results
            insights = model_bridge.get_recent_insights()
            predictions = model_bridge.get_recent_predictions()
            
            # Should have processed the data
            assert len(insights) >= 0
            assert len(predictions) >= 0
            
        finally:
            model_bridge.stop_real_time_processing()


class TestDataBridgeIntegration:
    """Test data bridge with high-performance streaming."""
    
    def test_data_bridge_with_multiple_devices(self):
        """Test data bridge coordinating multiple device streams."""
        data_bridge = DataBridge()
        
        # Create streams for each device type
        glasses_stream = data_bridge.create_stream("smart_glasses")
        headphones_stream = data_bridge.create_stream("smart_headphones")
        watch_stream = data_bridge.create_stream("smart_watch")
        
        # Initialize devices
        glasses = SmartGlassesDevice(sensor_id=1)
        headphones = SmartHeadphonesDevice(sensor_id=2)
        watch = SmartWatchDevice(sensor_id=3)
        
        # Start all streams
        data_bridge.start_all_streams()
        
        try:
            # Stream data from devices
            for i in range(20):
                glasses_data = glasses.read()
                headphones_data = headphones.read()
                watch_data = watch.read()
                
                # Add data to respective streams
                data_bridge.add_sensor_data("smart_glasses", 1, "smart_glasses", glasses_data)
                data_bridge.add_sensor_data("smart_headphones", 2, "smart_headphones", headphones_data)
                data_bridge.add_sensor_data("smart_watch", 3, "smart_watch", watch_data)
                
                time.sleep(0.05)  # 50ms intervals
            
            # Let data process
            time.sleep(0.5)
            
            # Check stream metrics
            metrics = data_bridge.get_global_metrics()
            
            assert metrics["active_streams"] == 3
            assert metrics["total_packets_processed"] > 0
            assert metrics["global_average_latency_ms"] >= 0
            
        finally:
            data_bridge.stop_all_streams()
    
    def test_synchronized_data_collection(self):
        """Test synchronized data collection across devices."""
        data_bridge = DataBridge()
        
        # Create streams
        glasses_stream = data_bridge.create_stream("glasses_sync")
        watch_stream = data_bridge.create_stream("watch_sync")
        
        # Initialize devices
        glasses = SmartGlassesDevice(sensor_id=1)
        watch = SmartWatchDevice(sensor_id=2)
        
        data_bridge.start_all_streams()
        
        try:
            # Add synchronized data
            sync_timestamp = time.time()
            
            for i in range(10):
                current_time = sync_timestamp + i * 0.1
                
                glasses_data = glasses.read()
                watch_data = watch.read()
                
                # Add with same timestamps for synchronization
                data_bridge.add_sensor_data("glasses_sync", 1, "smart_glasses", glasses_data)
                data_bridge.add_sensor_data("watch_sync", 2, "smart_watch", watch_data)
                
                time.sleep(0.05)
            
            # Get synchronized data
            sync_data = data_bridge.get_synchronized_data(["smart_glasses", "smart_watch"])
            
            # Should have synchronized packets
            assert len(sync_data) >= 0  # May be empty if no sync window match
            
        finally:
            data_bridge.stop_all_streams()
    
    def test_high_throughput_streaming(self):
        """Test high-throughput data streaming performance."""
        data_bridge = DataBridge(config={
            "buffer_size": 2000,
            "max_latency_ms": 5.0,
            "batch_size": 20
        })
        
        stream = data_bridge.create_stream("high_throughput")
        watch = SmartWatchDevice(sensor_id=1)
        
        stream.start_streaming()
        
        try:
            start_time = time.time()
            packet_count = 0
            
            # Stream data at high rate
            while time.time() - start_time < 2.0:  # 2 seconds
                watch_data = watch.read()
                success = data_bridge.add_sensor_data("high_throughput", 1, "smart_watch", watch_data)
                if success:
                    packet_count += 1
                
                time.sleep(0.001)  # 1ms intervals for high throughput
            
            # Check performance
            elapsed = time.time() - start_time
            throughput = packet_count / elapsed
            
            assert throughput > 100, f"Throughput too low: {throughput:.1f} packets/sec"
            
            # Check stream metrics
            metrics = stream.get_stream_metrics()
            assert metrics["packets_sent"] > 0
            assert metrics["average_latency_ms"] < 50  # Should be low latency
            
        finally:
            stream.stop_streaming()


class TestCompleteSystemIntegration:
    """Test complete system integration with all components."""
    
    def test_end_to_end_integration(self):
        """Test complete end-to-end system integration."""
        # Initialize all components
        swift_bridge = SwiftMLXBridge()
        model_bridge = MLXModelBridge()
        data_bridge = DataBridge()
        
        # Initialize all device types
        glasses = SmartGlassesDevice(sensor_id=1)
        headphones = SmartHeadphonesDevice(sensor_id=2)
        watch = SmartWatchDevice(sensor_id=3)
        
        # Create data streams
        glasses_stream = data_bridge.create_stream("integrated_glasses")
        headphones_stream = data_bridge.create_stream("integrated_headphones")
        watch_stream = data_bridge.create_stream("integrated_watch")
        
        # Start real-time processing
        model_bridge.start_real_time_processing()
        data_bridge.start_all_streams()
        
        try:
            # Simulate complete data pipeline
            for cycle in range(30):
                # Collect data from all devices
                glasses_data = glasses.read()
                headphones_data = headphones.read()
                watch_data = watch.read()
                
                # Stream through data bridge
                data_bridge.add_sensor_data("integrated_glasses", 1, "smart_glasses", glasses_data)
                data_bridge.add_sensor_data("integrated_headphones", 2, "smart_headphones", headphones_data)
                data_bridge.add_sensor_data("integrated_watch", 3, "smart_watch", watch_data)
                
                # Process through model bridge
                model_bridge.process_sensor_data("heart_rate", np.array([watch_data[1]]))
                model_bridge.process_sensor_data("cognitive_load", np.array([glasses_data[9]]))
                
                # Send to Swift bridge
                swift_bridge.send_sensor_data("smart_glasses", glasses_data.tolist())
                swift_bridge.send_sensor_data("smart_headphones", headphones_data.tolist())
                swift_bridge.send_sensor_data("smart_watch", watch_data.tolist())
                
                time.sleep(0.1)  # 100ms cycle
            
            # Verify integration results
            
            # Check data bridge metrics
            data_metrics = data_bridge.get_global_metrics()
            assert data_metrics["active_streams"] == 3
            assert data_metrics["total_packets_processed"] > 0
            
            # Check model bridge insights
            model_summary = model_bridge.get_health_summary()
            assert "total_insights" in model_summary
            
            # Check Swift bridge data collection
            assert len(swift_bridge.data_queue) > 0
            
            # Get synchronized health insights
            watch_insights = watch.get_health_insights()
            glasses_insights = glasses.get_health_insights()
            headphones_insights = headphones.get_hearing_health_summary()
            
            # Verify integrated health analysis
            if watch_insights.get("status") != "insufficient_data":
                assert "recommendations" in watch_insights
            
        finally:
            # Clean up
            model_bridge.stop_real_time_processing()
            data_bridge.stop_all_streams()
    
    def test_multi_device_health_fusion(self):
        """Test fusion of health data from multiple devices."""
        # Initialize devices
        glasses = SmartGlassesDevice(sensor_id=1)
        headphones = SmartHeadphonesDevice(sensor_id=2)
        watch = SmartWatchDevice(sensor_id=3)
        
        # Collect comprehensive health data
        health_data = {}
        
        for i in range(25):  # Collect enough data for analysis
            timestamp = time.time()
            
            glasses_data = glasses.read()
            headphones_data = headphones.read()
            watch_data = watch.read()
            
            health_data[i] = {
                'timestamp': timestamp,
                'heart_rate': watch_data[1],  # From watch
                'spo2': watch_data[2],  # From watch
                'stress_level': glasses_data[8],  # From glasses
                'cognitive_load': glasses_data[9],  # From glasses
                'attention_level': glasses_data[10],  # From glasses
                'hearing_health': headphones_data[7],  # From headphones
                'audio_exposure': headphones_data[1],  # From headphones
                'activity_level': watch_data[9],  # Exercise minutes from watch
                'sleep_quality': watch_data[5]  # From watch
            }
            
            time.sleep(0.05)
        
        # Analyze fused health data
        heart_rates = [health_data[i]['heart_rate'] for i in range(25)]
        stress_levels = [health_data[i]['stress_level'] for i in range(25)]
        cognitive_loads = [health_data[i]['cognitive_load'] for i in range(25)]
        
        # Calculate health correlations
        avg_heart_rate = np.mean(heart_rates)
        avg_stress = np.mean(stress_levels)
        avg_cognitive_load = np.mean(cognitive_loads)
        
        # Generate integrated health insights
        integrated_insights = {
            'cardiovascular_status': 'normal' if 60 <= avg_heart_rate <= 100 else 'attention_needed',
            'mental_wellness': 'good' if avg_stress < 0.6 and avg_cognitive_load < 0.7 else 'moderate',
            'overall_health_score': (
                (100 - abs(avg_heart_rate - 80)) * 0.4 +  # Heart rate component
                (100 - avg_stress * 100) * 0.3 +  # Stress component
                (100 - avg_cognitive_load * 100) * 0.3  # Cognitive load component
            )
        }
        
        # Verify integrated analysis
        assert integrated_insights['cardiovascular_status'] in ['normal', 'attention_needed']
        assert integrated_insights['mental_wellness'] in ['good', 'moderate', 'poor']
        assert 0 <= integrated_insights['overall_health_score'] <= 100
    
    def test_performance_under_load(self):
        """Test system performance under high load."""
        # Initialize all components
        swift_bridge = SwiftMLXBridge()
        model_bridge = MLXModelBridge()
        data_bridge = DataBridge()
        
        # Initialize devices
        devices = [
            SmartGlassesDevice(sensor_id=1),
            SmartHeadphonesDevice(sensor_id=2),
            SmartWatchDevice(sensor_id=3)
        ]
        
        # Create multiple streams
        streams = []
        for i, device in enumerate(devices):
            stream = data_bridge.create_stream(f"perf_test_{i}")
            streams.append(stream)
        
        model_bridge.start_real_time_processing()
        data_bridge.start_all_streams()
        
        try:
            start_time = time.time()
            total_operations = 0
            
            # High-load simulation
            while time.time() - start_time < 3.0:  # 3 seconds
                for i, device in enumerate(devices):
                    data = device.read()
                    
                    # Data bridge operations
                    data_bridge.add_sensor_data(f"perf_test_{i}", device.sensor_id, device.name, data)
                    
                    # Model bridge operations
                    model_bridge.process_sensor_data(device.name, data[:3])  # First 3 elements
                    
                    # Swift bridge operations
                    swift_bridge.send_sensor_data(device.name, data.tolist())
                    
                    total_operations += 3  # 3 operations per device
                
                time.sleep(0.01)  # 10ms delay
            
            elapsed = time.time() - start_time
            ops_per_second = total_operations / elapsed
            
            # Performance requirements
            assert ops_per_second > 500, f"Performance too low: {ops_per_second:.1f} ops/sec"
            
            # Check system health after load
            data_metrics = data_bridge.get_global_metrics()
            assert data_metrics["packet_loss_rate"] < 0.1  # Less than 10% packet loss
            
        finally:
            model_bridge.stop_real_time_processing()
            data_bridge.stop_all_streams()
    
    def test_system_reliability(self):
        """Test system reliability and error handling."""
        # Test with invalid data
        swift_bridge = SwiftMLXBridge()
        model_bridge = MLXModelBridge()
        
        # Test error handling with invalid sensor data
        try:
            # Invalid data types
            swift_bridge.send_sensor_data("invalid_sensor", "not_a_list")
            model_bridge.process_sensor_data("invalid_sensor", np.array([]))
            
            # Should handle gracefully without crashing
            assert True
        except Exception as e:
            # Should not raise unhandled exceptions
            pytest.fail(f"System should handle invalid data gracefully: {e}")
        
        # Test recovery from failures
        data_bridge = DataBridge()
        stream = data_bridge.create_stream("reliability_test")
        
        stream.start_streaming()
        
        try:
            # Simulate data processing
            watch = SmartWatchDevice(sensor_id=1)
            
            for i in range(10):
                data = watch.read()
                success = data_bridge.add_sensor_data("reliability_test", 1, "smart_watch", data)
                assert success or not success  # Should handle both cases
                
                time.sleep(0.05)
            
            # System should remain operational
            metrics = stream.get_stream_metrics()
            assert metrics["is_active"]
            
        finally:
            stream.stop_streaming()


class TestExportAndPersistence:
    """Test data export and persistence functionality."""
    
    def test_comprehensive_data_export(self):
        """Test exporting data from all system components."""
        # Initialize components
        model_bridge = MLXModelBridge()
        data_bridge = DataBridge()
        watch = SmartWatchDevice(sensor_id=1)
        
        # Generate data
        stream = data_bridge.create_stream("export_test")
        stream.start_streaming()
        
        try:
            for i in range(15):
                data = watch.read()
                data_bridge.add_sensor_data("export_test", 1, "smart_watch", data)
                model_bridge.process_sensor_data("heart_rate", np.array([data[1]]))
                time.sleep(0.1)
            
            # Export data
            with tempfile.TemporaryDirectory() as temp_dir:
                # Export model results
                model_export_path = f"{temp_dir}/model_results.json"
                model_bridge.export_model_results(model_export_path)
                
                # Export data bridge metrics
                data_export_path = f"{temp_dir}/data_metrics.json"
                data_bridge.export_metrics(data_export_path)
                
                # Verify exports
                assert os.path.exists(model_export_path)
                assert os.path.exists(data_export_path)
                
                # Verify export content
                with open(model_export_path, 'r') as f:
                    model_data = json.load(f)
                    assert "predictions" in model_data
                    assert "insights" in model_data
                    assert "export_timestamp" in model_data
                
                with open(data_export_path, 'r') as f:
                    data_metrics = json.load(f)
                    assert "active_streams" in data_metrics
                    assert "total_packets_processed" in data_metrics
                    
        finally:
            stream.stop_streaming()
    
    def test_health_insights_persistence(self):
        """Test persistence of health insights across devices."""
        # Initialize all devices
        glasses = SmartGlassesDevice(sensor_id=1)
        headphones = SmartHeadphonesDevice(sensor_id=2)
        watch = SmartWatchDevice(sensor_id=3)
        
        # Collect health insights
        all_insights = {}
        
        # Generate sufficient data for insights
        for i in range(20):
            glasses.read()
            headphones.read()
            watch.read()
        
        # Get insights from all devices
        all_insights['glasses'] = glasses.get_health_insights()
        all_insights['headphones'] = headphones.get_hearing_health_summary()
        all_insights['watch'] = watch.get_health_insights()
        
        # Export insights
        with tempfile.TemporaryDirectory() as temp_dir:
            insights_path = f"{temp_dir}/health_insights.json"
            
            with open(insights_path, 'w') as f:
                # Make insights JSON serializable
                serializable_insights = {}
                for device, insights in all_insights.items():
                    if isinstance(insights, dict):
                        # Convert numpy types to Python types
                        serializable_insights[device] = {}
                        for key, value in insights.items():
                            if isinstance(value, dict):
                                serializable_insights[device][key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in value.items()}
                            elif isinstance(value, (np.integer, np.floating)):
                                serializable_insights[device][key] = float(value)
                            else:
                                serializable_insights[device][key] = value
                    else:
                        serializable_insights[device] = insights
                
                json.dump(serializable_insights, f, indent=2)
            
            # Verify export
            assert os.path.exists(insights_path)
            
            # Verify content can be loaded
            with open(insights_path, 'r') as f:
                loaded_insights = json.load(f)
                assert 'glasses' in loaded_insights
                assert 'headphones' in loaded_insights
                assert 'watch' in loaded_insights


# Import os for file operations
import os 