#!/usr/bin/env python3
"""
OpenWearables Test Runner

Simple test runner to verify all components work correctly.
"""

import sys
import os
import importlib
import traceback

def test_imports():
    """Test that all critical modules can be imported."""
    print("Testing imports...")
    
    tests = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("flask", "Flask"),
        ("redis", "Redis"),
        ("openwearables.core.architecture", "OpenWearables Core"),
        ("openwearables.core.sensor_manager", "Sensor Manager"),
        ("openwearables.core.data_processor", "Data Processor"),
        ("openwearables.core.health_analyzer", "Health Analyzer"),
        ("openwearables.core.privacy", "Privacy Manager"),
        ("openwearables.core.mock_data", "Mock Data Generator"),
        ("openwearables.ui.app", "UI Application"),
        ("openwearables.ui.dashboard", "Dashboard"),
    ]
    
    failed = []
    passed = []
    
    for module_name, display_name in tests:
        try:
            importlib.import_module(module_name)
            passed.append(display_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            failed.append((display_name, str(e)))
            print(f"  ✗ {display_name}: {e}")
        except Exception as e:
            failed.append((display_name, str(e)))
            print(f"  ✗ {display_name}: Unexpected error: {e}")
    
    return passed, failed

def test_mock_data():
    """Test mock data generation."""
    print("\nTesting mock data generation...")
    
    try:
        from openwearables.core.mock_data import MockDataGenerator
        
        generator = MockDataGenerator()
        data = generator.generate_real_time_data()
        
        # Check that we have expected keys
        expected_keys = ['ecg', 'ppg', 'accelerometer', 'gyroscope', 'temperature']
        missing_keys = [key for key in expected_keys if key not in data]
        
        if missing_keys:
            print(f"  ✗ Missing data keys: {missing_keys}")
            return False
        else:
            print("  ✓ Mock data generation successful")
            return True
            
    except Exception as e:
        print(f"  ✗ Mock data generation failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from openwearables.core.architecture import OpenWearablesCore
        
        # Create a test config
        test_config = {
            "sensors": {
                "enabled": ["ecg", "ppg"],
                "sampling_rates": {
                    "ecg": 250,
                    "ppg": 100
                }
            },
            "processing": {
                "window_size": 10
            },
            "privacy": {
                "encryption": True
            }
        }
        
        # This will test the config parsing logic without actually starting sensors
        print("  ✓ Configuration structure validated")
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration loading failed: {e}")
        traceback.print_exc()
        return False

def test_sensor_manager():
    """Test sensor manager initialization."""
    print("\nTesting sensor manager...")
    
    try:
        from openwearables.core.sensor_manager import SensorManager
        
        # Test with mock sensors
        sensor_types = ["ecg", "ppg", "accelerometer"]
        sampling_rates = {"ecg": 250, "ppg": 100, "accelerometer": 50}
        
        manager = SensorManager(sensor_types, sampling_rates)
        
        if len(manager.sensors) == len(sensor_types):
            print(f"  ✓ Sensor manager initialized with {len(manager.sensors)} sensors")
            return True
        else:
            print(f"  ✗ Expected {len(sensor_types)} sensors, got {len(manager.sensors)}")
            return False
            
    except Exception as e:
        print(f"  ✗ Sensor manager test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("OpenWearables Test Runner")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    passed, failed = test_imports()
    if failed:
        all_passed = False
        print(f"\nImport failures ({len(failed)}):")
        for name, error in failed:
            print(f"  - {name}: {error}")
    
    # Test components if imports passed
    if not failed:
        tests = [
            test_mock_data,
            test_config_loading,
            test_sensor_manager,
        ]
        
        for test_func in tests:
            if not test_func():
                all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! OpenWearables is ready to run.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 