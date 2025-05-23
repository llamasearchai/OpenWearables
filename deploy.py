#!/usr/bin/env python3
"""
OpenWearables Deployment Script

Sets up the environment and validates that everything is working correctly.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    dirs = ['data', 'logs', 'config', 'static', 'templates']
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def setup_config():
    """Set up default configuration if it doesn't exist."""
    config_path = 'config/default.json'
    
    if os.path.exists(config_path):
        print(f"✓ Configuration file already exists: {config_path}")
        return
    
    default_config = {
        "device_name": "OpenWearables Professional Platform",
        "device_id": "ow-prod-001",
        "version": "1.0.0",
        "environment": "production",
        "hardware": {
            "use_mlx": True,
            "use_cuda": True,
            "fallback_cpu": True,
            "optimization_level": "maximum"
        },
        "logging": {
            "level": "INFO",
            "file": "logs/openwearables.log",
            "max_size": "100MB"
        },
        "sensors": {
            "enabled": ["ecg", "ppg", "accelerometer", "gyroscope", "temperature"],
            "sampling_rates": {
                "ecg": 250,
                "ppg": 100,
                "accelerometer": 50,
                "gyroscope": 50,
                "temperature": 1
            },
            "calibration": {
                "auto_calibrate": True,
                "calibration_interval": 3600
            }
        },
        "processing": {
            "window_size": 10,
            "overlap": 0.5,
            "features": ["time_domain", "frequency_domain"],
            "real_time": True
        },
        "privacy": {
            "encryption": {
                "algorithm": "AES-256-GCM",
                "key_rotation_hours": 24
            },
            "anonymization": {
                "enabled": True,
                "method": "differential_privacy"
            },
            "data_retention": {
                "raw_data_days": 30,
                "processed_data_days": 90
            }
        },
        "ui": {
            "theme": "professional",
            "refresh_rate": 500,
            "real_time_updates": True
        },
        "database": {
            "type": "sqlite",
            "path": "data/openwearables.db"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"✓ Created configuration file: {config_path}")

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    
    if version.major != 3 or version.minor < 10:
        print(f"✗ Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def test_imports():
    """Test critical imports."""
    print("Testing imports...")
    
    critical_imports = [
        'numpy',
        'pandas', 
        'flask',
        'openwearables.core.architecture',
        'openwearables.ui.app'
    ]
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            return False
    
    return True

def validate_installation():
    """Validate the installation."""
    print("Validating installation...")
    
    # Test mock data generation
    try:
        from openwearables.core.mock_data import MockDataGenerator
        generator = MockDataGenerator()
        data = generator.generate_real_time_data()
        print("  ✓ Mock data generation")
    except Exception as e:
        print(f"  ✗ Mock data generation: {e}")
        return False
    
    # Test sensor manager
    try:
        from openwearables.core.sensor_manager import SensorManager
        manager = SensorManager(['ecg', 'ppg'], {'ecg': 250, 'ppg': 100})
        print("  ✓ Sensor manager")
    except Exception as e:
        print(f"  ✗ Sensor manager: {e}")
        return False
    
    return True

def create_startup_scripts():
    """Create convenient startup scripts."""
    
    # Create a simple run script
    run_script = """#!/bin/bash
# OpenWearables Quick Start Script

echo "Starting OpenWearables..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start in mock mode by default
python start.py --mock --port 5001

echo "OpenWearables stopped."
"""
    
    with open('run.sh', 'w') as f:
        f.write(run_script)
    
    os.chmod('run.sh', 0o755)
    print("✓ Created run.sh script")
    
    # Create Windows batch file
    run_bat = """@echo off
echo Starting OpenWearables...

if exist venv (
    echo Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

python start.py --mock --port 5001

echo OpenWearables stopped.
pause
"""
    
    with open('run.bat', 'w') as f:
        f.write(run_bat)
    
    print("✓ Created run.bat script")

def main():
    """Main deployment function."""
    print("OpenWearables Deployment Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Set up configuration
    setup_config()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("✗ Import tests failed. Please check dependencies.")
        sys.exit(1)
    
    # Validate installation
    if not validate_installation():
        print("✗ Installation validation failed.")
        sys.exit(1)
    
    # Create startup scripts
    create_startup_scripts()
    
    print("\n" + "=" * 50)
    print("✓ OpenWearables deployment completed successfully!")
    print("\nTo start the application:")
    print("  Unix/macOS: ./run.sh")
    print("  Windows:    run.bat")
    print("  Manual:     python start.py --mock --port 5001")
    print("\nAccess the UI at: http://localhost:5001")

if __name__ == '__main__':
    main() 