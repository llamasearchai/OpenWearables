# OpenWearables Development Guide

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- Git
- Node.js 16+ (for frontend development)
- Docker (optional, for containerized development)

### Hardware Requirements

- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB+ RAM, 8-core CPU
- **GPU Support**: NVIDIA GPU with CUDA 11.8+ or Apple Silicon for MLX

### Development Environment Setup

1. **Clone the Repository**
```bash
git clone https://github.com/openwearables/openwearables.git
cd openwearables
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -e .
pip install -r requirements-dev.txt
```

4. **Install Pre-commit Hooks**
```bash
pre-commit install
```

5. **Run Tests**
```bash
pytest tests/ -v
```

## Project Structure

```
openwearables/
├── openwearables/           # Main package
│   ├── core/               # Core architecture
│   │   ├── architecture.py # Main orchestrator
│   │   ├── sensor_manager.py
│   │   ├── data_processor.py
│   │   ├── health_analyzer.py
│   │   └── privacy.py
│   ├── models/             # ML models
│   │   ├── mlx_models.py   # Apple Silicon models
│   │   ├── torch_models.py # CUDA/CPU models
│   │   └── model_utils.py
│   ├── cli/                # Command line interface
│   │   └── openwearables_cli.py
│   └── ui/                 # Web interface
│       ├── app.py
│       ├── static/
│       └── templates/
├── tests/                  # Test suite
├── docs/                   # Documentation
├── config/                 # Configuration files
├── data/                   # Data storage
└── logs/                   # Log files
```

## Development Workflow

### 1. Feature Development

1. **Create Feature Branch**
```bash
git checkout -b feature/new-sensor-support
```

2. **Implement Feature**
   - Write code following style guidelines
   - Add comprehensive tests
   - Update documentation

3. **Test Locally**
```bash
pytest tests/test_new_feature.py -v
black openwearables/
isort openwearables/
mypy openwearables/
```

4. **Commit Changes**
```bash
git add .
git commit -m "feat: add support for new sensor type"
```

5. **Push and Create PR**
```bash
git push origin feature/new-sensor-support
```

### 2. Testing Strategy

#### Unit Tests
```python
# tests/test_sensor_manager.py
import pytest
from openwearables.core.sensor_manager import SensorManager

def test_sensor_initialization():
    config = {"sensors": ["ecg", "ppg"]}
    manager = SensorManager(config=config)
    assert len(manager.sensors) == 2

def test_sensor_data_validation():
    manager = SensorManager()
    valid_data = {"timestamp": 1234567890, "value": 72}
    assert manager.validate_data(valid_data) is True
```

#### Integration Tests
```python
# tests/test_integration.py
import pytest
from openwearables.core.architecture import OpenWearablesCore

def test_full_pipeline():
    core = OpenWearablesCore()
    core.start()
    
    # Simulate sensor data
    mock_data = generate_mock_ecg_data()
    core.process_data(mock_data)
    
    # Verify analysis results
    analysis = core.get_latest_analysis()
    assert "heart_rate" in analysis
    assert analysis["heart_rate"] > 0
    
    core.stop()
```

#### Performance Tests
```python
# tests/test_performance.py
import time
import pytest
from openwearables.core.data_processor import DataProcessor

def test_processing_latency():
    processor = DataProcessor()
    large_dataset = generate_large_ecg_dataset(10000)
    
    start_time = time.time()
    result = processor.process_ecg(large_dataset)
    end_time = time.time()
    
    # Processing should complete within 1 second
    assert end_time - start_time < 1.0
    assert result is not None
```

### 3. Code Style Guidelines

#### Python Style
Follow PEP 8 with these specific guidelines:

```python
# Good: Clear function names and type hints
def calculate_heart_rate_variability(
    rr_intervals: List[float], 
    window_size: int = 300
) -> Dict[str, float]:
    """
    Calculate HRV metrics from RR intervals.
    
    Args:
        rr_intervals: List of RR intervals in milliseconds
        window_size: Analysis window size in seconds
        
    Returns:
        Dictionary containing HRV metrics
    """
    if not rr_intervals:
        raise ValueError("RR intervals cannot be empty")
    
    # Implementation here
    return {"sdnn": sdnn_value, "rmssd": rmssd_value}

# Good: Clear class structure
class ECGProcessor:
    """Process ECG signals for health analysis."""
    
    def __init__(self, sampling_rate: int = 250):
        self.sampling_rate = sampling_rate
        self.filters = self._initialize_filters()
    
    def _initialize_filters(self) -> Dict[str, Any]:
        """Initialize signal processing filters."""
        return {
            "lowpass": butter(4, 40, fs=self.sampling_rate),
            "highpass": butter(4, 0.5, fs=self.sampling_rate)
        }
```

#### Error Handling
```python
# Good: Specific exception handling
try:
    result = process_sensor_data(data)
except SensorCalibrationError as e:
    logger.error(f"Sensor calibration failed: {e}")
    return {"error": "sensor_calibration", "details": str(e)}
except DataValidationError as e:
    logger.warning(f"Invalid data received: {e}")
    return {"error": "invalid_data", "details": str(e)}
except Exception as e:
    logger.exception("Unexpected error in data processing")
    return {"error": "processing_failed", "details": "Internal error"}
```

#### Logging
```python
import logging

logger = logging.getLogger("OpenWearables.SensorManager")

# Good: Appropriate log levels
logger.debug("Processing ECG data with %d samples", len(data))
logger.info("Sensor calibration completed successfully")
logger.warning("Sensor quality degraded, consider recalibration")
logger.error("Failed to connect to sensor: %s", error_message)
logger.critical("System memory usage critical: %d%%", memory_percent)
```

### 4. Adding New Sensors

#### Step 1: Define Sensor Class
```python
# openwearables/core/sensors/new_sensor.py
from .base_sensor import BaseSensor
from typing import Dict, Any, List

class NewSensorType(BaseSensor):
    """Implementation for new sensor type."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sensor_type = "new_sensor"
        self.sampling_rate = config.get("sampling_rate", 100)
    
    def connect(self) -> bool:
        """Connect to the physical sensor."""
        # Implementation specific to sensor
        pass
    
    def read_data(self) -> List[float]:
        """Read raw data from sensor."""
        # Implementation specific to sensor
        pass
    
    def validate_data(self, data: List[float]) -> bool:
        """Validate sensor data quality."""
        # Check for reasonable value ranges
        return all(self.min_value <= x <= self.max_value for x in data)
```

#### Step 2: Add Processing Logic
```python
# openwearables/core/processors/new_sensor_processor.py
from .base_processor import BaseProcessor
import numpy as np

class NewSensorProcessor(BaseProcessor):
    """Process data from new sensor type."""
    
    def process(self, raw_data: List[float]) -> Dict[str, Any]:
        """Process raw sensor data into meaningful metrics."""
        
        # Filter noise
        filtered_data = self.apply_filters(raw_data)
        
        # Extract features
        features = self.extract_features(filtered_data)
        
        # Calculate metrics
        metrics = self.calculate_metrics(features)
        
        return {
            "timestamp": time.time(),
            "sensor_type": "new_sensor",
            "raw_data": raw_data,
            "filtered_data": filtered_data,
            "features": features,
            "metrics": metrics,
            "quality_score": self.assess_quality(filtered_data)
        }
    
    def extract_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract relevant features from sensor data."""
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            # Add sensor-specific features
        }
```

#### Step 3: Add Tests
```python
# tests/test_new_sensor.py
import pytest
from openwearables.core.sensors.new_sensor import NewSensorType

def test_new_sensor_initialization():
    config = {"sampling_rate": 100, "device_id": "test_001"}
    sensor = NewSensorType(config)
    assert sensor.sampling_rate == 100
    assert sensor.sensor_type == "new_sensor"

def test_new_sensor_data_validation():
    sensor = NewSensorType({})
    valid_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert sensor.validate_data(valid_data) is True
    
    invalid_data = [1000.0, -1000.0]  # Out of range
    assert sensor.validate_data(invalid_data) is False
```

### 5. Machine Learning Model Development

#### Model Structure
```python
# openwearables/models/custom_model.py
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class HealthAnalysisModel(nn.Module):
    """Custom model for health analysis."""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Define architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(hidden_size // 2, 3)  # 3 classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)
    
    def predict(self, data: np.ndarray) -> Dict[str, Any]:
        """Make prediction on input data."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(data).unsqueeze(0)
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            
            return {
                "prediction": int(prediction.item()),
                "confidence": float(torch.max(probabilities).item()),
                "probabilities": probabilities.squeeze().tolist()
            }
```

#### Training Script
```python
# scripts/train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from openwearables.models.custom_model import HealthAnalysisModel
from openwearables.data.dataset import HealthDataset

def train_model(config: Dict[str, Any]):
    # Load data
    train_dataset = HealthDataset(config["train_data_path"])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = HealthAnalysisModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"]
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), config["model_save_path"])
    print(f"Model saved to {config['model_save_path']}")

if __name__ == "__main__":
    config = {
        "train_data_path": "data/training_data.csv",
        "input_size": 100,
        "hidden_size": 128,
        "learning_rate": 0.001,
        "num_epochs": 50,
        "model_save_path": "models/health_analysis_model.pth"
    }
    train_model(config)
```

### 6. API Development

#### Adding New Endpoints
```python
# openwearables/ui/api/new_endpoint.py
from flask import Blueprint, request, jsonify
from openwearables.core.architecture import OpenWearablesCore

new_api = Blueprint('new_api', __name__)

@new_api.route('/api/new-feature', methods=['GET'])
def get_new_feature():
    """Get new feature data."""
    try:
        # Validate parameters
        param = request.args.get('param', default='default_value')
        
        # Get core instance
        core = current_app.config.get('CORE_INSTANCE')
        if not core:
            return jsonify({
                "success": False,
                "error": "System not initialized"
            }), 503
        
        # Process request
        result = core.get_new_feature_data(param)
        
        return jsonify({
            "success": True,
            "data": result,
            "timestamp": time.time()
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": "Invalid parameter",
            "details": str(e)
        }), 400
    
    except Exception as e:
        logger.exception("Error in new feature endpoint")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@new_api.route('/api/new-feature', methods=['POST'])
def update_new_feature():
    """Update new feature configuration."""
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        # Update configuration
        core = current_app.config.get('CORE_INSTANCE')
        success = core.update_new_feature_config(data)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Configuration updated successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to update configuration"
            }), 500
            
    except Exception as e:
        logger.exception("Error updating new feature")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
```

### 7. Database Migrations

```python
# scripts/migrate_database.py
import sqlite3
import logging
from pathlib import Path

def migrate_database(db_path: str, version: str):
    """Apply database migrations."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check current version
        cursor.execute("SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1")
        current_version = cursor.fetchone()
        current_version = current_version[0] if current_version else "0.0.0"
        
        # Apply migrations
        if version == "1.1.0" and current_version < "1.1.0":
            apply_migration_1_1_0(cursor)
        
        # Record migration
        cursor.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, datetime('now'))",
            (version,)
        )
        
        conn.commit()
        logging.info(f"Database migrated to version {version}")
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Migration failed: {e}")
        raise
    finally:
        conn.close()

def apply_migration_1_1_0(cursor):
    """Migration for version 1.1.0."""
    # Add new columns
    cursor.execute("ALTER TABLE sensor_data ADD COLUMN quality_score REAL DEFAULT 1.0")
    cursor.execute("ALTER TABLE health_analysis ADD COLUMN ai_confidence REAL DEFAULT 0.0")
    
    # Create new tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            preference_key TEXT NOT NULL,
            preference_value TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, preference_key)
        )
    """)
```

### 8. Deployment

#### Docker Development
```dockerfile
# docker/Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install in development mode
RUN pip install -e .

# Expose ports
EXPOSE 5000 8080

# Start development server
CMD ["python", "-m", "openwearables.ui.app"]
```

#### Development Docker Compose
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  openwearables:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    ports:
      - "5000:5000"
    volumes:
      - .:/app
      - /app/venv
    environment:
      - FLASK_ENV=development
      - FLASK_DEBUG=1
      - OPENWEARABLES_CONFIG=/app/config/development.json
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: openwearables_dev
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### 9. Continuous Integration

#### GitHub Actions Workflow
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Lint with flake8
      run: |
        flake8 openwearables tests
    
    - name: Format check with black
      run: |
        black --check openwearables tests
    
    - name: Import sort check
      run: |
        isort --check-only openwearables tests
    
    - name: Type check with mypy
      run: |
        mypy openwearables
    
    - name: Test with pytest
      run: |
        pytest tests/ --cov=openwearables --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 10. Documentation

#### Adding Documentation
```python
# docs/generate_docs.py
"""Generate API documentation from code."""

import inspect
import json
from pathlib import Path
from openwearables.core.architecture import OpenWearablesCore

def generate_api_docs():
    """Generate API documentation."""
    docs = {}
    
    # Inspect OpenWearablesCore
    core_methods = inspect.getmembers(OpenWearablesCore, predicate=inspect.ismethod)
    
    for name, method in core_methods:
        if not name.startswith('_'):  # Skip private methods
            docs[name] = {
                "name": name,
                "docstring": inspect.getdoc(method),
                "signature": str(inspect.signature(method)),
                "module": method.__module__
            }
    
    # Save documentation
    docs_path = Path("docs/generated/api.json")
    docs_path.parent.mkdir(exist_ok=True)
    
    with open(docs_path, 'w') as f:
        json.dump(docs, f, indent=2)
    
    print(f"API documentation generated at {docs_path}")

if __name__ == "__main__":
    generate_api_docs()
```

## Contributing Guidelines

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Follow code style** guidelines and run linters
4. **Update documentation** for any API changes
5. **Submit a pull request** with a clear description
6. **Respond to code review** feedback promptly

## Getting Help

- **Discord**: Join our development Discord server
- **GitHub Discussions**: Ask questions and share ideas
- **Documentation**: Read the full documentation at docs.openwearables.org
- **Email**: Contact the maintainers at dev@openwearables.org

## License

This project is licensed under the MIT License. See LICENSE file for details. 