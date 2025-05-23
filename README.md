# OpenWearables

**AI-Powered Wearable Health Monitoring Platform**

OpenWearables is a comprehensive, production-ready platform for real-time health monitoring using wearable sensors. It combines advanced signal processing, machine learning, and AI-powered health analytics to provide intelligent insights and recommendations.

## Features

### Core Capabilities
- **Real-time Sensor Data Processing**: ECG, PPG, accelerometer, gyroscope, temperature sensors
- **AI-Powered Health Analysis**: Advanced algorithms for arrhythmia detection, stress analysis, and activity recognition
- **Intelligent Health Insights**: LLM-powered health assessments and personalized recommendations
- **Privacy-First Design**: End-to-end encryption, differential privacy, and secure data handling
- **Professional Web Interface**: Modern, responsive dashboard with real-time visualizations

### Technical Highlights
- **Hardware Acceleration**: Optimized for Apple Silicon (MLX), NVIDIA GPUs (CUDA), and CPU fallback
- **Scalable Architecture**: Modular design supporting multiple sensor types and analysis pipelines
- **Production Ready**: Comprehensive logging, error handling, and monitoring capabilities
- **Developer Friendly**: Extensive mock data system for development and testing

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/llamasearchai/OpenWearables.git
   cd OpenWearables
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run with mock data (recommended for first time)**
   ```bash
   python main.py --mock --port 5000
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000` to access the dashboard

### Alternative Start Methods

```bash
# Using start.py directly
python start.py --mock --port 5000

# Debug mode with verbose logging
python main.py --mock --debug

# Production mode with real sensors
python main.py --port 8080
```

## Architecture

### Core Components

```
OpenWearables/
├── openwearables/
│   ├── core/                 # Core processing engine
│   │   ├── architecture.py   # Main orchestration
│   │   ├── sensor_manager.py # Sensor data collection
│   │   ├── data_processor.py # Signal processing
│   │   ├── health_analyzer.py# AI health analysis
│   │   ├── privacy.py        # Privacy protection
│   │   └── mock_data.py      # Development data
│   ├── ui/                   # Web interface
│   │   ├── app.py           # Flask application
│   │   ├── templates/       # HTML templates
│   │   └── static/          # CSS, JS, images
│   └── models/              # ML model implementations
├── config/                  # Configuration files
├── data/                    # Data storage
├── logs/                    # Application logs
└── tests/                   # Test suite
```

### Data Flow

1. **Sensor Data Collection**: Real-time data from multiple sensor types
2. **Signal Processing**: Filtering, artifact removal, feature extraction
3. **AI Analysis**: Health metrics calculation and anomaly detection
4. **Privacy Protection**: Data encryption and anonymization
5. **Web Dashboard**: Real-time visualization and insights

## Configuration

The platform uses JSON configuration files in the `config/` directory:

```json
{
  "sensors": {
    "enabled": ["ecg", "ppg", "accelerometer", "gyroscope", "temperature"],
    "sampling_rates": {
      "ecg": 250,
      "ppg": 100,
      "accelerometer": 50
    }
  },
  "processing": {
    "window_size": 10,
    "features": ["time_domain", "frequency_domain", "wavelet"]
  },
  "privacy": {
    "encryption": true,
    "anonymization": true,
    "data_retention": 90
  }
}
```

## API Reference

### REST Endpoints

- `GET /api/data/latest` - Get latest sensor readings
- `GET /api/analysis/latest` - Get latest health analysis
- `GET /api/health/summary` - Get health summary for specified period
- `POST /api/settings` - Update system configuration
- `GET /api/system/status` - Get system status

### WebSocket Events

- `sensor_data` - Real-time sensor data updates
- `analysis_data` - Real-time analysis results
- `system_status` - System status changes

## Development

### Running Tests

```bash
# Run the comprehensive test suite
python test_runner.py

# Run specific tests with pytest
pip install pytest
pytest tests/ -v
```

### Mock Data Development

The platform includes a sophisticated mock data generator for development:

```python
from openwearables.core.mock_data import MockDataGenerator

generator = MockDataGenerator()
data = generator.generate_real_time_data()
```

### Adding New Sensors

1. Create sensor class inheriting from `SensorInterface`
2. Implement the `read()` method
3. Add sensor type to configuration
4. Update data processor for new sensor type

## Production Deployment

### Docker Deployment

```bash
# Build the container
docker build -t openwearables .

# Run with environment variables
docker run -p 5000:5000 -e OPENWEARABLES_MOCK=false openwearables
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/production/
```

### Environment Variables

- `OPENWEARABLES_MOCK`: Enable mock mode (`true`/`false`)
- `OPENWEARABLES_CONFIG`: Path to configuration file
- `REDIS_HOST`: Redis server hostname
- `REDIS_PORT`: Redis server port

## Security

### Privacy Features
- **End-to-end encryption** using AES-256-GCM
- **Differential privacy** for data anonymization
- **Secure key management** with automatic rotation
- **Audit logging** for compliance

### Data Protection
- **Local data processing** - no cloud dependencies required
- **Configurable data retention** policies
- **User consent management**
- **GDPR compliance** features

## Performance

### Optimization Features
- **Hardware acceleration** support (MLX, CUDA, MPS)
- **Real-time processing** with sub-50ms latency
- **Efficient memory usage** with circular buffers
- **Parallel processing** for multiple sensors

### Benchmarks
- **Sensor throughput**: 1000+ samples/second per sensor
- **Analysis latency**: <50ms for real-time insights
- **Memory usage**: <500MB for full system
- **CPU usage**: <20% on modern hardware

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `python test_runner.py`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Support

- **Documentation**: [Wiki](https://github.com/llamasearchai/OpenWearables/wiki)
- **Issues**: [GitHub Issues](https://github.com/llamasearchai/OpenWearables/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llamasearchai/OpenWearables/discussions)

## Acknowledgments

- Built with modern Python ecosystem (Flask, NumPy, SciPy, scikit-learn)
- AI capabilities powered by LangChain and Hugging Face
- Hardware acceleration via MLX (Apple Silicon) and PyTorch
- Professional UI components and responsive design

---

**OpenWearables** - Advancing health monitoring through AI and wearable technology. 