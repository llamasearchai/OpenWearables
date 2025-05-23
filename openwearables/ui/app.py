import os
import json
import time
import logging
import secrets
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import redis
import threading
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("openwearables.log"), logging.StreamHandler()]
)
logger = logging.getLogger("OpenWearables.UI")

# Initialize Flask app with enhanced configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('HTTPS', 'false').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Enable CORS for API endpoints
CORS(app, resources={
    r"/api/*": {"origins": ["http://localhost:*", "https://localhost:*"]},
    r"/socket.io/*": {"origins": ["http://localhost:*", "https://localhost:*"]}
})

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, 
                   cors_allowed_origins=["http://localhost:*", "https://localhost:*"],
                   async_mode='threading',
                   ping_timeout=60,
                   ping_interval=25)

# Redis connection for session management and caching
try:
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6379)),
        password=os.environ.get('REDIS_PASSWORD'),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis not available: {e}")
    redis_client = None

# Import OpenWearables core if available
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from openwearables.core.architecture import OpenWearablesCore
    has_core = True
    logger.info("OpenWearables core module imported successfully")
except ImportError as e:
    has_core = False
    logger.warning(f"OpenWearables core module not found: {e}, running in demo mode")

# Import mock data generator
try:
    from openwearables.core.mock_data import (
        get_mock_data, get_mock_historical_data, 
        get_mock_insights, get_mock_device_status,
        MockDataGenerator
    )
    mock_data_generator = MockDataGenerator()
    logger.info("Mock data generator imported successfully")
except ImportError as e:
    logger.warning(f"Mock data generator not found: {e}")
    mock_data_generator = None

# Global variables
core_instance = None
config_path = os.environ.get('OPENWEARABLES_CONFIG', 'config/default.json')
mock_mode = os.environ.get('OPENWEARABLES_MOCK', 'false').lower() == 'true' or not has_core
data_broadcast_thread = None
stop_broadcast = threading.Event()

# Authentication system
class UserManager:
    """Simple user management for demo purposes."""
    
    def __init__(self):
        self.users = {
            'admin': {
                'password_hash': generate_password_hash('admin123'),
                'role': 'admin',
                'created_at': datetime.now().isoformat()
            }
        }
    
    def validate_user(self, username: str, password: str) -> bool:
        """Validate user credentials."""
        user = self.users.get(username)
        if user and check_password_hash(user['password_hash'], password):
            return True
        return False
    
    def get_user(self, username: str) -> dict:
        """Get user information."""
        return self.users.get(username)

user_manager = UserManager()

# Enhanced helper functions
def load_config():
    """Load configuration from JSON file with enhanced error handling."""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                # Validate configuration schema - check for nested structure
                required_keys = ['sensors', 'processing', 'privacy']
                for key in required_keys:
                    if key not in config:
                        logger.warning(f"Missing required config key: {key}")
                        # For sensors, check if it has the expected nested structure
                        if key == 'sensors' and 'sensors' not in config:
                            logger.warning("Config missing sensors section, will use defaults")
                        
                return config
        else:
            logger.warning(f"Config file {config_path} not found, creating defaults")
            
            # Create default config directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Enhanced default configuration that matches the expected structure
            default_config = {
                "device_name": "OpenWearables Device",
                "device_id": secrets.token_hex(8),
                "version": "1.0.0",
                "hardware": {
                    "use_mlx": True,
                    "use_cuda": True,
                    "fallback_cpu": True
                },
                "logging": {
                    "level": "INFO",
                    "file": "logs/openwearables.log",
                    "max_size": "100MB",
                    "backup_count": 5
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
                        "calibration_interval": 3600,
                        "calibration_samples": 100
                    }
                },
                "processing": {
                    "window_size": 10,
                    "overlap": 0.5,
                    "features": ["time_domain", "frequency_domain", "wavelet"],
                    "real_time": True,
                    "buffer_size": 1000
                },
                "privacy": {
                    "encryption": True,
                    "anonymization": True,
                    "data_retention": 90,
                    "secure_transmission": True,
                    "audit_logging": True
                },
                "alerts": {
                    "enabled": True,
                    "thresholds": {
                        "heart_rate_high": 100,
                        "heart_rate_low": 50,
                        "temperature_high": 38.5,
                        "spo2_low": 90
                    },
                    "notifications": ["email", "dashboard"]
                },
                "ui": {
                    "theme": "light",
                    "refresh_rate": 1000,
                    "charts": {
                        "ecg_duration": 10,
                        "ppg_duration": 10,
                        "update_interval": 250
                    }
                },
                "user_profile": {
                    "name": "",
                    "age": None,
                    "gender": "",
                    "height": None,
                    "weight": None,
                    "medical_conditions": "",
                    "medications": "",
                    "emergency_contact": "",
                    "physician": ""
                },
                "database": {
                    "type": "sqlite",
                    "path": "data/wearables.db",
                    "backup_enabled": True,
                    "backup_interval": 86400
                },
                "api": {
                    "rate_limit": 100,
                    "authentication": True,
                    "cors_enabled": True
                }
            }
            
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
            
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        # Return a minimal working config instead of empty dict
        return {
            "device_name": "OpenWearables Device",
            "sensors": {"enabled": ["ecg", "ppg", "accelerometer"]},
            "processing": {"real_time": True},
            "privacy": {"encryption": True}
        }

def save_config(config):
    """Save configuration to JSON file with backup."""
    try:
        # Create backup of current config
        if os.path.exists(config_path):
            backup_path = f"{config_path}.backup.{int(time.time())}"
            with open(config_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
        
        # Save new config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Cache config in Redis if available
        if redis_client:
            try:
                redis_client.set('openwearables:config', json.dumps(config), ex=3600)
            except Exception as e:
                logger.warning(f"Failed to cache config in Redis: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")
        return False

def get_cached_config():
    """Get configuration from cache if available."""
    if redis_client:
        try:
            cached_config = redis_client.get('openwearables:config')
            if cached_config:
                return json.loads(cached_config)
        except Exception as e:
            logger.warning(f"Failed to get cached config: {e}")
    return None

def init_core():
    """Initialize the OpenWearables core with enhanced error handling."""
    global core_instance
    
    if mock_mode:
        logger.info("Running in mock mode, core will not be initialized")
        return None
    
    if core_instance is not None:
        return core_instance
    
    try:
        logger.info("Initializing OpenWearables core...")
        core_instance = OpenWearablesCore(config_path)
        logger.info("Core initialized successfully")
        
        # Start data broadcasting thread
        start_data_broadcast()
        
        return core_instance
    except Exception as e:
        logger.error(f"Error initializing core: {str(e)}")
        return None

def start_data_broadcast():
    """Start background thread for broadcasting real-time data."""
    global data_broadcast_thread
    
    if data_broadcast_thread and data_broadcast_thread.is_alive():
        return
    
    stop_broadcast.clear()
    data_broadcast_thread = threading.Thread(target=data_broadcast_worker, daemon=True)
    data_broadcast_thread.start()
    logger.info("Started data broadcast thread")

def data_broadcast_worker():
    """Worker function for broadcasting real-time data to connected clients."""
    while not stop_broadcast.is_set():
        try:
            if mock_mode:
                data = get_mock_data()
                analysis = get_mock_analysis()
            else:
                if core_instance and core_instance.is_running():
                    data = {
                        'data': core_instance.get_latest_readings(),
                        'timestamp': time.time()
                    }
                    analysis = {
                        'results': core_instance.get_latest_analysis(),
                        'timestamp': time.time()
                    }
                else:
                    time.sleep(1)
                    continue
            
            # Broadcast to all connected clients
            socketio.emit('sensor_data', data, namespace='/live')
            socketio.emit('analysis_data', analysis, namespace='/live')
            
            # Cache latest data in Redis
            if redis_client:
                try:
                    redis_client.setex('openwearables:latest_data', 30, json.dumps(data))
                    redis_client.setex('openwearables:latest_analysis', 30, json.dumps(analysis))
                except Exception as e:
                    logger.warning(f"Failed to cache data in Redis: {e}")
            
            time.sleep(1)  # Broadcast every second
            
        except Exception as e:
            logger.error(f"Error in data broadcast worker: {e}")
            time.sleep(5)

def stop_data_broadcast():
    """Stop the data broadcast thread."""
    global data_broadcast_thread
    
    stop_broadcast.set()
    if data_broadcast_thread and data_broadcast_thread.is_alive():
        data_broadcast_thread.join(timeout=5)
        logger.info("Stopped data broadcast thread")

# Authentication decorator
def login_required(f):
    """Decorator to require authentication for routes."""
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Rate limiting decorator
def rate_limit(max_requests=100, per_seconds=60):
    """Simple rate limiting decorator."""
    def decorator(f):
        def decorated_function(*args, **kwargs):
            if redis_client:
                client_id = request.remote_addr
                key = f"rate_limit:{client_id}:{f.__name__}"
                current = redis_client.get(key)
                
                if current is None:
                    redis_client.setex(key, per_seconds, 1)
                elif int(current) >= max_requests:
                    return jsonify({'error': 'Rate limit exceeded'}), 429
                else:
                    redis_client.incr(key)
            
            return f(*args, **kwargs)
        decorated_function.__name__ = f.__name__
        return decorated_function
    return decorator

def get_mock_data():
    """Generate mock data for demo mode using the comprehensive mock data generator."""
    if mock_data_generator:
        return mock_data_generator.generate_real_time_data()
    else:
        # Fallback simple mock data
        return {
            'timestamp': datetime.now().isoformat(),
            'device_id': 'fallback_device',
            'ecg': {'heart_rate': 72, 'signal_quality': 0.95},
            'ppg': {'heart_rate': 72, 'spo2': 98},
            'vitals': {'temperature': 36.7},
            'motion': {'activity_type': 'resting'},
            'metadata': {'activity': 'resting', 'confidence': 0.9}
        }

def get_mock_analysis():
    """Generate mock analysis results for demo mode."""
    if mock_data_generator:
        insights = mock_data_generator.generate_health_insights()
        return {
            'insights': insights,
            'health_score': 85,
            'timestamp': datetime.now().isoformat()
        }
    else:
        # Fallback simple analysis
        return {
            'insights': [{
                'title': 'System Status',
                'message': 'All systems operating normally',
                'priority': 'low'
            }],
            'health_score': 80,
            'timestamp': datetime.now().isoformat()
        }

def get_mock_health_summary(days=1):
    """Generate mock health summary data for analysis page."""
    if mock_data_generator:
        historical_data = mock_data_generator.generate_historical_data(days)
        insights = mock_data_generator.generate_health_insights()
        
        # Process historical data for charts
        heart_rates = []
        hrv_data = []
        timestamps = []
        
        for data in historical_data[-24:]:  # Last 24 data points
            timestamp = data['timestamp']
            timestamps.append(timestamp)
            heart_rates.append(data.get('ecg', {}).get('heart_rate', 70))
            hrv_metrics = data.get('ecg', {}).get('hrv_metrics', {})
            hrv_data.append(hrv_metrics.get('rmssd', 25))
        
        return {
            'summary': {
                'average_hr': sum(heart_rates) / len(heart_rates) if heart_rates else 70,
                'hrv_rmssd': sum(hrv_data) / len(hrv_data) if hrv_data else 25,
                'total_data_points': len(historical_data)
            },
            'charts': {
                'heart_rate': list(zip(timestamps, heart_rates)),
                'hrv': list(zip(timestamps, hrv_data))
            },
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
    else:
        # Fallback simple summary
        return {
            'summary': {
                'average_hr': 72,
                'hrv_rmssd': 25,
                'total_data_points': 100
            },
            'charts': {
                'heart_rate': [],
                'hrv': []
            },
            'insights': [],
            'timestamp': datetime.now().isoformat()
        }

# Routes
@app.route('/')
def index():
    """Render the main dashboard page."""
    config = load_config()
    
    # Get system status for the template
    if mock_mode:
        # Simulate running status based on time to demo UI functionality
        is_running = (time.time() % 300) > 150
        system_status = {'is_running': is_running}
    else:
        if core_instance is None:
            system_status = {'is_running': False}
        else:
            try:
                # Check if at least one sensor is running
                sensors = core_instance.sensor_manager.sensors
                is_running = any(sensor.is_running for sensor in sensors.values())
                system_status = {'is_running': is_running}
            except Exception as e:
                logger.error(f"Error getting system status: {str(e)}")
                system_status = {'is_running': False}
    
    return render_template('index.html', config=config, system_status=system_status)

@app.route('/analysis')
def analysis():
    """Render the analysis page."""
    config = load_config()
    return render_template('analysis.html', config=config)

@app.route('/settings')
def settings():
    """Render the settings page."""
    config = load_config()
    return render_template('settings.html', config=config)

@app.route('/devices')
def devices():
    """Render the devices page."""
    config = load_config()
    
    # Get device information
    if mock_mode:
        devices_info = {
            'connected_devices': [
                {'name': 'OpenWearables Simulator', 'type': 'simulator', 'status': 'connected'},
                {'name': 'Mock ECG Sensor', 'type': 'ecg', 'status': 'active'},
                {'name': 'Mock PPG Sensor', 'type': 'ppg', 'status': 'active'},
                {'name': 'Mock Motion Sensor', 'type': 'motion', 'status': 'active'}
            ]
        }
    else:
        devices_info = {'connected_devices': []}
        if core_instance:
            try:
                # Get actual device information from core
                sensors = core_instance.sensor_manager.sensors
                devices_info['connected_devices'] = [
                    {'name': f'{name.upper()} Sensor', 'type': name, 'status': 'active' if sensor.is_running else 'inactive'}
                    for name, sensor in sensors.items()
                ]
            except Exception as e:
                logger.error(f"Error getting device info: {str(e)}")
    
    return render_template('devices.html', config=config, devices=devices_info)

@app.route('/health_insights')
def health_insights():
    """Render the health insights page."""
    config = load_config()
    
    # Get health insights
    if mock_mode:
        insights = get_mock_analysis().get('insights', [])
    else:
        insights = []
        if core_instance:
            try:
                analysis_results = core_instance.get_latest_analysis()
                insights = analysis_results.get('insights', [])
            except Exception as e:
                logger.error(f"Error getting health insights: {str(e)}")
    
    return render_template('health_insights.html', config=config, insights=insights)

@app.route('/alerts')
def alerts():
    """Render the alerts page."""
    config = load_config()
    
    # Get alerts
    if mock_mode:
        alerts_data = {
            'active_alerts': [
                {'id': 1, 'type': 'info', 'message': 'System running normally', 'timestamp': time.time()},
                {'id': 2, 'type': 'warning', 'message': 'Heart rate slightly elevated', 'timestamp': time.time() - 300}
            ],
            'alert_history': []
        }
    else:
        alerts_data = {'active_alerts': [], 'alert_history': []}
        if core_instance:
            try:
                # Get alerts from core
                alerts_data = core_instance.get_alerts()
            except Exception as e:
                logger.error(f"Error getting alerts: {str(e)}")
    
    return render_template('alerts.html', config=config, alerts=alerts_data)

@app.route('/reports')
def reports():
    """Render the reports page."""
    config = load_config()
    
    # Get reports data
    if mock_mode:
        reports_data = {
            'available_reports': [
                {'name': 'Daily Health Summary', 'type': 'daily', 'last_generated': time.time() - 86400},
                {'name': 'Weekly Trends', 'type': 'weekly', 'last_generated': time.time() - 604800},
                {'name': 'Monthly Analysis', 'type': 'monthly', 'last_generated': time.time() - 2592000}
            ]
        }
    else:
        reports_data = {'available_reports': []}
        if core_instance:
            try:
                # Get reports from core
                reports_data = core_instance.get_reports()
            except Exception as e:
                logger.error(f"Error getting reports: {str(e)}")
    
    return render_template('reports.html', config=config, reports=reports_data)

@app.route('/start', methods=['POST'])
def start_system():
    """Start the OpenWearables system."""
    if mock_mode:
        logger.info("Start requested in mock mode")
        return jsonify({'success': True})
    
    core = init_core()
    if core is None:
        return jsonify({'success': False, 'error': 'Failed to initialize core'})
    
    try:
        core.start()
        logger.info("System started successfully")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error starting system: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop', methods=['POST'])
def stop_system():
    """Stop the OpenWearables system."""
    global core_instance
    
    if mock_mode:
        logger.info("Stop requested in mock mode")
        return jsonify({'success': True})
    
    if core_instance is None:
        return jsonify({'success': False, 'error': 'System not initialized'})
    
    try:
        core_instance.stop()
        logger.info("System stopped successfully")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error stopping system: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get the current system status."""
    if mock_mode:
        # Simulate running status based on time to demo UI functionality
        is_running = (time.time() % 300) > 150
        return jsonify({'running': is_running})
    
    if core_instance is None:
        return jsonify({'running': False})
    
    try:
        # Check if at least one sensor is running
        sensors = core_instance.sensor_manager.sensors
        is_running = any(sensor.is_running for sensor in sensors.values())
        return jsonify({'running': is_running})
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return jsonify({'running': False, 'error': str(e)})

@app.route('/api/data/latest', methods=['GET'])
def latest_data():
    """Get the latest sensor data."""
    if mock_mode:
        return jsonify(get_mock_data())
    
    if core_instance is None:
        return jsonify({'error': 'System not initialized'})
    
    try:
        # Get latest readings from all sensors
        latest_data = core_instance.get_latest_readings()
        return jsonify({'data': latest_data, 'timestamp': time.time()})
    except Exception as e:
        logger.error(f"Error getting latest data: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/analysis/latest', methods=['GET'])
def latest_analysis():
    """Get the latest analysis results."""
    if mock_mode:
        return jsonify(get_mock_analysis())
    
    if core_instance is None:
        return jsonify({'error': 'System not initialized'})
    
    try:
        # Get latest analysis results
        analysis_results = core_instance.get_latest_analysis()
        return jsonify({'results': analysis_results, 'timestamp': time.time()})
    except Exception as e:
        logger.error(f"Error getting latest analysis: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/health/summary', methods=['GET'])
def health_summary():
    """Get health summary data for the analysis page."""
    # Get days parameter (default to 1 day)
    days = request.args.get('days', 1, type=int)
    
    if mock_mode:
        return jsonify(get_mock_health_summary(days))
    
    if core_instance is None:
        return jsonify({'error': 'System not initialized'})
    
    try:
        # Get health summary for specified days
        summary = core_instance.get_health_summary(days)
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting health summary: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update system settings."""
    try:
        # Get settings from request
        settings = request.json
        
        if not settings:
            return jsonify({'success': False, 'error': 'No settings provided'})
        
        # Load current config
        current_config = load_config()
        
        # Update with new settings
        if 'system' in settings:
            for key, value in settings['system'].items():
                current_config[key] = value
        
        if 'user_profile' in settings:
            current_config['user_profile'] = settings['user_profile']
        
        # Save updated config
        if save_config(current_config):
            restart_needed = False
            
            # Check if core needs to be restarted
            if not mock_mode and core_instance is not None:
                core_running = core_instance.is_running()
                restart_needed = core_running
                
                if core_running:
                    try:
                        core_instance.stop()
                    except:
                        pass
            
            return jsonify({
                'success': True, 
                'message': 'Settings saved. Please restart the system for changes to take effect.' if restart_needed else None
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save settings'})
    
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='OpenWearables UI Application')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)),
                       help='Port to run the server on (default: 5000)')
    parser.add_argument('--debug', action='store_true', 
                       default=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true',
                       help='Enable debug mode')
    parser.add_argument('--mock', action='store_true',
                       default=os.environ.get('OPENWEARABLES_MOCK', 'false').lower() == 'true',
                       help='Run in mock mode')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    # Override global mock_mode if specified via command line
    if args.mock:
        mock_mode = True
    
    # Load configuration
    config = load_config()
    
    # Initialize core if not in mock mode
    if not mock_mode:
        logger.info("Initializing OpenWearables core...")
        init_core()
    else:
        logger.info("Running in mock mode - no core initialization needed")
    
    # Start data broadcast thread
    start_data_broadcast()
    
    # Run the app
    logger.info(f"Starting OpenWearables UI on {args.host}:{args.port} (debug: {args.debug}, mock: {mock_mode})")
    
    try:
        if args.debug:
            # Development mode with SocketIO
            socketio.run(app, host=args.host, port=args.port, debug=True)
        else:
            # Production mode
            socketio.run(app, host=args.host, port=args.port, debug=False)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {args.port} is already in use. Try a different port with --port <number>")
            logger.info("On macOS, you may need to disable AirPlay Receiver in System Preferences > Sharing")
        else:
            logger.error(f"Failed to start server: {e}")
    except KeyboardInterrupt:
        logger.info("Shutting down OpenWearables UI...")
        stop_data_broadcast()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        stop_data_broadcast()