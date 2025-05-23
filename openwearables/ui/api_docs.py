"""
OpenWearables API Documentation System
Provides comprehensive REST API documentation with Swagger UI
"""

from flask import Blueprint, jsonify, request
from flask_restx import Api, Resource, fields, Namespace
from functools import wraps
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize Flask-RESTX
api = Api(
    api_bp,
    version='1.0',
    title='OpenWearables API',
    description='Comprehensive API for the OpenWearables platform - Advanced wearable device management, health analytics, and AI-powered insights',
    doc='/docs/',
    contact='OpenWearables Team',
    contact_email='api@openwearables.com',
    license='MIT',
    license_url='https://opensource.org/licenses/MIT',
    authorizations={
        'Bearer': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': 'JWT Bearer token for authentication'
        }
    },
    security='Bearer'
)

# Namespaces for organizing endpoints
devices_ns = Namespace('devices', description='Wearable device management operations')
health_ns = Namespace('health', description='Health analytics and monitoring')
analytics_ns = Namespace('analytics', description='Advanced AI analytics and insights')
sensors_ns = Namespace('sensors', description='Sensor data management')
ml_ns = Namespace('ml', description='Machine learning model operations')
privacy_ns = Namespace('privacy', description='Privacy and security operations')

api.add_namespace(devices_ns)
api.add_namespace(health_ns)
api.add_namespace(analytics_ns)
api.add_namespace(sensors_ns)
api.add_namespace(ml_ns)
api.add_namespace(privacy_ns)

# Common models for API documentation
error_model = api.model('Error', {
    'error': fields.String(required=True, description='Error message'),
    'code': fields.Integer(required=True, description='Error code'),
    'timestamp': fields.DateTime(required=True, description='Error timestamp'),
    'request_id': fields.String(description='Request ID for tracking')
})

success_model = api.model('Success', {
    'success': fields.Boolean(required=True, description='Operation success status'),
    'message': fields.String(description='Success message'),
    'data': fields.Raw(description='Response data'),
    'timestamp': fields.DateTime(required=True, description='Response timestamp')
})

# Device models
device_model = api.model('Device', {
    'id': fields.String(required=True, description='Unique device identifier'),
    'name': fields.String(required=True, description='Device name'),
    'type': fields.String(required=True, description='Device type (watch, glasses, headphones)'),
    'status': fields.String(required=True, description='Device status (connected, disconnected, error)'),
    'battery_level': fields.Float(description='Battery level (0-100)'),
    'firmware_version': fields.String(description='Firmware version'),
    'last_sync': fields.DateTime(description='Last synchronization time'),
    'capabilities': fields.List(fields.String, description='Device capabilities'),
    'location': fields.String(description='Device location/position')
})

device_list_model = api.model('DeviceList', {
    'devices': fields.List(fields.Nested(device_model)),
    'total': fields.Integer(description='Total number of devices'),
    'connected': fields.Integer(description='Number of connected devices')
})

# Health models
health_metric_model = api.model('HealthMetric', {
    'metric_type': fields.String(required=True, description='Type of health metric'),
    'value': fields.Float(required=True, description='Metric value'),
    'unit': fields.String(required=True, description='Unit of measurement'),
    'timestamp': fields.DateTime(required=True, description='Measurement timestamp'),
    'device_id': fields.String(description='Source device ID'),
    'confidence': fields.Float(description='Measurement confidence (0-1)')
})

health_summary_model = api.model('HealthSummary', {
    'user_id': fields.String(required=True, description='User identifier'),
    'date': fields.Date(required=True, description='Summary date'),
    'metrics': fields.List(fields.Nested(health_metric_model)),
    'insights': fields.List(fields.String, description='AI-generated health insights'),
    'recommendations': fields.List(fields.String, description='Health recommendations'),
    'risk_factors': fields.List(fields.String, description='Identified risk factors')
})

# Sensor models
sensor_data_model = api.model('SensorData', {
    'sensor_id': fields.String(required=True, description='Sensor identifier'),
    'sensor_type': fields.String(required=True, description='Type of sensor'),
    'timestamp': fields.DateTime(required=True, description='Data timestamp'),
    'values': fields.Raw(required=True, description='Sensor values (varies by type)'),
    'quality': fields.Float(description='Data quality score (0-1)'),
    'device_id': fields.String(description='Source device ID')
})

# ML models
ml_model_info = api.model('MLModel', {
    'model_id': fields.String(required=True, description='Model identifier'),
    'name': fields.String(required=True, description='Model name'),
    'type': fields.String(required=True, description='Model type'),
    'version': fields.String(required=True, description='Model version'),
    'accuracy': fields.Float(description='Model accuracy'),
    'last_trained': fields.DateTime(description='Last training timestamp'),
    'status': fields.String(description='Model status (active, training, deprecated)')
})

# Authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            api.abort(401, 'Authentication required')
        return f(*args, **kwargs)
    return decorated_function

# Device endpoints
@devices_ns.route('/')
class DeviceList(Resource):
    @api.doc('list_devices')
    @api.marshal_with(device_list_model)
    @require_auth
    def get(self):
        """Get all connected devices"""
        try:
            # Mock data for demonstration
            devices = [
                {
                    'id': 'watch_001',
                    'name': 'Apple Watch Series 9',
                    'type': 'watch',
                    'status': 'connected',
                    'battery_level': 85.0,
                    'firmware_version': '10.1.1',
                    'last_sync': datetime.now(),
                    'capabilities': ['heart_rate', 'ecg', 'blood_oxygen', 'activity'],
                    'location': 'left_wrist'
                },
                {
                    'id': 'glasses_001',
                    'name': 'OpenWearables Smart Glasses',
                    'type': 'glasses',
                    'status': 'connected',
                    'battery_level': 72.0,
                    'firmware_version': '2.3.1',
                    'last_sync': datetime.now(),
                    'capabilities': ['eye_tracking', 'ambient_light', 'head_movement'],
                    'location': 'head'
                }
            ]
            
            return {
                'devices': devices,
                'total': len(devices),
                'connected': len([d for d in devices if d['status'] == 'connected'])
            }
        except Exception as e:
            logger.error(f"Error fetching devices: {e}")
            api.abort(500, 'Internal server error')

@devices_ns.route('/<string:device_id>')
class Device(Resource):
    @api.doc('get_device')
    @api.marshal_with(device_model)
    @require_auth
    def get(self, device_id):
        """Get specific device information"""
        # Implementation would fetch from database
        return {
            'id': device_id,
            'name': 'Sample Device',
            'type': 'watch',
            'status': 'connected',
            'battery_level': 85.0
        }

# Health endpoints
@health_ns.route('/summary')
class HealthSummary(Resource):
    @api.doc('get_health_summary')
    @api.marshal_with(health_summary_model)
    @require_auth
    def get(self):
        """Get comprehensive health summary"""
        return {
            'user_id': 'user_001',
            'date': datetime.now().date(),
            'metrics': [
                {
                    'metric_type': 'heart_rate',
                    'value': 72.0,
                    'unit': 'bpm',
                    'timestamp': datetime.now(),
                    'confidence': 0.95
                }
            ],
            'insights': ['Heart rate variability is within normal range'],
            'recommendations': ['Consider increasing daily activity'],
            'risk_factors': []
        }

@health_ns.route('/metrics')
class HealthMetrics(Resource):
    @api.doc('get_health_metrics')
    @api.marshal_list_with(health_metric_model)
    @require_auth
    def get(self):
        """Get recent health metrics"""
        return [
            {
                'metric_type': 'heart_rate',
                'value': 72.0,
                'unit': 'bpm',
                'timestamp': datetime.now(),
                'confidence': 0.95
            }
        ]

# Analytics endpoints
@analytics_ns.route('/insights')
class AnalyticsInsights(Resource):
    @api.doc('get_analytics_insights')
    @require_auth
    def get(self):
        """Get AI-powered analytics insights"""
        return {
            'insights': [
                {
                    'type': 'pattern_detection',
                    'title': 'Sleep Pattern Improvement',
                    'description': 'Your sleep quality has improved by 15% over the past week',
                    'confidence': 0.87,
                    'timestamp': datetime.now()
                }
            ]
        }

# Sensor endpoints
@sensors_ns.route('/data')
class SensorData(Resource):
    @api.doc('get_sensor_data')
    @api.marshal_list_with(sensor_data_model)
    @require_auth
    def get(self):
        """Get recent sensor data"""
        return [
            {
                'sensor_id': 'hr_001',
                'sensor_type': 'heart_rate',
                'timestamp': datetime.now(),
                'values': {'bpm': 72, 'variability': 45},
                'quality': 0.95
            }
        ]

# ML endpoints
@ml_ns.route('/models')
class MLModels(Resource):
    @api.doc('list_ml_models')
    @api.marshal_list_with(ml_model_info)
    @require_auth
    def get(self):
        """Get available ML models"""
        return [
            {
                'model_id': 'health_predictor_v1',
                'name': 'Health Risk Predictor',
                'type': 'classification',
                'version': '1.2.3',
                'accuracy': 0.94,
                'last_trained': datetime.now(),
                'status': 'active'
            }
        ]

# Privacy endpoints
@privacy_ns.route('/status')
class PrivacyStatus(Resource):
    @api.doc('get_privacy_status')
    @require_auth
    def get(self):
        """Get privacy and security status"""
        return {
            'encryption_status': 'active',
            'data_anonymization': 'enabled',
            'consent_status': 'granted',
            'data_retention_days': 365,
            'last_audit': datetime.now()
        }

# Error handlers
@api.errorhandler(400)
def bad_request(error):
    return {'error': 'Bad request', 'code': 400, 'timestamp': datetime.now()}, 400

@api.errorhandler(401)
def unauthorized(error):
    return {'error': 'Unauthorized', 'code': 401, 'timestamp': datetime.now()}, 401

@api.errorhandler(404)
def not_found(error):
    return {'error': 'Resource not found', 'code': 404, 'timestamp': datetime.now()}, 404

@api.errorhandler(500)
def internal_error(error):
    return {'error': 'Internal server error', 'code': 500, 'timestamp': datetime.now()}, 500

# Health check endpoint
@api_bp.route('/health')
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now(),
        'services': {
            'database': 'connected',
            'redis': 'connected',
            'ml_models': 'loaded'
        }
    })

# API statistics endpoint
@api_bp.route('/stats')
def api_stats():
    """API usage statistics"""
    return jsonify({
        'total_requests': 12543,
        'active_devices': 156,
        'data_points_processed': 2847392,
        'ml_predictions_made': 8934,
        'uptime_hours': 720.5
    }) 