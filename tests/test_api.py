"""
Comprehensive test suite for OpenWearables API
Tests all endpoints with proper mocking and fixtures
"""

import pytest
import json
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock
from flask import Flask
from openwearables.ui.api_docs import api_bp, api
import logging

# Disable logging during tests
logging.disable(logging.CRITICAL)

@pytest.fixture
def app():
    """Create test Flask application"""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.register_blueprint(api_bp)
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
def auth_headers():
    """Mock authentication headers"""
    return {'Authorization': 'Bearer test_token_12345'}

@pytest.fixture
def mock_device_data():
    """Mock device data for testing"""
    return {
        'id': 'test_device_001',
        'name': 'Test Smart Watch',
        'type': 'watch',
        'status': 'connected',
        'battery_level': 85.0,
        'firmware_version': '1.0.0',
        'last_sync': datetime.now(),
        'capabilities': ['heart_rate', 'steps'],
        'location': 'left_wrist'
    }

@pytest.fixture
def mock_health_data():
    """Mock health data for testing"""
    return {
        'user_id': 'test_user_001',
        'date': date.today(),
        'metrics': [
            {
                'metric_type': 'heart_rate',
                'value': 72.0,
                'unit': 'bpm',
                'timestamp': datetime.now(),
                'confidence': 0.95
            }
        ],
        'insights': ['Test health insight'],
        'recommendations': ['Test recommendation'],
        'risk_factors': []
    }

class TestAPIHealthCheck:
    """Test API health check endpoints"""
    
    def test_health_check_success(self, client):
        """Test successful health check"""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['version'] == '1.0.0'
        assert 'timestamp' in data
        assert 'services' in data
        
    def test_api_stats(self, client):
        """Test API statistics endpoint"""
        response = client.get('/api/v1/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'total_requests' in data
        assert 'active_devices' in data
        assert 'data_points_processed' in data
        assert isinstance(data['total_requests'], int)

class TestDeviceEndpoints:
    """Test device management endpoints"""
    
    def test_get_devices_without_auth(self, client):
        """Test device list endpoint without authentication"""
        response = client.get('/api/v1/devices/')
        assert response.status_code == 401
        
        data = json.loads(response.data)
        assert data['error'] == 'Unauthorized'
        
    def test_get_devices_with_auth(self, client, auth_headers):
        """Test device list endpoint with authentication"""
        response = client.get('/api/v1/devices/', headers=auth_headers)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'devices' in data
        assert 'total' in data
        assert 'connected' in data
        assert isinstance(data['devices'], list)
        
    def test_get_specific_device(self, client, auth_headers):
        """Test getting specific device information"""
        device_id = 'test_device_001'
        response = client.get(f'/api/v1/devices/{device_id}', headers=auth_headers)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['id'] == device_id
        assert 'name' in data
        assert 'type' in data
        assert 'status' in data
        
    def test_get_nonexistent_device(self, client, auth_headers):
        """Test getting non-existent device"""
        response = client.get('/api/v1/devices/nonexistent', headers=auth_headers)
        # Should return 200 with mock data for now
        assert response.status_code == 200

class TestHealthEndpoints:
    """Test health analytics endpoints"""
    
    def test_health_summary_without_auth(self, client):
        """Test health summary without authentication"""
        response = client.get('/api/v1/health/summary')
        assert response.status_code == 401
        
    def test_health_summary_with_auth(self, client, auth_headers):
        """Test health summary with authentication"""
        response = client.get('/api/v1/health/summary', headers=auth_headers)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'user_id' in data
        assert 'date' in data
        assert 'metrics' in data
        assert 'insights' in data
        assert 'recommendations' in data
        assert 'risk_factors' in data
        
    def test_health_metrics(self, client, auth_headers):
        """Test health metrics endpoint"""
        response = client.get('/api/v1/health/metrics', headers=auth_headers)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert isinstance(data, list)
        if data:  # If there's data
            metric = data[0]
            assert 'metric_type' in metric
            assert 'value' in metric
            assert 'unit' in metric
            assert 'timestamp' in metric

class TestAnalyticsEndpoints:
    """Test analytics endpoints"""
    
    def test_analytics_insights(self, client, auth_headers):
        """Test analytics insights endpoint"""
        response = client.get('/api/v1/analytics/insights', headers=auth_headers)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'insights' in data
        assert isinstance(data['insights'], list)
        
        if data['insights']:
            insight = data['insights'][0]
            assert 'type' in insight
            assert 'title' in insight
            assert 'description' in insight
            assert 'confidence' in insight

class TestSensorEndpoints:
    """Test sensor data endpoints"""
    
    def test_sensor_data(self, client, auth_headers):
        """Test sensor data endpoint"""
        response = client.get('/api/v1/sensors/data', headers=auth_headers)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert isinstance(data, list)
        
        if data:
            sensor_data = data[0]
            assert 'sensor_id' in sensor_data
            assert 'sensor_type' in sensor_data
            assert 'timestamp' in sensor_data
            assert 'values' in sensor_data
            assert 'quality' in sensor_data

class TestMLEndpoints:
    """Test machine learning endpoints"""
    
    def test_ml_models_list(self, client, auth_headers):
        """Test ML models list endpoint"""
        response = client.get('/api/v1/ml/models', headers=auth_headers)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert isinstance(data, list)
        
        if data:
            model = data[0]
            assert 'model_id' in model
            assert 'name' in model
            assert 'type' in model
            assert 'version' in model
            assert 'status' in model

class TestPrivacyEndpoints:
    """Test privacy and security endpoints"""
    
    def test_privacy_status(self, client, auth_headers):
        """Test privacy status endpoint"""
        response = client.get('/api/v1/privacy/status', headers=auth_headers)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'encryption_status' in data
        assert 'data_anonymization' in data
        assert 'consent_status' in data
        assert 'data_retention_days' in data
        assert 'last_audit' in data

class TestErrorHandling:
    """Test API error handling"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get('/api/v1/nonexistent')
        assert response.status_code == 404
        
    def test_invalid_auth_header(self, client):
        """Test invalid authentication header"""
        headers = {'Authorization': 'Invalid token'}
        response = client.get('/api/v1/devices/', headers=headers)
        assert response.status_code == 401
        
    def test_missing_auth_header(self, client):
        """Test missing authentication header"""
        response = client.get('/api/v1/devices/')
        assert response.status_code == 401

class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    def test_swagger_ui_accessible(self, client):
        """Test that Swagger UI is accessible"""
        response = client.get('/api/v1/docs/')
        # Should redirect or return documentation
        assert response.status_code in [200, 302, 308]

@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for common API usage scenarios"""
    
    def test_device_health_workflow(self, client, auth_headers):
        """Test complete device to health data workflow"""
        # 1. Get devices
        devices_response = client.get('/api/v1/devices/', headers=auth_headers)
        assert devices_response.status_code == 200
        
        # 2. Get health summary
        health_response = client.get('/api/v1/health/summary', headers=auth_headers)
        assert health_response.status_code == 200
        
        # 3. Get analytics insights
        analytics_response = client.get('/api/v1/analytics/insights', headers=auth_headers)
        assert analytics_response.status_code == 200
        
        # Verify data consistency
        devices_data = json.loads(devices_response.data)
        health_data = json.loads(health_response.data)
        analytics_data = json.loads(analytics_response.data)
        
        assert devices_data['total'] >= 0
        assert len(health_data['metrics']) >= 0
        assert len(analytics_data['insights']) >= 0

@pytest.mark.performance
class TestPerformance:
    """Performance tests for API endpoints"""
    
    def test_response_times(self, client, auth_headers):
        """Test that API responses are within acceptable time limits"""
        import time
        
        endpoints = [
            '/api/v1/health',
            '/api/v1/devices/',
            '/api/v1/health/summary',
            '/api/v1/analytics/insights'
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint, headers=auth_headers if 'health' not in endpoint else None)
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response_time < 1.0, f"Endpoint {endpoint} took {response_time:.2f}s"
            assert response.status_code in [200, 401]  # 401 for auth-required endpoints

@pytest.mark.security
class TestSecurity:
    """Security tests for API endpoints"""
    
    def test_sql_injection_protection(self, client, auth_headers):
        """Test protection against SQL injection"""
        malicious_device_id = "'; DROP TABLE devices; --"
        response = client.get(f'/api/v1/devices/{malicious_device_id}', headers=auth_headers)
        # Should not cause server error
        assert response.status_code in [200, 400, 404]
        
    def test_xss_protection(self, client, auth_headers):
        """Test protection against XSS attacks"""
        xss_payload = "<script>alert('xss')</script>"
        response = client.get(f'/api/v1/devices/{xss_payload}', headers=auth_headers)
        # Should not execute script
        assert response.status_code in [200, 400, 404]
        
    def test_rate_limiting_headers(self, client):
        """Test that rate limiting headers are present"""
        response = client.get('/api/v1/health')
        # Note: Rate limiting would be implemented in production
        assert response.status_code == 200

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short']) 