# OpenWearables API Documentation

## Overview

The OpenWearables platform provides a comprehensive REST API for managing wearable health sensors, processing data, and generating health insights. All API endpoints support JSON format and include proper error handling, rate limiting, and authentication.

## Base URL

```
http://localhost:5000/api
```

## Authentication

Most API endpoints require authentication. Include the session token in your requests:

```bash
curl -H "Authorization: Bearer YOUR_SESSION_TOKEN" \
     -H "Content-Type: application/json" \
     http://localhost:5000/api/data/latest
```

## Rate Limiting

- Default rate limit: 100 requests per minute
- Rate limit headers included in responses:
  - `X-RateLimit-Limit`: Total requests allowed
  - `X-RateLimit-Remaining`: Requests remaining
  - `X-RateLimit-Reset`: Time when limit resets

## Error Handling

All API responses follow a consistent error format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": "Additional error details"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized  
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

## Endpoints

### System Management

#### Get System Status

```http
GET /api/system/status
```

Returns current system status and hardware information.

**Response:**
```json
{
  "success": true,
  "data": {
    "running": true,
    "device": "mlx",
    "sensors_active": 5,
    "processing_rate": 250.5,
    "memory_usage": 45.2,
    "cpu_usage": 23.1,
    "uptime": 3600,
    "version": "1.0.0"
  }
}
```

### Python SDK

```python
import requests

class OpenWearablesClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_latest_data(self, sensors=None):
        url = f"{self.base_url}/api/data/latest"
        params = {"sensors": ",".join(sensors)} if sensors else {}
        response = self.session.get(url, params=params)
        return response.json()

# Usage
client = OpenWearablesClient()
data = client.get_latest_data(sensors=["ecg", "ppg"])
``` 