# OpenWearables Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the OpenWearables platform in production environments with enterprise-grade reliability, security, and performance.

## Architecture

### System Components

- **Core Platform**: Python-based wearable health monitoring system
- **Web Interface**: Modern HTML5/CSS3/JavaScript dashboard
- **API Gateway**: RESTful APIs with comprehensive documentation
- **Data Pipeline**: Real-time sensor data processing
- **ML Engine**: Health analytics and AI-powered insights
- **Monitoring**: Comprehensive observability stack

### Technology Stack

- **Backend**: Python 3.11+, Flask, SQLAlchemy, Celery
- **Frontend**: HTML5, CSS3, JavaScript ES6+, Plotly.js
- **Database**: PostgreSQL (production), SQLite (development)
- **Cache**: Redis
- **ML/AI**: scikit-learn, TensorFlow, MLX (Apple Silicon)
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes
- **Monitoring**: Prometheus, Grafana
- **CI/CD**: GitHub Actions

## Prerequisites

### System Requirements

#### Minimum Requirements
- CPU: 4 cores, 2.5 GHz
- Memory: 8 GB RAM
- Storage: 100 GB SSD
- Network: 100 Mbps

#### Recommended Requirements
- CPU: 8 cores, 3.0 GHz
- Memory: 16 GB RAM
- Storage: 500 GB NVMe SSD
- Network: 1 Gbps

### Software Dependencies

- Docker 24.0+
- Docker Compose 2.0+
- Kubernetes 1.28+ (for K8s deployment)
- Python 3.11+
- Node.js 18+ (for asset building)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/openwearables/openwearables.git
cd openwearables
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Docker Deployment

```bash
# Production deployment
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Open dashboard
open http://localhost:8000
```

## Configuration

### Environment Variables

```bash
# Application
FLASK_ENV=production
SECRET_KEY=your-256-bit-secret-key
JWT_SECRET_KEY=your-jwt-secret-key

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/openwearables
REDIS_URL=redis://redis:6379/0

# Security
HTTPS=true
SSL_CERT_PATH=/etc/ssl/certs/openwearables.crt
SSL_KEY_PATH=/etc/ssl/private/openwearables.key

# Monitoring
PROMETHEUS_ENABLED=true
SENTRY_DSN=your-sentry-dsn

# External Services
OPENAI_API_KEY=your-openai-key
HF_TOKEN=your-huggingface-token
```

### Configuration Files

#### config/production.json
```json
{
  "device_name": "OpenWearables Production Platform",
  "environment": "production",
  "hardware": {
    "use_mlx": true,
    "use_cuda": true,
    "optimization_level": "maximum"
  },
  "sensors": {
    "enabled": ["ecg", "ppg", "accelerometer", "gyroscope", "temperature"],
    "sampling_rates": {
      "ecg": 250,
      "ppg": 100,
      "accelerometer": 50,
      "gyroscope": 50,
      "temperature": 1
    }
  },
  "processing": {
    "real_time": true,
    "buffer_size": 10000,
    "features": ["time_domain", "frequency_domain", "wavelet"]
  },
  "privacy": {
    "encryption": true,
    "anonymization": true,
    "data_retention": 90,
    "audit_logging": true
  },
  "api": {
    "rate_limit": 1000,
    "authentication": true,
    "cors_enabled": false
  }
}
```

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl apply -f k8s/production/namespace.yaml
```

### 2. Deploy Database

```bash
# PostgreSQL
kubectl apply -f k8s/production/postgres.yaml

# Redis
kubectl apply -f k8s/production/redis.yaml
```

### 3. Deploy Application

```bash
# Main application
kubectl apply -f k8s/production/deployment.yaml
kubectl apply -f k8s/production/service.yaml

# Ingress
kubectl apply -f k8s/production/ingress.yaml
```

### 4. Monitor Deployment

```bash
# Check pods
kubectl get pods -n openwearables-production

# Check logs
kubectl logs -f deployment/openwearables-app -n openwearables-production

# Check services
kubectl get svc -n openwearables-production
```

## Security Configuration

### SSL/TLS Setup

#### Generate Certificates

```bash
# Self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Let's Encrypt (production)
certbot certonly --nginx -d openwearables.example.com
```

#### Configure Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name openwearables.example.com;
    
    ssl_certificate /etc/letsencrypt/live/openwearables.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/openwearables.example.com/privkey.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://openwearables-app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Authentication Setup

```python
# JWT Configuration
JWT_SECRET_KEY = os.environ['JWT_SECRET_KEY']
JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'openwearables'
    static_configs:
      - targets: ['openwearables-app:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import the pre-built dashboard from `monitoring/grafana/openwearables-dashboard.json`

Key metrics monitored:
- Request rate and latency
- Error rate
- System resources (CPU, memory, disk)
- Database performance
- Health data quality metrics

### Alerting Rules

```yaml
# alerts.yml
groups:
  - name: openwearables
    rules:
      - alert: HighErrorRate
        expr: rate(flask_http_request_exceptions_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
          
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, flask_http_request_duration_seconds_bucket) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High response time detected
```

## Performance Optimization

### Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_sensor_data_timestamp ON sensor_data(timestamp);
CREATE INDEX idx_user_data_user_id ON user_data(user_id);
CREATE INDEX idx_health_analysis_date ON health_analysis(analysis_date);
```

### Caching Strategy

```python
# Redis caching configuration
CACHE_TYPE = "redis"
CACHE_REDIS_URL = os.environ['REDIS_URL']
CACHE_DEFAULT_TIMEOUT = 300
```

### Load Balancing

```yaml
# HAProxy configuration
backend openwearables_backend
    balance roundrobin
    server app1 openwearables-app-1:8000 check
    server app2 openwearables-app-2:8000 check
    server app3 openwearables-app-3:8000 check
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup-db.sh
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="openwearables"

# Create backup
pg_dump $DATABASE_URL > "$BACKUP_DIR/openwearables_$DATE.sql"

# Compress
gzip "$BACKUP_DIR/openwearables_$DATE.sql"

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "openwearables_*.sql.gz" -mtime +30 -delete
```

### Application Backup

```bash
#!/bin/bash
# backup-app.sh
tar -czf "openwearables_app_$(date +%Y%m%d).tar.gz" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='node_modules' \
    ./
```

## Scaling and High Availability

### Horizontal Scaling

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openwearables-app
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

### Auto-scaling

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: openwearables-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: openwearables-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Troubleshooting

### Common Issues

#### Port Already in Use (macOS)
```bash
# Disable AirPlay Receiver
sudo launchctl unload -w /System/Library/LaunchDaemons/com.apple.AirPlayXPCHelper.plist

# Or use different port
python -m openwearables.ui.app --port 5001
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
kubectl get pods -l app=postgres

# Check logs
kubectl logs -f deployment/postgres
```

#### Memory Issues
```bash
# Check memory usage
kubectl top pods -n openwearables-production

# Increase memory limits
kubectl patch deployment openwearables-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"openwearables","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
```

### Debug Mode

```bash
# Enable debug mode
export FLASK_DEBUG=true
python -m openwearables.ui.app --debug --port 5001
```

### Log Analysis

```bash
# View application logs
docker-compose logs -f openwearables

# Search for errors
docker-compose logs openwearables | grep ERROR

# Monitor real-time logs
tail -f logs/openwearables.log
```

## Maintenance

### Regular Tasks

1. **Daily**
   - Monitor system health
   - Check error logs
   - Verify backup completion

2. **Weekly**
   - Review performance metrics
   - Update security patches
   - Clean temporary files

3. **Monthly**
   - Security audit
   - Performance optimization
   - Capacity planning

### Update Procedure

```bash
# 1. Backup current deployment
kubectl create backup production-backup-$(date +%Y%m%d)

# 2. Deploy new version
kubectl set image deployment/openwearables-app openwearables=ghcr.io/openwearables/openwearables:v1.1.0

# 3. Monitor rollout
kubectl rollout status deployment/openwearables-app

# 4. Verify functionality
curl -f https://openwearables.example.com/api/v1/health

# 5. Rollback if needed
kubectl rollout undo deployment/openwearables-app
```

## Support and Documentation

- **Documentation**: https://docs.openwearables.com
- **API Reference**: https://api.openwearables.com/docs
- **Issues**: https://github.com/openwearables/openwearables/issues
- **Discussions**: https://github.com/openwearables/openwearables/discussions
- **Email Support**: support@openwearables.com

## License

This deployment guide is part of the OpenWearables project, licensed under the MIT License.

---

**OpenWearables Production Deployment Guide v1.0**  
*Last updated: 2024-01-15* 