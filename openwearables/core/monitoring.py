"""
OpenWearables Monitoring and Observability System
Comprehensive monitoring, metrics, logging, and alerting
"""

import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import asyncio
from functools import wraps
import traceback
import sys
import os

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('openwearables.log')
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class Metric:
    """Represents a system metric"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'unit': self.unit
        }

@dataclass
class Alert:
    """Represents a system alert"""
    id: str
    severity: str  # critical, warning, info
    title: str
    description: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics: deque = deque(maxlen=max_metrics)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
    def record_metric(self, metric: Metric):
        """Record a metric"""
        with self._lock:
            self.metrics.append(metric)
            
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        with self._lock:
            self.counters[name] += value
            self.record_metric(Metric(
                name=f"{name}_total",
                value=self.counters[name],
                timestamp=datetime.now(),
                tags=tags or {},
                unit="count"
            ))
            
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        with self._lock:
            self.gauges[name] = value
            self.record_metric(Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                unit="gauge"
            ))
            
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value"""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only last 1000 values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
                
            self.record_metric(Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                unit="histogram"
            ))
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self._lock:
            return {
                'total_metrics': len(self.metrics),
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histogram_counts': {k: len(v) for k, v in self.histograms.items()},
                'last_updated': datetime.now().isoformat()
            }

class SystemMonitor:
    """Monitors system resources and performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 30.0):
        """Start system monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"System monitoring started with {interval}s interval")
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
        
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
                
    def _collect_system_metrics(self):
        """Collect system metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics_collector.set_gauge("system_memory_percent", memory.percent)
        self.metrics_collector.set_gauge("system_memory_available_gb", memory.available / (1024**3))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics_collector.set_gauge("system_disk_percent", disk.percent)
        self.metrics_collector.set_gauge("system_disk_free_gb", disk.free / (1024**3))
        
        # Network metrics
        network = psutil.net_io_counters()
        self.metrics_collector.set_gauge("system_network_bytes_sent", network.bytes_sent)
        self.metrics_collector.set_gauge("system_network_bytes_recv", network.bytes_recv)
        
        # Process metrics
        process = psutil.Process()
        self.metrics_collector.set_gauge("process_memory_mb", process.memory_info().rss / (1024**2))
        self.metrics_collector.set_gauge("process_cpu_percent", process.cpu_percent())

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Callable] = []
        self._lock = threading.Lock()
        
    def add_alert_rule(self, rule: Callable[[Dict[str, Any]], Optional[Alert]]):
        """Add an alert rule"""
        self.alert_rules.append(rule)
        
    def create_alert(self, alert: Alert):
        """Create a new alert"""
        with self._lock:
            self.alerts[alert.id] = alert
            logger.warning(f"Alert created: {alert.title} - {alert.description}")
            
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
            
    def check_alerts(self, metrics_summary: Dict[str, Any]):
        """Check alert rules against current metrics"""
        for rule in self.alert_rules:
            try:
                alert = rule(metrics_summary)
                if alert:
                    self.create_alert(alert)
            except Exception as e:
                logger.error(f"Error checking alert rule: {e}")

class PerformanceTracer:
    """Traces performance of functions and operations"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_traces: Dict[str, float] = {}
        
    def trace_function(self, func_name: str = None):
        """Decorator to trace function performance"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                trace_id = f"{name}_{id(threading.current_thread())}"
                
                try:
                    self.active_traces[trace_id] = start_time
                    result = func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    self.metrics_collector.record_histogram(
                        f"function_duration_seconds",
                        duration,
                        tags={"function": name, "status": "success"}
                    )
                    self.metrics_collector.increment_counter(
                        "function_calls",
                        tags={"function": name, "status": "success"}
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    self.metrics_collector.record_histogram(
                        f"function_duration_seconds",
                        duration,
                        tags={"function": name, "status": "error"}
                    )
                    self.metrics_collector.increment_counter(
                        "function_calls",
                        tags={"function": name, "status": "error"}
                    )
                    raise
                    
                finally:
                    self.active_traces.pop(trace_id, None)
                    
            return wrapper
        return decorator
        
    def get_active_traces(self) -> Dict[str, float]:
        """Get currently active traces"""
        current_time = time.time()
        return {
            trace_id: current_time - start_time 
            for trace_id, start_time in self.active_traces.items()
        }

class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check"""
        self.health_checks[name] = check_func
        
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                healthy = check_func()
                duration = time.time() - start_time
                
                results[name] = {
                    'healthy': healthy,
                    'duration_ms': duration * 1000,
                    'timestamp': datetime.now().isoformat()
                }
                
                if not healthy:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                overall_healthy = False
                
        results['overall'] = {
            'healthy': overall_healthy,
            'timestamp': datetime.now().isoformat()
        }
        
        return results

class MonitoringSystem:
    """Main monitoring system that coordinates all components"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.alert_manager = AlertManager()
        self.performance_tracer = PerformanceTracer(self.metrics_collector)
        self.health_checker = HealthChecker()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
    def start(self):
        """Start the monitoring system"""
        self.system_monitor.start_monitoring()
        logger.info("OpenWearables monitoring system started")
        
    def stop(self):
        """Stop the monitoring system"""
        self.system_monitor.stop_monitoring()
        logger.info("OpenWearables monitoring system stopped")
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        return {
            'metrics_summary': self.metrics_collector.get_metrics_summary(),
            'active_alerts': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
            'health_status': self.health_checker.run_health_checks(),
            'active_traces': self.performance_tracer.get_active_traces(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
        }
        
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        
        def high_cpu_alert(metrics: Dict[str, Any]) -> Optional[Alert]:
            cpu_percent = metrics.get('gauges', {}).get('system_cpu_percent', 0)
            if cpu_percent > 90:
                return Alert(
                    id="high_cpu",
                    severity="warning",
                    title="High CPU Usage",
                    description=f"CPU usage is {cpu_percent:.1f}%",
                    timestamp=datetime.now()
                )
            return None
            
        def high_memory_alert(metrics: Dict[str, Any]) -> Optional[Alert]:
            memory_percent = metrics.get('gauges', {}).get('system_memory_percent', 0)
            if memory_percent > 90:
                return Alert(
                    id="high_memory",
                    severity="critical",
                    title="High Memory Usage",
                    description=f"Memory usage is {memory_percent:.1f}%",
                    timestamp=datetime.now()
                )
            return None
            
        def low_disk_space_alert(metrics: Dict[str, Any]) -> Optional[Alert]:
            disk_percent = metrics.get('gauges', {}).get('system_disk_percent', 0)
            if disk_percent > 95:
                return Alert(
                    id="low_disk_space",
                    severity="critical",
                    title="Low Disk Space",
                    description=f"Disk usage is {disk_percent:.1f}%",
                    timestamp=datetime.now()
                )
            return None
            
        self.alert_manager.add_alert_rule(high_cpu_alert)
        self.alert_manager.add_alert_rule(high_memory_alert)
        self.alert_manager.add_alert_rule(low_disk_space_alert)
        
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        def database_health() -> bool:
            # Mock database health check
            return True
            
        def redis_health() -> bool:
            # Mock Redis health check
            return True
            
        def ml_models_health() -> bool:
            # Mock ML models health check
            return True
            
        self.health_checker.register_health_check("database", database_health)
        self.health_checker.register_health_check("redis", redis_health)
        self.health_checker.register_health_check("ml_models", ml_models_health)

# Global monitoring instance
monitoring_system = MonitoringSystem()

# Convenience decorators
def monitor_performance(func_name: str = None):
    """Decorator to monitor function performance"""
    return monitoring_system.performance_tracer.trace_function(func_name)

def record_metric(name: str, value: float, tags: Dict[str, str] = None):
    """Record a custom metric"""
    monitoring_system.metrics_collector.record_metric(
        Metric(name=name, value=value, timestamp=datetime.now(), tags=tags or {})
    )

def increment_counter(name: str, value: int = 1, tags: Dict[str, str] = None):
    """Increment a counter"""
    monitoring_system.metrics_collector.increment_counter(name, value, tags)

def set_gauge(name: str, value: float, tags: Dict[str, str] = None):
    """Set a gauge value"""
    monitoring_system.metrics_collector.set_gauge(name, value, tags)

# Example usage
if __name__ == "__main__":
    # Start monitoring
    monitoring_system.start()
    
    # Example of using decorators
    @monitor_performance("example_function")
    def example_function():
        time.sleep(0.1)
        return "Hello World"
    
    # Run example
    result = example_function()
    
    # Record custom metrics
    record_metric("custom_metric", 42.0, {"type": "example"})
    increment_counter("api_requests", tags={"endpoint": "/health"})
    set_gauge("active_users", 150.0)
    
    # Get dashboard data
    dashboard = monitoring_system.get_dashboard_data()
    print(json.dumps(dashboard, indent=2, default=str))
    
    # Stop monitoring
    time.sleep(2)
    monitoring_system.stop() 