"""
OpenWearables Core Architecture

This module provides the main OpenWearablesCore class that orchestrates all
components of the wearable health monitoring platform.
"""

import os
import time
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import base64

# Conditional imports for hardware acceleration
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import core components
from .sensor_manager import SensorManager
from .data_processor import DataProcessor
from .health_analyzer import HealthAnalyzer
from .privacy import PrivacyManager

logger = logging.getLogger("OpenWearables.Core")

class OpenWearablesCore:
    """
    Core architecture for the OpenWearables platform.
    
    This class orchestrates all components including sensor management,
    data processing, health analysis, and privacy protection.
    """
    
    def __init__(self, config_path: str = "config/default.json", auto_optimize: bool = True):
        """
        Initialize the OpenWearables core system.
        
        Args:
            config_path: Path to configuration file
            auto_optimize: Whether to automatically optimize for available hardware
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.device = self._detect_device() if auto_optimize else "cpu"
        self.running = False
        self.data_lock = threading.Lock()
        
        logger.info(f"Initializing OpenWearables core on {self.device}")
        
        # Initialize database
        self.db_path = self.config.get("database", {}).get("path", "data/wearables.db")
        db_dir = os.path.dirname(self.db_path)
        if db_dir:  # Only create directory if there is one
            os.makedirs(db_dir, exist_ok=True)
        self._initialize_database()
        
        # Initialize core components
        sensors_config = self.config.get("sensors", [])
        
        # Handle different sensor configuration formats
        if isinstance(sensors_config, dict):
            # If sensors_config is a dictionary, extract the enabled sensors
            if "enabled" in sensors_config:
                # Configuration has explicit "enabled" list
                actual_sensor_types = sensors_config["enabled"]
            else:
                # Legacy format: assume keys are sensor types (excluding common config keys)
                config_keys = {"enabled", "sampling_rates", "calibration", "auto_calibrate", "calibration_interval", "calibration_samples"}
                actual_sensor_types = [key for key in sensors_config.keys() if key not in config_keys]
        elif isinstance(sensors_config, list):
            # If it's a list, assume it's a list of sensor type strings
            actual_sensor_types = sensors_config
        else:
            logger.warning(f"Unexpected type for 'sensors' in config: {type(sensors_config)}. Using default sensors.")
            actual_sensor_types = ["ecg", "ppg", "accelerometer", "gyroscope", "temperature"]

        # Ensure we have some sensors enabled
        if not actual_sensor_types:
            logger.warning("No sensors enabled in configuration. Using default sensors.")
            actual_sensor_types = ["ecg", "ppg", "accelerometer", "gyroscope", "temperature"]

        logger.info(f"Initializing sensors: {actual_sensor_types}")

        self.sensor_manager = SensorManager(
            sensor_types=actual_sensor_types,
            sampling_rates=self.config.get("sensors", {}).get("sampling_rates", {}),
            config=self.config
        )
        
        self.data_processor = DataProcessor(
            config=self.config.get("processing", {})
        )
        
        self.health_analyzer = HealthAnalyzer(
            config=self.config,
            device=self.device
        )
        
        self.privacy_manager = PrivacyManager(
            config=self.config.get("privacy", {})
        )
        
        # Initialize AI models
        self.models = {}
        self._load_models()
        
        # Data buffers for real-time processing
        self.data_buffers = {}
        self.latest_analysis = {}
        
        logger.info("OpenWearables core initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, creating default")
            
            # Create default configuration
            default_config = {
                "database": {"path": "data/wearables.db"},
                "sensors": ["ecg", "ppg", "accelerometer", "gyroscope", "temperature"],
                "sampling_rates": {
                    "ecg": 250,
                    "ppg": 100,
                    "accelerometer": 50,
                    "gyroscope": 50,
                    "temperature": 1
                },
                "models": {
                    "arrhythmia_detection": "openwearables/arrhythmia-detection",
                    "stress_analysis": "openwearables/stress-analysis",
                    "activity_recognition": "openwearables/activity-recognition",
                    "health_assessment": "microsoft/DialoGPT-medium"
                },
                "processing": {
                    "window_size": 10,
                    "overlap": 0.5,
                    "features": ["time_domain", "frequency_domain", "wavelet"]
                },
                "privacy": {
                    "encryption": True,
                    "anonymization": True,
                    "data_retention": 90
                },
                "logging": {
                    "level": "INFO",
                    "file": "logs/openwearables.log"
                },
                "user_profile": {
                    "name": "",
                    "age": None,
                    "gender": "",
                    "height": None,
                    "weight": None,
                    "medical_conditions": "",
                    "medications": ""
                }
            }
            
            # Ensure config directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            logger.info(f"Created default configuration at {config_path}")
            return default_config
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file {config_path}: {e}")
            raise
    
    def _detect_device(self) -> str:
        """Detect the best available computing device."""
        if HAS_MLX and self._is_apple_silicon():
            return "mlx"
        elif HAS_TORCH and torch.cuda.is_available():
            # Check for modern NVIDIA GPU capabilities
            if torch.cuda.get_device_capability()[0] >= 7:  # Ampere or newer
                return "cuda_amp"
            return "cuda"
        elif HAS_TORCH and torch.backends.mps.is_available():
            return "mps"  # Apple Metal Performance Shaders
        elif HAS_TORCH:
            return "torch_cpu"
        else:
            return "cpu"
    
    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        import platform
        return platform.processor() == 'arm' and platform.system() == 'Darwin'
    
    def _initialize_database(self) -> None:
        """Set up SQLite database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sensors metadata table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensors (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            sampling_rate REAL NOT NULL,
            last_calibration TEXT,
            status TEXT DEFAULT 'inactive'
        )
        ''')
        
        # Sensor readings table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY,
            sensor_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            value BLOB NOT NULL,
            processed INTEGER DEFAULT 0,
            FOREIGN KEY (sensor_id) REFERENCES sensors (id)
        )
        ''')
        
        # Analysis results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            analysis_type TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            user_id TEXT
        )
        ''')
        
        # Health metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_metrics (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            metric_type TEXT NOT NULL,
            value REAL NOT NULL,
            units TEXT,
            user_id TEXT
        )
        ''')
        
        # Alerts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            acknowledged INTEGER DEFAULT 0,
            user_id TEXT
        )
        ''')
        
        # User sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL,
            session_type TEXT,
            metadata TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized with required tables")
    
    def _load_models(self) -> None:
        """Load AI models based on configuration and device."""
        model_configs = self.config.get("models", {})
        
        for model_name, model_path in model_configs.items():
            try:
                if model_name == "health_assessment":
                    # LLM models are handled by HealthAnalyzer
                    continue
                
                if self.device == "mlx":
                    from openwearables.models.mlx_models import load_model
                    self.models[model_name] = load_model(model_path, model_name)
                elif self.device.startswith("cuda"):
                    from openwearables.models.torch_models import load_model
                    self.models[model_name] = load_model(
                        model_path, model_name, 
                        use_amp=(self.device == "cuda_amp")
                    )
                elif self.device == "mps":
                    from openwearables.models.torch_models import load_model
                    self.models[model_name] = load_model(
                        model_path, model_name, device="mps"
                    )
                else:
                    from openwearables.models.model_utils import load_fallback_model
                    self.models[model_name] = load_fallback_model(model_name)
                
                logger.info(f"Loaded model: {model_name} on {self.device}")
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                # Continue loading other models even if one fails
                continue
    
    def start(self) -> None:
        """Start the OpenWearables system."""
        if self.running:
            logger.warning("System is already running")
            return
        
        logger.info("Starting OpenWearables system")
        
        try:
            # Start sensor data collection
            self.sensor_manager.start()
            
            # Start data processing thread
            self.running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("OpenWearables system started successfully")
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.stop()
            raise
    
    def stop(self) -> None:
        """Stop the OpenWearables system."""
        logger.info("Stopping OpenWearables system")
        
        # Stop processing
        self.running = False
        
        # Stop sensors
        if hasattr(self, 'sensor_manager'):
            self.sensor_manager.stop()
        
        # Wait for processing thread to finish
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        logger.info("OpenWearables system stopped")
    
    def _processing_loop(self) -> None:
        """Main data processing loop."""
        logger.info("Starting data processing loop")
        
        while self.running:
            try:
                # Get latest sensor data
                sensor_data = self.sensor_manager.get_buffered_data(clear=True)
                
                if sensor_data:
                    # Process the data
                    processed_data = self.data_processor.process_batch(sensor_data)
                    
                    # Analyze health data
                    if processed_data:
                        analysis_results = self.health_analyzer.analyze_health_data(processed_data)
                        
                        # Apply privacy protections
                        safe_results = self.privacy_manager.sanitize_output(analysis_results)
                        
                        # Store results
                        self._store_analysis_results(safe_results)
                        
                        # Update latest analysis cache
                        with self.data_lock:
                            self.latest_analysis = safe_results
                
                # Sleep for processing interval
                time.sleep(1.0)  # Process every second
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(5.0)  # Wait longer before retrying on error
    
    def _store_analysis_results(self, results: Dict[str, Any]) -> None:
        """Store analysis results in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = time.time()
            
            for analysis_type, result in results.items():
                if isinstance(result, dict) and "confidence" in result:
                    confidence = result["confidence"]
                    result_str = json.dumps(result)
                else:
                    confidence = 0.0
                    result_str = str(result)
                
                cursor.execute(
                    "INSERT INTO analysis_results (timestamp, analysis_type, result, confidence) VALUES (?, ?, ?, ?)",
                    (timestamp, analysis_type, result_str, confidence)
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
    
    def get_latest_readings(self) -> Dict[str, Any]:
        """Get the latest sensor readings."""
        return self.sensor_manager.get_latest_readings()
    
    def get_latest_analysis(self) -> Dict[str, Any]:
        """Get the latest analysis results."""
        with self.data_lock:
            return self.latest_analysis.copy() if self.latest_analysis else {}
    
    def get_health_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get a comprehensive health summary over the specified number of days.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Dictionary containing health metrics and analysis summary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate timestamp for start of period
            start_time = time.time() - (days * 86400)
            
            # Get health metrics
            cursor.execute(
                "SELECT metric_type, AVG(value), MIN(value), MAX(value), COUNT(*) FROM health_metrics WHERE timestamp >= ? GROUP BY metric_type",
                (start_time,)
            )
            
            metrics = {}
            for row in cursor.fetchall():
                metrics[row[0]] = {
                    "average": row[1],
                    "minimum": row[2],
                    "maximum": row[3],
                    "count": row[4]
                }
            
            # Get recent analysis results
            cursor.execute(
                "SELECT analysis_type, result, confidence, timestamp FROM analysis_results WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 100",
                (start_time,)
            )
            
            analyses = {}
            for row in cursor.fetchall():
                analysis_type = row[0]
                if analysis_type not in analyses:
                    try:
                        result = json.loads(row[1])
                    except:
                        result = row[1]
                    
                    analyses[analysis_type] = {
                        "latest_result": result,
                        "confidence": row[2],
                        "timestamp": row[3]
                    }
            
            # Get alerts
            cursor.execute(
                "SELECT alert_type, COUNT(*), MAX(timestamp), severity FROM alerts WHERE timestamp >= ? GROUP BY alert_type, severity",
                (start_time,)
            )
            
            alerts = {}
            for row in cursor.fetchall():
                alert_type = row[0]
                if alert_type not in alerts:
                    alerts[alert_type] = []
                
                alerts[alert_type].append({
                    "count": row[1],
                    "latest": row[2],
                    "severity": row[3]
                })
            
            conn.close()
            
            return {
                "period_days": days,
                "metrics": metrics,
                "analyses": analyses,
                "alerts": alerts,
                "generated_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error generating health summary: {e}")
            return {
                "period_days": days,
                "metrics": {},
                "analyses": {},
                "alerts": {},
                "error": str(e)
            }
    
    def get_alerts(self, user_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get active and historical alerts from the database."""
        alerts_data = {"active_alerts": [], "alert_history": []}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get active alerts (not acknowledged)
            query = "SELECT id, timestamp, alert_type, severity, message FROM alerts WHERE acknowledged = 0"
            params = []
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " ORDER BY timestamp DESC"
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                alerts_data["active_alerts"].append({
                    "id": row[0],
                    "timestamp": row[1],
                    "type": row[2],
                    "severity": row[3],
                    "message": row[4]
                })
            
            # Get historical alerts (acknowledged)
            query = "SELECT id, timestamp, alert_type, severity, message FROM alerts WHERE acknowledged = 1"
            params = []
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " ORDER BY timestamp DESC LIMIT 50" # Limit history
            cursor.execute(query, params)

            for row in cursor.fetchall():
                alerts_data["alert_history"].append({
                    "id": row[0],
                    "timestamp": row[1],
                    "type": row[2],
                    "severity": row[3],
                    "message": row[4]
                })
            
            conn.close()
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
        return alerts_data

    def get_reports(self, user_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get available reports (placeholder implementation)."""
        # This is a placeholder. In a real system, this would query a table
        # of generated reports or have logic to list report types.
        reports_data = {
            "available_reports": [
                {"name": "Daily Health Summary", "type": "daily", "last_generated": time.time() - 86400, "description": "A summary of your health data from the past 24 hours."},
                {"name": "Weekly Trends", "type": "weekly", "last_generated": time.time() - 604800, "description": "Trends and insights from the past 7 days."},
                {"name": "Monthly Cardiac Analysis", "type": "monthly_cardiac", "last_generated": None, "description": "In-depth analysis of cardiac health over the past month."}
            ]
        }
        # If user_id is provided, one might filter reports or tailor the list.
        return reports_data
    
    def is_running(self) -> bool:
        """Check if the system is currently running."""
        return self.running and self.sensor_manager.is_running()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status information."""
        return {
            "running": self.is_running(),
            "device": self.device,
            "sensors": list(self.sensor_manager.sensors.keys()) if hasattr(self.sensor_manager, 'sensors') else [],
            "models": list(self.models.keys()),
            "database_path": self.db_path,
            "config_path": self.config_path,
            "version": "1.0.0"
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update system configuration.
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Merge with existing config
            self.config.update(new_config)
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def export_data(self, start_time: float, end_time: float, format: str = "json") -> Optional[str]:
        """
        Export health data for a specified time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            format: Export format ("json", "csv")
            
        Returns:
            Exported data as string or None if error
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            if format == "json":
                # Export as JSON
                cursor = conn.cursor()
                
                # Get readings
                cursor.execute(
                    "SELECT * FROM readings WHERE timestamp BETWEEN ? AND ?",
                    (start_time, end_time)
                )
                readings_raw = cursor.fetchall()
                columns = [col[0] for col in cursor.description]
                
                # Process readings to handle bytes
                readings = []
                for row in readings_raw:
                    reading = dict(zip(columns, row))
                    # Convert bytes to base64 string for JSON serialization
                    if 'value' in reading and isinstance(reading['value'], bytes):
                        reading['value'] = base64.b64encode(reading['value']).decode('utf-8')
                        reading['value_type'] = 'base64'
                    readings.append(reading)
                
                # Get analysis results
                cursor.execute(
                    "SELECT * FROM analysis_results WHERE timestamp BETWEEN ? AND ?",
                    (start_time, end_time)
                )
                analyses = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
                
                export_data = {
                    "export_timestamp": time.time(),
                    "start_time": start_time,
                    "end_time": end_time,
                    "readings": readings,
                    "analyses": analyses
                }
                
                conn.close()
                return json.dumps(export_data, indent=2)
                
            elif format == "csv":
                # Export as CSV (simplified)
                import pandas as pd
                
                readings_df = pd.read_sql_query(
                    "SELECT * FROM readings WHERE timestamp BETWEEN ? AND ?",
                    conn, params=(start_time, end_time)
                )
                
                # Handle bytes columns
                if 'value' in readings_df.columns:
                    readings_df['value'] = readings_df['value'].apply(
                        lambda x: base64.b64encode(x).decode('utf-8') if isinstance(x, bytes) else x
                    )
                
                conn.close()
                return readings_df.to_csv(index=False)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None 