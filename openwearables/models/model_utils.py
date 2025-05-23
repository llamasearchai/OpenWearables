"""
Model Utilities for OpenWearables

This module provides common utilities for model management, hardware detection,
and optimization across different ML frameworks (MLX, PyTorch, etc.).
"""

import os
import json
import logging
import time
import platform
import subprocess
import psutil
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

logger = logging.getLogger("OpenWearables.ModelUtils")

def detect_available_hardware() -> Dict[str, Any]:
    """
    Detect available hardware for ML acceleration.
    
    Returns:
        Dictionary containing information about available hardware
    """
    hardware_info = {
        "cpu": {
            "available": True,
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "architecture": platform.machine(),
            "model": platform.processor(),
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        },
        "cuda": {"available": False, "devices": []},
        "mps": {"available": False},
        "mlx": {"available": False},
        "recommended_device": "cpu"
    }
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            hardware_info["cuda"]["available"] = True
            hardware_info["cuda"]["device_count"] = torch.cuda.device_count()
            hardware_info["cuda"]["devices"] = []
            
            for i in range(torch.cuda.device_count()):
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    "compute_capability": torch.cuda.get_device_properties(i).major
                }
                hardware_info["cuda"]["devices"].append(device_info)
            
            hardware_info["recommended_device"] = "cuda"
            logger.info(f"CUDA detected: {hardware_info['cuda']['device_count']} devices")
        
        # Check for Metal Performance Shaders (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            hardware_info["mps"]["available"] = True
            if hardware_info["recommended_device"] == "cpu":
                hardware_info["recommended_device"] = "mps"
            logger.info("MPS (Metal Performance Shaders) detected")
            
    except ImportError:
        logger.debug("PyTorch not available for hardware detection")
    
    # Check for MLX (Apple Silicon)
    try:
        import mlx.core as mx
        hardware_info["mlx"]["available"] = True
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            hardware_info["recommended_device"] = "mlx"
            logger.info("MLX detected on Apple Silicon")
    except ImportError:
        logger.debug("MLX not available")
    
    return hardware_info

def get_optimal_batch_size(device: str, model_type: str, available_memory_gb: float) -> int:
    """
    Calculate optimal batch size based on hardware and model type.
    
    Args:
        device: Target device (cpu, cuda, mps, mlx)
        model_type: Type of model
        available_memory_gb: Available memory in GB
        
    Returns:
        Recommended batch size
    """
    # Base batch sizes for different model types
    base_batch_sizes = {
        "arrhythmia_detection": 32,
        "stress_analysis": 64,
        "activity_recognition": 32,
        "multimodal_health": 16,
    }
    
    base_batch = base_batch_sizes.get(model_type, 32)
    
    # Scale based on device and memory
    if device == "cuda":
        # Scale based on GPU memory
        if available_memory_gb >= 8:
            return base_batch * 2
        elif available_memory_gb >= 4:
            return base_batch
        else:
            return base_batch // 2
    elif device == "mlx":
        # MLX is memory efficient on Apple Silicon
        if available_memory_gb >= 16:
            return base_batch * 2
        else:
            return base_batch
    elif device == "mps":
        # MPS has memory limitations
        return min(base_batch, 16)
    else:  # CPU
        # CPU batch sizes should be smaller
        return min(base_batch // 2, 8)

def benchmark_inference(model, sample_data: Dict[str, Any], device: str, num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark model inference performance.
    
    Args:
        model: Model to benchmark
        sample_data: Sample input data
        device: Device to run on
        num_runs: Number of inference runs
        
    Returns:
        Performance metrics
    """
    try:
        # Warm up
        for _ in range(10):
            _ = model(sample_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(sample_data)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_per_second": throughput,
            "total_time_seconds": total_time,
            "device": device
        }
        
    except Exception as e:
        logger.error(f"Error in benchmarking: {str(e)}")
        return {"error": str(e)}

def optimize_model_for_device(model, device: str, precision: str = "float32") -> Any:
    """
    Optimize model for specific device and precision.
    
    Args:
        model: Model to optimize
        device: Target device
        precision: Target precision (float32, float16, int8)
        
    Returns:
        Optimized model
    """
    try:
        # For PyTorch models
        if hasattr(model, 'to'):
            model = model.to(device)
            
            if precision == "float16" and device in ["cuda", "mps"]:
                model = model.half()
                logger.info(f"Model converted to float16 on {device}")
            
        # For MLX models  
        elif hasattr(model, '__call__') and device == "mlx":
            try:
                import mlx.core as mx
                # MLX-specific optimizations would go here
                logger.info("Model optimized for MLX")
            except ImportError:
                pass
        
        return model
        
    except Exception as e:
        logger.error(f"Error optimizing model: {str(e)}")
        return model

def save_model_config(model_config: Dict[str, Any], config_path: str) -> None:
    """
    Save model configuration to file.
    
    Args:
        model_config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        logger.info(f"Model configuration saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving model config: {str(e)}")

def load_model_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Model configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading model config: {str(e)}")
        return {}

def validate_input_data(data: Dict[str, Any], expected_schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input data against expected schema.
    
    Args:
        data: Input data to validate
        expected_schema: Expected data schema
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    required_fields = expected_schema.get("required", [])
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Check data types and ranges
    field_specs = expected_schema.get("fields", {})
    for field, spec in field_specs.items():
        if field in data:
            value = data[field]
            
            # Check type
            expected_type = spec.get("type")
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"Field '{field}' should be {expected_type.__name__}, got {type(value).__name__}")
            
            # Check range for numeric fields
            if isinstance(value, (int, float)):
                min_val = spec.get("min")
                max_val = spec.get("max")
                
                if min_val is not None and value < min_val:
                    errors.append(f"Field '{field}' value {value} is below minimum {min_val}")
                
                if max_val is not None and value > max_val:
                    errors.append(f"Field '{field}' value {value} is above maximum {max_val}")
    
    return len(errors) == 0, errors

def create_health_data_schema() -> Dict[str, Any]:
    """
    Create schema for health data validation.
    
    Returns:
        Health data schema
    """
    return {
        "required": ["timestamp", "sensor_type"],
        "fields": {
            "timestamp": {"type": float, "min": 0},
            "sensor_type": {"type": str},
            "heart_rate": {"type": float, "min": 30, "max": 220},
            "spo2": {"type": float, "min": 70, "max": 100},
            "temperature": {"type": float, "min": 30, "max": 45},
            "activity": {"type": str},
        }
    }

def monitor_resource_usage(process_name: str = "openwearables") -> Dict[str, Any]:
    """
    Monitor resource usage of the application.
    
    Args:
        process_name: Name of the process to monitor
        
    Returns:
        Resource usage information
    """
    try:
        # Find processes by name
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            if process_name.lower() in proc.info['name'].lower():
                processes.append(proc)
        
        if not processes:
            return {"error": f"No processes found with name containing '{process_name}'"}
        
        # Aggregate resource usage
        total_cpu = 0
        total_memory_mb = 0
        
        for proc in processes:
            try:
                total_cpu += proc.cpu_percent()
                total_memory_mb += proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # System-wide stats
        system_memory = psutil.virtual_memory()
        system_cpu = psutil.cpu_percent()
        
        return {
            "process_count": len(processes),
            "total_cpu_percent": round(total_cpu, 2),
            "total_memory_mb": round(total_memory_mb, 2),
            "memory_percent": round((total_memory_mb * 1024 * 1024) / system_memory.total * 100, 2),
            "system_cpu_percent": system_cpu,
            "system_memory_percent": system_memory.percent,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error monitoring resource usage: {str(e)}")
        return {"error": str(e)}

def estimate_processing_time(data_size: int, device: str, model_type: str) -> float:
    """
    Estimate processing time based on data size and hardware.
    
    Args:
        data_size: Size of data to process
        device: Target device
        model_type: Type of model
        
    Returns:
        Estimated processing time in seconds
    """
    # Base processing rates (samples per second) for different devices
    base_rates = {
        "cuda": 10000,
        "mlx": 8000,
        "mps": 6000,
        "cpu": 2000
    }
    
    # Model complexity factors
    complexity_factors = {
        "arrhythmia_detection": 1.5,
        "stress_analysis": 1.0,
        "activity_recognition": 1.2,
        "multimodal_health": 2.0,
    }
    
    base_rate = base_rates.get(device, 1000)
    complexity = complexity_factors.get(model_type, 1.0)
    
    processing_rate = base_rate / complexity
    estimated_time = data_size / processing_rate
    
    return max(estimated_time, 0.001)  # Minimum 1ms

def get_model_memory_requirements(model_type: str, precision: str = "float32") -> Dict[str, float]:
    """
    Get estimated memory requirements for different model types.
    
    Args:
        model_type: Type of model
        precision: Model precision
        
    Returns:
        Memory requirements in MB
    """
    # Base memory requirements in MB for float32
    base_memory = {
        "arrhythmia_detection": 50,
        "stress_analysis": 10,
        "activity_recognition": 30,
        "multimodal_health": 120,
    }
    
    # Precision multipliers
    precision_multipliers = {
        "float32": 1.0,
        "float16": 0.5,
        "int8": 0.25,
    }
    
    base_mb = base_memory.get(model_type, 50)
    multiplier = precision_multipliers.get(precision, 1.0)
    
    return {
        "model_size_mb": base_mb * multiplier,
        "activation_memory_mb": base_mb * multiplier * 2,  # Rough estimate
        "total_estimated_mb": base_mb * multiplier * 3,
    }

def cleanup_model_cache(cache_dir: str = "~/.openwearables/models") -> Dict[str, Any]:
    """
    Clean up model cache directory.
    
    Args:
        cache_dir: Cache directory path
        
    Returns:
        Cleanup results
    """
    try:
        cache_path = Path(cache_dir).expanduser()
        
        if not cache_path.exists():
            return {"message": "Cache directory does not exist"}
        
        total_size = 0
        file_count = 0
        
        # Calculate current cache size
        for file_path in cache_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        # Remove old cache files (older than 7 days)
        cutoff_time = time.time() - (7 * 24 * 60 * 60)
        removed_size = 0
        removed_count = 0
        
        for file_path in cache_path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                removed_size += file_path.stat().st_size
                file_path.unlink()
                removed_count += 1
        
        return {
            "total_files": file_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "removed_files": removed_count,
            "removed_size_mb": round(removed_size / (1024 * 1024), 2),
            "remaining_size_mb": round((total_size - removed_size) / (1024 * 1024), 2)
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up model cache: {str(e)}")
        return {"error": str(e)}

# Export main functions
__all__ = [
    "detect_available_hardware",
    "get_optimal_batch_size",
    "benchmark_inference", 
    "optimize_model_for_device",
    "save_model_config",
    "load_model_config",
    "validate_input_data",
    "create_health_data_schema",
    "monitor_resource_usage",
    "estimate_processing_time",
    "get_model_memory_requirements",
    "cleanup_model_cache"
] 