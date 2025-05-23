"""
Data Bridge for Real-time Sensor Data Streaming

Provides high-performance data streaming and synchronization between Python
sensor components and Swift/MLX processing pipeline with minimal latency.
"""

import os
import time
import json
import logging
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import queue
import asyncio
from pathlib import Path

logger = logging.getLogger("OpenWearables.DataBridge")

@dataclass
class SensorDataPacket:
    """High-performance sensor data packet for streaming."""
    sensor_id: int
    sensor_type: str
    timestamp: float
    data: np.ndarray
    metadata: Dict[str, Any]
    sequence_number: int
    quality_score: float = 1.0

@dataclass
class DataStreamConfig:
    """Configuration for data streaming."""
    buffer_size: int = 1000
    max_latency_ms: float = 10.0
    compression_enabled: bool = True
    batch_size: int = 10
    sync_interval_ms: float = 100.0

@dataclass
class StreamMetrics:
    """Performance metrics for data streaming."""
    packets_sent: int = 0
    packets_received: int = 0
    average_latency_ms: float = 0.0
    dropped_packets: int = 0
    buffer_utilization: float = 0.0
    throughput_mbps: float = 0.0


class HighPerformanceBuffer:
    """Lock-free circular buffer for high-performance data streaming."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize high-performance buffer.
        
        Args:
            capacity: Maximum buffer capacity
        """
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.lock = threading.RLock()
        
        # Performance metrics
        self.total_writes = 0
        self.total_reads = 0
        self.overruns = 0
        
    def put(self, item: SensorDataPacket) -> bool:
        """
        Put item in buffer (thread-safe).
        
        Args:
            item: Sensor data packet to store
            
        Returns:
            True if successful, False if buffer full
        """
        with self.lock:
            if self.size >= self.capacity:
                # Buffer full - drop oldest item
                self.tail = (self.tail + 1) % self.capacity
                self.overruns += 1
            else:
                self.size += 1
            
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.capacity
            self.total_writes += 1
            
            return True
    
    def get(self) -> Optional[SensorDataPacket]:
        """
        Get item from buffer (thread-safe).
        
        Returns:
            Sensor data packet or None if empty
        """
        with self.lock:
            if self.size == 0:
                return None
            
            item = self.buffer[self.tail]
            self.buffer[self.tail] = None  # Help GC
            self.tail = (self.tail + 1) % self.capacity
            self.size -= 1
            self.total_reads += 1
            
            return item
    
    def get_batch(self, max_items: int) -> List[SensorDataPacket]:
        """
        Get multiple items from buffer efficiently.
        
        Args:
            max_items: Maximum number of items to retrieve
            
        Returns:
            List of sensor data packets
        """
        items = []
        
        with self.lock:
            for _ in range(min(max_items, self.size)):
                item = self.buffer[self.tail]
                if item is None:
                    break
                
                items.append(item)
                self.buffer[self.tail] = None
                self.tail = (self.tail + 1) % self.capacity
                self.size -= 1
                self.total_reads += 1
        
        return items
    
    def get_utilization(self) -> float:
        """Get buffer utilization percentage."""
        with self.lock:
            return (self.size / self.capacity) * 100.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get buffer performance metrics."""
        with self.lock:
            return {
                "capacity": self.capacity,
                "current_size": self.size,
                "utilization_percent": self.get_utilization(),
                "total_writes": self.total_writes,
                "total_reads": self.total_reads,
                "overruns": self.overruns,
                "write_rate": self.total_writes / max(1, time.time() - getattr(self, '_start_time', time.time())),
                "read_rate": self.total_reads / max(1, time.time() - getattr(self, '_start_time', time.time()))
            }


class DataCompressor:
    """High-performance data compression for sensor streams."""
    
    def __init__(self, compression_level: int = 6):
        """
        Initialize data compressor.
        
        Args:
            compression_level: Compression level (1-9)
        """
        self.compression_level = compression_level
        
        # Try to import compression libraries
        try:
            import zlib
            self.zlib = zlib
            self.compression_available = True
        except ImportError:
            self.compression_available = False
            logger.warning("Compression libraries not available")
    
    def compress_packet(self, packet: SensorDataPacket) -> bytes:
        """
        Compress sensor data packet.
        
        Args:
            packet: Sensor data packet to compress
            
        Returns:
            Compressed data bytes
        """
        if not self.compression_available:
            return self._serialize_packet(packet)
        
        # Serialize packet data
        serialized = self._serialize_packet(packet)
        
        # Compress if beneficial
        if len(serialized) > 100:  # Only compress if reasonably sized
            try:
                compressed = self.zlib.compress(serialized, self.compression_level)
                if len(compressed) < len(serialized) * 0.8:  # 20% savings threshold
                    return b'COMP' + compressed
            except Exception as e:
                logger.warning(f"Compression failed: {e}")
        
        return serialized
    
    def decompress_packet(self, data: bytes) -> SensorDataPacket:
        """
        Decompress sensor data packet.
        
        Args:
            data: Compressed data bytes
            
        Returns:
            Decompressed sensor data packet
        """
        if data.startswith(b'COMP') and self.compression_available:
            try:
                decompressed = self.zlib.decompress(data[4:])
                return self._deserialize_packet(decompressed)
            except Exception as e:
                logger.error(f"Decompression failed: {e}")
                raise
        
        return self._deserialize_packet(data)
    
    def _serialize_packet(self, packet: SensorDataPacket) -> bytes:
        """Serialize packet to bytes."""
        # Create JSON-serializable version
        packet_dict = {
            "sensor_id": packet.sensor_id,
            "sensor_type": packet.sensor_type,
            "timestamp": packet.timestamp,
            "data": packet.data.tolist(),
            "metadata": packet.metadata,
            "sequence_number": packet.sequence_number,
            "quality_score": packet.quality_score
        }
        
        return json.dumps(packet_dict).encode('utf-8')
    
    def _deserialize_packet(self, data: bytes) -> SensorDataPacket:
        """Deserialize packet from bytes."""
        packet_dict = json.loads(data.decode('utf-8'))
        
        return SensorDataPacket(
            sensor_id=packet_dict["sensor_id"],
            sensor_type=packet_dict["sensor_type"],
            timestamp=packet_dict["timestamp"],
            data=np.array(packet_dict["data"]),
            metadata=packet_dict["metadata"],
            sequence_number=packet_dict["sequence_number"],
            quality_score=packet_dict.get("quality_score", 1.0)
        )


class DataSynchronizer:
    """Synchronizes data streams from multiple sensors."""
    
    def __init__(self, sync_window_ms: float = 50.0):
        """
        Initialize data synchronizer.
        
        Args:
            sync_window_ms: Time window for synchronization in milliseconds
        """
        self.sync_window_ms = sync_window_ms
        self.sensor_streams = {}
        self.sync_lock = threading.Lock()
        
        # Synchronization buffers for each sensor type
        self.sync_buffers = {}
        
    def add_sensor_stream(self, sensor_type: str, buffer_size: int = 100):
        """
        Add sensor stream for synchronization.
        
        Args:
            sensor_type: Type of sensor
            buffer_size: Buffer size for this sensor
        """
        with self.sync_lock:
            self.sync_buffers[sensor_type] = deque(maxlen=buffer_size)
            logger.info(f"Added sensor stream: {sensor_type}")
    
    def add_data_point(self, packet: SensorDataPacket):
        """
        Add data point to synchronization.
        
        Args:
            packet: Sensor data packet
        """
        with self.sync_lock:
            sensor_type = packet.sensor_type
            
            if sensor_type not in self.sync_buffers:
                self.add_sensor_stream(sensor_type)
            
            self.sync_buffers[sensor_type].append(packet)
    
    def get_synchronized_data(self, target_timestamp: Optional[float] = None) -> Dict[str, SensorDataPacket]:
        """
        Get synchronized data from all sensors.
        
        Args:
            target_timestamp: Target timestamp for synchronization
            
        Returns:
            Dictionary mapping sensor types to synchronized data packets
        """
        if target_timestamp is None:
            target_timestamp = time.time()
        
        synchronized_data = {}
        
        with self.sync_lock:
            for sensor_type, buffer in self.sync_buffers.items():
                if not buffer:
                    continue
                
                # Find closest data point within sync window
                best_packet = None
                min_time_diff = float('inf')
                
                for packet in buffer:
                    time_diff = abs(packet.timestamp - target_timestamp) * 1000  # Convert to ms
                    
                    if time_diff <= self.sync_window_ms and time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_packet = packet
                
                if best_packet:
                    synchronized_data[sensor_type] = best_packet
        
        return synchronized_data
    
    def get_sync_quality(self) -> Dict[str, Any]:
        """Get synchronization quality metrics."""
        with self.sync_lock:
            metrics = {
                "active_streams": len(self.sync_buffers),
                "sync_window_ms": self.sync_window_ms,
                "buffer_levels": {}
            }
            
            for sensor_type, buffer in self.sync_buffers.items():
                metrics["buffer_levels"][sensor_type] = {
                    "current_size": len(buffer),
                    "max_size": buffer.maxlen,
                    "utilization": len(buffer) / buffer.maxlen if buffer.maxlen else 0
                }
        
        return metrics


class RealTimeDataStream:
    """Real-time data streaming with low latency guarantees."""
    
    def __init__(self, stream_id: str, config: DataStreamConfig):
        """
        Initialize real-time data stream.
        
        Args:
            stream_id: Unique identifier for this stream
            config: Stream configuration
        """
        self.stream_id = stream_id
        self.config = config
        
        # High-performance components
        self.buffer = HighPerformanceBuffer(config.buffer_size)
        self.compressor = DataCompressor()
        self.synchronizer = DataSynchronizer()
        
        # Stream state
        self.is_active = False
        self.sequence_counter = 0
        self.start_time = None
        
        # Performance metrics
        self.metrics = StreamMetrics()
        self.latency_samples = deque(maxlen=1000)
        
        # Threading
        self.stream_thread = None
        self.metrics_thread = None
        
        # Callbacks
        self.data_callbacks = []
        self.error_callbacks = []
        
        logger.info(f"Initialized real-time data stream: {stream_id}")
    
    def start_streaming(self):
        """Start real-time data streaming."""
        if self.is_active:
            return
        
        self.is_active = True
        self.start_time = time.time()
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._streaming_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
        # Start metrics thread
        self.metrics_thread = threading.Thread(target=self._metrics_loop)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        
        logger.info(f"Started streaming for {self.stream_id}")
    
    def stop_streaming(self):
        """Stop data streaming."""
        self.is_active = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        if self.metrics_thread:
            self.metrics_thread.join(timeout=1.0)
        
        logger.info(f"Stopped streaming for {self.stream_id}")
    
    def add_data(self, sensor_id: int, sensor_type: str, data: np.ndarray, 
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add sensor data to stream.
        
        Args:
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor
            data: Sensor data array
            metadata: Optional metadata
            
        Returns:
            True if data added successfully
        """
        packet = SensorDataPacket(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            timestamp=time.time(),
            data=data,
            metadata=metadata or {},
            sequence_number=self.sequence_counter,
            quality_score=self._calculate_quality_score(data)
        )
        
        self.sequence_counter += 1
        
        # Add to buffer
        success = self.buffer.put(packet)
        
        if success:
            self.metrics.packets_sent += 1
            # Add to synchronizer
            self.synchronizer.add_data_point(packet)
        
        return success
    
    def register_data_callback(self, callback: Callable[[List[SensorDataPacket]], None]):
        """Register callback for incoming data."""
        self.data_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable[[Exception], None]):
        """Register callback for errors."""
        self.error_callbacks.append(callback)
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get comprehensive stream metrics."""
        buffer_metrics = self.buffer.get_metrics()
        sync_metrics = self.synchronizer.get_sync_quality()
        
        # Calculate average latency
        if self.latency_samples:
            avg_latency = sum(self.latency_samples) / len(self.latency_samples)
        else:
            avg_latency = 0.0
        
        # Calculate throughput
        elapsed_time = time.time() - self.start_time if self.start_time else 1.0
        throughput = (self.metrics.packets_sent * 8 * 1024) / (elapsed_time * 1024 * 1024)  # Mbps estimate
        
        return {
            "stream_id": self.stream_id,
            "is_active": self.is_active,
            "uptime_seconds": elapsed_time,
            "packets_sent": self.metrics.packets_sent,
            "packets_received": self.metrics.packets_received,
            "dropped_packets": self.metrics.dropped_packets,
            "average_latency_ms": avg_latency,
            "throughput_mbps": throughput,
            "buffer_metrics": buffer_metrics,
            "sync_metrics": sync_metrics,
            "config": asdict(self.config)
        }
    
    def _streaming_loop(self):
        """Main streaming loop."""
        while self.is_active:
            try:
                # Get batch of data
                packets = self.buffer.get_batch(self.config.batch_size)
                
                if packets:
                    # Process callbacks
                    for callback in self.data_callbacks:
                        try:
                            callback(packets)
                        except Exception as e:
                            logger.error(f"Data callback error: {e}")
                            for error_callback in self.error_callbacks:
                                error_callback(e)
                    
                    # Update metrics
                    for packet in packets:
                        latency = (time.time() - packet.timestamp) * 1000
                        self.latency_samples.append(latency)
                
                # Sleep based on sync interval
                time.sleep(self.config.sync_interval_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Streaming loop error: {e}")
                for error_callback in self.error_callbacks:
                    error_callback(e)
    
    def _metrics_loop(self):
        """Metrics collection loop."""
        while self.is_active:
            try:
                # Update buffer utilization
                self.metrics.buffer_utilization = self.buffer.get_utilization()
                
                # Update average latency
                if self.latency_samples:
                    self.metrics.average_latency_ms = sum(self.latency_samples) / len(self.latency_samples)
                
                # Check for performance issues
                if self.metrics.buffer_utilization > 90:
                    logger.warning(f"High buffer utilization: {self.metrics.buffer_utilization:.1f}%")
                
                if self.metrics.average_latency_ms > self.config.max_latency_ms:
                    logger.warning(f"High latency detected: {self.metrics.average_latency_ms:.1f}ms")
                
                time.sleep(1.0)  # Update metrics every second
                
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
    
    def _calculate_quality_score(self, data: np.ndarray) -> float:
        """Calculate data quality score."""
        if len(data) == 0:
            return 0.0
        
        # Simple quality metrics
        finite_ratio = np.sum(np.isfinite(data)) / len(data)
        
        # Check for reasonable data ranges (sensor-specific logic could be added)
        range_score = 1.0
        if np.any(np.abs(data) > 1000):  # Very large values might indicate issues
            range_score = 0.8
        
        return finite_ratio * range_score


class DataBridge:
    """
    Main data bridge for high-performance sensor data streaming.
    
    Coordinates multiple data streams with synchronization, compression,
    and real-time processing capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data bridge.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default stream configuration
        self.default_stream_config = DataStreamConfig(
            buffer_size=self.config.get("buffer_size", 1000),
            max_latency_ms=self.config.get("max_latency_ms", 10.0),
            compression_enabled=self.config.get("compression_enabled", True),
            batch_size=self.config.get("batch_size", 10),
            sync_interval_ms=self.config.get("sync_interval_ms", 100.0)
        )
        
        # Active streams
        self.streams = {}
        self.stream_lock = threading.Lock()
        
        # Global synchronizer for cross-stream sync
        self.global_synchronizer = DataSynchronizer(
            sync_window_ms=self.config.get("global_sync_window_ms", 50.0)
        )
        
        # Performance monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("Data bridge initialized with high-performance streaming")
    
    def create_stream(self, stream_id: str, config: Optional[DataStreamConfig] = None) -> RealTimeDataStream:
        """
        Create new data stream.
        
        Args:
            stream_id: Unique identifier for the stream
            config: Optional stream configuration
            
        Returns:
            Real-time data stream instance
        """
        with self.stream_lock:
            if stream_id in self.streams:
                logger.warning(f"Stream {stream_id} already exists")
                return self.streams[stream_id]
            
            stream_config = config or self.default_stream_config
            stream = RealTimeDataStream(stream_id, stream_config)
            
            # Register global callbacks
            stream.register_data_callback(self._global_data_callback)
            stream.register_error_callback(self._global_error_callback)
            
            self.streams[stream_id] = stream
            logger.info(f"Created data stream: {stream_id}")
            
            return stream
    
    def get_stream(self, stream_id: str) -> Optional[RealTimeDataStream]:
        """Get existing data stream."""
        with self.stream_lock:
            return self.streams.get(stream_id)
    
    def start_all_streams(self):
        """Start all data streams."""
        with self.stream_lock:
            for stream in self.streams.values():
                stream.start_streaming()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info(f"Started {len(self.streams)} data streams")
    
    def stop_all_streams(self):
        """Stop all data streams."""
        with self.stream_lock:
            for stream in self.streams.values():
                stream.stop_streaming()
        
        # Stop monitoring
        self.stop_monitoring()
        
        logger.info("Stopped all data streams")
    
    def add_sensor_data(self, stream_id: str, sensor_id: int, sensor_type: str, 
                       data: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add sensor data to specific stream.
        
        Args:
            stream_id: Target stream identifier
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor
            data: Sensor data array
            metadata: Optional metadata
            
        Returns:
            True if data added successfully
        """
        stream = self.get_stream(stream_id)
        if not stream:
            logger.error(f"Stream not found: {stream_id}")
            return False
        
        success = stream.add_data(sensor_id, sensor_type, data, metadata)
        
        if success:
            # Also add to global synchronizer
            packet = SensorDataPacket(
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                timestamp=time.time(),
                data=data,
                metadata=metadata or {},
                sequence_number=0,  # Will be updated by stream
                quality_score=1.0
            )
            self.global_synchronizer.add_data_point(packet)
        
        return success
    
    def get_synchronized_data(self, sensor_types: List[str], 
                            timestamp: Optional[float] = None) -> Dict[str, SensorDataPacket]:
        """
        Get synchronized data across all streams.
        
        Args:
            sensor_types: List of sensor types to synchronize
            timestamp: Target timestamp for synchronization
            
        Returns:
            Dictionary mapping sensor types to synchronized packets
        """
        return self.global_synchronizer.get_synchronized_data(timestamp)
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all streams."""
        with self.stream_lock:
            stream_metrics = {}
            total_packets = 0
            total_dropped = 0
            avg_latencies = []
            
            for stream_id, stream in self.streams.items():
                metrics = stream.get_stream_metrics()
                stream_metrics[stream_id] = metrics
                
                total_packets += metrics["packets_sent"]
                total_dropped += metrics["dropped_packets"]
                if metrics["average_latency_ms"] > 0:
                    avg_latencies.append(metrics["average_latency_ms"])
            
            global_avg_latency = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0.0
            
            return {
                "active_streams": len(self.streams),
                "total_packets_processed": total_packets,
                "total_dropped_packets": total_dropped,
                "global_average_latency_ms": global_avg_latency,
                "packet_loss_rate": total_dropped / max(1, total_packets),
                "sync_metrics": self.global_synchronizer.get_sync_quality(),
                "stream_metrics": stream_metrics,
                "monitoring_active": self.monitoring_active
            }
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Started data bridge monitoring")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        logger.info("Stopped data bridge monitoring")
    
    def export_metrics(self, filepath: str):
        """Export comprehensive metrics to file."""
        metrics = self.get_global_metrics()
        metrics["export_timestamp"] = time.time()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def _global_data_callback(self, packets: List[SensorDataPacket]):
        """Global callback for all data packets."""
        # Could implement cross-stream analysis here
        pass
    
    def _global_error_callback(self, error: Exception):
        """Global callback for all errors."""
        logger.error(f"Global error callback: {error}")
    
    def _monitoring_loop(self):
        """Performance monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self.get_global_metrics()
                
                # Check for performance issues
                if metrics["global_average_latency_ms"] > 50:
                    logger.warning(f"High global latency: {metrics['global_average_latency_ms']:.1f}ms")
                
                if metrics["packet_loss_rate"] > 0.01:  # 1% loss rate
                    logger.warning(f"High packet loss rate: {metrics['packet_loss_rate']:.2%}")
                
                # Log periodic status
                if hasattr(self, '_last_status_log'):
                    if time.time() - self._last_status_log > 60:  # Every minute
                        logger.info(f"Data bridge status: {metrics['active_streams']} streams, "
                                  f"{metrics['total_packets_processed']} packets processed")
                        self._last_status_log = time.time()
                else:
                    self._last_status_log = time.time()
                
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_all_streams()
        self.stop_monitoring()
        
        with self.stream_lock:
            self.streams.clear()
        
        logger.info("Data bridge cleanup completed") 