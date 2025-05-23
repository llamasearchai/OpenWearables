"""
Swift MLX Bridge Architecture

Provides Python-Swift interoperability for OpenWearables platform with
MLX optimization and native Apple ecosystem integration.
"""

import os
import json
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict

logger = logging.getLogger("OpenWearables.SwiftBridge")

@dataclass
class SwiftSensorData:
    """Data structure for Swift-compatible sensor data."""
    timestamp: float
    sensor_type: str
    data: List[float]
    metadata: Dict[str, Any]

@dataclass 
class SwiftHealthInsight:
    """Data structure for Swift-compatible health insights."""
    timestamp: float
    insight_type: str
    confidence: float
    data: Dict[str, Any]
    recommendations: List[str]

@dataclass
class SwiftDeviceConfig:
    """Configuration structure for Swift device integration."""
    device_type: str
    device_id: str
    enable_healthkit: bool = True
    enable_background_processing: bool = True
    mlx_optimization: bool = True
    privacy_level: str = "high"
    sampling_rate: float = 50.0
    data_retention_hours: int = 24
    sensor_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.sensor_config is None:
            self.sensor_config = self._get_default_sensor_config()
    
    def _get_default_sensor_config(self) -> Dict[str, Any]:
        """Get default sensor configuration based on device type."""
        configs = {
            "smart_glasses": {
                "eye_tracking": {"enabled": True, "frequency": 30},
                "environmental": {"enabled": True, "frequency": 10},
                "biometric": {"enabled": True, "frequency": 50},
                "digital_wellness": {"enabled": True, "frequency": 1}
            },
            "smart_headphones": {
                "audio_health": {"enabled": True, "frequency": 44100},
                "spatial_audio": {"enabled": True, "frequency": 48000},
                "biometric_audio": {"enabled": True, "frequency": 50},
                "environmental_audio": {"enabled": True, "frequency": 10}
            },
            "smart_watch": {
                "ecg": {"enabled": True, "frequency": 250},
                "ppg": {"enabled": True, "frequency": 100},
                "accelerometer": {"enabled": True, "frequency": 50},
                "gyroscope": {"enabled": True, "frequency": 50},
                "temperature": {"enabled": True, "frequency": 1}
            }
        }
        
        return configs.get(self.device_type, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwiftDeviceConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_fields = ["device_type", "device_id"]
        for field in required_fields:
            if not getattr(self, field):
                return False
        
        valid_device_types = ["smart_glasses", "smart_headphones", "smart_watch"]
        if self.device_type not in valid_device_types:
            return False
        
        valid_privacy_levels = ["low", "medium", "high"]
        if self.privacy_level not in valid_privacy_levels:
            return False
        
        return True

class SwiftMLXBridge:
    """
    Main bridge class for Python-Swift interoperability.
    
    Provides:
    - Real-time data streaming between Python and Swift
    - MLX model sharing and inference
    - Native iOS/macOS integration
    - HealthKit connectivity
    - Background processing coordination
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Swift MLX Bridge.
        
        Args:
            config: Configuration dictionary for bridge settings
        """
        self.config = config or self._get_default_config()
        self.swift_process = None
        self.is_running = False
        self.data_queue = []
        self.insight_queue = []
        self.callback_registry = {}
        
        # Communication channels
        self.swift_input_pipe = None
        self.swift_output_pipe = None
        
        # Threading for async communication
        self.communication_thread = None
        self.data_lock = threading.Lock()
        
        # Apple platform detection
        self.platform_info = self._detect_apple_platform()
        
        logger.info(f"Swift MLX Bridge initialized for platform: {self.platform_info['platform']}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Swift bridge."""
        return {
            "swift_app_path": "openwearables/swift_apps/OpenWearablesApp",
            "data_buffer_size": 1000,
            "communication_interval": 0.1,  # 100ms
            "enable_healthkit": True,
            "enable_background_processing": True,
            "mlx_optimization": True,
            "privacy_level": "high"
        }
    
    def _detect_apple_platform(self) -> Dict[str, Any]:
        """Detect Apple platform capabilities."""
        platform_info = {
            "platform": "unknown",
            "has_mlx": False,
            "has_healthkit": False,
            "has_neural_engine": False,
            "device_model": "unknown"
        }
        
        try:
            # Check if running on macOS
            result = subprocess.run(["sw_vers"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                platform_info["platform"] = "macOS"
                
                # Check for Apple Silicon
                arch_result = subprocess.run(["arch"], capture_output=True, text=True, timeout=5)
                if "arm64" in arch_result.stdout:
                    platform_info["has_mlx"] = True
                    platform_info["has_neural_engine"] = True
                
                # Try to get device model
                try:
                    model_result = subprocess.run(
                        ["system_profiler", "SPHardwareDataType"], 
                        capture_output=True, text=True, timeout=10
                    )
                    if "Model Name" in model_result.stdout:
                        for line in model_result.stdout.split('\n'):
                            if "Model Name" in line:
                                platform_info["device_model"] = line.split(":")[1].strip()
                                break
                except Exception:
                    pass
                
                # Check for HealthKit (requires Xcode tools)
                try:
                    healthkit_check = subprocess.run(
                        ["xcrun", "--find", "swift"], 
                        capture_output=True, timeout=5
                    )
                    if healthkit_check.returncode == 0:
                        platform_info["has_healthkit"] = True
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning(f"Platform detection failed: {str(e)}")
        
        return platform_info
    
    def start(self) -> bool:
        """Start the Swift MLX bridge."""
        if self.is_running:
            logger.warning("Swift bridge is already running")
            return True
        
        try:
            # Verify Swift app exists
            swift_app_path = Path(self.config["swift_app_path"])
            if not swift_app_path.exists():
                logger.error(f"Swift app not found at {swift_app_path}")
                return False
            
            # Start Swift application process
            self._start_swift_process()
            
            # Start communication thread
            self.communication_thread = threading.Thread(
                target=self._communication_loop,
                daemon=True
            )
            self.communication_thread.start()
            
            self.is_running = True
            logger.info("Swift MLX bridge started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Swift bridge: {str(e)}")
            return False
    
    def stop(self) -> None:
        """Stop the Swift MLX bridge."""
        self.is_running = False
        
        # Stop Swift process
        if self.swift_process:
            try:
                self.swift_process.terminate()
                self.swift_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.swift_process.kill()
            except Exception as e:
                logger.error(f"Error stopping Swift process: {str(e)}")
        
        # Wait for communication thread
        if self.communication_thread and self.communication_thread.is_alive():
            self.communication_thread.join(timeout=2)
        
        logger.info("Swift MLX bridge stopped")
    
    def _start_swift_process(self) -> None:
        """Start the Swift application process."""
        swift_app_path = self.config["swift_app_path"]
        
        # Swift app command with bridge mode
        cmd = [
            "swift", "run", 
            "--package-path", swift_app_path,
            "OpenWearablesApp", 
            "--bridge-mode",
            "--config", json.dumps(self.config)
        ]
        
        try:
            self.swift_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0  # Unbuffered
            )
            
            self.swift_input_pipe = self.swift_process.stdin
            self.swift_output_pipe = self.swift_process.stdout
            
            logger.info("Swift process started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Swift process: {str(e)}")
            raise
    
    def _communication_loop(self) -> None:
        """Main communication loop between Python and Swift."""
        while self.is_running:
            try:
                # Send queued data to Swift
                self._send_queued_data()
                
                # Receive data from Swift
                self._receive_swift_data()
                
                # Sleep for communication interval
                threading.Event().wait(self.config["communication_interval"])
                
            except Exception as e:
                logger.error(f"Communication loop error: {str(e)}")
                if not self.is_running:
                    break
    
    def _send_queued_data(self) -> None:
        """Send queued sensor data to Swift application."""
        with self.data_lock:
            if not self.data_queue or not self.swift_input_pipe:
                return
            
            # Prepare data packet
            data_packet = {
                "type": "sensor_data",
                "timestamp": time.time(),
                "data": [asdict(item) for item in self.data_queue[:10]]  # Send up to 10 items
            }
            
            try:
                # Send JSON data to Swift
                json_data = json.dumps(data_packet) + "\n"
                self.swift_input_pipe.write(json_data)
                self.swift_input_pipe.flush()
                
                # Remove sent data
                self.data_queue = self.data_queue[10:]
                
            except Exception as e:
                logger.error(f"Failed to send data to Swift: {str(e)}")
    
    def _receive_swift_data(self) -> None:
        """Receive data and insights from Swift application."""
        if not self.swift_output_pipe:
            return
        
        try:
            # Check if data is available (non-blocking)
            import select
            ready, _, _ = select.select([self.swift_output_pipe], [], [], 0)
            
            if ready:
                line = self.swift_output_pipe.readline()
                if line.strip():
                    data = json.loads(line.strip())
                    self._process_swift_message(data)
                    
        except Exception as e:
            logger.error(f"Failed to receive Swift data: {str(e)}")
    
    def _process_swift_message(self, message: Dict[str, Any]) -> None:
        """Process message received from Swift application."""
        message_type = message.get("type")
        
        if message_type == "health_insight":
            insight = SwiftHealthInsight(
                timestamp=message["timestamp"],
                insight_type=message["insight_type"],
                confidence=message["confidence"],
                data=message["data"],
                recommendations=message.get("recommendations", [])
            )
            
            with self.data_lock:
                self.insight_queue.append(insight)
            
            # Trigger callbacks
            self._trigger_callbacks("health_insight", insight)
            
        elif message_type == "healthkit_data":
            # Process HealthKit data
            self._process_healthkit_data(message["data"])
            
        elif message_type == "status_update":
            logger.info(f"Swift app status: {message['status']}")
            
        elif message_type == "error":
            logger.error(f"Swift app error: {message['error']}")
    
    def _process_healthkit_data(self, healthkit_data: Dict[str, Any]) -> None:
        """Process HealthKit data received from Swift."""
        # Convert HealthKit data to OpenWearables format
        for data_type, values in healthkit_data.items():
            sensor_data = SwiftSensorData(
                timestamp=values.get("timestamp", time.time()),
                sensor_type=f"healthkit_{data_type}",
                data=values.get("data", []),
                metadata={"source": "HealthKit", "data_type": data_type}
            )
            
            with self.data_lock:
                self.data_queue.append(sensor_data)
    
    def send_sensor_data(self, sensor_type: str, data: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Send sensor data to Swift application.
        
        Args:
            sensor_type: Type of sensor data
            data: Sensor data array
            metadata: Additional metadata
        """
        sensor_data = SwiftSensorData(
            timestamp=time.time(),
            sensor_type=sensor_type,
            data=data,
            metadata=metadata or {}
        )
        
        with self.data_lock:
            self.data_queue.append(sensor_data)
            
            # Limit queue size
            if len(self.data_queue) > self.config["data_buffer_size"]:
                self.data_queue = self.data_queue[-self.config["data_buffer_size"]:]
    
    def get_health_insights(self, clear: bool = True) -> List[SwiftHealthInsight]:
        """
        Get health insights generated by Swift application.
        
        Args:
            clear: Whether to clear the insights queue
            
        Returns:
            List of health insights
        """
        with self.data_lock:
            insights = list(self.insight_queue)
            if clear:
                self.insight_queue.clear()
        
        return insights
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register callback for specific event types.
        
        Args:
            event_type: Type of event to listen for
            callback: Callback function to execute
        """
        if event_type not in self.callback_registry:
            self.callback_registry[event_type] = []
        
        self.callback_registry[event_type].append(callback)
        logger.info(f"Registered callback for {event_type}")
    
    def _trigger_callbacks(self, event_type: str, data: Any) -> None:
        """Trigger callbacks for specific event type."""
        callbacks = self.callback_registry.get(event_type, [])
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {str(e)}")
    
    def request_healthkit_data(self, data_types: List[str]) -> bool:
        """
        Request specific HealthKit data types from Swift app.
        
        Args:
            data_types: List of HealthKit data types to request
            
        Returns:
            True if request was sent successfully
        """
        if not self.platform_info["has_healthkit"]:
            logger.warning("HealthKit not available on this platform")
            return False
        
        request = {
            "type": "healthkit_request",
            "data_types": data_types,
            "timestamp": time.time()
        }
        
        try:
            if self.swift_input_pipe:
                json_data = json.dumps(request) + "\n"
                self.swift_input_pipe.write(json_data)
                self.swift_input_pipe.flush()
                return True
        except Exception as e:
            logger.error(f"Failed to request HealthKit data: {str(e)}")
        
        return False
    
    def enable_background_processing(self, enabled: bool = True) -> bool:
        """
        Enable/disable background processing in Swift app.
        
        Args:
            enabled: Whether to enable background processing
            
        Returns:
            True if command was sent successfully
        """
        command = {
            "type": "background_processing",
            "enabled": enabled,
            "timestamp": time.time()
        }
        
        try:
            if self.swift_input_pipe:
                json_data = json.dumps(command) + "\n"
                self.swift_input_pipe.write(json_data)
                self.swift_input_pipe.flush()
                return True
        except Exception as e:
            logger.error(f"Failed to set background processing: {str(e)}")
        
        return False
    
    def get_platform_capabilities(self) -> Dict[str, Any]:
        """Get information about Apple platform capabilities."""
        return self.platform_info.copy()
    
    def is_mlx_available(self) -> bool:
        """Check if MLX is available for Apple Silicon optimization."""
        return self.platform_info["has_mlx"]
    
    def is_healthkit_available(self) -> bool:
        """Check if HealthKit is available."""
        return self.platform_info["has_healthkit"]
    
    def get_swift_app_status(self) -> Dict[str, Any]:
        """Get status of the Swift application."""
        if not self.swift_process:
            return {"status": "not_running", "process_id": None}
        
        poll_result = self.swift_process.poll()
        if poll_result is None:
            return {
                "status": "running",
                "process_id": self.swift_process.pid,
                "running_time": time.time() - getattr(self, '_start_time', time.time())
            }
        else:
            return {
                "status": "terminated",
                "exit_code": poll_result,
                "process_id": self.swift_process.pid
            }
    
    def create_swift_package(self, app_name: str, output_dir: str) -> bool:
        """
        Create a new Swift package for OpenWearables integration.
        
        Args:
            app_name: Name of the Swift app
            output_dir: Directory to create the package
            
        Returns:
            True if package was created successfully
        """
        try:
            package_dir = Path(output_dir) / app_name
            package_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Package.swift
            package_swift = self._generate_package_swift(app_name)
            (package_dir / "Package.swift").write_text(package_swift)
            
            # Create Sources directory
            sources_dir = package_dir / "Sources" / app_name
            sources_dir.mkdir(parents=True, exist_ok=True)
            
            # Create main Swift file
            main_swift = self._generate_main_swift(app_name)
            (sources_dir / "main.swift").write_text(main_swift)
            
            # Create OpenWearablesBridge.swift
            bridge_swift = self._generate_bridge_swift()
            (sources_dir / "OpenWearablesBridge.swift").write_text(bridge_swift)
            
            logger.info(f"Swift package created at {package_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Swift package: {str(e)}")
            return False
    
    def _generate_package_swift(self, app_name: str) -> str:
        """Generate Package.swift content."""
        return f'''// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "{app_name}",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .executable(name: "{app_name}", targets: ["{app_name}"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.1.0")
    ],
    targets: [
        .executableTarget(
            name: "{app_name}",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ]
        )
    ]
)
'''
    
    def _generate_main_swift(self, app_name: str) -> str:
        """Generate main.swift content."""
        return f'''import Foundation
import HealthKit
import MLX

@main
struct {app_name} {{
    static func main() async {{
        let bridge = OpenWearablesBridge()
        
        if CommandLine.arguments.contains("--bridge-mode") {{
            await bridge.startBridgeMode()
        }} else {{
            await bridge.startStandaloneMode()
        }}
    }}
}}
'''
    
    def _generate_bridge_swift(self) -> str:
        """Generate OpenWearablesBridge.swift content."""
        return '''import Foundation
import HealthKit
import MLX
import OSLog

actor OpenWearablesBridge {
    private let logger = Logger(subsystem: "com.openwearables.app", category: "Bridge")
    private var healthStore: HKHealthStore?
    private var isRunning = false
    
    init() {
        if HKHealthStore.isHealthDataAvailable() {
            healthStore = HKHealthStore()
        }
    }
    
    func startBridgeMode() async {
        logger.info("Starting OpenWearables bridge mode")
        isRunning = true
        
        // Request HealthKit permissions
        await requestHealthKitPermissions()
        
        // Start main communication loop
        await communicationLoop()
    }
    
    func startStandaloneMode() async {
        logger.info("Starting OpenWearables standalone mode")
        // Implement standalone app functionality
    }
    
    private func communicationLoop() async {
        while isRunning {
            // Read from stdin for Python commands
            if let input = readLine() {
                await processCommand(input)
            }
            
            // Send any pending data to stdout
            await sendPendingData()
            
            // Sleep briefly
            try {
                await Task.sleep(nanoseconds: 100_000_000) // 100ms
            } catch {
                break
            }
        }
    }
    
    private func processCommand(_ input: String) async {
        guard let data = input.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else {
            return
        }
        
        switch type {
        case "healthkit_request":
            if let dataTypes = json["data_types"] as? [String] {
                await fetchHealthKitData(dataTypes)
            }
        case "background_processing":
            if let enabled = json["enabled"] as? Bool {
                configureBackgroundProcessing(enabled)
            }
        case "sensor_data":
            if let sensorData = json["data"] as? [[String: Any]] {
                await processSensorData(sensorData)
            }
        default:
            logger.warning("Unknown command type: \\(type)")
        }
    }
    
    private func requestHealthKitPermissions() async {
        guard let healthStore = healthStore else { return }
        
        let readTypes: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .stepCount)!,
            HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!
        ]
        
        do {
            try await healthStore.requestAuthorization(toShare: [], read: readTypes)
            logger.info("HealthKit permissions granted")
        } catch {
            logger.error("HealthKit permission error: \\(error)")
        }
    }
    
    private func fetchHealthKitData(_ dataTypes: [String]) async {
        guard let healthStore = healthStore else { return }
        
        for dataType in dataTypes {
            switch dataType {
            case "heart_rate":
                await fetchHeartRateData()
            case "steps":
                await fetchStepData()
            default:
                logger.warning("Unknown HealthKit data type: \\(dataType)")
            }
        }
    }
    
    private func fetchHeartRateData() async {
        guard let healthStore = healthStore,
              let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            return
        }
        
        let query = HKSampleQuery(
            sampleType: heartRateType,
            predicate: nil,
            limit: 10,
            sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
        ) { _, samples, error in
            if let error = error {
                self.logger.error("Heart rate query error: \\(error)")
                return
            }
            
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self.sendHealthKitData("heart_rate", samples: samples)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func fetchStepData() async {
        // Similar implementation for step data
    }
    
    private func sendHealthKitData(_ dataType: String, samples: [HKQuantitySample]) async {
        let data: [String: Any] = [
            "type": "healthkit_data",
            "data": [
                dataType: [
                    "timestamp": Date().timeIntervalSince1970,
                    "data": samples.map { $0.quantity.doubleValue(for: HKUnit.count()) }
                ]
            ]
        ]
        
        if let jsonData = try? JSONSerialization.data(withJSONObject: data),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            print(jsonString)
            fflush(stdout)
        }
    }
    
    private func processSensorData(_ sensorData: [[String: Any]]) async {
        // Process sensor data with MLX models
        for data in sensorData {
            if let sensorType = data["sensor_type"] as? String,
               let values = data["data"] as? [Double] {
                await processWithMLXModel(sensorType: sensorType, data: values)
            }
        }
    }
    
    private func processWithMLXModel(sensorType: String, data: [Double]) async {
        // Convert data to MLX array
        let mlxArray = MLXArray(data.map { Float($0) })
        
        // Process with appropriate MLX model based on sensor type
        switch sensorType {
        case "smart_glasses":
            await processGlassesData(mlxArray)
        case "smart_headphones":
            await processHeadphonesData(mlxArray)
        default:
            logger.info("Processing \\(sensorType) data with MLX")
        }
    }
    
    private func processGlassesData(_ data: MLXArray) async {
        // Implement glasses-specific MLX processing
        let insight: [String: Any] = [
            "type": "health_insight",
            "timestamp": Date().timeIntervalSince1970,
            "insight_type": "eye_strain",
            "confidence": 0.85,
            "data": ["strain_level": 0.3],
            "recommendations": ["Take a break from screen time"]
        ]
        
        await sendInsight(insight)
    }
    
    private func processHeadphonesData(_ data: MLXArray) async {
        // Implement headphones-specific MLX processing
    }
    
    private func sendInsight(_ insight: [String: Any]) async {
        if let jsonData = try? JSONSerialization.data(withJSONObject: insight),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            print(jsonString)
            fflush(stdout)
        }
    }
    
    private func sendPendingData() async {
        // Send any periodic updates or pending data
    }
    
    private func configureBackgroundProcessing(_ enabled: Bool) {
        // Configure background processing
        logger.info("Background processing \\(enabled ? "enabled" : "disabled")")
    }
}
'''

# Import time module that was missing
import time

# Export the main class
__all__ = ["SwiftMLXBridge"] 