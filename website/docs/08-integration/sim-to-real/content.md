---
title: System Integration for Humanoid Robotics
sidebar_position: 1
---

# System Integration for Humanoid Robotics

## Introduction

System integration is the critical phase where all individual components of a humanoid robot—locomotion, manipulation, perception, cognition, and communication—are combined into a cohesive, functional system. This chapter explores the challenges, architectures, and best practices for integrating the diverse subsystems that make up a humanoid robot, focusing on creating unified, stable, and efficient systems that can operate reliably in human environments.

## Integration Architecture

### Multi-Layer Integration Model

Effective humanoid robotics systems require integration at multiple levels:

1. **Component Level**: Individual sensors, actuators, and processing units
2. **Subsystem Level**: Locomotion, manipulation, perception modules
3. **Functional Level**: Navigation, object interaction, communication
4. **System Level**: Coherent behavior across all capabilities

```python
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

@dataclass
class SystemState:
    """Represents the overall state of the humanoid system"""
    timestamp: float
    locomotion_status: str
    manipulation_status: str
    perception_status: str
    cognitive_status: str
    communication_status: str
    battery_level: float
    system_health: Dict[str, float]  # Component health scores

class IntegrationLayer(Enum):
    """Levels of system integration"""
    COMPONENT = "component"
    SUBSYSTEM = "subsystem"
    FUNCTIONAL = "functional"
    SYSTEM = "system"

class SubsystemInterface:
    """Interface for different subsystems"""
    def __init__(self, name: str):
        self.name = name
        self.status = "idle"
        self.last_update = time.time()
        self.health_score = 1.0
        self.message_queue = []
        self.event_handlers = {}
    
    async def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a message and return a response if needed"""
        # Subsystems override this method
        raise NotImplementedError
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for a specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def trigger_event(self, event_type: str, data: Any) -> None:
        """Trigger an event and notify all registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                handler(data)

class LocomotionSubsystem(SubsystemInterface):
    """Handles all locomotion-related functions"""
    def __init__(self):
        super().__init__("locomotion")
        self.current_gait = "stand"
        self.target_velocity = [0.0, 0.0, 0.0]
        self.balance_controller = None  # Would be initialized with actual balance controller
        self.footstep_planner = None    # Would be initialized with actual planner
    
    async def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle locomotion-related messages"""
        msg_type = message.get("type")
        
        if msg_type == "move_to":
            target_pos = message.get("target_position")
            return await self._execute_navigation(target_pos)
        
        elif msg_type == "set_gait":
            gait_type = message.get("gait")
            self.current_gait = gait_type
            return {"status": "gait_changed", "new_gait": gait_type}
        
        elif msg_type == "stop":
            self.target_velocity = [0.0, 0.0, 0.0]
            return {"status": "stopped"}
        
        return {"status": "unknown_command"}
    
    async def _execute_navigation(self, target_pos: List[float]) -> Dict[str, Any]:
        """Execute navigation to target position"""
        # Simulate navigation execution
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # In a real implementation, this would interface with the actual locomotion controller
        return {
            "status": "navigation_started",
            "target": target_pos,
            "estimated_time": 5.0  # seconds
        }

class ManipulationSubsystem(SubsystemInterface):
    """Handles all manipulation-related functions"""
    def __init__(self):
        super().__init__("manipulation")
        self.left_arm_status = "ready"
        self.right_arm_status = "ready"
        self.end_effector_poses = {"left": [0, 0, 0, 0, 0, 0], "right": [0, 0, 0, 0, 0, 0]}
        self.gripper_states = {"left": 0.0, "right": 0.0}  # 0.0=open, 1.0=closed
    
    async def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle manipulation-related messages"""
        msg_type = message.get("type")
        
        if msg_type == "grasp_object":
            obj_info = message.get("object")
            arm = message.get("arm", "right")
            return await self._execute_grasp(obj_info, arm)
        
        elif msg_type == "move_arm":
            target_pose = message.get("target_pose")
            arm = message.get("arm", "right")
            return await self._execute_arm_movement(target_pose, arm)
        
        elif msg_type == "release_object":
            arm = message.get("arm", "right")
            return await self._execute_release(arm)
        
        return {"status": "unknown_command"}
    
    async def _execute_grasp(self, obj_info: Dict[str, Any], arm: str) -> Dict[str, Any]:
        """Execute grasping action"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Update internal state
        self.gripper_states[arm] = 1.0  # Closed
        
        return {
            "status": "grasp_executed",
            "arm": arm,
            "object": obj_info,
            "success": True
        }
    
    async def _execute_arm_movement(self, target_pose: List[float], arm: str) -> Dict[str, Any]:
        """Execute arm movement to target pose"""
        await asyncio.sleep(0.05)  # Simulate movement time
        
        # Update internal state
        self.end_effector_poses[arm] = target_pose
        
        return {
            "status": "movement_completed",
            "arm": arm,
            "target_pose": target_pose
        }
    
    async def _execute_release(self, arm: str) -> Dict[str, Any]:
        """Execute release action"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Update internal state
        self.gripper_states[arm] = 0.0  # Open
        
        return {
            "status": "release_executed",
            "arm": arm,
            "success": True
        }

class PerceptionSubsystem(SubsystemInterface):
    """Handles all perception-related functions"""
    def __init__(self):
        super().__init__("perception")
        self.camera_data = None
        self.lidar_data = None
        self.object_detections = []
        self.human_detections = []
        self.environment_map = None
    
    async def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle perception-related messages"""
        msg_type = message.get("type")
        
        if msg_type == "request_objects":
            return {"status": "objects_reported", "objects": self.object_detections}
        
        elif msg_type == "request_humans":
            return {"status": "humans_reported", "humans": self.human_detections}
        
        elif msg_type == "update_environment":
            new_map = message.get("environment_map")
            self.environment_map = new_map
            return {"status": "map_updated"}
        
        elif msg_type == "process_sensor_data":
            sensor_type = message.get("sensor_type")
            data = message.get("data")
            return await self._process_sensor_data(sensor_type, data)
        
        return {"status": "unknown_command"}
    
    async def _process_sensor_data(self, sensor_type: str, data: Any) -> Dict[str, Any]:
        """Process incoming sensor data"""
        await asyncio.sleep(0.02)  # Simulate processing time
        
        if sensor_type == "camera":
            self.camera_data = data
            # Simulate object detection
            self.object_detections = [{"name": "unknown_object", "position": [1, 2, 3]}]
        elif sensor_type == "lidar":
            self.lidar_data = data
        
        return {"status": "data_processed", "sensor_type": sensor_type}

class CognitiveSubsystem(SubsystemInterface):
    """Handles all cognitive functions"""
    def __init__(self):
        super().__init__("cognitive")
        self.conversation_state = {}
        self.task_plans = []
        self.memory_system = {}  # Simplified memory representation
        self.current_goals = []
    
    async def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle cognitive-related messages"""
        msg_type = message.get("type")
        
        if msg_type == "process_conversation":
            user_input = message.get("user_input")
            return await self._process_conversation(user_input)
        
        elif msg_type == "update_goals":
            new_goals = message.get("goals")
            self.current_goals.extend(new_goals)
            return {"status": "goals_updated", "total_goals": len(self.current_goals)}
        
        elif msg_type == "plan_task":
            task_spec = message.get("task")
            return await self._plan_task(task_spec)
        
        return {"status": "unknown_command"}
    
    async def _process_conversation(self, user_input: str) -> Dict[str, Any]:
        """Process conversational input"""
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Simplified natural language processing
        if "hello" in user_input.lower():
            response = "Hello! How can I assist you today?"
        elif "help" in user_input.lower():
            response = "I can help with navigation, object manipulation, and information."
        else:
            response = "I'm here to help. What would you like to do?"
        
        return {
            "status": "response_generated",
            "response": response
        }
    
    async def _plan_task(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a complex task"""
        await asyncio.sleep(0.05)  # Simulate planning time
        
        # Simplified task planning
        plan = [
            {"action": "navigate_to", "parameters": {"target": task_spec.get("target_location")}},
            {"action": "manipulate_object", "parameters": {"object": task_spec.get("object")}},
            {"action": "return_to_base", "parameters": {}}
        ]
        
        self.task_plans.append(plan)
        
        return {
            "status": "plan_generated",
            "plan_id": len(self.task_plans),
            "steps": len(plan)
        }

class CommunicationSubsystem(SubsystemInterface):
    """Handles all communication functions"""
    def __init__(self):
        super().__init__("communication")
        self.active_connections = []
        self.message_history = []
        self.language_settings = {"input": "en", "output": "en"}
    
    async def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle communication-related messages"""
        msg_type = message.get("type")
        
        if msg_type == "send_message":
            content = message.get("content")
            destination = message.get("destination", "local")
            return await self._send_message(content, destination)
        
        elif msg_type == "receive_message":
            source = message.get("source")
            content = message.get("content")
            return await self._receive_message(source, content)
        
        elif msg_type == "set_language":
            input_lang = message.get("input_language")
            output_lang = message.get("output_language")
            self.language_settings["input"] = input_lang
            self.language_settings["output"] = output_lang
            return {"status": "language_set"}
        
        return {"status": "unknown_command"}
    
    async def _send_message(self, content: str, destination: str) -> Dict[str, Any]:
        """Send a message to specified destination"""
        await asyncio.sleep(0.01)  # Simulate transmission time
        
        message_record = {
            "timestamp": time.time(),
            "content": content,
            "destination": destination,
            "status": "sent"
        }
        self.message_history.append(message_record)
        
        return {"status": "message_sent", "id": len(self.message_history)}
    
    async def _receive_message(self, source: str, content: str) -> Dict[str, Any]:
        """Receive a message from specified source"""
        await asyncio.sleep(0.01)  # Simulate processing time
        
        message_record = {
            "timestamp": time.time(),
            "source": source,
            "content": content,
            "status": "received"
        }
        self.message_history.append(message_record)
        
        return {"status": "message_received", "id": len(self.message_history)}

class SystemIntegrator:
    """Main integration layer that coordinates all subsystems"""
    def __init__(self):
        self.locomotion = LocomotionSubsystem()
        self.manipulation = ManipulationSubsystem()
        self.perception = PerceptionSubsystem()
        self.cognitive = CognitiveSubsystem()
        self.communication = CommunicationSubsystem()
        
        # All subsystems for easy iteration
        self.subsystems = {
            "locomotion": self.locomotion,
            "manipulation": self.manipulation,
            "perception": self.perception,
            "cognitive": self.cognitive,
            "communication": self.communication
        }
        
        # Message router
        self.message_router = {
            "locomotion": self.locomotion.process_message,
            "manipulation": self.manipulation.process_message,
            "perception": self.perception.process_message,
            "cognitive": self.cognitive.process_message,
            "communication": self.communication.process_message
        }
        
        # System-wide event handlers
        self.global_event_handlers = {}
    
    async def route_message(self, target_subsystem: str, message: Dict[str, Any]) -> Any:
        """Route a message to the appropriate subsystem"""
        if target_subsystem in self.message_router:
            try:
                result = await self.message_router[target_subsystem](message)
                return result
            except Exception as e:
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Unknown subsystem: {target_subsystem}"}
    
    async def broadcast_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast a message to all subsystems"""
        results = {}
        for name, subsystem in self.subsystems.items():
            try:
                result = await subsystem.process_message(message)
                results[name] = result
            except Exception as e:
                results[name] = {"status": "error", "message": str(e)}
        
        return results
    
    def get_system_state(self) -> SystemState:
        """Get the current state of all subsystems"""
        return SystemState(
            timestamp=time.time(),
            locomotion_status=self.locomotion.status,
            manipulation_status=self.manipulation.status,
            perception_status=self.perception.status,
            cognitive_status=self.cognitive.status,
            communication_status=self.communication.status,
            battery_level=0.85,  # Simulated battery level
            system_health={
                name: subsystem.health_score 
                for name, subsystem in self.subsystems.items()
            }
        )
    
    def register_global_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for system-wide events"""
        if event_type not in self.global_event_handlers:
            self.global_event_handlers[event_type] = []
        self.global_event_handlers[event_type].append(handler)
    
    async def trigger_global_event(self, event_type: str, data: Any) -> None:
        """Trigger a system-wide event"""
        if event_type in self.global_event_handlers:
            for handler in self.global_event_handlers[event_type]:
                # In a real implementation, this would handle async handlers properly
                handler(data)

# Example usage
async def demo_integration():
    integrator = SystemIntegrator()
    
    print("System integration demo:")
    print(f"Initial system state: {integrator.get_system_state()}")
    
    # Example: Request object detection
    perception_result = await integrator.route_message("perception", {
        "type": "request_objects"
    })
    print(f"Perception result: {perception_result}")
    
    # Example: Plan a task
    planning_result = await integrator.route_message("cognitive", {
        "type": "plan_task",
        "task": {
            "target_location": [1.0, 2.0, 0.0],
            "object": "red cup"
        }
    })
    print(f"Planning result: {planning_result}")
    
    # Example: Execute navigation
    nav_result = await integrator.route_message("locomotion", {
        "type": "move_to",
        "target_position": [1.0, 2.0, 0.0]
    })
    print(f"Navigation result: {nav_result}")
    
    # Broadcast a system event
    broadcast_results = await integrator.broadcast_message({
        "type": "system_update",
        "data": {"timestamp": time.time()}
    })
    print(f"Broadcast results: {broadcast_results}")

# Run the demo
if __name__ == "__main__":
    asyncio.run(demo_integration())
```

### Service-Oriented Integration

For large-scale humanoid systems, a service-oriented architecture provides flexibility and maintainability:

```python
import json
import uuid
from typing import Dict, List, Any, Optional
import zmq
import threading

class ServiceRegistry:
    """Registry for services in the humanoid system"""
    def __init__(self):
        self.services = {}  # name -> service info
        self.service_lock = threading.Lock()
    
    def register_service(self, name: str, address: str, capabilities: List[str]) -> bool:
        """Register a service with its capabilities"""
        with self.service_lock:
            if name in self.services:
                return False  # Service already registered
            
            self.services[name] = {
                "address": address,
                "capabilities": capabilities,
                "timestamp": time.time(),
                "status": "active"
            }
            return True
    
    def unregister_service(self, name: str) -> bool:
        """Unregister a service"""
        with self.service_lock:
            if name in self.services:
                del self.services[name]
                return True
            return False
    
    def get_service(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a service"""
        return self.services.get(name)
    
    def get_services_by_capability(self, capability: str) -> List[str]:
        """Get all services that support a specific capability"""
        matching_services = []
        for name, info in self.services.items():
            if capability in info["capabilities"]:
                matching_services.append(name)
        return matching_services
    
    def heartbeat(self, name: str) -> bool:
        """Update service heartbeat"""
        with self.service_lock:
            if name in self.services:
                self.services[name]["timestamp"] = time.time()
                return True
            return False

class ServiceClient:
    """Client for communicating with services"""
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.timeout = 5000  # 5 seconds in milliseconds
    
    def call_service(self, service_name: str, method: str, params: Dict[str, Any]) -> Optional[Any]:
        """Call a method on a service"""
        service_info = self.registry.get_service(service_name)
        if not service_info:
            return {"error": f"Service {service_name} not found"}
        
        try:
            # Connect to service
            self.socket.connect(service_info["address"])
            
            # Prepare request
            request = {
                "id": str(uuid.uuid4()),
                "method": method,
                "params": params,
                "timestamp": time.time()
            }
            
            # Send request
            self.socket.send_string(json.dumps(request))
            
            # Receive response
            response_str = self.socket.recv_string(flags=zmq.NOBLOCK, timeout=self.timeout)
            response = json.loads(response_str)
            
            return response
        except zmq.Again:
            return {"error": "Service call timed out"}
        except Exception as e:
            return {"error": f"Service call failed: {str(e)}"}
        finally:
            self.socket.disconnect(service_info["address"])

class BaseService:
    """Base class for humanoid services"""
    def __init__(self, name: str, address: str, capabilities: List[str]):
        self.name = name
        self.address = address
        self.capabilities = capabilities
        self.running = False
        self.context = zmq.Context()
        self.socket = None
    
    def start(self, registry: ServiceRegistry):
        """Start the service and register with registry"""
        if not registry.register_service(self.name, self.address, self.capabilities):
            raise Exception(f"Could not register service {self.name}")
        
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.address)
        self.running = True
        
        print(f"Service {self.name} started at {self.address}")
        
        # Service loop
        while self.running:
            try:
                # Receive request
                message = self.socket.recv_string()
                request = json.loads(message)
                
                # Process request
                response = self.handle_request(request["method"], request["params"])
                
                # Send response
                self.socket.send_string(json.dumps(response))
                
                # Update heartbeat in registry
                registry.heartbeat(self.name)
                
            except Exception as e:
                error_response = {"error": str(e)}
                self.socket.send_string(json.dumps(error_response))
    
    def stop(self):
        """Stop the service"""
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()
    
    def handle_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming request (to be implemented by subclasses)"""
        raise NotImplementedError

class LocomotionService(BaseService):
    """Locomotion service implementation"""
    def __init__(self):
        super().__init__(
            name="locomotion",
            address="tcp://*:5555",
            capabilities=["move_to", "set_gait", "stop", "get_position"]
        )
        self.current_position = [0.0, 0.0, 0.0]
        self.current_gait = "stand"
    
    def handle_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle locomotion-specific requests"""
        if method == "move_to":
            target = params.get("target_position", [0.0, 0.0, 0.0])
            # Simulate movement
            self.current_position = target
            return {
                "status": "success",
                "message": f"Moving to {target}",
                "estimated_time": 3.0
            }
        
        elif method == "set_gait":
            gait = params.get("gait", "stand")
            self.current_gait = gait
            return {
                "status": "success", 
                "message": f"Gait set to {gait}"
            }
        
        elif method == "get_position":
            return {
                "status": "success",
                "position": self.current_position,
                "gait": self.current_gait
            }
        
        else:
            return {"status": "error", "message": f"Unknown method: {method}"}

class ManipulationService(BaseService):
    """Manipulation service implementation"""
    def __init__(self):
        super().__init__(
            name="manipulation",
            address="tcp://*:5556", 
            capabilities=["grasp_object", "move_arm", "release_object", "get_gripper_state"]
        )
        self.gripper_states = {"left": "open", "right": "open"}
        self.arm_positions = {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]}
    
    def handle_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle manipulation-specific requests"""
        if method == "grasp_object":
            arm = params.get("arm", "right")
            obj = params.get("object", "unknown")
            self.gripper_states[arm] = "closed"
            return {
                "status": "success",
                "message": f"Grasped {obj} with {arm} arm",
                "gripper_state": self.gripper_states[arm]
            }
        
        elif method == "move_arm":
            arm = params.get("arm", "right")
            position = params.get("position", [0.0, 0.0, 0.0])
            self.arm_positions[arm] = position
            return {
                "status": "success",
                "message": f"Moved {arm} arm to {position}",
                "arm_position": self.arm_positions[arm]
            }
        
        elif method == "get_gripper_state":
            arm = params.get("arm", "right")
            return {
                "status": "success",
                "gripper_state": self.gripper_states[arm]
            }
        
        else:
            return {"status": "error", "message": f"Unknown method: {method}"}

class IntegrationManager:
    """Manages communication between services"""
    def __init__(self):
        self.registry = ServiceRegistry()
        self.client = ServiceClient(self.registry)
        
        # Start services in separate threads
        self.locomotion_service = LocomotionService()
        self.manipulation_service = ManipulationService()
        
        # Service threads
        self.service_threads = []
    
    def start_services(self):
        """Start all registered services"""
        services = [self.locomotion_service, self.manipulation_service]
        
        for service in services:
            thread = threading.Thread(
                target=service.start, 
                args=(self.registry,),
                daemon=True
            )
            thread.start()
            self.service_threads.append(thread)
        
        print("All services started")
    
    def coordinate_complex_action(self, action_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate a complex action across multiple services"""
        results = []
        success = True
        
        for action in action_sequence:
            service_name = action["service"]
            method = action["method"]
            params = action["params"]
            
            result = self.client.call_service(service_name, method, params)
            results.append(result)
            
            if result.get("status") != "success":
                success = False
                break  # Stop if any step fails
        
        return {
            "success": success,
            "results": results,
            "action_sequence": action_sequence
        }

# Example usage
def run_service_integration_demo():
    manager = IntegrationManager()
    manager.start_services()
    
    # Allow services to start
    time.sleep(1)
    
    print("Service integration demo:")
    
    # Coordinate a complex action: move to location and grasp object
    complex_action = [
        {
            "service": "locomotion",
            "method": "move_to",
            "params": {"target_position": [1.5, 2.0, 0.0]}
        },
        {
            "service": "manipulation", 
            "method": "grasp_object",
            "params": {"arm": "right", "object": "red cup"}
        }
    ]
    
    result = manager.coordinate_complex_action(complex_action)
    print(f"Complex action result: {result}")
    
    # Check robot position after movement
    position_result = manager.client.call_service("locomotion", "get_position", {})
    print(f"Robot position: {position_result}")
    
    # Check gripper state after grasp
    gripper_result = manager.client.call_service("manipulation", "get_gripper_state", {"arm": "right"})
    print(f"Right gripper state: {gripper_result}")

# Note: Running this would require ZMQ to be installed
# Uncomment the following line to run the demo
# run_service_integration_demo()
```

## Middleware and Communication

### Real-Time Communication Patterns

Humanoid robots require efficient, real-time communication between components:

```python
import asyncio
import time
from typing import Dict, Any, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import queue
import threading

class MessageType(Enum):
    """Types of messages in the humanoid system"""
    COMMAND = "command"
    STATUS = "status"
    SENSOR_DATA = "sensor_data"
    CONTROL_SIGNAL = "control_signal"
    EVENT = "event"

@dataclass
class Message:
    """Standard message format"""
    msg_type: MessageType
    source: str
    destination: str
    content: Dict[str, Any]
    timestamp: float
    correlation_id: str = None

class RealTimeMessageBus:
    """High-performance message bus for real-time communication"""
    def __init__(self, max_queue_size: int = 1000):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.topics: Dict[str, queue.Queue] = {}
        self.max_queue_size = max_queue_size
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "message_loss": 0
        }
        self.running = True
        
        # Start message processing threads
        self.processing_threads = []
        for i in range(4):  # 4 processing threads
            thread = threading.Thread(target=self._message_processor, daemon=True)
            thread.start()
            self.processing_threads.append(thread)
    
    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        
        # Create topic queue if it doesn't exist
        if topic not in self.topics:
            self.topics[topic] = queue.Queue(maxsize=self.max_queue_size)
    
    def publish(self, topic: str, message: Message) -> bool:
        """Publish a message to a topic"""
        if topic in self.topics:
            try:
                self.topics[topic].put_nowait(message)
                self.stats["messages_sent"] += 1
                return True
            except queue.Full:
                self.stats["message_loss"] += 1
                return False
        return False
    
    def _message_processor(self):
        """Process messages from queues"""
        while self.running:
            for topic, msg_queue in self.topics.items():
                try:
                    # Try to get a message without blocking
                    message = msg_queue.get_nowait()
                    self.stats["messages_received"] += 1
                    
                    # Notify subscribers
                    if topic in self.subscribers:
                        for callback in self.subscribers[topic]:
                            try:
                                callback(message)
                            except Exception as e:
                                print(f"Error in subscriber callback: {e}")
                except queue.Empty:
                    # No message available, sleep briefly
                    time.sleep(0.001)
                    continue
    
    def get_stats(self) -> Dict[str, int]:
        """Get communication statistics"""
        return self.stats.copy()

class ControlLoop:
    """Real-time control loop for humanoid systems"""
    def __init__(self, frequency: float = 100.0):  # 100 Hz by default
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.callbacks = []
        self.running = False
        self.loop_thread = None
        
        # Timing statistics
        self.timing_stats = {
            "min_cycle_time": float('inf'),
            "max_cycle_time": 0.0,
            "avg_cycle_time": 0.0,
            "cycle_count": 0
        }
    
    def add_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to the control loop"""
        self.callbacks.append(callback)
    
    def start(self) -> None:
        """Start the control loop"""
        if self.running:
            return
        
        self.running = True
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()
    
    def stop(self) -> None:
        """Stop the control loop"""
        self.running = False
        if self.loop_thread:
            self.loop_thread.join()
    
    def _run_loop(self) -> None:
        """Main control loop"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            cycle_start = time.time()
            
            # Execute all callbacks
            for callback in self.callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in control loop callback: {e}")
            
            # Calculate timing
            cycle_time = time.time() - cycle_start
            self.timing_stats["cycle_count"] += 1
            self.timing_stats["min_cycle_time"] = min(self.timing_stats["min_cycle_time"], cycle_time)
            self.timing_stats["max_cycle_time"] = max(self.timing_stats["max_cycle_time"], cycle_time)
            
            # Update average with exponential moving average
            alpha = 0.01
            avg = self.timing_stats["avg_cycle_time"]
            self.timing_stats["avg_cycle_time"] = alpha * cycle_time + (1 - alpha) * avg if avg > 0 else cycle_time
            
            # Sleep to maintain frequency
            sleep_time = self.period - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

class RealTimeIntegrator:
    """Integrator for real-time communication and control"""
    def __init__(self):
        self.message_bus = RealTimeMessageBus()
        self.control_loop = ControlLoop(frequency=200.0)  # 200 Hz for critical control
        self.high_freq_loop = ControlLoop(frequency=50.0)   # 50 Hz for perception
        
        # Component interfaces
        self.locomotion_interface = None
        self.manipulation_interface = None
        self.perception_interface = None
        
        # Timing critical data
        self.sensor_data_buffer = {}
        self.control_commands = {}
    
    def setup_integration(self) -> None:
        """Set up all integration components"""
        # Subscribe to sensor data topics
        self.message_bus.subscribe("sensor/imu", self._handle_imu_data)
        self.message_bus.subscribe("sensor/camera", self._handle_camera_data)
        self.message_bus.subscribe("sensor/lidar", self._handle_lidar_data)
        
        # Subscribe to control command topics
        self.message_bus.subscribe("control/locomotion", self._handle_locomotion_command)
        self.message_bus.subscribe("control/manipulation", self._handle_manipulation_command)
        
        # Add real-time callbacks
        self.control_loop.add_callback(self._critical_control_callback)
        self.high_freq_loop.add_callback(self._perception_callback)
        
        # Start loops
        self.control_loop.start()
        self.high_freq_loop.start()
    
    def _handle_imu_data(self, message: Message) -> None:
        """Handle IMU sensor data"""
        self.sensor_data_buffer["imu"] = message.content
        # Perform immediate processing if needed
        self._process_imu_for_balance(message.content)
    
    def _handle_camera_data(self, message: Message) -> None:
        """Handle camera sensor data"""
        self.sensor_data_buffer["camera"] = message.content
        # Queue for later processing by perception loop
        pass
    
    def _handle_lidar_data(self, message: Message) -> None:
        """Handle LIDAR sensor data"""
        self.sensor_data_buffer["lidar"] = message.content
        # Queue for navigation processing
        pass
    
    def _handle_locomotion_command(self, message: Message) -> None:
        """Handle locomotion commands"""
        self.control_commands["locomotion"] = message.content
        # Process immediately for real-time response
        self._execute_locomotion_command(message.content)
    
    def _handle_manipulation_command(self, message: Message) -> None:
        """Handle manipulation commands"""
        self.control_commands["manipulation"] = message.content
        # Process immediately for real-time response
        self._execute_manipulation_command(message.content)
    
    def _critical_control_callback(self) -> None:
        """High-frequency control callback for real-time response"""
        # Update balance control based on IMU data
        if "imu" in self.sensor_data_buffer:
            self._update_balance_control(self.sensor_data_buffer["imu"])
        
        # Process any pending control commands
        self._process_pending_commands()
    
    def _perception_callback(self) -> None:
        """Lower-frequency perception callback"""
        # Process camera data
        if "camera" in self.sensor_data_buffer:
            self._process_camera_data(self.sensor_data_buffer["camera"])
        
        # Process LIDAR data
        if "lidar" in self.sensor_data_buffer:
            self._process_lidar_data(self.sensor_data_buffer["lidar"])
    
    def _process_imu_for_balance(self, imu_data: Dict[str, Any]) -> None:
        """Process IMU data for immediate balance control"""
        # In a real system, this would interface with the balance controller
        pass
    
    def _update_balance_control(self, imu_data: Dict[str, Any]) -> None:
        """Update balance control system"""
        # Real-time balance control update
        pass
    
    def _execute_locomotion_command(self, command: Dict[str, Any]) -> None:
        """Execute locomotion command immediately"""
        # Send command to locomotion hardware
        pass
    
    def _execute_manipulation_command(self, command: Dict[str, Any]) -> None:
        """Execute manipulation command immediately"""
        # Send command to manipulation hardware
        pass
    
    def _process_pending_commands(self) -> None:
        """Process any pending control commands"""
        pass
    
    def _process_camera_data(self, camera_data: Dict[str, Any]) -> None:
        """Process camera data for perception"""
        # Object detection, scene understanding, etc.
        pass
    
    def _process_lidar_data(self, lidar_data: Dict[str, Any]) -> None:
        """Process LIDAR data for navigation"""
        # Obstacle detection, mapping, etc.
        pass

# Example usage
def demo_real_time_integration():
    integrator = RealTimeIntegrator()
    integrator.setup_integration()
    
    print("Real-time integration started")
    print("Publishing sensor data...")
    
    # Simulate publishing sensor data
    bus = integrator.message_bus
    
    # Publish IMU data (most critical for real-time control)
    imu_msg = Message(
        msg_type=MessageType.SENSOR_DATA,
        source="imu_sensor",
        destination="balance_controller",
        content={"acceleration": [0.1, 0.2, 9.8], "gyro": [0.01, 0.02, 0.03]},
        timestamp=time.time()
    )
    
    success = bus.publish("sensor/imu", imu_msg)
    print(f"IMU data published: {success}")
    
    # Publish camera data
    camera_msg = Message(
        msg_type=MessageType.SENSOR_DATA,
        source="camera",
        destination="object_detector", 
        content={"image_id": "frame_001", "timestamp": time.time()},
        timestamp=time.time()
    )
    
    success = bus.publish("sensor/camera", camera_msg)
    print(f"Camera data published: {success}")
    
    # Simulate timing statistics after some operations
    time.sleep(2)
    
    stats = bus.get_stats()
    print(f"Communication statistics: {stats}")
    
    # Stop the integrator
    integrator.control_loop.stop()
    integrator.high_freq_loop.stop()

demo_real_time_integration()
```

## Safety and Fault Tolerance

### Safety Integration Layer

Safety is paramount in humanoid robotics integration:

```python
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class SafetyConstraint:
    """Definition of a safety constraint"""
    name: str
    description: str
    check_function: callable
    severity: str  # "low", "medium", "high", "critical"
    enabled: bool = True

class SafetyManager:
    """Manages safety checks and constraints across the system"""
    def __init__(self):
        self.constraints: List[SafetyConstraint] = []
        self.safety_log = []
        self.emergency_stop_active = False
        self.safety_lock = threading.Lock()
        
        # Register default safety constraints
        self._register_default_constraints()
    
    def _register_default_constraints(self):
        """Register default safety constraints"""
        # Joint limit constraints
        self.add_constraint(
            name="joint_limits",
            description="Ensure joints stay within safe limits",
            check_function=self._check_joint_limits,
            severity="critical"
        )
        
        # Collision avoidance
        self.add_constraint(
            name="collision_avoidance", 
            description="Prevent robot from colliding with obstacles",
            check_function=self._check_collision,
            severity="critical"
        )
        
        # Balance constraints
        self.add_constraint(
            name="balance_stability",
            description="Maintain robot balance during operation",
            check_function=self._check_balance_stability, 
            severity="critical"
        )
    
    def add_constraint(self, name: str, description: str, 
                      check_function: callable, severity: str, enabled: bool = True) -> None:
        """Add a safety constraint"""
        constraint = SafetyConstraint(
            name=name,
            description=description,
            check_function=check_function,
            severity=severity,
            enabled=enabled
        )
        self.constraints.append(constraint)
    
    def check_all_constraints(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check all active safety constraints"""
        violations = []
        warnings = []
        
        with self.safety_lock:
            for constraint in self.constraints:
                if not constraint.enabled:
                    continue
                
                try:
                    result = constraint.check_function(system_state)
                    if not result["safe"]:
                        violation = {
                            "constraint": constraint.name,
                            "severity": constraint.severity,
                            "message": result.get("message", "Safety violation"),
                            "details": result
                        }
                        
                        if constraint.severity in ["critical", "high"]:
                            violations.append(violation)
                        else:
                            warnings.append(violation)
                        
                        # Log the violation
                        self._log_safety_event(violation)
                        
                except Exception as e:
                    self._log_safety_event({
                        "constraint": constraint.name,
                        "severity": "critical",
                        "message": f"Error checking constraint: {str(e)}",
                        "details": {}
                    })
        
        return {
            "violations": violations,
            "warnings": warnings,
            "safe": len(violations) == 0
        }
    
    def _check_joint_limits(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if joints are within safe limits"""
        current_joints = system_state.get("joints", {})
        limits = system_state.get("joint_limits", {})
        
        for joint_name, position in current_joints.items():
            if joint_name in limits:
                limit = limits[joint_name]
                if position < limit.get("min", -float('inf')) or position > limit.get("max", float('inf')):
                    return {
                        "safe": False,
                        "message": f"Joint {joint_name} exceeded limits: {position} not in [{limit.get('min')}, {limit.get('max')}]",
                        "joint": joint_name,
                        "position": position
                    }
        
        return {"safe": True}
    
    def _check_collision(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check for potential collisions"""
        obstacles = system_state.get("obstacles", [])
        robot_pose = system_state.get("robot_pose", [0, 0, 0])
        safety_margin = 0.3  # meters
        
        for obstacle in obstacles:
            distance = self._calculate_distance(robot_pose, obstacle.get("position", [0, 0, 0]))
            if distance < safety_margin:
                return {
                    "safe": False,
                    "message": f"Potential collision with obstacle at distance {distance:.2f}m",
                    "obstacle_id": obstacle.get("id"),
                    "distance": distance
                }
        
        return {"safe": True}
    
    def _check_balance_stability(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if robot is in stable balance"""
        com_position = system_state.get("com_position", [0, 0, 0.8])
        com_velocity = system_state.get("com_velocity", [0, 0, 0])
        support_polygon = system_state.get("support_polygon", [])
        
        # Check if CoM is within support polygon
        if support_polygon:
            if not self._point_in_polygon(com_position[0:2], support_polygon):
                return {
                    "safe": False,
                    "message": "Center of mass outside support polygon",
                    "com_position": com_position,
                    "support_polygon": support_polygon
                }
        
        # Check CoM velocity (too fast might indicate instability)
        velocity_magnitude = (com_velocity[0]**2 + com_velocity[1]**2)**0.5
        if velocity_magnitude > 0.5:  # m/s threshold
            return {
                "safe": False,
                "message": f"CoM velocity too high: {velocity_magnitude:.2f} m/s",
                "velocity": velocity_magnitude
            }
        
        return {"safe": True}
    
    def _calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate Euclidean distance between two 3D points"""
        return sum((a - b)**2 for a, b in zip(pos1, pos2))**0.5
    
    def _point_in_polygon(self, point: List[float], polygon: List[List[float]]) -> bool:
        """Check if a 2D point is inside a polygon using ray casting"""
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _log_safety_event(self, event: Dict[str, Any]) -> None:
        """Log a safety event"""
        log_entry = {
            "timestamp": time.time(),
            "event": event
        }
        self.safety_log.append(log_entry)
        
        # Keep only the last 1000 events
        if len(self.safety_log) > 1000:
            self.safety_log = self.safety_log[-1000:]
    
    def trigger_emergency_stop(self) -> None:
        """Trigger emergency stop for the entire system"""
        with self.safety_lock:
            self.emergency_stop_active = True
            self._log_safety_event({
                "type": "emergency_stop",
                "message": "Emergency stop activated by safety system"
            })
    
    def clear_emergency_stop(self) -> None:
        """Clear emergency stop"""
        with self.safety_lock:
            self.emergency_stop_active = False
            self._log_safety_event({
                "type": "emergency_stop_cleared",
                "message": "Emergency stop cleared"
            })
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        return {
            "emergency_stop_active": self.emergency_stop_active,
            "recent_violations": self.safety_log[-10:],  # Last 10 safety events
            "total_violations": len(self.safety_log),
            "active_constraints": [c.name for c in self.constraints if c.enabled]
        }

class FaultToleranceManager:
    """Manages fault detection and recovery in the system"""
    def __init__(self):
        self.faults = {}
        self.recovery_strategies = {}
        self.fault_history = []
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default fault recovery strategies"""
        # Locomotion fault recovery
        self.recovery_strategies["locomotion_failure"] = self._recover_locomotion
        
        # Manipulation fault recovery
        self.recovery_strategies["manipulation_failure"] = self._recover_manipulation
        
        # Sensor failure recovery
        self.recovery_strategies["sensor_failure"] = self._recover_sensor
        
        # Communication failure recovery
        self.recovery_strategies["communication_failure"] = self._recover_communication
    
    def register_fault(self, component: str, fault_type: str, details: Dict[str, Any]) -> None:
        """Register a fault detected in the system"""
        fault_id = f"{component}_{fault_type}_{int(time.time())}"
        
        fault = {
            "id": fault_id,
            "component": component,
            "type": fault_type,
            "details": details,
            "timestamp": time.time(),
            "status": "detected",
            "recovery_attempts": 0
        }
        
        self.faults[fault_id] = fault
        self.fault_history.append(fault)
        
        # Try to recover automatically
        self._attempt_automated_recovery(fault)
    
    def _attempt_automated_recovery(self, fault: Dict[str, Any]) -> bool:
        """Attempt automated recovery for a fault"""
        recovery_key = f"{fault['component']}_{fault['type']}"
        
        if recovery_key in self.recovery_strategies:
            try:
                success = self.recovery_strategies[recovery_key](fault)
                fault["status"] = "recovered" if success else "requires_manual"
                fault["recovery_attempts"] += 1
                return success
            except Exception as e:
                fault["status"] = "recovery_failed"
                fault["recovery_error"] = str(e)
                return False
        
        fault["status"] = "no_strategy"
        return False
    
    def _recover_locomotion(self, fault: Dict[str, Any]) -> bool:
        """Recovery strategy for locomotion failures"""
        print(f"Attempting to recover from locomotion fault: {fault['details']}")
        # In a real system, this would attempt to restart controllers,
        # switch to safe gait, etc.
        time.sleep(0.1)  # Simulate recovery time
        return True  # Simulate successful recovery
    
    def _recover_manipulation(self, fault: Dict[str, Any]) -> bool:
        """Recovery strategy for manipulation failures"""
        print(f"Attempting to recover from manipulation fault: {fault['details']}")
        # In a real system, this would release grippers,
        # attempt alternative grasps, etc.
        time.sleep(0.1)
        return True
    
    def _recover_sensor(self, fault: Dict[str, Any]) -> bool:
        """Recovery strategy for sensor failures"""
        print(f"Attempting to recover from sensor fault: {fault['details']}")
        # In a real system, this would restart sensor drivers,
        # switch to alternative sensors, etc.
        time.sleep(0.1)
        return True
    
    def _recover_communication(self, fault: Dict[str, Any]) -> bool:
        """Recovery strategy for communication failures"""
        print(f"Attempting to recover from communication fault: {fault['details']}")
        # In a real system, this would restart communication interfaces
        time.sleep(0.1)
        return True
    
    def get_fault_status(self) -> Dict[str, Any]:
        """Get current fault status"""
        active_faults = [f for f in self.faults.values() if f["status"] in ["detected", "recovery_failed"]]
        return {
            "active_faults": active_faults,
            "total_faults": len(self.faults),
            "recovery_success_rate": self._calculate_recovery_rate()
        }
    
    def _calculate_recovery_rate(self) -> float:
        """Calculate the overall fault recovery success rate"""
        if not self.fault_history:
            return 1.0  # No faults, so 100% success rate
        
        recovered = sum(1 for f in self.fault_history 
                       if f.get("status") in ["recovered", "requires_manual"])
        return recovered / len(self.fault_history)

class SafetyIntegrator:
    """Integrates safety and fault tolerance into the system"""
    def __init__(self, system_integrator):
        self.system_integrator = system_integrator
        self.safety_manager = SafetyManager()
        self.fault_manager = FaultToleranceManager()
        self.safety_thread = None
        self.running = False
    
    def start_safety_monitoring(self) -> None:
        """Start safety monitoring in a background thread"""
        self.running = True
        self.safety_thread = threading.Thread(target=self._safety_monitor_loop, daemon=True)
        self.safety_thread.start()
    
    def stop_safety_monitoring(self) -> None:
        """Stop safety monitoring"""
        self.running = False
        if self.safety_thread:
            self.safety_thread.join()
    
    def _safety_monitor_loop(self) -> None:
        """Main safety monitoring loop"""
        while self.running:
            try:
                # Get system state
                system_state = self._get_system_state()
                
                # Check safety constraints
                safety_results = self.safety_manager.check_all_constraints(system_state)
                
                # Handle safety violations
                if not safety_results["safe"]:
                    for violation in safety_results["violations"]:
                        print(f"Safety violation: {violation}")
                        
                        # For critical violations, trigger emergency stop
                        if violation["severity"] == "critical":
                            self.safety_manager.trigger_emergency_stop()
                
                # Check for component failures and register faults
                self._check_component_health()
                
                # Sleep briefly to avoid overwhelming the system
                time.sleep(0.05)  # 20 Hz safety checks
                
            except Exception as e:
                print(f"Error in safety monitor: {e}")
                time.sleep(0.1)
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for safety checking"""
        # In a real system, this would gather state from all components
        # For this example, we'll return simulated state
        return {
            "joints": {"hip_pitch": 0.1, "knee_pitch": 0.2},
            "joint_limits": {"hip_pitch": {"min": -1.5, "max": 1.5}},
            "obstacles": [{"position": [2, 0, 0], "id": "obstacle_1"}],
            "robot_pose": [1, 0, 0],
            "com_position": [0.05, 0.02, 0.8],
            "com_velocity": [0.01, 0.005, 0],
            "support_polygon": [[0.1, 0.05], [0.1, -0.05], [-0.1, -0.05], [-0.1, 0.05]]
        }
    
    def _check_component_health(self) -> None:
        """Check health of system components and register faults"""
        # In a real system, this would interface with component health monitors
        # For this example, we'll simulate occasional faults
        
        import random
        if random.random() < 0.01:  # 1% chance of fault per check
            components = ["locomotion", "manipulation", "perception"]
            fault_types = ["failure", "timeout", "out_of_range"]
            
            component = random.choice(components)
            fault_type = random.choice(fault_types)
            
            self.fault_manager.register_fault(
                component=component,
                fault_type=fault_type,
                details={"simulated": True, "timestamp": time.time()}
            )

# Example usage
def demo_safety_integration():
    # Create a basic system integrator (this would be the actual system)
    class MockSystemIntegrator:
        pass
    
    system_integrator = MockSystemIntegrator()
    safety_integrator = SafetyIntegrator(system_integrator)
    
    print("Starting safety monitoring...")
    safety_integrator.start_safety_monitoring()
    
    # Simulate running for a while
    time.sleep(3)
    
    # Get safety and fault status
    safety_status = safety_integrator.safety_manager.get_safety_status()
    fault_status = safety_integrator.fault_manager.get_fault_status()
    
    print(f"Safety status: {safety_status}")
    print(f"Fault status: {fault_status}")
    
    # Stop safety monitoring
    safety_integrator.stop_safety_monitoring()

demo_safety_integration()
```

## Integration Testing and Validation

### Testing Framework for Integrated Systems

Integration testing is crucial for humanoid robotics systems:

```python
import unittest
import asyncio
from typing import Dict, Any
import time

class IntegrationTestCase(unittest.TestCase):
    """Base class for integration testing"""
    def setUp(self):
        """Set up test environment"""
        self.test_timeout = 10.0  # seconds
        self.system_state = {}
        
        # Create a test system instance
        self.integrator = SystemIntegrator()
    
    def tearDown(self):
        """Clean up after test"""
        pass
    
    def wait_for_condition(self, condition_func, timeout=None):
        """Wait for a condition to be true"""
        if timeout is None:
            timeout = self.test_timeout
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(0.1)
        return False

class TestLocomotionIntegration(IntegrationTestCase):
    """Test locomotion system integration"""
    def test_navigation_integrity(self):
        """Test that navigation commands are properly processed"""
        # Send navigation command
        nav_result = asyncio.run(
            self.integrator.route_message("locomotion", {
                "type": "move_to",
                "target_position": [1.0, 1.0, 0.0]
            })
        )
        
        # Verify command was accepted
        self.assertEqual(nav_result["status"], "navigation_started")
        self.assertIn("estimated_time", nav_result)
    
    def test_gait_transition_safety(self):
        """Test safe transitions between different gaits"""
        # Start with standing gait
        stand_result = asyncio.run(
            self.integrator.route_message("locomotion", {
                "type": "set_gait", 
                "gait": "stand"
            })
        )
        self.assertEqual(stand_result["status"], "gait_changed")
        
        # Transition to walking gait
        walk_result = asyncio.run(
            self.integrator.route_message("locomotion", {
                "type": "set_gait",
                "gait": "walk"
            })
        )
        self.assertEqual(walk_result["status"], "gait_changed")
        
        # Verify system state reflects gait change
        state = self.integrator.get_system_state()
        # This would check internal state in a real implementation

class TestManipulationIntegration(IntegrationTestCase):
    """Test manipulation system integration"""
    def test_grasp_and_place_sequence(self):
        """Test a complete grasp and place sequence"""
        # First, move to object location
        nav_result = asyncio.run(
            self.integrator.route_message("locomotion", {
                "type": "move_to",
                "target_position": [0.5, 0.0, 0.0]
            })
        )
        self.assertEqual(nav_result["status"], "navigation_started")
        
        # Then, grasp the object
        grasp_result = asyncio.run(
            self.integrator.route_message("manipulation", {
                "type": "grasp_object",
                "arm": "right",
                "object": {"type": "cup", "position": [0.5, 0.0, 0.8]}
            })
        )
        self.assertEqual(grasp_result["status"], "grasp_executed")
        
        # Finally, place the object
        place_result = asyncio.run(
            self.integrator.route_message("manipulation", {
                "type": "move_arm",
                "arm": "right", 
                "target_pose": [0.8, 0.3, 0.8, 0, 0, 0]
            })
        )
        self.assertEqual(place_result["status"], "movement_completed")

class TestPerceptionIntegration(IntegrationTestCase):
    """Test perception system integration"""
    def test_object_detection_pipeline(self):
        """Test the complete object detection pipeline"""
        # Update environment map
        map_result = asyncio.run(
            self.integrator.route_message("perception", {
                "type": "update_environment",
                "environment_map": {"objects": [{"name": "chair", "position": [1, 1, 0]}]}
            })
        )
        self.assertEqual(map_result["status"], "map_updated")
        
        # Request object detection
        obj_result = asyncio.run(
            self.integrator.route_message("perception", {
                "type": "request_objects"
            })
        )
        self.assertIn("objects", obj_result)
    
    def test_sensor_data_processing(self):
        """Test that sensor data is properly processed"""
        # Process simulated camera data
        camera_result = asyncio.run(
            self.integrator.route_message("perception", {
                "type": "process_sensor_data",
                "sensor_type": "camera",
                "data": {"frame_id": "test_frame", "timestamp": time.time()}
            })
        )
        self.assertEqual(camera_result["status"], "data_processed")

class TestCognitiveIntegration(IntegrationTestCase):
    """Test cognitive system integration"""
    def test_conversation_and_action_sequence(self):
        """Test a sequence involving conversation and physical action"""
        # Process conversational input
        conv_result = asyncio.run(
            self.integrator.route_message("cognitive", {
                "type": "process_conversation",
                "user_input": "Please bring me the cup from the table"
            })
        )
        self.assertEqual(conv_result["status"], "response_generated")
        
        # The cognitive system should generate a plan
        plan_result = asyncio.run(
            self.integrator.route_message("cognitive", {
                "type": "plan_task",
                "task": {
                    "target_location": [1.5, 0.0, 0.0],
                    "object": "cup"
                }
            })
        )
        self.assertEqual(plan_result["status"], "plan_generated")
        self.assertGreater(plan_result["steps"], 0)

class TestSystemWideIntegration(IntegrationTestCase):
    """Test integration across all systems"""
    def test_complete_task_execution(self):
        """Test a complete task involving all subsystems"""
        # Step 1: Receive and understand command
        understanding_result = asyncio.run(
            self.integrator.route_message("cognitive", {
                "type": "process_conversation",
                "user_input": "Go to the kitchen and bring me a cup of water"
            })
        )
        
        # Step 2: Plan the task
        planning_result = asyncio.run(
            self.integrator.route_message("cognitive", {
                "type": "plan_task",
                "task": {
                    "target_location": [3.0, 2.0, 0.0],  # kitchen location
                    "object": "water_cup",
                    "actions": ["navigate", "grasp", "return"]
                }
            })
        )
        
        # Step 3: Execute navigation
        nav_result = asyncio.run(
            self.integrator.route_message("locomotion", {
                "type": "move_to",
                "target_position": [3.0, 2.0, 0.0]
            })
        )
        
        # Step 4: Execute manipulation
        manipulation_result = asyncio.run(
            self.integrator.route_message("manipulation", {
                "type": "grasp_object",
                "object": "water_cup",
                "arm": "right"
            })
        )
        
        # Verify all steps completed successfully
        self.assertEqual(understanding_result["status"], "response_generated")
        self.assertEqual(planning_result["status"], "plan_generated")
        self.assertEqual(nav_result["status"], "navigation_started")
        self.assertEqual(manipulation_result["status"], "grasp_executed")

class TestSafetyIntegration(IntegrationTestCase):
    """Test safety system integration"""
    def test_emergency_stop_propagation(self):
        """Test that emergency stop propagates to all subsystems"""
        # This would test that when safety system triggers emergency stop,
        # all other subsystems respond appropriately
        pass
    
    def test_constraint_violation_handling(self):
        """Test handling of safety constraint violations"""
        # This would test that constraint violations are properly detected and handled
        pass

# Run integration tests
def run_integration_tests():
    """Run all integration tests"""
    test_classes = [
        TestLocomotionIntegration,
        TestManipulationIntegration,
        TestPerceptionIntegration,
        TestCognitiveIntegration,
        TestSystemWideIntegration,
        TestSafetyIntegration
    ]
    
    all_tests = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        all_tests.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    # Print summary
    print(f"\nIntegration Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()

# Run the tests
if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
```

## Exercises

1. **Component Integration Exercise**: Integrate multiple robot components (locomotion, manipulation, perception) and verify their coordinated operation through a complex task.

2. **Middleware Implementation Exercise**: Design and implement a middleware system for a humanoid robot that efficiently routes messages between components with minimal latency.

3. **Safety System Exercise**: Create a comprehensive safety system for a humanoid robot that monitors joint limits, balance, and environmental constraints in real-time.

4. **Fault Tolerance Exercise**: Implement fault detection and recovery mechanisms for common humanoid robot failures like sensor malfunctions or actuator errors.

5. **System Optimization Exercise**: Profile and optimize an integrated humanoid system to reduce latency and improve real-time performance.

6. **Integration Testing Exercise**: Develop a comprehensive testing framework for verifying the correct integration of all humanoid robot subsystems.

7. **Scalability Exercise**: Design an integration architecture that can scale to accommodate additional subsystems and capabilities.

## Summary

System integration in humanoid robotics requires carefully coordinating multiple complex subsystems that must work together seamlessly. Success depends on implementing robust communication architectures, safety systems, and fault tolerance mechanisms. The integration must handle real-time requirements, manage system complexity, and ensure safe operation in human environments. Through proper testing, validation, and iterative improvement, integrated humanoid systems can achieve the reliability and performance needed for real-world deployment.