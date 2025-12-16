---
title: System Composition for Humanoid Robotics
sidebar_position: 1
---

# System Composition for Humanoid Robotics

## Introduction

System composition in humanoid robotics refers to the architectural approaches and methodologies used to build complex robotic systems from smaller, reusable components. This chapter explores how to compose humanoid robots from modular subsystems while maintaining system coherence, performance, and safety. We'll cover architectural patterns, component design principles, and methodologies for creating scalable and maintainable humanoid robot systems.

## Component-Based Architecture

### Modular Component Design

Effective system composition relies on well-designed, modular components that can be combined in various ways:

```python
import abc
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class ComponentInfo:
    """Information about a robot component"""
    name: str
    version: str
    description: str
    interfaces: List[str]
    dependencies: List[str]
    capabilities: List[str]

class ComponentInterface(abc.ABC):
    """Base interface for all robot components"""
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize the component, return True if successful"""
        pass
    
    @abc.abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return output"""
        pass
    
    @abc.abstractmethod
    def shutdown(self) -> bool:
        """Clean shutdown of the component"""
        pass
    
    def get_info(self) -> ComponentInfo:
        """Get information about the component"""
        return ComponentInfo(
            name=self.__class__.__name__,
            version="1.0.0",
            description=f"Component: {self.__class__.__name__}",
            interfaces=[],
            dependencies=[],
            capabilities=[]
        )

class ComponentManager:
    """Manages the lifecycle and composition of robot components"""
    def __init__(self):
        self.components: Dict[str, ComponentInterface] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.connections: List[Dict[str, str]] = []
        self.system_state = "uninitialized"
    
    def register_component(self, name: str, component: ComponentInterface) -> bool:
        """Register a component with the system"""
        if name in self.components:
            print(f"Component {name} already registered")
            return False
        
        if not component.initialize():
            print(f"Failed to initialize component {name}")
            return False
        
        self.components[name] = component
        self.dependencies[name] = component.get_info().dependencies
        return True
    
    def connect_components(self, source: str, destination: str, interface: str) -> bool:
        """Connect two components"""
        if source not in self.components or destination not in self.components:
            return False
        
        connection = {
            "source": source,
            "destination": destination,
            "interface": interface
        }
        self.connections.append(connection)
        return True
    
    def compose_system(self, component_config: List[Dict[str, Any]]) -> bool:
        """Compose a system from configuration"""
        success = True
        
        # Register all components
        for config in component_config:
            name = config["name"]
            component_class = config["class"]
            component = component_class()
            
            if not self.register_component(name, component):
                print(f"Failed to register component: {name}")
                success = False
                continue
            
            # Connect to dependencies if specified
            if "connect_to" in config:
                for target in config["connect_to"]:
                    if isinstance(target, str):
                        self.connect_components(name, target, "default")
                    elif isinstance(target, dict):
                        self.connect_components(name, target["name"], target["interface"])
        
        # Check for dependency cycles
        if not self._check_dependencies():
            print("Dependency cycle detected")
            success = False
        
        if success:
            self.system_state = "initialized"
        
        return success
    
    def _check_dependencies(self) -> bool:
        """Check for dependency cycles using topological sort"""
        # Simple cycle detection - in a real system, implement proper topological sort
        visited = set()
        rec_stack = set()
        
        def has_cycle(component: str) -> bool:
            if component in rec_stack:
                return True
            if component in visited:
                return False
            
            visited.add(component)
            rec_stack.add(component)
            
            for dep in self.dependencies.get(component, []):
                if has_cycle(dep):
                    return True
            
            rec_stack.remove(component)
            return False
        
        for component in self.components:
            if has_cycle(component):
                return False
        
        return True
    
    def execute_system(self) -> bool:
        """Execute the composed system"""
        if self.system_state != "initialized":
            print("System not properly initialized")
            return False
        
        # Execute in dependency order
        execution_order = self._get_execution_order()
        
        for component_name in execution_order:
            component = self.components[component_name]
            try:
                # This would be more sophisticated in a real implementation
                # with proper data flow management
                component.process(None)
            except Exception as e:
                print(f"Error executing component {component_name}: {e}")
                return False
        
        return True
    
    def _get_execution_order(self) -> List[str]:
        """Get the order in which components should be executed"""
        # Simple ordering - in a real system, implement topological sort
        # based on dependencies
        return list(self.components.keys())
    
    def decompose_system(self) -> bool:
        """Decompose and shutdown the system"""
        # Shutdown in reverse dependency order
        execution_order = self._get_execution_order()
        execution_order.reverse()
        
        success = True
        for component_name in execution_order:
            component = self.components[component_name]
            if not component.shutdown():
                print(f"Error shutting down component {component_name}")
                success = False
        
        self.components.clear()
        self.connections.clear()
        self.system_state = "uninitialized"
        
        return success

class LocomotionComponent(ComponentInterface):
    """Locomotion system component"""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_initialized = False
        self.current_gait = "stand"
        self.target_velocity = [0.0, 0.0, 0.0]
    
    def initialize(self) -> bool:
        """Initialize locomotion system"""
        print("Initializing locomotion system...")
        # Initialize physical interfaces, calibration, etc.
        self.is_initialized = True
        return True
    
    def process(self, input_data: Any) -> Any:
        """Process locomotion commands"""
        if input_data is None:
            return {"status": "idle", "gait": self.current_gait}
        
        command = input_data.get("command", "")
        
        if command == "move_to":
            target = input_data.get("target", [0, 0, 0])
            # Execute movement
            return {"status": "moving", "target": target}
        
        elif command == "set_gait":
            new_gait = input_data.get("gait", "stand")
            self.current_gait = new_gait
            return {"status": "gait_changed", "new_gait": new_gait}
        
        return {"status": "unknown_command"}
    
    def shutdown(self) -> bool:
        """Shutdown locomotion system"""
        print("Shutting down locomotion system...")
        # Stop all motion, power down motors, etc.
        self.current_gait = "stand"
        self.is_initialized = False
        return True
    
    def get_info(self) -> ComponentInfo:
        return ComponentInfo(
            name="locomotion",
            version="1.0.0",
            description="Locomotion Control System",
            interfaces=["command_interface", "feedback_interface"],
            dependencies=[],
            capabilities=["navigation", "balance_control", "gait_control"]
        )

class PerceptionComponent(ComponentInterface):
    """Perception system component"""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_initialized = False
        self.sensors = {}
        self.processed_data = {}
    
    def initialize(self) -> bool:
        """Initialize perception system"""
        print("Initializing perception system...")
        # Initialize sensors, calibrate cameras, etc.
        self.sensors = {
            "camera": "initialized",
            "lidar": "initialized",
            "imu": "initialized"
        }
        self.is_initialized = True
        return True
    
    def process(self, input_data: Any) -> Any:
        """Process sensor data"""
        if input_data is None:
            return {"status": "idle", "sensors": list(self.sensors.keys())}
        
        sensor_type = input_data.get("sensor_type")
        
        if sensor_type == "camera":
            # Process camera data
            return {"status": "camera_processed", "objects_detected": ["person", "cup"]}
        
        elif sensor_type == "lidar":
            # Process LiDAR data
            return {"status": "lidar_processed", "obstacles": [{"distance": 1.5, "angle": 0.2}]}
        
        return {"status": "sensor_not_specified"}
    
    def shutdown(self) -> bool:
        """Shutdown perception system"""
        print("Shutting down perception system...")
        # Power down sensors, release resources
        self.sensors.clear()
        self.is_initialized = False
        return True
    
    def get_info(self) -> ComponentInfo:
        return ComponentInfo(
            name="perception",
            version="1.0.0",
            description="Perception and Sensing System",
            interfaces=["sensor_interface", "detection_interface"],
            dependencies=[],
            capabilities=["object_detection", "environment_mapping", "obstacle_avoidance"]
        )

class ManipulationComponent(ComponentInterface):
    """Manipulation system component"""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_initialized = False
        self.arms = {"left": "ready", "right": "ready"}
        self.grippers = {"left": "open", "right": "open"}
    
    def initialize(self) -> bool:
        """Initialize manipulation system"""
        print("Initializing manipulation system...")
        # Initialize arm controllers, gripper motors, etc.
        self.is_initialized = True
        return True
    
    def process(self, input_data: Any) -> Any:
        """Process manipulation commands"""
        if input_data is None:
            return {"status": "idle", "arms": self.arms, "grippers": self.grippers}
        
        command = input_data.get("command", "")
        
        if command == "grasp_object":
            arm = input_data.get("arm", "right")
            obj = input_data.get("object", "unknown")
            self.grippers[arm] = "closed"
            return {"status": "grasping", "arm": arm, "object": obj}
        
        elif command == "release_object":
            arm = input_data.get("arm", "right")
            self.grippers[arm] = "open"
            return {"status": "releasing", "arm": arm}
        
        elif command == "move_arm":
            arm = input_data.get("arm", "right")
            pose = input_data.get("pose", [0, 0, 0, 0, 0, 0])
            return {"status": "moving_arm", "arm": arm, "pose": pose}
        
        return {"status": "unknown_command"}
    
    def shutdown(self) -> bool:
        """Shutdown manipulation system"""
        print("Shutting down manipulation system...")
        # Move arms to safe position, open grippers
        for arm in self.arms:
            self.grippers[arm] = "open"
        self.is_initialized = False
        return True
    
    def get_info(self) -> ComponentInfo:
        return ComponentInfo(
            name="manipulation",
            version="1.0.0",
            description="Manipulation and Grasping System",
            interfaces=["manipulation_interface", "gripper_interface"],
            dependencies=[],
            capabilities=["object_grasping", "arm_control", "precision_manipulation"]
        )

class CognitiveComponent(ComponentInterface):
    """Cognitive system component"""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_initialized = False
        self.current_goals = []
        self.memory = {}
        self.dialog_state = {}
    
    def initialize(self) -> bool:
        """Initialize cognitive system"""
        print("Initializing cognitive system...")
        # Initialize language models, memory structures, etc.
        self.is_initialized = True
        return True
    
    def process(self, input_data: Any) -> Any:
        """Process cognitive tasks"""
        if input_data is None:
            return {"status": "idle", "active_goals": len(self.current_goals)}
        
        task_type = input_data.get("task_type", "")
        
        if task_type == "natural_language":
            text = input_data.get("text", "")
            # Process natural language input
            intent = self._classify_intent(text)
            return {"status": "processed", "intent": intent, "response": self._generate_response(intent)}
        
        elif task_type == "task_planning":
            goal = input_data.get("goal", "")
            plan = self._plan_task(goal)
            return {"status": "planned", "plan": plan}
        
        return {"status": "unknown_task"}
    
    def _classify_intent(self, text: str) -> str:
        """Simple intent classification"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["hello", "hi", "hey"]):
            return "greeting"
        elif any(word in text_lower for word in ["move", "go", "walk", "navigate"]):
            return "navigation"
        elif any(word in text_lower for word in ["grasp", "pick", "take", "hold"]):
            return "manipulation"
        else:
            return "unknown"
    
    def _generate_response(self, intent: str) -> str:
        """Generate appropriate response"""
        responses = {
            "greeting": "Hello! How can I assist you today?",
            "navigation": "I can help with navigation. Where would you like to go?",
            "manipulation": "I can manipulate objects. What would you like me to do?",
            "unknown": "I'm not sure I understand. Could you please rephrase?"
        }
        return responses.get(intent, "I don't know how to respond to that.")
    
    def _plan_task(self, goal: str) -> List[str]:
        """Simple task planning"""
        if "navigation" in goal.lower():
            return ["move_to_target", "check_environment", "adjust_path"]
        elif "manipulation" in goal.lower():
            return ["locate_object", "approach_object", "grasp_object", "verify_grasp"]
        else:
            return ["analyze_request", "consult_memory", "generate_plan"]
    
    def shutdown(self) -> bool:
        """Shutdown cognitive system"""
        print("Shutting down cognitive system...")
        # Save memory, close models, etc.
        self.current_goals.clear()
        self.memory.clear()
        self.is_initialized = False
        return True
    
    def get_info(self) -> ComponentInfo:
        return ComponentInfo(
            name="cognitive",
            version="1.0.0",
            description="Cognitive and Decision Making System",
            interfaces=["dialog_interface", "planning_interface"],
            dependencies=["perception", "locomotion", "manipulation"],
            capabilities=["natural_language", "task_planning", "decision_making"]
        )

# Example usage
def demo_component_composition():
    """Demonstrate component-based composition"""
    print("=== Component-Based System Composition Demo ===\n")
    
    manager = ComponentManager()
    
    # Define system configuration
    system_config = [
        {
            "name": "locomotion_system",
            "class": LocomotionComponent,
            "connect_to": ["cognitive_system"]
        },
        {
            "name": "perception_system", 
            "class": PerceptionComponent,
            "connect_to": ["cognitive_system"]
        },
        {
            "name": "manipulation_system",
            "class": ManipulationComponent,
            "connect_to": ["cognitive_system"]
        },
        {
            "name": "cognitive_system",
            "class": CognitiveComponent,
            "connect_to": ["locomotion_system", "perception_system", "manipulation_system"]
        }
    ]
    
    # Compose the system
    success = manager.compose_system(system_config)
    print(f"System composition {'successful' if success else 'failed'}\n")
    
    if success:
        # Test the composed system
        print("Testing system components:")
        
        # Test locomotion
        locomotion_comp = manager.components["locomotion_system"]
        result = locomotion_comp.process({"command": "set_gait", "gait": "walk"})
        print(f"Locomotion: {result}")
        
        # Test perception
        perception_comp = manager.components["perception_system"]
        result = perception_comp.process({"sensor_type": "camera"})
        print(f"Perception: {result}")
        
        # Test manipulation
        manipulation_comp = manager.components["manipulation_system"]
        result = manipulation_comp.process({"command": "grasp_object", "arm": "right", "object": "cup"})
        print(f"Manipulation: {result}")
        
        # Test cognitive
        cognitive_comp = manager.components["cognitive_system"]
        result = cognitive_comp.process({"task_type": "natural_language", "text": "Hello robot"})
        print(f"Cognitive: {result}")
        
        # Execute the full system
        print(f"\nExecuting system: {'Success' if manager.execute_system() else 'Failed'}")
        
        # Decompose the system
        print(f"\nDecomposing system: {'Success' if manager.decompose_system() else 'Failed'}")

demo_component_composition()
```

### Component Interfaces and Adapters

Well-defined interfaces enable flexible composition:

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import asyncio

# Define protocols for component interfaces
@runtime_checkable
class MotionController(Protocol):
    """Protocol for motion control components"""
    def move_to(self, position: List[float]) -> bool: ...
    def stop(self) -> bool: ...
    def get_position(self) -> List[float]: ...

@runtime_checkable  
class Sensor(Protocol):
    """Protocol for sensor components"""
    def read(self) -> Any: ...
    def calibrate(self) -> bool: ...
    def is_connected(self) -> bool: ...

@runtime_checkable
class Actuator(Protocol):
    """Protocol for actuator components"""
    def activate(self, value: float) -> bool: ...
    def deactivate(self) -> bool: ...
    def get_state(self) -> str: ...

class ComponentAdapter(ABC):
    """Base adapter for connecting different component types"""
    
    @abstractmethod
    def adapt_input(self, input_data: Any) -> Any:
        """Adapt input data to component format"""
        pass
    
    @abstractmethod 
    def adapt_output(self, output_data: Any) -> Any:
        """Adapt component output to system format"""
        pass

class LocomotionAdapter(ComponentAdapter):
    """Adapter for locomotion components"""
    
    def adapt_input(self, input_data: Any) -> Dict[str, Any]:
        """Convert high-level navigation commands to locomotion format"""
        if isinstance(input_data, dict):
            if "destination" in input_data:
                # Convert destination to path planning format
                return {
                    "target_position": input_data["destination"],
                    "speed": input_data.get("speed", 0.5),
                    "avoidance_radius": input_data.get("avoidance_radius", 0.5)
                }
        return {"target_position": [0, 0, 0], "speed": 0.5}
    
    def adapt_output(self, output_data: Any) -> Dict[str, Any]:
        """Convert locomotion output to system format"""
        if isinstance(output_data, dict):
            return {
                "position": output_data.get("position", [0, 0, 0]),
                "status": output_data.get("status", "idle"),
                "remaining_distance": output_data.get("remaining_distance", 0.0),
                "estimated_time": output_data.get("estimated_time", 0.0)
            }
        return {"status": "unknown", "position": [0, 0, 0]}

class PerceptionAdapter(ComponentAdapter):
    """Adapter for perception components"""
    
    def adapt_input(self, input_data: Any) -> Dict[str, Any]:
        """Convert system requests to perception format"""
        if isinstance(input_data, str):
            # Convert simple commands to perception format
            if input_data == "detect_objects":
                return {"task": "object_detection", "mode": "all"}
            elif input_data == "detect_people":
                return {"task": "person_detection", "mode": "tracking"}
        return {"task": "idle"}
    
    def adapt_output(self, output_data: Any) -> Dict[str, Any]:
        """Convert perception results to system format"""
        if isinstance(output_data, dict):
            objects = output_data.get("objects", [])
            return {
                "detected_objects": objects,
                "object_count": len(objects),
                "timestamp": output_data.get("timestamp", time.time()),
                "confidence_threshold": output_data.get("confidence_threshold", 0.5)
            }
        return {"detected_objects": [], "object_count": 0}

class StandardizedComponent:
    """A component that uses standard interfaces and adapters"""
    
    def __init__(self, name: str, adapter: ComponentAdapter):
        self.name = name
        self.adapter = adapter
        self.is_running = False
    
    async def process_request(self, request: Any) -> Any:
        """Process a request through the adapter"""
        # Adapt input
        adapted_input = self.adapter.adapt_input(request)
        
        # Process (in a real implementation, this would interact with the actual component)
        result = await self._internal_process(adapted_input)
        
        # Adapt output
        adapted_output = self.adapter.adapt_output(result)
        
        return adapted_output
    
    async def _internal_process(self, adapted_input: Any) -> Any:
        """Internal processing implementation"""
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        # Return simulated result
        return {
            "status": "processed",
            "adapted_input": adapted_input,
            "timestamp": time.time()
        }

class CompositionManager:
    """Manages component composition with standardized interfaces"""
    
    def __init__(self):
        self.components: Dict[str, StandardizedComponent] = {}
        self.connections: Dict[str, List[str]] = {}
        self.data_flow_graph = {}
    
    def add_component(self, name: str, component: StandardizedComponent) -> bool:
        """Add a standardized component to the system"""
        if name in self.components:
            return False
        
        self.components[name] = component
        self.connections[name] = []
        return True
    
    def connect_components(self, source: str, destination: str, flow_type: str = "data") -> bool:
        """Connect two components for data flow"""
        if source not in self.components or destination not in self.components:
            return False
        
        if destination not in self.connections[source]:
            self.connections[source].append(destination)
        
        # Build data flow graph
        if source not in self.data_flow_graph:
            self.data_flow_graph[source] = []
        self.data_flow_graph[source].append({
            "destination": destination,
            "type": flow_type,
            "transform_func": None  # Optional data transformation
        })
        
        return True
    
    async def execute_request(self, component_name: str, request: Any) -> Any:
        """Execute a request on a specific component"""
        if component_name not in self.components:
            return {"error": f"Component {component_name} not found"}
        
        component = self.components[component_name]
        return await component.process_request(request)
    
    async def execute_pipeline(self, start_component: str, initial_request: Any) -> Dict[str, Any]:
        """Execute a pipeline of component interactions"""
        results = {}
        current_request = initial_request
        current_component = start_component
        
        # Simple pipeline execution - in a real system, this would be more sophisticated
        visited = set()
        
        while current_component and current_component not in visited:
            visited.add(current_component)
            
            # Process with current component
            result = await self.execute_request(current_component, current_request)
            results[current_component] = result
            
            # Find next component in the pipeline
            next_components = self.connections.get(current_component, [])
            
            if next_components:
                # For this example, just take the first connected component
                # In a real system, there would be more sophisticated routing
                current_component = next_components[0]
                current_request = result  # Pass result as input to next component
            else:
                current_component = None  # End of pipeline
        
        return {
            "pipeline_results": results,
            "final_result": result if result else None,
            "executed_components": list(results.keys())
        }

# Example usage with standardized interfaces
async def demo_standardized_composition():
    """Demonstrate standardized component composition"""
    print("=== Standardized Component Composition Demo ===\n")
    
    # Create components with adapters
    locomotion_adapter = LocomotionAdapter()
    locomotion_component = StandardizedComponent("locomotion", locomotion_adapter)
    
    perception_adapter = PerceptionAdapter()
    perception_component = StandardizedComponent("perception", perception_adapter)
    
    # Create composition manager
    composition_manager = CompositionManager()
    
    # Add components
    composition_manager.add_component("locomotion", locomotion_component)
    composition_manager.add_component("perception", perception_component)
    
    # Connect components
    composition_manager.connect_components("perception", "locomotion", "navigation_data")
    
    # Execute a simple request
    print("Executing locomotion request:")
    result = await composition_manager.execute_request(
        "locomotion", 
        {"destination": [2.0, 1.0, 0.0], "speed": 0.8}
    )
    print(f"Locomotion result: {result}")
    
    print("\nExecuting perception request:")
    result = await composition_manager.execute_request(
        "perception", 
        "detect_objects"
    )
    print(f"Perception result: {result}")
    
    print("\nExecuting pipeline:")
    pipeline_result = await composition_manager.execute_pipeline(
        "perception",
        "detect_objects"
    )
    print(f"Pipeline result: {pipeline_result}")

# Run the demo
asyncio.run(demo_standardized_composition())
```

## Architecture Patterns

### Service-Oriented Architecture

Service-oriented architecture enables flexible composition through well-defined services:

```python
import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Callable

class ServiceRegistry:
    """Registry for robot services"""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self.subscribers: Dict[str, List[str]] = {}  # topic -> list of service names
        self.service_lock = asyncio.Lock()
    
    async def register_service(self, service_name: str, service_info: Dict[str, Any]) -> bool:
        """Register a service"""
        async with self.service_lock:
            if service_name in self.services:
                return False  # Service already registered
            
            self.services[service_name] = {
                "info": service_info,
                "timestamp": time.time(),
                "status": "active"
            }
            return True
    
    async def unregister_service(self, service_name: str) -> bool:
        """Unregister a service"""
        async with self.service_lock:
            if service_name in self.services:
                del self.services[service_name]
                
                # Clean up subscriptions
                for topic, subscribers in self.subscribers.items():
                    if service_name in subscribers:
                        subscribers.remove(service_name)
                
                return True
            return False
    
    async def find_service(self, capability: str) -> Optional[str]:
        """Find a service that provides a specific capability"""
        async with self.service_lock:
            for name, service in self.services.items():
                if capability in service["info"].get("capabilities", []):
                    return name
        return None
    
    async def subscribe_to_topic(self, service_name: str, topic: str) -> bool:
        """Subscribe a service to a topic"""
        async with self.service_lock:
            if service_name not in self.services:
                return False
            
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            
            if service_name not in self.subscribers[topic]:
                self.subscribers[topic].append(service_name)
            
            return True
    
    async def get_subscribers(self, topic: str) -> List[str]:
        """Get all services subscribed to a topic"""
        async with self.service_lock:
            return self.subscribers.get(topic, []).copy()

class ServiceRequest:
    """Standardized service request"""
    
    def __init__(self, method: str, params: Dict[str, Any], service: str, 
                 correlation_id: Optional[str] = None):
        self.id = correlation_id or str(uuid.uuid4())
        self.method = method
        self.params = params
        self.service = service
        self.timestamp = time.time()

class ServiceResponse:
    """Standardized service response"""
    
    def __init__(self, request_id: str, result: Any, error: Optional[str] = None):
        self.request_id = request_id
        self.result = result
        self.error = error
        self.timestamp = time.time()

class RobotService:
    """Base class for robot services"""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.handlers: Dict[str, Callable] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    def register_handler(self, method: str, handler: Callable) -> None:
        """Register a method handler"""
        self.handlers[method] = handler
    
    async def handle_request(self, request: ServiceRequest) -> ServiceResponse:
        """Handle a service request"""
        if request.method not in self.handlers:
            return ServiceResponse(
                request.id, 
                None, 
                f"Method {request.method} not implemented"
            )
        
        try:
            result = await self._safe_call_handler(request.method, request.params)
            return ServiceResponse(request.id, result)
        except Exception as e:
            return ServiceResponse(request.id, None, str(e))
    
    async def _safe_call_handler(self, method: str, params: Dict[str, Any]) -> Any:
        """Safely call a handler with error handling"""
        handler = self.handlers[method]
        
        if asyncio.iscoroutinefunction(handler):
            return await handler(params)
        else:
            return handler(params)
    
    def get_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "methods": list(self.handlers.keys()),
            "timestamp": time.time()
        }

class LocomotionService(RobotService):
    """Locomotion service implementation"""
    
    def __init__(self):
        super().__init__("locomotion_service", [
            "navigation", "balance_control", "gait_control", "path_planning"
        ])
        
        # Register method handlers
        self.register_handler("move_to", self._handle_move_to)
        self.register_handler("get_position", self._handle_get_position) 
        self.register_handler("set_gait", self._handle_set_gait)
        self.register_handler("stop", self._handle_stop)
    
    async def _handle_move_to(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move_to request"""
        target = params.get("position", [0, 0, 0])
        speed = params.get("speed", 0.5)
        
        # Simulate movement execution
        await asyncio.sleep(0.05)  # Simulate processing time
        
        return {
            "status": "moving",
            "target": target,
            "estimated_time": 2.5,  # seconds
            "path": [[0, 0, 0], target]  # Simplified path
        }
    
    async def _handle_get_position(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_position request"""
        # In a real implementation, this would query actual position
        return {
            "position": [1.5, 2.0, 0.0],
            "orientation": [0, 0, 0, 1],
            "timestamp": time.time()
        }
    
    async def _handle_set_gait(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set_gait request"""
        gait = params.get("gait", "stand")
        # Simulate gait change
        await asyncio.sleep(0.02)
        
        return {
            "status": "success",
            "gait_set": gait,
            "transition_time": 0.3
        }
    
    async def _handle_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stop request"""
        # Simulate stopping
        await asyncio.sleep(0.01)
        
        return {
            "status": "stopped",
            "remaining_velocity": [0, 0, 0]
        }

class PerceptionService(RobotService):
    """Perception service implementation"""
    
    def __init__(self):
        super().__init__("perception_service", [
            "object_detection", "person_detection", "environment_mapping", "sensor_fusion"
        ])
        
        # Register method handlers
        self.register_handler("detect_objects", self._handle_detect_objects)
        self.register_handler("get_environment_map", self._handle_get_environment_map)
        self.register_handler("track_person", self._handle_track_person)
        self.register_handler("sensor_fusion", self._handle_sensor_fusion)
    
    async def _handle_detect_objects(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle object detection request"""
        # Simulate sensor processing
        await asyncio.sleep(0.03)
        
        # Return simulated detections
        return {
            "objects": [
                {"id": 1, "type": "person", "position": [2.5, 1.0, 0.0], "confidence": 0.95},
                {"id": 2, "type": "cup", "position": [1.0, 0.5, 0.8], "confidence": 0.87},
                {"id": 3, "type": "chair", "position": [0.5, 2.0, 0.0], "confidence": 0.92}
            ],
            "timestamp": time.time(),
            "sensor_source": "rgbd_camera"
        }
    
    async def _handle_get_environment_map(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle environment mapping request"""
        # Simulate map generation
        await asyncio.sleep(0.05)
        
        return {
            "map_type": "occupancy_grid",
            "resolution": 0.1,  # meters per cell
            "dimensions": [20, 20],  # 20x20 grid
            "origin": [-5, -5],  # map origin in world coordinates
            "occupied_cells": [[10, 10], [12, 8]],  # Sample occupied cells
            "timestamp": time.time()
        }
    
    async def _handle_track_person(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle person tracking request"""
        person_id = params.get("person_id")
        duration = params.get("duration", 10.0)  # seconds
        
        # Simulate tracking
        await asyncio.sleep(0.02)
        
        return {
            "status": "tracking",
            "person_id": person_id,
            "estimated_duration": duration,
            "tracking_accuracy": 0.94
        }
    
    async def _handle_sensor_fusion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sensor fusion request"""
        sensors = params.get("sensors", [])
        
        # Simulate fusion process
        await asyncio.sleep(0.04)
        
        return {
            "fused_data": {
                "position": [1.2, 0.8, 0.0],
                "orientation": [0.1, 0.05, 0.02, 0.99],
                "confidence": 0.89
            },
            "sensor_contributions": {sensor: 0.7 for sensor in sensors},
            "timestamp": time.time()
        }

class CognitiveService(RobotService):
    """Cognitive service implementation"""
    
    def __init__(self):
        super().__init__("cognitive_service", [
            "natural_language", "task_planning", "decision_making", "dialog_management"
        ])
        
        # Register method handlers
        self.register_handler("process_language", self._handle_process_language)
        self.register_handler("plan_task", self._handle_plan_task)
        self.register_handler("make_decision", self._handle_make_decision)
        self.register_handler("manage_dialog", self._handle_manage_dialog)
    
    async def _handle_process_language(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle natural language processing"""
        text = params.get("text", "")
        
        # Simulate language processing
        await asyncio.sleep(0.06)
        
        # Simple intent classification
        intent = "unknown"
        if "move" in text.lower() or "go" in text.lower():
            intent = "navigation"
        elif "grasp" in text.lower() or "pick" in text.lower():
            intent = "manipulation"
        elif "hello" in text.lower() or "hi" in text.lower():
            intent = "greeting"
        
        return {
            "intent": intent,
            "entities": [],
            "confidence": 0.85,
            "timestamp": time.time()
        }
    
    async def _handle_plan_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task planning"""
        goal = params.get("goal", "")
        
        # Simulate planning
        await asyncio.sleep(0.1)
        
        # Generate a simple plan
        if "navigation" in goal.lower():
            plan = [
                {"action": "locate_target", "parameters": {"target": goal}},
                {"action": "plan_path", "parameters": {"start": [0, 0, 0], "goal": [2, 1, 0]}},
                {"action": "execute_navigation", "parameters": {}}
            ]
        elif "manipulation" in goal.lower():
            plan = [
                {"action": "detect_object", "parameters": {"object_type": goal.split()[-1]}},
                {"action": "calculate_grasp", "parameters": {}},
                {"action": "execute_grasp", "parameters": {}}
            ]
        else:
            plan = [{"action": "analyze_request", "parameters": {"request": goal}}]
        
        return {
            "plan": plan,
            "plan_id": str(uuid.uuid4()),
            "estimated_time": len(plan) * 2.0,  # 2 seconds per action
            "resources_required": ["locomotion", "manipulation"]
        }
    
    async def _handle_make_decision(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle decision making"""
        situation = params.get("situation", {})
        
        # Simulate decision process
        await asyncio.sleep(0.05)
        
        # Simple decision logic
        if situation.get("object_detected") and situation.get("user_proximity", 0) < 2.0:
            decision = "greet_user"
        elif situation.get("battery_level", 1.0) < 0.2:
            decision = "return_to_charger"
        else:
            decision = "continue_current_task"
        
        return {
            "decision": decision,
            "confidence": 0.92,
            "rationale": "Based on current context and priorities",
            "timestamp": time.time()
        }
    
    async def _handle_manage_dialog(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dialog management"""
        user_input = params.get("user_input", "")
        
        # Simulate dialog processing
        await asyncio.sleep(0.04)
        
        # Generate response based on input
        if "hello" in user_input.lower():
            response = "Hello! How can I assist you today?"
        elif "help" in user_input.lower():
            response = "I can help with navigation, object manipulation, and information. What do you need?"
        else:
            response = f"I received your message: '{user_input}'. How can I help?"
        
        return {
            "response": response,
            "next_expected_input": "response_awaited",
            "engagement_level": 0.8,
            "timestamp": time.time()
        }

class ServiceOrchestrator:
    """Orchestrates service-based composition"""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self.services: Dict[str, RobotService] = {}
        self.message_queue = asyncio.Queue()
    
    async def start_service(self, service: RobotService) -> bool:
        """Start and register a service"""
        success = await self.registry.register_service(
            service.name, service.get_info()
        )
        
        if success:
            self.services[service.name] = service
            print(f"Service started: {service.name}")
        
        return success
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop and unregister a service"""
        success = await self.registry.unregister_service(service_name)
        
        if success and service_name in self.services:
            del self.services[service_name]
            print(f"Service stopped: {service_name}")
        
        return success
    
    async def call_service(self, service_name: str, method: str, 
                          params: Dict[str, Any]) -> ServiceResponse:
        """Call a specific service"""
        if service_name not in self.services:
            return ServiceResponse(
                str(uuid.uuid4()), 
                None, 
                f"Service {service_name} not found"
            )
        
        service = self.services[service_name]
        request = ServiceRequest(method, params, service_name)
        
        return await service.handle_request(request)
    
    async def find_and_call(self, capability: str, method: str, 
                           params: Dict[str, Any]) -> Optional[ServiceResponse]:
        """Find a service with the required capability and call it"""
        service_name = await self.registry.find_service(capability)
        
        if service_name is None:
            return ServiceResponse(
                str(uuid.uuid4()), 
                None, 
                f"No service found with capability: {capability}"
            )
        
        return await self.call_service(service_name, method, params)
    
    async def compose_complex_task(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Compose and execute a complex task using multiple services"""
        task_results = {}
        
        # Example task: Navigate to object, grasp it, and place it somewhere
        steps = task_spec.get("steps", [])
        
        for step in steps:
            step_type = step["type"]
            params = step["params"]
            
            if step_type == "navigation":
                result = await self.find_and_call("navigation", "move_to", params)
                task_results[f"nav_{step.get('id', len(task_results))}"] = result
                
            elif step_type == "detection":
                result = await self.find_and_call("object_detection", "detect_objects", params)
                task_results[f"detect_{step.get('id', len(task_results))}"] = result
                
            elif step_type == "manipulation":
                # This would call manipulation service in a real implementation
                task_results[f"manip_{step.get('id', len(task_results))}"] = {
                    "status": "simulated", "action": "grasp_object", "params": params
                }
            
            # Add small delay between steps
            await asyncio.sleep(0.1)
        
        return {
            "task_id": str(uuid.uuid4()),
            "results": task_results,
            "status": "completed",
            "timestamp": time.time()
        }

# Example usage
async def demo_service_oriented_architecture():
    """Demonstrate service-oriented architecture"""
    print("=== Service-Oriented Architecture Demo ===\n")
    
    orchestrator = ServiceOrchestrator()
    
    # Start services
    locomotion_service = LocomotionService()
    perception_service = PerceptionService()
    cognitive_service = CognitiveService()
    
    await orchestrator.start_service(locomotion_service)
    await orchestrator.start_service(perception_service)
    await orchestrator.start_service(cognitive_service)
    
    print("\nTesting individual services:")
    
    # Test locomotion service
    nav_result = await orchestrator.call_service(
        "locomotion_service", 
        "move_to", 
        {"position": [2.0, 1.0, 0.0], "speed": 0.8}
    )
    print(f"Navigation result: {nav_result.result}")
    
    # Test perception service
    detection_result = await orchestrator.find_and_call(
        "object_detection",
        "detect_objects",
        {}
    )
    print(f"Detection result: {detection_result.result}")
    
    # Test cognitive service
    language_result = await orchestrator.call_service(
        "cognitive_service",
        "process_language",
        {"text": "Please go to the kitchen and bring me a cup"}
    )
    print(f"Language processing result: {language_result.result}")
    
    print("\nExecuting complex task:")
    # Compose a complex task
    complex_task = {
        "steps": [
            {"type": "detection", "params": {}, "id": "detect_objects"},
            {"type": "navigation", "params": {"position": [3.0, 2.0, 0.0]}, "id": "navigate_to_kitchen"},
            {"type": "detection", "params": {}, "id": "detect_cup"}
        ]
    }
    
    task_result = await orchestrator.compose_complex_task(complex_task)
    print(f"Complex task completed: {task_result['status']}")
    print(f"Task steps executed: {len(task_result['results'])}")
    
    # Stop services
    await orchestrator.stop_service("locomotion_service")
    await orchestrator.stop_service("perception_service")
    await orchestrator.stop_service("cognitive_service")

# Run the demo
asyncio.run(demo_service_oriented_architecture())
```

### Event-Driven Architecture

Event-driven architecture enables asynchronous composition and loose coupling:

```python
import asyncio
import json
from typing import Dict, List, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    """Types of events in the system"""
    SENSOR_DATA = "sensor_data"
    ACTUATOR_COMMAND = "actuator_command"
    SYSTEM_STATE_CHANGE = "system_state_change"
    USER_COMMAND = "user_command"
    TASK_STATUS = "task_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class Event:
    """Standard event structure"""
    event_type: EventType
    source: str
    destination: str  # "broadcast" for all interested parties
    data: Dict[str, Any]
    timestamp: float
    correlation_id: str = None
    priority: int = 1  # Higher number = higher priority

class EventBus:
    """Central event bus for the system"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_queue = asyncio.Queue()
        self.running = False
        self.event_processors = []
    
    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe a handler to specific event types"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        if handler not in self.subscribers[event_type]:
            self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: Callable) -> bool:
        """Unsubscribe a handler from events"""
        if event_type in self.subscribers and handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            return True
        return False
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus"""
        await self.event_queue.put(event)
    
    async def start_processing(self) -> None:
        """Start processing events in the background"""
        self.running = True
        
        # Start event processing tasks
        for i in range(3):  # 3 parallel processors
            processor_task = asyncio.create_task(self._event_processor(i))
            self.event_processors.append(processor_task)
    
    async def stop_processing(self) -> None:
        """Stop processing events"""
        self.running = False
        
        # Cancel all processor tasks
        for processor in self.event_processors:
            processor.cancel()
        
        # Wait for tasks to complete
        for processor in self.event_processors:
            try:
                await processor
            except asyncio.CancelledError:
                pass  # Expected after cancellation
    
    async def _event_processor(self, processor_id: int) -> None:
        """Process events from the queue"""
        while self.running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Find subscribers for this event type
                subscribers = self.subscribers.get(event.event_type, [])
                
                # Process event with all subscribers
                for handler in subscribers:
                    try:
                        # Create a task to avoid blocking other handlers
                        await handler(event)
                    except Exception as e:
                        print(f"Error in event handler {handler.__name__}: {e}")
                
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                # No events in queue, continue loop
                continue
            except Exception as e:
                print(f"Error in event processor {processor_id}: {e}")

class EventDrivenComponent:
    """Base class for event-driven components"""
    
    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self.subscribed_events: Set[EventType] = set()
    
    async def start(self) -> bool:
        """Initialize the component"""
        await self._setup_event_handlers()
        return True
    
    async def stop(self) -> None:
        """Clean up the component"""
        for event_type in self.subscribed_events:
            self.event_bus.unsubscribe(event_type, self.handle_event)
    
    async def _setup_event_handlers(self) -> None:
        """Setup event handlers - to be implemented by subclasses"""
        pass
    
    async def handle_event(self, event: Event) -> None:
        """Handle an incoming event - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def publish_event(self, event_type: EventType, data: Dict[str, Any], 
                           destination: str = "broadcast") -> None:
        """Publish an event to the bus"""
        event = Event(
            event_type=event_type,
            source=self.name,
            destination=destination,
            data=data,
            timestamp=time.time(),
            priority=1
        )
        await self.event_bus.publish(event)

class LocomotionComponentED(EventDrivenComponent):
    """Locomotion component using event-driven architecture"""
    
    def __init__(self, name: str, event_bus: EventBus):
        super().__init__(name, event_bus)
        self.current_position = [0, 0, 0]
        self.current_gait = "stand"
        self.is_moving = False
    
    async def _setup_event_handlers(self) -> None:
        """Setup event handlers for locomotion component"""
        self.event_bus.subscribe(EventType.USER_COMMAND, self.handle_event)
        self.event_bus.subscribe(EventType.TASK_STATUS, self.handle_event)
        self.subscribed_events.update([EventType.USER_COMMAND, EventType.TASK_STATUS])
    
    async def handle_event(self, event: Event) -> None:
        """Handle incoming events for locomotion"""
        if event.event_type == EventType.USER_COMMAND:
            if event.data.get("command") == "move_to":
                await self._execute_movement(event.data)
        
        elif event.event_type == EventType.TASK_STATUS:
            if event.data.get("task") == "navigation_complete":
                await self._handle_navigation_complete(event.data)
    
    async def _execute_movement(self, data: Dict[str, Any]) -> None:
        """Execute movement command"""
        target = data.get("position", [0, 0, 0])
        speed = data.get("speed", 0.5)
        
        print(f"{self.name} moving to {target} at speed {speed}")
        
        # Simulate movement
        await asyncio.sleep(0.5)
        
        # Update position
        self.current_position = target
        
        # Publish completion event
        await self.publish_event(
            EventType.TASK_STATUS,
            {
                "task": "navigation",
                "status": "completed",
                "final_position": target
            }
        )
    
    async def _handle_navigation_complete(self, data: Dict[str, Any]) -> None:
        """Handle navigation completion events"""
        print(f"{self.name} received navigation completion: {data}")

class PerceptionComponentED(EventDrivenComponent):
    """Perception component using event-driven architecture"""
    
    def __init__(self, name: str, event_bus: EventBus):
        super().__init__(name, event_bus)
        self.detected_objects = []
        self.is_processing = False
    
    async def _setup_event_handlers(self) -> None:
        """Setup event handlers for perception component"""
        self.event_bus.subscribe(EventType.SENSOR_DATA, self.handle_event)
        self.event_bus.subscribe(EventType.USER_COMMAND, self.handle_event)
        self.subscribed_events.update([EventType.SENSOR_DATA, EventType.USER_COMMAND])
    
    async def handle_event(self, event: Event) -> None:
        """Handle incoming events for perception"""
        if event.event_type == EventType.SENSOR_DATA:
            if event.data.get("sensor_type") == "camera":
                await self._process_camera_data(event.data)
        
        elif event.event_type == EventType.USER_COMMAND:
            if event.data.get("command") == "detect_objects":
                await self._initiate_object_detection()
    
    async def _process_camera_data(self, data: Dict[str, Any]) -> None:
        """Process incoming camera data"""
        if self.is_processing:
            return  # Skip if already processing
        
        self.is_processing = True
        print(f"{self.name} processing camera data...")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate simulated detections
        detections = [
            {"id": 1, "type": "person", "position": [1.5, 2.0, 0.0], "confidence": 0.92}
        ]
        
        self.detected_objects.extend(detections)
        
        # Publish detection results
        await self.publish_event(
            EventType.SENSOR_DATA,
            {
                "sensor_type": "object_detection",
                "detections": detections,
                "timestamp": time.time()
            }
        )
        
        self.is_processing = False
    
    async def _initiate_object_detection(self) -> None:
        """Initiate object detection"""
        print(f"{self.name} initiating object detection...")
        
        # Publish request for sensor data
        await self.publish_event(
            EventType.USER_COMMAND,
            {
                "command": "request_sensor_data",
                "sensor_type": "camera"
            }
        )

class TaskManagerComponentED(EventDrivenComponent):
    """Task manager component using event-driven architecture"""
    
    def __init__(self, name: str, event_bus: EventBus):
        super().__init__(name, event_bus)
        self.active_tasks = {}
        self.task_queue = asyncio.Queue()
    
    async def _setup_event_handlers(self) -> None:
        """Setup event handlers for task manager"""
        self.event_bus.subscribe(EventType.USER_COMMAND, self.handle_event)
        self.event_bus.subscribe(EventType.TASK_STATUS, self.handle_event)
        self.subscribed_events.update([EventType.USER_COMMAND, EventType.TASK_STATUS])
    
    async def handle_event(self, event: Event) -> None:
        """Handle incoming events for task manager"""
        if event.event_type == EventType.USER_COMMAND:
            if event.data.get("command") == "execute_task":
                await self._queue_task(event.data)
        
        elif event.event_type == EventType.TASK_STATUS:
            await self._handle_task_status(event.data)
    
    async def _queue_task(self, data: Dict[str, Any]) -> None:
        """Queue a new task for execution"""
        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "description": data.get("description", "Unknown task"),
            "components_needed": data.get("components", []),
            "status": "queued",
            "timestamp": time.time()
        }
        
        self.active_tasks[task_id] = task_data
        
        # Publish task started event
        await self.publish_event(
            EventType.TASK_STATUS,
            {
                "task_id": task_id,
                "status": "started",
                "description": task_data["description"]
            }
        )
        
        print(f"{self.name} queued task {task_id}: {task_data['description']}")
    
    async def _handle_task_status(self, data: Dict[str, Any]) -> None:
        """Handle task status updates"""
        task_id = data.get("task_id")
        status = data.get("status")
        
        if task_id and task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = status
            print(f"{self.name} updated task {task_id} to status: {status}")

class EventDrivenSystem:
    """Main system using event-driven architecture"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.components: Dict[str, EventDrivenComponent] = {}
    
    async def add_component(self, name: str, component: EventDrivenComponent) -> bool:
        """Add a component to the system"""
        self.components[name] = component
        return await component.start()
    
    async def start_system(self) -> bool:
        """Start the event-driven system"""
        await self.event_bus.start_processing()
        
        # Start all components
        start_success = True
        for name, component in self.components.items():
            try:
                success = await component.start()
                if not success:
                    print(f"Failed to start component: {name}")
                    start_success = False
            except Exception as e:
                print(f"Error starting component {name}: {e}")
                start_success = False
        
        return start_success
    
    async def stop_system(self) -> None:
        """Stop the event-driven system"""
        # Stop all components
        for name, component in self.components.items():
            try:
                await component.stop()
            except Exception as e:
                print(f"Error stopping component {name}: {e}")
        
        # Stop event processing
        await self.event_bus.stop_processing()
    
    async def execute_demo_sequence(self) -> None:
        """Execute a demonstration sequence of events"""
        print("Starting demonstration sequence...")
        
        # Publish a user command to move to a position
        await self.event_bus.publish(Event(
            event_type=EventType.USER_COMMAND,
            source="user",
            destination="broadcast",
            data={
                "command": "move_to",
                "position": [2.5, 1.0, 0.0],
                "speed": 0.8
            },
            timestamp=time.time()
        ))
        
        # Wait for movement to complete
        await asyncio.sleep(1.0)
        
        # Request object detection
        await self.event_bus.publish(Event(
            event_type=EventType.USER_COMMAND,
            source="user",
            destination="broadcast",
            data={
                "command": "detect_objects"
            },
            timestamp=time.time()
        ))
        
        # Wait for detection
        await asyncio.sleep(0.5)
        
        # Execute a task
        await self.event_bus.publish(Event(
            event_type=EventType.USER_COMMAND,
            source="user",
            destination="broadcast",
            data={
                "command": "execute_task",
                "description": "Navigate to kitchen and detect objects",
                "components": ["locomotion", "perception"]
            },
            timestamp=time.time()
        ))
        
        # Wait for task completion
        await asyncio.sleep(1.5)

# Example usage
async def demo_event_driven_architecture():
    """Demonstrate event-driven architecture"""
    print("=== Event-Driven Architecture Demo ===\n")
    
    # Create system and components
    system = EventDrivenSystem()
    
    # Create components
    locomotion_comp = LocomotionComponentED("locomotion_manager", system.event_bus)
    perception_comp = PerceptionComponentED("perception_manager", system.event_bus)
    task_comp = TaskManagerComponentED("task_manager", system.event_bus)
    
    # Add components to system
    await system.add_component("locomotion", locomotion_comp)
    await system.add_component("perception", perception_comp)
    await system.add_component("task_manager", task_comp)
    
    # Start the system
    if await system.start_system():
        print("System started successfully\n")
        
        # Execute demonstration sequence
        await system.execute_demo_sequence()
        
        print("\nDemonstration completed")
        
        # Stop the system
        await system.stop_system()
        print("System stopped")
    else:
        print("Failed to start system")

# Run the demo
asyncio.run(demo_event_driven_architecture())
```

## Configuration and Deployment

### Component Configuration Management

Proper configuration management is essential for system composition:

```python
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import importlib

class ComponentConfig:
    """Manages configuration for robot components"""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.data = config_data
        self.component_configs = {}
        self.system_config = {}
        self._parse_config()
    
    def _parse_config(self) -> None:
        """Parse configuration data into structured format"""
        self.system_config = self.data.get("system", {})
        
        components = self.data.get("components", {})
        for name, config in components.items():
            self.component_configs[name] = {
                "type": config.get("type"),
                "class_path": config.get("class"),
                "parameters": config.get("parameters", {}),
                "dependencies": config.get("dependencies", []),
                "connections": config.get("connections", []),
                "enabled": config.get("enabled", True)
            }
    
    def get_component_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific component"""
        return self.component_configs.get(name)
    
    def get_system_parameter(self, key: str, default: Any = None) -> Any:
        """Get a system-level parameter"""
        return self.system_config.get(key, default)
    
    def get_all_component_names(self) -> List[str]:
        """Get names of all configured components"""
        return list(self.component_configs.keys())

class ConfigurationLoader:
    """Loads and validates system configurations"""
    
    @staticmethod
    def load_from_file(file_path: str) -> ComponentConfig:
        """Load configuration from a file"""
        path = Path(file_path)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return ComponentConfig(config_data)
    
    @staticmethod
    def validate_config(config: ComponentConfig) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check required system parameters
        required_system_params = ["name", "version", "robot_type"]
        for param in required_system_params:
            if param not in config.system_config:
                errors.append(f"Missing required system parameter: {param}")
        
        # Check component configurations
        for name, comp_config in config.component_configs.items():
            if not comp_config.get("type"):
                errors.append(f"Component {name} missing type")
            
            if not comp_config.get("class_path"):
                errors.append(f"Component {name} missing class_path")
        
        return errors
    
    @staticmethod
    def resolve_dependencies(config: ComponentConfig) -> Dict[str, List[str]]:
        """Resolve and validate component dependencies"""
        dependencies = {}
        
        for name, comp_config in config.component_configs.items():
            deps = comp_config.get("dependencies", [])
            dependencies[name] = deps
        
        # Check for circular dependencies
        if ConfigurationLoader._has_circular_dependencies(dependencies):
            raise ValueError("Circular dependencies detected in component configuration")
        
        return dependencies
    
    @staticmethod
    def _has_circular_dependencies(dependencies: Dict[str, List[str]]) -> bool:
        """Check for circular dependencies in the dependency graph"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for dep in dependencies.get(node, []):
                if has_cycle(dep):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in dependencies:
            if has_cycle(node):
                return True
        
        return False

class ComponentFactory:
    """Factory for creating configured components"""
    
    def __init__(self):
        self.component_classes = {}
    
    def register_component_class(self, class_name: str, class_obj: type) -> None:
        """Register a component class"""
        self.component_classes[class_name] = class_obj
    
    def create_component(self, config: Dict[str, Any]) -> Any:
        """Create a component from configuration"""
        class_path = config.get("class_path", "")
        
        # If we have the class registered, use it
        if class_path in self.component_classes:
            component_class = self.component_classes[class_path]
        else:
            # Otherwise, dynamically import the class
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                component_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not create component from class path {class_path}: {e}")
        
        # Create component with parameters
        parameters = config.get("parameters", {})
        return component_class(**parameters)

class SystemComposer:
    """Composes systems from configurations"""
    
    def __init__(self):
        self.factory = ComponentFactory()
        self.components = {}
        self.config = None
    
    def load_configuration(self, config_path: str) -> bool:
        """Load system configuration"""
        try:
            self.config = ConfigurationLoader.load_from_file(config_path)
            
            # Validate configuration
            errors = ConfigurationLoader.validate_config(self.config)
            if errors:
                print("Configuration validation errors:")
                for error in errors:
                    print(f"  - {error}")
                return False
            
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def register_standard_components(self) -> None:
        """Register standard component classes"""
        # Register standard humanoid components
        self.factory.register_component_class("LocomotionComponent", LocomotionComponent)
        self.factory.register_component_class("PerceptionComponent", PerceptionComponent)
        self.factory.register_component_class("ManipulationComponent", ManipulationComponent)
        self.factory.register_component_class("CognitiveComponent", CognitiveComponent)
    
    def compose_system(self) -> bool:
        """Compose the system from configuration"""
        if not self.config:
            print("No configuration loaded")
            return False
        
        # Resolve dependencies
        try:
            dependencies = ConfigurationLoader.resolve_dependencies(self.config)
        except ValueError as e:
            print(f"Dependency resolution error: {e}")
            return False
        
        # Create components in dependency order
        creation_order = self._get_creation_order(dependencies)
        
        for component_name in creation_order:
            comp_config = self.config.get_component_config(component_name)
            if not comp_config or not comp_config.get("enabled", True):
                continue
            
            try:
                component = self.factory.create_component(comp_config)
                self.components[component_name] = component
                print(f"Created component: {component_name}")
            except Exception as e:
                print(f"Error creating component {component_name}: {e}")
                return False
        
        return True
    
    def _get_creation_order(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """Get order to create components based on dependencies"""
        # Topological sort to determine creation order
        in_degree = {node: 0 for node in dependencies}
        for node, deps in dependencies.items():
            for dep in deps:
                if dep in in_degree:  # Only count registered dependencies
                    in_degree[node] += 1
        
        queue = [node for node, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            # Reduce in-degree of dependent nodes
            for other_node, deps in dependencies.items():
                if node in deps:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)
        
        return order
    
    def connect_components(self) -> bool:
        """Connect components based on configuration"""
        if not self.config:
            return False
        
        success = True
        for name, comp_config in self.config.component_configs.items():
            connections = comp_config.get("connections", [])
            
            for connection in connections:
                source = connection.get("source", name)
                destination = connection.get("destination")
                interface = connection.get("interface", "default")
                
                if source in self.components and destination in self.components:
                    # In a real system, this would establish actual connections
                    print(f"Connected {source} -> {destination} via {interface}")
                else:
                    print(f"Warning: Could not connect {source} -> {destination}, component not found")
                    success = False
        
        return success
    
    def get_component(self, name: str) -> Any:
        """Get a specific component"""
        return self.components.get(name)

# Example configuration files would look like this:
# config.yaml
example_config_yaml = """
system:
  name: "humanoid_robot"
  version: "1.0.0"
  robot_type: "humanoid"
  manufacturer: "Example Robotics"

components:
  locomotion:
    type: "motion"
    class: "LocomotionComponent"
    enabled: true
    parameters:
      max_velocity: 1.0
      step_height: 0.05
    dependencies: []
    connections: []
  
  perception:
    type: "sensor"
    class: "PerceptionComponent"
    enabled: true
    parameters:
      detection_range: 3.0
      confidence_threshold: 0.7
    dependencies: []
    connections: []
  
  manipulation:
    type: "actuator"
    class: "ManipulationComponent"
    enabled: true
    parameters:
      max_gripper_force: 50.0
      precision_mode: true
    dependencies: []
    connections: []
  
  cognitive:
    type: "intelligence"
    class: "CognitiveComponent"
    enabled: true
    parameters:
      reasoning_depth: 3
      memory_size: 1000
    dependencies: ["perception", "locomotion", "manipulation"]
    connections:
      - source: "cognitive"
        destination: "locomotion"
        interface: "command"
      - source: "cognitive"
        destination: "manipulation" 
        interface: "command"
"""

# Write example configuration to file
with open("example_config.yaml", "w") as f:
    f.write(example_config_yaml)

def demo_configuration_management():
    """Demonstrate configuration management"""
    print("=== Configuration Management Demo ===\n")
    
    composer = SystemComposer()
    
    # Register standard components
    composer.register_standard_components()
    
    # Load configuration
    if composer.load_configuration("example_config.yaml"):
        print("Configuration loaded successfully\n")
        
        # Compose the system
        if composer.compose_system():
            print("System composed successfully\n")
            
            # Connect components
            if composer.connect_components():
                print("Components connected successfully\n")
                
                # Verify system composition
                print("System components:")
                for name in composer.config.get_all_component_names():
                    comp = composer.get_component(name)
                    if comp:
                        print(f"  - {name}: {type(comp).__name__}")
                    else:
                        print(f"  - {name}: NOT CREATED (disabled or error)")
            else:
                print("Failed to connect components")
        else:
            print("Failed to compose system")
    else:
        print("Failed to load configuration")
    
    # Clean up
    import os
    os.remove("example_config.yaml")

demo_configuration_management()
```

## Exercises

1. **Component Design Exercise**: Design and implement modular components for a humanoid robot that can be composed in different configurations for various tasks.

2. **Service Integration Exercise**: Create a service-oriented architecture with multiple robot services and implement service discovery and communication.

3. **Event-Driven System Exercise**: Build an event-driven system for a humanoid robot that handles sensor data, user commands, and task execution asynchronously.

4. **Configuration Management Exercise**: Develop a configuration system that allows different robot capabilities to be loaded and configured at runtime.

5. **Integration Pattern Exercise**: Implement multiple architectural patterns (component-based, service-oriented, event-driven) and compare their effectiveness for different use cases.

6. **System Scalability Exercise**: Design an integration architecture that can scale from a single robot to a multi-robot system.

7. **Real-time Constraints Exercise**: Implement an integration that meets specific real-time performance requirements for critical robot functions.

## Summary

System composition in humanoid robotics requires careful consideration of architectural patterns, component design, and integration methodologies. Successful composition depends on well-defined interfaces, standardized communication protocols, and robust configuration management. The choice of architectural pattern—whether component-based, service-oriented, or event-driven—should align with the specific requirements of the robot application. Proper testing, validation, and performance optimization are essential to ensure that composed systems meet the real-time and safety requirements of humanoid robotics applications.