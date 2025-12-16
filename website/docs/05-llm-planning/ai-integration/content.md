---
title: LLM-Driven Planning for Humanoid Robotics
sidebar_position: 1
---

# LLM-Driven Planning for Humanoid Robotics

## Introduction

Large Language Models (LLMs) have emerged as powerful tools for robotics, particularly for humanoid robots that must operate in human-centric environments. LLMs excel at understanding natural language, reasoning about complex scenarios, and generating structured plans. This chapter explores how to integrate LLMs into humanoid robot planning systems to enable more natural, flexible, and intelligent behavior.

## The Role of LLMs in Robotics Planning

### Traditional vs. LLM-Driven Planning

Traditional robotics planning approaches rely on:
- Predefined task structures
- Hand-coded state machines
- Symbolic reasoning with limited context
- Domain-specific algorithms

In contrast, LLM-driven planning offers:
- Natural language interaction
- Commonsense reasoning
- Adaptability to novel situations
- Integration of world knowledge

### Advantages of LLM Integration

1. **Natural Language Understanding**: LLMs can interpret complex, ambiguous, or context-dependent commands
2. **World Knowledge**: LLMs contain extensive knowledge about objects, actions, and typical sequences of behavior
3. **Flexible Reasoning**: LLMs can reason through new scenarios using analogy and generalization
4. **Context Awareness**: LLMs can consider context and background knowledge in planning
5. **Human-Robot Communication**: LLMs enable natural dialogue for clarifying ambiguous instructions

### Limitations and Challenges

1. **Hallucination**: LLMs can generate factually incorrect information
2. **Temporal Inconsistency**: LLMs may forget previous statements in long interactions
3. **Lack of Grounding**: LLMs may generate plans not grounded in robot capabilities
4. **Real-time Performance**: LLM inference can be slow for real-time applications
5. **Safety Verification**: Ensuring LLM-generated plans are safe and appropriate

## Architecture for LLM-Integrated Planning

### Hybrid Planning Architecture

For robust humanoid robot operation, LLMs should be combined with traditional planning systems:

```
Human Command (Natural Language)
         ↓
[LLM-Based Task Planner] 
         ↓
[Task Decomposition & Sequencing]
         ↓
[Traditional Motion Planner]
         ↓
[Low-Level Controllers]
         ↓
[Robot Execution]
```

### System Components

1. **Language Interface**: Converts natural language to structured representations
2. **World Model**: Maintains current state and context for the LLM
3. **LLM Planner**: Generates high-level plans and decomposes tasks
4. **Action Executor**: Translates LLM outputs into robotic actions
5. **Feedback Loop**: Updates world model based on execution results

### Example Architecture Implementation

```python
import openai
import json
from typing import Dict, List, Any, Optional
import time

class LLMPlanningSystem:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.world_model = WorldModel()
        self.robot_interface = RobotInterface()
        self.task_decomposer = TaskDecomposer()
        
    def plan_task(self, natural_language_command: str) -> List[Dict]:
        """
        Generate a plan for a natural language command
        """
        # Get current world state
        world_state = self.world_model.get_state()
        
        # Create prompt for the LLM
        prompt = self._create_planning_prompt(natural_language_command, world_state)
        
        # Get plan from LLM
        response = self._query_llm(prompt)
        
        # Parse and validate the plan
        plan = self._parse_plan(response)
        validated_plan = self._validate_plan(plan)
        
        return validated_plan
    
    def _create_planning_prompt(self, command: str, world_state: Dict) -> str:
        """
        Create a prompt for the LLM with context
        """
        prompt = f"""
        You are an intelligent planning system for a humanoid robot. Your task is to create a detailed action plan to execute the given command.

        Current world state:
        {json.dumps(world_state, indent=2)}

        Robot capabilities:
        - Can move to locations in the environment
        - Can manipulate objects (grasp, release, move)
        - Can identify objects using vision
        - Can navigate around obstacles
        - Can open/close doors
        - Can interact with switches and controls

        Command: {command}

        Please respond with a detailed action plan in JSON format:
        {{
            "actions": [
                {{
                    "action": "action_type",
                    "parameters": {{"param1": "value1", "param2": "value2"}},
                    "description": "Brief description of the action"
                }}
            ],
            "reasoning": "Brief explanation of the planning approach"
        }}

        Actions should be atomic and feasible for a humanoid robot. Only include actions that match the robot's capabilities.
        """
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """
        Query the LLM with the provided prompt
        """
        # In a real implementation, this would call the LLM API
        # For this example, we'll simulate the response
        print(f"Querying LLM with prompt: {prompt[:100]}...")
        time.sleep(0.1)  # Simulate API call delay
        
        # Simulated response (in practice, this would come from the LLM)
        # This is just an example - real implementation would call the API
        return '''
        {
            "actions": [
                {
                    "action": "locate_object",
                    "parameters": {"object": "red cup"},
                    "description": "Look for the red cup on the table"
                },
                {
                    "action": "navigate_to",
                    "parameters": {"location": "table"},
                    "description": "Move to the table where the cup is located"
                },
                {
                    "action": "grasp_object", 
                    "parameters": {"object": "red cup", "position": "top"},
                    "description": "Pick up the red cup from the table"
                },
                {
                    "action": "navigate_to",
                    "parameters": {"location": "kitchen_counter"},
                    "description": "Move to the kitchen counter"
                },
                {
                    "action": "place_object",
                    "parameters": {"object": "red cup", "location": "kitchen_counter"},
                    "description": "Place the red cup on the kitchen counter"
                }
            ],
            "reasoning": "The command is to move the red cup from the table to the kitchen counter. The plan involves locating the object, navigating to it, grasping it, navigating to the destination, and placing it down."
        }
        '''
    
    def _parse_plan(self, response: str) -> List[Dict]:
        """
        Parse the JSON response from the LLM into an action plan
        """
        try:
            parsed = json.loads(response)
            return parsed.get("actions", [])
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {response}")
            return []
    
    def _validate_plan(self, plan: List[Dict]) -> List[Dict]:
        """
        Validate the plan against robot capabilities and world constraints
        """
        validated_plan = []
        
        for action in plan:
            if self._is_action_valid(action):
                validated_plan.append(action)
            else:
                print(f"Invalid action filtered out: {action}")
        
        return validated_plan
    
    def _is_action_valid(self, action: Dict) -> bool:
        """
        Check if an action is valid for this robot in the current context
        """
        action_type = action.get("action")
        
        # Check if action type is supported by the robot
        supported_actions = [
            "navigate_to", "grasp_object", "place_object", 
            "locate_object", "open_door", "close_door", 
            "turn_on", "turn_off"
        ]
        
        if action_type not in supported_actions:
            return False
        
        # Check if parameters are valid
        params = action.get("parameters", {})
        
        if action_type == "grasp_object":
            if "object" not in params:
                return False
        
        if action_type == "navigate_to":
            if "location" not in params:
                return False
                
        return True

class WorldModel:
    """
    Maintains current knowledge about the world state
    """
    def __init__(self):
        self.objects = {}
        self.locations = {}
        self.robot_state = {}
        self.update_timestamp = time.time()
    
    def get_state(self) -> Dict:
        """
        Get current world state as a dictionary
        """
        return {
            "objects": self.objects,
            "locations": self.locations,
            "robot_state": self.robot_state,
            "timestamp": self.update_timestamp
        }
    
    def update_object_location(self, obj_name: str, location: str):
        """
        Update the location of an object
        """
        self.objects[obj_name] = location
        self.update_timestamp = time.time()

class RobotInterface:
    """
    Interface to the actual robot or simulation
    """
    def execute_action(self, action: Dict) -> bool:
        """
        Execute an action on the robot
        Returns True if successful, False otherwise
        """
        action_type = action.get("action")
        params = action.get("parameters", {})
        
        print(f"Executing action: {action_type} with params: {params}")
        
        # In a real implementation, this would interface with the robot
        # For this example, we'll just simulate execution
        time.sleep(0.1)  # Simulate execution time
        return True

class TaskDecomposer:
    """
    Decomposes high-level LLM plans into executable robot actions
    """
    def decompose_task(self, llm_plan: List[Dict]) -> List[Dict]:
        """
        Convert LLM-generated actions to robot-executable actions
        """
        robot_actions = []
        
        for llm_action in llm_plan:
            action_type = llm_action.get("action")
            params = llm_action.get("parameters", {})
            
            if action_type == "navigate_to":
                robot_actions.extend(self._decompose_navigation(params))
            elif action_type == "grasp_object":
                robot_actions.extend(self._decompose_grasping(params))
            elif action_type == "locate_object":
                robot_actions.extend(self._decompose_locating(params))
            else:
                # For other actions, pass through directly
                robot_actions.append(llm_action)
        
        return robot_actions
    
    def _decompose_navigation(self, params: Dict) -> List[Dict]:
        """
        Decompose navigation action into specific robot commands
        """
        # This is a simplified example - in reality would involve path planning
        return [
            {
                "action": "path_plan_to",
                "parameters": {"destination": params.get("location")},
                "description": f"Plan path to {params.get('location')}"
            },
            {
                "action": "move_base",
                "parameters": {"path": "computed_path"},
                "description": f"Execute movement to {params.get('location')}"
            }
        ]
    
    def _decompose_grasping(self, params: Dict) -> List[Dict]:
        """
        Decompose grasping action into specific robot commands
        """
        obj_name = params.get("object")
        approach_pos = params.get("position", "top")
        
        return [
            {
                "action": "approach_object",
                "parameters": {"object": obj_name, "approach_direction": approach_pos},
                "description": f"Approach {obj_name} from {approach_pos}"
            },
            {
                "action": "align_gripper",
                "parameters": {"object": obj_name},
                "description": f"Align gripper to grasp {obj_name}"
            },
            {
                "action": "execute_grasp",
                "parameters": {"object": obj_name},
                "description": f"Grasp {obj_name}"
            },
            {
                "action": "lift_object",
                "parameters": {"object": obj_name},
                "description": f"Lift {obj_name} after grasping"
            }
        ]
    
    def _decompose_locating(self, params: Dict) -> List[Dict]:
        """
        Decompose object locating action into specific robot commands
        """
        obj_name = params.get("object")
        
        return [
            {
                "action": "scan_environment",
                "parameters": {"target_object": obj_name},
                "description": f"Scan environment for {obj_name}"
            },
            {
                "action": "object_detection",
                "parameters": {"target_object": obj_name},
                "description": f"Run object detection for {obj_name}"
            },
            {
                "action": "update_world_model",
                "parameters": {"object": obj_name, "location": "detected_location"},
                "description": f"Update world model with {obj_name} location"
            }
        ]

# Usage example
if __name__ == "__main__":
    # Initialize the LLM planning system
    planner = LLMPlanningSystem()
    
    # Example command
    command = "Take the red cup from the table and place it on the kitchen counter"
    
    # Generate and execute plan
    plan = planner.plan_task(command)
    print("Generated plan:")
    for i, action in enumerate(plan):
        print(f"Step {i+1}: {action}")
```

## Prompt Engineering for Robotics

### Best Practices for Robot-Oriented Prompts

1. **Specify Robot Capabilities**: Clearly define what the robot can and cannot do
2. **Provide Context**: Include information about the current environment
3. **Structure Requests**: Use consistent formats for better parsing
4. **Include Constraints**: Specify safety and operational constraints
5. **Request Validation**: Ask for confidence scores or explanations

### Example Prompt Templates

```python
class RobotPromptTemplates:
    @staticmethod
    def task_decomposition_template(task_description: str, robot_capabilities: List[str], environment_state: Dict) -> str:
        """
        Template for decomposing complex tasks
        """
        return f"""
        You are a task planning system for a humanoid robot. Decompose the following task into a sequence of actions.

        Task: {task_description}

        Robot Capabilities: {', '.join(robot_capabilities)}

        Environment State: {json.dumps(environment_state, indent=2)}

        Constraints:
        - Only use actions the robot is capable of
        - Ensure safety at each step
        - Consider physical limitations

        Provide the plan in JSON format:
        {{
            "actions": [
                {{"action": "...", "parameters": {{"...": "..."}}}}
            ],
            "safety_considerations": ["..."],
            "assumptions": ["..."]
        }}
        """
    
    @staticmethod
    def object_interaction_template(object_name: str, desired_action: str, current_environment: Dict) -> str:
        """
        Template for object interaction planning
        """
        return f"""
        Plan an interaction with the object '{object_name}' to achieve '{desired_action}'.

        Current environment:
        {json.dumps(current_environment, indent=2)}

        Object properties that might be relevant:
        - Manipulable: yes/no
        - Size: small/medium/large
        - Weight: light/medium/heavy
        - Location: relative to robot

        Provide specific action sequence considering:
        1. Robot's current position
        2. Object accessibility
        3. Required approach direction
        4. Grasping strategy

        Respond in this format:
        {{
            "action_sequence": [
                {{"action": "navigate_approach", "details": "..."}},
                {{"action": "grasp", "details": "..."}}
            ],
            "safety_checks": ["..."],
            "expected_outcome": "..."
        }}
        """
    
    @staticmethod
    def navigation_template(destination: str, environment_map: Dict, robot_position: Dict) -> str:
        """
        Template for navigation planning
        """
        return f"""
        Plan a path from current position to '{destination}'.

        Current position: {json.dumps(robot_position, indent=2)}
        Environment map: {json.dumps(environment_map, indent=2)}

        Consider:
        - Obstacle locations
        - Door states (open/closed)
        - Safety requirements
        - Efficiency

        Return:
        {{
            "path": ["waypoint1", "waypoint2", ...],
            "estimated_time": "...",
            "safety_annotations": {{"hazard1": "..."}},
            "alternatives": [["alt_path1"], ["alt_path2"]]
        }}
        """
```

## Grounding LLM Outputs in Physical Reality

### Perception Integration

LLMs must be grounded in real-world perception to be effective:

```python
class LLMPerceptionGrounding:
    def __init__(self, llm_planner: LLMPlanningSystem, perception_system: Any):
        self.llm_planner = llm_planner
        self.perception_system = perception_system
        
    def ground_command(self, command: str) -> str:
        """
        Ground a natural language command in perceived reality
        """
        # Get current scene understanding
        scene_description = self.perception_system.describe_scene()
        
        # Update world model with current perception
        self.llm_planner.world_model.update_scene(scene_description)
        
        # Create grounded command
        grounded_command = f"""
        Command: {command}
        
        Current scene:
        {scene_description}
        
        Please interpret this command in the context of the current environment.
        """
        
        return grounded_command
    
    def verify_plan_feasibility(self, plan: List[Dict]) -> (bool, str):
        """
        Verify if the LLM-generated plan is feasible given current perception
        """
        for action in plan:
            if action["action"] == "grasp_object":
                obj_name = action["parameters"].get("object")
                
                # Check if object exists and is accessible
                detected_objects = self.perception_system.get_detected_objects()
                if obj_name not in detected_objects:
                    return False, f"Object '{obj_name}' not detected in environment"
                
                obj_info = detected_objects[obj_name]
                if not obj_info.get("accessible", True):
                    return False, f"Object '{obj_name}' is not accessible"
        
        return True, "Plan is feasible"
```

### Execution Monitoring and Correction

```python
class ExecutionMonitor:
    def __init__(self, robot_interface: Any, perception_system: Any):
        self.robot_interface = robot_interface
        self.perception_system = perception_system
        self.action_history = []
    
    def monitor_execution(self, plan: List[Dict], llm_planner: LLMPlanningSystem) -> bool:
        """
        Monitor plan execution and handle failures
        """
        for i, action in enumerate(plan):
            print(f"Executing action {i+1}/{len(plan)}: {action}")
            
            # Execute action
            success = self.robot_interface.execute_action(action)
            
            # Record result
            self.action_history.append({
                "action": action,
                "success": success,
                "timestamp": time.time()
            })
            
            if not success:
                print(f"Action failed: {action}")
                
                # Update world model with current state
                current_state = self.perception_system.get_current_state()
                llm_planner.world_model.update_state(current_state)
                
                # Generate recovery plan
                recovery_plan = self.generate_recovery_plan(
                    failed_action=action,
                    current_state=current_state
                )
                
                if recovery_plan:
                    print(f"Executing recovery plan: {recovery_plan}")
                    return self.monitor_execution(recovery_plan, llm_planner)
                else:
                    print("No recovery plan available, stopping execution")
                    return False
        
        return True
    
    def generate_recovery_plan(self, failed_action: Dict, current_state: Dict) -> List[Dict]:
        """
        Generate a recovery plan when an action fails
        """
        # This could call the LLM to generate recovery actions
        # For this example, we'll implement simple recovery strategies
        
        action_type = failed_action.get("action")
        
        if action_type == "navigate_to":
            # Try alternative navigation strategies
            return [
                {
                    "action": "relocalize",
                    "parameters": {},
                    "description": "Re-localize robot in environment"
                },
                {
                    "action": "replan_path",
                    "parameters": failed_action.get("parameters"),
                    "description": "Regenerate navigation plan"
                }
            ]
        
        elif action_type == "grasp_object":
            # Try alternative grasping approaches
            obj_name = failed_action.get("parameters", {}).get("object")
            return [
                {
                    "action": "reposition_object",
                    "parameters": {"object": obj_name},
                    "description": "Adjust object position for easier grasp"
                },
                {
                    "action": "grasp_object",
                    "parameters": {"object": obj_name, "approach": "alternative"},
                    "description": "Try grasping with alternative approach"
                }
            ]
        
        # For other failures, return None (no recovery plan)
        return []
```

## Safety and Validation Considerations

### Safety-Aware Planning

```python
class SafetyValidator:
    def __init__(self):
        self.safety_rules = [
            self._check_physical_limits,
            self._check_collision_risk,
            self._check_object_properties,
            self._check_environment_constraints
        ]
    
    def validate_plan(self, plan: List[Dict], world_model: Dict) -> (bool, List[str]):
        """
        Validate a plan against safety constraints
        """
        issues = []
        
        for rule in self.safety_rules:
            rule_issues = rule(plan, world_model)
            issues.extend(rule_issues)
        
        return len(issues) == 0, issues
    
    def _check_physical_limits(self, plan: List[Dict], world_model: Dict) -> List[str]:
        """
        Check if actions exceed robot's physical capabilities
        """
        issues = []
        
        for action in plan:
            if action.get("action") == "grasp_object":
                obj_weight = action.get("parameters", {}).get("weight", 0.1)  # Default light
                if obj_weight > 5.0:  # Assume max grasp weight is 5kg
                    issues.append(f"Object might be too heavy to grasp safely: {obj_weight}kg")
        
        return issues
    
    def _check_collision_risk(self, plan: List[Dict], world_model: Dict) -> List[str]:
        """
        Check if actions might cause collisions
        """
        issues = []
        
        # Check if navigation paths are clear
        for action in plan:
            if action.get("action") == "navigate_to":
                location = action.get("parameters", {}).get("location")
                # In a real system, this would check the path to the location
                if location in world_model.get("obstacles", []):
                    issues.append(f"Navigation to {location} has obstacles in path")
        
        return issues
```

## Practical Examples and Use Cases

### Example 1: Object Retrieval Task

```python
def object_retrieval_example():
    """
    Example: Human tells robot "Bring me the blue water bottle on the desk"
    """
    # Initialize systems
    llm_planner = LLMPlanningSystem()
    perception_system = ObjectDetectionSystem()  # In practice, this would be a real perception system
    
    # Get current scene perception
    detected_objects = {
        "blue_water_bottle": {"position": [1.0, 2.0, 0.8], "accessible": True, "size": "medium"},
        "red_cup": {"position": [1.0, 2.1, 0.8], "accessible": True, "size": "small"},
        "desk": {"position": [1.0, 2.0, 0.0], "surface": True}
    }
    
    # Update world model with perception
    llm_planner.world_model.objects = detected_objects
    
    # Plan with LLM
    command = "Bring me the blue water bottle on the desk"
    plan = llm_planner.plan_task(command)
    
    print("Object Retrieval Plan:")
    for i, action in enumerate(plan):
        print(f"  {i+1}. {action['description']}")
    
    return plan

# Run example
object_retrieval_plan = object_retrieval_example()
```

### Example 2: Complex Task Execution

```python
def multi_step_task_example():
    """
    Example: Set a simple table for one person
    """
    llm_planner = LLMPlanningSystem()
    
    # Simulate current world state
    llm_planner.world_model.objects = {
        "plate": {"location": "kitchen_counter", "graspable": True},
        "fork": {"location": "kitchen_counter", "graspable": True},
        "knife": {"location": "kitchen_counter", "graspable": True},
        "table": {"location": "dining_area", "surface": True}
    }
    
    llm_planner.world_model.locations = {
        "kitchen_counter": {"objects": ["plate", "fork", "knife"]},
        "dining_area": {"furniture": ["table"]}
    }
    
    command = "Set the table with a plate, fork, and knife"
    plan = llm_planner.plan_task(command)
    
    print("Table Setting Plan:")
    for i, action in enumerate(plan):
        print(f"  {i+1}. {action['description']}")
    
    return plan

# Run example
table_setting_plan = multi_step_task_example()
```

## Integration with Traditional Planning Systems

### Combining Symbolic and Neural Planning

```python
class HybridPlanningSystem:
    def __init__(self):
        self.llm_planner = LLMPlanningSystem()
        self.symbolic_planner = SymbolicTaskPlanner()
        self.motion_planner = MotionPlanner()
        
    def plan_with_hybrid_approach(self, natural_language_goal: str):
        """
        Use LLM for high-level task decomposition and symbolic planner for low-level planning
        """
        # Use LLM to decompose high-level goal
        high_level_plan = self.llm_planner.plan_task(natural_language_goal)
        
        # Convert high-level actions to symbolic representations
        symbolic_goals = []
        for action in high_level_plan:
            if action["action"] in ["navigate_to", "grasp_object", "place_object"]:
                symbolic_goals.append(self._convert_to_symbolic(action))
        
        # Use symbolic planner to generate detailed motion plans
        motion_plans = []
        for goal in symbolic_goals:
            motion_plan = self.motion_planner.plan_to_goal(goal)
            motion_plans.extend(motion_plan)
        
        # Combine high-level and low-level plans
        complete_plan = {
            "high_level": high_level_plan,
            "motion_plans": motion_plans
        }
        
        return complete_plan
    
    def _convert_to_symbolic(self, action: Dict) -> Dict:
        """
        Convert LLM action to symbolic representation for traditional planning
        """
        action_type = action["action"]
        params = action["parameters"]
        
        if action_type == "navigate_to":
            return {
                "type": "navigation",
                "goal_position": self._get_location_coordinates(params["location"]),
                "constraints": ["collision_free", "safe_path"]
            }
        elif action_type == "grasp_object":
            obj_name = params["object"]
            return {
                "type": "manipulation",
                "goal": f"grasp_{obj_name}",
                "constraints": ["collision_free", "force_limited"]
            }
        # Add more conversion rules as needed
        return {"type": action_type, "params": params}
```

## Challenges and Solutions

### Handling Ambiguity

Natural language commands often contain ambiguity that must be resolved:

```python
class AmbiguityResolver:
    def __init__(self, llm_planner: LLMPlanningSystem, perception_system: Any):
        self.llm_planner = llm_planner
        self.perception_system = perception_system
    
    def resolve_ambiguity(self, command: str) -> (str, bool):
        """
        Detect and resolve ambiguities in natural language commands
        """
        # Query LLM to identify ambiguities
        prompt = f"""
        Analyze the following command for potential ambiguities:
        
        Command: {command}
        
        Identify what information is ambiguous or underspecified that would be needed for a robot to execute this command.
        
        Respond with a JSON object:
        {{
            "ambiguities": ["list of ambiguities"],
            "needed_clarifications": ["list of clarifying questions"]
        }}
        """
        
        response = self.llm_planner._query_llm(prompt)
        try:
            analysis = json.loads(response)
            ambiguities = analysis.get("ambiguities", [])
            clarifications = analysis.get("needed_clarifications", [])
            
            if ambiguities:
                # Generate clarifying questions
                questions = self._generate_clarifying_questions(clarifications, command)
                return questions, True  # Has ambiguity
            
            return command, False  # No ambiguity detected
            
        except json.JSONDecodeError:
            # If LLM response is malformed, assume no ambiguity
            return command, False
    
    def _generate_clarifying_questions(self, clarifications: List[str], original_command: str) -> List[str]:
        """
        Generate natural clarifying questions based on needed information
        """
        questions = []
        
        for clarification in clarifications:
            # In practice, this would use more sophisticated generation
            questions.append(f"Could you clarify: {clarification}")
        
        return questions
```

### Real-Time Performance Optimization

LLM queries can be slow, so optimization is critical:

```python
class OptimizedLLMInterface:
    def __init__(self, llm_planner: LLMPlanningSystem):
        self.llm_planner = llm_planner
        self.cache = {}  # Simple caching for repeated requests
        self.query_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._query_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def _query_worker(self):
        """
        Background worker to handle LLM queries
        """
        while True:
            try:
                query_id, prompt = self.query_queue.get(timeout=1.0)
                
                # Check cache first
                cache_key = hash(prompt)
                if cache_key in self.cache:
                    result = self.cache[cache_key]
                else:
                    # Call LLM
                    result = self.llm_planner._query_llm(prompt)
                    # Cache result
                    self.cache[cache_key] = result
                
                # Return result
                self.result_queue.put((query_id, result))
                
            except queue.Empty:
                continue  # Continue loop
    
    def async_query(self, prompt: str) -> str:
        """
        Asynchronously query the LLM
        """
        query_id = str(uuid.uuid4())
        self.query_queue.put((query_id, prompt))
        
        # Wait for result
        while True:
            try:
                result_id, result = self.result_queue.get(timeout=0.01)
                if result_id == query_id:
                    return result
            except queue.Empty:
                continue
```

## Evaluation and Validation

### Metrics for LLM-Driven Planning

```python
class LLMPlanningEvaluator:
    def __init__(self):
        self.metrics = {
            "task_success_rate": 0.0,
            "plan_accuracy": 0.0,
            "execution_time": 0.0,
            "human_intervention": 0.0,
            "safety_violations": 0.0
        }
    
    def evaluate_plan(self, plan: List[Dict], expected_outcome: Dict) -> Dict:
        """
        Evaluate the quality of an LLM-generated plan
        """
        results = {}
        
        # Calculate success metrics
        results["action_completeness"] = self._check_action_completeness(plan, expected_outcome)
        results["spatial_consistency"] = self._check_spatial_consistency(plan)
        results["temporal_feasibility"] = self._check_temporal_feasibility(plan)
        results["safety_compliance"] = self._check_safety_compliance(plan)
        
        return results
    
    def _check_action_completeness(self, plan: List[Dict], expected_outcome: Dict) -> float:
        """
        Check if the plan covers all necessary actions to achieve the goal
        """
        # Implementation would compare plan actions to required actions
        return 1.0  # Placeholder
    
    def _check_spatial_consistency(self, plan: List[Dict]) -> float:
        """
        Check if spatial relations in the plan are consistent
        """
        # Implementation would validate spatial constraints
        return 1.0  # Placeholder
    
    def _check_temporal_feasibility(self, plan: List[Dict]) -> float:
        """
        Check if the plan is temporally feasible
        """
        # Implementation would validate timing constraints
        return 1.0  # Placeholder
    
    def _check_safety_compliance(self, plan: List[Dict]) -> float:
        """
        Check if the plan complies with safety constraints
        """
        # Implementation would validate safety constraints
        return 1.0  # Placeholder
```

## Exercises

1. **LLM Integration Exercise**: Implement an LLM-based task planner for a simple humanoid robot. Test it with various natural language commands and evaluate its performance.

2. **Ambiguity Resolution Exercise**: Create a system that detects and resolves ambiguities in natural language commands for robot tasks.

3. **Safety Validation Exercise**: Implement safety checks for LLM-generated robot plans, ensuring they comply with physical and operational constraints.

4. **Perception Grounding Exercise**: Connect an LLM planning system with a perception module to ensure plans are grounded in the actual environment.

5. **Hybrid Planning Exercise**: Combine LLM-based high-level planning with traditional motion planning for complex humanoid robot tasks.

## Summary

LLM-driven planning offers exciting opportunities to make humanoid robots more accessible and capable of handling complex, natural language commands. However, safe and effective integration requires careful attention to grounding, validation, safety, and performance considerations. By combining the strengths of LLMs with traditional robotic planning and control systems, we can create more flexible and intelligent humanoid robots that can operate effectively in human environments.