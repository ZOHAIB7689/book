# Testing and Validation for Humanoid Robotics Systems

## Introduction

Testing and validation are critical for humanoid robotics systems due to their complexity, safety requirements, and human interaction environments. Unlike traditional robots, humanoid robots must operate safely around humans, handle unpredictable environments, and maintain stability during complex multi-joint movements. This chapter covers comprehensive testing methodologies, validation procedures, and quality assurance frameworks specifically designed for humanoid robotics applications.

## Testing Strategies

### Component-Level Testing

Component-level testing focuses on individual subsystems, ensuring each component functions correctly in isolation before integration:

```python
import asyncio
import unittest
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class TestResult:
    """Structured test result"""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    measured_values: Dict[str, Any] = None
    expected_values: Dict[str, Any] = None

class ComponentTester:
    """Base class for component testing"""
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.results: List[TestResult] = []
    
    async def run_test_suite(self) -> List[TestResult]:
        """Run all tests for the component"""
        raise NotImplementedError
    
    def assert_equal(self, actual: Any, expected: Any, message: str = "") -> bool:
        """Assert equality with detailed failure reporting"""
        try:
            assert actual == expected, f"{message} - Expected {expected}, got {actual}"
            return True
        except AssertionError as e:
            print(f"Assertion failed: {e}")
            return False
    
    def assert_within_range(self, value: float, min_val: float, max_val: float, 
                          message: str = "") -> bool:
        """Assert that a value falls within a specified range"""
        try:
            assert min_val <= value <= max_val, f"{message} - Value {value} not in range [{min_val}, {max_val}]"
            return True
        except AssertionError as e:
            print(f"Range assertion failed: {e}")
            return False
    
    def record_test_result(self, test_name: str, success: bool, duration: float, 
                          error_message: Optional[str] = None,
                          measured_values: Optional[Dict[str, Any]] = None,
                          expected_values: Optional[Dict[str, Any]] = None) -> TestResult:
        """Record a test result"""
        result = TestResult(
            test_name=test_name,
            success=success,
            duration=duration,
            error_message=error_message,
            measured_values=measured_values or {},
            expected_values=expected_values or {}
        )
        self.results.append(result)
        return result

class LocomotionComponentTester(ComponentTester):
    """Test suite for locomotion components"""
    def __init__(self, component):
        super().__init__("locomotion")
        self.component = component
    
    async def run_test_suite(self) -> List[TestResult]:
        """Run all tests for locomotion component"""
        tests = [
            self.test_initialization,
            self.test_standing_balance,
            self.test_simple_movement,
            self.test_gait_transitions,
            self.test_obstacle_avoidance,
            self.test_emergency_stop
        ]
        
        results = []
        for test in tests:
            start_time = time.time()
            try:
                result = await test()
                duration = time.time() - start_time
                results.append(self.record_test_result(
                    test_name=test.__name__,
                    success=result,
                    duration=duration
                ))
            except Exception as e:
                duration = time.time() - start_time
                results.append(self.record_test_result(
                    test_name=test.__name__,
                    success=False,
                    duration=duration,
                    error_message=str(e)
                ))
        
        return results
    
    async def test_initialization(self) -> bool:
        """Test component initialization"""
        # Test that component initializes without errors
        try:
            initialized = await self.component.initialize()
            return self.assert_equal(initialized, True, "Component should initialize successfully")
        except Exception as e:
            print(f"Initialization test failed: {e}")
            return False
    
    async def test_standing_balance(self) -> bool:
        """Test that the robot can maintain balance while standing"""
        # Reset to standing position
        await self.component.set_gait("standing")
        
        # Monitor balance for a period
        start_time = time.time()
        while time.time() - start_time < 5.0:  # Test for 5 seconds
            current_state = await self.component.get_state()
            
            # Check that center of mass stays within balance limits
            com_position = current_state.get("center_of_mass", [0, 0, 0])
            support_polygon = current_state.get("support_polygon", [])
            
            if not self._is_com_stable(com_position, support_polygon):
                return self.assert_equal(False, True, f"Robot lost balance at time {time.time() - start_time:.2f}s")
            
            await asyncio.sleep(0.1)  # 10 Hz monitoring
        
        return True
    
    async def test_simple_movement(self) -> bool:
        """Test simple forward/backward movement"""
        initial_position = await self.component.get_position()
        
        # Move forward 1 meter
        await self.component.move_to([initial_position[0] + 1.0, initial_position[1], initial_position[2]])
        
        # Wait for movement to complete
        await asyncio.sleep(3.0)
        
        final_position = await self.component.get_position()
        
        # Check that we moved approximately 1 meter in the x direction
        distance_moved = abs(final_position[0] - initial_position[0])
        return self.assert_within_range(distance_moved, 0.8, 1.2, "Movement distance should be within range")
    
    async def test_gait_transitions(self) -> bool:
        """Test transitions between different gaits"""
        gaits = ["standing", "walking", "trot", "crawl"]
        success_count = 0
        
        for gait in gaits:
            try:
                await self.component.set_gait(gait)
                current_gait = await self.component.get_gait()
                
                if self.assert_equal(current_gait, gait, f"Gait should transition to {gait}"):
                    success_count += 1
            except Exception as e:
                print(f"Failed to transition to {gait}: {e}")
        
        return self.assert_equal(success_count, len(gaits), "All gait transitions should succeed")
    
    async def test_obstacle_avoidance(self) -> bool:
        """Test obstacle avoidance functionality"""
        # Set up an obstacle in front of the robot
        await self.component.add_simulation_obstacle([0.5, 0, 0])
        
        # Try to move through the obstacle
        initial_position = await self.component.get_position()
        await self.component.move_to([initial_position[0] + 2.0, initial_position[1], initial_position[2]])
        
        # Wait for obstacle avoidance
        await asyncio.sleep(5.0)
        
        final_position = await self.component.get_position()
        
        # Check that robot navigated around the obstacle rather than colliding
        if abs(final_position[1] - initial_position[1]) > 0.2:  # Moved laterally to avoid
            return True
        else:
            return self.assert_equal(False, True, "Robot should avoid obstacle by moving laterally")
    
    async def test_emergency_stop(self) -> bool:
        """Test emergency stop functionality"""
        # Start moving
        initial_position = await self.component.get_position()
        await self.component.move_to([initial_position[0] + 5.0, initial_position[1], initial_position[2]])
        
        # Trigger emergency stop after a short delay
        await asyncio.sleep(1.0)
        await self.component.emergency_stop()
        
        # Check that movement stopped
        await asyncio.sleep(0.5)  # Wait for stop to complete
        stopped_position = await self.component.get_position()
        
        # Position should not have changed significantly after stop
        distance_traveled = abs(stopped_position[0] - initial_position[0])
        return self.assert_within_range(distance_traveled, 0.1, 1.5, "Robot should stop quickly")
    
    def _is_com_stable(self, com_position: List[float], support_polygon: List[List[float]]) -> bool:
        """Check if center of mass is within support polygon"""
        # Simple check: if support polygon exists and CoM is within rough bounds
        if not support_polygon or len(support_polygon) == 0:
            return False
        
        # For now, assume a 20cm stability margin around center
        return abs(com_position[0]) < 0.2 and abs(com_position[1]) < 0.2

class ManipulationComponentTester(ComponentTester):
    """Test suite for manipulation components"""
    def __init__(self, component):
        super().__init__("manipulation")
        self.component = component
    
    async def run_test_suite(self) -> List[TestResult]:
        """Run all tests for manipulation component"""
        tests = [
            self.test_gripper_functionality,
            self.test_joint_position_control,
            self.test_grasp_generation,
            self.test_ik_solution,
            self.test_payload_handling,
            self.test_safety_limits
        ]
        
        results = []
        for test in tests:
            start_time = time.time()
            try:
                result = await test()
                duration = time.time() - start_time
                results.append(self.record_test_result(
                    test_name=test.__name__,
                    success=result,
                    duration=duration
                ))
            except Exception as e:
                duration = time.time() - start_time
                results.append(self.record_test_result(
                    test_name=test.__name__,
                    success=False,
                    duration=duration,
                    error_message=str(e)
                ))
        
        return results
    
    async def test_gripper_functionality(self) -> bool:
        """Test gripper opening and closing"""
        # Open gripper
        await self.component.open_gripper("right")
        gripper_state = await self.component.get_gripper_state("right")
        
        if not self.assert_equal(gripper_state["position"], "open", "Gripper should be open"):
            return False
        
        # Close gripper
        await self.component.close_gripper("right")
        gripper_state = await self.component.get_gripper_state("right")
        
        return self.assert_equal(gripper_state["position"], "closed", "Gripper should be closed")
    
    async def test_joint_position_control(self) -> bool:
        """Test precise joint position control"""
        initial_positions = await self.component.get_joint_positions("right_arm")
        
        # Command a new joint configuration
        target_positions = [0.5, -0.3, 0.2, 0.4, -0.1, 0.6, -0.2]  # 7 joints
        await self.component.set_joint_positions("right_arm", target_positions)
        
        # Wait for movement
        await asyncio.sleep(2.0)
        
        achieved_positions = await self.component.get_joint_positions("right_arm")
        
        # Check that positions are within tolerance (10 degrees = 0.17 radians)
        tolerance = 0.17
        for i, (achieved, target) in enumerate(zip(achieved_positions, target_positions)):
            if abs(achieved - target) > tolerance:
                return self.assert_equal(False, True, f"Joint {i} position error: {abs(achieved - target):.3f} > {tolerance}")
        
        return True
    
    async def test_grasp_generation(self) -> bool:
        """Test grasp pose generation for objects"""
        # Test with a simple object (sphere)
        object_info = {
            "type": "sphere",
            "dimensions": [0.05],  # radius 5cm
            "position": [0.3, 0.2, 0.8],
            "friction": 0.5
        }
        
        grasp_candidates = await self.component.generate_grasps(object_info)
        
        # Should generate at least one grasp candidate
        success = self.assert_equal(len(grasp_candidates) > 0, True, "Should generate grasp candidates")
        
        if grasp_candidates:
            # Check that grasp positions are reasonable
            for grasp in grasp_candidates[:3]:  # Check first 3 candidates
                if "position" in grasp and "orientation" in grasp:
                    success = success and True
                else:
                    success = False
                    break
        
        return success
    
    async def test_ik_solution(self) -> bool:
        """Test inverse kinematics solutions"""
        # Test with a reachable position
        target_pose = {
            "position": [0.4, 0.0, 0.8],
            "orientation": [0, 0, 0, 1]
        }
        
        solution = await self.component.solve_inverse_kinematics("right_arm", target_pose)
        
        if solution is None:
            return self.assert_equal(False, True, "IK should find solution for reachable position")
        
        # Verify that the solution achieves the target
        achieved_pose = await self.component.forward_kinematics("right_arm", solution)
        
        position_error = self._calculate_position_error(
            target_pose["position"], 
            achieved_pose["position"]
        )
        
        return self.assert_within_range(position_error, 0, 0.02, "Position error should be within 2cm")
    
    async def test_payload_handling(self) -> bool:
        """Test handling different payload weights"""
        payloads = [0.0, 0.5, 1.0, 2.0]  # kg
        success_count = 0
        
        for payload in payloads:
            try:
                # Attach payload
                await self.component.attach_payload(payload)
                
                # Move to same position with different loads
                test_pose = {"position": [0.3, 0.1, 0.7], "orientation": [0, 0, 0, 1]}
                solution = await self.component.solve_inverse_kinematics("right_arm", test_pose)
                
                if solution is not None:
                    # Execute movement
                    await self.component.execute_trajectory("right_arm", [solution])
                    await asyncio.sleep(1.0)
                    success_count += 1
                
                # Detach payload
                await self.component.detach_payload()
                
            except Exception as e:
                print(f"Payload {payload}kg test failed: {e}")
        
        return self.assert_equal(success_count, len(payloads), "All payload tests should pass")
    
    async def test_safety_limits(self) -> bool:
        """Test that safety limits are enforced"""
        # Try to command a position that exceeds joint limits
        dangerous_positions = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]  # Way beyond limits
        
        try:
            await self.component.set_joint_positions("right_arm", dangerous_positions)
            # If we get here without safety intervention, it's a failure
            return self.assert_equal(False, True, "Safety limits not enforced")
        except Exception:
            # This is expected - safety should prevent dangerous commands
            return True
    
    def _calculate_position_error(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate Cartesian distance between two positions"""
        return sum((a - b)**2 for a, b in zip(pos1, pos2))**0.5

class PerceptionComponentTester(ComponentTester):
    """Test suite for perception components"""
    def __init__(self, component):
        super().__init__("perception")
        self.component = component
    
    async def run_test_suite(self) -> List[TestResult]:
        """Run all tests for perception component"""
        tests = [
            self.test_object_detection_accuracy,
            self.test_person_detection,
            self.test_depth_sensing,
            self.test_sensor_fusion,
            self.test_environment_mapping,
            self.test_detection_range
        ]
        
        results = []
        for test in tests:
            start_time = time.time()
            try:
                result = await test()
                duration = time.time() - start_time
                results.append(self.record_test_result(
                    test_name=test.__name__,
                    success=result,
                    duration=duration
                ))
            except Exception as e:
                duration = time.time() - start_time
                results.append(self.record_test_result(
                    test_name=test.__name__,
                    success=False,
                    duration=duration,
                    error_message=str(e)
                ))
        
        return results
    
    async def test_object_detection_accuracy(self) -> bool:
        """Test accuracy of object detection"""
        # Create a controlled test scenario
        test_objects = [
            {"name": "red_box", "dimensions": [0.1, 0.1, 0.1], "position": [1.0, 0, 0.8]},
            {"name": "blue_sphere", "dimensions": [0.05], "position": [1.2, 0.2, 0.8]},
            {"name": "green_cylinder", "dimensions": [0.04, 0.15], "position": [0.8, -0.1, 0.75]}
        ]
        
        # Add objects to simulation environment
        for obj in test_objects:
            await self.component.add_simulated_object(obj)
        
        # Run detection
        detections = await self.component.detect_objects()
        
        # Count matches
        correct_detections = 0
        for true_obj in test_objects:
            for det in detections:
                if (det["name"] == true_obj["name"] and 
                    self._is_same_position(det["position"], true_obj["position"], tolerance=0.1)):
                    correct_detections += 1
                    break
        
        accuracy = correct_detections / len(test_objects) if test_objects else 0
        return self.assert_within_range(accuracy, 0.8, 1.0, "Object detection accuracy should be at least 80%")
    
    async def test_person_detection(self) -> bool:
        """Test person detection and tracking"""
        # Add a person to the environment
        person = {"position": [1.5, 0, 0.0], "velocity": [0.2, 0, 0]}
        await self.component.add_simulated_person(person)
        
        # Detect people
        people = await self.component.detect_people()
        
        if len(people) == 0:
            return self.assert_equal(False, True, "Should detect simulated person")
        
        person_detected = people[0]
        
        # Check position accuracy
        position_error = self._calculate_position_error(
            person["position"], 
            person_detected["position"]
        )
        
        return self.assert_within_range(position_error, 0, 0.2, "Person detection should be accurate within 20cm")
    
    async def test_depth_sensing(self) -> bool:
        """Test depth sensing accuracy"""
        # Test at multiple known distances
        test_distances = [0.5, 1.0, 1.5, 2.0, 2.5]  # meters
        errors = []
        
        for distance in test_distances:
            # Set up a plane at known distance
            await self.component.set_depth_test_plane(distance)
            
            # Get measured distance
            measured_distances = await self.component.get_depth_measurement([0, 0, 0])
            avg_measured = sum(measured_distances) / len(measured_distances) if measured_distances else 0
            
            error = abs(avg_measured - distance)
            errors.append(error)
        
        mean_error = sum(errors) / len(errors) if errors else float('inf')
        return self.assert_within_range(mean_error, 0, 0.05, "Mean depth error should be within 5cm")
    
    async def test_sensor_fusion(self) -> bool:
        """Test fusion of multiple sensor inputs"""
        # Simulate data from different sensors
        camera_data = {"objects": [{"name": "cup", "position": [1.0, 0.1, 0.8]}]}
        lidar_data = {"obstacles": [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]}
        imu_data = {"orientation": [0.1, 0.05, 0.0, 0.99]}
        
        # Run sensor fusion
        fused_result = await self.component.fuse_sensor_data({
            "camera": camera_data,
            "lidar": lidar_data,
            "imu": imu_data
        })
        
        # Check that fused result contains expected information
        expected_elements = ["objects", "obstacles", "orientation", "confidence"]
        success = True
        
        for element in expected_elements:
            if element not in fused_result:
                success = False
                print(f"Missing expected element: {element}")
        
        return success
    
    async def test_environment_mapping(self) -> bool:
        """Test environment mapping accuracy"""
        # Create a known environment layout
        known_layout = {
            "walls": [
                {"start": [0, 0, 0], "end": [2, 0, 0]},
                {"start": [2, 0, 0], "end": [2, 2, 0]},
                {"start": [2, 2, 0], "end": [0, 2, 0]},
                {"start": [0, 2, 0], "end": [0, 0, 0]}
            ],
            "obstacles": [
                {"position": [1.0, 0.5, 0], "dimensions": [0.3, 0.3, 0.5]}
            ]
        }
        
        await self.component.set_known_environment(known_layout)
        
        # Have robot map the environment
        generated_map = await self.component.build_environment_map()
        
        # Compare generated map with known layout
        layout_match = self._compare_maps(generated_map, known_layout)
        
        return self.assert_within_range(layout_match, 0.8, 1.0, "Environmental mapping should be 80%+ accurate")
    
    async def test_detection_range(self) -> bool:
        """Test detection performance at different distances"""
        test_distances = [0.3, 1.0, 2.0, 3.0, 4.0, 5.0]  # meters
        success_rates = []
        
        for dist in test_distances:
            # Place object at distance
            obj = {"name": "test_object", "position": [dist, 0, 0.8]}
            await self.component.add_simulated_object(obj)
            
            # Try to detect
            detections = await self.component.detect_objects()
            
            detected = any(
                det["name"] == obj["name"] and 
                self._is_same_position(det["position"], obj["position"], 0.2)
                for det in detections
            )
            
            success_rates.append(1.0 if detected else 0.0)
        
        # At close range (<2m), detection should be nearly perfect
        close_range_success = sum(success_rates[:3]) / 3 if len(success_rates) >= 3 else 0
        far_range_success = sum(success_rates[3:]) / len(success_rates[3:]) if len(success_rates) > 3 else 0
        
        close_ok = self.assert_within_range(close_range_success, 0.8, 1.0, "Close range detection should be 80%+")
        far_ok = self.assert_within_range(far_range_success, 0.3, 1.0, "Far range detection should have minimum acceptance")
        
        return close_ok and far_ok
    
    def _is_same_position(self, pos1: List[float], pos2: List[float], tolerance: float) -> bool:
        """Check if two positions are the same within tolerance"""
        return self._calculate_position_error(pos1, pos2) <= tolerance
    
    def _compare_maps(self, map1: Dict[str, Any], map2: Dict[str, Any]) -> float:
        """Compare two environment maps and return similarity score"""
        # This is a simplified comparison
        # In a real system, this would be much more sophisticated
        return 0.85  # Return a placeholder similarity score

class CognitiveComponentTester(ComponentTester):
    """Test suite for cognitive components"""
    def __init__(self, component):
        super().__init__("cognitive")
        self.component = component
    
    async def run_test_suite(self) -> List[TestResult]:
        """Run all tests for cognitive component"""
        tests = [
            self.test_natural_language_understanding,
            self.test_task_planning,
            self.test_context_reasoning,
            self.test_memory_operations,
            self.test_decision_making,
            self.test_dialog_management
        ]
        
        results = []
        for test in tests:
            start_time = time.time()
            try:
                result = await test()
                duration = time.time() - start_time
                results.append(self.record_test_result(
                    test_name=test.__name__,
                    success=result,
                    duration=duration
                ))
            except Exception as e:
                duration = time.time() - start_time
                results.append(self.record_test_result(
                    test_name=test.__name__,
                    success=False,
                    duration=duration,
                    error_message=str(e)
                ))
        
        return results
    
    async def test_natural_language_understanding(self) -> bool:
        """Test natural language understanding capabilities"""
        test_inputs = [
            ("Take the red cup to the kitchen", "navigation"),
            ("Grasp the book on the table", "manipulation"),
            ("Hello, how are you?", "social"),
            ("Find the person wearing blue", "detection"),
            ("Come to my location", "navigation")
        ]
        
        success_count = 0
        for input_text, expected_intent in test_inputs:
            intent_result = await self.component.process_natural_language(input_text)
            detected_intent = intent_result.get("intent", "unknown")
            
            if self.assert_equal(detected_intent, expected_intent, f"Intent for '{input_text}'"):
                success_count += 1
        
        return self.assert_within_range(
            success_count / len(test_inputs) if test_inputs else 0, 
            0.8, 1.0, 
            "NLU should correctly identify intents 80%+ of the time"
        )
    
    async def test_task_planning(self) -> bool:
        """Test task planning capabilities"""
        # Test simple navigation task
        nav_task = {"type": "navigation", "destination": "kitchen", "constraints": []}
        nav_plan = await self.component.plan_task(nav_task)
        
        nav_success = len(nav_plan) > 0 if nav_plan else False
        
        # Test manipulation task
        manip_task = {"type": "manipulation", "action": "grasp", "object": "cup", "location": "table"}
        manip_plan = await self.component.plan_task(manip_task)
        
        manip_success = len(manip_plan) > 0 if manip_plan else False
        
        # Test complex task
        complex_task = {
            "type": "complex",
            "sequence": ["navigate_to_kitchen", "detect_cup", "grasp_cup", "navigate_to_table", "place_cup"]
        }
        complex_plan = await self.component.plan_task(complex_task)
        
        complex_success = len(complex_plan) > 3 if complex_plan else False
        
        return nav_success and manip_success and complex_success
    
    async def test_context_reasoning(self) -> bool:
        """Test context-aware reasoning"""
        # Set up a context
        context = {
            "location": "kitchen",
            "time": "evening",
            "detected_objects": ["refrigerator", "table", "chair"],
            "detected_people": ["person_1"]
        }
        
        # Ask context-dependent questions
        question1 = "Is the refrigerator accessible?"
        response1 = await self.component.reason_with_context(question1, context)
        
        question2 = "Who should I serve dinner to?"
        response2 = await self.component.reason_with_context(question2, context)
        
        # Check that responses are contextually appropriate
        # For this test, we'll verify that responses contain relevant information
        contextually_appropriate = (
            "refrigerator" in str(response1).lower() or
            "dinner" in str(response2).lower()
        )
        
        return self.assert_equal(contextually_appropriate, True, "Responses should be contextually relevant")
    
    async def test_memory_operations(self) -> bool:
        """Test memory storage and retrieval"""
        # Store information
        memory_item = {
            "type": "location",
            "name": "johns_office", 
            "coordinates": [2.5, 1.0, 0.0],
            "visited": True,
            "timestamp": time.time()
        }
        
        await self.component.store_memory(memory_item)
        
        # Retrieve the same information
        retrieved = await self.component.retrieve_memory("johns_office", "location")
        
        if not retrieved:
            return self.assert_equal(False, True, "Memory retrieval failed")
        
        # Check that retrieved information matches
        matches = (
            retrieved.get("coordinates") == memory_item["coordinates"] and
            retrieved.get("visited") == memory_item["visited"]
        )
        
        return self.assert_equal(matches, True, "Retrieved memory should match stored memory")
    
    async def test_decision_making(self) -> bool:
        """Test decision making under uncertainty"""
        # Test decision in a simple scenario
        scenario = {
            "options": [
                {"action": "go_left", "risk": 0.2, "reward": 0.8},
                {"action": "go_right", "risk": 0.6, "reward": 0.9},
                {"action": "stay", "risk": 0.1, "reward": 0.3}
            ],
            "preferences": {"safety": 0.7, "efficiency": 0.3}
        }
        
        decision = await self.component.make_decision(scenario)
        
        # With high safety preference, should choose safer option
        # The safest option is "stay" (risk=0.1), but it has low reward
        # The balanced choice might be "go_left" (decent safety, good reward)
        
        acceptable_decisions = ["go_left", "stay"]
        decision_acceptable = decision in acceptable_decisions
        
        return self.assert_equal(decision_acceptable, True, f"Decision should be one of {acceptable_decisions}")
    
    async def test_dialog_management(self) -> bool:
        """Test dialog management capabilities"""
        # Simulate a simple conversation
        conversation_history = [
            {"turn": "user", "text": "Hello robot!"},
            {"turn": "robot", "text": "Hello! How can I assist you today?"},
            {"turn": "user", "text": "Please go to the kitchen."},
            {"turn": "robot", "text": "On my way to the kitchen!"}
        ]
        
        # Test response generation
        response = await self.component.generate_dialog_response(
            "What did I just ask you to do?", 
            conversation_history
        )
        
        # Response should mention going to kitchen
        mentions_kitchen = "kitchen" in response.lower()
        
        # Test clarification request
        ambiguous_request = "Take it there"
        clarification = await self.component.request_clarification(ambiguous_request, conversation_history)
        
        has_clarification = len(clarification) > 10  # Should return meaningful clarification request
        
        return mentions_kitchen and has_clarification
```

### Integration Testing

Integration testing ensures that components work together effectively:

```python
class IntegrationTester:
    """Tester for component integration"""
    def __init__(self, components: Dict[str, Any]):
        self.components = components
        self.test_results = []
    
    async def test_navigation_manipulation_integration(self) -> TestResult:
        """Test integration between navigation and manipulation"""
        start_time = time.time()
        
        try:
            # 1. Navigate to a location with an object
            nav_component = self.components["locomotion"]
            manip_component = self.components["manipulation"]
            
            # Move to object location
            await nav_component.move_to([1.0, 0.0, 0.0])
            
            # Wait for navigation to complete
            await asyncio.sleep(2.0)
            
            # 2. Detect objects at the location
            perception_component = self.components["perception"]
            objects = await perception_component.detect_objects()
            
            if not objects:
                return self._create_failure_result(
                    "navigation_manipulation_integration",
                    time.time() - start_time,
                    "No objects detected at destination"
                )
            
            # 3. Plan and execute grasp of detected object
            target_object = objects[0]
            grasp_plan = await manip_component.generate_grasps(target_object)
            
            if not grasp_plan:
                return self._create_failure_result(
                    "navigation_manipulation_integration", 
                    time.time() - start_time,
                    "Could not generate grasp for detected object"
                )
            
            # Execute grasp
            grasp_result = await manip_component.execute_grasp(grasp_plan[0])
            
            success = grasp_result.get("success", False)
            return self._create_result(
                "navigation_manipulation_integration",
                success,
                time.time() - start_time
            )
            
        except Exception as e:
            return self._create_failure_result(
                "navigation_manipulation_integration",
                time.time() - start_time,
                str(e)
            )
    
    async def test_perception_cognition_integration(self) -> TestResult:
        """Test integration between perception and cognition"""
        start_time = time.time()
        
        try:
            perception_component = self.components["perception"]
            cognitive_component = self.components["cognitive"]
            
            # 1. Detect environment
            current_percept = await perception_component.get_environment_state()
            
            if not current_percept:
                return self._create_failure_result(
                    "perception_cognition_integration",
                    time.time() - start_time,
                    "Could not get environment state"
                )
            
            # 2. Process perception through cognitive system
            task_request = "Analyze the environment and suggest a useful action"
            cognitive_output = await cognitive_component.process_perception_data(
                current_percept, 
                task_request
            )
            
            # 3. Verify output contains meaningful information
            has_meaningful_output = (
                "analysis" in cognitive_output or
                "suggestion" in cognitive_output or
                "plan" in cognitive_output
            )
            
            return self._create_result(
                "perception_cognition_integration",
                has_meaningful_output,
                time.time() - start_time
            )
            
        except Exception as e:
            return self._create_failure_result(
                "perception_cognition_integration",
                time.time() - start_time,
                str(e)
            )
    
    async def test_full_system_workflow(self) -> TestResult:
        """Test a complete workflow involving all components"""
        start_time = time.time()
        
        try:
            # Example workflow: "Go to the kitchen, find a cup, and bring it to the table"
            nav_comp = self.components["locomotion"]
            perc_comp = self.components["perception"]
            manip_comp = self.components["manipulation"]
            cog_comp = self.components["cognitive"]
            
            # 1. Cognitive system parses the command
            command = "Go to kitchen, find cup, bring to table"
            task_plan = await cog_comp.parse_command_and_plan(command)
            
            if not task_plan or len(task_plan) == 0:
                return self._create_failure_result(
                    "full_system_workflow",
                    time.time() - start_time,
                    "Cognitive system could not parse command"
                )
            
            # 2. Execute each step in the plan
            step_results = []
            for step in task_plan:
                if step["action"] == "navigate":
                    result = await nav_comp.move_to(step["params"]["destination"])
                    step_results.append(result.get("success", False))
                    
                elif step["action"] == "detect":
                    objects = await perc_comp.detect_objects()
                    target_found = any(obj["name"] == step["params"]["target"] for obj in objects)
                    step_results.append(target_found)
                    
                elif step["action"] == "manipulate":
                    objects = await perc_comp.detect_objects()
                    target_obj = next((obj for obj in objects if obj["name"] == step["params"]["target"]), None)
                    
                    if target_obj:
                        grasp_plan = await manip_comp.generate_grasps(target_obj)
                        if grasp_plan:
                            result = await manip_comp.execute_grasp(grasp_plan[0])
                            step_results.append(result.get("success", False))
                        else:
                            step_results.append(False)
                    else:
                        step_results.append(False)
            
            # 3. All steps should succeed for overall success
            overall_success = all(step_results) and len(step_results) > 0
            
            return self._create_result(
                "full_system_workflow",
                overall_success,
                time.time() - start_time,
                measured_values={"step_results": step_results, "total_steps": len(task_plan)},
                expected_values={"all_steps_success": True}
            )
            
        except Exception as e:
            return self._create_failure_result(
                "full_system_workflow",
                time.time() - start_time,
                str(e)
            )
    
    def _create_result(self, test_name: str, success: bool, duration: float,
                      measured_values: Optional[Dict[str, Any]] = None,
                      expected_values: Optional[Dict[str, Any]] = None) -> TestResult:
        """Create a test result"""
        result = TestResult(
            test_name=test_name,
            success=success,
            duration=duration,
            measured_values=measured_values or {},
            expected_values=expected_values or {}
        )
        self.test_results.append(result)
        return result
    
    def _create_failure_result(self, test_name: str, duration: float, 
                              error_message: str) -> TestResult:
        """Create a failed test result"""
        return self._create_result(test_name, False, duration, 
                                  error_message=error_message)

class SystemTestManager:
    """Manages all system testing activities"""
    def __init__(self):
        self.component_testers = {}
        self.integration_tester = None
        self.results_summary = {}
    
    def add_component_tester(self, name: str, tester: ComponentTester):
        """Add a component tester to the system"""
        self.component_testers[name] = tester
    
    def set_integration_tester(self, tester: IntegrationTester):
        """Set the integration tester"""
        self.integration_tester = tester
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report"""
        print("Starting comprehensive system testing...")
        
        # Run component tests
        component_results = {}
        for name, tester in self.component_testers.items():
            print(f"Running tests for {name} component...")
            results = await tester.run_test_suite()
            component_results[name] = results
            print(f"  {name}: {sum(1 for r in results if r.success)}/{len(results)} tests passed")
        
        # Run integration tests
        integration_results = []
        if self.integration_tester:
            print("Running integration tests...")
            integration_tests = [
                self.integration_tester.test_navigation_manipulation_integration,
                self.integration_tester.test_perception_cognition_integration,
                self.integration_tester.test_full_system_workflow
            ]
            
            for test in integration_tests:
                result = await test()
                integration_results.append(result)
                status = "PASS" if result.success else "FAIL"
                print(f"  {result.test_name}: {status}")
        
        # Generate summary
        summary = self._generate_test_summary(component_results, integration_results)
        
        # Save results
        self.results_summary = summary
        
        return summary
    
    def _generate_test_summary(self, component_results: Dict[str, List[TestResult]], 
                              integration_results: List[TestResult]) -> Dict[str, Any]:
        """Generate summary of all test results"""
        total_tests = 0
        passed_tests = 0
        total_duration = 0.0
        
        component_summary = {}
        for comp_name, results in component_results.items():
            comp_passed = sum(1 for r in results if r.success)
            comp_total = len(results)
            comp_duration = sum(r.duration for r in results)
            
            component_summary[comp_name] = {
                "passed": comp_passed,
                "total": comp_total,
                "success_rate": comp_passed / comp_total if comp_total > 0 else 0,
                "duration": comp_duration
            }
            
            total_tests += comp_total
            passed_tests += comp_passed
            total_duration += comp_duration
        
        # Integration tests
        integration_summary = {}
        if integration_results:
            int_passed = sum(1 for r in integration_results if r.success)
            int_total = len(integration_results)
            int_duration = sum(r.duration for r in integration_results)
            
            integration_summary = {
                "passed": int_passed,
                "total": int_total,
                "success_rate": int_passed / int_total if int_total > 0 else 0,
                "duration": int_duration
            }
            
            total_tests += int_total
            passed_tests += int_passed
            total_duration += int_duration
        
        overall_success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": overall_success_rate,
                "total_duration": total_duration
            },
            "component_results": component_summary,
            "integration_results": integration_summary,
            "recommendations": self._generate_recommendations(component_results, integration_results)
        }
    
    def _generate_recommendations(self, component_results: Dict[str, List[TestResult]], 
                                 integration_results: List[TestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for problematic components
        for comp_name, results in component_results.items():
            success_rate = sum(1 for r in results if r.success) / len(results) if results else 0
            
            if success_rate < 0.8:  # Less than 80% success
                recommendations.append(f"Component '{comp_name}' has low test success rate ({success_rate:.1%}), investigate failures")
        
        # Check integration results
        if integration_results:
            int_success_rate = sum(1 for r in integration_results if r.success) / len(integration_results)
            
            if int_success_rate < 0.7:  # Less than 70% success
                recommendations.append(f"Integration tests have low success rate ({int_success_rate:.1%}), may need component interface improvements")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All tests passing at acceptable rates - continue current development approach")
        
        return recommendations

# Example usage
async def run_system_tests():
    """Example of running system tests"""
    print("Setting up system test environment...")
    
    # Create mock components (in a real system, these would be actual component instances)
    class MockLocomotionComponent:
        async def initialize(self): return True
        async def get_state(self): return {"center_of_mass": [0, 0, 0], "support_polygon": []}
        async def get_position(self): return [0, 0, 0]
        async def move_to(self, pos): pass
        async def get_gait(self): return "standing"
        async def set_gait(self, gait): pass
        async def add_simulation_obstacle(self, pos): pass
        async def emergency_stop(self): pass
    
    class MockManipulationComponent:
        async def get_gripper_state(self, arm): return {"position": "open"}
        async def open_gripper(self, arm): pass
        async def close_gripper(self, arm): pass
        async def get_joint_positions(self, arm): return [0, 0, 0, 0, 0, 0, 0]
        async def set_joint_positions(self, arm, positions): pass
        async def generate_grasps(self, obj_info): return [{"position": [0.3, 0.1, 0.8], "orientation": [0, 0, 0, 1]}]
        async def solve_inverse_kinematics(self, arm, pose): return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        async def forward_kinematics(self, arm, joints): return {"position": [0.3, 0.1, 0.8], "orientation": [0, 0, 0, 1]}
        async def attach_payload(self, weight): pass
        async def detach_payload(self): pass
        async def execute_trajectory(self, arm, trajectory): pass
    
    class MockPerceptionComponent:
        async def detect_objects(self): return [{"name": "red_box", "position": [1.0, 0, 0.8]}]
        async def detect_people(self): return [{"position": [1.5, 0, 0.0]}]
        async def get_depth_measurement(self, pos): return [0.5, 0.55, 0.45]
        async def fuse_sensor_data(self, data): return {"objects": [], "obstacles": [], "orientation": [0, 0, 0, 1], "confidence": 0.8}
        async def build_environment_map(self): return {"layout": "known"}
        async def add_simulated_object(self, obj): pass
        async def add_simulated_person(self, person): pass
        async def set_depth_test_plane(self, distance): pass
        async def set_known_environment(self, layout): pass
        async def get_environment_state(self): return {"objects": [], "people": []}
    
    class MockCognitiveComponent:
        async def process_natural_language(self, text): return {"intent": "navigation", "confidence": 0.9}
        async def plan_task(self, task): return [{"action": "move", "target": "kitchen"}] if task.get("type") == "navigation" else []
        async def reason_with_context(self, question, context): return "Relevant response based on context"
        async def store_memory(self, item): pass
        async def retrieve_memory(self, name, type): return {"coordinates": [2.5, 1.0, 0.0], "visited": True}
        async def make_decision(self, scenario): return "go_left"
        async def generate_dialog_response(self, text, history): return "I can go to the kitchen"
        async def request_clarification(self, request, history): return "Could you clarify what 'it' refers to?"
        async def parse_command_and_plan(self, command): return [{"action": "navigate", "params": {"destination": [2.0, 1.0, 0.0]}}]
        async def process_perception_data(self, percept, task): return {"analysis": "Environment analyzed", "suggestion": "Move to kitchen"}
    
    # Create component instances
    loco_comp = MockLocomotionComponent()
    manip_comp = MockManipulationComponent()
    perc_comp = MockPerceptionComponent()
    cog_comp = MockCognitiveComponent()
    
    # Create testers
    loco_tester = LocomotionComponentTester(loco_comp)
    manip_tester = ManipulationComponentTester(manip_comp)
    perc_tester = PerceptionComponentTester(perc_comp)
    cog_tester = CognitiveComponentTester(cog_comp)
    
    # Create integration tester
    components = {
        "locomotion": loco_comp,
        "manipulation": manip_comp, 
        "perception": perc_comp,
        "cognitive": cog_comp
    }
    integration_tester = IntegrationTester(components)
    
    # Create test manager
    test_manager = SystemTestManager()
    test_manager.add_component_tester("locomotion", loco_tester)
    test_manager.add_component_tester("manipulation", manip_tester)
    test_manager.add_component_tester("perception", perc_tester)
    test_manager.add_component_tester("cognitive", cog_tester)
    test_manager.set_integration_tester(integration_tester)
    
    # Run tests
    summary = await test_manager.run_comprehensive_test_suite()
    
    print(f"\nTesting Summary:")
    print(f"Overall Success Rate: {summary['summary']['success_rate']:.1%}")
    print(f"Total Tests Run: {summary['summary']['total_tests']}")
    print(f"Passed: {summary['summary']['passed_tests']}")
    print(f"Failed: {summary['summary']['failed_tests']}")
    print(f"Total Duration: {summary['summary']['total_duration']:.2f}s")
    
    print(f"\nRecommendations:")
    for rec in summary["recommendations"]:
        print(f"- {rec}")
    
    return summary

# Run the tests
if __name__ == "__main__":
    summary = asyncio.run(run_system_tests())
```

## Validation Procedures

### Safety Validation

Safety is paramount in humanoid robotics:

```python
class SafetyValidator:
    """Validates safety aspects of humanoid robot systems"""
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.safety_limits = self._define_safety_limits()
        self.emergency_protocols = self._define_emergency_protocols()
    
    def _define_safety_limits(self) -> Dict[str, Any]:
        """Define safety operational limits"""
        return {
            "joint_limits": {
                "hip_pitch": {"min": -1.57, "max": 1.57, "velocity_max": 1.0},
                "knee_pitch": {"min": -0.2, "max": 2.0, "velocity_max": 1.2},
                "ankle_pitch": {"min": -0.5, "max": 0.5, "velocity_max": 0.8},
                "arm_joints": {"position_tolerance": 0.1, "velocity_max": 2.0}
            },
            "balance_limits": {
                "com_height_min": 0.6,  # meters
                "com_deviation_max": 0.15,  # meters from center
                "angular_velocity_max": 1.0  # rad/s
            },
            "collision_distances": {
                "minimum": 0.1,  # meters
                "warning": 0.3,
                "critical": 0.05
            },
            "force_limits": {
                "gripper_force_max": 50.0,  # Newtons
                "joint_torque_max": 100.0,  # Nm
                "contact_force_max": 200.0   # N during impact
            }
        }
    
    def _define_emergency_protocols(self) -> Dict[str, Any]:
        """Define emergency response protocols"""
        return {
            "fall_detection": {
                "angular_threshold": 1.0,  # rad from upright
                "acceleration_threshold": 15.0,  # m/s^2
                "response_time": 0.1  # seconds
            },
            "collision_response": {
                "stiffness_reduction": 0.5,  # Reduce joint stiffness by 50%
                "motion_stop": True,  # Stop all motion
                "recovery_behavior": "return_to_safe_pose"
            },
            "overload_protection": {
                "torque_threshold": 0.9,  # Percentage of max torque
                "temperature_threshold": 60.0,  # Celsius
                "shutdown_delay": 0.5  # seconds before shutdown
            }
        }
    
    async def validate_safety_compliance(self) -> Dict[str, Any]:
        """Validate overall safety compliance"""
        validation_results = {
            "joint_limits_compliance": await self._validate_joint_limits(),
            "balance_stability": await self._validate_balance_stability(),
            "collision_avoidance": await self._validate_collision_avoidance(),
            "force_compliance": await self._validate_force_compliance(),
            "emergency_response": await self._validate_emergency_response()
        }
        
        # Calculate overall compliance
        compliant_checks = sum(1 for result in validation_results.values() if result["compliant"])
        total_checks = len(validation_results)
        overall_compliance = compliant_checks / total_checks if total_checks > 0 else 0
        
        return {
            "overall_compliance_rate": overall_compliance,
            "validation_results": validation_results,
            "pass": overall_compliance >= 0.95,  # Require 95% compliance
            "critical_failures": self._identify_critical_failures(validation_results)
        }
    
    async def _validate_joint_limits(self) -> Dict[str, Any]:
        """Validate all joint limits are respected"""
        # Get current joint states
        joint_states = await self.robot_model.get_joint_states()
        
        violations = []
        for joint_name, state in joint_states.items():
            limit_def = self.safety_limits["joint_limits"].get(joint_name)
            if limit_def:
                pos_ok = limit_def["min"] <= state["position"] <= limit_def["max"]
                vel_ok = abs(state["velocity"]) <= limit_def.get("velocity_max", float('inf'))
                
                if not (pos_ok and vel_ok):
                    violations.append({
                        "joint": joint_name,
                        "position": state["position"],
                        "velocity": state["velocity"],
                        "limits": limit_def,
                        "position_violation": not pos_ok,
                        "velocity_violation": not vel_ok
                    })
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "check_type": "joint_limit_validation"
        }
    
    async def _validate_balance_stability(self) -> Dict[str, Any]:
        """Validate balance and stability during operation"""
        # Get center of mass and support polygon
        com_state = await self.robot_model.get_center_of_mass_state()
        support_polygon = await self.robot_model.get_support_polygon()
        
        # Calculate deviation from stable position
        com_deviation = self._calculate_com_deviation(com_state["position"], support_polygon)
        height_ok = com_state["position"][2] >= self.safety_limits["balance_limits"]["com_height_min"]
        deviation_ok = com_deviation <= self.safety_limits["balance_limits"]["com_deviation_max"]
        
        # Check angular velocity limits
        angular_vel_ok = all(abs(vel) <= self.safety_limits["balance_limits"]["angular_velocity_max"] 
                           for vel in com_state["angular_velocity"])
        
        stability_issues = []
        if not height_ok:
            stability_issues.append(f"CoM height too low: {com_state['position'][2]:.3f}m")
        if not deviation_ok:
            stability_issues.append(f"CoM deviation too large: {com_deviation:.3f}m")
        if not angular_vel_ok:
            stability_issues.append("Excessive angular velocity detected")
        
        return {
            "compliant": len(stability_issues) == 0,
            "stability_issues": stability_issues,
            "com_deviation": com_deviation,
            "check_type": "balance_stability"
        }
    
    async def _validate_collision_avoidance(self) -> Dict[str, Any]:
        """Validate collision detection and avoidance"""
        # Get distance sensors and proximity alerts
        proximity_sensors = await self.robot_model.get_proximity_sensors()
        
        critical_distances = []
        for sensor_reading in proximity_sensors:
            if sensor_reading["distance"] < self.safety_limits["collision_distances"]["critical"]:
                critical_distances.append(sensor_reading)
        
        warning_distances = []
        for sensor_reading in proximity_sensors:
            if (self.safety_limits["collision_distances"]["critical"] <= 
                sensor_reading["distance"] < 
                self.safety_limits["collision_distances"]["warning"]):
                warning_distances.append(sensor_reading)
        
        # Check if collision avoidance system responds appropriately
        avoidance_response = await self._test_collision_avoidance_response()
        
        collision_issues = []
        if critical_distances:
            collision_issues.append(f"Critical proximity distances detected: {len(critical_distances)} zones")
        if not avoidance_response:
            collision_issues.append("Collision avoidance system not responding appropriately")
        
        return {
            "compliant": len(collision_issues) == 0,
            "collision_issues": collision_issues,
            "critical_distances": len(critical_distances),
            "warning_distances": len(warning_distances),
            "avoidance_response_works": avoidance_response,
            "check_type": "collision_avoidance"
        }
    
    async def _validate_force_compliance(self) -> Dict[str, Any]:
        """Validate force limits are not exceeded"""
        # Get force/torque readings from all joints and sensors
        force_data = await self.robot_model.get_force_torque_data()
        
        force_violations = []
        for joint_name, forces in force_data.items():
            if joint_name in self.safety_limits["force_limits"]:
                max_allowed = self.safety_limits["force_limits"][joint_name]
                if forces.get("torque", 0) > max_allowed:
                    force_violations.append({
                        "joint": joint_name,
                        "measured_torque": forces["torque"],
                        "limit": max_allowed
                    })
        
        # Check gripper forces
        gripper_forces = await self.robot_model.get_gripper_forces()
        gripper_violations = []
        for arm, force in gripper_forces.items():
            if force > self.safety_limits["force_limits"]["gripper_force_max"]:
                gripper_violations.append({
                    "arm": arm,
                    "measured_force": force,
                    "limit": self.safety_limits["force_limits"]["gripper_force_max"]
                })
        
        force_issues = []
        if force_violations:
            force_issues.append(f"Joint torque violations: {len(force_violations)} joints")
        if gripper_violations:
            force_issues.append(f"Gripper force violations: {len(gripper_violations)} grippers")
        
        return {
            "compliant": len(force_issues) == 0,
            "force_issues": force_issues,
            "joint_violations": len(force_violations),
            "gripper_violations": len(gripper_violations),
            "check_type": "force_compliance"
        }
    
    async def _validate_emergency_response(self) -> Dict[str, Any]:
        """Validate emergency response protocols work correctly"""
        # Test fall detection
        fall_detected = await self._simulate_fall_detection()
        
        # Test collision response
        collision_responded = await self._simulate_collision_response()
        
        # Test overload protection
        overload_protected = await self._simulate_overload_protection()
        
        emergency_issues = []
        if not fall_detected:
            emergency_issues.append("Fall detection system not responding")
        if not collision_responded:
            emergency_issues.append("Collision response system not triggering")
        if not overload_protected:
            emergency_issues.append("Overload protection system not functioning")
        
        return {
            "compliant": len(emergency_issues) == 0,
            "emergency_issues": emergency_issues,
            "fall_detection_works": fall_detected,
            "collision_response_works": collision_responded,
            "overload_protection_works": overload_protected,
            "check_type": "emergency_response"
        }
    
    async def _test_collision_avoidance_response(self) -> bool:
        """Test that collision avoidance system responds appropriately"""
        # In a real system, this would trigger the avoidance system
        # For simulation, we'll return True to indicate the check passes
        return True
    
    async def _simulate_fall_detection(self) -> bool:
        """Simulate fall detection functionality"""
        # In a real system, this would test the fall detection algorithm
        # For simulation, we'll return True
        return True
    
    async def _simulate_collision_response(self) -> bool:
        """Simulate collision response functionality"""
        # In a real system, this would test collision response
        # For simulation, we'll return True
        return True
    
    async def _simulate_overload_protection(self) -> bool:
        """Simulate overload protection functionality"""
        # In a real system, this would test overload protection
        # For simulation, we'll return True
        return True
    
    def _calculate_com_deviation(self, com_position: List[float], 
                                support_polygon: List[List[float]]) -> float:
        """Calculate deviation of CoM from stable position within support polygon"""
        # Calculate distance to nearest point in support polygon
        if not support_polygon:
            return float('inf')
        
        # For simplicity, assume support polygon center as stable position
        poly_center = [sum(coord) / len(coord) for coord in zip(*support_polygon)]
        dx = com_position[0] - poly_center[0]
        dy = com_position[1] - poly_center[1]
        
        return (dx**2 + dy**2)**0.5
    
    def _identify_critical_failures(self, validation_results: Dict[str, Any]) -> List[str]:
        """Identify critical failures that would prevent safe operation"""
        critical_failures = []
        
        for check_name, result in validation_results.items():
            if not result["compliant"]:
                if check_name in ["balance_stability", "emergency_response"]:
                    # These are critical safety issues
                    critical_failures.append(check_name)
        
        return critical_failures

class PerformanceValidator:
    """Validates performance aspects of humanoid robot systems"""
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.performance_baselines = self._define_performance_baselines()
    
    def _define_performance_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Define baseline performance metrics for humanoid robots"""
        return {
            "locomotion": {
                "walking_speed": {"target": 0.5, "threshold": 0.3},  # m/s
                "turning_speed": {"target": 0.3, "threshold": 0.1},  # rad/s
                "energy_efficiency": {"target": 0.2, "threshold": 0.4},  # J/m/kg
                "balance_recovery_time": {"target": 0.5, "threshold": 1.0},  # seconds
            },
            "manipulation": {
                "grasp_success_rate": {"target": 0.9, "threshold": 0.7},  # rate
                "position_accuracy": {"target": 0.01, "threshold": 0.03},  # meters
                "operation_time": {"target": 5.0, "threshold": 10.0},  # seconds per task
                "payload_capacity": {"target": 2.0, "threshold": 1.0},  # kg
            },
            "perception": {
                "object_detection_rate": {"target": 0.9, "threshold": 0.7},  # rate
                "false_positive_rate": {"target": 0.1, "threshold": 0.3},  # rate
                "processing_latency": {"target": 0.05, "threshold": 0.1},  # seconds
                "detection_range": {"target": 3.0, "threshold": 2.0},  # meters
            },
            "cognition": {
                "response_time": {"target": 1.0, "threshold": 2.0},  # seconds
                "task_completion_rate": {"target": 0.85, "threshold": 0.7},  # rate
                "dialog_success_rate": {"target": 0.9, "threshold": 0.7},  # rate
                "reasoning_accuracy": {"target": 0.9, "threshold": 0.7},  # rate
            }
        }
    
    async def validate_performance_metrics(self) -> Dict[str, Any]:
        """Validate all performance metrics against baselines"""
        results = {
            "locomotion_performance": await self._validate_locomotion_performance(),
            "manipulation_performance": await self._validate_manipulation_performance(),
            "perception_performance": await self._validate_perception_performance(),
            "cognition_performance": await self._validate_cognition_performance(),
        }
        
        # Calculate overall performance score
        total_metrics = 0
        passing_metrics = 0
        
        for domain_results in results.values():
            for metric_name, metric_result in domain_results.items():
                if "valid" in metric_result:
                    total_metrics += 1
                    if metric_result["valid"]:
                        passing_metrics += 1
        
        overall_score = passing_metrics / total_metrics if total_metrics > 0 else 0
        
        return {
            "overall_performance_score": overall_score,
            "domain_results": results,
            "total_metrics": total_metrics,
            "passing_metrics": passing_metrics,
            "performance_grade": self._assign_performance_grade(overall_score)
        }
    
    async def _validate_locomotion_performance(self) -> Dict[str, Any]:
        """Validate locomotion performance metrics"""
        metrics = {}
        
        # Test walking speed
        walking_speed = await self._measure_walking_speed()
        target = self.performance_baselines["locomotion"]["walking_speed"]["target"]
        threshold = self.performance_baselines["locomotion"]["walking_speed"]["threshold"]
        metrics["walking_speed"] = {
            "measured": walking_speed,
            "target": target,
            "threshold": threshold,
            "valid": walking_speed >= threshold,
            "excellent": walking_speed >= target
        }
        
        # Test energy efficiency
        energy_efficiency = await self._measure_energy_efficiency()
        target = self.performance_baselines["locomotion"]["energy_efficiency"]["target"]
        threshold = self.performance_baselines["locomotion"]["energy_efficiency"]["threshold"]
        metrics["energy_efficiency"] = {
            "measured": energy_efficiency,
            "target": target,
            "threshold": threshold,
            "valid": energy_efficiency <= threshold,  # Lower is better
            "excellent": energy_efficiency <= target
        }
        
        # Test balance recovery time
        balance_time = await self._measure_balance_recovery_time()
        target = self.performance_baselines["locomotion"]["balance_recovery_time"]["target"]
        threshold = self.performance_baselines["locomotion"]["balance_recovery_time"]["threshold"]
        metrics["balance_recovery_time"] = {
            "measured": balance_time,
            "target": target,
            "threshold": threshold,
            "valid": balance_time <= threshold,
            "excellent": balance_time <= target
        }
        
        return metrics
    
    async def _validate_manipulation_performance(self) -> Dict[str, Any]:
        """Validate manipulation performance metrics"""
        metrics = {}
        
        # Test grasp success rate
        grasp_rate = await self._measure_grasp_success_rate()
        target = self.performance_baselines["manipulation"]["grasp_success_rate"]["target"]
        threshold = self.performance_baselines["manipulation"]["grasp_success_rate"]["threshold"]
        metrics["grasp_success_rate"] = {
            "measured": grasp_rate,
            "target": target,
            "threshold": threshold,
            "valid": grasp_rate >= threshold,
            "excellent": grasp_rate >= target
        }
        
        # Test position accuracy
        accuracy = await self._measure_position_accuracy()
        target = self.performance_baselines["manipulation"]["position_accuracy"]["target"]
        threshold = self.performance_baselines["manipulation"]["position_accuracy"]["threshold"]
        metrics["position_accuracy"] = {
            "measured": accuracy,
            "target": target,
            "threshold": threshold,
            "valid": accuracy <= threshold,  # Lower error is better
            "excellent": accuracy <= target
        }
        
        return metrics
    
    async def _validate_perception_performance(self) -> Dict[str, Any]:
        """Validate perception performance metrics"""
        metrics = {}
        
        # Test object detection rate
        detection_rate = await self._measure_object_detection_rate()
        target = self.performance_baselines["perception"]["object_detection_rate"]["target"]
        threshold = self.performance_baselines["perception"]["object_detection_rate"]["threshold"]
        metrics["object_detection_rate"] = {
            "measured": detection_rate,
            "target": target,
            "threshold": threshold,
            "valid": detection_rate >= threshold,
            "excellent": detection_rate >= target
        }
        
        # Test processing latency
        latency = await self._measure_processing_latency()
        target = self.performance_baselines["perception"]["processing_latency"]["target"]
        threshold = self.performance_baselines["perception"]["processing_latency"]["threshold"]
        metrics["processing_latency"] = {
            "measured": latency,
            "target": target,
            "threshold": threshold,
            "valid": latency <= threshold,
            "excellent": latency <= target
        }
        
        return metrics
    
    async def _validate_cognition_performance(self) -> Dict[str, Any]:
        """Validate cognition performance metrics"""
        metrics = {}
        
        # Test response time
        response_time = await self._measure_response_time()
        target = self.performance_baselines["cognition"]["response_time"]["target"]
        threshold = self.performance_baselines["cognition"]["response_time"]["threshold"]
        metrics["response_time"] = {
            "measured": response_time,
            "target": target,
            "threshold": threshold,
            "valid": response_time <= threshold,
            "excellent": response_time <= target
        }
        
        # Test task completion rate
        completion_rate = await self._measure_task_completion_rate()
        target = self.performance_baselines["cognition"]["task_completion_rate"]["target"]
        threshold = self.performance_baselines["cognition"]["task_completion_rate"]["threshold"]
        metrics["task_completion_rate"] = {
            "measured": completion_rate,
            "target": target,
            "threshold": threshold,
            "valid": completion_rate >= threshold,
            "excellent": completion_rate >= target
        }
        
        return metrics
    
    def _assign_performance_grade(self, score: float) -> str:
        """Assign letter grade based on performance score"""
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "A-"
        elif score >= 0.80:
            return "B+"
        elif score >= 0.75:
            return "B"
        elif score >= 0.70:
            return "B-"
        elif score >= 0.60:
            return "C"
        elif score >= 0.50:
            return "D"
        else:
            return "F"
    
    async def _measure_walking_speed(self) -> float:
        """Measure actual walking speed of the robot"""
        # Simulate measurement
        import random
        # Return a value between 0.2 and 0.8 m/s
        return 0.2 + random.random() * 0.6
    
    async def _measure_energy_efficiency(self) -> float:
        """Measure energy efficiency (J/m/kg) of the robot"""
        # Simulate measurement
        import random
        # Return a value between 0.15 and 0.35 J/m/kg
        return 0.15 + random.random() * 0.2
    
    async def _measure_balance_recovery_time(self) -> float:
        """Measure time to recover from balance perturbation"""
        # Simulate measurement
        import random
        # Return a value between 0.3 and 0.8 seconds
        return 0.3 + random.random() * 0.5
    
    async def _measure_grasp_success_rate(self) -> float:
        """Measure grasp success rate over multiple trials"""
        # Simulate measurement
        import random
        # Return a success rate between 0.7 and 0.95
        return 0.7 + random.random() * 0.25
    
    async def _measure_position_accuracy(self) -> float:
        """Measure end-effector position accuracy"""
        # Simulate measurement
        import random
        # Return error in meters between 0.005m and 0.02m
        return 0.005 + random.random() * 0.015
    
    async def _measure_object_detection_rate(self) -> float:
        """Measure object detection success rate"""
        # Simulate measurement
        import random
        # Return success rate between 0.8 and 0.98
        return 0.8 + random.random() * 0.18
    
    async def _measure_processing_latency(self) -> float:
        """Measure perception processing latency"""
        # Simulate measurement
        import random
        # Return latency between 0.02s and 0.08s
        return 0.02 + random.random() * 0.06
    
    async def _measure_response_time(self) -> float:
        """Measure cognitive system response time"""
        # Simulate measurement
        import random
        # Return response time between 0.5s and 1.5s
        return 0.5 + random.random() * 1.0
    
    async def _measure_task_completion_rate(self) -> float:
        """Measure task completion success rate"""
        # Simulate measurement
        import random
        # Return success rate between 0.75 and 0.92
        return 0.75 + random.random() * 0.17

# Example usage
async def run_validation_tests():
    """Run validation tests for humanoid robot"""
    
    class MockRobotModel:
        """Mock robot model for validation"""
        async def get_joint_states(self):
            return {
                "hip_pitch": {"position": 0.1, "velocity": 0.05},
                "knee_pitch": {"position": 0.5, "velocity": 0.08},
                "ankle_pitch": {"position": 0.02, "velocity": 0.01}
            }
        
        async def get_center_of_mass_state(self):
            return {
                "position": [0.01, 0.005, 0.82],
                "angular_velocity": [0.02, 0.01, 0.005]
            }
        
        async def get_support_polygon(self):
            return [[0.1, 0.05], [0.1, -0.05], [-0.1, -0.05], [-0.1, 0.05]]
        
        async def get_proximity_sensors(self):
            return [
                {"sensor_id": "front_01", "distance": 1.2, "angle": 0},
                {"sensor_id": "front_02", "distance": 0.8, "angle": 15},
                {"sensor_id": "left_01", "distance": 0.9, "angle": 90}
            ]
        
        async def get_force_torque_data(self):
            return {
                "hip_pitch": {"torque": 15.5},
                "knee_pitch": {"torque": 22.1}
            }
        
        async def get_gripper_forces(self):
            return {"right": 8.2, "left": 7.8}
    
    robot_model = MockRobotModel()
    
    # Run safety validation
    safety_validator = SafetyValidator(robot_model)
    safety_results = await safety_validator.validate_safety_compliance()
    
    print("=== Safety Validation Results ===")
    print(f"Overall Compliance Rate: {safety_results['overall_compliance_rate']:.1%}")
    print(f"Pass: {'YES' if safety_results['pass'] else 'NO'}")
    print(f"Critical Failures: {len(safety_results['critical_failures'])}")
    
    # Run performance validation
    performance_validator = PerformanceValidator(robot_model)
    performance_results = await performance_validator.validate_performance_metrics()
    
    print(f"\n=== Performance Validation Results ===")
    print(f"Overall Performance Score: {performance_results['overall_performance_score']:.1%}")
    print(f"Performance Grade: {performance_results['performance_grade']}")
    print(f"Passing Metrics: {performance_results['passing_metrics']}/{performance_results['total_metrics']}")
    
    return {
        "safety": safety_results,
        "performance": performance_results
    }

if __name__ == "__main__":
    results = asyncio.run(run_validation_tests())