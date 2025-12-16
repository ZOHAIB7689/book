# Humanoid Locomotion and Manipulation

## Introduction

Humanoid robots are designed to interact with human environments, requiring sophisticated locomotion and manipulation capabilities. Unlike other robot types, humanoids must navigate spaces designed for humans, manipulate objects designed for human hands, and perform tasks in ways that align with human expectations. This chapter explores the unique challenges and solutions for locomotion and manipulation in humanoid robotics, discussing both the theoretical foundations and practical implementations.

## Humanoid Locomotion Principles

### Bipedal Gait Fundamentals

Bipedal locomotion is fundamental to humanoid robotics, allowing these robots to navigate human-designed environments effectively. The key principles of bipedal gait include:

1. **Center of Mass (CoM) Control**: Maintaining the CoM within the support polygon formed by the feet
2. **Zero Moment Point (ZMP) Stability**: Ensuring the ground reaction force passes through the support polygon
3. **Dynamic Balance**: Using active control to maintain balance during movement
4. **Energy Efficiency**: Minimizing energy consumption while maintaining stable locomotion

### Gait Phases

Humanoid walking consists of distinct phases:

1. **Double Support Phase**: Both feet are in contact with the ground
2. **Single Support Phase**: Only one foot is in contact with the ground
3. **Double Support Phase**: Both feet are in contact (for smooth transition)

```python
import numpy as np
from typing import Tuple, List

class HumanoidGaitGenerator:
    def __init__(self, step_length: float = 0.3, step_height: float = 0.05, 
                 step_duration: float = 1.0, com_height: float = 0.8):
        """
        Initialize gait generator for humanoid walking
        Args:
            step_length: Distance between consecutive footsteps
            step_height: Maximum height of swinging foot
            step_duration: Time for one complete step
            com_height: Height of center of mass
        """
        self.step_length = step_length
        self.step_height = step_height
        self.step_duration = step_duration
        self.com_height = com_height
        
        # Gait phase timing (in percentage of step duration)
        self.single_support_ratio = 0.8  # 80% of time in single support
        self.double_support_ratio = 0.1  # 10% at each transition
        
        # Current gait parameters
        self.current_left_foot = np.array([0.0, 0.1, 0.0])  # Initial foot positions
        self.current_right_foot = np.array([0.0, -0.1, 0.0])
        self.current_com = np.array([0.0, 0.0, self.com_height])
        self.phase = "double_support"  # Initial phase
        self.phase_time = 0.0  # Time elapsed in current phase
    
    def calculate_foot_trajectory(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                                phase_time: float, phase_duration: float) -> np.ndarray:
        """
        Calculate smooth trajectory for foot movement
        """
        # Cubic interpolation for smooth movement
        t = phase_time / phase_duration
        
        # Calculate intermediate points
        start_vel = np.array([0.0, 0.0, 0.0])
        end_vel = np.array([0.0, 0.0, 0.0])
        
        # Cubic polynomial coefficients
        a = start_pos
        b = start_vel * phase_duration
        c = 3 * (end_pos - start_pos) - 2 * start_vel * phase_duration - end_vel * phase_duration
        d = 2 * (start_pos - end_pos) + (start_vel + end_vel) * phase_duration
        
        # Calculate position
        t2 = t * t
        t3 = t2 * t
        
        pos = a + b * t + c * t2 + d * t3
        
        # Add step height for swinging foot
        if t < 0.5:
            pos[2] += self.step_height * np.sin(np.pi * t * 2)  # Arch up
        else:
            pos[2] += self.step_height * np.sin(np.pi * (1 - t) * 2)  # Arch down
        
        return pos
    
    def generate_step_trajectory(self, direction: np.ndarray) -> List[np.ndarray]:
        """
        Generate complete trajectory for one step in the given direction
        Args:
            direction: Normalized direction vector for the step
        Returns:
            List of positions for the stepping foot throughout the step
        """
        # Calculate target positions
        support_foot = self.current_left_foot if self.phase == "right_swing" else self.current_right_foot
        target_pos = support_foot + np.append(direction * self.step_length, [0])
        
        # Calculate trajectory for double support -> single support -> double support
        trajectory = []
        
        # Phase 1: Double support (first part)
        for i in range(int(self.double_support_ratio * self.step_duration * 100)):
            t = i / (self.double_support_ratio * self.step_duration * 100)
            trajectory.append(support_foot.copy())
        
        # Phase 2: Single support
        swing_foot = self.current_right_foot if self.phase == "right_swing" else self.current_left_foot
        for i in range(int(self.single_support_ratio * self.step_duration * 100)):
            t = i / (self.single_support_ratio * self.step_duration * 100)
            pos = self.calculate_foot_trajectory(swing_foot, target_pos, t, self.single_support_ratio * self.step_duration)
            trajectory.append(pos)
        
        # Phase 3: Double support (second part)
        for i in range(int(self.double_support_ratio * self.step_duration * 100)):
            t = i / (self.double_support_ratio * self.step_duration * 100)
            trajectory.append(target_pos.copy())
        
        return trajectory
    
    def update_com_position(self, time_step: float) -> np.ndarray:
        """
        Update Center of Mass position based on inverted pendulum model
        """
        # Simplified inverted pendulum model for CoM tracking
        # In real implementation, this would use more sophisticated models like LIPM (Linear Inverted Pendulum Model)
        
        # For forward walking, move CoM forward to support the next step
        if self.phase == "left_swing" or self.phase == "right_swing":
            # Move CoM in the direction of the step
            step_dir = (self.current_right_foot - self.current_left_foot) if self.phase == "right_swing" else (self.current_left_foot - self.current_right_foot)
            step_dir = step_dir / np.linalg.norm(step_dir[:2])  # Normalize to 2D
            step_dir = np.append(step_dir, [0])  # Make 3D
            
            self.current_com += step_dir * self.step_length * time_step / self.step_duration
        
        return self.current_com

# Example usage
gait_gen = HumanoidGaitGenerator()
step_trajectory = gait_gen.generate_step_trajectory(np.array([1.0, 0.0]))  # Step forward
print(f"Generated {len(step_trajectory)} trajectory points for one step")
```

### Walking Pattern Generation

Walking pattern generation involves creating coordinated movements for all joints:

```python
import numpy as np
from scipy import signal

class WalkingPatternGenerator:
    def __init__(self, step_length: float = 0.3, step_height: float = 0.05, 
                 step_duration: float = 1.0, hip_width: float = 0.2):
        """
        Generate coordinated walking patterns for all joints
        """
        self.step_length = step_length
        self.step_height = step_height
        self.step_duration = step_duration
        self.hip_width = hip_width  # Distance between hip joints
        
        # Joint angle parameters for natural walking
        self.joint_defaults = {
            'hip_pitch': 0.0,      # Forward/back tilt
            'hip_roll': 0.0,       # Side tilt
            'hip_yaw': 0.0,        # Rotation
            'knee': 0.0,           # Knee bend
            'ankle_pitch': 0.0,    # Foot tilt
            'ankle_roll': 0.0      # Ankle side tilt
        }
    
    def generate_joint_trajectories(self, num_steps: int) -> dict:
        """
        Generate joint angle trajectories for multiple steps
        """
        # Time vector for one step
        dt = 0.01  # 10ms time steps
        time_steps = int(self.step_duration / dt)
        time_vec = np.linspace(0, self.step_duration, time_steps)
        
        trajectories = {}
        
        # Generate patterns for each joint type
        for joint_type in self.joint_defaults.keys():
            trajectories[joint_type] = []
            
            for step in range(num_steps):
                # Generate pattern for this step
                step_pattern = self._generate_joint_pattern(joint_type, time_vec)
                trajectories[joint_type].extend(step_pattern)
        
        return trajectories
    
    def _generate_joint_pattern(self, joint_type: str, time_vec: np.ndarray) -> np.ndarray:
        """
        Generate pattern for a specific joint type
        """
        pattern = np.zeros_like(time_vec)
        
        if joint_type == 'hip_pitch':
            # Hip pitch follows the forward motion with slight forward lean
            # Sine wave pattern synchronized with gait cycle
            phase_shift = 0  # No phase shift for hip pitch
            pattern = 0.1 * np.sin(2 * np.pi * time_vec / self.step_duration + phase_shift)
            
            # Add forward lean for stability
            pattern += -0.05  # Forward lean for stability
            
        elif joint_type == 'hip_roll':
            # Hip roll alternates between steps for stability
            # Sine wave pattern shifted to create lateral sway
            phase_shift = np.pi / 2
            sway_amplitude = 0.05
            pattern = sway_amplitude * np.sin(2 * np.pi * time_vec / self.step_duration + phase_shift)
            
        elif joint_type == 'knee':
            # Knee flexes during swing phase, extends during stance
            # Use a combination of sine waves and step functions
            swing_phase = signal.sawtooth(2 * np.pi * time_vec / self.step_duration, 0.5)  # Square wave pattern
            knee_flexion = 0.3 * (1 + swing_phase) / 2  # Positive-only values
            
            # Add natural flexion curve during swing
            mid_swing = time_vec / self.step_duration
            knee_flexion += 0.1 * np.sin(np.pi * mid_swing)  # Additional flexion in mid-swing
            
            pattern = knee_flexion
            
        elif joint_type == 'ankle_pitch':
            # Ankle adjusts to keep the foot parallel to ground
            # Inverted pattern to hip pitch
            phase_shift = np.pi
            pattern = 0.05 * np.sin(2 * np.pi * time_vec / self.step_duration + phase_shift)
            
        elif joint_type == 'ankle_roll':
            # Ankle rolls to maintain balance, opposite to hip roll
            phase_shift = 0
            pattern = -0.03 * np.sin(2 * np.pi * time_vec / self.step_duration + phase_shift)
        
        # Add some natural variability
        noise = 0.01 * np.random.normal(0, 1, len(time_vec))
        pattern += noise
        
        return pattern

# Example usage
pattern_gen = WalkingPatternGenerator()
joint_patterns = pattern_gen.generate_joint_trajectories(num_steps=2)

print("Generated walking patterns for 2 steps:")
for joint, trajectory in joint_patterns.items():
    print(f"  {joint}: {len(trajectory)} points, range [{min(trajectory):.3f}, {max(trajectory):.3f}]")
```

## Balance Control Systems

### Zero Moment Point (ZMP) Control

ZMP control is a fundamental approach to maintaining balance in humanoid robots:

```python
import numpy as np
from scipy import signal

class ZMPController:
    def __init__(self, com_height: float = 0.8, gravity: float = 9.81):
        """
        Zero Moment Point (ZMP) controller for humanoid balance
        Args:
            com_height: Height of center of mass above ground
            gravity: Gravitational acceleration
        """
        self.com_height = com_height
        self.gravity = gravity
        self.com = np.array([0.0, 0.0, com_height])  # Initial CoM position
        self.com_velocity = np.array([0.0, 0.0, 0.0])
        self.com_acceleration = np.array([0.0, 0.0, 0.0])
        
        # Control parameters
        self.kp = 10.0  # Proportional gain
        self.ki = 1.0   # Integral gain
        self.kd = 5.0   # Derivative gain
        
        # Error history for integral term
        self.error_history = []
        self.max_history = 10  # Keep last 10 errors
    
    def compute_zmp(self, com_pos: np.ndarray, com_acc: np.ndarray) -> np.ndarray:
        """
        Compute ZMP position from CoM position and acceleration
        ZMP_x = CoM_x - h/g * CoM_acc_x
        ZMP_y = CoM_y - h/g * CoM_acc_y
        """
        zmp = np.zeros(3)  # We only care about x, y coordinates
        zmp[:2] = com_pos[:2] - (self.com_height / self.gravity) * com_acc[:2]
        zmp[2] = 0  # ZMP is on ground plane
        
        return zmp
    
    def control_step(self, desired_zmp: np.ndarray, actual_zmp: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform one control step to move actual ZMP to desired ZMP
        Args:
            desired_zmp: Desired ZMP position
            actual_zmp: Current ZMP position (from sensors)
            dt: Time step
        Returns:
            Force/torque command to apply to achieve balance
        """
        # Calculate error
        error = desired_zmp[:2] - actual_zmp[:2]
        
        # Add to error history for integral term
        self.error_history.append(error.copy())
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Calculate PID terms
        p_term = self.kp * error
        
        # Integral term
        if len(self.error_history) > 0:
            integral_error = np.sum(self.error_history, axis=0) * dt
            i_term = self.ki * integral_error
        else:
            i_term = np.zeros(2)
        
        # Derivative term
        if len(self.error_history) > 1:
            derivative_error = (self.error_history[-1] - self.error_history[-2]) / dt
            d_term = self.kd * derivative_error
        else:
            d_term = np.zeros(2)
        
        # Total control output
        control_output = p_term + i_term + d_term
        
        # Calculate required CoM acceleration to achieve desired ZMP
        required_com_acc = np.zeros(3)
        required_com_acc[:2] = -self.gravity / self.com_height * (actual_zmp[:2] - desired_zmp[:2])
        
        return required_com_acc

class CapturePointController:
    """
    Capture Point controller - an alternative to ZMP control
    A capture point is where a point mass would come to rest if it continued with its current velocity
    """
    def __init__(self, com_height: float = 0.8, gravity: float = 9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.sqrt_g_h = np.sqrt(gravity / com_height)
        
    def compute_capture_point(self, com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        """
        Calculate capture point from CoM position and velocity
        Capture Point = CoM_pos + CoM_vel / sqrt(g/h)
        """
        cp = np.zeros(3)
        cp[:2] = com_pos[:2] + com_vel[:2] / self.sqrt_g_h
        return cp
    
    def compute_balance_policy(self, capture_point: np.ndarray, target_point: np.ndarray) -> np.ndarray:
        """
        Compute balance policy based on capture point
        Returns the CoM acceleration needed to move capture point to target
        """
        # Acceleration needed to bring capture point to target
        acc = -self.sqrt_g_h * (capture_point[:2] - target_point[:2])
        
        # Return as acceleration vector
        com_acc = np.zeros(3)
        com_acc[:2] = acc
        return com_acc

# Example usage
zmp_controller = ZMPController()
cp_controller = CapturePointController()

# Simulate a simple balance correction
com_pos = np.array([0.01, 0.0, 0.8])  # Robot is slightly off-center
com_vel = np.array([0.0, 0.01, 0.0])  # With some velocity
com_acc = np.array([0.0, 0.0, 0.0])

# Compute current ZMP
current_zmp = zmp_controller.compute_zmp(com_pos, com_acc)
print(f"Current ZMP: {current_zmp[:2]}")

# Compute capture point
cp = cp_controller.compute_capture_point(com_pos, com_vel)
print(f"Current Capture Point: {cp[:2]}")

# Compute control to bring ZMP to center (0,0)
desired_zmp = np.array([0.0, 0.0, 0.0])
control_acc = zmp_controller.control_step(desired_zmp, current_zmp, 0.01)
print(f"Required CoM acceleration: {control_acc[:2]}")
```

### Whole-Body Control

For humanoid robots, balance requires coordination of all joints:

```python
import numpy as np
from typing import List, Dict

class WholeBodyController:
    def __init__(self, robot_config: Dict):
        """
        Whole body controller for coordinating all joints for balance
        Args:
            robot_config: Configuration dictionary with joint information
        """
        self.robot_config = robot_config
        self.joint_names = robot_config.get("joint_names", [])
        self.mass_properties = robot_config.get("mass_properties", {})
        
        # Initialize joint positions with neutral stance
        self.joint_positions = {name: 0.0 for name in self.joint_names}
        
        # Balance parameters
        self.com_position = np.array([0.0, 0.0, 0.8])
        self.support_polygon = []  # Vertices of support polygon
        
    def compute_jacobian(self, joint_angles: Dict[str, float]) -> np.ndarray:
        """
        Compute Jacobian matrix of end-effectors with respect to joint angles
        This is a simplified version - in practice, would use kinematic models
        """
        n_joints = len(self.joint_names)
        n_end_effs = 2  # Two feet
        
        # Initialize Jacobian (6 DoF per end-effector * 2 feet, n_joints)
        jacobian = np.zeros((n_end_effs * 6, n_joints))
        
        # Fill Jacobian based on kinematic model
        # This is a simplified approach - real implementation would use forward kinematics
        for i, joint_name in enumerate(self.joint_names):
            # Derivative of end-effector position w.r.t. each joint
            # Simplified: assume each joint affects end-effector with some influence
            if "ankle" in joint_name:
                jacobian[0:3, i] = [1.0, 0.0, 0.0]  # x, y, z translation
            elif "knee" in joint_name:
                jacobian[0:3, i] = [0.5, 0.0, 0.0]  # Less influence on translation
            # Add more joint-specific mappings
        
        return jacobian
    
    def compute_balance_control(self, target_com: np.ndarray) -> Dict[str, float]:
        """
        Compute joint position adjustments to achieve target CoM
        """
        # Calculate CoM error
        current_com = self._estimate_com(self.joint_positions)
        com_error = target_com - current_com
        
        # Compute required joint adjustments using Jacobian transpose method
        jacobian = self.compute_jacobian(self.joint_positions)
        
        # Use Jacobian transpose with damping for pseudoinverse
        damping = 0.01
        damped_jacobian = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping * np.eye(12))
        
        # Calculate joint adjustments
        joint_adjustments = damped_jacobian @ np.concatenate([com_error, np.zeros(3)])  # 3 for orientation
        
        # Apply adjustments to current positions
        new_positions = {}
        for i, joint_name in enumerate(self.joint_names):
            new_positions[joint_name] = self.joint_positions[joint_name] + joint_adjustments[i]
        
        return new_positions
    
    def _estimate_com(self, joint_pos: Dict[str, float]) -> np.ndarray:
        """
        Estimate center of mass from joint positions and mass properties
        """
        total_mass = sum(self.mass_properties.get(joint_name, 1.0) for joint_name in joint_pos.keys())
        weighted_sum = np.zeros(3)
        
        # This is a simplified CoM calculation
        # In real implementation, would use actual link masses and positions
        for joint_name, pos in joint_pos.items():
            mass = self.mass_properties.get(joint_name, 1.0)
            # Simplified position calculation
            weighted_sum[2] += mass * 0.8  # Assume all links at same height
        
        com = weighted_sum / total_mass if total_mass > 0 else np.zeros(3)
        return com

# Example usage
robot_config = {
    "joint_names": ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", 
                    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
                    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", 
                    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll"],
    "mass_properties": {
        "left_hip_yaw": 2.0, "left_hip_roll": 2.0, "left_hip_pitch": 2.0,
        "left_knee_pitch": 1.5, "left_ankle_pitch": 1.0, "left_ankle_roll": 1.0,
        "right_hip_yaw": 2.0, "right_hip_roll": 2.0, "right_hip_pitch": 2.0,
        "right_knee_pitch": 1.5, "right_ankle_pitch": 1.0, "right_ankle_roll": 1.0
    }
}

wbc = WholeBodyController(robot_config)
target_com = np.array([0.0, 0.0, 0.8])
joint_commands = wbc.compute_balance_control(target_com)

print("Computed joint positions for balance:")
for joint, pos in list(joint_commands.items())[:6]:  # Show first few joints
    print(f"  {joint}: {pos:.3f}")
```

## Manipulation Principles

### Grasping and Manipulation Fundamentals

Humanoid manipulation involves grasping objects using anthropomorphic hands and manipulating them in 3D space:

```python
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class GraspPoint:
    position: np.ndarray  # (x, y, z) in robot frame
    normal: np.ndarray    # Normal vector pointing out from object surface
    approach: np.ndarray  # Approach direction for grasping
    width: float         # Required gripper width

class GraspController:
    def __init__(self, hand_dof: int = 16):
        """
        Controller for humanoid hand grasping
        Args:
            hand_dof: Degrees of freedom in the hand to control
        """
        self.hand_dof = hand_dof
        self.max_gripper_width = 0.1  # 10cm max opening
        self.max_force = 50  # Max force in Newtons
    
    def plan_grasp_sequence(self, grasp_point: GraspPoint) -> List[Dict]:
        """
        Plan the sequence of movements to achieve a grasp
        Returns a list of actions with timing
        """
        sequence = []
        
        # 1. Approach with hand above the object
        sequence.append({
            "action": "move_to_approach",
            "position": grasp_point.position + grasp_point.approach * 0.1,  # 10cm above
            "orientation": self._calculate_approach_orientation(grasp_point),
            "gripper_width": self.max_gripper_width,  # Fully open
            "duration": 2.0  # seconds
        })
        
        # 2. Align with grasp point
        sequence.append({
            "action": "align_with_object",
            "position": grasp_point.position + grasp_point.normal * 0.02,  # 2cm from surface
            "orientation": self._calculate_grasp_orientation(grasp_point),
            "gripper_width": self.max_gripper_width,
            "duration": 1.0
        })
        
        # 3. Make contact
        sequence.append({
            "action": "move_to_contact",
            "position": grasp_point.position,
            "orientation": self._calculate_grasp_orientation(grasp_point),
            "gripper_width": grasp_point.width * 0.9,  # Slightly tighter than object width
            "duration": 0.5
        })
        
        # 4. Close gripper
        sequence.append({
            "action": "close_gripper",
            "gripper_width": grasp_point.width * 0.7,  # Apply force
            "force": min(20, self.max_force * 0.8),  # Apply grip force
            "duration": 0.3
        })
        
        # 5. Lift object
        sequence.append({
            "action": "lift_object",
            "position": grasp_point.position + np.array([0, 0, 0.05]),  # Lift 5cm
            "orientation": self._calculate_grasp_orientation(grasp_point),
            "duration": 1.0
        })
        
        return sequence
    
    def _calculate_approach_orientation(self, grasp_point: GraspPoint) -> np.ndarray:
        """
        Calculate approach orientation based on grasp normal
        """
        # For a simple approach, align gripper perpendicular to surface
        normal = grasp_point.normal
        approach_dir = grasp_point.approach
        
        # Calculate rotation to align gripper with normal
        # This is a simplified approach - in reality would solve for full orientation
        
        # Assuming gripper closes along z-axis of gripper frame
        # Try to align gripper z with surface normal
        z_axis = normal / np.linalg.norm(normal)
        
        # Choose an arbitrary x-axis perpendicular to z
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        else:
            x_axis = np.cross(z_axis, np.array([1, 0, 0]))
            
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Construct rotation matrix
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert to Euler angles (simplified)
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arcsin(-rotation_matrix[2, 0])
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        return np.array([roll, pitch, yaw])
    
    def _calculate_grasp_orientation(self, grasp_point: GraspPoint) -> np.ndarray:
        """
        Calculate final grasp orientation
        """
        # Similar to approach but with more precise alignment
        return self._calculate_approach_orientation(grasp_point)

class ManipulationController:
    def __init__(self, robot_arm_dof: int = 7):
        """
        Controller for arm manipulation tasks
        Args:
            robot_arm_dof: Degrees of freedom in the arm to control
        """
        self.arm_dof = robot_arm_dof
        self.reach_distance = 0.8  # Max reach distance
        self.workspace_bounds = {
            "x": [-0.5, 0.5],  # Reachable x range
            "y": [-0.5, 0.5],  # Reachable y range
            "z": [0.1, 1.2]    # Reachable z range
        }
    
    def plan_reach_trajectory(self, start_pos: np.ndarray, end_pos: np.ndarray) -> List[np.ndarray]:
        """
        Plan a smooth trajectory between two points
        """
        # Check if both points are within workspace
        if not self._is_in_workspace(start_pos) or not self._is_in_workspace(end_pos):
            raise ValueError("Start or end position outside workspace")
        
        # Create linear trajectory with intermediate waypoints
        num_waypoints = 20
        trajectory = []
        
        for t in np.linspace(0, 1, num_waypoints):
            pos = start_pos + t * (end_pos - start_pos)
            
            # Add slight arc to avoid obstacles
            if t > 0.1 and t < 0.9:  # Middle 80% of trajectory
                # Add small vertical displacement for arc
                arc_height = 0.05 * np.sin(np.pi * t)  # Up to 5cm arc
                pos[2] += arc_height
                
            trajectory.append(pos)
        
        return trajectory
    
    def plan_manipulation_sequence(self, task: str, object_pos: np.ndarray, 
                                 target_pos: np.ndarray) -> List[Dict]:
        """
        Plan a complete manipulation sequence
        Args:
            task: "pick_and_place", "move_to", etc.
            object_pos: Position of object to manipulate
            target_pos: Target position for manipulation
        Returns:
            Sequence of actions to complete the task
        """
        if task == "pick_and_place":
            # Pre-grasp position (above object)
            pre_grasp = object_pos + np.array([0, 0, 0.2])  # 20cm above object
            
            # Post-place position (above target)
            post_place = target_pos + np.array([0, 0, 0.2])  # 20cm above target
            
            return [
                {
                    "action": "move_to_pre_grasp",
                    "target": pre_grasp,
                    "gripper": "open",
                    "description": "Move arm to position above object"
                },
                {
                    "action": "move_to_object",
                    "target": object_pos,
                    "gripper": "pre_close",
                    "description": "Approach and grasp object"
                },
                {
                    "action": "lift_object",
                    "target": pre_grasp,
                    "gripper": "closed",
                    "description": "Lift object from surface"
                },
                {
                    "action": "move_to_target",
                    "target": post_place,
                    "gripper": "closed",
                    "description": "Move object to target location"
                },
                {
                    "action": "place_object",
                    "target": target_pos,
                    "gripper": "open",
                    "description": "Release object at target"
                },
                {
                    "action": "retract",
                    "target": post_place + np.array([0, 0, 0.1]),
                    "gripper": "open",
                    "description": "Retract from placed object"
                }
            ]
        
        return []
    
    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """
        Check if a position is within the robot's workspace
        """
        x, y, z = pos
        return (self.workspace_bounds["x"][0] <= x <= self.workspace_bounds["x"][1] and
                self.workspace_bounds["y"][0] <= y <= self.workspace_bounds["y"][1] and
                self.workspace_bounds["z"][0] <= z <= self.workspace_bounds["z"][1])

# Example usage
grasp_controller = GraspController()
manip_controller = ManipulationController()

# Define a grasp point for an object
grasp_point = GraspPoint(
    position=np.array([0.3, 0.1, 0.1]),
    normal=np.array([0, 0, 1]),  # Pointing up
    approach=np.array([0, 0, -1]),  # Approach from above
    width=0.04  # 4cm object width
)

# Plan grasp sequence
grasp_sequence = grasp_controller.plan_grasp_sequence(grasp_point)
print(f"Grasp sequence has {len(grasp_sequence)} steps")

# Plan manipulation sequence
manip_sequence = manip_controller.plan_manipulation_sequence(
    "pick_and_place",
    np.array([0.3, 0.1, 0.1]),  # Object position
    np.array([0.4, -0.2, 0.2])  # Target position
)
print(f"Manipulation sequence has {len(manip_sequence)} steps")
```

## Locomotion Gaits and Patterns

### Different Walking Styles

Humanoid robots may need different gaits for different situations:

```python
import numpy as np
from typing import Dict, List

class GaitLibrary:
    def __init__(self):
        """
        Library of different walking gaits for humanoid robots
        """
        self.gaits = {
            "normal_walk": self._generate_normal_walk,
            "slow_walk": self._generate_slow_walk,
            "fast_walk": self._generate_fast_walk,
            "cautious_walk": self._generate_cautious_walk,
            "trot": self._generate_trot,
        }
        
    def _generate_normal_walk(self, step_length: float = 0.3, step_height: float = 0.05, 
                            step_duration: float = 1.0) -> Dict[str, List[float]]:
        """
        Generate normal walking pattern
        """
        dt = 0.01
        time_vec = np.linspace(0, step_duration, int(step_duration/dt))
        
        # Hip trajectory (lateral sway for stability)
        hip_lateral = 0.05 * np.sin(2 * np.pi * time_vec / step_duration + np.pi/2)
        
        # Hip forward motion
        hip_forward = step_length/2 * (1 - np.cos(2 * np.pi * time_vec / step_duration))
        
        # Hip height (slight up/down motion)
        hip_height = 0.8 + 0.01 * np.sin(4 * np.pi * time_vec / step_duration)
        
        return {
            "hip_lateral": hip_lateral,
            "hip_forward": hip_forward,
            "hip_height": hip_height
        }
    
    def _generate_slow_walk(self, step_length: float = 0.2, step_height: float = 0.03, 
                          step_duration: float = 1.5) -> Dict[str, List[float]]:
        """
        Generate slow, careful walking pattern
        """
        dt = 0.01
        time_vec = np.linspace(0, step_duration, int(step_duration/dt))
        
        # More pronounced lateral sway for stability
        hip_lateral = 0.07 * np.sin(2 * np.pi * time_vec / step_duration + np.pi/2)
        
        # Slower forward motion
        hip_forward = step_length/2 * (1 - np.cos(2 * np.pi * time_vec / step_duration))
        
        # Higher step height for better clearance
        hip_height = 0.8 + 0.02 * np.sin(4 * np.pi * time_vec / step_duration)
        
        return {
            "hip_lateral": hip_lateral,
            "hip_forward": hip_forward,
            "hip_height": hip_height
        }
    
    def _generate_fast_walk(self, step_length: float = 0.4, step_height: float = 0.07, 
                          step_duration: float = 0.7) -> Dict[str, List[float]]:
        """
        Generate faster walking pattern
        """
        dt = 0.01
        time_vec = np.linspace(0, step_duration, int(step_duration/dt))
        
        # Reduced lateral sway for speed
        hip_lateral = 0.03 * np.sin(2 * np.pi * time_vec / step_duration + np.pi/2)
        
        # Faster forward motion
        hip_forward = step_length/2 * (1 - np.cos(2 * np.pi * time_vec / step_duration))
        
        # More dynamic height variation
        hip_height = 0.8 + 0.03 * np.sin(4 * np.pi * time_vec / step_duration)
        
        return {
            "hip_lateral": hip_lateral,
            "hip_forward": hip_forward,
            "hip_height": hip_height
        }
    
    def _generate_cautious_walk(self, step_length: float = 0.15, step_height: float = 0.08, 
                              step_duration: float = 1.8) -> Dict[str, List[float]]:
        """
        Generate very careful walking for uncertain terrain
        """
        dt = 0.01
        time_vec = np.linspace(0, step_duration, int(step_duration/dt))
        
        # Maximum lateral sway for stability
        hip_lateral = 0.08 * np.sin(2 * np.pi * time_vec / step_duration + np.pi/2)
        
        # Very slow, cautious forward motion
        hip_forward = step_length/2 * (1 - np.cos(2 * np.pi * time_vec / step_duration))
        
        # High step height for obstacle clearance
        hip_height = 0.8 + 0.02 * np.sin(4 * np.pi * time_vec / step_duration)
        
        return {
            "hip_lateral": hip_lateral,
            "hip_forward": hip_forward,
            "hip_height": hip_height
        }
    
    def get_gait_pattern(self, gait_type: str, **kwargs) -> Dict[str, List[float]]:
        """
        Get a specific gait pattern
        """
        if gait_type in self.gaits:
            return self.gaits[gait_type](**kwargs)
        else:
            raise ValueError(f"Unknown gait type: {gait_type}")

# Example usage
gait_lib = GaitLibrary()

# Generate different gait patterns
normal_pattern = gait_lib.get_gait_pattern("normal_walk")
slow_pattern = gait_lib.get_gait_pattern("slow_walk")
fast_pattern = gait_lib.get_gait_pattern("fast_walk")
cautious_pattern = gait_lib.get_gait_pattern("cautious_walk")

print("Gait patterns generated:")
print(f"  Normal walk: {len(normal_pattern['hip_forward'])} points")
print(f"  Slow walk: {len(slow_pattern['hip_forward'])} points")
print(f"  Fast walk: {len(fast_pattern['hip_forward'])} points")
print(f"  Cautious walk: {len(cautious_pattern['hip_forward'])} points")
```

## Integration of Locomotion and Manipulation

### Coordinated Movement Planning

For effective humanoid operation, locomotion and manipulation must be coordinated:

```python
import numpy as np
from typing import Dict, List, Tuple

class CoordinatedMotionPlanner:
    def __init__(self):
        """
        Plan coordinated locomotion and manipulation actions
        """
        self.locomotion_planner = HumanoidGaitGenerator()
        self.manipulation_planner = ManipulationController()
        self.balance_controller = ZMPController()
    
    def plan_coordinated_task(self, task_description: Dict) -> Dict[str, List]:
        """
        Plan a task that requires both locomotion and manipulation
        Args:
            task_description: Dictionary describing the task
        Returns:
            Dictionary with locomotion and manipulation plans
        """
        task_type = task_description.get("type", "")
        
        if task_type == "fetch_object":
            return self._plan_fetch_task(task_description)
        elif task_type == "set_table":
            return self._plan_table_setting_task(task_description)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _plan_fetch_task(self, task_description: Dict) -> Dict[str, List]:
        """
        Plan a task to fetch an object from one location to another
        """
        start_pos = task_description["robot_start_position"]
        object_pos = task_description["object_position"] 
        goal_pos = task_description["delivery_position"]
        
        plan = {
            "navigation_to_object": [],
            "grasping_sequence": [],
            "navigation_to_goal": [],
            "delivery_sequence": [],
            "balance_adjustments": []
        }
        
        # 1. Navigate to object
        if np.linalg.norm(start_pos - object_pos) > 0.5:  # If far away
            path_to_object = self._plan_navigation_path(start_pos, object_pos)
            plan["navigation_to_object"] = path_to_object
        
        # 2. Grasp object
        grasp_sequence = self.manipulation_planner.plan_manipulation_sequence(
            "pick_and_place", object_pos, object_pos  # For grasping at same location
        )
        plan["grasping_sequence"] = grasp_sequence
        
        # 3. Navigate to goal location (with object)
        path_to_goal = self._plan_navigation_path(object_pos, goal_pos)
        plan["navigation_to_goal"] = path_to_goal
        
        # 4. Deliver object
        delivery_sequence = self.manipulation_planner.plan_manipulation_sequence(
            "pick_and_place", object_pos, goal_pos  # Place at goal location
        )
        plan["delivery_sequence"] = delivery_sequence
        
        # 5. Plan balance adjustments for carrying object
        plan["balance_adjustments"] = self._calculate_balance_adjustments(
            len(path_to_goal), object_weight=task_description.get("object_weight", 0.5)
        )
        
        return plan
    
    def _plan_navigation_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """
        Plan a navigation path between two points
        """
        # Simplified path planning - in reality would use A*, RRT, etc.
        steps = max(int(np.linalg.norm(goal - start) / 0.3), 3)  # ~30cm per step
        path = []
        
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            pos = start + t * (goal - start)
            path.append(pos)
        
        return path
    
    def _calculate_balance_adjustments(self, num_steps: int, object_weight: float = 0.5) -> List[Dict]:
        """
        Calculate balance adjustments needed when carrying an object
        """
        adjustments = []
        
        # When carrying an object, CoM shifts - need to adjust stance
        for i in range(num_steps):
            # Calculate CoM offset based on object position relative to robot
            # This is simplified - in practice would consider exact object location
            
            adjustment = {
                "step_index": i,
                "com_offset": np.array([-0.02 * object_weight, 0.0, 0.0]),  # Shift backward to compensate
                "foot_position_offset": np.array([0.01 * object_weight, 0.0, 0.0]),  # Adjust foot position
                "required_stability_margin": 0.05  # Increase stability margin
            }
            adjustments.append(adjustment)
        
        return adjustments

# Example usage
coordinated_planner = CoordinatedMotionPlanner()

# Define a fetch task
fetch_task = {
    "type": "fetch_object",
    "robot_start_position": np.array([0.0, 0.0, 0.0]),
    "object_position": np.array([1.0, 0.5, 0.1]),
    "delivery_position": np.array([2.0, -0.5, 0.1]),
    "object_weight": 0.8  # 800g object
}

coordinated_plan = coordinated_planner.plan_coordinated_task(fetch_task)

print("Coordinated task plan:")
for key, value in coordinated_plan.items():
    print(f"  {key}: {len(value) if isinstance(value, list) else 'N/A'} elements")
```

## Exercises

1. **Gait Generation Exercise**: Implement different walking gaits (normal, slow, fast) for a humanoid model and evaluate their stability and energy efficiency.

2. **Balance Control Exercise**: Create a balance controller that can handle external disturbances and maintain stability during dynamic movements.

3. **Grasping Strategy Exercise**: Develop a system that can choose appropriate grasp points and strategies for objects of various shapes and sizes.

4. **Coordinated Motion Exercise**: Design and implement a system that coordinates locomotion with manipulation for a complex task like fetching and delivering an object.

5. **Terrain Adaptation Exercise**: Modify locomotion patterns to handle different terrains (slopes, stairs, uneven surfaces).

6. **Real-Time Adaptation Exercise**: Create a system that can modify ongoing movements based on real-time sensory feedback.

7. **Multi-Object Manipulation Exercise**: Extend manipulation capabilities to handle multiple objects simultaneously.

## Summary

Humanoid locomotion and manipulation represent some of the most challenging aspects of humanoid robotics. Successfully implementing these capabilities requires understanding of biomechanics, control theory, and the unique challenges of anthropomorphic movement. Advanced humanoid robots require sophisticated coordination between locomotion and manipulation systems to perform complex tasks in human environments effectively. The integration of balance control, gait generation, and manipulation planning enables humanoid robots to operate in diverse and unstructured environments.