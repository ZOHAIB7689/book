---
title: Balance Control for Humanoid Robots
sidebar_position: 1
---

# Balance Control for Humanoid Robots

## Introduction

Balance control is fundamental to humanoid robotics, as these robots must maintain stability while performing tasks in human environments. Unlike wheeled or tracked robots, humanoid robots are inherently unstable with a high center of mass and small support base. Effective balance control enables humanoid robots to walk, manipulate objects, and respond to disturbances while maintaining stable operation.

## Principles of Humanoid Balance

### Static vs. Dynamic Balance

Humanoid robots can maintain balance through two primary mechanisms:

1. **Static Balance**: Maintaining the center of mass (CoM) within the support polygon at all times
2. **Dynamic Balance**: Using motion to maintain balance, allowing the CoM to move outside the support polygon temporarily

Static balance is more stable but limits the robot's range of motion, while dynamic balance allows for more natural movement but requires sophisticated control algorithms.

### Support Polygon

The support polygon is the convex hull of all contact points with the ground. For a bipedal robot:
- **Double support**: Convex hull of both feet
- **Single support**: Area of the stance foot
- **Multi-contact**: Includes hands or other contact points during complex tasks

```python
import numpy as np
from typing import List, Tuple

class SupportPolygon:
    def __init__(self, contact_points: List[np.ndarray]):
        """
        Represents the support polygon based on contact points
        Args:
            contact_points: List of 2D points representing ground contact points
        """
        self.contact_points = contact_points
        self.vertices = self._compute_convex_hull(contact_points)
    
    def _compute_convex_hull(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute convex hull of contact points using Graham scan
        Simplified implementation for 2D points
        """
        if len(points) < 3:
            return points
        
        # Find the point with lowest y-coordinate (or leftmost in case of tie)
        start = min(points, key=lambda p: (p[1], p[0]))
        
        # Sort points by polar angle
        def polar_angle(p):
            # Calculate polar angle relative to start point
            if p[0] == start[0]:
                return np.pi / 2 if p[1] > start[1] else 3 * np.pi / 2
            angle = np.arctan2(p[1] - start[1], p[0] - start[0])
            return angle if angle >= 0 else angle + 2 * np.pi
        
        sorted_points = sorted(points, key=polar_angle)
        
        # Apply Graham scan
        hull = [sorted_points[0], sorted_points[1]]
        
        for point in sorted_points[2:]:
            while len(hull) > 1 and self._cross_product(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
        
        return hull
    
    def _cross_product(self, o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cross product of vectors OA and OB
        Positive if point B is counter-clockwise from A relative to O
        """
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a 2D point is inside the support polygon
        Uses ray-casting algorithm
        """
        x, y = point[0], point[1]
        n = len(self.vertices)
        inside = False
        
        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_centroid(self) -> np.ndarray:
        """
        Calculate centroid of the polygon
        """
        if len(self.vertices) == 0:
            return np.array([0.0, 0.0])
        
        cx, cy = 0.0, 0.0
        for vertex in self.vertices:
            cx += vertex[0]
            cy += vertex[1]
        
        n = len(self.vertices)
        return np.array([cx / n, cy / n])
    
    def get_area(self) -> float:
        """
        Calculate area of the polygon using the shoelace formula
        """
        if len(self.vertices) < 3:
            return 0.0
        
        area = 0.0
        n = len(self.vertices)
        
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        
        return abs(area) / 2.0

# Example usage
contact_points = [
    np.array([0.1, 0.05]),   # left foot front
    np.array([0.1, -0.05]),  # left foot back
    np.array([-0.1, 0.05]),  # right foot front
    np.array([-0.1, -0.05])  # right foot back
]

support_polygon = SupportPolygon(contact_points)
com_position = np.array([0.0, 0.0])

is_stable = support_polygon.contains_point(com_position)
print(f"CoM at {com_position} is {'stable' if is_stable else 'unstable'} with respect to support polygon")
print(f"Support polygon area: {support_polygon.get_area():.4f}")
```

### Zero Moment Point (ZMP) Theory

The Zero Moment Point (ZMP) is a critical concept in humanoid balance:

```python
class ZMPCalculator:
    def __init__(self, com_height: float = 0.8, gravity: float = 9.81):
        """
        Calculate Zero Moment Point for humanoid balance
        Args:
            com_height: Height of center of mass above ground (meters)
            gravity: Gravitational acceleration (m/s^2)
        """
        self.com_height = com_height
        self.gravity = gravity
    
    def calculate_zmp(self, com_pos: np.ndarray, com_accel: np.ndarray) -> np.ndarray:
        """
        Calculate ZMP position from CoM position and acceleration
        ZMP = CoM - (h/g) * CoM_acc
        Args:
            com_pos: Center of mass position [x, y, z]
            com_accel: Center of mass acceleration [x_acc, y_acc, z_acc]
        Returns:
            ZMP position [x, y, 0] (ZMP is always on ground)
        """
        zmp = np.zeros(3)
        # Project CoM position to ground level and adjust by acceleration
        zmp[0] = com_pos[0] - (self.com_height / self.gravity) * com_accel[0]
        zmp[1] = com_pos[1] - (self.com_height / self.gravity) * com_accel[1]
        zmp[2] = 0  # ZMP is always on the ground plane
        
        return zmp

class ZMPBasedBalancer:
    def __init__(self, com_height: float = 0.8, gravity: float = 9.81):
        """
        Balance controller based on ZMP theory
        """
        self.zmp_calc = ZMPCalculator(com_height, gravity)
        self.com_height = com_height
        self.gravity = gravity
        
        # PID controller parameters for ZMP tracking
        self.kp = 10.0  # Proportional gain
        self.ki = 1.0   # Integral gain  
        self.kd = 5.0   # Derivative gain
        
        # History for integral and derivative terms
        self.error_history = []
        self.error_derivatives = []
        self.max_history = 10
    
    def update_balance(self, current_com: np.ndarray, current_com_vel: np.ndarray,
                      current_com_acc: np.ndarray, desired_zmp: np.ndarray, 
                      dt: float) -> np.ndarray:
        """
        Calculate control correction based on ZMP error
        Args:
            current_com: Current CoM position
            current_com_vel: Current CoM velocity
            current_com_acc: Current CoM acceleration
            desired_zmp: Target ZMP position
            dt: Time step
        Returns:
            Correction to apply to CoM or joint positions
        """
        # Calculate current ZMP
        current_zmp = self.zmp_calc.calculate_zmp(current_com, current_com_acc)
        
        # Calculate tracking error
        error = desired_zmp[:2] - current_zmp[:2]
        
        # Update PID components
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.error_history.append(error.copy())
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        integral_error = np.zeros(2)
        for e in self.error_history:
            integral_error += e
        integral_error *= dt
        i_term = self.ki * integral_error
        
        # Derivative term
        if len(self.error_history) > 1:
            current_error = self.error_history[-1]
            previous_error = self.error_history[-2]
            derivative_error = (current_error - previous_error) / dt
            self.error_derivatives.append(derivative_error)
            if len(self.error_derivatives) > self.max_history:
                self.error_derivatives.pop(0)
            
            avg_derivative = np.mean(self.error_derivatives, axis=0)
        else:
            avg_derivative = np.zeros(2)
        
        d_term = self.kd * avg_derivative
        
        # Calculate total control output
        control_output = p_term + i_term + d_term
        
        # Calculate required CoM acceleration to achieve desired ZMP correction
        required_com_acc = np.zeros(3)
        required_com_acc[0] = -self.gravity / self.com_height * (current_zmp[0] - desired_zmp[0])
        required_com_acc[1] = -self.gravity / self.com_height * (current_zmp[1] - desired_zmp[1])
        
        return control_output, required_com_acc

# Example usage
balancer = ZMPBasedBalancer()
current_com = np.array([0.01, 0.01, 0.8])  # CoM slightly off center
current_com_vel = np.array([0.0, 0.0, 0.0])
current_com_acc = np.array([0.0, 0.0, 0.0])
desired_zmp = np.array([0.0, 0.0, 0.0])  # Want ZMP at origin

correction, required_acc = balancer.update_balance(current_com, current_com_vel, 
                                                   current_com_acc, desired_zmp, 0.01)

print(f"ZMP correction needed: [{correction[0]:.3f}, {correction[1]:.3f}]")
print(f"Required CoM acceleration: [{required_acc[0]:.3f}, {required_acc[1]:.3f}]")
```

## Control Strategies for Balance

### PID-Based Balance Control

PID controllers are commonly used for balance control:

```python
import numpy as np

class PIDBalanceController:
    def __init__(self, kp_com: float = 10.0, ki_com: float = 1.0, kd_com: float = 5.0,
                 kp_zmp: float = 5.0, ki_zmp: float = 0.5, kd_zmp: float = 2.0):
        """
        PID-based balance controller with separate control for CoM and ZMP
        Args:
            kp_com, ki_com, kd_com: PID gains for CoM control
            kp_zmp, ki_zmp, kd_zmp: PID gains for ZMP control
        """
        # CoM control PID
        self.kp_com = kp_com
        self.ki_com = ki_com
        self.kd_com = kd_com
        
        # ZMP control PID
        self.kp_zmp = kp_zmp
        self.ki_zmp = ki_zmp
        self.kd_zmp = kd_zmp
        
        # History for PID terms
        self.com_error_history = [[0, 0]] * 10
        self.com_last_error = [0, 0]
        
        self.zmp_error_history = [[0, 0]] * 10
        self.zmp_last_error = [0, 0]
    
    def compute_control(self, measured_com: np.ndarray, desired_com: np.ndarray,
                       measured_zmp: np.ndarray, desired_zmp: np.ndarray, dt: float) -> dict:
        """
        Compute balance control commands
        Args:
            measured_com: Current CoM position
            desired_com: Target CoM position
            measured_zmp: Current ZMP position
            desired_zmp: Target ZMP position
            dt: Time step
        Returns:
            Control commands as dictionary
        """
        # Calculate CoM error
        com_error = desired_com[:2] - measured_com[:2]
        
        # CoM PID control
        com_p_term = self.kp_com * com_error
        
        # Integral term
        self.com_error_history.pop(0)
        self.com_error_history.append(com_error)
        com_integral = np.mean(self.com_error_history, axis=0)
        com_i_term = self.ki_com * com_integral
        
        # Derivative term
        com_derivative = (com_error - self.com_last_error) / dt
        com_d_term = self.kd_com * com_derivative
        self.com_last_error = com_error.copy()
        
        com_control = com_p_term + com_i_term + com_d_term
        
        # Calculate ZMP error
        zmp_error = desired_zmp[:2] - measured_zmp[:2]
        
        # ZMP PID control
        zmp_p_term = self.kp_zmp * zmp_error
        
        # Integral term
        self.zmp_error_history.pop(0)
        self.zmp_error_history.append(zmp_error)
        zmp_integral = np.mean(self.zmp_error_history, axis=0)
        zmp_i_term = self.ki_zmp * zmp_integral
        
        # Derivative term
        zmp_derivative = (zmp_error - self.zmp_last_error) / dt
        zmp_d_term = self.kd_zmp * zmp_derivative
        self.zmp_last_error = zmp_error.copy()
        
        zmp_control = zmp_p_term + zmp_i_term + zmp_d_term
        
        # Combine controls
        total_control = com_control + zmp_control
        
        return {
            'com_control': com_control,
            'zmp_control': zmp_control,
            'total_control': total_control,
            'computed_dt': dt
        }

# Example usage
pid_balancer = PIDBalanceController()

# Simulate current state
current_com = np.array([0.02, -0.01, 0.8])
current_zmp = np.array([0.025, -0.015, 0.0])  # Different from CoM due to dynamics

# Define targets
target_com = np.array([0.0, 0.0, 0.8])
target_zmp = np.array([0.0, 0.0, 0.0])

# Compute control
control_result = pid_balancer.compute_control(current_com, target_com, current_zmp, target_zmp, 0.01)

print("Balance control results:")
for key, value in control_result.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: [{value[0]:.3f}, {value[1]:.3f}]")
    else:
        print(f"  {key}: {value}")
```

### Linear Inverted Pendulum Model (LIPM)

The Linear Inverted Pendulum Model is a simplified model that makes balance control computationally tractable:

```python
import numpy as np

class LIPMController:
    def __init__(self, com_height: float = 0.8, gravity: float = 9.81):
        """
        Controller based on Linear Inverted Pendulum Model
        The LIPM assumes constant CoM height, simplifying the dynamics
        """
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)  # Natural frequency of the pendulum
    
    def calculate_capture_point(self, com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        """
        Calculate the capture point - where the CoM would need to be placed
        to come to rest with zero velocity
        Capture Point = CoM + CoM_vel / omega
        Args:
            com_pos: CoM position [x, y, z]
            com_vel: CoM velocity [x_dot, y_dot, z_dot]
        Returns:
            Capture point position [x, y, 0] (on ground plane)
        """
        cp = np.zeros(3)
        cp[0] = com_pos[0] + com_vel[0] / self.omega
        cp[1] = com_pos[1] + com_vel[1] / self.omega
        # z-component is 0 (on ground)
        
        return cp
    
    def calculate_desired_com_trajectory(self, current_com: np.ndarray, 
                                       current_com_vel: np.ndarray,
                                       final_com_pos: np.ndarray, 
                                       final_time: float) -> callable:
        """
        Calculate desired CoM trajectory that ends at final position
        with zero velocity
        Args:
            current_com: Current CoM position
            current_com_vel: Current CoM velocity
            final_com_pos: Final CoM position
            final_time: Time to reach final position
        Returns:
            Function that gives desired CoM state at any time t
        """
        # Calculate LIPM trajectory parameters
        # x(t) = a * e^(omega*t) + b * e^(-omega*t)
        # Using boundary conditions to solve for a and b
        
        x0 = current_com[0]
        y0 = current_com[1]
        vx0 = current_com_vel[0]
        vy0 = current_com_vel[1]
        xf = final_com_pos[0]
        yf = final_com_pos[1]
        
        # Calculate coefficients for x direction
        a_x = (xf * np.exp(self.omega * final_time) - 
               x0 + (vx0 / self.omega)) / (np.exp(self.omega * final_time) - np.exp(-self.omega * final_time))
        b_x = x0 - (vx0 / self.omega) - a_x
        
        # Calculate coefficients for y direction
        a_y = (yf * np.exp(self.omega * final_time) - 
               y0 + (vy0 / self.omega)) / (np.exp(self.omega * final_time) - np.exp(-self.omega * final_time))
        b_y = y0 - (vy0 / self.omega) - a_y
        
        def trajectory_function(t):
            """
            Return desired CoM position and velocity at time t
            """
            if t >= final_time:
                return np.array([xf, yf, self.com_height]), np.array([0.0, 0.0, 0.0])
            
            # Position
            x_des = a_x * np.exp(self.omega * t) + b_x * np.exp(-self.omega * t)
            y_des = a_y * np.exp(self.omega * t) + b_y * np.exp(-self.omega * t)
            z_des = self.com_height  # Constant height in LIPM
            
            # Velocity (derivative of position)
            vx_des = self.omega * (a_x * np.exp(self.omega * t) - b_x * np.exp(-self.omega * t))
            vy_des = self.omega * (a_y * np.exp(self.omega * t) - b_y * np.exp(-self.omega * t))
            vz_des = 0.0
            
            return np.array([x_des, y_des, z_des]), np.array([vx_des, vy_des, vz_des])
        
        return trajectory_function

class LIPMBalancer:
    def __init__(self, com_height: float = 0.8, gravity: float = 9.81):
        """
        Balance controller using LIPM that provides footstep planning and CoM control
        """
        self.lipm = LIPMController(com_height, gravity)
        self.com_height = com_height
        self.gravity = gravity
        self.omega = self.lipm.omega
        
        # For feedback control
        self.kp = 5.0  # Proportional gain for CoM feedback
        self.kd = 1.0  # Derivative gain for CoM feedback
    
    def compute_balance_control(self, current_com: np.ndarray, current_com_vel: np.ndarray,
                              desired_com_trajectory: callable, t: float, dt: float) -> np.ndarray:
        """
        Compute balance control based on LIPM with feedback
        Args:
            current_com: Current CoM position
            current_com_vel: Current CoM velocity
            desired_com_trajectory: Function returning (desired_pos, desired_vel) at time t
            t: Current time
            dt: Time step
        Returns:
            Required CoM acceleration for balance
        """
        # Get desired state from trajectory
        desired_com, desired_vel = desired_com_trajectory(t)
        
        # Calculate CoM position error
        pos_error = desired_com[:2] - current_com[:2]
        vel_error = desired_vel[:2] - current_com_vel[:2]
        
        # Simple PD feedback control
        feedback_acc = self.kp * pos_error + self.kd * vel_error
        
        # Required acceleration to maintain LIPM dynamics
        # In LIPM: CoM_acc = omega^2 * (CoM - ZMP)
        # So: ZMP = CoM - CoM_acc / omega^2
        # For balance, we want ZMP in the support polygon, so:
        # CoM_acc[desired] = omega^2 * (CoM - ZMP[desired])
        
        # For stability, aim for ZMP at the center of the support polygon
        # This is a simplified approach - in practice, ZMP reference changes based on footstep plan
        zmp_desired = np.array([0.0, 0.0])  # Assume support at origin for this example
        feedforward_acc = self.omega**2 * (current_com[:2] - zmp_desired)
        
        # Combined control
        total_acc = feedforward_acc + feedback_acc
        
        # Return as 3D acceleration
        result_acc = np.zeros(3)
        result_acc[:2] = total_acc
        result_acc[2] = 0  # No vertical acceleration in LIPM
        
        return result_acc

# Example usage
lipm_balancer = LIPMBalancer()

# Define initial state
current_com = np.array([0.05, 0.02, 0.8])
current_com_vel = np.array([0.01, -0.005, 0.0])

# Define a desired trajectory that brings CoM back to center
initial_pos = np.array([0.05, 0.02, 0.8])
final_pos = np.array([0.0, 0.0, 0.8])
trajectory_func = lipm_balancer.lipm.calculate_desired_com_trajectory(
    initial_pos, current_com_vel, final_pos, final_time=2.0
)

# Compute balance control at t=0.5 seconds
balance_acc = lipm_balancer.compute_balance_control(
    current_com, current_com_vel, trajectory_func, 0.5, 0.01
)

print(f"LIPM balance control acceleration: [{balance_acc[0]:.3f}, {balance_acc[1]:.3f}, {balance_acc[2]:.3f}]")
```

## Advanced Balance Strategies

### Capture Point-Based Balance

The capture point is a key concept for dynamic balance recovery:

```python
class CapturePointBalancer:
    def __init__(self, com_height: float = 0.8, gravity: float = 9.81, 
                 max_foot_step: float = 0.3, max_ankle_torque: float = 100.0):
        """
        Balance controller using capture point concept
        Args:
            com_height: CoM height
            gravity: Gravity constant
            max_foot_step: Maximum foot step size for recovery
            max_ankle_torque: Maximum ankle torque available for balance
        """
        self.com_height = com_height
        self.gravity = gravity
        self.max_foot_step = max_foot_step
        self.max_ankle_torque = max_ankle_torque
        self.omega = np.sqrt(gravity / com_height)
        
        # Support polygon (simplified as a box around foot)
        self.support_polygon_half_width = 0.1  # 10cm half-width
        self.support_polygon_half_length = 0.15  # 15cm half-length
    
    def calculate_capture_point(self, com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        """
        Calculate capture point based on current CoM state
        """
        cp = np.zeros(3)
        cp[0] = com_pos[0] + com_vel[0] / self.omega
        cp[1] = com_pos[1] + com_vel[1] / self.omega
        return cp
    
    def is_balance_recoverable(self, current_cp: np.ndarray, support_center: np.ndarray) -> bool:
        """
        Check if the robot can recover balance by stepping to support center
        Args:
            current_cp: Current capture point
            support_center: Center of available support polygon
        Returns:
            True if balance can be recovered, False otherwise
        """
        # Check if capture point is within maximum recoverable distance
        distance_to_support = np.linalg.norm(current_cp[:2] - support_center[:2])
        return distance_to_support <= self.max_foot_step
    
    def compute_balance_strategy(self, com_pos: np.ndarray, com_vel: np.ndarray, 
                               support_pos: np.ndarray) -> dict:
        """
        Compute balance strategy based on current state and support
        Args:
            com_pos: Current CoM position
            com_vel: Current CoM velocity
            support_pos: Current support position (foot location)
        Returns:
            Balance strategy dictionary
        """
        # Calculate current capture point
        current_cp = self.calculate_capture_point(com_pos, com_vel)
        
        # Check if we're in the current support polygon
        in_support = (abs(current_cp[0] - support_pos[0]) <= self.support_polygon_half_length and
                      abs(current_cp[1] - support_pos[1]) <= self.support_polygon_half_width)
        
        strategy = {
            'current_capture_point': current_cp,
            'in_support': in_support,
            'next_step_needed': not in_support,
            'step_target': support_pos.copy(),
            'ankle_control_needed': True
        }
        
        if not in_support:
            # Calculate where to step to recover balance
            # Stepping to capture point location is one strategy
            step_target = current_cp[:2].copy()
            
            # Limit step size to robot capabilities
            step_vector = step_target - support_pos[:2]
            step_distance = np.linalg.norm(step_vector)
            
            if step_distance > self.max_foot_step:
                # Scale the step to maximum reachable distance
                step_vector = step_vector / step_distance * self.max_foot_step
                step_target = support_pos[:2] + step_vector
            
            strategy['step_target'] = step_target
            strategy['step_distance'] = np.linalg.norm(step_vector)
        
        # Calculate ankle control needed
        # In the support polygon, use ankle strategy
        if in_support:
            # Calculate required ankle moment to keep CP in place
            # This is a simplified approach
            cp_deviation = current_cp[:2] - support_pos[:2]
            ankle_control = {
                'required_torque_x': self.max_ankle_torque * min(1.0, abs(cp_deviation[0]) / 0.05),  # 5cm threshold
                'required_torque_y': self.max_ankle_torque * min(1.0, abs(cp_deviation[1]) / 0.05),
                'max_safe_ankle_torque': self.max_ankle_torque
            }
            strategy['ankle_control'] = ankle_control
        
        return strategy

# Example usage
cp_balancer = CapturePointBalancer()

# Simulate robot state
robot_com = np.array([0.1, 0.05, 0.8])  # CoM offset from foot position
robot_com_vel = np.array([0.02, 0.01, 0.0])
current_foot_pos = np.array([0.0, 0.0, 0.0])  # Foot position

# Compute balance strategy
strategy = cp_balancer.compute_balance_strategy(robot_com, robot_com_vel, current_foot_pos)

print("Capture point balance strategy:")
print(f"  Current capture point: [{strategy['current_capture_point'][0]:.3f}, {strategy['current_capture_point'][1]:.3f}]")
print(f"  In support polygon: {strategy['in_support']}")
print(f"  Next step needed: {strategy['next_step_needed']}")
print(f"  Step target: [{strategy['step_target'][0]:.3f}, {strategy['step_target'][1]:.3f}]")

if 'ankle_control' in strategy:
    print(f"  Required ankle torque X: {strategy['ankle_control']['required_torque_x']:.2f}")
    print(f"  Required ankle torque Y: {strategy['ankle_control']['required_torque_y']:.2f}")
```

### Whole-Body Balance Control

For effective balance, multiple parts of the robot must coordinate:

```python
import numpy as np

class WholeBodyBalanceController:
    def __init__(self, robot_mass: float = 60.0, com_height: float = 0.8):
        """
        Controller that coordinates multiple body parts for balance
        Args:
            robot_mass: Total mass of the robot
            com_height: Nominal CoM height
        """
        self.robot_mass = robot_mass
        self.com_height = com_height
        self.gravity = 9.81
        
        # Joint limits and weights for different parts
        self.joint_weights = {
            'left_ankle': 0.2,    # Ankle can make rapid corrections
            'right_ankle': 0.2,
            'left_hip': 0.15,     # Hip can shift weight
            'right_hip': 0.15,
            'torso': 0.2,         # Torso can orient for balance
            'left_arm': 0.05,     # Arms can be used for balance
            'right_arm': 0.05
        }
        
        # Maximum forces/torques for different joints
        self.max_torques = {
            'left_ankle_roll': 80.0,
            'left_ankle_pitch': 80.0,
            'right_ankle_roll': 80.0,
            'right_ankle_pitch': 80.0,
            'left_hip_roll': 120.0,
            'right_hip_roll': 120.0,
            'torso_yaw': 50.0,
            'torso_pitch': 50.0
        }
    
    def compute_balance_forces(self, com_error: np.ndarray, com_vel_error: np.ndarray,
                             capture_point_error: np.ndarray) -> dict:
        """
        Compute the distribution of forces needed across joints for balance
        Args:
            com_error: Error in CoM position [x, y, z]
            com_vel_error: Error in CoM velocity [x_dot, y_dot, z_dot]
            capture_point_error: Error in capture point [x, y]
        Returns:
            Dictionary of required joint torques
        """
        # Calculate required corrective forces based on errors
        # This is a simplified approach - in reality, would use full dynamics model
        
        # Primary correction: place capture point inside support polygon
        cp_force_x = 100 * capture_point_error[0]  # Proportional to CP error
        cp_force_y = 100 * capture_point_error[1]
        
        # Secondary correction: bring CoM to desired position
        com_force_x = 50 * com_error[0] + 10 * com_vel_error[0]
        com_force_y = 50 * com_error[1] + 10 * com_vel_error[1]
        
        # Combine forces
        total_force_x = cp_force_x + com_force_x
        total_force_y = cp_force_y + com_force_y
        
        # Distribute forces across joints based on their capabilities
        joint_commands = {}
        
        # Ankle control (primary balance mechanism)
        joint_commands['left_ankle_roll'] = min(
            max(total_force_y * self.joint_weights['left_ankle'], 
                -self.max_torques['left_ankle_roll']), 
            self.max_torques['left_ankle_roll']
        )
        joint_commands['left_ankle_pitch'] = min(
            max(total_force_x * self.joint_weights['left_ankle'], 
                -self.max_torques['left_ankle_pitch']), 
            self.max_torques['left_ankle_pitch']
        )
        
        joint_commands['right_ankle_roll'] = min(
            max(total_force_y * self.joint_weights['right_ankle'], 
                -self.max_torques['right_ankle_roll']), 
            self.max_torques['right_ankle_roll']
        )
        joint_commands['right_ankle_pitch'] = min(
            max(total_force_x * self.joint_weights['right_ankle'], 
                -self.max_torques['right_ankle_pitch']), 
            self.max_torques['right_ankle_pitch']
        )
        
        # Hip control (weight shifting)
        hip_correction_x = total_force_x * self.joint_weights['left_hip']  # Symmetric
        joint_commands['left_hip_roll'] = min(
            max(hip_correction_x, -self.max_torques['left_hip_roll']), 
            self.max_torques['left_hip_roll']
        )
        joint_commands['right_hip_roll'] = min(
            max(-hip_correction_x, -self.max_torques['right_hip_roll']), 
            self.max_torques['right_hip_roll']
        )
        
        # Torso control (orientation adjustment)
        joint_commands['torso_pitch'] = min(
            max(total_force_x * 0.1, -self.max_torques['torso_pitch']), 
            self.max_torques['torso_pitch']
        )
        joint_commands['torso_yaw'] = min(
            max(total_force_y * 0.1, -self.max_torques['torso_yaw']), 
            self.max_torques['torso_yaw']
        )
        
        # Arm control (swing arms for balance, if needed)
        arm_correction = (total_force_x + total_force_y) * 0.05
        joint_commands['left_arm_swing'] = arm_correction
        joint_commands['right_arm_swing'] = -arm_correction  # Opposite direction
        
        return joint_commands
    
    def compute_com_adjustment_for_manipulation(self, load_position: np.ndarray, 
                                               load_weight: float) -> np.ndarray:
        """
        Compute CoM adjustment needed when carrying a load
        Args:
            load_position: Position of the load in robot frame
            load_weight: Weight of the load in kg
        Returns:
            Required CoM adjustment vector
        """
        # Calculate moment created by the load
        moment_arm = load_position - np.array([0, 0, self.com_height])  # From CoM
        load_force = np.array([0, 0, -load_weight * self.gravity])
        
        # Calculate position offset needed to balance the load
        # This is a simplified calculation
        com_offset = np.zeros(3)
        
        # X offset to counteract moment in Y direction
        com_offset[0] = (load_weight * moment_arm[1]) / self.robot_mass
        # Y offset to counteract moment in X direction  
        com_offset[1] = -(load_weight * moment_arm[0]) / self.robot_mass
        
        return com_offset

# Example usage
wb_balancer = WholeBodyBalanceController()

# Simulate balance error
com_error = np.array([0.02, -0.01, 0.0])  # CoM offset
com_vel_error = np.array([0.01, 0.005, 0.0])  # CoM velocity error
cp_error = np.array([0.03, -0.02])  # Capture point offset

# Compute balance commands
balance_commands = wb_balancer.compute_balance_forces(com_error, com_vel_error, cp_error)

print("Whole-body balance commands:")
for joint, torque in list(balance_commands.items())[:6]:  # Show first few commands
    print(f"  {joint}: {torque:.2f} Nm")

# Simulate load handling
load_pos = np.array([0.3, 0.1, 0.8])  # Object in right hand
load_weight = 2.0  # 2kg object
com_adjustment = wb_balancer.compute_com_adjustment_for_manipulation(load_pos, load_weight)

print(f"\nCoM adjustment for carrying load: [{com_adjustment[0]:.3f}, {com_adjustment[1]:.3f}, {com_adjustment[2]:.3f}]")
```

## Disturbance Recovery

### Push Recovery Strategies

Humanoid robots must be able to recover from unexpected disturbances:

```python
class DisturbanceRecoveryController:
    def __init__(self, com_height: float = 0.8, max_ankle_moment: float = 80.0,
                 max_step_size: float = 0.3):
        """
        Controller for recovering from unexpected disturbances
        Args:
            com_height: Height of center of mass
            max_ankle_moment: Maximum ankle control moment
            max_step_size: Maximum step size for recovery
        """
        self.com_height = com_height
        self.max_ankle_moment = max_ankle_moment
        self.max_step_size = max_step_size
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / self.com_height)
        
        # State tracking for recovery
        self.is_recovering = False
        self.recovery_start_time = 0
        self.recovery_phase = 'initial'  # initial, ankle_control, stepping, finalizing
        
        # Recovery parameters
        self.ankle_control_duration = 0.3  # Time to try ankle control
        self.stance_switch_threshold = 0.2  # Threshold for switching stance foot
        self.recovery_timeout = 3.0  # Timeout for recovery attempt
    
    def detect_disturbance(self, current_com: np.ndarray, current_com_vel: np.ndarray,
                          current_capture_point: np.ndarray, support_polygon: SupportPolygon) -> bool:
        """
        Detect if a disturbance has occurred
        Args:
            current_com: Current CoM position
            current_com_vel: Current CoM velocity
            current_capture_point: Current capture point
            support_polygon: Current support polygon
        Returns:
            True if disturbance detected, False otherwise
        """
        # Check multiple criteria for disturbance
        # 1. Capture point outside support polygon
        is_cp_outside = not support_polygon.contains_point(current_capture_point[:2])
        
        # 2. High CoM velocity
        com_speed = np.linalg.norm(current_com_vel[:2])
        high_velocity = com_speed > 0.3  # m/s threshold
        
        # 3. Large CoM displacement
        com_displacement = np.linalg.norm(current_com[:2])
        large_displacement = com_displacement > 0.1  # 10cm threshold
        
        # 4. Rapid CoM acceleration
        # This would need to track acceleration over time
        
        return is_cp_outside or high_velocity or large_displacement
    
    def compute_recoverable_thresholds(self) -> dict:
        """
        Compute thresholds for different recovery strategies
        """
        # Ankle control range (capture point must be within this distance for ankle control)
        ankle_control_radius = self.max_ankle_moment / (self.robot_mass * self.gravity) * 0.1  # Simplified
        ankle_control_radius = min(ankle_control_radius, 0.05)  # Cap at 5cm for safety
        
        # Step recovery range (if capture point is within max step distance)
        step_recovery_radius = self.max_step_size * 0.8  # 80% of max step for safety
        
        return {
            'ankle_control_radius': ankle_control_radius,
            'step_recovery_radius': step_recovery_radius,
            'capture_point_threshold': step_recovery_radius
        }
    
    def plan_disturbance_recovery(self, current_state: dict, impact_magnitude: float = None) -> dict:
        """
        Plan a recovery sequence from disturbance
        Args:
            current_state: Dictionary with current robot state
            impact_magnitude: Estimated magnitude of disturbance (optional)
        Returns:
            Recovery plan dictionary
        """
        recovery_plan = {
            'actions': [],
            'strategy': 'none',
            'estimated_recovery_time': 0.0,
            'required_steps': 0
        }
        
        # Get current state
        com_pos = current_state['com_position']
        com_vel = current_state['com_velocity']
        current_cp = current_state['capture_point']
        support_polygon = current_state['support_polygon']
        support_pos = current_state['support_position']
        
        # Calculate thresholds
        thresholds = self.compute_recoverable_thresholds()
        
        # Calculate capture point distance from support center
        support_center = support_polygon.get_centroid()
        cp_distance = np.linalg.norm(current_cp[:2] - support_center[:2])
        
        if cp_distance <= thresholds['ankle_control_radius']:
            # Use ankle control to recover
            recovery_plan['strategy'] = 'ankle_control'
            recovery_plan['actions'].append({
                'type': 'ankle_adjustment',
                'magnitude': min(cp_distance / thresholds['ankle_control_radius'], 1.0),
                'direction': (support_center[:2] - current_cp[:2]) / cp_distance if cp_distance > 0 else np.array([0, 1]),
                'duration': self.ankle_control_duration
            })
            recovery_plan['estimated_recovery_time'] = self.ankle_control_duration
        
        elif cp_distance <= thresholds['step_recovery_radius']:
            # Plan a step to capture point
            recovery_plan['strategy'] = 'stepping'
            step_target = current_cp[:2]  # Step toward capture point
            
            # Limit step size
            step_vector = step_target - support_pos[:2]
            step_length = np.linalg.norm(step_vector)
            
            if step_length > self.max_step_size:
                step_vector = step_vector / step_length * self.max_step_size
                step_target = support_pos[:2] + step_vector
            
            recovery_plan['actions'].extend([
                {
                    'type': 'prepare_step',
                    'foot': 'swing_foot',
                    'duration': 0.2
                },
                {
                    'type': 'execute_step',
                    'target_position': step_target,
                    'duration': 0.4
                },
                {
                    'type': 'stabilize',
                    'duration': 0.5
                }
            ])
            recovery_plan['estimated_recovery_time'] = 1.1
            recovery_plan['required_steps'] = 1
        
        else:
            # Capture point is too far - use multiple steps or emergency strategy
            recovery_plan['strategy'] = 'multi_step_escape'
            
            # Calculate intermediate step positions to gradually move capture point
            steps_needed = int(np.ceil(cp_distance / (self.max_step_size * 0.7)))  # 70% of step for overlap
            current_target = support_pos[:2]
            
            for i in range(steps_needed):
                # Calculate direction toward capture point
                direction = (current_cp[:2] - current_target) / np.linalg.norm(current_cp[:2] - current_target)
                
                # Calculate next step position
                step_dist = min(self.max_step_size * 0.7, np.linalg.norm(current_cp[:2] - current_target))
                next_target = current_target + direction * step_dist
                
                recovery_plan['actions'].extend([
                    {
                        'type': 'prepare_step',
                        'foot': 'swing_foot',
                        'duration': 0.2
                    },
                    {
                        'type': 'execute_step',
                        'target_position': next_target,
                        'duration': 0.4
                    }
                ])
                
                current_target = next_target
            
            recovery_plan['actions'].append({
                'type': 'final_stabilize',
                'duration': 0.5
            })
            
            recovery_plan['estimated_recovery_time'] = steps_needed * 0.6 + 0.5  # 0.6s per step + final stabilize
            recovery_plan['required_steps'] = steps_needed
    
        return recovery_plan

# Example usage
recovery_controller = DisturbanceRecoveryController()

# Create a current state
current_state = {
    'com_position': np.array([0.05, 0.02, 0.8]),
    'com_velocity': np.array([0.1, 0.05, 0.0]),
    'capture_point': np.array([0.2, 0.1, 0.0]),
    'support_polygon': SupportPolygon([np.array([0.1, 0.05]), np.array([0.1, -0.05]), 
                                       np.array([-0.1, 0.05]), np.array([-0.1, -0.05])]),
    'support_position': np.array([0.0, 0.0, 0.0])
}

# Plan recovery
recovery_plan = recovery_controller.plan_disturbance_recovery(current_state)

print(f"Recovery strategy: {recovery_plan['strategy']}")
print(f"Estimated time: {recovery_plan['estimated_recovery_time']:.2f}s")
print(f"Steps required: {recovery_plan['required_steps']}")
print(f"Number of actions: {len(recovery_plan['actions'])}")
```

## Exercises

1. **Balance Control Simulation**: Implement and compare different balance control strategies (PID, LIPM, Capture Point) in a simulated environment. Evaluate their performance under various disturbances.

2. **Push Recovery Exercise**: Create a push recovery system that can detect and respond to pushes from different directions and magnitudes with appropriate recovery strategies.

3. **Walking Balance Integration**: Combine walking pattern generation with balance control to maintain stability during locomotion.

4. **Multi-Contact Balance**: Extend balance control to handle scenarios with multiple contact points (hands and feet).

5. **Compliance Control**: Implement compliant balance control that can handle uncertain or changing ground conditions.

6. **Load-Carrying Balance**: Design balance control system that adapts to carrying loads of different weights and positions.

7. **Learning-Based Balance**: Use reinforcement learning to improve balance control based on experience.

## Summary

Balance control in humanoid robots is a complex, multi-faceted challenge requiring sophisticated control algorithms. The key principles involve maintaining the center of mass within the support polygon or using dynamic strategies to maintain stability during motion. Advanced approaches like the Linear Inverted Pendulum Model and Capture Point theory provide effective frameworks for balance control, while whole-body strategies coordinate multiple joints for robust balance. For practical humanoid robotics, these control systems must be integrated with locomotion and manipulation systems to enable effective operation in human environments.