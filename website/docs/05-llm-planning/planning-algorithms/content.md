---
title: Planning Algorithms for Navigation and Manipulation in Humanoid Robotics
sidebar_position: 1
---

# Planning Algorithms for Navigation and Manipulation in Humanoid Robotics

## Introduction

Navigation and manipulation are fundamental capabilities for humanoid robots operating in human environments. These tasks require sophisticated planning algorithms that can handle the complexity of real-world scenarios while accounting for the unique characteristics of humanoid robots. This chapter explores various planning algorithms specifically tailored for navigation and manipulation tasks in humanoid robotics, covering both classical approaches and modern learning-based methods.

## Navigation Planning

### Classical Path Planning Approaches

Navigation planning for humanoid robots must consider their unique characteristics: bipedal locomotion, limited stability margins, and anthropomorphic capabilities.

#### A* Algorithm

A* remains a popular choice for pathfinding due to its optimality and efficiency:

```python
import heapq
import numpy as np
from typing import List, Tuple, Optional

class AStarPlanner:
    def __init__(self, grid_map: np.ndarray, resolution: float = 0.1):
        """
        Initialize A* planner
        Args:
            grid_map: 2D array representing occupancy grid (0=free, 1=occupied)
            resolution: Size of each grid cell in meters
        """
        self.grid_map = grid_map
        self.resolution = resolution
        self.height, self.width = grid_map.shape
    
    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Plan path from start to goal using A*
        Args:
            start: Start position (row, col)
            goal: Goal position (row, col)
        Returns:
            Path as list of grid coordinates or None if no path exists
        """
        # Check if start or goal are in collision
        if self.grid_map[start[0], start[1]] == 1 or self.grid_map[goal[0], goal[1]] == 1:
            return None
        
        # Initialize open and closed sets
        open_set = [(0, start)]  # Priority queue: (f_score, position)
        heapq.heapify(open_set)
        closed_set = set()
        
        # Initialize g_score and parent dictionaries
        g_score = {start: 0}
        parent = {start: None}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            # If we reached the goal, reconstruct path
            if current == goal:
                return self._reconstruct_path(parent, current)
            
            closed_set.add(current)
            
            # Get neighbors (8-connected for more flexible paths)
            neighbors = self._get_neighbors(current)
            for neighbor in neighbors:
                if neighbor in closed_set or self.grid_map[neighbor[0], neighbor[1]] == 1:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + self._calculate_cost(current, neighbor)
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    parent[neighbor] = current
        
        return None  # No path found
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 8-connected neighbors"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip current cell
                
                nr, nc = pos[0] + dr, pos[1] + dc
                
                # Check if neighbor is within map bounds
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    neighbors.append((nr, nc))
        
        return neighbors
    
    def _calculate_cost(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate movement cost between adjacent cells"""
        dr = pos1[0] - pos2[0]
        dc = pos1[1] - pos2[1]
        
        # Euclidean distance scaled by resolution
        return np.sqrt(dr**2 + dc**2) * self.resolution
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        dr = pos[0] - goal[0]
        dc = pos[1] - goal[1]
        return np.sqrt(dr**2 + dc**2) * self.resolution
    
    def _reconstruct_path(self, parent: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from parent dictionary"""
        path = [current]
        while current in parent and parent[current] is not None:
            current = parent[current]
            path.append(current)
        
        path.reverse()
        return path

# Example usage
if __name__ == "__main__":
    # Create a simple grid map (0=free, 1=occupied)
    grid_map = np.zeros((10, 10))
    
    # Add some obstacles
    grid_map[3, 3:7] = 1  # Horizontal wall
    grid_map[5:8, 5] = 1  # Vertical wall
    
    planner = AStarPlanner(grid_map, resolution=0.1)
    path = planner.plan_path((1, 1), (8, 8))
    
    print(f"Path: {path}")
    if path:
        print(f"Path length: {len(path) * 0.1:.2f} meters")
    else:
        print("No path found")
```

#### Dijkstra's Algorithm

Dijkstra's algorithm is useful when all edge weights are equal or when optimality is required:

```python
import heapq
import numpy as np
from typing import List, Tuple, Optional

class DijkstraPlanner:
    def __init__(self, graph: np.ndarray, resolution: float = 0.1):
        """
        Initialize Dijkstra planner for grid-based navigation
        Args:
            graph: Weighted adjacency matrix or occupancy grid
            resolution: Size of each grid cell in meters
        """
        self.graph = graph
        self.resolution = resolution
        self.rows, self.cols = graph.shape
    
    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Plan path using Dijkstra's algorithm
        """
        # Check if start or goal are in collision
        if self.graph[start[0], start[1]] == 1 or self.graph[goal[0], goal[1]] == 1:
            return None
        
        # Initialize distances and priority queue
        distances = np.full_like(self.graph, np.inf, dtype=float)
        distances[start[0], start[1]] = 0
        
        pq = [(0, start)]
        parent = {}
        visited = np.zeros_like(self.graph, dtype=bool)
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if visited[current[0], current[1]]:
                continue
                
            visited[current[0], current[1]] = True
            
            if current == goal:
                return self._reconstruct_path(parent, current)
            
            # Check neighbors
            for neighbor in self._get_neighbors(current):
                if visited[neighbor[0], neighbor[1]]:
                    continue
                
                # Calculate the distance to this neighbor
                edge_weight = self._get_edge_weight(current, neighbor)
                new_dist = distances[current[0], current[1]] + edge_weight
                
                if new_dist < distances[neighbor[0], neighbor[1]]:
                    distances[neighbor[0], neighbor[1]] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
                    parent[neighbor] = current
        
        return None  # No path found
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 4-connected neighbors (up, down, left, right)"""
        neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.graph[nr, nc] == 0:
                neighbors.append((nr, nc))
        return neighbors
    
    def _get_edge_weight(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate the weight of an edge between two positions"""
        # For grid maps, this is typically a constant
        return self.resolution
    
    def _reconstruct_path(self, parent: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from parent dictionary"""
        path = [current]
        while current in parent:
            current = parent[current]
            path.append(current)
        
        path.reverse()
        return path
```

### Sampling-Based Methods

For high-dimensional configuration spaces, sampling-based methods are often preferred.

#### RRT (Rapidly-exploring Random Tree)

RRT is particularly useful for path planning with complex constraints:

```python
import numpy as np
import random
from typing import List, Tuple, Optional

class RRTPlanner:
    def __init__(self, start: np.ndarray, goal: np.ndarray, bounds: Tuple[Tuple[float, float], Tuple[float, float]], 
                 obstacles: List[Tuple[float, float, float]], step_size: float = 0.1):
        """
        Initialize RRT planner
        Args:
            start: Start configuration
            goal: Goal configuration
            bounds: Configuration space bounds [(x_min, x_max), (y_min, y_max)]
            obstacles: List of circular obstacles [(x, y, radius)]
            step_size: Maximum distance for each step
        """
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        
        # Initialize tree
        self.vertices = [start]
        self.edges = {}  # Child -> Parent mapping
    
    def plan(self, max_samples: int = 1000) -> Optional[List[np.ndarray]]:
        """
        Plan path using RRT algorithm
        """
        for _ in range(max_samples):
            # Sample random point
            rand_point = self._sample_random()
            
            # Find nearest vertex in tree
            nearest_idx = self._find_nearest(rand_point)
            nearest_vertex = self.vertices[nearest_idx]
            
            # Steer towards random point
            new_vertex = self._steer(nearest_vertex, rand_point)
            
            # Check if new vertex is valid (not in collision)
            if self._is_valid(new_vertex):
                # Add to tree
                new_idx = len(self.vertices)
                self.vertices.append(new_vertex)
                self.edges[new_idx] = nearest_idx
                
                # Check if goal is reached
                if np.linalg.norm(new_vertex - self.goal) < self.step_size:
                    return self._extract_path(new_idx)
        
        return None  # Failed to find path
    
    def _sample_random(self) -> np.ndarray:
        """Sample random point in configuration space"""
        # With some probability, sample the goal (bias toward goal)
        if random.random() < 0.1:  # 10% chance to sample goal
            return self.goal
        
        # Otherwise, sample random point in bounds
        x = random.uniform(self.bounds[0][0], self.bounds[0][1])
        y = random.uniform(self.bounds[1][0], self.bounds[1][1])
        return np.array([x, y])
    
    def _find_nearest(self, point: np.ndarray) -> int:
        """Find index of nearest vertex to point"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, vertex in enumerate(self.vertices):
            dist = np.linalg.norm(point - vertex)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """Steer from from_point toward to_point by step_size"""
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_point
        
        # Normalize direction and scale by step_size
        normalized_direction = direction / distance
        return from_point + normalized_direction * self.step_size
    
    def _is_valid(self, point: np.ndarray) -> bool:
        """Check if point is collision-free"""
        # Check against obstacles
        for ox, oy, radius in self.obstacles:
            dist_to_obstacle = np.sqrt((point[0] - ox)**2 + (point[1] - oy)**2)
            if dist_to_obstacle <= radius:
                return False  # Point is in collision
        
        return True
    
    def _extract_path(self, goal_idx: int) -> List[np.ndarray]:
        """Extract path from goal to start"""
        path = []
        current_idx = goal_idx
        
        while current_idx is not None:
            path.append(self.vertices[current_idx])
            current_idx = self.edges.get(current_idx)
        
        path.reverse()
        return path

# Example usage
if __name__ == "__main__":
    # Define environment
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])
    bounds = [(-1, 6), (-1, 6)]  # x bounds, y bounds
    obstacles = [(2, 2, 0.5), (3, 3, 0.5), (1, 4, 0.3)]  # Circular obstacles
    
    # Plan path
    rrt = RRTPlanner(start, goal, bounds, obstacles)
    path = rrt.plan()
    
    if path:
        print(f"Path found with {len(path)} waypoints")
        for i, point in enumerate(path):
            print(f"Waypoint {i}: {point}")
    else:
        print("No path found")
```

#### RRT*

RRT* improves on RRT by providing asymptotic optimality:

```python
import numpy as np
import random
from typing import List, Tuple, Optional

class RRTStarPlanner:
    def __init__(self, start: np.ndarray, goal: np.ndarray, bounds: Tuple[Tuple[float, float], Tuple[float, float]], 
                 obstacles: List[Tuple[float, float, float]], step_size: float = 0.1, max_dist: float = 1.0):
        """
        Initialize RRT* planner
        Args:
            start: Start configuration
            goal: Goal configuration
            bounds: Configuration space bounds
            obstacles: List of circular obstacles [(x, y, radius)]
            step_size: Maximum distance for each step
            max_dist: Maximum distance for considering neighbors during rewiring
        """
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_dist = max_dist
        
        # Initialize tree
        self.vertices = [start]
        self.edges = {}  # Child -> Parent mapping
        self.costs = {0: 0.0}  # Vertex index -> cost from start
    
    def plan(self, max_samples: int = 1000) -> Optional[List[np.ndarray]]:
        """
        Plan path using RRT* algorithm
        """
        for _ in range(max_samples):
            # Sample random point
            rand_point = self._sample_random()
            
            # Find nearest vertex in tree
            nearest_idx = self._find_nearest(rand_point)
            nearest_vertex = self.vertices[nearest_idx]
            
            # Steer towards random point
            new_vertex = self._steer(nearest_vertex, rand_point)
            
            # Check if new vertex is valid
            if self._is_valid(new_vertex):
                # Find vertices within max_dist for rewiring
                neighbors = self._find_neighbors(new_vertex)
                
                # Choose parent from neighbors that minimizes cost
                min_cost = float('inf')
                parent_idx = nearest_idx
                for neighbor_idx in neighbors:
                    neighbor_vertex = self.vertices[neighbor_idx]
                    cost = self.costs[neighbor_idx] + np.linalg.norm(new_vertex - neighbor_vertex)
                    
                    if cost < min_cost:
                        min_cost = cost
                        parent_idx = neighbor_idx
                
                # Add new vertex to tree
                new_idx = len(self.vertices)
                self.vertices.append(new_vertex)
                self.edges[new_idx] = parent_idx
                self.costs[new_idx] = min_cost
                
                # Attempt to rewire neighbors
                for neighbor_idx in neighbors:
                    if neighbor_idx == parent_idx:
                        continue  # Skip parent
                    
                    neighbor_vertex = self.vertices[neighbor_idx]
                    new_cost = min_cost + np.linalg.norm(new_vertex - neighbor_vertex)
                    
                    if new_cost < self.costs[neighbor_idx]:
                        # Rewire through new vertex
                        self.edges[neighbor_idx] = new_idx
                        self.costs[neighbor_idx] = new_cost
        
        # Find best path to goal
        return self._find_best_path_to_goal()
    
    def _sample_random(self) -> np.ndarray:
        """Sample random point in configuration space"""
        if random.random() < 0.05:  # 5% chance to sample goal
            return self.goal
        
        x = random.uniform(self.bounds[0][0], self.bounds[0][1])
        y = random.uniform(self.bounds[1][0], self.bounds[1][1])
        return np.array([x, y])
    
    def _find_nearest(self, point: np.ndarray) -> int:
        """Find index of nearest vertex to point"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, vertex in enumerate(self.vertices):
            dist = np.linalg.norm(point - vertex)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """Steer from from_point toward to_point by step_size"""
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_point
        
        normalized_direction = direction / distance
        return from_point + normalized_direction * self.step_size
    
    def _find_neighbors(self, point: np.ndarray) -> List[int]:
        """Find all vertices within max_dist"""
        neighbors = []
        for i, vertex in enumerate(self.vertices):
            dist = np.linalg.norm(point - vertex)
            if dist < self.max_dist:
                neighbors.append(i)
        return neighbors
    
    def _is_valid(self, point: np.ndarray) -> bool:
        """Check if point is collision-free"""
        for ox, oy, radius in self.obstacles:
            dist_to_obstacle = np.sqrt((point[0] - ox)**2 + (point[1] - oy)**2)
            if dist_to_obstacle <= radius:
                return False
        return True
    
    def _find_best_path_to_goal(self) -> Optional[List[np.ndarray]]:
        """Find the best path to goal by looking for nearby vertices"""
        # Find vertices near the goal
        goal_neighbors = []
        for i, vertex in enumerate(self.vertices):
            dist_to_goal = np.linalg.norm(vertex - self.goal)
            if dist_to_goal < self.step_size:
                goal_neighbors.append((i, self.costs[i]))
        
        if not goal_neighbors:
            return None  # No vertex near goal
        
        # Select vertex with minimum cost
        best_neighbor_idx = min(goal_neighbors, key=lambda x: x[1])[0]
        
        # Extract path
        path = []
        current_idx = best_neighbor_idx
        while current_idx is not None:
            path.append(self.vertices[current_idx])
            current_idx = self.edges.get(current_idx)
        
        path.reverse()
        return path
```

### Humanoid-Specific Navigation Considerations

Humanoid robots have unique navigation requirements due to their bipedal nature:

#### Footstep Planning

```python
import numpy as np
from typing import List, Tuple

class FootstepPlanner:
    def __init__(self, step_length: float = 0.3, step_width: float = 0.2, max_yaw_rate: float = 0.2):
        """
        Plan footsteps for bipedal locomotion
        Args:
            step_length: Maximum distance between consecutive footsteps
            step_width: Lateral distance between feet
            max_yaw_rate: Maximum turning rate in radians
        """
        self.step_length = step_length
        self.step_width = step_width
        self.max_yaw_rate = max_yaw_rate
    
    def plan_footsteps(self, path: List[np.ndarray], start_pose: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Plan footsteps along a path
        Args:
            path: Global path as list of (x, y) coordinates
            start_pose: Starting pose (x, y, theta)
        Returns:
            List of footstep poses as (x, y, theta)
        """
        footsteps = []
        
        if len(path) < 2:
            return footsteps
        
        # Start with initial pose
        current_x, current_y, current_theta = start_pose
        left_support = True  # Start with left foot support
        
        for i in range(1, len(path)):
            target_x, target_y = path[i][0], path[i][1]
            
            # Calculate direction to next waypoint
            dx = target_x - current_x
            dy = target_y - current_y
            target_theta = np.arctan2(dy, dx)
            
            # Step in the direction, but within physical limits
            dist_to_target = np.sqrt(dx**2 + dy**2)
            
            if dist_to_target > self.step_length:
                # Take a full step
                step_x = current_x + np.cos(target_theta) * self.step_length
                step_y = current_y + np.sin(target_theta) * self.step_length
            else:
                # Take a partial step to reach the target
                step_x = target_x
                step_y = target_y
            
            # Calculate step orientation (blend current with target)
            step_theta = self._blend_orientation(current_theta, target_theta)
            
            # Add footstep based on support leg
            if left_support:
                # Right foot steps forward
                foot_x = step_x + np.sin(step_theta) * self.step_width / 2
                foot_y = step_y - np.cos(step_theta) * self.step_width / 2
                foot_theta = step_theta
            else:
                # Left foot steps forward
                foot_x = step_x - np.sin(step_theta) * self.step_width / 2
                foot_y = step_y + np.cos(step_theta) * self.step_width / 2
                foot_theta = step_theta
            
            footsteps.append((foot_x, foot_y, foot_theta))
            
            # Update current position and support leg
            current_x, current_y, current_theta = foot_x, foot_y, foot_theta
            left_support = not left_support
        
        return footsteps
    
    def _blend_orientation(self, current_theta: float, target_theta: float) -> float:
        """
        Blend current orientation with target, respecting maximum turning rate
        """
        # Calculate the difference in orientation
        diff = target_theta - current_theta
        
        # Normalize to [-π, π] range
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        # Limit the change based on maximum yaw rate
        if abs(diff) > self.max_yaw_rate:
            diff = np.sign(diff) * self.max_yaw_rate
        
        return current_theta + diff

# Example usage
if __name__ == "__main__":
    # Define a simple path
    path = [np.array([x, 0.0]) for x in np.linspace(0, 2, 10)]
    start_pose = (0.0, 0.0, 0.0)
    
    planner = FootstepPlanner()
    footsteps = planner.plan_footsteps(path, start_pose)
    
    print(f"Generated {len(footsteps)} footsteps:")
    for i, footstep in enumerate(footsteps):
        print(f"  Step {i+1}: ({footstep[0]:.2f}, {footstep[1]:.2f}, {footstep[2]:.2f} rad)")
```

## Manipulation Planning

### Inverse Kinematics for Humanoid Arms

Manipulation planning for humanoid robots requires solving inverse kinematics (IK) to determine joint configurations for desired end-effector poses:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, List

class HumanoidIKSolver:
    def __init__(self, arm_chain_lengths: List[float] = [0.3, 0.3, 0.2]):
        """
        Initialize IK solver for humanoid arm
        Args:
            arm_chain_lengths: Lengths of each arm segment
        """
        self.chain_lengths = arm_chain_lengths
        self.total_length = sum(arm_chain_lengths)
        
        # Joint limits (in radians)
        self.joint_limits = [
            (-2.0, 2.0),    # Shoulder yaw
            (-1.5, 1.5),    # Shoulder pitch
            (-2.0, 2.0),    # Shoulder roll
            (-2.0, 2.0),    # Elbow pitch
            (-1.0, 1.0),    # Wrist yaw
            (-2.0, 2.0),    # Wrist pitch
        ]
    
    def solve_ik(self, target_pos: np.ndarray, target_rot: np.ndarray, 
                 current_joints: np.ndarray = None, max_iterations: int = 100, tolerance: float = 1e-4) -> np.ndarray:
        """
        Solve inverse kinematics using Jacobian transpose method
        Args:
            target_pos: Target position (x, y, z)
            target_rot: Target orientation (3x3 rotation matrix)
            current_joints: Current joint angles as starting point
            max_iterations: Maximum number of iterations
            tolerance: Position/orientation tolerance
        Returns:
            Joint angles that achieve the target pose, or None if no solution found
        """
        if current_joints is None:
            # Start with neutral position
            current_joints = np.zeros(len(self.joint_limits))
        
        # Use Jacobian transpose method for IK
        for _ in range(max_iterations):
            # Calculate current end-effector pose
            current_pos, current_rot = self._forward_kinematics(current_joints)
            
            # Calculate errors
            pos_error = target_pos - current_pos
            rot_error = self._rotation_error(current_rot, target_rot)
            
            # Check if we're within tolerance
            if np.linalg.norm(pos_error) < tolerance and np.linalg.norm(rot_error) < tolerance:
                return current_joints
            
            # Calculate Jacobian
            jacobian = self._calculate_jacobian(current_joints)
            
            # Calculate pose error
            pose_error = np.concatenate([pos_error, rot_error])
            
            # Update joint angles using Jacobian transpose
            delta_joints = np.dot(jacobian.T, pose_error)
            
            # Apply joint limits
            current_joints += delta_joints * 0.1  # Learning rate
            
            # Enforce joint limits
            for i, (low, high) in enumerate(self.joint_limits):
                current_joints[i] = np.clip(current_joints[i], low, high)
        
        # If we couldn't converge, return None
        return None
    
    def _forward_kinematics(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate forward kinematics to get end-effector pose
        This is a simplified implementation - in practice, would use DH parameters or similar
        """
        # For this example, we'll use a simplified approach
        # In a real implementation, this would calculate the actual forward kinematics
        
        # Starting from a neutral pose
        pos = np.array([0.3, 0.0, 0.3])  # Starting position at shoulder
        rot = np.eye(3)  # Identity rotation
        
        # Simplified kinematic chain - in reality this would be based on actual DH parameters
        # This is just an example calculation
        for i, joint_angle in enumerate(joints[:3]):  # Simplified for first 3 joints
            # Apply rotation for this joint
            axis = np.array([1, 0, 0]) if i == 0 else np.array([0, 1, 0]) if i == 1 else np.array([0, 0, 1])
            rotation = R.from_rotvec(axis * joint_angle).as_matrix()
            rot = np.dot(rot, rotation)
            
            # Apply translation based on link length
            if i < len(self.chain_lengths):
                translation = np.dot(rot, np.array([self.chain_lengths[i], 0, 0]))
                pos += translation
        
        return pos, rot
    
    def _rotation_error(self, current_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        """
        Calculate rotation error as angle-axis representation
        """
        # Calculate relative rotation
        rel_rot = np.dot(target_rot, current_rot.T)
        
        # Convert to angle-axis representation
        angle_axis = R.from_matrix(rel_rot).as_rotvec()
        
        return angle_axis
    
    def _calculate_jacobian(self, joints: np.ndarray) -> np.ndarray:
        """
        Calculate the geometric Jacobian matrix
        """
        n_joints = len(joints)
        jacobian = np.zeros((6, n_joints))  # 6 DoF (3 pos, 3 rot)
        
        # Calculate Jacobian using numerical differentiation
        delta = 1e-6
        
        # Get current end-effector state
        current_pos, current_rot = self._forward_kinematics(joints)
        
        for i in range(n_joints):
            # Perturb joint i
            joints_plus = joints.copy()
            joints_plus[i] += delta
            pos_plus, rot_plus = self._forward_kinematics(joints_plus)
            
            # Calculate position change
            jacobian[:3, i] = (pos_plus - current_pos) / delta
            
            # Calculate orientation change
            rot_error = self._rotation_error(current_rot, rot_plus)
            jacobian[3:, i] = rot_error / delta
        
        return jacobian

# Example usage
if __name__ == "__main__":
    ik_solver = HumanoidIKSolver()
    
    # Define target pose
    target_pos = np.array([0.5, 0.2, 0.1])
    target_rot = np.eye(3)  # No rotation
    
    # Solve IK
    solution = ik_solver.solve_ik(target_pos, target_rot)
    
    if solution is not None:
        print(f"IK Solution found: {solution}")
        print(f"Joint angles: {[f'{angle:.3f}' for angle in solution]}")
    else:
        print("No IK solution found")
```

### Grasp Planning

Grasp planning is critical for humanoid manipulation:

```python
import numpy as np
from typing import List, Tuple, Optional

class GraspPlanner:
    def __init__(self):
        # Define basic grasp types for humanoid hands
        self.grasp_types = [
            "power_grasp",      # Large objects, high force
            "precise_grasp",    # Small objects, precise control
            "lateral_grasp",    # Grasping objects from the side
            "spherical_grasp",  # Grasping spherical objects
        ]
    
    def plan_grasp(self, object_info: dict) -> Optional[dict]:
        """
        Plan a suitable grasp for the given object
        Args:
            object_info: Dictionary containing object information:
                        - dimensions: [width, height, depth]
                        - shape: "cylinder", "box", "sphere", "irregular"
                        - weight: object weight in kg
                        - surface_properties: "smooth", "rough", "fragile"
        Returns:
            Dictionary with grasp information or None if no suitable grasp found
        """
        dimensions = object_info.get("dimensions", [0.1, 0.1, 0.1])
        shape = object_info.get("shape", "unknown")
        weight = object_info.get("weight", 0.1)
        surface = object_info.get("surface_properties", "regular")
        
        # Determine appropriate grasp based on object properties
        grasp_type = self._select_grasp_type(dimensions, shape, weight, surface)
        
        if grasp_type is None:
            return None
        
        # Calculate grasping pose
        grasp_pose = self._calculate_grasp_pose(object_info, grasp_type)
        
        # Calculate approach direction
        approach_dir = self._calculate_approach_direction(grasp_type)
        
        return {
            "type": grasp_type,
            "pose": grasp_pose,  # (x, y, z, roll, pitch, yaw)
            "approach_direction": approach_dir,
            "required_gripper_width": self._calculate_gripper_width(dimensions, grasp_type),
            "required_force": self._calculate_required_force(weight, surface),
        }
    
    def _select_grasp_type(self, dimensions: List[float], shape: str, weight: float, surface: str) -> Optional[str]:
        """
        Select appropriate grasp type based on object properties
        """
        max_dim = max(dimensions)
        min_dim = min(dimensions)
        
        # Weight-based considerations
        if weight > 2.0:  # Heavy objects
            return "power_grasp"
        
        # Shape-based considerations
        if shape == "sphere":
            return "spherical_grasp"
        elif shape == "cylinder" and max_dim / min_dim > 3:
            # Long cylinder - may need to grasp from the side
            if max_dim > 0.15:  # Very long - power grasp
                return "power_grasp"
            else:
                return "lateral_grasp"
        
        # Size-based considerations
        if max_dim < 0.05:  # Very small objects
            return "precise_grasp"
        elif max_dim < 0.1:  # Small-medium objects
            return "precise_grasp"
        else:  # Large objects
            return "power_grasp"
    
    def _calculate_grasp_pose(self, object_info: dict, grasp_type: str) -> List[float]:
        """
        Calculate the pose for grasping the object
        """
        # For this example, assume object center is at (0, 0, 0)
        # In practice, this would come from perception
        object_center = object_info.get("center", [0.0, 0.0, 0.0])
        
        # Calculate offset based on grasp type
        if grasp_type == "precise_grasp":
            # Grasp at a point appropriate for precision grip
            offset = [0.0, 0.0, 0.05]  # Slightly above center
        elif grasp_type == "power_grasp":
            # Grasp at a point appropriate for power grip
            offset = [0.05, 0.0, 0.0]  # At the side
        else:
            offset = [0.0, 0.0, 0.0]  # Default offset
        
        # Calculate pose (x, y, z, roll, pitch, yaw)
        pose = [
            object_center[0] + offset[0],
            object_center[1] + offset[1], 
            object_center[2] + offset[2],
            0.0,  # roll
            0.0,  # pitch
            0.0   # yaw
        ]
        
        return pose
    
    def _calculate_approach_direction(self, grasp_type: str) -> List[float]:
        """
        Calculate the approach direction for grasping
        """
        # Define approach direction based on grasp type
        if grasp_type == "precise_grasp":
            return [0, 0, -1]  # Approach from above
        elif grasp_type == "power_grasp":
            return [-1, 0, 0]  # Approach from the side
        else:
            return [0, -1, 0]  # Default approach direction
    
    def _calculate_gripper_width(self, dimensions: List[float], grasp_type: str) -> float:
        """
        Calculate required gripper width for the grasp
        """
        max_dim = max(dimensions)
        
        if grasp_type == "precise_grasp":
            return min(max_dim * 1.2, 0.08)  # Limit for precision grasp
        else:
            return max_dim * 1.5  # Allow for power grasp
    
    def _calculate_required_force(self, weight: float, surface: str) -> float:
        """
        Calculate required gripping force
        """
        base_force = weight * 9.81  # Weight in Newtons
        
        # Adjust based on surface properties
        if surface == "smooth":
            multiply_factor = 3.0  # Need more friction
        elif surface == "fragile":
            multiply_factor = 1.5  # Be gentle but secure
        else:
            multiply_factor = 2.0  # Regular grip
        
        return base_force * multiply_factor

# Example usage
if __name__ == "__main__":
    grasp_planner = GraspPlanner()
    
    # Example object
    cup_info = {
        "dimensions": [0.08, 0.1, 0.08],  # width, height, depth in meters
        "shape": "cylinder",
        "weight": 0.3,  # kg
        "surface_properties": "regular",
        "center": [0.5, 0.0, 0.2]  # Position in robot's coordinate frame
    }
    
    grasp_plan = grasp_planner.plan_grasp(cup_info)
    
    if grasp_plan:
        print("Grasp plan calculated:")
        for key, value in grasp_plan.items():
            print(f"  {key}: {value}")
    else:
        print("No suitable grasp found for the object")
```

### Task and Motion Planning (TAMP)

Combining high-level task planning with low-level motion planning:

```python
import numpy as np
from typing import List, Dict, Any

class TaskAndMotionPlanner:
    def __init__(self):
        self.motion_planner = RRTStarPlanner(
            np.array([0, 0]), np.array([1, 1]), 
            [(-1, 2), (-1, 2)], [], 0.1
        )
        self.grasp_planner = GraspPlanner()
    
    def plan_task(self, task_description: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan actions for a complex manipulation task
        Args:
            task_description: Contains high-level task information
        Returns:
            Complete plan with both task and motion components
        """
        # Decompose high-level task
        subtasks = self._decompose_task(task_description)
        
        # Generate complete plan
        complete_plan = {
            "task_sequence": [],
            "motion_sequences": [],
            "grasp_plans": [],
            "execution_order": []
        }
        
        for i, subtask in enumerate(subtasks):
            # Plan task-specific motions
            if subtask["type"] == "navigation":
                motion_plan = self._plan_navigation(subtask)
                complete_plan["motion_sequences"].append(motion_plan)
            
            elif subtask["type"] == "manipulation":
                grasp_plan = self.grasp_planner.plan_grasp(subtask["object_info"])
                motion_plan = self._plan_manipulation(subtask, grasp_plan)
                
                complete_plan["grasp_plans"].append(grasp_plan)
                complete_plan["motion_sequences"].append(motion_plan)
            
            complete_plan["task_sequence"].append(subtask)
            complete_plan["execution_order"].append(i)
        
        return complete_plan
    
    def _decompose_task(self, task_description: Dict[str, Any]) -> List[Dict]:
        """
        Decompose high-level task into subtasks
        """
        task_type = task_description.get("type", "")
        
        if "move" in task_type and "object" in task_description:
            # Example: "Move cup from table to counter"
            return [
                {
                    "type": "navigation",
                    "target_location": task_description["start_location"],
                    "description": f"Navigate to {task_description['start_location']}"
                },
                {
                    "type": "manipulation", 
                    "action": "pick_up",
                    "object_info": task_description["object_info"],
                    "description": f"Pick up {task_description['object_info']['name']}"
                },
                {
                    "type": "navigation",
                    "target_location": task_description["end_location"],
                    "description": f"Navigate to {task_description['end_location']}"
                },
                {
                    "type": "manipulation",
                    "action": "place_down",
                    "object_info": task_description["object_info"],
                    "target_location": task_description["end_location"],
                    "description": f"Place {task_description['object_info']['name']} at destination"
                }
            ]
        
        # Add more task decomposition rules
        return []
    
    def _plan_navigation(self, subtask: Dict) -> List[np.ndarray]:
        """
        Plan navigation for a subtask
        """
        # In practice, this would interface with the actual navigation system
        # For this example, we'll generate a simple path
        
        start_pos = np.array([0.0, 0.0])
        end_pos = np.array([1.0, 1.0])  # Example destination
        
        # Generate path (this would use actual motion planning in practice)
        path = [start_pos]
        steps = 10
        for i in range(1, steps + 1):
            t = i / steps
            pos = start_pos + t * (end_pos - start_pos)
            path.append(pos)
        
        return path
    
    def _plan_manipulation(self, subtask: Dict, grasp_plan: Dict) -> Dict:
        """
        Plan manipulation sequence
        """
        action = subtask["action"]
        
        if action == "pick_up":
            return self._plan_pickup_sequence(subtask, grasp_plan)
        elif action == "place_down":
            return self._plan_placement_sequence(subtask, grasp_plan)
        
        return {}
    
    def _plan_pickup_sequence(self, subtask: Dict, grasp_plan: Dict) -> Dict:
        """
        Plan sequence for picking up an object
        """
        return {
            "approach": {
                "position": [0.0, 0.0, 0.1],  # 10cm above grasp point
                "orientation": [0, 0, 0],
                "description": "Approach object from above"
            },
            "grasp": {
                "position": grasp_plan["pose"][:3],
                "orientation": grasp_plan["pose"][3:],
                "gripper_width": grasp_plan["required_gripper_width"],
                "gripper_force": grasp_plan["required_force"],
                "description": "Execute grasp"
            },
            "lift": {
                "position": [0.0, 0.0, 0.2],  # Lift 20cm
                "description": "Lift object"
            }
        }
    
    def _plan_placement_sequence(self, subtask: Dict, grasp_plan: Dict) -> Dict:
        """
        Plan sequence for placing down an object
        """
        target_pos = subtask.get("target_location", [0, 0, 0])
        return {
            "approach": {
                "position": [target_pos[0], target_pos[1], target_pos[2] + 0.1],
                "orientation": [0, 0, 0],
                "description": f"Approach {subtask.get('object_info', {}).get('name')} placement location"
            },
            "place": {
                "position": target_pos,
                "description": "Release object at target"
            },
            "retract": {
                "position": [target_pos[0], target_pos[1], target_pos[2] + 0.1],
                "description": "Retract gripper"
            }
        }

# Example usage
if __name__ == "__main__":
    tamp_planner = TaskAndMotionPlanner()
    
    # Define a complex task
    task = {
        "type": "move_object",
        "object_info": {
            "name": "cup",
            "dimensions": [0.08, 0.1, 0.08],
            "shape": "cylinder", 
            "weight": 0.3,
            "surface_properties": "regular",
            "center": [0.5, 0.0, 0.2]
        },
        "start_location": "table",
        "end_location": "counter"
    }
    
    complete_plan = tamp_planner.plan_task(task)
    
    print(f"Task plan generated with {len(complete_plan['task_sequence'])} subtasks")
    for i, task_seq in enumerate(complete_plan['task_sequence']):
        print(f"Subtask {i+1}: {task_seq['description']}")
```

## Real-time Planning Considerations

### Incremental Path Planning

For dynamic environments, incremental planning is essential:

```python
import numpy as np
from typing import List, Tuple, Optional

class IncrementalPlanner:
    def __init__(self, grid_map: np.ndarray, resolution: float = 0.1):
        """
        Initialize incremental planner that can update paths in real-time
        """
        self.grid_map = grid_map
        self.resolution = resolution
        self.last_path = None
        self.replan_threshold = 0.5  # Replan if >50% of path is blocked
    
    def update_map(self, new_grid_map: np.ndarray):
        """
        Update the map with new information
        """
        self.grid_map = new_grid_map
    
    def plan_with_replanning(self, start: np.ndarray, goal: np.ndarray, 
                           current_path: List[np.ndarray] = None) -> List[np.ndarray]:
        """
        Plan path considering that we might have a current path
        """
        # Check if we need to replan
        if current_path and not self._path_is_valid(current_path):
            return self._replan_path(start, goal)
        
        # If current path is still mostly valid, continue with it
        return current_path or self._replan_path(start, goal)
    
    def _path_is_valid(self, path: List[np.ndarray]) -> bool:
        """
        Check if the current path is still valid given the updated map
        """
        if not path or len(path) < 2:
            return True
        
        # Convert path points to grid coordinates
        grid_path = [(int(p[1] / self.resolution), int(p[0] / self.resolution)) for p in path]
        
        # Check each segment of the path
        blocked_segments = 0
        total_segments = len(grid_path) - 1
        
        for i in range(total_segments):
            r1, c1 = grid_path[i]
            r2, c2 = grid_path[i+1]
            
            # Check if this segment is blocked
            if self._segment_blocked(r1, c1, r2, c2):
                blocked_segments += 1
        
        # Calculate blocked percentage
        if total_segments > 0:
            blocked_percentage = blocked_segments / total_segments
            return blocked_percentage <= self.replan_threshold
        
        return True
    
    def _segment_blocked(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """
        Check if a segment between two points is blocked
        """
        # Use Bresenham's line algorithm to check intermediate points
        points = self._bresenham_line(r1, c1, r2, c2)
        
        for r, c in points:
            if 0 <= r < self.grid_map.shape[0] and 0 <= c < self.grid_map.shape[1]:
                if self.grid_map[r, c] == 1:  # Occupied
                    return True
        
        return False
    
    def _bresenham_line(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm to get all points on a line
        """
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x_step = 1 if x1 < x2 else -1
        y_step = 1 if y1 < y2 else -1
        
        error = dx - dy
        x, y = x1, y1
        
        while True:
            points.append((x, y))
            if x == x2 and y == y2:
                break
                
            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x += x_step
            if e2 < dx:
                error += dx
                y += y_step
        
        return points
    
    def _replan_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """
        Replan the entire path from scratch
        """
        # Convert to grid coordinates
        start_grid = (int(start[1] / self.resolution), int(start[0] / self.resolution))
        goal_grid = (int(goal[1] / self.resolution), int(goal[0] / self.resolution))
        
        # Use A* planner
        astar = AStarPlanner(self.grid_map, self.resolution)
        grid_path = astar.plan_path(start_grid, goal_grid)
        
        if grid_path:
            # Convert back to world coordinates
            world_path = []
            for r, c in grid_path:
                x = c * self.resolution
                y = r * self.resolution
                world_path.append(np.array([x, y]))
            return world_path
        
        return []

# Example usage
if __name__ == "__main__":
    # Initialize planner
    grid_map = np.zeros((20, 20))
    grid_map[10, 5:15] = 1  # Add an obstacle
    incremental_planner = IncrementalPlanner(grid_map, resolution=0.1)
    
    # Plan initial path
    start = np.array([1.0, 1.0])
    goal = np.array([1.0, 1.8])
    path = incremental_planner.plan_with_replanning(start, goal)
    
    print(f"Initial path has {len(path)} waypoints")
    
    # Update map with new obstacle and replan
    grid_map[5, 8:12] = 1  # Add another obstacle
    incremental_planner.update_map(grid_map)
    
    new_path = incremental_planner.plan_with_replanning(start, goal, path)
    print(f"New path after map update has {len(new_path)} waypoints")
```

## Exercises

1. **Path Planning Comparison Exercise**: Implement and compare different path planning algorithms (A*, Dijkstra, RRT) for humanoid navigation in various environments. Evaluate their performance in terms of computation time, path optimality, and success rate.

2. **Footstep Planning Exercise**: Develop a footstep planner that can handle complex terrain with obstacles, slopes, and varying surfaces. Test it with different walking patterns and gait parameters.

3. **Grasp Planning Exercise**: Create a grasp planner that works with a real humanoid robot model and can handle objects of various shapes, sizes, and weights. Validate the grasps with physics simulation.

4. **TAMP Integration Exercise**: Implement a Task and Motion Planner that integrates high-level task planning with low-level motion planning for a complex manipulation scenario.

5. **Real-Time Planning Exercise**: Develop a real-time planning system that can replan paths as new environmental information becomes available. Test with dynamic obstacle avoidance.

6. **Humanoid Constraints Exercise**: Modify a motion planner to account for humanoid-specific constraints such as balance, joint limits, and stability margins.

## Summary

Navigation and manipulation planning for humanoid robots requires specialized algorithms that account for their unique characteristics. Classical methods like A* and sampling-based approaches like RRT provide the foundation, but they must be adapted to address humanoid-specific challenges like bipedal locomotion, manipulation with anthropomorphic limbs, and complex whole-body coordination. The integration of task planning with motion planning allows humanoid robots to perform complex, multi-step operations in human environments effectively.