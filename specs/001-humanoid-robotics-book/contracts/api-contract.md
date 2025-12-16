# API Contract: Humanoid Robot Control Interface

**Feature**: 001-humanoid-robotics-book
**Version**: 1.0
**Date**: 2025-12-16

## Overview

This document defines the standard interface for controlling humanoid robots in simulation and real-world environments. The interface enables consistent interaction patterns between AI systems and physical robotic embodiments.

## Message Types

### JointState
Used to communicate joint angles and velocities
```
std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
```

### Twist
Used to command linear and angular velocity
```
geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular
```

### PointCloud2
Used for 3D sensing data
```
sensor_msgs/PointField[5] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
```

## Service Definitions

### MotorControl
Interface for low-level motor control
```
# Request
float64[] joint_positions
float64[] joint_velocities
float64[] joint_torques

# Response
bool success
string error_message
```

### PerceptionService
Interface for sensor data processing
```
# Request
sensor_msgs/Image image
sensor_msgs/PointCloud2 point_cloud

# Response
object_detection_msgs/DetectionArray detections
bool success
string error_message
```

## Topic Interface

### /humanoid/joint_states
- **Type**: sensor_msgs/JointState
- **Direction**: Publisher (Robot → Controller)
- **Rate**: 100 Hz
- **Description**: Current joint positions, velocities, and efforts

### /humanoid/cmd_vel
- **Type**: geometry_msgs/Twist
- **Direction**: Subscriber (Controller → Robot)
- **Rate**: 50 Hz
- **Description**: Desired linear and angular velocities

### /humanoid/head_camera/color/image_raw
- **Type**: sensor_msgs/Image
- **Direction**: Publisher (Robot → Controller)
- **Rate**: 30 Hz
- **Description**: Raw RGB image from humanoid head camera

### /humanoid/laser_scan
- **Type**: sensor_msgs/LaserScan
- **Direction**: Publisher (Robot → Controller)
- **Rate**: 40 Hz
- **Description**: 2D laser scan data for navigation

## Action Interface

### MoveHumanoidAction
High-level motion control action
```
# Goal
geometry_msgs/PoseStamped target_pose
float64 max_velocity
float64 timeout

# Result
bool success
string message
geometry_msgs/Pose final_pose

# Feedback
geometry_msgs/Pose current_pose
float64 distance_remaining
```

## Quality of Service Requirements

- **Latency**: Joint control messages < 10ms
- **Reliability**: 99.9% for safety-critical topics
- **Bandwidth**: 100 Mbps available for sensor streams
- **Synchronization**: Joint state timestamps must align with controller cycles

## Safety Constraints

1. **Velocity Limits**: All commanded velocities must stay within actuator limits
2. **Position Bounds**: Joint positions must remain within safe operational range
3. **Emergency Stop**: E-stop topic must always be monitored
4. **Collision Avoidance**: Motion plans must pass collision check before execution

## Test Requirements

### Unit Tests
- Verify message serialization/deserialization
- Test service call response times
- Validate topic data integrity

### Integration Tests
- Test complete robot control loop
- Verify safety constraint enforcement
- Validate graceful degradation on failure

### Performance Tests
- Measure latency under load
- Verify real-time performance characteristics
- Test system stability over extended periods