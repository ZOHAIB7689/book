---
title: ROS 2 Fundamentals for Humanoid Robotics
sidebar_position: 1
---

# ROS 2 Fundamentals for Humanoid Robotics

## Introduction

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms, hardware configurations, and applications. This chapter provides an introduction to ROS 2 concepts specifically relevant to humanoid robotics.

## What is ROS 2?

ROS 2 is the second generation of the Robot Operating System. It addresses many of the limitations of ROS 1, particularly around security, real-time performance, and integration with industrial systems. Unlike a traditional operating system, ROS 2 is a middleware that provides services designed for a heterogeneous computer cluster:

- Hardware abstraction
- Device drivers
- Libraries
- Message-passing
- Package management
- Tools for testing, building, and packaging code

### Key Improvements in ROS 2

- **Quality of Service (QoS) settings**: Allow fine-tuning of message delivery for reliability and performance
- **Security**: Built-in security features for safe robot deployment
- **Real-time support**: Better support for real-time systems
- **Multi-robot systems**: Enhanced capabilities for coordinating multiple robots
- **Official platform support**: No longer limited to Ubuntu/Linux systems

## Core Concepts

### Nodes

In ROS 2, a node is a process that performs computation. Nodes are the fundamental building blocks of a ROS program. In humanoid robotics, nodes typically handle specific functions like:

- Sensor data processing
- Motion control
- Perception algorithms
- Planning and decision making
- Human-robot interaction

Example of a simple node in Python:
```python
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.get_logger().info('Humanoid Controller node initialized')

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics and Messages

Topics are named buses over which nodes exchange messages. In humanoid robotics, common topics include:

- `/joint_states`: Current positions, velocities, and efforts of robot joints
- `/cmd_vel`: Desired linear and angular velocities
- `/imu`: Inertial measurement unit data
- `/camera/image_raw`: Raw image data from robot cameras
- `/scan`: Laser scan data for navigation

Messages are the data packets sent over topics. ROS messages have a specific structure defined in .msg files. Example of a Twist message structure for motion commands:

```
geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular
```

### Services

Services provide a request/response communication model. They're useful for operations that need a response, such as:

- Robot calibration
- Map saving/loading
- Action execution acknowledgment
- Parameter configuration

### Actions

Actions are used for long-running tasks that may take an appreciable amount of time to complete. They provide feedback during execution and can be canceled. In humanoid robotics, actions are commonly used for:

- Navigation to a goal location
- Manipulation tasks
- Complex motion sequences
- Planning operations

## ROS 2 Middleware Implementation

### DDS (Data Distribution Service)

ROS 2 uses DDS as its underlying middleware. DDS is a standard for distributed systems that provides:

- Publisher/subscriber communication
- Discovery mechanisms
- Quality of Service policies
- Data persistence capabilities

Different DDS implementations can be used with ROS 2:
- Fast DDS (default)
- Cyclone DDS
- RTI Connext DDS

### Quality of Service (QoS)

QoS settings allow fine-tuning of communication characteristics:

- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local data
- **History**: Keep all messages vs. keep last N messages
- **Deadline**: Maximum time between messages
- **Liveliness**: How to detect if a participant is alive

For humanoid robotics, QoS is critical for ensuring appropriate handling of different data types:
- Sensor data might use best-effort reliability with small history
- Control commands might use reliable delivery with small history
- Critical safety messages might use reliable delivery with keep-all history

## Creating a ROS 2 Package for Humanoid Robotics

A ROS 2 package is a reusable, shareable collection of code, data, and configuration files. For humanoid robotics, a typical package structure includes:

```
humanoid_control/
├── CMakeLists.txt
├── package.xml
├── src/
│   ├── humanoid_controller.cpp
│   └── joint_feedback_processor.cpp
├── include/
│   └── humanoid_control/
├── launch/
│   └── humanoid.launch.py
├── config/
│   └── humanoid_params.yaml
├── test/
└── scripts/
```

### package.xml

This file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_control</name>
  <version>0.0.0</version>
  <description>Humanoid robot control package</description>
  <maintainer email="maintainer@todo.todo">maintainer</maintainer>
  <license>Apache-2.0</license>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>message_runtime</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Launch Files

Launch files define how to start multiple nodes with specific configurations. For humanoid robotics, launch files often include:

- Robot state publisher to broadcast TF transforms
- Joint state publisher for simulation
- Control nodes for different subsystems
- Perception nodes for sensors
- Visualization tools like RViz

Example launch file:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('humanoid_control')
    
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': True},
                {'robot_description': os.path.join(pkg_share, 'urdf', 'humanoid.urdf')}
            ]
        ),
        Node(
            package='humanoid_control',
            executable='humanoid_controller',
            name='humanoid_controller',
            parameters=[os.path.join(pkg_share, 'config', 'humanoid_params.yaml')]
        )
    ])
```

## Parameter Management

ROS 2 provides flexible parameter management systems, which is important for humanoid robots with many configurable subsystems:

- Parameters can be set at startup via YAML files
- Parameters can be changed dynamically at runtime
- Parameters can be grouped and loaded from different sources

Example parameter file:
```yaml
humanoid_controller:
  ros__parameters:
    control_rate: 100.0  # Hz
    max_velocity: 1.0    # rad/s
    safety_timeout: 0.5  # seconds
    joint_limits:
      hip_pitch_min: -1.57
      hip_pitch_max: 1.57
      knee_pitch_min: 0.0
      knee_pitch_max: 2.0
```

## Working with TF (Transforms)

TF (Transform Frames) is crucial in humanoid robotics for tracking the position and orientation of robot parts relative to each other. The transform tree tracks relationships between different coordinate frames such as:

- `base_link`: The main body of the robot
- `odom`: Odometry frame for navigation
- `map`: Map frame for global localization
- Joint-specific frames for limbs

```python
import tf2_ros
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class TfPublisher(Node):
    def __init__(self):
        super().__init__('tf_publisher')
        self.tf_broadcaster = TransformBroadcaster(self)
        
    def broadcast_transform(self, translation, rotation, parent_frame, child_frame):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]
        
        self.tf_broadcaster.sendTransform(t)
```

## Best Practices for Humanoid Robotics

### Safety Considerations
- Implement safety supervisors that monitor joint limits and velocities
- Use appropriate QoS settings for safety-critical messages
- Include emergency stop mechanisms

### Performance
- Use multi-threaded executors appropriately
- Consider real-time performance requirements
- Optimize message rates for computational constraints

### Debugging
- Use ROS 2 tools like `rqt`, `rviz`, and `ros2 topic echo`
- Implement appropriate logging with different severity levels
- Create diagnostic nodes to monitor system health

## Exercises

1. **Node Creation Exercise**: Create a ROS 2 node that publishes joint position commands for a simple humanoid model (e.g., 6-DOF leg). Use appropriate message types and proper node structure.

2. **Topic Communication Exercise**: Create a publisher-subscriber pair that demonstrates communication between a sensor node and a controller node for a humanoid robot. Include proper error handling and message validation.

3. **Parameter Configuration Exercise**: Create a parameter file for configuring a humanoid robot's joint limits and control parameters. Implement a node that uses these parameters and can update them dynamically.

4. **Launch System Exercise**: Create a launch file that starts a humanoid robot simulation with multiple nodes (state publisher, controller, and a simple sensor). Include appropriate parameter files and ensure proper node startup order.

## Summary

ROS 2 provides a comprehensive framework for developing complex robotic systems like humanoid robots. Its distributed architecture, quality of service controls, and rich ecosystem of tools make it well-suited for the multi-modal, real-time requirements of humanoid robotics. Understanding these fundamentals is essential for building reliable and maintainable humanoid robot systems.