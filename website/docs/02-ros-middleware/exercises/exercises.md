---
title: "Exercises: ROS 2 Middleware for Humanoid Robotics"
sidebar_position: 101
---

# Exercises: ROS 2 Middleware for Humanoid Robotics

## Exercise 2.1: Basic Node Creation

### Objective
Create and run a basic ROS 2 node that simulates a humanoid robot's joint controller.

### Instructions
1. Create a new ROS 2 package named `humanoid_basics`
2. Implement a node named `joint_controller` that:
   - Publishes joint position commands to `/joint_commands` topic
   - Subscribes to `/joint_states` topic to receive current positions
   - Implements a simple control loop that adjusts joint positions toward a target
3. Use appropriate message types from `sensor_msgs` and `std_msgs`
4. Add proper logging to track control actions
5. Create a launch file to start your node

### Deliverable
- Source code for the node
- Launch file
- Brief report on the control approach used and challenges encountered

## Exercise 2.2: Quality of Service Configuration

### Objective
Implement and test different QoS profiles for various humanoid robot data streams.

### Instructions
1. Create a publisher that simulates sensor data (e.g., camera images) with "best effort" QoS
2. Create a publisher that simulates control commands with "reliable" QoS
3. Create a publisher for safety-critical messages with persistent durability
4. Set up appropriate subscribers for each topic
5. Test your system under different network conditions (use tools like `tc` to simulate packet loss)
6. Document the behavior differences between QoS configurations

### Deliverable
- Source code for publishers and subscribers with different QoS settings
- Test results showing behavior under different conditions
- Analysis of when to use each QoS configuration in humanoid robotics

## Exercise 2.3: Custom Message Types

### Objective
Create and use custom message types for humanoid-specific data.

### Instructions
1. Define a custom message type `HumanoidGait.msg` that contains:
   - Joint angles for walking pattern
   - Timing information for gait phases
   - Support foot designation (left/right/both)
2. Define a service type `BalanceCorrection.srv` that:
   - Takes current balance state as request
   - Returns correction commands as response
3. Implement a publisher that sends gait commands
4. Implement a service server that provides balance corrections
5. Test communication between these components

### Deliverable
- Definition files for custom message and service types
- Publisher and service server implementations
- Test code demonstrating communication
- Documentation of the message structure and rationale

## Exercise 2.4: Multi-Sensor Integration Node

### Objective
Create a node that integrates data from multiple humanoid robot sensors.

### Instructions
1. Create a node that subscribes to:
   - Joint states (`sensor_msgs/JointState`)
   - IMU data (`sensor_msgs/Imu`)
   - Camera images (`sensor_msgs/Image`)
2. Implement proper synchronization between different message types
3. Process the data to estimate the robot's state (position, balance, etc.)
4. Publish the fused state information
5. Implement appropriate error handling and data validation

### Deliverable
- Source code for the multi-sensor integration node
- Explanation of synchronization approach
- Test results showing state estimation accuracy
- Discussion of challenges in multi-sensor integration

## Exercise 2.5: Parameter-Driven Control Node

### Objective
Create a parameterized humanoid robot controller with runtime configuration.

### Instructions
1. Create a controller node with parameters for:
   - Control gains
   - Joint limits
   - Safety timeouts
   - Gait parameters
2. Use YAML files to configure different walking patterns
3. Implement dynamic parameter updates during execution
4. Add diagnostics to monitor controller health
5. Test parameter changes on the simulated robot

### Deliverable
- Controller source code with parameter handling
- YAML configuration files for different scenarios
- Demonstration of dynamic parameter updates
- Analysis of parameter tuning process

## Exercise 2.6: Performance Optimization Challenge

### Objective
Optimize a publisher-subscriber system for high-frequency humanoid control.

### Instructions
1. Start with a basic joint control system that publishes at 100Hz
2. Measure message latency and CPU usage
3. Apply optimization techniques such as:
   - Appropriate QoS settings
   - Efficient message structures
   - Proper threading configuration
   - Memory allocation optimization
4. Measure performance improvements
5. Identify bottlenecks and solutions

### Deliverable
- Original and optimized code versions
- Performance measurements before and after optimization
- Description of optimization techniques applied
- Analysis of remaining bottlenecks

## Exercise 2.7: Safety System Implementation

### Objective
Design and implement a safety system using ROS 2 publisher-subscriber patterns.

### Instructions
1. Create a safety supervisor node that:
   - Monitors joint states, velocities, and forces
   - Monitors system status (battery, temperature, etc.)
   - Publishes emergency stop commands when limits are exceeded
2. Implement nodes that publish system status information
3. Create a simple simulation that demonstrates safety responses
4. Test the system with various fault conditions

### Deliverable
- Safety supervisor implementation
- Status publisher implementations
- Test scenarios and results
- Discussion of safety system design considerations