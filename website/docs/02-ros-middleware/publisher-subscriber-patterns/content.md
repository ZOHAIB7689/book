---
title: Publisher-Subscriber Patterns in ROS 2 for Humanoid Robotics
sidebar_position: 1
---

# Publisher-Subscriber Patterns in ROS 2 for Humanoid Robotics

## Introduction

The publisher-subscriber pattern is one of the fundamental communication paradigms in ROS 2. It enables decoupled, asynchronous communication between nodes, making it ideal for the distributed nature of humanoid robotics applications where multiple sensors and actuators need to exchange information efficiently. This chapter explores how to implement and optimize publisher-subscriber patterns specifically for humanoid robotics systems.

## Understanding Publisher-Subscriber Pattern

### Basic Concept

In the publisher-subscriber pattern:
- **Publishers** send messages to named topics without knowledge of subscribers
- **Subscribers** receive messages from specific topics without knowledge of publishers
- **ROS 2 Middleware** handles message routing between publishers and subscribers

This decoupling allows for flexible system architectures where components can be added, removed, or modified without affecting other parts of the system.

### Why Publisher-Subscriber for Humanoid Robotics?

Humanoid robots generate and consume multiple streams of data simultaneously:
- Sensor streams (cameras, IMU, joint encoders, force/torque sensors)
- Control commands (joint positions, velocities, efforts)
- State information (battery level, temperature, system status)

The pub-sub pattern handles these concurrent data streams efficiently with appropriate Quality of Service (QoS) settings tailored to each stream's requirements.

## Implementing Publishers

### Basic Publisher Structure

Here's a basic publisher for joint commands in a humanoid robot:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class JointCommandPublisher : public rclcpp::Node
{
public:
    JointCommandPublisher() : Node("joint_command_publisher")
    {
        publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/joint_commands", 10);
        
        timer_ = this->create_wall_timer(
            50ms, std::bind(&JointCommandPublisher::publish_joint_commands, this));
    }

private:
    void publish_joint_commands()
    {
        auto message = std_msgs::msg::Float64MultiArray();
        // Update joint command values based on control logic
        message.data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};  // Example joint values
        publisher_->publish(message);
    }

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};
```

### Publisher with Custom Message Types

For humanoid robotics, you often need custom message types that represent robot-specific data:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "humanoid_msgs/msg/humanoid_state.hpp"  // Custom message

class HumanoidStatePublisher : public rclcpp::Node
{
public:
    HumanoidStatePublisher() : Node("humanoid_state_publisher")
    {
        publisher_ = this->create_publisher<humanoid_msgs::msg::HumanoidState>(
            "humanoid_state", 10);
    }

private:
    void publish_state()
    {
        auto message = humanoid_msgs::msg::HumanoidState();
        // Fill in robot state information
        message.header.stamp = this->get_clock()->now();
        message.header.frame_id = "base_link";
        
        // Fill joint states
        message.joint_names = {"left_hip", "left_knee", "right_hip", "right_knee"};
        message.joint_positions = {0.1, 0.2, 0.3, 0.4};
        message.joint_velocities = {0.01, 0.02, 0.03, 0.04};
        message.joint_efforts = {1.0, 2.0, 3.0, 4.0};
        
        // Fill balance information
        message.center_of_mass = {0.0, 0.0, 0.8};  // meters
        message.support_polygon = {{-0.1, -0.05}, {0.1, -0.05}, {0.1, 0.05}, {-0.1, 0.05}};
        
        publisher_->publish(message);
    }

    rclcpp::Publisher<humanoid_msgs::msg::HumanoidState>::SharedPtr publisher_;
};
```

## Implementing Subscribers

### Basic Subscriber Structure

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

class JointStateSubscriber : public rclcpp::Node
{
public:
    JointStateSubscriber() : Node("joint_state_subscriber")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&JointStateSubscriber::joint_state_callback, this, std::placeholders::_1));
    }

private:
    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received joint states for %zu joints", msg->name.size());
        
        // Process joint state data
        for (size_t i = 0; i < msg->name.size(); ++i)
        {
            RCLCPP_INFO(this->get_logger(), 
                "Joint %s: position=%.3f, velocity=%.3f, effort=%.3f",
                msg->name[i].c_str(), 
                msg->position[i], 
                msg->velocity[i], 
                msg->effort[i]);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
};
```

### Multiple Subscriptions

Humanoid robots often need to subscribe to multiple topics simultaneously:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/twist.hpp"

class MultiSubscriberNode : public rclcpp::Node
{
public:
    MultiSubscriberNode() : Node("multi_subscriber")
    {
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&MultiSubscriberNode::joint_callback, this, std::placeholders::_1));
        
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu/data", 10,
            std::bind(&MultiSubscriberNode::imu_callback, this, std::placeholders::_1));
            
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10,
            std::bind(&MultiSubscriberNode::cmd_vel_callback, this, std::placeholders::_1));
    }

private:
    void joint_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        joint_state_ = *msg;
        update_robot_state();
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        imu_data_ = *msg;
        update_balance_state();
    }

    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        cmd_vel_ = *msg;
        process_navigation_command();
    }

    void update_robot_state()
    {
        // Update internal state based on all sensor data
        RCLCPP_INFO(this->get_logger(), "Updated robot state with new data");
    }

    void update_balance_state()
    {
        // Update balance state based on IMU data
        RCLCPP_INFO(this->get_logger(), "Updated balance state");
    }

    void process_navigation_command()
    {
        // Process navigation command
        RCLCPP_INFO(this->get_logger(), "Processing navigation command");
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    
    sensor_msgs::msg::JointState joint_state_;
    sensor_msgs::msg::Imu imu_data_;
    geometry_msgs::msg::Twist cmd_vel_;
    std::mutex data_mutex_;
};
```

## Quality of Service (QoS) Settings for Humanoid Robotics

### Understanding QoS Policies

Different humanoid robot data streams have different requirements:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist.hpp"

class QoSDemo : public rclcpp::Node
{
public:
    QoSDemo() : Node("qos_demo")
    {
        // Sensor data: Best effort with small history (old data is irrelevant)
        rclcpp::QoS sensor_qos(10);
        sensor_qos.best_effort().keep_last(5);
        sensor_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("sensor_data", sensor_qos);
        
        // Control commands: Reliable with small history
        rclcpp::QoS control_qos(5);
        control_qos.reliable().keep_last(3).durability_volatile();
        control_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", control_qos);
        
        // Critical safety messages: Reliable with keep-all history
        rclcpp::QoS safety_qos(rclcpp::KeepAll());
        safety_qos.reliable().durability_transient_local();
        safety_pub_ = this->create_publisher<std_msgs::msg::Bool>("emergency_stop", safety_qos);
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr sensor_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr control_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr safety_pub_;
};
```

### Common QoS Patterns for Humanoid Robotics

1. **Sensor Data (Cameras, LiDAR, Joint States)**:
   - Reliability: Best effort (for real-time sensors) or Reliable (for critical sensors)
   - History: Keep last N messages (typically 1-10)
   - Deadline: Appropriate for sensor update frequency

2. **Control Commands**:
   - Reliability: Reliable
   - History: Keep last 1-3 messages
   - Deadline: Based on control loop frequency

3. **State Information**:
   - Reliability: Reliable
   - History: Keep last 1-5 messages
   - Durability: Volatile (unless persistence is needed)

4. **Safety-Critical Messages**:
   - Reliability: Reliable
   - History: Keep all or appropriate number based on use
   - Durability: Transient local for persistence across node restarts

## Advanced Patterns

### Latched Topics for Static Information

For static information that new subscribers need immediately:

```cpp
// Publisher for static TF transforms
rclcpp::QoS latched_qos(1);
latched_qos.transient_local().reliable().keep_last(1);

auto tf_pub = node->create_publisher<tf2_msgs::msg::TFMessage>("tf_static", latched_qos);
```

### Publisher with Callback Group

For complex nodes that need to control execution:

```cpp
#include "rclcpp/rclcpp.hpp"

class ComplexPublisher : public rclcpp::Node
{
public:
    ComplexPublisher() : Node("complex_publisher")
    {
        // Create callback group for high-priority control messages
        auto high_priority_group = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        
        // Publisher with specific callback group
        rclcpp::SubscriptionOptions options;
        options.callback_group = high_priority_group;
        
        control_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10,
            std::bind(&ComplexPublisher::control_callback, this, std::placeholders::_1),
            options);
    }

private:
    void control_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        // High-priority control processing
        RCLCPP_INFO(this->get_logger(), "Processing high-priority control command");
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr control_sub_;
};
```

## Performance Optimization

### Message Efficiency

```cpp
// Using efficient data structures in messages
#include "sensor_msgs/msg/joint_state.hpp"

void optimize_message_size(sensor_msgs::msg::JointState &msg)
{
    // Pre-allocate vectors with known sizes
    msg.name.reserve(28);  // For typical humanoid robot
    msg.position.reserve(28);
    msg.velocity.reserve(28);
    msg.effort.reserve(28);
    
    // Only send necessary data
    // Consider sending only changed values in some cases
}
```

### Threading Considerations

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

class ThreadingExample : public rclcpp::Node
{
public:
    ThreadingExample() : Node("threading_example")
    {
        // Use separate callback groups for different processing requirements
        auto sensor_group = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        
        auto control_group = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        
        rclcpp::SubscriptionOptions sensor_options;
        sensor_options.callback_group = sensor_group;
        
        rclcpp::SubscriptionOptions control_options;
        control_options.callback_group = control_group;
        
        // Sensor processing on separate thread from control
        sensor_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", 10,
            std::bind(&ThreadingExample::image_callback, this, std::placeholders::_1),
            sensor_options);
        
        control_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10,
            std::bind(&ThreadingExample::control_callback, this, std::placeholders::_1),
            control_options);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Image processing - potentially CPU intensive
        RCLCPP_INFO(this->get_logger(), "Processing image data");
    }
    
    void control_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        // Control processing - needs to be timely
        RCLCPP_INFO(this->get_logger(), "Processing control command");
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sensor_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr control_sub_;
};
```

## Error Handling and Robustness

### Publisher with Status Checking

```cpp
class RobustPublisher : public rclcpp::Node
{
public:
    RobustPublisher() : Node("robust_publisher")
    {
        publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
        
        // Timer to check for publisher match
        timer_ = this->create_wall_timer(
            1s, std::bind(&RobustPublisher::check_publisher_status, this));
    }

private:
    void check_publisher_status()
    {
        size_t num_awaiting_subscribers = publisher_->get_subscription_count();
        
        if (num_awaiting_subscribers == 0) {
            RCLCPP_WARN(this->get_logger(), "No subscribers for joint_states topic");
        } else {
            RCLCPP_DEBUG(this->get_logger(), 
                "Found %zu subscribers for joint_states topic", num_awaiting_subscribers);
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};
```

## Exercises

1. **Basic Pattern Exercise**: Create a simple publisher-subscriber pair that simulates a humanoid robot's walking gait. The publisher should send joint positions for walking, and the subscriber should log these positions with timestamps.

2. **QoS Configuration Exercise**: Implement a publisher-subscriber system for different types of humanoid robot data (sensors, controls, safety). Configure appropriate QoS settings for each data type and explain your choices.

3. **Multi-Subscriber Exercise**: Create a node that subscribes to joint states, IMU data, and camera images simultaneously. Implement proper threading and data synchronization between the different message types.

4. **Performance Exercise**: Create a publisher that sends large amounts of data (simulating high-resolution sensors) and optimize it using appropriate ROS 2 patterns and techniques.

## Summary

Publisher-subscriber patterns form the backbone of communication in ROS 2-based humanoid robotics systems. Understanding how to properly implement these patterns with appropriate Quality of Service settings, threading considerations, and error handling is essential for building robust, efficient humanoid robot applications. The decoupled nature of pub-sub communication makes it ideal for the complex, multi-component systems typical in humanoid robotics.