---
title: Gazebo Simulation Tutorials for Humanoid Robotics
sidebar_position: 1
---

# Gazebo Simulation Tutorials for Humanoid Robotics

## Introduction

Gazebo is a powerful 3D simulation environment that provides accurate physics simulation, realistic rendering, and convenient programmatic interfaces. For humanoid robotics, Gazebo offers an ideal platform to test control algorithms, perception systems, and planning approaches before deploying to real robots. This chapter provides comprehensive tutorials for creating and using humanoid robot simulations in Gazebo.

## Gazebo Fundamentals

### Key Components

Gazebo consists of several key components that work together to create realistic simulations:

1. **Physics Engine**: Currently uses ODE, Bullet, or DART for physics calculations
2. **Rendering Engine**: Uses OGRE for 3D visualization
3. **Sensor Simulation**: Simulates various sensor types with realistic noise models
4. **Plugin System**: Allows custom functionalities to be added to the simulation

### Gazebo vs Real Robot Development

Simulations play a critical role in humanoid robotics development:

- **Safety**: Test control algorithms without risk of damaging expensive hardware
- **Repeatability**: Run experiments multiple times with identical conditions
- **Cost-Effectiveness**: Develop and test algorithms without physical robot access
- **Speed**: Accelerate development by running simulations faster than real-time

## Setting Up a Humanoid Robot Model in Gazebo

### URDF to SDF Conversion

Gazebo natively uses SDF (Simulation Description Format), but ROS robots are typically described in URDF (Unified Robot Description Format). The `gazebo_ros` package provides tools to bridge between these formats.

Basic URDF snippet for a humanoid robot:
```xml
<robot name="simple_humanoid">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_leg"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <origin xyz="0 0.1 -0.3"/>
  </joint>

  <link name="left_leg">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.0025"/>
    </inertial>
  </link>
</robot>
```

### Gazebo-Specific Extensions

To make the robot work properly in Gazebo, add Gazebo-specific extensions to your URDF:

```xml
<!-- Gazebo plugin for ROS control -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/simple_humanoid</robotNamespace>
  </plugin>
</gazebo>

<!-- Gazebo-specific material properties -->
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
</gazebo>
```

## Creating Humanoid-Specific Plugins

### Joint Control Plugin

For humanoid robots, precise joint control is essential. Here's a basic joint controller plugin:

```cpp
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <thread>

namespace gazebo
{
  class HumanoidController : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      this->world = _model->GetWorld();
      
      // Get joint handles
      this->left_hip_joint = _model->GetJoint("left_hip_joint");
      this->right_hip_joint = _model->GetJoint("right_hip_joint");
      this->left_knee_joint = _model->GetJoint("left_knee_joint");
      this->right_knee_joint = _model->GetJoint("right_knee_joint");
      
      // Initialize ROS
      if (!ros::isInitialized()) {
        int argc = 0;
        char** argv = NULL;
        ros::init(argc, argv, "gazebo_humanoid_controller",
                 ros::init_options::NoSigintHandler);
      }
      
      this->rosNode.reset(new ros::NodeHandle);
      
      // Subscribe to joint commands
      this->joint_command_sub = this->rosNode->subscribe(
        "/simple_humanoid/joint_commands", 1000,
        &HumanoidController::JointCommandCallback, this);
      
      // Connect to Gazebo update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&HumanoidController::OnUpdate, this));
        
      gzdbg << "HumanoidController loaded for model [" << _model->GetName() << "]\n";
    }
    
    private: void JointCommandCallback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
      if (msg->data.size() >= 4) {
        joint_commands[0] = msg->data[0]; // left_hip
        joint_commands[1] = msg->data[1]; // right_hip
        joint_commands[2] = msg->data[2]; // left_knee
        joint_commands[3] = msg->data[3]; // right_knee
      }
    }
    
    private: void OnUpdate()
    {
      // Apply simple position control
      if (this->left_hip_joint) {
        double error = joint_commands[0] - this->left_hip_joint->GetAngle(0).Radian();
        this->left_hip_joint->SetForce(0, error * 10.0); // Simple P controller
      }
      
      if (this->right_hip_joint) {
        double error = joint_commands[1] - this->right_hip_joint->GetAngle(0).Radian();
        this->right_hip_joint->SetForce(0, error * 10.0);
      }
      
      if (this->left_knee_joint) {
        double error = joint_commands[2] - this->left_knee_joint->GetAngle(0).Radian();
        this->left_knee_joint->SetForce(0, error * 10.0);
      }
      
      if (this->right_knee_joint) {
        double error = joint_commands[3] - this->right_knee_joint->GetAngle(0).Radian();
        this->right_knee_joint->SetForce(0, error * 10.0);
      }
    }
    
    private: physics::ModelPtr model;
    private: physics::WorldPtr world;
    private: physics::JointPtr left_hip_joint, right_hip_joint;
    private: physics::JointPtr left_knee_joint, right_knee_joint;
    private: event::ConnectionPtr updateConnection;
    private: std::unique_ptr<ros::NodeHandle> rosNode;
    private: ros::Subscriber joint_command_sub;
    private: double joint_commands[4] = {0.0, 0.0, 0.0, 0.0};
  };
  
  GZ_REGISTER_MODEL_PLUGIN(HumanoidController)
}
```

### Sensor Plugins for Humanoid Perception

Humanoid robots require various sensors. Here's an example of a plugin for IMU data:

```cpp
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <thread>

namespace gazebo
{
  class HumanoidImuPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      this->link = _model->GetLink("torso"); // IMU mounted on torso
      
      // Initialize ROS
      if (!ros::isInitialized()) {
        int argc = 0;
        char** argv = NULL;
        ros::init(argc, argv, "gazebo_humanoid_imu",
                 ros::init_options::NoSigintHandler);
      }
      
      this->rosNode.reset(new ros::NodeHandle);
      this->imu_pub = this->rosNode->advertise<sensor_msgs::Imu>("/imu/data", 1000);
      
      // Connect to Gazebo update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&HumanoidImuPlugin::OnUpdate, this));
        
      // Set up update rate
      this->update_period_ = 1.0 / 100.0; // 100Hz
      this->last_update_time_ = this->model->GetWorld()->SimTime();
    }
    
    private: void OnUpdate()
    {
      common::Time current_time = this->model->GetWorld()->SimTime();
      if ((current_time - this->last_update_time_).Double() < this->update_period_)
        return;
        
      this->last_update_time_ = current_time;
      
      // Get orientation from link
      ignition::math::Pose3d pose = this->link->WorldPose();
      ignition::math::Vector3d linear_vel = this->link->WorldLinearVel();
      ignition::math::Vector3d angular_vel = this->link->WorldAngularVel();
      
      // Create IMU message
      sensor_msgs::Imu imu_msg;
      imu_msg.header.stamp = ros::Time::now();
      imu_msg.header.frame_id = "imu_link";
      
      // Fill orientation (convert from Gazebo format)
      imu_msg.orientation.x = pose.Rot().X();
      imu_msg.orientation.y = pose.Rot().Y();
      imu_msg.orientation.z = pose.Rot().Z();
      imu_msg.orientation.w = pose.Rot().W();
      
      // Fill angular velocity
      imu_msg.angular_velocity.x = angular_vel.X();
      imu_msg.angular_velocity.y = angular_vel.Y();
      imu_msg.angular_velocity.z = angular_vel.Z();
      
      // Fill linear acceleration (simplified - just gravity in the local frame)
      ignition::math::Vector3d gravity = this->world->Gravity();
      ignition::math::Vector3d linear_acc = this->link->WorldLinearAccel() - gravity;
      
      // Transform to local frame
      ignition::math::Quaterniond rot = pose.Rot().Inverse();
      linear_acc = rot.RotateVector(linear_acc);
      
      imu_msg.linear_acceleration.x = linear_acc.X();
      imu_msg.linear_acceleration.y = linear_acc.Y();
      imu_msg.linear_acceleration.z = linear_acc.Z();
      
      // Add covariance (simplified)
      for (int i = 0; i < 9; i++) {
        imu_msg.orientation_covariance[i] = 0.0;
        imu_msg.angular_velocity_covariance[i] = 0.0;
        imu_msg.linear_acceleration_covariance[i] = 0.0;
      }
      imu_msg.orientation_covariance[0] = 0.01;
      imu_msg.orientation_covariance[4] = 0.01;
      imu_msg.orientation_covariance[8] = 0.01;
      imu_msg.angular_velocity_covariance[0] = 0.01;
      imu_msg.angular_velocity_covariance[4] = 0.01;
      imu_msg.angular_velocity_covariance[8] = 0.01;
      imu_msg.linear_acceleration_covariance[0] = 0.1;
      imu_msg.linear_acceleration_covariance[4] = 0.1;
      imu_msg.linear_acceleration_covariance[8] = 0.1;
      
      this->imu_pub.publish(imu_msg);
    }
    
    private: physics::ModelPtr model;
    private: physics::LinkPtr link;
    private: physics::WorldPtr world;
    private: event::ConnectionPtr updateConnection;
    private: std::unique_ptr<ros::NodeHandle> rosNode;
    private: ros::Publisher imu_pub;
    private: double update_period_;
    private: common::Time last_update_time_;
  };
  
  GZ_REGISTER_MODEL_PLUGIN(HumanoidImuPlugin)
}
```

## Creating World Files

World files define the environment in which your humanoid robot will operate:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="humanoid_world">
    <!-- Include default Gazebo environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Add a simple maze for navigation -->
    <model name="maze_wall_1">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>4 0.2 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>4 0.2 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Your humanoid robot -->
    <include>
      <uri>model://simple_humanoid</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
    
    <!-- Physics parameters -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
  </world>
</sdf>
```

## Launching Gazebo with ROS Integration

### Creating a Launch File

To launch Gazebo with your humanoid robot and ROS integration:

```xml
<launch>
  <!-- Set Gazebo args to use a specific world file -->
  <arg name="world" default="worlds/empty.world"/>
  <arg name="paused" default="false"/>
  <arg name="use_sim_time" default="true"/>
  <arg name="gui" default="true"/>
  <arg name="headless" default="false"/>
  <arg name="debug" default="false"/>
  
  <!-- Launch Gazebo with ROS interface -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(arg world)"/>
    <arg name="paused" value="$(arg paused)"/>
    <arg name="use_sim_time" value="$(arg use_sim_time)"/>
    <arg name="gui" value="$(arg gui)"/>
    <arg name="headless" value="$(arg headless)"/>
    <arg name="debug" value="$(arg debug)"/>
  </include>
  
  <!-- Spawn the humanoid robot -->
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" 
        args="-file $(find simple_humanoid)/urdf/humanoid.urdf 
              -urdf -model simple_humanoid -x 0 -y 0 -z 1"/>
  
  <!-- Publish robot state -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" 
        type="robot_state_publisher" />
  
  <!-- Joint state publisher for GUI control -->
  <node name="joint_state_publisher" pkg="joint_state_publisher" 
        type="joint_state_publisher">
    <param name="use_gui" value="true" />
  </node>
</launch>
```

## Common Simulation Scenarios

### Balance Control Tutorial

Create a scenario to test balance control algorithms:

1. **Setup**: Create a narrow platform for the robot to stand on
2. **Perturbation**: Apply external forces to test balance recovery
3. **Metrics**: Measure center of mass position relative to support polygon

### Walking Gait Tutorial

Implement a basic walking controller in simulation:

```cpp
// Simplified walking pattern generator
class WalkingPatternGenerator {
  public:
    WalkingPatternGenerator() {
      step_length = 0.3;  // meters
      step_height = 0.1;  // meters
      step_duration = 2.0;  // seconds
      current_phase = 0.0;
    }
    
    void update(double dt) {
      current_phase += dt / step_duration;
      if (current_phase > 1.0) current_phase -= 1.0;
    }
    
    double get_left_foot_x() {
      if (current_phase < 0.5) {
        // Left foot swings forward
        return step_length * sin(current_phase * 2 * M_PI);
      } else {
        // Left foot supports
        return 0.0;
      }
    }
    
    double get_right_foot_x() {
      if (current_phase > 0.5) {
        // Right foot swings forward
        return step_length * sin((current_phase - 0.5) * 2 * M_PI);
      } else {
        // Right foot supports
        return 0.0;
      }
    }
    
  private:
    double step_length;
    double step_height;
    double step_duration;
    double current_phase;
};
```

## Debugging and Visualization

### Using Gazebo GUI

The Gazebo GUI provides several tools for debugging:

1. **Model States**: Visualize joint positions and forces
2. **Contacts**: See contact points and forces
3. **Transforms**: View coordinate frames and their relationships
4. **Performance**: Monitor simulation speed and resource usage

### ROS Tools Integration

Combine Gazebo with ROS visualization tools:

```bash
# Visualize robot state in RViz
rosrun rviz rviz

# Monitor topics
rostopic echo /joint_states
rostopic echo /imu/data

# Plot data
rosrun rqt_plot rqt_plot
```

## Performance Optimization

### Physics Parameters

Tune physics parameters for optimal performance:

```xml
<physics type="ode">
  <!-- Smaller step size for accuracy, larger for performance -->
  <max_step_size>0.001</max_step_size>
  
  <!-- Adjust for real-time performance -->
  <real_time_factor>1.0</real_time_factor>
  
  <!-- Update rate affects controller frequency -->
  <real_time_update_rate>1000</real_time_update_rate>
  
  <!-- Solver iterations affect stability -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Model Simplification

For better performance, consider:

1. **Simplified Collision Models**: Use simpler shapes for collision detection
2. **Reduced Visual Detail**: Lower polygon count for visualization
3. **Fewer Sensors**: Only simulate needed sensors
4. **Efficient Plugins**: Optimize update frequencies in plugins

## Exercises

1. **Model Creation Exercise**: Create a simple humanoid model with at least 12 DOF (6 per leg) and import it into Gazebo. Verify that all joints move correctly.

2. **Controller Integration Exercise**: Implement a position controller for your humanoid model that accepts commands via ROS topics and maintains stable joint positions.

3. **Sensor Simulation Exercise**: Add IMU and camera sensors to your humanoid model, publish appropriate ROS topics, and verify they function correctly in simulation.

4. **Balance Challenge**: Create a scenario where your humanoid robot must maintain balance on a narrow platform. Implement a balance controller and test its effectiveness.

5. **Walking Simulation Exercise**: Implement a basic walking gait generator for your humanoid robot in Gazebo. Test its stability on different terrains.

## Best Practices

1. **Start Simple**: Begin with a basic model and gradually add complexity
2. **Parameterize**: Use Xacro macros and parameters to easily modify model properties
3. **Validate Physics**: Check that masses, inertias, and joint limits are physically plausible
4. **Test Incrementally**: Verify each component before integrating with others
5. **Monitor Performance**: Keep an eye on simulation speed and adjust parameters as needed

## Troubleshooting Common Issues

### Robot Falls Through Ground
- Check that collision geometries are properly defined
- Verify that the model is above ground level at spawn
- Ensure inertial parameters are properly set

### Joints Behaving Erratically
- Check joint limits and safety controllers
- Verify physics parameters (solver iterations, CFM, ERP)
- Ensure appropriate damping and friction values

### ROS Communication Issues
- Verify that ROS IP addresses are correctly configured
- Check that necessary Gazebo-ROS plugins are loaded
- Ensure proper namespace usage in topics and parameters

## Summary

Gazebo provides a powerful and flexible platform for humanoid robotics simulation. By understanding how to properly configure models, integrate with ROS, and optimize simulation parameters, you can create realistic and effective simulation environments for developing and testing humanoid robot algorithms. The iterative process of simulation and real-world testing will help you build more robust and capable humanoid robots.