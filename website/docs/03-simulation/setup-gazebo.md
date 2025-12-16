---
title: Gazebo Simulation Environment Setup
sidebar_position: 1
---

# Gazebo Simulation Environment Setup

This document provides instructions for setting up the Gazebo simulation environment for use with the "Physical AI & Humanoid Robotics" textbook.

## Prerequisites

- ROS 2 (Humble Hawksbill) already installed and configured
- Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- At least 16GB RAM (32GB recommended for complex simulations)
- A graphics environment with OpenGL support (hardware acceleration recommended)

## Installing Gazebo

Gazebo has been restructured into multiple components. For this textbook, we'll focus on Gazebo Harmonic which is the most recent version.

### On Ubuntu:

1. First, make sure your Ubuntu packages are up to date:
   ```bash
   sudo apt update && sudo apt upgrade
   ```

2. Install Gazebo:
   ```bash
   sudo apt install gazebo
   ```

3. Install the ROS 2 Gazebo interfaces:
   ```bash
   sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins ros-humble-gazebo-dev
   ```

4. Install additional simulation packages for humanoid robotics:
   ```bash
   sudo apt install ros-humble-gazebo-ros-control ros-humble-gazebo-ros-control2
   sudo apt install ros-humble-hardware-interface ros-humble-controller-manager
   sudo apt install ros-humble-joint-state-broadcaster ros-humble-joint-trajectory-controller
   ```

## Configuring Gazebo

### Setting up Environment Variables

Add the following to your `.bashrc` to ensure Gazebo can find your custom models:

```bash
echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/humanoid_robot_ws/src/my_robot/models" >> ~/.bashrc
echo "export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/humanoid_robot_ws/src/my_robot/worlds" >> ~/.bashrc
```

Then source your `.bashrc`:
```bash
source ~/.bashrc
```

## Testing Gazebo Installation

1. Launch Gazebo to verify the installation:
   ```bash
   gazebo
   ```

2. If Gazebo launches successfully with an empty world, your installation is working.

3. To test with ROS 2 integration:
   ```bash
   # First source your ROS 2 environment and workspace
   source /opt/ros/humble/setup.bash
   cd ~/humanoid_robot_ws
   source install/setup.bash
   
   # Launch a sample world
   ros2 launch gazebo_ros empty_world.launch.py
   ```

## Installing Humanoid-Specific Models and Plugins

For humanoid robotics simulation, you'll need additional models and plugins:

1. Create a directory for humanoid models:
   ```bash
   mkdir -p ~/humanoid_robot_ws/src/humanoid_models
   cd ~/humanoid_robot_ws/src/humanoid_models
   ```

2. Download or create humanoid robot models in URDF format. You can start with simple humanoid models for initial testing.

3. Install physics plugins for realistic humanoid simulation:
   ```bash
   sudo apt install ros-humble-builtin_interfaces ros-humble-geometry-msgs ros-humble-sensor-msgs
   sudo apt install ros-humble-trajectory-msgs ros-humble-control-msgs
   ```

## Troubleshooting

- **Gazebo crashes or fails to start**: Ensure your graphics drivers are up to date and hardware acceleration is available.
- **ROS 2 plugins not loading**: Verify that you've properly sourced both ROS 2 and your workspace.
- **Performance issues**: Close other applications while running simulations, and consider reducing visual quality if needed.
- **Model not found errors**: Ensure the GAZEBO_MODEL_PATH environment variable is set correctly.

## Next Steps

With Gazebo installed and configured, you can now create your first humanoid robot simulation. The next section will cover setting up Unity for visualization and interaction.