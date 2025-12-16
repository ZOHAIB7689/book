---
title: Quickstart Guide
sidebar_position: 1
---

# Quickstart Guide: Physical AI & Humanoid Robotics Book

**For**: University students, AI engineers, Robotics beginners

## Getting Started

This guide will help you set up your development environment to work with the content in "Physical AI & Humanoid Robotics" textbook. By following these steps, you'll have a working environment for running exercises and simulations.

## Prerequisites

Before starting, ensure your system meets these requirements:

- Operating System: Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- RAM: 16GB minimum (32GB recommended for complex simulations)
- Storage: 50GB free space for ROS 2, Gazebo, and Isaac Sim
- GPU: Compatible with CUDA (for Isaac Sim and advanced AI components)

## Setup Steps

### 1. Install ROS 2 (Humble Hawksbill)

#### On Ubuntu:
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep python3-vcstool # For dependency management
```

#### On Windows with WSL2:
```cmd
# Install WSL2 with Ubuntu 22.04
wsl --install -d Ubuntu-22.04
```
Then follow the Ubuntu installation steps inside WSL2.

### 2. Install Gazebo Simulation

```bash
sudo apt install ros-humble-gazebo-*
```

### 3. Install NVIDIA Isaac Sim (Optional but Recommended)

1. Download Isaac Sim from NVIDIA Developer website
2. Follow installation instructions in the Isaac Sim documentation
3. Ensure your system has a compatible NVIDIA GPU with updated drivers

### 4. Create Workspace

```bash
# Create workspace directory
mkdir -p ~/humanoid_robot_ws/src
cd ~/humanoid_robot_ws

# Source ROS 2
source /opt/ros/humble/setup.bash

# Build workspace
colcon build
```

### 5. Clone Textbook Resources

```bash
cd ~/humanoid_robot_ws/src
git clone https://github.com/[your-repo]/physical-ai-textbook-resources.git
cd ~/humanoid_robot_ws
source /opt/ros/humble/setup.bash
colcon build
```

## First Exercise: Hello Humanoid

Let's run your first humanoid robot simulation with basic movement:

1. Source your workspace:
```bash
cd ~/humanoid_robot_ws
source install/setup.bash
```

2. Launch the basic humanoid simulation:
```bash
ros2 launch humanoid_basics hello_humanoid.launch.py
```

3. In another terminal, send a basic movement command:
```bash
# Source the workspace again in the new terminal
cd ~/humanoid_robot_ws
source install/setup.bash

# Send a command to move the humanoid
ros2 topic pub /humanoid/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}'
```

If successful, you should see your humanoid robot moving in the Gazebo simulation!

## Running Chapter Exercises

For each chapter in the textbook, navigate to the corresponding directory and follow the README:

```bash
cd ~/humanoid_robot_ws/src/physical-ai-textbook-resources/chapter_X/exercise_Y
source ~/humanoid_robot_ws/install/setup.bash
# Follow exercise-specific instructions
```

## Troubleshooting

### Common Issues:

**Issue**: Gazebo fails to launch or shows no graphics
**Solution**: Ensure GPU drivers are properly installed and run with:
```bash
export LIBGL_ALWAYS_SOFTWARE=1  # As fallback for OpenGL issues
```

**Issue**: Cannot find ROS packages
**Solution**: Always source your workspace before running ROS commands:
```bash
source ~/humanoid_robot_ws/install/setup.bash
```

**Issue**: Isaac Sim won't launch
**Solution**: Ensure NVIDIA GPU drivers are updated and Isaac Sim is properly licensed

## Development Tools

### Essential Commands:

- **Build workspace**: `cd ~/humanoid_robot_ws && colcon build`
- **Source environment**: `source ~/humanoid_robot_ws/install/setup.bash`
- **Check ROS nodes**: `ros2 node list`
- **Check ROS topics**: `ros2 topic list`
- **Visualize node graph**: `rqt_graph`

## Next Steps

Once you've successfully completed the quickstart:

1. Proceed to Chapter 1 exercises on Physical AI fundamentals
2. Explore the pre-built humanoid models in the simulation environment
3. Try modifying parameters in the sample configurations
4. Join the community forums for troubleshooting and collaboration