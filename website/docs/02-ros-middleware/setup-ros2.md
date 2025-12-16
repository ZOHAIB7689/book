---
title: ROS 2 Development Environment Setup
sidebar_position: 1
---

# ROS 2 Development Environment Setup

This document provides instructions for setting up a ROS 2 development environment for use with the "Physical AI & Humanoid Robotics" textbook.

## Prerequisites

- Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- At least 16GB RAM (32GB recommended for simulations)
- At least 50GB free disk space

## Installing ROS 2 (Humble Hawksbill)

### On Ubuntu:

1. Set up your sources list:
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   ```

2. Add the repository to your sources list:
   ```bash
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. Update your apt repository:
   ```bash
   sudo apt update
   sudo apt upgrade
   ```

4. Install ROS 2 packages:
   ```bash
   # Desktop version includes everything needed for a desktop environment
   sudo apt install ros-humble-desktop
   ```

5. Install development tools:
   ```bash
   sudo apt install python3-colcon-common-extensions
   sudo apt install python3-rosdep python3-vcstool
   ```

6. Source the ROS 2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

7. To automatically source the ROS 2 environment every time you open a new terminal, add the following line to your `~/.bashrc` file:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   ```

### On Windows with WSL2:

1. Install WSL2 with Ubuntu 22.04:
   ```cmd
   wsl --install -d Ubuntu-22.04
   ```

2. After WSL2 is set up, follow the Ubuntu installation instructions above.

## Setting Up Your Workspace

1. Create a ROS 2 workspace:
   ```bash
   mkdir -p ~/humanoid_robot_ws/src
   cd ~/humanoid_robot_ws
   ```

2. Build the workspace:
   ```bash
   colcon build
   ```

3. Source the workspace:
   ```bash
   source install/setup.bash
   ```

## Verifying Installation

Test your installation by running a simple ROS 2 command:

```bash
ros2 topic list
```

This should run without error (although the list will be empty since no nodes are running).

## Installing Additional Packages for Humanoid Robotics

For the specific needs of humanoid robotics covered in this textbook, install these additional packages:

```bash
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-joint-state-publisher
sudo apt install ros-humble-robot-state-publisher
sudo apt install ros-humble-teleop-tools
sudo apt install ros-humble-controllers
sudo apt install ros-humble-moveit
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
```

## Troubleshooting

- If ROS commands are not recognized, make sure you've sourced the setup.bash file or added it to your `.bashrc`.
- If you encounter issues with package installation, try updating your package lists: `sudo apt update`.
- For issues with Gazebo integration, ensure you have a functional graphics environment with hardware acceleration if possible.

## Next Steps

With ROS 2 installed, you're ready to proceed with the Gazebo simulation environment setup. See the next document for Gazebo installation instructions.