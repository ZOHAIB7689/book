---
title: NVIDIA Isaac Sim Setup
sidebar_position: 1
---

# NVIDIA Isaac Sim Setup

This document provides instructions for setting up NVIDIA Isaac Sim for advanced robotics simulation as described in the "Physical AI & Humanoid Robotics" textbook.

## Prerequisites

- NVIDIA GPU with Turing, Ampere, or newer architecture (RTX series recommended)
- NVIDIA Driver version 535 or later
- CUDA 11.8 or later
- At least 32GB RAM (64GB recommended for complex humanoid simulations)
- At least 40GB free disk space
- Ubuntu 20.04 or 22.04 LTS, or Windows 10/11 64-bit

## Installing Isaac Sim

Isaac Sim is part of the Isaac suite of tools from NVIDIA and is distributed through the NVIDIA Developer program.

### Getting Isaac Sim

1. Go to the NVIDIA Developer website (developer.nvidia.com) and register for an account if you don't have one.

2. Navigate to the Isaac Sim page and download the appropriate version for your OS:
   - Select "Isaac Sim 2023.1.1" or later for the textbook exercises
   - Download the full standalone package for local development

3. For containerized deployment, you can also pull the Isaac Sim Docker image:
   ```bash
   docker pull nvcr.io/nvidia/isaac-sim:2023.1.1
   ```

### Installing on Windows

1. Run the downloaded installer as Administrator.
2. Follow the installation wizard, accepting the license agreement.
3. Choose an installation directory (default is recommended).
4. The installer will handle all dependencies including Omniverse components.

### Installing on Linux

1. Extract the downloaded archive:
   ```bash
   tar -xzf isaac-sim-2023.1.1.tar.gz
   ```

2. Run the installation script:
   ```bash
   cd isaac-sim-2023.1.1
   bash install.sh
   ```

3. Follow the prompts to complete the installation.

## Post-Installation Setup

### Setting up Environment Variables (Linux)

Add the following to your `.bashrc` to make Isaac Sim available from any terminal:

```bash
export ISAACSIM_PATH=/path/to/isaac-sim
export PATH=$ISAACSIM_PATH/jdk/bin:$ISAACSIM_PATH/python/bin:$PATH
export PYTHONPATH=$ISAACSIM_PATH/python/lib/python3.10/site-packages:$ISAACSIM_PATH/apps:$PYTHONPATH
```

Then source your `.bashrc`:
```bash
source ~/.bashrc
```

### GPU Configuration

Isaac Sim relies heavily on GPU acceleration. Ensure your NVIDIA drivers are properly installed:

1. Check your driver version:
   ```bash
   nvidia-smi
   ```
   You should see a version number of 535 or later.

2. If you have multiple GPUs, you might want to specify which one to use:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

## Launching Isaac Sim

### Launching Standalone

- **Windows**: From the Start menu, search for "Isaac Sim" and launch "Isaac Sim App"
- **Linux**: Run the launch script from your installation directory:
  ```bash
  ./isaac-sim-2023.1.1/python.sh -m omniverse.launcher.core
  ```

### Launching via Isaac Sim Kit

Isaac Sim Kit is a simplified launcher that helps manage different configurations:

1. Launch Isaac Sim Kit (omniverse://isaacsim/isaac-sim-kit)
2. Select the Isaac Sim 2023.1.1 or later entry
3. Click Launch

## Initial Configuration

### Omniverse Connection

1. When launching for the first time, you might need to sign in to Omniverse if prompted.
2. Select "Create a New Account" or use an existing NVIDIA account.
3. Accept the terms and conditions.

### First Launch Settings

1. Upon first launch, Isaac Sim will set up configuration files.
2. It will also download additional assets for the first time, which may take a few minutes.
3. Ensure your firewall allows Isaac Sim to connect to necessary services.

## Isaac ROS Bridge

The Isaac ROS Bridge enables communication between Isaac Sim and ROS 2:

### Installing Isaac ROS Packages

1. Clone the Isaac ROS packages:
   ```bash
   cd ~/humanoid_robot_ws/src
   git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
   git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark.git
   git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git
   git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulators.git
   git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pointcloud_utils.git
   ```

2. Install dependencies:
   ```bash
   cd ~/humanoid_robot_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. Build the packages:
   ```bash
   colcon build --packages-select isaac_ros_common
   ```

## Basic Isaac Sim Operations

### Creating a New Scene

1. Launch Isaac Sim.
2. Go to File > New Scene to create an empty scene.
3. You'll see a default camera and lighting.

### Importing a Humanoid Robot

1. In the Isaac Sim interface, go to Window > Content > Content Browser.
2. Navigate to your robot's USD (Universal Scene Description) or URDF file.
3. Drag and drop it into the viewport.

### Using the Physics Simulation

1. Isaac Sim uses PhysX for physics simulation by default.
2. To enable physics for an object, select it and check the "Rigid Body" or "Articulation Root" in the Property panel.
3. For humanoid robots, use Articulation Root for the main body link to enable complex multi-joint simulation.

## Isaac Sim Python API

Isaac Sim supports Python scripting for automation and programmatic control:

1. In Isaac Sim, you can access the scripting console via Window > Script Editor.
2. You can also write external Python scripts that connect to Isaac Sim.

Example Python script to add a cube to the scene:
```python
import omni
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCube

# Create a world
my_world = World(stage_units_in_meters=1.0)

# Add a cube to the scene
my_world.scene.add(
    VisualCube(
        prim_path="/World/random_cube",
        name="my_cube",
        position=[0, 0, 1.0],
        size=0.5
    )
)

# Reset and step the world
my_world.reset()
for i in range(100):
    my_world.step()
```

## Troubleshooting

- **Isaac Sim fails to launch**: Ensure you have a compatible NVIDIA GPU and updated drivers installed.
- **Poor performance**: Close other GPU-intensive applications, reduce viewport quality, or increase physics substeps.
- **Cannot connect to Omniverse**: Check your internet connection and firewall settings.
- **ROS Bridge not working**: Verify both Isaac Sim and ROS environments are properly sourced.

## Next Steps

With Isaac Sim set up, you can now create advanced humanoid robot simulations with realistic physics, lighting, and sensor simulation. The next section will cover Vision-Language-Action system integration.