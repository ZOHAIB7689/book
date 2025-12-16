---
title: Unity Setup for Robotics Visualization
sidebar_position: 1
---

# Unity Setup for Robotics Visualization

This document provides instructions for setting up Unity for robotics visualization and interaction components as described in the "Physical AI & Humanoid Robotics" textbook.

## Prerequisites

- A Windows, macOS, or Linux system meeting Unity's system requirements
- At least 8GB RAM (16GB recommended)
- A graphics card compatible with DirectX 10 or later (Windows)
- At least 20GB free disk space

## Installing Unity

1. Go to the Unity website (https://unity.com/) and download the Unity Hub installer.

2. Install Unity Hub by following the on-screen instructions.

3. Open Unity Hub and sign in with a Unity ID (free account required).

4. In Unity Hub, go to the "Installs" tab and click "Add" to install a Unity version.

5. For robotics applications, we recommend Unity 2022.3 LTS (Long Term Support) version, which provides stability for long-running projects.

6. During installation, make sure to select the modules relevant to your development:
   - For Windows: "Windows Build Support (IL2CPP)"
   - For macOS: "macOS Build Support"
   - For Linux: "Linux Build Support"

## Unity Robotics Setup

### Installing Unity Robotics Tools

1. In Unity Hub, create a new 3D project (or open an existing one).

2. In the Unity editor, go to Window > Package Manager.

3. In the Package Manager, click on the "+" button in the top-left corner and select "Add package from git URL...".

4. Add the following packages:
   - Unity Robotics Hub: `com.unity.robotics.ros-tcp-connector` (for ROS communication)
   - Unity Robotics Simulation: `com.unity.robotics.urdf-importer` (for importing robot models)

5. For enhanced simulation capabilities, also install:
   - Unity Simulation: `com.unity.simulation`
   - ProBuilder: `com.unity.probuilder` (for creating simple environments)

### Setting up ROS-TCP-Connector

1. In your Unity project, import the ROS-TCP-Connector package if not done automatically.

2. In your scene hierarchy, create an empty GameObject and attach the "ROSConnection" script to it.

3. Configure the IP address and port for ROS communication:
   - IP Address: localhost (127.0.0.1) if running ROS on the same machine
   - Port: 10000 (default, but can be changed)

4. This allows Unity to communicate with ROS nodes for sending/receiving messages.

### Importing Robot Models

1. If you have URDF files for your humanoid robot, import the URDF Importer package.

2. Go to GameObject > Import Robot from URDF in the menu.

3. Select your URDF file, ensuring all referenced mesh files are in the Assets folder.

4. The robot model will be imported with its joint structure preserved.

## Basic Robotics Scene Setup

### Creating a Basic Robot Scene

1. Create a new 3D scene (File > New Scene).

2. Import your humanoid robot model using the URDF Importer.

3. Add a ground plane for the robot to stand on:
   - Create a 3D object > Plane
   - Position it at Y=0
   - Scale as needed

4. Add lighting to the scene:
   - Add a Directional Light to simulate sunlight
   - Add additional lights as needed for visualization

5. Add a camera to view the scene:
   - Position it to get a good view of the robot
   - Consider adding multiple cameras for different perspectives

### Setting up Physics

1. Make sure the ground plane has a static collider component.

2. For the robot, ensure each link has appropriate colliders:
   - Use Box Collider for simple geometric shapes
   - Use Mesh Collider for complex shapes (set as Convex for dynamic objects)

3. Configure the physics properties in Edit > Project Settings > Physics.

## Testing the Setup

1. Create a simple test scene with a basic humanoid model.

2. Add a simple script to control one of the robot's joints using the Transform component.

3. Run the scene to ensure the robot displays correctly and can be controlled.

4. To test ROS integration, run a simple publisher in ROS that sends joint position commands, and have Unity subscribe to these messages to move the robot.

## Troubleshooting

- **URDF Importer not working**: Make sure all mesh files referenced in the URDF are in the Assets folder.
- **Performance issues**: Reduce the complexity of meshes, use lower resolution textures, or reduce the number of lights.
- **ROS connection issues**: Verify IP addresses and ports match between Unity and ROS, check firewall settings.

## Next Steps

With Unity set up for robotics visualization, you can begin creating more complex scenes and integrating with ROS for real-time communication. The next section will cover NVIDIA Isaac Sim setup for advanced simulation capabilities.