# Unity Integration for Humanoid Robotics Visualization

## Introduction

Unity is a powerful real-time 3D development platform that has gained significant traction in robotics for visualization, simulation, and human-robot interaction applications. For humanoid robotics, Unity provides realistic rendering, intuitive visual development tools, and flexible integration with ROS (Robot Operating System) through the ROS-TCP-Connector. This chapter covers how to use Unity to visualize, simulate, and interact with humanoid robots.

## Why Unity for Robotics?

Unity offers several advantages for humanoid robotics applications:

1. **High-Quality Visualization**: Realistic rendering with advanced lighting and materials
2. **Interactive Environment**: Intuitive tools for creating and modifying environments
3. **Cross-Platform Deployment**: Build for PC, mobile, VR, and AR platforms
4. **Asset Ecosystem**: Large marketplace of 3D models, animations, and tools
5. **Robotics Integration**: Direct ROS communication through dedicated packages

## Setting Up Unity for Robotics

### Required Packages

1. **ROS-TCP-Connector**: Enables communication between Unity and ROS
2. **URDF-Importer**: Imports robot models from URDF files
3. **Robotics XR Interaction Package**: For VR/AR applications (optional)
4. **ProBuilder**: For creating simple environments (optional)

### Basic Scene Setup

Create a new Unity 3D project and set up a basic robotics scene:

1. **Create a ground plane**:
   - Right-click in Hierarchy → 3D Object → Plane
   - Scale to appropriate size (e.g., 20x20 units)
   - Add Physics Material to control friction

2. **Add lighting**:
   - Add a Directional Light to simulate sun
   - Adjust intensity and color as needed
   - Consider adding additional lights for indoor environments

3. **Set up camera**:
   - Position the Main Camera to view the workspace
   - Consider adding multiple cameras for different views
   - Set appropriate field of view and clipping planes

4. **Configure physics**:
   - Go to Edit → Project Settings → Physics
   - Adjust Time.fixedDeltaTime for physics simulation
   - Set default material properties (bounciness, friction)

### ROS Connection Setup

1. **Add ROS Connection GameObject**:
   - Create an empty GameObject
   - Add the ROSConnection component
   - Configure IP address and port (typically 10000 for localhost)

2. **Example connection script**:
```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class HumanoidController : MonoBehaviour
{
    ROSConnection ros;
    string rosIP = "127.0.0.1";
    int rosPort = 10000;

    // Joint positions
    float[] jointPositions = new float[28]; // For a typical humanoid
    
    // Start is called before the first frame update
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);
        
        // Subscribe to joint position updates
        ros.Subscribe<sensor_msgs.JointState>("/joint_states", JointStateCallback);
    }

    void JointStateCallback(sensor_msgs.JointState jointState)
    {
        // Update joint positions
        for (int i = 0; i < jointState.position.Count; i++)
        {
            if (i < jointPositions.Length)
            {
                jointPositions[i] = (float)jointState.position[i];
            }
        }
    }

    void Update()
    {
        // Update robot joints based on received positions
        UpdateRobotJoints();
    }

    void UpdateRobotJoints()
    {
        // Example: Update the first joint (assuming it's a child GameObject)
        Transform joint1 = transform.Find("Joint1");
        if (joint1 != null)
        {
            joint1.localEulerAngles = new Vector3(jointPositions[0] * Mathf.Rad2Deg, 0, 0);
        }
        
        // Update other joints similarly...
    }
}
```

## Importing Humanoid Robot Models

### Using URDF Importer

The URDF Importer is a Unity package that allows you to import robot models from URDF files:

1. **Install the URDF Importer package** through the Package Manager

2. **Import the robot**:
   - Go to GameObject → Import Robot from URDF
   - Select your URDF file
   - The URDF importer will automatically create the kinematic structure

3. **Configure joint controllers**:
   - The imported robot will have configurable joint controllers
   - You can set joint limits, motor properties, and PID controllers
   - Joint controllers can be controlled via scripts

### Manual Robot Creation

For complex humanoid robots, you may need to create them manually:

1. **Create link GameObjects**:
   - Represent each physical link as a GameObject
   - Use appropriate 3D primitives (cylinders, boxes, spheres)
   - Set proper scale and orientation

2. **Add colliders**:
   - Add Collider components for physics interactions
   - Use appropriate collider types (Mesh, Capsule, Box)

3. **Create joints**:
   - Use Unity's Joint components (HingeJoint, ConfigurableJoint)
   - Configure joint limits and motor properties
   - Set appropriate anchor points and axes

## Creating Interactive Environments

### Environment Design

Create realistic environments for humanoid robot testing:

```csharp
using UnityEngine;

public class EnvironmentCreator : MonoBehaviour
{
    public GameObject[] obstaclePrefabs;
    public Transform environmentBounds;

    void Start()
    {
        GenerateObstacles();
        AddInteractiveElements();
    }

    void GenerateObstacles()
    {
        // Randomly place obstacles within bounds
        for (int i = 0; i < 10; i++)
        {
            GameObject obstacle = Instantiate(
                obstaclePrefabs[Random.Range(0, obstaclePrefabs.Length)]);
            
            // Position within bounds
            float x = Random.Range(
                environmentBounds.position.x - environmentBounds.localScale.x / 2,
                environmentBounds.position.x + environmentBounds.localScale.x / 2);
            float z = Random.Range(
                environmentBounds.position.z - environmentBounds.localScale.z / 2,
                environmentBounds.position.z + environmentBounds.localScale.z / 2);
                
            obstacle.transform.position = new Vector3(x, 0, z);
        }
    }

    void AddInteractiveElements()
    {
        // Add elements like doors, switches, or objects to manipulate
        // These can have custom scripts for interaction
    }
}
```

### Navigation Mesh

For navigation planning, create a navigation mesh:

1. **Select static environment objects**
2. **Mark them as "Navigation Static" in Inspector**
3. **Go to Window → AI → Navigation**
4. **Select the Bake tab and click Bake**

This allows Unity to calculate walkable areas for navigation planning algorithms.

## Real-time Visualization of Robot Data

### Sensor Data Visualization

Visualize sensor data from the robot:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SensorVisualizer : MonoBehaviour
{
    public GameObject lidarPointPrefab;
    private List<GameObject> lidarPoints = new List<GameObject>();
    
    public void UpdateLidarVisualization(List<float[]> lidarData)
    {
        // Clear previous points
        foreach (GameObject point in lidarPoints)
        {
            DestroyImmediate(point);
        }
        lidarPoints.Clear();
        
        // Create new visualization points
        foreach (float[] point in lidarData)
        {
            GameObject lidarPoint = Instantiate(lidarPointPrefab);
            lidarPoint.transform.position = new Vector3(point[0], point[1], point[2]);
            lidarPoints.Add(lidarPoint);
        }
    }
    
    public void UpdateCameraFeed(Texture2D cameraImage, Renderer cameraRenderer)
    {
        if (cameraRenderer != null && cameraImage != null)
        {
            cameraRenderer.material.mainTexture = cameraImage;
        }
    }
}
```

### Robot State Feedback

Display robot state information using Unity UI:

```csharp
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector;

public class RobotStateUI : MonoBehaviour
{
    public Text jointStateText;
    public Text batteryLevelText;
    public Text statusText;
    
    // Update UI with robot state
    public void UpdateRobotUI(Dictionary<string, float> jointPositions, 
                             float batteryLevel, string status)
    {
        // Format and display joint positions
        string jointInfo = "Joint Positions:\n";
        foreach (var joint in jointPositions)
        {
            jointInfo += $"{joint.Key}: {joint.Value:F3}\n";
        }
        jointStateText.text = jointInfo;
        
        batteryLevelText.text = $"Battery: {batteryLevel:F1}%";
        statusText.text = $"Status: {status}";
    }
}
```

## Advanced Robotics Features

### Inverse Kinematics

Implement inverse kinematics for more natural humanoid movement:

```csharp
using UnityEngine;

public class HumanoidIKController : MonoBehaviour
{
    public Transform leftHandTarget;
    public Transform rightHandTarget;
    public Transform leftFootTarget;
    public Transform rightFootTarget;
    
    public Transform leftHand;
    public Transform rightHand;
    public Transform leftFoot;
    public Transform rightFoot;
    
    void LateUpdate()
    {
        // Simple example - in practice, use Unity's built-in IK solver
        // or a custom implementation
        
        if (leftHandTarget != null && leftHand != null)
        {
            leftHand.position = leftHandTarget.position;
            leftHand.rotation = leftHandTarget.rotation;
        }
        
        if (rightHandTarget != null && rightHand != null)
        {
            rightHand.position = rightHandTarget.position;
            rightHand.rotation = rightHandTarget.rotation;
        }
        
        // Apply similar logic for feet
        if (leftFootTarget != null && leftFoot != null)
        {
            leftFoot.position = leftFootTarget.position;
            leftFoot.rotation = leftFootTarget.rotation;
        }
        
        if (rightFootTarget != null && rightFoot != null)
        {
            rightFoot.position = rightFootTarget.position;
            rightFoot.rotation = rightFootTarget.rotation;
        }
    }
}
```

### Physics Simulation

For physics-based interactions, ensure proper physics configuration:

```csharp
using UnityEngine;

public class HumanoidPhysicsController : MonoBehaviour
{
    public bool enablePhysics = true;
    public float gravityScale = 1.0f;
    
    void Start()
    {
        ConfigurePhysics();
    }
    
    void ConfigurePhysics()
    {
        // Adjust physics properties for the humanoid
        Rigidbody[] rigidbodies = GetComponentsInChildren<Rigidbody>();
        foreach (Rigidbody rb in rigidbodies)
        {
            rb.useGravity = enablePhysics;
            rb.drag = 0.1f;
            rb.angularDrag = 0.05f;
        }
        
        // Adjust gravity scale if needed
        Physics.gravity = new Vector3(0, -9.81f * gravityScale, 0);
    }
    
    void FixedUpdate()
    {
        if (enablePhysics)
        {
            ApplyPhysicsConstraints();
        }
    }
    
    void ApplyPhysicsConstraints()
    {
        // Apply constraints to maintain humanoid structure
        // For example, limit joint angles or prevent self-collision
    }
}
```

## Human-Robot Interaction in Unity

### VR/AR Integration

For immersive interaction, Unity supports VR and AR development:

1. **Install XR packages**:
   - XR Plugin Management
   - OpenXR or Oculus packages depending on target platform

2. **Set up VR camera rig**:
   - Use XR Origin (requires XR packages)

3. **Interaction systems**:
   - Use XR Interaction Toolkit
   - Implement hand tracking or controller-based interaction

### User Interface for Robot Control

Create intuitive interfaces for robot supervision:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class RobotControlPanel : MonoBehaviour
{
    public Slider speedSlider;
    public Button walkButton;
    public Button stopButton;
    public Button resetButton;
    
    void Start()
    {
        SetupControls();
    }
    
    void SetupControls()
    {
        speedSlider.onValueChanged.AddListener(OnSpeedChanged);
        walkButton.onClick.AddListener(OnWalkClicked);
        stopButton.onClick.AddListener(OnStopClicked);
        resetButton.onClick.AddListener(OnResetClicked);
    }
    
    void OnSpeedChanged(float speed)
    {
        // Send speed command to robot via ROS
        // Example: ros.Publish("/cmd_speed", speed);
    }
    
    void OnWalkClicked()
    {
        // Send walk command
        // Example: ros.Publish("/cmd_walk", true);
    }
    
    void OnStopClicked()
    {
        // Send stop command
        // Example: ros.Publish("/cmd_stop", true);
    }
    
    void OnResetClicked()
    {
        // Send reset command
        // Example: ros.Publish("/cmd_reset", true);
    }
}
```

## Performance Optimization

### Rendering Optimization

For real-time robot visualization:

1. **Use Level of Detail (LOD)**:
   - Create simplified versions of the robot model
   - Switch between LODs based on camera distance

2. **Optimize materials**:
   - Use shader variants to reduce complexity
   - Limit texture sizes for performance

3. **Occlusion culling**:
   - Enable occlusion culling in Unity
   - Ensure environment is properly set up for it

### Script Optimization

Optimize scripts for real-time performance:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class OptimizedRobotController : MonoBehaviour
{
    // Cache components to avoid repeated lookups
    private Dictionary<string, Transform> jointCache = new Dictionary<string, Transform>();
    private List<Renderer> robotRenderers = new List<Renderer>();
    
    void Start()
    {
        CacheRobotComponents();
    }
    
    void CacheRobotComponents()
    {
        // Cache all joint transforms
        Transform[] allTransforms = GetComponentsInChildren<Transform>();
        foreach (Transform t in allTransforms)
        {
            if (t.name.Contains("Joint") || t.name.Contains("Link"))
            {
                jointCache[t.name] = t;
            }
            
            Renderer renderer = t.GetComponent<Renderer>();
            if (renderer != null)
            {
                robotRenderers.Add(renderer);
            }
        }
    }
    
    void UpdateRobotJoints(Dictionary<string, float> jointPositions)
    {
        // Apply joint positions using cached transforms
        foreach (var joint in jointPositions)
        {
            if (jointCache.ContainsKey(joint.Key))
            {
                Transform jointTransform = jointCache[joint.Key];
                jointTransform.localEulerAngles = 
                    new Vector3(joint.Value * Mathf.Rad2Deg, 0, 0);
            }
        }
    }
}
```

## Integration with External Systems

### ROS Bridge Configuration

Configure the ROS bridge for efficient communication:

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class ROSBridgeManager : MonoBehaviour
{
    ROSConnection ros;
    
    // Topic names
    string jointStateTopic = "/joint_states";
    string cmdVelTopic = "/cmd_vel";
    string sensorDataTopic = "/robot/sensors";
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize("127.0.0.1", 10000);
        
        // Subscribe to topics
        ros.Subscribe<sensor_msgs.JointState>(jointStateTopic, OnJointStateReceived);
        ros.Subscribe<geometry_msgs.Twist>(cmdVelTopic, OnCmdVelReceived);
    }
    
    void OnJointStateReceived(sensor_msgs.JointState jointState)
    {
        // Process joint state data
        // Update Unity representation of the robot
    }
    
    void OnCmdVelReceived(geometry_msgs.Twist cmdVel)
    {
        // Process velocity commands
        // Update robot visualization or simulation
    }
    
    public void SendRobotCommand(string topic, object message)
    {
        ros.Publish(topic, message);
    }
}
```

## Exercises

1. **Environment Setup Exercise**: Create a Unity project with ROS connection and import a simple humanoid model. Set up a basic scene with ground plane and lighting.

2. **Robot Visualization Exercise**: Implement a script that updates a humanoid robot's joint positions based on ROS joint state messages. Verify that the visual representation matches the reported joint angles.

3. **Sensor Data Visualization Exercise**: Create visualizations for common robot sensors (LiDAR, camera feed, IMU). Display sensor data in the Unity scene.

4. **Interactive Control Exercise**: Implement a UI panel that allows users to control the humanoid robot (e.g., move to positions, change walking gait). Send commands through ROS.

5. **VR/AR Extension Exercise**: (Advanced) Set up a VR environment where users can interact with the humanoid robot using hand tracking or controllers.

## Best Practices

1. **Coordinate System Consistency**: Ensure Unity and ROS coordinate systems are properly aligned (commonly Unity's Y-up vs ROS's Z-up)

2. **Performance Monitoring**: Regularly check frame rate and optimize as needed

3. **Modular Design**: Create reusable components for different robot types

4. **Error Handling**: Implement robust error handling for ROS connection failures

5. **Testing**: Test visualization with various robot configurations and sensor data

## Troubleshooting Common Issues

### ROS Connection Issues
- Verify IP address and port settings
- Check firewall settings
- Ensure ROS bridge is running

### Model Import Problems
- Check that URDF files are properly formatted
- Verify all referenced mesh files exist
- Check that joint limits are reasonable

### Performance Problems
- Reduce model complexity if needed
- Optimize shader usage
- Limit update frequency for high-frequency data

## Summary

Unity provides a powerful platform for humanoid robotics visualization and interaction. By properly setting up ROS integration, importing robot models, and creating optimized visualization systems, you can create compelling real-time applications for robot monitoring, teleoperation, and simulation. The combination of Unity's rendering capabilities and the ROS ecosystem enables rich, interactive experiences for developing and working with humanoid robots.