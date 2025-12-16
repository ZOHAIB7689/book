---
title: Perception Modules for Vision-Language-Action Systems
sidebar_position: 1
---

# Perception Modules for Vision-Language-Action Systems

## Introduction

Perception modules form the foundation of Vision-Language-Action (VLA) systems in humanoid robotics. These modules are responsible for transforming raw sensor data into meaningful representations that can be used by language understanding and action generation components. For humanoid robots operating in human environments, perception systems must be robust, real-time capable, and capable of understanding complex scenes with multiple objects, agents, and dynamic elements.

## Core Perception Components

### Visual Perception Pipeline

The visual perception pipeline in humanoid robots typically consists of multiple stages:

1. **Low-Level Processing**:
   - Image rectification and distortion correction
   - Color space conversion and normalization
   - Noise reduction and enhancement

2. **Feature Extraction**:
   - Edge detection and segmentation
   - Keypoint detection and description
   - Deep feature extraction using CNNs

3. **Object Recognition**:
   - Object detection and classification
   - Instance segmentation
   - 3D object pose estimation

4. **Scene Understanding**:
   - Semantic segmentation
   - Scene graph construction
   - Affordance detection

### Multi-Modal Sensor Fusion

Humanoid robots are equipped with diverse sensors that must be integrated for comprehensive perception:

1. **RGB Cameras**: Visual information for object recognition and scene understanding
2. **Depth Sensors**: 3D information for navigation and manipulation
3. **IMU (Inertial Measurement Unit)**: Robot orientation and acceleration
4. **Joint Encoders**: Robot configuration and self-awareness
5. **Force/Torque Sensors**: Interaction and manipulation feedback
6. **Microphones**: Audio for speech and sound recognition

## Object Recognition and Detection

### Deep Learning Approaches

Modern object recognition in humanoid robots relies heavily on deep learning:

1. **Convolutional Neural Networks (CNNs)**:
   - Feature extraction and object classification
   - Real-time processing capabilities
   - Transfer learning from pre-trained models

2. **Object Detection Networks**:
   - YOLO (You Only Look Once) for real-time detection
   - R-CNN variants for high-accuracy detection
   - Single shot detectors for speed-accuracy tradeoff

### Example: Real-time Object Detection

```python
import torch
import torchvision
import cv2
import numpy as np

class RealTimeObjectDetector:
    def __init__(self, model_name="yolov5s", confidence_threshold=0.5):
        # Load pre-trained model
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.confidence_threshold = confidence_threshold
        self.model.eval()
        
    def detect_objects(self, image):
        """
        Detect objects in an image
        Args:
            image: numpy array of shape (H, W, C) in BGR format
        Returns:
            detections: list of detected objects with bounding boxes and class labels
        """
        # Convert BGR to RGB for the model
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(rgb_image)
        
        # Extract detections
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            if conf > self.confidence_threshold:
                detection = {
                    'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    'confidence': conf,
                    'class_id': int(cls),
                    'class_name': self.model.names[int(cls)]
                }
                detections.append(detection)
        
        return detections
    
    def annotate_image(self, image, detections):
        """
        Annotate image with detection results
        """
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_image, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated_image

# Usage example
detector = RealTimeObjectDetector()
image = cv2.imread("scene.jpg")
detections = detector.detect_objects(image)
annotated_image = detector.annotate_image(image, detections)
cv2.imshow("Detections", annotated_image)
```

### 3D Object Detection and Pose Estimation

For humanoid robotics, 3D object information is crucial:

1. **Depth-Based Detection**: Combining RGB and depth data for 3D localization
2. **Multi-View Fusion**: Using multiple cameras for complete object models
3. **Pose Estimation**: Estimating object pose (position and orientation) in 3D space
4. **Shape Reconstruction**: Building 3D models of objects for manipulation planning

### Example: 3D Pose Estimation

```python
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class ObjectPoseEstimator:
    def __init__(self, camera_matrix):
        """
        Initialize with camera calibration matrix
        camera_matrix: 3x3 camera intrinsic matrix
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = None  # Assuming no distortion for simplicity
        
    def estimate_pose(self, object_points_3d, image_points_2d):
        """
        Estimate object pose using PnP algorithm
        Args:
            object_points_3d: 3D points of the object in object coordinate frame
            image_points_2d: Corresponding 2D points in image
        Returns:
            rotation_vector: Rotation vector (Rodrigues format)
            translation_vector: Translation vector [x, y, z]
        """
        if len(object_points_3d) != len(image_points_2d) or len(object_points_3d) < 4:
            raise ValueError("Need at least 4 coplanar points to estimate pose")
            
        # Solve Perspective-n-Point problem
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points_3d, 
            image_points_2d, 
            self.camera_matrix, 
            self.dist_coeffs
        )
        
        if not success:
            raise ValueError("Pose estimation failed")
            
        return rotation_vector, translation_vector
    
    def convert_to_transform_matrix(self, rotation_vector, translation_vector):
        """
        Convert rotation vector and translation vector to 4x4 transformation matrix
        """
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Create transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation_vector.flatten()
        
        return transform_matrix

# Usage example
# Define 3D model points of an object (e.g., a cube)
object_points_3d = np.array([
    [0, 0, 0],      # Point 1: Origin
    [0.1, 0, 0],    # Point 2: X-axis
    [0.1, 0.1, 0],  # Point 3: XY-plane
    [0, 0.1, 0],    # Point 4: Y-axis
    [0, 0, 0.1]     # Point 5: Z-axis
], dtype=np.float32)

# Define corresponding 2D image points (this would come from feature matching)
image_points_2d = np.array([
    [100, 150],  # 2D point corresponding to 3D point 1
    [120, 150],  # 2D point corresponding to 3D point 2
    [120, 170],  # 2D point corresponding to 3D point 3
    [100, 170],  # 2D point corresponding to 3D point 4
    [100, 150]   # 2D point corresponding to 3D point 5
], dtype=np.float32)

# Camera calibration matrix (replace with actual values from calibration)
camera_matrix = np.array([
    [525.0, 0, 319.5],
    [0, 525.0, 239.5],
    [0, 0, 1]
])

# Estimate pose
pose_estimator = ObjectPoseEstimator(camera_matrix)
rvec, tvec = pose_estimator.estimate_pose(object_points_3d, image_points_2d)
transform_matrix = pose_estimator.convert_to_transform_matrix(rvec, tvec)

print("Object pose transformation matrix:")
print(transform_matrix)
```

## Scene Understanding and Semantic Segmentation

### Semantic Segmentation for Robotics

Semantic segmentation assigns a class label to each pixel in an image, enabling detailed scene understanding:

```python
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

class SemanticSegmentation:
    def __init__(self, model_name='deeplabv3_resnet101'):
        # Load pre-trained DeepLabV3 model
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.eval()
        
        # Define preprocessing transforms
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(520),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # COCO dataset class names (for reference)
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def segment_image(self, image):
        """
        Perform semantic segmentation on an image
        Args:
            image: numpy array of shape (H, W, C) in RGB format
        Returns:
            segmentation_mask: 2D array with class IDs
            color_mask: 3D array with color-coded segmentation
        """
        input_tensor = self.preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            segmentation_mask = output.argmax(0).detach().cpu().numpy()
        
        # Create color-coded mask
        color_mask = self._create_color_mask(segmentation_mask)
        
        return segmentation_mask, color_mask
    
    def _create_color_mask(self, segmentation_mask):
        """
        Create a color-coded mask for visualization
        """
        # Simple color mapping (in practice, use more sophisticated coloring)
        unique_labels = np.unique(segmentation_mask)
        color_map = np.zeros((len(self.class_names), 3))
        
        # Generate random colors for each class
        np.random.seed(42)  # For reproducible results
        for i in range(len(self.class_names)):
            color_map[i] = np.random.randint(0, 255, 3)
        
        # Create color mask
        color_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
        for label in unique_labels:
            if label < len(color_map):
                color_mask[segmentation_mask == label] = color_map[label]
        
        return color_mask

# Usage example
segmenter = SemanticSegmentation()
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Placeholder image
seg_mask, color_mask = segmenter.segment_image(image)
```

### Affordance Detection

Affordance detection identifies what actions are possible with objects:

1. **Grasp Affordances**: Where and how to grasp objects
2. **Support Affordances**: Where objects can be placed
3. **Functional Affordances**: What an object is used for
4. **Navigation Affordances**: Where the robot can move

### Spatial Relation Understanding

Understanding spatial relationships is crucial for humanoid robots:

1. **Object-Object Relations**: "the cup is on the table"
2. **Object-Agent Relations**: "person is sitting on the chair"  
3. **Object-Environment Relations**: "the door is at the end of the hallway"
4. **Temporal Relations**: "the person was standing then sat down"

## Human Detection and Tracking

### Person Detection and Pose Estimation

For humanoid robots interacting with humans, understanding human presence and pose is essential:

```python
import cv2
import numpy as np

class HumanPoseEstimator:
    def __init__(self, model_path="pose_model.onnx"):
        # Load OpenPose model
        self.pose_net = cv2.dnn.readNetFromONNX(model_path)
        self.pose_pairs = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], 
                          [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], 
                          [1,0], [0,14], [14,16], [0,15], [15,17]]
    
    def estimate_human_pose(self, image):
        """
        Estimate human pose in an image
        Returns:
            pose_info: List of keypoints for detected humans
        """
        # Prepare image for pose estimation
        blob = cv2.dnn.blobFromImage(
            image, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False
        )
        
        # Run pose estimation
        self.pose_net.setInput(blob)
        output = self.pose_net.forward()
        
        # Process output to extract keypoints
        # (This is a simplified example - actual implementation would be more complex)
        height, width = image.shape[:2]
        keypoints = []
        
        # Placeholder for processing confidence maps to extract keypoints
        for i in range(output.shape[1]):  # For each keypoint
            prob_map = output[0, i, :, :]
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            
            # Scale keypoint to original image size
            x = (width * point[0]) / output.shape[3]
            y = (height * point[1]) / output.shape[2]
            
            if prob > 0.1:  # Confidence threshold
                keypoints.append((x, y, prob))
            else:
                keypoints.append((0, 0, 0))
        
        return keypoints
    
    def visualize_pose(self, image, keypoints):
        """
        Visualize human pose on image
        """
        output_image = image.copy()
        
        # Draw keypoints
        for i, (x, y, prob) in enumerate(keypoints):
            if prob > 0.1:
                cv2.circle(output_image, (int(x), int(y)), 5, (0, 255, 255), -1)
                cv2.putText(output_image, str(i), (int(x), int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw connections between keypoints
        for pair in self.pose_pairs:
            part_a, part_b = pair
            ka = keypoints[part_a]
            kb = keypoints[part_b]
            
            if ka[2] > 0.1 and kb[2] > 0.1:  # Both keypoints have sufficient confidence
                cv2.line(output_image, (int(ka[0]), int(ka[1])), 
                        (int(kb[0]), int(kb[1])), (255, 0, 0), 2)
        
        return output_image

# Usage example
pose_estimator = HumanPoseEstimator()
image = cv2.imread("human_image.jpg")
keypoints = pose_estimator.estimate_human_pose(image)
visualized_image = pose_estimator.visualize_pose(image, keypoints)
```

### Multi-Person Tracking

Tracking multiple humans in dynamic environments:

```python
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import cv2

class MultiPersonTracker:
    def __init__(self):
        self.trackers = []  # List of active trackers
        self.next_id = 0    # ID for next person
        self.max_disappeared = 10  # Max frames to keep inactive tracker
        self.max_distance = 50     # Max distance for association
    
    def init_kalman_filter(self):
        """Initialize Kalman filter for tracking"""
        kf = KalmanFilter(dim_x=7, dim_z=4)  # State: [x, y, s, r, vx, vy, vs] (x, y, scale, aspect_ratio)
        
        # State transition matrix
        kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1]])
        
        # Measurement function
        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]])
        
        # Covariance matrices
        kf.R[2:, 2:] *= 10.  # Measurement uncertainty
        kf.P[4:, 4:] *= 1000.  # Initial uncertainty
        kf.P *= 10.  # Initial uncertainty
        kf.Q[-1, -1] *= 0.01  # Process noise
        kf.Q[4:, 4:] *= 0.01
        
        return kf
    
    def update(self, detections):
        """
        Update tracker with new detections
        Args:
            detections: List of bounding boxes [x, y, w, h]
        """
        # If no active trackers, initialize from detections
        if len(self.trackers) == 0:
            for det in detections:
                tracker = self.init_kalman_filter()
                # Initialize tracker with detection
                tracker.x[:4] = det[:4].tolist()
                self.trackers.append({
                    'kf': tracker,
                    'id': self.next_id,
                    'disappeared': 0,
                    'bbox': det
                })
                self.next_id += 1
        else:
            # Calculate distances between trackers and detections
            if len(detections) > 0:
                # Predict new positions
                for tracker in self.trackers:
                    tracker['kf'].predict()
                
                # Calculate distances
                distances = np.zeros((len(self.trackers), len(detections)))
                for i, tracker in enumerate(self.trackers):
                    for j, det in enumerate(detections):
                        # Calculate distance between predicted position and detection
                        pred = tracker['kf'].x[:4].astype(int)
                        dist = np.sqrt((pred[0] - det[0])**2 + (pred[1] - det[1])**2)
                        distances[i, j] = dist
                
                # Associate trackers with detections
                row_indices, col_indices = linear_sum_assignment(distances)
                
                # Update matched trackers
                used_det_indices = set()
                for i, j in zip(row_indices, col_indices):
                    if distances[i, j] < self.max_distance:
                        # Update tracker with detection
                        self.trackers[i]['kf'].update(detections[j])
                        self.trackers[i]['bbox'] = detections[j]
                        self.trackers[i]['disappeared'] = 0
                        used_det_indices.add(j)
                
                # Mark unmatched trackers as disappeared
                for i in range(len(self.trackers)):
                    if i not in row_indices:
                        self.trackers[i]['disappeared'] += 1
                        if self.trackers[i]['disappeared'] > self.max_disappeared:
                            del self.trackers[i]
                
                # Create new trackers for unmatched detections
                for j in range(len(detections)):
                    if j not in used_det_indices:
                        tracker = self.init_kalman_filter()
                        tracker.x[:4] = detections[j].tolist()
                        self.trackers.append({
                            'kf': tracker,
                            'id': self.next_id,
                            'disappeared': 0,
                            'bbox': detections[j]
                        })
                        self.next_id += 1
            else:
                # No detections, mark all trackers as disappeared
                for tracker in self.trackers:
                    tracker['disappeared'] += 1
                
                # Remove trackers that have disappeared too long
                self.trackers = [t for t in self.trackers if t['disappeared'] <= self.max_disappeared]
    
    def get_tracked_objects(self):
        """Return current tracked objects"""
        return [{'id': t['id'], 'bbox': t['bbox']} for t in self.trackers]

# Usage example
tracker = MultiPersonTracker()
# detections would come from person detector
detections = np.array([[100, 150, 50, 100], [300, 200, 45, 95]])  # Example detections
tracker.update(detections)
tracked_objects = tracker.get_tracked_objects()
print(tracked_objects)
```

## Sensor Fusion Techniques

### Kalman Filtering for Multi-Sensor Integration

Kalman filters are essential for fusing information from multiple sensors:

```python
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

class MultiSensorFusion:
    def __init__(self):
        # Initialize Kalman filter for state estimation
        # State: [x, y, z, vx, vy, vz] (position and velocity)
        self.kf = KalmanFilter(dim_x=6, dim_z=3)  # 3D position measurements
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + vx*dt
            [0, 1, 0, 0, 1, 0],  # y = y + vy*dt  
            [0, 0, 1, 0, 0, 1],  # z = z + vz*dt
            [0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 1]   # vz = vz
        ])
        
        # Measurement function (only position is measured)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # Measure x
            [0, 1, 0, 0, 0, 0],  # Measure y
            [0, 0, 1, 0, 0, 0]   # Measure z
        ])
        
        # Initial uncertainty
        self.kf.P *= 1000.0
        
        # Process noise (how much we expect the model to deviate)
        self.kf.Q = np.eye(6) * 0.1
        
        # Measurement noise (sensor accuracy)
        self.kf.R = np.eye(3) * 10  # Position measurement noise
        
        # Initial state
        self.kf.x = np.array([0, 0, 0, 0, 0, 0]).T  # x, y, z, vx, vy, vz
    
    def update_with_camera(self, camera_pos):
        """
        Update state estimate with camera measurement
        Args:
            camera_pos: [x, y, z] position from camera
        """
        self.kf.update(camera_pos)
    
    def update_with_imu(self, acceleration, dt):
        """
        Update with IMU acceleration data
        Args:
            acceleration: [ax, ay, az] acceleration from IMU
            dt: Time step
        """
        # Integrate acceleration to update velocity prediction
        self.kf.x[3] += acceleration[0] * dt  # Update vx
        self.kf.x[4] += acceleration[1] * dt  # Update vy
        self.kf.x[5] += acceleration[2] * dt  # Update vz
        
        # Update state transition matrix with dt
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt
    
    def predict(self, dt):
        """
        Predict next state based on current state
        """
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt
        self.kf.predict()
        return self.kf.x
```

### Particle Filtering for Non-Linear Systems

For non-linear perception problems, particle filtering can be more appropriate:

```python
import numpy as np
import matplotlib.pyplot as plt

class ParticleFilter:
    def __init__(self, num_particles=1000, state_dim=2):
        self.num_particles = num_particles
        self.state_dim = state_dim
        
        # Initialize particles
        self.particles = np.random.normal(0, 1, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles
    
    def predict(self, control_input, noise_std=0.1):
        """
        Predict next state based on control input
        """
        # Add process noise
        noise = np.random.normal(0, noise_std, self.particles.shape)
        self.particles += control_input + noise
    
    def update(self, measurement, measurement_error_std=0.5):
        """
        Update particle weights based on measurement
        """
        # Calculate likelihood of each particle given measurement
        diff = self.particles - measurement
        distances = np.sum(diff**2, axis=1)  # Squared distances
        
        # Calculate weights based on measurement likelihood
        likelihoods = np.exp(-distances / (2 * measurement_error_std**2))
        self.weights *= likelihoods
        
        # Normalize weights
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)
    
    def resample(self):
        """
        Resample particles based on their weights
        """
        # Systematic resampling
        indices = []
        step = 1.0 / self.num_particles
        start = np.random.uniform(0, step)
        
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.num_particles:
            if start + j * step <= cumulative_sum[i]:
                indices.append(i)
                j += 1
            else:
                i += 1
        
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def estimate(self):
        """
        Calculate state estimate as weighted mean of particles
        """
        estimate = np.average(self.particles, axis=0, weights=self.weights)
        return estimate

# Example: Object tracking with particle filter
pf = ParticleFilter(num_particles=1000, state_dim=2)
true_position = np.array([5, 7])  # True position to track

# Simulate tracking loop
for t in range(100):
    # Simulate measurement with noise
    measurement = true_position + np.random.normal(0, 0.5, 2)
    
    # Predict based on motion model
    control_input = np.array([0.1, 0.05])  # Expected motion
    pf.predict(control_input)
    
    # Update with measurement
    pf.update(measurement)
    
    # Resample if needed (low effective sample size)
    effective_sample_size = 1.0 / np.sum(pf.weights**2)
    if effective_sample_size < pf.num_particles / 2:
        pf.resample()
    
    # Estimate current position
    estimated_position = pf.estimate()
    print(f"Step {t}: True={true_position}, Est={estimated_position}")
```

## Perception for VLA Integration

### Attention Mechanisms in Perception

Attention mechanisms help focus computational resources on relevant visual information:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualAttention(nn.Module):
    def __init__(self, feature_dim):
        super(VisualAttention, self).__init__()
        self.feature_dim = feature_dim
        self.attention_layer = nn.Linear(feature_dim, 1)
        
    def forward(self, features):
        """
        Compute attention weights for spatial features
        Args:
            features: Tensor of shape (batch, channels, height, width)
        Returns:
            attended_features: Features weighted by attention
            attention_weights: Attention weights
        """
        batch_size, channels, height, width = features.shape
        
        # Reshape to (batch, spatial_locations, channels)
        features_flat = features.view(batch_size, channels, -1).permute(0, 2, 1)
        
        # Compute attention scores
        attention_scores = self.attention_layer(features_flat)  # (batch, spatial_locations, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention
        attended_features = features_flat * attention_weights
        attended_features = attended_features.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        return attended_features, attention_weights.squeeze(-1)

class CrossModalAttention(nn.Module):
    def __init__(self, visual_dim, language_dim):
        super(CrossModalAttention, self).__init__()
        self.visual_transform = nn.Linear(visual_dim, language_dim)
        self.language_transform = nn.Linear(language_dim, language_dim)
        
    def forward(self, visual_features, language_features):
        """
        Compute attention between visual and language features
        Args:
            visual_features: Tensor of shape (batch, num_regions, visual_dim)
            language_features: Tensor of shape (batch, num_words, language_dim)
        Returns:
            aligned_features: Features aligned between modalities
        """
        # Transform features to same dimension
        visual_proj = self.visual_transform(visual_features)
        language_proj = self.language_transform(language_features)
        
        # Compute attention weights
        attention_weights = torch.bmm(visual_proj, language_proj.transpose(1, 2))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Apply attention: each visual region attended to all language tokens
        aligned_features = torch.bmm(attention_weights, language_proj)
        
        return aligned_features
```

## Real-Time Considerations

### Optimized Perception Pipeline

For humanoid robots, perception must run in real-time:

```python
import time
import threading
import queue
from collections import deque

class RealTimePerceptionPipeline:
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=10)  # Prevent memory buildup
        self.result_queue = queue.Queue(maxsize=10)
        
        # Processing threads
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        
        # Frame rate control
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps
        
        # Processing history for performance analysis
        self.processing_times = deque(maxlen=30)  # Last 30 frames
        
        self.running = False
    
    def start(self):
        self.running = True
        self.processing_thread.start()
    
    def stop(self):
        self.running = False
        self.processing_thread.join()
    
    def submit_frame(self, frame):
        """Submit a frame for processing"""
        try:
            self.input_queue.put_nowait(frame)
        except queue.Full:
            # Drop frame if queue is full
            print("Warning: Dropped frame due to full queue")
    
    def get_result(self):
        """Get the latest processed result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            start_time = time.time()
            
            try:
                # Get next frame to process
                frame = self.input_queue.get(timeout=0.1)
                
                # Process frame (this would call actual perception algorithms)
                result = self._process_frame(frame)
                
                # Add result to queue
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Discard old result if queue full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                
                # Record processing time
                self.processing_times.append(time.time() - start_time)
                
            except queue.Empty:
                continue  # No frame to process, continue loop
            
            # Throttle processing rate
            elapsed = time.time() - start_time
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_frame(self, frame):
        """Process a single frame with perception algorithms"""
        # This is where actual perception processing would happen
        # For example: object detection, segmentation, etc.
        result = {
            'frame_id': time.time(),
            'objects': [],
            'processed_at': time.time()
        }
        return result
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.processing_times:
            return 0, 0
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return avg_time, avg_fps

# Usage example
pipeline = RealTimePerceptionPipeline()
pipeline.start()

# Simulate frame submission
for i in range(100):
    fake_frame = np.random.random((480, 640, 3))  # Simulated frame
    pipeline.submit_frame(fake_frame)
    
    result = pipeline.get_result()
    if result:
        print(f"Processed frame at {result['processed_at']}")
    
    time.sleep(0.033)  # Simulate time between frames

pipeline.stop()
```

## Quality Assessment and Error Handling

### Uncertainty Quantification

Assessing the reliability of perception results is crucial for VLA systems:

```python
import numpy as np
from scipy.stats import entropy

class PerceptionUncertainty:
    def __init__(self):
        pass
    
    def classification_uncertainty(self, prediction_probs):
        """
        Calculate uncertainty of classification predictions
        Args:
            prediction_probs: Array of prediction probabilities
        Returns:
            uncertainty: Measure of prediction uncertainty
        """
        # Entropy-based uncertainty (higher entropy = higher uncertainty)
        return entropy(prediction_probs)
    
    def detection_confidence_interval(self, bbox, confidence):
        """
        Estimate detection confidence intervals
        Args:
            bbox: Bounding box [x, y, w, h]
            confidence: Detection confidence score
        Returns:
            confidence_interval: Estimate of position uncertainty
        """
        # Simple approach: uncertainty decreases with confidence
        base_uncertainty = 0.1  # meters
        scale_factor = 1.0 - confidence
        return base_uncertainty * scale_factor
    
    def pose_estimation_uncertainty(self, rotation_matrix, translation_vector):
        """
        Estimate uncertainty in pose estimation
        Args:
            rotation_matrix: 3x3 rotation matrix
            translation_vector: 3x1 translation vector
        Returns:
            position_uncertainty: Estimate of position uncertainty
            orientation_uncertainty: Estimate of orientation uncertainty
        """
        # For this example, we'll return fixed estimates
        # In practice, this would come from the pose estimation algorithm
        position_uncertainty = np.linalg.norm(translation_vector) * 0.05  # 5% of distance
        orientation_uncertainty = 0.05  # 0.05 radians
        
        return position_uncertainty, orientation_uncertainty
    
    def fusion_confidence(self, sensor_readings, sensor_variances):
        """
        Calculate confidence in fused sensor readings
        Args:
            sensor_readings: Array of readings from different sensors
            sensor_variances: Array of variances for each sensor
        Returns:
            fused_estimate: Weighted average of sensor readings
            fused_variance: Variance of fused estimate
        """
        # Weighted average based on sensor precision (inverse of variance)
        weights = 1.0 / (sensor_variances + 1e-8)  # Add small value to avoid division by zero
        weights = weights / np.sum(weights)  # Normalize weights
        
        fused_estimate = np.sum(sensor_readings * weights)
        fused_variance = 1.0 / np.sum(1.0 / sensor_variances)
        
        return fused_estimate, fused_variance

# Example usage
uncertainty_estimator = PerceptionUncertainty()

# Classification uncertainty example
pred_probs = np.array([0.6, 0.3, 0.1])  # Classification with 60% confidence in class 0
class_uncertainty = uncertainty_estimator.classification_uncertainty(pred_probs)
print(f"Classification uncertainty: {class_uncertainty:.3f}")

# Detection confidence example
bbox = [100, 150, 50, 100]
confidence = 0.8
pos_uncertainty = uncertainty_estimator.detection_confidence_interval(bbox, confidence)
print(f"Position uncertainty for detection: {pos_uncertainty:.3f}")

# Sensor fusion example
readings = np.array([1.0, 1.1, 0.95])  # Three different sensors
variances = np.array([0.01, 0.02, 0.015])  # Different variances for different sensors
fused_val, fused_var = uncertainty_estimator.fusion_confidence(readings, variances)
print(f"Fused estimate: {fused_val:.3f}, Variance: {fused_var:.3f}")
```

## Exercises

1. **Real-time Object Detection Exercise**: Implement and optimize an object detection system that can run in real-time on humanoid robot hardware, with performance monitoring.

2. **Multi-Sensor Fusion Exercise**: Design and implement a sensor fusion system that combines camera, IMU, and joint encoder data to improve robot self-localization.

3. **Human Pose Tracking Exercise**: Create a system that can track human poses in a dynamic environment and predict their future movements.

4. **Perception Uncertainty Exercise**: Implement uncertainty quantification for your perception modules and design a system that requests human confirmation when uncertainty is high.

5. **Cross-Modal Attention Exercise**: Build a system that demonstrates visual attention guided by language commands, highlighting relevant objects in the scene.

## Summary

Perception modules are critical for Vision-Language-Action systems in humanoid robotics, providing the foundation for understanding the environment and enabling appropriate action selection. These modules must be robust, efficient, and capable of handling the complex, dynamic environments where humanoid robots operate. With proper sensor fusion, uncertainty quantification, and real-time optimization, perception systems can enable humanoid robots to operate effectively in human environments and respond appropriately to natural language commands.