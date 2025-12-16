# Data Model: Physical AI & Humanoid Robotics Book

**Feature**: 001-humanoid-robotics-book
**Spec**: [Link to spec.md](spec.md)
**Plan**: [Link to plan.md](plan.md)

## Core Entities

### Physical AI Principles
- **Concept**: Fundamental ideas that bridge digital AI and physical robot embodiment
- **Attributes**: 
  - Definition
  - Historical context
  - Key researchers/practitioners
  - Related theories
- **Relationships**: Connected to Humanoid Robot Models and Simulation Environments

### Humanoid Robot Model
- **Concept**: The simulated robot that integrates AI systems with physical body mechanics
- **Attributes**:
  - Kinematic structure
  - Sensor configuration
  - Actuator specifications
  - Control interfaces
- **Relationships**: Used in Simulation Environments and AI Control Systems

### Simulation Environment
- **Concept**: Virtual world where the humanoid robot operates (Gazebo/Isaac Sim/Unity)
- **Attributes**:
  - Physics engine
  - Scene configuration
  - Object properties
  - Environmental conditions
- **Relationships**: Contains Humanoid Robot Models and interfaces with AI Control Systems

### AI Control System
- **Concept**: Software components that process input (voice, sensors) and generate robot actions
- **Attributes**:
  - Input processing modules
  - Decision-making algorithms
  - Output generation
  - Learning capabilities
- **Relationships**: Interfaces with Humanoid Robot Models and Simulation Environments

### Educational Module
- **Concept**: Organized content sections that build upon each other from basic to advanced topics
- **Attributes**:
  - Learning objectives
  - Content type (theory, practice, assessment)
  - Prerequisites
  - Estimated duration
- **Relationships**: Contains exercises and assessments tied to specific concepts

## Relationships

```
Physical AI Principles ←→ Educational Modules
(Principles inform module content)

Humanoid Robot Model ↔ Simulation Environment
(Robot operates in simulation)

AI Control System ↔ Humanoid Robot Model
(Control system drives robot)

Simulation Environment ↔ AI Control System
(Simulation provides feedback to control)

Educational Module → Exercises
(Modules contain hands-on exercises)
```

## State Transitions

### For Humanoid Robot Model:
- **Idle**: Robot initialized but awaiting commands
- **Calibrating**: Adjusting sensors and actuators
- **Active**: Responding to control inputs
- **Learning**: Updating behavior based on experience
- **Safe Mode**: Emergency state when anomalies detected

### For Educational Module:
- **Draft**: Content being created
- **Review**: Undergoing technical validation
- **Published**: Available for students
- **Deprecated**: Superseded by newer content

## Validation Rules

1. Each Humanoid Robot Model must have valid kinematic definitions
2. Simulation Environments must support the configured robot model
3. AI Control Systems must provide safety checks before actuation
4. Educational Modules must align with specified learning objectives
5. All code examples in modules must be tested and verified