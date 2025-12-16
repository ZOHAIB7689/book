# Physical AI & Humanoid Robotics Textbook

This repository contains the content, code examples, and exercises for the textbook "Physical AI & Humanoid Robotics: From Embodied Intelligence to Autonomous Humanoids", published as a Docusaurus website.

## Table of Contents

The textbook is organized into 9 chapters, accessible through the website navigation:

1. [Introduction](https://zohaib7689.github.io/book/docs/01-introduction/physical-ai-principles/content)
2. [ROS 2 Middleware](https://zohaib7689.github.io/book/docs/02-ros-middleware/fundamentals/content)
3. [Simulation Environments](https://zohaib7689.github.io/book/docs/03-simulation/gazebo-basics/content)
4. [Vision-Language-Action Systems](https://zohaib7689.github.io/book/docs/04-vla-systems/vla-theory/content)
5. [LLM-Driven Planning](https://zohaib7689.github.io/book/docs/05-llm-planning/planning-algorithms/content)
6. [Humanoid Locomotion](https://zohaib7689.github.io/book/docs/06-humanoid-locomotion/walking-algorithms/content)
7. [Cognitive Robotics](https://zohaib7689.github.io/book/docs/07-cognitive-robotics/cognitive-architectures/content)
8. [System Integration](https://zohaib7689.github.io/book/docs/08-integration/system-composition/content)
9. [Capstone Project](https://zohaib7689.github.io/book/docs/09-capstone/autonomous-humanoid-project/content)

## Overview

This textbook teaches students how to design, simulate, and control humanoid robots by integrating AI systems with physical bodies. The content is organized to guide readers from basic concepts to complex implementation, bridging the gap between digital AI systems and physical robotic embodiment.

## Technologies

- ROS 2 (Humble / Iron)
- Gazebo
- Unity (visualization & interaction)
- NVIDIA Isaac Sim and Isaac ROS
- Vision-Language-Action (VLA) systems
- LLM-driven planning for robotics

## Development

This Docusaurus website was generated from content in the `book/` directory using migration scripts. To run the website locally:

1. Install Node.js (version 20.0 or above)
2. Navigate to the `website` directory: `cd website`
3. Install dependencies: `npm install`
4. Start the development server: `npm run start`

## Deployment

The site is configured for deployment to GitHub Pages at https://zohaib7689.github.io/book/ using the following command:
```
cd website
GIT_USER=ZOHAIB7689 CURRENT_BRANCH=main USE_SSH=true npm run deploy
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. 
