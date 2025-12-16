import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar structure for the textbook
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'introduction/physical-ai-principles/content',
        'introduction/embodied-intelligence/content',
        'introduction/exercises/exercises',
      ],
      link: {
        type: 'doc',
        id: 'introduction/physical-ai-principles/content',
      },
    },
    {
      type: 'category',
      label: 'ROS 2 Middleware',
      items: [
        'ros-middleware/fundamentals/content',
        'ros-middleware/publisher-subscriber-patterns/content',
        'ros-middleware/setup-ros2',
        'ros-middleware/exercises/exercises',
      ],
      link: {
        type: 'doc',
        id: 'ros-middleware/fundamentals/content',
      },
    },
    {
      type: 'category',
      label: 'Simulation Environments',
      items: [
        'simulation/gazebo-basics/content',
        'simulation/isaac-sim/content',
        'simulation/unity-integration/content',
        'simulation/setup-gazebo',
        'simulation/setup-isaac-sim',
        'simulation/setup-unity',
        'simulation/exercises/exercises',
      ],
      link: {
        type: 'doc',
        id: 'simulation/gazebo-basics/content',
      },
    },
    {
      type: 'category',
      label: 'Vision-Language-Action Systems',
      items: [
        'vla-systems/vla-theory/content',
        'vla-systems/perception-modules/content',
        'vla-systems/exercises/exercises',
      ],
      link: {
        type: 'doc',
        id: 'vla-systems/vla-theory/content',
      },
    },
    {
      type: 'category',
      label: 'LLM-Driven Planning',
      items: [
        'llm-planning/planning-algorithms/content',
        'llm-planning/ai-integration/content',
        'llm-planning/exercises/exercises',
      ],
      link: {
        type: 'doc',
        id: 'llm-planning/planning-algorithms/content',
      },
    },
    {
      type: 'category',
      label: 'Humanoid Locomotion',
      items: [
        'humanoid-locomotion/walking-algorithms/content',
        'humanoid-locomotion/balance-control/content',
        'humanoid-locomotion/exercises/exercises',
      ],
      link: {
        type: 'doc',
        id: 'humanoid-locomotion/walking-algorithms/content',
      },
    },
    {
      type: 'category',
      label: 'Cognitive Robotics',
      items: [
        'cognitive-robotics/cognitive-architectures/content',
        'cognitive-robotics/conversational-systems/content',
        'cognitive-robotics/exercises/exercises',
      ],
      link: {
        type: 'doc',
        id: 'cognitive-robotics/cognitive-architectures/content',
      },
    },
    {
      type: 'category',
      label: 'System Integration',
      items: [
        'integration/system-composition/content',
        'integration/sim-to-real/content',
        'integration/exercises/testing-validation',
        'integration/exercises/exercises',
      ],
      link: {
        type: 'doc',
        id: 'integration/system-composition/content',
      },
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone/autonomous-humanoid-project/content',
      ],
      link: {
        type: 'doc',
        id: 'capstone/autonomous-humanoid-project/content',
      },
    },
  ],
};

export default sidebars;
