# Contributing to Physical AI & Humanoid Robotics Textbook

We welcome contributions to the "Physical AI & Humanoid Robotics" textbook project! This document outlines the process for contributing content, code, and exercises to the textbook.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Content Contribution Guidelines](#content-contribution-guidelines)
4. [Technical Requirements](#technical-requirements)
5. [Pull Request Process](#pull-request-process)
6. [Style Guide](#style-guide)
7. [Questions?](#questions)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [project-email@example.com].

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork to your local machine
3. Create a new branch for your contribution: `git checkout -b feature/amazing-content`
4. Make your changes
5. Push your branch to GitHub: `git push origin feature/amazing-content`
6. Open a Pull Request to the main repository

### Prerequisites

Before contributing, ensure you have:

- Git installed
- A development environment with ROS 2 (Humble/Iron), Gazebo, and Unity
- Familiarity with humanoid robotics concepts
- Understanding of Physical AI principles

## Content Contribution Guidelines

### What We're Looking For

- Clear, well-structured textbook content
- Accurate and tested code examples
- Comprehensive exercises with solutions
- High-quality simulation models and environments
- Pedagogically sound explanations of complex concepts

### Content Structure

Each chapter should follow this structure:

1. Theory: Foundational concepts and principles
2. Examples: Practical demonstrations and use cases
3. Exercises: Hands-on activities for students
4. Assessment: Self-evaluation components

### What We're NOT Looking For

- Implementation details that should remain in code comments
- Content outside the scope of Physical AI and Humanoid Robotics
- Proprietary data or code that can't be distributed under the project's license

## Technical Requirements

### Code Examples

- All code examples must be tested in the specified ROS 2 environment
- Follow ROS 2 best practices and naming conventions
- Include appropriate error handling and safety checks
- Be documented with meaningful comments

### Simulation Environments

- Gazebo worlds and models must be compatible with ROS 2
- Unity scenes should be exported with appropriate settings
- Isaac Sim environments must include proper configuration files

## Pull Request Process

1. Ensure your PR addresses a specific issue or need in the textbook
2. Update the README.md with details of changes if applicable
3. Increase the version number in the relevant files if the changes affect the textbook's API
4. Ensure all code examples have been tested in the target environments
5. You may merge the Pull Request once you have the sign-off of another developer, or if you do not have permission to do that, you may request the reviewer to merge it for you

## Style Guide

### Writing Style

- Write for university students, AI engineers transitioning to robotics, and robotics beginners with programming experience
- Use clear, concise language
- Provide concrete examples before abstract concepts
- Link to external resources when appropriate

### Code Style

- Follow ROS 2 C++/Python style guidelines
- Use meaningful variable and function names
- Include docstrings for all functions and classes
- Write unit tests for all new code

## Questions?

If you have any questions about contributing to the textbook, feel free to open an issue in the GitHub repository or contact the maintainers.

Thank you for contributing to the Physical AI & Humanoid Robotics textbook!