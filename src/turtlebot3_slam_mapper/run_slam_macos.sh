#!/bin/bash

# macOS-specific environment variables to help with RViz2 GLSL shader issues
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

# Additional OpenGL and graphics settings for macOS
export OGRE_RTT_MODE=Copy
export OGRE_GLSUPPORT_USE_GLX=0
export OGRE_GLSUPPORT_USE_COCOA=1
export OGRE_GLSUPPORT_USE_NSVIEW=1

# Disable hardware acceleration for better compatibility
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

# Additional ROS2 environment variables for better macOS compatibility
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Force software rendering for RViz2
export QT_QPA_PLATFORM=offscreen

echo "Starting SLAM with macOS optimizations..."
echo "Environment variables set:"
echo "  LIBGL_ALWAYS_SOFTWARE=1"
echo "  MESA_GL_VERSION_OVERRIDE=3.3"
echo "  MESA_GLSL_VERSION_OVERRIDE=330"
echo "  OGRE_RTT_MODE=Copy"
echo "  QT_QPA_PLATFORM=offscreen"

# Launch the SLAM system
ros2 launch turtlebot3_slam_mapper python_slam_maze.launch.py 