#!/bin/bash
# ============================================================
# Surgical Instrument Digital Twin — Full Pipeline Launcher
# Run this script to start all nodes in the correct order
# ============================================================

echo "Starting Surgical Instrument Digital Twin Pipeline..."
echo "============================================================"

# Source ROS 2
source /opt/ros/jazzy/setup.bash
source ~/surgical_twin_ws/install/setup.bash

# Export Gazebo model path
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$HOME/surgical_twin_ws/models

echo ""
echo "Open 6 separate terminals and run:"
echo ""
echo "Terminal 1 (Gazebo):"
echo "  gz sim -v 4 empty.sdf"
echo ""
echo "Terminal 2 (Video Publisher):"
echo "  ros2 run surgical_perception video_publisher"
echo ""
echo "Terminal 3 (Perception Node):"
echo "  ros2 run surgical_perception perception_node"
echo ""
echo "Terminal 4 (Stereo Depth Node):"
echo "  ros2 run surgical_perception stereo_depth_node"
echo ""
echo "Terminal 5 (Pose Estimator):"
echo "  ros2 run surgical_perception pose_estimator"
echo ""
echo "Terminal 6 (Twin Sync Node):"
echo "  ros2 run surgical_perception twin_sync_node"
echo ""
echo "Optional Terminal 7 (Node Graph):"
echo "  rqt_graph"
echo "============================================================"
