#!/bin/bash

# ROS Setup
# Update the following line with your actual home directory if needed
gnome-terminal -- bash -c 'conda activate serl && cd ~/franka && bash frankapy/bash_scripts/start_control_pc.sh -u katefgroup -i franka-control -d /home/katefgroup/franka-interface -a 192.168.1.1; exec bash'

# Camera Setup
gnome-terminal -- bash -c 'conda activate serl && cd ~/serl && roslaunch azure_kinect_ros_driver kinect_rgbd.launch fps:=30 color_resolution:=720P depth_unit:=32FC1; exec bash'

# Serl server setup
gnome-terminal -- bash -c 'conda activate serl && cd ~/serl && python serl_robot_infra/robot_servers/franka_server.py --gripper_type=Franka --robot_ip=192.168.1.1; exec bash'

# Data collection
gnome-terminal -- bash -c 'cd ~/serl/examples/genecollections && sudo bash -c "source /opt/ros/noetic/setup.bash && source /home/katefgroup/Documents/franka/frankapy/catkin_ws/devel/setup.bash --extend && source /home/katefgroup/catkin_ws/devel/setup.bash && /home/katefgroup/miniconda3/envs/serl/bin/python3 record_demo.py"; exec bash'

echo "All setup commands have been launched. Follow the instructions in each terminal window as needed."
