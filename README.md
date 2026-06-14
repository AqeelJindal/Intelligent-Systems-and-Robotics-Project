INTELLIGENT SYSTEMS & ROBOTICS — AUTONOMOUS COLOUR SEARCH ROBOT
----------------------------------------------------------------
ROS2 Python package implementing autonomous robot navigation with real-time
computer vision for colour-based target detection and approach.

This project implements a full autonomous robotics pipeline within the ROS2 framework.
A mobile robot navigates a pre-mapped environment, visiting a sequence of waypoints while
continuously analysing its camera feed for coloured objects. Upon detecting a blue target,
it cancels active navigation, performs a local sweep scan if needed, and switches to a
proportional visual-serving controller to approach and stop at the target. The implementation
integrates the Nav2 action stack, OpenCV HSV colour masking, and velocity control via ROS2
topics — all within a single, well-structured Python node.

FEATURES
--------
- Autonomous waypoint navigation using the Nav2 NavigateToPose action client
- Real-time HSV colour detection with morphological noise filtering via OpenCV
- Hue-wrap handling for accurate red detection across the 0/360 degree boundary
- Local left-right sweep scanning at each waypoint to maximise target detection
- Proportional visual-serving controller for smooth, clamped target approach
- Live annotated camera display with bounding boxes, centroids, and detection summary

TECHNOLOGIES
------------
- Python 3
- ROS2 (rclpy, Nav2, sensor_msgs, geometry_msgs)
- OpenCV (cv2) and CvBridge
- Nav2 NavigateToPose action and ActionClient
- Gazebo (TurtleBot3 simulation)
- ament_python (ROS2 colcon build system)

GETTING STARTED
---------------
This project runs inside a Singularity container provided by the COMP3631 module,
which bundles ROS2 Humble, Gazebo, Nav2, and all required dependencies on Ubuntu 22.04.

1. Enter the Singularity environment
   Open a terminal and run:
     ros
   You should see the [COMP3631] Singularity> prompt. All subsequent commands
   must be run inside this environment.

2. Clone the repository into your ROS2 workspace
     cd ~/ros2_ws/src
     git clone https://github.com/AqeelJindal/Intelligent-Systems-and-Robotics-Project ros2_project_sc23aj2

3. Build the package
     cd ~/ros2_ws
     colcon build --packages-select ros2_project_sc23aj2
     source ~/.bashrc

4. Launch the task world in Gazebo (separate terminal, inside Singularity)
     ros2 launch turtlebot3_gazebo turtlebot3_task_world_2026.launch.py

5. Launch the Nav2 navigation stack with the provided map (separate terminal)
     ros2 launch turtlebot3_navigation2 navigation2.launch.py \
       use_sim_time:=True \
       map:=$HOME/ros2_ws/src/ros2_project_sc23aj2/map/map.yaml

   Once Rviz opens, use the "2D Pose Estimate" button to set the robot's
   approximate starting position in the bottom-right compartment of the map.

6. Run the node (separate terminal)
     ros2 run ros2_project_sc23aj2 main

   The robot will begin navigating through its waypoints, scanning for coloured
   boxes and approaching the blue target when detected. An annotated OpenCV
   window will open showing the live camera feed.

Note: If you are not using the university Singularity environment, you will need
ROS2 Humble, Nav2, and the turtlebot3 simulation packages installed manually on
Ubuntu 22.04. Refer to the official ROS2 and Nav2 installation documentation.

LEARNING OUTCOMES
-----------------
This project demonstrates practical understanding of:
- ROS2 node architecture with concurrent publishers, subscribers, and action clients
- HSV colour space transforms, contour analysis, and morphological operations on live camera feeds
- Reactive autonomy combining deliberate waypoint planning with sensor-driven interrupts
- Proportional control systems with clamped velocity output for safe robot motion
- Clean separation of sensing, navigation, and control responsibilities in a robotics codebase
