import sys
if sys.prefix == '/Users/Colegio/miniforge3/envs/ros_env':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/Users/Colegio/miniforge3/envs/ros_env/share/turtlebot3_gazebo/pra_ws/install/turtlebot3_slam_mapper'
