set(PYTHON_EXECUTABLE "/usr/bin/python3")
cmake_minimum_required(VERSION 3.0.2)
project(forest_slam_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf
  cv_bridge
  message_generation
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs geometry_msgs nav_msgs tf cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS src/mono_slam.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS src/stereo_slam.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
