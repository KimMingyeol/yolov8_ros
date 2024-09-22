cmake_minimum_required(VERSION 3.8)
project(object_msgs)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Object.msg"
  "msg/ObjectInBox.msg"
  "msg/ObjectsInBoxes.msg"
  DEPENDENCIES std_msgs sensor_msgs
)

ament_package()
