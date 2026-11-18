cmake_minimum_required(VERSION 3.22)
project(drone_nav_deploy LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Milestone 3: find_package(rclcpp), px4_msgs, onnxruntime; add the policy node executable here.

install(DIRECTORY launch config models DESTINATION share/${PROJECT_NAME})

ament_package()
