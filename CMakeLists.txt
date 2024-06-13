cmake_minimum_required(VERSION 3.14)
project(nnetwork)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_library(${PROJECT_NAME} SHARED
  include/nnetwork/nnetwork.hpp
  include/nnetwork/visibility_control.hpp
  src/nnetwork.cpp
)

ament_auto_add_library(${PROJECT_NAME}_node SHARED
  include/nnetwork/nnetwork_node.hpp
  src/nnetwork_node.cpp
)

rclcpp_components_register_node(${PROJECT_NAME}_node
  PLUGIN "nnetwork::NnetworkNode"
  EXECUTABLE ${PROJECT_NAME}_node_exe
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_ros_isolated_gtest(test_nnetwork
    test/test_nnetwork.cpp
  )
  ament_target_dependencies(test_nnetwork ${TEST_SOURCES})
  target_link_libraries(test_nnetwork ${PROJECT_NAME})
endif()

ament_auto_package(INSTALL_TO_SHARE
  launch
  config
)
