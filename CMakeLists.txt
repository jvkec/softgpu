cmake_minimum_required(VERSION 3.22)
project(softgpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# SG_SANITIZE=thread|address|undefined|off
set(SG_SANITIZE "off" CACHE STRING "Sanitizer to build with")
if(NOT SG_SANITIZE STREQUAL "off")
  add_compile_options(-fsanitize=${SG_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${SG_SANITIZE})
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)

find_package(Threads REQUIRED)

# ---- layering ---------------------------------------------------------------
# runtime -> driver -> device. Each library only links the one below it and
# only exposes its own headers, so a violation is a build error, not a review
# comment.

add_library(sg_device STATIC src/device/device.cpp)
target_include_directories(sg_device
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(sg_device PUBLIC Threads::Threads)

add_library(sg_driver STATIC src/driver/driver.cpp src/driver/vram_alloc.cpp)
target_include_directories(sg_driver
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(sg_driver PRIVATE sg_device)

add_library(sg_runtime STATIC src/runtime/runtime.cpp)
target_include_directories(sg_runtime
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(sg_runtime PRIVATE sg_driver)

# ---- executables ------------------------------------------------------------
add_executable(sgbench bench/main.cpp bench/benches.cpp)
target_link_libraries(sgbench PRIVATE sg_runtime Threads::Threads)

add_executable(sgtest tests/test_runtime.cpp)
target_link_libraries(sgtest PRIVATE sg_runtime Threads::Threads)

enable_testing()
add_test(NAME runtime COMMAND sgtest)
