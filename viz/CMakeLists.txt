cmake_minimum_required(VERSION 3.10)
project(OrbitMLViewer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Aurora defaults to gprof instrumentation, which is not wanted for the viewer
set(AURORA_ENABLE_GPROF OFF CACHE BOOL "Build with gprof instrumentation (-pg)")

# Imported targets are directory scoped, so make them visible here as well
find_package(Vulkan REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(glm REQUIRED)
find_package(Freetype REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)

# Only the Aurora libraries the viewer links against get built
add_subdirectory(Aurora aurora EXCLUDE_FROM_ALL)

add_subdirectory(viewer)
