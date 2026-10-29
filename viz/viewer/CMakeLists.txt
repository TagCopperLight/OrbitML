cmake_minimum_required(VERSION 3.10)
project(orbit_viewer)

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(orbit_viewer ${SOURCES})

target_link_libraries(orbit_viewer PRIVATE
    aurora_ui
    aurora_debug
    aurora_engine
    Freetype::Freetype
    PNG::PNG
    ZLIB::ZLIB
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(orbit_viewer PRIVATE -Wall -Wextra -Wpedantic)
endif()
