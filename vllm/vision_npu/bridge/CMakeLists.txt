# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

cmake_minimum_required(VERSION 3.18)
project(_vision_flexmlrt)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# FlexMLRT paths (must be configured via command line or environment variables)
set(FLEXMLRT_INCLUDE_DIR "" CACHE PATH "FlexMLRT include directory")
set(FLEXMLRT_LIB_DIR "" CACHE PATH "FlexMLRT library directory")

# Check if FlexMLRT paths are configured
if(NOT FLEXMLRT_INCLUDE_DIR OR NOT FLEXMLRT_LIB_DIR)
    message(FATAL_ERROR
        "FlexMLRT paths not configured. Please specify:\n"
        "  -DFLEXMLRT_INCLUDE_DIR=/path/to/flexmlRT/include\n"
        "  -DFLEXMLRT_LIB_DIR=/path/to/flexmlRT/build/lib\n"
        "Or set environment variables:\n"
        "  export FLEXMLRT_INCLUDE_DIR=/path/to/flexmlRT/include\n"
        "  export FLEXMLRT_LIB_DIR=/path/to/flexmlRT/build/lib")
endif()

# Verify paths exist
if(NOT EXISTS "${FLEXMLRT_INCLUDE_DIR}")
    message(FATAL_ERROR "FlexMLRT include directory does not exist: ${FLEXMLRT_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${FLEXMLRT_LIB_DIR}")
    message(FATAL_ERROR "FlexMLRT library directory does not exist: ${FLEXMLRT_LIB_DIR}")
endif()

message(STATUS "FlexMLRT include: ${FLEXMLRT_INCLUDE_DIR}")
message(STATUS "FlexMLRT library: ${FLEXMLRT_LIB_DIR}")

# Create pybind11 module for FlexMLRT NPU vision inference
pybind11_add_module(_vision_flexmlrt_npu vision_flexmlrt.cpp)
target_include_directories(_vision_flexmlrt_npu PRIVATE
    ${FLEXMLRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(_vision_flexmlrt_npu PRIVATE ${FLEXMLRT_LIB_DIR})
target_link_libraries(_vision_flexmlrt_npu PRIVATE flexmlrt)
target_compile_features(_vision_flexmlrt_npu PRIVATE cxx_std_17)

# Install to parent directory (vllm/vision_npu/)
install(TARGETS _vision_flexmlrt_npu LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/..)
