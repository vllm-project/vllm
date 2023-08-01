cmake_minimum_required(VERSION 3.15)
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")

project(PageAttCPU)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-mfma -mavx512f -mavx512bf16 -mavx512vl)

set(PYTHON_LIB_PATH "/usr/local/lib/python3.10/dist-packages/")
set(PYTHON_INCLUDE_PATH "/usr/include/python3.10")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYTHON_LIB_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/FindModules")

include_directories(${PYTHON_INCLUDE_PATH})
include_directories("/usr/local/lib/python3.10/dist-packages/torch/include/")
include_directories("/usr/local/lib/python3.10/dist-packages/torch/include/torch/csrc/api/include/")

option(USE_STATIC_MKL "Prefer to link with MKL statically (Unix only)" OFF)

add_subdirectory(csrc/cpu)