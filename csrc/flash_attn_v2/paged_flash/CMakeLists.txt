cmake_minimum_required(VERSION 3.11)
project(DeepSpeedBlockedFlash CXX CUDA)

Set(CMAKE_CXX_STANDARD 17)
Set(CMAKE_CUDA_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

if (NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake"
                EXPECTED_HASH SHA256=396e16d0f5eabdc6a14afddbcfff62a54a7ee75c6da23f32f7a31bc85db23484
                TLS_VERIFY ON)
endif()

find_package(CUDA)

if (NOT WIN32)
    list(APPEND CMAKE_CXX_FLAGS "-fmax-errors=1 -Wfatal-errors")
    Set(LIB_NAME "blockedflash")
else()
    Set(LIB_NAME "libblockedflash")
endif()

add_library(${LIB_NAME} SHARED)

target_include_directories(${LIB_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass/include
)

list(APPEND CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})

list(APPEND NVCC_FLAGS "-O3")
list(APPEND NVCC_FLAGS "-U__CUDA_NO_HALF_OPERATORS__")
list(APPEND NVCC_FLAGS "-U__CUDA_NO_HALF_CONVERSIONS__")
list(APPEND NVCC_FLAGS "-U__CUDA_NO_HALF2_OPERATORS__")
list(APPEND NVCC_FLAGS "-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
list(APPEND NVCC_FLAGS "-expt-relaxed-constexpr")
list(APPEND NVCC_FLAGS "--use_fast_math")

set(VERBOSE_BUILD 0)

if (VERBOSE_BUILD)
    list(APPEND NVCC_FLAGS "--ptxas-options=-v")
endif()

file(GLOB SRC_FILES *.cu)

target_sources(${LIB_NAME} PRIVATE ${SRC_FILES})
target_link_libraries(${LIB_NAME} "${TORCH_LIBRARIES}")
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
set_target_properties(${LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_OUTPUT_DIR})


# Show timings?
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
