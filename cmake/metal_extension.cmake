# Metal extension for Apple Silicon (MPS)
message(STATUS "Configuring Metal extension for Apple Silicon")

# Find Metal framework
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

if(NOT METAL_FRAMEWORK)
    message(FATAL_ERROR "Metal framework not found")
endif()

message(STATUS "Metal framework: ${METAL_FRAMEWORK}")

# Define Metal kernel source files
set(METAL_KERNEL_SOURCES
    csrc/metal/paged_attention_v1.metal
    csrc/metal/paged_attention_v2.metal
    csrc/metal/cache_ops.metal
)

# Define C++/Objective-C++ source files
set(METAL_CXX_SOURCES
    csrc/metal/metal_context.mm
    csrc/metal/metal_kernels.cpp
    csrc/metal/metal_bindings.cpp
)

# Compile Metal shaders to metallib
set(METAL_LIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/vllm_metal_kernels.metallib")

# Metal compiler flags
set(METAL_COMPILE_FLAGS
    -std=metal3.0
    -ffast-math
    -O3
)

# Custom command to compile Metal kernels
# Each shader must be compiled separately to its own .air file
add_custom_command(
    OUTPUT ${METAL_LIB_PATH}
    COMMAND xcrun -sdk macosx metal ${METAL_COMPILE_FLAGS} -c
            ${CMAKE_CURRENT_SOURCE_DIR}/csrc/metal/paged_attention_v1.metal
            -o ${CMAKE_CURRENT_BINARY_DIR}/paged_attention_v1.air
    COMMAND xcrun -sdk macosx metal ${METAL_COMPILE_FLAGS} -c
            ${CMAKE_CURRENT_SOURCE_DIR}/csrc/metal/paged_attention_v2.metal
            -o ${CMAKE_CURRENT_BINARY_DIR}/paged_attention_v2.air
    COMMAND xcrun -sdk macosx metal ${METAL_COMPILE_FLAGS} -c
            ${CMAKE_CURRENT_SOURCE_DIR}/csrc/metal/cache_ops.metal
            -o ${CMAKE_CURRENT_BINARY_DIR}/cache_ops.air
    COMMAND xcrun -sdk macosx metallib
            ${CMAKE_CURRENT_BINARY_DIR}/paged_attention_v1.air
            ${CMAKE_CURRENT_BINARY_DIR}/paged_attention_v2.air
            ${CMAKE_CURRENT_BINARY_DIR}/cache_ops.air
            -o ${METAL_LIB_PATH}
    DEPENDS ${METAL_KERNEL_SOURCES}
    COMMENT "Compiling Metal kernels"
    VERBATIM
)

# Custom target for Metal library
add_custom_target(metal_kernels ALL DEPENDS ${METAL_LIB_PATH})

# Create Metal Python extension
define_extension_target(
    _metal_C
    DESTINATION vllm
    LANGUAGE CXX
    SOURCES ${METAL_CXX_SOURCES}
    LIBRARIES ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
    WITH_SOABI
)

# Set Objective-C++ compile flags
set_source_files_properties(
    csrc/metal/metal_context.mm
    PROPERTIES
    COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
)

# Add Metal framework to include path
target_include_directories(_metal_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/metal
)

# Link Metal and Foundation frameworks
target_link_libraries(_metal_C PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Add dependency on Metal kernels compilation
add_dependencies(_metal_C metal_kernels)

# Install Metal library alongside Python extension
install(
    FILES ${METAL_LIB_PATH}
    DESTINATION vllm
    COMPONENT _metal_C
)

message(STATUS "Metal extension configuration complete")
