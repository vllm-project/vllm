# DeepGEMM requires CUDA 12.8+ and only works on sm90 and sm100 architectures
# Check CUDA version requirement
if(NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    message(STATUS "Not installing DeepGEMM as CUDA Compiler version is "
                   "not >= 12.8 (current: ${CMAKE_CUDA_COMPILER_VERSION}). "
                   "DeepGEMM requires CUDA 12.8 or later.")
    # Create an empty target for setup.py when CUDA version is insufficient
    add_custom_target(_deepgemm_C)
    return()
endif()

# Only install DeepGEMM if we are building for compatible architectures (sm90, sm100)
cuda_archs_loose_intersection(DEEPGEMM_ARCHS "9.0;9.0a;10.0;10.0a;10.1;10.1a" "${CUDA_ARCHS}")
if(NOT DEEPGEMM_ARCHS)
    message(STATUS "Not installing DeepGEMM as no compatible archs found "
                   "in CUDA target architectures. DeepGEMM requires sm90 or sm100.")
    # Create an empty target for setup.py when not targeting compatible systems
    add_custom_target(_deepgemm_C)
    return()
endif()

message(STATUS "Installing DeepGEMM using tools/install_deepgemm.sh for archs: ${DEEPGEMM_ARCHS}")

# Create a timestamp file to track when DeepGEMM was last installed
set(DEEPGEMM_INSTALL_STAMP "${CMAKE_BINARY_DIR}/deepgemm_install.stamp")

# Create a custom target that runs the install script
add_custom_command(
    OUTPUT ${DEEPGEMM_INSTALL_STAMP}
    COMMAND ${CMAKE_COMMAND} -E echo "Installing DeepGEMM using tools/install_deepgemm.sh..."
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_deepgemm.sh
    COMMAND ${CMAKE_COMMAND} -E touch ${DEEPGEMM_INSTALL_STAMP}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Installing DeepGEMM Python package"
    VERBATIM
)

# Create the _deepgemm_C target that depends on the install step
add_custom_target(_deepgemm_C
    DEPENDS ${DEEPGEMM_INSTALL_STAMP}
    COMMENT "DeepGEMM installation target"
)