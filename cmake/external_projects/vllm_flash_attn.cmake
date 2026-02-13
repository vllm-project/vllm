# vLLM Flash-Attention External Project
include(ExternalProject)

# GPU Architecture Handling
if(VLLM_GPU_LANG STREQUAL "CUDA")
  foreach(_ARCH ${CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    list(APPEND VLLM_GPU_ARCHES "${_ARCH}-real")
  endforeach()
endif()

# Support local development of vllm-flash-attn
if (DEFINED ENV{VLLM_FLASH_ATTN_SRC_DIR})
    set(VLLM_FLASH_ATTN_SRC_DIR $ENV{VLLM_FLASH_ATTN_SRC_DIR})
endif()

# Setup the arguments based on source availability
if(VLLM_FLASH_ATTN_SRC_DIR)
    message(STATUS "Building vllm-flash-attn from local source: ${VLLM_FLASH_ATTN_SRC_DIR}")
    set(VLLM_FA_SOURCE_ARGS 
        SOURCE_DIR ${VLLM_FLASH_ATTN_SRC_DIR}
        UPDATE_COMMAND "" 
        PATCH_COMMAND ""
    )
    set(FLASH_ATTN_PYTHON_SRC
        ${VLLM_FLASH_ATTN_SRC_DIR}/vllm_flash_attn)
else()
    set(VLLM_FA_SOURCE_ARGS 
        GIT_REPOSITORY https://github.com/vllm-project/vllm-flash-attn.git
        GIT_TAG 7d346be62004163f0b59f965761b122cc40bd0a3
        GIT_PROGRESS TRUE
        SOURCE_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn-src
    )
    set(FLASH_ATTN_PYTHON_SRC
        ${CMAKE_BINARY_DIR}/vllm-flash-attn-src/vllm_flash_attn)
endif()

ExternalProject_Add(vllm_flash_attn_external
    ${VLLM_FA_SOURCE_ARGS}
    PREFIX       ${CMAKE_BINARY_DIR}/vllm-flash-attn
    BINARY_DIR   ${CMAKE_BINARY_DIR}/vllm-flash-attn-build
    INSTALL_DIR  ${CMAKE_BINARY_DIR}/vllm-flash-attn-install
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}
        -DPython_EXECUTABLE=${Python_EXECUTABLE}
        -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
        -DTORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
        -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}

    # Inform Ninja what file proves the install step completed
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/vllm_flash_attn/_vllm_fa2_C*.so
        <INSTALL_DIR>/vllm_flash_attn/_vllm_fa3_C*.so

    STEP_TARGETS build install
)

# Install the directory into the final wheel layout
install(
    DIRECTORY
        ${CMAKE_BINARY_DIR}/vllm-flash-attn-install/vllm_flash_attn/
    DESTINATION vllm/vllm_flash_attn
    COMPONENT _vllm_fa2_C
)

install(
    DIRECTORY
        ${CMAKE_BINARY_DIR}/vllm-flash-attn-install/vllm_flash_attn/
    DESTINATION vllm/vllm_flash_attn
    COMPONENT _vllm_fa3_C
)
