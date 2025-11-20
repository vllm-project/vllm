include(FetchContent)

# If FLASH_MLA_SRC_DIR is set, flash-mla is installed from that directory 
# instead of downloading.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{FLASH_MLA_SRC_DIR})
  set(FLASH_MLA_SRC_DIR $ENV{FLASH_MLA_SRC_DIR})
endif()

if(FLASH_MLA_SRC_DIR)
  FetchContent_Declare(
        flashmla 
        SOURCE_DIR ${FLASH_MLA_SRC_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
else()
  FetchContent_Declare(
        flashmla
        GIT_REPOSITORY https://github.com/vllm-project/FlashMLA
        GIT_TAG 46d64a8ebef03fa50b4ae74937276a5c940e3f95
        GIT_PROGRESS TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
endif()


FetchContent_MakeAvailable(flashmla)
message(STATUS "FlashMLA is available at ${flashmla_SOURCE_DIR}")

# The FlashMLA kernels only work on hopper and require CUDA 12.3 or later.
# Only build FlashMLA kernels if we are building for something compatible with 
# sm90a

set(SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.3)
    list(APPEND SUPPORT_ARCHS 9.0a)
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.8)
    list(APPEND SUPPORT_ARCHS 10.0a)
endif()


cuda_archs_loose_intersection(FLASH_MLA_ARCHS "${SUPPORT_ARCHS}" "${CUDA_ARCHS}")
if(FLASH_MLA_ARCHS)
    set(VLLM_FLASHMLA_GPU_FLAGS ${VLLM_GPU_FLAGS})
    list(APPEND VLLM_FLASHMLA_GPU_FLAGS "--expt-relaxed-constexpr" "--expt-extended-lambda" "--use_fast_math")

    set(FlashMLA_SOURCES
        ${flashmla_SOURCE_DIR}/csrc/torch_api.cpp
        ${flashmla_SOURCE_DIR}/csrc/pybind.cpp
        ${flashmla_SOURCE_DIR}/csrc/smxx/get_mla_metadata.cu
        ${flashmla_SOURCE_DIR}/csrc/smxx/mla_combine.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/splitkv_mla.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/splitkv_mla.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/fwd.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/decode/sparse_fp8/splitkv_mla.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd.cu
    )

    set(FlashMLA_Extension_SOURCES
        ${flashmla_SOURCE_DIR}/csrc/extension/torch_api.cpp
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/pybind.cpp
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_fp8_sm90.cu
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_metadata.cu
    )

    set(FlashMLA_INCLUDES
        ${flashmla_SOURCE_DIR}/csrc
        ${flashmla_SOURCE_DIR}/csrc/sm90
        ${flashmla_SOURCE_DIR}/csrc/cutlass/include
        ${flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
    )

    set(FlashMLA_Extension_INCLUDES
        ${flashmla_SOURCE_DIR}/csrc
        ${flashmla_SOURCE_DIR}/csrc/sm90
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/
        ${flashmla_SOURCE_DIR}/csrc/cutlass/include
        ${flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
    )

    set_gencode_flags_for_srcs(
        SRCS "${FlashMLA_SOURCES}"
        CUDA_ARCHS "${FLASH_MLA_ARCHS}")

    set_gencode_flags_for_srcs(
        SRCS "${FlashMLA_Extension_SOURCES}"
        CUDA_ARCHS "${FLASH_MLA_ARCHS}")

    define_extension_target(
        _flashmla_C
        DESTINATION vllm
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES ${FlashMLA_SOURCES}
        COMPILE_FLAGS ${VLLM_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        INCLUDE_DIRECTORIES ${FlashMLA_INCLUDES}
        USE_SABI 3
        WITH_SOABI)

    # Keep Stable ABI for the module, but *not* for CUDA/C++ files.
    # This prevents Py_LIMITED_API from affecting nvcc and C++ compiles.
    target_compile_options(_flashmla_C PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API>
        $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>)

    define_extension_target(
        _flashmla_extension_C
        DESTINATION vllm
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES ${FlashMLA_Extension_SOURCES}
        COMPILE_FLAGS ${VLLM_FLASHMLA_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        INCLUDE_DIRECTORIES ${FlashMLA_Extension_INCLUDES}
        USE_SABI 3
        WITH_SOABI)

    # Keep Stable ABI for the module, but *not* for CUDA/C++ files.
    # This prevents Py_LIMITED_API from affecting nvcc and C++ compiles.
    target_compile_options(_flashmla_extension_C PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API>
        $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>)
else()
    # Create empty targets for setup.py when not targeting sm90a systems
    add_custom_target(_flashmla_C)
    add_custom_target(_flashmla_extension_C)
endif()

