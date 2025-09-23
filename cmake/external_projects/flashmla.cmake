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
        GIT_REPOSITORY https://github.com/vllm-model-0920/FlashMLA
        GIT_TAG a25b977fae6925c45c3d0404c98c6ce6f4563dac
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
cuda_archs_loose_intersection(FLASH_MLA_ARCHS "9.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.3 AND FLASH_MLA_ARCHS)
    #######################################################################
    # FlashMLA Dense -- _flashmla_C
    #######################################################################

    set(FlashMLA_SOURCES
        ${flashmla_SOURCE_DIR}/csrc/flash_api.cpp
        ${flashmla_SOURCE_DIR}/csrc/kernels/get_mla_metadata.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/mla_combine.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/splitkv_mla.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels_fp8/flash_fwd_mla_fp8_sm90.cu)

    set(FlashMLA_INCLUDES
        ${flashmla_SOURCE_DIR}/csrc/cutlass/include
        ${flashmla_SOURCE_DIR}/csrc)

    set_gencode_flags_for_srcs(
        SRCS "${FlashMLA_SOURCES}"
        CUDA_ARCHS "${FLASH_MLA_ARCHS}")

    define_gpu_extension_target(
        _flashmla_C
        DESTINATION vllm
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES ${FlashMLA_SOURCES}
        COMPILE_FLAGS ${VLLM_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        INCLUDE_DIRECTORIES ${FlashMLA_INCLUDES}
        USE_SABI 3
        WITH_SOABI)
    
    #######################################################################
    # FlashMLA Sparse -- _flashmla_sparse_C
    #######################################################################

    # We use seperate libraries to avoid crosss contaminating includes,
    # namely kernels/utils.h

    set(DECODE_FOLDER  ${flashmla_SOURCE_DIR}/csrc/sparse/decode)
    set(PREFILL_FOLDER ${flashmla_SOURCE_DIR}/csrc/sparse/prefill)

    # ---- Decode object library ----
    set(SPARSE_FLASHMLA_DECODE_SOURCES
        ${DECODE_FOLDER}/flash_api.cpp
        ${DECODE_FOLDER}/kernels/get_mla_metadata.cu
        ${DECODE_FOLDER}/kernels/mla_combine.cu
        ${DECODE_FOLDER}/kernels/fp8_sparse/splitkv_mla.cu
    )

    add_library(_flashmla_sparse_decode OBJECT ${SPARSE_FLASHMLA_DECODE_SOURCES})
    set_property(TARGET _flashmla_sparse_decode PROPERTY POSITION_INDEPENDENT_CODE ON)

    set_gencode_flags_for_srcs(
        SRCS       "${SPARSE_FLASHMLA_DECODE_SOURCES}"
        CUDA_ARCHS "${FLASH_MLA_ARCHS}"
    )

    # Include paths for decode ONLY (do not leak DECODE_FOLDER to others)
    target_include_directories(_flashmla_sparse_decode
        PRIVATE
            ${flashmla_SOURCE_DIR}/csrc/cutlass/include
            ${TORCH_INCLUDE_DIRS}
            ${Python_INCLUDE_DIRS}
            ${DECODE_FOLDER}
    )
    target_compile_options(_flashmla_sparse_decode  PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${VLLM_GPU_FLAGS}>)

    # ---- Prefill object library ----
    set(SPARSE_FLASHMLA_PREFILL_SOURCES
        ${PREFILL_FOLDER}/api.cpp
        ${PREFILL_FOLDER}/kernels/sm90/fwd/fwd.cu
    )

    add_library(_flashmla_sparse_prefill OBJECT ${SPARSE_FLASHMLA_PREFILL_SOURCES})
    set_property(TARGET _flashmla_sparse_prefill PROPERTY POSITION_INDEPENDENT_CODE ON)

    set_gencode_flags_for_srcs(
        SRCS       "${SPARSE_FLASHMLA_PREFILL_SOURCES}"
        CUDA_ARCHS "${FLASH_MLA_ARCHS}"
    )

    target_include_directories(_flashmla_sparse_prefill
        PRIVATE
            ${flashmla_SOURCE_DIR}/csrc/cutlass/include
            ${TORCH_INCLUDE_DIRS}
            ${Python_INCLUDE_DIRS}
            ${PREFILL_FOLDER}
    )
    target_compile_options(_flashmla_sparse_prefill PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${VLLM_GPU_FLAGS}>)

    # ---- Final extension target with unified API ----
    define_gpu_extension_target(
        _flashmla_sparse_C
        DESTINATION vllm
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sparse/api.cpp
            $<TARGET_OBJECTS:_flashmla_sparse_decode>
            $<TARGET_OBJECTS:_flashmla_sparse_prefill>
        COMPILE_FLAGS ${VLLM_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        # Only the common/public includes here; do NOT add decode/prefill folders
        INCLUDE_DIRECTORIES
            ${flashmla_SOURCE_DIR}/csrc/
            ${CUTLASS_INCLUDE_DIR}
            ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
        USE_SABI 3
        WITH_SOABI
    )
else()
    # Create an empty target for setup.py when not targeting sm90a systems
    add_custom_target(_flashmla_C)
    add_custom_target(_flashmla_sparse_C)
endif()

