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
  # vllm-project/FlashMLA synced through deepseek-ai/FlashMLA "Add kernels for
  # DeepSeek v4.1 (#221)": V4.1 fp8 / fp4 paged KV formats, native head128
  # sparse decode and the fused norm + RoPE + sparse attention + RoPE + FP8-cast
  # kernels, registered as torch.ops._flashmla_C.
  FetchContent_Declare(
        flashmla
        GIT_REPOSITORY https://github.com/vllm-project/FlashMLA
        GIT_TAG c112cc1ed1c61bfdb7dbf25e53753bd272edb767
        GIT_PROGRESS TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
endif()


FetchContent_MakeAvailable(flashmla)
message(STATUS "FlashMLA is available at ${flashmla_SOURCE_DIR}")

# Vendor the FlashMLA Python interfaces into vLLM with the torch-ops shim.
set(FLASHMLA_VENDOR_DIR "${CMAKE_SOURCE_DIR}/vllm/third_party/flashmla")
file(MAKE_DIRECTORY "${FLASHMLA_VENDOR_DIR}")
foreach(FLASHMLA_PY_FILE flash_mla_interface.py fused_norm_rope_attn_rope_cast.py)
  file(READ "${flashmla_SOURCE_DIR}/flash_mla/${FLASHMLA_PY_FILE}"
       FLASHMLA_PY_CONTENT)
  string(REPLACE "flash_mla_cuda = torch.ops._flashmla_C"
                 "import vllm._flashmla_C\nflash_mla_cuda = torch.ops._flashmla_C"
                 FLASHMLA_PY_CONTENT
                 "${FLASHMLA_PY_CONTENT}")
  file(WRITE "${FLASHMLA_VENDOR_DIR}/${FLASHMLA_PY_FILE}"
       "${FLASHMLA_PY_CONTENT}")

  # Install the generated file to the wheel with the C extension.
  install(FILES "${FLASHMLA_VENDOR_DIR}/${FLASHMLA_PY_FILE}"
          DESTINATION vllm/third_party/flashmla/
          COMPONENT _flashmla_C)
endforeach()

# The FlashMLA kernels only work on hopper and require CUDA 12.3 or later.
# Only build FlashMLA kernels if we are building for something compatible with
# sm90a

set(SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3)
    list(APPEND SUPPORT_ARCHS "9.0a")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
    # CUDA 12.9 has introduced "Family-Specific Architecture Features"
    # this supports all compute_10x family
    list(APPEND SUPPORT_ARCHS "10.0f")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.4)
        list(APPEND SUPPORT_ARCHS "10.7f")
    endif()
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    list(APPEND SUPPORT_ARCHS "10.0a")
endif()


cuda_archs_loose_intersection(FLASH_MLA_ARCHS "${SUPPORT_ARCHS}" "${CUDA_ARCHS}")
if(FLASH_MLA_ARCHS)
    message(STATUS "FlashMLA CUDA architectures: ${FLASH_MLA_ARCHS}")
    set(VLLM_FLASHMLA_GPU_FLAGS ${VLLM_GPU_FLAGS})
    list(APPEND VLLM_FLASHMLA_GPU_FLAGS "--expt-relaxed-constexpr" "--expt-extended-lambda" "--use_fast_math")

    # Mirrors the source list in FlashMLA's setup.py.
    set(FlashMLA_SOURCES
        ${flashmla_SOURCE_DIR}/csrc/api/api.cpp
        ${flashmla_SOURCE_DIR}/csrc/api/sparse_prefill.cpp
        ${flashmla_SOURCE_DIR}/csrc/api/sparse_decode.cpp
        ${flashmla_SOURCE_DIR}/csrc/api/dense_decode.cpp
        ${flashmla_SOURCE_DIR}/csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp

        # Misc kernels for decoding
        ${flashmla_SOURCE_DIR}/csrc/kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/smxx/decode/combine/combine.cu

        # sm90 dense decode
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/dense/instantiations/fp16.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/dense/instantiations/bf16.cu

        # sm90 sparse decode
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v4_persistent_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v4_persistent_h128.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v32_persistent_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v32_persistent_h128.cu

        # sm90 sparse prefill
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/prefill/sparse/instantiations/phase1_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/prefill/sparse/instantiations/phase1_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu

        # sm100 dense prefill & backward
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu

        # sm100 sparse prefill
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_k512.cu

        # sm100 fused norm + RoPE + sparse attn + RoPE + FP8 cast (DeepSeek V4 /
        # V4.1); the API dispatches on enable_q_norm at runtime.
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h64_prefill_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h64_prefill_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h128_prefill_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h128_prefill_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h64_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h64_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h128_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h128_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41_h64_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41_h64_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41_h128_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41_h128_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41fp4_h64_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41fp4_h64_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41fp4_h128_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41fp4_h128_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.cu

        # sm100 sparse decode (head64 and native head128)
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v32_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v32_h64_no_split.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v4_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v4_h64_no_split.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41_h64_no_split.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64_no_split.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/nvfp4_head64/instantiations/v32_nvfp4_fp8rope.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_splitkv.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41_splitkv.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4_splitkv.cu
    )

    set(FlashMLA_Extension_SOURCES
        ${flashmla_SOURCE_DIR}/csrc/extension/torch_api.cpp
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/pybind.cpp
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_fp8_sm90.cu
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_metadata.cu
    )

    set(FlashMLA_INCLUDES
        ${flashmla_SOURCE_DIR}/csrc
        ${flashmla_SOURCE_DIR}/csrc/kerutils/include
        ${flashmla_SOURCE_DIR}/csrc/cutlass/include
        ${flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
    )
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
        # CUDA 13 moved cuda/std headers under include/cccl; nvcc finds them,
        # the host compiler building csrc/api/*.cpp does not.
        list(APPEND FlashMLA_INCLUDES ${CUDA_TOOLKIT_ROOT_DIR}/include/cccl)
    endif()

    set(FlashMLA_Extension_INCLUDES
        ${flashmla_SOURCE_DIR}/csrc
        ${flashmla_SOURCE_DIR}/csrc/kerutils/include
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
        COMPILE_FLAGS ${VLLM_FLASHMLA_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        INCLUDE_DIRECTORIES ${FlashMLA_INCLUDES}
        USE_SABI 3
        WITH_SOABI)

    # Enable C++20 for the FlashMLA sources (required for std::span, requires, etc.)
    target_compile_options(_flashmla_C PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>)

    # _flashmla_C is now ABI-stable torch 2.11+
    target_compile_definitions(_flashmla_C PRIVATE
        TORCH_TARGET_VERSION=0x020B000000000000ULL)
    if(VLLM_GPU_LANG STREQUAL "CUDA")
        target_compile_definitions(_flashmla_C PRIVATE USE_CUDA)
    endif()

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

    # _flashmla_extension_C is now ABI-stable w/ torch 2.11+
    target_compile_definitions(_flashmla_extension_C PRIVATE
        TORCH_TARGET_VERSION=0x020B000000000000ULL)
    if(VLLM_GPU_LANG STREQUAL "CUDA")
        target_compile_definitions(_flashmla_extension_C PRIVATE USE_CUDA)
    endif()
else()
    message(STATUS "FlashMLA will not compile: unsupported CUDA architecture ${CUDA_ARCHS}")
    # Create empty targets for setup.py on unsupported systems
    add_custom_target(_flashmla_C)
    add_custom_target(_flashmla_extension_C)
endif()
