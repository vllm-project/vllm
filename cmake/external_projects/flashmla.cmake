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
        GIT_TAG c2afa9cb93e674d5a9120a170a6da57b89267208
        GIT_PROGRESS TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
endif()


FetchContent_MakeAvailable(flashmla)
message(STATUS "FlashMLA is available at ${flashmla_SOURCE_DIR}")

# Vendor FlashMLA interface into vLLM with torch-ops shim.
set(FLASHMLA_VENDOR_DIR "${CMAKE_SOURCE_DIR}/vllm/third_party/flashmla")
file(MAKE_DIRECTORY "${FLASHMLA_VENDOR_DIR}")
file(READ "${flashmla_SOURCE_DIR}/flash_mla/flash_mla_interface.py"
     FLASHMLA_INTERFACE_CONTENT)
string(REPLACE "import flash_mla.cuda as flash_mla_cuda"
               "import vllm._flashmla_C\nflash_mla_cuda = torch.ops._flashmla_C"
               FLASHMLA_INTERFACE_CONTENT
               "${FLASHMLA_INTERFACE_CONTENT}")
file(WRITE "${FLASHMLA_VENDOR_DIR}/flash_mla_interface.py"
     "${FLASHMLA_INTERFACE_CONTENT}")

# Install the generated flash_mla_interface.py to the wheel
# Use COMPONENT _flashmla_C to ensure it's installed with the C extension
install(FILES "${FLASHMLA_VENDOR_DIR}/flash_mla_interface.py"
        DESTINATION vllm/third_party/flashmla/
        COMPONENT _flashmla_C)

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
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    list(APPEND SUPPORT_ARCHS "10.0a")
endif()


cuda_archs_loose_intersection(FLASH_MLA_ARCHS "${SUPPORT_ARCHS}" "${CUDA_ARCHS}")
if(FLASH_MLA_ARCHS)
    message(STATUS "FlashMLA CUDA architectures: ${FLASH_MLA_ARCHS}")
    set(VLLM_FLASHMLA_GPU_FLAGS ${VLLM_GPU_FLAGS})
    list(APPEND VLLM_FLASHMLA_GPU_FLAGS "--expt-relaxed-constexpr" "--expt-extended-lambda" "--use_fast_math")

    set(FlashMLA_SOURCES
        ${flashmla_SOURCE_DIR}/csrc/torch_api.cpp

        # Misc kernels for decoding
        ${flashmla_SOURCE_DIR}/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
        ${flashmla_SOURCE_DIR}/csrc/smxx/decode/combine/combine.cu

        # sm90 dense decode
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/instantiations/fp16.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/instantiations/bf16.cu

        # sm90 sparse decode
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu

        # sm90 sparse prefill
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/fwd.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu

        # sm100 dense prefill & backward
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu

        # sm100 sparse prefill
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu

        # sm100 sparse decode
        ${flashmla_SOURCE_DIR}/csrc/sm100/decode/head64/instantiations/v32.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/decode/head64/instantiations/model1.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu
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
        ${flashmla_SOURCE_DIR}/csrc/sm90
        ${flashmla_SOURCE_DIR}/csrc/cutlass/include
        ${flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
    )

    set(FlashMLA_Extension_INCLUDES
        ${flashmla_SOURCE_DIR}/csrc
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
    # Also enable C++20 for the FlashMLA sources (required for std::span, requires, etc.)
    target_compile_options(_flashmla_C PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API>
        $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>)

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
    message(STATUS "FlashMLA will not compile: unsupported CUDA architecture ${CUDA_ARCHS}")
    # Create empty targets for setup.py on unsupported systems
    add_custom_target(_flashmla_C)
    add_custom_target(_flashmla_extension_C)
endif()

