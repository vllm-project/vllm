#
# _C_stable_libtorch extension (ops registered via STABLE_TORCH_LIBRARY)
#
# This file is included from the top-level CMakeLists.txt when building for
# CUDA. It defines the _C_stable_libtorch target and its CUTLASS / scaled_mm /
# CUTLASS-MoE / NVFP4 / W4A8 kernel sources.
#
# Note: this file also calls target_sources(_C ...) and
# target_compile_definitions(_C ...) on the _C target defined in the top-level
# CMakeLists.txt, so it must be included after _C is defined.
#

set(VLLM_STABLE_EXT_SRC
  "csrc/libtorch_stable/torch_bindings.cpp"
  "csrc/cutlass_extensions/common.cpp"
  "csrc/cuda_utils_kernels.cu"
  "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu"
  "csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu"
  "csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_STABLE_EXT_SRC
    "csrc/libtorch_stable/permute_cols.cu"
    "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu"
    "csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu")
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set_gencode_flags_for_srcs(
    SRCS "${VLLM_STABLE_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")
endif()

#
# CUTLASS scaled_mm kernels (moved from _C to _C_stable_libtorch)
#
set(SCALED_MM_3X_ARCHS)
# The cutlass_scaled_mm kernels for Hopper (c3x, i.e. CUTLASS 3.x) require
# CUDA 12.0 or later
cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
  set(SRCS
     "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm90.cu"
     "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8.cu"
     "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_int8.cu"
     "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_azp_sm90_int8.cu"
     "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM90=1")
  # Let scaled_mm_c2x know it doesn't need to build these arches
  list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
  message(STATUS "Building scaled_mm_c3x_sm90 for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
    message(STATUS "Not building scaled_mm_c3x_sm90 as CUDA Compiler version is "
                   "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                   "later if you intend on running FP8 quantized models on "
                   "Hopper.")
  else()
    message(STATUS "Not building scaled_mm_c3x_sm90 as no compatible archs found "
                   "in CUDA target architectures")
  endif()
endif()


# The cutlass_scaled_mm kernels for Blackwell SM12x (c3x, i.e. CUTLASS 3.x) require
# CUDA 12.8 or later
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
  set(SRCS
    "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu"
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu"
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu"
  )
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM120=1")
  # Let scaled_mm_c2x know it doesn't need to build these arches
  list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
  message(STATUS "Building scaled_mm_c3x_sm120 for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    message(STATUS "Not building scaled_mm_c3x_sm120 as CUDA Compiler version is "
                   "not >= 12.8, we recommend upgrading to CUDA 12.8 or "
                   "later if you intend on running FP8 quantized models on "
                   "Blackwell.")
  else()
    message(STATUS "Not building scaled_mm_c3x_120 as no compatible archs found "
                   "in CUDA target architectures")
  endif()
endif()


# The cutlass_scaled_mm kernels for Blackwell SM100 (c3x, i.e. CUTLASS 3.x)
# require CUDA 12.8 or later
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
  set(SRCS
    "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm100.cu"
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu"
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8.cu"
  )
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM100=1")
  # Let scaled_mm_c2x know it doesn't need to build these arches
  list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
  message(STATUS "Building scaled_mm_c3x_sm100 for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    message(STATUS "Not building scaled_mm_c3x_sm100 as CUDA Compiler version is "
                   "not >= 12.8, we recommend upgrading to CUDA 12.8 or "
                   "later if you intend on running FP8 quantized models on "
                   "Blackwell.")
  else()
    message(STATUS "Not building scaled_mm_c3x_100 as no compatible archs found "
                   "in CUDA target architectures")
  endif()
endif()

#
# For the cutlass_scaled_mm kernels we want to build the c2x (CUTLASS 2.x)
# kernels for the remaining archs that are not already built for 3x.
# (Build 8.9 for FP8)
cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
  "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
# subtract out the archs that are already built for 3x
list(REMOVE_ITEM SCALED_MM_2X_ARCHS ${SCALED_MM_3X_ARCHS})
if (SCALED_MM_2X_ARCHS)
  set(SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_2X_ARCHS}")
  list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_C2X=1")
  message(STATUS "Building scaled_mm_c2x for archs: ${SCALED_MM_2X_ARCHS}")
else()
  if (SCALED_MM_3X_ARCHS)
    message(STATUS "Not building scaled_mm_c2x as all archs are already built"
                   " for and covered by scaled_mm_c3x")
  else()
    message(STATUS "Not building scaled_mm_c2x as no compatible archs found "
                  "in CUDA target architectures")
  endif()
endif()

#
# CUTLASS MoE kernels (moved from _C to _C_stable_libtorch)
#

# The MoE kernel cutlass_moe_mm requires CUDA 12.3 or later (and ONLY works
# on Hopper). get_cutlass_(batched_)moe_mm_data should only be compiled
# if it's possible to compile MoE kernels that use its output.
cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
  set(SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm90.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM90=1")
  message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
    message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
                   "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
                   "if you intend on running FP8 quantized MoE models on Hopper.")
  else()
    message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
                   "in CUDA target architectures.")
  endif()
endif()

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
  set(SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
  message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
                   "not >= 12.8, we recommend upgrading to CUDA 12.8 or later "
                   "if you intend on running FP8 quantized MoE models on Blackwell.")
  else()
    message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
                   "in CUDA target architectures.")
  endif()
endif()

# moe_data.cu is used by all CUTLASS MoE kernels.
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
  set(SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/moe/moe_data.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${CUTLASS_MOE_DATA_ARCHS}")
  list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
  message(STATUS "Building moe_data for archs: ${CUTLASS_MOE_DATA_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
    message(STATUS "Not building moe_data as CUDA Compiler version is "
                   "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
                   "if you intend on running FP8 quantized MoE models on Hopper or Blackwell.")
  else()
    message(STATUS "Not building moe_data as no compatible archs found "
                   "in CUDA target architectures.")
  endif()
endif()

#
# FP4/NVFP4 kernels (moved from _C to _C_stable_libtorch)
#

# The nvfp4_scaled_mm_sm120 kernels for Blackwell SM12x require
# CUDA 12.8 or later
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(FP4_ARCHS "12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(FP4_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
  set(SRCS
    "csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu"
    "csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
    "csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu"
    "csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu"
    "csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${FP4_ARCHS}")
  list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
  # nvfp4_kv_cache_kernels uses non-stable torch API and is called directly
  # from cache_kernels.cu, so it belongs in _C rather than _C_stable.
  set(NVFP4_KV_SRC "csrc/nvfp4_kv_cache_kernels.cu")
  set_gencode_flags_for_srcs(
    SRCS "${NVFP4_KV_SRC}"
    CUDA_ARCHS "${FP4_ARCHS}")
  target_sources(_C PRIVATE ${NVFP4_KV_SRC})
  target_compile_definitions(_C PRIVATE ENABLE_NVFP4_SM120=1)
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM120=1")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM120=1")
  message(STATUS "Building NVFP4 for archs: ${FP4_ARCHS}")
else()
  message(STATUS "Not building NVFP4 as no compatible archs were found.")
  # clear FP4_ARCHS
  set(FP4_ARCHS)
endif()

# FP4 Archs and flags
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(FP4_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(FP4_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
  set(SRCS
    "csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu"
    "csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
    "csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu"
    "csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_kernels.cu"
    "csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu"
    "csrc/libtorch_stable/quantization/fp4/mxfp4_experts_quant.cu"
    "csrc/libtorch_stable/quantization/fp4/mxfp4_blockwise_moe_kernel.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${FP4_ARCHS}")
  list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
  set(NVFP4_KV_SRC "csrc/nvfp4_kv_cache_kernels.cu")
  set_gencode_flags_for_srcs(
    SRCS "${NVFP4_KV_SRC}"
    CUDA_ARCHS "${FP4_ARCHS}")
  target_sources(_C PRIVATE ${NVFP4_KV_SRC})
  target_compile_definitions(_C PRIVATE ENABLE_NVFP4_SM100=1)
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM100=1")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
  message(STATUS "Building NVFP4 for archs: ${FP4_ARCHS}")
else()
  message(STATUS "Not building NVFP4 as no compatible archs were found.")
  # clear FP4_ARCHS
  set(FP4_ARCHS)
endif()

#
# W4A8 kernels (moved from _C to _C_stable_libtorch)
#

# Only build W4A8 kernels if we are building for something compatible with sm90a
cuda_archs_loose_intersection(W4A8_ARCHS "9.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND W4A8_ARCHS)
  set(SRCS
     "csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu"
     "csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_grouped_mm_entry.cu"
     "csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_utils.cu"
     )

  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${W4A8_ARCHS}")

  list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")

  message(STATUS "Building W4A8 kernels for archs: ${W4A8_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0
      AND W4A8_ARCHS)
    message(STATUS "Not building W4A8 kernels as CUDA Compiler version is "
                   "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                   "later if you intend on running w4a16 quantized models on "
                   "Hopper.")
  else()
    message(STATUS "Not building W4A8 kernels as no compatible archs "
                   "found in CUDA target architectures")
  endif()
endif()

message(STATUS "Enabling C_stable extension.")
define_extension_target(
  _C_stable_libtorch
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_STABLE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR} ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

# Set TORCH_TARGET_VERSION for stable ABI compatibility.
# This ensures we only use C-shim APIs available in PyTorch 2.10.
# _C_stable_libtorch is abi compatible with PyTorch >= TORCH_TARGET_VERSION
# which is currently set to 2.10.
target_compile_definitions(_C_stable_libtorch PRIVATE
  TORCH_TARGET_VERSION=0x020A000000000000ULL)

# Needed to use cuda APIs from C-shim
target_compile_definitions(_C_stable_libtorch PRIVATE
  USE_CUDA)

# Needed by CUTLASS kernels
target_compile_definitions(_C_stable_libtorch PRIVATE
  CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
