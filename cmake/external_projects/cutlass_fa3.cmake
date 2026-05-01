# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# CUTLASS FA3 MLA Sparse Attention — requires CUDA >= 12.4, SM90a
#
# Vendors the sgl-attn CUTLASS FlashAttention3 kernel from SGLang into vLLM
# as a self-contained extension (_cutlass_fa3_C). This provides a high-
# performance sparse MLA attention kernel for SM90 (Hopper) GPUs.
#
# Source: https://github.com/sgl-project/sgl-attn (commit bcf72ccc)
# CUTLASS: https://github.com/NVIDIA/cutlass (commit 57e3cfb4)

# Guard: CUDA >= 12.4 required for SM90a features used by FA3
if(NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "12.4")
  message(STATUS "Skipping CUTLASS FA3: requires CUDA >= 12.4")
  # Create empty target so setup.py doesn't fail on unsupported systems
  add_custom_target(_cutlass_fa3_C)
  return()
endif()

# Guard: SM90 architecture required
set(CUTLASS_FA3_SUPPORT_ARCHS)
list(APPEND CUTLASS_FA3_SUPPORT_ARCHS "9.0a")
cuda_archs_loose_intersection(
  CUTLASS_FA3_ARCHS "${CUTLASS_FA3_SUPPORT_ARCHS}" "${CUDA_ARCHS}")
if(NOT CUTLASS_FA3_ARCHS)
  message(STATUS "Skipping CUTLASS FA3: requires SM90 (CUDA_ARCHS=${CUDA_ARCHS})")
  add_custom_target(_cutlass_fa3_C)
  return()
endif()

include(FetchContent)

# Fetch sgl-attn (Flash Attention 3 kernels from SGLang)
# We only need the source files, not the build system, so we use
# FetchContent_Populate to download without building.
if (DEFINED ENV{SGL_ATTN_SRC_DIR})
  set(SGL_ATTN_SRC_DIR $ENV{SGL_ATTN_SRC_DIR})
endif()
if(SGL_ATTN_SRC_DIR)
  FetchContent_Declare(cutlass_fa3
    SOURCE_DIR ${SGL_ATTN_SRC_DIR})
else()
  FetchContent_Declare(cutlass_fa3
    GIT_REPOSITORY https://github.com/sgl-project/sgl-attn.git
    GIT_TAG bcf72ccc6816b36a5fae2c5a3c027604629785e0
    GIT_PROGRESS TRUE
    GIT_SHALLOW FALSE)
endif()
FetchContent_GetProperties(cutlass_fa3)
if(NOT cutlass_fa3_POPULATED)
  FetchContent_Populate(cutlass_fa3)
endif()
message(STATUS "CUTLASS FA3 sgl-attn source: ${cutlass_fa3_SOURCE_DIR}")

# Fetch CUTLASS for FA3 (headers only, separate from vLLM's main CUTLASS
# to avoid version conflicts). Use FetchContent_Populate to avoid running
# CUTLASS's own CMakeLists.txt which would create conflicting targets.
if (DEFINED ENV{CUTLASS_FA3_CUTLASS_SRC_DIR})
  set(CUTLASS_FA3_CUTLASS_SRC_DIR $ENV{CUTLASS_FA3_CUTLASS_SRC_DIR})
endif()
if(CUTLASS_FA3_CUTLASS_SRC_DIR)
  FetchContent_Declare(cutlass_for_fa3
    SOURCE_DIR ${CUTLASS_FA3_CUTLASS_SRC_DIR})
else()
  FetchContent_Declare(cutlass_for_fa3
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG 57e3cfb47a2d9e0d46eb6335c3dc411498efa198
    GIT_PROGRESS TRUE
    GIT_SHALLOW FALSE)
endif()
FetchContent_GetProperties(cutlass_for_fa3)
if(NOT cutlass_for_fa3_POPULATED)
  FetchContent_Populate(cutlass_for_fa3)
endif()
message(STATUS "CUTLASS FA3 cutlass source: ${cutlass_for_fa3_SOURCE_DIR}")

set(FA3_SRC "${cutlass_fa3_SOURCE_DIR}/hopper")

# flash_api.cpp dispatches to all head dimensions + dtypes (BF16, FP16, FP8)
# at compile time. With FLASHATTENTION_DISABLE_SM8x, only SM90 instantiations
# are needed. We exclude hdimall_* (fails on CUDA 13+) and backward files.
file(GLOB FA3_INSTANTIATION_SOURCES
  # BF16 instantiations
  "${FA3_SRC}/instantiations/flash_fwd_hdim64_bf16*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim96_bf16*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim128_bf16*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim192_bf16*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim256_bf16*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdimdiff_bf16*_sm90.cu"
  # FP16 instantiations
  "${FA3_SRC}/instantiations/flash_fwd_hdim64_fp16*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim96_fp16*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim128_fp16*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim192_fp16*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim256_fp16*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdimdiff_fp16*_sm90.cu"
  # FP8 (e4m3) instantiations
  "${FA3_SRC}/instantiations/flash_fwd_hdim64_e4m3*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim96_e4m3*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim128_e4m3*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim192_e4m3*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdim256_e4m3*_sm90.cu"
  "${FA3_SRC}/instantiations/flash_fwd_hdimdiff_e4m3*_sm90.cu")

set(FA3_CORE_SOURCES
  "${FA3_SRC}/flash_api.cpp"
  "${FA3_SRC}/flash_prepare_scheduler.cu"
  "${FA3_SRC}/flash_fwd_combine.cu")

set(FA3_ALL_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass_fa3_extension.cc"
  ${FA3_CORE_SOURCES}
  ${FA3_INSTANTIATION_SOURCES})

set(FA3_INCLUDE_DIRS
  ${FA3_SRC}
  ${cutlass_fa3_SOURCE_DIR}/include
  ${cutlass_for_fa3_SOURCE_DIR}/include
  ${cutlass_for_fa3_SOURCE_DIR}/tools/util/include
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

# Set SM90a gencode flags for all FA3 CUDA sources
set_gencode_flags_for_srcs(
  SRCS "${FA3_ALL_SOURCES}"
  CUDA_ARCHS "${CUTLASS_FA3_ARCHS}")

define_extension_target(_cutlass_fa3_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${FA3_ALL_SOURCES}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${FA3_INCLUDE_DIRS}
  USE_SABI 3
  WITH_SOABI)

# FA3-specific compile options for CUDA and C++ source files:
# - C++17 required by CUTLASS
# - Fast math for performance
# - Relaxed constexpr for CUTLASS template metaprogramming
# - Disable backward pass, dropout, uneven K (not needed for inference)
# - Enable varlen-only mode (all our use cases are variable-length)
target_compile_options(_cutlass_fa3_C PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API>
  $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>
  $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_compile_definitions(_cutlass_fa3_C PRIVATE
  CUTE_USE_PACKED_TUPLE=1
  CUTLASS_ENABLE_GDC_FOR_SM90
  CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
  CUTLASS_ENABLE_TENSOR_CORE_MMA=1
  FLASHATTENTION_DISABLE_BACKWARD
  FLASHATTENTION_DISABLE_DROPOUT
  FLASHATTENTION_DISABLE_UNEVEN_K
  FLASHATTENTION_DISABLE_SM8x
  FLASHATTENTION_VARLEN_ONLY)

message(STATUS "CUTLASS FA3 MLA Sparse: enabled for SM90 (${CUTLASS_FA3_ARCHS})")
