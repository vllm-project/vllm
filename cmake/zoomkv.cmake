# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#[[
CMake target for ZoomKV GPU-only CUDA extension (vllm._zoomkv_C).

Build (from vLLM repo root, after configuring the main project):
  cmake -DVLLM_BUILD_ZOOMKV_EXT=ON ...
  cmake --build . --target _zoomkv_C
  cmake --install . --component _zoomkv_C

Sources live under vllm/v1/attention/ops/zoomkv/ and cuda/.
]]

option(VLLM_BUILD_ZOOMKV_EXT "Build the optional ZoomKV CUDA extension" OFF)
option(
  VLLM_ZOOMKV_DEBUG_CUDA
  "Build ZoomKV CUDA kernels with line info and optimization disabled"
  OFF
)

set(
  ZOOMKV_SRC_DIR
  ${CMAKE_CURRENT_LIST_DIR}/../vllm/v1/attention/ops/zoomkv
)

set(VLLM_ZOOMKV_SRCS
  ${ZOOMKV_SRC_DIR}/cuda/bindings.cpp
  ${ZOOMKV_SRC_DIR}/cuda/quest_chunk_score.cu
  ${ZOOMKV_SRC_DIR}/cuda/physical_retrieval.cu
  ${ZOOMKV_SRC_DIR}/kivi_qk_dot.cu
  ${ZOOMKV_SRC_DIR}/cuda/rerank_topk.cu
  ${ZOOMKV_SRC_DIR}/cuda/float_topk.cu
  ${ZOOMKV_SRC_DIR}/cuda/h2d_gather_tokens.cu
)

# Optional extension — only built when explicitly requested.
if(VLLM_BUILD_ZOOMKV_EXT AND VLLM_GPU_LANG STREQUAL "CUDA")
  define_extension_target(
    _zoomkv_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_ZOOMKV_SRCS}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    WITH_SOABI
  )
  target_compile_definitions(_zoomkv_C PRIVATE ZOOMKV_UNIFIED_EXTENSION=1)
  if(VLLM_ZOOMKV_DEBUG_CUDA)
    target_compile_options(
      _zoomkv_C
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-O0>
    )
  endif()
  # Some development environments carry a stale libtorch C++ API include in
  # TorchConfig.cmake while linking the active Python torch package. Put the
  # active package headers first so CUDA/C++ objects use one ABI consistently.
  get_filename_component(
    ZOOMKV_TORCH_PACKAGE_DIR
    "${Torch_DIR}/../../.."
    ABSOLUTE
  )
  target_include_directories(
    _zoomkv_C
    BEFORE PRIVATE
    "${ZOOMKV_TORCH_PACKAGE_DIR}/include"
    "${ZOOMKV_TORCH_PACKAGE_DIR}/include/torch/csrc/api/include"
  )
endif()
