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

set(
  ZOOMKV_SRC_DIR
  ${CMAKE_CURRENT_LIST_DIR}/../vllm/v1/attention/ops/zoomkv
)

set(VLLM_ZOOMKV_SRCS
  ${ZOOMKV_SRC_DIR}/cuda/bindings.cpp
  ${ZOOMKV_SRC_DIR}/cuda/quest_chunk_score.cu
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
endif()
