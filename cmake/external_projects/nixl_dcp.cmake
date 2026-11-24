# NIXL's device API is header-only. Build against the same UCX revision as
# the NIXL release wheel, without linking another UCX runtime into vLLM.
include(FetchContent)
FetchContent_Declare(nixl_dcp_sdk
  GIT_REPOSITORY https://github.com/ai-dynamo/nixl.git
  GIT_TAG e72c52c78478d2e1ca9e9c57b872114e113ad767 # v1.5.2
  GIT_SUBMODULES "")
FetchContent_Declare(nixl_dcp_ucx
  GIT_REPOSITORY https://github.com/openucx/ucx.git
  GIT_TAG 65e342a82dd2609a28c8c2b8c45ed7fe2cb33fa0
  GIT_SUBMODULES external/gpunetio)
FetchContent_MakeAvailable(nixl_dcp_sdk nixl_dcp_ucx)

find_path(NIXL_DCP_VERBS_INCLUDE_DIR infiniband/mlx5dv.h REQUIRED)

# Reproduce a CUDA SDK header install. In particular, do not expose ROCm
# headers: UCX detects the enabled device transports using __has_include.
set(_nixl_headers "${nixl_dcp_ucx_BINARY_DIR}/include")
file(COPY "${nixl_dcp_ucx_SOURCE_DIR}/src/ucp"
          "${nixl_dcp_ucx_SOURCE_DIR}/src/ucs"
          "${nixl_dcp_ucx_SOURCE_DIR}/src/uct"
     DESTINATION "${_nixl_headers}"
     FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh" PATTERN "rocm" EXCLUDE)
file(COPY "${nixl_dcp_ucx_SOURCE_DIR}/external/gpunetio/include/"
     DESTINATION "${_nixl_headers}/gpunetio"
     FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")

function(nixl_dcp_configure_ucx_version)
  set(MAJOR_VERSION 1)
  set(MINOR_VERSION 23)
  set(PATCH_VERSION 0)
  set(EXTRA_VERSION "")
  set(VERSION "1.23.0")
  set(SCM_VERSION "65e342a82")
  set(SCM_BRANCH "")
  configure_file("${nixl_dcp_ucx_SOURCE_DIR}/src/uct/api/version.h.in"
                 "${_nixl_headers}/uct/api/version.h" @ONLY)
endfunction()
nixl_dcp_configure_ucx_version()

set(NIXL_DCP_SRC "csrc/attention/dcp_utils/dcp_nixl_a2a.cu")
list(APPEND VLLM_STABLE_EXT_SRC "${NIXL_DCP_SRC}")
set_source_files_properties("${NIXL_DCP_SRC}" PROPERTIES
  INCLUDE_DIRECTORIES "${nixl_dcp_sdk_SOURCE_DIR}/src/api/cpp;${nixl_dcp_sdk_SOURCE_DIR}/src/api/gpu/ucx;${_nixl_headers};${nixl_dcp_ucx_SOURCE_DIR}/external/gpunetio/include;${NIXL_DCP_VERBS_INCLUDE_DIR}"
  COMPILE_DEFINITIONS "UCP_DEVICE_ENABLE_PARAMS_CHECK=1")
