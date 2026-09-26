#
# _moe_C_stable_libtorch extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/libtorch_stable/moe/torch_bindings.cpp"
  "csrc/libtorch_stable/moe/moe_align_sum_kernels.cu"
  "csrc/libtorch_stable/moe/topk_softmax_kernels.cu"
  "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_MOE_EXT_SRC
    "csrc/libtorch_stable/moe/moe_wna16.cu"
    "csrc/libtorch_stable/moe/grouped_topk_kernels.cu")
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(MOE_PERMUTE_SRC
      "csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
      "csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu")

  list(APPEND VLLM_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")
endif()

set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  include(${CMAKE_CURRENT_LIST_DIR}/moe_extension_cuda.cmake)
endif()

message(STATUS "Enabling MoE C_stable extension.")
define_extension_target(
  _moe_C_stable_libtorch
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

# Set TORCH_TARGET_VERSION for stable ABI compatibility.
# This ensures we only use C-shim APIs available in PyTorch 2.11.
# _moe_C_stable_libtorch is abi compatible with PyTorch >= TORCH_TARGET_VERSION
# which is currently set to 2.11.
target_compile_definitions(_moe_C_stable_libtorch PRIVATE
  TORCH_TARGET_VERSION=0x020B000000000000ULL)

# Needed to use cuda/hip APIs from C-shim
if(VLLM_GPU_LANG STREQUAL "CUDA")
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE USE_CUDA)
  # Needed by CUTLASS kernels
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE
    CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
elseif(VLLM_GPU_LANG STREQUAL "HIP")
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE USE_ROCM)
endif()

# On ROCm, _moe_C_stable_libtorch calls raw HIP APIs (e.g. hipGetDevice in
# get_device_prop()) which must resolve to the same libamdhip64.so that
# PyTorch uses.  When PyTorch bundles its own copy (pip/conda wheels),
# the raw HIP calls would otherwise resolve to the system ROCm copy,
# initializing a second HIP runtime that corrupts device state (wrong
# device on DeviceGuard, core dumps on multi-GPU tests).
#
# If PyTorch doesn't bundle libamdhip64 (built from source against system
# ROCm), there is only one copy in the process and no action is needed —
# the HIP compiler already links the system libamdhip64 automatically.
if(VLLM_GPU_LANG STREQUAL "HIP")
  find_library(_MOE_STABLE_TORCH_AMDHIP64 amdhip64
    PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
  if(_MOE_STABLE_TORCH_AMDHIP64)
    message(STATUS "Found PyTorch-bundled libamdhip64 for _moe_C_stable_libtorch at ${_MOE_STABLE_TORCH_AMDHIP64}")
    target_link_libraries(_moe_C_stable_libtorch PRIVATE ${_MOE_STABLE_TORCH_AMDHIP64})
  endif()
endif()
