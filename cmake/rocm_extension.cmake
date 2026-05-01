#
# _rocm_C extension
#
# This file is included from the top-level CMakeLists.txt when building for
# ROCm/HIP. It defines the _rocm_C target with ROCm-specific kernels.
#

set(VLLM_ROCM_EXT_SRC
  "csrc/rocm/torch_bindings.cpp"
  "csrc/rocm/skinny_gemms.cu"
  "csrc/rocm/attention.cu")

define_extension_target(
  _rocm_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_ROCM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)
