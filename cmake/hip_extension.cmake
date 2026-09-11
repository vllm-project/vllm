#
# Legacy _C extension (ROCm only — CUDA ops migrated to _C_stable_libtorch)
#

if(VLLM_GPU_LANG STREQUAL "HIP")
  set(VLLM_EXT_SRC
    "csrc/rocm/quickreduce/torch_bindings.cpp"
    "csrc/rocm/quickreduce/custom_quickreduce.cu")

  message(STATUS "Enabling C extension.")
  define_extension_target(
    _C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    USE_SABI 3
    WITH_SOABI)
endif() # _C HIP endif
