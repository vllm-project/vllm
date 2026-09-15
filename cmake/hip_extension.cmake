#
# Legacy _C extension (ROCm only — CUDA ops migrated to _C_stable_libtorch)
#

if(VLLM_GPU_LANG STREQUAL "HIP")
  set(VLLM_EXT_SRC
    "csrc/torch_bindings.cpp"
    "csrc/custom_quickreduce.cu")

  message(STATUS "Enabling C extension.")
  define_extension_target(
    _C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
    USE_SABI 3
    WITH_SOABI)

  # If CUTLASS is compiled on NVCC >= 12.5, it by default uses
  # cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
  # driver API. This causes problems when linking with earlier versions of CUDA.
  # Setting this variable sidesteps the issue by calling the driver directly.
  target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
endif() # _C HIP endif
