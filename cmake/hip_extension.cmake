#
# Legacy _C extension (ROCm only — CUDA ops migrated to _C_stable_libtorch)
#

if(VLLM_GPU_LANG STREQUAL "HIP")
  set(VLLM_EXT_SRC
    "csrc/rocm/quickreduce/torch_bindings.cpp"
    "csrc/rocm/quickreduce/custom_quickreduce.cu")
  # Octave KV cache (store, decode, rotation, prefill): only built for the
  # validated architectures, gfx1100 (RDNA3) and gfx942 (MI300).
  set(VLLM_OCTAVE_KV OFF)
  if(VLLM_GPU_ARCHES MATCHES "gfx942|gfx1100")
    set(VLLM_OCTAVE_KV ON)
    list(APPEND VLLM_EXT_SRC
      "csrc/attention/octave_torch_bindings.cpp"
      "csrc/attention/octave_attn.cu"
      "csrc/attention/octave_decode_wmma_rdna3.cu"
      "csrc/attention/octave_prefill_rdna3.cu"
      "csrc/attention/octave_prefill_cdna.cu"
      "csrc/attention/octave_decode_mfma_cdna.cu"
    )
  endif()

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
  if(VLLM_OCTAVE_KV)
    # MFMA accumulators in VGPRs: in AGPRs every online-softmax rescale of the
    # output copies it through 64 extra VGPRs and halves the occupancy.
    set_source_files_properties(
      "${CMAKE_CURRENT_BINARY_DIR}/csrc/attention/octave_decode_mfma_cdna.hip"
      PROPERTIES COMPILE_FLAGS "-mllvm -amdgpu-mfma-vgpr-form")
  endif()
endif() # _C HIP endif
