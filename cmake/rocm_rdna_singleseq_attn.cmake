#
# Builds the RDNA3.5 attention kernels into the object library ${TARGET} for
# ${ARCHES}. Each (kernel, head dim, dtype) is a generated translation unit
# that builds the entries of csrc/rocm/rdna_singleseq_<kernel>_configs.h with
# that head dim and dtype and registers them, so the builds compile in
# parallel. The head dims and dtypes are the enumerators of
# csrc/rocm/rdna_singleseq_attn_types.h; an entry without a unit makes
# rdna_singleseq_<kernel>_configs() raise.
#
function(rdna_singleseq_attn_target TARGET ARCHES)
  set(_KERNELS decode prefill)
  set(_HEAD_DIMS 64 96 128 256 512)
  set(_DTYPES Fp16 Bf16)
  set(_OUT ${CMAKE_CURRENT_BINARY_DIR}/rdna_singleseq_attn)
  set(_UNITS "")
  foreach(_KERNEL IN LISTS _KERNELS)
    foreach(_HEAD_DIM IN LISTS _HEAD_DIMS)
      foreach(_DTYPE IN LISTS _DTYPES)
        string(TOLOWER ${_DTYPE} _DTYPE_NAME)
        set(_UNIT ${_OUT}/${_KERNEL}_d${_HEAD_DIM}_${_DTYPE_NAME}.hip)
        file(CONFIGURE OUTPUT ${_UNIT} CONTENT
"#include \"rdna_singleseq_attn_instances.cuh\"
using namespace rdna_singleseq;
static const bool kRegistered =
    register_${_KERNEL}<HeadDim::k${_HEAD_DIM}, Dtype::k${_DTYPE}>();
")
        list(APPEND _UNITS ${_UNIT})
      endforeach()
    endforeach()
  endforeach()

  add_library(${TARGET} OBJECT ${_UNITS})
  set_source_files_properties(${_UNITS} PROPERTIES LANGUAGE ${VLLM_GPU_LANG})
  set_target_properties(${TARGET} PROPERTIES
    ${VLLM_GPU_LANG}_ARCHITECTURES "${ARCHES}"
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/csrc/rocm)
  # -O3 regardless of build type. Some builds leave values unused.
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:${VLLM_GPU_LANG}>:${VLLM_ROCM_EXT_FLAGS}>
    -O3 -g0 -Wno-unused-variable)
endfunction()
