#
# Builds the RDNA attention kernels of one GPU family into the object
# library ${TARGET} for ${ARCHES}; FAMILY names the family's config tables
# (Arch::k${FAMILY} in csrc/rocm/rdna_attn_types.h). Each (kernel, head dim,
# dtype) is a generated translation unit that builds the entries of the
# family's table in csrc/rocm/rdna_<kernel>_configs.h with that head dim and
# dtype and registers them, so the builds compile in parallel. The head dims
# and dtypes are the enumerators of rdna_attn_types.h; an entry without a unit
# makes rdna_<kernel>_configs() raise.
#
function(rdna_attn_target TARGET FAMILY ARCHES)
  set(_KERNELS decode decode_batch prefill)
  set(_HEAD_DIMS 64 96 128 256 512)
  set(_DTYPES Fp16 Bf16)
  string(TOLOWER ${FAMILY} _FAMILY_NAME)
  set(_OUT ${CMAKE_CURRENT_BINARY_DIR}/rdna_attn/${_FAMILY_NAME})
  set(_UNITS "")
  foreach(_KERNEL IN LISTS _KERNELS)
    foreach(_HEAD_DIM IN LISTS _HEAD_DIMS)
      foreach(_DTYPE IN LISTS _DTYPES)
        string(TOLOWER ${_DTYPE} _DTYPE_NAME)
        set(_UNIT ${_OUT}/${_KERNEL}_d${_HEAD_DIM}_${_DTYPE_NAME}.hip)
        file(CONFIGURE OUTPUT ${_UNIT} CONTENT
"#include \"rdna_attn_instances.cuh\"
using namespace rdna;
static const bool kRegistered = register_${_KERNEL}<
    Arch::k${FAMILY}, HeadDim::k${_HEAD_DIM}, Dtype::k${_DTYPE}>();
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
  # The archs the registry offers the family on.
  foreach(_ARCH IN LISTS ARCHES)
    string(TOUPPER ${_ARCH} _ARCH)
    target_compile_definitions(${TARGET} INTERFACE VLLM_RDNA_ATTN_${_ARCH})
  endforeach()
endfunction()

# The object libraries, in OUT, of the GPU families with an arch in
# VLLM_GPU_ARCHES: Gfx1151 (gfx1151), Gfx1150 (gfx1150, gfx1152, gfx1153) and
# Gfx1201 (gfx1200, gfx1201).
function(rdna_attn_targets OUT)
  set(_TARGETS "")
  foreach(_FAMILY_ARCHES "Gfx1151 gfx1151" "Gfx1150 gfx1150 gfx1152 gfx1153"
                         "Gfx1201 gfx1200 gfx1201")
    separate_arguments(_FAMILY_ARCHES)
    list(POP_FRONT _FAMILY_ARCHES _FAMILY)
    set(_ARCHES "")
    foreach(_ARCH IN LISTS _FAMILY_ARCHES)
      if(_ARCH IN_LIST VLLM_GPU_ARCHES)
        list(APPEND _ARCHES ${_ARCH})
      endif()
    endforeach()
    if(_ARCHES)
      string(TOLOWER ${_FAMILY} _NAME)
      rdna_attn_target(_rocm_C_rdna_attn_${_NAME} ${_FAMILY} "${_ARCHES}")
      list(APPEND _TARGETS _rocm_C_rdna_attn_${_NAME})
    endif()
  endforeach()
  set(${OUT} ${_TARGETS} PARENT_SCOPE)
endfunction()
