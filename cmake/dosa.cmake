# Optional AOT extension; operators share vLLM's existing _C Torch namespace.
set(_dosa_root "${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/dosa")
set(_dosa_sources "${_dosa_root}/module.cpp")
cuda_archs_loose_intersection(DOSA_ARCHS "10.0a" "${CUDA_ARCHS}")
if(DOSA_ARCHS AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  set(_dosa_generated "${CMAKE_CURRENT_BINARY_DIR}/dosa")
  file(MAKE_DIRECTORY "${_dosa_generated}")
  foreach(_heads 32 64)
    foreach(_role compact direct_abs high low planner refine sample select)
      set(_namespace "dosa_h${_heads}")
      set(_source "${_dosa_root}/${_role}.cu")
      set(_role_defines "")
      if(_role STREQUAL "direct_abs")
        set(_namespace "dosa_h${_heads}_direct")
        set(_source "${_dosa_root}/planner.cu")
        set(_role_defines "#define DOSA_DIRECT_ABS 1\n")
      endif()
      set(_unit "${_dosa_generated}/h${_heads}_${_role}.cu")
      string(CONCAT _dosa_unit_content
        "${_role_defines}"
        "#define DOSA_SCORE_HEADS ${_heads}\n"
        "#define DOSA_NAMESPACE ${_namespace}\n"
        "#define DOSA_OP_PREFIX \"dosa_h${_heads}_${_role}_\"\n"
        "#include \"${_source}\"\n")
      file(GENERATE OUTPUT "${_unit}" CONTENT "${_dosa_unit_content}")
      set_gencode_flags_for_srcs(SRCS "${_unit}" CUDA_ARCHS "${DOSA_ARCHS}")
      set_source_files_properties("${_unit}" PROPERTIES INCLUDE_DIRECTORIES
        "${deepgemm_SOURCE_DIR}/deep_gemm/include;${deepgemm_SOURCE_DIR}/third-party/cutlass/include")
      set_property(SOURCE "${_unit}" APPEND PROPERTY COMPILE_OPTIONS
        -O3 -lineinfo --expt-relaxed-constexpr -std=c++20
        --prec-div=true --prec-sqrt=true --ftz=false --fmad=true)
      if(_role MATCHES "^(low|direct_abs)$")
        set_property(SOURCE "${_unit}" APPEND PROPERTY COMPILE_OPTIONS --fmad=false)
      endif()
      if(_role STREQUAL "low")
        set_property(SOURCE "${_unit}" APPEND PROPERTY COMPILE_OPTIONS --expt-extended-lambda)
      endif()
      list(APPEND _dosa_sources "${_unit}")
    endforeach()
  endforeach()
endif()

set(_dosa_gpu_flags ${VLLM_GPU_FLAGS})
list(REMOVE_ITEM _dosa_gpu_flags --use_fast_math -use_fast_math)
define_extension_target(
  _dosa_C
  DESTINATION vllm
  LANGUAGE CUDA
  SOURCES ${_dosa_sources}
  COMPILE_FLAGS ${_dosa_gpu_flags}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)
target_compile_definitions(_dosa_C PRIVATE
  TORCH_TARGET_VERSION=0x020B000000000000ULL
  USE_CUDA CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
