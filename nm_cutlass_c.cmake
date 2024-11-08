function(build_nm_cutlass_c)

  message (STATUS "Project root dir ${PROJECT_ROOT_DIR}")
  file(GLOB full_path_generated_dirs LIST_DIRECTORIES true "${PROJECT_ROOT_DIR}/csrc/sparse/cutlass/generator/generated/*")
  
  message (STATUS "fullpath generated dirs ${full_path_generated_dirs}")
  
  set(generated_dirs)
  foreach(d ${full_path_generated_dirs})
    get_filename_component(d_name ${d} NAME)
    list(APPEND generated_dirs ${d_name})
  endforeach()
  
  set(NM_CUTLASS_C_ARCHS "9.0;9.0a")
  
  foreach(d ${generated_dirs})
  
      set(SRCS_DIR "csrc/sparse/cutlass/generator/generated/${d}")
      set(SRCS)
      file(GLOB SRCS "${SRCS_DIR}/*cu")
      list(APPEND SRCS "${SRCS_DIR}/torch_bindings.cpp")
  
      set_gencode_flags_for_srcs(
        SRCS "${SRCS}"
        CUDA_ARCHS "${NM_CUTLASS_C_ARCHS}")
  
      set(EXT_NAME "_nm_cutlass_${d}_C")
      message(STATUS "Enabling ${EXT_NAME} extension.")
      define_gpu_extension_target(
        ${EXT_NAME}
        DESTINATION vllm
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES ${SRCS}
        COMPILE_FLAGS ${VLLM_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
        USE_SABI 3
        WITH_SOABI)
  
      target_compile_definitions(${EXT_NAME} PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
  
  endforeach()

endfunction()
