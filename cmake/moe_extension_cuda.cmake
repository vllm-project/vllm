# CUDA-specific MoE sources and per-source architecture flags.
set(VLLM_MOE_WNA16_SRC
  "csrc/libtorch_stable/moe/moe_wna16.cu")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_WNA16_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

list(APPEND VLLM_MOE_EXT_SRC "${VLLM_MOE_WNA16_SRC}")
# moe marlin arches
# note that we always set `use_atomic_add=False` for moe marlin now,
# so we don't need 9.0 for bf16 atomicAdd PTX
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX;12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX;12.0a;12.1a" "${CUDA_ARCHS}")
endif()
# moe marlin has limited support for turing
cuda_archs_loose_intersection(MARLIN_MOE_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
# moe marlin arches for fp8 input
# - sm80 doesn't support fp8 computation
# - sm90 and sm100 don't support QMMA.16832.F32.E4M3.E4M3 SAAS instruction
# so we only enable fp8 computation for SM89 (e.g. RTX 40x0)  and 12.0 (e.g. RTX 50x0)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(MARLIN_MOE_FP8_ARCHS "8.9;12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(MARLIN_MOE_FP8_ARCHS "8.9;12.0a;12.1a" "${CUDA_ARCHS}")
endif()
# moe marlin arches for other files
cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")
if (MARLIN_MOE_OTHER_ARCHS)

  #
  # For the Marlin MOE kernels we automatically generate sources for various
  # preselected input type pairs and schedules.
  # Generate sources:
  set(MOE_MARLIN_GEN_SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py)
  file(MD5 ${MOE_MARLIN_GEN_SCRIPT} MOE_MARLIN_GEN_SCRIPT_HASH)
  list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
  set(MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH "${MOE_MARLIN_GEN_SCRIPT_HASH}(ARCH:${CUDA_ARCHS_STR})")

  message(STATUS "Marlin MOE generation script hash with arch: ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")
  message(STATUS "Last run Marlin MOE generate script hash with arch: $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")

  if (NOT DEFINED CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
      OR NOT $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH} STREQUAL ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=$ENV{PYTHONPATH}
        ${Python_EXECUTABLE} ${MOE_MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
      RESULT_VARIABLE moe_marlin_generation_result
      OUTPUT_VARIABLE moe_marlin_generation_output
      OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
      ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
    )

    if (NOT moe_marlin_generation_result EQUAL 0)
      message(FATAL_ERROR "Marlin MOE generation failed."
                          " Result: \"${moe_marlin_generation_result}\""
                          "\nCheck the log for details: "
                          "${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log")
    else()
      set(MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
          CACHE STRING "Last run Marlin MOE generate script hash" FORCE)
      message(STATUS "Marlin MOE generation completed successfully.")
    endif()
  else()
    message(STATUS "Marlin MOE generation script has not changed, skipping generation.")
  endif()

  if (MARLIN_MOE_ARCHS)
    file(GLOB MARLIN_MOE_SRC "csrc/libtorch_stable/moe/marlin_moe_wna16/sm80_kernel_*.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_MOE_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SRC})
  endif()

  if (MARLIN_MOE_SM75_ARCHS)
    file(GLOB MARLIN_MOE_SM75_SRC "csrc/libtorch_stable/moe/marlin_moe_wna16/sm75_kernel_*.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_SM75_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_SM75_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_MOE_SM75_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SM75_SRC})
  endif()

  if (MARLIN_MOE_FP8_ARCHS)
    file(GLOB MARLIN_MOE_FP8_SRC "csrc/libtorch_stable/moe/marlin_moe_wna16/sm89_kernel_*.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_FP8_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_FP8_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_MOE_FP8_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_FP8_SRC})
  endif()

  set(MARLIN_MOE_OTHER_SRC "csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_MOE_OTHER_SRC}"
    CUDA_ARCHS "${MARLIN_MOE_OTHER_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    set_source_files_properties(${MARLIN_MOE_OTHER_SRC}
      PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
  endif()
  list(APPEND VLLM_MOE_EXT_SRC "${MARLIN_MOE_OTHER_SRC}")

  message(STATUS "Building Marlin MOE kernels for archs: ${MARLIN_MOE_OTHER_ARCHS}")
else()
  message(STATUS "Not building Marlin MOE kernels as no compatible archs found"
                 " in CUDA target architectures")
endif()
