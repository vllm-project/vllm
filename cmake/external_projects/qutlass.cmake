include(FetchContent)

set(CUTLASS_INCLUDE_DIR "${CUTLASS_INCLUDE_DIR}" CACHE PATH "Path to CUTLASS include/ directory")

if(DEFINED ENV{QUTLASS_SRC_DIR})
  set(QUTLASS_SRC_DIR $ENV{QUTLASS_SRC_DIR})
endif()

if(QUTLASS_SRC_DIR)
  FetchContent_Declare(
    qutlass
    SOURCE_DIR ${QUTLASS_SRC_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
else()
  FetchContent_Declare(
    qutlass
    GIT_REPOSITORY https://github.com/IST-DASLab/qutlass.git
    GIT_TAG 830d2c4537c7396e14a02a46fbddd18b5d107c65
    GIT_PROGRESS TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
endif()

FetchContent_Populate(qutlass)

if(NOT qutlass_SOURCE_DIR)
  message(FATAL_ERROR "[QUTLASS] source directory could not be resolved.")
endif()
message(STATUS "[QUTLASS] QuTLASS is available at ${qutlass_SOURCE_DIR}")

cuda_archs_loose_intersection(QUTLASS_ARCHS "12.0a;10.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.8 AND QUTLASS_ARCHS)

  if(QUTLASS_ARCHS MATCHES "10\\.0a")
    set(QUTLASS_TARGET_CC 100)
  elseif(QUTLASS_ARCHS MATCHES "12\\.0a")
    set(QUTLASS_TARGET_CC 120)
  else()
    message(FATAL_ERROR "[QUTLASS] internal error parsing CUDA_ARCHS='${QUTLASS_ARCHS}'.")
  endif()

  set(QUTLASS_SOURCES
    ${qutlass_SOURCE_DIR}/qutlass/csrc/bindings.cpp
    ${qutlass_SOURCE_DIR}/qutlass/csrc/gemm.cu
    ${qutlass_SOURCE_DIR}/qutlass/csrc/gemm_ada.cu
    ${qutlass_SOURCE_DIR}/qutlass/csrc/fused_quantize_mx.cu
    ${qutlass_SOURCE_DIR}/qutlass/csrc/fused_quantize_nv.cu
    ${qutlass_SOURCE_DIR}/qutlass/csrc/fused_quantize_mx_sm100.cu
    ${qutlass_SOURCE_DIR}/qutlass/csrc/fused_quantize_nv_sm100.cu
  )

  set(QUTLASS_INCLUDES
    ${qutlass_SOURCE_DIR}
    ${qutlass_SOURCE_DIR}/qutlass
    ${qutlass_SOURCE_DIR}/qutlass/csrc/include
    ${qutlass_SOURCE_DIR}/qutlass/csrc/include/cutlass_extensions
  )

  if(CUTLASS_INCLUDE_DIR AND EXISTS "${CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h")
    list(APPEND QUTLASS_INCLUDES "${CUTLASS_INCLUDE_DIR}")
  elseif(EXISTS "${qutlass_SOURCE_DIR}/qutlass/third_party/cutlass/include/cutlass/cutlass.h")
    list(APPEND QUTLASS_INCLUDES "${qutlass_SOURCE_DIR}/qutlass/third_party/cutlass/include")
    message(STATUS "[QUTLASS] Using QuTLASS vendored CUTLASS headers (no vLLM CUTLASS detected).")
  else()
    message(FATAL_ERROR "[QUTLASS] CUTLASS headers not found. "
                        "Set -DCUTLASS_INCLUDE_DIR=/path/to/cutlass/include")
  endif()

  set_gencode_flags_for_srcs(
    SRCS "${QUTLASS_SOURCES}"
    CUDA_ARCHS "${QUTLASS_ARCHS}"
  )

  target_sources(_C PRIVATE ${QUTLASS_SOURCES})
  target_include_directories(_C PRIVATE ${QUTLASS_INCLUDES})
  target_compile_definitions(_C PRIVATE
    QUTLASS_DISABLE_PYBIND=1
    TARGET_CUDA_ARCH=${QUTLASS_TARGET_CC}
  )

  set_property(SOURCE ${QUTLASS_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math -O3>
  )

else()
  if("${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS "12.8")
    message(STATUS
      "[QUTLASS] Skipping build: CUDA 12.8 or newer is required (found ${CMAKE_CUDA_COMPILER_VERSION}).")
  else()
    message(STATUS
      "[QUTLASS] Skipping build: no supported arch (12.0a / 10.0a) found in "
      "CUDA_ARCHS='${CUDA_ARCHS}'.")
  endif()
endif()
