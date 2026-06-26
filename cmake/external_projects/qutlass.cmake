include(FetchContent)

set(CUTLASS_INCLUDE_DIR "${CUTLASS_INCLUDE_DIR}" CACHE PATH "Path to CUTLASS include/ directory")

if(DEFINED ENV{QUTLASS_SRC_DIR})
  set(QUTLASS_SRC_DIR $ENV{QUTLASS_SRC_DIR})
endif()

# CMP0169 NEW: one-argument FetchContent_Populate(name) after Declare is invalid.
# Use explicit Populate(...) for git, or set SOURCE_DIR for local trees.
if(QUTLASS_SRC_DIR)
  set(_qutlass_user_src "${QUTLASS_SRC_DIR}")
  cmake_path(ABSOLUTE_PATH _qutlass_user_src
    BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
    NORMALIZE)
  set(QUTLASS_SRC_DIR "${_qutlass_user_src}")
  if(NOT IS_DIRECTORY "${QUTLASS_SRC_DIR}")
    message(FATAL_ERROR
      "[QUTLASS] QUTLASS_SRC_DIR is not an existing directory: '${QUTLASS_SRC_DIR}'")
  endif()
  set(qutlass_SOURCE_DIR "${QUTLASS_SRC_DIR}")
  set(qutlass_BINARY_DIR "${CMAKE_BINARY_DIR}/qutlass-binary-dir-unused")
else()
  set(_QUTLASS_UPSTREAM_REPO "https://github.com/IST-DASLab/qutlass.git")
  set(_QUTLASS_UPSTREAM_TAG "830d2c4537c7396e14a02a46fbddd18b5d107c65")

  set(_qutlass_fc_root "${FETCHCONTENT_BASE_DIR}")
  if(NOT _qutlass_fc_root)
    set(_qutlass_fc_root "${CMAKE_BINARY_DIR}/_deps")
  endif()
  set(_qutlass_src "${_qutlass_fc_root}/qutlass-src")
  set(_qutlass_bin "${_qutlass_fc_root}/qutlass-build")
  set(_qutlass_sub "${_qutlass_fc_root}/qutlass-subbuild")

  if(EXISTS "${_qutlass_src}/qutlass/csrc/bindings.cpp")
    set(qutlass_SOURCE_DIR "${_qutlass_src}")
    set(qutlass_BINARY_DIR "${_qutlass_bin}")
  else()
    FetchContent_Populate(
      qutlass
      SUBBUILD_DIR "${_qutlass_sub}"
      SOURCE_DIR "${_qutlass_src}"
      BINARY_DIR "${_qutlass_bin}"
      GIT_REPOSITORY "${_QUTLASS_UPSTREAM_REPO}"
      GIT_TAG "${_QUTLASS_UPSTREAM_TAG}"
      GIT_PROGRESS TRUE
    )
  endif()
endif()

if(NOT qutlass_SOURCE_DIR)
  message(FATAL_ERROR "[QUTLASS] source directory could not be resolved.")
endif()
message(STATUS "[QUTLASS] QuTLASS is available at ${qutlass_SOURCE_DIR}")

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(QUTLASS_SM120_ARCHS "12.0f" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(QUTLASS_SM100_ARCHS "10.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(QUTLASS_SM120_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(QUTLASS_SM100_ARCHS "10.0a;10.3a" "${CUDA_ARCHS}")
endif()

# QUTLASS uses TARGET_CUDA_ARCH as a single preprocessor selector for all its
# sources. Do not compile a mixed SM100/SM120 arch list with one selector; prefer
# SM100 when both families are requested because that is the primary deployed
# target for this extension today.
if(QUTLASS_SM100_ARCHS)
  set(QUTLASS_ARCHS "${QUTLASS_SM100_ARCHS}")
  set(QUTLASS_TARGET_CC 100)
  if(QUTLASS_SM120_ARCHS)
    message(WARNING
      "[QUTLASS] Both SM100 and SM120 archs were requested; selecting SM100 "
      "because TARGET_CUDA_ARCH is a single compile-time selector.")
  endif()
elseif(QUTLASS_SM120_ARCHS)
  set(QUTLASS_ARCHS "${QUTLASS_SM120_ARCHS}")
  set(QUTLASS_TARGET_CC 120)
else()
  set(QUTLASS_ARCHS)
endif()

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND QUTLASS_ARCHS)
  set(QUTLASS_SOURCES
    csrc/qutlass_registration.cpp
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
    if(CUTLASS_TOOLS_UTIL_INCLUDE_DIR AND
       EXISTS "${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}/cutlass/util/packed_stride.hpp")
      list(APPEND QUTLASS_INCLUDES "${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}")
    else()
      get_filename_component(_qutlass_cutlass_root "${CUTLASS_INCLUDE_DIR}" DIRECTORY)
      if(EXISTS "${_qutlass_cutlass_root}/tools/util/include/cutlass/util/packed_stride.hpp")
        list(APPEND QUTLASS_INCLUDES "${_qutlass_cutlass_root}/tools/util/include")
      endif()
    endif()
  elseif(EXISTS "${qutlass_SOURCE_DIR}/qutlass/third_party/cutlass/include/cutlass/cutlass.h")
    list(APPEND QUTLASS_INCLUDES
      "${qutlass_SOURCE_DIR}/qutlass/third_party/cutlass/include"
      "${qutlass_SOURCE_DIR}/qutlass/third_party/cutlass/tools/util/include")
    message(STATUS "[QUTLASS] Using QuTLASS vendored CUTLASS headers (no vLLM CUTLASS detected).")
  else()
    message(FATAL_ERROR "[QUTLASS] CUTLASS headers not found. "
                        "Set -DCUTLASS_INCLUDE_DIR=/path/to/cutlass/include")
  endif()

  set_gencode_flags_for_srcs(
    SRCS "${QUTLASS_SOURCES}"
    CUDA_ARCHS "${QUTLASS_ARCHS}"
  )

  # QuTLASS uses legacy ATen headers and cannot be built with TORCH_TARGET_VERSION.
  # Keep it as its own extension (registers torch.ops._qutlass_C).
  define_extension_target(
    _qutlass_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${QUTLASS_SOURCES}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${QUTLASS_INCLUDES}
    USE_SABI 3
    WITH_SOABI)

  target_compile_definitions(_qutlass_C PRIVATE
    QUTLASS_DISABLE_PYBIND=1
    TARGET_CUDA_ARCH=${QUTLASS_TARGET_CC}
    CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

  set_property(SOURCE ${QUTLASS_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math -O3>
  )

else()
  if("${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS "12.8")
    message(STATUS
      "[QUTLASS] Skipping build: CUDA 12.8 or newer is required (found ${CMAKE_CUDA_COMPILER_VERSION}).")
  else()
    message(STATUS
      "[QUTLASS] Skipping build: no supported arch (12.0f / 10.0f) found in "
      "CUDA_ARCHS='${CUDA_ARCHS}'.")
  endif()
  add_custom_target(_qutlass_C)
endif()
