include(FetchContent)

# If DEEPGEMM_SRC_DIR is set, DeepGEMM is built from that directory
# instead of downloading.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{DEEPGEMM_SRC_DIR})
  set(DEEPGEMM_SRC_DIR $ENV{DEEPGEMM_SRC_DIR})
endif()

if(DEEPGEMM_SRC_DIR)
  FetchContent_Declare(
    deepgemm
    SOURCE_DIR ${DEEPGEMM_SRC_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
else()
  # This ref should be kept in sync with tools/install_deepgemm.sh
  FetchContent_Declare(
    deepgemm
    GIT_REPOSITORY https://github.com/deepseek-ai/DeepGEMM.git
    GIT_TAG 891d57b4db1071624b5c8fa0d1e51cb317fa709f
    GIT_SUBMODULES "third-party/cutlass" "third-party/fmt"
    GIT_PROGRESS TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
endif()

# Use FetchContent_Populate (not MakeAvailable) to avoid processing
# DeepGEMM's own CMakeLists.txt which has incompatible find_package calls.
FetchContent_GetProperties(deepgemm)
if(NOT deepgemm_POPULATED)
  FetchContent_Populate(deepgemm)
endif()
message(STATUS "DeepGEMM is available at ${deepgemm_SOURCE_DIR}")

# DeepGEMM requires CUDA 12.3+ for SM90, 12.9+ for SM100
set(DEEPGEMM_SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3)
  list(APPEND DEEPGEMM_SUPPORT_ARCHS "9.0a")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
  list(APPEND DEEPGEMM_SUPPORT_ARCHS "10.0f")
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
  list(APPEND DEEPGEMM_SUPPORT_ARCHS "10.0a")
endif()

cuda_archs_loose_intersection(DEEPGEMM_ARCHS
  "${DEEPGEMM_SUPPORT_ARCHS}" "${CUDA_ARCHS}")

if(DEEPGEMM_ARCHS)
  message(STATUS "DeepGEMM CUDA architectures: ${DEEPGEMM_ARCHS}")

  # Build _C once per interpreter in DEEPGEMM_PYTHON_INTERPRETERS (":"-
  # separated paths) so the wheel imports cleanly on every supported Python.
  # Unset → fall back to the build interpreter (editable / source builds).
  # The compile is delegated to tools/build_deepgemm_C.py and always runs
  # against the build interpreter's torch — target Pythons don't need torch.
  # Note: empty-but-set env vars are still DEFINED in cmake; treat empty as
  # unset so an empty interpreter list falls back to the build interpreter
  # rather than silently skipping the per-Python build.
  if(NOT "$ENV{DEEPGEMM_PYTHON_INTERPRETERS}" STREQUAL "")
    string(REPLACE ":" ";" _dg_pythons "$ENV{DEEPGEMM_PYTHON_INTERPRETERS}")
  else()
    set(_dg_pythons "${Python_EXECUTABLE}")
  endif()
  message(STATUS "DeepGEMM _C will be built for: ${_dg_pythons}")

  # Header set fed to add_custom_command's DEPENDS so a header-only edit
  # (in upstream DeepGEMM or its vendored cutlass/fmt) re-triggers the
  # rebuild. add_custom_command does no implicit header scanning, unlike
  # add_library.
  file(GLOB_RECURSE _dg_headers
    "${deepgemm_SOURCE_DIR}/csrc/*.h"
    "${deepgemm_SOURCE_DIR}/csrc/*.hpp"
    "${deepgemm_SOURCE_DIR}/deep_gemm/include/*.h"
    "${deepgemm_SOURCE_DIR}/deep_gemm/include/*.hpp"
    "${deepgemm_SOURCE_DIR}/deep_gemm/include/*.cuh")

  set(_dg_markers)
  set(_dg_seen_soabis)
  foreach(_pybin IN LISTS _dg_pythons)
    execute_process(
      COMMAND "${_pybin}" -c
        "import sysconfig; print(sysconfig.get_config_var('SOABI'))"
      OUTPUT_VARIABLE _dg_soabi
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
    # Dedup so duplicate paths (or two paths resolving to the same CPython)
    # don't register conflicting build rules.
    if(_dg_soabi IN_LIST _dg_seen_soabis)
      continue()
    endif()
    list(APPEND _dg_seen_soabis "${_dg_soabi}")
    set(_dg_dir "${CMAKE_CURRENT_BINARY_DIR}/deepgemm_C_${_dg_soabi}")
    set(_dg_marker "${_dg_dir}/.built")
    add_custom_command(
      OUTPUT "${_dg_marker}"
      COMMAND "${Python_EXECUTABLE}"
              "${CMAKE_SOURCE_DIR}/tools/build_deepgemm_C.py"
              "${deepgemm_SOURCE_DIR}" "${_dg_dir}" "${_pybin}"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_dg_marker}"
      DEPENDS "${CMAKE_SOURCE_DIR}/tools/build_deepgemm_C.py"
              "${deepgemm_SOURCE_DIR}/csrc/python_api.cpp"
              ${_dg_headers}
      COMMENT "Building DeepGEMM _C for ${_pybin}"
      VERBATIM)
    list(APPEND _dg_markers "${_dg_marker}")
    install(DIRECTORY "${_dg_dir}/"
      DESTINATION vllm/third_party/deep_gemm
      COMPONENT _deep_gemm_C
      FILES_MATCHING PATTERN "_C.cpython-*.so")
  endforeach()
  add_custom_target(_deep_gemm_C ALL DEPENDS ${_dg_markers})

  #
  # Vendor DeepGEMM Python package files
  #
  install(FILES
    "${deepgemm_SOURCE_DIR}/deep_gemm/__init__.py"
    DESTINATION vllm/third_party/deep_gemm
    COMPONENT _deep_gemm_C)

  install(DIRECTORY "${deepgemm_SOURCE_DIR}/deep_gemm/utils/"
    DESTINATION vllm/third_party/deep_gemm/utils
    COMPONENT _deep_gemm_C
    FILES_MATCHING PATTERN "*.py")

  install(DIRECTORY "${deepgemm_SOURCE_DIR}/deep_gemm/testing/"
    DESTINATION vllm/third_party/deep_gemm/testing
    COMPONENT _deep_gemm_C
    FILES_MATCHING PATTERN "*.py")

  install(DIRECTORY "${deepgemm_SOURCE_DIR}/deep_gemm/legacy/"
    DESTINATION vllm/third_party/deep_gemm/legacy
    COMPONENT _deep_gemm_C
    FILES_MATCHING PATTERN "*.py")

  install(DIRECTORY "${deepgemm_SOURCE_DIR}/deep_gemm/mega/"
    DESTINATION vllm/third_party/deep_gemm/mega
    COMPONENT _deep_gemm_C
    FILES_MATCHING PATTERN "*.py")

  # Generate envs.py (normally generated by DeepGEMM's setup.py build step)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/deep_gemm_envs.py"
    "# Pre-installed environment variables\npersistent_envs = dict()\n")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/deep_gemm_envs.py"
    DESTINATION vllm/third_party/deep_gemm
    RENAME envs.py
    COMPONENT _deep_gemm_C)

  #
  # Install include files needed for JIT compilation at runtime.
  # The JIT compiler finds these relative to the package directory.
  #

  # DeepGEMM's own CUDA headers
  install(DIRECTORY "${deepgemm_SOURCE_DIR}/deep_gemm/include/"
    DESTINATION vllm/third_party/deep_gemm/include
    COMPONENT _deep_gemm_C)

  # CUTLASS and CuTe headers (vendored for JIT, separate from vLLM's CUTLASS)
  install(DIRECTORY "${deepgemm_SOURCE_DIR}/third-party/cutlass/include/"
    DESTINATION vllm/third_party/deep_gemm/include
    COMPONENT _deep_gemm_C)

else()
  message(STATUS "DeepGEMM will not compile: "
    "unsupported CUDA architecture ${CUDA_ARCHS}")
  # Create empty target so setup.py doesn't fail on unsupported systems
  add_custom_target(_deep_gemm_C)
endif()
