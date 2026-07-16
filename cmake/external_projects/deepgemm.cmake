include(FetchContent)

# If DEEPGEMM_SRC_DIR is set, DeepGEMM is built from that directory
# instead of downloading.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{DEEPGEMM_SRC_DIR})
  set(DEEPGEMM_SRC_DIR $ENV{DEEPGEMM_SRC_DIR})
endif()

# Local tree: set deepgemm_SOURCE_DIR directly (no FetchContent download).
# Upstream git: use FetchContent_Populate with explicit options (CMP0169 NEW
# disallows one-argument Populate(dep) after Declare; MakeAvailable would run
# DeepGEMM's top-level CMakeLists.txt, which vLLM must not load).
if(DEEPGEMM_SRC_DIR)
  # cmake_path(ABSOLUTE_PATH <var> ...) reads the path from <var>; NORMALIZE is a
  # flag (no trailing path argument). Resolve relative paths against vLLM root.
  set(_deepgemm_user_src "${DEEPGEMM_SRC_DIR}")
  cmake_path(ABSOLUTE_PATH _deepgemm_user_src
    BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
    NORMALIZE)
  set(DEEPGEMM_SRC_DIR "${_deepgemm_user_src}")
  if(NOT IS_DIRECTORY "${DEEPGEMM_SRC_DIR}")
    message(FATAL_ERROR
      "DEEPGEMM_SRC_DIR is not an existing directory: '${DEEPGEMM_SRC_DIR}'")
  endif()
  set(deepgemm_SOURCE_DIR "${DEEPGEMM_SRC_DIR}")
  message(STATUS "DeepGEMM using local DEEPGEMM_SRC_DIR: ${deepgemm_SOURCE_DIR}")
else()
  # Keep in sync with tools/install_deepgemm.sh
  set(_DEEPGEMM_UPSTREAM_REPO "https://github.com/cleonard530/DeepGEMM.git")
  # TORCH_LIBRARY migration (migrate_pybind_to_torch_library); see cleonard530/DeepGEMM#2
  set(_DEEPGEMM_UPSTREAM_TAG "5b266fb39a1c147793dea19d53d21bf8e8e3a256")

  set(_deepgemm_fc_root "${FETCHCONTENT_BASE_DIR}")
  if(NOT _deepgemm_fc_root)
    set(_deepgemm_fc_root "${CMAKE_BINARY_DIR}/_deps")
  endif()
  set(_deepgemm_src "${_deepgemm_fc_root}/deepgemm-src")
  set(_deepgemm_bin "${_deepgemm_fc_root}/deepgemm-build")
  set(_deepgemm_sub "${_deepgemm_fc_root}/deepgemm-subbuild")

  if(EXISTS "${_deepgemm_src}/deep_gemm/_C.py")
    set(deepgemm_SOURCE_DIR "${_deepgemm_src}")
    set(deepgemm_BINARY_DIR "${_deepgemm_bin}")
  else()
    FetchContent_Populate(
      deepgemm
      SUBBUILD_DIR "${_deepgemm_sub}"
      SOURCE_DIR "${_deepgemm_src}"
      BINARY_DIR "${_deepgemm_bin}"
      GIT_REPOSITORY "${_DEEPGEMM_UPSTREAM_REPO}"
      GIT_TAG "${_DEEPGEMM_UPSTREAM_TAG}"
      GIT_SUBMODULES "third-party/cutlass" "third-party/fmt"
      GIT_PROGRESS TRUE
    )
  endif()
  message(STATUS "DeepGEMM is available at ${deepgemm_SOURCE_DIR}")
endif()

# DeepGEMM requires CUDA 12.3+ for SM90, 12.9+ for SM100 (official upstream),
# and 12.8+ for SM120 / SM12x. CUDA 13+ can use the family-specific SM12x
# arch; CUDA 12.x builds the arch-specific SM120/SM121 variants.
set(DEEPGEMM_SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3)
  list(APPEND DEEPGEMM_SUPPORT_ARCHS "9.0a")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
    list(APPEND DEEPGEMM_SUPPORT_ARCHS "10.0f")
  else()
    list(APPEND DEEPGEMM_SUPPORT_ARCHS "10.0a")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    list(APPEND DEEPGEMM_SUPPORT_ARCHS "12.0f")
  else()
    list(APPEND DEEPGEMM_SUPPORT_ARCHS "12.0a" "12.1a")
  endif()
endif()

cuda_archs_loose_intersection(DEEPGEMM_ARCHS
  "${DEEPGEMM_SUPPORT_ARCHS}" "${CUDA_ARCHS}")

if(DEEPGEMM_ARCHS)
  message(STATUS "DeepGEMM CUDA architectures: ${DEEPGEMM_ARCHS}")

  #
  # DeepGEMM integration notes
  # --------------------------
  # We vendor DeepGEMM into vllm/third_party/deep_gemm/ and bundle:
  #   - deep_gemm/_C.py (Python shim over torch.ops.deep_gemm)
  #   - deep_gemm/_C_extension.abi3.so (single limited-API extension)
  # The build is delegated to tools/build_deepgemm_C.py (setup.py build_ext).
  #
  # TODO: AOT-compile DeepGEMM's CUDA kernels instead of runtime JIT to drop
  # the vendored CUTLASS/CCCL headers and the CUDA-toolkit-at-runtime
  # requirement.
  #

  message(STATUS "DeepGEMM extension will be built with: ${Python_EXECUTABLE}")

  # add_custom_command does no implicit header scanning; glob explicitly so
  # header-only edits in DeepGEMM/cutlass/fmt re-trigger the rebuild.
  file(GLOB_RECURSE _dg_headers
    "${deepgemm_SOURCE_DIR}/csrc/*.h"
    "${deepgemm_SOURCE_DIR}/csrc/*.hpp"
    "${deepgemm_SOURCE_DIR}/deep_gemm/include/*.h"
    "${deepgemm_SOURCE_DIR}/deep_gemm/include/*.hpp"
    "${deepgemm_SOURCE_DIR}/deep_gemm/include/*.cuh")

  set(_dg_dir "${CMAKE_CURRENT_BINARY_DIR}/deepgemm_C")
  set(_dg_marker "${_dg_dir}/.built")
  add_custom_command(
    OUTPUT "${_dg_marker}"
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tools/build_deepgemm_C.py"
            "${deepgemm_SOURCE_DIR}" "${_dg_dir}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_dg_marker}"
    DEPENDS "${CMAKE_SOURCE_DIR}/tools/build_deepgemm_C.py"
            "${deepgemm_SOURCE_DIR}/csrc/python_api.cpp"
            "${deepgemm_SOURCE_DIR}/deep_gemm/_C.py"
            "${deepgemm_SOURCE_DIR}/setup.py"
            ${_dg_headers}
    COMMENT "Building DeepGEMM _C_extension (abi3)"
    VERBATIM)
  add_custom_target(_deep_gemm_C ALL DEPENDS "${_dg_marker}")

  install(DIRECTORY "${_dg_dir}/"
    DESTINATION vllm/third_party/deep_gemm
    COMPONENT _deep_gemm_C
    FILES_MATCHING
      PATTERN "_C.py"
      PATTERN "_C_extension*.so")

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
