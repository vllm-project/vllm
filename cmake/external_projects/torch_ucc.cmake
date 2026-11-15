# PyTorch's ProcessGroupUCC sources, for the optional _ucc_C extension.
#
# PyPI torch is built without UCC, so ProcessGroupUCC is not in libtorch. Its
# sources are fetched unmodified from the PyTorch commit the installed torch
# was built from (torch.version.git_version) and compiled into vllm._ucc_C.
#
# If TORCH_UCC_SRC_DIR is set (environment variable or cmake argument), the
# sources are taken from that PyTorch checkout instead of being downloaded.
# The environment variable takes precedence.

if(DEFINED ENV{TORCH_UCC_SRC_DIR})
  set(TORCH_UCC_SRC_DIR $ENV{TORCH_UCC_SRC_DIR})
endif()

set(_TORCH_UCC_FILES
  torch/csrc/distributed/c10d/ProcessGroupUCC.cpp
  torch/csrc/distributed/c10d/ProcessGroupUCC.hpp
  torch/csrc/distributed/c10d/ucc/UCCTracing.cpp
  torch/csrc/distributed/c10d/ucc/UCCTracing.hpp
  torch/csrc/distributed/c10d/ucc/UCCUtils.cpp
  torch/csrc/distributed/c10d/ucc/UCCUtils.hpp)

if(TORCH_UCC_SRC_DIR)
  set(TORCH_UCC_ROOT "${TORCH_UCC_SRC_DIR}")
else()
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
      "import torch; print(torch.version.git_version)"
    OUTPUT_VARIABLE _TORCH_GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _TORCH_GIT_VERSION_RESULT)
  if(NOT _TORCH_GIT_VERSION_RESULT EQUAL 0
     OR NOT _TORCH_GIT_VERSION MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Cannot read torch.version.git_version "
      "('${_TORCH_GIT_VERSION}'); set TORCH_UCC_SRC_DIR to a PyTorch checkout "
      "matching the installed torch.")
  endif()
  set(TORCH_UCC_ROOT "${CMAKE_BINARY_DIR}/torch_ucc-${_TORCH_GIT_VERSION}")
  foreach(_f IN LISTS _TORCH_UCC_FILES)
    if(NOT EXISTS "${TORCH_UCC_ROOT}/${_f}")
      file(DOWNLOAD
        "https://raw.githubusercontent.com/pytorch/pytorch/${_TORCH_GIT_VERSION}/${_f}"
        "${TORCH_UCC_ROOT}/${_f}.tmp"
        STATUS _status)
      list(GET _status 0 _code)
      if(NOT _code EQUAL 0)
        message(FATAL_ERROR "Downloading ${_f} at PyTorch "
          "${_TORCH_GIT_VERSION} failed: ${_status}. Set TORCH_UCC_SRC_DIR to "
          "a PyTorch checkout to build offline.")
      endif()
      file(RENAME "${TORCH_UCC_ROOT}/${_f}.tmp" "${TORCH_UCC_ROOT}/${_f}")
    endif()
  endforeach()
endif()

foreach(_f IN LISTS _TORCH_UCC_FILES)
  if(NOT EXISTS "${TORCH_UCC_ROOT}/${_f}")
    message(FATAL_ERROR "${TORCH_UCC_ROOT}/${_f} not found")
  endif()
endforeach()
message(STATUS "ProcessGroupUCC sources at ${TORCH_UCC_ROOT}")

set(TORCH_UCC_SOURCES
  "${TORCH_UCC_ROOT}/torch/csrc/distributed/c10d/ProcessGroupUCC.cpp"
  "${TORCH_UCC_ROOT}/torch/csrc/distributed/c10d/ucc/UCCTracing.cpp"
  "${TORCH_UCC_ROOT}/torch/csrc/distributed/c10d/ucc/UCCUtils.cpp")

# ProcessGroupUCC.cpp includes ATen/cuda/nvrtc_stub/ATenNVRTC.h, which needs
# the CUDA toolkit headers (and is missing from CPU torch), but uses NVRTC only
# under USE_CUDA. _ucc_C is CPU-only, so an empty stand-in shadows it.
set(_stub_dir "${CMAKE_BINARY_DIR}/torch_ucc_stub")
file(WRITE "${_stub_dir}/ATen/cuda/nvrtc_stub/ATenNVRTC.h"
  "// Empty: _ucc_C is built without USE_CUDA and does not use NVRTC.\n")
set(TORCH_UCC_INCLUDE_DIRS "${TORCH_UCC_ROOT}" "${_stub_dir}")
