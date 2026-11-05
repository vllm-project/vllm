# Install OpenAI triton_kernels from https://github.com/triton-lang/triton/tree/main/python/triton_kernels

# Set TRITON_KERNELS_SRC_DIR for use with local development with vLLM. We expect TRITON_KERNELS_SRC_DIR to
# be directly set to the triton_kernels python directory.
if (DEFINED ENV{TRITON_KERNELS_SRC_DIR})
  message(STATUS "[triton_kernels] Fetch from $ENV{TRITON_KERNELS_SRC_DIR}")
  FetchContent_Declare(
          triton_kernels
          SOURCE_DIR $ENV{TRITON_KERNELS_SRC_DIR}
  )

else()
  if (VLLM_TARGET_DEVICE STREQUAL "rocm")
    set(TRITON_GIT "https://github.com/ROCm/triton.git")
    # Pinned from release/internal/3.8.x
    set(TRITON_KERNELS_TAG "669b31acc1dd1b3fd93286afd5db67f65d9f7557")
  else()
    set(TRITON_GIT "https://github.com/triton-lang/triton.git")
    set(TRITON_KERNELS_TAG "v3.5.1")
  endif()
  # Fetch the source archive of the pinned revision instead of git-cloning the
  # whole Triton repo: FetchContent cannot shallow-clone a raw commit sha, so a
  # clone pulls ~406 MiB of history (443 MB on disk) to install a few .py files,
  # while this archive is a 7.7 MB download. docker/Dockerfile.rocm prefetches
  # the same archive for TRITON_KERNELS_SRC_DIR, so Docker and non-Docker builds
  # get identical sources; keep the URL form in sync with that Dockerfile.
  string(REGEX REPLACE "\\.git$" "" TRITON_REPO_URL "${TRITON_GIT}")
  string(REPLACE "https://github.com/" "https://codeload.github.com/"
         TRITON_REPO_URL "${TRITON_REPO_URL}")
  set(TRITON_KERNELS_URL "${TRITON_REPO_URL}/tar.gz/${TRITON_KERNELS_TAG}")
  message (STATUS "[triton_kernels] Fetch from ${TRITON_KERNELS_URL}")
  FetchContent_Declare(
          triton_kernels
          URL ${TRITON_KERNELS_URL}
          DOWNLOAD_EXTRACT_TIMESTAMP TRUE
          SOURCE_SUBDIR python/triton_kernels/triton_kernels
  )
endif()

# Fetch content
FetchContent_MakeAvailable(triton_kernels)

if (NOT triton_kernels_SOURCE_DIR)
  message (FATAL_ERROR "[triton_kernels] Cannot resolve triton_kernels_SOURCE_DIR")
endif()

if (DEFINED ENV{TRITON_KERNELS_SRC_DIR})
  set(TRITON_KERNELS_PYTHON_DIR "${triton_kernels_SOURCE_DIR}/")
else()
  set(TRITON_KERNELS_PYTHON_DIR "${triton_kernels_SOURCE_DIR}/python/triton_kernels/triton_kernels/")
endif()

message (STATUS "[triton_kernels] triton_kernels is available at ${TRITON_KERNELS_PYTHON_DIR}")

add_custom_target(triton_kernels)

# Ensure the vllm/third_party directory exists before installation
install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/third_party/triton_kernels\")")

## Copy .py files to install directory.
install(DIRECTORY
        ${TRITON_KERNELS_PYTHON_DIR}
        DESTINATION
        vllm/third_party/triton_kernels/
        COMPONENT triton_kernels
        FILES_MATCHING PATTERN "*.py")
