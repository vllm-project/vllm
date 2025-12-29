# Install OpenAI triton_kernels from https://github.com/triton-lang/triton/tree/main/python/triton_kernels

set(DEFAULT_TRITON_KERNELS_TAG "v3.5.0")

# Set TRITON_KERNELS_SRC_DIR for use with local development with vLLM. We expect TRITON_KERNELS_SRC_DIR to
# be directly set to the triton_kernels python directory. 
if (DEFINED ENV{TRITON_KERNELS_SRC_DIR})
  message(STATUS "[triton_kernels] Fetch from $ENV{TRITON_KERNELS_SRC_DIR}")
  FetchContent_Declare(
          triton_kernels
          SOURCE_DIR $ENV{TRITON_KERNELS_SRC_DIR}
  )

else()
  set(TRITON_GIT "https://github.com/triton-lang/triton.git")
  message (STATUS "[triton_kernels] Fetch from ${TRITON_GIT}:${DEFAULT_TRITON_KERNELS_TAG}")
  FetchContent_Declare(
          triton_kernels
          # TODO (varun) : Fetch just the triton_kernels directory from Triton
          GIT_REPOSITORY https://github.com/triton-lang/triton.git
          GIT_TAG ${DEFAULT_TRITON_KERNELS_TAG}
          GIT_PROGRESS TRUE
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
