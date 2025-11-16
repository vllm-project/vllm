# Install OpenAI triton_kernels from https://github.com/triton-lang/triton/tree/main/python/triton_kernels

set(DEFAULT_TRITON_KERNELS_TAG "v3.5.0")

# Set TRITON_KERNELS_SRC_DIR for use with local development with vLLM.
if (DEFINED ENV{TRITON_KERNELS_SRC_DIR})
  set(TRITON_KERNELS_SRC_DIR $ENV{TRITON_KERNELS_SRC_DIR})
endif()

if(TRITON_KERNELS_SRC_DIR)
  FetchContent_Declare(
          triton_kernels
          SOURCE_DIR ${TRITON_KERNELS_SRC_DIR}
  )
else()
  FetchContent_Declare(
          triton_kernels
          # TODO (varun) : Fetch just the triton_kernels directory from Triton
          GIT_REPOSITORY https://github.com/triton-lang/triton.git
          GIT_TAG ${DEFAULT_TRITON_KERNELS_TAG}
          GIT_PROGRESS TRUE
  )
endif()

# Fetch content 
FetchContent_Populate(triton_kernels)

if (NOT triton_kernels_SOURCE_DIR)
  message (FATAL_ERROR "[triton_kernels] Cannot resolve triton_kernels_SOURCE_DIR")
endif()
  
if (TRITON_KERNELS_SRC_DIR)
  # When passed in explicitly we expect TRITON_KERNELS_SRC_DIR to point to the python directory directly.
  set(TRITON_KERNELS_PYTHON_DIR "${triton_kernels_SOURCE_DIR}")
else()
  # Triton kernels lives in the python directory
  set(TRITON_KERNELS_PYTHON_DIR "${triton_kernels_SOURCE_DIR}/python/triton_kernels/triton_kernels/")
endif()

add_custom_target(triton_kernels)

## Copy .py files to install directory.
install(DIRECTORY
        ${TRITON_KERNELS_PYTHON_DIR}
        DESTINATION 
        vllm/third_party/
        COMPONENT triton_kernels
        FILES_MATCHING PATTERN "*.py")
