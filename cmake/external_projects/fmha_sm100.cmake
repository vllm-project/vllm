include(FetchContent)

# If FMHA_SM100_SRC_DIR is set, fmha_sm100 is installed from that directory
# instead of downloading. This is useful for local MSA development.
if(DEFINED ENV{FMHA_SM100_SRC_DIR})
  set(FMHA_SM100_SRC_DIR $ENV{FMHA_SM100_SRC_DIR})
endif()

if(FMHA_SM100_SRC_DIR)
  FetchContent_Declare(
    fmha_sm100
    SOURCE_DIR ${FMHA_SM100_SRC_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
else()
  FetchContent_Declare(
    fmha_sm100
    GIT_REPOSITORY https://github.com/vllm-project/MSA.git
    GIT_TAG e32b23be48ccdbab952ff68db9fa31dc12733084
    GIT_SUBMODULES python/fmha_sm100/cutlass
    GIT_SUBMODULES_RECURSE TRUE
    GIT_PROGRESS TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
endif()

FetchContent_GetProperties(fmha_sm100)
if(NOT fmha_sm100_POPULATED)
  FetchContent_Populate(fmha_sm100)
endif()
message(STATUS "fmha_sm100 is available at ${fmha_sm100_SOURCE_DIR}")

if(NOT EXISTS "${fmha_sm100_SOURCE_DIR}/python/fmha_sm100/cutlass/include")
  message(FATAL_ERROR
    "fmha_sm100 CUTLASS submodule is missing. "
    "If using FMHA_SM100_SRC_DIR, run "
    "`git submodule update --init --recursive` in that MSA checkout.")
endif()

add_custom_target(fmha_sm100)

install(DIRECTORY "${fmha_sm100_SOURCE_DIR}/python/fmha_sm100/"
  DESTINATION vllm/third_party/fmha_sm100
  COMPONENT fmha_sm100
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.cu"
    PATTERN "*.cuh"
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.inl"
    PATTERN "*.jinja"
    PATTERN "VERSION"
    PATTERN "*.md"
    PATTERN "*.ini"
    PATTERN "*.txt"
    PATTERN "Makefile"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE)
