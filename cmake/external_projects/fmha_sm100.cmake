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
    GIT_TAG 544eee5e09ae2dfa774d5b06739013f9b7402c57
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

add_custom_target(fmha_sm100)

install(FILES
  "${fmha_sm100_SOURCE_DIR}/python/fmha_sm100/__init__.py"
  "${fmha_sm100_SOURCE_DIR}/python/fmha_sm100/sparse.py"
  DESTINATION vllm/third_party/fmha_sm100
  COMPONENT fmha_sm100)

install(DIRECTORY "${fmha_sm100_SOURCE_DIR}/python/fmha_sm100/cute/"
  DESTINATION vllm/third_party/fmha_sm100/cute
  COMPONENT fmha_sm100
  FILES_MATCHING
    REGEX "/__pycache__(/.*)?$" EXCLUDE
    REGEX ".*\\.pyc$" EXCLUDE
    PATTERN "example.py" EXCLUDE
    PATTERN "test_*.py" EXCLUDE
    PATTERN "*.py"
    PATTERN "build_k2q_csr.cu")
