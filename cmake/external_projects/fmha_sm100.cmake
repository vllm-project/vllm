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
    GIT_TAG 890aaa1a37a598ad17ccff0827fea21540d381fa
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

set(FMHA_SM100_PY_ROOT "${fmha_sm100_SOURCE_DIR}/python/fmha_sm100")

install(FILES
  "${FMHA_SM100_PY_ROOT}/__init__.py"
  "${FMHA_SM100_PY_ROOT}/api.py"
  "${FMHA_SM100_PY_ROOT}/bench_utils.py"
  "${FMHA_SM100_PY_ROOT}/jit.py"
  "${FMHA_SM100_PY_ROOT}/sparse.py"
  "${FMHA_SM100_PY_ROOT}/sparse_fmha_adapter.py"
  DESTINATION vllm/third_party/fmha_sm100
  COMPONENT fmha_sm100)

install(DIRECTORY "${FMHA_SM100_PY_ROOT}/csrc/"
  DESTINATION vllm/third_party/fmha_sm100/csrc
  COMPONENT fmha_sm100
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".git*" EXCLUDE)

install(DIRECTORY "${FMHA_SM100_PY_ROOT}/cute/"
  DESTINATION vllm/third_party/fmha_sm100/cute
  COMPONENT fmha_sm100
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".git*" EXCLUDE)

install(DIRECTORY "${FMHA_SM100_PY_ROOT}/cutlass/include/"
  DESTINATION vllm/third_party/fmha_sm100/cutlass/include
  COMPONENT fmha_sm100
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".git*" EXCLUDE)

install(DIRECTORY "${FMHA_SM100_PY_ROOT}/cutlass/tools/util/include/"
  DESTINATION vllm/third_party/fmha_sm100/cutlass/tools/util/include
  COMPONENT fmha_sm100
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".git*" EXCLUDE)
