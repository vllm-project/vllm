include(FetchContent)

# FlashTopK (SM90 fused MQA scoring + exact top-k, CuTe DSL / Triton): its
# Python package is installed into vllm/third_party/flashtopk. If
# FLASHTOPK_SRC_DIR is set, it is installed from that directory instead.
if(DEFINED ENV{FLASHTOPK_SRC_DIR})
  set(FLASHTOPK_SRC_DIR $ENV{FLASHTOPK_SRC_DIR})
endif()

if(FLASHTOPK_SRC_DIR)
  FetchContent_Declare(
    flashtopk
    SOURCE_DIR ${FLASHTOPK_SRC_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND "")
else()
  FetchContent_Declare(
    flashtopk
    GIT_REPOSITORY https://github.com/Tencent-Hunyuan/FlashTopK.git
    GIT_TAG 122b661cf3e48b9f12a66f484ed7049ff49b8779
    GIT_PROGRESS TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND "")
endif()

FetchContent_GetProperties(flashtopk)
if(NOT flashtopk_POPULATED)
  FetchContent_Populate(flashtopk)
endif()
message(STATUS "FlashTopK is available at ${flashtopk_SOURCE_DIR}")

add_custom_target(flashtopk)

# The package's imports are relative, so it installs unchanged.
install(DIRECTORY "${flashtopk_SOURCE_DIR}/flashtopk/"
  DESTINATION vllm/third_party/flashtopk
  COMPONENT flashtopk
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE)

install(FILES
  "${flashtopk_SOURCE_DIR}/LICENSE"
  "${flashtopk_SOURCE_DIR}/NOTICE"
  DESTINATION vllm/third_party/flashtopk
  COMPONENT flashtopk)
