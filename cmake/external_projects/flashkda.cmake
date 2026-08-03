include(FetchContent)

if(DEFINED ENV{FLASH_KDA_SRC_DIR})
  set(FLASH_KDA_SRC_DIR $ENV{FLASH_KDA_SRC_DIR})
endif()

if(FLASH_KDA_SRC_DIR)
  FetchContent_Declare(
    flashkda
    SOURCE_DIR ${FLASH_KDA_SRC_DIR}
  )
else()
  FetchContent_Declare(
    flashkda
    GIT_REPOSITORY https://github.com/vllm-project/FlashKDA.git
    GIT_TAG a3e42bbbece3bb38f7c426b880315294a336e82f
    GIT_PROGRESS TRUE
    GIT_SUBMODULES cutlass
  )
endif()

FetchContent_MakeAvailable(flashkda)
message(STATUS "FlashKDA is available at ${flashkda_SOURCE_DIR}")

set(FLASH_KDA_SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0)
  list(APPEND FLASH_KDA_SUPPORT_ARCHS "9.0a")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  list(APPEND FLASH_KDA_SUPPORT_ARCHS "10.0f" "12.0f")
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
  list(APPEND FLASH_KDA_SUPPORT_ARCHS "10.0a" "10.3a" "12.0a")
endif()

cuda_archs_loose_intersection(
  FLASH_KDA_ARCHS "${FLASH_KDA_SUPPORT_ARCHS}" "${CUDA_ARCHS}")

if(FLASH_KDA_ARCHS)
  message(STATUS "FlashKDA CUDA architectures: ${FLASH_KDA_ARCHS}")

  set(FLASH_KDA_SOURCES
    csrc/flashkda_registration.cpp
    ${flashkda_SOURCE_DIR}/csrc/flash_kda.cpp
    ${flashkda_SOURCE_DIR}/csrc/smxx/fwd_launch.cu)
  set(FLASH_KDA_INCLUDES
    ${flashkda_SOURCE_DIR}/csrc
    ${flashkda_SOURCE_DIR}/cutlass/include
    ${flashkda_SOURCE_DIR}/cutlass/examples/common
    ${flashkda_SOURCE_DIR}/cutlass/tools/util/include)

  set_gencode_flags_for_srcs(
    SRCS "${FLASH_KDA_SOURCES}"
    CUDA_ARCHS "${FLASH_KDA_ARCHS}")

  define_extension_target(
    _flashkda_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${FLASH_KDA_SOURCES}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${FLASH_KDA_INCLUDES}
    USE_SABI 3
    WITH_SOABI)

  target_compile_options(_flashkda_C PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -O3>
    $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>)
else()
  message(STATUS
    "FlashKDA will not compile: CUDA >=12.0 and a supported architecture "
    "(SM90, SM10x, or SM12x) are required")
  add_custom_target(_flashkda_C)
endif()
