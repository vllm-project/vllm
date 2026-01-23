cmake_minimum_required(VERSION 3.26)

# When building directly using CMake, make sure you run the install step
# (it places the .so files in the correct location).
#
# Example:
# mkdir build && cd build
# cmake -G Ninja -DVLLM_PYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=.. ..
# cmake --build . --target install
#
# If you want to only build one target, make sure to install it manually:
# cmake --build . --target _C
# cmake --install . --component _C
project(vllm_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Prevent installation of dependencies (cutlass) by default.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12" "3.13")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1200;gfx1201;gfx1150;gfx1151")

# ROCm installation prefix. Default to /opt/rocm but allow override via
# -DROCM_PATH=/your/rocm/path when invoking cmake.
if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation prefix")
else()
  set(ROCM_PATH ${ROCM_PATH} CACHE PATH "ROCm installation prefix" FORCE)
endif()
#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from docker/Dockerfile.rocm
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.9.1")
set(TORCH_SUPPORTED_VERSION_ROCM "2.9.1")

#
# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

# Supported NVIDIA architectures.
# This check must happen after find_package(Torch) because that's when CMAKE_CUDA_COMPILER_VERSION gets defined
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")
else()
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

#
# Forward the non-CUDA device extensions to external CMake scripts.
#
if (NOT VLLM_TARGET_DEVICE STREQUAL "cuda" AND
    NOT VLLM_TARGET_DEVICE STREQUAL "rocm")
    if (VLLM_TARGET_DEVICE STREQUAL "cpu")
        include(${CMAKE_CURRENT_LIST_DIR}/cmake/cpu_extension.cmake)
    else()
        return()
    endif()
    return()
endif()

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (NOT HIP_FOUND AND CUDA_FOUND)
  set(VLLM_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
elseif(HIP_FOUND)
  set(VLLM_GPU_LANG "HIP")

  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  # ROCm 5.X and 6.X
  if (ROCM_VERSION_DEV_MAJOR GREATER_EQUAL 5 AND
      Torch_VERSION VERSION_LESS ${TORCH_SUPPORTED_VERSION_ROCM})
    message(WARNING "Pytorch version >= ${TORCH_SUPPORTED_VERSION_ROCM} "
      "expected for ROCm build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()


if(VLLM_GPU_LANG STREQUAL "CUDA")
  #
  # For cuda we want to be able to control which architectures we compile for on
  # a per-file basis in order to cut down on compile time. So here we extract
  # the set of architectures we want to compile for and remove the from the
  # CMAKE_CUDA_FLAGS so that they are not applied globally.
  #
  clear_cuda_arches(CUDA_ARCH_FLAGS)
  extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
  message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
  # Filter the target architectures by the supported supported archs
  # since for some files we will build for all CUDA_ARCHS.
  cuda_archs_loose_intersection(CUDA_ARCHS
    "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
  message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")
else()
  #
  # For other GPU targets override the GPU architectures detected by cmake/torch
  # and filter them by the supported versions for the current language.
  # The final set of arches is stored in `VLLM_GPU_ARCHES`.
  #
  override_gpu_arches(VLLM_GPU_ARCHES
    ${VLLM_GPU_LANG}
    "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")
endif()

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

#
# Set compression mode for CUDA >=13.x.
#
if(VLLM_GPU_LANG STREQUAL "CUDA" AND
   DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  list(APPEND VLLM_GPU_FLAGS "--compress-mode=size")
endif()

#
# Set CUDA include flags for CXX compiler.
#
if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include")
  if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
  endif()
endif()

#
# Use FetchContent for C++ dependencies that are compiled as part of vLLM's build process.
# setup.py will override FETCHCONTENT_BASE_DIR to play nicely with sccache.
# Each dependency that produces build artifacts should override its BINARY_DIR to avoid
# conflicts between build types. It should instead be set to ${CMAKE_BINARY_DIR}/<dependency>.
#
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

if(VLLM_GPU_LANG STREQUAL "HIP")
  #
  # Overriding the default -O set up by cmake, adding ggdb3 for the most verbose devug info
  #
  set(CMAKE_${VLLM_GPU_LANG}_FLAGS_DEBUG "${CMAKE_${VLLM_GPU_LANG}_FLAGS_DEBUG} -O0 -ggdb3")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb3")

  #
  # Certain HIP functions are marked as [[nodiscard]], yet vllm ignores the result which generates
  # a lot of warnings that always mask real issues. Suppressing until this is properly addressed.
  #
  set(CMAKE_${VLLM_GPU_LANG}_FLAGS "${CMAKE_${VLLM_GPU_LANG}_FLAGS} -Wno-unused-result")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
endif()

#
# Define other extension targets
#

#
# cumem_allocator extension
#

set(VLLM_CUMEM_EXT_SRC
  "csrc/cumem_allocator.cpp")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_CUMEM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
  message(STATUS "Enabling cumem allocator extension.")
  if(VLLM_GPU_LANG STREQUAL "CUDA")
    # link against cuda driver library
    list(APPEND CUMEM_LIBS CUDA::cuda_driver)
  else()
    # link against rocm driver library. Prefer an absolute path to
    # libamdhip64.so inside ${ROCM_PATH}/lib if available, otherwise fall
    # back to linking by name "amdhip64".
    find_library(AMDHIP64_LIB
      NAMES amdhip64 libamdhip64.so
      PATHS ${ROCM_PATH}/lib
      NO_DEFAULT_PATH)
    if(AMDHIP64_LIB)
      message(STATUS "Found libamdhip64 at ${AMDHIP64_LIB}")
      list(APPEND CUMEM_LIBS ${AMDHIP64_LIB})
    else()
      message(WARNING "libamdhip64 not found in ${ROCM_PATH}/lib; falling back to linking 'amdhip64' by name")
      list(APPEND CUMEM_LIBS amdhip64)
    endif()
  endif()
  define_extension_target(
    cumem_allocator
    DESTINATION vllm
    LANGUAGE CXX
    SOURCES ${VLLM_CUMEM_EXT_SRC}
    LIBRARIES ${CUMEM_LIBS}
    USE_SABI 3.8
    WITH_SOABI)
endif()

#
# _C extension
#

set(VLLM_EXT_SRC
  "csrc/mamba/mamba_ssm/selective_scan_fwd.cu"
  "csrc/quantization/gptq/q_gemm.cu"
  "csrc/quantization/gguf/gguf_kernel.cu"
  "csrc/cuda_utils_kernels.cu"
  "csrc/custom_all_reduce.cu"
  "csrc/torch_bindings.cpp")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

  # Set CUTLASS_REVISION. Used for FetchContent. Also fixes some bogus messages when building.
  set(CUTLASS_REVISION "v4.2.1")

  # Use the specified CUTLASS source directory for compilation if VLLM_CUTLASS_SRC_DIR is provided
  if (DEFINED ENV{VLLM_CUTLASS_SRC_DIR})
    set(VLLM_CUTLASS_SRC_DIR $ENV{VLLM_CUTLASS_SRC_DIR})
  endif()

  if(VLLM_CUTLASS_SRC_DIR)
    if(NOT IS_ABSOLUTE VLLM_CUTLASS_SRC_DIR)
      get_filename_component(VLLM_CUTLASS_SRC_DIR "${VLLM_CUTLASS_SRC_DIR}" ABSOLUTE)
    endif()
    message(STATUS "The VLLM_CUTLASS_SRC_DIR is set, using ${VLLM_CUTLASS_SRC_DIR} for compilation")
    FetchContent_Declare(cutlass SOURCE_DIR ${VLLM_CUTLASS_SRC_DIR})
  else()
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/nvidia/cutlass.git
        # Please keep this in sync with CUTLASS_REVISION line above.
        GIT_TAG ${CUTLASS_REVISION}
        GIT_PROGRESS TRUE

        # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
        # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
        # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
        GIT_SHALLOW TRUE
    )
  endif()
  FetchContent_MakeAvailable(cutlass)

list(APPEND VLLM_EXT_SRC
    "csrc/quantization/awq/gemm_kernels.cu"
    "csrc/stable/cutlass_extensions/common.cpp")

  set_gencode_flags_for_srcs(
    SRCS "${VLLM_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  # Only build Marlin kernels if we are building for at least some compatible archs.
  # Keep building Marlin for 9.0 as there are some group sizes and shapes that
  # are not supported by Machete yet.

  # marlin arches for fp16 output
  cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
  # marlin has limited support for turing
  cuda_archs_loose_intersection(MARLIN_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
  # marlin arches for bf16 output (we need 9.0 for bf16 atomicAdd PTX)
  cuda_archs_loose_intersection(MARLIN_BF16_ARCHS "8.0+PTX;9.0+PTX" "${CUDA_ARCHS}")
  # marlin arches for fp8 input
  # - sm80 doesn't support fp8 computation
  # - sm90 and sm100 don't support QMMA.16832.F32.E4M3.E4M3 SAAS instruction
  # so we only enable fp8 computation for SM89 (e.g. RTX 40x0)  and 12.0 (e.g. RTX 50x0)
  cuda_archs_loose_intersection(MARLIN_FP8_ARCHS "8.9;12.0" "${CUDA_ARCHS}")
  # marlin arches for other files
  cuda_archs_loose_intersection(MARLIN_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")

  if (MARLIN_OTHER_ARCHS)

    #
    # For the Marlin kernels we automatically generate sources for various
    # preselected input type pairs and schedules.
    # Generate sources:
    set(MARLIN_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/marlin/generate_kernels.py)
    file(MD5 ${MARLIN_GEN_SCRIPT} MARLIN_GEN_SCRIPT_HASH)
    list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
    set(MARLIN_GEN_SCRIPT_HASH_AND_ARCH "${MARLIN_GEN_SCRIPT_HASH}(ARCH:${CUDA_ARCHS_STR})")

    message(STATUS "Marlin generation script hash: ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")
    message(STATUS "Last run Marlin generate script hash: $CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")

    if (NOT DEFINED CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
        OR NOT $CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH} STREQUAL ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH})
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=$ENV{PYTHONPATH}
          ${Python_EXECUTABLE} ${MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
        RESULT_VARIABLE marlin_generation_result
        OUTPUT_VARIABLE marlin_generation_result
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
        ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
      )

      if (NOT marlin_generation_result EQUAL 0)
        message(FATAL_ERROR "Marlin generation failed."
                            " Result: \"${marlin_generation_result}\""
                            "\nCheck the log for details: "
                            "${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log")
      else()
        set(MARLIN_GEN_SCRIPT_HASH_AND_ARCH ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
            CACHE STRING "Last run Marlin generate script hash and arch" FORCE)
        message(STATUS "Marlin generation completed successfully.")
      endif()
    else()
      message(STATUS "Marlin generation script has not changed, skipping generation.")
    endif()

    if (MARLIN_ARCHS)
      file(GLOB MARLIN_TEMPLATE_KERNEL_SRC "csrc/quantization/marlin/sm80_kernel_*_float16.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_TEMPLATE_KERNEL_SRC}"
        CUDA_ARCHS "${MARLIN_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_TEMPLATE_KERNEL_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_KERNEL_SRC})

      file(GLOB MARLIN_TEMPLATE_BF16_KERNEL_SRC "csrc/quantization/marlin/sm80_kernel_*_bfloat16.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_TEMPLATE_BF16_KERNEL_SRC}"
        CUDA_ARCHS "${MARLIN_BF16_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_TEMPLATE_BF16_KERNEL_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_BF16_KERNEL_SRC})
    endif()

    if (MARLIN_SM75_ARCHS) 
      file(GLOB MARLIN_TEMPLATE_SM75_KERNEL_SRC "csrc/quantization/marlin/sm75_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_TEMPLATE_SM75_KERNEL_SRC}"
        CUDA_ARCHS "${MARLIN_SM75_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_TEMPLATE_SM75_KERNEL_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_SM75_KERNEL_SRC})
    endif()

    if (MARLIN_FP8_ARCHS) 
      file(GLOB MARLIN_TEMPLATE_FP8_KERNEL_SRC "csrc/quantization/marlin/sm89_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_TEMPLATE_FP8_KERNEL_SRC}"
        CUDA_ARCHS "${MARLIN_FP8_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_TEMPLATE_FP8_KERNEL_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_FP8_KERNEL_SRC})
    endif()

    set(MARLIN_SRCS
       "csrc/quantization/marlin/marlin.cu"
       "csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu"
       "csrc/quantization/marlin/gptq_marlin_repack.cu"
       "csrc/quantization/marlin/awq_marlin_repack.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_SRCS}"
      CUDA_ARCHS "${MARLIN_OTHER_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_SRCS}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_EXT_SRC "${MARLIN_SRCS}")

    message(STATUS "Building Marlin kernels for archs: ${MARLIN_OTHER_ARCHS}")
  else()
    message(STATUS "Not building Marlin kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()

  # Only build AllSpark kernels if we are building for at least some compatible archs.
  cuda_archs_loose_intersection(ALLSPARK_ARCHS "8.0;8.6;8.7;8.9" "${CUDA_ARCHS}")
  if (ALLSPARK_ARCHS)
    set(ALLSPARK_SRCS
       "csrc/quantization/gptq_allspark/allspark_repack.cu"
       "csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu")
    set_gencode_flags_for_srcs(
      SRCS "${ALLSPARK_SRCS}"
      CUDA_ARCHS "${ALLSPARK_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${ALLSPARK_SRCS}")
    message(STATUS "Building AllSpark kernels for archs: ${ALLSPARK_ARCHS}")
  else()
    message(STATUS "Not building AllSpark kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()


  #
  # 2:4 Sparse Kernels
  # NOTE: Migrated to csrc/stable/sparse/cutlass/ for libtorch stable ABI.
  # Build logic is in the _C_stable_libtorch section below.
  #

  #
  # NVFP4 Kernels
  # NOTE: Migrated to csrc/stable/quantization/fp4/ for libtorch stable ABI.
  # Build logic is in the _C_stable_libtorch section below.
  #

  #
  # CUTLASS MLA Kernels
  # NOTE: Migrated to csrc/stable/attention/mla/ for libtorch stable ABI.
  # Build logic is in the _C_stable_libtorch section below.
  #

  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()

  #
  # Machete kernels - MIGRATED to stable ABI (csrc/stable/quantization/machete/)
  # See the VLLM_STABLE_EXT_SRC section for the stable ABI version

  # Hadacore kernels
  cuda_archs_loose_intersection(HADACORE_ARCHS "8.0+PTX;9.0+PTX" "${CUDA_ARCHS}")
  if(HADACORE_ARCHS)
    set(SRCS "csrc/quantization/hadamard/hadacore/hadamard_transform_cuda.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${HADACORE_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    message(STATUS "Building hadacore")
  endif()

# if CUDA endif
endif()

if (VLLM_GPU_LANG STREQUAL "HIP")
  # Add QuickReduce kernels
  list(APPEND VLLM_EXT_SRC
    "csrc/custom_quickreduce.cu"
  )
# if ROCM endif
endif()

message(STATUS "Enabling C extension.")
define_extension_target(
  _C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)


if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
  #
  # _C_stable_libtorch extension (ops registered via STABLE_TORCH_LIBRARY)
  #
  set(VLLM_STABLE_EXT_SRC
    "csrc/cuda_utils_kernels.cu"
    "csrc/stable/activation_kernels.cu"
    "csrc/stable/cache_kernels.cu"
    "csrc/stable/cache_kernels_fused.cu"
    "csrc/stable/cuda_view.cu"
    "csrc/stable/fused_qknorm_rope_kernel.cu"
    "csrc/stable/layernorm_kernels.cu"
    "csrc/stable/layernorm_quant_kernels.cu"
    "csrc/stable/pos_encoding_kernels.cu"
    "csrc/stable/sampler.cu"
    "csrc/stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
    "csrc/stable/quantization/activation_kernels.cu"
    "csrc/stable/quantization/w8a8/int8/scaled_quant.cu"
    "csrc/stable/quantization/w8a8/fp8/common.cu"
    "csrc/stable/attention/paged_attention_v1.cu"
    "csrc/stable/attention/paged_attention_v2.cu"
    "csrc/stable/attention/merge_attn_states.cu"
    "csrc/stable/attention/vertical_slash_index.cu"
    "csrc/stable/torch_bindings.cpp")


  if(VLLM_GPU_LANG STREQUAL "CUDA")
    list(APPEND VLLM_STABLE_EXT_SRC "csrc/stable/permute_cols.cu"
    "csrc/stable/quantization/w8a8/fp8/per_token_group_quant.cu"
    "csrc/stable/quantization/w8a8/int8/per_token_group_quant.cu"
    "csrc/stable/quantization/w8a8/cutlass/scaled_mm_entry.cu"
    "csrc/stable/quantization/fp4/nvfp4_quant_entry.cu"
    "csrc/stable/quantization/fp4/nvfp4_scaled_mm_entry.cu"
"csrc/stable/sparse/cutlass/sparse_scaled_mm_entry.cu"
    "csrc/stable/cutlass_extensions/common.cpp")
  endif()

  # Only build W4A8 kernels if we are building for something compatible with sm90a
  cuda_archs_loose_intersection(W4A8_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND W4A8_ARCHS)
    set(SRCS
       "csrc/stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu"
       "csrc/stable/quantization/cutlass_w4a8/w4a8_grouped_mm_entry.cu"
       "csrc/stable/quantization/cutlass_w4a8/w4a8_utils.cu"
       )

    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${W4A8_ARCHS}")

    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")

    message(STATUS "Building W4A8 kernels for archs: ${W4A8_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0
        AND W4A8_ARCHS)
      message(STATUS "Not building W4A8 kernels as CUDA Compiler version is "
                     "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                     "later if you intend on running w4a16 quantized models on "
                     "Hopper.")
    else()
      message(STATUS "Not building W4A8 kernels as no compatible archs "
                     "found in CUDA target architectures")
    endif()
  endif()

  if(VLLM_GPU_LANG STREQUAL "CUDA")
    set_gencode_flags_for_srcs(
      SRCS "${VLLM_STABLE_EXT_SRC}"
      CUDA_ARCHS "${CUDA_ARCHS}")
  endif()

  #
  # Stable CUTLASS kernels (architecture-specific builds)
  #

  # Track which archs are built for 3x so c2x can exclude them
  set(STABLE_SCALED_MM_3X_ARCHS)
  # Track stable GPU flags separately
  set(VLLM_STABLE_GPU_FLAGS)

  # Stable cutlass_scaled_mm kernels for Hopper (c3x, CUTLASS 3.x) - SM90
  cuda_archs_loose_intersection(STABLE_SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND STABLE_SCALED_MM_ARCHS)
    set(SRCS
       "csrc/stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm90.cu"
       "csrc/stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8.cu"
       "csrc/stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_int8.cu"
       "csrc/stable/quantization/w8a8/cutlass/c3x/scaled_mm_azp_sm90_int8.cu"
       "csrc/stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    list(APPEND STABLE_SCALED_MM_3X_ARCHS "${STABLE_SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_SCALED_MM_SM90=1")
    message(STATUS "Building stable scaled_mm_c3x_sm90 for archs: ${STABLE_SCALED_MM_ARCHS}")
  else()
    message(STATUS "Not building stable scaled_mm_c3x_sm90")
  endif()

  # Stable cutlass_scaled_mm kernels for Blackwell SM120 (c3x, CUTLASS 3.x)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(STABLE_SCALED_MM_ARCHS "12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(STABLE_SCALED_MM_ARCHS "12.0a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND STABLE_SCALED_MM_ARCHS)
    set(SRCS
      "csrc/stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu"
      "csrc/stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu"
      "csrc/stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    list(APPEND STABLE_SCALED_MM_3X_ARCHS "${STABLE_SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_SCALED_MM_SM120=1")
    message(STATUS "Building stable scaled_mm_c3x_sm120 for archs: ${STABLE_SCALED_MM_ARCHS}")
  else()
    message(STATUS "Not building stable scaled_mm_c3x_sm120")
  endif()

  # Stable cutlass_scaled_mm kernels for Blackwell SM100 (c3x, CUTLASS 3.x)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(STABLE_SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(STABLE_SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND STABLE_SCALED_MM_ARCHS)
    set(SRCS
      "csrc/stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm100.cu"
      "csrc/stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu"
      "csrc/stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    list(APPEND STABLE_SCALED_MM_3X_ARCHS "${STABLE_SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_SCALED_MM_SM100=1")
    message(STATUS "Building stable scaled_mm_c3x_sm100 for archs: ${STABLE_SCALED_MM_ARCHS}")
  else()
    message(STATUS "Not building stable scaled_mm_c3x_sm100")
  endif()

  # Stable cutlass_scaled_mm C2x kernels (for older archs not covered by 3x)
  cuda_archs_loose_intersection(STABLE_SCALED_MM_2X_ARCHS
    "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
  list(REMOVE_ITEM STABLE_SCALED_MM_2X_ARCHS ${STABLE_SCALED_MM_3X_ARCHS})
  if(STABLE_SCALED_MM_2X_ARCHS)
    set(SRCS "csrc/stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_SCALED_MM_2X_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_SCALED_MM_C2X=1")
    message(STATUS "Building stable scaled_mm_c2x for archs: ${STABLE_SCALED_MM_2X_ARCHS}")
  else()
    message(STATUS "Not building stable scaled_mm_c2x")
  endif()

  # Stable CUTLASS MoE kernels for Hopper (SM90)
  cuda_archs_loose_intersection(STABLE_MOE_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND STABLE_MOE_ARCHS)
    set(SRCS "csrc/stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm90.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_MOE_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM90=1")
    message(STATUS "Building stable grouped_mm_c3x_sm90 for archs: ${STABLE_MOE_ARCHS}")
  else()
    message(STATUS "Not building stable grouped_mm_c3x_sm90")
  endif()

  # Stable CUTLASS MoE kernels for Blackwell (SM100)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(STABLE_MOE_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(STABLE_MOE_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND STABLE_MOE_ARCHS)
    set(SRCS "csrc/stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_MOE_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
    message(STATUS "Building stable grouped_mm_c3x_sm100 for archs: ${STABLE_MOE_ARCHS}")
  else()
    message(STATUS "Not building stable grouped_mm_c3x_sm100")
  endif()

  # Stable MoE data kernel (shared by all MoE kernels)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(STABLE_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(STABLE_MOE_DATA_ARCHS "9.0a;10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND STABLE_MOE_DATA_ARCHS)
    set(SRCS "csrc/stable/quantization/w8a8/cutlass/moe/moe_data.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_MOE_DATA_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    message(STATUS "Building stable moe_data for archs: ${STABLE_MOE_DATA_ARCHS}")
  else()
    message(STATUS "Not building stable moe_data")
  endif()

  #
  # Stable 2:4 Sparse Kernels (SM90)
  #
  cuda_archs_loose_intersection(STABLE_SPARSE_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.2 AND STABLE_SPARSE_MM_ARCHS)
    set(SRCS "csrc/stable/sparse/cutlass/sparse_scaled_mm_c3x.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_SPARSE_MM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_SPARSE_SCALED_MM_C3X=1")
    message(STATUS "Building stable sparse_scaled_mm_c3x for archs: ${STABLE_SPARSE_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.2 AND STABLE_SPARSE_MM_ARCHS)
      message(STATUS "Not building stable sparse_scaled_mm_c3x kernels as CUDA Compiler version is "
                     "not >= 12.2, we recommend upgrading to CUDA 12.2 or later "
                     "if you intend on running FP8 sparse quantized models on Hopper.")
    else()
      message(STATUS "Not building stable sparse_scaled_mm_c3x as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()

  #
  # Stable NVFP4 kernels for Geforce Blackwell SM120
  #
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(STABLE_FP4_SM120_ARCHS "12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(STABLE_FP4_SM120_ARCHS "12.0a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND STABLE_FP4_SM120_ARCHS)
    set(SRCS
      "csrc/stable/quantization/fp4/nvfp4_quant_kernels.cu"
      "csrc/stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
      "csrc/stable/quantization/fp4/nvfp4_experts_quant.cu"
      "csrc/stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu"
      "csrc/stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_FP4_SM120_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_NVFP4_SM120=1")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM120=1")
    message(STATUS "Building stable NVFP4 SM120 for archs: ${STABLE_FP4_SM120_ARCHS}")
  else()
    message(STATUS "Not building stable NVFP4 SM120 as no compatible archs were found.")
  endif()

  #
  # Stable NVFP4 kernels for Blackwell SM100
  #
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(STABLE_FP4_SM100_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(STABLE_FP4_SM100_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND STABLE_FP4_SM100_ARCHS)
    set(SRCS
      "csrc/stable/quantization/fp4/nvfp4_quant_kernels.cu"
      "csrc/stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
      "csrc/stable/quantization/fp4/nvfp4_experts_quant.cu"
      "csrc/stable/quantization/fp4/nvfp4_scaled_mm_kernels.cu"
      "csrc/stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_FP4_SM100_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_NVFP4_SM100=1")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
    message(STATUS "Building stable NVFP4 SM100 for archs: ${STABLE_FP4_SM100_ARCHS}")
  else()
    message(STATUS "Not building stable NVFP4 SM100 as no compatible archs were found.")
  endif()

  #
  # Stable CUTLASS MLA kernels for Blackwell (SM100+)
  #
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(STABLE_MLA_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(STABLE_MLA_ARCHS "10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND STABLE_MLA_ARCHS)
    set(SRCS
      "csrc/stable/attention/mla/sm100_cutlass_mla_kernel.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${STABLE_MLA_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
    list(APPEND VLLM_STABLE_GPU_FLAGS "-DENABLE_CUTLASS_MLA=1")
    # Add MLA-specific include directories only to MLA source files
    set_source_files_properties(${SRCS}
      PROPERTIES INCLUDE_DIRECTORIES "${CUTLASS_DIR}/examples/77_blackwell_fmha;${CUTLASS_DIR}/examples/common")
    message(STATUS "Building stable CUTLASS MLA for archs: ${STABLE_MLA_ARCHS}")
  else()
    message(STATUS "Not building stable CUTLASS MLA as no compatible archs were found.")
  endif()

  #
  # Stable Machete kernels (SM90+)
  #
  cuda_archs_loose_intersection(STABLE_MACHETE_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND STABLE_MACHETE_ARCHS)
    # For the Machete kernels we automatically generate sources for various
    # preselected input type pairs and schedules.
    set(STABLE_MACHETE_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/stable/quantization/machete/generate.py)
    file(MD5 ${STABLE_MACHETE_GEN_SCRIPT} STABLE_MACHETE_GEN_SCRIPT_HASH)

    message(STATUS "Stable Machete generation script hash: ${STABLE_MACHETE_GEN_SCRIPT_HASH}")
    message(STATUS "Last run stable machete generate script hash: $CACHE{STABLE_MACHETE_GEN_SCRIPT_HASH}")

    if (NOT DEFINED CACHE{STABLE_MACHETE_GEN_SCRIPT_HASH}
        OR NOT $CACHE{STABLE_MACHETE_GEN_SCRIPT_HASH} STREQUAL ${STABLE_MACHETE_GEN_SCRIPT_HASH})
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass_extensions/:${CUTLASS_DIR}/python/:${VLLM_PYTHON_PATH}:$ENV{PYTHONPATH}
          ${Python_EXECUTABLE} ${STABLE_MACHETE_GEN_SCRIPT}
        RESULT_VARIABLE stable_machete_generation_result
        OUTPUT_VARIABLE stable_machete_generation_output
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/stable_machete_generation.log
        ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/stable_machete_generation.log
      )

      if (NOT stable_machete_generation_result EQUAL 0)
        message(FATAL_ERROR "Stable Machete generation failed."
                            " Result: \"${stable_machete_generation_result}\""
                            "\nCheck the log for details: "
                            "${CMAKE_CURRENT_BINARY_DIR}/stable_machete_generation.log")
      else()
        set(STABLE_MACHETE_GEN_SCRIPT_HASH ${STABLE_MACHETE_GEN_SCRIPT_HASH}
            CACHE STRING "Last run stable machete generate script hash" FORCE)
        message(STATUS "Stable Machete generation completed successfully.")
      endif()
    else()
      message(STATUS "Stable Machete generation script has not changed, skipping generation.")
    endif()

    # Add stable machete generated sources
    file(GLOB STABLE_MACHETE_GEN_SOURCES "csrc/stable/quantization/machete/generated/*.cu")
    list(APPEND VLLM_STABLE_EXT_SRC ${STABLE_MACHETE_GEN_SOURCES})

    set_gencode_flags_for_srcs(
      SRCS "${STABLE_MACHETE_GEN_SOURCES}"
      CUDA_ARCHS "${STABLE_MACHETE_ARCHS}")

    list(APPEND VLLM_STABLE_EXT_SRC
      "csrc/stable/quantization/machete/machete_pytorch.cu")

    message(STATUS "Building stable Machete kernels for archs: ${STABLE_MACHETE_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0
        AND STABLE_MACHETE_ARCHS)
      message(STATUS "Not building stable Machete kernels as CUDA Compiler version is "
                     "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                     "later if you intend on running w4a16 quantized models on "
                     "Hopper.")
    else()
      message(STATUS "Not building stable Machete kernels as no compatible archs "
                     "found in CUDA target architectures")
    endif()
  endif()

  message(STATUS "Enabling C_stable extension.")
  define_extension_target(
    _C_stable_libtorch
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_STABLE_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS} ${VLLM_STABLE_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
    USE_SABI 3
    WITH_SOABI)

  # Set TORCH_TARGET_VERSION for stable ABI compatibility.
  # This ensures we only use C-shim APIs available in PyTorch 2.10+.
  target_compile_definitions(_C_stable_libtorch PRIVATE
    TORCH_TARGET_VERSION=0x020A000000000000ULL)

  # Needed to use cuda APIs from C-shim
  target_compile_definitions(_C_stable_libtorch PRIVATE
    USE_CUDA)

  # If CUTLASS is compiled on NVCC >= 12.5, it by default uses
  # cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
  # driver API. This causes problems when linking with earlier versions of CUDA.
  # Setting this variable sidesteps the issue by calling the driver directly.
  target_compile_definitions(_C_stable_libtorch PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
endif()

#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/moe/torch_bindings.cpp"
  "csrc/moe/moe_align_sum_kernels.cu"
  "csrc/moe/topk_softmax_kernels.cu")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_MOE_EXT_SRC
    "csrc/moe/moe_wna16.cu"
    "csrc/moe/grouped_topk_kernels.cu")
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(MOE_PERMUTE_SRC
      "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
      "csrc/moe/moe_permute_unpermute_op.cu")

  list(APPEND VLLM_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")
endif()

set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(VLLM_MOE_WNA16_SRC
    "csrc/moe/moe_wna16.cu")

  set_gencode_flags_for_srcs(
    SRCS "${VLLM_MOE_WNA16_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  list(APPEND VLLM_MOE_EXT_SRC "${VLLM_MOE_WNA16_SRC}")
  # moe marlin arches
  # note that we always set `use_atomic_add=False` for moe marlin now,
  # so we don't need 9.0 for bf16 atomicAdd PTX
  cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
  # moe marlin has limited support for turing
  cuda_archs_loose_intersection(MARLIN_MOE_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
  # moe marlin arches for fp8 input
  # - sm80 doesn't support fp8 computation
  # - sm90 and sm100 don't support QMMA.16832.F32.E4M3.E4M3 SAAS instruction
  # so we only enable fp8 computation for SM89 (e.g. RTX 40x0)  and 12.0 (e.g. RTX 50x0)
  cuda_archs_loose_intersection(MARLIN_MOE_FP8_ARCHS "8.9;12.0" "${CUDA_ARCHS}")
  # moe marlin arches for other files
  cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")
  if (MARLIN_MOE_OTHER_ARCHS)

    #
    # For the Marlin MOE kernels we automatically generate sources for various
    # preselected input type pairs and schedules.
    # Generate sources:
    set(MOE_MARLIN_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/marlin_moe_wna16/generate_kernels.py)
    file(MD5 ${MOE_MARLIN_GEN_SCRIPT} MOE_MARLIN_GEN_SCRIPT_HASH)
    list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
    set(MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH "${MOE_MARLIN_GEN_SCRIPT_HASH}(ARCH:${CUDA_ARCHS_STR})")

    message(STATUS "Marlin MOE generation script hash with arch: ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")
    message(STATUS "Last run Marlin MOE generate script hash with arch: $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")

    if (NOT DEFINED CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
        OR NOT $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH} STREQUAL ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH})
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=$ENV{PYTHONPATH}
          ${Python_EXECUTABLE} ${MOE_MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
        RESULT_VARIABLE moe_marlin_generation_result
        OUTPUT_VARIABLE moe_marlin_generation_output
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
        ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
      )

      if (NOT moe_marlin_generation_result EQUAL 0)
        message(FATAL_ERROR "Marlin MOE generation failed."
                            " Result: \"${moe_marlin_generation_result}\""
                            "\nCheck the log for details: "
                            "${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log")
      else()
        set(MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
            CACHE STRING "Last run Marlin MOE generate script hash" FORCE)
        message(STATUS "Marlin MOE generation completed successfully.")
      endif()
    else()
      message(STATUS "Marlin MOE generation script has not changed, skipping generation.")
    endif()

    if (MARLIN_MOE_ARCHS)
      file(GLOB MARLIN_MOE_SRC "csrc/moe/marlin_moe_wna16/sm80_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_MOE_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SRC})
    endif()

    if (MARLIN_MOE_SM75_ARCHS) 
      file(GLOB MARLIN_MOE_SM75_SRC "csrc/moe/marlin_moe_wna16/sm75_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_SM75_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_SM75_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_MOE_SM75_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SM75_SRC})
    endif()

    if (MARLIN_MOE_FP8_ARCHS)
      file(GLOB MARLIN_MOE_FP8_SRC "csrc/moe/marlin_moe_wna16/sm89_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_FP8_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_FP8_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_MOE_FP8_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_FP8_SRC})
    endif()

    set(MARLIN_MOE_OTHER_SRC "csrc/moe/marlin_moe_wna16/ops.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_OTHER_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_OTHER_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_MOE_OTHER_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_MOE_EXT_SRC "${MARLIN_MOE_OTHER_SRC}")

    message(STATUS "Building Marlin MOE kernels for archs: ${MARLIN_MOE_OTHER_ARCHS}")
  else()
    message(STATUS "Not building Marlin MOE kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()
endif()

message(STATUS "Enabling moe extension.")
define_extension_target(
  _moe_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

if(VLLM_GPU_LANG STREQUAL "HIP")
  #
  # _rocm_C extension
  #
  set(VLLM_ROCM_EXT_SRC
    "csrc/rocm/torch_bindings.cpp"
    "csrc/rocm/skinny_gemms.cu"
    "csrc/rocm/attention.cu")

  define_extension_target(
    _rocm_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_ROCM_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    USE_SABI 3
    WITH_SOABI)
endif()

# For CUDA and HIP builds also build the triton_kernels external package.
if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
    include(cmake/external_projects/triton_kernels.cmake)
endif()

# For CUDA we also build and ship some external projects.
if (VLLM_GPU_LANG STREQUAL "CUDA")
    include(cmake/external_projects/flashmla.cmake)
    include(cmake/external_projects/qutlass.cmake)

    # vllm-flash-attn should be last as it overwrites some CMake functions
    include(cmake/external_projects/vllm_flash_attn.cmake)
endif ()
