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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# PyTorch headers require C++20; GCC < 11.3 has incomplete C++20 support.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.3")
  message(FATAL_ERROR
    "GCC >= 11.3 is required to build vLLM (found ${CMAKE_CXX_COMPILER_VERSION}). "
    "PyTorch's C++20 headers require a compiler with full C++20 support. "
    "See: https://github.com/pytorch/pytorch/pull/167929")
endif()


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
set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12" "3.13" "3.14")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")

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
set(TORCH_SUPPORTED_VERSION_CUDA "2.11.0")
set(TORCH_SUPPORTED_VERSION_ROCM "2.11.0")
# TORCH_NIGHTLY=1 builds run against unpinned nightly wheels, so the supported-
# version check would always warn. Only treat it as a nightly build when the
# value is exactly "1" (the bootstrap exports TORCH_NIGHTLY=0 by default, which
# must NOT suppress the warning for normal builds).
if (DEFINED ENV{TORCH_NIGHTLY} AND "$ENV{TORCH_NIGHTLY}" STREQUAL "1")
  set(TORCH_NIGHTLY_BUILD TRUE)
else()
  set(TORCH_NIGHTLY_BUILD FALSE)
endif()

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
   # starting from CUDA 12.9 and Blackwell (10.0), we use family-specific targets (10.0f, 12.0f, etc)
   # to support the whole generation without specifying all sub-architectures
   # see: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1")
else()
  set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

#
# spinloop extension (pure CXX; must stay above the non-CUDA device branch so
# CPU builds define the target before the early return)
# This extension requires SABI 3.11 since it relies on Py_buffer support. Loading
# failure is handled gracefully on vLLM side for lower Python versions.
# Skip the target entirely on Python < 3.11 so the build doesn't break.
#
if(Python_VERSION VERSION_GREATER_EQUAL "3.11")
  set(VLLM_SPINLOOP_EXT_SRC "csrc/spinloop.cpp")
  set(SPINLOOP_COMPILE_FLAGS "")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    list(APPEND SPINLOOP_COMPILE_FLAGS "-mmwaitx")
  endif()
  define_extension_target(
    spinloop
    DESTINATION vllm
    LANGUAGE CXX
    SOURCES ${VLLM_SPINLOOP_EXT_SRC}
    COMPILE_FLAGS ${SPINLOOP_COMPILE_FLAGS}
    USE_SABI 3.11
    WITH_SOABI)
endif()

#
# fs_io extension (pure CXX; must stay above the non-CUDA device branch
# so CPU builds define the target before the early return).
# GIL-releasing filesystem helpers for FileSystemTierManager.
#
if(Python_VERSION VERSION_GREATER_EQUAL "3.11")
  define_extension_target(
    fs_io_C
    DESTINATION vllm
    LANGUAGE CXX
    SOURCES csrc/fs_io.cpp
    USE_SABI 3.11
    WITH_SOABI)
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
if (NOT HIP_FOUND AND NOT PYTORCH_FOUND_HIP AND CUDA_FOUND)
  set(VLLM_GPU_LANG "CUDA")

  if (NOT TORCH_NIGHTLY_BUILD AND NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
elseif(HIP_FOUND OR PYTORCH_FOUND_HIP)
  set(VLLM_GPU_LANG "HIP")

  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  # ROCm 5.X and 6.X
  if (NOT TORCH_NIGHTLY_BUILD AND ROCM_VERSION_DEV_MAJOR GREATER_EQUAL 5 AND
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
  # `+PTX` in TORCH_CUDA_ARCH_LIST is not preserved here. It is emitted by torch
  # as `code=compute_*`, while extract_unique_cuda_archs_ascending() records only
  # `arch=compute_*`. If a kernel really needs PTX, add `+PTX` to that kernel's
  # component-specific arch list below.
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
  set(CMAKE_${VLLM_GPU_LANG}_FLAGS "${CMAKE_${VLLM_GPU_LANG}_FLAGS} -Wno-unused-result -Wno-unused-value")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result -Wno-unused-value")

  # When using LTO then *.cpp files must be compiled with same compiler as used linker
  # So if HIP uses clang linker we also must use it
  # Otherwise symbols will be missing from .so
  if (CMAKE_CXX_FLAGS MATCHES "\-flto")
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL CMAKE_HIP_COMPILER_ID)
      message(FATAL_ERROR "LTO is enabled for ROCm build, but the C++ compiler (${CMAKE_CXX_COMPILER_ID}) and HIP compiler (${CMAKE_HIP_COMPILER_ID}) are different which is not supported. "
                          "Please ensure they are same by setting CXX=${CMAKE_HIP_COMPILER} environment variable. Or alternatively disable LTO.")
    endif()
   endif()
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
# Legacy _C extension (ROCm only — CUDA ops migrated to _C_stable_libtorch)
#

if(VLLM_GPU_LANG STREQUAL "HIP")
  set(VLLM_EXT_SRC
    "csrc/torch_bindings.cpp"
    "csrc/custom_quickreduce.cu")

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
endif() # _C HIP endif

if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
  #
  # _C_stable_libtorch extension (ops registered via STABLE_TORCH_LIBRARY)
  #
  set(VLLM_STABLE_EXT_SRC
    "csrc/libtorch_stable/torch_bindings.cpp"
    "csrc/libtorch_stable/cuda_view.cu"
    "csrc/libtorch_stable/cuda_utils_kernels.cu"
    "csrc/libtorch_stable/activation_kernels.cu"
    "csrc/libtorch_stable/quantization/activation_kernels.cu"
    "csrc/libtorch_stable/quantization/w8a8/int8/scaled_quant.cu"
    "csrc/libtorch_stable/quantization/w8a8/fp8/common.cu"
    "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu"
    "csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu"
    "csrc/libtorch_stable/permute_cols.cu"
    "csrc/libtorch_stable/quantization/gptq/q_gemm.cu"
    "csrc/libtorch_stable/pos_encoding_kernels.cu"
    "csrc/libtorch_stable/fused_qknorm_rope_kernel.cu"
    "csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu"
    "csrc/libtorch_stable/layernorm_kernels.cu"
    "csrc/libtorch_stable/layernorm_quant_kernels.cu"
    "csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
    "csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu"
    "csrc/libtorch_stable/attention/merge_attn_states.cu"
    "csrc/libtorch_stable/sampler.cu"
    "csrc/libtorch_stable/topk.cu"
    "csrc/libtorch_stable/mamba/selective_scan_fwd.cu"
    "csrc/libtorch_stable/attention/paged_attention_v1.cu"
    "csrc/libtorch_stable/attention/paged_attention_v2.cu"
    "csrc/libtorch_stable/cache_kernels.cu"
    "csrc/libtorch_stable/cache_kernels.cu"
    "csrc/libtorch_stable/cache_kernels_fused.cu"
    "csrc/libtorch_stable/custom_all_reduce.cu"
    "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu")

  if(VLLM_GPU_LANG STREQUAL "CUDA" AND
     DEFINED CMAKE_CUDA_COMPILER_VERSION AND
     CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)

    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
      cuda_archs_loose_intersection(COOPERATIVE_TOPK_ARCHS
        "9.0a;10.0f;10.1f;10.3f;11.0f;12.0f;12.1f" "${CUDA_ARCHS}")
    else()
      cuda_archs_loose_intersection(COOPERATIVE_TOPK_ARCHS
        "9.0a;10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
    endif()

    if(COOPERATIVE_TOPK_ARCHS)
      list(APPEND VLLM_GPU_FLAGS "-DVLLM_ENABLE_COOPERATIVE_TOPK=1")

    endif()
  endif()

  if(VLLM_GPU_LANG STREQUAL "CUDA")
    SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

    # Set CUTLASS_REVISION. Used for FetchContent. Also fixes some bogus messages when building.
    set(CUTLASS_REVISION "v4.4.2")

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

    list(APPEND VLLM_STABLE_EXT_SRC
      "csrc/libtorch_stable/cutlass_extensions/common.cpp"
      "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu"
      "csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu"
      "csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu"
      "csrc/libtorch_stable/quantization/awq/gemm_kernels.cu"
      "csrc/libtorch_stable/minimax_reduce_rms_kernel.cu")

    #
    # Machete kernels
    #
    # The machete kernels only work on hopper and require CUDA 12.0 or later.
    # Only build Machete kernels if we are building for something compatible with sm90a
    cuda_archs_loose_intersection(MACHETE_ARCHS "9.0a" "${CUDA_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND MACHETE_ARCHS)
      #
      # For the Machete kernels we automatically generate sources for various
      # preselected input type pairs and schedules.
      # Generate sources:
      set(MACHETE_GEN_SCRIPT
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/quantization/machete/generate.py)
      file(MD5 ${MACHETE_GEN_SCRIPT} MACHETE_GEN_SCRIPT_HASH)

      message(STATUS "Machete generation script hash: ${MACHETE_GEN_SCRIPT_HASH}")
      message(STATUS "Last run machete generate script hash: $CACHE{MACHETE_GEN_SCRIPT_HASH}")

      if (NOT DEFINED CACHE{MACHETE_GEN_SCRIPT_HASH}
          OR NOT $CACHE{MACHETE_GEN_SCRIPT_HASH} STREQUAL ${MACHETE_GEN_SCRIPT_HASH})
        execute_process(
          COMMAND ${CMAKE_COMMAND} -E env
          PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass_extensions/:${CUTLASS_DIR}/python/:${VLLM_PYTHON_PATH}:$ENV{PYTHONPATH}
            ${Python_EXECUTABLE} ${MACHETE_GEN_SCRIPT}
          RESULT_VARIABLE machete_generation_result
          OUTPUT_VARIABLE machete_generation_output
          OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
          ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
        )

        if (NOT machete_generation_result EQUAL 0)
          message(FATAL_ERROR "Machete generation failed."
                              " Result: \"${machete_generation_result}\""
                              "\nCheck the log for details: "
                              "${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log")
        else()
          set(MACHETE_GEN_SCRIPT_HASH ${MACHETE_GEN_SCRIPT_HASH}
              CACHE STRING "Last run machete generate script hash" FORCE)
          message(STATUS "Machete generation completed successfully.")
        endif()
      else()
        message(STATUS "Machete generation script has not changed, skipping generation.")
      endif()

      # Add machete generated sources
      file(GLOB MACHETE_GEN_SOURCES "csrc/libtorch_stable/quantization/machete/generated/*.cu")
      list(APPEND VLLM_STABLE_EXT_SRC ${MACHETE_GEN_SOURCES})

      # forward compatible
      set_gencode_flags_for_srcs(
        SRCS "${MACHETE_GEN_SOURCES}"
        CUDA_ARCHS "${MACHETE_ARCHS}")

      list(APPEND VLLM_STABLE_EXT_SRC
        csrc/libtorch_stable/quantization/machete/machete_pytorch.cu)
      message(STATUS "Building Machete kernels for archs: ${MACHETE_ARCHS}")
    else()
      if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0
          AND MACHETE_ARCHS)
        message(STATUS "Not building Machete kernels as CUDA Compiler version is "
                       "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                       "later if you intend on running w4a16 quantized models on "
                       "Hopper.")
      else()
        message(STATUS "Not building Machete kernels as no compatible archs "
                       "found in CUDA target architectures")
      endif()
    endif()

    set_gencode_flags_for_srcs(
      SRCS "${VLLM_STABLE_EXT_SRC}"
      CUDA_ARCHS "${CUDA_ARCHS}")

    if(COOPERATIVE_TOPK_ARCHS)
      list(APPEND VLLM_STABLE_EXT_SRC
        "csrc/libtorch_stable/cooperative_topk.cu")
      set_gencode_flags_for_srcs(
        SRCS "csrc/libtorch_stable/cooperative_topk.cu"
        CUDA_ARCHS "${COOPERATIVE_TOPK_ARCHS}")
    endif()

    # Only build Marlin kernels if we are building for at least some compatible archs.
    # Keep building Marlin for 9.0 as there are some group sizes and shapes that
    # are not supported by Machete yet.

    # marlin arches for fp16 output
    # Family-conditional 12.0f (one cubin for SM12x family) requires CUDA >= 13.0;
    # fall back to architecture-specific 12.0a;12.1a on CUDA < 13.0 (e.g. 12.8).
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
      cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX;12.0f" "${CUDA_ARCHS}")
    else()
      cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX;12.0a;12.1a" "${CUDA_ARCHS}")
    endif()
    # marlin has limited support for turing
    cuda_archs_loose_intersection(MARLIN_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
    # marlin arches for bf16 output (we need 9.0 for bf16 atomicAdd PTX)
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
      cuda_archs_loose_intersection(MARLIN_BF16_ARCHS "8.0+PTX;9.0+PTX;12.0f" "${CUDA_ARCHS}")
    else()
      cuda_archs_loose_intersection(MARLIN_BF16_ARCHS "8.0+PTX;9.0+PTX;12.0a;12.1a" "${CUDA_ARCHS}")
    endif()
    # marlin arches for fp8 input
    # - sm80 doesn't support fp8 computation
    # - sm90 and sm100 don't support QMMA.16832.F32.E4M3.E4M3 SAAS instruction
    # so we only enable fp8 computation for SM89 (e.g. RTX 40x0)  and 12.0 (e.g. RTX 50x0)
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
      cuda_archs_loose_intersection(MARLIN_FP8_ARCHS "8.9;12.0f" "${CUDA_ARCHS}")
    else()
      cuda_archs_loose_intersection(MARLIN_FP8_ARCHS "8.9;12.0a;12.1a" "${CUDA_ARCHS}")
    endif()
    # marlin arches for other files
    cuda_archs_loose_intersection(MARLIN_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")

    if (MARLIN_OTHER_ARCHS)

      #
      # For the Marlin kernels we automatically generate sources for various
      # preselected input type pairs and schedules.
      # Generate sources:
      set(MARLIN_GEN_SCRIPT
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/quantization/marlin/generate_kernels.py)
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
        file(GLOB MARLIN_TEMPLATE_KERNEL_SRC "csrc/libtorch_stable/quantization/marlin/sm80_kernel_*_float16.cu")
        set_gencode_flags_for_srcs(
          SRCS "${MARLIN_TEMPLATE_KERNEL_SRC}"
          CUDA_ARCHS "${MARLIN_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
          set_source_files_properties(${MARLIN_TEMPLATE_KERNEL_SRC}
            PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_STABLE_EXT_SRC ${MARLIN_TEMPLATE_KERNEL_SRC})

        file(GLOB MARLIN_TEMPLATE_BF16_KERNEL_SRC "csrc/libtorch_stable/quantization/marlin/sm80_kernel_*_bfloat16.cu")
        set_gencode_flags_for_srcs(
          SRCS "${MARLIN_TEMPLATE_BF16_KERNEL_SRC}"
          CUDA_ARCHS "${MARLIN_BF16_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
          set_source_files_properties(${MARLIN_TEMPLATE_BF16_KERNEL_SRC}
            PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_STABLE_EXT_SRC ${MARLIN_TEMPLATE_BF16_KERNEL_SRC})
      endif()

      if (MARLIN_SM75_ARCHS)
        file(GLOB MARLIN_TEMPLATE_SM75_KERNEL_SRC "csrc/libtorch_stable/quantization/marlin/sm75_kernel_*.cu")
        set_gencode_flags_for_srcs(
          SRCS "${MARLIN_TEMPLATE_SM75_KERNEL_SRC}"
          CUDA_ARCHS "${MARLIN_SM75_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
          set_source_files_properties(${MARLIN_TEMPLATE_SM75_KERNEL_SRC}
            PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_STABLE_EXT_SRC ${MARLIN_TEMPLATE_SM75_KERNEL_SRC})
      endif()

      if (MARLIN_FP8_ARCHS)
        file(GLOB MARLIN_TEMPLATE_FP8_KERNEL_SRC "csrc/libtorch_stable/quantization/marlin/sm89_kernel_*.cu")
        set_gencode_flags_for_srcs(
          SRCS "${MARLIN_TEMPLATE_FP8_KERNEL_SRC}"
          CUDA_ARCHS "${MARLIN_FP8_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
          set_source_files_properties(${MARLIN_TEMPLATE_FP8_KERNEL_SRC}
            PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_STABLE_EXT_SRC ${MARLIN_TEMPLATE_FP8_KERNEL_SRC})
      endif()

      set(MARLIN_SRCS
         "csrc/libtorch_stable/quantization/marlin/marlin.cu"
         "csrc/libtorch_stable/quantization/marlin/marlin_int4_fp8_preprocess.cu"
         "csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu"
         "csrc/libtorch_stable/quantization/marlin/awq_marlin_repack.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_SRCS}"
        CUDA_ARCHS "${MARLIN_OTHER_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_SRCS}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_STABLE_EXT_SRC "${MARLIN_SRCS}")

      message(STATUS "Building Marlin kernels for archs: ${MARLIN_OTHER_ARCHS}")
    else()
      message(STATUS "Not building Marlin kernels as no compatible archs found"
                     " in CUDA target architectures")
    endif()

  # DeepSeek V3 fused A GEMM kernel (requires SM 9.0+, Hopper and later)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(DSV3_FUSED_A_GEMM_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(DSV3_FUSED_A_GEMM_ARCHS "9.0a;10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND DSV3_FUSED_A_GEMM_ARCHS)
    set(DSV3_FUSED_A_GEMM_SRCS "csrc/libtorch_stable/dsv3_fused_a_gemm.cu")
    set_gencode_flags_for_srcs(
      SRCS "${DSV3_FUSED_A_GEMM_SRCS}"
      CUDA_ARCHS "${DSV3_FUSED_A_GEMM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${DSV3_FUSED_A_GEMM_SRCS}")
    message(STATUS "Building dsv3_fused_a_gemm for archs: ${DSV3_FUSED_A_GEMM_ARCHS}")
  else()
    message(STATUS "Not building dsv3_fused_a_gemm as no compatible archs found "
                   "in CUDA target architectures.")
  endif()

  # FP32 router GEMM (H=3072, E=256, M<=32). Requires SM90+ and CUDA >= 12.0.
  cuda_archs_sm90plus(FP32_ROUTER_GEMM_ARCHS "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND FP32_ROUTER_GEMM_ARCHS)
    set(FP32_ROUTER_GEMM_SRCS
      "csrc/libtorch_stable/fp32_router_gemm_entry.cu"
      "csrc/libtorch_stable/fp32_router_gemm.cu")
    set_gencode_flags_for_srcs(
      SRCS "${FP32_ROUTER_GEMM_SRCS}"
      CUDA_ARCHS "${FP32_ROUTER_GEMM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${FP32_ROUTER_GEMM_SRCS}")
    message(STATUS "Building fp32_router_gemm for archs: ${FP32_ROUTER_GEMM_ARCHS}")
  else()
    message(STATUS "Not building fp32_router_gemm as no compatible archs found "
                   "(requires SM90+ and CUDA >= 12.0).")
  endif()

  # Only build AllSpark kernels if we are building for at least some compatible archs.
  cuda_archs_loose_intersection(ALLSPARK_ARCHS "8.0;8.6;8.7;8.9" "${CUDA_ARCHS}")
  if (ALLSPARK_ARCHS)
    set(ALLSPARK_SRCS
       "csrc/libtorch_stable/quantization/gptq_allspark/allspark_repack.cu"
       "csrc/libtorch_stable/quantization/gptq_allspark/allspark_qgemm_w8a16.cu")
    set_gencode_flags_for_srcs(
      SRCS "${ALLSPARK_SRCS}"
      CUDA_ARCHS "${ALLSPARK_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${ALLSPARK_SRCS}")
    message(STATUS "Building AllSpark kernels for archs: ${ALLSPARK_ARCHS}")
  else()
    message(STATUS "Not building AllSpark kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()

  #
  # CUTLASS scaled_mm kernels (moved from _C to _C_stable_libtorch)
  #
  set(SCALED_MM_3X_ARCHS)
  # The cutlass_scaled_mm kernels for Hopper (c3x, i.e. CUTLASS 3.x) require
  # CUDA 12.0 or later
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
    set(SCALED_MM_SM90_SRCS
       "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm90.cu"
       "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8.cu"
       "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_int8.cu"
       "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_azp_sm90_int8.cu"
       "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SCALED_MM_SM90_SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SCALED_MM_SM90_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM90=1")
    # Let scaled_mm_c2x know it doesn't need to build these arches
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
    message(STATUS "Building scaled_mm_c3x_sm90 for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
      message(STATUS "Not building scaled_mm_c3x_sm90 as CUDA Compiler version is "
                     "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                     "later if you intend on running FP8 quantized models on "
                     "Hopper.")
    else()
      message(STATUS "Not building scaled_mm_c3x_sm90 as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()


  # The cutlass_scaled_mm kernels for Blackwell SM12x (c3x, i.e. CUTLASS 3.x) require
  # CUDA 12.8 or later
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SCALED_MM_SM120_SRCS
      "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu"
      "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu"
      "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu"
    )
    set_gencode_flags_for_srcs(
      SRCS "${SCALED_MM_SM120_SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SCALED_MM_SM120_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM120=1")
    # Let scaled_mm_c2x know it doesn't need to build these arches
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
    message(STATUS "Building scaled_mm_c3x_sm120 for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
      message(STATUS "Not building scaled_mm_c3x_sm120 as CUDA Compiler version is "
                     "not >= 12.8, we recommend upgrading to CUDA 12.8 or "
                     "later if you intend on running FP8 quantized models on "
                     "Blackwell.")
    else()
      message(STATUS "Not building scaled_mm_c3x_120 as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()


  # The cutlass_scaled_mm kernels for Blackwell SM100 (c3x, i.e. CUTLASS 3.x)
  # require CUDA 12.8 or later
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SCALED_MM_SM100_SRCS
      "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm100.cu"
      "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu"
      "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8.cu"
    )
    set_gencode_flags_for_srcs(
      SRCS "${SCALED_MM_SM100_SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SCALED_MM_SM100_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM100=1")
    # Let scaled_mm_c2x know it doesn't need to build these arches
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
    message(STATUS "Building scaled_mm_c3x_sm100 for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
      message(STATUS "Not building scaled_mm_c3x_sm100 as CUDA Compiler version is "
                     "not >= 12.8, we recommend upgrading to CUDA 12.8 or "
                     "later if you intend on running FP8 quantized models on "
                     "Blackwell.")
    else()
      message(STATUS "Not building scaled_mm_c3x_100 as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()

  #
  # For the cutlass_scaled_mm kernels we want to build the c2x (CUTLASS 2.x)
  # kernels for the remaining archs that are not already built for 3x.
  # (Build 8.9 for FP8)
  cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
    "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
  # subtract out the archs that are already built for 3x
  list(REMOVE_ITEM SCALED_MM_2X_ARCHS ${SCALED_MM_3X_ARCHS})
  if (SCALED_MM_2X_ARCHS)
    set(SCALED_MM_C2X_SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SCALED_MM_C2X_SRCS}"
      CUDA_ARCHS "${SCALED_MM_2X_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SCALED_MM_C2X_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_C2X=1")
    message(STATUS "Building scaled_mm_c2x for archs: ${SCALED_MM_2X_ARCHS}")
  else()
    if (SCALED_MM_3X_ARCHS)
      message(STATUS "Not building scaled_mm_c2x as all archs are already built"
                     " for and covered by scaled_mm_c3x")
    else()
      message(STATUS "Not building scaled_mm_c2x as no compatible archs found "
                    "in CUDA target architectures")
    endif()
  endif()

  #
  # CUTLASS MoE kernels (moved from _C to _C_stable_libtorch)
  #

  # The MoE kernel cutlass_moe_mm requires CUDA 12.3 or later (and ONLY works
  # on Hopper). get_cutlass_(batched_)moe_mm_data should only be compiled
  # if it's possible to compile MoE kernels that use its output.
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
    set(CUTLASS_MOE_SM90_SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm90.cu")
    set_gencode_flags_for_srcs(
      SRCS "${CUTLASS_MOE_SM90_SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${CUTLASS_MOE_SM90_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM90=1")
    message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
      message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
                     "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
                     "if you intend on running FP8 quantized MoE models on Hopper.")
    else()
      message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
                     "in CUDA target architectures.")
    endif()
  endif()

  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(CUTLASS_MOE_SM100_SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu")
    set_gencode_flags_for_srcs(
      SRCS "${CUTLASS_MOE_SM100_SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${CUTLASS_MOE_SM100_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
    message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
      message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
                     "not >= 12.8, we recommend upgrading to CUDA 12.8 or later "
                     "if you intend on running FP8 quantized MoE models on Blackwell.")
    else()
      message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
                     "in CUDA target architectures.")
    endif()
  endif()

  # moe_data.cu is used by all CUTLASS MoE kernels.
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
    set(CUTLASS_MOE_DATA_SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/moe/moe_data.cu")
    set_gencode_flags_for_srcs(
      SRCS "${CUTLASS_MOE_DATA_SRCS}"
      CUDA_ARCHS "${CUTLASS_MOE_DATA_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${CUTLASS_MOE_DATA_SRCS}")
    message(STATUS "Building moe_data for archs: ${CUTLASS_MOE_DATA_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
      message(STATUS "Not building moe_data as CUDA Compiler version is "
                     "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
                     "if you intend on running FP8 quantized MoE models on Hopper or Blackwell.")
    else()
      message(STATUS "Not building moe_data as no compatible archs found "
                     "in CUDA target architectures.")
    endif()
  endif()

  #
  # FP4/NVFP4 kernels (moved from _C to _C_stable_libtorch)
  #

  # SM12x FP4 kernels. These share some generic NVFP4 quantization entry
  # sources with the SM10x/11x block below; set_gencode_flags_for_srcs appends
  # per-source flags, so shared files accumulate both SM12x and SM10x/11x
  # gencodes when both families are requested.
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(FP4_SM120_ARCHS "12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(FP4_SM120_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_SM120_ARCHS)
    set(FP4_SM120_SRCS
      "csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu"
      "csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
      "csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu"
      "csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu"
      "csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu"
      "csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu")
    set_gencode_flags_for_srcs(
      SRCS "${FP4_SM120_SRCS}"
      CUDA_ARCHS "${FP4_SM120_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${FP4_SM120_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM120=1")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM120=1")
    message(STATUS "Building SM12x NVFP4 for archs: ${FP4_SM120_ARCHS}")
  else()
    message(STATUS "Not building SM12x NVFP4 as no compatible archs were found.")
  endif()

  # SM10x/11x FP4 kernels. MXFP4 experts quantization is currently compiled
  # only in this block; SM12x has separate NVFP4 matmul/MoE kernels above.
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(FP4_SM100_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(FP4_SM100_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_SM100_ARCHS)
    set(FP4_SM100_SRCS
      "csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu"
      "csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
      "csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu"
      "csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_kernels.cu"
      "csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu"
      "csrc/libtorch_stable/quantization/fp4/mxfp4_experts_quant.cu"
      "csrc/libtorch_stable/quantization/fp4/mxfp4_blockwise_moe_kernel.cu"
      "csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu")
    if(NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
      message(STATUS
        "Building mxfp4_experts_quant unsupported stubs because CUDA compiler version is not >= 12.9 (found ${CMAKE_CUDA_COMPILER_VERSION}).")
    endif()
    set_gencode_flags_for_srcs(
      SRCS "${FP4_SM100_SRCS}"
      CUDA_ARCHS "${FP4_SM100_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${FP4_SM100_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM100=1")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
    message(STATUS "Building SM10x/11x NVFP4/MXFP4 for archs: ${FP4_SM100_ARCHS}")
  else()
    message(STATUS "Not building SM10x/11x NVFP4/MXFP4 as no compatible archs were found.")
  endif()

  #
  # W4A8 kernels (moved from _C to _C_stable_libtorch)
  #

  # Only build W4A8 kernels if we are building for something compatible with sm90a
  cuda_archs_loose_intersection(W4A8_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND W4A8_ARCHS)
    set(W4A8_SRCS
       "csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu"
       "csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_grouped_mm_entry.cu"
       "csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_utils.cu"
       )

    set_gencode_flags_for_srcs(
      SRCS "${W4A8_SRCS}"
      CUDA_ARCHS "${W4A8_ARCHS}")

    list(APPEND VLLM_STABLE_EXT_SRC "${W4A8_SRCS}")

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

  # CUTLASS MLA Archs and flags.
  # Runtime dispatch is gated in
  # vllm/v1/attention/backends/mla/cutlass_mla.py.
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(MLA_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(MLA_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND MLA_ARCHS)
    set(CUTLASS_MLA_SRCS
      "csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu")
    set_gencode_flags_for_srcs(
      SRCS "${CUTLASS_MLA_SRCS}"
      CUDA_ARCHS "${MLA_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${CUTLASS_MLA_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MLA=1")
    # Add MLA-specific include directories only to MLA source files
    set_source_files_properties(${CUTLASS_MLA_SRCS}
      PROPERTIES INCLUDE_DIRECTORIES "${CUTLASS_DIR}/examples/77_blackwell_fmha;${CUTLASS_DIR}/examples/common")
    message(STATUS "Building CUTLASS MLA for archs: ${MLA_ARCHS}")
  else()
    message(STATUS "Not building CUTLASS MLA as no compatible archs were found.")
    # clear MLA_ARCHS
    set(MLA_ARCHS)
  endif()

  # Hadacore kernels
  cuda_archs_loose_intersection(HADACORE_ARCHS "8.0+PTX;9.0+PTX" "${CUDA_ARCHS}")
  if(HADACORE_ARCHS)
    set(HADACORE_SRCS "csrc/libtorch_stable/quantization/hadamard/hadacore/hadamard_transform_cuda.cu")
    set_gencode_flags_for_srcs(
      SRCS "${HADACORE_SRCS}"
      CUDA_ARCHS "${HADACORE_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${HADACORE_SRCS}")
    message(STATUS "Building hadacore")
  endif()

  # if CUDA endif
  endif()

  message(STATUS "Enabling C_stable extension.")
  list(REMOVE_DUPLICATES VLLM_STABLE_EXT_SRC)
  define_extension_target(
    _C_stable_libtorch
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_STABLE_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR} ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
    USE_SABI 3
    WITH_SOABI)

  # Set TORCH_TARGET_VERSION for stable ABI compatibility.
  # This ensures we only use C-shim APIs available in PyTorch 2.11.
  # _C_stable_libtorch is abi compatible with PyTorch >= TORCH_TARGET_VERSION
  # which is currently set to 2.11.
  target_compile_definitions(_C_stable_libtorch PRIVATE
    TORCH_TARGET_VERSION=0x020B000000000000ULL)

  # Needed to use cuda/hip APIs from C-shim
  if(VLLM_GPU_LANG STREQUAL "CUDA")
    target_compile_definitions(_C_stable_libtorch PRIVATE USE_CUDA)
    if(COOPERATIVE_TOPK_ARCHS)
      target_compile_definitions(_C_stable_libtorch PRIVATE
        VLLM_ENABLE_COOPERATIVE_TOPK=1)
    endif()
    # Needed by CUTLASS kernels
    target_compile_definitions(_C_stable_libtorch PRIVATE
      CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
  elseif(VLLM_GPU_LANG STREQUAL "HIP")
    target_compile_definitions(_C_stable_libtorch PRIVATE USE_ROCM)
  endif()

  # On ROCm, _C_stable_libtorch calls raw HIP APIs (e.g. hipGetDevice in
  # get_device_prop()) which must resolve to the same libamdhip64.so that
  # PyTorch uses.  When PyTorch bundles its own copy (pip/conda wheels),
  # the raw HIP calls would otherwise resolve to the system ROCm copy,
  # initializing a second HIP runtime that corrupts device state (wrong
  # device on DeviceGuard, core dumps on multi-GPU tests).
  #
  # If PyTorch doesn't bundle libamdhip64 (built from source against system
  # ROCm), there is only one copy in the process and no action is needed —
  # the HIP compiler already links the system libamdhip64 automatically.
  if(VLLM_GPU_LANG STREQUAL "HIP")
    find_library(_STABLE_TORCH_AMDHIP64 amdhip64
      PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
    if(_STABLE_TORCH_AMDHIP64)
      message(STATUS "Found PyTorch-bundled libamdhip64 at ${_STABLE_TORCH_AMDHIP64}")
      target_link_libraries(_C_stable_libtorch PRIVATE ${_STABLE_TORCH_AMDHIP64})
    endif()
  endif()
endif()

#
# _moe_C_stable_libtorch extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/libtorch_stable/moe/torch_bindings.cpp"
  "csrc/libtorch_stable/moe/moe_align_sum_kernels.cu"
  "csrc/libtorch_stable/moe/topk_softmax_kernels.cu"
  "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_MOE_EXT_SRC
    "csrc/libtorch_stable/moe/moe_wna16.cu"
    "csrc/libtorch_stable/moe/grouped_topk_kernels.cu")
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(MOE_PERMUTE_SRC
      "csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
      "csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu")

  list(APPEND VLLM_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")
endif()

set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(VLLM_MOE_WNA16_SRC
    "csrc/libtorch_stable/moe/moe_wna16.cu")

  set_gencode_flags_for_srcs(
    SRCS "${VLLM_MOE_WNA16_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  list(APPEND VLLM_MOE_EXT_SRC "${VLLM_MOE_WNA16_SRC}")
  # moe marlin arches
  # note that we always set `use_atomic_add=False` for moe marlin now,
  # so we don't need 9.0 for bf16 atomicAdd PTX
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX;12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX;12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  # moe marlin has limited support for turing
  cuda_archs_loose_intersection(MARLIN_MOE_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
  # moe marlin arches for fp8 input
  # - sm80 doesn't support fp8 computation
  # - sm90 and sm100 don't support QMMA.16832.F32.E4M3.E4M3 SAAS instruction
  # so we only enable fp8 computation for SM89 (e.g. RTX 40x0)  and 12.0 (e.g. RTX 50x0)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(MARLIN_MOE_FP8_ARCHS "8.9;12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(MARLIN_MOE_FP8_ARCHS "8.9;12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  # moe marlin arches for other files
  cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")
  if (MARLIN_MOE_OTHER_ARCHS)

    #
    # For the Marlin MOE kernels we automatically generate sources for various
    # preselected input type pairs and schedules.
    # Generate sources:
    set(MOE_MARLIN_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py)
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
      file(GLOB MARLIN_MOE_SRC "csrc/libtorch_stable/moe/marlin_moe_wna16/sm80_kernel_*.cu")
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
      file(GLOB MARLIN_MOE_SM75_SRC "csrc/libtorch_stable/moe/marlin_moe_wna16/sm75_kernel_*.cu")
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
      file(GLOB MARLIN_MOE_FP8_SRC "csrc/libtorch_stable/moe/marlin_moe_wna16/sm89_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_FP8_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_FP8_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_MOE_FP8_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_FP8_SRC})
    endif()

    set(MARLIN_MOE_OTHER_SRC "csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu")
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

  # DeepSeek V3 router GEMM kernel requires SM90+ and CUDA >= 12.0.
  # (fp32_router_gemm has been migrated to _C_stable_libtorch above.)
  cuda_archs_sm90plus(SM90PLUS_ROUTER_GEMM_ARCHS "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SM90PLUS_ROUTER_GEMM_ARCHS)
    set(DSV3_ROUTER_GEMM_SRC
      "csrc/libtorch_stable/moe/dsv3_router_gemm_entry.cu"
      "csrc/libtorch_stable/moe/dsv3_router_gemm_float_out.cu"
      "csrc/libtorch_stable/moe/dsv3_router_gemm_bf16_out.cu")
    set_gencode_flags_for_srcs(
      SRCS "${DSV3_ROUTER_GEMM_SRC}"
      CUDA_ARCHS "${SM90PLUS_ROUTER_GEMM_ARCHS}")
    list(APPEND VLLM_MOE_EXT_SRC "${DSV3_ROUTER_GEMM_SRC}")

    message(STATUS "Building DSV3 router GEMM kernels for archs: ${SM90PLUS_ROUTER_GEMM_ARCHS}")
  else()
    message(STATUS "Not building DSV3 router GEMM kernels as no compatible archs found"
                   " (requires SM90+ and CUDA >= 12.0)")
  endif()
endif()

message(STATUS "Enabling MoE C_stable extension.")
define_extension_target(
  _moe_C_stable_libtorch
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

# Set TORCH_TARGET_VERSION for stable ABI compatibility.
# This ensures we only use C-shim APIs available in PyTorch 2.11.
# _moe_C_stable_libtorch is abi compatible with PyTorch >= TORCH_TARGET_VERSION
# which is currently set to 2.11.
target_compile_definitions(_moe_C_stable_libtorch PRIVATE
  TORCH_TARGET_VERSION=0x020B000000000000ULL)

# Needed to use cuda/hip APIs from C-shim
if(VLLM_GPU_LANG STREQUAL "CUDA")
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE USE_CUDA)
  # Needed by CUTLASS kernels
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE
    CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
elseif(VLLM_GPU_LANG STREQUAL "HIP")
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE USE_ROCM)
endif()

# On ROCm, _moe_C_stable_libtorch calls raw HIP APIs (e.g. hipGetDevice in
# get_device_prop()) which must resolve to the same libamdhip64.so that
# PyTorch uses.  When PyTorch bundles its own copy (pip/conda wheels),
# the raw HIP calls would otherwise resolve to the system ROCm copy,
# initializing a second HIP runtime that corrupts device state (wrong
# device on DeviceGuard, core dumps on multi-GPU tests).
#
# If PyTorch doesn't bundle libamdhip64 (built from source against system
# ROCm), there is only one copy in the process and no action is needed —
# the HIP compiler already links the system libamdhip64 automatically.
if(VLLM_GPU_LANG STREQUAL "HIP")
  find_library(_MOE_STABLE_TORCH_AMDHIP64 amdhip64
    PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
  if(_MOE_STABLE_TORCH_AMDHIP64)
    message(STATUS "Found PyTorch-bundled libamdhip64 for _moe_C_stable_libtorch at ${_MOE_STABLE_TORCH_AMDHIP64}")
    target_link_libraries(_moe_C_stable_libtorch PRIVATE ${_MOE_STABLE_TORCH_AMDHIP64})
  endif()
endif()

if(VLLM_GPU_LANG STREQUAL "HIP")
  #
  # _rocm_C extension
  #
  set(VLLM_ROCM_EXT_SRC
    "csrc/rocm/torch_bindings.cpp"
    "csrc/rocm/skinny_gemms.cu"
    "csrc/rocm/attention.cu")

  set(VLLM_ROCM_HAS_GFX1100 OFF)
  if(VLLM_GPU_ARCHES MATCHES "gfx1100")
    set(VLLM_ROCM_HAS_GFX1100 ON)
    list(APPEND VLLM_ROCM_EXT_SRC
      "csrc/rocm/q_gemm_rdna3.cu"
      "csrc/rocm/q_gemm_rdna3_wmma.cu"
      "csrc/rocm/moe_q_gemm_rdna3.cu")
  endif()

  define_extension_target(
    _rocm_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_ROCM_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    USE_SABI 3
    WITH_SOABI)

  if(VLLM_ROCM_HAS_GFX1100)
    target_compile_definitions(_rocm_C PRIVATE VLLM_ROCM_GFX1100)
  endif()
endif()

# Must run after the last HIP `define_extension_target` so every extension
# has registered its sources.
if (VLLM_GPU_LANG STREQUAL "HIP")
  vllm_finalize_hipify_target()
endif()

# For CUDA and HIP builds also build the triton_kernels external package.
if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
    include(cmake/external_projects/triton_kernels.cmake)
endif()

# For CUDA we also build and ship some external projects.
if (VLLM_GPU_LANG STREQUAL "CUDA")
    include(cmake/external_projects/deepgemm.cmake)
    include(cmake/external_projects/fmha_sm100.cmake)
    include(cmake/external_projects/flashmla.cmake)
    include(cmake/external_projects/qutlass.cmake)

    # vllm-flash-attn should be last as it overwrites some CMake functions
    include(cmake/external_projects/vllm_flash_attn.cmake)
endif ()
