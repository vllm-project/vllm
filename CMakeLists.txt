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

# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1200;gfx1201")

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
set(TORCH_SUPPORTED_VERSION_CUDA "2.7.0")
set(TORCH_SUPPORTED_VERSION_ROCM "2.7.0")

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
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM})
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
# Use FetchContent for C++ dependencies that are compiled as part of vLLM's build process.
# setup.py will override FETCHCONTENT_BASE_DIR to play nicely with sccache.
# Each dependency that produces build artifacts should override its BINARY_DIR to avoid
# conflicts between build types. It should instead be set to ${CMAKE_BINARY_DIR}/<dependency>.
#
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

#
# Set rocm version dev int.
#
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

if(VLLM_GPU_LANG STREQUAL "CUDA")
  message(STATUS "Enabling cumem allocator extension.")
  # link against cuda driver library
  list(APPEND CUMEM_LIBS CUDA::cuda_driver)
  define_gpu_extension_target(
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
  "csrc/mamba/causal_conv1d/causal_conv1d.cu"
  "csrc/cache_kernels.cu"
  "csrc/attention/paged_attention_v1.cu"
  "csrc/attention/paged_attention_v2.cu"
  "csrc/attention/merge_attn_states.cu"
  "csrc/attention/vertical_slash_index.cu"
  "csrc/pos_encoding_kernels.cu"
  "csrc/activation_kernels.cu"
  "csrc/layernorm_kernels.cu"
  "csrc/layernorm_quant_kernels.cu"
  "csrc/cuda_view.cu"
  "csrc/quantization/gptq/q_gemm.cu"
  "csrc/quantization/compressed_tensors/int8_quant_kernels.cu"
  "csrc/quantization/fp8/common.cu"
  "csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
  "csrc/quantization/gguf/gguf_kernel.cu"
  "csrc/quantization/activation_kernels.cu"
  "csrc/cuda_utils_kernels.cu"
  "csrc/prepare_inputs/advance_step.cu"
  "csrc/custom_all_reduce.cu"
  "csrc/torch_bindings.cpp")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

  # Set CUTLASS_REVISION. Used for FetchContent. Also fixes some bogus messages when building.
  set(CUTLASS_REVISION "v3.9.2" CACHE STRING "CUTLASS revision to use")

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
    "csrc/quantization/aqlm/gemm_kernels.cu"
    "csrc/quantization/awq/gemm_kernels.cu"
    "csrc/permute_cols.cu"
    "csrc/quantization/cutlass_w8a8/scaled_mm_entry.cu"
    "csrc/quantization/fp4/nvfp4_quant_entry.cu"
    "csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu"
    "csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu"
    "csrc/sparse/cutlass/sparse_scaled_mm_entry.cu"
    "csrc/cutlass_extensions/common.cpp"
    "csrc/attention/mla/cutlass_mla_entry.cu")

  set_gencode_flags_for_srcs(
    SRCS "${VLLM_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  # Only build Marlin kernels if we are building for at least some compatible archs.
  # Keep building Marlin for 9.0 as there are some group sizes and shapes that
  # are not supported by Machete yet.
  # 9.0 for latest bf16 atomicAdd PTX
  # Marlin kernels: generate and build for supported architectures
  cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
  optional_cuda_sources(
    NAME Marlin
    ARCHS "8.0;9.0+PTX"
    GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/gptq_marlin/generate_kernels.py"
    GEN_GLOB "csrc/quantization/gptq_marlin/kernel_*.cu"
    SRCS
      "csrc/quantization/marlin/dense/marlin_cuda_kernel.cu"
      "csrc/quantization/marlin/sparse/marlin_24_cuda_kernel.cu"
      "csrc/quantization/marlin/qqq/marlin_qqq_gemm_kernel.cu"
      "csrc/quantization/gptq_marlin/gptq_marlin.cu"
      "csrc/quantization/gptq_marlin/gptq_marlin_repack.cu"
      "csrc/quantization/gptq_marlin/awq_marlin_repack.cu"
  )

  # AllSpark kernels
  optional_cuda_sources(
    NAME AllSpark
    ARCHS "8.0;8.6;8.7;8.9"
    SRCS
      "csrc/quantization/gptq_allspark/allspark_repack.cu"
      "csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu"
  )


  set(SCALED_MM_3X_ARCHS)
  # The cutlass_scaled_mm kernels for Hopper (c3x, i.e. CUTLASS 3.x) require CUDA 12.0 or later
  optional_cuda_sources(
    NAME scaled_mm_c3x_sm90
    MIN_VERSION 12.0
    ARCHS "9.0a"
    SRCS
      "csrc/quantization/cutlass_w8a8/scaled_mm_c3x_sm90.cu"
      "csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm90_fp8.cu"
      "csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm90_int8.cu"
      "csrc/quantization/cutlass_w8a8/c3x/scaled_mm_azp_sm90_int8.cu"
      "csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm90_fp8.cu"
    FLAGS "-DENABLE_SCALED_MM_SM90=1"
    APPEND_ARCHS SCALED_MM_3X_ARCHS
    VERSION_MSG
      "Not building scaled_mm_c3x_sm90: CUDA Compiler version is not >= 12.0.\n"
      "Please upgrade to CUDA 12.0 or later to run FP8 quantized models on Hopper."
  )

  # The cutlass_scaled_mm kernels for Blackwell (c3x, i.e. CUTLASS 3.x) require CUDA 12.8 or later
  optional_cuda_sources(
    NAME scaled_mm_c3x_sm100
    MIN_VERSION 12.8
    ARCHS "10.0a;10.1a;12.0a"
    SRCS
      "csrc/quantization/cutlass_w8a8/scaled_mm_c3x_sm100.cu"
      "csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm100_fp8.cu"
      "csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm100_fp8.cu"
    FLAGS "-DENABLE_SCALED_MM_SM100=1"
    APPEND_ARCHS SCALED_MM_3X_ARCHS
    VERSION_MSG
      "Not building scaled_mm_c3x_sm100: CUDA Compiler version is not >= 12.8.\n"
      "Please upgrade to CUDA 12.8 or later to run FP8 quantized models on Blackwell."
  )

  # For the cutlass_scaled_mm kernels we want to build the c2x (CUTLASS 2.x)
  # kernels for the remaining archs that are not already built for 3x.
  # (Build 8.9 for FP8)
  cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
    "7.5;8.0;8.9+PTX" "${CUDA_ARCHS}")
  # subtract out the archs that are already built for 3x
  list(REMOVE_ITEM SCALED_MM_2X_ARCHS ${SCALED_MM_3X_ARCHS})
  optional_cuda_sources(
    NAME scaled_mm_c2x
    ARCHS "${SCALED_MM_2X_ARCHS}"
    SRCS "csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cu"
    FLAGS "-DENABLE_SCALED_MM_C2X=1"
    NO_ARCH_MSG "Not building scaled_mm_c2x as no compatible archs found in CUDA target architectures, "
                "or is already covered by scaled_mm_c3x."
  )

  #
  # 2:4 Sparse Kernels
  optional_cuda_sources(
    NAME sparse_scaled_mm_c3x
    MIN_VERSION 12.2
    ARCHS "9.0a;"
    SRCS "csrc/sparse/cutlass/sparse_scaled_mm_c3x.cu"
    FLAGS "-DENABLE_SPARSE_SCALED_MM_C3X=1"
    VERSION_MSG
      "Not building sparse_scaled_mm_c3x: CUDA Compiler version is not >= 12.2.\n"
      "Please upgrade to CUDA 12.2 or later to run FP8 sparse quantized models on Hopper."
  )

  # FP4 Archs and flags
  optional_cuda_sources(
    NAME NVFP4
    MIN_VERSION 12.8
    ARCHS "10.0a"
    SRCS
      "csrc/quantization/fp4/nvfp4_quant_kernels.cu"
      "csrc/quantization/fp4/nvfp4_experts_quant.cu"
      "csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu"
      "csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu"
    FLAGS "-DENABLE_NVFP4=1"
  )

  # CUTLASS MLA Archs and flags
  optional_cuda_sources(
    NAME CUTLASS_MLA
    MIN_VERSION 12.8
    ARCHS "10.0a"
    SRCS "csrc/attention/mla/cutlass_mla_kernels.cu"
    FLAGS "-DENABLE_CUTLASS_MLA=1"
  )
  # Add MLA-specific include directories only to MLA source files
  set_source_files_properties(
    "csrc/attention/mla/cutlass_mla_kernels.cu"
    PROPERTIES INCLUDE_DIRECTORIES "${CUTLASS_DIR}/examples/77_blackwell_fmha;${CUTLASS_DIR}/examples/common"
  )

  # CUTLASS MoE kernels
  optional_cuda_sources(
    NAME grouped_mm_c3x
    MIN_VERSION 12.3
    ARCHS "9.0a;10.0a"
    SRCS
      "csrc/quantization/cutlass_w8a8/moe/grouped_mm_c3x.cu"
      "csrc/quantization/cutlass_w8a8/moe/moe_data.cu"
    FLAGS "-DENABLE_CUTLASS_MOE_SM90=1"
    VERSION_MSG "Not building grouped_mm_c3x kernels as CUDA Compiler version is not >= 12.3, we recommend upgrading to CUDA 12.3 or later if you intend on running FP8 quantized MoE models on Hopper."
  )

  #
  # Machete kernels

  # Machete kernels: generate and build for supported architectures
  cuda_archs_loose_intersection(MACHETE_ARCHS "9.0a" "${CUDA_ARCHS}")
  optional_cuda_sources(
    NAME Machete
    MIN_VERSION 12.0
    ARCHS "${MACHETE_ARCHS}"
    GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/machete/generate.py"
    GEN_GLOB "csrc/quantization/machete/generated/*.cu"
    SRCS "csrc/quantization/machete/machete_pytorch.cu"
    VERSION_MSG
      "Not building Machete kernels as CUDA Compiler version is less than 12.0."
      "We recommend upgrading to CUDA 12.0 or later to run w4a16 quantized models on Hopper."
  )
# if CUDA endif
endif()

message(STATUS "Enabling C extension.")
define_gpu_extension_target(
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

#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/moe/torch_bindings.cpp"
  "csrc/moe/moe_align_sum_kernels.cu"
  "csrc/moe/topk_softmax_kernels.cu")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_MOE_EXT_SRC "csrc/moe/moe_wna16.cu")
endif()

# Apply gencode flags to base MOE extension sources
set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

## Marlin MOE kernels: generate and include for supported architectures
if(VLLM_GPU_LANG STREQUAL "CUDA")
  optional_cuda_sources(
    NAME "Marlin MOE"
    ARCHS "8.0;9.0+PTX"
    GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/marlin_moe_wna16/generate_kernels.py"
    GEN_GLOB "csrc/moe/marlin_moe_wna16/*.cu"
    OUT_SRCS_VAR VLLM_MOE_EXT_SRC
  )
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(MOE_PERMUTE_SRC
      "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
      "csrc/moe/moe_permute_unpermute_op.cu")

  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_PERMUTE_SRC}"
    CUDA_ARCHS "${MOE_PERMUTE_ARCHS}")

  list(APPEND VLLM_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")
endif()
message(STATUS "Enabling moe extension.")
define_gpu_extension_target(
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

  define_gpu_extension_target(
    _rocm_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_ROCM_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    USE_SABI 3
    WITH_SOABI)
endif()

# For CUDA we also build and ship some external projects.
if (VLLM_GPU_LANG STREQUAL "CUDA")
    include(cmake/external_projects/flashmla.cmake)
    include(cmake/external_projects/vllm_flash_attn.cmake)
endif ()
