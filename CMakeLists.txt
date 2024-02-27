cmake_minimum_required(VERSION 3.21)

project(vllm_extensions LANGUAGES CXX)

include(${CMAKE_SOURCE_DIR}/utils.cmake)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

# Supported NVIDIA architectures.
set(NVIDIA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

# Supported AMD GPU architectures.
set(ROCM_SUPPORTED_ARCHS "gfx908;gfx90a;gfx942;gfx1100")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from Dockerfile.rocm
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.1.2")
set(TORCH_SUPPORTED_VERSION_ROCM_5X "2.0.1")
set(TORCH_SUPPORTED_VERSION_ROCM_6X "2.1.1")

#
# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the desired python version before "
    "running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

# For some reason torch does not add libtorch_python.so to the list of torch
# libraries to link.  Find it by hand using `append_torchlib_if_found` from
# torch's cmake setup.
append_torchlib_if_found(torch_python)

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

  # ROCm 5.x
  if (ROCM_VERSION_DEV_MAJOR EQUAL 5 AND
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM_5X})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_ROCM_5X} "
      "expected for ROCMm 5.x build, saw ${Torch_VERSION} instead.")
  endif()

  # ROCm 6.x
  if (ROCM_VERSION_DEV_MAJOR EQUAL 6 AND
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM_6X})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_ROCM_6X} "
      "expected for ROCMm 6.x build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()

#
# Setup extra platform specific GPU compilation flags, e.g. NVCC flags for CUDA
# and hip flags for ROCm.
#
# Note: CUDA (and partially HIP) is detected by pytorch package so there's no
# need to repeat detecting it explicitly with check_language, etc.
#
if (HIP_FOUND)
  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  #
  # `VLLM_HIP_ARCHITECUTRES` controls the `--offload-arch` flags.
  # `CMAKE_HIP_ARCHITECTURES` is set up by torch and can be controlled
  # via the `PYTORCH_ROCM_ARCH` env variable.
  #

  #
  # Find the intersection of the supported + detected architectures to
  # set the module architecture flags.
  #
  set(VLLM_HIP_ARCHITECTURES)
  foreach (ARCH ${CMAKE_HIP_ARCHITECTURES})
    if (ARCH IN_LIST ROCM_SUPPORTED_ARCHS)
      list(APPEND VLLM_HIP_ARCHITECTURES ${ARCH})
    endif()
  endforeach()

  if(NOT VLLM_HIP_ARCHITECTURES)
    message(FATAL_ERROR
      "None of the detected ROCm architectures: ${CMAKE_HIP_ARCHITECTURES} is"
      " supported. Supported ROCm architectures are: ${ROCM_SUPPORTED_ARCHS}.")
  endif()

  set(VLLM_GPU_ARCHES ${VLLM_HIP_ARCHITECTURES})
  set(VLLM_PUNICA_GPU_ARCHES ${VLLM_HIP_ARCHITECTURES})

  # Get common HIP/HIPCC flags from torch.
  run_python(VLLM_GPU_FLAGS
    "from torch.utils.cpp_extension import COMMON_HIP_FLAGS; print(';'.join(COMMON_HIP_FLAGS))"
    "Failed to determine torch nvcc compiler flags")

  run_python(VLLM_HIPCC_FLAGS
    "from torch.utils.cpp_extension import COMMON_HIPCC_FLAGS; print(';'.join(COMMON_HIPCC_FLAGS))"
    "Failed to determine torch nvcc compiler flags")

  list(APPEND VLLM_GPU_FLAGS
    ${VLLM_HIPCC_FLAGS}
    "-DUSE_ROCM"
    "-U__HIP_NO_HALF_CONVERSIONS__"
    "-U__HIP_NO_HALF_OPERATORS__"
    "-fno-gpu-rdc")

else()
  #
  # Get common NVCC flags from torch.
  #
  run_python(VLLM_GPU_FLAGS
    "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
    "Failed to determine torch nvcc compiler flags")

  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_FP8_E5M2")
  endif()

  if(NVCC_THREADS)
    list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
  endif()

  set(VLLM_PUNICA_GPU_FLAGS ${VLLM_GPU_FLAGS})

  #
  # Copy flags+update for punica
  #
  list(REMOVE_ITEM VLLM_PUNICA_GPU_FLAGS
    "-D__CUDA_NO_HALF_OPERATORS__"
    "-D__CUDA_NO_HALF_CONVERSIONS__"
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
    "-D__CUDA_NO_HALF2_OPERATORS__")

  #
  # Setup/process CUDA arch flags.
  #
  # The torch cmake setup hardcodes the detected architecture flags in
  # `CMAKE_CUDA_FLAGS`.  Since `CMAKE_CUDA_FLAGS` is a "global" variable, it
  # can't modified on a per-target basis, e.g. for the `punica` extension.
  # So, all the `-gencode` flags need to be extracted and removed from
  # `CMAKE_CUDA_FLAGS` for processing so they can be passed by another method.
  # Since it's not possible to use `target_compiler_options` for adding target
  # specific `-gencode` arguments, the target's `CUDA_ARCHITECTURES` property
  # must be used instead.  This requires repackaging the architecture flags
  # into a format that cmake expects for `CUDA_ARCHITECTURES`.
  #
  # This is a bit fragile in that it depends on torch using `-gencode` as opposed
  # to one of the other nvcc options to specify architectures.
  #
  # Note: torch uses the `TORCH_CUDA_ARCH_LIST` environment variable to override
  # detected architectures.
  #
  message(DEBUG "initial CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

  # Extract all `-gencode` flags from `CMAKE_CUDA_FLAGS`
  string(REGEX MATCHALL "-gencode arch=[^ ]+" VLLM_CUDA_ARCH_FLAGS
    ${CMAKE_CUDA_FLAGS})

  # Remove all `-gencode` flags from `CMAKE_CUDA_FLAGS` since they will be modified
  # and passed back via the `CUDA_ARCHITECTURES` property.
  string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS
    ${CMAKE_CUDA_FLAGS})

  # If this error is triggered, it might mean that torch has changed how it sets
  # up nvcc architecture code generation flags.
  if (NOT VLLM_CUDA_ARCH_FLAGS)
    message(FATAL_ERROR
      "Could not find any architecture related code generation flags in "
      "CMAKE_CUDA_FLAGS. (${CMAKE_CUDA_FLAGS})")
  endif()

  message(DEBUG "final CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
  message(DEBUG "arch flags: ${VLLM_CUDA_ARCH_FLAGS}")

  # Macro for converting a `gencode` version number to a cmake version number.
  macro(string_to_ver OUT_VER IN_STR)
    string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" ${OUT_VER} ${IN_STR})
  endmacro()

  # Initialize the architecture lists to empty.
  set(VLLM_GPU_ARCHES)
  set(VLLM_PUNICA_GPU_ARCHES)

  # Process each `gencode` flag.
  foreach(ARCH ${VLLM_CUDA_ARCH_FLAGS})
    # For each flag, extract the version number and whether it refers to PTX
    # or native code.
    # Note: if a regex matches then `CMAKE_MATCH_1` holds the binding
    # for that match.

    string(REGEX MATCH "arch=compute_\([0-9]+a?\)" COMPUTE ${ARCH})
    if (COMPUTE)
      set(COMPUTE ${CMAKE_MATCH_1})
    endif()

    string(REGEX MATCH "code=sm_\([0-9]+a?\)" SM ${ARCH})
    if (SM)
      set(SM ${CMAKE_MATCH_1})
    endif()

    string(REGEX MATCH "code=compute_\([0-9]+a?\)" CODE ${ARCH})
    if (CODE)
      set(CODE ${CMAKE_MATCH_1})
    endif()

    # Make sure the virtual architecture can be matched.
    if (NOT COMPUTE)
      message(FATAL_ERROR
        "Could not determine virtual architecture from: ${ARCH}.")
    endif()

    # One of sm_ or compute_ must exist.
    if ((NOT SM) AND (NOT CODE))
      message(FATAL_ERROR
        "Could not determine a codegen architecture from: ${ARCH}.")
    endif()

    if (SM)
      set(VIRT "")
      set(CODE_ARCH ${SM})
    else()
      set(VIRT "-virtual")
      set(CODE_ARCH ${CODE})
    endif()

    # Check if the current version is in the supported arch list.
    string_to_ver(CODE_VER ${CODE_ARCH})
    if (NOT CODE_VER IN_LIST NVIDIA_SUPPORTED_ARCHS)
      message(STATUS "discarding unsupported CUDA arch ${VER}.")
      continue()
    endif()

    # Add it to the arch list.
    list(APPEND VLLM_GPU_ARCHES "${CODE_ARCH}${VIRT}")

    # Add it to punica arch list if the version is >= 8.0.
    if (CODE_VER GREATER_EQUAL 8.0)
      list(APPEND VLLM_PUNICA_GPU_ARCHES "${CODE_ARCH}${VIRT}")
    endif()
  endforeach()

  message(DEBUG "nvcc arch: ${VLLM_GPU_ARCHES}")
  message(DEBUG "punica arch: ${VLLM_PUNICA_GPU_ARCHES}")
endif()

#
# Define extension targets
#

#
# _C extension
#

set(VLLM_EXT_SRC
  "csrc/cache_kernels.cu"
  "csrc/attention/attention_kernels.cu"
  "csrc/pos_encoding_kernels.cu"
  "csrc/activation_kernels.cu"
  "csrc/layernorm_kernels.cu"
  "csrc/quantization/squeezellm/quant_cuda_kernel.cu"
  "csrc/quantization/gptq/q_gemm.cu"
  "csrc/cuda_utils_kernels.cu"
  "csrc/moe_align_block_size_kernels.cu"
  "csrc/pybind.cpp")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_EXT_SRC
    "csrc/quantization/awq/gemm_kernels.cu"
    "csrc/custom_all_reduce.cu")
endif()

define_gpu_extension_target(
  _C
  vllm
  "${VLLM_GPU_LANG}"
  "${VLLM_EXT_SRC}"
  "${VLLM_GPU_FLAGS}"
  "${VLLM_GPU_ARCHES}")

#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/moe/moe_ops.cpp"
  "csrc/moe/topk_softmax_kernels.cu")

define_gpu_extension_target(
  _moe_C
  vllm
  "${VLLM_GPU_LANG}"
  "${VLLM_MOE_EXT_SRC}"
  "${VLLM_GPU_FLAGS}"
  "${VLLM_GPU_ARCHES}")

#
# _punica_C extension
#

set(VLLM_PUNICA_EXT_SRC
  "csrc/punica/bgmv/bgmv_bf16_bf16_bf16.cu"
  "csrc/punica/bgmv/bgmv_bf16_bf16_fp16.cu"
  "csrc/punica/bgmv/bgmv_bf16_fp16_bf16.cu"
  "csrc/punica/bgmv/bgmv_bf16_fp16_fp16.cu"
  "csrc/punica/bgmv/bgmv_bf16_fp32_bf16.cu"
  "csrc/punica/bgmv/bgmv_bf16_fp32_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp16_bf16_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp16_bf16_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp16_fp16_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp16_fp16_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp16_fp32_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp16_fp32_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp32_bf16_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp32_bf16_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp32_fp16_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp32_fp16_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp32_fp32_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp32_fp32_fp16.cu"
  "csrc/punica/punica_ops.cc")

define_gpu_extension_target(
  _punica_C
  vllm
  "${VLLM_GPU_LANG}"
  "${VLLM_PUNICA_EXT_SRC}"
  "${VLLM_PUNICA_GPU_FLAGS}"
  "${VLLM_PUNICA_GPU_ARCHES}")

#
# Add the `default` target which detects which extensions should be
# built based on platform/architecture.  This is the same logic that
# setup.py uses to select which extensions should be built and should
# be kept in sync.
#
# The `default` target makes direct use of cmake easier since knowledge
# of which extensions are supported has been factored in, e.g.
#
# mkdir build && cd build
# cmake -G Ninja ..
# cmake --build . --target default
#
add_custom_target(default)

if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
  message(STATUS "Enabling C extension.")
  add_dependencies(default _C)
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
  message(STATUS "Enabling moe extension.")
  add_dependencies(default _moe_C)

  set(ENABLE_PUNICA)
  # Enable punica if -DVLLM_INSTALL_PUNICA_KERNELS=1 or
  # VLLM_INSTALL_PUNICA_KERNELS is set in the environment.
  if (DEFINED VLLM_INSTALL_PUNICA_KERNELS OR ENV{VLLM_INSTALL_PUNICA_KERNELS})
    set(ENABLE_PUNICA true)
    foreach (ARCH ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
      string(REGEX MATCH "\([0-9]+\)" ARCH_VER_STR ${ARCH})
      string_to_ver(ARCH_VER ${ARCH_VER_STR})
      if (ARCH_VER VERSION_LESS 8.0)
        message(STATUS
          "Unable to add punica extension due to device version ${ARCH_VER} < 8.0.")
        set(ENABLE_PUNICA false)
        break()
      endif()
    endforeach()
  endif()

  if (ENABLE_PUNICA)
    message(STATUS "Enabling punica extension.")
    add_dependencies(default _punica_C)
  endif()
endif()
