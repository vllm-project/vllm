cmake_minimum_required(VERSION 3.21)

project(vllm_extensions LANGUAGES CXX)

#
# Supported python verions.  These versions will be searched in order, the
# first match will be selected.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

# Supported NVIDIA architectures
set(NVIDIA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

# Supported AMD GPU architectures
set(ROCM_SUPPORTED_ARCHS "gfx90a;gfx942;gfx1100")

#
# Loop thru all supported python versions until we find the first match.
# Cmake is unable to pick the lowest supported version when multiple
# versions are available, even with CMAKE_FIND_PACKAGE_SORT_ORDER.
#
foreach(VER ${PYTHON_SUPPORTED_VERSIONS})
  message(STATUS "Attempting to find python ${VER} package.")
  find_package(Python ${VER} EXACT COMPONENTS Interpreter Development.Module)
  if (Python_FOUND)
    message(STATUS "Found python version ${VER} (${Python_EXECUTABLE}).")
    break()
  endif()
endforeach()

if (NOT Python_FOUND)
  message(FATAL_ERROR
    "No supported version of python found. ('${PYTHON_SUPPORTED_VERSIONS}')")
endif()

#
# Run EXPR in python.  The standard output of python is stored in OUT and has
# trailing whitespace stripped.  If an error is encountered when running python,
# a fatal message ERR_MSG is issued.
#
macro (run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "${Python_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE ${OUT}
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
endmacro()

#
# Try to find MPI package
#
find_package(MPI)

#
# Find where user site-packages and torch are installed and add it to cmake's
# search path.
#

# Run EXPR in python after importing PKG. Use the result of this to extend
# CMAKE_PREFIX_PATH so we can import the torch cmake configuration.
macro (append_cmake_prefix_path PKG EXPR)
  run_python(PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${PREFIX_PATH})
endmacro()

# Add user site-packages and torch path to CMAKE_PREFIX_PATH
append_cmake_prefix_path("site" "site.getusersitepackages()")
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

#
# Import torch cmake configuration.
# Torch also imports CUDA/HIP packages with some customizations, so we do not
# need to do this explicitly with check_language/enable_language, etc.
#
find_package(Torch 2.1.2 EXACT REQUIRED)

# For some reason torch does not add libtorch_python.so to the list of torch
# libraries to link.  Find it by hand using 'append_torchlib_if_found' from
# torch's cmake setup.
append_torchlib_if_found(torch_python)

if ((NOT HIP_FOUND) AND (NOT CUDA_FOUND))
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()

if (NOT HIP_FOUND AND CUDA_FOUND)
  set(IS_CUDA true)
endif()

#
# Setup extra NVCC flags
#
# Note: CUDA + HIP are detected by pytorch package so there's no need to repeat
# detect them explicitly with check_language, etc.
#
if (HIP_FOUND)
  list(APPEND VLLM_NVCC_FLAGS
    "-DUSE_ROCM"
    "-U__HIP_NO_HALF_CONVERSIONS__"
    "-U__HIP_NO_HALF_OPERATORS__")

  # TODO: intersect with this list?
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(VLLM_SUPPORTED_HIP_ARCHITECTURES "gfx90a;gfx942;gfx1100")
  endif()

  foreach(HIP_ARCH ${CMAKE_HIP_ARCHITECTURES})
    list(APPEND VLLM_NVCC_FLAGS "--offload-arch=${HIP_ARCH}")
  endforeach()
endif()

# TODO: IS_CUDA only?

# Get common NVCC flags from torch.
run_python(VLLM_NVCC_FLAGS
  "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
  "Failed to determine torch nvcc compiler flags")

if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
  list(APPEND VLLM_NVCC_FLAGS "-DENABLE_FP8_E5M2")
endif()

if(NVCC_THREADS)
  list(APPEND VLLM_NVCC_FLAGS "--threads=${NVCC_THREADS}")
endif()

set(VLLM_PUNICA_NVCC_FLAGS ${VLLM_NVCC_FLAGS})

#
# Copy flags+update for punica
#

list(REMOVE_ITEM VLLM_PUNICA_NVCC_FLAGS
  "-D__CUDA_NO_HALF_OPERATORS__"
  "-D__CUDA_NO_HALF_CONVERSIONS__"
  "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
  "-D__CUDA_NO_HALF2_OPERATORS__")

message(DEBUG "nvcc: ${VLLM_NVCC_FLAGS}")
message(DEBUG "punica nvcc: ${VLLM_PUNICA_NVCC_FLAGS}")

#
# Setup/process CUDA arch flags
#
# The torch cmake setup detects and hardcodes the detected architecture flags
# in CMAKE_CUDA_FLAGS but we will need to filter/modify them for the supported
# architectures and the punica target.  So we have to extract and remove all
# the '-gencode' flags from CMAKE_CUDA_FLAGS for processing.  We can't use
# 'target_compiler_options' for adding '-gencode' arguments so we will use the
# target's CUDA_ARCHITECTURES property instead. This requires repackaging
# the architecture flags into a format that cmake expects for
# CUDA_ARCHITECTURES.
#
# This is a bit fragile in that it depends on torch using -gencode as opposed
# to one of the other nvcc options to specify architectures.
#
# Note: torch uses the TORCH_CUDA_ARCH_LIST environment variable to override
# detected architectures.
#
message(DEBUG "initial CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# Extract all '-gencode' flags from CMAKE_CUDA_FLAGS
string(REGEX MATCHALL "-gencode arch=[^ ]+" VLLM_CUDA_ARCH_FLAGS
  ${CMAKE_CUDA_FLAGS})

# Remove all '-gencode' flags from CMAKE_CUDA_FLAGS since we will be modifying
# them and passing them back in via the CUDA_ARCHITECTURES property.
string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS
  ${CMAKE_CUDA_FLAGS})

# If this error is triggered, it might mean that torch has changed how it sets
# up nvcc architecture code generation flags.
if (NOT VLLM_CUDA_ARCH_FLAGS)
  message(FATAL_ERROR
    "Could not find any architecture related code generation flags in "
    "CMAKE_CUDA_FLAGS. (${CMAKE_CUDA_FLAGS})")
endif()

message(DEBUG "arch flags: ${VLLM_CUDA_ARCH_FLAGS}")

# Macro for converting a 'gencode' version number to a cmake version number.
macro(string_to_ver OUT_VER IN_STR)
  string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" ${OUT_VER} ${IN_STR})
endmacro()

# Initialize the architecure lists to empty.
set(VLLM_CUDA_ARCHES)
set(VLLM_PUNICA_CUDA_ARCHES)

# Process each 'gencode' flag.
foreach(ARCH ${VLLM_CUDA_ARCH_FLAGS})
  # For each flag we want to extract the version number and whether
  # it refers to PTX or native code.
  # Note: if a regex matches then CMAKE_MATCH_1 will hold the binding
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

  # Check if the current version is in the supported arch list
  string_to_ver(CODE_VER ${CODE_ARCH})
  if (NOT CODE_VER IN_LIST NVIDIA_SUPPORTED_ARCHS)
    message(STATUS "discarding unsupported CUDA arch ${VER}.")
    continue()
  endif()

  # Add it to the arch list
  list(APPEND VLLM_CUDA_ARCHES "${CODE_ARCH}${VIRT}")

  # Add it to punica arch list if the version is >= 8.0
  if (CODE_VER GREATER_EQUAL 8.0)
    list(APPEND VLLM_PUNICA_CUDA_ARCHES "${CODE_ARCH}${VIRT}")
  endif()
endforeach()

message(DEBUG "nvcc arch: ${VLLM_CUDA_ARCHES}")
message(DEBUG "punica arch: ${VLLM_PUNICA_CUDA_ARCHES}")

#
# Define targets
#

# add comment
# Note: optimization level/debug info is set via cmake build type.
function(define_module_target MOD_NAME MOD_SRC MOD_EXTRA_NVCC_FLAGS
    MOD_CUDA_ARCHES)

  Python_add_library(${MOD_NAME} MODULE ${MOD_SRC} WITH_SOABI)

  # TODO: needed for rocm?
  if (IS_CUDA)
    set(CUDA_LANG "CUDA")
  else()
    set(CUDA_LANG "HIP")
  endif()

  set_property(TARGET ${MOD_NAME} PROPERTY CXX_STANDARD 17)

  set_target_properties(${MOD_NAME} PROPERTIES CUDA_ARCHITECTURES
    "${MOD_CUDA_ARCHES}")

  target_compile_options(${MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:${CUDA_LANG}>:${MOD_EXTRA_NVCC_FLAGS}>)

  target_compile_definitions(${MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${MOD_NAME}")

  target_include_directories(${MOD_NAME} PRIVATE
    csrc PRIVATE ${TORCH_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})

  target_link_libraries(${MOD_NAME} PRIVATE ${TORCH_LIBRARIES})

  install(TARGETS ${MOD_NAME} LIBRARY DESTINATION vllm)
endfunction()

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

if(IS_CUDA)
  list(APPEND VLLM_EXT_SRC
    "csrc/quantization/awq/gemm_kernels.cu"
    "csrc/custom_all_reduce.cu")
endif()

define_module_target(_C
  "${VLLM_EXT_SRC}"
  "${VLLM_NVCC_FLAGS}"
  "${VLLM_CUDA_ARCHES}")

#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/moe/moe_ops.cpp"
  "csrc/moe/topk_softmax_kernels.cu")

define_module_target(_moe_C
  "${VLLM_MOE_EXT_SRC}"
  "${VLLM_NVCC_FLAGS}"
  "${VLLM_CUDA_ARCHES}")

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

define_module_target(_punica_C
  "${VLLM_PUNICA_EXT_SRC}"
  "${VLLM_PUNICA_NVCC_FLAGS}"
  "${VLLM_PUNICA_CUDA_ARCHES}")
