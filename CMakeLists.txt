cmake_minimum_required(VERSION 3.21)

project(vllm_extensions LANGUAGES CXX)

# add comment why it comes before append_cmake_prefix_path
# add comment why we need to do this multiple times (TODO: maybe use a loop?)
find_package(Python 3.8 EXACT COMPONENTS Interpreter Development.Module)
if (NOT Python_FOUND)
  find_package(Python 3.8...3.11 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

if(NOT DEFINED PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()


find_package(MPI)

#
# Find where user site-packages and torch are installed and add it to cmake's search path.
#

# add comment
macro (append_cmake_prefix_path PKG EXPR)
  execute_process(
    COMMAND
    "${PYTHON_EXECUTABLE}" "-c" "import ${PKG}; print(${EXPR})"
    OUTPUT_VARIABLE PREFIX_PATH
    ERROR_VARIABLE PREFIX_PATH_ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(PREFIX_PATH STREQUAL "")
    message(FATAL_ERROR "Failed to locate ${PKG} path,"
            " full error message:\n${PREFIX_PATH_ERR}")
  endif()

  list(APPEND CMAKE_PREFIX_PATH ${PREFIX_PATH})
endmacro()

append_cmake_prefix_path("site" "site.getusersitepackages()")
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")


find_package(Torch 2.1.2 EXACT REQUIRED)
append_torchlib_if_found(torch_python)

#
# Setup extra NVCC flags
#

# TODO: IS_CUDA only?

# add comment
execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" "-c"
      "import torch.utils.cpp_extension as torch_cpp_ext; print(';'.join(torch_cpp_ext.COMMON_NVCC_FLAGS))"
    OUTPUT_VARIABLE TORCH_NVCC_FLAGS
    ERROR_VARIABLE TORCH_NVCC_FLAGS_ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(TORCH_NVCC_FLAGS STREQUAL "")
  message(FATAL_ERROR "Unable to determine torch nvcc compiler flags,"
                      " full error message:\n${TORCH_NVCC_FLAGS_ERR}")
endif()

set(VLLM_NVCC_FLAGS ${TORCH_NVCC_FLAGS})

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

foreach(OPT
    "-D__CUDA_NO_HALF_OPERATORS__"
    "-D__CUDA_NO_HALF_CONVERSIONS__"
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
    "-D__CUDA_NO_HALF2_OPERATORS__"
  )
  list(REMOVE_ITEM VLLM_PUNICA_NVCC_FLAGS ${OPT})
endforeach()

#
# deal with arch flags here
#

#
# CUDA_NVCC_FLAGS holds the complete + canonical flags at this point
# make two versions: regular, punica
# strip out stuff from punica + update versions
#

message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")


# remove gencode flags added by pytorch
#list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX "-gencode")
#list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX "arch=compute.*")
#string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})


#
# Setup arch flags
#

string(REGEX MATCHALL "-gencode arch=[^ ]+" VLLM_CUDA_ARCH_FLAGS ${CMAKE_CUDA_FLAGS})
string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
message("arch flags: ${VLLM_CUDA_ARCH_FLAGS}")
# filter ARCH_FLAGS and add them back into CMAKE_CUDA_FLAGS

#set(VLLM_PUNICA_CUDA_ARCH_FLAGS ${VLLM_CUDA_ARCH_FLAGS})
#list(FILTER VLLM_PUNICA_CUDA_ARCH_FLAGS EXCLUDE REGEX "compute_[1-7][0-9]")

message("post arch flags: ${VLLM_CUDA_ARCH_FLAGS}")
message("post punica arch flags: ${VLLM_PUNICA_CUDA_ARCH_FLAGS}")

message("nvcc: ${VLLM_NVCC_FLAGS}")
message("punica nvcc: ${VLLM_PUNICA_NVCC_FLAGS}")

#list(APPEND VLLM_NVCC_FLAGS ${VLLM_CUDA_ARCH_FLAGS})
#list(APPEND VLLM_PUNICA_NVCC_FLAGS ${VLLM_PUNICA_CUDA_ARCH_FLAGS})

# the painful way: NOTE needs to only happen w/CUDA
set(VLLM_CUDA_ARCHES)
set(VLLM_PUNICA_CUDA_ARCHES)

macro(string_to_ver VER STR)
  string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" ${VER} ${STR})
endmacro()

set(NVIDIA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")
set(ROCM_SUPPORTED_ARCHS "gfx90a;gfx942;gfx1100")

foreach(ARCH ${VLLM_CUDA_ARCH_FLAGS})
  string(REGEX MATCH "arch=compute_\([0-9]+a*\)" COMPUTE ${ARCH})
  if (COMPUTE)
    set(COMPUTE ${CMAKE_MATCH_1})
#    message("arch: ${COMPUTE}")
  endif()

  string(REGEX MATCH "code=sm_\([0-9]+a*\)" SM ${ARCH})
  if (SM)
    set(SM ${CMAKE_MATCH_1})
#    message("sm: ${SM}")
  endif()

  string(REGEX MATCH "code=compute_\([0-9]+a*\)" CODE ${ARCH})
  if (CODE)
    set(CODE ${CMAKE_MATCH_1})
#    message("code: ${CODE}")
  endif()

  if (COMPUTE AND SM)
    string_to_ver(SM_VER ${SM})
    if (NOT SM_VER IN_LIST NVIDIA_SUPPORTED_ARCHS)
      # TODO: issue warning?
      continue()
    endif()
    list(APPEND VLLM_CUDA_ARCHES ${SM})
    if (SM_VER GREATER_EQUAL 8.0)
      list(APPEND VLLM_PUNICA_CUDA_ARCHES ${SM})
    endif()
  else()
    string_to_ver(CODE_VER ${CODE})
    if (NOT CODE_VER IN_LIST NVIDIA_SUPPORTED_ARCHS)
      # TODO: issue warning?
      continue()
    endif()
    list(APPEND VLLM_CUDA_ARCHES "${CODE}-virtual")
    if (CODE_VER GREATER_EQUAL 8.0)
      list(APPEND VLLM_PUNICA_CUDA_ARCHES "${CODE}-virtual")
    endif()
  endif()
endforeach()

message("post nvcc: ${VLLM_NVCC_FLAGS}")
message("post punica nvcc: ${VLLM_PUNICA_NVCC_FLAGS}")
message("post nvcc arch: ${VLLM_CUDA_ARCHES}")
message("post punica arch: ${VLLM_PUNICA_CUDA_ARCHES}")


#
# Check for existence of CUDA/HIP language support
#
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
#include(CheckLanguage)
#check_language(HIP)
#check_language(CUDA)  # picked up by torch

# Note: CUDA + HIP are detected by pytorch package so there's no need to repeat them.

if(HIP_FOUND)
  enable_language(HIP)
  list(APPEND VLLM_NVCC_FLAGS "-DUSE_ROCM" "-U__HIP_NO_HALF_CONVERSIONS__" "-U__HIP_NO_HALF_OPERATORS__")

  # TODO: intersect with this list?
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(VLLM_SUPPORTED_HIP_ARCHITECTURES "gfx90a;gfx942;gfx1100")
  endif()

  foreach(HIP_ARCH ${CMAKE_HIP_ARCHITECTURES})
    list(APPEND VLLM_NVCC_FLAGS "--offload-arch=${HIP_ARCH}")
  endforeach()
elseif(CUDA_FOUND)
  enable_language(CUDA)
  set(IS_CUDA true)

  # TODO: check supported?

else()
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()

#
# Define target source files
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

#TODO: list files
File(GLOB VLLM_MOE_EXT_SRC "csrc/moe/*.cu" "csrc/moe/*.cpp")
File(GLOB VLLM_PUNICA_EXT_SRC "csrc/punica/bgmv/*.cu" "csrc/punica/*.cpp")

#
# Define targets
#
set(CMAKE_CXX_STANDARD 17)

# add comment
function(define_module_target MOD_NAME MOD_SRC MOD_EXTRA_NVCC_FLAGS MOD_CUDA_ARCHES)
  Python_add_library(${MOD_NAME} MODULE ${MOD_SRC} WITH_SOABI)

  set_target_properties(${MOD_NAME} PROPERTIES CUDA_ARCHITECTURES "${MOD_CUDA_ARCHES}")

  if (IS_CUDA)
    set(CUDA_LANG "CUDA")
  else()
    set(CUDA_LANG "HIP")
  endif()

  # Note: optimization level/debug info is set by build type
  #  target_compile_options(${MOD_NAME} BEFORE PRIVATE $<$<COMPILE_LANGUAGE:${CUDA_LANG}>:${MOD_EXTRA_NVCC_FLAGS}>)
  target_compile_options(${MOD_NAME} PRIVATE $<$<COMPILE_LANGUAGE:${CUDA_LANG}>:${MOD_EXTRA_NVCC_FLAGS}>)

#  get_target_property(XXX ${MOD_NAME} COMPILE_OPTIONS)
#  message("XXX: ${XXX}")
#  get_target_property(XXX ${MOD_NAME} COMPILE_FEATURES)
#  message("XXX: ${XXX}")

  target_compile_definitions(${MOD_NAME} PRIVATE "-DTORCH_EXTENSION_NAME=${MOD_NAME}")
  target_include_directories(${MOD_NAME} PRIVATE csrc PRIVATE ${TORCH_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(${MOD_NAME} PRIVATE ${TORCH_LIBRARIES})
  install(TARGETS ${MOD_NAME} LIBRARY DESTINATION vllm)
endfunction()

define_module_target(_C "${VLLM_EXT_SRC}" "${VLLM_NVCC_FLAGS}" "${VLLM_CUDA_ARCHES}")
define_module_target(_moe_C "${VLLM_MOE_EXT_SRC}" "${VLLM_NVCC_FLAGS}" "${VLLM_CUDA_ARCHES}")
define_module_target(_punica_C "${VLLM_PUNICA_EXT_SRC}" "${VLLM_PUNICA_NVCC_FLAGS}" "${VLLM_PUNICA_CUDA_ARCHES}")

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
