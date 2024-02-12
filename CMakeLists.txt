cmake_minimum_required(VERSION 3.21)

project(vllm_extensions LANGUAGES CXX)

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

# add comment why it comes before append_cmake_prefix_path
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(MPI)

#
# Find where user site-packages and torch are installed and add it to cmake's search path.
# Find packages needed to compile
#
append_cmake_prefix_path("site" "site.getusersitepackages()")
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")


find_package(Torch 2.1.2 EXACT REQUIRED)
append_torchlib_if_found(torch_python)

#set(ENV{TORCH_CUDA_ARCH_LIST} "70;75;80;86;89;90")

#
# Setup NVCC flags
#

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

set(NVCC_FLAGS ${TORCH_NVCC_FLAGS})

if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
  list(APPEND NVCC_FLAGS "-DENABLE_FP8_E5M2")
endif()

if(NVCC_THREADS)
  list(APPEND NVCC_FLAGS "--threads=${NVCC_THREADS}")
endif()

#
# Copy flags+update for punica
#
set(PUNICA_NVCC_FLAGS ${NVCC_FLAGS})
foreach(OPT
    "-D__CUDA_NO_HALF_OPERATORS__"
    "-D__CUDA_NO_HALF_CONVERSIONS__"
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
    "-D__CUDA_NO_HALF2_OPERATORS__"
  )
  list(REMOVE_ITEM PUNICA_NVCC_FLAGS ${OPT})
endforeach()

# remove gencode flags added by pytorch
list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX "-gencode")
list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX "arch=compute.*")
#list(FILTER CMAKE_CUDA_FLAGS EXCLUDE REGEX "-gencode")
#list(FILTER CMAKE_CUDA_FLAGS EXCLUDE REGEX "arch=compute.*")
string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})


message("torch nvcc: ${CUDA_NVCC_FLAGS}")
message("torch cuda_flags: ${CMAKE_CUDA_FLAGS}")
message("nvcc: ${NVCC_FLAGS}")
message("punica nvcc: ${PUNICA_NVCC_FLAGS}")

#
# Check for existence of CUDA/HIP language support
#
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
include(CheckLanguage)
check_language(HIP)
#check_language(CUDA)  # picked up by torch

if(NOT CMAKE_HIP_COMPILER STREQUAL "NOTFOUND")
  enable_language(HIP)
  list(APPEND NVCC_FLAGS "-DUSE_ROCM" "-U__HIP_NO_HALF_CONVERSIONS__" "-U__HIP_NO_HALF_OPERATORS__")

  # TODO: intersect with this list?
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx942;gfx1100")
  endif()

  foreach(HIP_ARCH ${CMAKE_HIP_ARCHITECTURES})
    list(APPEND NVCC_FLAGS "--offload-arch=${HIP_ARCH}")
  endforeach()
elseif(NOT CMAKE_CUDA_COMPILER STREQUAL "NOTFOUND")
  enable_language(CUDA)
  set(IS_CUDA true)

  # TODO: parse TORCH_CUDA_ARCH_LIST -> CMAKE_CUDA_ARCHITECTURES?
  # cmake env var CUDAARCHS

  # https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
  # set_target_properties(tgt PROPERTIES CUDA_ARCHITECTURES "35;50;72")
  # TODO: PTX stuff
#  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#    # This indicates support for both real architectures (i.e, no ptx).
#    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
#  endif()

  set(VLLM_CUDA_ARCHES "70;75;80;86;89;90")
  set(VLLM_PUNICA_CUDA_ARCHES "80;86;89;90")  # >8.0 of VLLM_CUDA_ARCHITECTURES

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

File(GLOB VLLM_MOE_EXT_SRC "csrc/moe/*.cu" "csrc/moe/*.cpp")
File(GLOB VLLM_PUNICA_EXT_SRC "csrc/punica/bgmv/*.cu" "csrc/punica/*.cpp")

#
# Define targets
#
set(CMAKE_CXX_STANDARD 17)

#this doesn't seem to work
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# add comment
function(define_module_target MOD_NAME MOD_SRC MOD_NVCC_FLAGS MOD_CUDA_ARCHES)
  Python_add_library(${MOD_NAME} MODULE ${MOD_SRC} WITH_SOABI)
  #add_library(${MOD_NAME} MODULE ${MOD_SRC})
  set_target_properties(${MOD_NAME} PROPERTIES CUDA_ARCHITECTURES "${MOD_CUDA_ARCHES}")
  # Note: optimization level/debug info is set by build type
  if (IS_CUDA)
    set(CUDA_LANG "CUDA")
#    target_compile_options(${MOD_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MOD_NVCC_FLAGS}>)
  else()
    set(CUDA_LANG "HIP")
#    target_compile_options(${MOD_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${MOD_NVCC_FLAGS}>)
  endif()
  target_compile_options(${MOD_NAME} PRIVATE $<$<COMPILE_LANGUAGE:${CUDA_LANG}>:${MOD_NVCC_FLAGS}>)
  target_compile_definitions(${MOD_NAME} PRIVATE "-DTORCH_EXTENSION_NAME=${MOD_NAME}")
  target_include_directories(${MOD_NAME} PRIVATE csrc PRIVATE ${TORCH_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(${MOD_NAME} PRIVATE ${TORCH_LIBRARIES})
  install(TARGETS ${MOD_NAME} LIBRARY DESTINATION vllm)
endfunction()

define_module_target(_C "${VLLM_EXT_SRC}" "${NVCC_FLAGS}" "${VLLM_CUDA_ARCHES}")
define_module_target(_moe_C "${VLLM_MOE_EXT_SRC}" "${NVCC_FLAGS}" "${VLLM_CUDA_ARCHES}")
define_module_target(_punica_C "${VLLM_PUNICA_EXT_SRC}" "${PUNICA_NVCC_FLAGS}" "${VLLM_PUNICA_CUDA_ARCHES}")

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
