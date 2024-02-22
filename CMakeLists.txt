cmake_minimum_required(VERSION 3.21)

project(vllm_extensions LANGUAGES CXX)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

# Supported NVIDIA architectures
set(NVIDIA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

# Supported AMD GPU architectures
set(ROCM_SUPPORTED_ARCHS "gfx908;gfx90a;gfx942;gfx1100")

#
# Loop thru all supported python versions until we find the first suitable
# version that has torch installed.
#
# Cmake is unable to pick the lowest supported version when multiple
# versions are available, even with CMAKE_FIND_PACKAGE_SORT_ORDER.
#
foreach(VER ${PYTHON_SUPPORTED_VERSIONS})
  message(STATUS "Attempting to find python ${VER} package.")
  find_package(Python ${VER} COMPONENTS Interpreter Development.Module)
  if (Python_FOUND)
    execute_process(
      COMMAND
      "${Python_EXECUTABLE}" "-c" "import torch"
      OUTPUT_VARIABLE PYTHON_OUT
      RESULT_VARIABLE PYTHON_ERROR_CODE
      ERROR_VARIABLE PYTHON_STDERR)

    if(PYTHON_ERROR_CODE EQUAL 0)
      message(STATUS "Found python version ${Python_VERSION} (${Python_EXECUTABLE}).")
      break()
    endif()
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
find_package(Torch REQUIRED)

# For some reason torch does not add libtorch_python.so to the list of torch
# libraries to link.  Find it by hand using 'append_torchlib_if_found' from
# torch's cmake setup.
append_torchlib_if_found(torch_python)

if ((NOT HIP_FOUND) AND (NOT CUDA_FOUND))
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()

if (NOT HIP_FOUND AND CUDA_FOUND)
  set(IS_CUDA true)

  # Verify torch version and warn if it is not expected.
  if (NOT Torch_VERSION VERSION_EQUAL 2.1.2)
    message(WARNING "Pytorch version 2.1.2 expected for CUDA build, "
      "saw ${Torch_VERSION} instead.")
  endif()
else()
  # Verify torch version and warn if it is not expected (derived from Dockerfile.rocm)
  # ROCm 5.7 -> torch 2.0.1
  if (ROCM_VERSION_DEV_MAJOR EQUAL 5 AND NOT Torch_VERSION VERSION_EQUAL 2.0.1)
    message(WARNING "Pytorch version 2.0.1 expected for ROCMm 5.x build, "
      "saw ${Torch_VERSION} instead.")
  endif()

  # ROCm 6.0 -> torch 2.1.1
  if (ROCM_VERSION_DEV_MAJOR EQUAL 6 AND NOT Torch_VERSION VERSION_EQUAL 2.1.1)
    message(WARNING "Pytorch version 2.1.1 expected for ROCMm 6.x build, "
      "saw ${Torch_VERSION} instead.")
  endif()
endif()

#
# Setup extra platform specific GPU compilation flags, e.g. NVCC flags for CUDA
# and hip flags for ROCm.
#
# Note: CUDA + HIP are detected by pytorch package so there's no need to repeat
# detect them explicitly with check_language, etc.
#
if (HIP_FOUND)
  # Importing torch recognizes and sets up some HIP/ROCm stuff but not all.
  # If we want cmake to be able to understand the .hip extension automatically,
  # we need to enable HIP explicitly.
  enable_language(HIP)

  #
  # VLLM_HIP_ARCHITECUTRES will control the offload-arch flags.
  # CMAKE_HIP_ARCHITECTURES is setup by pytorch and can be controlled
  # via the PYTORCH_ROCM_ARCH env variable.
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
      "None of the detected ROCM architectures: ${CMAKE_HIP_ARCHITECTURES} is"
      " supported. Supported ROCM architectures are: ${ROCM_SUPPORTED_ARCHS}.")
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

  list(APPEND VLLM_GPU_FLAGS ${VLLM_HIPCC_FLAGS})

  list(APPEND VLLM_GPU_FLAGS
    "-DUSE_ROCM"
    "-U__HIP_NO_HALF_CONVERSIONS__"
    "-U__HIP_NO_HALF_OPERATORS__"
    "-fno-gpu-rdc")

else()
  # Get common NVCC flags from torch.
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

  message(DEBUG "nvcc: ${VLLM_GPU_FLAGS}")
  message(DEBUG "punica nvcc: ${VLLM_PUNICA_GPU_FLAGS}")

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

  # Initialize the architecture lists to empty.
  set(VLLM_GPU_ARCHES)
  set(VLLM_PUNICA_GPU_ARCHES)

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
    list(APPEND VLLM_GPU_ARCHES "${CODE_ARCH}${VIRT}")

    # Add it to punica arch list if the version is >= 8.0
    if (CODE_VER GREATER_EQUAL 8.0)
      list(APPEND VLLM_PUNICA_GPU_ARCHES "${CODE_ARCH}${VIRT}")
    endif()
  endforeach()

  message(DEBUG "nvcc arch: ${VLLM_GPU_ARCHES}")
  message(DEBUG "punica arch: ${VLLM_PUNICA_GPU_ARCHES}")
endif()

#
# Define targets
#

# add comment
macro(hipify_sources_target OUT_SRCS NAME ORIG_SRCS)
  #
  # Split into C++ and non-C++ (i.e. CUDA) sources
  #
  set(SRCS ${ORIG_SRCS})
  set(CXX_SRCS ${ORIG_SRCS})
  list(FILTER SRCS EXCLUDE REGEX "\.(cc)|(cpp)$")
  list(FILTER CXX_SRCS INCLUDE REGEX "\.(cc)|(cpp)$")

  message(DEBUG "sources: ${SRCS}")
  message(DEBUG "cxx sources: ${CXX_SRCS}")

  #
  # Generate ROCM/HIP source file names from CUDA file names.
  #
  set(HIP_SRCS)
  foreach (SRC ${SRCS})
    string(REGEX REPLACE "\.cu$" "\.hip" SRC ${SRC})
    string(REGEX REPLACE "cuda" "hip" SRC ${SRC})
    list(APPEND HIP_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${SRC}")
  endforeach()
  message("rocm src: ${HIP_SRCS}")

  # TODO: hipify script needs a bunch more work
  # TODO: move copy into python (or figure out how to make hipify work properly)
  set(CSRC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/csrc)
  add_custom_target(
    hipify${NAME}
    COMMAND cp -r ${CMAKE_SOURCE_DIR}/csrc ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_SOURCE_DIR}/hipify.py -b ${CSRC_BUILD_DIR} -o ${CSRC_BUILD_DIR} -i ${CSRC_BUILD_DIR} ${SRCS}
    DEPENDS hipify.py ${SRCS}
    BYPRODUCTS ${HIP_SRCS}
    COMMENT "Running hipify on extension source files.")

  # Swap out original extension sources with hipified sources.
  set(${OUT_SRCS} ${HIP_SRCS})
  list(APPEND ${OUT_SRCS} ${CXX_SRCS})

  message(DEBUG "final ext src: ${OUT_SRCS}")
endmacro()

# add comment
# Note: optimization level/debug info is set via cmake build type.
function(define_module_target MOD_NAME MOD_SRC MOD_EXTRA_GPU_FLAGS
    MOD_GPU_ARCHES)

  if (NOT IS_CUDA AND HIP_FOUND)
    # Add hipify preprocessing step if we are running on AMD.
    hipify_sources_target(MOD_SRC ${MOD_NAME} "${MOD_SRC}")
  endif()

  # Note: for ROCm builds we let the proper flags and libraries get
  # pulled in by torch.
  Python_add_library(${MOD_NAME} MODULE ${MOD_SRC} WITH_SOABI)

  if (IS_CUDA)
    set(GPU_LANG "CUDA")
  else()
    set(GPU_LANG "HIP")

    # Make this target dependent on the hipify preprocessor step.
    add_dependencies(${MOD_NAME} hipify${MOD_NAME})
  endif()

  set_target_properties(${MOD_NAME} PROPERTIES ${GPU_LANG}_ARCHITECTURES
      "${MOD_GPU_ARCHES}")

  set_property(TARGET ${MOD_NAME} PROPERTY CXX_STANDARD 17)

  target_compile_options(${MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:${GPU_LANG}>:${MOD_EXTRA_GPU_FLAGS}>)

  target_compile_definitions(${MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${MOD_NAME}")

  target_include_directories(${MOD_NAME} PRIVATE
    csrc PRIVATE ${MPI_CXX_INCLUDE_DIRS})

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

elseif(HIP_FOUND)



endif()

define_module_target(_C
  "${VLLM_EXT_SRC}"
  "${VLLM_GPU_FLAGS}"
  "${VLLM_GPU_ARCHES}")

#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/moe/moe_ops.cpp"
  "csrc/moe/topk_softmax_kernels.cu")

define_module_target(_moe_C
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

define_module_target(_punica_C
  "${VLLM_PUNICA_EXT_SRC}"
  "${VLLM_PUNICA_GPU_FLAGS}"
  "${VLLM_PUNICA_GPU_ARCHES}")
