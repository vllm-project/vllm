#
# Attempt to find the python package that uses the same python executable as
# `EXECUTABLE` and is one of the `SUPPORTED_VERSIONS`.
#
macro (find_python_from_executable EXECUTABLE SUPPORTED_VERSIONS)
  file(REAL_PATH ${EXECUTABLE} EXECUTABLE)
  set(Python_EXECUTABLE ${EXECUTABLE})
  find_package(Python COMPONENTS Interpreter Development.Module)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
  set(VER "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
  set(SUPPORTED_VERSIONS_LIST ${SUPPORTED_VERSIONS} ${ARGN})
  if (NOT VER IN_LIST SUPPORTED_VERSIONS_LIST)
    message(FATAL_ERROR
      "Python version (${VER}) is not one of the supported versions: "
      "${SUPPORTED_VERSIONS}.")
  endif()
  message(STATUS "Found python matching: ${EXECUTABLE}.")
endmacro()

#
# Run `EXPR` in python.  The standard output of python is stored in `OUT` and
# has trailing whitespace stripped.  If an error is encountered when running
# python, a fatal message `ERR_MSG` is issued.
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

# Run `EXPR` in python after importing `PKG`. Use the result of this to extend
# `CMAKE_PREFIX_PATH` so the torch cmake configuration can be imported.
macro (append_cmake_prefix_path PKG EXPR)
  run_python(PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${PREFIX_PATH})
endmacro()

#
# Add a target named `hipify${NAME}` that runs the hipify preprocessor on a set
# of CUDA source files. The names of the corresponding "hipified" sources are
# stored in `OUT_SRCS`.
#
macro (hipify_sources_target OUT_SRCS NAME ORIG_SRCS)
  #
  # Split into C++ and non-C++ (i.e. CUDA) sources.
  #
  set(SRCS ${ORIG_SRCS})
  set(CXX_SRCS ${ORIG_SRCS})
  list(FILTER SRCS EXCLUDE REGEX "\.(cc)|(cpp)$")
  list(FILTER CXX_SRCS INCLUDE REGEX "\.(cc)|(cpp)$")

  #
  # Generate ROCm/HIP source file names from CUDA file names.
  # Since HIP files are generated code, they will appear in the build area
  # `CMAKE_CURRENT_BINARY_DIR` directory rather than the original csrc dir.
  #
  set(HIP_SRCS)
  foreach (SRC ${SRCS})
    string(REGEX REPLACE "\.cu$" "\.hip" SRC ${SRC})
    string(REGEX REPLACE "cuda" "hip" SRC ${SRC})
    list(APPEND HIP_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${SRC}")
  endforeach()

  set(CSRC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/csrc)
  add_custom_target(
    hipify${NAME}
    COMMAND ${CMAKE_SOURCE_DIR}/cmake/hipify.py -p ${CMAKE_SOURCE_DIR}/csrc -o ${CSRC_BUILD_DIR} ${SRCS}
    DEPENDS ${CMAKE_SOURCE_DIR}/cmake/hipify.py ${SRCS}
    BYPRODUCTS ${HIP_SRCS}
    COMMENT "Running hipify on ${NAME} extension source files.")

  # Swap out original extension sources with hipified sources.
  set(${OUT_SRCS} ${HIP_SRCS})
  list(APPEND ${OUT_SRCS} ${CXX_SRCS})
endmacro()

#
# Get additional GPU compiler flags from torch.
#
macro(get_torch_gpu_compiler_flags GPU_FLAGS GPU_LANG)
  if (${GPU_LANG} STREQUAL "CUDA")
    #
    # Get common NVCC flags from torch.
    #
    run_python(${GPU_FLAGS}
      "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND ${GPU_FLAGS} "-DENABLE_FP8_E5M2")
    endif()

  elseif(${GPU_LANG} STREQUAL "HIP")
    #
    # Get common HIP/HIPCC flags from torch.
    #
    run_python(${GPU_FLAGS}
      "import torch.utils.cpp_extension as t; print(';'.join(t.COMMON_HIP_FLAGS + t.COMMON_HIPCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    list(APPEND ${GPU_FLAGS}
      "-DUSE_ROCM"
      "-U__HIP_NO_HALF_CONVERSIONS__"
      "-U__HIP_NO_HALF_OPERATORS__"
      "-fno-gpu-rdc")

  endif()
endmacro()

# Macro for converting a `gencode` version number to a cmake version number.
macro(string_to_ver OUT_VER IN_STR)
  string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" ${OUT_VER} ${IN_STR})
endmacro()

#
# Override the GPU architectures detected by cmake/torch and filter them by
# `GPU_SUPPORTED_ARCHES`. Sets the final set of architectures in
# `GPU_ARCHES`.
#
macro(override_gpu_arches GPU_ARCHES GPU_LANG GPU_SUPPORTED_ARCHES)
  set(GPU_SUPPORTED_ARCHES_LIST ${GPU_SUPPORTED_ARCHES} ${ARGN})
  message(STATUS "${GPU_LANG} supported arches: ${GPU_SUPPORTED_ARCHES_LIST}")

  if (${GPU_LANG} STREQUAL "HIP")
    #
    # `GPU_ARCHES` controls the `--offload-arch` flags.
    # `CMAKE_HIP_ARCHITECTURES` is set up by torch and can be controlled
    # via the `PYTORCH_ROCM_ARCH` env variable.
    #

    #
    # Find the intersection of the supported + detected architectures to
    # set the module architecture flags.
    #
    set(${GPU_ARCHES})
    foreach (ARCH ${CMAKE_HIP_ARCHITECTURES})
      if (ARCH IN_LIST GPU_SUPPORTED_ARCHES_LIST)
        list(APPEND ${GPU_ARCHES} ${ARCH})
      endif()
    endforeach()

    if(NOT ${GPU_ARCHES})
      message(FATAL_ERROR
        "None of the detected ROCm architectures: ${CMAKE_HIP_ARCHITECTURES} is"
        " supported. Supported ROCm architectures are: ${GPU_SUPPORTED_ARCHES_LIST}.")
    endif()

  elseif(${GPU_LANG} STREQUAL "CUDA")
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
    string(REGEX MATCHALL "-gencode arch=[^ ]+" _CUDA_ARCH_FLAGS
      ${CMAKE_CUDA_FLAGS})

    # Remove all `-gencode` flags from `CMAKE_CUDA_FLAGS` since they will be modified
    # and passed back via the `CUDA_ARCHITECTURES` property.
    string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS
      ${CMAKE_CUDA_FLAGS})

    # If this error is triggered, it might mean that torch has changed how it sets
    # up nvcc architecture code generation flags.
    if (NOT _CUDA_ARCH_FLAGS)
      message(FATAL_ERROR
        "Could not find any architecture related code generation flags in "
        "CMAKE_CUDA_FLAGS. (${CMAKE_CUDA_FLAGS})")
    endif()

    message(DEBUG "final CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    message(DEBUG "arch flags: ${_CUDA_ARCH_FLAGS}")

    # Initialize the architecture lists to empty.
    set(${GPU_ARCHES})

    # Process each `gencode` flag.
    foreach(ARCH ${_CUDA_ARCH_FLAGS})
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
      if (NOT CODE_VER IN_LIST GPU_SUPPORTED_ARCHES_LIST)
        message(STATUS "discarding unsupported CUDA arch ${VER}.")
        continue()
      endif()

      # Add it to the arch list.
      list(APPEND ${GPU_ARCHES} "${CODE_ARCH}${VIRT}")
    endforeach()
  endif()
  message(STATUS "${GPU_LANG} target arches: ${${GPU_ARCHES}}")
endmacro()

#
# Define a target named `MOD_NAME` for a single extension. The
# arguments are:
#
# MOD_DEST            - module destination directory.
# MOD_GPU_LANG        - the GPU language for this module, e.g CUDA, HIP, etc.
# MOD_SRC             - the list of source files relative to CMakeLists.txt
#                       directory.
# MOD_EXTRA_GPU_FLAGS - extra compiler flags passed to NVCC/hip.
# MOD_GPU_ARCHES      - a list of target GPU architectures in cmake format.
#                       Refer to documentation on `CMAKE_CUDA_ARCHITECTURES`
#                       and `CMAKE_HIP_ARCHITECTURES` for more info.
#
# Note: optimization level/debug info is set via cmake build type.
#
function (define_gpu_extension_target MOD_NAME MOD_DEST MOD_GPU_LANG MOD_SRC
    MOD_EXTRA_GPU_FLAGS MOD_GPU_ARCHES)

  # Add hipify preprocessing step when building with HIP/ROCm.
  if (MOD_GPU_LANG STREQUAL "HIP")
    hipify_sources_target(MOD_SRC ${MOD_NAME} "${MOD_SRC}")
  endif()

  Python_add_library(${MOD_NAME} MODULE ${MOD_SRC} WITH_SOABI)

  if (MOD_GPU_LANG STREQUAL "HIP")
    # Make this target dependent on the hipify preprocessor step.
    add_dependencies(${MOD_NAME} hipify${MOD_NAME})
  endif()

  if (MOD_GPU_ARCHES)
    set_target_properties(${MOD_NAME} PROPERTIES ${MOD_GPU_LANG}_ARCHITECTURES
      "${MOD_GPU_ARCHES}")
  endif()

  set_property(TARGET ${MOD_NAME} PROPERTY CXX_STANDARD 17)

  target_compile_options(${MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:${MOD_GPU_LANG}>:${MOD_EXTRA_GPU_FLAGS}>)

  target_compile_definitions(${MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${MOD_NAME}")

  target_include_directories(${MOD_NAME} PRIVATE csrc)

  target_link_libraries(${MOD_NAME} PRIVATE ${TORCH_LIBRARIES})

  install(TARGETS ${MOD_NAME} LIBRARY DESTINATION ${MOD_DEST})
endfunction()
