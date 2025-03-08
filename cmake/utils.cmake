#
# Attempt to find the python package that uses the same python executable as
# `EXECUTABLE` and is one of the `SUPPORTED_VERSIONS`.
#
macro (find_python_from_executable EXECUTABLE SUPPORTED_VERSIONS)
  file(REAL_PATH ${EXECUTABLE} EXECUTABLE)
  set(Python_EXECUTABLE ${EXECUTABLE})
  find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
  set(_VER "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
  set(_SUPPORTED_VERSIONS_LIST ${SUPPORTED_VERSIONS} ${ARGN})
  if (NOT _VER IN_LIST _SUPPORTED_VERSIONS_LIST)
    message(FATAL_ERROR
      "Python version (${_VER}) is not one of the supported versions: "
      "${_SUPPORTED_VERSIONS_LIST}.")
  endif()
  message(STATUS "Found python matching: ${EXECUTABLE}.")
endmacro()

#
# Run `EXPR` in python.  The standard output of python is stored in `OUT` and
# has trailing whitespace stripped.  If an error is encountered when running
# python, a fatal message `ERR_MSG` is issued.
#
function (run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "${Python_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

# Run `EXPR` in python after importing `PKG`. Use the result of this to extend
# `CMAKE_PREFIX_PATH` so the torch cmake configuration can be imported.
macro (append_cmake_prefix_path PKG EXPR)
  run_python(_PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

#
# Add a target named `hipify${NAME}` that runs the hipify preprocessor on a set
# of CUDA source files. The names of the corresponding "hipified" sources are
# stored in `OUT_SRCS`.
#
function (hipify_sources_target OUT_SRCS NAME ORIG_SRCS)
  #
  # Split into C++ and non-C++ (i.e. CUDA) sources.
  #
  set(SRCS ${ORIG_SRCS})
  set(CXX_SRCS ${ORIG_SRCS})
  list(FILTER SRCS EXCLUDE REGEX "\.(cc)|(cpp)|(hip)$")
  list(FILTER CXX_SRCS INCLUDE REGEX "\.(cc)|(cpp)|(hip)$")

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
  list(APPEND HIP_SRCS ${CXX_SRCS})
  set(${OUT_SRCS} ${HIP_SRCS} PARENT_SCOPE)
endfunction()

#
# Get additional GPU compiler flags from torch.
#
function (get_torch_gpu_compiler_flags OUT_GPU_FLAGS GPU_LANG)
  if (${GPU_LANG} STREQUAL "CUDA")
    #
    # Get common NVCC flags from torch.
    #
    run_python(GPU_FLAGS
      "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND GPU_FLAGS "-DENABLE_FP8")
    endif()
    if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
      list(REMOVE_ITEM GPU_FLAGS
        "-D__CUDA_NO_HALF_OPERATORS__"
        "-D__CUDA_NO_HALF_CONVERSIONS__"
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
        "-D__CUDA_NO_HALF2_OPERATORS__")
    endif()

  elseif(${GPU_LANG} STREQUAL "HIP")
    #
    # Get common HIP/HIPCC flags from torch.
    #
    run_python(GPU_FLAGS
      "import torch.utils.cpp_extension as t; print(';'.join(t.COMMON_HIP_FLAGS + t.COMMON_HIPCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    list(APPEND GPU_FLAGS
      "-DUSE_ROCM"
      "-DENABLE_FP8"
      "-U__HIP_NO_HALF_CONVERSIONS__"
      "-U__HIP_NO_HALF_OPERATORS__"
      "-fno-gpu-rdc")

  endif()
  set(${OUT_GPU_FLAGS} ${GPU_FLAGS} PARENT_SCOPE)
endfunction()

# Macro for converting a `gencode` version number to a cmake version number.
macro(string_to_ver OUT_VER IN_STR)
  string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" ${OUT_VER} ${IN_STR})
endmacro()

#
# Clear all `-gencode` flags from `CMAKE_CUDA_FLAGS` and store them in
# `CUDA_ARCH_FLAGS`.
#
# Example:
#   CMAKE_CUDA_FLAGS="-Wall -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75"
#   clear_cuda_arches(CUDA_ARCH_FLAGS)
#   CUDA_ARCH_FLAGS="-gencode arch=compute_70,code=sm_70;-gencode arch=compute_75,code=sm_75"
#   CMAKE_CUDA_FLAGS="-Wall"
#
macro(clear_cuda_arches CUDA_ARCH_FLAGS)
    # Extract all `-gencode` flags from `CMAKE_CUDA_FLAGS`
    string(REGEX MATCHALL "-gencode arch=[^ ]+" CUDA_ARCH_FLAGS
      ${CMAKE_CUDA_FLAGS})

    # Remove all `-gencode` flags from `CMAKE_CUDA_FLAGS` since they will be modified
    # and passed back via the `CUDA_ARCHITECTURES` property.
    string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS
      ${CMAKE_CUDA_FLAGS})
endmacro()

#
# Extract unique CUDA architectures from a list of compute capabilities codes in 
# the form `<major><minor>[<letter>]`, convert them to the form sort 
# `<major>.<minor>`, dedupes them and then sorts them in ascending order and 
# stores them in `OUT_ARCHES`.
#
# Example:
#   CUDA_ARCH_FLAGS="-gencode arch=compute_75,code=sm_75;...;-gencode arch=compute_90a,code=sm_90a" 
#   extract_unique_cuda_archs_ascending(OUT_ARCHES CUDA_ARCH_FLAGS)
#   OUT_ARCHES="7.5;...;9.0"
function(extract_unique_cuda_archs_ascending OUT_ARCHES CUDA_ARCH_FLAGS)
  set(_CUDA_ARCHES)
  foreach(_ARCH ${CUDA_ARCH_FLAGS})
    string(REGEX MATCH "arch=compute_\([0-9]+a?\)" _COMPUTE ${_ARCH})
    if (_COMPUTE)
      set(_COMPUTE ${CMAKE_MATCH_1})
    endif()

    string_to_ver(_COMPUTE_VER ${_COMPUTE})
    list(APPEND _CUDA_ARCHES ${_COMPUTE_VER})
  endforeach()

  list(REMOVE_DUPLICATES _CUDA_ARCHES)
  list(SORT _CUDA_ARCHES COMPARE NATURAL ORDER ASCENDING)
  set(${OUT_ARCHES} ${_CUDA_ARCHES} PARENT_SCOPE)
endfunction()

#
# For a specific file set the `-gencode` flag in compile options conditionally 
# for the CUDA language. 
#
# Example:
#   set_gencode_flag_for_srcs(
#     SRCS "foo.cu"
#     ARCH "compute_75"
#     CODE "sm_75")
#   adds: "-gencode arch=compute_75,code=sm_75" to the compile options for 
#    `foo.cu` (only for the CUDA language).
#
macro(set_gencode_flag_for_srcs)
  set(options)
  set(oneValueArgs ARCH CODE)
  set(multiValueArgs SRCS)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )
  set(_FLAG -gencode arch=${arg_ARCH},code=${arg_CODE})
  set_property(
    SOURCE ${arg_SRCS}
    APPEND PROPERTY
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${_FLAG}>"
  )

  message(DEBUG "Setting gencode flag for ${arg_SRCS}: ${_FLAG}")
endmacro(set_gencode_flag_for_srcs)

#
# For a list of source files set the `-gencode` flags in the files specific 
#  compile options (specifically for the CUDA language).
#
# arguments are:
#  SRCS: list of source files
#  CUDA_ARCHS: list of CUDA architectures in the form `<major>.<minor>[letter]`
#  BUILD_PTX_FOR_ARCH: if set to true, then the PTX code will be built
#    for architecture `BUILD_PTX_FOR_ARCH` if there is a CUDA_ARCH in CUDA_ARCHS 
#    that is larger than BUILD_PTX_FOR_ARCH.
#
macro(set_gencode_flags_for_srcs)
  set(options)
  set(oneValueArgs BUILD_PTX_FOR_ARCH)
  set(multiValueArgs SRCS CUDA_ARCHS)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

  foreach(_ARCH ${arg_CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    set_gencode_flag_for_srcs(
      SRCS ${arg_SRCS}
      ARCH "compute_${_ARCH}"
      CODE "sm_${_ARCH}")
  endforeach()

  if (${arg_BUILD_PTX_FOR_ARCH})
    list(SORT arg_CUDA_ARCHS COMPARE NATURAL ORDER ASCENDING)
    list(GET arg_CUDA_ARCHS -1 _HIGHEST_ARCH)
    if (_HIGHEST_ARCH VERSION_GREATER_EQUAL ${arg_BUILD_PTX_FOR_ARCH})
      string(REPLACE "." "" _PTX_ARCH "${arg_BUILD_PTX_FOR_ARCH}")
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_PTX_ARCH}"
        CODE "compute_${_PTX_ARCH}")
    endif()
  endif()
endmacro()

#
# For the given `SRC_CUDA_ARCHS` list of gencode versions in the form 
#  `<major>.<minor>[letter]` compute the "loose intersection" with the 
#  `TGT_CUDA_ARCHS` list of gencodes. 
# The loose intersection is defined as:
#   { max{ x \in tgt | x <= y } | y \in src, { x \in tgt | x <= y } != {} }
#  where `<=` is the version comparison operator.
# In other words, for each version in `TGT_CUDA_ARCHS` find the highest version
#  in `SRC_CUDA_ARCHS` that is less or equal to the version in `TGT_CUDA_ARCHS`.
# We have special handling for x.0a, if x.0a is in `SRC_CUDA_ARCHS` and x.0 is
#  in `TGT_CUDA_ARCHS` then we should remove x.0a from `SRC_CUDA_ARCHS` and add
#  x.0a to the result (and remove x.0 from TGT_CUDA_ARCHS). 
# The result is stored in `OUT_CUDA_ARCHS`.
#
# Example:
#   SRC_CUDA_ARCHS="7.5;8.0;8.6;9.0;9.0a"
#   TGT_CUDA_ARCHS="8.0;8.9;9.0"
#   cuda_archs_loose_intersection(OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
#   OUT_CUDA_ARCHS="8.0;8.6;9.0;9.0a"
#
function(cuda_archs_loose_intersection OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
  list(REMOVE_DUPLICATES SRC_CUDA_ARCHS)
  set(TGT_CUDA_ARCHS_ ${TGT_CUDA_ARCHS})

  # if x.0a is in SRC_CUDA_ARCHS and x.0 is in CUDA_ARCHS then we should
  # remove x.0a from SRC_CUDA_ARCHS and add x.0a to _CUDA_ARCHS
  set(_CUDA_ARCHS)
  if ("9.0a" IN_LIST SRC_CUDA_ARCHS)
    list(REMOVE_ITEM SRC_CUDA_ARCHS "9.0a")
    if ("9.0" IN_LIST TGT_CUDA_ARCHS_)
      list(REMOVE_ITEM TGT_CUDA_ARCHS_ "9.0")
      set(_CUDA_ARCHS "9.0a")
    endif()
  endif()

  if ("10.0a" IN_LIST SRC_CUDA_ARCHS)
    list(REMOVE_ITEM SRC_CUDA_ARCHS "10.0a")
    if ("10.0" IN_LIST TGT_CUDA_ARCHS)
      list(REMOVE_ITEM TGT_CUDA_ARCHS_ "10.0")
      set(_CUDA_ARCHS "10.0a")
    endif()
  endif()

  list(SORT SRC_CUDA_ARCHS COMPARE NATURAL ORDER ASCENDING)

  # for each ARCH in TGT_CUDA_ARCHS find the highest arch in SRC_CUDA_ARCHS that
  # is less or equal to ARCH (but has the same major version since SASS binary
  # compatibility is only forward compatible within the same major version).
  foreach(_ARCH ${TGT_CUDA_ARCHS_})
    set(_TMP_ARCH)
    # Extract the major version of the target arch
    string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" TGT_ARCH_MAJOR "${_ARCH}")
    foreach(_SRC_ARCH ${SRC_CUDA_ARCHS})
      # Extract the major version of the source arch
      string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" SRC_ARCH_MAJOR "${_SRC_ARCH}")
      # Check major-version match AND version-less-or-equal
      if (_SRC_ARCH VERSION_LESS_EQUAL _ARCH)
        if (SRC_ARCH_MAJOR STREQUAL TGT_ARCH_MAJOR)
          set(_TMP_ARCH "${_SRC_ARCH}")
        endif()
      else()
        # If we hit a version greater than the target, we can break
        break()
      endif()
    endforeach()

    # If we found a matching _TMP_ARCH, append it to _CUDA_ARCHS
    if (_TMP_ARCH)
      list(APPEND _CUDA_ARCHS "${_TMP_ARCH}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _CUDA_ARCHS)
  set(${OUT_CUDA_ARCHS} ${_CUDA_ARCHS} PARENT_SCOPE)
endfunction()

#
# Override the GPU architectures detected by cmake/torch and filter them by
# `GPU_SUPPORTED_ARCHES`. Sets the final set of architectures in
# `GPU_ARCHES`. This only applies to the HIP language since for CUDA we set 
# the architectures on a per file basis.
#
# Note: this is defined as a macro since it updates `CMAKE_CUDA_FLAGS`.
#
macro(override_gpu_arches GPU_ARCHES GPU_LANG GPU_SUPPORTED_ARCHES)
  set(_GPU_SUPPORTED_ARCHES_LIST ${GPU_SUPPORTED_ARCHES} ${ARGN})
  message(STATUS "${GPU_LANG} supported arches: ${_GPU_SUPPORTED_ARCHES_LIST}")

  if (${GPU_LANG} STREQUAL "HIP")
    #
    # `GPU_ARCHES` controls the `--offload-arch` flags.
    #
    # If PYTORCH_ROCM_ARCH env variable exists, then we take it as a list,
    # if not, then we use CMAKE_HIP_ARCHITECTURES which was generated by calling
    # "rocm_agent_enumerator" in "enable_language(HIP)"
    # (in file Modules/CMakeDetermineHIPCompiler.cmake)
    #
    if(DEFINED ENV{PYTORCH_ROCM_ARCH})
      set(HIP_ARCHITECTURES $ENV{PYTORCH_ROCM_ARCH})
    else()
      set(HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
    endif()
    #
    # Find the intersection of the supported + detected architectures to
    # set the module architecture flags.
    #
    set(${GPU_ARCHES})
    foreach (_ARCH ${HIP_ARCHITECTURES})
      if (_ARCH IN_LIST _GPU_SUPPORTED_ARCHES_LIST)
        list(APPEND ${GPU_ARCHES} ${_ARCH})
      endif()
    endforeach()

    if(NOT ${GPU_ARCHES})
      message(FATAL_ERROR
        "None of the detected ROCm architectures: ${HIP_ARCHITECTURES} is"
        " supported. Supported ROCm architectures are: ${_GPU_SUPPORTED_ARCHES_LIST}.")
    endif()
  endif()
endmacro()

#
# Define a target named `GPU_MOD_NAME` for a single extension. The
# arguments are:
#
# DESTINATION <dest>         - Module destination directory.
# LANGUAGE <lang>            - The GPU language for this module, e.g CUDA, HIP,
#                              etc.
# SOURCES <sources>          - List of source files relative to CMakeLists.txt
#                              directory.
#
# Optional arguments:
#
# ARCHITECTURES <arches>     - A list of target GPU architectures in cmake
#                              format.
#                              Refer `CMAKE_CUDA_ARCHITECTURES` documentation
#                              and `CMAKE_HIP_ARCHITECTURES` for more info.
#                              ARCHITECTURES will use cmake's defaults if
#                              not provided.
# COMPILE_FLAGS <flags>      - Extra compiler flags passed to NVCC/hip.
# INCLUDE_DIRECTORIES <dirs> - Extra include directories.
# LIBRARIES <libraries>      - Extra link libraries.
# WITH_SOABI                 - Generate library with python SOABI suffix name.
# USE_SABI <version>         - Use python stable api <version>
#
# Note: optimization level/debug info is set via cmake build type.
#
function (define_gpu_extension_target GPU_MOD_NAME)
  cmake_parse_arguments(PARSE_ARGV 1
    GPU
    "WITH_SOABI"
    "DESTINATION;LANGUAGE;USE_SABI"
    "SOURCES;ARCHITECTURES;COMPILE_FLAGS;INCLUDE_DIRECTORIES;LIBRARIES")

  # Add hipify preprocessing step when building with HIP/ROCm.
  if (GPU_LANGUAGE STREQUAL "HIP")
    hipify_sources_target(GPU_SOURCES ${GPU_MOD_NAME} "${GPU_SOURCES}")
  endif()

  if (GPU_WITH_SOABI)
    set(GPU_WITH_SOABI WITH_SOABI)
  else()
    set(GPU_WITH_SOABI)
  endif()

  if (GPU_USE_SABI)
    Python_add_library(${GPU_MOD_NAME} MODULE USE_SABI ${GPU_USE_SABI} ${GPU_WITH_SOABI} "${GPU_SOURCES}")
  else()
    Python_add_library(${GPU_MOD_NAME} MODULE ${GPU_WITH_SOABI} "${GPU_SOURCES}")
  endif()

  if (GPU_LANGUAGE STREQUAL "HIP")
    # Make this target dependent on the hipify preprocessor step.
    add_dependencies(${GPU_MOD_NAME} hipify${GPU_MOD_NAME})
  endif()

  if (GPU_ARCHITECTURES)
    set_target_properties(${GPU_MOD_NAME} PROPERTIES
      ${GPU_LANGUAGE}_ARCHITECTURES "${GPU_ARCHITECTURES}")
  endif()

  set_property(TARGET ${GPU_MOD_NAME} PROPERTY CXX_STANDARD 17)

  target_compile_options(${GPU_MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:${GPU_LANGUAGE}>:${GPU_COMPILE_FLAGS}>)

  target_compile_definitions(${GPU_MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${GPU_MOD_NAME}")

  target_include_directories(${GPU_MOD_NAME} PRIVATE csrc
    ${GPU_INCLUDE_DIRECTORIES})

  target_link_libraries(${GPU_MOD_NAME} PRIVATE torch ${GPU_LIBRARIES})

  # Don't use `TORCH_LIBRARIES` for CUDA since it pulls in a bunch of
  # dependencies that are not necessary and may not be installed.
  if (GPU_LANGUAGE STREQUAL "CUDA")
    target_link_libraries(${GPU_MOD_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)
  else()
    target_link_libraries(${GPU_MOD_NAME} PRIVATE ${TORCH_LIBRARIES})
  endif()

  install(TARGETS ${GPU_MOD_NAME} LIBRARY DESTINATION ${GPU_DESTINATION} COMPONENT ${GPU_MOD_NAME})
endfunction()
