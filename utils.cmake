#
# Attempt to find the python pacakge that uses the same python executable as
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
    COMMAND ${CMAKE_SOURCE_DIR}/hipify.py -p ${CMAKE_SOURCE_DIR}/csrc -o ${CSRC_BUILD_DIR} ${SRCS}
    DEPENDS hipify.py ${SRCS}
    BYPRODUCTS ${HIP_SRCS}
    COMMENT "Running hipify on ${NAME} extension source files.")

  # Swap out original extension sources with hipified sources.
  set(${OUT_SRCS} ${HIP_SRCS})
  list(APPEND ${OUT_SRCS} ${CXX_SRCS})
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

  set_target_properties(${MOD_NAME} PROPERTIES ${MOD_GPU_LANG}_ARCHITECTURES
      "${MOD_GPU_ARCHES}")

  set_property(TARGET ${MOD_NAME} PROPERTY CXX_STANDARD 17)

  target_compile_options(${MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:${MOD_GPU_LANG}>:${MOD_EXTRA_GPU_FLAGS}>)

  target_compile_definitions(${MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${MOD_NAME}")

  target_include_directories(${MOD_NAME} PRIVATE
    csrc PRIVATE ${MPI_CXX_INCLUDE_DIRS})

  target_link_libraries(${MOD_NAME} PRIVATE ${TORCH_LIBRARIES})

  install(TARGETS ${MOD_NAME} LIBRARY DESTINATION ${MOD_DEST})
endfunction()
