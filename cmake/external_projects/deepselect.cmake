include(FetchContent)

# If DEEPSELECT_SRC_DIR is set, DeepSelect is built from that directory
# instead of downloading.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{DEEPSELECT_SRC_DIR})
  set(DEEPSELECT_SRC_DIR $ENV{DEEPSELECT_SRC_DIR})
endif()

if(DEEPSELECT_SRC_DIR)
  FetchContent_Declare(
        deepselect
        SOURCE_DIR ${DEEPSELECT_SRC_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
else()
  FetchContent_Declare(
        deepselect
        GIT_REPOSITORY https://github.com/deepseek-ai/DeepSelect.git
        GIT_TAG 0f03b68748b304863fdf0181a11458d04ae533a9 # v1.0.0
        GIT_SUBMODULES "csrc/3rdparty/cutlass"
        GIT_PROGRESS TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
endif()

FetchContent_MakeAvailable(deepselect)
message(STATUS "DeepSelect is available at ${deepselect_SOURCE_DIR}")

# DeepSelect kernels only support sm_100a/sm_103a and require CUDA 12.9+.
set(DEEPSELECT_SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
    list(APPEND DEEPSELECT_SUPPORT_ARCHS "10.0a" "10.3a")
endif()

cuda_archs_loose_intersection(DEEPSELECT_ARCHS "${DEEPSELECT_SUPPORT_ARCHS}" "${CUDA_ARCHS}")
if(DEEPSELECT_ARCHS)
    message(STATUS "DeepSelect CUDA architectures: ${DEEPSELECT_ARCHS}")

    # All explicit template instantiations must be compiled: api.cpp
    # references every dispatch combination.
    file(GLOB DeepSelect_SOURCES CONFIGURE_DEPENDS
        ${deepselect_SOURCE_DIR}/csrc/cuda_kernels/v3/instantiations/*.cu
        ${deepselect_SOURCE_DIR}/csrc/cuda_kernels/v3_fp32/instantiations/*.cu
        ${deepselect_SOURCE_DIR}/csrc/cuda_kernels/v3_cluster/instantiations/*.cu
    )
    list(PREPEND DeepSelect_SOURCES ${deepselect_SOURCE_DIR}/csrc/api.cpp)

    set(DeepSelect_INCLUDES
        ${deepselect_SOURCE_DIR}/csrc
        ${deepselect_SOURCE_DIR}/csrc/3rdparty/cutlass/include
        ${deepselect_SOURCE_DIR}/csrc/3rdparty/kerutils/include
    )

    set_gencode_flags_for_srcs(
        SRCS "${DeepSelect_SOURCES}"
        CUDA_ARCHS "${DEEPSELECT_ARCHS}")

    set(VLLM_DEEPSELECT_GPU_FLAGS ${VLLM_GPU_FLAGS})
    list(APPEND VLLM_DEEPSELECT_GPU_FLAGS
        "-O3" "--expt-relaxed-constexpr" "--expt-extended-lambda"
        "--use_fast_math" "--ftz=false"
        "-U__CUDA_NO_HALF_OPERATORS__" "-U__CUDA_NO_HALF_CONVERSIONS__"
        "-U__CUDA_NO_HALF2_OPERATORS__" "-U__CUDA_NO_BFLOAT16_CONVERSIONS__")

    # DeepSelect exposes a raw PYBIND11_MODULE (no TORCH_LIBRARY shim), so it
    # cannot use the stable ABI; see the deepgemm.cmake comment for details.
    define_extension_target(
        _deepselect_C
        DESTINATION vllm
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES ${DeepSelect_SOURCES}
        COMPILE_FLAGS ${VLLM_DEEPSELECT_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        INCLUDE_DIRECTORIES ${DeepSelect_INCLUDES}
        WITH_SOABI)

    # PYBIND11_MODULE bindings need pybind11's at::Tensor casters, which live
    # in libtorch_python (loaded RTLD_LOCAL by `import torch`).
    find_library(DEEPSELECT_TORCH_PYTHON torch_python
        PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(_deepselect_C PRIVATE ${DEEPSELECT_TORCH_PYTHON})

    # DeepSelect requires C++20 (std::format, template lambdas, etc.)
    target_compile_options(_deepselect_C PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>
        # CUDA TUs get this from __CUDACC__; only the host TU needs it.
        $<$<COMPILE_LANGUAGE:CXX>:-DKERUTILS_IS_BUILD_ON_CUDA>)
else()
    message(STATUS "DeepSelect will not compile: unsupported CUDA architecture ${CUDA_ARCHS}")
    # Create an empty target for setup.py on unsupported systems
    add_custom_target(_deepselect_C)
endif()
