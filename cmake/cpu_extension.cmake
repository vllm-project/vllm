include(FetchContent)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
# Define environment variables for special configurations
#
if(DEFINED ENV{VLLM_CPU_AVX512BF16})
    set(ENABLE_AVX512BF16 ON)
endif()

include_directories("${CMAKE_SOURCE_DIR}/csrc")

#
# Check the compile flags
#
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
    list(APPEND CXX_COMPILE_FLAGS
        "-fopenmp"
        "-DVLLM_CPU_EXTENSION")
else()
    list(APPEND CXX_COMPILE_FLAGS
        "-fopenmp"
        "-mf16c"
        "-DVLLM_CPU_EXTENSION")
endif()

execute_process(COMMAND cat /proc/cpuinfo
                RESULT_VARIABLE CPUINFO_RET
                OUTPUT_VARIABLE CPUINFO)

if (NOT CPUINFO_RET EQUAL 0)
    message(FATAL_ERROR "Failed to check CPU features via /proc/cpuinfo")
endif()

function (find_isa CPUINFO TARGET OUT)
    string(FIND ${CPUINFO} ${TARGET} ISA_FOUND)
    if(NOT ISA_FOUND EQUAL -1)
        set(${OUT} ON PARENT_SCOPE)
    else()
        set(${OUT} OFF PARENT_SCOPE)
    endif()
endfunction()

function (is_avx512_disabled OUT)
    set(DISABLE_AVX512 $ENV{VLLM_CPU_DISABLE_AVX512})
    if(DISABLE_AVX512 AND DISABLE_AVX512 STREQUAL "true")
        set(${OUT} ON PARENT_SCOPE)
    else()
        set(${OUT} OFF PARENT_SCOPE)
    endif()
endfunction()

is_avx512_disabled(AVX512_DISABLED)

find_isa(${CPUINFO} "avx2" AVX2_FOUND)
find_isa(${CPUINFO} "avx512f" AVX512_FOUND)
find_isa(${CPUINFO} "POWER10" POWER10_FOUND)
find_isa(${CPUINFO} "POWER9" POWER9_FOUND)

if (AVX512_FOUND AND NOT AVX512_DISABLED)
    list(APPEND CXX_COMPILE_FLAGS
        "-mavx512f"
        "-mavx512vl"
        "-mavx512bw"
        "-mavx512dq")

    find_isa(${CPUINFO} "avx512_bf16" AVX512BF16_FOUND)
    if (AVX512BF16_FOUND OR ENABLE_AVX512BF16)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.3)
            list(APPEND CXX_COMPILE_FLAGS "-mavx512bf16")
        else()
            message(WARNING "Disable AVX512-BF16 ISA support, requires gcc/g++ >= 12.3")
        endif()
    else()
        message(WARNING "Disable AVX512-BF16 ISA support, no avx512_bf16 found in local CPU flags." " If cross-compilation is required, please set env VLLM_CPU_AVX512BF16=1.")
    endif()
elseif (AVX2_FOUND)
    list(APPEND CXX_COMPILE_FLAGS "-mavx2")
    message(WARNING "vLLM CPU backend using AVX2 ISA")
elseif (POWER9_FOUND OR POWER10_FOUND)
    message(STATUS "PowerPC detected")
    # Check for PowerPC VSX support
    list(APPEND CXX_COMPILE_FLAGS
        "-mvsx"
        "-mcpu=native"
        "-mtune=native")
else()
    message(FATAL_ERROR "vLLM CPU backend requires AVX512 or AVX2 or Power9+ ISA support.")
endif()

#
# Build oneDNN for W8A8 GEMM kernels (only for x86-AVX512 platforms)
#
if (AVX512_FOUND AND NOT AVX512_DISABLED)
    FetchContent_Declare(
        oneDNN
        GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
        GIT_TAG  v3.6
        GIT_PROGRESS TRUE
        GIT_SHALLOW TRUE
    )

    set(ONEDNN_LIBRARY_TYPE "STATIC")
    set(ONEDNN_BUILD_DOC "OFF")
    set(ONEDNN_BUILD_EXAMPLES "OFF")
    set(ONEDNN_BUILD_TESTS "OFF")
    set(ONEDNN_ENABLE_WORKLOAD "INFERENCE")
    set(ONEDNN_ENABLE_PRIMITIVE "MATMUL;REORDER")
    set(ONEDNN_BUILD_GRAPH "OFF")
    set(ONEDNN_ENABLE_JIT_PROFILING "OFF")
    set(ONEDNN_ENABLE_ITT_TASKS "OFF")
    set(ONEDNN_ENABLE_MAX_CPU_ISA "OFF")
    set(ONEDNN_ENABLE_CPU_ISA_HINTS "OFF")
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    FetchContent_MakeAvailable(oneDNN)
    
    list(APPEND LIBS dnnl)
endif()

message(STATUS "CPU extension compile flags: ${CXX_COMPILE_FLAGS}")

list(APPEND LIBS numa)

#
# _C extension
#
set(VLLM_EXT_SRC
    "csrc/cpu/activation.cpp"
    "csrc/cpu/attention.cpp"
    "csrc/cpu/cache.cpp"
    "csrc/cpu/utils.cpp"
    "csrc/cpu/layernorm.cpp"
    "csrc/cpu/pos_encoding.cpp"
    "csrc/cpu/torch_bindings.cpp")

if (AVX512_FOUND AND NOT AVX512_DISABLED)
    set(VLLM_EXT_SRC
        "csrc/cpu/quant.cpp"
        ${VLLM_EXT_SRC})
endif()

#
# Define extension targets
#

define_gpu_extension_target(
    _C
    DESTINATION vllm
    LANGUAGE CXX
    SOURCES ${VLLM_EXT_SRC}
    LIBRARIES ${LIBS}
    COMPILE_FLAGS ${CXX_COMPILE_FLAGS}
    USE_SABI 3
    WITH_SOABI
)

message(STATUS "Enabling C extension.")
