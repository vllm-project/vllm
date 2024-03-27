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
list(APPEND CXX_COMPILE_FLAGS 
    "-fopenmp"
    "-DVLLM_CPU_EXTENSION")

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

find_isa(${CPUINFO} "avx512f" AVX512_FOUND)

if (AVX512_FOUND)
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
else()
    message(FATAL_ERROR "vLLM CPU backend requires AVX512 ISA support.")
endif()

message(STATUS "CPU extension compile flags: ${CXX_COMPILE_FLAGS}")


#
# Define extension targets
#

#
# _C extension
#
set(VLLM_EXT_SRC
    "csrc/cpu/activation.cpp"
    "csrc/cpu/attention.cpp"
    "csrc/cpu/cache.cpp"
    "csrc/cpu/layernorm.cpp"
    "csrc/cpu/pos_encoding.cpp"
    "csrc/cpu/pybind.cpp")

define_gpu_extension_target(
    _C
    DESTINATION vllm
    LANGUAGE CXX
    SOURCES ${VLLM_EXT_SRC}
    COMPILE_FLAGS ${CXX_COMPILE_FLAGS}
    WITH_SOABI 
)

add_custom_target(default)
message(STATUS "Enabling C extension.")
add_dependencies(default _C)

