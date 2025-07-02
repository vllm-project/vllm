include(FetchContent)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX_FOUND TRUE)
endif()


#
# Define environment variables for special configurations
#
if(DEFINED ENV{VLLM_CPU_AVX512BF16})
    set(ENABLE_AVX512BF16 ON)
endif()

include_directories("${CMAKE_SOURCE_DIR}/csrc")


set (ENABLE_NUMA TRUE)

#
# Check the compile flags
#

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    list(APPEND CXX_COMPILE_FLAGS
        "-mf16c"
    )
endif()

if(MACOSX_FOUND)
    list(APPEND CXX_COMPILE_FLAGS
        "-DVLLM_CPU_EXTENSION")
else()
    list(APPEND CXX_COMPILE_FLAGS
        "-fopenmp"
        "-DVLLM_CPU_EXTENSION")
endif()

if (NOT MACOSX_FOUND)
    execute_process(COMMAND cat /proc/cpuinfo
                    RESULT_VARIABLE CPUINFO_RET
                    OUTPUT_VARIABLE CPUINFO)
    if (NOT CPUINFO_RET EQUAL 0)
        message(FATAL_ERROR "Failed to check CPU features via /proc/cpuinfo")
    endif()
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

if (MACOSX_FOUND AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(APPLE_SILICON_FOUND TRUE)
else()
    find_isa(${CPUINFO} "avx2" AVX2_FOUND)
    find_isa(${CPUINFO} "avx512f" AVX512_FOUND)
    find_isa(${CPUINFO} "Power11" POWER11_FOUND)
    find_isa(${CPUINFO} "POWER10" POWER10_FOUND)
    find_isa(${CPUINFO} "POWER9" POWER9_FOUND)
    find_isa(${CPUINFO} "asimd" ASIMD_FOUND) # Check for ARM NEON support
    find_isa(${CPUINFO} "bf16" ARM_BF16_FOUND) # Check for ARM BF16 support
    find_isa(${CPUINFO} "S390" S390_FOUND)
endif()


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
            set(ENABLE_AVX512BF16 ON)
        else()
            set(ENABLE_AVX512BF16 OFF)
            message(WARNING "Disable AVX512-BF16 ISA support, requires gcc/g++ >= 12.3")
        endif()
    else()
        set(ENABLE_AVX512BF16 OFF)
        message(WARNING "Disable AVX512-BF16 ISA support, no avx512_bf16 found in local CPU flags." " If cross-compilation is required, please set env VLLM_CPU_AVX512BF16=1.")
    endif()

    find_isa(${CPUINFO} "avx512_vnni" AVX512VNNI_FOUND)
    if (AVX512VNNI_FOUND)
        list(APPEND CXX_COMPILE_FLAGS "-mavx512vnni")
        set(ENABLE_AVX512VNNI ON) 
    endif() 
    
elseif (AVX2_FOUND)
    list(APPEND CXX_COMPILE_FLAGS "-mavx2")
    message(WARNING "vLLM CPU backend using AVX2 ISA")
    
elseif (POWER9_FOUND OR POWER10_FOUND OR POWER11_FOUND)
    message(STATUS "PowerPC detected")
    if (POWER9_FOUND)
        list(APPEND CXX_COMPILE_FLAGS
            "-mvsx"
            "-mcpu=power9"
            "-mtune=power9")
    elseif (POWER10_FOUND OR POWER11_FOUND)
        list(APPEND CXX_COMPILE_FLAGS
            "-mvsx"
            "-mcpu=power10"
            "-mtune=power10")
    endif()

elseif (ASIMD_FOUND)
    message(STATUS "ARMv8 or later architecture detected")
    if(ARM_BF16_FOUND)
        message(STATUS "BF16 extension detected")
        set(MARCH_FLAGS "-march=armv8.2-a+bf16+dotprod+fp16")
        add_compile_definitions(ARM_BF16_SUPPORT)
    else()
        message(WARNING "BF16 functionality is not available")
        set(MARCH_FLAGS "-march=armv8.2-a+dotprod+fp16")  
    endif()
    list(APPEND CXX_COMPILE_FLAGS ${MARCH_FLAGS})     
elseif(APPLE_SILICON_FOUND)
    message(STATUS "Apple Silicon Detected")
    set(ENABLE_NUMA OFF)
elseif (S390_FOUND)
    message(STATUS "S390 detected")
    # Check for S390 VXE support
    list(APPEND CXX_COMPILE_FLAGS
        "-mvx"
        "-mzvector"
        "-march=native"
        "-mtune=native")
else()
    message(FATAL_ERROR "vLLM CPU backend requires AVX512, AVX2, Power9+ ISA, S390X ISA or ARMv8 support.")
endif()

#
# Build oneDNN for W8A8 GEMM kernels (only for x86-AVX512 platforms)
#
if (AVX512_FOUND AND NOT AVX512_DISABLED)
    FetchContent_Declare(
        oneDNN
        GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
        GIT_TAG  v3.7.1
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
elseif(POWER10_FOUND)
    FetchContent_Declare(
        oneDNN
        GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
        GIT_TAG v3.7.2
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

    set(DNNL_CPU_RUNTIME "OMP")

    FetchContent_MakeAvailable(oneDNN)

    list(APPEND LIBS dnnl)
endif()

message(STATUS "CPU extension compile flags: ${CXX_COMPILE_FLAGS}")

if(ENABLE_NUMA)
    list(APPEND LIBS numa)
else()
    message(STATUS "NUMA is disabled")
    add_compile_definitions(-DVLLM_NUMA_DISABLED)
endif()

#
# _C extension
#
set(VLLM_EXT_SRC
    "csrc/cpu/activation.cpp"
    "csrc/cpu/attention.cpp"
    "csrc/cpu/cache.cpp"
    "csrc/cpu/utils.cpp"
    "csrc/cpu/layernorm.cpp"
    "csrc/cpu/mla_decode.cpp"
    "csrc/cpu/pos_encoding.cpp"
    "csrc/cpu/torch_bindings.cpp")

if (AVX512_FOUND AND NOT AVX512_DISABLED)
    set(VLLM_EXT_SRC
        "csrc/cpu/quant.cpp"
        "csrc/cpu/shm.cpp"
        ${VLLM_EXT_SRC})
    if (ENABLE_AVX512BF16 AND ENABLE_AVX512VNNI)
        set(VLLM_EXT_SRC
            "csrc/cpu/sgl-kernels/gemm.cpp"
            "csrc/cpu/sgl-kernels/gemm_int8.cpp"
            "csrc/cpu/sgl-kernels/gemm_fp8.cpp"
            "csrc/cpu/sgl-kernels/moe.cpp"
            "csrc/cpu/sgl-kernels/moe_int8.cpp"
            "csrc/cpu/sgl-kernels/moe_fp8.cpp"
            ${VLLM_EXT_SRC})
        add_compile_definitions(-DCPU_CAPABILITY_AVX512)
    endif()
elseif(POWER10_FOUND)
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
