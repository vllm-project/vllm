include(FetchContent)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX_FOUND TRUE)
endif()


#
# Define environment variables for special configurations
#
set(ENABLE_AVX512BF16 $ENV{VLLM_CPU_AVX512BF16})
set(ENABLE_AVX512VNNI $ENV{VLLM_CPU_AVX512VNNI})
set(ENABLE_AMXBF16 $ENV{VLLM_CPU_AMXBF16})

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


function(check_sysctl TARGET OUT)
    execute_process(COMMAND sysctl -n "${TARGET}"
                    RESULT_VARIABLE SYSCTL_RET
                    OUTPUT_VARIABLE SYSCTL_INFO
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(SYSCTL_RET EQUAL 0 AND
      (SYSCTL_INFO STREQUAL "1" OR SYSCTL_INFO GREATER 0))
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
    message(STATUS "Apple Silicon Detected")
    set(APPLE_SILICON_FOUND TRUE)
    set(ENABLE_NUMA OFF)
    check_sysctl(hw.optional.neon ASIMD_FOUND)
    check_sysctl(hw.optional.arm.FEAT_BF16 ARM_BF16_FOUND)
else()
    find_isa(${CPUINFO} "avx2" AVX2_FOUND)
    find_isa(${CPUINFO} "avx512f" AVX512_FOUND)
    find_isa(${CPUINFO} "Power11" POWER11_FOUND)
    find_isa(${CPUINFO} "POWER10" POWER10_FOUND)
    find_isa(${CPUINFO} "POWER9" POWER9_FOUND)
    find_isa(${CPUINFO} "asimd" ASIMD_FOUND) # Check for ARM NEON support
    find_isa(${CPUINFO} "bf16" ARM_BF16_FOUND) # Check for ARM BF16 support
    find_isa(${CPUINFO} "S390" S390_FOUND)
    find_isa(${CPUINFO} "v" RVV_FOUND) # Check for RISC-V RVV support
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
    if (AVX512VNNI_FOUND OR ENABLE_AVX512VNNI)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.3)
            list(APPEND CXX_COMPILE_FLAGS "-mavx512vnni")
            set(ENABLE_AVX512VNNI ON)
        else()
            set(ENABLE_AVX512VNNI OFF)
            message(WARNING "Disable AVX512-VNNI ISA support, requires gcc/g++ >= 12.3")
        endif()
    else()
        set(ENABLE_AVX512VNNI OFF)
        message(WARNING "Disable AVX512-VNNI ISA support, no avx512_vnni found in local CPU flags." " If cross-compilation is required, please set env VLLM_CPU_AVX512VNNI=1.")
    endif()

    find_isa(${CPUINFO} "amx_bf16" AMXBF16_FOUND)
    if (AMXBF16_FOUND OR ENABLE_AMXBF16)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.3)
            list(APPEND CXX_COMPILE_FLAGS "-mamx-bf16" "-mamx-tile")
            set(ENABLE_AMXBF16 ON)
            add_compile_definitions(-DCPU_CAPABILITY_AMXBF16)
        else()
            set(ENABLE_AMXBF16 OFF)
            message(WARNING "Disable AMX_BF16 ISA support, requires gcc/g++ >= 12.3")
        endif()
    else()
        set(ENABLE_AMXBF16 OFF)
        message(WARNING "Disable AMX_BF16 ISA support, no amx_bf16 found in local CPU flags." " If cross-compilation is required, please set env VLLM_CPU_AMXBF16=1.")
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
elseif (S390_FOUND)
    message(STATUS "S390 detected")
    # Check for S390 VXE support
    list(APPEND CXX_COMPILE_FLAGS
        "-mvx"
        "-mzvector"
        "-march=native"
        "-mtune=native")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    if(RVV_FOUND)
	    message(FAIL_ERROR "Can't support rvv now.")
    else()
        list(APPEND CXX_COMPILE_FLAGS "-march=rv64gc")
    endif()
else()
    message(FATAL_ERROR "vLLM CPU backend requires AVX512, AVX2, Power9+ ISA, S390X ISA, ARMv8 or RISC-V support.")
endif()


# Build oneDNN for GEMM kernels (only for x86-AVX512 /ARM platforms)
if ((AVX512_FOUND AND NOT AVX512_DISABLED) OR (ASIMD_FOUND AND NOT APPLE_SILICON_FOUND) OR POWER9_FOUND OR POWER10_FOUND OR POWER11_FOUND)
    # Fetch and build Arm Compute Library (ACL) as oneDNN's backend for AArch64
    # TODO [fadara01]: remove this once ACL can be fetched and built automatically as a dependency of oneDNN
    if(ASIMD_FOUND)
        if(DEFINED ENV{ACL_ROOT_DIR} AND IS_DIRECTORY "$ENV{ACL_ROOT_DIR}")
            message(STATUS "Using ACL from specified source directory: $ENV{ACL_ROOT_DIR}")
        else()
            message(STATUS "Downloading Arm Compute Library (ACL) from GitHub")
            FetchContent_Populate(arm_compute
                SUBBUILD_DIR "${FETCHCONTENT_BASE_DIR}/arm_compute-subbuild"
                SOURCE_DIR   "${FETCHCONTENT_BASE_DIR}/arm_compute-src"
                GIT_REPOSITORY https://github.com/ARM-software/ComputeLibrary.git
                GIT_TAG        v52.2.0
                GIT_SHALLOW    TRUE
                GIT_PROGRESS   TRUE
            )
            set(ENV{ACL_ROOT_DIR} "${arm_compute_SOURCE_DIR}")
        endif()

        # Build ACL with scons
        include(ProcessorCount)
        ProcessorCount(_NPROC)
        set(_scons_cmd
        scons -j${_NPROC}
            Werror=0 debug=0 neon=1 examples=0 embed_kernels=0 os=linux
            arch=armv8.2-a build=native benchmark_examples=0 fixed_format_kernels=1
            multi_isa=1 openmp=1 cppthreads=0
        )

        # locate PyTorch's libgomp (e.g. site-packages/torch.libs/libgomp-947d5fa1.so.1.0.0)
        # and create a local shim dir with it
        include("${CMAKE_CURRENT_LIST_DIR}/utils.cmake")
        vllm_prepare_torch_gomp_shim(VLLM_TORCH_GOMP_SHIM_DIR)

        if(NOT VLLM_TORCH_GOMP_SHIM_DIR STREQUAL "")
            list(APPEND _scons_cmd extra_link_flags=-L${VLLM_TORCH_GOMP_SHIM_DIR})
        endif()

        execute_process(
            COMMAND ${_scons_cmd}
            WORKING_DIRECTORY "$ENV{ACL_ROOT_DIR}"
            RESULT_VARIABLE _acl_rc
        )
        if(NOT _acl_rc EQUAL 0)
            message(FATAL_ERROR "ACL SCons build failed (exit ${_acl_rc}).")
        endif()

        set(ONEDNN_AARCH64_USE_ACL "ON")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ENV{ACL_ROOT_DIR}/build/")
        add_compile_definitions(VLLM_USE_ACL)
    endif()

    set(FETCHCONTENT_SOURCE_DIR_ONEDNN "$ENV{FETCHCONTENT_SOURCE_DIR_ONEDNN}" CACHE PATH "Path to a local oneDNN source directory.")

    if(FETCHCONTENT_SOURCE_DIR_ONEDNN)
        message(STATUS "Using oneDNN from specified source directory: ${FETCHCONTENT_SOURCE_DIR_ONEDNN}")
        FetchContent_Declare(
            oneDNN
            SOURCE_DIR ${FETCHCONTENT_SOURCE_DIR_ONEDNN}
        )
    else()
        message(STATUS "Downloading oneDNN from GitHub")
        FetchContent_Declare(
            oneDNN
            GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
            GIT_TAG v3.9
            GIT_PROGRESS TRUE
            GIT_SHALLOW TRUE
        )
    endif()

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
    set(ONEDNN_VERBOSE "OFF")
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    set(VLLM_BUILD_TYPE ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE "Release") # remove oneDNN debug symbols to reduce size
    FetchContent_MakeAvailable(oneDNN)
    set(CMAKE_BUILD_TYPE ${VLLM_BUILD_TYPE})
    add_library(dnnl_ext OBJECT "csrc/cpu/dnnl_helper.cpp")
    target_include_directories(
        dnnl_ext
        PUBLIC ${oneDNN_SOURCE_DIR}/include
        PUBLIC ${oneDNN_BINARY_DIR}/include
        PRIVATE ${oneDNN_SOURCE_DIR}/src
    )
    target_link_libraries(dnnl_ext dnnl)
    target_compile_options(dnnl_ext PRIVATE ${CXX_COMPILE_FLAGS} -fPIC)
    list(APPEND LIBS dnnl_ext)
    set(USE_ONEDNN ON)
else()
    set(USE_ONEDNN OFF)
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
    "csrc/cpu/utils.cpp"
    "csrc/cpu/layernorm.cpp"
    "csrc/cpu/mla_decode.cpp"
    "csrc/cpu/pos_encoding.cpp"
    "csrc/moe/dynamic_4bit_int_moe_cpu.cpp"
    "csrc/cpu/cpu_attn.cpp"
    "csrc/cpu/scratchpad_manager.cpp"
    "csrc/cpu/torch_bindings.cpp")

if (AVX512_FOUND AND NOT AVX512_DISABLED)
    set(VLLM_EXT_SRC
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
endif()

if(USE_ONEDNN)
    set(VLLM_EXT_SRC
        "csrc/cpu/dnnl_kernels.cpp"
        ${VLLM_EXT_SRC})
endif()

message(STATUS "CPU extension source files: ${VLLM_EXT_SRC}")

#
# Define extension targets
#

define_extension_target(
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
