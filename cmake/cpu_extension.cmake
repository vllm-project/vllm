include(FetchContent)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX_FOUND TRUE)
endif()


#
# Define environment variables for special configurations
#
set(ENABLE_X86_ISA $ENV{VLLM_CPU_X86})
set(ENABLE_ARM_BF16 $ENV{VLLM_CPU_ARM_BF16})

include_directories("${CMAKE_SOURCE_DIR}/csrc")

set (ENABLE_NUMA TRUE)

#
# Check the compile flags
#
if(MACOSX_FOUND)
    list(APPEND CXX_COMPILE_FLAGS
        "-DVLLM_CPU_EXTENSION")
else()
    list(APPEND CXX_COMPILE_FLAGS
        "-fopenmp"
        "-DVLLM_CPU_EXTENSION")

    # locate PyTorch's libgomp (e.g. site-packages/torch.libs/libgomp-947d5fa1.so.1.0.0)
    # and create a local shim dir with it. When PyTorch is built from source or packaged
    # by a distro (common on RISC-V, s390x, Fedora/RHEL aarch64), no vendored libgomp
    # exists and the shim dir is empty; fall back to the system libgomp in that case.
    vllm_prepare_torch_gomp_shim(VLLM_TORCH_GOMP_SHIM_DIR)

    if(VLLM_TORCH_GOMP_SHIM_DIR)
        find_library(OPEN_MP
            NAMES gomp
            PATHS "${VLLM_TORCH_GOMP_SHIM_DIR}"
            NO_DEFAULT_PATH
            REQUIRED
        )
        # Use the same libgomp as PyTorch at runtime
        set(ENV{LD_LIBRARY_PATH} "${VLLM_TORCH_GOMP_SHIM_DIR}:$ENV{LD_LIBRARY_PATH}")
    else()
        # Fall back to system / toolchain libgomp
        find_library(OPEN_MP NAMES gomp REQUIRED)
    endif()
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

if (MACOSX_FOUND AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(STATUS "Apple Silicon Detected")
    set(APPLE_SILICON_FOUND TRUE)
    set(ENABLE_NUMA OFF)
    check_sysctl(hw.optional.neon ASIMD_FOUND)
    check_sysctl(hw.optional.arm.FEAT_BF16 ARM_BF16_FOUND)
else()
    find_isa(${CPUINFO} "Power11" POWER11_FOUND)
    find_isa(${CPUINFO} "POWER10" POWER10_FOUND)
    find_isa(${CPUINFO} "POWER9" POWER9_FOUND)
    find_isa(${CPUINFO} "asimd" ASIMD_FOUND) # Check for ARM NEON support
    find_isa(${CPUINFO} "bf16" ARM_BF16_FOUND) # Check for ARM BF16 support
    find_isa(${CPUINFO} "S390" S390_FOUND)
    find_isa(${CPUINFO} "zvfhmin" RVV_FP16_FOUND) # Check for RISC-V Vector FP16 support
    find_isa(${CPUINFO} "zvfbfmin" RVV_BF16_FOUND) # Check for RISC-V Vector BF16 support

    # Support cross-compilation by allowing override via environment variables
    if (ENABLE_ARM_BF16)
        set(ARM_BF16_FOUND ON)
        message(STATUS "ARM BF16 support enabled via VLLM_CPU_ARM_BF16 environment variable")
    endif()
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64" OR ENABLE_X86_ISA)
    set(ENABLE_X86_ISA ON)
    if (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.3))
        message(FATAL_ERROR "X86 backend requires gcc/g++ >= 12.3")
    endif()
    list(APPEND CXX_COMPILE_FLAGS "-mf16c")
    list(APPEND CXX_COMPILE_FLAGS_AVX512 ${CXX_COMPILE_FLAGS})
    list(APPEND CXX_COMPILE_FLAGS_AVX2 ${CXX_COMPILE_FLAGS})
    list(APPEND CXX_COMPILE_FLAGS_AVX512
        "-mavx512f"
        "-mavx512vl"
        "-mavx512bw"
        "-mavx512dq")
    list(APPEND CXX_COMPILE_FLAGS_AVX512_AMX 
        ${CXX_COMPILE_FLAGS_AVX512}
        "-mamx-bf16"
        "-mamx-tile"
        "-mavx512bf16"
        "-mavx512vnni")
    list(APPEND CXX_COMPILE_FLAGS_AVX2
        "-mavx2")
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
    message(STATUS "RISC-V detected")
    # VLLM_RVV_VLEN selects the target VLEN. Auto-detected from /proc/cpuinfo
    # by default; override with -DVLLM_RVV_VLEN=128 or -DVLLM_RVV_VLEN=256.
    if(NOT DEFINED VLLM_RVV_VLEN)
        # Auto-detect: find the largest zvl<N>b in /proc/cpuinfo isa line.
        if(EXISTS /proc/cpuinfo)
            file(READ /proc/cpuinfo _cpuinfo)
            set(_best 0)
            foreach(_n IN ITEMS 128 256 512 1024)
                if(_cpuinfo MATCHES "zvl${_n}b")
                    set(_best ${_n})
                endif()
            endforeach()
            if(_best GREATER 0)
                set(VLLM_RVV_VLEN ${_best})
            endif()
        endif()
        # If auto-detect failed (no /proc/cpuinfo or no zvl<N>b reported)
        # but the compiler supports RVV, require explicit specification.
        if(NOT DEFINED VLLM_RVV_VLEN AND (RVV_FP16_FOUND OR RVV_BF16_FOUND))
            message(FATAL_ERROR
                "RISC-V RVV is available but VLEN could not be auto-detected. "
                "Please specify VLEN explicitly:\n"
                "  -DVLLM_RVV_VLEN=128   (for VLEN=128 hardware)\n"
                "  -DVLLM_RVV_VLEN=256   (for VLEN=256 hardware, e.g. Spacemit X100)\n"
                "  -DVLLM_RVV_VLEN=0     (force scalar, no RVV)")
        endif()
    endif()
    if(VLLM_RVV_VLEN AND VLLM_RVV_VLEN GREATER 0)
        message(STATUS "RISC-V RVV VLEN=${VLLM_RVV_VLEN}")
        if(RVV_BF16_FOUND)
            message(STATUS "BF16 extension detected")
            set(MARCH_FLAGS -march=rv64gcv_zvfh_zfbfmin_zvfbfmin_zvl${VLLM_RVV_VLEN}b -mrvv-vector-bits=zvl -mabi=lp64d)
            add_compile_definitions(RISCV_BF16_SUPPORT)
        elseif(RVV_FP16_FOUND)
            message(WARNING "BF16 functionality is not available")
            set(MARCH_FLAGS -march=rv64gcv_zvfh_zvl${VLLM_RVV_VLEN}b -mrvv-vector-bits=zvl -mabi=lp64d)
        else()
            message(STATUS "compile riscv with scalar (no FP16/BF16)")
            set(MARCH_FLAGS -march=rv64gc)
        endif()
    else()
        message(STATUS "compile riscv with scalar")
        set(MARCH_FLAGS -march=rv64gc)
    endif()
    list(APPEND CXX_COMPILE_FLAGS ${MARCH_FLAGS})
else()
    message(FATAL_ERROR "vLLM CPU backend requires X86, Power9+ ISA, S390X ISA, ARMv8 or RISC-V support.")
endif()


# Build oneDNN for GEMM kernels
if (ENABLE_X86_ISA OR (ASIMD_FOUND AND NOT APPLE_SILICON_FOUND) OR POWER9_FOUND OR POWER10_FOUND OR POWER11_FOUND)
    # Fetch and build Arm Compute Library (ACL) as oneDNN's backend for AArch64
    # TODO [fadara01]: remove this once ACL can be fetched and built automatically as a dependency of oneDNN
    set(ONEDNN_AARCH64_USE_ACL OFF CACHE BOOL "")
    if(ASIMD_FOUND)
        # Set number of parallel build processes
        include(ProcessorCount)
        ProcessorCount(NPROC)
        if(NOT NPROC)
            set(NPROC 4)
        endif()

        # Fetch and populate ACL
        if(DEFINED ENV{ACL_ROOT_DIR} AND IS_DIRECTORY "$ENV{ACL_ROOT_DIR}")
            message(STATUS "Using ACL from specified source directory: $ENV{ACL_ROOT_DIR}")
        else()
            message(STATUS "Downloading Arm Compute Library (ACL) from GitHub")
            FetchContent_Populate(arm_compute
                SUBBUILD_DIR "${FETCHCONTENT_BASE_DIR}/arm_compute-subbuild"
                SOURCE_DIR   "${FETCHCONTENT_BASE_DIR}/arm_compute-src"
                GIT_REPOSITORY https://github.com/ARM-software/ComputeLibrary.git
                GIT_TAG        v52.6.0
                GIT_SHALLOW    TRUE
                GIT_PROGRESS   TRUE
            )
            set(ENV{ACL_ROOT_DIR} "${arm_compute_SOURCE_DIR}")
            set(ACL_LIB_DIR "$ENV{ACL_ROOT_DIR}/build")
        endif()

        # Build ACL with CMake
        set(_cmake_config_cmd
             ${CMAKE_COMMAND} -G Ninja -B build 
            -DARM_COMPUTE_BUILD_SHARED_LIB=OFF 
            -DCMAKE_BUILD_TYPE=Release 
            -DARM_COMPUTE_ARCH=armv8.2-a 
            -DARM_COMPUTE_ENABLE_ASSERTS=OFF 
            -DARM_COMPUTE_ENABLE_CPPTHREADS=OFF 
            -DARM_COMPUTE_ENABLE_OPENMP=ON 
            -DARM_COMPUTE_ENABLE_WERROR=OFF 
            -DARM_COMPUTE_BUILD_EXAMPLES=OFF 
            -DARM_COMPUTE_BUILD_TESTING=OFF)
        set(_cmake_build_cmd
            ${CMAKE_COMMAND} --build build -- -j${NPROC}
        )

        execute_process(
            COMMAND ${_cmake_config_cmd}
            WORKING_DIRECTORY "$ENV{ACL_ROOT_DIR}"
        )
        execute_process(
            COMMAND ${_cmake_build_cmd}
            WORKING_DIRECTORY "$ENV{ACL_ROOT_DIR}"
            RESULT_VARIABLE _acl_rc
        )

        if(NOT _acl_rc EQUAL 0)
            message(FATAL_ERROR "ACL SCons build failed (exit ${_acl_rc}).")
        endif()
        message(STATUS "Arm Compute Library (ACL) built successfully.")

        # VLLM/oneDNN settings for ACL
        set(ONEDNN_AARCH64_USE_ACL ON CACHE BOOL "" FORCE)
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
        if(ASIMD_FOUND AND NOT APPLE_SILICON_FOUND)
            message(STATUS "aarch64 detected: using pinned oneDNN commit 9c5be1cc59e368aebf0909e6cf20f981ea61462a")
            FetchContent_Declare(
                oneDNN
                GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
                GIT_TAG        9c5be1cc59e368aebf0909e6cf20f981ea61462a
                GIT_PROGRESS   TRUE
                GIT_SHALLOW    FALSE
            )
        else()
            FetchContent_Declare(
                oneDNN
                GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
                GIT_TAG        v3.10
                GIT_PROGRESS   TRUE
                GIT_SHALLOW    TRUE
            )
        endif()
    endif()

    set(ONEDNN_LIBRARY_TYPE "STATIC")
    set(ONEDNN_BUILD_DOC "OFF")
    set(ONEDNN_BUILD_EXAMPLES "OFF")
    set(ONEDNN_BUILD_TESTS "OFF")
    set(ONEDNN_ENABLE_WORKLOAD "INFERENCE")
    set(ONEDNN_ENABLE_PRIMITIVE "MATMUL;REORDER")
    set(ONEDNN_BUILD_GRAPH "OFF")
    set(ONEDNN_ENABLE_JIT_PROFILING "ON")
    set(ONEDNN_ENABLE_ITT_TASKS "OFF")
    set(ONEDNN_ENABLE_MAX_CPU_ISA "ON")
    set(ONEDNN_ENABLE_CPU_ISA_HINTS "ON")
    set(ONEDNN_VERBOSE "ON")
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
    target_link_libraries(dnnl_ext dnnl torch)
    if (ENABLE_X86_ISA)
        target_compile_options(dnnl_ext PRIVATE ${CXX_COMPILE_FLAGS_AVX2} -fPIC)
    else()
        target_compile_options(dnnl_ext PRIVATE ${CXX_COMPILE_FLAGS} -fPIC)
    endif()
    list(APPEND LIBS dnnl_ext)


    set(USE_ONEDNN ON)
else()
    set(USE_ONEDNN OFF)
endif()

# TODO: Refactor this
if (ENABLE_X86_ISA)
    message(STATUS "CPU extension (AVX512F + BF16 + VNNI + AMX) compile flags: ${CXX_COMPILE_FLAGS_AVX512_AMX}")
    message(STATUS "CPU extension (AVX512F) compile flags: ${CXX_COMPILE_FLAGS_AVX512}")
    message(STATUS "CPU extension (AVX2) compile flags: ${CXX_COMPILE_FLAGS_AVX2}")
else()
    message(STATUS "CPU extension compile flags: ${CXX_COMPILE_FLAGS}")
endif()

if(ENABLE_NUMA)
    list(APPEND LIBS numa)
else()
    message(STATUS "NUMA is disabled")
    add_compile_definitions(-DVLLM_NUMA_DISABLED)
endif()

#
# Generate CPU attention dispatch header
#
message(STATUS "Generating CPU attention dispatch header")
execute_process(
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/csrc/cpu/generate_cpu_attn_dispatch.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/csrc/cpu
    RESULT_VARIABLE GEN_RESULT
)
if(NOT GEN_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate CPU attention dispatch header")
endif()

#
# _C extension
#
set(VLLM_EXT_SRC
    "csrc/cpu/activation.cpp"
    "csrc/cpu/utils.cpp"
    "csrc/cpu/spec_decode_utils.cpp"
    "csrc/cpu/layernorm.cpp"
    "csrc/cpu/mla_decode.cpp"
    "csrc/cpu/pos_encoding.cpp"
    "csrc/moe/dynamic_4bit_int_moe_cpu.cpp"
    "csrc/cpu/cpu_attn.cpp"
    "csrc/cpu/torch_bindings.cpp")

if (ASIMD_FOUND AND NOT APPLE_SILICON_FOUND)
    set(VLLM_EXT_SRC
        "csrc/cpu/shm.cpp"
        "csrc/cpu/activation_lut_bf16.cpp"
        ${VLLM_EXT_SRC})
endif()

if(USE_ONEDNN)
    set(VLLM_EXT_SRC
        "csrc/cpu/dnnl_kernels.cpp"
        ${VLLM_EXT_SRC})
endif()

if (ENABLE_X86_ISA)
    set(VLLM_EXT_SRC_SGL
        "csrc/cpu/sgl-kernels/fla.cpp"
        "csrc/cpu/sgl-kernels/conv.cpp"
        "csrc/cpu/sgl-kernels/gemm.cpp"
        "csrc/cpu/sgl-kernels/gemm_int8.cpp"
        "csrc/cpu/sgl-kernels/gemm_fp8.cpp"
        "csrc/cpu/sgl-kernels/gemm_int4.cpp"
        "csrc/cpu/sgl-kernels/moe.cpp"
        "csrc/cpu/sgl-kernels/moe_int8.cpp"
        "csrc/cpu/sgl-kernels/moe_int4.cpp"
        "csrc/cpu/sgl-kernels/moe_fp8.cpp")

    set(VLLM_EXT_SRC_AVX512
        "csrc/cpu/shm.cpp"
        "csrc/cpu/cpu_wna16.cpp"
        "csrc/cpu/cpu_fused_moe.cpp"
        "csrc/cpu/utils.cpp"
        "csrc/cpu/spec_decode_utils.cpp"
        "csrc/cpu/cpu_attn.cpp"
        "csrc/cpu/dnnl_kernels.cpp"
        "csrc/cpu/torch_bindings.cpp"
        # TODO: Remove these files
        "csrc/cpu/activation.cpp"
        "csrc/cpu/layernorm.cpp"
        "csrc/cpu/mla_decode.cpp"
        "csrc/cpu/pos_encoding.cpp"
        "csrc/moe/dynamic_4bit_int_moe_cpu.cpp") 

    set(VLLM_EXT_SRC_AVX2
        "csrc/cpu/utils.cpp"
        "csrc/cpu/spec_decode_utils.cpp"
        "csrc/cpu/cpu_attn.cpp"
        "csrc/cpu/dnnl_kernels.cpp"
        "csrc/cpu/torch_bindings.cpp"
        # TODO: Remove these files
        "csrc/cpu/activation.cpp"
        "csrc/cpu/layernorm.cpp"
        "csrc/cpu/mla_decode.cpp"
        "csrc/cpu/pos_encoding.cpp"
        "csrc/moe/dynamic_4bit_int_moe_cpu.cpp") 

    message(STATUS "CPU extension (AVX512F + BF16 + VNNI + AMX) source files: ${VLLM_EXT_SRC_AVX512} ${VLLM_EXT_SRC_SGL}")
    message(STATUS "CPU extension (AVX512F) source files: ${VLLM_EXT_SRC_AVX512}")
    message(STATUS "CPU extension (AVX2) source files: ${VLLM_EXT_SRC_AVX2}")

    set(_C_LIBS numa dnnl_ext)
    set(_C_AVX512_LIBS numa dnnl_ext)
    set(_C_AVX2_LIBS numa dnnl_ext)

    # AMX + AVX512F + AVX512BF16 + AVX512VNNI
    define_extension_target(
        _C
        DESTINATION vllm
        LANGUAGE CXX
        SOURCES ${VLLM_EXT_SRC_AVX512} ${VLLM_EXT_SRC_SGL}
        LIBRARIES ${_C_LIBS}
        COMPILE_FLAGS ${CXX_COMPILE_FLAGS_AVX512_AMX}
        USE_SABI 3
        WITH_SOABI
    )

    # For AMX kernels
    target_compile_definitions(_C PRIVATE "-DCPU_CAPABILITY_AMXBF16")

    # AVX512F 
    define_extension_target(
        _C_AVX512
        DESTINATION vllm
        LANGUAGE CXX
        SOURCES ${VLLM_EXT_SRC_AVX512}
        LIBRARIES ${_C_AVX512_LIBS}
        COMPILE_FLAGS ${CXX_COMPILE_FLAGS_AVX512}
        USE_SABI 3
        WITH_SOABI
    )

    # AVX2 
    define_extension_target(
        _C_AVX2
        DESTINATION vllm
        LANGUAGE CXX
        SOURCES ${VLLM_EXT_SRC_AVX2}
        LIBRARIES ${_C_AVX2_LIBS}
        COMPILE_FLAGS ${CXX_COMPILE_FLAGS_AVX2}
        USE_SABI 3
        WITH_SOABI
    )
else()
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
endif()

message(STATUS "Enabling C extension.")
