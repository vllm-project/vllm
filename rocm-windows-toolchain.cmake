# ROCm Windows GNU-Style Clang Toolchain
# Prevents Windows-MSVC.cmake from injecting cl-style flags into GNU clang++
# Runs BEFORE cmake compiler detection (via -DCMAKE_TOOLCHAIN_FILE=)

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)
set(CMAKE_SYSTEM_VERSION 10.0)

# Force GNU frontend variant — prevents cmake 4.3 from rejecting
# mixed HIP/CXX front ends when Windows-Clang.cmake dispatches
set(CMAKE_C_COMPILER_FRONTEND_VARIANT GNU)
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT GNU)
# HIP frontend variant: NOT set — cmake 3.29's HIP detection fails if
# this is preset. HIP uses hipcc/clang++ which always speaks GNU dialect.

# Windows SDK resource compiler (rc.exe) -- needed by enable_language(RC)
# in Windows-Clang.cmake's GNU path. Not in standard search paths without vcvars64.
set(CMAKE_RC_COMPILER "C:/Program Files (x86)/Windows Kits/10/bin/10.0.19041.0/x64/rc.exe" CACHE FILEPATH "")

# ROCm 7.13 compiler paths (GNU-style dirver, not clang-cl)
set(CMAKE_C_COMPILER "E:/ROCM-7.13.0-Windows/bin/clang.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "E:/ROCM-7.13.0-Windows/bin/clang++.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_HIP_COMPILER "E:/ROCM-7.13.0-Windows/bin/clang++.exe" CACHE FILEPATH "" FORCE)

# Prevent MSVC flag injection by zeroing C/CXX _INIT variables before
# Windows-Clang.cmake's GNU path populates them. DO NOT zero HIP _INIT
# variables — cmake's HIP detection needs the platform-init flags to
# compile the detection test program successfully.
set(CMAKE_C_FLAGS_INIT "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_INIT "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_DEBUG_INIT "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG_INIT "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE_INIT "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE_INIT "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_MINSIZEREL_INIT "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT "" CACHE STRING "" FORCE)
