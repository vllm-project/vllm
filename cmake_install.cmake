# Install script for directory: /mnt/d/Programming/Projects/vllm-lm-format-enforcer

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

set(CMAKE_INSTALL_LOCAL_ONLY TRUE)
if(CMAKE_INSTALL_COMPONENT STREQUAL "cumem_allocator" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm" TYPE MODULE FILES "/mnt/d/Programming/Projects/vllm-lm-format-enforcer/cumem_allocator.abi3.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so"
         OLD_RPATH "/lib/intel64:/lib/intel64_win:/lib/win-x64:/usr/lib/wsl/lib:/home/noamgat/mambaforge/envs/vllmdev312/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/mnt/d/Programming/Projects/vllm-lm-format-enforcer/_deps/cutlass-build/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_C" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C.abi3.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C.abi3.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm" TYPE MODULE FILES "/mnt/d/Programming/Projects/vllm-lm-format-enforcer/_C.abi3.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C.abi3.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C.abi3.so"
         OLD_RPATH "/lib/intel64:/lib/intel64_win:/lib/win-x64:/home/noamgat/mambaforge/envs/vllmdev312/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/lib/wsl/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C.abi3.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_moe_C" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C.abi3.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C.abi3.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm" TYPE MODULE FILES "/mnt/d/Programming/Projects/vllm-lm-format-enforcer/_moe_C.abi3.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C.abi3.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C.abi3.so"
         OLD_RPATH "/lib/intel64:/lib/intel64_win:/lib/win-x64:/home/noamgat/mambaforge/envs/vllmdev312/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/lib/wsl/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C.abi3.so")
    endif()
  endif()
endif()

file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn")
set(CMAKE_INSTALL_LOCAL_ONLY FALSE)
set(OLD_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/vllm/")
if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/mnt/d/Programming/Projects/vllm-lm-format-enforcer/vllm-flash-attn/cmake_install.cmake")
endif()

set(CMAKE_INSTALL_PREFIX "${OLD_CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_LOCAL_ONLY TRUE)
if(CMAKE_INSTALL_COMPONENT STREQUAL "_vllm_fa2_C" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn" TYPE DIRECTORY FILES "/mnt/d/Programming/Projects/vllm-lm-format-enforcer/_deps/vllm-flash-attn-src/vllm_flash_attn/" FILES_MATCHING REGEX "/[^/]*\\.py$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_vllm_fa3_C" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn" TYPE DIRECTORY FILES "/mnt/d/Programming/Projects/vllm-lm-format-enforcer/_deps/vllm-flash-attn-src/vllm_flash_attn/" FILES_MATCHING REGEX "/[^/]*\\.py$")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/mnt/d/Programming/Projects/vllm-lm-format-enforcer/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/mnt/d/Programming/Projects/vllm-lm-format-enforcer/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
