# vLLM flash attention requires VLLM_GPU_ARCHES to contain the set of target
# arches in the CMake syntax (75-real, 89-virtual, etc), since we clear the
# arches in the CUDA case (and instead set the gencodes on a per file basis)
# we need to manually set VLLM_GPU_ARCHES here.
if(VLLM_GPU_LANG STREQUAL "CUDA")
  foreach(_ARCH ${CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    list(APPEND VLLM_GPU_ARCHES "${_ARCH}-real")
  endforeach()
endif()

#
# Build vLLM flash attention from source
#
# IMPORTANT: This has to be the last thing we do, because vllm-flash-attn uses the same macros/functions as vLLM.
# Because functions all belong to the global scope, vllm-flash-attn's functions overwrite vLLMs.
# They should be identical but if they aren't, this is a massive footgun.
#
# The vllm-flash-attn install rules are nested under vllm to make sure the library gets installed in the correct place.
# To only install vllm-flash-attn, use --component _vllm_fa2_C (for FA2), --component _vllm_fa3_C (for FA3),
# or --component _vllm_fa4_cutedsl_C (for FA4 CuteDSL Python files).
# If no component is specified, vllm-flash-attn is still installed.

# If VLLM_FLASH_ATTN_SRC_DIR is set, vllm-flash-attn is installed from that directory instead of downloading.
# This is to enable local development of vllm-flash-attn within vLLM.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{VLLM_FLASH_ATTN_SRC_DIR})
  set(VLLM_FLASH_ATTN_SRC_DIR $ENV{VLLM_FLASH_ATTN_SRC_DIR})
endif()

if(VLLM_FLASH_ATTN_SRC_DIR)
  FetchContent_Declare(
          vllm-flash-attn SOURCE_DIR 
          ${VLLM_FLASH_ATTN_SRC_DIR}
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn
  )
else()
  FetchContent_Declare(
          vllm-flash-attn
          GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
          GIT_TAG 7d346be62004163f0b59f965761b122cc40bd0a3
          GIT_PROGRESS TRUE
          # Don't share the vllm-flash-attn build between build types
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn
  )
endif()


# Install rules for FA components need the install prefix nested under vllm/
# These run at install time, before the FA library's own install rules
foreach(_FA_COMPONENT _vllm_fa2_C _vllm_fa3_C)
  install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY FALSE)" COMPONENT ${_FA_COMPONENT})
  install(CODE "set(OLD_CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}\")" COMPONENT ${_FA_COMPONENT})
  install(CODE "set(CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}/vllm/\")" COMPONENT ${_FA_COMPONENT})
endforeach()

# Fetch the vllm-flash-attn library
FetchContent_MakeAvailable(vllm-flash-attn)
message(STATUS "vllm-flash-attn is available at ${vllm-flash-attn_SOURCE_DIR}")

# Restore the install prefix after FA's install rules
foreach(_FA_COMPONENT _vllm_fa2_C _vllm_fa3_C)
  install(CODE "set(CMAKE_INSTALL_PREFIX \"\${OLD_CMAKE_INSTALL_PREFIX}\")" COMPONENT ${_FA_COMPONENT})
  install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" COMPONENT ${_FA_COMPONENT})
endforeach()

# Install shared Python files for both FA2 and FA3 components
foreach(_FA_COMPONENT _vllm_fa2_C _vllm_fa3_C)
  # Ensure the vllm/vllm_flash_attn directory exists before installation
  install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn\")"
    COMPONENT ${_FA_COMPONENT})

  # Copy vllm_flash_attn python files (except flash_attn_interface.py which is source-controlled in vllm)
  install(
    DIRECTORY ${vllm-flash-attn_SOURCE_DIR}/vllm_flash_attn/
    DESTINATION vllm/vllm_flash_attn
    COMPONENT ${_FA_COMPONENT}
    FILES_MATCHING PATTERN "*.py"
    PATTERN "flash_attn_interface.py" EXCLUDE
  )

endforeach()

#
# FA4 CuteDSL component
# This is a Python-only component that copies the flash_attn/cute directory
# and transforms imports to match our package structure.
#
add_custom_target(_vllm_fa4_cutedsl_C)

# Copy flash_attn/cute directory (needed for FA4) and transform imports
# The cute directory uses flash_attn.cute imports internally, which we replace
# with vllm.vllm_flash_attn.cute to match our package structure.
install(CODE "
  file(GLOB_RECURSE CUTE_PY_FILES \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute/*.py\")
  foreach(SRC_FILE \${CUTE_PY_FILES})
    file(RELATIVE_PATH REL_PATH \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute\" \${SRC_FILE})
    set(DST_FILE \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn/cute/\${REL_PATH}\")
    get_filename_component(DST_DIR \${DST_FILE} DIRECTORY)
    file(MAKE_DIRECTORY \${DST_DIR})
    file(READ \${SRC_FILE} FILE_CONTENTS)
    string(REPLACE \"flash_attn.cute\" \"vllm.vllm_flash_attn.cute\" FILE_CONTENTS \"\${FILE_CONTENTS}\")
    file(WRITE \${DST_FILE} \"\${FILE_CONTENTS}\")
  endforeach()
" COMPONENT _vllm_fa4_cutedsl_C)
