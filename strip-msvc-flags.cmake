# CMAKE_USER_MAKE_RULES_OVERRIDE
# Runs AFTER platform modules but BEFORE targets are defined
# Strips any cl-style flags that leaked through from Windows-MSVC.cmake

set(_MSVC_BAD_FLAGS
    /permissive- /Z7 /Zi /Zo /EHsc /EHa /EHs /GR /GR- /W3 /W4 /Wall
    /bigobj /nologo /errorReport:queue /diagnostics:column
    /MP /Od /O2 /Ob0 /Ob1 /Ob2 /Os /Ot /Ox /Oy /fp:precise /fp:fast
    /GF /Gm- /GS /GS- /Gy /Gy- /sdl /sdl- /guard:cf /RTC1 /RTCc
    /Gd /Gr /Gv /Gm /Gy /openmp /openmp:experimental
    /wd4267 /wd4251 /wd4275 /wd4018 /wd4190 /wd4624 /wd4068
    /wd4996 /wd4067 /wd4627 /wd4819 /wd4141 /wd4291 /wd4244
    /wd4305 /wd4800 /wd4065 /wd4355 /wd4506 /wd4505 /wd4146
    /wd4805 /wd4554 /wd4351 /wd4503 /wd4099 /wd4722 /wd4723
    /wd4172 /wd4005 /wd4611 /wd4577 /wd4701 /wd4702 /wd4242
    /wd5045 /wd5054 /wd4706 /wd4456 /wd4457 /wd4458 /wd4459
    /wd4310 /wd4100 /wd4201 /wd4324 /wd4127 /wd4312
)

set(_FLAG_VARS
    CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_HIP_FLAGS
    CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG CMAKE_HIP_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE CMAKE_HIP_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_HIP_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_HIP_FLAGS_MINSIZEREL
    CMAKE_STATIC_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS
    CMAKE_STATIC_LINKER_FLAGS_DEBUG CMAKE_SHARED_LINKER_FLAGS_DEBUG CMAKE_EXE_LINKER_FLAGS_DEBUG
    CMAKE_STATIC_LINKER_FLAGS_RELEASE CMAKE_SHARED_LINKER_FLAGS_RELEASE CMAKE_EXE_LINKER_FLAGS_RELEASE
)

foreach(_var ${_FLAG_VARS})
    if(DEFINED ${_var})
        foreach(_flag ${_MSVC_BAD_FLAGS})
            string(REPLACE "${_flag}" "" ${_var} "${${_var}}")
            string(REPLACE "${_flag} " "" ${_var} "${${_var}}")
        endforeach()
        string(REGEX REPLACE "  +" " " ${_var} "${${_var}}")
        string(STRIP "${${_var}}" ${_var})
        set(${_var} "${${_var}}" CACHE STRING "" FORCE)
    endif()
endforeach()
