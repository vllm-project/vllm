# CUDA architecture selection helpers, also usable with cmake -P.
include_guard(GLOBAL)

# Parse <major>.<minor>[a|f][+PTX].
macro(_vllm_cuda_arch_parse ARCH VER SFX PTX)
  if(NOT "${ARCH}" MATCHES "^([0-9]+)\\.([0-9]+)([af]?)(\\+PTX)?$")
    message(FATAL_ERROR "Invalid CUDA arch: '${ARCH}'")
  endif()
  set(${VER} "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
  set(${SFX} "${CMAKE_MATCH_3}")
  if(CMAKE_MATCH_4)
    set(${PTX} TRUE)
  else()
    set(${PTX} FALSE)
  endif()
endmacro()

# Approximate CUDA families by major and minimum minor, except sm_101
# (renamed sm_110 in CUDA 13), which belongs to a separate family.
function(_vllm_cuda_family_covers OUT BASE TGT)
  string(REGEX MATCH "^[0-9]+" _bm "${BASE}")
  string(REGEX MATCH "^[0-9]+" _tm "${TGT}")
  set(_ok FALSE)
  if(_bm STREQUAL _tm AND TGT VERSION_GREATER_EQUAL BASE)
    set(_ok TRUE)
    if(NOT BASE VERSION_EQUAL TGT AND
       (BASE VERSION_EQUAL 10.1 OR TGT VERSION_EQUAL 10.1))
      set(_ok FALSE)
    endif()
  endif()
  set(${OUT} ${_ok} PARENT_SCOPE)
endfunction()

# For each TGT, select the highest runnable SRC version, breaking ties by
# feature requirements (a > f > plain), then by preserving requested PTX.
# An 'a' SRC requires the exact version; 'f' requires the same CUDA family.
# Both emit the TGT version with a suffix at least as strong as SRC's.
# Plain SRC runs on later minors of the same major (across majors with +PTX);
# emit SRC's version, taking TGT's suffix only on an exact version match.
# For a plain TGT, also retain a same-version plain SRC alongside a selected
# suffixed SRC, preserving the plain code's forward compatibility.
# Keep SRC's +PTX, deduplicate outputs and preserve TGT order.
# Example: SRC="7.5;8.0;8.6;9.0;9.0a", TGT="8.0;8.9;9.0"
# yields "8.0;8.6;9.0;9.0a". SRC="10.0f", TGT="10.7" yields "10.7f".
function(cuda_archs_loose_intersection OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
  set(_out)
  foreach(_tgt IN LISTS TGT_CUDA_ARCHS)
    if(_tgt STREQUAL "")
      continue()
    endif()
    _vllm_cuda_arch_parse("${_tgt}" _tv _ts _tp)
    string(REGEX MATCH "^[0-9]+" _tmaj "${_tv}")
    set(_best "")
    set(_best_v "0.0")
    set(_best_r -1)
    set(_best_ptx FALSE)
    set(_plain "")
    set(_plain_ptx FALSE)

    foreach(_src IN LISTS SRC_CUDA_ARCHS)
      if(_src STREQUAL "")
        continue()
      endif()
      _vllm_cuda_arch_parse("${_src}" _sv _ss _sp)
      string(REGEX MATCH "^[0-9]+" _smaj "${_sv}")
      if(_sv VERSION_GREATER _tv)
        continue()
      endif()

      if(_ss STREQUAL "a")
        if(NOT _sv VERSION_EQUAL _tv)
          continue()
        endif()
        set(_cand "${_tv}a")
        set(_r 2)
      elseif(_ss STREQUAL "f")
        _vllm_cuda_family_covers(_ok "${_sv}" "${_tv}")
        if(NOT _ok)
          continue()
        endif()
        if(_ts STREQUAL "a")
          set(_cand "${_tv}a")
        else()
          set(_cand "${_tv}f")
        endif()
        set(_r 1)
      else()
        if(NOT _smaj STREQUAL _tmaj AND NOT _sp)
          continue()
        endif()
        if(_sv VERSION_EQUAL _tv)
          set(_cand "${_tv}${_ts}")
        else()
          set(_cand "${_sv}")
        endif()
        set(_r 0)
      endif()

      if(_sp)
        string(APPEND _cand "+PTX")
      endif()
      if(_r EQUAL 0 AND _ts STREQUAL "" AND _sv VERSION_EQUAL _tv)
        if(_plain STREQUAL "" OR (_sp AND NOT _plain_ptx))
          set(_plain "${_cand}")
          set(_plain_ptx ${_sp})
        endif()
      endif()
      if(_sv VERSION_GREATER _best_v OR
         (_sv VERSION_EQUAL _best_v AND _r GREATER _best_r) OR
         (_sv VERSION_EQUAL _best_v AND _r EQUAL _best_r AND
          _sp AND NOT _best_ptx))
        set(_best "${_cand}")
        set(_best_v "${_sv}")
        set(_best_r ${_r})
        set(_best_ptx ${_sp})
      endif()
    endforeach()

    if(_best_r GREATER 0 AND NOT _plain STREQUAL "")
      list(APPEND _out "${_plain}")
    endif()
    if(NOT _best STREQUAL "")
      list(APPEND _out "${_best}")
    endif()
  endforeach()

  if(_out)
    list(REMOVE_DUPLICATES _out)
  endif()
  set(${OUT_CUDA_ARCHS} ${_out} PARENT_SCOPE)
endfunction()
