#include "bgmv_config.h"
#include "bgmv_impl.cuh"

FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, nv_half, nv_half)
FOR_INST_BGMV_WIDE_NARROW(INST_BGMV_ONESIDE, float, nv_half, nv_half)
