#pragma once

#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #if CUB_VERSION >= 200800
    #include <cuda/std/functional>
using AddOp = cuda::std::plus<>;
using MaxOp = cuda::maximum<>;
  #else   // if CUB_VERSION < 200800
using AddOp = cub::Sum;
using MaxOp = cub::Max;
  #endif  // CUB_VERSION
#else
  #include <hipcub/hipcub.hpp>
using AddOp = cub::Sum;
using MaxOp = cub::Max;
#endif  // USE_ROCM
