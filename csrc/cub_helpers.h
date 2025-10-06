#pragma once

#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #if CUB_VERSION >= 200800
    #include <cuda/std/functional>
using CubAddOp = cuda::std::plus<>;
using CubMaxOp = cuda::maximum<>;
  #else   // if CUB_VERSION < 200800
using CubAddOp = cub::Sum;
using CubMaxOp = cub::Max;
  #endif  // CUB_VERSION
#else
  #include <hipcub/hipcub.hpp>
using CubAddOp = cub::Sum;
using CubMaxOp = cub::Max;
#endif  // USE_ROCM
