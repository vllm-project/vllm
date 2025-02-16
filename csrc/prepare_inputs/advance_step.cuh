#pragma once

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

namespace prepare_inputs {

static constexpr int max_threads = 256;
static constexpr bool logging = false;

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

}  // namespace prepare_inputs
