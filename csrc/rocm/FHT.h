/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct HadamardParamsBase {
  using index_t = int64_t;

  int batch, dim, log_N;

  index_t x_batch_stride;
  index_t out_batch_stride;

  float scale;

  // Common data pointers.
  void* __restrict__ x_ptr;
  void* __restrict__ out_ptr;

  // Print method
  void print() const {
    std::cout << "batch: " << batch << std::endl;
    std::cout << "dim: " << dim << std::endl;
    std::cout << "log_N: " << log_N << std::endl;
    std::cout << "x_batch_stride: " << x_batch_stride << std::endl;
    std::cout << "out_batch_stride: " << out_batch_stride << std::endl;
    std::cout << "scale: " << scale << std::endl;
    std::cout << "x_ptr: " << x_ptr << std::endl;
    std::cout << "out_ptr: " << out_ptr << std::endl;
  }
};

at::Tensor fast_hadamard_transform(at::Tensor& x, double scale);