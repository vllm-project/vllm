#pragma once

#include <torch/all.h>

//! Check if the tensor is contiguous except for the last dimension.
//!
//! \param tensor The tensor to check.
//! \return True if the tensor is contiguous except for the last dimension.
inline bool is_contiguous_except_last_dim(const torch::Tensor& tensor) {
  // If the tensor has 2 or fewer dimensions, it is contiguous.
  size_t ndim = tensor.ndimension();
  if (ndim <= 2) {
    return true;
  }

  // Check if the tensor is contiguous except for the last dimension.
  auto strides = tensor.strides();
  auto sizes = tensor.sizes();
  for (size_t i = 0; i < ndim - 2; ++i) {
    if (strides[i] != sizes[i + 1] * strides[i + 1]) {
      return false;
    }
  }
  return true;
}
