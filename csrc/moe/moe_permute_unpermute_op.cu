#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "permute_unpermute_kernels/moe_permute_unpermute_kernel.h"
#include "permute_unpermute_kernels/dispatch.h"
#include "core/registration.h"

template <typename T>
__global__ void shuffleInputRowsKernel(const T* input,
                                       const int32_t* dst2src_map, T* output,
                                       int64_t num_src_rows,
                                       int64_t num_dst_rows, int64_t num_cols) {
  int64_t dest_row_idx = blockIdx.x;
  int64_t const source_row_idx = dst2src_map[dest_row_idx];

  if (blockIdx.x < num_dst_rows) {
    // Load 128-bits per thread
    constexpr int64_t ELEM_PER_THREAD = 128 / sizeof(T) / 8;
    using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

    // Duplicate and permute rows
    auto const* source_row_ptr =
        reinterpret_cast<DataElem const*>(input + source_row_idx * num_cols);
    auto* dest_row_ptr =
        reinterpret_cast<DataElem*>(output + dest_row_idx * num_cols);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = blockDim.x;
    int64_t const num_elems_in_col = num_cols / ELEM_PER_THREAD;

    for (int elem_index = start_offset; elem_index < num_elems_in_col;
         elem_index += stride) {
      dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
  }
}

template <typename T>
__global__ void shuffleInputRowsKernelSlow(const T* input,
                                           const int32_t* dst2src_map,
                                           T* output, int64_t num_src_rows,
                                           int64_t num_dst_rows,
                                           int64_t num_cols) {
  int64_t dest_row_idx = blockIdx.x;
  int64_t const source_row_idx = dst2src_map[dest_row_idx];

  if (blockIdx.x < num_dst_rows) {
    // Duplicate and permute rows
    auto const* source_row_ptr = input + source_row_idx * num_cols;
    auto* dest_row_ptr = output + dest_row_idx * num_cols;

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = blockDim.x;

    for (int elem_index = start_offset; elem_index < num_cols;
         elem_index += stride) {
      dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
  }
}

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor) {
  TORCH_CHECK(input_tensor.scalar_type() == output_tensor.scalar_type(),
              "Input and output tensors must have the same data type");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t const blocks = output_tensor.size(0);
  int64_t const threads = 256;
  int64_t const num_dest_rows = output_tensor.size(0);
  int64_t const num_src_rows = input_tensor.size(0);
  int64_t const num_cols = input_tensor.size(1);

  if (num_cols % (128 / sizeof(input_tensor.scalar_type()) / 8)) {
    // use slow kernel if num_cols can't be aligned to 128 bits
    MOE_DISPATCH(input_tensor.scalar_type(), [&] {
      shuffleInputRowsKernelSlow<scalar_t><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<scalar_t*>(input_tensor.data_ptr()),
          dst2src_map.data_ptr<int32_t>(),
          reinterpret_cast<scalar_t*>(output_tensor.data_ptr()), num_src_rows,
          num_dest_rows, num_cols);
    });
  } else {
    MOE_DISPATCH(input_tensor.scalar_type(), [&] {
      shuffleInputRowsKernel<scalar_t><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<scalar_t*>(input_tensor.data_ptr()),
          dst2src_map.data_ptr<int32_t>(),
          reinterpret_cast<scalar_t*>(output_tensor.data_ptr()), num_src_rows,
          num_dest_rows, num_cols);
    });
  }
}
