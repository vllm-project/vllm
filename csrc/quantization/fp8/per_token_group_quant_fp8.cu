#include "common.cuh"
#include "dispatch_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

template <typename scalar_t, bool is_column_major>
__global__ void per_token_group_quant_fp8_kernel(
    const scalar_t* __restrict__ input, FP8_TYPE* __restrict__ output_q,
    float* __restrict__ output_s, const int group_size,
    const int groups_per_row, const int y_s_stride) {
  float const min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);

  // Each block processes one group
  const int tid = threadIdx.x;

  // Calculate row and group within row
  const int row = blockIdx.x;
  const int row_group_id = blockIdx.y;

  // Calculate group_idx
  const int64_t group_idx = row * groups_per_row + row_group_id;

  // Calculate input and output pointers
  const int64_t input_offset =
      row * (groups_per_row * group_size) + row_group_id * group_size;
  scalar_t const* __restrict__ group_input = input + input_offset;
  FP8_TYPE* __restrict__ group_output = output_q + input_offset;
  // FP8_TYPE* __restrict__ group_output = output_q + group_idx * group_size;

  // Calculate scale output pointer based on layout
  float* scale_output;
  if (is_column_major) {
    // Column-major: scales[col][row]
    scale_output = output_s + row_group_id * y_s_stride + row;
  } else {
    // Row-major: scales[row][col]
    scale_output = output_s + group_idx;
  }

  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = group_size % 4 == 0;

  // Find maximum absolute value in this group
  float absmax_val = 0.0f;
  if (can_vectorize) {
    absmax_val = vllm::thread_max_vec(group_input, group_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < group_size; i += blockDim.x) {
      float const x = static_cast<float>(group_input[i]);
      absmax_val = max(absmax_val, fabs(x));
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float group_scale;
  if (tid == 0) {
    group_scale = max(block_absmax_val / FP8_E4M3_MAX, min_scaling_factor);
    *scale_output = group_scale;
  }
  __syncthreads();

  // Quantize the data
  if (can_vectorize) {
    vllm::scaled_fp8_conversion_vec<scalar_t, false>(
        group_output, group_input, group_scale, group_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < group_size; i += blockDim.x) {
      group_output[i] = vllm::scaled_fp8_conversion<false>(
          static_cast<float>(group_input[i]), group_scale);
    }
  }
}

void per_token_group_quant_fp8(torch::Tensor const& input,
                               torch::Tensor& output_q, torch::Tensor& output_s,
                               int64_t group_size, bool column_major_scales) {
  TORCH_CHECK(input.dim() == 2, "Input tensor must have 2 dimensions");
  TORCH_CHECK(input.stride(-1) == 1, "Last dimension must be contiguous");
  TORCH_CHECK(input.size(-1) % group_size == 0,
              "Last dimension must be divisible by group_size");

  // Dimensions for kernel launch
  const int num_rows = input.size(0);
  const int groups_per_row = input.size(-1) / group_size;
  const int y_s_stride = column_major_scales ? output_s.stride(-1) : 1;

  // Launch parameters
  const int block_size = std::min(1024, static_cast<int>(group_size));
  dim3 grid(num_rows, groups_per_row);
  dim3 block(block_size);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "per_token_group_quant_fp8_kernel", [&] {
        if (column_major_scales) {
          per_token_group_quant_fp8_kernel<scalar_t, true>
              <<<grid, block, 0, stream>>>(
                  static_cast<scalar_t*>(input.data_ptr()),
                  static_cast<FP8_TYPE*>(output_q.data_ptr()),
                  static_cast<float*>(output_s.data_ptr()), group_size,
                  groups_per_row, y_s_stride);
        } else {
          per_token_group_quant_fp8_kernel<scalar_t, false>
              <<<grid, block, 0, stream>>>(
                  static_cast<scalar_t*>(input.data_ptr()),
                  static_cast<FP8_TYPE*>(output_q.data_ptr()),
                  static_cast<float*>(output_s.data_ptr()), group_size,
                  groups_per_row, y_s_stride);
        }
        return true;
      });
}