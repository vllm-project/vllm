#include "common.cuh"
#include "dispatch_utils.h"
#include "../vectorization_utils.cuh"
#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

template <typename scalar_t, typename fp8_type>
__global__ void scaled_fp8_quant_kernel(fp8_type* __restrict__ out,
                                        const scalar_t* __restrict__ input,
                                        const float* __restrict__ scale,
                                        int64_t num_elems) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.
  const float inverted_scale = 1.0f / (*scale);
  vectorize_with_alignment<16>(
      input, out, num_elems, tid, blockDim.x * gridDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                    inverted_scale);
      });
}

template <typename scalar_t, typename fp8_type>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel(
    fp8_type* __restrict__ out, float* __restrict__ scale,
    scalar_t const* __restrict__ input, float const* __restrict__ scale_ub,
    const int hidden_size) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;

  // Use int64 to avoid overflowing an int32 when calculating this offset
  int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
  scalar_t const* __restrict__ token_input = &input[offset];
  fp8_type* __restrict__ token_output = &out[offset];

  // 1) compute per-token absmax
  float absmax_val = 0.0f;
  vectorize_read_with_alignment<16>(token_input, hidden_size, tid, blockDim.x,
                                    [&] __device__(const scalar_t& src) {
                                      const float v =
                                          fabsf(static_cast<float>(src));
                                      absmax_val = fmaxf(absmax_val, v);
                                    });

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float token_scale;
  if (tid == 0) {
    if (scale_ub) {
      token_scale = fminf(block_absmax_val_maybe, *scale_ub);
    } else {
      token_scale = block_absmax_val_maybe;
    }
    // token scale computation
    token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>,
                        min_scaling_factor<fp8_type>::val());
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // 2) quantize
  // Note that we don't use inverted scales so we can match FBGemm impl.
  vectorize_with_alignment<16>(
      token_input, token_output, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src),
                                                     token_scale);
      });
}

template <typename scalar_t, typename fp8_type>
__global__ void scaled_fp8_quant_kernel_strided(
    fp8_type* __restrict__ out, const scalar_t* __restrict__ input,
    const float* __restrict__ scale, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride) {
  const int token_idx = blockIdx.x;  // one token per block
  const int tid = threadIdx.x;

  const scalar_t* token_in = input + token_idx * in_row_stride;
  fp8_type* token_out = out + token_idx * out_row_stride;

  const float inv_scale = 1.0f / (*scale);

  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                    inv_scale);
      });
}

template <typename scalar_t, typename fp8_type>
__global__ void segmented_max_reduction_strided(
    float* __restrict__ scale, const scalar_t* __restrict__ input,
    int hidden_size, int64_t in_row_stride, int64_t num_tokens) {
  __shared__ float cache[256];
  int tid = threadIdx.x;

  // Each thread processes multiple rows in a round-robin fashion.
  float local_max = 0.0f;
  for (int64_t token = blockIdx.x * blockDim.x + tid; token < num_tokens;
       token += blockDim.x * gridDim.x) {
    const scalar_t* row_ptr = input + token * in_row_stride;
    // Traverse the row
#pragma unroll 4
    for (int e = 0; e < hidden_size; ++e) {
      float v = fabsf(static_cast<float>(row_ptr[e]));
      local_max = fmaxf(local_max, v);
    }
  }

  cache[tid] = local_max;
  __syncthreads();

  // Reduction inside block
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      cache[tid] = fmaxf(cache[tid], cache[tid + offset]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicMaxFloat(scale, cache[0] / quant_type_max_v<fp8_type>);
  }
}

template <typename scalar_t, typename fp8_type>
__global__ void scaled_fp8_quant_kernel_strided_dynamic(
    fp8_type* __restrict__ out, const scalar_t* __restrict__ input,
    const float* __restrict__ scale, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const scalar_t* token_in = input + token_idx * in_row_stride;
  fp8_type* token_out = out + token_idx * out_row_stride;

  const float reciprocal_scale = 1.0f / (*scale);
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                    reciprocal_scale);
      });
}

template <typename scalar_t, typename fp8_type>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel_strided(
    fp8_type* __restrict__ out, float* __restrict__ scale,
    const scalar_t* __restrict__ input, const float* __restrict__ scale_ub,
    int hidden_size, int64_t in_row_stride, int64_t out_row_stride) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const scalar_t* token_in = input + token_idx * in_row_stride;
  fp8_type* token_out = out + token_idx * out_row_stride;

  // 1) per-token absmax
  float absmax_val = 0.f;
  vectorize_read_with_alignment<16>(
      token_in, hidden_size, tid, blockDim.x, [&] __device__(scalar_t v) {
        absmax_val = fmaxf(absmax_val, fabsf(static_cast<float>(v)));
      });

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float block_max = BlockReduce(tmp).Reduce(absmax_val, cub::Max{}, blockDim.x);

  __shared__ float token_scale;
  if (tid == 0) {
    token_scale = scale_ub ? fminf(block_max, *scale_ub) : block_max;
    token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>,
                        min_scaling_factor<fp8_type>::val());
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // 2) quantize
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src),
                                                     token_scale);
      });
}

}  // namespace vllm

void static_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                             torch::Tensor const& input,  // [..., d]
                             torch::Tensor const& scale)  // [1]
{
  TORCH_CHECK(input.stride(-1) == 1,
              "last dimension of input must be contiguous");
  TORCH_CHECK(out.stride(-1) == 1,
              "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int block_size = 256;
  dim3 grid(num_tokens);
  dim3 block(block_size);

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);
  const bool is_contig_rows =
      (in_row_stride == hidden_size) && (out_row_stride == hidden_size);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              if (is_contig_rows) {
                const int num_elems = input.numel();
                vllm::scaled_fp8_quant_kernel<scalar_t, fp8_t>
                    <<<grid, block, 0, stream>>>(
                        out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                        scale.data_ptr<float>(), num_elems);
              } else {
                vllm::scaled_fp8_quant_kernel_strided<scalar_t, fp8_t>
                    <<<grid, block, 0, stream>>>(
                        out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                        scale.data_ptr<float>(), hidden_size, in_row_stride,
                        out_row_stride);
              }
            });
      });
}

void dynamic_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                              torch::Tensor const& input,  // [..., d]
                              torch::Tensor& scale)        // [1]
{
  TORCH_CHECK(input.stride(-1) == 1,
              "last dimension of input must be contiguous");
  TORCH_CHECK(out.stride(-1) == 1,
              "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int block_size = 256;
  dim3 grid(num_tokens);
  dim3 block(block_size);

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);
  const bool is_contig_rows =
      (in_row_stride == hidden_size) && (out_row_stride == hidden_size);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // scale tensor should be initialised to <=0 before reduction
  if (!is_contig_rows) {
    scale.fill_(0.0f);
  }

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              if (is_contig_rows) {
                const int num_elems = input.numel();
                vllm::segmented_max_reduction<scalar_t, fp8_t>
                    <<<grid, block, 0, stream>>>(scale.data_ptr<float>(),
                                                 input.data_ptr<scalar_t>(),
                                                 num_elems);
                vllm::scaled_fp8_quant_kernel<scalar_t, fp8_t>
                    <<<grid, block, 0, stream>>>(
                        out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                        scale.data_ptr<float>(), num_elems);
              } else {
                vllm::segmented_max_reduction_strided<scalar_t, fp8_t>
                    <<<grid, block, 0, stream>>>(
                        scale.data_ptr<float>(), input.data_ptr<scalar_t>(),
                        hidden_size, in_row_stride, (int64_t)num_tokens);

                vllm::scaled_fp8_quant_kernel_strided_dynamic<scalar_t, fp8_t>
                    <<<grid, block, 0, stream>>>(
                        out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                        scale.data_ptr<float>(), hidden_size, in_row_stride,
                        out_row_stride);
              }
            });
      });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor& scales, std::optional<at::Tensor> const& scale_ub) {
  TORCH_CHECK(input.stride(-1) == 1,
              "last dimension of input must be contiguous");
  TORCH_CHECK(out.stride(-1) == 1,
              "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int block_size = 256;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, block_size));

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);
  const bool is_contig_rows =
      (in_row_stride == hidden_size) && (out_row_stride == hidden_size);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "dynamic_per_token_scaled_fp8_quant_kernel_fp8_type", [&] {
              if (is_contig_rows) {
                vllm::dynamic_per_token_scaled_fp8_quant_kernel<scalar_t, fp8_t>
                    <<<grid, block, 0, stream>>>(
                        out.data_ptr<fp8_t>(), scales.data_ptr<float>(),
                        input.data_ptr<scalar_t>(),
                        scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                             : nullptr,
                        hidden_size);
              } else {
                vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<
                    scalar_t, fp8_t><<<grid, block, 0, stream>>>(
                    out.data_ptr<fp8_t>(), scales.data_ptr<float>(),
                    input.data_ptr<scalar_t>(),
                    scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                         : nullptr,
                    hidden_size, in_row_stride, out_row_stride);
              }
            });
      });
}
