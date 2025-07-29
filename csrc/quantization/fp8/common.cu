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

}  // namespace vllm

void static_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                             torch::Tensor const& input,  // [..., d]
                             torch::Tensor const& scale)  // [1]
{
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  int const block_size = 256;
  int const num_tokens = input.numel() / input.size(-1);
  int const num_elems = input.numel();
  dim3 const grid(num_tokens);
  dim3 const block(block_size);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::scaled_fp8_quant_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                      scale.data_ptr<float>(), num_elems);
            });
      });
}

void dynamic_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                              torch::Tensor const& input,  // [..., d]
                              torch::Tensor& scale)        // [1]
{
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  int const block_size = 256;
  int const num_tokens = input.numel() / input.size(-1);
  int const num_elems = input.numel();
  dim3 const grid(num_tokens);
  dim3 const block(block_size);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::segmented_max_reduction<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(scale.data_ptr<float>(),
                                               input.data_ptr<scalar_t>(),
                                               num_elems);
              vllm::scaled_fp8_quant_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                      scale.data_ptr<float>(), num_elems);
            });
      });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor& scales, std::optional<at::Tensor> const& scale_ub) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  int const block_size = 256;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, block_size));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "dynamic_per_token_scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::dynamic_per_token_scaled_fp8_quant_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), scales.data_ptr<float>(),
                      input.data_ptr<scalar_t>(),
                      scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                           : nullptr,
                      hidden_size);
            });
      });
}
