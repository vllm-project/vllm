// layernorm_quant_kernels.cu — host launchers for fp8-quantised RMS norm.
//
// The actual CUDA kernel templates live in layernorm_kernels.cuh (shared with
// layernorm_kernels.cu).  This file instantiates the fp8-quantised variant
// (out_t == fp8_t) of each template.
//
// Previously this file contained full copies of the rms_norm and
// fused_add_rms_norm kernel bodies, differing from layernorm_kernels.cu only
// in the output write (scalar vs. scaled fp8 conversion).  The duplication
// (~180 lines) is eliminated by the shared header.
//
// New in this refactor:
//   · The fp8 output write is now vectorised via q8_n_t<fp8_t, VEC_SIZE>
//     (previously element-by-element scalar writes).
//   · fused_add_rms_norm_static_fp8_quant now also vectorises FP32 inputs
//     (VEC_SIZE=4) thanks to the vec_n_t migration.
#include "layernorm_kernels.cuh"
#include "dispatch_utils.h"
#include "core/batch_invariant.hpp"
#include "libtorch_stable/quantization/vectorization_utils.cuh"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

void rms_norm_static_fp8_quant(torch::Tensor& out,     // [..., hidden_size]
                               torch::Tensor& input,   // [..., hidden_size]
                               torch::Tensor& weight,  // [hidden_size]
                               torch::Tensor& scale,   // [1]
                               double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = input.contiguous();
  }
  TORCH_CHECK(input.stride(-1) == 1);
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_static_fp8_quant_kernel_scalar", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "rms_norm_static_fp8_quant_kernel_fp8", [&] {
              // VEC_SIZE based on input scalar type (not fp8 output type).
              const int calculated_vec_size =
                  std::gcd((int)(16 / sizeof(scalar_t)), hidden_size);
              const int block_size =
                  std::min(hidden_size / calculated_vec_size, max_block_size);
              dim3 block(block_size);
              VLLM_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
                // fp8 variant with NUM_DIMS=2 (quant norm is always 2-D).
                // Unused stride/shape params are zero for the 2-D
                // specialisation.
                vllm::rms_norm_kernel<scalar_t, fp8_t, vec_size, 2>
                    <<<grid, block, 0, stream>>>(
                        out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                        /*input_stride_d2=*/input_stride,
                        /*input_stride_d3=*/0, /*input_stride_d4=*/0,
                        /*input_shape_d2=*/0, /*input_shape_d3=*/0,
                        weight.data_ptr<scalar_t>(), scale.data_ptr<float>(),
                        (float)epsilon, num_tokens, hidden_size);
              });
            });
      });
}

void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,       // [..., hidden_size]  — fp8 output
    torch::Tensor& input,     // [..., hidden_size]  — strided input
    torch::Tensor& residual,  // [..., hidden_size]  — updated in-place
    torch::Tensor& weight,    // [hidden_size]
    torch::Tensor& scale,     // [1]
    double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(residual.scalar_type() == input.scalar_type());
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());

  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool batch_invariant_launch = vllm::vllm_is_batch_invariant();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "fused_add_rms_norm_static_fp8_quant_scalar", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "fused_add_rms_norm_static_fp8_quant_fp8", [&] {
              const int max_vec = (int)(16 / sizeof(scalar_t));
              const int req_align = max_vec * (int)sizeof(scalar_t);
              bool ptrs_aligned = (inp_ptr % req_align == 0) &&
                                  (res_ptr % req_align == 0) &&
                                  (wt_ptr % req_align == 0);
              bool dims_aligned =
                  (hidden_size % max_vec == 0) && (input_stride % max_vec == 0);
              const int calculated_vec_size =
                  (ptrs_aligned && dims_aligned && !batch_invariant_launch)
                      ? max_vec
                      : 1;

              const int block_size =
                  std::min(hidden_size / calculated_vec_size, max_block_size);
              dim3 block(block_size);

              VLLM_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
                vllm::fused_add_rms_norm_kernel<scalar_t, fp8_t, vec_size>
                    <<<grid, block, 0, stream>>>(
                        out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                        input_stride, residual.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(), scale.data_ptr<float>(),
                        (float)epsilon, num_tokens, hidden_size);
              });
            });
      });
}
