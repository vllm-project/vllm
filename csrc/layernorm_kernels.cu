// layernorm_kernels.cu — host launchers for plain (non-quantised) RMS norm.
//
// Kernel templates live in layernorm_kernels.cuh (shared with
// layernorm_quant_kernels.cu).  This file instantiates the scalar variant
// (out_t == scalar_t) of each template.
//
// Changes vs. original:
//   • Removed the duplicated rms_norm_kernel template (now in .cuh).
//   • Replaced the two enable_if fused_add_rms_norm_kernel specialisations
//     (_f16Vec<scalar_t,8> / scalar fallback) with the unified vec_n_t-based
//     template from .cuh.  This also enables FP32 vectorisation (VEC_SIZE=4)
//     which was previously scalar-only.
#include "layernorm_kernels.cuh"
#include "dispatch_utils.h"
#include "core/batch_invariant.hpp"
#include "libtorch_stable/quantization/vectorization_utils.cuh"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = input.contiguous();
  }
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int num_dims = input.dim();
  int64_t input_stride_d2 = input.stride(-2);
  int64_t input_stride_d3 = (num_dims >= 3) ? input.stride(-3) : 0;
  int64_t input_stride_d4 = (num_dims >= 4) ? input.stride(-4) : 0;
  int64_t input_shape_d2 = (num_dims >= 3) ? input.size(-2) : 0;
  int64_t input_shape_d3 = (num_dims >= 4) ? input.size(-3) : 0;

  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_RANK234(num_dims, [&] {
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
      const int calculated_vec_size =
          std::gcd((int)(16 / sizeof(scalar_t)), hidden_size);
      const int block_size =
          std::min(hidden_size / calculated_vec_size, max_block_size);
      dim3 block(block_size);
      VLLM_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
        // Scalar variant: out_t == scalar_t, scale == nullptr
        vllm::rms_norm_kernel<scalar_t, scalar_t, vec_size, tensor_rank>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                input_stride_d2, input_stride_d3, input_stride_d4,
                input_shape_d2, input_shape_d3, weight.data_ptr<scalar_t>(),
                /*scale=*/nullptr, (float)epsilon, num_tokens, hidden_size);
      });
    });
  });
}

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  TORCH_CHECK(input.scalar_type() == residual.scalar_type());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool batch_invariant_launch = vllm::vllm_is_batch_invariant();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {
        // Max vector width: 16 bytes / element_size
        //   fp16/bf16 → 8  (128-bit load, same as the previous _f16Vec<,8>)
        //   fp32      → 4  (128-bit load, NEW: previously fell back to scalar)
        const int max_vec = (int)(16 / sizeof(scalar_t));
        const int req_align = max_vec * (int)sizeof(scalar_t);  // always 16 B
        bool ptrs_aligned = (inp_ptr % req_align == 0) &&
                            (res_ptr % req_align == 0) &&
                            (wt_ptr % req_align == 0);
        bool dims_aligned =
            (hidden_size % max_vec == 0) && (input_stride % max_vec == 0);
        // Use max_vec when all alignment conditions are met; otherwise scalar.
        const int calculated_vec_size =
            (ptrs_aligned && dims_aligned && !batch_invariant_launch) ? max_vec
                                                                      : 1;

        const int block_size =
            std::min(hidden_size / calculated_vec_size, max_block_size);
        dim3 block(block_size);

        VLLM_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
          // Scalar variant: out_t == scalar_t.
          // out_quant = nullptr (output written back to input by the kernel).
          // scale     = nullptr (unused for scalar output).
          vllm::fused_add_rms_norm_kernel<scalar_t, scalar_t, vec_size>
              <<<grid, block, 0, stream>>>(
                  /*out_quant=*/static_cast<scalar_t*>(nullptr),
                  input.data_ptr<scalar_t>(), input_stride,
                  residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                  /*scale=*/nullptr, (float)epsilon, num_tokens, hidden_size);
        });
      });
}
