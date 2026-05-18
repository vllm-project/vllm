#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cuda_compat.h"
#include "flashinfer_fp8_activation.cuh"
#include "silu_mul_fp8_quant_launcher.h"

void silu_mul_fp8_quant_baseline(const at::Tensor& input,
                                 const at::Tensor& input_scales,
                                 at::Tensor& output, at::Tensor& output_scales,
                                 const at::Tensor& n_tokens,
                                 bool use_tanh_silu) {
  TORCH_CHECK(input.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(input_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(output.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(output_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(n_tokens.dtype() == torch::kInt32);

  const int64_t inner_dim = input.size(1);
  const int32_t max_padded = static_cast<int32_t>(input.size(0));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  flashinfer::launch_fp8_silu_mul_baseline(
      input.data_ptr(), input_scales.data_ptr<float>(), output.data_ptr(),
      output_scales.data_ptr<float>(), n_tokens.data_ptr<int32_t>(),
      static_cast<int32_t>(inner_dim), max_padded, use_tanh_silu, stream);
}

void silu_mul_fp8_quant_tma_ws_persistent(
    const at::Tensor& input, const at::Tensor& input_scales, at::Tensor& output,
    at::Tensor& output_scales, const at::Tensor& n_tokens, int64_t n_compute,
    int64_t batch_size, bool use_tanh_silu) {
  TORCH_CHECK(input.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(input_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(output.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(output_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(n_tokens.dtype() == torch::kInt32);

  const auto input_sizes = input.sizes();
  const int64_t N = input_sizes[0];
  const int64_t H = input_sizes[1] / 2;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t scale_stride = input.size(0);

  vllm::launch_silu_mul_fp8_quant_tma_ws_persistent(
      input.data_ptr(), input_scales.data_ptr<float>(), output.data_ptr(),
      output_scales.data_ptr<float>(), n_tokens.item<int32_t>(), H,
      scale_stride, n_compute, batch_size, use_tanh_silu, input.size(0),
      stream);
}
