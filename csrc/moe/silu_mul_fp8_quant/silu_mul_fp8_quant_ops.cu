#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cuda_compat.h"
#include "flashinfer_fp8_activation.cuh"
#include "nvfp4_silu_mul_quant_launcher.h"
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
      output_scales.data_ptr<float>(), static_cast<int32_t>(N), H, scale_stride,
      n_compute, batch_size, use_tanh_silu, N, stream);
}

void nvfp4_silu_mul_quant(at::Tensor& output, at::Tensor& output_scale,
                          const at::Tensor& input,
                          const at::Tensor& input_global_scale,
                          const at::Tensor& mask, int64_t n_experts) {
  TORCH_CHECK(input.dtype() == torch::kBFloat16);
  TORCH_CHECK(input_global_scale.dtype() == torch::kFloat32);
  TORCH_CHECK(mask.dtype() == torch::kInt32);

  int32_t m_topk = static_cast<int32_t>(input.size(0));
  int32_t k = static_cast<int32_t>(input.size(1) / 2);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  nvfp4::launch_silu_mul_nvfp4_quant(
      output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
      input_global_scale.data_ptr<float>(), mask.data_ptr<int32_t>(), m_topk, k,
      static_cast<int32_t>(n_experts), stream);
}

void silu_mul_nvfp4_quant_tma_ws_persistent_bf16(
    const at::Tensor& input, at::Tensor& output, at::Tensor& output_sf,
    const at::Tensor& global_scale, const at::Tensor& n_tokens,
    int64_t n_compute, int64_t batch_size, bool use_tanh_silu) {
  TORCH_CHECK(input.dtype() == torch::kBFloat16);
  TORCH_CHECK(output.dtype() == torch::kUInt8);
  TORCH_CHECK(output_sf.dtype() == torch::kUInt8);
  TORCH_CHECK(global_scale.dtype() == torch::kFloat32);
  TORCH_CHECK(n_tokens.dtype() == torch::kInt32);

  int64_t H = input.size(1) / 2;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t N = input.size(0);
  vllm::launch_silu_mul_nvfp4_quant_tma_ws_persistent_bf16(
      input.data_ptr(), output.data_ptr(), output_sf.data_ptr(),
      global_scale.data_ptr<float>(), static_cast<int32_t>(N), H, N, n_compute,
      batch_size, use_tanh_silu, stream);
}
