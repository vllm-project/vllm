
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../dispatch_utils.h"
#include "layernorm_utils.cuh"
#include "quant_conversions.cuh"

namespace vllm {

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void rms_norm_dynamic_per_token_quant_vec(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr) {
  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute rms
  vllm::vectorized::compute_rms<scalar_t, has_residual>(
      &rms, input, hidden_size, var_epsilon, residual);

  // Compute scale
  vllm::vectorized::compute_dynamic_per_token_scales<scalar_t, scalar_out_t,
                                                     has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub, hidden_size,
      residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, true,
                                     has_residual>(
        out, input, weight, rms, 1.0f / token_scale, hidden_size, residual);
  } else {
    // FP8 - Do not invert token_scale for exact match with FBGemm
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, false,
                                     has_residual>(
        out, input, weight, rms, token_scale, hidden_size, residual);
  }
}

// RMS norm + quant kernel
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__global__ void rms_norm_dynamic_per_token_quant_kernel(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr) {
  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0;

  if (can_vectorize) {
    return rms_norm_dynamic_per_token_quant_vec<scalar_t, scalar_out_t,
                                                has_residual>(
        out, scales, input, weight, scale_ub, var_epsilon, hidden_size,
        residual);
  }

  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute RMS
  vllm::compute_rms<scalar_t, has_residual>(&rms, input, hidden_size,
                                            var_epsilon, residual);
  // Compute Scale
  vllm::compute_dynamic_per_token_scales<scalar_t, scalar_out_t, has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub, hidden_size,
      residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    token_scale = 1.0f / token_scale;
    vllm::norm_and_quant<scalar_t, scalar_out_t, true, has_residual>(
        out, input, weight, rms, &token_scale, hidden_size, residual);
  } else {
    // FP8 - Do not invert s_token_scale for exact match with FBGemm
    vllm::norm_and_quant<scalar_t, scalar_out_t, false, has_residual>(
        out, input, weight, rms, &token_scale, hidden_size, residual);
  }
}

// RMS norm + quant kernel
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__global__ void rms_norm_per_block_quant_kernel_1(
    float* __restrict__ rms,
    scalar_out_t* __restrict__ out,  // [..., hidden_size]
    float* __restrict__ scales,      // [num_tokens, hidden_size / group_size]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr, int32_t const group_size = 0) {
  // Compute RMS
  vllm::compute_rms<scalar_t, has_residual>(rms + blockIdx.x, input,
                                            hidden_size, var_epsilon, residual);
}

// RMS norm + quant kernel
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__global__ void rms_norm_per_block_quant_kernel_2(
    float* rms, float* token_scale,
    scalar_out_t* __restrict__ out,  // [..., hidden_size]
    float* __restrict__ scales,      // [num_tokens, hidden_size / group_size]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr, int32_t const group_size = 0) {
  // Compute Scale
  vllm::compute_dynamic_per_token_scales<scalar_t, scalar_out_t, has_residual>(
      token_scale + blockIdx.x, scales, input, weight,
      rms[blockIdx.x / (hidden_size / group_size)], scale_ub, hidden_size,
      residual, group_size);
}

// RMS norm + quant kernel
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__global__ void rms_norm_per_block_quant_kernel_3(
    float* rms, float* token_scale,
    scalar_out_t* __restrict__ out,  // [..., hidden_size]
    float* __restrict__ scales,      // [num_tokens, hidden_size / group_size]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr, int32_t const group_size = 0) {
  // RMS Norm + Quant
  int token_idx = blockIdx.x * hidden_size / group_size;
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      auto token_group_idx = token_idx + i / group_size;
      token_scale[token_group_idx] = 1.0f / token_scale[token_group_idx];
    }
    vllm::norm_and_quant<scalar_t, scalar_out_t, true, has_residual>(
        out, input, weight, rms[blockIdx.x], token_scale + token_idx,
        hidden_size, residual, group_size);
  } else {
    // FP8 - Do not invert s_token_scale for exact match with FBGemm
    vllm::norm_and_quant<scalar_t, scalar_out_t, false, has_residual>(
        out, input, weight, rms[blockIdx.x], token_scale + token_idx,
        hidden_size, residual, group_size);
  }
}

}  // namespace vllm

// Residual add + RMS norm + dynamic per token
template <typename scalar_in_t>
void rms_norm_dynamic_per_token_quant_dispatch(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    std::optional<at::Tensor> const& scale_ub,
    std::optional<at::Tensor>& residual) {
  int32_t hidden_size = input.size(-1);
  auto num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (residual.has_value()) {
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,
                                                        true>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, residual->data_ptr<scalar_in_t>());
        });

  } else {
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,
                                                        false>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, nullptr);
        });
  }
}

void rms_norm_dynamic_per_token_quant(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    std::optional<at::Tensor> scale_ub, std::optional<at::Tensor> residual) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;
  TORCH_CHECK(out.dtype() == kFp8Type || out.dtype() == torch::kInt8);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());

  if (scale_ub.has_value()) {
    TORCH_CHECK(out.dtype() == kFp8Type);
  }
  TORCH_CHECK(weight.dtype() == input.dtype());
  TORCH_CHECK(scales.dtype() == torch::kFloat32);
  if (residual) {
    TORCH_CHECK(residual->scalar_type() == input.scalar_type());
  }

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_dynamic_per_token_quant_dispatch", [&] {
        rms_norm_dynamic_per_token_quant_dispatch<scalar_t>(
            out, input, weight, scales, var_epsilon, scale_ub, residual);
      });
}

// Residual add + RMS norm + dynamic per token
// TODO think up better names than kernel_1, kernel_2, kernel_3, cleanup args
// TODO vectorized kernels
template <typename scalar_in_t>
void rms_norm_per_block_quant_dispatch(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor& scales,        // [num_tokens, hidden_size / group_size]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    std::optional<at::Tensor> const& scale_ub,
    std::optional<at::Tensor>& residual, int64_t group_size) {
  int32_t hidden_size = input.size(-1);
  auto num_tokens = input.numel() / hidden_size;

  dim3 grid13(num_tokens);
  dim3 block13(std::min(hidden_size, 1024));
  dim3 grid2(num_tokens * hidden_size / group_size);
  dim3 block2(std::min(group_size, 1024l));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto const fp_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
  torch::Tensor rms = torch::zeros({num_tokens}, fp_options);
  torch::Tensor token_scale =
      torch::zeros({num_tokens * hidden_size / group_size}, fp_options);

  if (residual.has_value()) {
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_per_block_quant_kernel_1", [&] {
          vllm::rms_norm_per_block_quant_kernel_1<scalar_in_t, scalar_t, true>
              <<<grid13, block13, 0, stream>>>(
                  rms.data_ptr<float>(), out.data_ptr<scalar_t>(),
                  scales.data_ptr<float>(), input.data_ptr<scalar_in_t>(),
                  weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, residual->data_ptr<scalar_in_t>(),
                  group_size);
        });
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_per_block_quant_kernel_2", [&] {
          vllm::rms_norm_per_block_quant_kernel_2<scalar_in_t, scalar_t, true>
              <<<grid2, block2, 0, stream>>>(
                  rms.data_ptr<float>(), token_scale.data_ptr<float>(),
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, residual->data_ptr<scalar_in_t>(),
                  group_size);
        });
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_per_block_quant_kernel_3", [&] {
          vllm::rms_norm_per_block_quant_kernel_3<scalar_in_t, scalar_t, true>
              <<<grid13, block13, 0, stream>>>(
                  rms.data_ptr<float>(), token_scale.data_ptr<float>(),
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, residual->data_ptr<scalar_in_t>(),
                  group_size);
        });
  } else {
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_per_block_quant_kernel_1", [&] {
          vllm::rms_norm_per_block_quant_kernel_1<scalar_in_t, scalar_t, false>
              <<<grid13, block13, 0, stream>>>(
                  rms.data_ptr<float>(), out.data_ptr<scalar_t>(),
                  scales.data_ptr<float>(), input.data_ptr<scalar_in_t>(),
                  weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, nullptr, group_size);
        });
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_per_block_quant_kernel_2", [&] {
          vllm::rms_norm_per_block_quant_kernel_2<scalar_in_t, scalar_t, false>
              <<<grid2, block2, 0, stream>>>(
                  rms.data_ptr<float>(), token_scale.data_ptr<float>(),
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, nullptr, group_size);
        });
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_per_block_quant_kernel_3", [&] {
          vllm::rms_norm_per_block_quant_kernel_3<scalar_in_t, scalar_t, false>
              <<<grid13, block13, 0, stream>>>(
                  rms.data_ptr<float>(), token_scale.data_ptr<float>(),
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, nullptr, group_size);
        });
  }
}

void rms_norm_per_block_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& weight,
                              torch::Tensor& scales, double const var_epsilon,
                              std::optional<torch::Tensor> scale_ub,
                              std::optional<torch::Tensor> residual,
                              int64_t group_size) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;
  TORCH_CHECK(out.dtype() == kFp8Type || out.dtype() == torch::kInt8);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());

  if (scale_ub.has_value()) {
    TORCH_CHECK(out.dtype() == kFp8Type);
  }
  TORCH_CHECK(weight.dtype() == input.dtype());
  TORCH_CHECK(scales.dtype() == torch::kFloat32);
  if (residual) {
    TORCH_CHECK(residual->scalar_type() == input.scalar_type());
  }

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_per_block_quant_dispatch", [&] {
        rms_norm_per_block_quant_dispatch<scalar_t>(out, input, weight, scales,
                                                    var_epsilon, scale_ub,
                                                    residual, group_size);
      });
}