#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include "../../dispatch_utils.h"
#include "../../torch_utils.h"
#include "../../../quantization/fused_kernels/layernorm_utils.cuh"
#include "../../../quantization/fused_kernels/quant_conversions.cuh"

#include <optional>

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
    token_scale = 1.0f / token_scale;
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, true,
                                     has_residual>(
        out, input, weight, rms, &token_scale, hidden_size, residual);
  } else {
    // FP8 - Do not invert token_scale for exact match with FBGemm
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, false,
                                     has_residual>(
        out, input, weight, rms, &token_scale, hidden_size, residual);
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
template <typename scalar_t, typename scalar_out_t, bool has_residual = false,
          bool is_scale_transposed = false, int32_t group_size = 0>
__global__ void rms_norm_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,  // [..., hidden_size]
    float* __restrict__ scales,      // [num_tokens, hidden_size / group_size]
                                     // or
                                     // [hidden_size / group_size, num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr) {
  float rms;
  // Compute RMS
  // Always able to vectorize due to constraints on hidden_size
  vllm::vectorized::compute_rms<scalar_t, has_residual>(
      &rms, input, hidden_size, var_epsilon, residual);

  // Compute Scale
  // Always able to vectorize due to constraints on hidden_size and group_size
  vllm::vectorized::compute_dynamic_per_token_scales<
      scalar_t, scalar_out_t, has_residual, is_scale_transposed, group_size>(
      nullptr, scales, input, weight, rms, scale_ub, hidden_size, residual);

  // RMS Norm + Quant
  // Always able to vectorize due to constraints on hidden_size
  // For int8, don't invert token_scale here: do it inside the norm_and_quant
  // kernel. We do it because particular elements of token_scale can be shared
  // between multiple threads, so this way, we avoid extra synchronization
  // overhead.
  vllm::vectorized::norm_and_quant<
      scalar_t, scalar_out_t, std::is_same_v<scalar_out_t, int8_t>,
      has_residual, is_scale_transposed, group_size>(
      out, input, weight, rms, scales, hidden_size, residual);
}

}  // namespace vllm

// Residual add + RMS norm + dynamic per token
template <typename scalar_in_t>
void rms_norm_dynamic_per_token_quant_dispatch(
    torch::stable::Tensor& out,           // [..., hidden_size]
    torch::stable::Tensor const& input,   // [..., hidden_size]
    torch::stable::Tensor const& weight,  // [hidden_size]
    torch::stable::Tensor& scales,        // [num_tokens]
    double const var_epsilon,  // Variance epsilon used in norm calculation
    std::optional<torch::stable::Tensor> const& scale_ub,
    std::optional<torch::stable::Tensor>& residual) {
  int32_t hidden_size = input.size(-1);
  auto num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device());

  VLLM_STABLE_DISPATCH_BOOL(residual.has_value(), has_residual, [&] {
    VLLM_STABLE_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,
                                                        has_residual>
              <<<grid, block, 0, stream>>>(
                  out.mutable_data_ptr<scalar_t>(),
                  scales.mutable_data_ptr<float>(),
                  input.const_data_ptr<scalar_in_t>(),
                  weight.const_data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->const_data_ptr<float>()
                                       : nullptr,
                  var_epsilon, hidden_size,
                  has_residual ? residual->mutable_data_ptr<scalar_in_t>()
                               : nullptr);
        });
  });
}

void rms_norm_dynamic_per_token_quant(
    torch::stable::Tensor& out,           // [..., hidden_size]
    torch::stable::Tensor const& input,   // [..., hidden_size]
    torch::stable::Tensor const& weight,  // [hidden_size]
    torch::stable::Tensor& scales,        // [num_tokens]
    double const var_epsilon,  // Variance epsilon used in norm calculation
    std::optional<torch::stable::Tensor> scale_ub,
    std::optional<torch::stable::Tensor> residual) {
  static torch::headeronly::ScalarType kFp8Type =
      is_fp8_ocp() ? torch::headeronly::ScalarType::Float8_e4m3fn
                   : torch::headeronly::ScalarType::Float8_e4m3fnuz;
  auto kInt8Type = torch::headeronly::ScalarType::Char;
  STD_TORCH_CHECK(out.scalar_type() == kFp8Type ||
                  out.scalar_type() == kInt8Type);
  STD_TORCH_CHECK(out.is_contiguous() && input.is_contiguous());

  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(out.scalar_type() == kFp8Type);
  }
  STD_TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  STD_TORCH_CHECK(scales.scalar_type() == torch::headeronly::ScalarType::Float);
  if (residual.has_value()) {
    STD_TORCH_CHECK(residual->scalar_type() == input.scalar_type());
  }

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_dynamic_per_token_quant_dispatch", [&] {
        rms_norm_dynamic_per_token_quant_dispatch<scalar_t>(
            out, input, weight, scales, var_epsilon, scale_ub, residual);
      });
}

// Residual add + RMS norm + dynamic per token
void rms_norm_per_block_quant_dispatch(
    torch::stable::Tensor& out,           // [..., hidden_size]
    torch::stable::Tensor const& input,   // [..., hidden_size]
    torch::stable::Tensor const& weight,  // [hidden_size]
    torch::stable::Tensor& scales,        // [num_tokens, hidden_size /
                                          // group_size] or [hidden_size /
                                          // group_size, num_tokens]
    int32_t group_size,
    double const var_epsilon,  // Variance epsilon used in norm calculation
    std::optional<torch::stable::Tensor> const& scale_ub,
    std::optional<torch::stable::Tensor>& residual, bool is_scale_transposed) {
  int32_t hidden_size = input.size(-1);
  auto num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  const int max_block_size = (num_tokens <= 256) ? 512 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device());

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_per_block_quant_fp_dispatch", [&] {
        using scalar_in_t = scalar_t;
        VLLM_STABLE_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
          VLLM_STABLE_DISPATCH_BOOL(residual.has_value(), has_residual, [&] {
            VLLM_STABLE_DISPATCH_BOOL(
                is_scale_transposed, transpose_scale, [&] {
                  VLLM_STABLE_DISPATCH_QUANT_TYPES(
                      out.scalar_type(), "rms_norm_per_block_quant_kernel",
                      [&] {
                        vllm::rms_norm_per_block_quant_kernel<
                            scalar_in_t, scalar_t, has_residual,
                            transpose_scale, gs><<<grid, block, 0, stream>>>(
                            out.mutable_data_ptr<scalar_t>(),
                            scales.mutable_data_ptr<float>(),
                            input.const_data_ptr<scalar_in_t>(),
                            weight.const_data_ptr<scalar_in_t>(),
                            scale_ub.has_value()
                                ? scale_ub->const_data_ptr<float>()
                                : nullptr,
                            var_epsilon, hidden_size,
                            has_residual
                                ? residual->mutable_data_ptr<scalar_in_t>()
                                : nullptr);
                      });
                });
          });
        });
      });
}

void rms_norm_per_block_quant(torch::stable::Tensor& out,
                              torch::stable::Tensor const& input,
                              torch::stable::Tensor const& weight,
                              torch::stable::Tensor& scales,
                              double const var_epsilon,
                              std::optional<torch::stable::Tensor> scale_ub,
                              std::optional<torch::stable::Tensor> residual,
                              int64_t group_size, bool is_scale_transposed) {
  static torch::headeronly::ScalarType kFp8Type =
      is_fp8_ocp() ? torch::headeronly::ScalarType::Float8_e4m3fn
                   : torch::headeronly::ScalarType::Float8_e4m3fnuz;
  auto kInt8Type = torch::headeronly::ScalarType::Char;
  STD_TORCH_CHECK(out.scalar_type() == kFp8Type ||
                  out.scalar_type() == kInt8Type);
  STD_TORCH_CHECK(out.is_contiguous() && input.is_contiguous());

  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(out.scalar_type() == kFp8Type);
  }
  STD_TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  STD_TORCH_CHECK(scales.scalar_type() == torch::headeronly::ScalarType::Float);
  if (residual.has_value()) {
    STD_TORCH_CHECK(residual->scalar_type() == input.scalar_type());
  }

  STD_TORCH_CHECK(group_size == 128 || group_size == 64,
                  "Unsupported group size: ", group_size);

  rms_norm_per_block_quant_dispatch(out, input, weight, scales, group_size,
                                    var_epsilon, scale_ub, residual,
                                    is_scale_transposed);
}
