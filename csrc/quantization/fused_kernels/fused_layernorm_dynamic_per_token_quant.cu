
#include "../../torch_utils.h"

#include "../../dispatch_utils.h"
#include "core/utils.hpp"
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
    int32_t const input_stride, scalar_t* __restrict__ residual = nullptr) {
  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute rms
  vllm::vectorized::compute_rms<scalar_t, has_residual>(
      &rms, input, hidden_size, input_stride, var_epsilon, residual);

  // Compute scale
  vllm::vectorized::compute_dynamic_per_token_scales<scalar_t, scalar_out_t,
                                                     has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub, hidden_size,
      input_stride, residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    token_scale = 1.0f / token_scale;
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, true,
                                     has_residual>(out, input, weight, rms,
                                                   &token_scale, hidden_size,
                                                   input_stride, residual);
  } else {
    // FP8 - Do not invert token_scale for exact match with FBGemm
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, false,
                                     has_residual>(out, input, weight, rms,
                                                   &token_scale, hidden_size,
                                                   input_stride, residual);
  }
}

constexpr int REG_KERNEL_MAX_THREADS = 512;

// Same as rms_norm_dynamic_per_token_quant_vec, but each thread keeps its
// part of the token in registers, so input and residual are read once.
// hidden_size must be <= 4 * max_vecs_per_thread * blockDim.x (checked by
// the dispatch).
template <typename scalar_t, typename scalar_out_t, bool has_residual,
          int max_vecs_per_thread>
__global__ void __launch_bounds__(REG_KERNEL_MAX_THREADS)
    rms_norm_dynamic_per_token_quant_reg_kernel(
        scalar_out_t* __restrict__ out,       // [..., hidden_size]
        float* __restrict__ scales,           // [num_tokens]
        scalar_t const* __restrict__ input,   // [..., hidden_size]
        scalar_t const* __restrict__ weight,  // [hidden_size]
        float const* scale_ub, float const var_epsilon,
        int32_t const hidden_size, int32_t const input_stride,
        scalar_t* __restrict__ residual) {
  constexpr int VEC_SIZE = 4;
  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};
  constexpr bool is_int8 = std::is_same_v<scalar_out_t, int8_t>;
  int32_t const num_vec_elems = hidden_size >> 2;
  int64_t const input_token_offset =
      blockIdx.x * static_cast<int64_t>(input_stride);
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);

  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[input_token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  q8x4_t<scalar_out_t>* vec_output =
      reinterpret_cast<q8x4_t<scalar_out_t>*>(&out[token_offset]);
  vec4_t<scalar_t>* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
  }

  vec4_t<float> x[max_vecs_per_thread];
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < max_vecs_per_thread; ++k) {
    int32_t const i = threadIdx.x + k * blockDim.x;
    if (i < num_vec_elems) {
      vec4_t<scalar_t> const in = vec_input[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        x[k].val[j] = static_cast<float>(in.val[j]);
      }
      if constexpr (has_residual) {
        vec4_t<scalar_t> const r = vec_residual[i];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          x[k].val[j] += static_cast<float>(r.val[j]);
        }
      }
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        ss += x[k].val[j] * x[k].val[j];
      }
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, CubAddOp{}, blockDim.x);
  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + var_epsilon);
  }
  __syncthreads();
  float const rms = s_rms;

  float absmax = 0.0f;
#pragma unroll
  for (int k = 0; k < max_vecs_per_thread; ++k) {
    int32_t const i = threadIdx.x + k * blockDim.x;
    if (i < num_vec_elems) {
      vec4_t<scalar_t> const w = vec_weight[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        absmax = fmaxf(
            absmax, fabs(static_cast<scalar_t>(x[k].val[j] * rms) * w.val[j]));
      }
    }
  }
  absmax = BlockReduce(reduceStore).Reduce(absmax, CubMaxOp{}, blockDim.x);
  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    float scale = scale_ub ? min(absmax, *scale_ub) : absmax;
    scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
    s_token_scale = scale;
    scales[blockIdx.x] = scale;
  }
  __syncthreads();
  // int8 multiplies by the inverse scale; fp8 divides for FBGemm parity.
  float const scale_val = is_int8 ? 1.0f / s_token_scale : s_token_scale;

#pragma unroll
  for (int k = 0; k < max_vecs_per_thread; ++k) {
    int32_t const i = threadIdx.x + k * blockDim.x;
    if (i < num_vec_elems) {
      vec4_t<scalar_t> const w = vec_weight[i];
      if constexpr (has_residual) {
        vec4_t<scalar_t> r;
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          r.val[j] = static_cast<scalar_t>(x[k].val[j]);
        }
        vec_residual[i] = r;
      }
      q8x4_t<scalar_out_t> q;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        q.val[j] = ScaledQuant<scalar_out_t, is_int8>::quant_fn(
            static_cast<scalar_t>(x[k].val[j] * rms) * w.val[j], scale_val);
      }
      vec_output[i] = q;
    }
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
    int32_t const input_stride, scalar_t* __restrict__ residual = nullptr) {
  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0 and input_stride % 4 == 0;

  if (can_vectorize) {
    return rms_norm_dynamic_per_token_quant_vec<scalar_t, scalar_out_t,
                                                has_residual>(
        out, scales, input, weight, scale_ub, var_epsilon, hidden_size,
        input_stride, residual);
  }

  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute RMS
  vllm::compute_rms<scalar_t, has_residual>(
      &rms, input, hidden_size, input_stride, var_epsilon, residual);
  // Compute Scale
  vllm::compute_dynamic_per_token_scales<scalar_t, scalar_out_t, has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub, hidden_size,
      input_stride, residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    token_scale = 1.0f / token_scale;
    vllm::norm_and_quant<scalar_t, scalar_out_t, true, has_residual>(
        out, input, weight, rms, &token_scale, hidden_size, input_stride,
        residual);
  } else {
    // FP8 - Do not invert s_token_scale for exact match with FBGemm
    vllm::norm_and_quant<scalar_t, scalar_out_t, false, has_residual>(
        out, input, weight, rms, &token_scale, hidden_size, input_stride,
        residual);
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
    int32_t const input_stride, scalar_t* __restrict__ residual = nullptr,
    int64_t outer_scale_stride = 1) {
  float rms;
  // Compute RMS
  // Always able to vectorize due to constraints on hidden_size
  vllm::vectorized::compute_rms<scalar_t, has_residual>(
      &rms, input, hidden_size, input_stride, var_epsilon, residual);

  // Compute Scale
  // Always able to vectorize due to constraints on hidden_size and group_size
  vllm::vectorized::compute_dynamic_per_token_scales<
      scalar_t, scalar_out_t, has_residual, is_scale_transposed, group_size>(
      nullptr, scales, input, weight, rms, scale_ub, hidden_size, input_stride,
      residual, outer_scale_stride);

  // RMS Norm + Quant
  // Always able to vectorize due to constraints on hidden_size
  // For int8, don't invert token_scale here: do it inside the norm_and_quant
  // kernel. We do it because particular elements of token_scale can be shared
  // between multiple threads, so this way, we avoid extra synchronization
  // overhead.
  vllm::vectorized::norm_and_quant<
      scalar_t, scalar_out_t, std::is_same_v<scalar_out_t, int8_t>,
      has_residual, is_scale_transposed, group_size>(
      out, input, weight, rms, scales, hidden_size, input_stride, residual,
      outer_scale_stride);
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
  int32_t input_stride =
      torch::stable::view(input, {-1, hidden_size}).stride(0);
  auto num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  // Use the register kernel (4 or 8 vec4s per thread) from two tokens per SM;
  // with fewer tokens its smaller blocks underfill the GPU. Batch-invariant
  // mode keeps the existing block so the reduction order does not depend on
  // num_tokens.
  bool const can_vectorize = hidden_size % 4 == 0 and input_stride % 4 == 0;
  bool const use_register_path =
      can_vectorize && !vllm::vllm_is_batch_invariant() &&
      num_tokens >= 2 * get_device_prop()->multiProcessorCount &&
      hidden_size <= 4 * 8 * vllm::REG_KERNEL_MAX_THREADS;
  int const vecs_per_thread =
      use_register_path ? (hidden_size <= 4096 ? 4 : 8) : 0;
  if (use_register_path) {
    block = dim3(round_to_next_multiple_of(
        div_ceil(hidden_size / 4, vecs_per_thread), 32));
  }

  VLLM_STABLE_DISPATCH_BOOL(residual.has_value(), has_residual, [&] {
    VLLM_STABLE_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
          auto kernel = vllm::rms_norm_dynamic_per_token_quant_kernel<
              scalar_in_t, scalar_t, has_residual>;
          if (vecs_per_thread == 4) {
            kernel = vllm::rms_norm_dynamic_per_token_quant_reg_kernel<
                scalar_in_t, scalar_t, has_residual, 4>;
          } else if (vecs_per_thread == 8) {
            kernel = vllm::rms_norm_dynamic_per_token_quant_reg_kernel<
                scalar_in_t, scalar_t, has_residual, 8>;
          }
          kernel<<<grid, block, 0, stream>>>(
              out.mutable_data_ptr<scalar_t>(),
              scales.mutable_data_ptr<float>(),
              input.const_data_ptr<scalar_in_t>(),
              weight.const_data_ptr<scalar_in_t>(),
              scale_ub.has_value() ? scale_ub->const_data_ptr<float>()
                                   : nullptr,
              var_epsilon, hidden_size, input_stride,
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
  STD_TORCH_CHECK(out.scalar_type() == kFp8Type ||
                  out.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(out.is_contiguous());
  STD_TORCH_CHECK(input.stride(-1) == 1,
                  "Input must be contiguous in the last dimension");

  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(out.scalar_type() == kFp8Type);
  }
  STD_TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  STD_TORCH_CHECK(scales.scalar_type() == torch::headeronly::ScalarType::Float);
  if (residual) {
    STD_TORCH_CHECK(residual->scalar_type() == input.scalar_type());
    STD_TORCH_CHECK(residual->is_contiguous());
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
                                          // group_size] or
                                          // [hidden_size / group_size,
                                          // num_tokens]
    int32_t group_size,
    double const var_epsilon,  // Variance epsilon used in norm calculation
    std::optional<torch::stable::Tensor> const& scale_ub,
    std::optional<torch::stable::Tensor>& residual, bool is_scale_transposed) {
  int32_t hidden_size = input.size(-1);
  int32_t input_stride =
      torch::stable::view(input, {-1, hidden_size}).stride(0);

  STD_TORCH_CHECK(hidden_size % 4 == 0,
                  "Hidden size must be divisible by 4 for vectorized access");
  STD_TORCH_CHECK(input_stride % 4 == 0,
                  "Input stride must be divisible by 4 for vectorized access");
  STD_TORCH_CHECK(group_size % 4 == 0,
                  "Group size must be divisible by 4 for vectorized access");

  auto num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  const bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
  const int max_block_size =
      batch_invariant_launch ? 512 : ((num_tokens <= 256) ? 512 : 256);
  dim3 block(std::min(hidden_size, max_block_size));
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

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
                            var_epsilon, hidden_size, input_stride,
                            has_residual
                                ? residual->mutable_data_ptr<scalar_in_t>()
                                : nullptr,
                            scales.stride(1));
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
  STD_TORCH_CHECK(out.scalar_type() == kFp8Type ||
                  out.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(out.is_contiguous());
  STD_TORCH_CHECK(input.stride(-1) == 1,
                  "Input must be contiguous in the last dimension");

  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(out.scalar_type() == kFp8Type);
  }
  STD_TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  STD_TORCH_CHECK(scales.scalar_type() == torch::headeronly::ScalarType::Float);
  if (residual) {
    STD_TORCH_CHECK(residual->scalar_type() == input.scalar_type());
    STD_TORCH_CHECK(residual->is_contiguous());
  }

  STD_TORCH_CHECK(group_size == 128 || group_size == 64,
                  "Unsupported group size: ", group_size);

  if (scales.stride(1) > 1) {
    STD_TORCH_CHECK(
        is_scale_transposed,
        "Outer scale stride must be 1 when scales are not transposed");
  }

  int64_t hidden_size = input.size(-1);
  STD_TORCH_CHECK(hidden_size > 0 && hidden_size % group_size == 0,
                  "hidden_size must be a positive multiple of group_size");
  int64_t num_tokens = input.numel() / hidden_size;
  int64_t num_groups = hidden_size / group_size;
  STD_TORCH_CHECK(scales.numel() >= num_tokens * num_groups,
                  "scales buffer too small: need ", num_tokens * num_groups,
                  " elements, got ", scales.numel());

  rms_norm_per_block_quant_dispatch(out, input, weight, scales, group_size,
                                    var_epsilon, scale_ub, residual,
                                    is_scale_transposed);
}
