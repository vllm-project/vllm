#include <optional>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "attention_dtypes.h"
#include "attention_utils.cuh"
#include "../quantization/w8a8/fp8/common.cuh"
#include "../dispatch_utils.h"

namespace vllm {

// Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
// can be used to combine partial attention results (in the split-KV case)
template <typename scalar_t, typename output_t, const uint NUM_THREADS,
          bool USE_FP8_OUTPUT>
__global__ void merge_attn_states_kernel(
    output_t* output, float* output_lse, const scalar_t* prefix_output,
    const float* prefix_lse, const scalar_t* suffix_output,
    const float* suffix_lse, const float* output_scale, const uint num_tokens,
    const uint num_heads, const uint head_size, const uint prefix_head_stride,
    const uint output_head_stride) {
  // Inputs always load 128-bit packs (pack_size elements of scalar_t).
  // Outputs store pack_size elements of output_t, which is smaller for FP8.
  using input_pack_t = uint4;
  using output_pack_t =
      std::conditional_t<USE_FP8_OUTPUT,
                         std::conditional_t<sizeof(scalar_t) == 4, uint, uint2>,
                         uint4>;
  const uint pack_size = 16 / sizeof(scalar_t);
  const uint threads_per_head = head_size / pack_size;

  const uint global_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

  // global_idx -> token_idx + head_idx + pack_idx
  const uint token_head_idx = global_idx / threads_per_head;
  const uint pack_idx = global_idx % threads_per_head;

  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  const uint pack_offset = pack_idx * pack_size;  // (0~15)*8, etc.
  const uint src_head_offset = token_idx * num_heads * prefix_head_stride +
                               head_idx * prefix_head_stride;
  const uint dst_head_offset = token_idx * num_heads * output_head_stride +
                               head_idx * output_head_stride;
  const scalar_t* prefix_head_ptr = prefix_output + src_head_offset;
  const scalar_t* suffix_head_ptr = suffix_output + src_head_offset;
  output_t* output_head_ptr = output + dst_head_offset;

  // Pre-invert scale: multiplication is faster than division
  float fp8_scale_inv = 1.0f;
  if constexpr (USE_FP8_OUTPUT) {
    fp8_scale_inv = 1.0f / *output_scale;
  }

  float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
  float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
  p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

  const float max_lse = fmaxf(p_lse, s_lse);

  /* In certain edge cases, MLA can produce p_lse = s_lse = -inf;
     continuing the pipeline then yields NaN. Root cause: with chunked prefill
     a batch may be split into two chunks; if a request in that batch has no
     prefix hit, every LSE entry for that request's position is -inf, and at
     this moment we merge cross-attention at first. For now we simply emit
     prefix_output (expected to be all zeros) and prefix_lse (-inf) to fix
     this problem.
  */
  if (std::isinf(max_lse)) {
    if (pack_offset < head_size) {
      input_pack_t p_out_pack = reinterpret_cast<const input_pack_t*>(
          prefix_head_ptr)[pack_offset / pack_size];

      if constexpr (USE_FP8_OUTPUT) {
        // Convert prefix values to FP8 (since -inf means no data,
        // prefix_output is expected to be zeros)
        output_t o_out_pack[pack_size];
#pragma unroll
        for (uint i = 0; i < pack_size; ++i) {
          const float val =
              vllm::to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
          o_out_pack[i] =
              vllm::scaled_fp8_conversion<true, output_t>(val, fp8_scale_inv);
        }
        reinterpret_cast<output_pack_t*>(
            output_head_ptr)[pack_offset / pack_size] =
            *reinterpret_cast<output_pack_t*>(o_out_pack);
      } else {
        reinterpret_cast<output_pack_t*>(
            output_head_ptr)[pack_offset / pack_size] = p_out_pack;
      }
    }
    // We only need to write to output_lse once per head.
    if (output_lse != nullptr && pack_idx == 0) {
      output_lse[head_idx * num_tokens + token_idx] = max_lse;
    }
    return;
  }

  p_lse = p_lse - max_lse;
  s_lse = s_lse - max_lse;
  const float p_se = expf(p_lse);
  const float s_se = expf(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;

  if (pack_offset < head_size) {
    input_pack_t p_out_pack = reinterpret_cast<const input_pack_t*>(
        prefix_head_ptr)[pack_offset / pack_size];
    input_pack_t s_out_pack = reinterpret_cast<const input_pack_t*>(
        suffix_head_ptr)[pack_offset / pack_size];

    // Compute merged values in float32
    float o_out_f[pack_size];
#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
      const float p_out_f =
          vllm::to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
      const float s_out_f =
          vllm::to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
      o_out_f[i] = p_out_f * p_scale + (s_out_f * s_scale);
    }

    // Convert and store
    if constexpr (USE_FP8_OUTPUT) {
      output_t o_out_pack[pack_size];
#pragma unroll
      for (uint i = 0; i < pack_size; ++i) {
        o_out_pack[i] = vllm::scaled_fp8_conversion<true, output_t>(
            o_out_f[i], fp8_scale_inv);
      }
      reinterpret_cast<output_pack_t*>(
          output_head_ptr)[pack_offset / pack_size] =
          *reinterpret_cast<output_pack_t*>(o_out_pack);
    } else {
      output_pack_t o_out_pack;
#pragma unroll
      for (uint i = 0; i < pack_size; ++i) {
        vllm::from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i],
                         o_out_f[i]);
      }
      reinterpret_cast<output_pack_t*>(
          output_head_ptr)[pack_offset / pack_size] = o_out_pack;
    }
  }
  // We only need to write to output_lse once per head.
  if (output_lse != nullptr && pack_idx == 0) {
    float out_lse = logf(out_se) + max_lse;
    output_lse[head_idx * num_tokens + token_idx] = out_lse;
  }
}

}  // namespace vllm

// The following macro is used to dispatch the conversion function based on
// the output data type. The FN is a macro that calls a function with
// template<typename scalar_t>.
#define DISPATCH_BY_SCALAR_DTYPE(scalar_dtype, fn)                      \
  {                                                                     \
    if (scalar_dtype == at::ScalarType::Float) {                        \
      fn(float);                                                        \
    } else if (scalar_dtype == at::ScalarType::Half) {                  \
      fn(uint16_t);                                                     \
    } else if (scalar_dtype == at::ScalarType::BFloat16) {              \
      fn(__nv_bfloat16);                                                \
    } else {                                                            \
      TORCH_CHECK(false, "Unsupported data type of O: ", scalar_dtype); \
    }                                                                   \
  }

#define LAUNCH_MERGE_ATTN_STATES(scalar_t, output_t, NUM_THREADS,              \
                                 USE_FP8_OUTPUT)                               \
  {                                                                            \
    vllm::merge_attn_states_kernel<scalar_t, output_t, NUM_THREADS,            \
                                   USE_FP8_OUTPUT>                             \
        <<<grid, block, 0, stream>>>(                                          \
            reinterpret_cast<output_t*>(output.data_ptr()), output_lse_ptr,    \
            reinterpret_cast<scalar_t*>(prefix_output.data_ptr()),             \
            reinterpret_cast<float*>(prefix_lse.data_ptr()),                   \
            reinterpret_cast<scalar_t*>(suffix_output.data_ptr()),             \
            reinterpret_cast<float*>(suffix_lse.data_ptr()), output_scale_ptr, \
            num_tokens, num_heads, head_size, prefix_head_stride,              \
            output_head_stride);                                               \
  }

/*@brief Merges the attention states from prefix and suffix
 * into the output tensor. NUM_TOKENS: n, NUM_HEADS: h, HEAD_SIZE: d
 *
 * @param output [n,h,d] The output tensor to store the merged attention states.
 * @param output_lse [h,d] Optional tensor to store the log-sum-exp values.
 * @param prefix_output [n,h,d] The prefix attention states.
 * @param prefix_lse [h,n] The log-sum-exp values for the prefix attention
 * states.
 * @param suffix_output [n,h,d] The suffix attention states.
 * @param suffix_lse [h,n] The log-sum-exp values for the suffix attention
 * states.
 * @param output_scale Optional scalar tensor for FP8 static quantization.
 * When provided, output must be FP8 dtype.
 */
template <typename scalar_t>
void merge_attn_states_launcher(
    torch::Tensor& output, std::optional<torch::Tensor> output_lse,
    const torch::Tensor& prefix_output, const torch::Tensor& prefix_lse,
    const torch::Tensor& suffix_output, const torch::Tensor& suffix_lse,
    const std::optional<torch::Tensor>& output_scale) {
  constexpr uint NUM_THREADS = 128;
  const uint num_tokens = prefix_output.size(0);
  const uint num_heads = prefix_output.size(1);
  const uint head_size = prefix_output.size(2);
  const uint prefix_head_stride = prefix_output.stride(1);
  const uint output_head_stride = output.stride(1);
  // Thread mapping is based on input BF16 pack_size
  const uint pack_size = 16 / sizeof(scalar_t);
  TORCH_CHECK(head_size % pack_size == 0,
              "headsize must be multiple of pack_size:", pack_size);
  float* output_lse_ptr = nullptr;
  if (output_lse.has_value()) {
    output_lse_ptr = output_lse.value().data_ptr<float>();
  }
  float* output_scale_ptr = nullptr;
  if (output_scale.has_value()) {
    output_scale_ptr = output_scale.value().data_ptr<float>();
  }
  // Process one pack elements per thread. for float, the
  // pack_size is 4 for half/bf16, the pack_size is 8.
  const uint threads_per_head = head_size / pack_size;
  const uint total_threads = num_tokens * num_heads * threads_per_head;

  dim3 block(NUM_THREADS);
  dim3 grid((total_threads + NUM_THREADS - 1) / NUM_THREADS);

  const c10::cuda::OptionalCUDAGuard device_guard(prefix_output.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (output_scale.has_value()) {
    // FP8 output path - dispatch on output FP8 type
    VLLM_DISPATCH_FP8_TYPES(output.scalar_type(), "merge_attn_states_fp8", [&] {
      LAUNCH_MERGE_ATTN_STATES(scalar_t, fp8_t, NUM_THREADS, true);
    });
  } else {
    // Original BF16/FP16/FP32 output path
    LAUNCH_MERGE_ATTN_STATES(scalar_t, scalar_t, NUM_THREADS, false);
  }
}

#define CALL_MERGE_ATTN_STATES_LAUNCHER(scalar_t)                           \
  {                                                                         \
    merge_attn_states_launcher<scalar_t>(output, output_lse, prefix_output, \
                                         prefix_lse, suffix_output,         \
                                         suffix_lse, output_scale);         \
  }

void merge_attn_states(torch::Tensor& output,
                       std::optional<torch::Tensor> output_lse,
                       const torch::Tensor& prefix_output,
                       const torch::Tensor& prefix_lse,
                       const torch::Tensor& suffix_output,
                       const torch::Tensor& suffix_lse,
                       const std::optional<torch::Tensor>& output_scale) {
  if (output_scale.has_value()) {
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                    output.scalar_type() == at::ScalarType::Float8_e4m3fnuz,
                "output must be FP8 when output_scale is provided, got: ",
                output.scalar_type());
  } else {
    TORCH_CHECK(output.scalar_type() == prefix_output.scalar_type(),
                "output dtype (", output.scalar_type(),
                ") must match prefix_output dtype (",
                prefix_output.scalar_type(), ") when output_scale is not set");
  }
  // Always dispatch on prefix_output (input) dtype
  DISPATCH_BY_SCALAR_DTYPE(prefix_output.dtype(),
                           CALL_MERGE_ATTN_STATES_LAUNCHER);
}
