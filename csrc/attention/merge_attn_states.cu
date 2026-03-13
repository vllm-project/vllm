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
  // 128-bit pack type for input loads (and full output stores).
  using pack_128b_t = uint4;
  constexpr uint pack_size = 16 / sizeof(scalar_t);  // elements per input pack
  // FP8: process multiple input packs per thread so output writes are 128-bit.
  // sizeof(scalar_t) gives: bf16/fp16 -> 2 packs, fp32 -> 4 packs.
  constexpr uint packs_per_group = USE_FP8_OUTPUT ? sizeof(scalar_t) : 1;
  constexpr uint elems_per_group = pack_size * packs_per_group;  // 16 for FP8
  // Fallback pack type for boundary threads that can't fill a full 128-bit
  // output store (pack_size fp8 bytes: uint2 for bf16, uint for fp32).
  using small_pack_t =
      std::conditional_t<USE_FP8_OUTPUT,
                         std::conditional_t<sizeof(scalar_t) == 4, uint, uint2>,
                         pack_128b_t>;
  // Use ceiling division so the last thread handles any remainder.
  // One thread per pack group.
  const uint threads_per_head =
      (head_size + elems_per_group - 1) / elems_per_group;

  const uint global_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

  // global_idx -> token_idx + head_idx + pack_group_idx
  const uint token_head_idx = global_idx / threads_per_head;
  const uint pack_group_idx = global_idx % threads_per_head;

  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  // pack idx and elem offset of base (first) pack in thread
  const uint base_pack_idx = pack_group_idx * packs_per_group;
  const uint base_elem_offset = pack_group_idx * elems_per_group;
  // How many elements this thread actually processes (may be < elems_per_group
  // for the last thread when head_size is not a multiple of elems_per_group).
  const uint num_elems = min(elems_per_group, head_size - base_elem_offset);
  const uint num_packs = num_elems / pack_size;

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
     prefix hit, every LSE entry for that request’s position is -inf, and at
     this moment we merge cross-attention at first. For now we simply emit
     prefix_output (expected to be all zeros) and prefix_lse (-inf) to fix
     this problem.
  */
  if (std::isinf(max_lse)) {
    if constexpr (USE_FP8_OUTPUT) {
      // Load input packs, convert to FP8, store
      output_t o_fp8[elems_per_group];
#pragma unroll
      for (uint p = 0; p < packs_per_group; ++p) {
        if (p < num_packs) {
          pack_128b_t p_pack = reinterpret_cast<const pack_128b_t*>(
              prefix_head_ptr)[base_pack_idx + p];
#pragma unroll
          for (uint i = 0; i < pack_size; ++i) {
            const float val =
                vllm::to_float(reinterpret_cast<const scalar_t*>(&p_pack)[i]);
            o_fp8[p * pack_size + i] =
                vllm::scaled_fp8_conversion<true, output_t>(val, fp8_scale_inv);
          }
        }
      }
      // Full 128-bit store when we have all packs, otherwise fall back
      if (num_packs == packs_per_group) {
        reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_group_idx] =
            *reinterpret_cast<pack_128b_t*>(o_fp8);
      } else {
        for (uint p = 0; p < num_packs; ++p) {
          reinterpret_cast<small_pack_t*>(output_head_ptr)[base_pack_idx + p] =
              *reinterpret_cast<small_pack_t*>(&o_fp8[p * pack_size]);
        }
      }
    } else {
      // Non-FP8: single 128-bit load and store
      pack_128b_t p_pack =
          reinterpret_cast<const pack_128b_t*>(prefix_head_ptr)[base_pack_idx];
      reinterpret_cast<pack_128b_t*>(output_head_ptr)[base_pack_idx] = p_pack;
    }
    // We only need to write to output_lse once per head.
    if (output_lse != nullptr && pack_group_idx == 0) {
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

  // Merge, convert, and store
  if constexpr (USE_FP8_OUTPUT) {
    output_t o_fp8[elems_per_group];
#pragma unroll
    for (uint p = 0; p < packs_per_group; ++p) {
      if (p < num_packs) {
        pack_128b_t p_pack = reinterpret_cast<const pack_128b_t*>(
            prefix_head_ptr)[base_pack_idx + p];
        pack_128b_t s_pack = reinterpret_cast<const pack_128b_t*>(
            suffix_head_ptr)[base_pack_idx + p];
#pragma unroll
        for (uint i = 0; i < pack_size; ++i) {
          const float p_out_f =
              vllm::to_float(reinterpret_cast<const scalar_t*>(&p_pack)[i]);
          const float s_out_f =
              vllm::to_float(reinterpret_cast<const scalar_t*>(&s_pack)[i]);
          const float merged = p_out_f * p_scale + (s_out_f * s_scale);
          o_fp8[p * pack_size + i] =
              vllm::scaled_fp8_conversion<true, output_t>(merged,
                                                          fp8_scale_inv);
        }
      }
    }
    // Full 128-bit store when we have all packs, otherwise fall back
    if (num_packs == packs_per_group) {
      reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_group_idx] =
          *reinterpret_cast<pack_128b_t*>(o_fp8);
    } else {
      for (uint p = 0; p < num_packs; ++p) {
        reinterpret_cast<small_pack_t*>(output_head_ptr)[base_pack_idx + p] =
            *reinterpret_cast<small_pack_t*>(&o_fp8[p * pack_size]);
      }
    }
  } else {
    pack_128b_t p_pack =
        reinterpret_cast<const pack_128b_t*>(prefix_head_ptr)[base_pack_idx];
    pack_128b_t s_pack =
        reinterpret_cast<const pack_128b_t*>(suffix_head_ptr)[base_pack_idx];
    pack_128b_t o_pack;
#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
      const float p_out_f =
          vllm::to_float(reinterpret_cast<const scalar_t*>(&p_pack)[i]);
      const float s_out_f =
          vllm::to_float(reinterpret_cast<const scalar_t*>(&s_pack)[i]);
      const float merged = p_out_f * p_scale + (s_out_f * s_scale);
      vllm::from_float(reinterpret_cast<scalar_t*>(&o_pack)[i], merged);
    }
    reinterpret_cast<pack_128b_t*>(output_head_ptr)[base_pack_idx] = o_pack;
  }
  // We only need to write to output_lse once per head.
  if (output_lse != nullptr && pack_group_idx == 0) {
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
 * @param output_lse [h,n] Optional tensor to store the log-sum-exp values.
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
  constexpr uint pack_size = 16 / sizeof(scalar_t);
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

  const c10::cuda::OptionalCUDAGuard device_guard(prefix_output.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (output_scale.has_value()) {
    // FP8 output: each thread targets 16 elements for 128-bit output writes.
    // Ceiling division so the last thread handles any remainder.
    constexpr uint elems_per_group = 16;
    const uint threads_per_head =
        (head_size + elems_per_group - 1) / elems_per_group;
    const uint total_threads = num_tokens * num_heads * threads_per_head;
    dim3 grid((total_threads + NUM_THREADS - 1) / NUM_THREADS);
    dim3 block(NUM_THREADS);
    VLLM_DISPATCH_FP8_TYPES(output.scalar_type(), "merge_attn_states_fp8", [&] {
      LAUNCH_MERGE_ATTN_STATES(scalar_t, fp8_t, NUM_THREADS, true);
    });
  } else {
    // Non-FP8: each thread processes pack_size elements (128-bit I/O)
    const uint threads_per_head = head_size / pack_size;
    const uint total_threads = num_tokens * num_heads * threads_per_head;
    dim3 grid((total_threads + NUM_THREADS - 1) / NUM_THREADS);
    dim3 block(NUM_THREADS);
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
