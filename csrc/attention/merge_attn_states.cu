#include <optional>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

namespace vllm {

// Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
// can be used to combine partial attention results (in the split-KV case)
template <typename scalar_t>
__device__ __forceinline__ void merge_attn_states_common(
    scalar_t* output, float* output_lse,
    const scalar_t* __restrict__ prefix_output,
    const float* __restrict__ prefix_lse,
    const scalar_t* __restrict__ suffix_output,
    const float* __restrict__ suffix_lse, const uint num_tokens,
    const uint num_heads, const uint head_size, const uint token_idx,
    const uint head_idx, const uint thr_idx) {
  using pack_128b_t = uint4;  // float -> 4, half/bf16 -> 8
  constexpr uint pack_size = 16 / sizeof(scalar_t);

  const uint thr_offset = thr_idx * pack_size;  // (0~15)*8, etc.
  const uint blk_offset =
      token_idx * num_heads * head_size + head_idx * head_size;
  const scalar_t* prefix_output_blk = prefix_output + blk_offset;
  const scalar_t* suffix_output_blk = suffix_output + blk_offset;
  scalar_t* output_blk = output + blk_offset;

  float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
  float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
  p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

  const float max_lse = fmaxf(p_lse, s_lse);
  p_lse = p_lse - max_lse;
  s_lse = s_lse - max_lse;
  const float p_se = expf(p_lse);
  const float s_se = expf(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;

  if (thr_offset < head_size) {
    // Pack 128b load
    pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(
        prefix_output_blk)[thr_offset / pack_size];
    pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(
        suffix_output_blk)[thr_offset / pack_size];
    pack_128b_t o_out_pack;

#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
      // Always use float for FMA to keep precision.
      // half(uint16_t), bfloat16, float -> float.
      const float p_out_f =
          vllm::to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
      const float s_out_f =
          vllm::to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
      // fma: a * b + c = p_out_f * p_scale + (s_out_f * s_scale)
      const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);
      // float -> half(uint16_t), bfloat16, float.
      vllm::from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i], o_out_f);
    }

    // Pack 128b storage
    reinterpret_cast<pack_128b_t*>(output_blk)[thr_offset / pack_size] =
        o_out_pack;
  }
  // We only need to write to output_lse once per head.
  if (output_lse != nullptr && thr_idx == 0) {
    float out_lse = logf(out_se) + max_lse;
    output_lse[head_idx * num_tokens + token_idx] = out_lse;
  }
}

template <typename scalar_t>
__global__ void merge_attn_states_kernel(
    scalar_t* output, float* output_lse, const scalar_t* prefix_output,
    const float* prefix_lse, const scalar_t* suffix_output,
    const float* suffix_lse, const uint num_tokens, const uint num_heads,
    const uint head_size) {
  constexpr uint BLOCK_SIZE = 128;
  const uint pack_size = 16 / sizeof(scalar_t);
  const uint threads_per_head = head_size / pack_size;

  const uint global_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

  // global_idx -> token_idx + head_idx + thr_idx
  const uint token_head_idx = global_idx / threads_per_head;
  const uint thr_idx = global_idx % threads_per_head;

  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  merge_attn_states_common<scalar_t>(
      output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse,
      num_tokens, num_heads, head_size, token_idx, head_idx, thr_idx);
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

#define LAUNCH_MERGE_ATTN_STATES(scalar_t)                              \
  {                                                                     \
    vllm::merge_attn_states_kernel<scalar_t><<<grid, block>>>(          \
        reinterpret_cast<scalar_t*>(output.data_ptr()), output_lse_ptr, \
        reinterpret_cast<scalar_t*>(prefix_output.data_ptr()),          \
        reinterpret_cast<float*>(prefix_lse.data_ptr()),                \
        reinterpret_cast<scalar_t*>(suffix_output.data_ptr()),          \
        reinterpret_cast<float*>(suffix_lse.data_ptr()), num_tokens,    \
        num_heads, head_size);                                          \
  }

template <typename scalar_t>
void merge_attn_states_launcher(
    torch::Tensor& output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    std::optional<torch::Tensor> output_lse,  // [NUM_HEADS, NUM_TOKENS]
    const torch::Tensor& prefix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const torch::Tensor& prefix_lse,     // [NUM_HEADS, NUM_TOKENS]
    const torch::Tensor& suffix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const torch::Tensor& suffix_lse      // [NUM_HEADS, NUM_TOKENS]
) {
  const uint num_tokens = output.size(0);
  const uint num_heads = output.size(1);
  const uint head_size = output.size(2);
  const uint pack_size = 16 / sizeof(scalar_t);
  TORCH_CHECK(head_size % pack_size == 0,
              "headsize must be multiple of pack_size:", pack_size);
  const uint threads_per_head = head_size / pack_size;
  const uint total_threads = num_tokens * num_heads * threads_per_head;
  float* output_lse_ptr = nullptr;
  if (output_lse.has_value()) {
    output_lse_ptr = output_lse.value().data_ptr<float>();
  }

  dim3 block(128);
  dim3 grid((total_threads + block.x - 1) / block.x);
  LAUNCH_MERGE_ATTN_STATES(scalar_t);
}

#define CALL_MERGE_ATTN_STATES_LAUNCHER(scalar_t)                           \
  {                                                                         \
    merge_attn_states_launcher<scalar_t>(output, output_lse, prefix_output, \
                                         prefix_lse, suffix_output,         \
                                         suffix_lse);                       \
  }

void merge_attn_states(
    torch::Tensor& output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    std::optional<torch::Tensor> output_lse,  // [NUM_HEADS, NUM_TOKENS]
    const torch::Tensor& prefix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const torch::Tensor& prefix_lse,     // [NUM_HEADS, NUM_TOKENS]
    const torch::Tensor& suffix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const torch::Tensor& suffix_lse      // [NUM_HEADS, NUM_TOKENS]
) {
  DISPATCH_BY_SCALAR_DTYPE(output.dtype(), CALL_MERGE_ATTN_STATES_LAUNCHER);
}
