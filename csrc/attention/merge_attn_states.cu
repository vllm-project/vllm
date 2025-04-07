#include <optional>
#include <cuda_runtime.h>
#include <cmath>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
// can be used to combine partial attention results (in the split-KV case)
// May loop over num heads for large NUM_TOKENS
template <const bool LOOP_OVER_HEAD, const bool OUTPUT_LSE>
__global__ void merge_attn_states_kernel(
    float* output,      // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    float* output_lse,  // [NUM_HEADS, NUM_TOKENS]
    const float* __restrict__ prefix_output,  // [NUM_TOKENS, NUM_HEADS,
                                              // HEAD_SIZE]
    const float* __restrict__ prefix_lse,     // [NUM_HEADS, NUM_TOKENS]
    const float* __restrict__ suffix_output,  // [NUM_TOKENS, NUM_HEADS,
                                              // HEAD_SIZE]
    const float* __restrict__ suffix_lse,     // [NUM_HEADS, NUM_TOKENS]
    const uint NUM_TOKENS,                    // NUM_TOKENS
    const uint NUM_HEADS,                     // NUM QUERY HEADS
    const uint HEAD_SIZE                      // HEAD_SIZE, 32,48,64,...,512,etc
) {
  if constexpr (LOOP_OVER_HEAD) {
    const uint token_idx = blockIdx.x;
    const uint thread_idx = threadIdx.x;

#pragma unroll
    for (uint head_idx = 0; head_idx < NUM_HEADS; ++head_idx) {
      float p_lse = prefix_lse[head_idx * NUM_TOKENS + token_idx];
      float s_lse = suffix_lse[head_idx * NUM_TOKENS + token_idx];
      p_lse =
          std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
      s_lse =
          std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

      const float max_lse = fmaxf(p_lse, s_lse);
      p_lse = p_lse - max_lse;
      s_lse = s_lse - max_lse;
      const float p_se = expf(p_lse);
      const float s_se = expf(s_lse);
      const float out_se = p_se + s_se;

      if constexpr (OUTPUT_LSE) {
        if (output_lse != nullptr) {
          float out_lse = logf(out_se) + max_lse;
          output_lse[head_idx * NUM_TOKENS + token_idx] = out_lse;
        }
      }

      const uint blk_offset =
          token_idx * NUM_HEADS * HEAD_SIZE + head_idx * HEAD_SIZE;
      const uint thr_offset = thread_idx * 4;
      const float p_scale = p_se / out_se;
      const float s_scale = s_se / out_se;
      const float* prefix_output_blk = prefix_output + blk_offset;
      const float* suffix_output_blk = suffix_output + blk_offset;
      float* output_blk = output + blk_offset;

      if (thr_offset < HEAD_SIZE) {
        float4 p_out4 = ((const float4*)(prefix_output_blk))[thr_offset / 4];
        float4 s_out4 = ((const float4*)(suffix_output_blk))[thr_offset / 4];

        float4 out4 = make_float4(p_out4.x * p_scale + s_out4.x * s_scale,
                                  p_out4.y * p_scale + s_out4.y * s_scale,
                                  p_out4.z * p_scale + s_out4.z * s_scale,
                                  p_out4.w * p_scale + s_out4.w * s_scale);
        ((float4*)(output_blk))[thr_offset / 4] = out4;
      }
    }  // end loop over heads
  } else {
    const uint token_idx = blockIdx.x;
    const uint head_idx = blockIdx.y;
    const uint thread_idx = threadIdx.x;

    float p_lse = prefix_lse[head_idx * NUM_TOKENS + token_idx];
    float s_lse = suffix_lse[head_idx * NUM_TOKENS + token_idx];
    p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
    s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

    const float max_lse = fmaxf(p_lse, s_lse);
    p_lse = p_lse - max_lse;
    s_lse = s_lse - max_lse;
    const float p_se = expf(p_lse);
    const float s_se = expf(s_lse);
    const float out_se = p_se + s_se;

    if constexpr (OUTPUT_LSE) {
      if (output_lse != nullptr) {
        float out_lse = logf(out_se) + max_lse;
        output_lse[head_idx * NUM_TOKENS + token_idx] = out_lse;
      }
    }

    const uint blk_offset =
        token_idx * NUM_HEADS * HEAD_SIZE + head_idx * HEAD_SIZE;
    const uint thr_offset = thread_idx * 4;
    const float p_scale = p_se / out_se;
    const float s_scale = s_se / out_se;
    const float* prefix_output_blk = prefix_output + blk_offset;
    const float* suffix_output_blk = suffix_output + blk_offset;
    float* output_blk = output + blk_offset;

    if (thr_offset < HEAD_SIZE) {
      float4 p_out4 = ((const float4*)(prefix_output_blk))[thr_offset / 4];
      float4 s_out4 = ((const float4*)(suffix_output_blk))[thr_offset / 4];

      float4 out4 = make_float4(p_out4.x * p_scale + s_out4.x * s_scale,
                                p_out4.y * p_scale + s_out4.y * s_scale,
                                p_out4.z * p_scale + s_out4.z * s_scale,
                                p_out4.w * p_scale + s_out4.w * s_scale);

      ((float4*)(output_blk))[thr_offset / 4] = out4;
    }
  }
}

#define LAUNCH_MERGE_ATTN_STATES_KERNEL(LOOP_OVER_HEAD, OUTPUT_LSE)        \
  {                                                                        \
    merge_attn_states_kernel<LOOP_OVER_HEAD, OUTPUT_LSE><<<grid, block>>>( \
        output.data_ptr<float>(), output_lse_ptr,                          \
        prefix_output.data_ptr<float>(), prefix_lse.data_ptr<float>(),     \
        suffix_output.data_ptr<float>(), suffix_lse.data_ptr<float>(),     \
        NUM_TOKENS, NUM_HEADS, HEAD_SIZE);                                 \
  }

void merge_attn_states(
    torch::Tensor& output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    std::optional<torch::Tensor> output_lse,  // [NUM_HEADS, NUM_TOKENS]
    const torch::Tensor& prefix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const torch::Tensor& prefix_lse,     // [NUM_HEADS, NUM_TOKENS]
    const torch::Tensor& suffix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const torch::Tensor& suffix_lse,     // [NUM_HEADS, NUM_TOKENS]
    const bool disable_loop_over_head) {
  const uint NUM_TOKENS = output.size(0);
  const uint NUM_HEADS = output.size(1);  // num query heads
  const uint HEAD_SIZE = output.size(2);
  assert(HEAD_SIZE % 4 == 0);     // headsize must be multiple of 4
  assert(HEAD_SIZE / 4 <= 1024);  // headsize must be <= of 4096
  float* output_lse_ptr = nullptr;
  if (output_lse.has_value()) {
    output_lse_ptr = output_lse.value().data_ptr<float>();
  }

  if (NUM_TOKENS <= 1024 || NUM_HEADS >= 64 || disable_loop_over_head) {
    dim3 grid(NUM_TOKENS, NUM_HEADS);
    dim3 block(HEAD_SIZE / 4);
    if (output_lse_ptr != nullptr) {
      LAUNCH_MERGE_ATTN_STATES_KERNEL(false, true);
    } else {
      LAUNCH_MERGE_ATTN_STATES_KERNEL(false, false);
    }
  } else {
    // try loop over num heads for large NUM_TOKENS
    dim3 grid(NUM_TOKENS);
    dim3 block(HEAD_SIZE / 4);
    if (output_lse_ptr != nullptr) {
      LAUNCH_MERGE_ATTN_STATES_KERNEL(true, true);
    } else {
      LAUNCH_MERGE_ATTN_STATES_KERNEL(true, false);
    }
  }
}
