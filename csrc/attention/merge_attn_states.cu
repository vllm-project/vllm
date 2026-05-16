#include <optional>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdint>
#include <limits>

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
    const float* suffix_lse, const uint num_tokens, const uint num_heads,
    const uint head_size, const uint prefix_head_stride,
    const uint output_head_stride, const uint prefix_num_tokens,
    const float* output_scale) {
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

  // If token_idx >= prefix_num_tokens, just copy from suffix
  if (token_idx >= prefix_num_tokens) {
    if (pack_offset < head_size) {
      input_pack_t s_out_pack = reinterpret_cast<const input_pack_t*>(
          suffix_head_ptr)[pack_offset / pack_size];

      if constexpr (USE_FP8_OUTPUT) {
        output_t o_out_pack[pack_size];
#pragma unroll
        for (uint i = 0; i < pack_size; ++i) {
          const float val =
              vllm::to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
          o_out_pack[i] =
              vllm::scaled_fp8_conversion<true, output_t>(val, fp8_scale_inv);
        }
        reinterpret_cast<output_pack_t*>(
            output_head_ptr)[pack_offset / pack_size] =
            *reinterpret_cast<output_pack_t*>(o_out_pack);
      } else {
        reinterpret_cast<output_pack_t*>(
            output_head_ptr)[pack_offset / pack_size] = s_out_pack;
      }
    }
    if (output_lse != nullptr && pack_idx == 0) {
      float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
      output_lse[head_idx * num_tokens + token_idx] = s_lse;
    }
    return;
  }

  // For tokens within prefix range, merge prefix and suffix
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

// Per-token-per-group FP8 merge kernel.
//
// Each thread handles `pack_size` elements (= 16 / sizeof(scalar_t), e.g. 8 for
// bf16) and `THREADS_PER_GROUP = GROUP_SIZE / pack_size` threads cooperate to
// quantize one group. Threads within a group share an absmax via warp shuffle,
// the lowest-id thread in the group writes the FP8 scale into
// output_block_scale (using caller-provided strides so any of row-major /
// col-major / TMA-aligned layouts work), and every thread writes its own
// quantized FP8 pack.
template <typename scalar_t, typename fp8_t, const uint NUM_THREADS,
          const int GROUP_SIZE, bool USE_UE8M0>
__global__ void merge_attn_states_group_fp8_kernel(
    fp8_t* output, float* output_lse, const scalar_t* prefix_output,
    const float* prefix_lse, const scalar_t* suffix_output,
    const float* suffix_lse, const uint num_tokens, const uint num_heads,
    const uint head_size, const uint prefix_head_stride,
    const uint output_head_stride, const uint prefix_num_tokens,
    float* output_block_scale, const int64_t sf_token_stride,
    const int64_t sf_group_stride) {
  using input_pack_t = uint4;  // 128-bit load (matches existing kernel)
  using output_pack_t = std::conditional_t<sizeof(scalar_t) == 4, uint, uint2>;

  constexpr uint pack_size = 16 / sizeof(scalar_t);
  constexpr uint threads_per_group = GROUP_SIZE / pack_size;
  static_assert(threads_per_group > 0 && threads_per_group <= 32 &&
                    (threads_per_group & (threads_per_group - 1)) == 0,
                "GROUP_SIZE/pack_size must be a power-of-2 in [1, 32]");
  // Full warp mask: butterfly shuffles with offset < threads_per_group stay
  // within each threads_per_group-aligned chunk of lanes.
  constexpr uint shfl_mask = 0xffffffffu;
  const uint threads_per_head = head_size / pack_size;
  const uint groups_per_head = head_size / GROUP_SIZE;

  const uint global_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;
  if (global_idx >= token_head_threads) return;

  const uint token_head_idx = global_idx / threads_per_head;
  const uint pack_idx = global_idx % threads_per_head;
  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  const uint pack_offset = pack_idx * pack_size;
  const uint group_idx_in_head = pack_offset / GROUP_SIZE;
  const uint global_group_idx = head_idx * groups_per_head + group_idx_in_head;
  const uint thread_in_group = threadIdx.x % threads_per_group;

  const uint src_head_offset = token_idx * num_heads * prefix_head_stride +
                               head_idx * prefix_head_stride;
  const uint dst_head_offset = token_idx * num_heads * output_head_stride +
                               head_idx * output_head_stride;
  const scalar_t* prefix_head_ptr = prefix_output + src_head_offset;
  const scalar_t* suffix_head_ptr = suffix_output + src_head_offset;
  fp8_t* output_head_ptr = output + dst_head_offset;

  // Stage 1: compute fp32 merged values for this thread's pack.
  float merged_f[pack_size];

  if (token_idx >= prefix_num_tokens) {
    input_pack_t s_pack = reinterpret_cast<const input_pack_t*>(
        suffix_head_ptr)[pack_offset / pack_size];
#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
      merged_f[i] =
          vllm::to_float(reinterpret_cast<const scalar_t*>(&s_pack)[i]);
    }
    if (output_lse != nullptr && pack_idx == 0) {
      output_lse[head_idx * num_tokens + token_idx] =
          suffix_lse[head_idx * num_tokens + token_idx];
    }
  } else {
    float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
    float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
    p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
    s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;
    const float max_lse = fmaxf(p_lse, s_lse);

    if (std::isinf(max_lse)) {
      input_pack_t p_pack = reinterpret_cast<const input_pack_t*>(
          prefix_head_ptr)[pack_offset / pack_size];
#pragma unroll
      for (uint i = 0; i < pack_size; ++i) {
        merged_f[i] =
            vllm::to_float(reinterpret_cast<const scalar_t*>(&p_pack)[i]);
      }
      if (output_lse != nullptr && pack_idx == 0) {
        output_lse[head_idx * num_tokens + token_idx] = max_lse;
      }
    } else {
      const float p_lse_n = p_lse - max_lse;
      const float s_lse_n = s_lse - max_lse;
      const float p_se = expf(p_lse_n);
      const float s_se = expf(s_lse_n);
      const float out_se = p_se + s_se;
      const float p_scale = p_se / out_se;
      const float s_scale = s_se / out_se;

      input_pack_t p_pack = reinterpret_cast<const input_pack_t*>(
          prefix_head_ptr)[pack_offset / pack_size];
      input_pack_t s_pack = reinterpret_cast<const input_pack_t*>(
          suffix_head_ptr)[pack_offset / pack_size];
#pragma unroll
      for (uint i = 0; i < pack_size; ++i) {
        const float pf =
            vllm::to_float(reinterpret_cast<const scalar_t*>(&p_pack)[i]);
        const float sf =
            vllm::to_float(reinterpret_cast<const scalar_t*>(&s_pack)[i]);
        merged_f[i] = pf * p_scale + sf * s_scale;
      }
      if (output_lse != nullptr && pack_idx == 0) {
        output_lse[head_idx * num_tokens + token_idx] = logf(out_se) + max_lse;
      }
    }
  }

  // Stage 2: per-thread absmax + intra-group reduction.
  float local_max = 0.0f;
#pragma unroll
  for (uint i = 0; i < pack_size; ++i) {
    local_max = fmaxf(local_max, fabsf(merged_f[i]));
  }
#pragma unroll
  for (uint offset = threads_per_group / 2; offset > 0; offset /= 2) {
    local_max = fmaxf(local_max, __shfl_xor_sync(shfl_mask, local_max, offset));
  }

  // Stage 3: derive scale (optionally UE8M0) and write SF.
  constexpr float eps = 1e-10f;
  const float fp8_max = quant_type_max_v<fp8_t>;
  float scale_raw = fmaxf(local_max, eps) / fp8_max;
  float scale;
  if constexpr (USE_UE8M0) {
    scale = exp2f(ceilf(log2f(scale_raw)));
  } else {
    scale = scale_raw;
  }
  if (thread_in_group == 0) {
    output_block_scale[static_cast<int64_t>(token_idx) * sf_token_stride +
                       static_cast<int64_t>(global_group_idx) *
                           sf_group_stride] = scale;
  }

  // Stage 4: quantize + store.
  const float inv_scale = 1.0f / scale;
  output_pack_t out_pack;
  fp8_t* out_ptr = reinterpret_cast<fp8_t*>(&out_pack);
#pragma unroll
  for (uint i = 0; i < pack_size; ++i) {
    out_ptr[i] =
        vllm::scaled_fp8_conversion<true, fp8_t>(merged_f[i], inv_scale);
  }
  reinterpret_cast<output_pack_t*>(output_head_ptr)[pack_offset / pack_size] =
      out_pack;
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

#define LAUNCH_MERGE_ATTN_STATES(scalar_t, output_t, NUM_THREADS,           \
                                 USE_FP8_OUTPUT)                            \
  {                                                                         \
    vllm::merge_attn_states_kernel<scalar_t, output_t, NUM_THREADS,         \
                                   USE_FP8_OUTPUT>                          \
        <<<grid, block, 0, stream>>>(                                       \
            reinterpret_cast<output_t*>(output.data_ptr()), output_lse_ptr, \
            reinterpret_cast<scalar_t*>(prefix_output.data_ptr()),          \
            reinterpret_cast<float*>(prefix_lse.data_ptr()),                \
            reinterpret_cast<scalar_t*>(suffix_output.data_ptr()),          \
            reinterpret_cast<float*>(suffix_lse.data_ptr()), num_tokens,    \
            num_heads, head_size, prefix_head_stride, output_head_stride,   \
            prefix_num_tokens, output_scale_ptr);                           \
  }

#define LAUNCH_MERGE_ATTN_STATES_GROUP_FP8(scalar_t, fp8_t, NUM_THREADS,   \
                                           GROUP_SIZE, USE_UE8M0)          \
  {                                                                        \
    vllm::merge_attn_states_group_fp8_kernel<scalar_t, fp8_t, NUM_THREADS, \
                                             GROUP_SIZE, USE_UE8M0>        \
        <<<grid, block, 0, stream>>>(                                      \
            reinterpret_cast<fp8_t*>(output.data_ptr()), output_lse_ptr,   \
            reinterpret_cast<scalar_t*>(prefix_output.data_ptr()),         \
            reinterpret_cast<float*>(prefix_lse.data_ptr()),               \
            reinterpret_cast<scalar_t*>(suffix_output.data_ptr()),         \
            reinterpret_cast<float*>(suffix_lse.data_ptr()), num_tokens,   \
            num_heads, head_size, prefix_head_stride, output_head_stride,  \
            prefix_num_tokens, output_block_scale_ptr, sf_token_stride,    \
            sf_group_stride);                                              \
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
 * @param prefill_tokens_with_context Number of prefill tokens with context
 * For the first p tokens (0 <= token_idx < prefill_tokens_with_context), output
 * is computed by merging prefix_output and suffix_output. For remaining tokens
 * (prefill_tokens_with_context <= token_idx < n), output is copied directly
 * from suffix_output.
 * @param output_scale Optional scalar tensor for FP8 static quantization.
 * When provided, output must be FP8 dtype.
 */
template <typename scalar_t>
void merge_attn_states_launcher(
    torch::Tensor& output, std::optional<torch::Tensor> output_lse,
    const torch::Tensor& prefix_output, const torch::Tensor& prefix_lse,
    const torch::Tensor& suffix_output, const torch::Tensor& suffix_lse,
    const std::optional<int64_t> prefill_tokens_with_context,
    const std::optional<torch::Tensor>& output_scale,
    const std::optional<torch::Tensor>& output_block_scale,
    const std::optional<int64_t> quant_group_size,
    const bool quant_scale_ue8m0) {
  constexpr uint NUM_THREADS = 128;
  const uint num_tokens = output.size(0);
  const uint num_heads = output.size(1);
  const uint head_size = output.size(2);
  const uint prefix_head_stride = prefix_output.stride(1);
  const uint output_head_stride = output.stride(1);
  // Thread mapping is based on input BF16 pack_size
  const uint pack_size = 16 / sizeof(scalar_t);
  TORCH_CHECK(head_size % pack_size == 0,
              "headsize must be multiple of pack_size:", pack_size);

  const uint prefix_num_tokens =
      prefill_tokens_with_context.has_value()
          ? static_cast<uint>(prefill_tokens_with_context.value())
          : num_tokens;
  TORCH_CHECK(prefix_num_tokens <= num_tokens,
              "prefix_num_tokens must be <= num_tokens");

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

  if (output_block_scale.has_value()) {
    TORCH_CHECK(!output_scale.has_value(),
                "output_scale must be None when output_block_scale is set");
    TORCH_CHECK(quant_group_size.has_value(),
                "quant_group_size is required for per-group FP8 output");
    const int group_size = static_cast<int>(quant_group_size.value());
    TORCH_CHECK(head_size % group_size == 0, "head_size (", head_size,
                ") must be a multiple of quant_group_size (", group_size, ")");
    TORCH_CHECK(group_size % pack_size == 0, "quant_group_size (", group_size,
                ") must be a multiple of pack_size (", pack_size, ")");
    // Intra-group warp shuffle uses a full-warp mask, so every lane in a
    // warp must take the same early-return decision. We need the total
    // launched thread count to be a multiple of warpSize. (num_tokens can
    // compensate for an odd per-token count.)
    TORCH_CHECK(
        (total_threads % 32) == 0,
        "total_threads (num_tokens * num_heads * head_size / pack_size) must "
        "be a multiple of warpSize (32) for group FP8 merge; got ",
        total_threads);
    const torch::Tensor& sf = output_block_scale.value();
    TORCH_CHECK(sf.scalar_type() == at::ScalarType::Float,
                "output_block_scale must be FP32, got: ", sf.scalar_type());
    TORCH_CHECK(sf.dim() == 2,
                "output_block_scale must be 2D [num_tokens, num_groups]");
    const int64_t sf_token_stride = sf.stride(0);
    const int64_t sf_group_stride = sf.stride(1);
    float* output_block_scale_ptr = sf.data_ptr<float>();

#define LAUNCH_MERGE_GROUP_FP8(GROUP_SIZE)                                 \
  if (quant_scale_ue8m0) {                                                 \
    VLLM_DISPATCH_FP8_TYPES(                                               \
        output.scalar_type(), "merge_attn_states_group_fp8", [&] {         \
          LAUNCH_MERGE_ATTN_STATES_GROUP_FP8(scalar_t, fp8_t, NUM_THREADS, \
                                             GROUP_SIZE, true);            \
        });                                                                \
  } else {                                                                 \
    VLLM_DISPATCH_FP8_TYPES(                                               \
        output.scalar_type(), "merge_attn_states_group_fp8", [&] {         \
          LAUNCH_MERGE_ATTN_STATES_GROUP_FP8(scalar_t, fp8_t, NUM_THREADS, \
                                             GROUP_SIZE, false);           \
        });                                                                \
  }

    if (group_size == 128) {
      LAUNCH_MERGE_GROUP_FP8(128);
    } else if (group_size == 64) {
      LAUNCH_MERGE_GROUP_FP8(64);
    } else {
      TORCH_CHECK(false, "Unsupported quant_group_size for group FP8 merge: ",
                  group_size, ". Supported: 64, 128");
    }
#undef LAUNCH_MERGE_GROUP_FP8
  } else if (output_scale.has_value()) {
    // FP8 output path - dispatch on output FP8 type
    VLLM_DISPATCH_FP8_TYPES(output.scalar_type(), "merge_attn_states_fp8", [&] {
      LAUNCH_MERGE_ATTN_STATES(scalar_t, fp8_t, NUM_THREADS, true);
    });
  } else {
    // Original BF16/FP16/FP32 output path
    LAUNCH_MERGE_ATTN_STATES(scalar_t, scalar_t, NUM_THREADS, false);
  }
}

#define CALL_MERGE_ATTN_STATES_LAUNCHER(scalar_t)                     \
  {                                                                   \
    merge_attn_states_launcher<scalar_t>(                             \
        output, output_lse, prefix_output, prefix_lse, suffix_output, \
        suffix_lse, prefill_tokens_with_context, output_scale,        \
        output_block_scale, quant_group_size, quant_scale_ue8m0);     \
  }

void merge_attn_states(torch::Tensor& output,
                       std::optional<torch::Tensor> output_lse,
                       const torch::Tensor& prefix_output,
                       const torch::Tensor& prefix_lse,
                       const torch::Tensor& suffix_output,
                       const torch::Tensor& suffix_lse,
                       std::optional<int64_t> prefill_tokens_with_context,
                       const std::optional<torch::Tensor>& output_scale,
                       const std::optional<torch::Tensor>& output_block_scale,
                       const std::optional<int64_t> quant_group_size,
                       const bool quant_scale_ue8m0) {
  if (output_scale.has_value() || output_block_scale.has_value()) {
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                    output.scalar_type() == at::ScalarType::Float8_e4m3fnuz,
                "output must be FP8 when output_scale or output_block_scale "
                "is provided, got: ",
                output.scalar_type());
  } else {
    TORCH_CHECK(output.scalar_type() == prefix_output.scalar_type(),
                "output dtype (", output.scalar_type(),
                ") must match prefix_output dtype (",
                prefix_output.scalar_type(), ") when no quant scale is set");
  }
  // Always dispatch on prefix_output (input) dtype
  DISPATCH_BY_SCALAR_DTYPE(prefix_output.dtype(),
                           CALL_MERGE_ATTN_STATES_LAUNCHER);
}
