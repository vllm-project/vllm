#include "torch_utils.h"
#include "dispatch_utils.h"

#include "../cuda_compat.h"

#include "../quantization/w8a8/fp8/common.cuh"
#ifdef USE_ROCM
  #include "../quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

namespace vllm {

// Fused RoPE + reshape_and_cache_flash kernel.
//
// One thread block per token. The block first rotates query and key in place
// using cos/sin loaded from cos_sin_cache, then (if slot_idx != -1) writes the
// rotated key and the value tensor into the paged KV cache (flash NHD layout:
// [num_blocks, block_size, num_kv_heads, head_size]).
//
// raw_kv_scalar_t is the integer backing for qk_t (uint16_t for fp16/bf16),
// matching the convention used in concat_and_cache_mla_rope_fused_kernel.
//
// Why fp32 intermediates for the RoPE math:
// Matches apply_token_rotary_embedding's upcast -> multiply -> downcast
// convention (csrc/libtorch_stable/pos_encoding_kernels.cu). The MLA fused
// kernel does the rotation in bf16 and accumulates ~3e-2 of rounding error
// vs the existing rotary kernel; we deliberately don't mirror that.
//
// Why scalar writes for Phase 3 (instead of vectorize_with_alignment as in
// reshape_and_cache_flash_kernel):
// Empirically, the helper's inlined template machinery raised register
// pressure enough to regress small-N cases by 30-60% on H200. Scalar writes
// give a flatter curve. Vectorization would be welcome as a follow-up if
// profiling shows headroom.
//
// Bit-identical to the unfused path:
// RoPE is per-element and the cache write is per-element scatter -- there
// are no reductions whose order could change between the fused and unfused
// paths, so each output element comes from the same arithmetic sequence in
// both. The parameterized correctness test in
// tests/kernels/attention/test_fused_rope_kvcache.py asserts rtol=0, atol=0.
template <typename qk_t, typename cos_sin_t, bool IS_NEOX,
          typename raw_kv_scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void fused_rope_and_reshape_cache_flash_kernel(
    const int64_t* __restrict__ positions,  // [num_padded_tokens]
    qk_t* __restrict__ query,  // [num_padded_tokens, num_q_heads, head_size]
    qk_t* __restrict__ key,    // [num_padded_tokens, num_kv_heads, head_size]
    const qk_t* __restrict__ value,  // [num_padded_tokens, num_kv_heads,
                                     // head_size]
    const cos_sin_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    cache_t* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads,
                                      // head_size]
    cache_t* __restrict__ value_cache,         //   same shape
    const int64_t* __restrict__ slot_mapping,  // [num_actual_tokens]
    const float* __restrict__ k_scale,         // [1]
    const float* __restrict__ v_scale,         // [1]
    const int rot_dim, const int64_t query_stride_token,
    const int64_t key_stride_token, const int64_t value_stride_token,
    const int64_t cache_block_stride, const int64_t cache_page_stride,
    const int num_q_heads, const int num_kv_heads, const int head_size,
    const int block_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t pos = positions[token_idx];

  const cos_sin_t* cos_sin_ptr = cos_sin_cache + pos * rot_dim;
  const int embed_dim = rot_dim / 2;

  // Phase 1: in-place RoPE on Q. Math is done in fp32 to match the precision
  // of vllm::rotary_embedding_kernel (apply_token_rotary_embedding upcasts
  // cos/sin/x/y to float, multiplies, and downcasts to scalar_t on store).
  const int nq = num_q_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int pair_idx = i % embed_dim;

    const float cos_f = static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx));
    const float sin_f =
        static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim));

    qk_t* q_head_ptr =
        query + token_idx * query_stride_token + head_idx * head_size;

    int idx_x, idx_y;
    if constexpr (IS_NEOX) {
      idx_x = pair_idx;
      idx_y = embed_dim + pair_idx;
    } else {
      idx_x = pair_idx * 2;
      idx_y = pair_idx * 2 + 1;
    }

    const float x_f = static_cast<float>(q_head_ptr[idx_x]);
    const float y_f = static_cast<float>(q_head_ptr[idx_y]);
    q_head_ptr[idx_x] = static_cast<qk_t>(x_f * cos_f - y_f * sin_f);
    q_head_ptr[idx_y] = static_cast<qk_t>(y_f * cos_f + x_f * sin_f);
  }

  // Phase 2: in-place RoPE on K (same precision treatment as Phase 1).
  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int pair_idx = i % embed_dim;

    const float cos_f = static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx));
    const float sin_f =
        static_cast<float>(VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim));

    qk_t* k_head_ptr =
        key + token_idx * key_stride_token + head_idx * head_size;

    int idx_x, idx_y;
    if constexpr (IS_NEOX) {
      idx_x = pair_idx;
      idx_y = embed_dim + pair_idx;
    } else {
      idx_x = pair_idx * 2;
      idx_y = pair_idx * 2 + 1;
    }

    const float x_f = static_cast<float>(k_head_ptr[idx_x]);
    const float y_f = static_cast<float>(k_head_ptr[idx_y]);
    k_head_ptr[idx_x] = static_cast<qk_t>(x_f * cos_f - y_f * sin_f);
    k_head_ptr[idx_y] = static_cast<qk_t>(y_f * cos_f + x_f * sin_f);
  }

  // K writes from Phase 2 must be visible to Phase 3 readers in this block.
  __syncthreads();

  // slot_idx < 0 means "padded token, do not write to cache" -- Phase 1/2
  // above still rotated query/key for the padded slot, which matches the
  // unfused path (rotary_embedding runs over all tokens, then
  // reshape_and_cache_flash skips writes for negative slots).
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t page_offset = slot_idx % block_size;

  cache_t* k_dst = key_cache + block_idx * cache_block_stride +
                   page_offset * cache_page_stride;
  cache_t* v_dst = value_cache + block_idx * cache_block_stride +
                   page_offset * cache_page_stride;

  const qk_t* k_src = key + token_idx * key_stride_token;
  const qk_t* v_src = value + token_idx * value_stride_token;

  const int n_elems = num_kv_heads * head_size;

  // Scalar writes. Earlier experiments with vectorize_with_alignment regressed
  // small-N cases (kernel-launch bound) by ~30-60% while only marginally
  // helping MHA N=2048 fp8. Kept scalar for now; revisit if profiling shows
  // HBM-bandwidth headroom we can recover.
  if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
    for (int i = threadIdx.x; i < n_elems; i += blockDim.x) {
      const raw_kv_scalar_t k_raw =
          *reinterpret_cast<const raw_kv_scalar_t*>(k_src + i);
      const raw_kv_scalar_t v_raw =
          *reinterpret_cast<const raw_kv_scalar_t*>(v_src + i);
      *reinterpret_cast<raw_kv_scalar_t*>(k_dst + i) = k_raw;
      *reinterpret_cast<raw_kv_scalar_t*>(v_dst + i) = v_raw;
    }
  } else {
    const float k_scale_val = *k_scale;
    const float v_scale_val = *v_scale;
    for (int i = threadIdx.x; i < n_elems; i += blockDim.x) {
      const raw_kv_scalar_t k_raw =
          *reinterpret_cast<const raw_kv_scalar_t*>(k_src + i);
      const raw_kv_scalar_t v_raw =
          *reinterpret_cast<const raw_kv_scalar_t*>(v_src + i);
      k_dst[i] = fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
          k_raw, k_scale_val);
      v_dst[i] = fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
          v_raw, v_scale_val);
    }
  }
}

}  // namespace vllm

// KV_T  = raw integer type backing qk_t (uint16_t for fp16/bf16)
// CACHE_T = stored cache element type
// KV_DTYPE = Fp8KVCacheDataType enum value
#define CALL_FUSED_ROPE_AND_RESHAPE_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)        \
  do {                                                                          \
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                        \
        query.scalar_type(), "qk_scalar_type", [&] {                            \
          using qk_t = scalar_t;                                                \
          VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                  \
              cos_sin_cache.scalar_type(), "cos_sin_cache_scalar_type", [&] {   \
                using cos_sin_t = scalar_t;                                     \
                if (is_neox) {                                                  \
                  vllm::fused_rope_and_reshape_cache_flash_kernel<              \
                      qk_t, cos_sin_t, true, KV_T, CACHE_T, KV_DTYPE>           \
                      <<<grid, block, 0, stream>>>(                             \
                          positions.const_data_ptr<int64_t>(),                  \
                          query.mutable_data_ptr<qk_t>(),                       \
                          key.mutable_data_ptr<qk_t>(),                         \
                          value.const_data_ptr<qk_t>(),                         \
                          cos_sin_cache.const_data_ptr<cos_sin_t>(),            \
                          reinterpret_cast<CACHE_T*>(                           \
                              key_cache.mutable_data_ptr()),                    \
                          reinterpret_cast<CACHE_T*>(                           \
                              value_cache.mutable_data_ptr()),                  \
                          slot_mapping.const_data_ptr<int64_t>(),               \
                          k_scale.const_data_ptr<float>(),                      \
                          v_scale.const_data_ptr<float>(), rot_dim,             \
                          query_stride_token, key_stride_token,                 \
                          value_stride_token, cache_block_stride,               \
                          cache_page_stride, num_q_heads, num_kv_heads,         \
                          head_size, block_size);                               \
                } else {                                                        \
                  vllm::fused_rope_and_reshape_cache_flash_kernel<              \
                      qk_t, cos_sin_t, false, KV_T, CACHE_T, KV_DTYPE>          \
                      <<<grid, block, 0, stream>>>(                             \
                          positions.const_data_ptr<int64_t>(),                  \
                          query.mutable_data_ptr<qk_t>(),                       \
                          key.mutable_data_ptr<qk_t>(),                         \
                          value.const_data_ptr<qk_t>(),                         \
                          cos_sin_cache.const_data_ptr<cos_sin_t>(),            \
                          reinterpret_cast<CACHE_T*>(                           \
                              key_cache.mutable_data_ptr()),                    \
                          reinterpret_cast<CACHE_T*>(                           \
                              value_cache.mutable_data_ptr()),                  \
                          slot_mapping.const_data_ptr<int64_t>(),               \
                          k_scale.const_data_ptr<float>(),                      \
                          v_scale.const_data_ptr<float>(), rot_dim,             \
                          query_stride_token, key_stride_token,                 \
                          value_stride_token, cache_block_stride,               \
                          cache_page_stride, num_q_heads, num_kv_heads,         \
                          head_size, block_size);                               \
                }                                                               \
              });                                                               \
        });                                                                     \
  } while (false)

// Replaces a back-to-back `rotary_embedding(positions, q, k, ...)` followed
// by `reshape_and_cache_flash(k, v, key_cache, value_cache, slot_mapping,
// ...)`. query and key are modified in place.
//
// This PR supports the flash NHD cache layout
// ([num_blocks, block_size, num_kv_heads, head_size]) with scalar per-tensor
// k_scale/v_scale and kv_cache_dtype in {auto, fp8/fp8_e4m3, fp8_e5m2}.
// NVFP4 cache, per-token-head FP8 quant, and the HND layout fall back to the
// unfused path via FlashAttentionImpl.fused_rope_kvcache_supported().
void fused_rope_and_reshape_cache_flash(
    torch::stable::Tensor& query,    // [num_tokens, num_q_heads, head_size]
    torch::stable::Tensor& key,      // [num_tokens, num_kv_heads, head_size]
    torch::stable::Tensor& value,    // [num_tokens, num_kv_heads, head_size]
    torch::stable::Tensor& positions,      // [num_tokens]
    torch::stable::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox,
    torch::stable::Tensor&
        key_cache,  // [num_blocks, block_size, num_kv_heads, head_size]
    torch::stable::Tensor& value_cache,   //   same shape
    torch::stable::Tensor& slot_mapping,  // [num_actual_tokens]
    torch::stable::Tensor& k_scale,       // [1]
    torch::stable::Tensor& v_scale,       // [1]
    const std::string& kv_cache_dtype) {
  // V1 CUDA graphs pad query/key/value/positions; slot_mapping carries the
  // unpadded count, mirroring reshape_and_cache_flash semantics.
  const int64_t num_tokens = slot_mapping.size(0);
  const int64_t num_padded_tokens = query.size(0);
  STD_TORCH_CHECK(num_padded_tokens >= num_tokens);
  STD_TORCH_CHECK(key.size(0) >= num_tokens);
  STD_TORCH_CHECK(value.size(0) >= num_tokens);

  STD_TORCH_CHECK(query.dim() == 3);
  STD_TORCH_CHECK(key.dim() == 3);
  STD_TORCH_CHECK(value.dim() == 3);

  const int num_q_heads = query.size(1);
  const int num_kv_heads = key.size(1);
  const int head_size = query.size(2);
  const int rot_dim = cos_sin_cache.size(1);
  STD_TORCH_CHECK(key.size(2) == head_size);
  STD_TORCH_CHECK(value.size(2) == head_size);
  STD_TORCH_CHECK(value.size(1) == num_kv_heads);
  STD_TORCH_CHECK(rot_dim > 0);
  STD_TORCH_CHECK(rot_dim <= head_size);
  STD_TORCH_CHECK(rot_dim % 2 == 0);

  // The kernel assumes per-token rows of (q,k,v) are contiguous head-major:
  // stride(2) == 1, stride(1) == head_size. This is the flash NHD layout that
  // FlashAttentionImpl writes today.
  STD_TORCH_CHECK(query.stride(2) == 1);
  STD_TORCH_CHECK(query.stride(1) == head_size);
  STD_TORCH_CHECK(key.stride(2) == 1);
  STD_TORCH_CHECK(key.stride(1) == head_size);
  STD_TORCH_CHECK(value.stride(2) == 1);
  STD_TORCH_CHECK(value.stride(1) == head_size);

  STD_TORCH_CHECK(positions.dim() == 1);
  STD_TORCH_CHECK(positions.scalar_type() ==
                  torch::headeronly::ScalarType::Long);
  STD_TORCH_CHECK(positions.size(0) == num_padded_tokens);

  STD_TORCH_CHECK(slot_mapping.dim() == 1);
  STD_TORCH_CHECK(slot_mapping.scalar_type() ==
                  torch::headeronly::ScalarType::Long);

  STD_TORCH_CHECK(key_cache.dim() == 4);
  STD_TORCH_CHECK(value_cache.dim() == 4);
  STD_TORCH_CHECK(key_cache.size(3) == head_size);
  STD_TORCH_CHECK(key_cache.size(2) == num_kv_heads);
  STD_TORCH_CHECK(value_cache.size(3) == head_size);
  STD_TORCH_CHECK(value_cache.size(2) == num_kv_heads);
  STD_TORCH_CHECK(key_cache.stride(3) == 1);
  STD_TORCH_CHECK(key_cache.stride(2) == head_size);
  STD_TORCH_CHECK(key_cache.stride(1) ==
                  static_cast<int64_t>(num_kv_heads) * head_size);
  STD_TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));
  STD_TORCH_CHECK(key_cache.stride(1) == value_cache.stride(1));

  // This PR supports scalar per-tensor scales only; per-head FP8 scales are
  // out of scope.
  STD_TORCH_CHECK(k_scale.numel() == 1);
  STD_TORCH_CHECK(v_scale.numel() == 1);
  STD_TORCH_CHECK(k_scale.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(v_scale.scalar_type() ==
                  torch::headeronly::ScalarType::Float);

  const int block_size = key_cache.size(1);
  const int64_t query_stride_token = query.stride(0);
  const int64_t key_stride_token = key.stride(0);
  const int64_t value_stride_token = value.stride(0);
  const int64_t cache_block_stride = key_cache.stride(0);
  const int64_t cache_page_stride = key_cache.stride(1);

  const int embed_dim = rot_dim / 2;
  const int rope_work =
      std::max(num_q_heads * embed_dim, num_kv_heads * embed_dim);
  const int cache_work = num_kv_heads * head_size;
  const int thread_block_size = std::min(std::max(rope_work, cache_work), 512);

  dim3 grid(num_tokens, 1, 1);
  dim3 block(thread_block_size, 1, 1);

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  DISPATCH_BY_KV_CACHE_DTYPE(query.scalar_type(), kv_cache_dtype,
                             CALL_FUSED_ROPE_AND_RESHAPE_CACHE_FLASH);
}
