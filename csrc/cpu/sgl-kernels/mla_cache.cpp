// vLLM-native CPU cache-write op for MLA's single-latent-buffer KV cache.
//
// `concat_and_cache_mla` (the generic MLA cache-write op used by every GPU
// backend) is registered CUDA-only. This is the CPU counterpart, adapted in
// spirit from SGLang's `store_cache_cpu` (csrc/cpu/kvcache.cpp) but
// generalized to write two source tensors (`kv_c_normed`, `k_pe`) into two
// different column-offset ranges of the SAME destination row -- SGLang's
// version assumes k/v land in two independent, equal-row-width cache
// tensors, which doesn't hold here since MLA's cache is one 576-wide buffer
// and the two column ranges (512-wide, 64-wide) don't match the buffer's
// true per-token stride, so the write can't reuse `store_cache_cpu` as-is.

#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ dst,
                      const scalar_t* __restrict__ src, int64_t size) {
  int64_t d = 0;
#if defined(CPU_CAPABILITY_AVX512)
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int64_t kVecSize = Vec::size();
  for (; d <= size - kVecSize; d += kVecSize) {
    Vec data = Vec::loadu(src + d);
    data.store(dst + d);
  }
#endif
  for (; d < size; ++d) {
    dst[d] = src[d];
  }
}

template <typename scalar_t, typename index_t>
void concat_and_cache_mla_kernel_impl(
    const scalar_t* __restrict__ kv_c_normed,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,         // [num_tokens, qk_rope_head_dim]
    scalar_t* __restrict__ kv_cache,  // [.., kv_lora_rank + qk_rope_head_dim]
    const index_t* __restrict__ slot_mapping,  // [num_tokens]
    int64_t num_tokens, int64_t kv_lora_rank, int64_t qk_rope_head_dim,
    int64_t kv_c_stride, int64_t k_pe_stride, int64_t cache_stride) {
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t slot = static_cast<int64_t>(slot_mapping[i]);
      if (slot < 0) {
        // padded/invalid token, matches `concat_and_cache_mla`'s semantics.
        continue;
      }
      scalar_t* __restrict__ cache_row = kv_cache + slot * cache_stride;
      copy_stub(cache_row, kv_c_normed + i * kv_c_stride, kv_lora_rank);
      copy_stub(cache_row + kv_lora_rank, k_pe + i * k_pe_stride,
                qk_rope_head_dim);
    }
  });
}

}  // namespace

// kv_c_normed : [num_tokens, kv_lora_rank]
// k_pe        : [num_tokens, qk_rope_head_dim] or [num_tokens, 1,
//               qk_rope_head_dim]
// kv_cache    : [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
// slot_mapping: [num_tokens] int32/int64, absolute physical row index
//               (block_id * block_size + block_offset); negative entries are
//               skipped (padded tokens), matching `concat_and_cache_mla`.
void concat_and_cache_mla_cpu(const at::Tensor& kv_c_normed,
                              const at::Tensor& k_pe, at::Tensor& kv_cache,
                              const at::Tensor& slot_mapping) {
  TORCH_CHECK(kv_c_normed.dim() == 2,
              "kv_c_normed must be 2D [num_tokens, kv_lora_rank]");
  TORCH_CHECK(k_pe.dim() == 2 || k_pe.dim() == 3, "k_pe must be 2D or 3D");
  TORCH_CHECK(kv_cache.dim() == 3,
              "kv_cache must be 3D [num_blocks, block_size, head_size]");
  TORCH_CHECK(kv_c_normed.stride(-1) == 1,
              "kv_c_normed innermost dim must be contiguous");
  TORCH_CHECK(k_pe.stride(-1) == 1, "k_pe innermost dim must be contiguous");
  TORCH_CHECK(kv_cache.stride(-1) == 1,
              "kv_cache innermost dim must be contiguous");

  int64_t num_tokens = kv_c_normed.size(0);
  int64_t kv_lora_rank = kv_c_normed.size(1);
  int64_t qk_rope_head_dim = k_pe.size(-1);
  int64_t head_size = kv_cache.size(-1);
  TORCH_CHECK(head_size == kv_lora_rank + qk_rope_head_dim,
              "kv_cache head_size must equal kv_lora_rank + qk_rope_head_dim");
  TORCH_CHECK(slot_mapping.size(0) == num_tokens, "slot_mapping size mismatch");

  // Real physical per-token stride of the cache, read from the tensor's own
  // strides rather than assumed to equal head_size (paged/pooled buffers are
  // not guaranteed to have zero inter-row padding in general, even though in
  // practice a freshly-allocated MLA cache is fully contiguous).
  int64_t cache_stride = kv_cache.stride(1);

  const auto dtype = kv_cache.scalar_type();
  TORCH_CHECK(dtype == kv_c_normed.scalar_type() && dtype == k_pe.scalar_type(),
              "concat_and_cache_mla_cpu: dtype mismatch");
  const auto index_dtype = slot_mapping.scalar_type();
  TORCH_CHECK(index_dtype == at::kLong || index_dtype == at::kInt,
              "slot_mapping must be int32 or int64");

  AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "concat_and_cache_mla_cpu", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "concat_and_cache_mla_cpu_index", [&] {
      concat_and_cache_mla_kernel_impl<scalar_t, index_t>(
          kv_c_normed.data_ptr<scalar_t>(), k_pe.data_ptr<scalar_t>(),
          kv_cache.data_ptr<scalar_t>(), slot_mapping.data_ptr<index_t>(),
          num_tokens, kv_lora_rank, qk_rope_head_dim, kv_c_normed.stride(0),
          k_pe.stride(0), cache_stride);
    });
  });
}
