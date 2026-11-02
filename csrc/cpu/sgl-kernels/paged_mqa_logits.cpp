// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu:
// `fp8_paged_mqa_logits_cpu`/`fp8_paged_mqa_logits_cpu_impl` (DECODE path).
//
// Reads the paged indexer K-cache directly via `page_table` -- no separate
// eager gather step -- and processes the whole decode batch (one query token
// per request; the CPU indexer decode path doesn't support native MTP/spec
// decode yet, see `sparse_attn_indexer_cpu`) in a single `at::parallel_for`
// flattening `(batch, token)`, exactly like upstream. AVX512 `dot_fp8_128`
// dequant-dot-product and the split K-region-then-scale-region page byte
// layout are verbatim (that part of the upstream contract already matches
// vLLM's cache, written by `compress_norm_rope_store_indexer_cpu` in
// compressor.cpp).
//
// Differences from upstream:
//  - `block_size` is a runtime argument instead of the hardcoded
//    `kBlockSize = 64`. vLLM's indexer K-cache shares the model's single
//    global `--block-size` KV-cache-group setting (confirmed empirically:
//    the DeepSeek-V4-Flash TP1/TP4 baseline configs run with
//    `block_size=256`, not 64 -- see `.agent-workspace/perf_baseline_tp*_
//    server.log`), so the upstream `kvcache_fp8.size(1) == 64` shape
//    contract does not hold in general.
//  - `q_fp8`/`weight` drop the redundant `S_q=1` dimension (`[batch, heads,
//    128]`/`[batch, heads]` instead of `[batch, 1, heads, 128]`) since S_q is
//    always 1 here; `weight` is fp32-only (already folds q_scale/softmax_
//    scale/head_scale, matching this call site's actual dtype -- unlike
//    upstream's bf16/fp16/fp32 dispatch, which vLLM's CPU indexer never
//    produces).
//  - `index_head_dim` (128) stays a compile-time constant, matching
//    upstream: it's `config.index_head_dim`, a genuine DeepSeek-V4 model
//    invariant, not a vLLM KV-cache config knob like `block_size`.

#include "common.h"
#include "vec.h"

#if !defined(CPU_CAPABILITY_AVX512)
static_assert(false,
              "paged_mqa_logits.cpp is only ever compiled into the "
              "AVX512+AMX-flagged _C target (see cmake/cpu_extension.cmake's "
              "VLLM_EXT_SRC_SGL) and relies on that unconditionally -- "
              "CPU_CAPABILITY_AVX512 must be defined here.");
#endif

namespace {

constexpr int64_t kIndexHeadDim = 128;
constexpr int64_t kIndexHeadDimWithScaleBytes = 132;

inline float dot_fp8_128(const uint8_t* k, const uint8_t* q) {
  __m512 acc = _mm512_setzero_ps();
  for (int64_t d = 0; d < kIndexHeadDim; d += 32) {
    const __m256i k8 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + d));
    const __m256i q8 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d));
    acc = _mm512_dpbf16_ps(acc, CVT_FP8_TO_BF16(k8), CVT_FP8_TO_BF16(q8));
  }
  return _mm512_reduce_add_ps(acc);
}

template <typename seq_t, typename page_t>
void fp8_paged_mqa_logits_cpu_impl(const at::Tensor& q_fp8,
                                   const at::Tensor& kvcache_fp8,
                                   const at::Tensor& weight,
                                   const at::Tensor& seq_lens,
                                   const at::Tensor& page_table,
                                   at::Tensor& logits, int64_t block_size,
                                   int64_t max_seq_len) {
  const int64_t batch_size = q_fp8.size(0);
  const int64_t num_heads = q_fp8.size(1);
  const int64_t num_pages = kvcache_fp8.size(0);
  // Row stride is threaded through explicitly rather than assumed equal to
  // the page byte width: the K-cache may be a page of a larger pooled
  // allocation shared across layers (matches store_cache.cpp/flash_mla.cpp's
  // precedent).
  const int64_t page_stride = kvcache_fp8.stride(0);
  const int64_t pages_per_batch = page_table.size(1);
  const int64_t scale_offset_bytes = block_size * kIndexHeadDim;

  const auto* q_ptr = reinterpret_cast<const uint8_t*>(q_fp8.const_data_ptr());
  const auto* cache_ptr =
      reinterpret_cast<const uint8_t*>(kvcache_fp8.const_data_ptr());
  const auto* weight_ptr = weight.const_data_ptr<float>();
  const auto* seq_ptr = seq_lens.const_data_ptr<seq_t>();
  const auto* page_ptr = page_table.const_data_ptr<page_t>();
  auto* out_ptr = logits.data_ptr<float>();

  at::parallel_for(
      0, batch_size * max_seq_len, GRAIN_SIZE / kIndexHeadDim,
      [&](int64_t begin, int64_t end) {
        int64_t b{0}, token{0};
        data_index_init(begin, b, batch_size, token, max_seq_len);
        for (int64_t i = begin; i < end; ++i) {
          const int64_t seq_len = static_cast<int64_t>(seq_ptr[b]);

          if (token >= seq_len) {
            data_index_step(b, batch_size, token, max_seq_len);
            continue;
          }

          const int64_t q_batch_offset = b * num_heads * kIndexHeadDim;
          const int64_t weight_batch_offset = b * num_heads;
          const int64_t page_batch_offset = b * pages_per_batch;
          float* out_row = out_ptr + b * max_seq_len;

          const int64_t logical_page = token / block_size;
          const int64_t token_in_page = token % block_size;

          const int64_t physical_page =
              static_cast<int64_t>(page_ptr[page_batch_offset + logical_page]);

          const uint8_t* block = cache_ptr + physical_page * page_stride;
          const uint8_t* k_token = block + token_in_page * kIndexHeadDim;
          const float* scale_ptr =
              reinterpret_cast<const float*>(block + scale_offset_bytes);
          const float k_scale = scale_ptr[token_in_page];

          float score_sum = 0.0f;
          for (int64_t h = 0; h < num_heads; ++h) {
            const uint8_t* q_head = q_ptr + q_batch_offset + h * kIndexHeadDim;
            float dot = dot_fp8_128(k_token, q_head);
            dot = std::max(dot, 0.0f);
            score_sum += dot * weight_ptr[weight_batch_offset + h];
          }

          out_row[token] = score_sum * k_scale;
          data_index_step(b, batch_size, token, max_seq_len);
        }
      });
}

template <typename seq_t>
void dispatch_page_type(const at::Tensor& q_fp8, const at::Tensor& kvcache_fp8,
                        const at::Tensor& weight, const at::Tensor& seq_lens,
                        const at::Tensor& page_table, at::Tensor& logits,
                        int64_t block_size, int64_t max_seq_len) {
  if (page_table.scalar_type() == at::kInt) {
    fp8_paged_mqa_logits_cpu_impl<seq_t, int32_t>(q_fp8, kvcache_fp8, weight,
                                                  seq_lens, page_table, logits,
                                                  block_size, max_seq_len);
  } else if (page_table.scalar_type() == at::kLong) {
    fp8_paged_mqa_logits_cpu_impl<seq_t, int64_t>(q_fp8, kvcache_fp8, weight,
                                                  seq_lens, page_table, logits,
                                                  block_size, max_seq_len);
  } else {
    TORCH_CHECK(false, "page_table must be int32 or int64");
  }
}

}  // namespace

at::Tensor fp8_paged_mqa_logits_cpu(at::Tensor& q_fp8, at::Tensor& kvcache_fp8,
                                    at::Tensor& weight, at::Tensor& seq_lens,
                                    at::Tensor& page_table, int64_t block_size,
                                    int64_t max_seq_len) {
  CHECK_INPUT(q_fp8);
  CHECK_CPU(kvcache_fp8);
  CHECK_INPUT(weight);
  CHECK_INPUT(seq_lens);
  CHECK_INPUT(page_table);
  TORCH_CHECK(
      q_fp8.dim() == 3 && q_fp8.scalar_type() == at::kFloat8_e4m3fn,
      "fp8_paged_mqa_logits_cpu: q_fp8 must be a 3D float8_e4m3fn tensor "
      "[batch, heads, 128]");
  TORCH_CHECK(q_fp8.size(2) == kIndexHeadDim,
              "fp8_paged_mqa_logits_cpu: q_fp8 head_dim must be 128");
  TORCH_CHECK(kvcache_fp8.dim() == 2 && kvcache_fp8.scalar_type() == at::kByte,
              "fp8_paged_mqa_logits_cpu: kvcache_fp8 must be a 2D uint8 tensor "
              "[num_pages, buf_numel_per_page]");
  TORCH_CHECK(kvcache_fp8.stride(1) == 1 &&
                  kvcache_fp8.stride(0) >= kvcache_fp8.size(1),
              "fp8_paged_mqa_logits_cpu: kvcache_fp8 must be contiguous "
              "within a page (may be a page of a larger pooled allocation "
              "between pages)");
  TORCH_CHECK(block_size > 0,
              "fp8_paged_mqa_logits_cpu: block_size must be positive");
  TORCH_CHECK(
      kvcache_fp8.size(1) >= block_size * kIndexHeadDimWithScaleBytes,
      "fp8_paged_mqa_logits_cpu: kvcache_fp8 page byte width too small for "
      "block_size");

  const int64_t batch_size = q_fp8.size(0);
  const int64_t num_heads = q_fp8.size(1);
  TORCH_CHECK(
      weight.dim() == 2 && weight.scalar_type() == at::kFloat &&
          weight.size(0) == batch_size && weight.size(1) == num_heads,
      "fp8_paged_mqa_logits_cpu: weight must be a 2D fp32 tensor [batch, "
      "heads]");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == batch_size,
              "fp8_paged_mqa_logits_cpu: seq_lens must be a 1D tensor [batch]");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) == batch_size,
              "fp8_paged_mqa_logits_cpu: page_table must be a 2D tensor "
              "[batch, pages]");
  TORCH_CHECK(max_seq_len >= 0,
              "fp8_paged_mqa_logits_cpu: max_seq_len must be non-negative");

  // Positions with token >= seq_len are never written by the kernel below,
  // and topk_transform_512_cpu (the only consumer) never reads them either
  // -- left uninitialized, matching upstream.
  auto logits =
      at::empty({batch_size, max_seq_len}, q_fp8.options().dtype(at::kFloat));
  if (batch_size == 0 || max_seq_len == 0) {
    return logits;
  }

  if (seq_lens.scalar_type() == at::kInt) {
    dispatch_page_type<int32_t>(q_fp8, kvcache_fp8, weight, seq_lens,
                                page_table, logits, block_size, max_seq_len);
  } else if (seq_lens.scalar_type() == at::kLong) {
    dispatch_page_type<int64_t>(q_fp8, kvcache_fp8, weight, seq_lens,
                                page_table, logits, block_size, max_seq_len);
  } else {
    TORCH_CHECK(false, "seq_lens must be int32 or int64");
  }

  return logits;
}
