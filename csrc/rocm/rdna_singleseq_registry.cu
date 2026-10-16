// Torch ops over the RDNA3.5 attention kernels built into _rocm_C.
// rdna_singleseq_{decode,prefill}_configs list the builds of a kernel and
// rdna_singleseq_{decode,prefill}_attn launch one by its index.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <string>
#include <vector>

#include "rdna_singleseq_attn_launch.h"
#include "rdna_singleseq_decode_configs.h"
#include "rdna_singleseq_prefill_configs.h"

rdna_singleseq::LaunchFn* rdna_singleseq::decode_launchers() {
  static LaunchFn launchers[std::size(kDecodeConfigs)] = {};
  return launchers;
}

rdna_singleseq::PrefillFn* rdna_singleseq::prefill_launchers() {
  static PrefillFn launchers[std::size(kPrefillConfigs)] = {};
  return launchers;
}

namespace {

constexpr int64_t kNumDecodeConfigs = std::size(rdna_singleseq::kDecodeConfigs);
constexpr int64_t kNumPrefillConfigs =
    std::size(rdna_singleseq::kPrefillConfigs);

bool on_gfx1151() {
  const std::string arch = at::cuda::getCurrentDeviceProperties()->gcnArchName;
  return arch.find("gfx1151") != std::string::npos;
}

// The kernels expect (num_pages, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM).
void check_kv_cache(const torch::Tensor& kv_cache, int num_kv_heads,
                    int page_size, int head_dim) {
  const int64_t row = 2 * head_dim;
  TORCH_CHECK(kv_cache.dim() == 4 && kv_cache.size(1) == num_kv_heads &&
                  kv_cache.size(2) == page_size && kv_cache.size(3) == row &&
                  kv_cache.stride(3) == 1 && kv_cache.stride(2) == row &&
                  kv_cache.stride(1) == page_size * row &&
                  kv_cache.stride(0) == num_kv_heads * page_size * row,
              "kv_cache must be packed HND (pages, ", num_kv_heads, ", ",
              page_size, ", ", row, "), got sizes ", kv_cache.sizes(),
              " strides ", kv_cache.strides());
}

void check_device(const torch::Tensor& q,
                  std::initializer_list<const torch::Tensor*> others) {
  for (const torch::Tensor* t : others)
    TORCH_CHECK(t->device() == q.device(), "every tensor must be on ",
                q.device(), ", got one on ", t->device());
}

}  // namespace

// Every decode build, its DecodeConfig fields in order; none off gfx1151.
// Raises if a build has no instantiation unit.
std::vector<int64_t> rdna_singleseq_decode_configs() {
  std::vector<int64_t> fields;
  if (!on_gfx1151()) return fields;
  for (int64_t i = 0; i < kNumDecodeConfigs; ++i) {
    const rdna_singleseq::DecodeConfig& c = rdna_singleseq::kDecodeConfigs[i];
    TORCH_CHECK(rdna_singleseq::decode_launchers()[i], "decode build ", i,
                " (head dim ", static_cast<int>(c.head_dim), ", bf16 ",
                c.dtype == rdna_singleseq::Dtype::kBf16,
                ") has no unit in cmake/rocm_rdna_singleseq_attn.cmake");
    fields.insert(
        fields.end(),
        {static_cast<int>(c.head_dim), c.num_q_heads, c.num_kv_heads, c.window,
         c.max_query_len, static_cast<int>(c.dtype), c.page_size,
         c.max_segments, c.row_groups, c.min_segment_blocks, c.waves,
         c.head_dim_split, c.prefetch, c.v_in_lds, c.dot_product});
  }
  return fields;
}

// Every prefill build, its PrefillConfig fields in order; none off gfx1151.
// Raises if a build has no instantiation unit.
std::vector<int64_t> rdna_singleseq_prefill_configs() {
  std::vector<int64_t> fields;
  if (!on_gfx1151()) return fields;
  for (int64_t i = 0; i < kNumPrefillConfigs; ++i) {
    const rdna_singleseq::PrefillConfig& c = rdna_singleseq::kPrefillConfigs[i];
    TORCH_CHECK(rdna_singleseq::prefill_launchers()[i], "prefill build ", i,
                " (head dim ", static_cast<int>(c.head_dim), ", bf16 ",
                c.dtype == rdna_singleseq::Dtype::kBf16,
                ") has no unit in cmake/rocm_rdna_singleseq_attn.cmake");
    fields.insert(
        fields.end(),
        {static_cast<int>(c.head_dim), c.num_q_heads, c.num_kv_heads, c.window,
         static_cast<int>(c.dtype), c.page_size, c.min_query_len, c.waves,
         c.key_tile, c.head_dim_split, c.v_t, c.double_buffer, c.key_waves,
         c.max_segments, c.head_group});
  }
  return fields;
}

// The caller allocates the scratch: this runs under CUDA-graph capture.
void rdna_singleseq_decode_attn(int64_t config, torch::Tensor& q,
                                torch::Tensor& kv_cache,
                                torch::Tensor& block_table, torch::Tensor& out,
                                torch::Tensor& acc, torch::Tensor& m,
                                torch::Tensor& l, torch::Tensor& cnt,
                                torch::Tensor& seq_lens, double scale) {
  using namespace rdna_singleseq;
  TORCH_CHECK(
      config >= 0 && config < kNumDecodeConfigs && decode_launchers()[config],
      "no decode build ", config);
  const DecodeConfig& c = kDecodeConfigs[config];
  const int maxm = c.max_query_len;
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "q and out must be contiguous");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  (block_table.dim() == 1 || block_table.dim() == 2) &&
                  block_table.stride(-1) == 1,
              "block table must be int32 rows, contiguous within a row");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() >= 1,
              "seq_lens must be an int32 tensor holding the sequence length");
  // One sequence: row 0 of the block table and seq_lens[0].
  TORCH_CHECK(q.size(0) == maxm, "q has ", q.size(0),
              " tokens, not one sequence of MAX_QUERY_LEN=", maxm);
  TORCH_CHECK(block_table.dim() == 1 || block_table.size(0) >= 1,
              "block table has no row");
  TORCH_CHECK(
      q.size(1) == c.num_q_heads && q.size(2) == static_cast<int>(c.head_dim),
      "q shape does not match the build");
  const auto dtype = c.dtype == Dtype::kBf16 ? at::kBFloat16 : at::kHalf;
  TORCH_CHECK(q.scalar_type() == dtype && kv_cache.scalar_type() == dtype &&
                  out.scalar_type() == dtype,
              "kernel built for ", dtype, ", got q=", q.scalar_type(),
              " kv_cache=", kv_cache.scalar_type(), " out=", out.scalar_type());
  TORCH_CHECK(out.numel() == q.numel(), "out must hold q's ", q.numel(),
              " elements, got ", out.numel());
  const int hq = c.num_q_heads, hkv = c.num_kv_heads,
            d = static_cast<int>(c.head_dim);
  check_kv_cache(kv_cache, hkv, c.page_size, d);
  check_device(q,
               {&kv_cache, &block_table, &out, &acc, &m, &l, &cnt, &seq_lens});
  // As KernelVariant.scratch_shapes.
  const int segments = c.max_segments;
  int64_t rows, counters;
  if (c.dot_product) {
    rows = (int64_t)hq * segments * maxm;
    counters = hq;
  } else {
    const int groups = c.row_groups;
    const int64_t padded = ((hq / hkv) * maxm / groups + 15) / 16 * 16;
    rows = (int64_t)hkv * groups * segments * padded;
    counters = 4 * hkv * groups;
  }
  TORCH_CHECK(acc.scalar_type() == torch::kFloat32 &&
                  m.scalar_type() == torch::kFloat32 &&
                  l.scalar_type() == torch::kFloat32 &&
                  cnt.scalar_type() == torch::kInt32,
              "scratch must be float32 partials and int32 counters");
  TORCH_CHECK(acc.numel() >= rows * d && m.numel() >= rows &&
                  l.numel() >= rows && cnt.numel() >= counters,
              "scratch too small for this build: needs ", rows * d, ", ", rows,
              ", ", rows, ", ", counters, " elements");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  decode_launchers()[config](
      {q.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr<int>(),
       acc.data_ptr<float>(), m.data_ptr<float>(), l.data_ptr<float>(),
       cnt.data_ptr<int>(), out.data_ptr(), seq_lens.data_ptr<int>(),
       (int)block_table.size(-1), (float)scale,
       at::cuda::getCurrentCUDAStream()});
}

// One sequence; runs outside CUDA graphs, so max_seq_len may size the split.
void rdna_singleseq_prefill_attn(
    int64_t config, torch::Tensor& q, torch::Tensor& kv_cache,
    torch::Tensor& block_table, torch::Tensor& out, torch::Tensor& partial_o,
    torch::Tensor& partial_ml, torch::Tensor& counters, torch::Tensor& seq_lens,
    int64_t max_seq_len, double scale) {
  using namespace rdna_singleseq;
  TORCH_CHECK(
      config >= 0 && config < kNumPrefillConfigs && prefill_launchers()[config],
      "no prefill build ", config);
  const PrefillConfig& c = kPrefillConfigs[config];
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "q and out must be contiguous");
  TORCH_CHECK(
      block_table.scalar_type() == torch::kInt32 && block_table.stride(-1) == 1,
      "block table must be an int32 row, contiguous");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() >= 1,
              "seq_lens must be an int32 tensor holding the sequence length");
  TORCH_CHECK(q.size(0) > 0 && max_seq_len >= q.size(0),
              "max_seq_len must cover the query tokens");
  TORCH_CHECK(
      q.size(1) == c.num_q_heads && q.size(2) == static_cast<int>(c.head_dim),
      "q shape does not match the build");
  TORCH_CHECK(partial_o.scalar_type() == torch::kFloat32 &&
                  partial_ml.scalar_type() == torch::kFloat32 &&
                  counters.scalar_type() == torch::kInt32,
              "scratch must be float32 partials and int32 counters");
  const auto dtype = c.dtype == Dtype::kBf16 ? at::kBFloat16 : at::kHalf;
  TORCH_CHECK(q.scalar_type() == dtype && kv_cache.scalar_type() == dtype &&
                  out.scalar_type() == dtype,
              "kernel built for ", dtype, ", got q=", q.scalar_type(),
              " kv_cache=", kv_cache.scalar_type(), " out=", out.scalar_type());
  TORCH_CHECK(out.numel() == q.numel(), "out must hold q's ", q.numel(),
              " elements, got ", out.numel());
  const int d = static_cast<int>(c.head_dim);
  check_kv_cache(kv_cache, c.num_kv_heads, c.page_size, d);
  check_device(q, {&kv_cache, &block_table, &out, &partial_o, &partial_ml,
                   &counters, &seq_lens});
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  if (c.max_segments > 1) {
    // As PrefillVariant.scratch_rows.
    const int64_t target =
        4 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t block_rows = c.waves / (c.head_dim_split * c.key_waves) * 16;
    const int64_t rows = 2 * target * block_rows;
    TORCH_CHECK(partial_o.numel() >= rows * d &&
                    partial_ml.numel() >= 2 * rows &&
                    counters.numel() >= target,
                "scratch too small for this build: needs ", rows * d, ", ",
                2 * rows, ", ", target, " elements");
  }
  prefill_launchers()[config](
      {q.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr<int>(),
       out.data_ptr(), seq_lens.data_ptr<int>(), partial_o.data_ptr<float>(),
       partial_ml.data_ptr<float>(), counters.data_ptr<int>(), (int)q.size(0),
       (int)max_seq_len, (float)scale, at::cuda::getCurrentCUDAStream()});
}
