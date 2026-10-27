// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Qwen4Exp QSA indexer scores on RDNA3 (gfx1100) WMMA:
//
//   logits[r, c] = sum_h relu(q[r, h] . k[c]) / divisor   for c < visible[r]
//
// over the paged compressed-key cache of row r's request (one KV head, H = 4
// query heads, head dim 128). Same contract as _qsa_mqa_paged_kernel in
// vllm/models/qwen4_exp/amd/ops/qsa.py: visible[r] is written for every row;
// columns visible[r] <= c are -inf in the tiles that are written; tiles past
// every row's visible range of a block are left untouched (top-k only reads
// the first visible[r] columns).
//
// A block takes 64 query rows: each of its 16 waves holds 4 rows x 4 heads
// as one 16-row WMMA M tile, its Q fragments kept in registers. The block
// walks BN-column K tiles through double-buffered LDS. Every tile lies inside
// one cache page (page_size % BN == 0, tiles BN-aligned), so the page-table
// lookup is a single load per tile. Rows of different requests in one block
// (decode batches) are handled in one pass per request.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1100__)
  #define __HIP__RDNA3__
#endif

namespace vllm {
namespace qsa_index_rdna3 {

constexpr int kHeads = 4;
constexpr int kHeadDim = 128;
constexpr int kWaves = 16;
constexpr int kThreads = kWaves * 32;
constexpr int kRows = 4 * kWaves;  // query rows per block
constexpr int kTileN = 32;         // columns per K tile
// One 16-byte K chunk per thread per tile. More than one per thread puts
// the prefetch registers on the stack.
static_assert(kTileN * kHeadDim / 8 == kThreads);

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

using v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using v8fp32 = float __attribute__((ext_vector_type(8)));

template <int COLS_PER_BLOCK>
__global__ void __launch_bounds__(kThreads)
    scores_kernel(const half* __restrict__ q, const half* __restrict__ k_cache,
                  const int32_t* __restrict__ page_table,
                  const int32_t* __restrict__ token_to_req,
                  const int64_t* __restrict__ query_positions,
                  const int32_t* __restrict__ sequence_lengths,
                  int32_t* __restrict__ visible_out, float* __restrict__ logits,
                  int64_t stride_q_row, int64_t stride_cache_block,
                  int64_t stride_cache_token, int64_t stride_table_req,
                  int64_t stride_logits_row, int num_rows, int num_columns,
                  int num_pages, int num_requests, int page_size,
                  int table_width, int compress_ratio, float inv_divisor) {
  constexpr int KP = kHeadDim + 8;  // padded LDS row
  const int t = threadIdx.x;
  const int wave = t >> 5;
  const int lane = t & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;
  const int row0 = blockIdx.x * kRows;
  const int col_begin = blockIdx.y * COLS_PER_BLOCK;

  __shared__ int s_req[kRows];
  __shared__ int s_vis[kRows];
  __shared__ int s_reqs[kRows];  // distinct valid requests of the block
  __shared__ int s_nreq, s_max_vis;
  __shared__ __attribute__((aligned(16))) half sk[2][kTileN * KP];

  if (t < kRows) {
    const int r = row0 + t;
    int req = -1, vis = 0;
    if (r < num_rows) {
      req = token_to_req[r];
      if (req >= 0 && req < num_requests) {
        const int64_t pos = query_positions[r];
        vis = (int)min((pos + 1) / compress_ratio,
                       (int64_t)(sequence_lengths[req] / compress_ratio));
      } else {
        req = -1;
      }
      if (blockIdx.y == 0) visible_out[r] = vis;
    }
    s_req[t] = req;
    s_vis[t] = vis;
  }
  __syncthreads();
  // Done by one thread: unrolled loops over s_req / s_vis in every lane
  // would pin all of them in VGPRs.
  if (t == 0) {
    int n = 0, max_vis = 0;
    for (int i = 0; i < kRows; ++i) {
      max_vis = max(max_vis, s_vis[i]);
      const int r = s_req[i];
      if (r < 0) continue;
      bool seen = false;
      for (int j = 0; j < n; ++j) seen |= s_reqs[j] == r;
      if (!seen) s_reqs[n++] = r;
    }
    s_nreq = n;
    s_max_vis = max_vis;
  }
  __syncthreads();
  const int nreq = s_nreq;
  const int col_end = min(min(col_begin + COLS_PER_BLOCK, num_columns),
                          (s_max_vis + kTileN - 1) / kTileN * kTileN);
  if (col_begin >= col_end) return;

  // A fragments of this wave: m = 4 * row + head.
  v16fp16 af[kHeadDim / 16];
  {
    const int r = row0 + wave * 4 + (lane_lo >> 2);
    const bool ok = r < num_rows;
    const half* qp =
        q + (int64_t)(ok ? r : 0) * stride_q_row + (lane_lo & 3) * kHeadDim;
  #pragma unroll
    for (int ks = 0; ks < kHeadDim / 16; ++ks) {
      if (ok) {
        af[ks] = *reinterpret_cast<const v16fp16*>(qp + ks * 16);
      } else {
  #pragma unroll
        for (int i = 0; i < 16; ++i) af[ks][i] = 0;
      }
    }
  }

  // C fragment lane holds m = 2 * i + lane_hi, i.e. row i / 2 of the wave's
  // four, head 2 * (i & 1) + lane_hi. Lanes of half lane_hi store rows
  // 2 * lane_hi and 2 * lane_hi + 1.
  int out_row[2], out_req[2], out_vis[2];
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int rl = wave * 4 + 2 * lane_hi + j;
    out_row[j] = row0 + rl;
    out_req[j] = s_req[rl];
    out_vis[j] = s_vis[rl];
  }

  const int k_row = t >> 4;
  const int k_ch = (t & 15) * 8;
  for (int pass = 0; pass < nreq; ++pass) {
    const int req = s_reqs[pass];
    const int32_t* table = page_table + (int64_t)req * stride_table_req;
    auto load_k = [&](int c0) {
      const int lp = c0 / page_size;
      const int pp = lp < table_width ? table[lp] : -1;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (c0 < num_columns && pp >= 0 && pp < num_pages)
        v = *reinterpret_cast<const uint4*>(
            k_cache + (int64_t)pp * stride_cache_block +
            (int64_t)(c0 - lp * page_size + k_row) * stride_cache_token + k_ch);
      return v;
    };

    uint4 rk = load_k(col_begin);
    __syncthreads();  // the previous pass may still read sk
    *reinterpret_cast<uint4*>(&sk[0][k_row * KP + k_ch]) = rk;
    __syncthreads();
    int buf = 0;
    for (int c0 = col_begin; c0 < col_end; c0 += kTileN) {
      const bool more = c0 + kTileN < col_end;
      if (more) rk = load_k(c0 + kTileN);

      v8fp32 acc[kTileN / 16];
  #pragma unroll
      for (int j = 0; j < kTileN / 16; ++j)
  #pragma unroll
        for (int i = 0; i < 8; ++i) acc[j][i] = 0.0f;
  #pragma unroll
      for (int ks = 0; ks < kHeadDim / 16; ++ks)
  #pragma unroll
        for (int j = 0; j < kTileN / 16; ++j) {
          const v16fp16 bf = *reinterpret_cast<const v16fp16*>(
              &sk[buf][(j * 16 + lane_lo) * KP + ks * 16]);
          acc[j] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[ks], bf, acc[j]);
        }

  #pragma unroll
      for (int j = 0; j < kTileN / 16; ++j) {
        const int col = c0 + j * 16 + lane_lo;
        float s[4];
  #pragma unroll
        for (int rl = 0; rl < 4; ++rl) {
          float v =
              fmaxf(acc[j][2 * rl], 0.0f) + fmaxf(acc[j][2 * rl + 1], 0.0f);
          s[rl] = v + __shfl_xor(v, 16, 32);
        }
  #pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          if (out_row[jj] >= num_rows || out_req[jj] != req ||
              col >= num_columns)
            continue;
          const float v = (lane_hi ? s[2 + jj] : s[jj]) * inv_divisor;
          logits[(int64_t)out_row[jj] * stride_logits_row + col] =
              col < out_vis[jj] ? v : -INFINITY;
        }
      }

      if (more) {
        *reinterpret_cast<uint4*>(&sk[buf ^ 1][k_row * KP + k_ch]) = rk;
        __syncthreads();
        buf ^= 1;
      }
    }
  }
}

#else  // non-RDNA3: empty stub for symbol parity

template <int COLS_PER_BLOCK>
__global__ void scores_kernel(const half*, const half*, const int32_t*,
                              const int32_t*, const int64_t*, const int32_t*,
                              int32_t*, float*, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int, int, int, int, int, int,
                              int, float) {}

#endif  // __HIP__RDNA3__ || !__HIP_DEVICE_COMPILE__

}  // namespace qsa_index_rdna3
}  // namespace vllm

// q [rows, 4, 128] half; k_cache [pages, page_size, 1, 128] half with
// page_size % 32 == 0; page_table [requests, width] int32; token_to_req
// [rows] int32; query_positions [rows] int64; sequence_lengths [requests]
// int32. Writes visible [rows] int32 and logits [rows, columns] float32.
void qsa_index_scores_rdna3(torch::Tensor q, torch::Tensor k_cache,
                            torch::Tensor page_table,
                            torch::Tensor token_to_req,
                            torch::Tensor query_positions,
                            torch::Tensor sequence_lengths,
                            torch::Tensor visible, torch::Tensor logits,
                            int64_t compress_ratio, double divisor) {
  using namespace vllm::qsa_index_rdna3;
  TORCH_CHECK(
      q.scalar_type() == torch::kHalf && k_cache.scalar_type() == torch::kHalf,
      "QSA index scores run in float16");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeads && q.size(2) == kHeadDim &&
                  q.stride(2) == 1 && q.stride(1) == kHeadDim,
              "q must be [rows, 4, 128] with contiguous heads");
  TORCH_CHECK(
      k_cache.dim() == 4 && k_cache.size(2) == 1 &&
          k_cache.size(3) == kHeadDim && k_cache.stride(3) == 1 &&
          k_cache.size(1) % kTileN == 0,
      "k_cache must be [pages, page_size, 1, 128], page_size % 32 == 0");
  TORCH_CHECK(page_table.scalar_type() == torch::kInt32 &&
                  token_to_req.scalar_type() == torch::kInt32 &&
                  query_positions.scalar_type() == torch::kInt64 &&
                  sequence_lengths.scalar_type() == torch::kInt32 &&
                  visible.scalar_type() == torch::kInt32 &&
                  logits.scalar_type() == torch::kFloat32,
              "unexpected index dtypes");
  TORCH_CHECK(page_table.stride(1) == 1 && logits.stride(1) == 1 &&
                  token_to_req.is_contiguous() &&
                  query_positions.is_contiguous() &&
                  sequence_lengths.is_contiguous() && visible.is_contiguous(),
              "unexpected index strides");
  const int rows = q.size(0);
  const int columns = logits.size(1);
  if (rows == 0 || columns == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
#define QSA_INDEX_LAUNCH(CPB)                                                 \
  scores_kernel<CPB>                                                          \
      <<<dim3((rows + kRows - 1) / kRows, (columns + CPB - 1) / CPB),         \
         kThreads, 0, stream>>>(                                              \
          (const half*)q.data_ptr(), (const half*)k_cache.data_ptr(),         \
          page_table.data_ptr<int32_t>(), token_to_req.data_ptr<int32_t>(),   \
          query_positions.data_ptr<int64_t>(),                                \
          sequence_lengths.data_ptr<int32_t>(), visible.data_ptr<int32_t>(),  \
          logits.data_ptr<float>(), q.stride(0), k_cache.stride(0),           \
          k_cache.stride(1), page_table.stride(0), logits.stride(0), rows,    \
          columns, (int)k_cache.size(0), (int)page_table.size(0),             \
          (int)k_cache.size(1), (int)page_table.size(1), (int)compress_ratio, \
          (float)(1.0 / divisor))
  // Fewer rows leave fewer row blocks; split the columns finer to fill the
  // GPU.
  if (rows >= 256) {
    QSA_INDEX_LAUNCH(1024);
  } else {
    QSA_INDEX_LAUNCH(512);
  }
#undef QSA_INDEX_LAUNCH
}
