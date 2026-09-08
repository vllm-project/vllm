/*
 * Persistent TopK Scheduler for DSA Indexer
 */

#ifndef PERSISTENT_TOPK_CUH_
#define PERSISTENT_TOPK_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <type_traits>

namespace vllm {
namespace persistent {

// ============================================================================
// Constants
// ============================================================================

constexpr int kThreadsPerBlock = 1024;
constexpr int RADIX = 256;

// Medium path: all shared state in dynamic smem (no static __shared__,
// which would inflate the kernel's smem footprint and kill occupancy
// for the decode/trivial paths).
constexpr size_t kMediumHistBytes = 2 * (RADIX + 128) * sizeof(int);  // 3072
constexpr size_t kMediumScalarsBytes = 5 * sizeof(int);               // 20
constexpr size_t kMediumHeaderSize =
    (kMediumHistBytes + kMediumScalarsBytes + 127) & ~size_t(127);  // 3200
constexpr int MAX_BUFFERED_ITEMS = 4096;
constexpr size_t kSmemMedium =
    kMediumHeaderSize + 2 * MAX_BUFFERED_ITEMS * sizeof(int);  // 35968
// Rows at or below this width take the single-CTA cached select; wider rows take the multi-CTA
// cooperative radix. The bound is shared memory, not speed: det_select_row caches the row's
// ordered keys at 4 bytes each and needs fixed + 4n <= the device opt-in, i.e. n <= 24,280 on a
// 101,376 B part -- so 24,576 would silently fall to the uncached path. Measured on GB10, every
// width in 16,384 < n <= 22,016 costs 16-53 % more on the multi-CTA path (worst at 64 rows:
// 17,408 is 53.3 -> 34.8 us), while n = 16,384 and n >= 24,576 are unchanged.
constexpr uint32_t RADIX_THRESHOLD = 22016;

// Decode path constants
constexpr int kDecodeBins = 2048;
constexpr uint32_t HIST2048_THRESHOLD = 8192;

// Large path: fixed shared memory for histograms + scalars
constexpr size_t kFixedSmemLarge =
    ((RADIX + RADIX + 5) * sizeof(uint32_t) + 15) & ~size_t(15);
// The blocked emission loads `shared_ordered` 128 bits at a time.
static_assert(kFixedSmemLarge % 16 == 0,
              "shared_ordered must stay 16-byte aligned");

// ============================================================================
// Common helpers
// ============================================================================

__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  // -0.0 and +0.0 are numerically equal, so the documented rule (value
  // descending, ties by index ascending) must treat them as a tie. Their bit
  // patterns differ, which would otherwise order +0.0 above -0.0. Canonicalise
  // to +0 before the order-preserving transform.
  if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

// ============================================================================
// Vectorized load helpers
// ============================================================================

// Unconditional float4 load with cache hint (.cg = cache at global level only).
__device__ __forceinline__ void load_float4(const float* ptr, float& v0,
                                            float& v1, float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

// Per-element predicated scalar loads with -inf default.
__device__ __forceinline__ void load_float4_predicated(const float* ptr,
                                                       int base, int seq_len,
                                                       float& v0, float& v1,
                                                       float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  int p0 = (base < seq_len);
  int p1 = (base + 1 < seq_len);
  int p2 = (base + 2 < seq_len);
  int p3 = (base + 3 < seq_len);
  asm volatile(
      "{\n"
      "  .reg .pred pr0, pr1, pr2, pr3;\n"
      "  setp.ne.u32 pr0, %4, 0;\n"
      "  setp.ne.u32 pr1, %5, 0;\n"
      "  setp.ne.u32 pr2, %6, 0;\n"
      "  setp.ne.u32 pr3, %7, 0;\n"
      "  mov.u32 %0, 0xFF800000;\n"
      "  mov.u32 %1, 0xFF800000;\n"
      "  mov.u32 %2, 0xFF800000;\n"
      "  mov.u32 %3, 0xFF800000;\n"
      "  @pr0 ld.global.cg.u32 %0, [%8];\n"
      "  @pr1 ld.global.cg.u32 %1, [%8+4];\n"
      "  @pr2 ld.global.cg.u32 %2, [%8+8];\n"
      "  @pr3 ld.global.cg.u32 %3, [%8+12];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

// ============================================================================
// Deterministic selection.
// The output of this op must not depend on thread scheduling: the QSA
// consumer sums the selected keys in output order, so a scheduling-dependent
// order (or, under threshold ties, a scheduling-dependent set) makes identical
// requests diverge (vllm#54521). Hence no output slot is ever assigned from an
// arrival counter and no tie is ever resolved first-come.
//
// Every single-CTA row goes through `det_select_row`: a radix select that
// rescans the row for each of the four key bytes (no candidate buffers, so no
// truncation and an exact pivot), then one index-ordered block scan that emits
// all elements above the pivot and the lowest-index `fin` elements equal to it
// **directly into their final ascending positions**: the rank of a selected
// element is (# greater before) + min(# equal before, fin), both of which the
// emission's packed scan already carries, so no reordering pass exists at all.
// Cost: at most five reads of the row, fewer when a radix pass can be skipped.
// Rows longer than RADIX_THRESHOLD take the multi-CTA path, which computes the
// same position with per-CTA prefixes added.
// ============================================================================
// Deterministic single-CTA top-k of one row (n > TopK).
// Shared memory layout (bytes): [0,1024) hist, [1024,2048) hist2 (suffix
// scratch), [2048, +scan) BlockScan storage, then — when `smem_bytes` allows —
// the row's ordered keys
// (4*n bytes), so passes 1..3 and the emission read shared memory and the
// row is fetched from global memory exactly once. Otherwise every pass
// rescans global memory (still deterministic, just slower).
// Single source of truth for the fixed part of det_select_row's shared-memory
// layout; usable from the launcher (host) and the kernel (device).
template <int TopK, int N_THREADS>
__host__ __device__ constexpr size_t det_select_row_fixed_bytes() {
  using ScanT = cub::BlockScan<uint32_t, N_THREADS>;
  return 2048 + ((sizeof(typename ScanT::TempStorage) + 127) & ~size_t(127));
}
// Bytes needed to keep a row of n keys cached (host side sizing helper).
template <int TopK, int N_THREADS>
__host__ __device__ constexpr size_t det_select_row_bytes(size_t n) {
  return det_select_row_fixed_bytes<TopK, N_THREADS>() + n * sizeof(uint32_t);
}
template <int TopK, int N_THREADS>
__device__ void det_select_row(const float* __restrict__ row, int n,
                               int32_t* __restrict__ out, void* smem,
                               size_t smem_bytes) {
  static_assert(N_THREADS <= 0xFFFF,
                "the emission packs two per-tile counts into one uint32");
  static_assert(N_THREADS >= 32, "the bin scan runs in one full warp");
  using ScanT = cub::BlockScan<uint32_t, N_THREADS>;
  uint32_t* hist = reinterpret_cast<uint32_t*>(smem);  // [256]
  uint32_t* hist2 = hist + 256;                        // [256]
  auto* scan_tmp = reinterpret_cast<typename ScanT::TempStorage*>(
      reinterpret_cast<char*>(smem) + 2048);
  const size_t fixed = det_select_row_fixed_bytes<TopK, N_THREADS>();
  uint32_t* keys =
      reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(smem) + fixed);
  const bool cached =
      (fixed + static_cast<size_t>(n) * sizeof(uint32_t) <= smem_bytes);
  const int tx = threadIdx.x;
  uint32_t prefix = 0;
  uint32_t remaining = TopK;
  for (int pass = 0; pass < 4; pass++) {
    const int shift = 24 - pass * 8;
    const uint32_t hi_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
    for (int i = tx; i < 256; i += N_THREADS) hist[i] = 0;
    __syncthreads();
    if (pass == 0) {
      // first pass: read the row once; keep the ordered keys if they fit
      const int n4 = n & ~3;
      const bool aligned = ((reinterpret_cast<uintptr_t>(row) & 15) == 0);
      for (int i = tx * 4; i < n4; i += N_THREADS * 4) {
        float v0, v1, v2, v3;
        if (aligned)
          load_float4(row + i, v0, v1, v2, v3);
        else {
          v0 = row[i];
          v1 = row[i + 1];
          v2 = row[i + 2];
          v3 = row[i + 3];
        }
        const uint32_t k0 = convert_to_uint32_v2(v0),
                       k1 = convert_to_uint32_v2(v1);
        const uint32_t k2 = convert_to_uint32_v2(v2),
                       k3 = convert_to_uint32_v2(v3);
        if (cached) {
          keys[i] = k0;
          keys[i + 1] = k1;
          keys[i + 2] = k2;
          keys[i + 3] = k3;
        }
        atomicAdd(&hist[k0 >> 24], 1);
        atomicAdd(&hist[k1 >> 24], 1);
        atomicAdd(&hist[k2 >> 24], 1);
        atomicAdd(&hist[k3 >> 24], 1);
      }
      for (int i = n4 + tx; i < n; i += N_THREADS) {
        const uint32_t k = convert_to_uint32_v2(row[i]);
        if (cached) keys[i] = k;
        atomicAdd(&hist[k >> 24], 1);
      }
    } else {
      for (int i = tx; i < n; i += N_THREADS) {
        const uint32_t key = cached ? keys[i] : convert_to_uint32_v2(row[i]);
        if ((key & hi_mask) == prefix)
          atomicAdd(&hist[(key >> shift) & 0xFF], 1);
      }
    }
    __syncthreads();
    // Suffix sum over the 256 bins and the threshold search, in ONE warp:
    // lane l owns bins [8l, 8l+8), sums them serially, then a Hillis-Steele
    // suffix scan over the 32 lane totals gives what lies above the lane.
    // suf[b] = #elements in this prefix group with byte >= b, and the first
    // bin of the next lane has suf = `above`, so the search is lane-local too.
    // The previous 8-step double-buffered version cost 8 __syncthreads() here,
    // 32 over the four passes; this costs none.
    if (tx < 32) {
      uint32_t local[8];
      uint32_t total = 0;
#pragma unroll
      for (int j = 7; j >= 0; j--) {
        total += hist[tx * 8 + j];
        local[j] = total;
      }
      uint32_t s_suf = total;  // inclusive suffix over lane totals
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const uint32_t v = __shfl_down_sync(0xFFFFFFFFu, s_suf, off);
        if (tx + off < 32) s_suf += v;
      }
      const uint32_t above = s_suf - total;  // strictly higher lanes
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const uint32_t suf_b = local[j] + above;
        const uint32_t suf_b1 = (j < 7) ? (local[j + 1] + above) : above;
        if (suf_b >= remaining && suf_b1 < remaining) {
          hist2[0] = static_cast<uint32_t>(tx * 8 + j);
          hist2[1] = suf_b1;
          hist2[2] = suf_b - suf_b1;  // population of the threshold bin
        }
      }
    }
    __syncthreads();
    const uint32_t thr = hist2[0];
    const uint32_t bin_pop = hist2[2];
    remaining -= hist2[1];
    prefix |= thr << shift;
    // Early exit: the threshold bin holds exactly what is still needed, so all
    // of it is selected and the lower key bytes cannot change the answer. The
    // selection becomes `key >= prefix`, i.e. `key > prefix - 1`, with no ties
    // to rank. (`remaining == 0` can never happen: the bin search guarantees
    // suf_b1 < remaining, so the subtraction above always leaves at least 1.)
    if (bin_pop == remaining && prefix != 0u) {
      prefix -= 1u;
      remaining = 0u;
      __syncthreads();
      break;
    }
    __syncthreads();
  }
  const uint32_t pivot = prefix;
  const uint32_t fin = remaining;
  // Blocked emission, four elements per thread: thread `tx` owns the four
  // consecutive indices [base + 4*tx, base + 4*tx + 4). Blocked (not striped)
  // ownership is what keeps the position formula below valid unchanged -- it
  // needs the scan to run in index order, and blocked layout preserves index
  // order both within a thread and across threads. This runs one BlockScan and
  // one barrier per 4*N_THREADS elements instead of per N_THREADS: at
  // n = 16384 that is 4 of each rather than 16. The selected set and its order
  // are untouched, since the position is a pure function of index, pivot and
  // `fin`.
  constexpr int kItems = 4;
  constexpr int kTile = kItems * N_THREADS;
  static_assert(kTile <= 0xFFFF,
                "the emission packs two per-tile counts into one uint32");
  // `keys` sits at `smem + fixed`, and `fixed` is 2048 plus a multiple of 128,
  // so the 128-bit blocked load below is aligned and conflict-free.
  const bool row_aligned = ((reinterpret_cast<uintptr_t>(row) & 15) == 0);
  uint32_t run_gt = 0, run_eq = 0;
  for (int base = 0; base < n; base += kTile) {
    const int mine = base + tx * kItems;
    uint32_t key[kItems];
    bool valid[kItems];
    if (mine + kItems <= n) {
      if (cached) {
        const uint4 v = *reinterpret_cast<const uint4*>(keys + mine);
        key[0] = v.x;
        key[1] = v.y;
        key[2] = v.z;
        key[3] = v.w;
      } else if (row_aligned) {
        float v0, v1, v2, v3;
        load_float4(row + mine, v0, v1, v2, v3);
        key[0] = convert_to_uint32_v2(v0);
        key[1] = convert_to_uint32_v2(v1);
        key[2] = convert_to_uint32_v2(v2);
        key[3] = convert_to_uint32_v2(v3);
      } else {
#pragma unroll
        for (int j = 0; j < kItems; ++j)
          key[j] = convert_to_uint32_v2(row[mine + j]);
      }
#pragma unroll
      for (int j = 0; j < kItems; ++j) valid[j] = true;
    } else {
#pragma unroll
      for (int j = 0; j < kItems; ++j) {
        const int i = mine + j;
        valid[j] = (i < n);
        key[j] =
            valid[j] ? (cached ? keys[i] : convert_to_uint32_v2(row[i])) : 0u;
      }
    }
    uint32_t fgt[kItems], feq[kItems];
    uint32_t agg = 0;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      fgt[j] = (valid[j] && key[j] > pivot) ? 1u : 0u;
      feq[j] = (valid[j] && key[j] == pivot) ? 1u : 0u;
      // Both flags in one scan: they are mutually exclusive and a tile holds at
      // most kTile elements, so each count fits in 16 bits.
      agg += fgt[j] | (feq[j] << 16);
    }
    uint32_t packed_rank, packed_total;
    ScanT(*scan_tmp).ExclusiveSum(agg, packed_rank, packed_total);
    // Final ascending position directly. The number of selected elements at
    // lower indices is (# greater before) + min(# equal before, fin), since
    // exactly the first `fin` equal elements by index are kept. True for a `>`
    // element (the min saturates) and for a kept `==` element (it does not),
    // so one expression serves both and no reordering pass is needed. The
    // block-wide exclusive prefix is continued serially over this thread's own
    // four, in index order.
    uint32_t g = run_gt + (packed_rank & 0xFFFFu);
    uint32_t e = run_eq + (packed_rank >> 16);
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      if (fgt[j] || (feq[j] && e < fin)) out[g + (e < fin ? e : fin)] = mine + j;
      g += fgt[j];
      e += feq[j];
    }
    run_gt += packed_total & 0xFFFFu;
    run_eq += packed_total >> 16;
    __syncthreads();
  }
}

// ============================================================================
// Large path: inter-CTA coordination state (one per group)
// ============================================================================

constexpr uint32_t kDetMaxCtasPerGroup = 64;
struct RadixRowState {
  uint32_t histogram[3][256];  // Triple-buffered histograms
  int arrival_counter;
  uint32_t det_gt_counts[kDetMaxCtasPerGroup];  // per-CTA > pivot
  uint32_t det_eq_counts[kDetMaxCtasPerGroup];  // per-CTA == pivot
};

// ============================================================================
// Kernel parameters
// ============================================================================

struct PersistentTopKParams {
  const float* __restrict__ input;      // [num_rows, stride]
  int32_t* __restrict__ output;         // [num_rows, top_k]
  const int32_t* __restrict__ lengths;  // [num_rows]
  RadixRowState* row_states;            // large path: per-group state
  uint32_t num_rows;
  uint32_t stride;
  uint32_t top_k;           // actual k value for output stride
  uint32_t chunk_size;      // large path: elements per CTA
  uint32_t ctas_per_group;  // 1=medium, >1=large
  uint32_t max_seq_len;     // max seq_len across all rows (for early CTA exit)
  uint32_t det_smem_bytes;  // dynamic smem available to det_select_row
  uint32_t force_single_cta;  // low-smem fallback: one CTA per row, no coop
};

// ============================================================================
// Decode path: 2048-bin histogram for short sequences (seq_len <= 8192)
// Uses 11-bit half-precision bins for fine granularity.
// One histogram pass typically suffices since 8192/2048 = 4 elements/bin avg.
// ============================================================================

// 11-bit bin from half-precision representation (ascending: high values -> high
// bins)
__device__ __forceinline__ uint32_t decode_bin(float x) {
  __half hx = __float2half(x);
  uint16_t bits = __half_as_ushort(hx);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return key >> 5;
}

template <int TopK>
__device__ __noinline__ void histogram_2048_topk(
    const float* __restrict__ logits, int32_t* __restrict__ output_indices,
    int32_t seq_len) {
  extern __shared__ int decode_smem[];
  const int tx = threadIdx.x;
  const int lane = tx & 31;

  // ---- Layout constants ----
  constexpr int SBASE = 8192 - 8;           // 8184
  constexpr int RHIST = RADIX + 128;        // 384
  constexpr int BOFF = 2 * RHIST;           // 768
  constexpr int DBUF = (SBASE - BOFF) / 2;  // 3708
  constexpr int MAX_ITEMS_PER_THREAD =
      (HIST2048_THRESHOLD + kThreadsPerBlock - 1) / kThreadsPerBlock;

  enum : int { sTHR = 0, sOUT = 1, sREF = 2, sFIN = 3, sBUF0 = 4, sBUF1 = 5 };

  // ---- Initialize scalars (prevents stale data from prior rows) ----
  if (tx < 8) {
    decode_smem[SBASE + tx] = 0;
  }

  // ---- Phase 1: Build 2048-bin histogram with float4 vectorized loads ----
  int* histo = decode_smem;
  uint16_t reg_bins[MAX_ITEMS_PER_THREAD];
  int nitems = 0;

  for (int i = tx; i < kDecodeBins; i += kThreadsPerBlock) {
    histo[i] = 0;
  }
  __syncthreads();

  const int n_vec = (seq_len + 3) >> 2;
  const bool row_aligned = ((reinterpret_cast<uintptr_t>(logits) & 15) == 0);

  for (int i = tx; i < n_vec; i += kThreadsPerBlock) {
    const int base = i << 2;
    float v0, v1, v2, v3;

    if (row_aligned && base + 3 < seq_len) {
      load_float4(logits + base, v0, v1, v2, v3);
    } else {
      load_float4_predicated(logits + base, base, seq_len, v0, v1, v2, v3);
    }

    const uint16_t b0 = static_cast<uint16_t>(decode_bin(v0));
    const uint16_t b1 = static_cast<uint16_t>(decode_bin(v1));
    const uint16_t b2 = static_cast<uint16_t>(decode_bin(v2));
    const uint16_t b3 = static_cast<uint16_t>(decode_bin(v3));
    reg_bins[nitems++] = b0;
    reg_bins[nitems++] = b1;
    reg_bins[nitems++] = b2;
    reg_bins[nitems++] = b3;
    atomicAdd(&histo[b0], 1);
    atomicAdd(&histo[b1], 1);
    atomicAdd(&histo[b2], 1);
    atomicAdd(&histo[b3], 1);
  }
  __syncthreads();

  // ---- CUB suffix sum ----
  using BlockScanT = cub::BlockScan<int, kThreadsPerBlock>;
  const int h0 = histo[2 * tx];
  const int pair_sum = h0 + histo[2 * tx + 1];

  auto& scan_storage = *reinterpret_cast<typename BlockScanT::TempStorage*>(
      decode_smem + kDecodeBins);

  int pair_prefix, total;
  BlockScanT(scan_storage).ExclusiveSum(pair_sum, pair_prefix, total);

  // Find threshold bin purely from registers
  const int pair_suffix = total - pair_prefix;

  if (pair_suffix >= TopK && (pair_suffix - h0) < TopK) {
    decode_smem[SBASE + sTHR] = 2 * tx;
  }
  {
    const int right_suf = pair_suffix - h0;
    const int next_suf = pair_suffix - pair_sum;
    if (right_suf >= TopK && next_suf < TopK) {
      decode_smem[SBASE + sTHR] = 2 * tx + 1;
    }
  }
  __syncthreads();

  const int threshold = decode_smem[SBASE + sTHR];

  // ---- Phase 2: Collection with warp-aggregated atomicAdds ----
  int* bufs[2] = {decode_smem + BOFF, decode_smem + BOFF + DBUF};
  const int sOUT_abs = SBASE + sOUT;
  const int sBUF0_abs = SBASE + sBUF0;

  {
    const uint32_t uthr = static_cast<uint32_t>(threshold);
    int item = 0;
    const int n_vec_iters = (n_vec + kThreadsPerBlock - 1) / kThreadsPerBlock;

    for (int iter = 0; iter < n_vec_iters; iter++) {
      const int i = tx + iter * kThreadsPerBlock;
      const bool vec_valid = (i < n_vec);
      const int base_idx = i << 2;

#pragma unroll 4
      for (int sub = 0; sub < 4; sub++) {
        const int elem_idx = base_idx + sub;
        uint32_t bin = 0;
        if (vec_valid) bin = reg_bins[item++];
        const bool is_above = vec_valid && (bin > uthr);
        const bool is_equal = vec_valid && (bin == uthr);

        const uint32_t above_mask = __ballot_sync(0xffffffff, is_above);
        if (above_mask) {
          const int above_count = __popc(above_mask);
          const int above_rank = __popc(above_mask & ((1u << lane) - 1));
          int above_base;
          if (lane == 0) {
            above_base = atomicAdd(&decode_smem[sOUT_abs], above_count);
          }
          above_base = __shfl_sync(0xffffffff, above_base, 0);
          if (is_above) {
            output_indices[above_base + above_rank] = elem_idx;
          }
        }

        const uint32_t equal_mask = __ballot_sync(0xffffffff, is_equal);
        if (equal_mask) {
          const int equal_count = __popc(equal_mask);
          const int equal_rank = __popc(equal_mask & ((1u << lane) - 1));
          int equal_base;
          if (lane == 0) {
            equal_base = atomicAdd(&decode_smem[sBUF0_abs], equal_count);
          }
          equal_base = __shfl_sync(0xffffffff, equal_base, 0);
          if (is_equal && __builtin_expect(equal_base + equal_rank < DBUF, 1)) {
            bufs[0][equal_base + equal_rank] = elem_idx;
          }
        }
      }
    }
  }
  __syncthreads();

  int remaining_k = TopK - decode_smem[SBASE + sOUT];
  if (remaining_k <= 0) return;

  // If all buffered elements fit, output them all (common for short seqs)
  const int raw_buf0 = decode_smem[SBASE + sBUF0];
  if (raw_buf0 <= remaining_k) {
    const int nb = (raw_buf0 < DBUF) ? raw_buf0 : DBUF;
    const int base = decode_smem[SBASE + sOUT];
    for (int i = tx; i < nb; i += kThreadsPerBlock) {
      output_indices[base + i] = bufs[0][i];
    }
    __syncthreads();
    return;
  }

  // ---- Phase 3: Deferred refinement (rare path) ----
  int* refine[2] = {decode_smem, decode_smem + RHIST};
  const int num_buf0 = (raw_buf0 < DBUF) ? raw_buf0 : DBUF;

  for (int i = tx; i < RHIST; i += kThreadsPerBlock) {
    refine[0][i] = 0;
  }
  __syncthreads();

  for (int i = tx; i < num_buf0; i += kThreadsPerBlock) {
    const uint32_t fp32 = convert_to_uint32_v2(logits[bufs[0][i]]);
    atomicAdd(&refine[0][(fp32 >> 24) & 0xFF], 1);
  }
  __syncthreads();

  auto compute_suffix_sum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const int stride = 1 << i;
        const int s = i & 1;
        const int d = s ^ 1;
        int value = refine[s][tx];
        if (tx < RADIX - stride) value += refine[s][tx + stride];
        refine[d][tx] = value;
      }
      __syncthreads();
    }
  };

#pragma unroll 4
  for (int pass = 0; pass < 4; ++pass) {
    const int src = pass & 1;
    const int dst = src ^ 1;

    const int raw_buf = decode_smem[SBASE + sBUF0 + src];
    const int num_buffered = (raw_buf < DBUF) ? raw_buf : DBUF;

    compute_suffix_sum();

    if (tx < RADIX && refine[0][tx] > remaining_k &&
        refine[0][tx + 1] <= remaining_k) {
      decode_smem[SBASE + sREF] = tx;
      decode_smem[SBASE + sBUF0 + dst] = 0;
      decode_smem[SBASE + sFIN] = remaining_k - refine[0][tx + 1];
    }
    __syncthreads();

    const int ref_thr = decode_smem[SBASE + sREF];
    remaining_k -= refine[0][ref_thr + 1];
    const int bit_offset = 24 - pass * 8;

    if (remaining_k == 0) {
      for (int i = tx; i < num_buffered; i += kThreadsPerBlock) {
        const int idx = bufs[src][i];
        const uint32_t fp32 = convert_to_uint32_v2(logits[idx]);
        if (((fp32 >> bit_offset) & 0xFF) > static_cast<uint32_t>(ref_thr)) {
          const int pos = atomicAdd(&decode_smem[SBASE + sOUT], 1);
          output_indices[pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    __syncthreads();
    if (tx < RADIX + 1) refine[0][tx] = 0;
    __syncthreads();

    for (int i = tx; i < num_buffered; i += kThreadsPerBlock) {
      const int idx = bufs[src][i];
      const float logit_val = logits[idx];
      const uint32_t fp32 = convert_to_uint32_v2(logit_val);
      const int bin = (fp32 >> bit_offset) & 0xFF;

      if (bin > ref_thr) {
        const int pos = atomicAdd(&decode_smem[SBASE + sOUT], 1);
        output_indices[pos] = idx;
      } else if (bin == ref_thr) {
        if (pass == 3) {
          const int slot = atomicAdd(&decode_smem[SBASE + sFIN], -1);
          if (slot > 0) output_indices[TopK - slot] = idx;
        } else {
          const int bp = atomicAdd(&decode_smem[SBASE + sBUF0 + dst], 1);
          if (__builtin_expect(bp < DBUF, 1)) {
            bufs[dst][bp] = idx;
            const int nbo = bit_offset - 8;
            atomicAdd(&refine[0][(fp32 >> nbo) & 0xFF], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

// ============================================================================
// Medium path: coarse FP16 histogram + 4-pass FP32 radix refinement
// For sequences 8K < seq_len <= 64K.
// ============================================================================

// Adapted from:
// https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/elementwise/topk.cu#L87
// by: DarkSharpness
// which at the same time is an optimized topk kernel copied from tilelang
// kernel
template <int TopK>
__device__ __noinline__ void histogram_256_topk(
    const float* __restrict__ logits, int* __restrict__ output_indices,
    int logits_offset, int seq_len) {
  // All shared state lives in dynamic shared memory to avoid static
  extern __shared__ char medium_smem[];

  int (*shared_histogram)[RADIX + 128] =
      reinterpret_cast<int (*)[RADIX + 128]>(medium_smem);
  int* medium_scalars = reinterpret_cast<int*>(medium_smem + kMediumHistBytes);
  int& shared_output_count = medium_scalars[0];
  int& shared_threshold_bin = medium_scalars[1];
  int* shared_buffered_count = &medium_scalars[2];
  int& shared_final_k = medium_scalars[4];
  int (*buffered_indices)[MAX_BUFFERED_ITEMS] =
      reinterpret_cast<int (*)[MAX_BUFFERED_ITEMS]>(medium_smem +
                                                    kMediumHeaderSize);

  const int thread_id = threadIdx.x;
  int remaining_k = TopK;

  if (thread_id < RADIX + 1) {
    shared_histogram[0][thread_id] = 0;
  }
  __syncthreads();

  for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
    const auto bin = convert_to_uint8(logits[idx + logits_offset]);
    atomicAdd(&shared_histogram[0][bin], 1);
  }
  __syncthreads();

  auto compute_cumulative_sum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (__builtin_expect(thread_id < RADIX, 1)) {
        const int stride = 1 << i;
        const int src_buffer = i & 1;
        const int dst_buffer = src_buffer ^ 1;
        int value = shared_histogram[src_buffer][thread_id];
        if (thread_id < RADIX - stride) {
          value += shared_histogram[src_buffer][thread_id + stride];
        }
        shared_histogram[dst_buffer][thread_id] = value;
      }
      __syncthreads();
    }
  };

  compute_cumulative_sum();

  if (thread_id < RADIX && shared_histogram[0][thread_id] > remaining_k &&
      shared_histogram[0][thread_id + 1] <= remaining_k) {
    shared_threshold_bin = thread_id;
    shared_buffered_count[0] = 0;
    shared_output_count = 0;
  }
  __syncthreads();

  const int threshold_bin = shared_threshold_bin;
  remaining_k -= shared_histogram[0][threshold_bin + 1];

  if (remaining_k == 0) {
    for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
      const int bin = convert_to_uint8(logits[idx + logits_offset]);
      if (bin > threshold_bin) {
        const int output_pos = atomicAdd(&shared_output_count, 1);
        output_indices[output_pos] = idx;
      }
    }
    __syncthreads();
    return;
  }

  __syncthreads();
  if (thread_id < RADIX + 1) {
    shared_histogram[0][thread_id] = 0;
  }
  __syncthreads();

  for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
    const float logit_value = logits[idx + logits_offset];
    const int bin = convert_to_uint8(logit_value);
    if (bin > threshold_bin) {
      const int output_pos = atomicAdd(&shared_output_count, 1);
      output_indices[output_pos] = idx;
    } else if (bin == threshold_bin) {
      const int buffer_pos = atomicAdd(&shared_buffered_count[0], 1);
      if (__builtin_expect(buffer_pos < MAX_BUFFERED_ITEMS, 1)) {
        buffered_indices[0][buffer_pos] = idx;
        const uint32_t fp32_bits = convert_to_uint32_v2(logit_value);
        const int next_bin = (fp32_bits >> 24) & 0xFF;
        atomicAdd(&shared_histogram[0][next_bin], 1);
      }
    }
  }
  __syncthreads();

#pragma unroll 4
  for (int pass = 0; pass < 4; ++pass) {
    const int src_buffer = pass % 2;
    const int dst_buffer = src_buffer ^ 1;
    const int raw_buffered = shared_buffered_count[src_buffer];
    const int num_buffered =
        (raw_buffered < MAX_BUFFERED_ITEMS) ? raw_buffered : MAX_BUFFERED_ITEMS;

    compute_cumulative_sum();

    if (thread_id < RADIX && shared_histogram[0][thread_id] > remaining_k &&
        shared_histogram[0][thread_id + 1] <= remaining_k) {
      shared_threshold_bin = thread_id;
      shared_buffered_count[dst_buffer] = 0;
      shared_final_k = remaining_k - shared_histogram[0][thread_id + 1];
    }
    __syncthreads();

    const int threshold_bin = shared_threshold_bin;
    remaining_k -= shared_histogram[0][threshold_bin + 1];
    const int bit_offset = 24 - pass * 8;

    if (remaining_k == 0) {
      for (int i = thread_id; i < num_buffered; i += kThreadsPerBlock) {
        const int idx = buffered_indices[src_buffer][i];
        const uint32_t fp32_bits =
            convert_to_uint32_v2(logits[idx + logits_offset]);
        const int bin = (fp32_bits >> bit_offset) & 0xFF;
        if (bin > threshold_bin) {
          const int output_pos = atomicAdd(&shared_output_count, 1);
          output_indices[output_pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    __syncthreads();
    if (thread_id < RADIX + 1) {
      shared_histogram[0][thread_id] = 0;
    }
    __syncthreads();

    for (int i = thread_id; i < num_buffered; i += kThreadsPerBlock) {
      const int idx = buffered_indices[src_buffer][i];
      const float logit_value = logits[idx + logits_offset];
      const uint32_t fp32_bits = convert_to_uint32_v2(logit_value);
      const int bin = (fp32_bits >> bit_offset) & 0xFF;
      if (bin > threshold_bin) {
        const int output_pos = atomicAdd(&shared_output_count, 1);
        output_indices[output_pos] = idx;
      } else if (bin == threshold_bin) {
        if (pass == 3) {
          const int slot = atomicAdd(&shared_final_k, -1);
          if (slot > 0) {
            output_indices[TopK - slot] = idx;
          }
        } else {
          const int buffer_pos =
              atomicAdd(&shared_buffered_count[dst_buffer], 1);
          if (__builtin_expect(buffer_pos < MAX_BUFFERED_ITEMS, 1)) {
            buffered_indices[dst_buffer][buffer_pos] = idx;
            const int next_bit_offset = bit_offset - 8;
            const int next_bin = (fp32_bits >> next_bit_offset) & 0xFF;
            atomicAdd(&shared_histogram[0][next_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

// ============================================================================
// Inter-CTA sync primitives
// ============================================================================

__device__ __forceinline__ int ld_acquire(int* ptr) {
  int state = 0;
#if (__CUDA_ARCH__ >= 700)
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(state)
               : "l"(ptr));
#else
  asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif
  return state;
}

__device__ __forceinline__ void red_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
               :
               : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicAdd(ptr, val);
#endif
}

__device__ __forceinline__ void st_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicExch(ptr, val);
#endif
}

__device__ __forceinline__ void wait_ge(int* ptr, int target_val,
                                        int thread_idx) {
  if (thread_idx == 0) {
#pragma unroll 1
    while (ld_acquire(ptr) < target_val) {
    }
  }
  __syncthreads();
}

// ============================================================================
// Large path: multi-CTA radix select for sequences > 64K
//
// Each row is processed by a group of CTAs. Each CTA loads its chunk into
// shared memory as ordered uint32, then participates in 4 rounds of
// coordinated radix select via global-memory histograms and barriers.
// ============================================================================

// ============================================================================
// Multi-CTA cooperative RadixTopK for a single large row.
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================

template <int TopK, uint32_t VEC_SIZE>
__device__ void radix_topk(const float* __restrict__ row_input,
                           int32_t* __restrict__ row_output, uint32_t seq_len,
                           uint32_t my_chunk_start, uint32_t chunk_size,
                           uint32_t* local_histogram, uint32_t* suffix_sum,
                           uint32_t* shared_scalars, uint32_t* shared_ordered,
                           RadixRowState* state, uint32_t cta_in_group,
                           uint32_t ctas_per_group, int& barrier_phase,
                           uint32_t radix_iter, uint32_t tx) {
  const uint32_t my_chunk_end = (my_chunk_start + chunk_size < seq_len)
                                    ? my_chunk_start + chunk_size
                                    : seq_len;
  const uint32_t actual_chunk_size =
      (my_chunk_start < seq_len) ? (my_chunk_end - my_chunk_start) : 0;

  // -- Stage 1: Load chunk to shared memory as ordered uint32 --
  {
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

    for (uint32_t i = tx * VEC_SIZE; i < aligned_size;
         i += kThreadsPerBlock * VEC_SIZE) {
      const float* src = row_input + my_chunk_start + i;
      if constexpr (VEC_SIZE == 4) {
        float4 v = *reinterpret_cast<const float4*>(src);
        shared_ordered[i] = convert_to_uint32_v2(v.x);
        shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
        shared_ordered[i + 2] = convert_to_uint32_v2(v.z);
        shared_ordered[i + 3] = convert_to_uint32_v2(v.w);
      } else if constexpr (VEC_SIZE == 2) {
        float2 v = *reinterpret_cast<const float2*>(src);
        shared_ordered[i] = convert_to_uint32_v2(v.x);
        shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
      } else {
        shared_ordered[i] = convert_to_uint32_v2(*src);
      }
    }
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size;
         i += kThreadsPerBlock) {
      shared_ordered[i] = convert_to_uint32_v2(row_input[my_chunk_start + i]);
    }
  }
  __syncthreads();

  // -- Init radix select state --
  if (tx == 0) {
    shared_scalars[0] = 0;     // prefix
    shared_scalars[1] = TopK;  // remaining_k
  }
  __syncthreads();

  // -- Initial barrier --
  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  // -- Stage 2: 4 rounds of radix select --
  for (uint32_t round = 0; round < 4; round++) {
    const uint32_t global_round = radix_iter * 4 + round;
    const uint32_t shift = 24 - round * 8;
    const uint32_t prefix = shared_scalars[0];
    const uint32_t remaining_k = shared_scalars[1];

    uint32_t* current_hist = state->histogram[global_round % 3];
    uint32_t* next_hist = state->histogram[(global_round + 1) % 3];

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      local_histogram[i] = 0;
    }
    __syncthreads();

    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      uint32_t ordered = shared_ordered[i];
      uint32_t mask = (round == 0) ? 0u : (~0u << (32 - round * 8));
      if ((ordered & mask) == prefix) {
        uint32_t bucket = (ordered >> shift) & 0xFF;
        atomicAdd(&local_histogram[bucket], 1);
      }
    }
    __syncthreads();

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      if (local_histogram[i] > 0) {
        atomicAdd(&current_hist[i], local_histogram[i]);
      }
    }

    if (cta_in_group == 0) {
      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        next_hist[i] = 0;
      }
    }

    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    wait_ge(&state->arrival_counter,
            (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
    barrier_phase++;
    __syncthreads();

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      suffix_sum[i] = current_hist[i];
    }
    __syncthreads();

    for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
      uint32_t val = 0;
      if (tx < RADIX) {
        val = suffix_sum[tx];
        if (tx + stride < RADIX) val += suffix_sum[tx + stride];
      }
      __syncthreads();
      if (tx < RADIX) suffix_sum[tx] = val;
      __syncthreads();
    }

    if (tx == 0) {
      shared_scalars[2] = 0;
      shared_scalars[3] = remaining_k;
    }
    __syncthreads();

    if (tx < RADIX) {
      uint32_t count_ge = suffix_sum[tx];
      uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
      if (count_ge >= remaining_k && count_gt < remaining_k) {
        shared_scalars[2] = tx;
        shared_scalars[3] = remaining_k - count_gt;
      }
    }
    __syncthreads();

    if (tx == 0) {
      shared_scalars[0] = prefix | (shared_scalars[2] << shift);
      shared_scalars[1] = shared_scalars[3];
    }
    __syncthreads();
  }  // end 4 radix rounds

  // -- Count local > pivot elements --
  const uint32_t ordered_pivot = shared_scalars[0];

  if (tx == 0) suffix_sum[0] = 0;
  __syncthreads();

  uint32_t my_gt_count = 0;
  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] > ordered_pivot) my_gt_count++;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    my_gt_count += __shfl_down_sync(0xffffffff, my_gt_count, offset);
  }
  if (tx % 32 == 0 && my_gt_count > 0) {
    atomicAdd(&suffix_sum[0], my_gt_count);
  }
  __syncthreads();
  const uint32_t local_gt_count = suffix_sum[0];

  // -- Stage 3: Collect top-k indices --
  // Publish this CTA's counts; slots come from a prefix over CTAs, never
  // from a global arrival counter. Both groups are ranked by index within the
  // CTA and ordered across CTAs by chunk, so `> pivot` and `== pivot` each come
  // out as one ascending run and CTA 0 merges them.
  uint32_t my_eq_count = 0;
  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] == ordered_pivot) my_eq_count++;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    my_eq_count += __shfl_down_sync(0xffffffff, my_eq_count, offset);
  }
  if (tx == 0) suffix_sum[1] = 0;
  __syncthreads();
  if (tx % 32 == 0 && my_eq_count > 0) atomicAdd(&suffix_sum[1], my_eq_count);
  __syncthreads();
  const uint32_t local_eq_count = suffix_sum[1];
  if (tx == 0) {
    state->det_gt_counts[cta_in_group] = local_gt_count;
    state->det_eq_counts[cta_in_group] = local_eq_count;
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();
  // All three values are CTA-uniform, so one thread reads the count table and
  // publishes them. Every thread doing it costs kThreadsPerBlock ctas_per_group
  // acquire loads per CTA to produce three scalars; the barrier above already
  // orders the publication.
  if (tx == 0) {
    uint32_t gb = 0, gtot = 0, eb = 0;
    for (uint32_t c = 0; c < ctas_per_group; c++) {
      const uint32_t g =
          ld_acquire(reinterpret_cast<int*>(&state->det_gt_counts[c]));
      const uint32_t e =
          ld_acquire(reinterpret_cast<int*>(&state->det_eq_counts[c]));
      if (c < cta_in_group) {
        gb += g;
        eb += e;
      }
      gtot += g;
    }
    shared_scalars[2] = gb;
    shared_scalars[3] = gtot;
    shared_scalars[4] = eb;
  }
  __syncthreads();
  const uint32_t gt_before = shared_scalars[2];
  const uint32_t gt_total = shared_scalars[3];
  const uint32_t eq_before = shared_scalars[4];
  const uint32_t remaining_eq =
      (gt_total < static_cast<uint32_t>(TopK)) ? (TopK - gt_total) : 0u;
  // Both groups are emitted in ascending index order, in one packed scan per
  // tile. The `> pivot` group used to take its slots with atomicAdd, i.e. in
  // thread-arrival order, which left that region unsorted and made the final
  // sort load-bearing. Ranking it by index instead makes each CTA's slice
  // ascending, and CTA c covers a lower index range than CTA c+1, so the whole
  // region is ascending — which lets the merge below replace the sort.
  {
    using ScanT = cub::BlockScan<uint32_t, kThreadsPerBlock>;
    __shared__ typename ScanT::TempStorage det_scan_tmp;
    static_assert(kThreadsPerBlock <= 0xFFFF,
                  "the emission packs two per-tile counts into one uint32");
    // Blocked emission, four elements per thread -- see det_select_row for why
    // blocked ownership is required and why it cannot change the result.
    constexpr uint32_t kItems = 4;
    constexpr uint32_t kTile = kItems * kThreadsPerBlock;
    static_assert(kTile <= 0xFFFF,
                  "the emission packs two per-tile counts into one uint32");
    uint32_t run_gt = 0, run_eq = 0;
    for (uint32_t base = 0; base < actual_chunk_size; base += kTile) {
      const uint32_t mine = base + tx * kItems;
      uint32_t key[kItems];
      bool valid[kItems];
      if (mine + kItems <= actual_chunk_size) {
        // `shared_ordered` starts at `smem_raw + kFixedSmemLarge`, a multiple
        // of 16, so this 128-bit blocked load is aligned and conflict-free.
        const uint4 v = *reinterpret_cast<const uint4*>(shared_ordered + mine);
        key[0] = v.x;
        key[1] = v.y;
        key[2] = v.z;
        key[3] = v.w;
#pragma unroll
        for (uint32_t j = 0; j < kItems; ++j) valid[j] = true;
      } else {
#pragma unroll
        for (uint32_t j = 0; j < kItems; ++j) {
          const uint32_t i = mine + j;
          valid[j] = (i < actual_chunk_size);
          key[j] = valid[j] ? shared_ordered[i] : 0u;
        }
      }
      uint32_t fgt[kItems], feq[kItems];
      uint32_t agg = 0;
#pragma unroll
      for (uint32_t j = 0; j < kItems; ++j) {
        fgt[j] = (valid[j] && key[j] > ordered_pivot) ? 1u : 0u;
        feq[j] = (valid[j] && key[j] == ordered_pivot) ? 1u : 0u;
        agg += fgt[j] | (feq[j] << 16);
      }
      uint32_t packed_rank, packed_total;
      ScanT(det_scan_tmp).ExclusiveSum(agg, packed_rank, packed_total);
      // Same direct placement, with the per-CTA prefixes folded in. CTA c owns
      // a lower contiguous index interval than CTA c+1, so CTA order and index
      // order agree and the position computed here is final.
      uint32_t g = gt_before + run_gt + (packed_rank & 0xFFFFu);
      uint32_t e = eq_before + run_eq + (packed_rank >> 16);
#pragma unroll
      for (uint32_t j = 0; j < kItems; ++j) {
        if (fgt[j] || (feq[j] && e < remaining_eq)) {
          const uint32_t pos = g + (e < remaining_eq ? e : remaining_eq);
          if (pos < static_cast<uint32_t>(TopK))
            row_output[pos] = static_cast<int32_t>(my_chunk_start + mine + j);
        }
        g += fgt[j];
        e += feq[j];
      }
      run_gt += packed_total & 0xFFFFu;
      run_eq += packed_total >> 16;
      __syncthreads();
    }
  }
  // One barrier closes the row: every CTA wrote its own final positions, so
  // there is no second phase publishing a reordering.
  if (tx == 0) red_release(&state->arrival_counter, 1);
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();
}

// ============================================================================
// Persistent kernel — BS≤32, decode/medium/large paths with RadixTopK
// BS>32 uses standalone histogram_256_buffered_topk (separate kernel,
// see filtered_topk.cuh)
// ============================================================================

template <int TopK = 2048, uint32_t VEC_SIZE = 1>
__global__ void __launch_bounds__(kThreadsPerBlock, 2)
    persistent_topk_kernel(PersistentTopKParams params) {
  const uint32_t tx = threadIdx.x;
  extern __shared__ uint8_t smem_raw[];

  // ========================================================================
  // Group mode: multi-CTA groups with static round-robin row assignment.
  // Non-large rows: CTA-0 handles trivial/decode/medium.
  // Large rows: all CTAs in the group cooperate via RadixTopK.
  // ========================================================================
  const uint32_t ctas_per_group = params.ctas_per_group;
  const uint32_t group_id = blockIdx.x / ctas_per_group;
  const uint32_t cta_in_group = blockIdx.x % ctas_per_group;
  const uint32_t num_groups = gridDim.x / ctas_per_group;
  const uint32_t chunk_size = params.chunk_size;

  if (blockIdx.x >= num_groups * ctas_per_group) return;

  // Early exit: non-CTA-0 threads are never needed if no large rows exist
  if (cta_in_group != 0 &&
      (params.force_single_cta || params.max_seq_len <= RADIX_THRESHOLD))
    return;

  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  uint32_t* shared_ordered =
      reinterpret_cast<uint32_t*>(smem_raw + kFixedSmemLarge);

  // RadixRowState for multi-CTA cooperative radix.
  // Zero-initialization is done host-side via cudaMemsetAsync in topk.cu
  // before launch — that gives a stream-ordered happens-before edge for all
  // CTAs, which the previous in-kernel init (CTA-0 only + intra-CTA
  // __syncthreads) did not provide and which manifested as a race against
  // CTA-1+'s first red_release on arrival_counter.
  RadixRowState* state = &params.row_states[group_id];

  int barrier_phase = 0;
  uint32_t radix_iter = 0;
  const uint32_t total_iters = (params.num_rows + num_groups - 1) / num_groups;

  for (uint32_t iter = 0; iter < total_iters; iter++) {
    // Static round-robin: all CTAs in the group implicitly agree on the row
    uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= params.num_rows) break;

    // Clamp the row length before any decision is made on it.
    //
    // `lengths` is int32 and is consumed here as uint32, so a negative value
    // (e.g. a padded decode slot whose per-token context length underflowed)
    // would reinterpret as ~4e9 and sail past every threshold below. Any
    // value beyond the row width would also read into the next row.
    //
    // Clamping to max_seq_len additionally keeps this per-row decision
    // consistent with the `cta_in_group != 0` early exit above, which is
    // taken from the host-side scalar: when max_seq_len <= RADIX_THRESHOLD
    // the non-leader CTAs return immediately, so a leader that reached the
    // cooperative radix path would wait on the inter-CTA barrier for peers
    // that no longer exist and spin until the kernel is killed.
    const int32_t raw_len = params.lengths[row_idx];
    const uint32_t row_bound =
        params.stride < params.max_seq_len ? params.stride : params.max_seq_len;
    const uint32_t non_negative_len =
        raw_len > 0 ? static_cast<uint32_t>(raw_len) : 0u;
    const uint32_t seq_len =
        non_negative_len < row_bound ? non_negative_len : row_bound;
    int32_t* row_output = params.output + row_idx * params.top_k;
    const float* row_input = params.input + row_idx * params.stride;

    // force_single_cta is the low-smem fallback: the cooperative launch does
    // not fit, so a single CTA runs the same deterministic select over the
    // whole row (uncached, hence slower) rather than deferring to a kernel
    // that does not guarantee ordering.
    if (params.force_single_cta || seq_len <= RADIX_THRESHOLD) {
      if (cta_in_group == 0) {
        if (seq_len <= static_cast<uint32_t>(TopK)) {
          // Trivial case: seq_len <= TopK
          for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
               i += kThreadsPerBlock) {
            row_output[i] = (i < seq_len) ? static_cast<int32_t>(i) : -1;
          }
        } else {
          // Single-CTA rows: rescanning select (exact pivot, index-ranked
          // ties, two ascending runs merged).
          det_select_row<TopK, kThreadsPerBlock>(
              row_input, static_cast<int>(seq_len), row_output, smem_raw,
              params.det_smem_bytes);
        }
      }
      continue;
    }

    const uint32_t my_chunk_start = cta_in_group * chunk_size;
    radix_topk<TopK, VEC_SIZE>(
        row_input, row_output, seq_len, my_chunk_start, chunk_size,
        local_histogram, suffix_sum, shared_scalars, shared_ordered, state,
        cta_in_group, ctas_per_group, barrier_phase, radix_iter, tx);
    radix_iter++;
  }
}

}  // namespace persistent

// ============================================================================
// ============================================================================
// Optimized FilteredTopK — single CTA per row for bs > 32.
// Kept with persistent_topk so the portable fallback owns the non-cluster path.
// ============================================================================
namespace filtered_topk {

// ============================================================================
// FilteredTopK — single CTA per row for bs > 32
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================

#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename T, size_t N>
struct vec_t {
  T data[N];

  FLASHINFER_INLINE T& operator[](size_t i) { return data[i]; }
  FLASHINFER_INLINE const T& operator[](size_t i) const { return data[i]; }

  FLASHINFER_INLINE void cast_load(const T* ptr) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      data[i] = ptr[i];
    }
  }
};
#undef FLASHINFER_INLINE

// FilteredTopK traits for different data types
template <typename DType>
struct FilteredTopKTraits;

// Specialization for float (32-bit): coarse histogram uses FP16 high 8 bits, 4
// refinement rounds
template <>
struct FilteredTopKTraits<float> {
  using OrderedType = uint32_t;
  static constexpr int NUM_REFINE_ROUNDS = 4;
  static constexpr int FIRST_REFINE_SHIFT = 24;

  __device__ __forceinline__ static uint8_t ToCoarseKey(float x) {
    // Convert to FP16 representation and extract high 8 bits
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
  }

  __device__ __forceinline__ static OrderedType ToOrdered(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  }
};

constexpr uint32_t FILTERED_TOPK_BLOCK_THREADS = 1024;
constexpr uint32_t FILTERED_TOPK_SMEM_INPUT_SIZE =
    16 * 1024;  // 16K indices per buffer
// 128 KB: the size of the two candidate buffers this path used to carry. Those
// buffers are gone, and the shared memory now caches the row for
// det_select_row, so the useful size is a function of the row width and the
// device -- not a constant. Kept as the minimum request so the request never
// shrinks below what the old code asked for.
constexpr size_t FILTERED_TOPK_SMEM_DYNAMIC =
    sizeof(int) * 2 * FILTERED_TOPK_SMEM_INPUT_SIZE;  // 128KB

/*!
 * \brief Filtered Top-K kernel for ragged sequences.
 *
 * \tparam DType Data type (float, half, nv_bfloat16)
 * \tparam IdType Index type (int32_t)
 * \tparam VEC_SIZE Vector size for input loads (1, 2, 4, or 8)
 */
template <typename DType, typename IdType, int VEC_SIZE, uint32_t MAX_K = 2048,
          bool UsePredicatedShortLoads = false>
__global__ void __launch_bounds__(FILTERED_TOPK_BLOCK_THREADS)
    FilteredTopKUnifiedKernel(const DType* __restrict__ input,
                              IdType* __restrict__ output,
                              const IdType* __restrict__ lengths,
                              uint32_t num_rows, uint32_t top_k,
                              uint32_t max_len, uint32_t max_seq_len,
                              uint32_t smem_bytes) {
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;

  const uint32_t bid = blockIdx.x;
  const int tx = threadIdx.x;

  if (bid >= num_rows) return;

  // Same row-bound contract as the persistent kernel: `max_len` is the row
  // pitch, `max_seq_len` the logical width, and a length outside
  // [0, min(pitch, logical)] must not be trusted -- an oversized lengths[bid]
  // would otherwise read past the row.
  const uint32_t row_bound = max_len < max_seq_len ? max_len : max_seq_len;
  const int raw_len = (lengths != nullptr) ? static_cast<int>(lengths[bid])
                                           : static_cast<int>(row_bound);
  const uint32_t non_negative_len =
      raw_len > 0 ? static_cast<uint32_t>(raw_len) : 0u;
  const int length = static_cast<int>(
      non_negative_len < row_bound ? non_negative_len : row_bound);
  const DType* score = input + bid * max_len;
  IdType* dst = output + bid * top_k;

  // Trivial case: length <= top_k
  if (length <= static_cast<int>(top_k)) {
    for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
      dst[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
    }
    return;
  }

  // Short path
  // Every filtered row (any length, including the launcher's occupancy
  // fallback) goes through the rescanning select. vLLM instantiates this
  // kernel for float only.
  static_assert(std::is_same<DType, float>::value,
                "FilteredTopKUnifiedKernel: deterministic path is float-only");
  extern __shared__ uint8_t _smem_reg[];
  vllm::persistent::det_select_row<static_cast<int>(MAX_K),
                                   static_cast<int>(BLOCK_SIZE)>(
      reinterpret_cast<const float*>(score), length,
      reinterpret_cast<int32_t*>(dst), _smem_reg, smem_bytes);
}

// Helper to compute GCD for VEC_SIZE selection
constexpr uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    uint32_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Compute optimal VEC_SIZE based on max_len and dtype
// Returns 1, 2, 4, or 8
template <typename DType>
constexpr int ComputeFilteredTopKVecSize(uint32_t max_len) {
  constexpr int MAX_VEC = 16 / sizeof(DType);  // 4 for float32, 8 for fp16/bf16
  // Use GCD to find largest power-of-2 divisor
  const uint32_t g = gcd(max_len, static_cast<uint32_t>(MAX_VEC));
  return static_cast<int>(g);
}

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKRaggedTransform(const DType* input,
                                        IdType* output_indices,
                                        const IdType* lengths,
                                        uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len, uint32_t max_seq_len,
                                        int max_smem_per_block,
                                        cudaStream_t stream = 0) {
  constexpr int MAX_VEC = 16 / sizeof(DType);

  // det_select_row re-reads the row from GLOBAL memory on each of its four
  // radix passes unless the row fits in shared memory. The old fixed 128 KB
  // request caches rows up to ~32K keys; every device that reaches this path
  // offers more (A100 163 KiB, H100/H200 227 KiB), and asking for it moves the
  // cutoff to ~41K / ~57K. Ask for what the widest row needs, capped by the
  // device, floored at the historical request. The caller passes the device's
  // sharedMemPerBlockOptin -- it already has it from get_device_prop().
  const int device_optin = max_smem_per_block;
  const uint32_t row_width = max_len < max_seq_len ? max_len : max_seq_len;
  size_t want = vllm::persistent::det_select_row_bytes<
      static_cast<int>(MAX_K), static_cast<int>(FILTERED_TOPK_BLOCK_THREADS)>(
      row_width);
  if (want < FILTERED_TOPK_SMEM_DYNAMIC) want = FILTERED_TOPK_SMEM_DYNAMIC;

  dim3 grid(num_rows);
  dim3 block(FILTERED_TOPK_BLOCK_THREADS);

  const int vec_size = ComputeFilteredTopKVecSize<DType>(max_len);

#define DISPATCH_VEC_SIZE(VS)                                                 \
  if (vec_size == VS) {                                                       \
    auto kernel =                                                             \
        FilteredTopKUnifiedKernel<DType, IdType, VS, MAX_K, (VS != MAX_VEC)>; \
    cudaFuncAttributes fa{};                                                  \
    FLASHINFER_CUDA_CALL(cudaFuncGetAttributes(&fa, kernel));                 \
    size_t cap = static_cast<size_t>(device_optin) > fa.sharedSizeBytes       \
                     ? static_cast<size_t>(device_optin) - fa.sharedSizeBytes \
                     : 0;                                                     \
    uint32_t smem_size = static_cast<uint32_t>(want < cap ? want : cap);      \
    void* args[] = {&input,     &output_indices, &lengths,    &num_rows,      \
                    &top_k_val, &max_len,        &max_seq_len, &smem_size};   \
    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(                                \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));     \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args,   \
                                          smem_size, stream));                \
    return cudaSuccess;                                                       \
  }

  DISPATCH_VEC_SIZE(1)
  DISPATCH_VEC_SIZE(2)
  DISPATCH_VEC_SIZE(4)
  if constexpr (MAX_VEC >= 8) {
    DISPATCH_VEC_SIZE(8)
  }
#undef DISPATCH_VEC_SIZE

  return cudaSuccess;
}

}  // namespace filtered_topk

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKRaggedTransform(const DType* input,
                                        IdType* output_indices,
                                        const IdType* lengths,
                                        uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len, uint32_t max_seq_len,
                                        int max_smem_per_block,
                                        cudaStream_t stream = 0) {
  return filtered_topk::FilteredTopKRaggedTransform<DType, IdType, MAX_K>(
      input, output_indices, lengths, num_rows, top_k_val, max_len, max_seq_len,
      max_smem_per_block,
      stream);
}

}  // namespace vllm

#endif  // PERSISTENT_TOPK_CUH_
