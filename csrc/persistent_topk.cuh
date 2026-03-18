/*
 * Persistent TopK Scheduler for DSA Indexer
 *
 * Single persistent kernel with dynamic per-row path selection:
 *   - Trivial  (seq_len <= TopK):  direct index copy
 *   - Decode   (seq_len <= 8K):    2048-bin FP16 histogram + FP32 radix refine
 *   - Medium   (seq_len <= 64K):   256-bin FP16 histogram + FP32 radix refine
 *   - Large    (seq_len > 64K):    multi-CTA cooperative radix select
 *
 * CUDAGraph-safe: fixed grid configuration handles all seq_lens.
 * The grid is always set up for the large path (worst case). For non-large
 * rows, only CTA 0 of each group does work — others skip with no barrier
 * overhead. The inter-CTA barrier protocol naturally handles timing
 * differences between fast-skipping and working CTAs.
 */

#ifndef PERSISTENT_TOPK_CUH_
#define PERSISTENT_TOPK_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>

namespace vllm {
namespace persistent {

// ============================================================================
// Constants
// ============================================================================

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr int RADIX = 256;
constexpr size_t kSmemMedium = 8 * 1024 * sizeof(uint32_t);          // 32KB
constexpr int MAX_BUFFERED_ITEMS = kSmemMedium / (2 * sizeof(int));  // 4096
constexpr uint32_t LARGE_THRESHOLD = 65536;                          // 64K

// Decode path constants
constexpr int kDecodeBins = 2048;
constexpr uint32_t DECODE_THRESHOLD = 8192;

// Large path: fixed shared memory for histograms + scalars
constexpr size_t kFixedSmemLarge =
    ((RADIX + RADIX + 5) * sizeof(uint32_t) + 15) & ~size_t(15);

// ============================================================================
// Common helpers
// ============================================================================

__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
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
// Large path: inter-CTA coordination state (one per group)
// ============================================================================

struct RadixRowState {
  uint32_t histogram[3][256];  // Triple-buffered histograms
  uint32_t remaining_k;
  uint32_t prefix;
  int arrival_counter;
  int output_counter;
};

// ============================================================================
// Kernel parameters
// ============================================================================

struct PersistentTopKParams {
  const float* __restrict__ input;  // [num_rows, stride]
  int32_t* __restrict__ output;     // [num_rows, TopK]
  int32_t* __restrict__ lengths;    // [num_rows]
  RadixRowState* row_states;        // large path: per-group state
  uint32_t num_rows;
  uint32_t stride;
  uint32_t chunk_size;      // large path: elements per CTA
  uint32_t ctas_per_group;  // 1=medium, >1=large
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

__device__ __noinline__ void decode_topk_cuda(
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
      (DECODE_THRESHOLD + kThreadsPerBlock - 1) / kThreadsPerBlock;

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

__device__ __noinline__ void fast_topk_cuda_tl(const float* __restrict__ logits,
                                               int* __restrict__ output_indices,
                                               int logits_offset, int seq_len) {
  alignas(128) __shared__ int shared_histogram[2][RADIX + 128];
  alignas(128) __shared__ int shared_output_count;
  alignas(128) __shared__ int shared_threshold_bin;
  alignas(128) __shared__ int shared_buffered_count[2];

  extern __shared__ int buffered_indices[][MAX_BUFFERED_ITEMS];

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
    __shared__ int shared_final_k;
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
// Inter-CTA sync primitives (from FlashInfer)
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

template <uint32_t VEC_SIZE>
__device__ void large_topk_cuda(PersistentTopKParams params) {
  const uint32_t ctas_per_group = params.ctas_per_group;
  const uint32_t group_id = blockIdx.x / ctas_per_group;
  const uint32_t cta_in_group = blockIdx.x % ctas_per_group;
  const uint32_t num_groups = gridDim.x / ctas_per_group;
  const uint32_t tx = threadIdx.x;
  const uint32_t chunk_size = params.chunk_size;

  if (blockIdx.x >= num_groups * ctas_per_group) return;

  // -- Dynamic shared memory layout --
  extern __shared__ uint8_t smem_raw[];
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  uint32_t* shared_ordered =
      reinterpret_cast<uint32_t*>(smem_raw + kFixedSmemLarge);

  RadixRowState* state = &params.row_states[group_id];

  // -- Initialize state (CTA 0 of each group) --
  if (cta_in_group == 0) {
    for (uint32_t buf = 0; buf < 3; buf++) {
      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        state->histogram[buf][i] = 0;
      }
    }
    if (tx == 0) {
      state->remaining_k = 0;
      state->prefix = 0;
      state->arrival_counter = 0;
      state->output_counter = 0;
    }
  }
  __syncthreads();

  int barrier_phase = 0;
  const uint32_t total_iters = (params.num_rows + num_groups - 1) / num_groups;

  for (uint32_t iter = 0; iter < total_iters; iter++) {
    const uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= params.num_rows) break;

    const uint32_t seq_len = params.lengths[row_idx];
    int32_t* row_output = params.output + row_idx * TopK;
    const float* row_input = params.input + row_idx * params.stride;

    // -- Non-large rows: only CTA 0 processes, no barriers needed --
    if (seq_len <= LARGE_THRESHOLD) {
      if (cta_in_group == 0) {
        if (seq_len <= static_cast<uint32_t>(TopK)) {
          for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
               i += kThreadsPerBlock) {
            row_output[i] = (i < seq_len) ? static_cast<int32_t>(i) : -1;
          }
        } else if (seq_len <= static_cast<uint32_t>(DECODE_THRESHOLD)) {
          decode_topk_cuda(row_input, row_output, seq_len);
        } else {
          fast_topk_cuda_tl(row_input, row_output, 0, seq_len);
        }
      }
      // Clean histogram for potential next large row
      if (cta_in_group == 0) {
        uint32_t next_hist_idx = ((iter + 1) * 4) % 3;
        for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
          state->histogram[next_hist_idx][i] = 0;
        }
      }
      continue;
    }

    // -- Compute this CTA's chunk bounds --
    const uint32_t my_chunk_start = cta_in_group * chunk_size;
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

    if (cta_in_group == 0 && tx == 0) {
      st_release(&state->output_counter, 0);
    }

    // -- Stage 2: 4 rounds of radix select --
    for (uint32_t round = 0; round < 4; round++) {
      const uint32_t global_round = iter * 4 + round;
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
    if (tx == 0) {
      local_histogram[0] = 0;
      if (local_gt_count > 0) {
        local_histogram[1] =
            atomicAdd(&state->output_counter, static_cast<int>(local_gt_count));
      }
    }
    __syncthreads();

    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      if (shared_ordered[i] > ordered_pivot) {
        uint32_t local_pos = atomicAdd(&local_histogram[0], 1);
        int pos = static_cast<int>(local_histogram[1]) + local_pos;
        row_output[pos] = static_cast<int32_t>(my_chunk_start + i);
      }
    }

    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    wait_ge(&state->arrival_counter,
            (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
    barrier_phase++;
    __syncthreads();

    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      if (shared_ordered[i] == ordered_pivot) {
        int pos = atomicAdd(&state->output_counter, 1);
        if (pos < TopK) {
          row_output[pos] = static_cast<int32_t>(my_chunk_start + i);
        }
      }
    }
  }  // end row loop

  // -- Cleanup: CTA 0 resets state --
  if (cta_in_group == 0) {
    for (uint32_t buf = 0; buf < 3; buf++) {
      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        state->histogram[buf][i] = 0;
      }
    }
    if (tx == 0) {
      st_release(&state->arrival_counter, 0);
    }
  }
}

// ============================================================================
// Persistent kernel — unified, CUDAGraph-safe
// ============================================================================

template <uint32_t VEC_SIZE = 1>
__global__ void __launch_bounds__(kThreadsPerBlock)
    persistent_topk_kernel(PersistentTopKParams params) {
  large_topk_cuda<VEC_SIZE>(params);
}

}  // namespace persistent
}  // namespace vllm

#endif  // PERSISTENT_TOPK_CUH_
