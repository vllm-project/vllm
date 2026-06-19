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
constexpr uint32_t RADIX_THRESHOLD = 32768;

// Decode path constants
constexpr int kDecodeBins = 2048;
constexpr uint32_t HIST2048_THRESHOLD = 8192;

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
                           uint32_t iter, uint32_t tx) {
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
  if (cta_in_group != 0 && params.max_seq_len <= RADIX_THRESHOLD) return;

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
  const uint32_t total_iters = (params.num_rows + num_groups - 1) / num_groups;

  for (uint32_t iter = 0; iter < total_iters; iter++) {
    // Static round-robin: all CTAs in the group implicitly agree on the row
    uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= params.num_rows) break;

    const uint32_t seq_len = params.lengths[row_idx];
    int32_t* row_output = params.output + row_idx * params.top_k;
    const float* row_input = params.input + row_idx * params.stride;

    if (seq_len <= RADIX_THRESHOLD) {
      if (cta_in_group == 0) {
        if (seq_len <= static_cast<uint32_t>(TopK)) {
          // Trivial case: seq_len <= TopK
          for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
               i += kThreadsPerBlock) {
            row_output[i] = (i < seq_len) ? static_cast<int32_t>(i) : -1;
          }
        } else if (seq_len <= static_cast<uint32_t>(HIST2048_THRESHOLD)) {
          histogram_2048_topk<TopK>(row_input, row_output, seq_len);
        } else {
          histogram_256_topk<TopK>(row_input, row_output, 0, seq_len);
        }
      }
      continue;
    }

    const uint32_t my_chunk_start = cta_in_group * chunk_size;
    radix_topk<TopK, VEC_SIZE>(
        row_input, row_output, seq_len, my_chunk_start, chunk_size,
        local_histogram, suffix_sum, shared_scalars, shared_ordered, state,
        cta_in_group, ctas_per_group, barrier_phase, iter, tx);
  }
}

}  // namespace persistent

// ============================================================================
// ============================================================================
// Optimized FilteredTopK — single CTA per row for bs > 32.
// Kept with persistent_topk so the portable fallback owns the non-cluster path.
// ============================================================================
namespace filtered_topk {

constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kHistBits = 10;
constexpr uint32_t kHistBins = 1 << kHistBits;
constexpr uint32_t RADIX = 256;
constexpr uint32_t kMaxTies = 1024;
static_assert(kMaxTies <= kBlockSize,
              "tie_handle requires kMaxTies <= kBlockSize");
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;

constexpr uint32_t kElemPerStage = 16;
constexpr uint32_t kSizePerStage = kElemPerStage * kBlockSize;  // 16384

// CS=4: 2 TMA stages (double buffer), two-pass
constexpr uint32_t kNumStages4 = 2;
// CS=8: 4 TMA stages, single-pass (all data stays in smem)
constexpr uint32_t kNumStages8 =
    2;  // 2 stages × 16K = 32K per block (enough for CS=8)
// Max data per block with CS=8: ceil(262144/8) = 32768 ≤ 4 × 8192 = 32768
constexpr uint32_t kMaxSeqLen8 = kNumStages8 * kSizePerStage * 8;  // 262144
constexpr uint32_t kNumStages16 = 2;  // double-buffer for CS=16
constexpr uint32_t kMaxSeqLen16 = kNumStages16 * kSizePerStage * 16;  // 524288

// Register path
constexpr uint32_t kHist4096Bits = 12;
constexpr uint32_t kHist4096Bins = 1 << kHist4096Bits;  // 4096
constexpr uint32_t kHist4096VecsPerThread = 4;
constexpr uint32_t kHist4096MaxLen =
    kHist4096VecsPerThread * 4 * kBlockSize;                     // 16384
constexpr uint32_t kHist4096Items = kHist4096Bins / kBlockSize;  // 4

// CS=4 single-pass path
constexpr uint32_t kMaxSinglePassStages = 3;
constexpr uint32_t kMaxSinglePassPerBlock =
    kMaxSinglePassStages * kSizePerStage;  // 49152

constexpr uint32_t kStreamNumStages = 2;

struct alignas(16) MatchBin {
  uint32_t bin, above_count, equal_count;
};
struct alignas(8) Tie {
  uint32_t idx;
  float score;
};

__device__ __forceinline__ void load_float4_predicated(const float* ptr,
                                                       int base, int seq_len,
                                                       float& v0, float& v1,
                                                       float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  const int p0 = (base < seq_len);
  const int p1 = (base + 1 < seq_len);
  const int p2 = (base + 2 < seq_len);
  const int p3 = (base + 3 < seq_len);
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

// converts the float32 score to a 32-bit ordered unsigned integer — the full
// precision key for radix sorting
__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Converts each score to a 12-bit bin (FP16 sign-magnitude -> top 12 bits ->
// bin 0-4095)
template <uint32_t kBits>
__device__ __forceinline__ uint32_t extract_coarse_bin_N(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

// running sum within each warp — thread 0 gets its own value, thread 1 gets
// thread 0 + thread 1, thread 2 gets threads 0+1+2, etc.
__device__ __forceinline__ uint32_t warp_inclusive_sum(uint32_t lane,
                                                       uint32_t v) {
#pragma unroll
  for (uint32_t o = 1; o < 32; o *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, v, o);
    if (lane >= o) v += n;
  }
  return v;
}

// Returns the sum of a value across all 32 threads in the warp, and every
// thread gets the same result SM90+ PTX instruction that does a hardware
// warp-wide reduction in a single instruction w.r.t. warp::reduce_sum(), which
// uses a __shfl_xor_sync butterfly tree (5 shuffles for 32 lanes)
__device__ __forceinline__ uint32_t warp_reduce_sum_full(uint32_t v) {
  uint32_t r;
  asm("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
  return r;
}

// ============================================================================
// Tie refinement (single CTA): 4-round radix-256 topK on the full FP32 ordered
// key Each round narrows by 8 bits until ties are fully resolved
// ============================================================================

template <uint32_t TopK>
__device__ void tie_handle(const Tie* ties, uint32_t num_ties,
                           uint32_t num_above, int32_t* output, void* _smem) {
  struct TS {
    alignas(128) uint32_t counter;
    alignas(128) MatchBin match;
    uint32_t histogram[RADIX];
    uint32_t warp_sum[kNumWarps];
  };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize, wi = tx / kWarpSize;

  // Each thread loads one tie element.
  const bool has = tx < num_ties;
  const auto tie = has ? ties[tx] : Tie{0, 0.0f};
  const uint32_t key = convert_to_uint32_v2(tie.score);

  bool active = has;  // tracks whether this thread's tie is still a candidate.
  uint32_t remain =
      TopK - num_above;  // decreases each round as ties are resolved.
  uint32_t wpos = TopK;  // wpos will hold the final output position.
  s->counter = 0;
  __syncthreads();

  // The 4-round radix loop - each round narrows by 8 bits until ties are fully
  // resolved
#pragma unroll
  for (int r = 0; r < 4; r++) {
    uint32_t sh = 24 - r * 8;  // round 0: bits 31-24, round 1: 23-16, etc.
    uint32_t bin = (key >> sh) & 0xFF;  // this tie's 8-bit bin for this round

    // Step 1: Build 256-bin histogram.
    if (tx < RADIX) s->histogram[tx] = 0;
    __syncthreads();
    if (active) atomicAdd(&s->histogram[bin], 1);
    __syncthreads();

    // Step 2: Prefix scan to find threshold
    uint32_t hv = 0, wi2 = 0;
    if (tx < RADIX) {
      hv = s->histogram[tx];
      wi2 = warp_inclusive_sum(li, hv);
      if (li == kWarpSize - 1) s->warp_sum[wi] = wi2;
    }
    __syncthreads();

    if (tx < RADIX) {
      auto tmp = (li < RADIX / kWarpSize) ? s->warp_sum[li] : 0;
      auto tot = warp_reduce_sum_full(tmp);
      auto inter = warp_reduce_sum_full(li < wi ? tmp : 0);
      auto above = tot - (inter + wi2);
      if (above < remain && above + hv >= remain) {
        s->match = {tx, above, remain - above};
      }
    }
    __syncthreads();

    // Step 3: Scatter
    auto [thr, na, _] = s->match;  // threshold bin, num above, unused
    if (active) {
      if (bin > thr) {
        wpos = num_above +
               atomicAdd(&s->counter, 1);  // above -> place in output directly
        active = false;
      } else if (bin < thr)
        active = false;  // below -> discard
      else if (r == 3)
        wpos = TopK - atomicAdd(&s->match.equal_count,
                                -1u);  // last round: place remaining
    }
    remain -= na;
    if (!remain) break;  // all ties resolved early
  }
  // Final write
  if (wpos < TopK) output[wpos] = tie.idx;
}

// Extended tie_handle for TopK > kBlockSize (e.g. TopK=2048).
// tie_handle assumes 1 tie per thread (max 1024).
// This version handles 2 ties per thread via kPerThread=2
template <uint32_t TopK>
__device__ void tie_handle_large(const Tie* ties, uint32_t num_ties,
                                 uint32_t num_above, int32_t* output,
                                 void* _smem) {
  static_assert(TopK > kBlockSize);
  struct TS {
    alignas(128) uint32_t counter;
    alignas(128) MatchBin match;
    uint32_t histogram[RADIX];
    uint32_t warp_sum[kNumWarps];
  };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize;
  const auto wi = tx / kWarpSize;

  constexpr uint32_t kPerThread = (TopK + kBlockSize - 1) / kBlockSize;
  Tie my_ties[kPerThread];
  uint32_t keys[kPerThread];
  bool active[kPerThread];

  for (uint32_t e = 0; e < kPerThread; e++) {
    uint32_t idx = e * kBlockSize + tx;
    if (idx < num_ties) {
      my_ties[e] = ties[idx];
      keys[e] = convert_to_uint32_v2(ties[idx].score);
      active[e] = true;
    } else {
      my_ties[e] = {0, 0.0f};
      keys[e] = 0;
      active[e] = false;
    }
  }

  uint32_t remain = TopK - num_above;
  s->counter = 0;
  __syncthreads();

  for (int r = 0; r < 4; r++) {
    uint32_t sh = 24 - r * 8;
    if (tx < RADIX) {
      s->histogram[tx] = 0;
    }
    __syncthreads();

    for (uint32_t e = 0; e < kPerThread; e++) {
      if (active[e]) {
        atomicAdd(&s->histogram[(keys[e] >> sh) & 0xFF], 1);
      }
    }
    __syncthreads();

    uint32_t hv = 0;
    if (tx < RADIX) {
      hv = s->histogram[tx];
      auto wi2 = warp_inclusive_sum(li, hv);
      if (li == kWarpSize - 1) {
        s->warp_sum[wi] = wi2;
      }
    }
    __syncthreads();
    if (tx < RADIX) {
      auto tmp2 = (li < RADIX / kWarpSize) ? s->warp_sum[li] : 0;
      auto total = warp_reduce_sum_full(tmp2);
      auto inter = warp_reduce_sum_full(li < wi ? tmp2 : 0);
      auto wi2 = warp_inclusive_sum(li, hv);
      auto above = total - (inter + wi2);
      if (above < remain && above + hv >= remain) {
        s->match = {
            .bin = tx, .above_count = above, .equal_count = remain - above};
      }
    }
    __syncthreads();

    auto thr = s->match.bin;
    auto na = s->match.above_count;

    for (uint32_t e = 0; e < kPerThread; e++) {
      if (!active[e]) {
        continue;
      }
      uint32_t bin = (keys[e] >> sh) & 0xFF;
      if (bin > thr) {
        uint32_t wpos = num_above + atomicAdd(&s->counter, 1);
        if (wpos < TopK) {
          output[wpos] = my_ties[e].idx;
        }
        active[e] = false;
      } else if (bin < thr) {
        active[e] = false;
      } else if (r == 3) {
        uint32_t wpos = TopK - atomicAdd(&s->match.equal_count, -1u);
        if (wpos < TopK) {
          output[wpos] = my_ties[e].idx;
        }
      }
    }

    num_above += na;
    remain -= na;
    __syncthreads();
    s->counter = 0;
    __syncthreads();
  }
}

// ============================================================================
// Register-based single-CTA fast path for seq_len <= 16384
// 4 float4 per thread × 1024 threads = 16384 elements max
// Uses 4096-bin (12-bit) histogram for better precision
// ============================================================================

template <uint32_t TopK, uint32_t HIST_BITS>
struct Histogram4096Smem {
  static constexpr uint32_t HIST_BINS = 1 << HIST_BITS;
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBin match;
  uint32_t warp_sum[kNumWarps];
  union {
    uint32_t histogram[HIST_BINS];
    Tie tie_buffer[kMaxTies];
  };
  static_assert(sizeof(uint32_t) * HIST_BINS >= TopK * sizeof(Tie),
                "histogram union must be large enough for TopK ties");
};

template <uint32_t TopK, uint32_t HIST_BITS,
          uint32_t VECS_PER_THREAD = kHist4096VecsPerThread,
          bool UsePredicatedLoads = false>
__device__ void histogram_4096_topk(const float* __restrict__ scores,
                                    int32_t* __restrict__ output,
                                    uint32_t length, void* _smem) {
  constexpr uint32_t HIST_BINS = 1 << HIST_BITS;
  constexpr uint32_t ITEMS_PER_THREAD = HIST_BINS / kBlockSize;
  static_assert(HIST_BINS >= kBlockSize,
                "HIST_BITS must give >= kBlockSize bins");

  using Smem = Histogram4096Smem<TopK, HIST_BITS>;
  auto* smem = static_cast<Smem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpSize;
  const auto warp_id = tx / kWarpSize;

  // Phase 1: Load all data into RF + build histogram
  float4
      vecs[VECS_PER_THREAD];  // 4 vectors x 4 floats = 16 elements per thread
  if constexpr (ITEMS_PER_THREAD >= 4) {
    // Zero the histogram (SMEM writes)
    for (uint32_t i = 0; i < ITEMS_PER_THREAD / 4; i++)
      reinterpret_cast<uint4*>(
          smem->histogram)[tx * (ITEMS_PER_THREAD / 4) + i] =
          make_uint4(0, 0, 0, 0);
  } else {
    if (tx < HIST_BINS) smem->histogram[tx] = 0;
  }
  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  if constexpr (UsePredicatedLoads) {
    const bool row_aligned = (reinterpret_cast<uintptr_t>(scores) & 0xFu) == 0;
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
      const uint32_t base = (tx + v * kBlockSize) * 4;
      if (base < length) {
        if (row_aligned && base + 3 < length) {
          vecs[v] = *reinterpret_cast<const float4*>(scores + base);
        } else {
          load_float4_predicated(scores + base, static_cast<int>(base),
                                 static_cast<int>(length), vecs[v].x, vecs[v].y,
                                 vecs[v].z, vecs[v].w);
        }
      }
    }
  } else {
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
      const uint32_t base = (tx + v * kBlockSize) * 4;
      if (base < length) {
        vecs[v] = *reinterpret_cast<const float4*>(scores + base);
      }
    }
  }
  __syncthreads();

  // Build histogram from RF via atomic adds into the shared histogram
  bool done = false;
#pragma unroll
  for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
    const float* elems = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
    for (uint32_t e = 0; e < 4 && !done; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) {
        done = true;
      } else {
        atomicAdd(&smem->histogram[extract_coarse_bin_N<HIST_BITS>(elems[e])],
                  1);
      }
    }
  }
  __syncthreads();

  // Phase 2: Prefix scan to find threshold bin
  // Multi-element scan (4096 bins: 4 per thread)
  uint32_t orig[ITEMS_PER_THREAD];
  uint32_t local_sum = 0;

  // Step 1: Each thread sums its 4 bins
#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    orig[i] = smem->histogram[tx * ITEMS_PER_THREAD + i];
    local_sum += orig[i];
  }

  // Step 2: Warp-level inclusive prefix sum on local_sum
  const auto warp_inc = warp_inclusive_sum(lane_id, local_sum);
  if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
  __syncthreads();

  // Step 3: Inter-warp prefix via redux.sync
  const auto tmp = smem->warp_sum[lane_id];
  uint32_t prefix = warp_reduce_sum_full(
      lane_id < warp_id ? tmp : 0);  // sum of all prior warps
  prefix +=
      warp_inc - local_sum;  // exclusive prefix within this thread's position

  // Step 4: Find threshold - scan 4 bins, accumulate prefix
#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    prefix += orig[i];
    const auto above = length - prefix;  // elements in bins ABOVE this one
    if (above < TopK && above + orig[i] >= TopK) {
      smem->match = {.bin = tx * ITEMS_PER_THREAD + i,
                     .above_count = above,
                     .equal_count = orig[i]};
    }
  }

  __syncthreads();

  // Phase 3: Scatter from registers
  const auto [thr_bin, num_above, num_equal] = smem->match;
  const bool need_tie = (num_equal + num_above > TopK);

  done = false;
#pragma unroll
  for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
    const float* elems = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
    for (uint32_t e = 0; e < 4 && !done; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) {
        done = true;
      } else {
        const uint32_t bin = extract_coarse_bin_N<HIST_BITS>(elems[e]);
        if (bin > thr_bin) {
          output[atomicAdd(&smem->counter_gt, 1)] =
              idx;  // above -> output directly
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (!need_tie) {
            if (pos + num_above < TopK) {
              output[pos + num_above] = idx;  // all fit
            }
          } else {
            if (pos < TopK) {
              smem->tie_buffer[pos] = {idx, elems[e]};  // store for refirement
            }
          }
        }
        // else: bin < thr_bin - discard (not in top-k)
      }
    }
  }

  // Phase 4: Tie-breaking
  if (!need_tie) return;
  __syncthreads();

  // Fast warp-ballot tie-breaking for small tie counts
  const uint32_t num_ties = min(num_equal, static_cast<uint32_t>(TopK));
  const uint32_t topk_remain =
      TopK - num_above;  // pick exactly remaining elements to fill topK

  auto is_greater = [](const Tie& a, const Tie& b) {
    return (a.score > b.score) || (a.score == b.score && a.idx < b.idx);
  };

  if (num_ties <= kWarpSize) {
    // <=32 ties - Use warp ballot
    // All-to-all comparison in one __ballot_sync. 32 ties x 32 warps = 1024
    // comparisons in one instruction per warp. O(1) work.
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    if (lane_id >= num_ties || warp_id >= num_ties) return;
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = smem->tie_buffer[lane_id];  // each lane holds one tie
    const auto target =
        smem->tie_buffer[warp_id];  // each warp evaluates one candidate
    const bool pred =
        is_greater(tie, target);  // compare all ties against target
    const auto rank = static_cast<uint32_t>(
        __popc(__ballot_sync(mask, pred)));  // count how many are greater
    if (lane_id == 0 && rank < topk_remain) {
      output[num_above + rank] = target.idx;  // place at correct position
    }
  } else if (num_ties <=
             kWarpSize *
                 2) {  // TODO (roberto): try to refactor this with <=32 case
    //  Same idea but each thread handles 2 tie elements
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    const auto lane1 = lane_id + kWarpSize;
    const auto warp1 = warp_id + kWarpSize;
    const auto invalid = Tie{0xFFFFFFFF, -__FLT_MAX__};
    const auto tie0 = smem->tie_buffer[lane_id];
    const auto tie1 = lane1 < num_ties ? smem->tie_buffer[lane1] : invalid;
    if (warp_id < num_ties) {
      const auto target = smem->tie_buffer[warp_id];
      const auto r0 =
          __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie0, target)));
      const auto r1 =
          __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie1, target)));
      if (lane_id == 0 && r0 + r1 < topk_remain)
        output[num_above + r0 + r1] = target.idx;
    }
    if (warp1 < num_ties) {
      const auto target = smem->tie_buffer[warp1];
      const auto r0 =
          __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie0, target)));
      const auto r1 =
          __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie1, target)));
      if (lane_id == 0 && r0 + r1 < topk_remain)
        output[num_above + r0 + r1] = target.idx;
    }
  } else {
    // Large tie count: fall back to 4-round radix-256 sort
    if constexpr (TopK <= kBlockSize) {
      tie_handle<TopK>(smem->tie_buffer, num_ties, num_above, output, smem);
    } else {
      tie_handle_large<TopK>(smem->tie_buffer, num_ties, num_above, output,
                             smem);
    }
  }
}

template <uint32_t TopK, uint32_t HIST_BITS,
          uint32_t VECS_PER_THREAD = kHist4096VecsPerThread>
__device__ __noinline__ void histogram_4096_topk_predicated(
    const float* __restrict__ scores, int32_t* __restrict__ output,
    uint32_t length, void* _smem) {
  histogram_4096_topk<TopK, HIST_BITS, VECS_PER_THREAD, true>(scores, output,
                                                              length, _smem);
}

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

  FLASHINFER_INLINE void cast_store(T* ptr) const {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      ptr[i] = data[i];
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
                              uint32_t max_len) {
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
  constexpr int RADIX = 256;
  constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;

  const uint32_t bid = blockIdx.x;
  const int tx = threadIdx.x;

  if (bid >= num_rows) return;

  const int length =
      (lengths != nullptr) ? lengths[bid] : static_cast<int>(max_len);
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
  if (length <= 32768) {
    extern __shared__ uint8_t _smem_reg[];
    if constexpr (UsePredicatedShortLoads) {
      histogram_4096_topk_predicated<MAX_K, 12, 8>(score, dst, length,
                                                   _smem_reg);
    } else {
      histogram_4096_topk<MAX_K, 12, 8>(score, dst, length, _smem_reg);
    }
    return;
  }

  // Static shared memory
  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_indices[MAX_K];

  auto& s_histogram = s_histogram_buf[0];

  // Dynamic shared memory for input double buffer
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  using Traits = FilteredTopKTraits<DType>;
  int topk = top_k;

  // Stage 1: 8-bit coarse histogram with vectorized loads
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  vec_t<DType, VEC_SIZE> score_vec;

  const int aligned_length = (length / VEC_SIZE) * VEC_SIZE;
#pragma unroll 2
  for (int base = tx * VEC_SIZE; base < aligned_length;
       base += BLOCK_SIZE * VEC_SIZE) {
    score_vec.cast_load(&score[base]);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const auto bin = Traits::ToCoarseKey(score_vec[j]);
      atomicAdd(&s_histogram[bin], 1);
    }
  }
  // Handle tail
  for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
    const auto bin = Traits::ToCoarseKey(score[i]);
    atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  // Suffix sum
  const auto run_cumsum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  constexpr int NUM_ROUNDS = Traits::NUM_REFINE_ROUNDS;
  constexpr int FIRST_SHIFT = Traits::FIRST_REFINE_SHIFT;

  if (topk == 0) {
    // Collect indices where bin > threshold
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length;
         base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const auto bin = static_cast<int>(Traits::ToCoarseKey(score_vec[j]));
        if (bin > threshold_bin) {
          const auto pos = atomicAdd(&s_counter, 1);
          s_indices[pos] = base + j;
        }
      }
    }
    // Handle tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(score[i]));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = i;
      }
    }
    __syncthreads();
  } else {
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    // Filter + histogram for refinement
    auto filter_and_add_to_histogram = [&](auto raw_input, int index) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = index;
      } else if (bin == threshold_bin) {
        const auto pos = atomicAdd(&s_num_input[0], 1);
        if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
          s_input_idx[0][pos] = index;
          const auto ordered = Traits::ToOrdered(raw_input);
          const auto sub_bin = (ordered >> FIRST_SHIFT) & 0xFF;
          atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    };
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length;
         base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        filter_and_add_to_histogram(score_vec[j], base + j);
      }
    }
    // Handle tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      filter_and_add_to_histogram(score[i], i);
    }
    __syncthreads();

    // Stage 2: refine with 8bit radix passes
#pragma unroll
    for (int round = 0; round < NUM_ROUNDS; ++round) {
      __shared__ int s_last_remain;
      const auto r_idx = round % 2;

      const auto _raw_num_input = s_num_input[r_idx];
      const auto num_input =
          (_raw_num_input < SMEM_INPUT_SIZE) ? _raw_num_input : SMEM_INPUT_SIZE;

      run_cumsum();
      if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
        s_threshold_bin_id = tx;
        s_num_input[r_idx ^ 1] = 0;
        s_last_remain = topk - s_histogram[tx + 1];
      }
      __syncthreads();

      const auto threshold = s_threshold_bin_id;
      topk -= s_histogram[threshold + 1];

      const int offset = FIRST_SHIFT - round * 8;
      const bool is_last_round = (round == NUM_ROUNDS - 1);

      if (topk == 0) {
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto bin = (Traits::ToOrdered(score[idx]) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          }
        }
        __syncthreads();
        break;
      } else {
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto raw_input = score[idx];
          const auto bin = (Traits::ToOrdered(raw_input) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          } else if (static_cast<int>(bin) == threshold) {
            if (is_last_round) {
              const auto pos = atomicAdd(&s_last_remain, -1);
              if (pos > 0) {
                s_indices[top_k - pos] = idx;
              }
            } else {
              const auto pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
              if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
                s_input_idx[r_idx ^ 1][pos] = idx;
                const auto bin32 = Traits::ToOrdered(raw_input);
                const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
                atomicAdd(&s_histogram[sub_bin], 1);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

  // Output phase - mode-specific
#pragma unroll 2
  for (int base = tx; base < static_cast<int>(top_k); base += BLOCK_SIZE) {
    const int idx = s_indices[base];
    dst[base] = static_cast<IdType>(idx);
  }
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
                                        uint32_t max_len,
                                        cudaStream_t stream = 0) {
  constexpr size_t smem_size = FILTERED_TOPK_SMEM_DYNAMIC;
  constexpr int MAX_VEC = 16 / sizeof(DType);

  dim3 grid(num_rows);
  dim3 block(FILTERED_TOPK_BLOCK_THREADS);
  void* args[] = {&input,    &output_indices, &lengths,
                  &num_rows, &top_k_val,      &max_len};

  const int vec_size = ComputeFilteredTopKVecSize<DType>(max_len);

#define DISPATCH_VEC_SIZE(VS)                                                 \
  if (vec_size == VS) {                                                       \
    auto kernel =                                                             \
        FilteredTopKUnifiedKernel<DType, IdType, VS, MAX_K, (VS != MAX_VEC)>; \
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
                                        uint32_t max_len,
                                        cudaStream_t stream = 0) {
  return filtered_topk::FilteredTopKRaggedTransform<DType, IdType, MAX_K>(
      input, output_indices, lengths, num_rows, top_k_val, max_len, stream);
}

}  // namespace vllm

#endif  // PERSISTENT_TOPK_CUH_
