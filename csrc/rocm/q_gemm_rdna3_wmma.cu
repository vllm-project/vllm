// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 GPTQ WMMA prefill kernel for AMD RDNA3 (gfx1100). This is the
// matrix-instruction path for M >= 16; small-M decode lives in the sibling
// file q_gemm_rdna3.cu and is exposed via a different op
// (`gptq_gemm_rdna3`). Keeping the two paths in separate translation units
// is intentional: an earlier attempt at putting WMMA in the same TU as the
// scalar dot-product kernel introduced a compile-time interaction that
// silently miscompiled the M=1 path even though the WMMA template was never
// instantiated for M=1. Hipcc's optimizer appears to scope some decisions
// at the TU level (likely register file / SGPR pressure heuristics across
// all kernels in the TU), so we isolate.
//
// Hardware notes (RDNA3 / gfx1100 / RX 7900 XTX):
//   * v_wmma_f32_16x16x16_{f16,bf16}_w32 — 16×16×16 GEMM in one instruction
//     (~16 cycles per WMMA on wave32). Accumulator dtype is FP32; inputs
//     are fp16 or bf16. There is NO 16x16x16 with fp16/bf16 accumulator on
//     gfx11 that we'd want here (we always need fp32 accum to avoid loss
//     across many K iterations).
//   * Wave32 input fragment storage is "doubled" — lanes 16..31 hold a
//     copy of lanes 0..15 for the A and B fragments. The output C
//     fragment uses a different mapping: lane t holds COLUMN n=lane_lo
//     of the 16x16 output, with 8 elements alternating M rows by lane_hi
//     (lanes 0..15 = even rows, lanes 16..31 = odd rows). See the layout
//     diagram on `gemm_q4_wmma_kernel_16x16_1w` below for the full mapping.
//   * No native v_global_atomic_pk_add_{f16,bf16} on gfx11; the K-split
//     epilogue (gridDim.z > 1) emulates packed atomic add via a CAS-32
//     retry loop on a uint32 word covering 2 fp16/bf16 lanes. Within a
//     block we shuffle adjacent lanes via shfl_xor first so each pair of
//     output cols goes through a single atomic — no intra-block
//     contention. K_SPLIT == 1 keeps the original direct-write path with
//     each block owning its 16M × 16N output tile.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "qdq_4_rdna3.cuh"

#if defined(__HIPCC__) && defined(__gfx1100__)
  #define __HIP__RDNA3__
#endif

namespace vllm {
namespace gptq_rdna3_wmma {

// Pull dequant types from the sibling namespace.
using vllm::gptq_rdna3::bf162_t;
using vllm::gptq_rdna3::bf16_t;

// Device code below uses RDNA3-only __builtin_amdgcn_wmma_* intrinsics;
// non-RDNA3 device passes fall through to empty __global__ stubs at the
// #else block at the end of this TU.
#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// PRECISE dequant variants live HERE, not in the shared qdq_4_rdna3.cuh
// header. Reason: hipcc takes different register/scheduling decisions when
// the shared header grows, which caused a measurable decode-tk/s regression
// in the scalar kernel even when these new functions were never called from
// it. Keeping them in the WMMA TU only restores scalar's binary identity
// to its tuned baseline.
//
// Numerics: the classic fp16 bit-trick (FMA form: q*scale +
// (-(1024+zero)*scale)) loses up to ~0.025 per cell at scale=0.1 because
// fp16(scale) ≠ scale and the FMA amplifies that by 1024× without cancelling
// against the precomputed (1024+zero)*scale (which is rounded to fp16 BEFORE
// the FMA).
//
// Fix: subtract (1024+zero) as an integer FIRST — exact in fp16 because
// integers in [1024, 2047] are exactly representable — then multiply by
// scale, incurring at most one half-ULP rounding. Costs one extra
// instruction per dequant pair (sub+mul vs single FMA), worth it for the
// WMMA path because K can be small (16) and errors don't average.
__forceinline__ __device__ void prep_zero_scale_fp16_precise(uint32_t zero,
                                                             half scale,
                                                             half2& z_prep,
                                                             half2& y_prep) {
  union {
    uint16_t u;
    half h;
  } zu;
  zu.u = (uint16_t)(0x6400 | zero);
  z_prep = __half2half2(zu.h);
  y_prep = __half2half2(scale);
}

__forceinline__ __device__ void dequant_4bit_8_fp16_precise(uint32_t qa,
                                                            half2 (&dq)[4],
                                                            half2 z_prep,
                                                            half2 y_prep) {
  const uint32_t c0 = 0x64006400;
  union {
    uint32_t u;
    half2 h2;
  } q0, q1, q2, q3;
  q0.u = ((qa >> 0) & 0x000F000F) | c0;
  q1.u = ((qa >> 4) & 0x000F000F) | c0;
  q2.u = ((qa >> 8) & 0x000F000F) | c0;
  q3.u = ((qa >> 12) & 0x000F000F) | c0;
  dq[0] = __hmul2(__hsub2(q0.h2, z_prep), y_prep);
  dq[1] = __hmul2(__hsub2(q1.h2, z_prep), y_prep);
  dq[2] = __hmul2(__hsub2(q2.h2, z_prep), y_prep);
  dq[3] = __hmul2(__hsub2(q3.h2, z_prep), y_prep);
}

// fp32 prep for the bf16→bf16 fp32-internal dequant. Returns:
//   z_prep = -(128 + zero) * scale   (folded bias for FMA)
//   y_prep = scale
// Per-element FMA `q*y_prep + z_prep` then yields scale * (nibble - zero).
__forceinline__ __device__ void prep_zero_scale_bf16_f32(uint32_t zero,
                                                         bf16_t scale,
                                                         float& z_prep,
                                                         float& y_prep) {
  float scale_f = __bfloat162float(scale);
  z_prep = -(128.0f + (float)zero) * scale_f;
  y_prep = scale_f;
}

// fp32 → bf16 narrow that skips the defensive NaN-canonicalisation hipcc
// emits for __float2bfloat16. Round-half-to-even via add (0x7FFF + lsb)
// + truncate; no NaN check, since dequant outputs are bounded products
// of (nibble - zero) ∈ [-15, 15] and bf16 scales — never NaN/Inf.
__forceinline__ __device__ bf16_t f32_to_bf16_no_canon(float f) {
  uint32_t fu = __float_as_uint(f);
  uint32_t lsb = (fu >> 16) & 1u;
  uint16_t out_u = (uint16_t)((fu + 0x7FFFu + lsb) >> 16);
  bf16_t out;
  __builtin_memcpy(&out, &out_u, sizeof(out));
  return out;
}

// 4-bit GPTQ dequant for the bf16 WMMA path: 8 nibbles per int32 in qa,
// outputs bf162_t dq[4]. Implementation rationale:
//
// gfx11 has no v_pk_fma_bf16, so __hmul2/__hsub2 on bf16 lower to a
// widen-fp32-op-narrow chain that hipcc decorates with NaN canonicalisation
// (28 extra VALU ops per call observed in the v5 ISA dump: v_cmp_u_f32 +
// v_cndmask_b32 around every bf16 sub/mul). Doing the math in fp32 directly
// — bit-cast widening (`__uint_as_float((bf16_bits) << 16)` is just a shift,
// no NaN canon), one fused FMA per element, single `__float2bfloat16` narrow
// at the end — eliminates the canon and is also strictly more precise (one
// rounding step instead of two).
//
// Inputs match prep_zero_scale_bf16_f32:
//   z_prep = -(128 + zero) * scale   (folded bias for FMA)
//   y_prep = scale
// Per-element FMA `q*y_prep + z_prep` then yields scale * (nibble - zero).
__forceinline__ __device__ void dequant_4bit_8_bf16_to_bf16(uint32_t qa,
                                                            bf162_t (&dq)[4],
                                                            float z_prep,
                                                            float y_prep) {
  const uint32_t c0 = 0x43004300;
  const uint32_t q0 = ((qa >> 0) & 0x000F000F) | c0;
  const uint32_t q1 = ((qa >> 4) & 0x000F000F) | c0;
  const uint32_t q2 = ((qa >> 8) & 0x000F000F) | c0;
  const uint32_t q3 = ((qa >> 12) & 0x000F000F) | c0;
  // bf16(128+nibble) bits → fp32 via left-shift by 16 (zero-extends mantissa).
  const float q0x = __uint_as_float((q0 & 0xFFFFu) << 16);
  const float q0y = __uint_as_float(q0 & 0xFFFF0000u);
  const float q1x = __uint_as_float((q1 & 0xFFFFu) << 16);
  const float q1y = __uint_as_float(q1 & 0xFFFF0000u);
  const float q2x = __uint_as_float((q2 & 0xFFFFu) << 16);
  const float q2y = __uint_as_float(q2 & 0xFFFF0000u);
  const float q3x = __uint_as_float((q3 & 0xFFFFu) << 16);
  const float q3y = __uint_as_float(q3 & 0xFFFF0000u);
  // r = q*scale + (-(128+zero)*scale) = (nibble - zero)*scale, then narrow.
  dq[0].x = f32_to_bf16_no_canon(__fmaf_rn(q0x, y_prep, z_prep));
  dq[0].y = f32_to_bf16_no_canon(__fmaf_rn(q0y, y_prep, z_prep));
  dq[1].x = f32_to_bf16_no_canon(__fmaf_rn(q1x, y_prep, z_prep));
  dq[1].y = f32_to_bf16_no_canon(__fmaf_rn(q1y, y_prep, z_prep));
  dq[2].x = f32_to_bf16_no_canon(__fmaf_rn(q2x, y_prep, z_prep));
  dq[2].y = f32_to_bf16_no_canon(__fmaf_rn(q2y, y_prep, z_prep));
  dq[3].x = f32_to_bf16_no_canon(__fmaf_rn(q3x, y_prep, z_prep));
  dq[3].y = f32_to_bf16_no_canon(__fmaf_rn(q3y, y_prep, z_prep));
}

// ---------------------------------------------------------------------------
// Packed atomic-add helpers used by the K-split epilogue.
//
// When the kernel is launched with gridDim.z > 1, multiple K-segments
// accumulate into the same 16x16 output tile and need atomic write-back.
// gfx11 has no native v_global_atomic_pk_add_{f16,bf16}, so we issue a
// CAS-loop on a 32-bit word covering 2 packed fp16/bf16 lanes. Within a
// block the kernel pairs adjacent lanes via shfl_xor first, so each pair
// of cols (n=lane_lo even, lane_lo+1) goes through a SINGLE atomic — no
// intra-block contention on the same uint32 target. Inter-block
// contention from gridDim.z (4-way at K_SPLIT=4) is the residual cost.
// ---------------------------------------------------------------------------

__forceinline__ __device__ void atomic_add_pk_f16(half2* addr, half2 val) {
  uint32_t* addr_u = reinterpret_cast<uint32_t*>(addr);
  uint32_t old = *addr_u;
  while (true) {
    half2 cur;
    __builtin_memcpy(&cur, &old, sizeof(cur));
    half2 sum = __hadd2(cur, val);
    uint32_t sum_u;
    __builtin_memcpy(&sum_u, &sum, sizeof(sum_u));
    uint32_t prev = atomicCAS(addr_u, old, sum_u);
    if (prev == old) break;
    old = prev;
  }
}

__forceinline__ __device__ void atomic_add_pk_bf16(bf162_t* addr, bf162_t val) {
  uint32_t* addr_u = reinterpret_cast<uint32_t*>(addr);
  uint32_t old = *addr_u;
  while (true) {
    bf162_t cur;
    __builtin_memcpy(&cur, &old, sizeof(cur));
    bf162_t sum = __hadd2(cur, val);
    uint32_t sum_u;
    __builtin_memcpy(&sum_u, &sum, sizeof(sum_u));
    uint32_t prev = atomicCAS(addr_u, old, sum_u);
    if (prev == old) break;
    old = prev;
  }
}

#endif  // helpers guard; K-split heuristics below are pure host/device
        // arithmetic, called from launch_* on non-RDNA3 device passes too.

// K-split factor heuristic. Returns the gridDim.z to use for a given K.
// Aim: each block does at least ~16 K-tiles (= K=256) so the per-block
// constant overhead (LDS init, kernel prologue) is amortised. Upper
// bound K_SPLIT=4 to cap inter-block atomic contention to 4-way.
//
// For typical Qwen-class shapes K ∈ {4096, 5120, 11008}, all return 4.
// Smaller K (e.g., embedding lookups) fall back to 1 (no split, no
// atomic). K must be divisible by (K_SPLIT × 16) for the split to be
// valid; the heuristic checks divisibility before raising the factor.
__host__ __device__ static inline int compute_wmma_k_split(int size_k) {
  if (size_k >= 1024 && size_k % 64 == 0) return 4;
  if (size_k >= 512 && size_k % 32 == 0) return 2;
  return 1;
}

// M-and-N-aware K-split heuristic for the v3/v4/v5 launchers.
//
// The original `compute_wmma_k_split` was K-only and always returns 4 for
// Qwen-class K, which over-subscribes wave slots and pays the atomic CAS
// epilogue once-per-K-segment per output cell. With v3/v4/v5's larger
// tiles (64M × 16/32/64N) and 4 resident waves per block, the no-split
// grid is often already well-saturated on gfx1100's 96 CUs / 3072 wave
// slots — adding gridDim.z just adds atomic overhead.
//
// Heuristic: compute the no-split block count gridDim.x × gridDim.y, then
// pick the smallest K_SPLIT that brings total waves to at least
// ~2× over-subscription (~6000 waves for our 3072 slots, i.e. 1500 blocks
// at 4 waves/block). Above that threshold, K_SPLIT=1 — direct write, no
// atomic.
//
// Args:
//   size_m, size_n, size_k     — GEMM dims
//   m_tile, n_tile             — block-level M and N tile (64×16 for v3,
//                                64×32 for v4, 64×64 for v5)
//
// Returns: gridDim.z divisor (1, 2, or 4), respecting K-divisibility.
__host__ __device__ static inline int compute_wmma_k_split_mn(
    int size_m, int size_n, int size_k, int m_tile, int n_tile) {
  const int blocks_xy =
      ((size_n + n_tile - 1) / n_tile) * ((size_m + m_tile - 1) / m_tile);
  // Target: enough blocks to keep ~2× oversubscription on 96 CUs / 3072
  // wave slots at 4 waves/block ⇒ ~1500 blocks no-split.
  constexpr int kTargetBlocksXY = 1500;
  if (blocks_xy >= kTargetBlocksXY) return 1;
  if (blocks_xy * 2 >= kTargetBlocksXY && size_k >= 512 && size_k % 32 == 0)
    return 2;
  if (blocks_xy * 4 >= kTargetBlocksXY && size_k >= 1024 && size_k % 64 == 0)
    return 4;
  // Fall back to the K-only heuristic when blocks are very few (small
  // models with small N): K-split is the only way to add parallelism.
  return compute_wmma_k_split(size_k);
}

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// Native AMDGPU vector types expected by the WMMA built-ins.
using v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using v16bf16 = __bf16 __attribute__((ext_vector_type(16)));
using v8fp32 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8fp32 wmma_mma(v16fp16 a, v16fp16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ v8fp32 wmma_mma(v16bf16 a, v16bf16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

// Map HIP wrapper types (half, __hip_bfloat16) to native compiler types
// (_Float16, __bf16) used by the WMMA built-ins. Bitcast is a register
// reinterpret in practice.
template <typename T>
struct WmmaNative;
template <>
struct WmmaNative<half> {
  using elem = _Float16;
  using v16 = v16fp16;
};
template <>
struct WmmaNative<bf16_t> {
  using elem = __bf16;
  using v16 = v16bf16;
};

template <typename FROM, typename TO>
__device__ __forceinline__ TO bitcast_elem(FROM x) {
  static_assert(sizeof(FROM) == sizeof(TO),
                "bitcast_elem requires equal-sized types");
  TO r;
  __builtin_memcpy(&r, &x, sizeof(TO));
  return r;
}

// Per-T tzero (matches the helper in the scalar TU).
template <typename T>
__device__ __forceinline__ T tzero();
template <>
__device__ __forceinline__ half tzero<half>() {
  return __float2half_rn(0.0f);
}
template <>
__device__ __forceinline__ bf16_t tzero<bf16_t>() {
  return __float2bfloat16(0.0f);
}

#endif  // helpers guard (each __global__ below has its own guard so launch_*
        // host code remains visible to the parser on non-RDNA3 device passes)

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// ===========================================================================
// WMMA kernel: 16M × 16N tile per block, 1 wave, full K traversal.
//
// Wave32 fragment layout (verified empirically with all-modes diagnostic
// against a random A and B and the eight candidate output mappings):
//
//   * A frag (row-major in M, K in slot):
//       lane t, slot i → A[lane_lo][k = i]
//       Lane axis encodes M (A's row), slot encodes K.
//   * B frag (col-major in N, K in slot):
//       lane t, slot i → B[k = i][lane_lo]
//       Lane axis encodes N (B's column), slot encodes K. K-axis aligns
//       with A's K-axis (same slot index).
//   * C frag (output, lane = N, slot = M with hi-bit interleave):
//       lane t, slot i → C[m = 2*i + lane_hi][n = lane_lo]
//       Lane axis encodes N (C's column). Each lane holds 8 elements of
//       its output column, alternating rows: lanes 0..15 (hi=0) hold even
//       rows m=0,2,4,...,14; lanes 16..31 (hi=1) hold odd rows
//       m=1,3,5,...,15.
//   * Both halves of the wave (lanes 0..15 and 16..31) hold IDENTICAL input
//     fragments (AMD's "doubled" wave32 input layout). Output is split
//     between halves via lane_hi.
//
// History note: an earlier version of this kernel loaded B row-major in K
// and assumed the output was C[lane_lo][2*i+lane_hi]. That layout passes
// all-A=identity tests because A=I makes the K-axis sum collapse, but
// implements C = A @ B^T for non-trivial A — the bug only shows up against
// random A. A layout probe iterating all four
// {row,col} × {row,col} loadings identified mode 1 (A row, B col) with
// output [m=2*i+hi][n=lane_lo] as the unique mapping that yields A @ B.
// ===========================================================================

template <typename T>
__global__ void gemm_q4_wmma_kernel_16x16_1w(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 16;
  const int n_tile = blockIdx.x * 16;
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int lane = threadIdx.x;   // 0..31
  const int lane_lo = lane & 15;  // row index within fragment
  const int lane_hi = lane >> 4;  // 0 or 1

  v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;

  // K-split: each block in the gridDim.z dimension processes a contiguous
  // K-segment [k_start, k_end). With gridDim.z > 1, multiple blocks
  // accumulate into the same output tile and need atomic write-back at
  // the end. With gridDim.z == 1 the kernel falls back to the original
  // behaviour: full K range, single writer per cell, direct write.
  //
  // The split multiplies the wave count by gridDim.z and proportionally
  // raises CU saturation — this is the dominant lever for closing the
  // throughput gap to the fp16 scalar kernel at M >= 16, which already
  // uses K-split natively (gridDim.z = K/256). At gridDim.z = 4 with
  // K=4096, WMMA jumps from 17% to ~67% wave-slot saturation.
  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  // LDS tile of dequantized B. 16 K rows × 16 N cols.
  __shared__ T b_lds[16][16];

  for (int k_tile = k_start; k_tile < k_end; k_tile += 16) {
    // ---- Dequant 16x16 B tile into LDS ----
    // 32 lanes split 16 N cols × 2 K-octets per col = 32 dequant tasks.
    const int my_n = lane_lo;
    const int my_k_octet = lane_hi;  // 0 → K[0..7], 1 → K[8..15]
    const int actual_n = n_tile + my_n;

    if (actual_n < size_n) {
      const int qk_row = (k_tile / 8) + my_k_octet;
      const uint32_t qa = b_q[qk_row * size_n + actual_n];

      const int g = k_tile / groupsize;
      const int qz_idx = g * (size_n / 8) + actual_n / 8;
      const int qz_shift = (actual_n & 7) * 4;
      const uint32_t zero_v =
          ((b_qzeros[qz_idx] >> qz_shift) & 0xF) + (uint32_t)zero_offset;
      const T scale_t = b_scales[g * size_n + actual_n];

      const int k_base = my_k_octet * 8;

      if constexpr (std::is_same<T, half>::value) {
        half2 z_prep, y_prep;
        prep_zero_scale_fp16_precise(zero_v, scale_t, z_prep, y_prep);
        half2 dq[4];
        dequant_4bit_8_fp16_precise(qa, dq, z_prep, y_prep);
        b_lds[k_base + 0][my_n] = __low2half(dq[0]);
        b_lds[k_base + 1][my_n] = __high2half(dq[0]);
        b_lds[k_base + 2][my_n] = __low2half(dq[1]);
        b_lds[k_base + 3][my_n] = __high2half(dq[1]);
        b_lds[k_base + 4][my_n] = __low2half(dq[2]);
        b_lds[k_base + 5][my_n] = __high2half(dq[2]);
        b_lds[k_base + 6][my_n] = __low2half(dq[3]);
        b_lds[k_base + 7][my_n] = __high2half(dq[3]);
      } else {
        float z_f, y_f;
        prep_zero_scale_bf16_f32(zero_v, scale_t, z_f, y_f);
        bf162_t dq[4];
        dequant_4bit_8_bf16_to_bf16(qa, dq, z_f, y_f);
        b_lds[k_base + 0][my_n] = dq[0].x;
        b_lds[k_base + 1][my_n] = dq[0].y;
        b_lds[k_base + 2][my_n] = dq[1].x;
        b_lds[k_base + 3][my_n] = dq[1].y;
        b_lds[k_base + 4][my_n] = dq[2].x;
        b_lds[k_base + 5][my_n] = dq[2].y;
        b_lds[k_base + 6][my_n] = dq[3].x;
        b_lds[k_base + 7][my_n] = dq[3].y;
      }
    }

    // No __syncthreads() needed: the launch is `dim3 block(32)` = exactly one
    // wave32, so there is no inter-wave concurrency. Within a wave the
    // compiler emits `s_waitcnt lgkmcnt(0)` between dependent ds_write/ds_read
    // pairs automatically, so cross-lane LDS reads (lane 0 reading what
    // lane 16 wrote into b_lds[8..15][0]) still observe the writes. Keeping
    // the explicit `__syncthreads()` would emit a wave-level `s_barrier` that
    // costs ~10-20 cycles every iteration but provides no semantic guarantee
    // we don't already have for free in single-wave mode.

    // ---- Build A and B fragments, run WMMA ----
    V16 a_frag, b_frag;
    const int m_row = m_tile + lane_lo;

    if (m_row < size_m) {
      const T* a_row = a + m_row * size_k;
      if (b_q_perm) {
        // Permuted (act-order): scattered global reads, no vectorization.
  #pragma unroll
        for (int i = 0; i < 16; i++) {
          T v = a_row[b_q_perm[k_tile + i]];
          a_frag[i] = bitcast_elem<T, E>(v);
        }
      } else {
        // Sequential A reads: replace 16 single-element global_load_b16 with
        // a bulk 32-byte copy. The AMDGPU backend lowers a memcpy of this
        // size + alignment to two `global_load_b128` instructions. size_k is
        // a multiple of 16 (TORCH_CHECK above) and k_tile increments by 16,
        // so k_tile + 16 is always within bounds — no tail handling needed.
        // Note: we memcpy into the whole vector (`&a_frag`) rather than
        // `&a_frag[0]`; ext_vector_type element addresses aren't reliably
        // valid C pointers across compiler versions.
        static_assert(sizeof(a_frag) == 32, "V16 must be 32 bytes (16 × 2)");
        __builtin_memcpy(&a_frag, a_row + k_tile, sizeof(a_frag));
      }
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }

    // B fragment: lane t holds COLUMN n=lane_lo of the B tile (K-axis in
    // slot, N-axis in lane). This is the AMD WMMA convention for the right
    // operand of a matrix multiply — K-axis aligns with A's K-axis (also
    // in slot), enabling per-lane inner products.
  #pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag[i] = bitcast_elem<T, E>(b_lds[i][lane_lo]);
    }

  #ifdef VLLM_WMMA_LAYOUT_DEBUG
    // Diagnostic: skip WMMA, force c_acc to encode (lane, slot) so the
    // store pattern reveals the C-output lane→matrix mapping. Compile with
    // -DVLLM_WMMA_LAYOUT_DEBUG to enable. Output: c[m][n] = lane + slot/16.
    (void)a_frag;
    (void)b_frag;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      c_acc[i] = (float)lane + (float)i / 16.0f;
    }
    // Run only one K iteration in debug mode so c_acc isn't overwritten.
    if (k_tile == 0) {
      k_tile = size_k;  // exit loop on next check
    }
  #else
    c_acc = wmma_mma(a_frag, b_frag, c_acc);
  #endif

    // No __syncthreads() needed before the next iter overwrites b_lds:
    // single-wave block, and the next iter's ds_write to b_lds is preceded
    // by a `s_waitcnt lgkmcnt(0)` from the compiler that ensures the WMMA's
    // ds_read of b_frag has completed before the new ds_write issues.
  }

  // ---- Store C ----
  // Lane t holds column n=lane_lo of the output tile, 8 rows determined by
  // slot i and lane_hi:
  //   lane_hi == 0  →  rows 0, 2, 4, ..., 14   (even rows)
  //   lane_hi == 1  →  rows 1, 3, 5, ..., 15   (odd rows)
  // c_acc[i] corresponds to actual row m = 2*i + lane_hi at column lane_lo.
  if (gridDim.z > 1) {
    // K-split path: 4 (or whatever the split factor is) K-segments per
    // output cell contend → atomic accumulation. Caller has zero-init'd c.
    //
    // Pair-shuffle to avoid intra-block CAS contention: lanes lane_lo and
    // lane_lo+1 share the same uint32 atomic target (4 bytes = 2 fp16),
    // so without pairing they'd hammer the same word. Instead, swap the
    // c_acc[i] value with the lane_lo+1 neighbour via shfl_xor and have
    // ONLY the even lane issue a single packed CAS. Inter-block contention
    // (gridDim.z-way per cell) remains and is the residual atomic cost.
    const bool is_even_lane = (lane_lo & 1) == 0;
    const int out_n_pair = n_tile + lane_lo;  // valid only on even lane
  #pragma unroll
    for (int i = 0; i < 8; i++) {
      // Wave-wide shuffle: every lane participates so the side-effect is
      // visible. Only even lanes use the result. shfl_xor with mask 1
      // swaps with the lane_lo XOR 1 neighbour (same lane_hi → same row).
      float other_f = __shfl_xor(c_acc[i], 1);
      if (!is_even_lane) continue;

      const int out_m = m_tile + 2 * i + lane_hi;
      if (out_m >= size_m || out_n_pair >= size_n) continue;

      T* dst = c + out_m * size_n + out_n_pair;
      if constexpr (std::is_same<T, half>::value) {
        // Pack: .x = mine (col=lane_lo even), .y = neighbour (col=lane_lo+1)
        half2 packed =
            __halves2half2(__float2half_rn(c_acc[i]), __float2half_rn(other_f));
        atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
      } else {
        bf162_t packed;
        packed.x = __float2bfloat16(c_acc[i]);
        packed.y = __float2bfloat16(other_f);
        atomic_add_pk_bf16(reinterpret_cast<bf162_t*>(dst), packed);
      }
    }
  } else {
    // gridDim.z == 1: single writer per cell, direct non-atomic write.
    // Caller can leave c uninitialised (torch::empty) since every cell is
    // assigned exactly once.
    const int out_n = n_tile + lane_lo;
    if (out_n < size_n) {
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value) {
            *dst = __float2half_rn(c_acc[i]);
          } else {
            *dst = __float2bfloat16(c_acc[i]);
          }
        }
      }
    }
  }
}

#else  // non-RDNA3 device pass: empty kernel for symbol parity.
template <typename T>
__global__ void gemm_q4_wmma_kernel_16x16_1w(const T*, const uint32_t*,
                                             const uint32_t*, const T*, T*,
                                             const int, const int, const int,
                                             const int, const int, const int*) {
}
#endif

template <typename T>
void launch_gemm_q4_wmma_16x16_1w(const T* a, const uint32_t* b_q_weight,
                                  const uint32_t* b_qzeros, const T* b_scales,
                                  const int* b_q_perm, T* c, int size_m,
                                  int size_n, int size_k, int groups,
                                  int zero_offset, cudaStream_t stream) {
  // 1 wave per block (32 lanes), 16x16 C tile per block. gridDim.z splits
  // K so that more blocks (and therefore more waves) are in flight; with
  // K_SPLIT > 1 the kernel switches to atomic write-back at the epilogue.
  const int k_split = compute_wmma_k_split(size_k);
  dim3 block(32);
  dim3 grid((size_n + 15) / 16, (size_m + 15) / 16, k_split);
  gemm_q4_wmma_kernel_16x16_1w<T><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
}

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// ===========================================================================
// 32x16_2w kernel: 2 waves per block, 32M × 16N tile, double-buffered LDS.
//
// Targets the bf16-WMMA prefill regime (M >= 128) where the v1 single-wave
// kernel saturates only ~24% of WMMA peak because each wave does roughly
// one v_wmma every ~30-40 cycles (16-cycle wmma latency + dequant + LDS +
// global A load all serial inside the single resident wave).
//
// Two structural changes vs v1:
//
//   * 2 waves per block (64 threads). Both waves cooperate on a 32M×16N
//     output tile: wave 0 produces rows [0..15], wave 1 rows [16..31].
//     The B-tile in LDS is shared (only wave 0 dequants); each wave loads
//     its own A slice from global. With two resident waves the SIMD
//     scheduler can keep the WMMA pipeline full by interleaving wmmas
//     from the two waves while the other does dequant / LDS / load work.
//
//   * Double-buffered LDS B-tile (b_lds[2][16][16]). Wave 0 dequants
//     the K-tile for iter k+1 while both waves consume the K-tile for
//     iter k. Pulls dequant out of the WMMA-critical path. Costs ~512 B
//     extra LDS per block (irrelevant — gfx1100 has 64 KB LDS/CU).
//
// One __syncthreads() per K-iter remains: it ensures wave 0's dequant of
// the next K-tile has committed AND both waves have finished reading the
// current K-tile before wave 0 wraps around and overwrites it.
//
// The v2 launcher (`launch_gemm_q4_wmma_32x16_2w`) is the production entry on
// the WMMA path and falls back to v1 internally for size_m < 32 — see the
// comment at the top of `launch_gemm_q4_wmma_32x16_2w` for the M=16 regression
// rationale that justifies the fallback.
// ===========================================================================

template <typename T>
__global__ void gemm_q4_wmma_kernel_32x16_2w(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 32;  // 32-row stride per block
  const int n_tile = blockIdx.x * 16;
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int tid = threadIdx.x;   // 0..63
  const int wave_id = tid >> 5;  // 0 or 1
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;

  // K-split: each block handles a contiguous K-segment when gridDim.z > 1.
  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  // Double-buffered LDS B-tile. 2 × 16K × 16N × sizeof(T) = 1024 B for
  // fp16/bf16.
  __shared__ T b_lds[2][16][16];

  // Dequant a 16K × 16N B-tile into b_lds[buf]. Only wave 0 participates
  // (32 lanes do 32 dequant tasks: 16 N-cols × 2 K-octets per col).
  auto dequant_into = [&](int buf, int k_tile) {
    if (wave_id != 0) return;

    const int my_n = lane_lo;
    const int my_k_octet = lane_hi;
    const int actual_n = n_tile + my_n;

    if (actual_n >= size_n) return;

    const int qk_row = (k_tile / 8) + my_k_octet;
    const uint32_t qa = b_q[qk_row * size_n + actual_n];

    const int g = k_tile / groupsize;
    const int qz_idx = g * (size_n / 8) + actual_n / 8;
    const int qz_shift = (actual_n & 7) * 4;
    const uint32_t zero_v =
        ((b_qzeros[qz_idx] >> qz_shift) & 0xF) + (uint32_t)zero_offset;
    const T scale_t = b_scales[g * size_n + actual_n];

    const int k_base = my_k_octet * 8;

    if constexpr (std::is_same<T, half>::value) {
      half2 z_prep, y_prep;
      prep_zero_scale_fp16_precise(zero_v, scale_t, z_prep, y_prep);
      half2 dq[4];
      dequant_4bit_8_fp16_precise(qa, dq, z_prep, y_prep);
      b_lds[buf][k_base + 0][my_n] = __low2half(dq[0]);
      b_lds[buf][k_base + 1][my_n] = __high2half(dq[0]);
      b_lds[buf][k_base + 2][my_n] = __low2half(dq[1]);
      b_lds[buf][k_base + 3][my_n] = __high2half(dq[1]);
      b_lds[buf][k_base + 4][my_n] = __low2half(dq[2]);
      b_lds[buf][k_base + 5][my_n] = __high2half(dq[2]);
      b_lds[buf][k_base + 6][my_n] = __low2half(dq[3]);
      b_lds[buf][k_base + 7][my_n] = __high2half(dq[3]);
    } else {
      float z_f, y_f;
      prep_zero_scale_bf16_f32(zero_v, scale_t, z_f, y_f);
      bf162_t dq[4];
      dequant_4bit_8_bf16_to_bf16(qa, dq, z_f, y_f);
      b_lds[buf][k_base + 0][my_n] = dq[0].x;
      b_lds[buf][k_base + 1][my_n] = dq[0].y;
      b_lds[buf][k_base + 2][my_n] = dq[1].x;
      b_lds[buf][k_base + 3][my_n] = dq[1].y;
      b_lds[buf][k_base + 4][my_n] = dq[2].x;
      b_lds[buf][k_base + 5][my_n] = dq[2].y;
      b_lds[buf][k_base + 6][my_n] = dq[3].x;
      b_lds[buf][k_base + 7][my_n] = dq[3].y;
    }
  };

  // Pre-fill buffer 0 with the first K-tile so iter 0 has data to consume.
  dequant_into(0, k_start);
  __syncthreads();

  int cur_buf = 0;
  for (int k_tile = k_start; k_tile < k_end; k_tile += 16) {
    const int next_buf = 1 - cur_buf;
    const int k_next = k_tile + 16;

    // Issue dequant of next K-tile (wave 0 only) — overlaps with current
    // iter's WMMA work below. The LDS write is non-blocking; the sync at
    // end of iter ensures wave 1 sees it before iter k+1.
    if (k_next < k_end) {
      dequant_into(next_buf, k_next);
    }

    // Load A: each wave loads its own M slice in parallel.
    const int m_row = m_tile + wave_id * 16 + lane_lo;
    V16 a_frag, b_frag;
    if (m_row < size_m) {
      const T* a_row = a + m_row * size_k;
      if (b_q_perm) {
  #pragma unroll
        for (int i = 0; i < 16; i++) {
          T v = a_row[b_q_perm[k_tile + i]];
          a_frag[i] = bitcast_elem<T, E>(v);
        }
      } else {
        static_assert(sizeof(a_frag) == 32, "V16 must be 32 bytes");
        __builtin_memcpy(&a_frag, a_row + k_tile, sizeof(a_frag));
      }
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }

    // Load B from current buffer (both waves read identical data).
  #pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo]);
    }

    // Each wave issues its own WMMA against its own a_frag + shared b_frag.
    c_acc = wmma_mma(a_frag, b_frag, c_acc);

    // Sync ensures: (a) wave 0's dequant into next_buf has committed,
    // (b) both waves are done reading cur_buf — so next iter's overwrite
    // (cur_buf becomes the previous next_buf and gets reused two iters
    // later) is race-free.
    __syncthreads();
    cur_buf = next_buf;
  }

  // ---- Store C ----
  // Each wave owns rows [m_tile + wave_id*16 .. + 16) of the output tile.
  const int m_tile_wave = m_tile + wave_id * 16;

  if (gridDim.z > 1) {
    // K-split atomic path. Pair-shuffle within wave to halve atomic count.
    // shfl_xor here is wave-local (wave32 semantics) so each wave does its
    // own pairing — the two waves don't interact during the store.
    const bool is_even_lane = (lane_lo & 1) == 0;
    const int out_n_pair = n_tile + lane_lo;
  #pragma unroll
    for (int i = 0; i < 8; i++) {
      float other_f = __shfl_xor(c_acc[i], 1);
      if (!is_even_lane) continue;

      const int out_m = m_tile_wave + 2 * i + lane_hi;
      if (out_m >= size_m || out_n_pair >= size_n) continue;

      T* dst = c + out_m * size_n + out_n_pair;
      if constexpr (std::is_same<T, half>::value) {
        half2 packed =
            __halves2half2(__float2half_rn(c_acc[i]), __float2half_rn(other_f));
        atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
      } else {
        bf162_t packed;
        packed.x = __float2bfloat16(c_acc[i]);
        packed.y = __float2bfloat16(other_f);
        atomic_add_pk_bf16(reinterpret_cast<bf162_t*>(dst), packed);
      }
    }
  } else {
    // Single writer per cell, direct non-atomic write.
    const int out_n = n_tile + lane_lo;
    if (out_n < size_n) {
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value) {
            *dst = __float2half_rn(c_acc[i]);
          } else {
            *dst = __float2bfloat16(c_acc[i]);
          }
        }
      }
    }
  }
}

#else  // non-RDNA3 device pass: empty kernel for symbol parity.
template <typename T>
__global__ void gemm_q4_wmma_kernel_32x16_2w(const T*, const uint32_t*,
                                             const uint32_t*, const T*, T*,
                                             const int, const int, const int,
                                             const int, const int, const int*) {
}
#endif

template <typename T>
void launch_gemm_q4_wmma_32x16_2w(const T* a, const uint32_t* b_q_weight,
                                  const uint32_t* b_qzeros, const T* b_scales,
                                  const int* b_q_perm, T* c, int size_m,
                                  int size_n, int size_k, int groups,
                                  int zero_offset, cudaStream_t stream) {
  // Fallback to v1 for size_m < 32. With M-tile=32 the v2 block has 2 waves
  // working on rows [0..15] and [16..31]; at M < 32 the second wave processes
  // out-of-range M rows (zero-padded a_frag → wmma produces nothing useful)
  // and just wastes SIMD cycles. Bench measured a +47 % regression at M=16
  // vs v1 for this reason. The M < 32 case is rare in serving (decode at
  // max-num-seqs=32 lands at M≈32 steady-state; the M=16 sliver is edge),
  // but the fallback costs nothing and is the right shape.
  if (size_m < 32) {
    launch_gemm_q4_wmma_16x16_1w<T>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
    return;
  }

  // 2 waves per block (64 threads), 32M × 16N C tile per block.
  // K-split heuristic shared with v1. With M-tile=32, the natural
  // grid blocks are halved on Y vs v1 — but each block does 2× the work,
  // so total wave count is unchanged at the same M when K_SPLIT is equal.
  const int k_split = compute_wmma_k_split(size_k);
  dim3 block(64);
  dim3 grid((size_n + 15) / 16, (size_m + 31) / 32, k_split);
  gemm_q4_wmma_kernel_32x16_2w<T><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
}

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// ===========================================================================
// 64x16_4w kernel: 4 waves per block, 64M × 16N tile, double-buffered LDS.
//
// Targets the prefill plateau observed at M >= 128 in v2 (~144 K tk/s bf16,
// ~28 % of WMMA peak). The bottleneck is wmma issue rate per resident wave:
// each wave issues at most 1 wmma per ~30-40 cycles. With only 2 waves per
// block, two wmmas overlap; the wmma pipeline (16-cycle latency) is mostly
// idle.
//
// Doubling the resident wave count (2 → 4) targets ~2× wmma throughput by
// keeping the pipeline closer to full. 64M tile keeps N-tile at 16 (so the
// b_lds layout, dequant pattern, and store mapping carry over from v2) — the
// only structural changes are:
//
//   * 4 waves cooperate on the 64M × 16N output tile. Wave w produces rows
//     [16w .. 16w+15] of the M tile.
//   * Dequant remains on wave 0 only (32 lanes do 32 dequant slots, identical
//     to v2). Waves 1-3 idle through the dequant phase but their wmmas can
//     issue concurrently with wave 0's dequant of the *next* K-tile thanks
//     to the double buffer — net wave occupancy is dominated by the wmma
//     phase, not the dequant phase.
//   * One __syncthreads() per K-iter still required (same race as v2).
//
// Costs: same LDS as v2 (1024 B for the b_lds double buffer). Block has 128
// threads vs 64 in v2; gfx1100 supports up to 1024 threads/block so this is
// well within budget. Doubles VGPR pressure slightly because the four waves
// each hold their own a_frag + c_acc — but each wave's working set is
// independent so per-thread VGPR is unchanged.
// ===========================================================================

template <typename T>
__global__ void gemm_q4_wmma_kernel_64x16_4w(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile =
      blockIdx.y * 64;  // 64-row stride per block (4 waves × 16M)
  const int n_tile = blockIdx.x * 16;
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int tid = threadIdx.x;   // 0..127
  const int wave_id = tid >> 5;  // 0..3
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;

  // K-split: each block handles a contiguous K-segment when gridDim.z > 1.
  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  // Double-buffered LDS B-tile. Same layout as v2.
  __shared__ T b_lds[2][16][16];

  // Dequant a 16K × 16N B-tile into b_lds[buf]. Only wave 0's 32 lanes
  // participate (16 N-cols × 2 K-octets = 32 dequant slots). Waves 1-3
  // skip — their wmma can run concurrently with wave 0's next-iter dequant
  // through the double buffer.
  auto dequant_into = [&](int buf, int k_tile) {
    if (wave_id != 0) return;

    const int my_n = lane_lo;
    const int my_k_octet = lane_hi;
    const int actual_n = n_tile + my_n;

    if (actual_n >= size_n) return;

    const int qk_row = (k_tile / 8) + my_k_octet;
    const uint32_t qa = b_q[qk_row * size_n + actual_n];

    const int g = k_tile / groupsize;
    const int qz_idx = g * (size_n / 8) + actual_n / 8;
    const int qz_shift = (actual_n & 7) * 4;
    const uint32_t zero_v =
        ((b_qzeros[qz_idx] >> qz_shift) & 0xF) + (uint32_t)zero_offset;
    const T scale_t = b_scales[g * size_n + actual_n];

    const int k_base = my_k_octet * 8;

    if constexpr (std::is_same<T, half>::value) {
      half2 z_prep, y_prep;
      prep_zero_scale_fp16_precise(zero_v, scale_t, z_prep, y_prep);
      half2 dq[4];
      dequant_4bit_8_fp16_precise(qa, dq, z_prep, y_prep);
      b_lds[buf][k_base + 0][my_n] = __low2half(dq[0]);
      b_lds[buf][k_base + 1][my_n] = __high2half(dq[0]);
      b_lds[buf][k_base + 2][my_n] = __low2half(dq[1]);
      b_lds[buf][k_base + 3][my_n] = __high2half(dq[1]);
      b_lds[buf][k_base + 4][my_n] = __low2half(dq[2]);
      b_lds[buf][k_base + 5][my_n] = __high2half(dq[2]);
      b_lds[buf][k_base + 6][my_n] = __low2half(dq[3]);
      b_lds[buf][k_base + 7][my_n] = __high2half(dq[3]);
    } else {
      float z_f, y_f;
      prep_zero_scale_bf16_f32(zero_v, scale_t, z_f, y_f);
      bf162_t dq[4];
      dequant_4bit_8_bf16_to_bf16(qa, dq, z_f, y_f);
      b_lds[buf][k_base + 0][my_n] = dq[0].x;
      b_lds[buf][k_base + 1][my_n] = dq[0].y;
      b_lds[buf][k_base + 2][my_n] = dq[1].x;
      b_lds[buf][k_base + 3][my_n] = dq[1].y;
      b_lds[buf][k_base + 4][my_n] = dq[2].x;
      b_lds[buf][k_base + 5][my_n] = dq[2].y;
      b_lds[buf][k_base + 6][my_n] = dq[3].x;
      b_lds[buf][k_base + 7][my_n] = dq[3].y;
    }
  };

  // Pre-fill buffer 0 with the first K-tile so iter 0 has data to consume.
  dequant_into(0, k_start);
  __syncthreads();

  int cur_buf = 0;
  for (int k_tile = k_start; k_tile < k_end; k_tile += 16) {
    const int next_buf = 1 - cur_buf;
    const int k_next = k_tile + 16;

    // Issue dequant of next K-tile (wave 0 only) — overlaps with the wmma
    // work below across all 4 waves.
    if (k_next < k_end) {
      dequant_into(next_buf, k_next);
    }

    // Each wave loads its own 16M slice of A (wave w handles M-rows
    // [m_tile + 16w .. m_tile + 16w + 15]).
    const int m_row = m_tile + wave_id * 16 + lane_lo;
    V16 a_frag, b_frag;
    if (m_row < size_m) {
      const T* a_row = a + m_row * size_k;
      if (b_q_perm) {
  #pragma unroll
        for (int i = 0; i < 16; i++) {
          T v = a_row[b_q_perm[k_tile + i]];
          a_frag[i] = bitcast_elem<T, E>(v);
        }
      } else {
        static_assert(sizeof(a_frag) == 32, "V16 must be 32 bytes");
        __builtin_memcpy(&a_frag, a_row + k_tile, sizeof(a_frag));
      }
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }

    // Load B from current buffer (all 4 waves read identical data).
  #pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo]);
    }

    // Each wave issues its own WMMA against its own a_frag + shared b_frag.
    // 4 wmmas in flight per block per K-iter.
    c_acc = wmma_mma(a_frag, b_frag, c_acc);

    __syncthreads();
    cur_buf = next_buf;
  }

  // ---- Store C ---- Each wave owns 16 M-rows of the output tile.
  const int m_tile_wave = m_tile + wave_id * 16;

  if (gridDim.z > 1) {
    // K-split atomic path. Pair-shuffle within wave to halve atomic count.
    const bool is_even_lane = (lane_lo & 1) == 0;
    const int out_n_pair = n_tile + lane_lo;
  #pragma unroll
    for (int i = 0; i < 8; i++) {
      float other_f = __shfl_xor(c_acc[i], 1);
      if (!is_even_lane) continue;

      const int out_m = m_tile_wave + 2 * i + lane_hi;
      if (out_m >= size_m || out_n_pair >= size_n) continue;

      T* dst = c + out_m * size_n + out_n_pair;
      if constexpr (std::is_same<T, half>::value) {
        half2 packed =
            __halves2half2(__float2half_rn(c_acc[i]), __float2half_rn(other_f));
        atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
      } else {
        bf162_t packed;
        packed.x = __float2bfloat16(c_acc[i]);
        packed.y = __float2bfloat16(other_f);
        atomic_add_pk_bf16(reinterpret_cast<bf162_t*>(dst), packed);
      }
    }
  } else {
    // Single writer per cell, direct non-atomic write.
    const int out_n = n_tile + lane_lo;
    if (out_n < size_n) {
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value) {
            *dst = __float2half_rn(c_acc[i]);
          } else {
            *dst = __float2bfloat16(c_acc[i]);
          }
        }
      }
    }
  }
}

#else  // non-RDNA3 device pass: empty kernel for symbol parity.
template <typename T>
__global__ void gemm_q4_wmma_kernel_64x16_4w(const T*, const uint32_t*,
                                             const uint32_t*, const T*, T*,
                                             const int, const int, const int,
                                             const int, const int, const int*) {
}
#endif

template <typename T>
void launch_gemm_q4_wmma_64x16_4w(const T* a, const uint32_t* b_q_weight,
                                  const uint32_t* b_qzeros, const T* b_scales,
                                  const int* b_q_perm, T* c, int size_m,
                                  int size_n, int size_k, int groups,
                                  int zero_offset, cudaStream_t stream) {
  // Fall back to v2 for M < 64 (would waste 1+ waves on out-of-range rows).
  if (size_m < 64) {
    launch_gemm_q4_wmma_32x16_2w<T>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
    return;
  }

  // 4 waves per block (128 threads), 64M × 16N tile per block.
  const int k_split = compute_wmma_k_split_mn(size_m, size_n, size_k, 64, 16);
  dim3 block(128);
  dim3 grid((size_n + 15) / 16, (size_m + 63) / 64, k_split);
  gemm_q4_wmma_kernel_64x16_4w<T><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
}

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// ===========================================================================
// 64x32_4w kernel: 4 waves per block, 64M × 32N tile, double-buffered LDS.
//
// Builds on v3 by doubling the N-tile from 16 → 32. Each wave now issues
// 2 wmmas per K-iter (one for cols 0-15, one for cols 16-31, sharing the
// same a_frag). With 4 waves × 2 wmmas = 8 wmmas in flight per K-iter,
// the wmma pipeline gets ~2× more in-flight work than v3 — targeting the
// remaining wmma issue gap on Qwen-class shapes (gate/up, down) where v3
// plateaus at ~22 TFLOPS effective.
//
// Costs:
//   * LDS B-tile doubles (2 × 16K × 32N × sizeof(T) = 2048 B for fp16/bf16).
//   * Dequant doubles (32 N-cols × 2 K-octets = 64 slots/K-tile). Distributed
//     across waves 0-1: wave 0 dequants n=[0..15], wave 1 dequants n=[16..31].
//     Waves 2-3 idle on dequant but do wmma work.
//   * Per-wave registers: 2 × v8fp32 accumulator (16 fp32 = 32 VGPRs) plus
//     b_frag0 + b_frag1 (32 VGPRs total). Within budget.
//
// Mapping invariant (wave-id → output tile slice):
//   * Wave w produces M rows [m_tile + 16w .. m_tile + 16w + 15]
//   * c_acc0 holds N cols [n_tile + 0 .. n_tile + 15]
//   * c_acc1 holds N cols [n_tile + 16 .. n_tile + 31]
// ===========================================================================

template <typename T>
__global__ void gemm_q4_wmma_kernel_64x32_4w(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 64;
  const int n_tile = blockIdx.x * 32;  // 32-col stride per block (was 16 in v3)
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int tid = threadIdx.x;   // 0..127
  const int wave_id = tid >> 5;  // 0..3
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  // Two accumulators per wave: c_acc0 covers cols [n_tile..n_tile+15],
  // c_acc1 covers cols [n_tile+16..n_tile+31].
  v8fp32 c_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;

  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  // Doubled LDS B-tile: [buf][k][n_in_tile=0..31].
  __shared__ T b_lds[2][16][32];

  // Dequant: 64 slots per K-tile (32 N-cols × 2 K-octets). Distributed
  // across waves 0,1: wave w handles n_in_tile in [16w..16w+15], k_oct in
  // {0,1}. Each dequanting wave has 32 lanes for 32 dequant slots — perfect
  // mapping, identical layout to v2/v3 dequant per wave. Waves 2,3 stay idle
  // during dequant; they catch up via the double buffer overlapping the
  // wmma of iter k with dequant of iter k+1.
  //
  // Tried distributing dequant across all 4 waves (16 slots/wave): regressed
  // 3% on gate/up M=2048 (357K → 347K tk/s). The dequant is not on the
  // critical path; spreading it just adds LDS bank pressure with no gain.
  auto dequant_into = [&](int buf, int k_tile) {
    if (wave_id >= 2) return;

    const int my_n_local = lane_lo;  // 0..15 (within wave's N-half)
    const int my_n_in_tile = wave_id * 16 + my_n_local;  // 0..31 (in 32N tile)
    const int my_k_octet = lane_hi;
    const int actual_n = n_tile + my_n_in_tile;

    if (actual_n >= size_n) return;

    const int qk_row = (k_tile / 8) + my_k_octet;
    const uint32_t qa = b_q[qk_row * size_n + actual_n];

    const int g = k_tile / groupsize;
    const int qz_idx = g * (size_n / 8) + actual_n / 8;
    const int qz_shift = (actual_n & 7) * 4;
    const uint32_t zero_v =
        ((b_qzeros[qz_idx] >> qz_shift) & 0xF) + (uint32_t)zero_offset;
    const T scale_t = b_scales[g * size_n + actual_n];

    const int k_base = my_k_octet * 8;

    if constexpr (std::is_same<T, half>::value) {
      half2 z_prep, y_prep;
      prep_zero_scale_fp16_precise(zero_v, scale_t, z_prep, y_prep);
      half2 dq[4];
      dequant_4bit_8_fp16_precise(qa, dq, z_prep, y_prep);
      b_lds[buf][k_base + 0][my_n_in_tile] = __low2half(dq[0]);
      b_lds[buf][k_base + 1][my_n_in_tile] = __high2half(dq[0]);
      b_lds[buf][k_base + 2][my_n_in_tile] = __low2half(dq[1]);
      b_lds[buf][k_base + 3][my_n_in_tile] = __high2half(dq[1]);
      b_lds[buf][k_base + 4][my_n_in_tile] = __low2half(dq[2]);
      b_lds[buf][k_base + 5][my_n_in_tile] = __high2half(dq[2]);
      b_lds[buf][k_base + 6][my_n_in_tile] = __low2half(dq[3]);
      b_lds[buf][k_base + 7][my_n_in_tile] = __high2half(dq[3]);
    } else {
      float z_f, y_f;
      prep_zero_scale_bf16_f32(zero_v, scale_t, z_f, y_f);
      bf162_t dq[4];
      dequant_4bit_8_bf16_to_bf16(qa, dq, z_f, y_f);
      b_lds[buf][k_base + 0][my_n_in_tile] = dq[0].x;
      b_lds[buf][k_base + 1][my_n_in_tile] = dq[0].y;
      b_lds[buf][k_base + 2][my_n_in_tile] = dq[1].x;
      b_lds[buf][k_base + 3][my_n_in_tile] = dq[1].y;
      b_lds[buf][k_base + 4][my_n_in_tile] = dq[2].x;
      b_lds[buf][k_base + 5][my_n_in_tile] = dq[2].y;
      b_lds[buf][k_base + 6][my_n_in_tile] = dq[3].x;
      b_lds[buf][k_base + 7][my_n_in_tile] = dq[3].y;
    }
  };

  dequant_into(0, k_start);
  __syncthreads();

  int cur_buf = 0;
  for (int k_tile = k_start; k_tile < k_end; k_tile += 16) {
    const int next_buf = 1 - cur_buf;
    const int k_next = k_tile + 16;

    if (k_next < k_end) {
      dequant_into(next_buf, k_next);
    }

    // Load A: each wave loads its 16M slice, shared across both wmmas.
    const int m_row = m_tile + wave_id * 16 + lane_lo;
    V16 a_frag, b_frag0, b_frag1;
    if (m_row < size_m) {
      const T* a_row = a + m_row * size_k;
      if (b_q_perm) {
  #pragma unroll
        for (int i = 0; i < 16; i++) {
          T v = a_row[b_q_perm[k_tile + i]];
          a_frag[i] = bitcast_elem<T, E>(v);
        }
      } else {
        static_assert(sizeof(a_frag) == 32, "V16 must be 32 bytes");
        __builtin_memcpy(&a_frag, a_row + k_tile, sizeof(a_frag));
      }
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }

    // Load B for cols [0..15] and cols [16..31]. Both halves of the 32N tile.
  #pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag0[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo]);
      b_frag1[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo + 16]);
    }

    // Two wmmas per wave per K-iter, sharing a_frag.
    c_acc0 = wmma_mma(a_frag, b_frag0, c_acc0);
    c_acc1 = wmma_mma(a_frag, b_frag1, c_acc1);

    __syncthreads();
    cur_buf = next_buf;
  }

  // ---- Store C ----
  // Each wave owns 16M rows × 32N cols. c_acc0 → cols [n_tile..n_tile+15],
  // c_acc1 → cols [n_tile+16..n_tile+31].
  const int m_tile_wave = m_tile + wave_id * 16;

  // Helper: store one v8fp32 accumulator's 8 outputs (covers 16 N-cols at
  // n_base via lane_lo + interleaved M rows m_tile_wave + 2i + lane_hi).
  auto store_acc = [&](const v8fp32& acc, int n_base) {
    if (gridDim.z > 1) {
      const bool is_even_lane = (lane_lo & 1) == 0;
      const int out_n_pair = n_base + lane_lo;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        float other_f = __shfl_xor(acc[i], 1);
        if (!is_even_lane) continue;

        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m >= size_m || out_n_pair >= size_n) continue;

        T* dst = c + out_m * size_n + out_n_pair;
        if constexpr (std::is_same<T, half>::value) {
          half2 packed =
              __halves2half2(__float2half_rn(acc[i]), __float2half_rn(other_f));
          atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
        } else {
          bf162_t packed;
          packed.x = __float2bfloat16(acc[i]);
          packed.y = __float2bfloat16(other_f);
          atomic_add_pk_bf16(reinterpret_cast<bf162_t*>(dst), packed);
        }
      }
    } else {
      const int out_n = n_base + lane_lo;
      if (out_n >= size_n) return;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value) {
            *dst = __float2half_rn(acc[i]);
          } else {
            *dst = __float2bfloat16(acc[i]);
          }
        }
      }
    }
  };

  store_acc(c_acc0, n_tile);
  store_acc(c_acc1, n_tile + 16);
}

#else  // non-RDNA3 device pass: empty kernel for symbol parity.
template <typename T>
__global__ void gemm_q4_wmma_kernel_64x32_4w(const T*, const uint32_t*,
                                             const uint32_t*, const T*, T*,
                                             const int, const int, const int,
                                             const int, const int, const int*) {
}
#endif

template <typename T>
void launch_gemm_q4_wmma_64x32_4w(const T* a, const uint32_t* b_q_weight,
                                  const uint32_t* b_qzeros, const T* b_scales,
                                  const int* b_q_perm, T* c, int size_m,
                                  int size_n, int size_k, int groups,
                                  int zero_offset, cudaStream_t stream) {
  // Fall back to v3 when M < 64 (small-M decode/prefill stays on the
  // narrower 64M × 16N path) or when N < 32 (tile would waste a wave on
  // out-of-range cols).
  if (size_m < 64 || size_n < 32) {
    launch_gemm_q4_wmma_64x16_4w<T>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
    return;
  }

  // 4 waves per block (128 threads), 64M × 32N tile per block.
  const int k_split = compute_wmma_k_split_mn(size_m, size_n, size_k, 64, 32);
  dim3 block(128);
  dim3 grid((size_n + 31) / 32, (size_m + 63) / 64, k_split);
  gemm_q4_wmma_kernel_64x32_4w<T><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
}

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// ===========================================================================
// 64x64_4w kernel: 4 waves per block, 64M × 64N tile, 4 wmmas per wave per
// K-iter.
//
// Doubles the N-tile from 32 → 64. Each wave issues 4 wmmas per K-iter
// (cols 0-15, 16-31, 32-47, 48-63), all sharing the same a_frag. With
// 4 waves × 4 wmmas = 16 wmmas in flight per K-iter, the wmma pipeline
// is fully saturated (16-cycle latency × 1 issue/cycle = 16 wmmas in
// flight at peak).
//
// Costs:
//   * LDS B-tile: 2 × 16K × 64N × sizeof(T) = 4 KB. Within budget.
//   * Dequant: 64 N × 2 K-oct = 128 slots/K-tile. Distributed across all
//     4 waves: 32 slots/wave (16 N-cols × 2 K-octets per wave) — perfect
//     32-lane fit, full lane utilization on dequant.
//   * Per-wave registers: 4 × v8fp32 acc (64 VGPRs) + 4 b_frag (64 VGPRs)
//     + a_frag (16 VGPRs) ≈ 144 VGPRs/thread + locals ≈ ~170 total.
//     Under the 192-VGPR gfx1100 cap.
//
// Mapping invariant:
//   * Wave w produces M rows [m_tile + 16w .. m_tile + 16w + 15]
//   * c_acc[i] holds N cols [n_tile + 16i .. n_tile + 16i + 15] for i=0..3
// ===========================================================================

template <typename T>
__global__ void gemm_q4_wmma_kernel_64x64_4w(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 64;
  const int n_tile = blockIdx.x * 64;  // 64-col stride per block
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int tid = threadIdx.x;   // 0..127
  const int wave_id = tid >> 5;  // 0..3
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  // Four accumulators per wave, each covering 16 N-cols.
  v8fp32 c_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;

  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  // Larger LDS B-tile: [buf][k][n_in_tile=0..63].
  __shared__ T b_lds[2][16][64];

  // Dequant: 128 slots per K-tile (64 N-cols × 2 K-octets). All 4 waves
  // participate; wave w covers n_in_tile in [16w..16w+15], k_oct in {0,1}
  // — 32 slots per wave (16 N-cols × 2 K-octets), perfect 32-lane fit.
  auto dequant_into = [&](int buf, int k_tile) {
    const int my_n_local = lane_lo;                      // 0..15 within wave
    const int my_n_in_tile = wave_id * 16 + my_n_local;  // 0..63
    const int my_k_octet = lane_hi;
    const int actual_n = n_tile + my_n_in_tile;

    if (actual_n >= size_n) return;

    const int qk_row = (k_tile / 8) + my_k_octet;
    const uint32_t qa = b_q[qk_row * size_n + actual_n];

    const int g = k_tile / groupsize;
    const int qz_idx = g * (size_n / 8) + actual_n / 8;
    const int qz_shift = (actual_n & 7) * 4;
    const uint32_t zero_v =
        ((b_qzeros[qz_idx] >> qz_shift) & 0xF) + (uint32_t)zero_offset;
    const T scale_t = b_scales[g * size_n + actual_n];

    const int k_base = my_k_octet * 8;

    if constexpr (std::is_same<T, half>::value) {
      half2 z_prep, y_prep;
      prep_zero_scale_fp16_precise(zero_v, scale_t, z_prep, y_prep);
      half2 dq[4];
      dequant_4bit_8_fp16_precise(qa, dq, z_prep, y_prep);
      b_lds[buf][k_base + 0][my_n_in_tile] = __low2half(dq[0]);
      b_lds[buf][k_base + 1][my_n_in_tile] = __high2half(dq[0]);
      b_lds[buf][k_base + 2][my_n_in_tile] = __low2half(dq[1]);
      b_lds[buf][k_base + 3][my_n_in_tile] = __high2half(dq[1]);
      b_lds[buf][k_base + 4][my_n_in_tile] = __low2half(dq[2]);
      b_lds[buf][k_base + 5][my_n_in_tile] = __high2half(dq[2]);
      b_lds[buf][k_base + 6][my_n_in_tile] = __low2half(dq[3]);
      b_lds[buf][k_base + 7][my_n_in_tile] = __high2half(dq[3]);
    } else {
      float z_f, y_f;
      prep_zero_scale_bf16_f32(zero_v, scale_t, z_f, y_f);
      bf162_t dq[4];
      dequant_4bit_8_bf16_to_bf16(qa, dq, z_f, y_f);
      b_lds[buf][k_base + 0][my_n_in_tile] = dq[0].x;
      b_lds[buf][k_base + 1][my_n_in_tile] = dq[0].y;
      b_lds[buf][k_base + 2][my_n_in_tile] = dq[1].x;
      b_lds[buf][k_base + 3][my_n_in_tile] = dq[1].y;
      b_lds[buf][k_base + 4][my_n_in_tile] = dq[2].x;
      b_lds[buf][k_base + 5][my_n_in_tile] = dq[2].y;
      b_lds[buf][k_base + 6][my_n_in_tile] = dq[3].x;
      b_lds[buf][k_base + 7][my_n_in_tile] = dq[3].y;
    }
  };

  dequant_into(0, k_start);
  __syncthreads();

  int cur_buf = 0;
  for (int k_tile = k_start; k_tile < k_end; k_tile += 16) {
    const int next_buf = 1 - cur_buf;
    const int k_next = k_tile + 16;

    if (k_next < k_end) {
      dequant_into(next_buf, k_next);
    }

    // Load A: each wave loads its 16M slice, shared across all 4 wmmas.
    const int m_row = m_tile + wave_id * 16 + lane_lo;
    V16 a_frag, b_frag0, b_frag1, b_frag2, b_frag3;
    if (m_row < size_m) {
      const T* a_row = a + m_row * size_k;
      if (b_q_perm) {
  #pragma unroll
        for (int i = 0; i < 16; i++) {
          T v = a_row[b_q_perm[k_tile + i]];
          a_frag[i] = bitcast_elem<T, E>(v);
        }
      } else {
        static_assert(sizeof(a_frag) == 32, "V16 must be 32 bytes");
        __builtin_memcpy(&a_frag, a_row + k_tile, sizeof(a_frag));
      }
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }

    // Load B for all four 16-col halves.
  #pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag0[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo + 0]);
      b_frag1[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo + 16]);
      b_frag2[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo + 32]);
      b_frag3[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo + 48]);
    }

    // Four wmmas per wave per K-iter, sharing a_frag.
    c_acc0 = wmma_mma(a_frag, b_frag0, c_acc0);
    c_acc1 = wmma_mma(a_frag, b_frag1, c_acc1);
    c_acc2 = wmma_mma(a_frag, b_frag2, c_acc2);
    c_acc3 = wmma_mma(a_frag, b_frag3, c_acc3);

    __syncthreads();
    cur_buf = next_buf;
  }

  // ---- Store C ---- Each wave owns 16M × 64N. Helper writes one acc slice.
  const int m_tile_wave = m_tile + wave_id * 16;
  auto store_acc = [&](const v8fp32& acc, int n_base) {
    if (gridDim.z > 1) {
      const bool is_even_lane = (lane_lo & 1) == 0;
      const int out_n_pair = n_base + lane_lo;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        float other_f = __shfl_xor(acc[i], 1);
        if (!is_even_lane) continue;

        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m >= size_m || out_n_pair >= size_n) continue;

        T* dst = c + out_m * size_n + out_n_pair;
        if constexpr (std::is_same<T, half>::value) {
          half2 packed =
              __halves2half2(__float2half_rn(acc[i]), __float2half_rn(other_f));
          atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
        } else {
          bf162_t packed;
          packed.x = __float2bfloat16(acc[i]);
          packed.y = __float2bfloat16(other_f);
          atomic_add_pk_bf16(reinterpret_cast<bf162_t*>(dst), packed);
        }
      }
    } else {
      const int out_n = n_base + lane_lo;
      if (out_n >= size_n) return;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value) {
            *dst = __float2half_rn(acc[i]);
          } else {
            *dst = __float2bfloat16(acc[i]);
          }
        }
      }
    }
  };

  store_acc(c_acc0, n_tile + 0);
  store_acc(c_acc1, n_tile + 16);
  store_acc(c_acc2, n_tile + 32);
  store_acc(c_acc3, n_tile + 48);
}

// ===========================================================================
// 128x64_k16 kernel: 8 waves per block, 128M × 64N tile, K=16 per iteration.
//
// Doubles M-tile from 64 → 128. Each B-tile in LDS is reused by 8 waves
// (8 independent A-row slices) instead of 4, halving the effective B-load
// cost per output element. This matches Hybrid Triton's BLOCK_M=128.
//
// Dequant: same 128 slots as V5 (64N × 2 K-octets). Only waves 0-3
//   participate in dequant (same mapping). Waves 4-7 are pure compute.
// LDS: [2][16][64] × sizeof(T) — unchanged from V5.
// ===========================================================================

template <typename T>
__global__ void gemm_q4_wmma_kernel_128x64_k16(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 128;  // 128-row M tile
  const int n_tile = blockIdx.x * 64;
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int tid = threadIdx.x;   // 0..255
  const int wave_id = tid >> 5;  // 0..7
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  v8fp32 c_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;

  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  // LDS with K-contiguous layout [buf][N][K] — enables vectorized ds_load_b128.
  // Triton uses this layout to read 8 bf16 per instruction instead of 1.
  __shared__ T b_lds[2][64][16];

  // Dequant with scale/zero caching: 1 global_load per iter (7/8 of the time).
  const bool dq_ok =
      (wave_id < 4) && (n_tile + wave_id * 16 + lane_lo < size_n);
  const int dq_n = wave_id * 16 + lane_lo;
  const int dq_an = n_tile + dq_n;
  const int dq_oct = lane_hi;
  const int dq_kb = dq_oct * 8;
  half2 ch_z = {}, ch_y = {};
  float cf_z = 0, cf_y = 0;
  int cached_g = -1;

  auto dequant_into = [&](int buf, int k_tile) __attribute__((always_inline)) {
    if (!dq_ok) return;

    // Reload scale/zero only on group boundary (every groupsize/16 iters).
    const int g = k_tile / groupsize;
    if (g != cached_g) {
      cached_g = g;
      const int qz_idx = g * (size_n / 8) + dq_an / 8;
      const uint32_t zero_v = ((b_qzeros[qz_idx] >> ((dq_an & 7) * 4)) & 0xF) +
                              (uint32_t)zero_offset;
      const T sc = b_scales[g * size_n + dq_an];
      if constexpr (std::is_same<T, half>::value)
        prep_zero_scale_fp16_precise(zero_v, sc, ch_z, ch_y);
      else
        prep_zero_scale_bf16_f32(zero_v, sc, cf_z, cf_y);
    }

    const int qk_row = (k_tile / 8) + dq_oct;
    const uint32_t qa = b_q[qk_row * size_n + dq_an];
    const int k_base = dq_kb;

    if constexpr (std::is_same<T, half>::value) {
      half2 dq[4];
      dequant_4bit_8_fp16_precise(qa, dq, ch_z, ch_y);
      b_lds[buf][dq_n][k_base + 0] = __low2half(dq[0]);
      b_lds[buf][dq_n][k_base + 1] = __high2half(dq[0]);
      b_lds[buf][dq_n][k_base + 2] = __low2half(dq[1]);
      b_lds[buf][dq_n][k_base + 3] = __high2half(dq[1]);
      b_lds[buf][dq_n][k_base + 4] = __low2half(dq[2]);
      b_lds[buf][dq_n][k_base + 5] = __high2half(dq[2]);
      b_lds[buf][dq_n][k_base + 6] = __low2half(dq[3]);
      b_lds[buf][dq_n][k_base + 7] = __high2half(dq[3]);
    } else {
      bf162_t dq[4];
      dequant_4bit_8_bf16_to_bf16(qa, dq, cf_z, cf_y);
      b_lds[buf][dq_n][k_base + 0] = dq[0].x;
      b_lds[buf][dq_n][k_base + 1] = dq[0].y;
      b_lds[buf][dq_n][k_base + 2] = dq[1].x;
      b_lds[buf][dq_n][k_base + 3] = dq[1].y;
      b_lds[buf][dq_n][k_base + 4] = dq[2].x;
      b_lds[buf][dq_n][k_base + 5] = dq[2].y;
      b_lds[buf][dq_n][k_base + 6] = dq[3].x;
      b_lds[buf][dq_n][k_base + 7] = dq[3].y;
    }
  };

  dequant_into(0, k_start);
  __syncthreads();

  int cur_buf = 0;
  const int m_row = m_tile + wave_id * 16 + lane_lo;
  const T* a_row_ptr = (m_row < size_m) ? (a + m_row * size_k) : nullptr;

  for (int k_tile = k_start; k_tile < k_end; k_tile += 16) {
    const int next_buf = 1 - cur_buf;
    const int k_next = k_tile + 16;

    if (k_next < k_end) {
      dequant_into(next_buf, k_next);
    }

    // A-load: vectorized 256-bit. b_q_perm branch removed from V7 hot path
    // to eliminate ~450 ISA instructions of dead code (6× icache bloat).
    V16 a_frag, b_frag0, b_frag1, b_frag2, b_frag3;
    if (a_row_ptr) {
      __builtin_memcpy(&a_frag, a_row_ptr + k_tile, sizeof(a_frag));
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }

    // Vectorized LDS reads: 32 bytes (16 bf16) per b_frag → ds_load_b128 × 2.
    static_assert(sizeof(V16) == 32, "V16 must be 32 bytes for memcpy");
    __builtin_memcpy(&b_frag0, &b_lds[cur_buf][lane_lo + 0][0], 32);
    __builtin_memcpy(&b_frag1, &b_lds[cur_buf][lane_lo + 16][0], 32);
    __builtin_memcpy(&b_frag2, &b_lds[cur_buf][lane_lo + 32][0], 32);
    __builtin_memcpy(&b_frag3, &b_lds[cur_buf][lane_lo + 48][0], 32);

    c_acc0 = wmma_mma(a_frag, b_frag0, c_acc0);
    c_acc1 = wmma_mma(a_frag, b_frag1, c_acc1);
    c_acc2 = wmma_mma(a_frag, b_frag2, c_acc2);
    c_acc3 = wmma_mma(a_frag, b_frag3, c_acc3);

    __syncthreads();
    cur_buf = next_buf;
  }

  // ---- Store C ---- Each wave owns 16M × 64N.
  const int m_tile_wave = m_tile + wave_id * 16;
  auto store_acc = [&](const v8fp32& acc, int n_base) {
    if (gridDim.z > 1) {
      const bool is_even_lane = (lane_lo & 1) == 0;
      const int out_n_pair = n_base + lane_lo;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        float other_f = __shfl_xor(acc[i], 1);
        if (!is_even_lane) continue;

        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m >= size_m || out_n_pair >= size_n) continue;

        T* dst = c + out_m * size_n + out_n_pair;
        if constexpr (std::is_same<T, half>::value) {
          half2 packed =
              __halves2half2(__float2half_rn(acc[i]), __float2half_rn(other_f));
          atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
        } else {
          bf162_t packed;
          packed.x = __float2bfloat16(acc[i]);
          packed.y = __float2bfloat16(other_f);
          atomic_add_pk_bf16(reinterpret_cast<bf162_t*>(dst), packed);
        }
      }
    } else {
      const int out_n = n_base + lane_lo;
      if (out_n >= size_n) return;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value) {
            *dst = __float2half_rn(acc[i]);
          } else {
            *dst = __float2bfloat16(acc[i]);
          }
        }
      }
    }
  };

  store_acc(c_acc0, n_tile + 0);
  store_acc(c_acc1, n_tile + 16);
  store_acc(c_acc2, n_tile + 32);
  store_acc(c_acc3, n_tile + 48);
}

// ===========================================================================
// 128x64_k32 kernel: K=32 per iteration, all 8 waves dequant.
//
// Same 128M × 64N tile as V7, but processes 32 K-elements per iteration
// instead of 16.  Halves iteration count and __syncthreads() calls.
//
// Dequant mapping:
//   Waves 0-3 dequant N[wave*16 +: 16] × K[0:15]  (same as V7)
//   Waves 4-7 dequant N[(wave-4)*16 +: 16] × K[16:31]  (new)
//
// LDS layout: b_lds[2][64][34] — 2 extra padding elements per row to
// avoid 8-way bank conflicts (row stride 68 bytes gives 16 unique banks
// across the 16 lanes).
//
// Per iteration: 8 WMMAs (2 A-frags × 4 N-groups) vs V7's 4.
// Requires K divisible by 32 and groupsize ≥ 32.
// ===========================================================================
template <typename T>
__global__ void gemm_q4_wmma_kernel_128x64_k32(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 128;
  const int n_tile = blockIdx.x * 64;
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int tid = threadIdx.x;   // 0..255
  const int wave_id = tid >> 5;  // 0..7
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  v8fp32 c_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;

  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  // Padded LDS: +2 elements per row to break bank conflicts.
  // Row stride = 34 elements × 2 bytes = 68 bytes → 16 unique banks.
  __shared__ T b_lds[2][64][34];

  // All 8 waves dequant. Waves 0-3 fill K[0:15], waves 4-7 fill K[16:31].
  const int dq_wave4 = wave_id & 3;  // 0-3 for both halves
  const int dq_k_half = (wave_id >= 4) ? 1 : 0;
  const int dq_n = dq_wave4 * 16 + lane_lo;  // N position 0..63
  const int dq_an = n_tile + dq_n;
  const bool dq_ok = (dq_an < size_n);
  const int dq_oct = lane_hi + dq_k_half * 2;  // K octet 0-3
  const int dq_kb = dq_oct * 8;                // K base 0,8,16,24
  half2 ch_z = {}, ch_y = {};
  float cf_z = 0, cf_y = 0;
  int cached_g = -1;

  auto dequant_into = [&](int buf, int k_tile) __attribute__((always_inline)) {
    if (!dq_ok) return;

    const int g = k_tile / groupsize;
    if (g != cached_g) {
      cached_g = g;
      const int qz_idx = g * (size_n / 8) + dq_an / 8;
      const uint32_t zero_v = ((b_qzeros[qz_idx] >> ((dq_an & 7) * 4)) & 0xF) +
                              (uint32_t)zero_offset;
      const T sc = b_scales[g * size_n + dq_an];
      if constexpr (std::is_same<T, half>::value)
        prep_zero_scale_fp16_precise(zero_v, sc, ch_z, ch_y);
      else
        prep_zero_scale_bf16_f32(zero_v, sc, cf_z, cf_y);
    }

    const int qk_row = (k_tile / 8) + dq_oct;
    const uint32_t qa = b_q[qk_row * size_n + dq_an];

    if constexpr (std::is_same<T, half>::value) {
      half2 dq[4];
      dequant_4bit_8_fp16_precise(qa, dq, ch_z, ch_y);
      b_lds[buf][dq_n][dq_kb + 0] = __low2half(dq[0]);
      b_lds[buf][dq_n][dq_kb + 1] = __high2half(dq[0]);
      b_lds[buf][dq_n][dq_kb + 2] = __low2half(dq[1]);
      b_lds[buf][dq_n][dq_kb + 3] = __high2half(dq[1]);
      b_lds[buf][dq_n][dq_kb + 4] = __low2half(dq[2]);
      b_lds[buf][dq_n][dq_kb + 5] = __high2half(dq[2]);
      b_lds[buf][dq_n][dq_kb + 6] = __low2half(dq[3]);
      b_lds[buf][dq_n][dq_kb + 7] = __high2half(dq[3]);
    } else {
      bf162_t dq[4];
      dequant_4bit_8_bf16_to_bf16(qa, dq, cf_z, cf_y);
      b_lds[buf][dq_n][dq_kb + 0] = dq[0].x;
      b_lds[buf][dq_n][dq_kb + 1] = dq[0].y;
      b_lds[buf][dq_n][dq_kb + 2] = dq[1].x;
      b_lds[buf][dq_n][dq_kb + 3] = dq[1].y;
      b_lds[buf][dq_n][dq_kb + 4] = dq[2].x;
      b_lds[buf][dq_n][dq_kb + 5] = dq[2].y;
      b_lds[buf][dq_n][dq_kb + 6] = dq[3].x;
      b_lds[buf][dq_n][dq_kb + 7] = dq[3].y;
    }
  };

  dequant_into(0, k_start);
  __syncthreads();

  int cur_buf = 0;
  const int m_row = m_tile + wave_id * 16 + lane_lo;
  const T* a_row_ptr = (m_row < size_m) ? (a + m_row * size_k) : nullptr;

  for (int k_tile = k_start; k_tile < k_end; k_tile += 32) {
    const int next_buf = 1 - cur_buf;
    const int k_next = k_tile + 32;

    if (k_next < k_end) {
      dequant_into(next_buf, k_next);
    }

    // A-load: two 16-element fragments for K[0:15] and K[16:31].
    V16 a_frag_lo, a_frag_hi;
    V16 b_frag0, b_frag1, b_frag2, b_frag3;
    if (a_row_ptr) {
      __builtin_memcpy(&a_frag_lo, a_row_ptr + k_tile, sizeof(V16));
      __builtin_memcpy(&a_frag_hi, a_row_ptr + k_tile + 16, sizeof(V16));
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag_lo[i] = (E)0;
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag_hi[i] = (E)0;
    }

    // --- Lower K half [0:15]: 4 WMMAs ---
    static_assert(sizeof(V16) == 32, "V16 must be 32 bytes for memcpy");
    __builtin_memcpy(&b_frag0, &b_lds[cur_buf][lane_lo + 0][0], 32);
    __builtin_memcpy(&b_frag1, &b_lds[cur_buf][lane_lo + 16][0], 32);
    __builtin_memcpy(&b_frag2, &b_lds[cur_buf][lane_lo + 32][0], 32);
    __builtin_memcpy(&b_frag3, &b_lds[cur_buf][lane_lo + 48][0], 32);

    c_acc0 = wmma_mma(a_frag_lo, b_frag0, c_acc0);
    c_acc1 = wmma_mma(a_frag_lo, b_frag1, c_acc1);
    c_acc2 = wmma_mma(a_frag_lo, b_frag2, c_acc2);
    c_acc3 = wmma_mma(a_frag_lo, b_frag3, c_acc3);

    // --- Upper K half [16:31]: 4 WMMAs ---
    __builtin_memcpy(&b_frag0, &b_lds[cur_buf][lane_lo + 0][16], 32);
    __builtin_memcpy(&b_frag1, &b_lds[cur_buf][lane_lo + 16][16], 32);
    __builtin_memcpy(&b_frag2, &b_lds[cur_buf][lane_lo + 32][16], 32);
    __builtin_memcpy(&b_frag3, &b_lds[cur_buf][lane_lo + 48][16], 32);

    c_acc0 = wmma_mma(a_frag_hi, b_frag0, c_acc0);
    c_acc1 = wmma_mma(a_frag_hi, b_frag1, c_acc1);
    c_acc2 = wmma_mma(a_frag_hi, b_frag2, c_acc2);
    c_acc3 = wmma_mma(a_frag_hi, b_frag3, c_acc3);

    __syncthreads();
    cur_buf = next_buf;
  }

  // ---- Store C ---- Same as V7.
  const int m_tile_wave = m_tile + wave_id * 16;
  auto store_acc = [&](const v8fp32& acc, int n_base) {
    if (gridDim.z > 1) {
      const bool is_even_lane = (lane_lo & 1) == 0;
      const int out_n_pair = n_base + lane_lo;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        float other_f = __shfl_xor(acc[i], 1);
        if (!is_even_lane) continue;

        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m >= size_m || out_n_pair >= size_n) continue;

        T* dst = c + out_m * size_n + out_n_pair;
        if constexpr (std::is_same<T, half>::value) {
          half2 packed =
              __halves2half2(__float2half_rn(acc[i]), __float2half_rn(other_f));
          atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
        } else {
          bf162_t packed;
          packed.x = __float2bfloat16(acc[i]);
          packed.y = __float2bfloat16(other_f);
          atomic_add_pk_bf16(reinterpret_cast<bf162_t*>(dst), packed);
        }
      }
    } else {
      const int out_n = n_base + lane_lo;
      if (out_n >= size_n) return;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value) {
            *dst = __float2half_rn(acc[i]);
          } else {
            *dst = __float2bfloat16(acc[i]);
          }
        }
      }
    }
  };

  store_acc(c_acc0, n_tile + 0);
  store_acc(c_acc1, n_tile + 16);
  store_acc(c_acc2, n_tile + 32);
  store_acc(c_acc3, n_tile + 48);
}

#else  // non-RDNA3 device pass: empty kernels for symbol parity (covers the
       // three kernels that share this launcher).
template <typename T>
__global__ void gemm_q4_wmma_kernel_64x64_4w(const T*, const uint32_t*,
                                             const uint32_t*, const T*, T*,
                                             const int, const int, const int,
                                             const int, const int, const int*) {
}
template <typename T>
__global__ void gemm_q4_wmma_kernel_128x64_k16(const T*, const uint32_t*,
                                               const uint32_t*, const T*, T*,
                                               const int, const int, const int,
                                               const int, const int,
                                               const int*) {}
template <typename T>
__global__ void gemm_q4_wmma_kernel_128x64_k32(const T*, const uint32_t*,
                                               const uint32_t*, const T*, T*,
                                               const int, const int, const int,
                                               const int, const int,
                                               const int*) {}
#endif

template <typename T>
void launch_gemm_q4_wmma_64x64_4w(const T* a, const uint32_t* b_q_weight,
                                  const uint32_t* b_qzeros, const T* b_scales,
                                  const int* b_q_perm, T* c, int size_m,
                                  int size_n, int size_k, int groups,
                                  int zero_offset, cudaStream_t stream) {
  // Fall back to v4 when N < 64 (would waste 1+ waves on out-of-range cols).
  if (size_m < 64 || size_n < 64) {
    launch_gemm_q4_wmma_64x32_4w<T>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
    return;
  }

  // V8 (128M × 64N, K=32/iter, 8-wave dequant) when K%32==0 and gs≥32.
  // Falls back to V7 otherwise. V7/V8 read A sequentially, so act-order
  // (b_q_perm != null) must skip them and use v5, which honors the perm.
  if (size_m >= 128 && b_q_perm == nullptr) {
    const int k_split =
        compute_wmma_k_split_mn(size_m, size_n, size_k, 128, 64);
    const int groupsize = size_k / groups;
    dim3 block(256);
    dim3 grid((size_n + 63) / 64, (size_m + 127) / 128, k_split);
    if (size_k % 32 == 0 && groupsize >= 32 && (size_k / k_split) % 32 == 0) {
      gemm_q4_wmma_kernel_128x64_k32<T><<<grid, block, 0, stream>>>(
          a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
          zero_offset, b_q_perm);
    } else {
      gemm_q4_wmma_kernel_128x64_k16<T><<<grid, block, 0, stream>>>(
          a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
          zero_offset, b_q_perm);
    }
    return;
  }

  // 4 waves per block (128 threads), 64M × 64N tile per block.
  const int k_split = compute_wmma_k_split_mn(size_m, size_n, size_k, 64, 64);
  dim3 block(128);
  dim3 grid((size_n + 63) / 64, (size_m + 63) / 64, k_split);
  gemm_q4_wmma_kernel_64x64_4w<T><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
}

}  // namespace gptq_rdna3_wmma
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
//
// Inputs:
//   a         [M, K]            half or bfloat16
//   b_q_weight[K/8, N]          uint32 (already shuffled via gptq_shuffle)
//   b_qzeros  [groups, N/8]     uint32 (packed 4-bit zeros)
//   b_scales  [groups, N]       half or bfloat16
//   b_g_idx   [K] or empty      int32 (act-order permutation; empty=identity)
//   use_v2_format               bool   (true = GPTQv2, no +1 zero offset)
//
// Output:
//   c         [M, N]            same dtype as a
//
// Requirements:
//   * size_m >= 16 (otherwise prefer the scalar gptq_gemm_rdna3 op)
//   * size_n % 16 == 0 (WMMA tile size)
//   * size_k % 16 == 0 (WMMA tile size)

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales,
                                   torch::Tensor b_g_idx, bool use_v2_format) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_qzeros.is_cuda(), "b_qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N]");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
      "a must be half or bfloat16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(1);
  int groups = (int)b_qzeros.size(0);

  TORCH_CHECK(b_q_weight.size(0) * 8 == size_k,
              "b_q_weight first dim must be K/8");
  TORCH_CHECK(b_scales.size(0) == groups,
              "b_scales must have same group count as qzeros");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales last dim must be N");
  TORCH_CHECK(size_n % 16 == 0, "WMMA path requires N % 16 == 0");
  TORCH_CHECK(size_k % 16 == 0, "WMMA path requires K % 16 == 0");

  auto opts = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  // Always zero-init the output: some V3-V8 boundary threads may exit
  // without writing their output cell (e.g. out_m >= size_m), leaving
  // uninitialized garbage when torch::empty is used.  The cost is
  // negligible (< 1.5% of prefill time on gfx1100).
  at::Tensor c = torch::zeros({size_m, size_n}, opts);

  const int* g_idx_ptr = nullptr;
  if (!b_g_idx.device().is_meta() && b_g_idx.numel() > 0) {
    TORCH_CHECK(b_g_idx.scalar_type() == torch::kInt32,
                "b_g_idx must be int32");
    g_idx_ptr = (const int*)b_g_idx.data_ptr();
  }

  const int zero_offset = use_v2_format ? 0 : 1;

  // launch_gemm_q4_wmma_64x64_4w dispatches:
  //   M >= 128           → 128x64_k32 / 128x64_k16 (8 waves, K=32/16 per iter)
  //   64 <= M < 128 & N >= 64 → 64x64_4w (4 waves, 4 wmma/wave/K-iter)
  //   M >= 64 && 32 <= N < 64 → 64x32_4w (4 waves, 2 wmma/wave/K-iter)
  //   M >= 64 && N <  32 → 64x16_4w (4 waves)
  //   32 <= M < 64       → 32x16_2w (2 waves)
  //   M < 32             → 16x16_1w (1 wave)
  if (a.scalar_type() == torch::kHalf) {
    vllm::gptq_rdna3_wmma::launch_gemm_q4_wmma_64x64_4w<half>(
        (const half*)a.data_ptr(), (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(), (const half*)b_scales.data_ptr(),
        g_idx_ptr, (half*)c.data_ptr(), size_m, size_n, size_k, groups,
        zero_offset, stream);
  } else {
    vllm::gptq_rdna3_wmma::launch_gemm_q4_wmma_64x64_4w<
        vllm::gptq_rdna3_wmma::bf16_t>(
        (const vllm::gptq_rdna3_wmma::bf16_t*)a.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(),
        (const vllm::gptq_rdna3_wmma::bf16_t*)b_scales.data_ptr(), g_idx_ptr,
        (vllm::gptq_rdna3_wmma::bf16_t*)c.data_ptr(), size_m, size_n, size_k,
        groups, zero_offset, stream);
  }

  return c;
}
