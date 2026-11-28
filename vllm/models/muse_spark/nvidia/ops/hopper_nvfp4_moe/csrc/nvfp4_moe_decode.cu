// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <atomic>
#include <climits>
#include <cstdint>

#include "nvfp4_moe_common.cuh"

namespace vllm::muse_spark_hopper_nvfp4_moe::decode {

using bf16 = __nv_bfloat16;

// ---------------------------------------------------------------------------
// Architecture and shared-memory layout constants (Muse Spark NVFP4 MoE)
// ---------------------------------------------------------------------------

constexpr int kHiddenDim = 4096;
constexpr int kIntermediateDim = 4096;
constexpr int kNumLocalExperts = 32;
constexpr int kTopK = 8;
constexpr int kMaxTokensBase = 1023;
constexpr int kMaxRowsBase = kMaxTokensBase * kTopK;  // 8184
constexpr int kMaxTokens = 1536;
constexpr int kMaxRowsExt = kMaxTokens * kTopK;  // 12288

constexpr int kKbn = 64;        // 4096 / 64 K-blocks
constexpr int kChunk = 9216;    // Bytes per 64-K x 256-row packed chunk
constexpr int kUnit = 1152;     // kChunk / 8 (32 output rows per unit)
constexpr int kWgChunk = 2304;  // kChunk / 4 (64 output rows per warpgroup)
constexpr int kWBytes = 36864;  // 4 * kChunk per weight pipeline stage
constexpr int kWStages = 3;     // TMA weight pipeline depth
constexpr int kXSlot = 16384;   // Activation SMEM stage bytes
constexpr int kXStages = 4;     // cp.async activation pipeline depth
constexpr int kItemRing = 4;    // Producer-to-consumer item ring depth
constexpr int kItemWords = 8;   // int32 words per work item descriptor

// Weight-packing geometry constants (inverse of dequant_fp4x8 / decode_scale).
constexpr int kRawBytesPerRow = kHiddenDim / 2;  // 2048
constexpr int kScalesPerRow = kHiddenDim / 16;   // 256
constexpr int kSubTileFp4Bytes = 2048;           // 4 warps * 512 B
constexpr int kSubTileWarpFp4Bytes = 512;        // 32 lanes * 16 B (uint4)
constexpr int kSubTileWarpScaleBytes = 64;       // 32 lanes * 2 B (uint16)
constexpr int kE5Bias = 16;
// Largest biased E5M3 exponent that decode_scale_pair decodes to a finite
// BF16: (code << 4) + 0x7300 has BF16 exponent e5 + 230 <= 254 (e5 <= 24,
// |scale| <= 480; float8_e4m3fn tops out at 448).
constexpr int kE5MaxBiasedExp = 24;

constexpr int kThreads = 640;
constexpr int kNumWarps = kThreads / 32;  // 20 warps
constexpr int kNumConsumerWarps = 16;     // Warps 4..19
constexpr int kCompletionWarp = 3;        // Warp 3
constexpr int kNumGatherThreads = 64;     // Warps 1..2

constexpr int kNmax = 24;       // Max tokens per tile for cold MMA path
constexpr int kNmaxSmall = 16;  // Max tokens per tile for T <= 16 MMA path
constexpr int kHotMin = 8;      // Threshold above which moe_main uses WGMMA
constexpr int kNww = 128;       // Max tokens per tile for WGMMA path

// Hopper SM90 register-file partition (zero spill budget).
constexpr int kRegProducer = 64;
constexpr int kRegConsumer = 104;
static_assert((96 - kRegProducer) * 128 >= (kRegConsumer - 96) * 512,
              "Producer register release must cover consumer register demand");

// Fixed SMEM prefix offsets (relative to 1024-byte-aligned base).
constexpr int kXchBytes = 24576;
constexpr int kOffX = 0;
constexpr int kOffW = kOffX + kXStages * kXSlot;
constexpr int kOffXch = kOffW + kWStages * kWBytes;
constexpr int kOffTok = kOffXch + kXchBytes;
constexpr int kGsmOff =
    128;  // Inside kOffSch; reuses order[32] after sort read
constexpr int kNumPstRows = 16;
constexpr int kNumBars = 2 * kWStages + 2 * kXStages + 2 * kItemRing;

static_assert(kMaxRowsBase <= kXchBytes,
              "Routing local-expert scratch borrows exchange buffer");

// Base SMEM layout for 1 <= T <= 1023 (moe_main and moe_main_small_batch).
// Layout inside kOffSch:
//   Contiguous == true (512 B):
//     [0, 128)      cnt[32] (int32)
//     [128, 256)    order[32] (int32, dead once warp 0 reads er = order[lane])
//     [128, 512)    gsm[96] = gs13_gate[32], gs13_up[32], gs2[32] (float)
//   Contiguous == false (832 B, non-contiguous expert_map fallback):
//     [0, 512)      same as above
//     [512, 768)    pxm[256] (uint8)
//     [768, 800)    psl[32] (uint8)
template <bool Contiguous>
struct SmemLayout {
  static constexpr int kOffCntw = kOffTok + kMaxRowsBase * 2;
  static constexpr int kOffSch = kOffCntw + kNumWarps * 32 * 4;
  static constexpr int kSchBytes = Contiguous ? 512 : 832;
  static constexpr int kOffGsm = kOffSch + kGsmOff;
  static constexpr int kOffPxm = 512;
  static constexpr int kOffPsl = 768;

  static constexpr int kOffItem = kOffSch + kSchBytes;
  static constexpr int kOffPst = kOffItem + kItemRing * kItemWords * 4;
  static constexpr int kOffCmb = kOffPst + kNumPstRows * 32 * 4;
  // Combine scratch (216 int32s = 864 B):
  //   [0..31]     need2[32]
  //   [32..63]    done[32]
  //   [64]        tcl[0]
  //   [65]        ctrl_off (0 or kCtrlSet)
  //   [66]        tcl[1]
  //   [67]        padding
  //   [68..195]   red[128] (float)
  //   [196..203]  twv[8] (float)
  //   [204..211]  slv[8] (int)
  //   [212..213]  has_loc[2] (uint32)
  //   [214..215]  padding (16-byte alignment for kOffBar)
  static constexpr int kOffBar = kOffCmb + 216 * 4;
  static constexpr int kSmemUsed = kOffBar + kNumBars * 8;
  static constexpr int kSmemBytes = kSmemUsed + 1024;

  static_assert(
      kGsmOff + 3 * kNumLocalExperts * static_cast<int>(sizeof(float)) <= 512,
      "Split global scales must fit inside the 512-byte kOffSch block");
  static_assert((kOffBar % 16) == 0, "mbarrier array must be 16-byte aligned");
  static_assert(kSmemBytes <= 232448,
                "Dynamic SMEM exceeds Hopper 227 KB opt-in limit");
};

// Zero-overflow extended SMEM layout for 1024 <= T <= 1536 (moe_main_ext).
// Compacts local routes into kOffX (65536 B >= 12288 * 4 B) during Pass 1 and
// stores per-warp counts/prefix sums in uint16_t cntw[20 * 32] (1280 B; every
// per-warp count <= 1024 and every prefix sum <= 12288 < 65535).
// Uses unused padding slot kOffCmb + 214 * 4 for nloc_smem so it never
// overlaps cnt, order, gsm, pxm, psl, or any combine scratch entry.
template <bool Contiguous>
struct SmemLayoutExt {
  static constexpr int kOffCntw = kOffTok + kMaxRowsExt * 2;
  static constexpr int kOffSch = kOffCntw + kNumWarps * 32 * 2;
  static constexpr int kSchBytes = Contiguous ? 512 : 832;
  static constexpr int kOffGsm = kOffSch + kGsmOff;
  static constexpr int kOffPxm = 512;
  static constexpr int kOffPsl = 768;

  static constexpr int kOffItem = kOffSch + kSchBytes;
  static constexpr int kOffPst = kOffItem + kItemRing * kItemWords * 4;
  static constexpr int kOffCmb = kOffPst + kNumPstRows * 32 * 4;
  static constexpr int kOffNloc = kOffCmb + 214 * 4;
  static constexpr int kOffBar = kOffCmb + 216 * 4;
  static constexpr int kSmemUsed = kOffBar + kNumBars * 8;
  static constexpr int kSmemBytes = kSmemUsed + 1024;

  static_assert(kMaxRowsExt * static_cast<int>(sizeof(uint32_t)) <=
                    kXStages * kXSlot,
                "Compacted local-route scratch must fit in activation SMEM");
  static_assert(kMaxRowsExt <= 65535,
                "Extended route counts and prefix sums must fit in uint16_t");
  static_assert(
      kGsmOff + 3 * kNumLocalExperts * static_cast<int>(sizeof(float)) <= 512,
      "Split global scales must fit inside the 512-byte kOffSch block");
  static_assert((kOffSch % 16) == 0, "kOffSch must be 16-byte aligned");
  static_assert((kOffBar % 16) == 0, "mbarrier array must be 16-byte aligned");
  static_assert(kSmemBytes <= 232448,
                "Extended dynamic SMEM exceeds Hopper 227 KB opt-in limit");
};

static_assert(SmemLayout<true>::kSmemBytes <= 232448,
              "SmemLayout<true> exceeds 227 KB opt-in limit");
static_assert(SmemLayout<false>::kSmemBytes <= 232448,
              "SmemLayout<false> exceeds 227 KB opt-in limit");
static_assert(SmemLayoutExt<true>::kSmemBytes <= 232448,
              "SmemLayoutExt<true> exceeds 227 KB opt-in limit");
static_assert(SmemLayoutExt<false>::kSmemBytes <= 232448,
              "SmemLayoutExt<false> exceeds 227 KB opt-in limit");

// Persistent control buffer layout (256 uint32 entries):
//   [0..33]     moe_main_small_batch work counter + W13 completion counts
//   [72..143]   moe_main / moe_main_ext epoch-0 control set
//   [144..215]  moe_main / moe_main_ext epoch-1 control set
//   [216]       moe_main / moe_main_ext epoch counter (bit 20 selects set)
constexpr int kCtrlW2 = 34;
constexpr int kCtrlSlot = 66;
constexpr int kCtrlTok = 70;
constexpr int kCtrlSet = 72;
constexpr int kCtrlBase = 72;
constexpr int kCtrlEpoch = kCtrlBase + 2 * kCtrlSet;  // 216

// Tail-split lookup table for small batches (1 <= T <= 16), indexed by
// [T > 8][active_experts - 4] for active_experts in [4, 11].
constexpr unsigned char kTailSplit[2][8] = {
    {0, 3, 5, 5, 0, 2, 0, 0},
    {0, 5, 5, 5, 0, 4, 0, 0},
};

constexpr uint64_t pack_tail_split() {
  uint64_t packed = 0;
  for (int i = 0; i < 16; ++i) {
    packed |= static_cast<uint64_t>(kTailSplit[i / 8][i % 8]) << (4 * i);
  }
  return packed;
}

__host__ __device__ constexpr int kx_of(int np) {
  return np <= 8 ? 16 : (np <= 16 ? 8 : (np <= 32 ? 4 : (np <= 64 ? 2 : 1)));
}

__host__ __device__ constexpr int hw_bucket(int n) {
  return n <= 64 ? 3 + (n > 9 ? n - 9 : 0) / 8 : 9 + (n - 49) / 16;
}

// ---------------------------------------------------------------------------
// Single-launch NVFP4 (E2M1) + E5M3 block-scale weight packer
// ---------------------------------------------------------------------------

// Convert an IEEE/OFP8 float8_e4m3fn byte to exact IEEE-754 float32.
// Handles +/-0, subnormals (0x01..0x07, 0x81..0x87 -> +/-m * 2^-9), normals
// ((1 + m/8) * 2^(e - 7)), and NaN (0x7F, 0xFF).
__device__ __forceinline__ float fp8_e4m3fn_to_f32(uint8_t u) {
  const uint32_t sign = static_cast<uint32_t>(u & 0x80u) << 24;
  const uint32_t exp_m = static_cast<uint32_t>(u & 0x7Fu);
  if (exp_m == 0x7Fu) {
    return __uint_as_float(sign | 0x7FC00000u);  // NaN
  }
  if (exp_m == 0u) {
    return __uint_as_float(sign);  // +/-0.0f
  }
  const uint32_t exp = exp_m >> 3;
  const uint32_t mant = exp_m & 7u;
  if (exp == 0u) {
    const float mag = static_cast<float>(mant) * (1.0f / 512.0f);
    return __uint_as_float(sign | __float_as_uint(mag));
  }
  return __uint_as_float(sign | ((exp + 120u) << 23) | (mant << 20));
}

// Load one block scale element and convert to float32.
// sf_dtype_tag: 0 = float8_e4m3fn, 1 = bfloat16, 2 = float32.
__device__ __forceinline__ float load_scale_f32(const void* __restrict__ raw_sf,
                                                size_t idx, int sf_dtype_tag) {
  if (sf_dtype_tag == 0) {
    return fp8_e4m3fn_to_f32(reinterpret_cast<const uint8_t*>(raw_sf)[idx]);
  }
  if (sf_dtype_tag == 1) {
    const uint16_t raw = reinterpret_cast<const uint16_t*>(raw_sf)[idx];
    return __uint_as_float(static_cast<uint32_t>(raw) << 16);
  }
  return reinterpret_cast<const float*>(raw_sf)[idx];
}

// Validate and encode one FP32 scale value into E5M3 (bias = 16).
// Returns true iff finite and exactly representable in E5M3 (or +/-0.0f).
// Pure bit-level logic (immune to --use_fast_math):
//   - Non-finite (NaN / +/-Inf) -> invalid (returns false).
//   - s == 0.0f (+/-0.0f) -> zero_out = true, neg_out = false, e5m3 = 0x80.
//   - s != 0.0f -> neg_out = (s < 0), exact iff 0 <= E + 16 <= kE5MaxBiasedExp
//     (24) and all 20 trailing FP32 fraction bits below the top 3 mantissa bits
//     are zero.
__device__ __forceinline__ bool encode_scale_f32(float s, uint8_t& e5m3_out,
                                                 bool& neg_out,
                                                 bool& zero_out) {
  const uint32_t bits = __float_as_uint(s);
  if ((bits & 0x7F800000u) == 0x7F800000u) {
    e5m3_out = 0x80u;
    neg_out = false;
    zero_out = true;
    return false;
  }
  const uint32_t abits = bits & 0x7FFFFFFFu;
  if (abits == 0u) {
    e5m3_out = static_cast<uint8_t>(kE5Bias << 3);  // 0x80
    neg_out = false;
    zero_out = true;
    return true;
  }
  const uint32_t biased_e5 =
      (abits >> 23) - static_cast<uint32_t>(127 - kE5Bias);
  const uint32_t trailing = abits & 0x000FFFFFu;
  if (biased_e5 > static_cast<uint32_t>(kE5MaxBiasedExp) || trailing != 0u) {
    e5m3_out = 0x80u;
    neg_out = false;
    zero_out = true;
    return false;
  }
  const uint32_t m3 = (abits >> 20) & 7u;
  e5m3_out = static_cast<uint8_t>((biased_e5 << 3) | m3);
  neg_out = (bits >> 31) != 0u;
  zero_out = false;
  return true;
}

// Place one pair of FP4 codes (b = c0 | (c1 << 4)) into pair-0 bit positions:
//   (0, 0): bits 0,1,2,3 -> (6, 7, 8, 15)
//   (0, 1): bits 0,1,2,3 -> (22, 23, 24, 31)
__device__ __forceinline__ uint32_t place_bf16_pair(uint32_t b) {
  return ((b & 0x07u) << 6) | ((b & 0x08u) << 12) | ((b & 0x70u) << 18) |
         ((b & 0x80u) << 24);
}

// Place one pair of FP4 codes (b = c0 | (c1 << 4)) into pair-3 bit positions:
//   (3, 0): bits 0,1,2,3 -> (3, 4, 0, 1)
//   (3, 1): bits 0,1,2,3 -> (19, 20, 16, 17)
__device__ __forceinline__ uint32_t place_p3_pair(uint32_t b) {
  return ((b & 0x03u) << 3) | ((b & 0x0Cu) >> 2) | ((b & 0x30u) << 15) |
         ((b & 0xC0u) << 10);
}

// Pack 4 FP4 pairs (b0..b3) into one 32-bit word (exact inverse of
// dequant_fp4x8):
//   pair 1 = ror32(place_bf16_pair(b1), 10) -> (28,29,30,5) & (12,13,14,21)
//   pair 2 = ror32(place_bf16_pair(b2), 13) -> (25,26,27,2) & (9,10,11,18)
__device__ __forceinline__ uint32_t pack_4pairs(uint32_t b0, uint32_t b1,
                                                uint32_t b2, uint32_t b3) {
  const uint32_t y0 = place_bf16_pair(b0);
  const uint32_t y1 = place_bf16_pair(b1);
  const uint32_t y2 = place_bf16_pair(b2);
  const uint32_t y3 = place_p3_pair(b3);
  return y0 | __funnelshift_r(y1, y1, 10) | __funnelshift_r(y2, y2, 13) | y3;
}

__device__ __forceinline__ uint32_t pick4(int idx, uint32_t a0, uint32_t a1,
                                          uint32_t a2, uint32_t a3) {
  const uint32_t lo = (idx & 1) ? a1 : a0;
  const uint32_t hi = (idx & 1) ? a3 : a2;
  return (idx & 2) ? hi : lo;
}

// Single-launch coalesced NVFP4 + E5M3 weight packer (0 B SMEM, 0 B temp VRAM).
// Grid:  n_experts * n_outer * 64 CTAs (n_outer = 32 for w13, 16 for w2).
// Block: 128 threads (4 warps). Warp wq (0..3) iterates over sub-tiles 0..3.
__global__ void __launch_bounds__(128)
    pack_experts_k(const uint8_t* __restrict__ raw_u8,
                   const void* __restrict__ raw_sf, int sf_dtype_tag,
                   uint8_t* __restrict__ out, int n_experts, int is_w13,
                   int* __restrict__ err_flag) {
  const int n_outer = is_w13 ? 32 : 16;
  const int num_rows = is_w13 ? (2 * kIntermediateDim) : kHiddenDim;

  const int blk = static_cast<int>(blockIdx.x);
  const int kb = blk & (kKbn - 1);
  const int em = blk >> 6;
  const int mb = is_w13 ? (em & 31) : (em & 15);
  const int e = is_w13 ? (em >> 5) : (em >> 4);
  if (e >= n_experts) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int wq = tid >> 5;    // 0..3
  const int lane = tid & 31;  // 0..31
  const int g = lane >> 2;    // 0..7
  const int t = lane & 3;     // 0..3 (ks during load/pack, kq during store)

  uint8_t* const out_chunk =
      out +
      (static_cast<size_t>(e) * static_cast<size_t>(n_outer) +
       static_cast<size_t>(mb)) *
          static_cast<size_t>(kKbn * kChunk) +
      static_cast<size_t>(kb) * static_cast<size_t>(kChunk);

  const size_t expert_row_base =
      static_cast<size_t>(e) * static_cast<size_t>(num_rows);
  const size_t sf_col = static_cast<size_t>(4 * kb + t);
  const size_t u8_col = static_cast<size_t>(32 * kb + 8 * t);

  bool any_bad = false;

#pragma unroll
  for (int tile = 0; tile < 4; ++tile) {
    // Closed-form row permutation:
    //   w13: src_row = 128*mb + 32*tile + 8*wq + g + 4096*hi
    //   w2 : src_row = 256*mb + 64*tile + 16*wq + 8*hi + g (identity)
    const int row0 = is_w13 ? (128 * mb + 32 * tile + 8 * wq + g)
                            : (256 * mb + 64 * tile + 16 * wq + g);
    const int row1 = is_w13 ? (row0 + kIntermediateDim) : (row0 + 8);

    const size_t sf_idx0 =
        (expert_row_base + static_cast<size_t>(row0)) * kScalesPerRow + sf_col;
    const size_t sf_idx1 =
        (expert_row_base + static_cast<size_t>(row1)) * kScalesPerRow + sf_col;

    uint8_t e5m3_0 = 0x80u;
    uint8_t e5m3_1 = 0x80u;
    bool neg0 = false;
    bool zero0 = false;
    bool neg1 = false;
    bool zero1 = false;
    const bool ok0 = encode_scale_f32(
        load_scale_f32(raw_sf, sf_idx0, sf_dtype_tag), e5m3_0, neg0, zero0);
    const bool ok1 = encode_scale_f32(
        load_scale_f32(raw_sf, sf_idx1, sf_dtype_tag), e5m3_1, neg1, zero1);
    any_bad = any_bad || !ok0 || !ok1;

    const size_t u8_off0 =
        (expert_row_base + static_cast<size_t>(row0)) * kRawBytesPerRow +
        u8_col;
    const size_t u8_off1 =
        (expert_row_base + static_cast<size_t>(row1)) * kRawBytesPerRow +
        u8_col;
    const uint2 v0 = *reinterpret_cast<const uint2*>(raw_u8 + u8_off0);
    const uint2 v1 = *reinterpret_cast<const uint2*>(raw_u8 + u8_off1);

    const uint32_t flip0 = neg0 ? 0x88888888u : 0u;
    const uint32_t mask0 = zero0 ? 0u : 0xFFFFFFFFu;
    const uint32_t lo0 = (v0.x ^ flip0) & mask0;
    const uint32_t hi0 = (v0.y ^ flip0) & mask0;

    const uint32_t flip1 = neg1 ? 0x88888888u : 0u;
    const uint32_t mask1 = zero1 ? 0u : 0xFFFFFFFFu;
    const uint32_t lo1 = (v1.x ^ flip1) & mask1;
    const uint32_t hi1 = (v1.y ^ flip1) & mask1;

    // Pack 4 words (kq = 0..3) for this lane's scale block ks = t.
    const uint32_t w0 =
        pack_4pairs(lo0 & 0xFFu, lo1 & 0xFFu, hi0 & 0xFFu, hi1 & 0xFFu);
    const uint32_t w1 = pack_4pairs((lo0 >> 8) & 0xFFu, (lo1 >> 8) & 0xFFu,
                                    (hi0 >> 8) & 0xFFu, (hi1 >> 8) & 0xFFu);
    const uint32_t w2 = pack_4pairs((lo0 >> 16) & 0xFFu, (lo1 >> 16) & 0xFFu,
                                    (hi0 >> 16) & 0xFFu, (hi1 >> 16) & 0xFFu);
    const uint32_t w3 = pack_4pairs(lo0 >> 24, lo1 >> 24, hi0 >> 24, hi1 >> 24);

    // 4x4 transpose across 4-lane sub-warp (ks <-> kq) in 3 XOR shuffles.
    const uint32_t r0 = pick4(t, w0, w1, w2, w3);
    const uint32_t r1 =
        __shfl_xor_sync(kFullWarpMask, pick4(t ^ 1, w0, w1, w2, w3), 1, 4);
    const uint32_t r2 =
        __shfl_xor_sync(kFullWarpMask, pick4(t ^ 2, w0, w1, w2, w3), 2, 4);
    const uint32_t r3 =
        __shfl_xor_sync(kFullWarpMask, pick4(t ^ 3, w0, w1, w2, w3), 3, 4);

    const uint4 out_w =
        make_uint4(pick4(t, r0, r1, r2, r3), pick4(t ^ 1, r0, r1, r2, r3),
                   pick4(t ^ 2, r0, r1, r2, r3), pick4(t ^ 3, r0, r1, r2, r3));

    uint8_t* const out_subtile =
        out_chunk + static_cast<size_t>(tile) * kWgChunk;

    // Coalesced 512 B FP4 store (32 lanes x 16 B) and 64 B scale store.
    reinterpret_cast<uint4*>(out_subtile + static_cast<size_t>(wq) *
                                               kSubTileWarpFp4Bytes)[lane] =
        out_w;
    const uint16_t sc_pair =
        static_cast<uint16_t>(e5m3_0) | (static_cast<uint16_t>(e5m3_1) << 8);
    reinterpret_cast<uint16_t*>(out_subtile + kSubTileFp4Bytes +
                                static_cast<size_t>(wq) *
                                    kSubTileWarpScaleBytes)[lane] = sc_pair;
  }

  // Coalesce error reporting to at most one atomicOr per warp on invalid input.
  if (__any_sync(kFullWarpMask, any_bad) && lane == 0) {
    atomicOr(err_flag, 1);
  }
}

void pack_experts(torch::Tensor raw_w, torch::Tensor raw_sf, bool is_w13,
                  torch::Tensor packed_out, torch::Tensor err_flag) {
  const int w13_flag = is_w13 ? 1 : 0;
  const int n_outer = w13_flag ? 32 : 16;
  const int expected_rows = w13_flag ? (2 * kIntermediateDim) : kHiddenDim;

  TORCH_CHECK(raw_w.is_cuda(), "raw_w must be a CUDA tensor");
  const auto dev = raw_w.device();
  TORCH_CHECK(raw_sf.is_cuda() && raw_sf.device() == dev,
              "raw_sf must be on the same CUDA device as raw_w");
  TORCH_CHECK(packed_out.is_cuda() && packed_out.device() == dev,
              "packed_out must be on the same CUDA device as raw_w");
  TORCH_CHECK(err_flag.is_cuda() && err_flag.device() == dev,
              "err_flag must be on the same CUDA device as raw_w");

  TORCH_CHECK(raw_w.is_contiguous(), "raw_w must be contiguous");
  TORCH_CHECK(raw_sf.is_contiguous(), "raw_sf must be contiguous");
  TORCH_CHECK(packed_out.is_contiguous(), "packed_out must be contiguous");
  TORCH_CHECK(err_flag.is_contiguous(), "err_flag must be contiguous");

  TORCH_CHECK(raw_w.element_size() == 1,
              "raw_w must be a 1-byte dtype (uint8 or float4_e2m1fn_x2)");
  TORCH_CHECK(raw_w.dim() == 3 && raw_w.size(0) > 0 &&
                  raw_w.size(1) == expected_rows &&
                  raw_w.size(2) == kRawBytesPerRow,
              "raw_w must have shape [n_experts, ", expected_rows, ", ",
              kRawBytesPerRow, "], got ", raw_w.sizes());

  const int64_t n_experts = raw_w.size(0);
  TORCH_CHECK(n_experts * n_outer * kKbn <= INT_MAX,
              "n_experts exceeds grid limit: ", n_experts);

  TORCH_CHECK(raw_sf.dim() == 3 && raw_sf.size(0) == n_experts &&
                  raw_sf.size(1) == expected_rows &&
                  raw_sf.size(2) == kScalesPerRow,
              "raw_sf must have shape [", n_experts, ", ", expected_rows, ", ",
              kScalesPerRow, "], got ", raw_sf.sizes());

  int sf_dtype_tag = -1;
  if (raw_sf.scalar_type() == torch::kFloat8_e4m3fn) {
    sf_dtype_tag = 0;
  } else if (raw_sf.scalar_type() == torch::kBFloat16) {
    sf_dtype_tag = 1;
  } else if (raw_sf.scalar_type() == torch::kFloat32) {
    sf_dtype_tag = 2;
  } else {
    TORCH_CHECK(
        false,
        "raw_sf must have dtype float8_e4m3fn, bfloat16, or float32, got ",
        raw_sf.scalar_type());
  }

  TORCH_CHECK(packed_out.scalar_type() == torch::kUInt8,
              "packed_out must have dtype uint8");
  TORCH_CHECK(packed_out.dim() == 4 && packed_out.size(0) == n_experts &&
                  packed_out.size(1) == n_outer && packed_out.size(2) == kKbn &&
                  packed_out.size(3) == kChunk,
              "packed_out must have shape [", n_experts, ", ", n_outer, ", ",
              kKbn, ", ", kChunk, "], got ", packed_out.sizes());

  TORCH_CHECK(err_flag.scalar_type() == torch::kInt32 && err_flag.numel() >= 1,
              "err_flag must be a non-empty int32 CUDA tensor");

  TORCH_CHECK((reinterpret_cast<uintptr_t>(raw_w.data_ptr()) & 7u) == 0u,
              "raw_w data pointer must be 8-byte aligned");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(packed_out.data_ptr()) & 15u) == 0u,
              "packed_out data pointer must be 16-byte aligned");

  const c10::cuda::CUDAGuard device_guard(dev);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev.index());

  const uint32_t grid_blocks =
      static_cast<uint32_t>(n_experts * n_outer * kKbn);
  pack_experts_k<<<dim3(grid_blocks), dim3(128), 0, stream>>>(
      reinterpret_cast<const uint8_t*>(raw_w.data_ptr()), raw_sf.data_ptr(),
      sf_dtype_tag, packed_out.data_ptr<uint8_t>(), static_cast<int>(n_experts),
      w13_flag, err_flag.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Kernel parameter structures and low-level helpers
// ---------------------------------------------------------------------------

struct Params {
  const uint8_t* w13;
  const uint8_t* w2;
  const float* gs13_gate;
  const float* gs13_up;
  const float* gs2;
  const bf16* xm;
  const int* ids;
  bf16* act;
  bf16* y;
  int* slot_of;
  uint32_t* ctrl;
  const float* tw;
  const float* postw;
  bf16* out;
  int T;
  float post_eps;
  uint64_t atab;
  const uint8_t* plan;
  uint32_t smul;
  int local_start;
};

struct Ring {
  int ws;
  int xs;
  uint32_t wph;
  uint32_t xph;
};

struct DecodedItem {
  int ph;
  int e;
  int mg;
  int tt;
  int rl;
  int uoff;
  int ce;
  int oe;
  int tce;
  int nte;
};

__device__ __forceinline__ void red_release_add_u32(uint32_t* ptr,
                                                    uint32_t val) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
}

__device__ __noinline__ void mbar_wait_slow_ni(uint32_t bar, uint32_t parity) {
  while (!mbar_try_hint(bar, parity)) {
  }
}

template <bool InlineSpin>
__device__ __forceinline__ void mbar_wait_t(uint32_t bar, uint32_t parity) {
  if constexpr (InlineSpin) {
    mbar_wait_spin(bar, parity);
  } else {
    if (!mbar_try_hint(bar, parity)) {
      mbar_wait_slow_ni(bar, parity);
    }
  }
}

__device__ __forceinline__ int map_expert_contiguous(int local_start,
                                                     int global_id) {
  const uint32_t d =
      static_cast<uint32_t>(global_id) - static_cast<uint32_t>(local_start);
  return d < static_cast<uint32_t>(kNumLocalExperts) ? static_cast<int>(d)
                                                     : 255;
}

__device__ __forceinline__ int map_expert_plan(const uint8_t* pxm,
                                               int global_id) {
  return static_cast<uint32_t>(global_id) < 256u
             ? static_cast<int>(pxm[global_id])
             : 255;
}

template <bool Contiguous>
__device__ __forceinline__ int map_expert(const Params& p, const uint8_t* pxm,
                                          int global_id) {
  if constexpr (Contiguous) {
    return map_expert_contiguous(p.local_start, global_id);
  } else {
    return map_expert_plan(pxm, global_id);
  }
}

template <int OffCmb>
__device__ __forceinline__ uint32_t* active_ctrl(const Params& p,
                                                 const uint8_t* sm) {
  return p.ctrl + kCtrlBase + reinterpret_cast<const int*>(sm + OffCmb)[65];
}

template <bool Unroll>
__device__ __forceinline__ int warp_inclusive_scan(int v, int lane) {
  if constexpr (Unroll) {
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int yv = __shfl_up_sync(kFullWarpMask, v, d);
      if (lane >= d) {
        v += yv;
      }
    }
  } else {
#pragma unroll 1
    for (int d = 1; d < 32; d <<= 1) {
      const int yv = __shfl_up_sync(kFullWarpMask, v, d);
      if (lane >= d) {
        v += yv;
      }
    }
  }
  return v;
}

// Compute descending sort rank across 32 lanes (ties broken by ascending lane).
__device__ __forceinline__ int warp_descending_rank(int v, int lane) {
  const int mx = __reduce_max_sync(kFullWarpMask, v);
  uint32_t eq = kFullWarpMask;
  int gt = 0;
  for (int bt = 31 - __clz(mx | 1); bt >= 0; --bt) {
    const uint32_t ones = __ballot_sync(kFullWarpMask, (v >> bt) & 1);
    if ((v >> bt) & 1) {
      eq &= ones;
    } else {
      gt += __popc(eq & ones);
      eq &= ~ones;
    }
  }
  return gt + __popc(eq & ((1u << lane) - 1u));
}

// ---------------------------------------------------------------------------
// Warp MMA (cold / small-tile path) and WGMMA (hot-tile path) compute kernels
// ---------------------------------------------------------------------------

template <int N, bool UseFmaMul>
__device__ __forceinline__ void mma_k64(float (&acc0)[N / 2],
                                        float (&acc1)[N / 2], const uint4 q0,
                                        const uint4 q1, uint2 c0, uint2 c1,
                                        uint32_t xs, int lane, uint32_t smul) {
  const int r = lane & 7;
  const int mq = lane >> 3;
  const uint32_t w0[4] = {q0.x, q0.y, q0.z, q0.w};
  const uint32_t w1[4] = {q1.x, q1.y, q1.z, q1.w};
  const uint32_t mul = UseFmaMul ? smul : 16u;
  const uint32_t m0 = decode_scale_quad(c0, lane, mul);
  const uint32_t m1 = decode_scale_quad(c1, lane, mul);

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint4 bh[N / 8];
#pragma unroll
    for (int i = 0; i < N / 8; ++i) {
      uint32_t r_ld[4];
      ldmatrix_x4(xs + (8 * i + r) * 128 + (((4 * h + mq) ^ r) << 4), r_ld);
      bh[i] = make_uint4(r_ld[0], r_ld[1], r_ld[2], r_ld[3]);
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int s = 2 * h + u;
      const uint32_t r0 = scale_for_step(m0, s);
      const uint32_t r1 = scale_for_step(m1, s);
      const uint4 a0 = dequant_fp4x8(w0[s], r0);
      const uint4 a1 = dequant_fp4x8(w1[s], r1);
      const uint32_t a0_r[4] = {a0.x, a0.y, a0.z, a0.w};
      const uint32_t a1_r[4] = {a1.x, a1.y, a1.z, a1.w};
#pragma unroll
      for (int i = 0; i < N / 8; ++i) {
        const uint32_t b0 = u ? bh[i].z : bh[i].x;
        const uint32_t b1 = u ? bh[i].w : bh[i].y;
        mma16816(acc0 + 4 * i, a0_r, b0, b1);
        mma16816(acc1 + 4 * i, a1_r, b0, b1);
      }
    }
  }
}

template <int N>
__device__ __forceinline__ void store_act_col(bf16* dst, int n, int t,
                                              const float* a, float gs_gate,
                                              float gs_up) {
#pragma unroll
  for (int j = 0; j < N / 8; ++j) {
    const int tok = 8 * j + 2 * t;
    if (tok < n) {
      dst[static_cast<size_t>(tok) * kHiddenDim] = __float2bfloat16_rn(
          silu_mul(gs_gate * a[4 * j], gs_up * a[4 * j + 2]));
    }
    if (tok + 1 < n) {
      dst[static_cast<size_t>(tok + 1) * kHiddenDim] = __float2bfloat16_rn(
          silu_mul(gs_gate * a[4 * j + 1], gs_up * a[4 * j + 3]));
    }
  }
}

template <int N>
__device__ __forceinline__ void store_y_col(bf16* dst, int n, int t,
                                            const float* a, float gs) {
#pragma unroll
  for (int j = 0; j < N / 8; ++j) {
    const int tok = 8 * j + 2 * t;
    if (tok < n) {
      dst[static_cast<size_t>(tok) * kHiddenDim] =
          __float2bfloat16_rn(gs * a[4 * j]);
      dst[static_cast<size_t>(tok) * kHiddenDim + 8] =
          __float2bfloat16_rn(gs * a[4 * j + 2]);
    }
    if (tok + 1 < n) {
      dst[static_cast<size_t>(tok + 1) * kHiddenDim] =
          __float2bfloat16_rn(gs * a[4 * j + 1]);
      dst[static_cast<size_t>(tok + 1) * kHiddenDim + 8] =
          __float2bfloat16_rn(gs * a[4 * j + 3]);
    }
  }
}

template <int N, int OffGsm>
__device__ __forceinline__ void epilogue_dual(const Params& p, uint32_t sbase,
                                              int phase, int expert, int mg,
                                              int slot0, int n, int cw, int wl,
                                              int lane,
                                              const float (&acc0)[N / 2],
                                              const float (&acc1)[N / 2]) {
  const int g = lane >> 2;
  const int t = lane & 3;
  const uint32_t gsm_base = sbase + OffGsm;
  if (phase == 1) {
    const float gs_gate =
        ld_shared_f32(gsm_base + 4u * static_cast<uint32_t>(expert));
    const float gs_up =
        ld_shared_f32(gsm_base + 4u * static_cast<uint32_t>(32 + expert));
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const float* a = b ? acc1 : acc0;
      const int col =
          (mg * 4 + 2 * cw + (wl >> 1)) * 32 + (2 * (wl & 1) + b) * 8 + g;
      store_act_col<N>(p.act + static_cast<size_t>(slot0) * kHiddenDim + col, n,
                       t, a, gs_gate, gs_up);
    }
  } else {
    const float gs_down =
        ld_shared_f32(gsm_base + 4u * static_cast<uint32_t>(64 + expert));
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const float* a = b ? acc1 : acc0;
      const int o =
          (mg * 4 + 2 * cw + (wl >> 1)) * 64 + (2 * (wl & 1) + b) * 16 + g;
      store_y_col<N>(p.y + static_cast<size_t>(slot0) * kHiddenDim + o, n, t, a,
                     gs_down);
    }
  }
}

template <int N, int OffGsm>
__device__ __forceinline__ void epilogue_single(const Params& p, uint32_t sbase,
                                                int phase, int expert, int mg,
                                                int slot0, int n, int mb,
                                                int wl, int lane,
                                                const float (&a)[N / 2]) {
  const int g = lane >> 2;
  const int t = lane & 3;
  const uint32_t gsm_base = sbase + OffGsm;
  if (phase == 1) {
    const float gs_gate =
        ld_shared_f32(gsm_base + 4u * static_cast<uint32_t>(expert));
    const float gs_up =
        ld_shared_f32(gsm_base + 4u * static_cast<uint32_t>(32 + expert));
    const int col = (mg * 4 + mb) * 32 + wl * 8 + g;
    store_act_col<N>(p.act + static_cast<size_t>(slot0) * kHiddenDim + col, n,
                     t, a, gs_gate, gs_up);
  } else {
    const float gs_down =
        ld_shared_f32(gsm_base + 4u * static_cast<uint32_t>(64 + expert));
    const int o = (mg * 4 + mb) * 64 + wl * 16 + g;
    store_y_col<N>(p.y + static_cast<size_t>(slot0) * kHiddenDim + o, n, t, a,
                   gs_down);
  }
}

template <int N, int OffGsm, bool InlineSpin = true>
__device__ __forceinline__ void run_item(const Params& p, const int* rec,
                                         Ring& rg, uint32_t sbase, uint32_t bar,
                                         int cwi, int lane) {
  constexpr int kKx = kx_of(N);
  // Cross-K-group partials use (16 - (8 >> rl)) slots of N * 128 B in the
  // exchange buffer; planners emit rl <= 2 for N == 8 and rl <= 1 for N == 16
  // (an exact 24576 B fit).
  static_assert(N == 8 || N == 16, "run_item serves 8- or 16-token tiles");
  static_assert(
      8 * 128 * (16 - 2) <= kXchBytes && 16 * 128 * (16 - 4) <= kXchBytes,
      "exchange buffer too small for scheduled run_item splits");
  const int rl = rec[7] & 15;
  const int U = 8 >> rl;
  const int nkg = 2 << rl;
  const int kps = 4 << rl;
  const int spx = kKx >> (2 + rl);

  const uint32_t b_wf = bar;
  const uint32_t b_we = bar + 8 * kWStages;
  const uint32_t b_xf = bar + 8 * (2 * kWStages);
  const uint32_t b_xe = bar + 8 * (2 * kWStages + kXStages);

  const int uo = cwi & (U - 1);
  const int kg = cwi >> (3 - rl);

  float acc0[N / 2];
  float acc1[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) {
    acc0[i] = 0.f;
    acc1[i] = 0.f;
  }

  const uint32_t wbase =
      sbase + kOffW + static_cast<uint32_t>(cwi) * kUnit + lane * 16;
  const uint32_t xoff = static_cast<uint32_t>(kg * (N * 128));
  const uint32_t xstep = static_cast<uint32_t>(kps * (N * 128));
  const uint32_t xhalf = static_cast<uint32_t>(nkg * (N * 128));

#pragma unroll 1
  for (int xsl = 0; xsl < kKbn / kKx; ++xsl) {
    uint32_t xb = sbase + kOffX + rg.xs * kXSlot + xoff;
#pragma unroll 1
    for (int j = 0; j < spx; ++j) {
      mbar_wait_t<InlineSpin>(b_wf + 8 * rg.ws, rg.wph);
      if (j == 0) {
        mbar_wait_t<InlineSpin>(b_xf + 8 * rg.xs, rg.xph);
      }
      // cwi * kUnit addresses unit u = cwi % U of stage chunk cwi / U. A
      // packed 64-row tile (kWgChunk = 2304 B) is 2048 B of FP4 words ordered
      // [wq 4][group 8][t 4][kstep 4] (16 B per lane), then 256 B of E5M3
      // scale pairs ordered [wq 4][group 8][kstep 4][gate, up]. Unit u is
      // tile u / 2 with rows wq = 2 * (u & 1) + {0 (acc0), 1 (acc1)}. For odd
      // u (U is even, so u & 1 == cwi & 1), u * kUnit lands 128 B past that
      // half's FP4 words (tile + 1024 B) and its scales (tile + 2176 B) sit
      // 896 B below wb + 2048 - 16 * lane.
      const uint32_t wb =
          wbase + rg.ws * kWBytes - static_cast<uint32_t>(cwi & 1) * 128u;
      const uint4 qa0 = lds128(wb);
      const uint4 qa1 = lds128(wb + 512);
      const uint4 qb0 = lds128(wb + 16 * kUnit);
      const uint4 qb1 = lds128(wb + 16 * kUnit + 512);
      const uint32_t sc = wb + 2048 - 16 * lane -
                          static_cast<uint32_t>(cwi & 1) * 896u +
                          8 * (lane >> 2);
      const uint2 sa0 = lds64(sc);
      const uint2 sa1 = lds64(sc + 64);
      const uint2 sb0 = lds64(sc + 16 * kUnit);
      const uint2 sb1 = lds64(sc + 16 * kUnit + 64);

      mma_k64<N, InlineSpin>(acc0, acc1, qa0, qa1, sa0, sa1, xb, lane, p.smul);
      mma_k64<N, InlineSpin>(acc0, acc1, qb0, qb1, sb0, sb1, xb + xhalf, lane,
                             p.smul);

      if (lane == 0) {
        mbar_arrive(b_we + 8 * rg.ws);
      }
      ring_next<kWStages>(rg.ws, rg.wph);
      xb += xstep;
    }
    if (lane == 0) {
      mbar_arrive(b_xe + 8 * rg.xs);
    }
    ring_next<kXStages>(rg.xs, rg.xph);
  }

  // Deterministic FP32 K-order reduction across K-groups (k = 1 .. nkg - 1).
  const uint32_t xc = sbase + kOffXch;
  if (kg > 0) {
    const uint32_t a =
        xc + static_cast<uint32_t>(((kg - 1) * U + uo) * N * 32 + lane) * 4;
#pragma unroll
    for (int f = 0; f < N / 2; ++f) {
      st_shared_f32(a + f * 128, acc0[f]);
      st_shared_f32(a + (N / 2 + f) * 128, acc1[f]);
    }
  }
  named_bar(2, kNumConsumerWarps * 32);
  if (kg == 0) {
#pragma unroll 1
    for (int k = 1; k < nkg; ++k) {
      const uint32_t a =
          xc + static_cast<uint32_t>(((k - 1) * U + uo) * N * 32 + lane) * 4;
#pragma unroll
      for (int f = 0; f < N / 2; ++f) {
        acc0[f] += ld_shared_f32(a + f * 128);
        acc1[f] += ld_shared_f32(a + (N / 2 + f) * 128);
      }
    }
    const int ug = ((rec[7] >> 4) & 15) + uo;
    epilogue_dual<N, OffGsm>(p, sbase, rec[0], rec[1], rec[2], rec[3], rec[4],
                             ug >> 2, ug & 3, lane, acc0, acc1);
  }
}

template <int N, int SubGroup, int OffGsm>
__device__ __forceinline__ void run_wgw(const Params& p, const int* rec,
                                        Ring& rg, uint32_t sbase, uint32_t bar,
                                        int cwi, int lane) {
  constexpr int kKx = kx_of(N);
  const uint32_t b_wf = bar;
  const uint32_t b_we = bar + 8 * kWStages;
  const uint32_t b_xf = bar + 8 * (2 * kWStages);
  const uint32_t b_xe = bar + 8 * (2 * kWStages + kXStages);

  const int mb = cwi >> 2;
  const int wl = cwi & 3;
  const int g = lane >> 2;

  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) {
    acc[i] = 0.f;
  }

  // Materialize per-lane weight/scale base offsets before the K loop to keep
  // consumer warps within the 104-register budget.
  uint32_t wofs;
  uint32_t sofs;
  asm volatile(
      "mov.u32 %0, %1;"
      : "=r"(wofs)
      : "r"(sbase + kOffW +
            static_cast<uint32_t>(mb * kWgChunk + wl * 512 + lane * 16)));
  asm volatile(
      "mov.u32 %0, %1;"
      : "=r"(sofs)
      : "r"(sbase + kOffW +
            static_cast<uint32_t>(mb * kWgChunk + 2048 + wl * 64 + g * 8)));

  const uint64_t dx0 = desc_swizzle<1024, 1>(sbase + kOffX);

#pragma unroll 1
  for (int st = 0; st < kKbn / 4; ++st) {
    mbar_wait_spin(b_wf + 8 * rg.ws, rg.wph);
    const uint32_t wb = static_cast<uint32_t>(rg.ws * kWBytes);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kx = (4 * st + j) & (kKx - 1);
      if (kx == 0) {
        mbar_wait_spin(b_xf + 8 * rg.xs, rg.xph);
        fence_proxy_async();
      }
      const uint32_t wo = wb + static_cast<uint32_t>(j * kChunk);
      const uint4 q = lds128(wofs + wo);
      const uint32_t w[4] = {q.x, q.y, q.z, q.w};
      const uint32_t m = decode_scale_quad(lds64(sofs + wo), lane, p.smul);
      const uint64_t d =
          dx0 + static_cast<uint32_t>(rg.xs * (kXSlot / 16) + kx * (N * 8));
#pragma unroll
      for (int s0 = 0; s0 < 4; s0 += SubGroup) {
        uint4 a_dq[SubGroup];
#pragma unroll
        for (int u = 0; u < SubGroup; ++u) {
          a_dq[u] = dequant_fp4x8(w[s0 + u], scale_for_step(m, s0 + u));
        }
        wgmma_fence();
#pragma unroll
        for (int u = 0; u < SubGroup; ++u) {
          const uint32_t a_regs[4] = {a_dq[u].x, a_dq[u].y, a_dq[u].z,
                                      a_dq[u].w};
          wgmma_bf16_rs<N>(acc, a_regs, d + 2 * (s0 + u));
        }
        wgmma_commit();
        wgmma_wait<0>();
      }
      if (kx == kKx - 1) {
        if (lane == 0) {
          mbar_arrive(b_xe + 8 * rg.xs);
        }
        ring_next<kXStages>(rg.xs, rg.xph);
      }
    }
    if (lane == 0) {
      mbar_arrive(b_we + 8 * rg.ws);
    }
    ring_next<kWStages>(rg.ws, rg.wph);
  }

  epilogue_single<N, OffGsm>(p, sbase, rec[0], rec[1], rec[2], rec[3], rec[4],
                             mb, wl, lane, acc);
}

// ---------------------------------------------------------------------------
// Activation gather (cp.async) and weight producer (TMA bulk copy)
// ---------------------------------------------------------------------------

template <int NP, int KX, int NR, bool ZeroFill, int K = 0>
__device__ __forceinline__ void gcopy(uint32_t d, const bf16* s,
                                      const uint32_t (&o)[NR],
                                      const bool (&a)[NR]) {
  if constexpr (K < KX * NR) {
    constexpr int kbl = K / NR;
    constexpr int i = K % NR;
    if constexpr (ZeroFill) {
      asm volatile("cp.async.cg.shared.global [%0+%2], [%1+%3], 16, %4;"
                   :
                   : "r"(d), "l"(s + o[i]), "n"((kbl * NP + 8 * i) * 128),
                     "n"(kbl * 128), "r"(a[i] ? 16u : 0u)
                   : "memory");
    } else if (a[i]) {
      asm volatile("cp.async.cg.shared.global [%0+%2], [%1+%3], 16;"
                   :
                   : "r"(d), "l"(s + o[i]), "n"((kbl * NP + 8 * i) * 128),
                     "n"(kbl * 128)
                   : "memory");
    }
    gcopy<NP, KX, NR, ZeroFill, K + 1>(d, s, o, a);
  }
}

template <int NP, bool Lean = false>
__device__ __forceinline__ void gather_t(const Params& p, const uint16_t* toks,
                                         uint32_t sbase, uint32_t b_xf,
                                         uint32_t b_xe, int ph, int slot0,
                                         int n, int j0, int c, int& xs,
                                         uint32_t& xph) {
  constexpr int kKx = kx_of(NP);
  constexpr int kNr = NP / 8;
  static_assert(kKx * NP * 128 <= kXSlot,
                "Activation tile exceeds SMEM slot capacity");

  const bf16* rb = (ph == 1 ? p.xm : p.act) + c * 8;
  uint32_t ro[kNr];
  bool act[kNr];
#pragma unroll
  for (int i = 0; i < kNr; ++i) {
    const int j = j0 + 8 * i;
    act[i] = j < n;
    const int jj = act[i] ? j : 0;
    ro[i] = static_cast<uint32_t>(ph == 1 ? static_cast<int>(toks[slot0 + jj])
                                          : (slot0 + jj)) *
            static_cast<uint32_t>(kHiddenDim);
  }

  const uint32_t sw = static_cast<uint32_t>(j0 * 128 + ((c ^ j0) << 4));
#pragma unroll 1
  for (int kb0 = 0; kb0 < kKbn; kb0 += kKx) {
    if constexpr (Lean) {
      mbar_wait_t<false>(b_xe + 8 * xs, xph ^ 1);
    } else {
      mbar_wait_hint(b_xe + 8 * xs, xph ^ 1);
    }
    gcopy<NP, kKx, kNr, !Lean>(sbase + kOffX + xs * kXSlot + sw, rb, ro, act);
    rb += kKx * 64;
    cp_async_mbar_arrive(b_xf + 8 * xs);
    ring_next<kXStages>(xs, xph);
  }
}

template <bool HotBuckets, bool Lean, int PollSleep = 64>
__device__ __forceinline__ void run_gather_warps(
    const Params& p, const uint16_t* toks, const int* items, uint32_t sbase,
    uint32_t b_xf, uint32_t b_xe, uint32_t b_if, uint32_t b_ie,
    const uint32_t* ctl, int tid, int lane) {
  const int xt = tid - 32;
  const int c = xt & 7;
  const int j0 = xt >> 3;
  int xs = 0;
  uint32_t xph = 0;

  for (int it = 0;; ++it) {
    const int sl = it & (kItemRing - 1);
    mbar_wait_spin(b_if + 8 * sl, (it / kItemRing) & 1);
    const int* rec = items + sl * kItemWords;
    const int ph = rec[0];
    if (!ph) {
      break;
    }
    const int e = rec[1];
    const int slot0 = rec[3];
    const int n = rec[4];
    const int need = rec[6];
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(b_ie + 8 * sl);
    }
    if (ph == 2) {
      if (xt == 0) {
        const uint32_t* cw13 = ctl + 2 + e;
        while (ld_acquire(cw13) < static_cast<uint32_t>(need)) {
          __nanosleep(PollSleep);
        }
      }
      named_bar(1, kNumGatherThreads);
    }
    if constexpr (HotBuckets) {
      // In moe_main / moe_main_ext, c <= kHotMin(8) maps to bucket 0 and
      // c > 8 maps to hw_bucket(n) in 3..13.
      switch (rec[5]) {
        case 0:
          gather_t<8, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c, xs,
                            xph);
          break;
        case 3:
          gather_t<16, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                             xs, xph);
          break;
        case 4:
          gather_t<24, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                             xs, xph);
          break;
        case 5:
          gather_t<32, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                             xs, xph);
          break;
        case 6:
          gather_t<40, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                             xs, xph);
          break;
        case 7:
          gather_t<48, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                             xs, xph);
          break;
        case 8:
          gather_t<56, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                             xs, xph);
          break;
        case 9:
          gather_t<64, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                             xs, xph);
          break;
        case 10:
          gather_t<80, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                             xs, xph);
          break;
        case 11:
          gather_t<96, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                             xs, xph);
          break;
        case 12:
          gather_t<112, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                              xs, xph);
          break;
        default:
          gather_t<128, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                              xs, xph);
          break;
      }
    } else {
      switch (rec[5]) {
        case 0:
          gather_t<8, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c, xs,
                            xph);
          break;
        default:
          gather_t<16, Lean>(p, toks, sbase, b_xf, b_xe, ph, slot0, n, j0, c,
                             xs, xph);
          break;
      }
    }
  }
}

__device__ __noinline__ uint32_t produce_item(const uint8_t* src, uint64_t polw,
                                              int rl, int uoff, uint32_t sbase,
                                              uint32_t bar, uint32_t ring,
                                              int st) {
  const uint32_t b_wf = bar;
  const uint32_t b_we = bar + 8 * kWStages;
  int ws = static_cast<int>(ring & 0xffu);
  uint32_t wph = ring >> 8;
  const int U = 8 >> rl;
  const int kps = 4 << rl;
  const int nst = kKbn / kps;

  for (;;) {
    if (rl == 0 && ws == 0) {
#pragma unroll 1
      for (; st + kWStages <= nst; st += kWStages) {
#pragma unroll
        for (int s = 0; s < kWStages; ++s) {
          mbar_wait_spin(b_we + 8 * s, wph ^ 1);
          if (elect_one()) {
            mbar_expect_tx(b_wf + 8 * s, kWBytes);
            bulk_g2s_hint(sbase + kOffW + s * kWBytes,
                          src + static_cast<size_t>(st + s) * kWBytes, kWBytes,
                          b_wf + 8 * s, polw);
          }
          __syncwarp();
        }
        wph ^= 1;
      }
    }
    if (st >= nst) {
      break;
    }
    mbar_wait_spin(b_we + 8 * ws, wph ^ 1);
    if (elect_one()) {
      mbar_expect_tx(b_wf + 8 * ws, kWBytes);
      if (rl == 0) {
        bulk_g2s_hint(sbase + kOffW + ws * kWBytes,
                      src + static_cast<size_t>(st) * kWBytes, kWBytes,
                      b_wf + 8 * ws, polw);
      } else {
        for (int j = 0; j < kps; ++j) {
          bulk_g2s_hint(
              sbase + kOffW + ws * kWBytes + j * U * kUnit,
              src + static_cast<size_t>(st * kps + j) * kChunk + uoff * kUnit,
              U * kUnit, b_wf + 8 * ws, polw);
        }
      }
    }
    __syncwarp();
    ring_next<kWStages>(ws, wph);
    ++st;
  }
  return static_cast<uint32_t>(ws) | (wph << 8);
}

// ---------------------------------------------------------------------------
// Work-item decoding, slot scatter, persistent engine, and inline combine
// ---------------------------------------------------------------------------

template <bool UnsignedDiv>
__device__ __forceinline__ DecodedItem decode_work_item(
    int cur, int tot1, int tot2, int c, int offe, int tc, int nt, int er,
    int ntr, int r1, int i1, int s1, int s2, int r2, int i2) {
  DecodedItem d{};
  if (cur < tot1) {
    const int r = __popc(__ballot_sync(kFullWarpMask, s1 <= cur));
    d.e = __shfl_sync(kFullWarpMask, er, r);
    int lo = cur - __shfl_sync(kFullWarpMask, s1 - i1, r);
    const int ntl = __shfl_sync(kFullWarpMask, ntr, r);
    d.rl = __shfl_sync(kFullWarpMask, r1, r);
    d.uoff = (lo & ((1 << d.rl) - 1)) * (8 >> d.rl);
    lo >>= d.rl;
    if (ntl == 1) {
      d.mg = lo;
      d.tt = 0;
    } else {
      d.mg = UnsignedDiv ? static_cast<int>(static_cast<uint32_t>(lo) /
                                            static_cast<uint32_t>(ntl))
                         : (lo / ntl);
      d.tt = lo - d.mg * ntl;
    }
    d.ph = 1;
  } else if (cur - tot1 < tot2) {
    const int i2c = cur - tot1;
    const int r = __popc(__ballot_sync(kFullWarpMask, s2 <= i2c));
    d.e = __shfl_sync(kFullWarpMask, er, r);
    d.rl = __shfl_sync(kFullWarpMask, r2, r);
    int lo = i2c - __shfl_sync(kFullWarpMask, s2 - i2, r);
    const int ntl = __shfl_sync(kFullWarpMask, ntr, r);
    d.uoff = (lo & ((1 << d.rl) - 1)) * (8 >> d.rl);
    lo >>= d.rl;
    if (ntl == 1) {
      d.mg = lo;
      d.tt = 0;
    } else {
      d.mg = UnsignedDiv ? static_cast<int>(static_cast<uint32_t>(lo) /
                                            static_cast<uint32_t>(ntl))
                         : (lo / ntl);
      d.tt = lo - d.mg * ntl;
    }
    d.ph = 2;
  }
  d.ce = __shfl_sync(kFullWarpMask, c, d.e);
  d.oe = __shfl_sync(kFullWarpMask, offe, d.e);
  d.tce = __shfl_sync(kFullWarpMask, tc, d.e);
  d.nte = __shfl_sync(kFullWarpMask, nt, d.e);
  return d;
}

template <bool StoreUnits>
__device__ __forceinline__ void write_item_rec(int* rec, const DecodedItem& d,
                                               bool use_hot_bucket) {
  if (d.ph) {
    const int tok0 = d.tt * d.tce;
    const int n = min(d.ce - tok0, d.tce);
    rec[0] = d.ph;
    rec[1] = d.e;
    rec[2] = d.mg;
    rec[3] = d.oe + tok0;
    rec[4] = n;
    rec[5] = use_hot_bucket ? hw_bucket(n) : ((n - 1) >> 3);
    rec[6] = 256 * d.nte;
    rec[7] = d.rl | (d.uoff << 4) | (StoreUnits ? ((8 >> d.rl) << 8) : 0);
  } else {
    rec[0] = 0;
    rec[7] = 0;
  }
}

template <bool Contiguous>
__device__ __forceinline__ const uint8_t* weight_source(const Params& p,
                                                        const uint8_t* psl,
                                                        int ph, int e, int mg) {
  const size_t se =
      Contiguous ? static_cast<size_t>(e) : static_cast<size_t>(psl[e]);
  return ph == 1 ? p.w13 + (se * 32 + mg) * kKbn * kChunk
                 : p.w2 + (se * 16 + mg) * kKbn * kChunk;
}

template <bool Unroll, bool WriteSlotOf>
__device__ __forceinline__ void scatter_slots(const Params& p,
                                              const uint8_t* loc,
                                              const int* cnt, int* cntw,
                                              uint16_t* toks, int nf, int warp,
                                              int lane) {
  const int w = warp - 1;
  const int cw =
      ((nf + (kNumWarps - 1) * 32 - 1) / ((kNumWarps - 1) * 32)) * 32;
  const int f0 = w * cw;
  const int f1 = min(nf, f0 + cw);

  for (int fb = f0; fb < f1; fb += 32) {
    const int f = fb + lane;
    const int e = f < f1 ? static_cast<int>(loc[f]) : 255;
    const uint32_t peers = __match_any_sync(kFullWarpMask, e);
    if (e < kNumLocalExperts && lane == __ffs(peers) - 1) {
      cntw[w * 32 + e] += __popc(peers);
    }
  }
  named_bar(3, (kNumWarps - 1) * 32);

  if (warp == 1) {
    const int c = cnt[lane];
    const int x = warp_inclusive_scan<Unroll>(c, lane);
    int run = x - c;
    for (int i = 0; i < kNumWarps - 1; ++i) {
      const int v = cntw[i * 32 + lane];
      cntw[i * 32 + lane] = run;
      run += v;
    }
  }
  named_bar(3, (kNumWarps - 1) * 32);

  for (int fb = f0; fb < f1; fb += 32) {
    const int f = fb + lane;
    const int e = f < f1 ? static_cast<int>(loc[f]) : 255;
    const uint32_t peers = __match_any_sync(kFullWarpMask, e);
    const int rank = __popc(peers & ((1u << lane) - 1u));
    int slot = -1;
    if (e < kNumLocalExperts) {
      slot = cntw[w * 32 + e] + rank;
      toks[slot] = static_cast<uint16_t>(f >> 3);
    }
    if constexpr (WriteSlotOf) {
      if (blockIdx.x == 0 && f < f1) {
        p.slot_of[f] = slot;
      }
    }
    __syncwarp();
    if (e < kNumLocalExperts && lane == __ffs(peers) - 1) {
      cntw[w * 32 + e] += __popc(peers);
    }
    __syncwarp();
  }
  named_bar(3, (kNumWarps - 1) * 32);
}

template <typename CntwT>
__device__ __forceinline__ void init_barriers_and_counts(
    uint32_t b_wf, uint32_t b_we, uint32_t b_xf, uint32_t b_xe, uint32_t b_if,
    uint32_t b_ie, int* cnt, CntwT* cntw, int tid) {
  if (tid == 0) {
    for (int i = 0; i < kWStages; ++i) {
      mbar_init(b_wf + 8 * i, 1);
      mbar_init(b_we + 8 * i, kNumConsumerWarps);
    }
    for (int i = 0; i < kXStages; ++i) {
      mbar_init(b_xf + 8 * i, kNumGatherThreads);
      mbar_init(b_xe + 8 * i, kNumConsumerWarps);
    }
    for (int i = 0; i < kItemRing; ++i) {
      mbar_init(b_if + 8 * i, 1);
      mbar_init(b_ie + 8 * i, 3 + kNumConsumerWarps);
    }
    mbar_init_fence();
  }
  if (tid < kNumLocalExperts) {
    cnt[tid] = 0;
  }
  for (int i = tid; i < kNumWarps * 32; i += kThreads) {
    cntw[i] = CntwT{0};
  }
}

template <bool Contiguous, typename Layout>
__device__ __forceinline__ void schedule_and_prefetch_item0(
    const Params& p, uint8_t* sm, const int* cnt, int* order,
    const uint8_t* psl, float* gsm, int* items, int* pst, uint32_t sbase,
    uint32_t b_wf, uint32_t b_if, float g13_gate, float g13_up, float g2,
    int lane) {
  const int c = cnt[lane];
  const int x = warp_inclusive_scan<true>(c, lane);
  const int offe = x - c;
  const int rank = warp_descending_rank(c, lane);
  order[rank] = lane;

  const bool hot = c > kHotMin;
  const int nt = hot ? (c + kNww - 1) / kNww : (c + kNmax - 1) / kNmax;
  const int tc = nt <= 1 ? c : (c + nt - 1) / nt;
  {
    int* cm = reinterpret_cast<int*>(sm + Layout::kOffCmb);
    cm[lane] = 128 * nt;
    cm[32 + lane] = 0;
  }
  __syncwarp();

  int er = order[lane];
  __syncwarp();
  // order[32] is dead now; publish split global scales into kOffSch + 128
  // before arriving on b_if so consumers always observe initialized scales.
  gsm[lane] = g13_gate;
  gsm[32 + lane] = g13_up;
  gsm[64 + lane] = g2;
  __syncwarp();

  const int tcr2 = __shfl_sync(kFullWarpMask, tc, er);
  const bool hotr = __shfl_sync(kFullWarpMask, c, er) > kHotMin;
  int r2 = 0;
  {
    const int ea = __popc(__ballot_sync(kFullWarpMask, c > 0));
    if (tcr2 > 0 && !hotr) {
      r2 = (lane >= ea - 2 && tcr2 <= 8) ? 2 : (lane >= ea - 4 ? 1 : 0);
    }
    const int u = ea - __popc(__ballot_sync(kFullWarpMask, r2 > 0));
    er = __shfl_sync(
        kFullWarpMask, er,
        lane < u ? ((lane & 1) ? (u - 1 - (lane >> 1)) : (lane >> 1)) : lane);
  }
  const int ntr = __shfl_sync(kFullWarpMask, nt, er);
  int i1 = 32 * ntr;
  int r1 = 0;
  {
    const int f = warp_inclusive_scan<true>(i1, lane);
    const int G = static_cast<int>(gridDim.x);
    const int tot = __shfl_sync(kFullWarpMask, f, 31);
    const int tail = tot % G;
    const int ce1 = __shfl_sync(kFullWarpMask, c, er);
    r1 = (tot > G && 4 * tail <= G && f > tot - tail && ce1 <= kHotMin) ? 1 : 0;
    i1 <<= r1;
  }
  const int i2 = (16 * ntr) << r2;
  const int sp = warp_inclusive_scan<true>((i2 << 16) | i1, lane);
  const int s1 = sp & 0xFFFF;
  const int s2 = sp >> 16;

  int* q = pst + lane;
  q[0] = c;
  q[32] = offe;
  q[64] = nt;
  q[96] = tc;
  q[128] = er;
  q[160] = ntr;
  q[192] = r1;
  q[224] = i1;
  q[256] = s1;
  q[288] = s2;
  q[320] = r2;
  q[352] = i2;

  // Publish item 0 and prefetch up to kWStages weight stages while warps
  // 1..19 finish scattering slots.
  const int tot1 = __shfl_sync(kFullWarpMask, s1, 31);
  const int tot2 = __shfl_sync(kFullWarpMask, s2, 31);
  const int cur = static_cast<int>(blockIdx.x);
  const DecodedItem d = decode_work_item<false>(
      cur, tot1, tot2, c, offe, tc, nt, er, ntr, r1, i1, s1, s2, r2, i2);
  if (lane == 0) {
    write_item_rec<true>(items, d, d.ce > kHotMin);
    mbar_arrive(b_if);
  }
  int pre = 0;
  if (d.ph) {
    const int U = 8 >> d.rl;
    const int kps = 4 << d.rl;
    const int nst = kKbn / kps;
    pre = min(kWStages, nst);
    if (lane == 0) {
      const uint8_t* src = weight_source<Contiguous>(p, psl, d.ph, d.e, d.mg);
      const uint64_t polw =
          d.nte > 1 ? policy_evict_normal() : policy_evict_first();
      for (int st = 0; st < pre; ++st) {
        mbar_expect_tx(b_wf + 8 * st, kWBytes);
        if (d.rl == 0) {
          bulk_g2s_hint(sbase + kOffW + st * kWBytes,
                        src + static_cast<size_t>(st) * kWBytes, kWBytes,
                        b_wf + 8 * st, polw);
        } else {
          for (int j = 0; j < kps; ++j) {
            bulk_g2s_hint(sbase + kOffW + st * kWBytes + j * U * kUnit,
                          src + static_cast<size_t>(st * kps + j) * kChunk +
                              d.uoff * kUnit,
                          U * kUnit, b_wf + 8 * st, polw);
          }
        }
      }
    }
  }
  pst[416 + lane] = pre;
  __syncwarp();
}

template <int OffGsm>
__device__ __forceinline__ void run_consumer_loop(const Params& p,
                                                  const int* items,
                                                  uint32_t sbase, uint32_t bar,
                                                  uint32_t b_if, uint32_t b_ie,
                                                  int cwi, int lane) {
  Ring rg{};
  for (int it = 0;; ++it) {
    const int sl = it & (kItemRing - 1);
    mbar_wait_spin(b_if + 8 * sl, (it / kItemRing) & 1);
    const int* rec = items + sl * kItemWords;
    if (!rec[0]) {
      break;
    }
    // Keep bar uniform (derived from sbase + Layout::kOffBar, never from sbi).
    uint32_t sbi;
    asm volatile("mov.u32 %0, %1;" : "=r"(sbi) : "r"(sbase));
    switch (rec[5]) {
      case 0:
        run_item<8, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      case 3:
        run_wgw<16, 4, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      case 4:
        run_wgw<24, 4, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      case 5:
        run_wgw<32, 4, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      case 6:
        run_wgw<40, 2, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      case 7:
        run_wgw<48, 2, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      case 8:
        run_wgw<56, 2, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      case 9:
        run_wgw<64, 4, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      case 10:
        run_wgw<80, 2, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      case 11:
        run_wgw<96, 2, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      case 12:
        run_wgw<112, 1, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      default:
        run_wgw<128, 1, OffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
    }
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(b_ie + 8 * sl);
    }
  }
}

// Inline post-expert RMSNorm + FP32 weighted combine across consumer warps:
// 3 named_bar(6) per non-empty token + 1-barrier __ballot_sync fast-skip for
// tokens with 0 local routes (0 ULP bitwise exact).
template <bool Contiguous, int OffCmb>
__device__ __forceinline__ void combine_inline(const Params& p, uint8_t* sm,
                                               const uint8_t* pxm, int tid,
                                               int lane) {
  int* cm = reinterpret_cast<int*>(sm + OffCmb);
  int* need2 = cm;
  int* done = cm + 32;
  float* red = reinterpret_cast<float*>(cm + 68);
  float* twv = reinterpret_cast<float*>(cm + 196);
  int* slv = cm + 204;
  uint32_t* has_loc = reinterpret_cast<uint32_t*>(cm + 212);

  const int T = p.T;
  const int ctid = tid - 128;
  const int cw = ctid >> 5;
  const int c0 = ctid * 8;

  float pw[8];
  {
    const float4 a = *reinterpret_cast<const float4*>(p.postw + c0);
    const float4 b = *reinterpret_cast<const float4*>(p.postw + c0 + 4);
    pw[0] = a.x;
    pw[1] = a.y;
    pw[2] = a.z;
    pw[3] = a.w;
    pw[4] = b.x;
    pw[5] = b.y;
    pw[6] = b.z;
    pw[7] = b.w;
  }

  if (ctid == 0) {
    while (ld_acquire(active_ctrl<OffCmb>(p, sm) + kCtrlSlot) == 0u) {
      __nanosleep(64);
    }
    cm[64] =
        static_cast<int>(atomicAdd(active_ctrl<OffCmb>(p, sm) + kCtrlTok, 1u));
  }
  named_bar(6, kNumConsumerWarps * 32);

  int par = 0;
  int t = cm[64];
  while (t < T) {
    const int nxt_par = par ^ 1;
    if (cw == 0) {
      int slot = -1;
      if (lane < kTopK) {
        slot = ld_global_cg_s32(p.slot_of + t * kTopK + lane);
      }
      const uint32_t loc_mask = __ballot_sync(kFullWarpMask, slot >= 0);
      if (loc_mask != 0u) {
        if (lane < kTopK) {
          slv[lane] = slot;
          if (slot >= 0) {
            const float w = ld_global_cg_f32(p.tw + t * kTopK + lane);
            const int e = map_expert<Contiguous>(
                p, pxm, ld_global_cg_s32(p.ids + t * kTopK + lane));
            if (e < kNumLocalExperts && !done[e]) {
              const uint32_t* cw2 = active_ctrl<OffCmb>(p, sm) + kCtrlW2 + e;
              while (ld_acquire(cw2) < static_cast<uint32_t>(need2[e])) {
                __nanosleep(128);
              }
              done[e] = 1;
            }
            twv[lane] = w;
          } else {
            twv[lane] = 0.f;
          }
        }
      } else if (lane == 0) {
        cm[64 + (nxt_par << 1)] = static_cast<int>(
            atomicAdd(active_ctrl<OffCmb>(p, sm) + kCtrlTok, 1u));
      }
      if (lane == 0) {
        has_loc[par] = loc_mask;
      }
    }
    named_bar(6, kNumConsumerWarps * 32);

    if (has_loc[par] == 0u) {
      *reinterpret_cast<uint4*>(p.out + static_cast<size_t>(t) * kHiddenDim +
                                c0) = make_uint4(0, 0, 0, 0);
      t = cm[64 + (nxt_par << 1)];
      par = nxt_par;
      continue;
    }

    uint32_t nxt = 0u;
    if (ctid == 0) {
      nxt = atomicAdd(active_ctrl<OffCmb>(p, sm) + kCtrlTok, 1u);
    }

    uint4 u[8];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const int sk = slv[k];
      u[k] =
          sk >= 0
              ? ld_global_cg_u4(p.y + static_cast<size_t>(sk) * kHiddenDim + c0)
              : make_uint4(0, 0, 0, 0);
    }

    float ss[8];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const uint32_t uu[4] = {u[k].x, u[k].y, u[k].z, u[k].w};
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float v0 = bf16_lo_to_f32(uu[i]) * pw[2 * i];
        const float v1 = bf16_hi_to_f32(uu[i]) * pw[2 * i + 1];
        a += v0 * v0 + v1 * v1;
      }
      ss[k] = a;
    }

#pragma unroll
    for (int dd = 16; dd; dd >>= 1) {
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        ss[k] += __shfl_xor_sync(kFullWarpMask, ss[k], dd);
      }
    }

    if (lane < kTopK) {
      float val = ss[0];
#pragma unroll
      for (int k = 1; k < kTopK; ++k) {
        val = (lane == k) ? ss[k] : val;
      }
      if (ctid == 0) {
        cm[64 + (nxt_par << 1)] = static_cast<int>(nxt);
      }
      red[cw * kTopK + lane] = val;
    }
    named_bar(6, kNumConsumerWarps * 32);

    if (ctid < kTopK) {
      float tot = 0.f;
#pragma unroll
      for (int i = 0; i < kNumConsumerWarps; ++i) {
        tot += red[i * kTopK + ctid];
      }
      red[ctid] = slv[ctid] >= 0
                      ? rsqrtf(tot * (1.f / 4096.f) + p.post_eps) * twv[ctid]
                      : 0.f;
    }
    named_bar(6, kNumConsumerWarps * 32);

    float acc[8] = {};
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const float r = red[k];
      const uint32_t uu[4] = {u[k].x, u[k].y, u[k].z, u[k].w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[2 * i] += bf16_lo_to_f32(uu[i]) * pw[2 * i] * r;
        acc[2 * i + 1] += bf16_hi_to_f32(uu[i]) * pw[2 * i + 1] * r;
      }
    }
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      o[i] = pack_bf16x2(acc[2 * i], acc[2 * i + 1]);
    }
    *reinterpret_cast<uint4*>(p.out + static_cast<size_t>(t) * kHiddenDim +
                              c0) = make_uint4(o[0], o[1], o[2], o[3]);
    t = cm[64 + (nxt_par << 1)];
    par = nxt_par;
  }
}

template <bool Contiguous, typename Layout>
__device__ __forceinline__ void run_persistent_engine(
    const Params& p, uint8_t* sm, const uint8_t* pxm, const uint8_t* psl,
    const uint16_t* toks, int* items, const int* pst, uint32_t sbase,
    uint32_t bar, uint32_t b_xf, uint32_t b_xe, uint32_t b_if, uint32_t b_ie,
    int tid, int warp, int lane) {
  if (warp < 4) {
    setmaxnreg_dec<kRegProducer>();
    if (warp == 0) {
      const int* q = pst + lane;
      const int c = q[0];
      const int offe = q[32];
      const int nt = q[64];
      const int tc = q[96];
      const int er = q[128];
      const int ntr = q[160];
      const int r1 = q[192];
      const int i1 = q[224];
      const int s1 = q[256];
      const int s2 = q[288];
      const int r2 = q[320];
      const int i2 = q[352];
      const int tot1 = __shfl_sync(kFullWarpMask, s1, 31);
      const int tot2 = __shfl_sync(kFullWarpMask, s2, 31);
      const uint64_t pol = policy_evict_first();
      const uint64_t pol_normal = policy_evict_normal();
      const int pre0 = q[416];
      int ws = pre0 == kWStages ? 0 : pre0;
      uint32_t wph = pre0 == kWStages ? 1u : 0u;

      for (int it = 0;; ++it) {
        int cur = static_cast<int>(blockIdx.x);
        if (it) {
          if (lane == 0) {
            cur = static_cast<int>(gridDim.x) +
                  static_cast<int>(
                      atomicAdd(active_ctrl<Layout::kOffCmb>(p, sm), 1u));
          }
          cur = __shfl_sync(kFullWarpMask, cur, 0);
        }
        const int sl = it & (kItemRing - 1);
        const DecodedItem d = decode_work_item<false>(
            cur, tot1, tot2, c, offe, tc, nt, er, ntr, r1, i1, s1, s2, r2, i2);
        if (lane == 0) {
          int* rec = items + sl * kItemWords;
          if (it) {
            mbar_wait_hint(b_ie + 8 * sl, ((it / kItemRing) & 1) ^ 1);
            write_item_rec<true>(rec, d, d.ce > kHotMin);
            mbar_arrive(b_if + 8 * sl);
          }
        }
        __syncwarp();
        if (d.ph) {
          const uint8_t* src =
              weight_source<Contiguous>(p, psl, d.ph, d.e, d.mg);
          const uint64_t polw = d.nte > 1 ? pol_normal : pol;
          const uint32_t st2 = produce_item(
              src, polw, d.rl, d.uoff, sbase, bar,
              static_cast<uint32_t>(ws) | (wph << 8), it == 0 ? pre0 : 0);
          ws = static_cast<int>(st2 & 0xffu);
          wph = st2 >> 8;
        }
        __syncwarp();
        if (!d.ph) {
          break;
        }
      }
    } else if (warp == kCompletionWarp) {
      if (blockIdx.x == 0 && lane == 0) {
        __threadfence();
        atomicAdd(active_ctrl<Layout::kOffCmb>(p, sm) + kCtrlSlot, 1u);
      }
      for (int it = 0;; ++it) {
        const int sl = it & (kItemRing - 1);
        mbar_wait_spin(b_if + 8 * sl, (it / kItemRing) & 1);
        const int* rec = items + sl * kItemWords;
        const int ph = rec[0];
        const int e = rec[1];
        const int units = (rec[7] >> 8) & 255;
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(b_ie + 8 * sl);
        }
        if (!ph) {
          break;
        }
        while (!mbar_try_hint(b_ie + 8 * sl, (it / kItemRing) & 1)) {
        }
        if (lane == 0) {
          __threadfence();
          atomicAdd(
              active_ctrl<Layout::kOffCmb>(p, sm) + (ph == 1 ? 2 : kCtrlW2) + e,
              static_cast<uint32_t>(units));
        }
      }
      named_bar(5, kNumConsumerWarps * 32 + 32);
    } else {
      run_gather_warps<true, false, 64>(
          p, toks, items, sbase, b_xf, b_xe, b_if, b_ie,
          active_ctrl<Layout::kOffCmb>(p, sm), tid, lane);
    }
    return;
  }

  setmaxnreg_inc<kRegConsumer>();
  run_consumer_loop<Layout::kOffGsm>(p, items, sbase, bar, b_if, b_ie, warp - 4,
                                     lane);
  combine_inline<Contiguous, Layout::kOffCmb>(p, sm, pxm, tid, lane);
  named_bar(5, kNumConsumerWarps * 32 + 32);
}

// ---------------------------------------------------------------------------
// Persistent MoE kernel for 17 <= T <= 1023 (W13 + SwiGLU + W2 + inline
// combine)
// ---------------------------------------------------------------------------

template <bool Contiguous>
__global__ void __launch_bounds__(kThreads, 1) moe_main(const Params p) {
  using Layout = SmemLayout<Contiguous>;

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  const uint32_t sbase = smem_u32(sm);

  int* cnt = reinterpret_cast<int*>(sm + Layout::kOffSch);
  int* order = cnt + 32;
  uint16_t* toks = reinterpret_cast<uint16_t*>(sm + kOffTok);
  uint8_t* loc = sm + kOffXch;
  uint8_t* pxm =
      Contiguous ? nullptr : (sm + Layout::kOffSch + Layout::kOffPxm);
  uint8_t* psl =
      Contiguous ? nullptr : (sm + Layout::kOffSch + Layout::kOffPsl);
  float* gsm = reinterpret_cast<float*>(sm + Layout::kOffGsm);
  int* cntw = reinterpret_cast<int*>(sm + Layout::kOffCntw);
  int* items = reinterpret_cast<int*>(sm + Layout::kOffItem);

  const uint32_t bar = sbase + Layout::kOffBar;
  const uint32_t b_wf = bar;
  const uint32_t b_we = bar + 8 * kWStages;
  const uint32_t b_xf = bar + 8 * (2 * kWStages);
  const uint32_t b_xe = bar + 8 * (2 * kWStages + kXStages);
  const uint32_t b_if = bar + 8 * (2 * kWStages + 2 * kXStages);
  const uint32_t b_ie = b_if + 8 * kItemRing;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = __shfl_sync(kFullWarpMask, tid >> 5, 0);

  // No predecessor pdl_wait() (launched normally on stream) and no redundant
  // 64 KB kOffX zeroing (gather_t<NP, Lean=false> hardware zero-fills padding).
  pdl_launch();
  init_barriers_and_counts(b_wf, b_we, b_xf, b_xe, b_if, b_ie, cnt, cntw, tid);

  const int T = p.T;
  const int nf = T * kTopK;
  float g13_gate = 0.f;
  float g13_up = 0.f;
  float g2 = 0.f;

  // Read current epoch before updating it so every CTA agrees on the control
  // half; CTA 0 clears the opposite half for the next launch.
  const uint32_t epo =
      tid == 0 ? static_cast<uint32_t>(ld_global_cg_s32(
                     reinterpret_cast<const int*>(p.ctrl) + kCtrlEpoch))
               : 0u;

  constexpr int kPer = (kMaxRowsBase + kThreads - 1) / kThreads;
  int v[kPer];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int f = tid + i * kThreads;
    v[i] = f < nf ? ld_global_cg_s32(p.ids + f) : 0;
  }
  if constexpr (!Contiguous) {
    if (tid < 256) {
      pxm[tid] = p.plan[tid];
    }
    if (tid < kNumLocalExperts) {
      const int s = p.plan[256 + tid];
      psl[tid] = static_cast<uint8_t>(s);
      g13_gate = p.gs13_gate[s];
      g13_up = p.gs13_up[s];
      g2 = p.gs2[s];
    }
  } else {
    if (tid < kNumLocalExperts) {
      g13_gate = p.gs13_gate[tid];
      g13_up = p.gs13_up[tid];
      g2 = p.gs2[tid];
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int f = tid + i * kThreads;
    if (f < nf) {
      const int e = map_expert<Contiguous>(p, pxm, v[i]);
      loc[f] = static_cast<uint8_t>(e < kNumLocalExperts ? e : 255);
      if (e < kNumLocalExperts) {
        atomicAdd(&cnt[e], 1);
      }
    }
  }
  if (tid == 0) {
    red_add_u32(p.ctrl + kCtrlEpoch,
                blockIdx.x == 0 ? (1u << 20) - (gridDim.x - 1u) : 1u);
    reinterpret_cast<int*>(sm + Layout::kOffCmb)[65] =
        ((epo >> 20) & 1u) ? kCtrlSet : 0;
  }
  __syncthreads();

  if (blockIdx.x == 0 && warp == kNumWarps - 1 && lane < kCtrlSet / 4) {
    reinterpret_cast<uint4*>(p.ctrl + kCtrlBase +
                             (kCtrlSet - reinterpret_cast<const int*>(
                                             sm + Layout::kOffCmb)[65]))[lane] =
        make_uint4(0, 0, 0, 0);
  }

  int* pst = reinterpret_cast<int*>(sm + Layout::kOffPst);
  if (warp == 0) {
    schedule_and_prefetch_item0<Contiguous, Layout>(
        p, sm, cnt, order, psl, gsm, items, pst, sbase, b_wf, b_if, g13_gate,
        g13_up, g2, lane);
  } else {
    scatter_slots<true, true>(p, loc, cnt, cntw, toks, nf, warp, lane);
  }

  run_persistent_engine<Contiguous, Layout>(p, sm, pxm, psl, toks, items, pst,
                                            sbase, bar, b_xf, b_xe, b_if, b_ie,
                                            tid, warp, lane);
}

// ---------------------------------------------------------------------------
// Zero-overflow extended persistent MoE kernel for 1024 <= T <= 1536
// Single-pass warp compaction into kOffX + uint16_t cntw + toks[12288].
// ---------------------------------------------------------------------------

template <bool Contiguous>
__global__ void __launch_bounds__(kThreads, 1) moe_main_ext(const Params p) {
  using Layout = SmemLayoutExt<Contiguous>;

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  const uint32_t sbase = smem_u32(sm);

  int* cnt = reinterpret_cast<int*>(sm + Layout::kOffSch);
  int* order = cnt + 32;
  int* nloc_smem = reinterpret_cast<int*>(sm + Layout::kOffNloc);
  uint16_t* toks = reinterpret_cast<uint16_t*>(sm + kOffTok);
  uint32_t* comp = reinterpret_cast<uint32_t*>(sm + kOffX);
  uint8_t* pxm =
      Contiguous ? nullptr : (sm + Layout::kOffSch + Layout::kOffPxm);
  uint8_t* psl =
      Contiguous ? nullptr : (sm + Layout::kOffSch + Layout::kOffPsl);
  float* gsm = reinterpret_cast<float*>(sm + Layout::kOffGsm);
  uint16_t* cntw = reinterpret_cast<uint16_t*>(sm + Layout::kOffCntw);
  int* items = reinterpret_cast<int*>(sm + Layout::kOffItem);

  const uint32_t bar = sbase + Layout::kOffBar;
  const uint32_t b_wf = bar;
  const uint32_t b_we = bar + 8 * kWStages;
  const uint32_t b_xf = bar + 8 * (2 * kWStages);
  const uint32_t b_xe = bar + 8 * (2 * kWStages + kXStages);
  const uint32_t b_if = bar + 8 * (2 * kWStages + 2 * kXStages);
  const uint32_t b_ie = b_if + 8 * kItemRing;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = __shfl_sync(kFullWarpMask, tid >> 5, 0);

  pdl_launch();
  init_barriers_and_counts(b_wf, b_we, b_xf, b_xe, b_if, b_ie, cnt, cntw, tid);
  if (tid == 0) {
    *nloc_smem = 0;
  }

  float g13_gate = 0.f;
  float g13_up = 0.f;
  float g2 = 0.f;
  if constexpr (!Contiguous) {
    if (tid < 256) {
      pxm[tid] = p.plan[tid];
    }
    if (tid < kNumLocalExperts) {
      const int s = p.plan[256 + tid];
      psl[tid] = static_cast<uint8_t>(s);
      g13_gate = p.gs13_gate[s];
      g13_up = p.gs13_up[s];
      g2 = p.gs2[s];
    }
  } else {
    if (tid < kNumLocalExperts) {
      g13_gate = p.gs13_gate[tid];
      g13_up = p.gs13_up[tid];
      g2 = p.gs2[tid];
    }
  }
  __syncthreads();

  const int T = p.T;
  const int nf = T * kTopK;  // <= 12288 = kMaxRowsExt
  const uint32_t epo =
      tid == 0 ? static_cast<uint32_t>(ld_global_cg_s32(
                     reinterpret_cast<const int*>(p.ctrl) + kCtrlEpoch))
               : 0u;

  // Deterministic power-of-2 warp partition: 20 * (1 << cw_shift) >= nf.
  // For nf <= 10240 -> cw_step = 512; for 10241 <= nf <= 12288 -> cw_step =
  // 1024. Every per-warp count <= 1024 < 2^11 (fits in 11-bit lrank and
  // uint16_t cntw), and every f < 12288 < 2^16.
  const int cw_shift = (nf <= 10240) ? 9 : 10;
  const int cw_step = 1 << cw_shift;
  const int f0 = warp * cw_step;
  const int f1 = min(nf, f0 + cw_step);

  // Pass 1: single coalesced scan over ids; discard off-rank routes in
  // registers and compact local (f, lrank, e) packets into kOffX.
  // Because n_local <= nf <= 12288 == kMaxRowsExt, pos < 12288 unconditionally.
  for (int fb = f0; fb < f1; fb += 32) {
    const int f = fb + lane;
    const bool valid = f < f1;
    const int id = valid ? ld_global_cg_s32(p.ids + f) : -1;
    const int e = valid ? map_expert<Contiguous>(p, pxm, id) : 255;
    const bool is_loc = e < kNumLocalExperts;
    if (blockIdx.x == 0 && valid && !is_loc) {
      p.slot_of[f] = -1;
    }
    const uint32_t lmask = __ballot_sync(kFullWarpMask, is_loc);
    if (lmask) {
      const uint32_t peers = __match_any_sync(kFullWarpMask, is_loc ? e : 255);
      const int rank_step = __popc(peers & ((1u << lane) - 1u));
      const int lrank =
          is_loc ? (static_cast<int>(cntw[warp * 32 + e]) + rank_step) : 0;
      const int leader = __ffs(lmask) - 1;
      int base_pos = 0;
      if (lane == leader) {
        base_pos = atomicAdd(nloc_smem, __popc(lmask));
      }
      base_pos = __shfl_sync(kFullWarpMask, base_pos, leader);
      if (is_loc) {
        const int pos = base_pos + __popc(lmask & ((1u << lane) - 1u));
        comp[pos] = static_cast<uint32_t>(f) |
                    (static_cast<uint32_t>(lrank) << 16) |
                    (static_cast<uint32_t>(e) << 27);
      }
      __syncwarp();
      if (is_loc && lane == __ffs(peers) - 1) {
        cntw[warp * 32 + e] = static_cast<uint16_t>(
            static_cast<int>(cntw[warp * 32 + e]) + __popc(peers));
      }
      __syncwarp();
    }
  }
  __syncthreads();

  const int n_local = *nloc_smem;

  if (tid == 0) {
    red_add_u32(p.ctrl + kCtrlEpoch,
                blockIdx.x == 0 ? (1u << 20) - (gridDim.x - 1u) : 1u);
    reinterpret_cast<int*>(sm + Layout::kOffCmb)[65] =
        ((epo >> 20) & 1u) ? kCtrlSet : 0;
  }
  if (warp == 0) {
    int sum_e = 0;
#pragma unroll
    for (int wi = 0; wi < kNumWarps; ++wi) {
      sum_e += static_cast<int>(cntw[wi * 32 + lane]);
    }
    cnt[lane] = sum_e;
    const int x = warp_inclusive_scan<true>(sum_e, lane);
    int run = x - sum_e;
#pragma unroll
    for (int wi = 0; wi < kNumWarps; ++wi) {
      const int v = static_cast<int>(cntw[wi * 32 + lane]);
      cntw[wi * 32 + lane] = static_cast<uint16_t>(run);
      run += v;
    }
  }
  __syncthreads();

  if (blockIdx.x == 0 && warp == kNumWarps - 1 && lane < kCtrlSet / 4) {
    reinterpret_cast<uint4*>(p.ctrl + kCtrlBase +
                             (kCtrlSet - reinterpret_cast<const int*>(
                                             sm + Layout::kOffCmb)[65]))[lane] =
        make_uint4(0, 0, 0, 0);
  }

  int* pst = reinterpret_cast<int*>(sm + Layout::kOffPst);
  if (warp == 0) {
    schedule_and_prefetch_item0<Contiguous, Layout>(
        p, sm, cnt, order, psl, gsm, items, pst, sbase, b_wf, b_if, g13_gate,
        g13_up, g2, lane);
  } else {
    // Pass 2: scatter compacted local routes into toks[0..n_local) and
    // slot_of. named_bar(3) guarantees all reads of comp[] (in kOffX) finish
    // before gather warps 1..2 overwrite kOffX with hardware zero-fill.
    for (int idx = tid - 32; idx < n_local; idx += (kNumWarps - 1) * 32) {
      const uint32_t pkt = comp[idx];
      const int f = static_cast<int>(pkt & 0xFFFFu);
      const int lrank = static_cast<int>((pkt >> 16) & 0x7FFu);
      const int e = static_cast<int>(pkt >> 27);
      const int w_orig = f >> cw_shift;
      const int slot = static_cast<int>(cntw[w_orig * 32 + e]) + lrank;
      toks[slot] = static_cast<uint16_t>(f >> 3);
      if (blockIdx.x == 0) {
        p.slot_of[f] = slot;
      }
    }
    named_bar(3, (kNumWarps - 1) * 32);
  }

  run_persistent_engine<Contiguous, Layout>(p, sm, pxm, psl, toks, items, pst,
                                            sbase, bar, b_xf, b_xe, b_if, b_ie,
                                            tid, warp, lane);
}

// ---------------------------------------------------------------------------
// Small-batch MoE kernel for 1 <= T <= 16 (paired with moe_combine via PDL)
// Includes E in {1, 2, 3} kTailSplit and low-latency W13->W2 release/poll.
// ---------------------------------------------------------------------------

template <bool Contiguous>
__global__ void __launch_bounds__(kThreads, 1)
    moe_main_small_batch(const Params p) {
  using Layout = SmemLayout<Contiguous>;

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  const uint32_t sbase = smem_u32(sm);

  int* cnt = reinterpret_cast<int*>(sm + Layout::kOffSch);
  int* order = cnt + 32;
  uint16_t* toks = reinterpret_cast<uint16_t*>(sm + kOffTok);
  uint8_t* loc = sm + kOffXch;
  uint8_t* pxm =
      Contiguous ? nullptr : (sm + Layout::kOffSch + Layout::kOffPxm);
  uint8_t* psl =
      Contiguous ? nullptr : (sm + Layout::kOffSch + Layout::kOffPsl);
  float* gsm = reinterpret_cast<float*>(sm + Layout::kOffGsm);
  int* cntw = reinterpret_cast<int*>(sm + Layout::kOffCntw);
  int* items = reinterpret_cast<int*>(sm + Layout::kOffItem);

  const uint32_t bar = sbase + Layout::kOffBar;
  const uint32_t b_wf = bar;
  const uint32_t b_we = bar + 8 * kWStages;
  const uint32_t b_xf = bar + 8 * (2 * kWStages);
  const uint32_t b_xe = bar + 8 * (2 * kWStages + kXStages);
  const uint32_t b_if = bar + 8 * (2 * kWStages + 2 * kXStages);
  const uint32_t b_ie = b_if + 8 * kItemRing;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = __shfl_sync(kFullWarpMask, tid >> 5, 0);

  // No predecessor pdl_wait(); retain pdl_launch() to overlap moe_combine.
  init_barriers_and_counts(b_wf, b_we, b_xf, b_xe, b_if, b_ie, cnt, cntw, tid);
  pdl_launch();

  const int T = p.T;
  const int nf = T * kTopK;  // <= 128 < kThreads (1 element per thread)
  float g13_gate = 0.f;
  float g13_up = 0.f;
  float g2 = 0.f;
  {
    const int v0 = tid < nf ? ld_global_cg_s32(p.ids + tid) : 0;
    if constexpr (!Contiguous) {
      if (tid < 256) {
        pxm[tid] = p.plan[tid];
      }
      if (tid < kNumLocalExperts) {
        const int s = p.plan[256 + tid];
        psl[tid] = static_cast<uint8_t>(s);
        g13_gate = p.gs13_gate[s];
        g13_up = p.gs13_up[s];
        g2 = p.gs2[s];
      }
    } else {
      if (tid < kNumLocalExperts) {
        g13_gate = p.gs13_gate[tid];
        g13_up = p.gs13_up[tid];
        g2 = p.gs2[tid];
      }
    }
    __syncthreads();
    if (tid < nf) {
      const int e = map_expert<Contiguous>(p, pxm, v0);
      loc[tid] = static_cast<uint8_t>(e < kNumLocalExperts ? e : 255);
      if (e < kNumLocalExperts) {
        atomicAdd(&cnt[e], 1);
      }
    }
  }
  __syncthreads();

  if (warp < 4) {
    setmaxnreg_dec<kRegProducer>();
  }

  if (warp == 0) {
    const int c = cnt[lane];
    const int x = warp_inclusive_scan<false>(c, lane);
    const int offe = x - c;
    const int rank = warp_descending_rank(c, lane);
    order[rank] = lane;

    // Tile cap 16 matches run_item<8>/run_item<16> dispatch: exact even on
    // duplicate-ID adversarial cases with c > 16.
    const int nt = (c + kNmaxSmall - 1) >> 4;
    const int tc = nt <= 1
                       ? c
                       : static_cast<int>(static_cast<uint32_t>(c + nt - 1) /
                                          static_cast<uint32_t>(nt));
    __syncwarp();

    const int er = order[lane];
    __syncwarp();
    // order[32] is dead now; publish split global scales into kOffSch + 128
    // before arriving on b_if.
    gsm[lane] = g13_gate;
    gsm[32 + lane] = g13_up;
    gsm[64 + lane] = g2;
    __syncwarp();

    const int ntr = __shfl_sync(kFullWarpMask, nt, er);
    const int tcr2 = __shfl_sync(kFullWarpMask, tc, er);
    int i1 = 32 * ntr;
    int r1 = 0;
    int r2 = 0;
    int i2 = 0;
    int s1 = 0;
    int s2 = 0;
    {
      if (tcr2 > 0) {
        r2 = 1;
      }
      const int active_e = __popc(__ballot_sync(kFullWarpMask, c > 0));
      if (active_e >= 1 && active_e <= 3 && tcr2 > 0 && ntr == 1) {
        // Pure-decode E in {1, 2, 3} split: clamp rl <= 2 at N=8 (kKx=16)
        // and rl <= 1 at N=16 (kKx=8) so spx >= 1 and kOffXch <= 24576 B.
        const int max_rl = tcr2 <= 8 ? 2 : 1;
        if (active_e == 1) {
          r1 = max_rl;
          r2 = max_rl;
        } else if (active_e == 2) {
          r1 = 1;
          r2 = max_rl;
        } else {
          // E == 3: split lane 0 (first in W2 queue) so tot1 = 128 fits in
          // one wave and E0 W2 runs without head-of-line blocking while E1/E2
          // finish W13; E1+E2 W2 then run in a single 128-CTA wave.
          r1 = (lane == 0) ? 1 : 0;
          r2 = max_rl;
        }
      }
      const int f = warp_inclusive_scan<false>(i1, lane);
      const bool late = ntr > 0 && f > static_cast<int>(gridDim.x);
      const int am =
          (active_e >= 4 && active_e <= 11)
              ? static_cast<int>(
                    (p.atab >> (4 * ((p.T <= 8 ? 0 : 8) + active_e - 4))) &
                    15ull)
              : 0;
      if ((am & 1) && late) {
        r1 = 1;
      }
      const int pw = (am >> 1) & 3;
      if (tcr2 > 0 && pw) {
        const int fin = tcr2 <= 8 ? 2 : 1;
        if (pw == 1) {
          if (!late) {
            r2 = 0;
          }
        } else if (pw == 2) {
          if (late) {
            r2 = fin;
          }
        } else {
          r2 = late ? fin : 0;
        }
      }
      i1 <<= r1;
      i2 = (16 * ntr) << r2;
      const int sp = warp_inclusive_scan<false>((i2 << 16) | i1, lane);
      s1 = sp & 0xFFFF;
      s2 = sp >> 16;
    }

    const int tot1 = __shfl_sync(kFullWarpMask, s1, 31);
    const int tot2 = __shfl_sync(kFullWarpMask, s2, 31);
    const uint64_t pol = policy_evict_first();
    const uint64_t pol_normal = policy_evict_normal();
    int ws = 0;
    uint32_t wph = 0;

    for (int it = 0;; ++it) {
      int cur = static_cast<int>(blockIdx.x);
      if (it) {
        if (lane == 0) {
          cur = static_cast<int>(gridDim.x) +
                static_cast<int>(atomicAdd(p.ctrl, 1u));
        }
        cur = __shfl_sync(kFullWarpMask, cur, 0);
      }
      const int sl = it & (kItemRing - 1);
      const DecodedItem d = decode_work_item<true>(
          cur, tot1, tot2, c, offe, tc, nt, er, ntr, r1, i1, s1, s2, r2, i2);
      if (lane == 0) {
        mbar_wait_spin(b_ie + 8 * sl, ((it / kItemRing) & 1) ^ 1);
        write_item_rec<false>(items + sl * kItemWords, d, false);
        mbar_arrive(b_if + 8 * sl);
      }
      __syncwarp();
      if (d.ph) {
        const uint8_t* src = weight_source<Contiguous>(p, psl, d.ph, d.e, d.mg);
        const uint64_t polw = d.nte > 1 ? pol_normal : pol;
        const uint32_t st2 =
            produce_item(src, polw, d.rl, d.uoff, sbase, bar,
                         static_cast<uint32_t>(ws) | (wph << 8), 0);
        ws = static_cast<int>(st2 & 0xffu);
        wph = st2 >> 8;
      }
      __syncwarp();
      if (!d.ph) {
        break;
      }
    }
    return;
  }

  // Skip dead slot_of global writes in T <= 16 path (moe_combine computes
  // slots directly from ids during PDL overlap).
  scatter_slots<false, false>(p, loc, cnt, cntw, toks, nf, warp, lane);

  if (warp == kCompletionWarp) {
    for (int it = 0;; ++it) {
      const int sl = it & (kItemRing - 1);
      mbar_wait_spin(b_if + 8 * sl, (it / kItemRing) & 1);
      const int* rec = items + sl * kItemWords;
      const int ph = rec[0];
      const int e = rec[1];
      const int rl = rec[7] & 15;
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(b_ie + 8 * sl);
      }
      if (!ph) {
        break;
      }
      if (ph == 1) {
        while (!mbar_try_hint(b_ie + 8 * sl, (it / kItemRing) & 1)) {
        }
        if (lane == 0) {
          red_release_add_u32(p.ctrl + 2 + e, static_cast<uint32_t>(8 >> rl));
        }
      }
    }
    named_bar(5, kNumConsumerWarps * 32 + 32);
    return;
  }

  if (warp < 4) {
    run_gather_warps<false, true, 32>(p, toks, items, sbase, b_xf, b_xe, b_if,
                                      b_ie, p.ctrl, tid, lane);
    return;
  }

  setmaxnreg_inc<kRegConsumer>();
  const int cwi = warp - 4;
  const uint32_t sbi = sbase;
  Ring rg{};
  for (int it = 0;; ++it) {
    const int sl = it & (kItemRing - 1);
    mbar_wait_spin(b_if + 8 * sl, (it / kItemRing) & 1);
    const int* rec = items + sl * kItemWords;
    if (!rec[0]) {
      break;
    }
    switch (rec[5]) {
      case 0:
        run_item<8, Layout::kOffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
      default:
        run_item<16, Layout::kOffGsm>(p, rec, rg, sbi, bar, cwi, lane);
        break;
    }
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(b_ie + 8 * sl);
    }
  }
  named_bar(5, kNumConsumerWarps * 32 + 32);
}

// ---------------------------------------------------------------------------
// Post-expert RMSNorm + FP32 weighted combine kernel for 1 <= T <= 16
//   Contiguous == true: contiguous local_start arithmetic (no plan load, no
//                       pxm[256] SMEM, no prologue __syncthreads()).
// ---------------------------------------------------------------------------

template <bool Contiguous>
__global__ void __launch_bounds__(256)
    moe_combine(const bf16* y, const int* ids, int T, const float* tw,
                const float* __restrict__ postw, float post_eps, bf16* out,
                uint32_t* ctrl, const uint8_t* plan, int local_start) {
  __shared__ float red[8][8];
  __shared__ uint8_t pxm[Contiguous ? 1 : 256];

  pdl_launch();

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int c0 = tid * 16;
  const int wp = tid >> 5;
  const int ln = tid & 31;

  float pw[16];
  {
    const float4* pp = reinterpret_cast<const float4*>(postw + c0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 v = pp[i];
      pw[4 * i] = v.x;
      pw[4 * i + 1] = v.y;
      pw[4 * i + 2] = v.z;
      pw[4 * i + 3] = v.w;
    }
  }

  int sl[8];
  float w[8];
  {
    const int nf = T * kTopK;
    if constexpr (!Contiguous) {
      pxm[tid] = plan[tid];
      __syncthreads();
    }
    int key[8];
    bool any_local = false;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const int gid = ld_global_cg_s32(ids + t * kTopK + k);
      const int e = Contiguous ? map_expert_contiguous(local_start, gid)
                               : map_expert_plan(pxm, gid);
      if (e < kNumLocalExperts) {
        w[k] = ld_global_cg_f32(tw + t * kTopK + k);
        key[k] = (e << 12) | (t * kTopK + k);
        any_local = true;
      } else {
        w[k] = 0.f;
        key[k] = -1;
      }
    }
    // Uniform CTA fast-skip when token t has 0 local routes (0 ULP exact).
    if (!any_local) {
      pdl_wait();
      if (ctrl != nullptr && blockIdx.x == 0 && tid < 34) {
        ctrl[tid] = 0u;
      }
      uint4* dst = reinterpret_cast<uint4*>(
          out + static_cast<size_t>(t) * kHiddenDim + c0);
      dst[0] = make_uint4(0, 0, 0, 0);
      dst[1] = make_uint4(0, 0, 0, 0);
      return;
    }

    uint32_t c[8] = {};
#pragma unroll 1
    for (int f = tid; f < nf; f += 256) {
      const int gid = ld_global_cg_s32(ids + f);
      const int e = Contiguous ? map_expert_contiguous(local_start, gid)
                               : map_expert_plan(pxm, gid);
      const int kf = e < kNumLocalExperts ? ((e << 12) | f) : 0x7fffffff;
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        c[k] += (kf < key[k]) ? 1u : 0u;
      }
    }
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      c[k] = __reduce_add_sync(kFullWarpMask, c[k]);
    }
    if (ln == 0) {
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        red[wp][k] = __uint_as_float(c[k]);
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      uint32_t n = 0u;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        n += __float_as_uint(red[i][k]);
      }
      sl[k] = key[k] >= 0 ? static_cast<int>(n) : -1;
    }
    __syncthreads();
  }

  // Keep postw global loads scheduled ahead of pdl_wait().
  {
    uint32_t pin = 0xffffffffu;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      pin &= __float_as_uint(pw[i]);
    }
    if (pin == 0xffffffffu) {
      __trap();
    }
  }

  int nloc = 0;
  const bf16* sp[8];
  float sw[8];
#pragma unroll
  for (int j = 0; j < kTopK; ++j) {
    sp[j] = y + c0;
    sw[j] = 0.f;
  }
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    if (sl[k] >= 0) {
      const bf16* pk = y + static_cast<size_t>(sl[k]) * kHiddenDim + c0;
#pragma unroll
      for (int j = 0; j < kTopK; ++j) {
        if (j == nloc) {
          sp[j] = pk;
          sw[j] = w[k];
        }
      }
      ++nloc;
    }
  }

  pdl_wait();

  uint4 u[8][2];
#pragma unroll
  for (int j = 0; j < kTopK; ++j) {
    if (j < nloc) {
      u[j][0] = ld_global_cg_u4(sp[j]);
      u[j][1] = ld_global_cg_u4(sp[j] + 8);
    } else {
      u[j][0] = make_uint4(0, 0, 0, 0);
      u[j][1] = make_uint4(0, 0, 0, 0);
    }
  }

  if (ctrl != nullptr && blockIdx.x == 0 && tid < 34) {
    ctrl[tid] = 0u;
  }

  float v[8][16];
  float ss[8] = {};
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    if (2 * g < nloc) {
      const uint32_t ua[8] = {u[2 * g][0].x, u[2 * g][0].y, u[2 * g][0].z,
                              u[2 * g][0].w, u[2 * g][1].x, u[2 * g][1].y,
                              u[2 * g][1].z, u[2 * g][1].w};
      const uint32_t ub[8] = {u[2 * g + 1][0].x, u[2 * g + 1][0].y,
                              u[2 * g + 1][0].z, u[2 * g + 1][0].w,
                              u[2 * g + 1][1].x, u[2 * g + 1][1].y,
                              u[2 * g + 1][1].z, u[2 * g + 1][1].w};
      float a0 = 0.f;
      float a1 = 0.f;
      float* va = v[2 * g];
      float* vb = v[2 * g + 1];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        va[2 * i] = bf16_lo_to_f32(ua[i]) * pw[2 * i];
        va[2 * i + 1] = bf16_hi_to_f32(ua[i]) * pw[2 * i + 1];
        vb[2 * i] = bf16_lo_to_f32(ub[i]) * pw[2 * i];
        vb[2 * i + 1] = bf16_hi_to_f32(ub[i]) * pw[2 * i + 1];
        a0 += va[2 * i] * va[2 * i] + va[2 * i + 1] * va[2 * i + 1];
        a1 += vb[2 * i] * vb[2 * i] + vb[2 * i + 1] * vb[2 * i + 1];
      }
#pragma unroll
      for (int d = 16; d; d >>= 1) {
        a0 += __shfl_xor_sync(kFullWarpMask, a0, d);
        a1 += __shfl_xor_sync(kFullWarpMask, a1, d);
      }
      ss[2 * g] = a0;
      ss[2 * g + 1] = a1;
    }
  }

  if (ln < kTopK) {
    float vv = ss[0];
#pragma unroll
    for (int j = 1; j < kTopK; ++j) {
      vv = (ln == j) ? ss[j] : vv;
    }
    red[wp][ln] = vv;
  }
  __syncthreads();

  float r[8] = {};
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    if (2 * g < nloc) {
      float t0 = 0.f;
      float t1 = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        t0 += red[i][2 * g];
        t1 += red[i][2 * g + 1];
      }
      r[2 * g] = rsqrtf(t0 * (1.f / 4096.f) + post_eps) * sw[2 * g];
      r[2 * g + 1] =
          (2 * g + 1 < nloc)
              ? rsqrtf(t1 * (1.f / 4096.f) + post_eps) * sw[2 * g + 1]
              : 0.f;
    }
  }

  float acc[16] = {};
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    if (2 * g < nloc) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        acc[i] += v[2 * g][i] * r[2 * g];
        acc[i] += v[2 * g + 1][i] * r[2 * g + 1];
      }
    }
  }

  uint32_t o[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    o[i] = pack_bf16x2(acc[2 * i], acc[2 * i + 1]);
  }
  uint4* dst =
      reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * kHiddenDim + c0);
  dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
  dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

// ---------------------------------------------------------------------------
// Device dynamic-SMEM configuration and host entrypoint
// ---------------------------------------------------------------------------

std::atomic<uint64_t> g_smem_configured{0};

void configure_device_impl(int dev_idx) {
  const uint64_t dev_bit =
      (dev_idx >= 0 && dev_idx < 64) ? (uint64_t{1} << dev_idx) : 0ull;
  if (dev_bit != 0ull &&
      (g_smem_configured.load(std::memory_order_relaxed) & dev_bit) != 0ull) {
    return;
  }

  auto set_attr = [](const void* fn, int bytes, const char* name) {
    const cudaError_t err = cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute(", name,
                ") failed: ", cudaGetErrorString(err));
  };

  set_attr(reinterpret_cast<const void*>(moe_main<true>),
           SmemLayout<true>::kSmemBytes, "moe_main<true>");
  set_attr(reinterpret_cast<const void*>(moe_main<false>),
           SmemLayout<false>::kSmemBytes, "moe_main<false>");
  set_attr(reinterpret_cast<const void*>(moe_main_small_batch<true>),
           SmemLayout<true>::kSmemBytes, "moe_main_small_batch<true>");
  set_attr(reinterpret_cast<const void*>(moe_main_small_batch<false>),
           SmemLayout<false>::kSmemBytes, "moe_main_small_batch<false>");
  set_attr(reinterpret_cast<const void*>(moe_main_ext<true>),
           SmemLayoutExt<true>::kSmemBytes, "moe_main_ext<true>");
  set_attr(reinterpret_cast<const void*>(moe_main_ext<false>),
           SmemLayoutExt<false>::kSmemBytes, "moe_main_ext<false>");

  if (dev_bit != 0ull) {
    g_smem_configured.fetch_or(dev_bit, std::memory_order_relaxed);
  }
}

void configure_device(int64_t dev_idx) {
  int dev_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&dev_count);
  TORCH_CHECK(err == cudaSuccess,
              "cudaGetDeviceCount failed: ", cudaGetErrorString(err));
  TORCH_CHECK(dev_idx >= 0 && dev_idx < dev_count, "dev_idx must be in [0, ",
              dev_count - 1, "], got ", dev_idx);
  const c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(dev_idx)));
  configure_device_impl(static_cast<int>(dev_idx));
}

void run(torch::Tensor w13, torch::Tensor w2, torch::Tensor gs13_gate,
         torch::Tensor gs13_up, torch::Tensor gs2, torch::Tensor x,
         torch::Tensor topk_ids, torch::Tensor topk_weights,
         torch::Tensor post_weight, double post_eps, torch::Tensor act,
         torch::Tensor y, torch::Tensor slot_of, torch::Tensor ctrl,
         torch::Tensor out, int64_t grid, c10::optional<torch::Tensor> plan,
         int64_t local_start = -1) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(),
              "x must be a contiguous CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "x must have dtype bfloat16");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kHiddenDim,
              "x must have shape [T, 4096]");

  const int64_t T = x.size(0);
  if (T == 0) {
    return;
  }
  TORCH_CHECK(T >= 1 && T <= kMaxTokens,
              "Hopper NVFP4 MoE decode supports 1 <= T <= ", kMaxTokens,
              ", got T=", T);
  TORCH_CHECK(grid > 0 && grid < (1 << 20),
              "grid must be in [1, 2^20 - 1], got ", grid);
  TORCH_CHECK(local_start == -1 ||
                  (local_start >= 0 && local_start + kNumLocalExperts <= 256),
              "local_start must be -1 or in [0, ", 256 - kNumLocalExperts,
              "], got ", local_start);

  const bool contiguous = local_start >= 0;
  const bool has_plan = plan.has_value() && plan->defined();
  TORCH_CHECK(
      contiguous || has_plan,
      "Either local_start >= 0 or a valid plan tensor must be provided");

  const auto dev = x.device();
  auto check_tensor = [&](const torch::Tensor& t, torch::ScalarType dtype,
                          const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == dev, name,
                " must be on the same CUDA device as x");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has unexpected dtype");
  };

  check_tensor(w13, torch::kUInt8, "w13");
  check_tensor(w2, torch::kUInt8, "w2");
  check_tensor(gs13_gate, torch::kFloat32, "gs13_gate");
  check_tensor(gs13_up, torch::kFloat32, "gs13_up");
  check_tensor(gs2, torch::kFloat32, "gs2");
  check_tensor(topk_ids, torch::kInt32, "topk_ids");
  check_tensor(topk_weights, torch::kFloat32, "topk_weights");
  check_tensor(post_weight, torch::kFloat32, "post_weight");
  check_tensor(act, torch::kBFloat16, "act");
  check_tensor(y, torch::kBFloat16, "y");
  check_tensor(slot_of, torch::kInt32, "slot_of");
  check_tensor(ctrl, torch::kInt32, "ctrl");
  check_tensor(out, torch::kBFloat16, "out");

  const uint8_t* plan_ptr = nullptr;
  if (has_plan) {
    check_tensor(*plan, torch::kUInt8, "plan");
    TORCH_CHECK(plan->numel() >= 288,
                "plan must hold at least 288 uint8 elements");
    plan_ptr = plan->data_ptr<uint8_t>();
  }

  TORCH_CHECK(w13.dim() == 4 && w13.size(0) >= kNumLocalExperts &&
                  w13.size(1) == 32 && w13.size(2) == kKbn &&
                  w13.size(3) == kChunk,
              "w13 must have shape [>=32, 32, 64, 9216]");
  TORCH_CHECK(w2.dim() == 4 && w2.size(0) >= kNumLocalExperts &&
                  w2.size(1) == 16 && w2.size(2) == kKbn &&
                  w2.size(3) == kChunk,
              "w2 must have shape [>=32, 16, 64, 9216]");
  TORCH_CHECK(gs13_gate.dim() == 1 && gs13_gate.size(0) >= kNumLocalExperts,
              "gs13_gate must have shape [>=32]");
  TORCH_CHECK(gs13_up.dim() == 1 && gs13_up.size(0) >= kNumLocalExperts,
              "gs13_up must have shape [>=32]");
  TORCH_CHECK(gs2.dim() == 1 && gs2.size(0) >= kNumLocalExperts,
              "gs2 must have shape [>=32]");
  TORCH_CHECK(
      topk_ids.dim() == 2 && topk_ids.size(0) == T && topk_ids.size(1) == kTopK,
      "topk_ids must have shape [T, 8]");
  TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.size(0) == T &&
                  topk_weights.size(1) == kTopK,
              "topk_weights must have shape [T, 8]");
  TORCH_CHECK(post_weight.dim() == 1 && post_weight.size(0) == kHiddenDim,
              "post_weight must have shape [4096]");
  TORCH_CHECK(act.numel() >= T * kTopK * kHiddenDim,
              "act workspace must hold at least T * 8 * 4096 elements");
  TORCH_CHECK(y.numel() >= T * kTopK * kHiddenDim,
              "y workspace must hold at least T * 8 * 4096 elements");
  TORCH_CHECK(slot_of.numel() >= T * kTopK,
              "slot_of must hold at least T * 8 int32 elements");
  TORCH_CHECK(ctrl.numel() >= kCtrlEpoch + 1,
              "ctrl must hold at least 217 int32 elements");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == T && out.size(1) == kHiddenDim,
              "out must have shape [T, 4096]");

  const c10::cuda::CUDAGuard device_guard(dev);
  const int dev_idx = dev.index();
  // Persistent decode kernels require all CTAs co-resident (one 640-thread CTA
  // per SM): CTA b owns static item b and peers spin on its completion counts.
  const int num_sms = get_device_sm_count(dev_idx);
  TORCH_CHECK(grid <= num_sms, "grid must not exceed the SM count (", num_sms,
              "), got ", grid);
  auto check_aligned16 = [](const torch::Tensor& t, const char* name) {
    TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0u, name,
                " must be 16-byte aligned");
  };
  check_aligned16(x, "x");
  check_aligned16(w13, "w13");
  check_aligned16(w2, "w2");
  check_aligned16(act, "act");
  check_aligned16(y, "y");
  check_aligned16(out, "out");
  check_aligned16(post_weight, "post_weight");
  check_aligned16(ctrl, "ctrl");

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev_idx);
  configure_device_impl(dev_idx);

  constexpr uint64_t kAtab = pack_tail_split();
  const Params p{
      w13.data_ptr<uint8_t>(),
      w2.data_ptr<uint8_t>(),
      gs13_gate.data_ptr<float>(),
      gs13_up.data_ptr<float>(),
      gs2.data_ptr<float>(),
      reinterpret_cast<const bf16*>(x.data_ptr()),
      topk_ids.data_ptr<int>(),
      reinterpret_cast<bf16*>(act.data_ptr()),
      reinterpret_cast<bf16*>(y.data_ptr()),
      slot_of.data_ptr<int>(),
      reinterpret_cast<uint32_t*>(ctrl.data_ptr<int>()),
      topk_weights.data_ptr<float>(),
      post_weight.data_ptr<float>(),
      reinterpret_cast<bf16*>(out.data_ptr()),
      static_cast<int>(T),
      static_cast<float>(post_eps),
      kAtab,
      plan_ptr,
      16u,
      contiguous ? static_cast<int>(local_start) : 0,
  };

  const dim3 gdim(static_cast<uint32_t>(grid));
  const dim3 bdim(kThreads);

  if (T <= 16) {
    if (contiguous) {
      moe_main_small_batch<true>
          <<<gdim, bdim, SmemLayout<true>::kSmemBytes, stream>>>(p);
    } else {
      moe_main_small_batch<false>
          <<<gdim, bdim, SmemLayout<false>::kSmemBytes, stream>>>(p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const cudaError_t err = launch_pdl(
        contiguous ? moe_combine<true> : moe_combine<false>,
        dim3(static_cast<uint32_t>(T)), dim3(256), 0, stream, p.y, p.ids, p.T,
        p.tw, p.postw, p.post_eps, p.out, p.ctrl, p.plan, p.local_start);
    TORCH_CHECK(err == cudaSuccess,
                "Hopper NVFP4 MoE combine kernel launch failed: ",
                cudaGetErrorString(err));
    return;
  }

  if (T <= kMaxTokensBase) {
    if (contiguous) {
      moe_main<true><<<gdim, bdim, SmemLayout<true>::kSmemBytes, stream>>>(p);
    } else {
      moe_main<false><<<gdim, bdim, SmemLayout<false>::kSmemBytes, stream>>>(p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  if (contiguous) {
    moe_main_ext<true>
        <<<gdim, bdim, SmemLayoutExt<true>::kSmemBytes, stream>>>(p);
  } else {
    moe_main_ext<false>
        <<<gdim, bdim, SmemLayoutExt<false>::kSmemBytes, stream>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::muse_spark_hopper_nvfp4_moe::decode

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace impl = vllm::muse_spark_hopper_nvfp4_moe::decode;
  m.def("run", &impl::run,
        "Run Hopper SM90a NVFP4 MoE decode/verify engine (1 <= T <= 1536).",
        py::arg("w13"), py::arg("w2"), py::arg("gs13_gate"), py::arg("gs13_up"),
        py::arg("gs2"), py::arg("x"), py::arg("topk_ids"),
        py::arg("topk_weights"), py::arg("post_weight"), py::arg("post_eps"),
        py::arg("act"), py::arg("y"), py::arg("slot_of"), py::arg("ctrl"),
        py::arg("out"), py::arg("grid"), py::arg("plan"),
        py::arg("local_start") = -1);
  m.def("pack_experts", &impl::pack_experts,
        "Pack NVFP4 (E2M1) weights and E5M3 block scales into Hopper MoE "
        "layout in a single CUDA launch (0 B SMEM, 0 B temp VRAM). "
        "Caller must zero err_flag before calling; any non-finite or "
        "non-E5M3-exact scale sets err_flag |= 1.",
        py::arg("raw_w"), py::arg("raw_sf"), py::arg("is_w13"),
        py::arg("packed_out"), py::arg("err_flag"));
  m.def("configure_device", &impl::configure_device,
        "Pre-configure Hopper SM90a dynamic shared memory attributes for all "
        "decode kernels on the given CUDA device index.",
        py::arg("dev_idx"));
}
