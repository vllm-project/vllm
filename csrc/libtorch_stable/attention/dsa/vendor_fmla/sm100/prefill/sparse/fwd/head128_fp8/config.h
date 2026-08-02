#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/float8.h>
#include <kerutils/kerutils.cuh>

#include "params.h"
#include "defines.h"
#include "mma_fp8_noelect.h"

namespace sm100::fwd::head128_fp8 {

using namespace cute;
using fp8_e4m3 = cutlass::float_e4m3_t;

// fp8 variant of the masked head128 sparse prefill kernel.
// Differences from the bf16 kernel (../head128):
//  - q/kv/P are e4m3; both GEMMs run tcgen05 kind::f8f6f4 (2x accum steps of
//    32 elems vs bf16's 16).
//  - Q shrinks to 36.9KB and lives ENTIRELY in smem -> the TMEM-Q + UTCCP
//    machinery is deleted (all-SS MMAs). TMEM only holds O and P accums.
//  - Q/K tiles keep the 64-element column geometry but use 64B swizzle
//    (SW64); V uses 128-element boxes with SW128 (D_V/2 = 256 = 2x128).
//    KV therefore needs TWO tensor maps (same data, different box/swizzle).
//  - P is stored to smem as e4m3 scaled by 448 (bake log2(448) into the
//    softmax exponent); the 448 cancels between O and li, lse subtracts
//    ln(448), and params.out_scale carries the V dequant scale (k_scale).

template <typename Shape_Q, typename TMA_Q, typename Shape_O, typename TMA_O>
struct TmaParams {
  Shape_Q shape_Q;
  TMA_Q tma_Q;
  Shape_O shape_O;
  TMA_O tma_O;
  CUtensorMap tensor_map_kv;    // K gathers: box {64,1},  SWIZZLE_64B
  CUtensorMap tensor_map_kv_v;  // V gathers: box {128,1}, SWIZZLE_128B
};

template <int D_QK>
struct KernelTemplate {
  static constexpr int D_Q = D_QK;
  static constexpr int D_K = D_QK;
  static constexpr int D_V = 512;
  static constexpr float MAX_INIT_VAL = -1e30;

  static constexpr int B_H = 128;     // For 2 CTAs
  static constexpr int B_TOPK = 128;  // For 2 CTAs
  static constexpr int NUM_BUFS = 2;
  static constexpr int NUM_THREADS = 256 + 128 + 128;

  // K pipeline split: part0 = 128 dims (one 128B gather box, early QK start),
  // part1 = 448 (3x128B + 1x64B boxes). 128-elem SW128 boxes for the first
  // 512 dims cut K's TMA issues from 9 to 5 per token-quad and double the
  // DRAM request size (the 64B boxes were 10% of stall samples via
  // mio_throttle).
  static constexpr int D_PART0 = 128;
  static constexpr int D_PART1 = D_QK - D_PART0;  // 448 = 384(SW128) + 64(SW64)
  static constexpr int D_SW128 = 512;             // 128-elem box region
  static constexpr int NUM_P0_TILES = D_PART0 / 64, NUM_P1_TILES = D_PART1 / 64;

  // P is stored as e4m3 scaled by 7 (= 448 / 2^6): the online-softmax
  // hysteresis lets logits exceed mi by up to 6, i.e. exp2 values up to 64,
  // so the pre-scale must leave 2^6 headroom below e4m3's 448 max.
  static constexpr float P_SCALE_LOG2 = 2.8073549220576042f;  // log2(7)
  static constexpr float P_SCALE_LN = 1.9459101090932196f;    // ln(7)

  // Tensor memory columns (fp32 accums only; no Q in TMEM)
  struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 320: P buffer 0
    // 320 ~ 384: P buffer 1 (double-buffered: QK(k+1) overlaps softmax(k))
    static constexpr int o = 0;
    static constexpr int p = 256;
    static constexpr int p1 = 320;
  };

  // Q/K: 64-element column tiles, 64B swizzle (same element geometry as the
  // bf16 kernel's SW128 tiles, so all element-offset math carries over).
  template <int NUM_TILES>
  using SmemLayoutQTiles = decltype(coalesce(
      tile_to_shape(UMMA::Layout_K_SW64_Atom<fp8_e4m3>{},
                    Shape<Int<B_H / 2>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));

  template <int NUM_TILES>
  using SmemLayoutOTiles = decltype(coalesce(
      tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{},
                    Shape<Int<B_H / 2>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));

  using SmemLayoutO = SmemLayoutOTiles<8>;

  template <int NUM_TILES>
  using SmemLayoutKTiles = decltype(coalesce(
      tile_to_shape(UMMA::Layout_K_SW64_Atom<fp8_e4m3>{},
                    Shape<Int<B_TOPK / 2>, Int<64 * NUM_TILES>>{},
                    Step<_1, _2>{}),
      Shape<_1, _1>{}));

  // K region layouts: 128-elem SW128 tiles for dims [0, 512), one SW64 tile
  // for the rope tail [512, 576).
  template <int NUM_TILES128>
  using SmemLayoutK128 = decltype(coalesce(
      tile_to_shape(UMMA::Layout_K_SW128_Atom<fp8_e4m3>{},
                    Shape<Int<B_TOPK / 2>, Int<128 * NUM_TILES128>>{},
                    Step<_1, _2>{}),
      Shape<_1, _1>{}));
  using SmemLayoutKTail = decltype(coalesce(
      tile_to_shape(UMMA::Layout_K_SW64_Atom<fp8_e4m3>{},
                    Shape<Int<B_TOPK / 2>, Int<64>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));

  using SmemLayoutV = decltype(coalesce(
      tile_to_shape(UMMA::Layout_MN_SW128_Atom<fp8_e4m3>{},
                    Shape<Int<256>, Int<B_TOPK>>{}, Step<_2, _1>{}),
      Shape<_1, _1>{}));

  // P/S: e4m3 [B_H/2, B_TOPK] per CTA, INTER (unswizzled) K-major atoms.
  // The (8,16)-element fp8 atom is byte-identical to the bf16 (8,8) atom, so
  // the u128 store pattern in the softmax warps mirrors the bf16 kernel's.
  using SmemLayoutS = decltype(coalesce(
      tile_to_shape(UMMA::Layout_K_INTER_Atom<fp8_e4m3>{},
                    Shape<Int<B_H / 2>, Int<B_TOPK>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));

  struct SharedMemoryPlan {
    union {
      struct {
        array_aligned<fp8_e4m3, cosize_v<SmemLayoutQTiles<D_Q / 64>>> q_full;
        array_aligned<fp8_e4m3, cosize_v<SmemLayoutKTiles<D_K / 64>>> k[2];
        array_aligned<fp8_e4m3, cosize_v<SmemLayoutV>> v[2];
      } s;
      array_aligned<bf16, cosize_v<SmemLayoutO>> o;  // epilogue only
    } u;
    array_aligned<fp8_e4m3, cosize_v<SmemLayoutS>> s[2];  // double-buffered
    char is_k_valid[NUM_BUFS][B_TOPK / 8];
    char is_kq_valid[NUM_BUFS][16][B_TOPK / 8];
    transac_bar_t bar_prologue_q;
    transac_bar_t bar_qk_part_done[NUM_BUFS], bar_qk_done[NUM_BUFS];
    transac_bar_t bar_sv_part_done[NUM_BUFS], bar_sv_done[NUM_BUFS];
    transac_bar_t bar_k_part0_ready[NUM_BUFS], bar_k_part1_ready[NUM_BUFS];
    transac_bar_t bar_v_part0_ready[NUM_BUFS], bar_v_part1_ready[NUM_BUFS];
    transac_bar_t bar_p_free[NUM_BUFS];
    transac_bar_t bar_so_ready[NUM_BUFS];
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[2][128],
        rowwise_li_buf[128];  // max-buf parity-double-buffered: one
                              // barrier/block
  };

  using TiledMMA_P = decltype(make_tiled_mma(
      SM100_MMA_F8F6F4_2x1SM_SS_NOELECT<fp8_e4m3, fp8_e4m3, float, B_H, B_TOPK,
                                        UMMA::Major::K, UMMA::Major::K>{}));

  using TiledMMA_O = decltype(make_tiled_mma(
      SM100_MMA_F8F6F4_2x1SM_SS_NOELECT<fp8_e4m3, fp8_e4m3, float, B_H, 256,
                                        UMMA::Major::K, UMMA::Major::MN>{},
      Layout<Shape<_1, _1, _1>>{},
      Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _32>{}
      // CTA0 takes V[:, 0:256], CTA1 takes V[:, 256:512]; K-mode = 32 (8bit)
      ));

  template <typename TmaParams>
  static __device__ void sparse_attn_fwd_kernel_devfunc(
      const SparseAttnFwdParams& params, const TmaParams& tma_params);
};

}  // namespace sm100::fwd::head128_fp8
