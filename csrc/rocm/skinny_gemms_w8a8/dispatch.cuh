#pragma once

#include "kernel.cuh"

// Production launch dispatch: (thrds, ytile, unrl, achunk) -> kernel template.
// WvPrGrp is baked at 16; the heuristic in the parent .cu varies
// (yt, ur, ac). Each (yt, ur, ac) tuple referenced here must be reachable
// from WVSPLIT_W8A8_TILE — extra tuples bloat the binary, missing tuples
// fall through to the TORCH_CHECK.
template <typename scalar_t, int N_VAL>
inline void dispatch_w8a8(dim3 grid, cudaStream_t stream, int K, int M, int Bx,
                          int By, const int8_t* B, const void* A_raw,
                          const scalar_t* w_scale, const float* a_scale,
                          const scalar_t* BIAS, scalar_t* C, int CuCount,
                          int quant_mode, int thrds, int ytile, int unrl,
                          int achunk) {
#define LAUNCH(_THRDS, _YTILE, _UNRL, _ACHUNK)                                 \
  do {                                                                         \
    int __wvPrGrp = mindiv_w8a8(M, CuCount * (_YTILE), 16);                    \
    wvSplitK_w8a8_hf_sml_<scalar_t, _THRDS, _YTILE, 16, _ACHUNK, _UNRL, N_VAL> \
        <<<grid, dim3(_THRDS, 16), 0, stream>>>(                               \
            K, M, Bx, By, B, A_raw, w_scale, a_scale, BIAS, C, __wvPrGrp,      \
            CuCount, quant_mode);                                              \
    return;                                                                    \
  } while (0)

  // Enumerate exactly the (yt, ur, ac) tuples WVSPLIT_W8A8_TILE picks.
  if (thrds == 32) {
    // gfx11 wave32
    if (ytile == 1 && unrl == 4 && achunk == 32) LAUNCH(32, 1, 4, 32);
    if (ytile == 4 && unrl == 1 && achunk == 16) LAUNCH(32, 4, 1, 16);
    if (ytile == 4 && unrl == 4 && achunk == 32) LAUNCH(32, 4, 4, 32);
    if (ytile == 2 && unrl == 4 && achunk == 32) LAUNCH(32, 2, 4, 32);
    if (ytile == 4 && unrl == 2 && achunk == 32) LAUNCH(32, 4, 2, 32);
    if (ytile == 2 && unrl == 2 && achunk == 32) LAUNCH(32, 2, 2, 32);
    if (ytile == 4 && unrl == 4 && achunk == 16) LAUNCH(32, 4, 4, 16);
    if (ytile == 4 && unrl == 4 && achunk == 8) LAUNCH(32, 4, 4, 8);
    if (ytile == 2 && unrl == 4 && achunk == 8) LAUNCH(32, 2, 4, 8);
  } else {  // thrds == 64, gfx9 wave64
    if (ytile == 1 && unrl == 4 && achunk == 32) LAUNCH(64, 1, 4, 32);
    if (ytile == 4 && unrl == 1 && achunk == 16) LAUNCH(64, 4, 1, 16);
    if (ytile == 4 && unrl == 4 && achunk == 8) LAUNCH(64, 4, 4, 8);
    if (ytile == 2 && unrl == 4 && achunk == 8) LAUNCH(64, 2, 4, 8);
  }
  TORCH_CHECK(false, "wvSplitK_w8a8: unhandled (thrds=", thrds,
              ", ytile=", ytile, ", unrl=", unrl, ", achunk=", achunk,
              "). Add to dispatch.cuh.");

#undef LAUNCH
}

#ifdef VLLM_SKINNY_GEMM_SWEEP

// Sweep dispatch: full (yt, ur, ac, wv) cross-product. Built only when the
// sweep flag is set so the production binary never pays for the wider
// instantiation space.
template <typename scalar_t, int N_VAL>
inline void dispatch_w8a8_sweep(dim3 grid, cudaStream_t stream, int K, int M,
                                int Bx, int By, const int8_t* B,
                                const void* A_raw, const scalar_t* w_scale,
                                const float* a_scale, const scalar_t* BIAS,
                                scalar_t* C, int CuCount, int quant_mode,
                                int thrds, int ytile, int wvprgrp, int achunk,
                                int unrl) {
  #define SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL)            \
    do {                                                                    \
      int __wvPrGrp = mindiv_w8a8(M, CuCount * (_YTILE), _WVPRGRP);         \
      wvSplitK_w8a8_hf_sml_<scalar_t, _THRDS, _YTILE, _WVPRGRP, _ACHUNK,    \
                            _UNRL, N_VAL>                                   \
          <<<grid, dim3(_THRDS, _WVPRGRP), 0, stream>>>(                    \
              K, M, Bx, By, B, A_raw, w_scale, a_scale, BIAS, C, __wvPrGrp, \
              CuCount, quant_mode);                                         \
      return;                                                               \
    } while (0)

  #define SWEEP_UNRL(_THRDS, _YTILE, _WVPRGRP, _ACHUNK)                  \
    do {                                                                 \
      if (unrl == 1) SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 1); \
      if (unrl == 2) SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 2); \
      if (unrl == 4) SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 4); \
    } while (0)

  #define SWEEP_YTILE(_THRDS, _WVPRGRP, _ACHUNK)                \
    do {                                                        \
      if (ytile == 1) SWEEP_UNRL(_THRDS, 1, _WVPRGRP, _ACHUNK); \
      if (ytile == 2) SWEEP_UNRL(_THRDS, 2, _WVPRGRP, _ACHUNK); \
      if (ytile == 4) SWEEP_UNRL(_THRDS, 4, _WVPRGRP, _ACHUNK); \
    } while (0)

  #define SWEEP_WV(_THRDS, _ACHUNK)                        \
    do {                                                   \
      if (wvprgrp == 8) SWEEP_YTILE(_THRDS, 8, _ACHUNK);   \
      if (wvprgrp == 12) SWEEP_YTILE(_THRDS, 12, _ACHUNK); \
      if (wvprgrp == 16) SWEEP_YTILE(_THRDS, 16, _ACHUNK); \
    } while (0)

  #define SWEEP_AC(_THRDS)                    \
    do {                                      \
      if (achunk == 8) SWEEP_WV(_THRDS, 8);   \
      if (achunk == 16) SWEEP_WV(_THRDS, 16); \
      if (achunk == 32) SWEEP_WV(_THRDS, 32); \
    } while (0)

  if (thrds == 32) {
    SWEEP_AC(32);
  } else {
    SWEEP_AC(64);
  }
  TORCH_CHECK(false, "wvSplitK_w8a8_sweep: unhandled (thrds=", thrds,
              ", ytile=", ytile, ", wvprgrp=", wvprgrp, ", achunk=", achunk,
              ", unrl=", unrl, ")");

  #undef SWEEP_LAUNCH
  #undef SWEEP_UNRL
  #undef SWEEP_YTILE
  #undef SWEEP_WV
  #undef SWEEP_AC
}

#endif  // VLLM_SKINNY_GEMM_SWEEP
