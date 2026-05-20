// Sweep wrapper: parallel-compiled TU sharing kernel templates with the
// production .cu via skinny_gemms_int4_kernels.cuh.
#ifdef VLLM_SKINNY_GEMM_SWEEP
  #include "skinny_gemms_int4_kernels.cuh"

torch::Tensor wvSplitK_int4g_hf_sweep(
    const at::Tensor& in_a, const at::Tensor& in_b, const at::Tensor& in_scale,
    const int64_t CuCount, const int64_t group_size, const int64_t ytile,
    const int64_t unrl, const int64_t achunk, const int64_t wvprgrp) {
  auto M_in = in_a.size(0);
  auto K_in = in_b.size(1);
  auto N_in = in_b.size(0);

  int64_t expected_weight_bytes = M_in * K_in / 2;
  int64_t actual_weight_bytes = in_a.numel() * in_a.element_size();
  TORCH_CHECK(actual_weight_bytes == expected_weight_bytes,
              "Weight tensor must contain M*K/2 bytes for int4 packing");
  TORCH_CHECK(in_b.dtype() == torch::kFloat16,
              "Sweep only supports float16 activations");
  TORCH_CHECK(in_scale.dtype() == torch::kFloat16,
              "Sweep only supports float16 scale");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "group_size must be 32, 64, or 128, got ", group_size);
  TORCH_CHECK(K_in % group_size == 0,
              "K must be divisible by group_size=", group_size);
  int64_t num_groups = K_in / group_size;
  TORCH_CHECK(in_scale.size(0) == M_in && in_scale.size(1) == num_groups,
              "Scale must be [M, K/group_size]");
  TORCH_CHECK(K_in % achunk == 0, "K must be divisible by achunk=", achunk);

  const int max_lds_len = get_lds_size_int4() / 2;
  TORCH_CHECK(K_in * N_in <= (int64_t)(max_lds_len * 1.2),
              "K*N exceeds medium LDS capacity. K=", K_in, " N=", N_in,
              " K*N=", K_in * N_in, " max=", (int64_t)(max_lds_len * 1.2));

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using fptype = half;
  const uint8_t* wptr = reinterpret_cast<const uint8_t*>(in_a.data_ptr());
  const fptype* aptr = reinterpret_cast<const fptype*>(in_b.data_ptr());
  const fptype* sptr = reinterpret_cast<const fptype*>(in_scale.data_ptr());
  const fptype* biasptr = nullptr;
  fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());

  const int THRDS = is_gfx1x_int4() ? 32 : 64;

  #define SWEEP_GHF_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, _N, _GS) \
    {                                                                         \
      dim3 block(_THRDS, _WVPRGRP);                                           \
      int __wvPrGrp = mindiv_int4(M_in, CuCount * _YTILE, _WVPRGRP);          \
      wvSplitK_int4_hf_<fptype, _THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, _N, \
                        _GS><<<grid, block, 0, stream>>>(                     \
          K_in, M_in, 1, 1, wptr, aptr, sptr, nullptr, biasptr, cptr,         \
          __wvPrGrp, CuCount);                                                \
    }

  #define SWEEP_GHF_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, _GS)       \
    switch (N_in) {                                                        \
      case 1:                                                              \
        SWEEP_GHF_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 1, _GS) \
        break;                                                             \
      case 2:                                                              \
        SWEEP_GHF_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 2, _GS) \
        break;                                                             \
      case 3:                                                              \
        SWEEP_GHF_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 3, _GS) \
        break;                                                             \
      case 4:                                                              \
        SWEEP_GHF_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 4, _GS) \
        break;                                                             \
      default:                                                             \
        TORCH_CHECK(false, "Unsupported N=", N_in);                        \
    }

  #define SWEEP_GHF_UNRL(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _GS) \
    if (unrl == 1) {                                             \
      SWEEP_GHF_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 1, _GS)     \
    } else if (unrl == 2) {                                      \
      SWEEP_GHF_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 2, _GS)     \
    } else if (unrl == 4) {                                      \
      SWEEP_GHF_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 4, _GS)     \
    } else {                                                     \
      TORCH_CHECK(false, "Unsupported unrl=", unrl);             \
    }

  #define SWEEP_GHF_YTILE(_THRDS, _WVPRGRP, _ACHUNK, _GS) \
    if (ytile == 1) {                                     \
      SWEEP_GHF_UNRL(_THRDS, 1, _WVPRGRP, _ACHUNK, _GS)   \
    } else if (ytile == 2) {                              \
      SWEEP_GHF_UNRL(_THRDS, 2, _WVPRGRP, _ACHUNK, _GS)   \
    } else if (ytile == 4) {                              \
      SWEEP_GHF_UNRL(_THRDS, 4, _WVPRGRP, _ACHUNK, _GS)   \
    } else {                                              \
      TORCH_CHECK(false, "Unsupported ytile=", ytile);    \
    }

  #define SWEEP_GHF_WVPRGRP(_THRDS, _ACHUNK, _GS)          \
    if (wvprgrp == 8) {                                    \
      SWEEP_GHF_YTILE(_THRDS, 8, _ACHUNK, _GS)             \
    } else if (wvprgrp == 12) {                            \
      SWEEP_GHF_YTILE(_THRDS, 12, _ACHUNK, _GS)            \
    } else if (wvprgrp == 16) {                            \
      SWEEP_GHF_YTILE(_THRDS, 16, _ACHUNK, _GS)            \
    } else {                                               \
      TORCH_CHECK(false, "Unsupported wvprgrp=", wvprgrp); \
    }

  #define SWEEP_GHF_ACHUNK(_THRDS, _GS)                  \
    if (achunk == 8) {                                   \
      SWEEP_GHF_WVPRGRP(_THRDS, 8, _GS)                  \
    } else if (achunk == 16) {                           \
      SWEEP_GHF_WVPRGRP(_THRDS, 16, _GS)                 \
    } else if (achunk == 32) {                           \
      SWEEP_GHF_WVPRGRP(_THRDS, 32, _GS)                 \
    } else {                                             \
      TORCH_CHECK(false, "Unsupported achunk=", achunk); \
    }

  if (THRDS == 32) {
    if (group_size == 128) {
      SWEEP_GHF_ACHUNK(32, 128)
    } else if (group_size == 64) {
      SWEEP_GHF_ACHUNK(32, 64)
    } else if (group_size == 32) {
      SWEEP_GHF_ACHUNK(32, 32)
    } else {
      TORCH_CHECK(false, "Unsupported group_size=", group_size);
    }
  } else {
    if (group_size == 128) {
      SWEEP_GHF_ACHUNK(64, 128)
    } else if (group_size == 64) {
      SWEEP_GHF_ACHUNK(64, 64)
    } else if (group_size == 32) {
      SWEEP_GHF_ACHUNK(64, 32)
    } else {
      TORCH_CHECK(false, "Unsupported group_size=", group_size);
    }
  }

  #undef SWEEP_GHF_LAUNCH
  #undef SWEEP_GHF_N
  #undef SWEEP_GHF_UNRL
  #undef SWEEP_GHF_YTILE
  #undef SWEEP_GHF_WVPRGRP
  #undef SWEEP_GHF_ACHUNK

  return out_c;
}

// MoE int4 sweep op (bf16/fp16).  Lets a benchmark harness pick
// (ytile, unrl, achunk, wvprgrp) at runtime and route through the
// existing MOE_WVSPLITK_INT4G_LAUNCH_W_AC macro so the (Y, U, W, AC)
// space can be explored on real expert-routed shapes.  Mirrors the
// argument list of fused_moe_wvSplitK_int4_gemm so the harness can pass
// the same tensors and just append four runtime knobs.  The dispatcher's
// MOE_WVSPLIT_INT4G_DISPATCH macro only exposes (Y, U) in production —
// (W, AC) are bound to the (W=16, AC=16) defaults except for one
// hand-tuned tiny-K branch.  This sweep op opens the rest of the space.

#endif  // VLLM_SKINNY_GEMM_SWEEP
