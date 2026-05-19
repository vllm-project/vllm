// Sweep wrapper: parallel-compiled TU sharing kernel templates with the
// production .cu via skinny_gemms_int4_kernels.cuh.
#ifdef VLLM_SKINNY_GEMM_SWEEP
  #include "skinny_gemms_int4_kernels.cuh"

void fused_moe_wvSplitK_int4_gemm_sweep(
    torch::Tensor a, torch::Tensor w, torch::Tensor scales, torch::Tensor c,
    torch::Tensor expert_ids, int64_t block_size_m, int64_t CuCount,
    int64_t group_size, torch::Tensor zero_points,
    torch::Tensor sorted_token_ids, int64_t top_k, bool fuse_silu_mul,
    int64_t ytile, int64_t unrl, int64_t achunk, int64_t wvprgrp) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int M_in = static_cast<int>(w.size(1));
  int K_in = static_cast<int>(w.size(2)) * 8;
  int N_in = static_cast<int>(block_size_m);
  int num_expert_blocks = static_cast<int>(expert_ids.size(0));

  bool has_zp = zero_points.numel() > 0;

  long expert_stride_w = w.stride(0) * static_cast<long>(sizeof(int32_t));
  long expert_stride_s = scales.stride(0);
  long expert_stride_zp = has_zp ? zero_points.stride(0) : 0;

  bool scattered = sorted_token_ids.numel() > 0;
  int top_k_in = scattered ? static_cast<int>(top_k) : 1;

  if (fuse_silu_mul) {
    TORCH_CHECK(N_in == 1, "fuse_silu_mul requires block_size_m == 1, got ",
                N_in);
    TORCH_CHECK(K_in * 1 <= MOE_LDS_ELEMS && M_in % 1 == 0,
                "fuse_silu_mul requires the sml LDS path");
    TORCH_CHECK(a.size(-1) == 2 * K_in,
                "fuse_silu_mul expects A's last dim to be 2*K_in");
  }
  TORCH_CHECK(M_in % ytile == 0, "M must be divisible by ytile=", ytile);
  TORCH_CHECK(K_in % achunk == 0, "K must be divisible by achunk=", achunk);
  TORCH_CHECK(group_size == 32 || group_size == 128,
              "group_size must be 32 or 128 for the MoE sweep");

  // Macro-switch chains over the four runtime knobs.  Same pattern as
  // wvSplitK_int4g_sweep above; expand only what the kernel template
  // instantiations actually reach (gfx1x THRDS=32 only).  Defined at file
  // scope because the C preprocessor doesn't process #define directives
  // that appear inside another macro's argument (AT_DISPATCH_* is itself
  // a macro, so its lambda body would not see these defines).
  #define MOE_SWP_LAUNCH(_YT, _W, _AC, _UN, _N, _GS, _HASZP) \
    MOE_WVSPLITK_INT4G_LAUNCH_W_AC(32, _YT, _W, _AC, _UN, _N, _GS, _HASZP)

  #define MOE_SWP_HASZP(_YT, _W, _AC, _UN, _N, _GS)     \
    if (has_zp) {                                       \
      MOE_SWP_LAUNCH(_YT, _W, _AC, _UN, _N, _GS, true)  \
    } else {                                            \
      MOE_SWP_LAUNCH(_YT, _W, _AC, _UN, _N, _GS, false) \
    }

  #define MOE_SWP_N(_YT, _W, _AC, _UN, _GS)                    \
    switch (N_in) {                                            \
      case 1:                                                  \
        MOE_SWP_HASZP(_YT, _W, _AC, _UN, 1, _GS);              \
        break;                                                 \
      case 2:                                                  \
        MOE_SWP_HASZP(_YT, _W, _AC, _UN, 2, _GS);              \
        break;                                                 \
      case 4:                                                  \
        MOE_SWP_HASZP(_YT, _W, _AC, _UN, 4, _GS);              \
        break;                                                 \
      default:                                                 \
        TORCH_CHECK(false, "Unsupported block_size_m=", N_in); \
    }

  #define MOE_SWP_UN(_YT, _W, _AC, _GS)              \
    if (unrl == 1) {                                 \
      MOE_SWP_N(_YT, _W, _AC, 1, _GS)                \
    } else if (unrl == 2) {                          \
      MOE_SWP_N(_YT, _W, _AC, 2, _GS)                \
    } else if (unrl == 4) {                          \
      MOE_SWP_N(_YT, _W, _AC, 4, _GS)                \
    } else {                                         \
      TORCH_CHECK(false, "Unsupported unrl=", unrl); \
    }

  #define MOE_SWP_YT(_W, _AC, _GS)                     \
    if (ytile == 1) {                                  \
      MOE_SWP_UN(1, _W, _AC, _GS)                      \
    } else if (ytile == 2) {                           \
      MOE_SWP_UN(2, _W, _AC, _GS)                      \
    } else if (ytile == 4) {                           \
      MOE_SWP_UN(4, _W, _AC, _GS)                      \
    } else {                                           \
      TORCH_CHECK(false, "Unsupported ytile=", ytile); \
    }

  #define MOE_SWP_W(_AC, _GS)                              \
    if (wvprgrp == 8) {                                    \
      MOE_SWP_YT(8, _AC, _GS)                              \
    } else if (wvprgrp == 12) {                            \
      MOE_SWP_YT(12, _AC, _GS)                             \
    } else if (wvprgrp == 16) {                            \
      MOE_SWP_YT(16, _AC, _GS)                             \
    } else if (wvprgrp == 32) {                            \
      MOE_SWP_YT(32, _AC, _GS)                             \
    } else {                                               \
      TORCH_CHECK(false, "Unsupported wvprgrp=", wvprgrp); \
    }

  #define MOE_SWP_AC(_GS)                                \
    if (achunk == 8) {                                   \
      MOE_SWP_W(8, _GS)                                  \
    } else if (achunk == 16) {                           \
      MOE_SWP_W(16, _GS)                                 \
    } else if (achunk == 32) {                           \
      MOE_SWP_W(32, _GS)                                 \
    } else {                                             \
      TORCH_CHECK(false, "Unsupported achunk=", achunk); \
    }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      a.scalar_type(), "fused_moe_wvSplitK_int4_gemm_sweep", [&] {
        using fptype = typename scalar<scalar_t>::type;

        const uint8_t* wptr = reinterpret_cast<const uint8_t*>(w.data_ptr());
        const fptype* aptr = reinterpret_cast<const fptype*>(a.data_ptr());
        const fptype* sptr = reinterpret_cast<const fptype*>(scales.data_ptr());
        const fptype* zpptr =
            has_zp ? reinterpret_cast<const fptype*>(zero_points.data_ptr())
                   : nullptr;
        fptype* cptr = reinterpret_cast<fptype*>(c.data_ptr());
        const int* eidptr = expert_ids.data_ptr<int32_t>();
        const int* stidptr =
            scattered ? sorted_token_ids.data_ptr<int32_t>() : nullptr;

        if (group_size == 128) {
          MOE_SWP_AC(128)
        } else {
          MOE_SWP_AC(32)
        }
      });
}

  #undef MOE_SWP_LAUNCH
  #undef MOE_SWP_HASZP
  #undef MOE_SWP_N
  #undef MOE_SWP_UN
  #undef MOE_SWP_YT
  #undef MOE_SWP_W
  #undef MOE_SWP_AC

#endif  // VLLM_SKINNY_GEMM_SWEEP
