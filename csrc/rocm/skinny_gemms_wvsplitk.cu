// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-side launcher of the wvSplitK skinny GEMM; kernels in skinny_gemms.cuh.
#include "skinny_gemms.cuh"

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount) {
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);
  auto Kap_in = in_a.stride(0);
  auto Kbp_in = in_b.stride(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == in_b.dtype());
  TORCH_CHECK(K_in % 8 == 0, "k % 8 == 0");
  TORCH_CHECK(in_a.dtype() == torch::kFloat16 ||
              in_a.dtype() == torch::kBFloat16);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int max_lds_len = get_lds_size() / 2;

#define WVSPLITK_CFG(_THRDS, _WVPRGRP, _YTILE, _UNRL, _N)                     \
  {                                                                           \
    dim3 block(_THRDS, _WVPRGRP);                                             \
    int __wvPrGrp = mindiv(M_in, CuCount * _YTILE, _WVPRGRP);                 \
    if ((Kbp_in * N_in <= max_lds_len) && (M_in % _YTILE == 0))               \
      wvSplitK_hf_sml_<fptype, _THRDS, _YTILE, _WVPRGRP, 8, _UNRL, _N>        \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, af4, bf4, biasf4, c, __wvPrGrp, \
                                       CuCount);                              \
    else if (Kbp_in * N_in <= max_lds_len * 1.2)                              \
      wvSplitK_hf_<fptype, _THRDS, _YTILE, _WVPRGRP, 8, _UNRL, _N>            \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, af4, bf4, biasf4, c, __wvPrGrp, \
                                       CuCount);                              \
    else                                                                      \
      wvSplitK_hf_big_<fptype, _THRDS, _YTILE, _WVPRGRP, 8, _UNRL, _N>        \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, af4, bf4, biasf4, c, __wvPrGrp, \
                                       CuCount);                              \
  }

#define WVSPLIT_TILE_CFG(_THRDS, _WVPRGRP, _sYT, __N)     \
  {                                                       \
    bool fit_lds = (Kbp_in * N_in <= max_lds_len);        \
    if (_sYT <= 1)                                        \
      WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 4, __N)           \
    else if ((__N == 1) || (!fit_lds) || (_sYT <= 4 * 2)) \
      WVSPLITK_CFG(_THRDS, _WVPRGRP, 2, 2, __N)           \
    else if (_sYT <= 4 * 3)                               \
      WVSPLITK_CFG(_THRDS, _WVPRGRP, 3, 2, __N)           \
    else if (__N == 4)                                    \
      WVSPLITK_CFG(_THRDS, _WVPRGRP, 4, 1, __N)           \
    else                                                  \
      WVSPLITK_CFG(_THRDS, _WVPRGRP, 4, 2, __N)           \
  }

// WVSPLITK_CFG arguments are: (THRDS, WVPRGRP, YTILE, UNRL, N).
//   THRDS  = wavefront width (32 on GFX11/GFX12, 64 on GFX9)
//   WVPRGRP= waves per group (always 16)
//   YTILE  = output rows per thread tile
//   UNRL   = K-loop unroll factor
//   N      = batch size (passed through from the switch in wvSplitK)
#define WVSPLIT_TILE(_sYT, __N)                                             \
  {                                                                         \
    if (on_gfx1151()) {                                                     \
      bool fit_lds = (Kbp_in * N_in <= max_lds_len);                        \
      if (_sYT <= 1)                                                        \
        WVSPLITK_CFG(/*THRDS=*/32, /*WVPRGRP=*/16, /*YTILE=*/1, /*UNRL=*/4, \
                     __N)                                                   \
      else if ((K_in % 1024 == 512) && K_in >= 1536 &&                      \
               (_sYT >= 40 || K_in >= 4096))                                \
        WVSPLITK_CFG(/*THRDS=*/32, /*WVPRGRP=*/16, /*YTILE=*/4, /*UNRL=*/1, \
                     __N)                                                   \
      else if (K_in < 1024)                                                 \
        WVSPLITK_CFG(/*THRDS=*/32, /*WVPRGRP=*/16, /*YTILE=*/2, /*UNRL=*/4, \
                     __N)                                                   \
      else if (K_in <= 2048 && (__N >= 2 || _sYT <= 26))                    \
        WVSPLITK_CFG(/*THRDS=*/32, /*WVPRGRP=*/16, /*YTILE=*/1, /*UNRL=*/4, \
                     __N)                                                   \
      else if (__N >= 2 && !fit_lds)                                        \
        WVSPLITK_CFG(/*THRDS=*/32, /*WVPRGRP=*/16, /*YTILE=*/1, /*UNRL=*/4, \
                     __N)                                                   \
      else if (__N == 1)                                                    \
        WVSPLITK_CFG(/*THRDS=*/32, /*WVPRGRP=*/16, /*YTILE=*/1, /*UNRL=*/2, \
                     __N)                                                   \
      else                                                                  \
        WVSPLITK_CFG(/*THRDS=*/32, /*WVPRGRP=*/16, /*YTILE=*/1, /*UNRL=*/1, \
                     __N)                                                   \
    } else if (on_gfx1x()) { /* gfx1100/gfx1150/GFX12, wave32 */            \
      WVSPLIT_TILE_CFG(/*THRDS=*/32, /*WVPRGRP=*/16, _sYT, __N)             \
    } else { /* GFX9, wave64 */                                             \
      WVSPLIT_TILE_CFG(/*THRDS=*/64, /*WVPRGRP=*/16, _sYT, __N)             \
    }                                                                       \
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_b.scalar_type(), "wvSplitK", [&] {
    using fptype = typename scalar<scalar_t>::type;
    fptype* af4 = reinterpret_cast<fptype*>(in_a.data_ptr());
    const fptype* bf4 = reinterpret_cast<const fptype*>(in_b.data_ptr());
    const fptype* biasf4 =
        (in_bias.has_value() && in_bias->numel() > 0)
            ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
            : nullptr;
    fptype* c = reinterpret_cast<fptype*>(out_c.data_ptr());

    // first shoot for biggest tile-size that keeps all simd busy,
    // then cut the active waves to balance their distribution...
    int sYT = (M_in + CuCount * 4 - 1) / (CuCount * 4);

    switch (N_in) {
      case 1:
        WVSPLIT_TILE(sYT, 1)
        break;
      case 2:
        WVSPLIT_TILE(sYT, 2)
        break;
      case 3:
        WVSPLIT_TILE(sYT, 3)
        break;
      case 4:
        WVSPLIT_TILE(sYT, 4)
        break;
      case 5:
        WVSPLIT_TILE(sYT, 5)
        break;
      default:
        throw std::runtime_error(
            "Unsupported N value: " + std::to_string(M_in) + "," +
            std::to_string(K_in) + "," + std::to_string(N_in));
    }
  });
  return out_c;
}
