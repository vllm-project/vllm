#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

#include "../cuda_compat.h"
#include "dispatch_utils.h"

#include "skinny_gemms_w8a8/kernel.cuh"  // arch helpers + scalar<> traits
#include "skinny_gemms_w8a8/launch.h"    // per-N launchers (defined in shards)

torch::Tensor wvSplitK_w8a8(const at::Tensor& in_a, const at::Tensor& in_b,
                            const at::Tensor& in_w_scale,
                            const std::optional<at::Tensor>& in_a_scale,
                            const std::optional<at::Tensor>& in_bias,
                            const int64_t CuCount) {
  // in_a: int8 weights [M, K]
  // in_b: int8 or fp16/bf16 activations [N, K]
  // in_w_scale: per-channel weight scale [M] in fp16/bf16
  // in_a_scale: per-tensor activation scale (scalar) in float32 — optional
  //             (None for dynamic quantization)
  // in_bias: optional bias
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == torch::kInt8, "Weight must be int8");

  // Determine quantization mode:
  //   1 = fused static quant (bf16 in, known a_scale)
  //   2 = fused dynamic quant (bf16 in, compute a_scale per row)
  TORCH_CHECK(in_b.dtype() != torch::kInt8,
              "Pre-quantized int8 activations not supported; pass bf16/fp16");
  TORCH_CHECK(in_b.dtype() == in_w_scale.dtype(),
              "Activation dtype must match weight scale dtype, got ",
              in_b.dtype(), " vs ", in_w_scale.dtype());
  int quant_mode;
  if (in_a_scale.has_value() && in_a_scale->numel() > 0) {
    quant_mode = 1;  // fused static
  } else {
    quant_mode = 2;  // fused dynamic
  }
  if (in_a_scale.has_value() && in_a_scale->numel() > 0) {
    TORCH_CHECK(in_a_scale->dtype() == torch::kFloat32,
                "Activation scale must be float32");
  }
  TORCH_CHECK(in_w_scale.dtype() == torch::kFloat16 ||
                  in_w_scale.dtype() == torch::kBFloat16,
              "Weight scale must be float16 or bfloat16");
  TORCH_CHECK(in_w_scale.size(0) == M_in, "Weight scale size must match M");
  TORCH_CHECK(K_in % 16 == 0, "K must be divisible by 16 for w8a8 kernel");

  // LDS stores int8 activations: 1 byte each (full LDS capacity)
  const int max_lds_ints = get_lds_size_w8a8();
  TORCH_CHECK(K_in * N_in <= max_lds_ints,
              "K*N exceeds LDS capacity; only sml variant is supported. "
              "K=",
              K_in, " N=", N_in, " K*N=", K_in * N_in, " max=", max_lds_ints);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_w_scale.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int thrds = is_gfx11_w8a8() ? 32 : 64;

  // Heuristic: pick (ytile, unrl, achunk) per shape and arch class.
  // The (yt, ur, ac) tuples reachable here MUST also be enumerated in
  // dispatch.cuh's dispatch_w8a8 helper.
  //
  // Tuned from the w8a8 sweep on gfx1151/gfx1150/gfx1103 (Apr 2026):
  //   - gfx1151 default is (1, 4, 32); small-K shapes (K<=1024) prefer
  //     ac=8 with yt scaling on M; lm_head/large-gate_up at K=2560-3584
  //     prefers (4, 1, 16). gfx9 shares this branch (no sweep data).
  //   - gfx1103 (Radeon 760M, RDNA 3 mobile) prefers ur=4 across the board;
  //     yt drops as K grows. For N>=2 ac also shrinks with K — small-K
  //     batched GEMMs need finer K-loop granularity than ac=32 provides.
  //   - Low-bandwidth gfx11 (gfx1150/1152/1153, RDNA 3.5 mobile) defaults
  //     to (4, 2, 32); K-heavy down/MLP-back shapes prefer (2, 2, 32);
  //     very-large-M N=1 shapes prefer (4, 4, 32), but the same shapes at
  //     N>=2 want (4, 2, 32) — ur=4 is a clear loser there.
  int ytile, unrl, achunk;
  if (is_gfx1103_w8a8()) {
    if (K_in >= 9728) {
      ytile = 1;
      unrl = 4;
      achunk = 32;
    } else if (N_in >= 2) {
      // Batched paths want smaller ac for smaller K on gfx1103.
      if (K_in >= 3584) {
        ytile = 2;
        unrl = 4;
        achunk = 32;
      } else if (K_in >= 2048) {
        ytile = 4;
        unrl = 4;
        achunk = 16;
      } else {  // K_in <= 1024
        ytile = 4;
        unrl = 4;
        achunk = 8;
      }
    } else if (K_in >= 4096) {
      // N == 1
      ytile = 2;
      unrl = 4;
      achunk = 32;
    } else {
      ytile = 4;
      unrl = 4;
      achunk = 32;
    }
  } else if (is_low_bandwidth_gfx11_w8a8()) {
    if (K_in >= 11008) {
      ytile = 2;
      unrl = 2;
      achunk = 32;
    } else if (M_in >= 19000 && K_in <= 3584) {
      // ur=4 wins on N=1 here, but loses to ur=2 on batched (N>=2).
      ytile = 4;
      unrl = (N_in == 1) ? 4 : 2;
      achunk = 32;
    } else {
      ytile = 4;
      unrl = 2;
      achunk = 32;
    }
  } else {
    // gfx1151 and gfx9
    if (K_in <= 1024) {
      // Small-K shapes prefer ac=8; yt scales with M.
      unrl = 4;
      achunk = 8;
      ytile = (M_in >= 4096) ? 4 : 2;
    } else if (N_in == 1 && M_in >= 19000 && K_in >= 2560 && K_in <= 3584) {
      // lm_head / large-gate_up at K=2560-3584 (N=1 only).
      ytile = 4;
      unrl = 1;
      achunk = 16;
    } else {
      ytile = 1;
      unrl = 4;
      achunk = 32;
    }
  }
  TORCH_CHECK(M_in % ytile == 0, "M must be divisible by ytile=", ytile);
  TORCH_CHECK(K_in % achunk == 0, "K must be divisible by achunk=", achunk);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      in_w_scale.scalar_type(), "wvSplitK_w8a8", [&] {
        using fptype = typename scalar<scalar_t>::type;
        const int8_t* wptr = in_a.data_ptr<int8_t>();
        const void* aptr = in_b.data_ptr();
        const fptype* wsptr =
            reinterpret_cast<const fptype*>(in_w_scale.data_ptr());
        const float* asptr = (in_a_scale.has_value() && in_a_scale->numel() > 0)
                                 ? in_a_scale->data_ptr<float>()
                                 : nullptr;
        const fptype* biasptr =
            (in_bias.has_value() && in_bias->numel() > 0)
                ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
                : nullptr;
        fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());

        switch (N_in) {
          case 1:
            launch_w8a8_n1<fptype>(grid, stream, K_in, M_in, Bx_in, By_in, wptr,
                                   aptr, wsptr, asptr, biasptr, cptr, CuCount,
                                   quant_mode, thrds, ytile, unrl, achunk);
            break;
          case 2:
            launch_w8a8_n2<fptype>(grid, stream, K_in, M_in, Bx_in, By_in, wptr,
                                   aptr, wsptr, asptr, biasptr, cptr, CuCount,
                                   quant_mode, thrds, ytile, unrl, achunk);
            break;
          case 3:
            launch_w8a8_n3<fptype>(grid, stream, K_in, M_in, Bx_in, By_in, wptr,
                                   aptr, wsptr, asptr, biasptr, cptr, CuCount,
                                   quant_mode, thrds, ytile, unrl, achunk);
            break;
          case 4:
            launch_w8a8_n4<fptype>(grid, stream, K_in, M_in, Bx_in, By_in, wptr,
                                   aptr, wsptr, asptr, biasptr, cptr, CuCount,
                                   quant_mode, thrds, ytile, unrl, achunk);
            break;
          case 5:
            launch_w8a8_n5<fptype>(grid, stream, K_in, M_in, Bx_in, By_in, wptr,
                                   aptr, wsptr, asptr, biasptr, cptr, CuCount,
                                   quant_mode, thrds, ytile, unrl, achunk);
            break;
          default:
            throw std::runtime_error(
                "Unsupported N value: " + std::to_string(M_in) + "," +
                std::to_string(K_in) + "," + std::to_string(N_in));
        }
      });

  return out_c;
}

// Sweep function disabled by default to reduce compile time.
// Build with -DVLLM_SKINNY_GEMM_SWEEP to enable.
#ifdef VLLM_SKINNY_GEMM_SWEEP
torch::Tensor wvSplitK_w8a8_sweep(const at::Tensor& in_a,
                                  const at::Tensor& in_b,
                                  const at::Tensor& in_w_scale,
                                  const std::optional<at::Tensor>& in_a_scale,
                                  const std::optional<at::Tensor>& in_bias,
                                  const int64_t CuCount, const int64_t ytile,
                                  const int64_t unrl, const int64_t achunk,
                                  const int64_t wvprgrp) {
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == torch::kInt8, "Weight must be int8");
  TORCH_CHECK(in_b.dtype() != torch::kInt8,
              "Pre-quantized int8 activations not supported; pass bf16/fp16");
  TORCH_CHECK(in_b.dtype() == in_w_scale.dtype(),
              "Activation dtype must match weight scale dtype, got ",
              in_b.dtype(), " vs ", in_w_scale.dtype());
  int quant_mode;
  if (in_a_scale.has_value() && in_a_scale->numel() > 0) {
    quant_mode = 1;
  } else {
    quant_mode = 2;
  }
  if (in_a_scale.has_value() && in_a_scale->numel() > 0) {
    TORCH_CHECK(in_a_scale->dtype() == torch::kFloat32,
                "Activation scale must be float32");
  }
  TORCH_CHECK(in_w_scale.dtype() == torch::kFloat16 ||
                  in_w_scale.dtype() == torch::kBFloat16,
              "Weight scale must be float16 or bfloat16");
  TORCH_CHECK(in_w_scale.size(0) == M_in, "Weight scale size must match M");
  TORCH_CHECK(K_in % achunk == 0, "K must be divisible by achunk=", achunk);
  TORCH_CHECK(M_in % ytile == 0, "M must be divisible by ytile=", ytile);

  const int max_lds_ints = get_lds_size_w8a8();
  TORCH_CHECK(K_in * N_in <= max_lds_ints, "K*N exceeds LDS capacity. K=", K_in,
              " N=", N_in);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_w_scale.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int thrds = is_gfx11_w8a8() ? 32 : 64;

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      in_w_scale.scalar_type(), "wvSplitK_w8a8_sweep", [&] {
        using fptype = typename scalar<scalar_t>::type;
        const int8_t* wptr = in_a.data_ptr<int8_t>();
        const void* aptr = in_b.data_ptr();
        const fptype* wsptr =
            reinterpret_cast<const fptype*>(in_w_scale.data_ptr());
        const float* asptr = (in_a_scale.has_value() && in_a_scale->numel() > 0)
                                 ? in_a_scale->data_ptr<float>()
                                 : nullptr;
        const fptype* biasptr =
            (in_bias.has_value() && in_bias->numel() > 0)
                ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
                : nullptr;
        fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());

        switch (N_in) {
          case 1:
            launch_w8a8_n1_sweep<fptype>(
                grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, wsptr,
                asptr, biasptr, cptr, CuCount, quant_mode, thrds, (int)ytile,
                (int)wvprgrp, (int)achunk, (int)unrl);
            break;
          case 2:
            launch_w8a8_n2_sweep<fptype>(
                grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, wsptr,
                asptr, biasptr, cptr, CuCount, quant_mode, thrds, (int)ytile,
                (int)wvprgrp, (int)achunk, (int)unrl);
            break;
          case 3:
            launch_w8a8_n3_sweep<fptype>(
                grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, wsptr,
                asptr, biasptr, cptr, CuCount, quant_mode, thrds, (int)ytile,
                (int)wvprgrp, (int)achunk, (int)unrl);
            break;
          case 4:
            launch_w8a8_n4_sweep<fptype>(
                grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, wsptr,
                asptr, biasptr, cptr, CuCount, quant_mode, thrds, (int)ytile,
                (int)wvprgrp, (int)achunk, (int)unrl);
            break;
          case 5:
            launch_w8a8_n5_sweep<fptype>(
                grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, wsptr,
                asptr, biasptr, cptr, CuCount, quant_mode, thrds, (int)ytile,
                (int)wvprgrp, (int)achunk, (int)unrl);
            break;
          default:
            TORCH_CHECK(false, "Unsupported N=", N_in);
        }
      });

  return out_c;
}
#endif  // VLLM_SKINNY_GEMM_SWEEP
