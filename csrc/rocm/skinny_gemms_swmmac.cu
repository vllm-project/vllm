// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-side launcher of the SWMMAC skinny GEMM; kernels in skinny_gemms.cuh.
#include "skinny_gemms.cuh"

torch::Tensor swmmacGEMM(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t logical_M, const int64_t CuCount) {
  TORCH_CHECK(on_gfx120x(), "swmmacGEMM is only supported on GFX120x");
  TORCH_CHECK(in_a.is_cuda() && in_b.is_cuda(),
              "swmmacGEMM inputs must be CUDA tensors");
  TORCH_CHECK(in_b.scalar_type() == torch::kFloat16 ||
                  in_b.scalar_type() == torch::kBFloat16,
              "B must be FP16 or BF16");
  TORCH_CHECK(in_a.scalar_type() == in_b.scalar_type(),
              "A and B must use the same dtype");
  TORCH_CHECK(in_b.dim() == 2, "B must be a 2D tensor");
  TORCH_CHECK(in_a.dim() == 2, "A must be a 2D tensor");
  TORCH_CHECK(in_a.device() == in_b.device(),
              "A and B must be on the same device");
  TORCH_CHECK(in_a.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(in_b.is_contiguous(), "B must be contiguous");

  constexpr int tile_columns = 16;
  constexpr int m_tiles_per_wave = 1;
  constexpr int load_group = 4;
  constexpr int columns_per_wave = tile_columns * m_tiles_per_wave;
  constexpr int k_per_instruction = 32;
  constexpr int split_k = 4;
  constexpr int regular_waves_per_block = 4;
  constexpr int threads_per_wave = 32;

  const int M = static_cast<int>(in_b.size(0));
  const int K = static_cast<int>(in_b.size(1));
  const int N = static_cast<int>(in_a.size(0));
  const int A_stride = static_cast<int>(in_a.stride(0));

  TORCH_CHECK(logical_M == M, "logical_M must match the row count of raw B");
  TORCH_CHECK(N >= 5 && N <= 8, "swmmacGEMM supports N in [5, 8]");
  TORCH_CHECK(in_a.size(1) == K, "A and B inner dimensions (K) must match");
  TORCH_CHECK(K % k_per_instruction == 0, "K must be divisible by ",
              k_per_instruction);

  const bool has_bias = in_bias.has_value() && in_bias->numel() > 0;

  if (has_bias) {
    TORCH_CHECK(in_bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(in_bias->device() == in_a.device(),
                "bias and A must be on the same device");
    TORCH_CHECK(in_bias->scalar_type() == in_a.scalar_type(),
                "bias and A must use the same dtype");
    TORCH_CHECK(in_bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(in_bias->dim() == 1 || in_bias->dim() == 2,
                "bias must be a 1D or 2D tensor");
  }

  // Bias may be scalar, per-column, per-row, or full [N, M]. Bx is the
  // logical bias width used by swmmac_bias_index().
  const int Bx = !has_bias
                     ? 1
                     : static_cast<int>(in_bias->dim() == 2 ? in_bias->size(1)
                                                            : in_bias->size(0));

  // By is the logical bias height. A 1D bias is treated as [1, M].
  const int By =
      !has_bias || in_bias->dim() == 1 ? 1 : static_cast<int>(in_bias->size(0));

  TORCH_CHECK((Bx == 1 || Bx == M) && (By == 1 || By == N),
              "bias must be broadcastable to the [N, M] output");

  auto out_c = torch::empty(
      {N, M}, torch::TensorOptions().dtype(in_a.dtype()).device(in_a.device()));

  const auto ceil_div = [](const int value, const int divisor) {
    return (value + divisor - 1) / divisor;
  };

  const int cu_count = static_cast<int>(CuCount);
  const int total_num_kblocks = K / k_per_instruction;

  const int regular_columns_per_block =
      regular_waves_per_block * columns_per_wave;

  const int regular_workgroups = ceil_div(M, regular_columns_per_block);
  const int regular_useful_waves = ceil_div(M, columns_per_wave);

  // Use split-K when regular M tiling does not create enough waves to fill
  // the GPU and K is long enough to make splitting worthwhile.
  const bool regular_is_underfilled = regular_useful_waves < cu_count * 4;
  const bool splitk_is_profitable = total_num_kblocks >= 64;
  const bool use_splitk = regular_is_underfilled && splitk_is_profitable;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define VLLM_LAUNCH_SWMMAC_REGULAR(_N, _HAS_BIAS, _ALIGNED)                    \
  swmmacGemmRegular<fptype, _N, _HAS_BIAS, load_group, _ALIGNED, tile_columns, \
                    m_tiles_per_wave, k_per_instruction,                       \
                    regular_waves_per_block, threads_per_wave>                 \
      <<<grid, block, 0, stream>>>(K, A_stride, M, Bx, By, A, B, BIAS, C)

#define VLLM_DISPATCH_SWMMAC_REGULAR(_N, _HAS_BIAS)     \
  do {                                                  \
    if (aligned_regular) {                              \
      VLLM_LAUNCH_SWMMAC_REGULAR(_N, _HAS_BIAS, true);  \
    } else {                                            \
      VLLM_LAUNCH_SWMMAC_REGULAR(_N, _HAS_BIAS, false); \
    }                                                   \
  } while (0)

#define VLLM_DISPATCH_SWMMAC_REGULAR_BY_N(_HAS_BIAS) \
  do {                                               \
    switch (N) {                                     \
      case 5:                                        \
        VLLM_DISPATCH_SWMMAC_REGULAR(5, _HAS_BIAS);  \
        break;                                       \
      case 6:                                        \
        VLLM_DISPATCH_SWMMAC_REGULAR(6, _HAS_BIAS);  \
        break;                                       \
      case 7:                                        \
        VLLM_DISPATCH_SWMMAC_REGULAR(7, _HAS_BIAS);  \
        break;                                       \
      case 8:                                        \
        VLLM_DISPATCH_SWMMAC_REGULAR(8, _HAS_BIAS);  \
        break;                                       \
      default:                                       \
        TORCH_CHECK(false, "unsupported N: ", N);    \
    }                                                \
  } while (0)

#define VLLM_LAUNCH_SWMMAC_SPLITK(_N, _HAS_BIAS, _ALIGNED)                     \
  swmmacGemmSplitK<fptype, _N, _HAS_BIAS, split_k, load_group, _ALIGNED,       \
                   tile_columns, m_tiles_per_wave, k_per_instruction, split_k, \
                   threads_per_wave>                                           \
      <<<fused_splitk_grid, fused_splitk_block, 0, stream>>>(                  \
          K, A_stride, M, Bx, By, A, B, BIAS, C)

#define VLLM_DISPATCH_SWMMAC_SPLITK(_N, _HAS_BIAS)     \
  do {                                                 \
    if (aligned_splitk) {                              \
      VLLM_LAUNCH_SWMMAC_SPLITK(_N, _HAS_BIAS, true);  \
    } else {                                           \
      VLLM_LAUNCH_SWMMAC_SPLITK(_N, _HAS_BIAS, false); \
    }                                                  \
  } while (0)

#define VLLM_DISPATCH_SWMMAC_SPLITK_BY_N(_HAS_BIAS) \
  do {                                              \
    switch (N) {                                    \
      case 5:                                       \
        VLLM_DISPATCH_SWMMAC_SPLITK(5, _HAS_BIAS);  \
        break;                                      \
      case 6:                                       \
        VLLM_DISPATCH_SWMMAC_SPLITK(6, _HAS_BIAS);  \
        break;                                      \
      case 7:                                       \
        VLLM_DISPATCH_SWMMAC_SPLITK(7, _HAS_BIAS);  \
        break;                                      \
      case 8:                                       \
        VLLM_DISPATCH_SWMMAC_SPLITK(8, _HAS_BIAS);  \
        break;                                      \
      default:                                      \
        TORCH_CHECK(false, "unsupported N: ", N);   \
    }                                               \
  } while (0)

  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_a.scalar_type(), "swmmacGEMM", [&] {
    using fptype = typename scalar<scalar_t>::type;

    const auto* B = reinterpret_cast<const fptype*>(in_b.data_ptr());
    const auto* A = reinterpret_cast<const fptype*>(in_a.data_ptr());

    const auto* BIAS =
        has_bias ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
                 : nullptr;

    auto* C = reinterpret_cast<fptype*>(out_c.data_ptr());

    if (!use_splitk) {
      const int grid_x =
          std::max(1, std::min(regular_workgroups, cu_count * 2));

      const dim3 grid(grid_x);
      const dim3 block(threads_per_wave, regular_waves_per_block);

      const bool aligned_regular = total_num_kblocks % load_group == 0;

      if (has_bias) {
        VLLM_DISPATCH_SWMMAC_REGULAR_BY_N(true);
      } else {
        VLLM_DISPATCH_SWMMAC_REGULAR_BY_N(false);
      }

      return;
    }

    const int fused_splitk_workgroups = ceil_div(M, columns_per_wave);

    const int fused_splitk_grid_x =
        std::max(1, std::min(fused_splitk_workgroups, cu_count * split_k * 2));

    const dim3 fused_splitk_grid(fused_splitk_grid_x);
    const dim3 fused_splitk_block(threads_per_wave, split_k);

    const bool aligned_splitk = total_num_kblocks % split_k == 0 &&
                                (total_num_kblocks / split_k) % load_group == 0;

    if (has_bias) {
      VLLM_DISPATCH_SWMMAC_SPLITK_BY_N(true);
    } else {
      VLLM_DISPATCH_SWMMAC_SPLITK_BY_N(false);
    }
  });

#undef VLLM_LAUNCH_SWMMAC_REGULAR
#undef VLLM_DISPATCH_SWMMAC_REGULAR
#undef VLLM_DISPATCH_SWMMAC_REGULAR_BY_N
#undef VLLM_LAUNCH_SWMMAC_SPLITK
#undef VLLM_DISPATCH_SWMMAC_SPLITK
#undef VLLM_DISPATCH_SWMMAC_SPLITK_BY_N

  return out_c;
}
