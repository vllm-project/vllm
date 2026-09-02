// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

template <typename scalar_t, typename packed_t>
void bmm_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const packed_t* __restrict__ mat2,
    int64_t B,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideB,
    int64_t mat1_strideM,
    int64_t out_strideB,
    int64_t out_strideM,
    float scale = 0.f) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  // mat2 contiguous in [B, N, K]
  int64_t mat2_strideB = N * K;
  int64_t mat2_strideN = K;

  const bool use_brgemm = can_use_brgemm<scalar_t>(M);

  // parallel on [B, MB, NB]
  at::parallel_for(0, B * MB * NB, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, mb{0}, nb{0};
    data_index_init(begin, bs, B, mb, MB, nb, NB);

    // for brgemm, use float32 for accumulate
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

    for (int i = begin; i < end; ++i) {
      UNUSED(i);
      int mb_start = mb * BLOCK_M;
      int mb_size = std::min(M - mb_start, BLOCK_M);
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(N - nb_start, BLOCK_N);

      tinygemm_kernel<scalar_t>(
          /*   A */ mat1 + bs * mat1_strideB + mb_start * mat1_strideM,
          /*   B */ mat2 + bs * mat2_strideB + nb_start * mat2_strideN /* nb * BLOCK_N * K */,
          /*   C */ out + bs * out_strideB + mb_start * out_strideM + nb_start,
          /* Ctmp*/ Ctmp,
          /*   M */ mb_size,
          /*   N */ nb_size,
          /*   K */ K,
          /* lda */ mat1_strideM,
          /* ldb */ nb_size,
          /* ldc */ out_strideM,
          /* brg */ use_brgemm);

      // move to the next index
      data_index_step(bs, B, mb, MB, nb, NB);
    }

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

template <>
void bmm_kernel_impl(
    at::BFloat16* __restrict__ out,
    const at::BFloat16* __restrict__ mat1,
    const at::Float8_e4m3fn* __restrict__ mat2,
    int64_t B,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideB,
    int64_t mat1_strideM,
    int64_t out_strideB,
    int64_t out_strideM,
    float scale) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  // mat2 contiguous in [B, N, K]
  int64_t mat2_strideB = N * K;
  int64_t mat2_strideN = K;

  const bool use_brgemm = can_use_brgemm<at::BFloat16>(M);

  // parallel on [B, MB, NB]
  parallel_2d(B * MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // for brgemm, use float32 for accumulate
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];
    // for brgemm when mat2 is float8_e4m3
    alignas(64) at::BFloat16 Btmp[BLOCK_N * BLOCK_K];

    loop_2d<at::Float8_e4m3fn>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t bs = mb / MB;
      int64_t mb_start = (mb % MB) * BLOCK_M;
      int64_t mb_size = std::min(M - mb_start, BLOCK_M);
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(N - nb_start, BLOCK_N);

      tinygemm_kernel(
          /*   A */ mat1 + bs * mat1_strideB + mb_start * mat1_strideM,
          /*   B */ mat2 + bs * mat2_strideB + nb_start * mat2_strideN /* nb * BLOCK_N * K */,
          /*   C */ out + bs * out_strideB + mb_start * out_strideM + nb_start,
          /* Btmp*/ Btmp,
          /* Ctmp*/ Ctmp,
          /*scale*/ scale,
          /*   M */ mb_size,
          /*   N */ nb_size,
          /*   K */ K,
          /* lda */ mat1_strideM,
          /* ldb */ nb_size,
          /* ldc */ out_strideM,
          /* brg */ use_brgemm);
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

}  // anonymous namespace

// mat1 : [B, M, K]
// mat2 : [B, N, K] or [B, OC, IC]
// out  : [B, M, N]
// scale: [] 0-dim tensor for per tensor quant
//
void bmm_cpu(
    at::Tensor& out, at::Tensor& mat1, at::Tensor& mat2, bool is_vnni, const std::optional<at::Tensor>& scale) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  // input and out could be non-contiguous
  // weight needs to be contiguous in [OC, IC] order
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(out);
  CHECK_INPUT(mat2);
  CHECK_DIM(3, out);
  CHECK_DIM(3, mat1);
  CHECK_DIM(3, mat2);

  int64_t B = mat1.size(0);
  int64_t M = mat1.size(1);
  int64_t N = mat2.size(1);
  int64_t K = mat1.size(2);

  const bool use_fp8_w8a16 = scale.has_value();
  TORCH_CHECK(N % 32 == 0, "tinygemm requires N to be 32x.");

  int64_t mat1_strideB = mat1.stride(0);
  int64_t mat1_strideM = mat1.stride(1);
  int64_t out_strideB = out.stride(0);
  int64_t out_strideM = out.stride(1);

  // check shapes
  TORCH_CHECK(mat2.size(0) == B && mat2.size(2) == K, "bmm: mat2 shape mismatch!");
  TORCH_CHECK(out.size(0) == B && out.size(1) == M, "bmm: out shape mismatch!");
  if (!use_fp8_w8a16) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(mat1.scalar_type(), "bmm_kernel_impl", [&] {
      bmm_kernel_impl<scalar_t, scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          packed_w.data_ptr<scalar_t>(),
          B,
          M,
          N,
          K,
          mat1_strideB,
          mat1_strideM,
          out_strideB,
          out_strideM);
    });
  } else {  // fp8 bmm
    float scale_val = 0.f;

    auto scale_tensor = scale.value();
    TORCH_CHECK(scale_tensor.ndimension() == 0, "bmm: expect scale to be 0-dim tensor.");
    scale_val = scale_tensor.item<float>();

    bmm_kernel_impl<at::BFloat16, at::Float8_e4m3fn>(
        out.data_ptr<at::BFloat16>(),
        mat1.data_ptr<at::BFloat16>(),
        packed_w.data_ptr<at::Float8_e4m3fn>(),
        B,
        M,
        N,
        K,
        mat1_strideB,
        mat1_strideM,
        out_strideB,
        out_strideM,
        scale_val);
  }
}
