// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu
//
// Modified for vLLM: this fused kernel stops before the MLA Q-absorption step
// (SGLang's version additionally folds `w_kc` into Q here, producing an
// already-absorbed latent-space query). vLLM's shared MLA abstraction
// (`MLACommonImpl`) performs weight absorption generically for every backend
// via `torch.bmm`/`bmm_cpu` against `W_UK_T`, so this kernel instead returns
// the standard *unabsorbed* query (`q_nope || q_pe`, RoPE already applied),
// matching what `MultiHeadLatentAttentionWrapper.forward()` computes via its
// unfused path. `w_kc` is therefore not a parameter here.

// clang-format off

#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

// [NOTE]: Fused kernel for QKV projection with RoPE
//
//   1. `q_a_proj` and `kv_a_proj_with_mqa` fused into one gemm,
//      otherwise we need to split IC for the 2nd gemm.
//   2. `q_a_layernorm` and `kv_a_layernorm` fused into one parallel loop.
//   3. k_input and v_input share the same storage, the torch API did
//      this in `set_kv_buffer`. No additional memory movement.
//

// [C0, C1] = A @ [B0, B1]
template <typename scalar_t>
void segment_gemm_kernel_impl(
    scalar_t* __restrict__ C0,
    scalar_t* __restrict__ C1,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B0,
    const scalar_t* __restrict__ B1,
    int64_t M,
    int64_t N0,
    int64_t N1,
    int64_t K) {
  // convert_weight_packed make sure N0 and N1 are 32x
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB0 = div_up(N0, BLOCK_N);
  const int64_t NB1 = div_up(N1, BLOCK_N);
  const int64_t NB = NB0 + NB1;

  const bool use_brgemm = can_use_brgemm<scalar_t>(M);

  // parallel on [MB, NB0 + NB1]
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    int64_t mb{0}, nb{0};
    data_index_init(begin, mb, MB, nb, NB);

    // for brgemm, use float32 for accumulate
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

    for (int64_t i = begin; i < end; ++i) {
      UNUSED(i);
      int mb_start = mb * BLOCK_M;
      int mb_size = std::min(M - mb_start, BLOCK_M);
      int nb_start = nb * BLOCK_N;
      int nb_size = BLOCK_N;

      const scalar_t* __restrict__ B = nb < NB0 ? B0 : B1;
      scalar_t* __restrict__ C = nb < NB0 ? C0 : C1;
      int64_t ldc = nb < NB0 ? N0 : N1;
      int64_t local_nb_start = nb < NB0 ? nb_start : nb_start - N0;

      tinygemm_kernel<scalar_t>(
          /*   A */ A + mb_start * K,
          /*   B */ B + local_nb_start * K /* nb * BLOCK_N * K */,
          /*   C */ C + mb_start * ldc + local_nb_start,
          /* Ctmp*/ Ctmp,
          /*   M */ mb_size,
          /*   N */ nb_size,
          /*   K */ K,
          /* lda */ K,
          /* ldb */ nb_size,
          /* ldc */ ldc,
          /* brg */ use_brgemm);

      // move to the next index
      data_index_step(mb, MB, nb, NB);
    }

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

// [C0, C1] = A @ [B0, B1]
template <typename scalar_t>
void segment_gemm_kernel_impl(
    scalar_t* __restrict__ C0,
    scalar_t* __restrict__ C1,
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B0,
    const int8_t* __restrict__ B1,
    const float* __restrict__ As,
    const float* __restrict__ Bs0,
    const float* __restrict__ Bs1,
    int64_t M,
    int64_t N0,
    int64_t N1,
    int64_t K) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB0 = div_up(N0, BLOCK_N);
  const int64_t NB1 = div_up(N1, BLOCK_N);
  const int64_t NB = NB0 + NB1;

  const bool use_brgemm = can_use_brgemm<int8_t>(M);

  // K + 4 after compensation
  const int64_t packed_row_size = get_row_size<int8_t>(K);

  // parallel on [MB, NB0 + NB1]
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    int64_t mb{0}, nb{0};
    data_index_init(begin, mb, MB, nb, NB);

    // for brgemm, use float32 for accumulate
    alignas(64) int32_t Ctmp[BLOCK_M * BLOCK_N];

    for (int64_t i = begin; i < end; ++i) {
      UNUSED(i);
      int mb_start = mb * BLOCK_M;
      int mb_size = std::min(M - mb_start, BLOCK_M);
      int nb_start = nb * BLOCK_N;
      int nb_size = BLOCK_N;

      const int8_t* __restrict__ B = nb < NB0 ? B0 : B1;
      const float* __restrict__ Bs = nb < NB0 ? Bs0 : Bs1;
      scalar_t* __restrict__ C = nb < NB0 ? C0 : C1;
      int64_t ldc = nb < NB0 ? N0 : N1;
      int64_t local_nb_start = nb < NB0 ? nb_start : nb_start - N0;

      tinygemm_kernel<scalar_t>(
          /*   A */ A + mb_start * K,
          /*   B */ B + local_nb_start * packed_row_size /* nb * BLOCK_N * (K + 4) */,
          /*   C */ C + mb_start * ldc + local_nb_start,
          /* Ctmp*/ Ctmp,
          /*  As */ As + mb_start,
          /*  Bs */ Bs + local_nb_start,
          /*   M */ mb_size,
          /*   N */ nb_size,
          /*   K */ K,
          /* lda */ K,
          /* ldb */ nb_size,
          /* ldc */ ldc,
          /* brg */ use_brgemm);

      // move to the next index
      data_index_step(mb, MB, nb, NB);
    }

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

// [C0, C1] = A @ [B0, B1]
template <typename scalar_t>
void segment_gemm_kernel_impl(
    scalar_t* __restrict__ C0,
    scalar_t* __restrict__ C1,
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B0,
    const at::Float8_e4m3fn* __restrict__ B1,
    const float* __restrict__ Bs0,
    const float* __restrict__ Bs1,
    scalar_t* __restrict__ Btmp,
    int64_t M,
    int64_t N0,
    int64_t N1,
    int64_t K,
    int64_t block_size_N,
    int64_t block_size_K) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB0 = div_up(N0, BLOCK_N);
  const int64_t NB1 = div_up(N1, BLOCK_N);
  const int64_t NB = NB0 + NB1;

  const int64_t scale_size_K = div_up(K, block_size_K);
  const int64_t blocks_n_per_group = block_size_N / BLOCK_N;

  const bool use_brgemm = can_use_brgemm<at::Float8_e4m3fn>(M);

  // parallel on [MB, NB0 + NB1]
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    int64_t mb{0}, nb{0};
    data_index_init(begin, mb, MB, nb, NB);

    int tid = at::get_thread_num();
    // for brgemm, use float32 for accumulate
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

    for (int64_t i = begin; i < end; ++i) {
      UNUSED(i);

      int mb_start = mb * BLOCK_M;
      int mb_size = std::min(M - mb_start, BLOCK_M);
      int nb_start = nb * BLOCK_N;
      int nb_size = BLOCK_N;

      const at::Float8_e4m3fn* __restrict__ B = nb < NB0 ? B0 : B1;
      const float* __restrict__ Bs = nb < NB0 ? Bs0 : Bs1;
      scalar_t* __restrict__ C = nb < NB0 ? C0 : C1;
      int64_t ldc = nb < NB0 ? N0 : N1;
      int64_t local_nb_start = nb < NB0 ? nb_start : nb_start - N0;
      int64_t new_nb = nb < NB0 ? nb : nb - NB0;

      tinygemm_kernel<scalar_t>(
          /*   A */ A + mb_start * K,
          /*   B */ B + local_nb_start * K /* nb * BLOCK_N * K */,
          /*   C */ C + mb_start * ldc + local_nb_start,
          /* Btmp*/ Btmp + tid * BLOCK_N * K,
          /* Ctmp*/ Ctmp,
          /*Bbias*/ nullptr,
          /*  Bs */ Bs + (new_nb / blocks_n_per_group) * scale_size_K,
          /*   M */ mb_size,
          /*   N */ nb_size,
          /*   K */ K,
          /* lda */ K,
          /* ldb */ nb_size,
          /* ldc */ ldc,
          /* brg */ use_brgemm,
          /* block_size_K */ block_size_K);

      // move to the next index
      data_index_step(mb, MB, nb, NB);
    }

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

template <typename scalar_t>
inline float reduce(const scalar_t* __restrict__ x, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  fVec sum_fvec = fVec(float(0));

// no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += bVec::size()) {
    auto [x_fvec0, x_fvec1] = load_float_vec2(x + d);
    sum_fvec += x_fvec0 * x_fvec0;
    sum_fvec += x_fvec1 * x_fvec1;
  }
  return vec_reduce_sum(sum_fvec);
}

// map2 from aten functional doesn't have fast bf16->fp32 conversion
template <typename scalar_t>
inline void map2(scalar_t* y, const scalar_t* x, const scalar_t* __restrict__ w, float scale, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  fVec scale_fvec = fVec(scale);

// no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += bVec::size()) {
    auto [x_fvec0, x_fvec1] = load_float_vec2(x + d);
    auto [w_fvec0, w_fvec1] = load_float_vec2(w + d);
    x_fvec0 = x_fvec0 * scale_fvec * w_fvec0;
    x_fvec1 = x_fvec1 * scale_fvec * w_fvec1;
    bVec out_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
    out_bvec.store(y + d);
  }
}

template <typename scalar_t>
void rms_norm_kernel_impl(
    scalar_t* __restrict__ input0,
    scalar_t* __restrict__ input1,
    const scalar_t* __restrict__ weight0,
    const scalar_t* __restrict__ weight1,
    int64_t M,
    int64_t N0,
    int64_t N1,
    int64_t stride1,
    float eps = 1e-5) {
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      scalar_t* x0 = input0 + m * N0;
      scalar_t* x1 = input1 + m * stride1;
      float scale0 = reduce(x0, N0);
      float scale1 = reduce(x1, N1);
      scale0 = float(1) / std::sqrt(scale0 / N0 + eps);
      scale1 = float(1) / std::sqrt(scale1 / N1 + eps);
      map2(x0, x0, weight0, scale0, N0);
      map2(x1, x1, weight1, scale1, N1);
    }
  });
}

template <typename scalar_t>
inline void rotary(const scalar_t* input, scalar_t* out, const scalar_t* cos, const scalar_t* sin, int64_t size) {
  TORCH_CHECK(false, "rotary scalar path not implemented.");
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void rotary<at::BFloat16>(
    const at::BFloat16* input, at::BFloat16* out, const at::BFloat16* cos, const at::BFloat16* sin, int64_t size) {
  // permute indices
  const __m512i idx1 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  const __m512i idx2 = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
  const __m512i idy1 = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
  const __m512i idy2 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8);

// rotary dim is 64, just 2 iters
#pragma GCC unroll 2
  for (int64_t d = 0; d < size; d += 32) {
    int64_t d2 = d >> 1;
    // load coefs
    __m512 vcos = CVT_BF16_TO_FP32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cos + d2)));
    __m512 vsin = CVT_BF16_TO_FP32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sin + d2)));
    // load input
    __m512i a16 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input + d));
    __m512 a = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 0));
    __m512 b = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 1));
    // from [16, 2] to [2, 16]
    __m512 in1 = _mm512_mask_permutex2var_ps(a, 0xffff, idx1, b);
    __m512 in2 = _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b);
    // out1 = in1 * cos - in2 * sin;
    // out2 = in2 * cos + in1 * sin
    __m512 out1 = _mm512_sub_ps(_mm512_mul_ps(in1, vcos), _mm512_mul_ps(in2, vsin));
    __m512 out2 = _mm512_add_ps(_mm512_mul_ps(in2, vcos), _mm512_mul_ps(in1, vsin));
    // from [2, 16] to [16, 2]
    a = _mm512_mask_permutex2var_ps(out1, 0xffff, idy1, out2);
    b = _mm512_mask_permutex2var_ps(out1, 0xffff, idy2, out2);

    _mm512_storeu_si512(reinterpret_cast<__m512i*>((out + d)), (__m512i)(_mm512_cvtne2ps_pbh(b, a)));
  }
}
#endif

template <typename scalar_t>
void rotary_emb_kernel_impl(
    scalar_t* q_pe_out,
    scalar_t* k_pe_out,
    const scalar_t* q_pe,
    const scalar_t* k_pe,
    const int64_t* pos,
    const scalar_t* cos_sin,
    int64_t num_seqs,
    int64_t num_heads,
    int64_t rotary_dim,
    int64_t q_strideB,
    int64_t q_strideH,
    int64_t k_strideB,
    int64_t oq_strideB,
    int64_t oq_strideH,
    int64_t ok_strideB) {
  TORCH_CHECK(rotary_dim % 32 == 0, "rotary_dim is not 32x.");
  const int64_t rotary_offset = rotary_dim / 2;

  // parallel on [num_seqs, num_heads + 1]
  // top [num_heads] handle q_pe and bottom [1] handle k_pe
  at::parallel_for(0, num_seqs * (num_heads + 1), GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t seq{0}, head_id{0};
    data_index_init(begin, seq, num_seqs, head_id, num_heads + 1);

    for (int64_t i = begin; i < end; ++i) {
      UNUSED(i);
      // get cos and sin cache ptr
      int64_t index = pos[seq];
      const scalar_t* cos = cos_sin + index * rotary_dim;
      const scalar_t* sin = cos + rotary_offset;

      const scalar_t* input =
          (head_id < num_heads) ? q_pe + seq * q_strideB + head_id * q_strideH : k_pe + seq * k_strideB;
      scalar_t* out =
          (head_id < num_heads) ? q_pe_out + seq * oq_strideB + head_id * oq_strideH : k_pe_out + seq * ok_strideB;
      rotary<scalar_t>(input, out, cos, sin, rotary_dim);

      // move to the next index
      data_index_step(seq, num_seqs, head_id, num_heads + 1);
    }
  });
}

}  // anonymous namespace

extern at::Tensor
weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, const std::optional<at::Tensor>& bias, bool is_vnni);

extern at::Tensor int8_scaled_mm_with_quant(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

extern at::Tensor fp8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    std::vector<int64_t> block_size,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

// NB: shapes in DeepSeek R1
//
//   hidden_states    : [num_seqs, hidden_size] [1, 7168]
//   q_a_proj_weight  : [q_lora_rank, hidden_size] [1536, 7168]
//   q_b_proj_weight  : [num_heads * qk_head_dim, q_lora_rank] [4224, 1536]
//   kv_a_proj_weight : [kv_lora_rank + qk_rope_head_dim, hidden_size] [576, 7168]
//   q_a_layernorm_weight  : [q_lora_rank] [1536]
//   kv_a_layernorm_weight : [kv_lora_rank] [512]
//
// Unlike upstream SGLang, there is no `w_kc` argument here: this kernel
// returns the unabsorbed query (q_nope || q_pe, RoPE applied), and absorption
// is left to vLLM's shared `MLACommonImpl`/`bmm_cpu` path (see file header).
std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope(
    at::Tensor& hidden_states,
    at::Tensor& q_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& kv_a_proj_weight,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor> q_a_proj_scale,
    std::optional<at::Tensor> q_b_proj_scale,
    std::optional<at::Tensor> kv_a_proj_scale,
    std::optional<at::Tensor> w_scale,
    bool is_vnni,
    std::optional<std::vector<int64_t>> block_size,
    int64_t num_heads,
    int64_t qk_nope_head_dim,
    int64_t kv_lora_rank) {
  const auto st = hidden_states.scalar_type();
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(positions);
  CHECK_INPUT(cos_sin_cache);
  CHECK_EQ(q_a_layernorm_weight.scalar_type(), st);
  CHECK_EQ(kv_a_layernorm_weight.scalar_type(), st);
  CHECK_EQ(positions.scalar_type(), at::kLong);
  CHECK_EQ(cos_sin_cache.scalar_type(), st);
  CHECK_DIM(2, hidden_states);
  CHECK_DIM(1, q_a_layernorm_weight);
  CHECK_DIM(1, kv_a_layernorm_weight);
  CHECK_DIM(1, positions);
  CHECK_DIM(2, cos_sin_cache);

  // skip contiguous checks for weights, expect prepacked
  TORCH_CHECK(is_vnni, "qkv_proj_with_rope: expect weights are prepacked!");

  int64_t num_seqs = hidden_states.size(0);
  int64_t hidden_size = hidden_states.size(1);
  int64_t q_lora_rank = q_a_proj_weight.size(0);
  int64_t qk_head_dim = q_b_proj_weight.size(0) / num_heads;
  int64_t qk_rope_head_dim = qk_head_dim - qk_nope_head_dim;
  int64_t rotary_dim = cos_sin_cache.size(1);

  CHECK_EQ(positions.numel(), num_seqs);
  CHECK_EQ(rotary_dim, qk_rope_head_dim);
  CHECK_EQ(q_a_layernorm_weight.numel(), q_lora_rank);
  CHECK_EQ(kv_a_layernorm_weight.numel(), kv_lora_rank);
  CHECK_EQ(kv_a_proj_weight.size(0), kv_lora_rank + qk_rope_head_dim);

  if (use_int8_w8a8) {
    TORCH_CHECK(q_a_proj_scale.has_value(), "missing q_a_proj_scale for int8 w8a8.");
    TORCH_CHECK(q_b_proj_scale.has_value(), "missing q_b_proj_scale for int8 w8a8.");
    TORCH_CHECK(kv_a_proj_scale.has_value(), "missing kv_a_proj_scale for int8 w8a8.");
  }
  if (use_fp8_w8a16) {
    TORCH_CHECK(q_a_proj_scale.has_value(), "missing q_a_proj_scale for fp8 w8a16.");
    TORCH_CHECK(q_b_proj_scale.has_value(), "missing q_b_proj_scale for fp8 w8a16.");
    TORCH_CHECK(kv_a_proj_scale.has_value(), "missing kv_a_proj_scale for fp8 w8a16.");
    TORCH_CHECK(block_size.has_value(), "missing block_size for fp8 w8a16.");
    TORCH_CHECK(block_size.value().size() == 2, "block_size should be 2D for fp8 w8a16.");
  }
  // outputs and temp buffer
  const auto options = hidden_states.options();
  // k_input/v_input share one 576-wide latent buffer (v_input is a view of
  // k_input's first `kv_lora_rank` columns) -- same layout as
  // MLACommonBackend.get_kv_cache_shape's per-token row, so the cache-write
  // op can consume this directly.
  auto k_input = at::empty({num_seqs, 1, kv_lora_rank + qk_rope_head_dim}, options);
  auto v_input = k_input.narrow(-1, 0, kv_lora_rank);

  // outputs of q_a_proj and q_b_proj
  auto qa = at::empty({num_seqs, q_lora_rank}, options);

  // stage 1: q_a_proj and kv_a_proj
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "qkv_proj_kernel_impl", [&] {
    if (use_int8_w8a8) {
      auto q_a_proj_s = q_a_proj_scale.value();
      auto kv_a_proj_s = kv_a_proj_scale.value();
      TORCH_CHECK(q_a_proj_s.numel() == q_lora_rank);
      TORCH_CHECK(kv_a_proj_s.numel() == kv_lora_rank + qk_rope_head_dim);

      auto buffer = at::empty({num_seqs * hidden_size + num_seqs * 4}, options.dtype(at::kByte));
      uint8_t* __restrict__ Aq_data = buffer.data_ptr<uint8_t>();
      float* __restrict__ As_data = (float*)((void*)(Aq_data + num_seqs * hidden_size));
      const scalar_t* __restrict__ A_data = hidden_states.data_ptr<scalar_t>();

      at::parallel_for(0, num_seqs, 0, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
          quantize_row_int8<scalar_t>(Aq_data + m * hidden_size, As_data[m], A_data + m * hidden_size, hidden_size);
        }
      });

      segment_gemm_kernel_impl<scalar_t>(
          qa.data_ptr<scalar_t>(),
          k_input.data_ptr<scalar_t>(),
          Aq_data,
          q_a_proj_weight.data_ptr<int8_t>(),
          kv_a_proj_weight.data_ptr<int8_t>(),
          As_data,
          q_a_proj_s.data_ptr<float>(),
          kv_a_proj_s.data_ptr<float>(),
          num_seqs,
          q_lora_rank,
          kv_lora_rank + qk_rope_head_dim,
          hidden_size);
    } else if (use_fp8_w8a16) {
      int64_t block_size_N = block_size.value()[0];
      int64_t block_size_K = block_size.value()[1];
      auto q_a_proj_s = q_a_proj_scale.value();
      auto kv_a_proj_s = kv_a_proj_scale.value();
      CHECK_EQ(q_a_proj_s.size(0), div_up(q_lora_rank, block_size_N));
      CHECK_EQ(q_a_proj_s.size(1), div_up(hidden_size, block_size_K));
      CHECK_EQ(kv_a_proj_s.size(0), div_up(kv_lora_rank + qk_rope_head_dim, block_size_N));
      CHECK_EQ(kv_a_proj_s.size(1), div_up(hidden_size, block_size_K));

      const int BLOCK_N = block_size_n();
      const int num_threads = at::get_num_threads();
      auto buffer = at::empty({num_threads, BLOCK_N * hidden_size}, options);
      segment_gemm_kernel_impl<scalar_t>(
          qa.data_ptr<scalar_t>(),
          k_input.data_ptr<scalar_t>(),
          hidden_states.data_ptr<scalar_t>(),
          q_a_proj_weight.data_ptr<at::Float8_e4m3fn>(),
          kv_a_proj_weight.data_ptr<at::Float8_e4m3fn>(),
          q_a_proj_s.data_ptr<float>(),
          kv_a_proj_s.data_ptr<float>(),
          buffer.data_ptr<scalar_t>(),
          num_seqs,
          q_lora_rank,
          kv_lora_rank + qk_rope_head_dim,
          hidden_size,
          block_size_N,
          block_size_K);
    } else {
      segment_gemm_kernel_impl<scalar_t>(
          qa.data_ptr<scalar_t>(),
          k_input.data_ptr<scalar_t>(),
          hidden_states.data_ptr<scalar_t>(),
          q_a_proj_weight.data_ptr<scalar_t>(),
          kv_a_proj_weight.data_ptr<scalar_t>(),
          num_seqs,
          q_lora_rank,
          kv_lora_rank + qk_rope_head_dim,
          hidden_size);
    }
  });

  // stage 2: apply rmsnorm inplace
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "rms_norm_kernel_impl", [&] {
    rms_norm_kernel_impl<scalar_t>(
        qa.data_ptr<scalar_t>(),
        v_input.data_ptr<scalar_t>(),
        q_a_layernorm_weight.data_ptr<scalar_t>(),
        kv_a_layernorm_weight.data_ptr<scalar_t>(),
        num_seqs,
        q_lora_rank,
        kv_lora_rank,
        kv_lora_rank + qk_rope_head_dim,
        eps);
  });

  // stage 3: q_b_proj -- qb is already [num_seqs, num_heads, qk_head_dim],
  // exactly the unabsorbed query layout we return (no separate q_input
  // allocation needed, unlike upstream).
  at::Tensor qb;
  std::optional<at::Tensor> bias;
  if (use_int8_w8a8) {
    qb = int8_scaled_mm_with_quant(qa, q_b_proj_weight, q_b_proj_scale.value(), bias, at::kBFloat16, is_vnni);
  } else if (use_fp8_w8a16) {
    qb = fp8_scaled_mm_cpu(
        qa, q_b_proj_weight, q_b_proj_scale.value(), block_size.value(), bias, at::kBFloat16, is_vnni);
  } else {
    qb = weight_packed_linear(qa, q_b_proj_weight, bias, is_vnni);
  }
  qb.as_strided_({num_seqs, num_heads, qk_head_dim}, {num_heads * qk_head_dim, qk_head_dim, 1});

  // stage 4: rope, applied in place onto qb's q_pe slice (mirrors how k_pe
  // is already rope'd in place below with input == output pointers).
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "rotary_emb_kernel_impl", [&] {
    rotary_emb_kernel_impl<scalar_t>(
        qb.data_ptr<scalar_t>() + qk_nope_head_dim,
        k_input.data_ptr<scalar_t>() + kv_lora_rank,
        qb.data_ptr<scalar_t>() + qk_nope_head_dim,
        k_input.data_ptr<scalar_t>() + kv_lora_rank,
        positions.data_ptr<int64_t>(),
        cos_sin_cache.data_ptr<scalar_t>(),
        num_seqs,
        num_heads,
        rotary_dim,
        num_heads * qk_head_dim,
        qk_head_dim,
        kv_lora_rank + qk_rope_head_dim,
        num_heads * qk_head_dim,
        qk_head_dim,
        kv_lora_rank + qk_rope_head_dim);
  });

  return std::make_tuple(qb, k_input, v_input);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope_fused_weight(
    at::Tensor& hidden_states,
    at::Tensor& qkv_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor> qkv_a_proj_scale,
    std::optional<at::Tensor> q_b_proj_scale,
    std::optional<at::Tensor> w_scale,
    bool is_vnni,
    std::optional<std::vector<int64_t>> block_size,
    int64_t q_lora_rank,
    int64_t kv_lora_rank,
    int64_t qk_rope_head_dim,
    int64_t num_heads,
    int64_t qk_nope_head_dim) {
  int64_t hidden_size = hidden_states.size(1);
  CHECK_EQ(qkv_a_proj_weight.size(0), q_lora_rank + kv_lora_rank + qk_rope_head_dim);
  CHECK_EQ(qkv_a_proj_weight.size(1), get_row_size(hidden_size, use_int8_w8a8));

  std::vector<at::Tensor> weight_chunks =
      at::split(qkv_a_proj_weight, {q_lora_rank, kv_lora_rank + qk_rope_head_dim}, 0);
  at::Tensor q_a_proj_weight = weight_chunks[0];
  at::Tensor kv_a_proj_weight = weight_chunks[1];
  at::Tensor q_a_proj_s;
  at::Tensor kv_a_proj_s;

  if (use_int8_w8a8) {
    TORCH_CHECK(qkv_a_proj_scale.has_value(), "missing qkv_a_proj_scale for int8 w8a8.");
    std::vector<at::Tensor> scale_chunks =
        at::split(qkv_a_proj_scale.value(), {q_lora_rank, kv_lora_rank + qk_rope_head_dim}, 0);
    q_a_proj_s = scale_chunks[0];
    kv_a_proj_s = scale_chunks[1];
  }
  if (use_fp8_w8a16) {
    TORCH_CHECK(qkv_a_proj_scale.has_value(), "missing qkv_a_proj_scale for fp8 w8a16.");
    int64_t block_size_N = block_size.value()[0];
    int64_t q_a_proj_s_dim0 = div_up(q_lora_rank, block_size_N);
    int64_t kv_a_proj_s_dim0 = div_up(kv_lora_rank + qk_rope_head_dim, block_size_N);
    std::vector<at::Tensor> scale_chunks = at::split(qkv_a_proj_scale.value(), {q_a_proj_s_dim0, kv_a_proj_s_dim0}, 0);
    q_a_proj_s = scale_chunks[0];
    kv_a_proj_s = scale_chunks[1];
  }

  return qkv_proj_with_rope(
      hidden_states,
      q_a_proj_weight,
      q_b_proj_weight,
      kv_a_proj_weight,
      q_a_layernorm_weight,
      kv_a_layernorm_weight,
      positions,
      cos_sin_cache,
      eps,
      use_int8_w8a8,
      use_fp8_w8a16,
      q_a_proj_s,
      q_b_proj_scale,
      kv_a_proj_s,
      w_scale,
      is_vnni,
      block_size,
      num_heads,
      qk_nope_head_dim,
      kv_lora_rank);
}
