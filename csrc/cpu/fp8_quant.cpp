#include <torch/all.h>

#include <algorithm>
#include <cmath>

#if defined(__AVX512F__)
  #include <immintrin.h>
#endif

namespace {

template <typename fp8_t>
constexpr float fp8_max() {
  return std::is_same_v<fp8_t, c10::Float8_e5m2> ? 57344.0f : 448.0f;
}

template <typename fp8_t>
inline fp8_t to_fp8(float x) {
  constexpr float max = fp8_max<fp8_t>();
  return static_cast<fp8_t>(std::min(std::max(x, -max), max));
}

#if defined(__AVX512F__)
inline __m512 load_as_fp32(const float* p) { return _mm512_loadu_ps(p); }

inline __m512 load_as_fp32(const c10::BFloat16* p) {
  const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
}

inline __m512 load_as_fp32(const c10::Half* p) {
  return _mm512_cvtph_ps(
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}

// Bit-exact vector form of c10::Float8_e4m3fn(float) (round-to-nearest-even)
// for inputs already clamped to [-448, 448] or NaN.
inline __m128i fp32_to_e4m3fn(__m512 x) {
  const __m512i bits = _mm512_castps_si512(x);
  const __m512i sign =
      _mm512_and_si512(bits, _mm512_set1_epi32(static_cast<int>(0x80000000u)));
  const __m512i abs_bits = _mm512_xor_si512(bits, sign);

  const __mmask16 nan_mask =
      _mm512_cmpge_epu32_mask(abs_bits, _mm512_set1_epi32(1087 << 20));
  const __mmask16 denorm_mask =
      _mm512_cmplt_epu32_mask(abs_bits, _mm512_set1_epi32(121 << 23));

  const __m512i denorm_offset = _mm512_set1_epi32(141 << 23);
  const __m512i denorm = _mm512_sub_epi32(
      _mm512_castps_si512(_mm512_add_ps(_mm512_castsi512_ps(abs_bits),
                                        _mm512_castsi512_ps(denorm_offset))),
      denorm_offset);

  const __m512i mant_odd =
      _mm512_and_si512(_mm512_srli_epi32(abs_bits, 20), _mm512_set1_epi32(1));
  const __m512i normal = _mm512_srli_epi32(
      _mm512_add_epi32(
          _mm512_add_epi32(abs_bits, _mm512_set1_epi32(static_cast<int>(
                                         (uint32_t(7 - 127) << 23) + 0x7FFFF))),
          mant_odd),
      20);

  __m512i res = _mm512_mask_blend_epi32(denorm_mask, normal, denorm);
  res = _mm512_mask_mov_epi32(res, nan_mask, _mm512_set1_epi32(0x7F));
  res = _mm512_or_si512(res, _mm512_srli_epi32(sign, 24));
  return _mm512_cvtepi32_epi8(res);
}

inline __m512 clamp_fp8_range(__m512 x, float max) {
  // Operand order propagates NaN, matching the scalar path.
  return _mm512_min_ps(_mm512_set1_ps(max),
                       _mm512_max_ps(_mm512_set1_ps(-max), x));
}
#endif

// out[i] = fp8(in[i] * scale) when kDivide is false, fp8(in[i] / scale)
// otherwise.
template <bool kDivide, typename scalar_t, typename fp8_t>
inline void quant_row(fp8_t* __restrict__ out, const scalar_t* __restrict__ in,
                      int64_t n, float scale) {
  int64_t i = 0;
#if defined(__AVX512F__)
  if constexpr (std::is_same_v<fp8_t, c10::Float8_e4m3fn> &&
                !std::is_same_v<scalar_t, double>) {
    const __m512 vscale = _mm512_set1_ps(scale);
    for (; i + 16 <= n; i += 16) {
      const __m512 v = load_as_fp32(in + i);
      const __m512 scaled =
          kDivide ? _mm512_div_ps(v, vscale) : _mm512_mul_ps(v, vscale);
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(out + i),
          fp32_to_e4m3fn(clamp_fp8_range(scaled, fp8_max<fp8_t>())));
    }
  }
#endif
  for (; i < n; ++i) {
    const float v = static_cast<float>(in[i]);
    out[i] = to_fp8<fp8_t>(kDivide ? v / scale : v * scale);
  }
}

template <typename scalar_t>
inline float row_absmax(const scalar_t* __restrict__ in, int64_t n) {
  int64_t i = 0;
  float absmax = 0.0f;
#if defined(__AVX512F__)
  if constexpr (!std::is_same_v<scalar_t, double>) {
    __m512 vmax = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
      // Operand order skips NaN, matching fmaxf in the CUDA kernel.
      vmax = _mm512_max_ps(_mm512_abs_ps(load_as_fp32(in + i)), vmax);
    }
    absmax = _mm512_reduce_max_ps(vmax);
  }
#endif
  for (; i < n; ++i) {
    absmax = std::max(absmax, std::fabs(static_cast<float>(in[i])));
  }
  return absmax;
}

template <typename scalar_t, typename fp8_t>
void static_quant_impl(fp8_t* __restrict__ out, const scalar_t* __restrict__ in,
                       const float* __restrict__ scale, int64_t num_tokens,
                       int64_t hidden_size, int64_t in_row_stride,
                       int64_t out_row_stride, int64_t group_m, int64_t group_n,
                       int64_t scale_stride_i, int64_t scale_stride_j) {
#pragma omp parallel for
  for (int64_t m = 0; m < num_tokens; ++m) {
    const scalar_t* row_in = in + m * in_row_stride;
    fp8_t* row_out = out + m * out_row_stride;
    const float* row_scale = scale + (m / group_m) * scale_stride_i;
    if (scale_stride_j == 0) {
      quant_row<false>(row_out, row_in, hidden_size, 1.0f / row_scale[0]);
    } else {
      for (int64_t n = 0; n < hidden_size; n += group_n) {
        quant_row<false>(row_out + n, row_in + n, group_n,
                         1.0f / row_scale[(n / group_n) * scale_stride_j]);
      }
    }
  }
}

template <typename scalar_t, typename fp8_t>
void dynamic_per_token_quant_impl(fp8_t* __restrict__ out,
                                  float* __restrict__ scale,
                                  const scalar_t* __restrict__ in,
                                  const float* scale_ub, int64_t num_tokens,
                                  int64_t hidden_size, int64_t in_row_stride,
                                  int64_t out_row_stride) {
  constexpr float min_scale = 1.0f / (fp8_max<fp8_t>() * 512.0f);
#pragma omp parallel for
  for (int64_t m = 0; m < num_tokens; ++m) {
    const scalar_t* row_in = in + m * in_row_stride;
    fp8_t* row_out = out + m * out_row_stride;
    float absmax = row_absmax(row_in, hidden_size);
    if (scale_ub != nullptr) {
      absmax = std::min(absmax, *scale_ub);
    }
    const float token_scale = std::max(absmax / fp8_max<fp8_t>(), min_scale);
    scale[m] = token_scale;
    quant_row<true>(row_out, row_in, hidden_size, token_scale);
  }
}

#define DISPATCH_FP8_QUANT(OUT_DTYPE, IN_DTYPE, NAME, ...)                   \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                           \
      at::ScalarType::BFloat16, at::ScalarType::Half, IN_DTYPE, NAME, [&] {  \
        switch (OUT_DTYPE) {                                                 \
          case at::ScalarType::Float8_e4m3fn: {                              \
            using fp8_t = c10::Float8_e4m3fn;                                \
            __VA_ARGS__();                                                   \
            break;                                                           \
          }                                                                  \
          case at::ScalarType::Float8_e5m2: {                                \
            using fp8_t = c10::Float8_e5m2;                                  \
            __VA_ARGS__();                                                   \
            break;                                                           \
          }                                                                  \
          default:                                                           \
            TORCH_CHECK(false, NAME, ": unsupported fp8 dtype ", OUT_DTYPE); \
        }                                                                    \
      })

void check_fp8_quant_args(const torch::Tensor& out,
                          const torch::Tensor& input) {
  TORCH_CHECK(input.dim() >= 2 && input.is_cpu() && out.is_cpu(),
              "fp8 quant expects CPU tensors with at least 2 dims");
  TORCH_CHECK(input.stride(-1) == 1 && out.stride(-1) == 1,
              "last dimension of input and output must be contiguous");
  TORCH_CHECK(out.size(-1) == input.size(-1) &&
                  out.numel() / out.size(-1) >= input.numel() / input.size(-1),
              "output must match input hidden size and hold all tokens");
}

}  // namespace

void static_scaled_fp8_quant(
    torch::Tensor& out, const torch::Tensor& input, const torch::Tensor& scale,
    const std::optional<std::vector<int64_t>>& opt_group_shape) {
  check_fp8_quant_args(out, input);
  TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");

  const int64_t hidden_size = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_size;
  int64_t group_m = num_tokens;
  int64_t group_n = hidden_size;
  int64_t scale_stride_i = 0;
  int64_t scale_stride_j = 0;

  auto resolve_group_shape = [&]() {
    TORCH_CHECK(opt_group_shape.has_value() && opt_group_shape->size() == 2,
                "group_shape with exactly 2 elements is required");
    group_m = (*opt_group_shape)[0] == -1 ? num_tokens : (*opt_group_shape)[0];
    group_n = (*opt_group_shape)[1] == -1 ? hidden_size : (*opt_group_shape)[1];
  };

  if (scale.dim() == 0 || (scale.dim() == 1 && scale.numel() == 1)) {
    // per-tensor
  } else if (scale.dim() == 1) {
    resolve_group_shape();
    const int64_t scale_m = num_tokens / group_m;
    const int64_t scale_n = hidden_size / group_n;
    TORCH_CHECK(num_tokens % group_m == 0 && hidden_size % group_n == 0,
                "input shape must be divisible by group_shape");
    TORCH_CHECK(scale.numel() == scale_m * scale_n,
                "1D scale length does not match group_shape");
    TORCH_CHECK(scale_m == 1 || scale_n == 1,
                "1D scale requires one trivial scale dimension; use a 2D "
                "scale for group scaling");
    if (scale_m == 1) {
      scale_stride_j = scale.stride(0);
    } else {
      scale_stride_i = scale.stride(0);
    }
  } else if (scale.dim() == 2) {
    TORCH_CHECK(
        num_tokens % scale.size(0) == 0 && hidden_size % scale.size(1) == 0,
        "input shape must be divisible by 2D scale shape");
    group_m = num_tokens / scale.size(0);
    group_n = hidden_size / scale.size(1);
    scale_stride_i = scale.stride(0);
    scale_stride_j = scale.stride(1);
  } else {
    TORCH_CHECK(false, "scale must be 0D, 1D, or 2D, got ", scale.dim(), "D");
  }

  DISPATCH_FP8_QUANT(
      out.scalar_type(), input.scalar_type(), "static_scaled_fp8_quant", [&] {
        static_quant_impl<scalar_t, fp8_t>(
            out.data_ptr<fp8_t>(), input.const_data_ptr<scalar_t>(),
            scale.const_data_ptr<float>(), num_tokens, hidden_size,
            input.stride(-2), out.stride(-2), group_m, group_n, scale_stride_i,
            scale_stride_j);
      });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, const torch::Tensor& input, torch::Tensor& scale,
    const std::optional<torch::Tensor>& scale_ub) {
  check_fp8_quant_args(out, input);
  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.is_contiguous(),
              "scale must be a contiguous float32 tensor");

  const int64_t hidden_size = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_size;
  TORCH_CHECK(scale.numel() >= num_tokens,
              "scale must hold one element per token");
  const float* scale_ub_ptr = nullptr;
  if (scale_ub.has_value()) {
    TORCH_CHECK(scale_ub->scalar_type() == at::kFloat && scale_ub->numel() == 1,
                "scale_ub must be a single float32 element");
    scale_ub_ptr = scale_ub->const_data_ptr<float>();
  }

  DISPATCH_FP8_QUANT(out.scalar_type(), input.scalar_type(),
                     "dynamic_per_token_scaled_fp8_quant", [&] {
                       dynamic_per_token_quant_impl<scalar_t, fp8_t>(
                           out.data_ptr<fp8_t>(), scale.data_ptr<float>(),
                           input.const_data_ptr<scalar_t>(), scale_ub_ptr,
                           num_tokens, hidden_size, input.stride(-2),
                           out.stride(-2));
                     });
}
