// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include <ATen/native/CPUBlas.h>
#include <c10/util/Unroll.h>
#include <torch/all.h>

#include "common.h"
#include "vec.h"

namespace {

#define BLOCK_N 32

#define PER_TENSOR 1
#define PER_ROW 2
#define PER_GROUP 3

static bool cpublas_checked = false;
static bool cpublas_can_pack = false;

bool cpublas_could_pack() {
  // the could_pack check requires AMX support implicitly
  if (cpublas_checked) {
    return cpublas_can_pack;
  }
#ifdef CPUBLAS_BRGEMM_F8F8BF16
  std::cout << "Using F8F8 packing..." << std::endl;
  cpublas_can_pack = at::native::cpublas::could_pack(at::kFloat8_e4m3fn);
#else
  cpublas_can_pack = at::native::cpublas::could_pack(at::kBFloat16);
#endif
  cpublas_checked = true;
  return cpublas_can_pack;
}

#if defined(CPU_CAPABILITY_AVX512)
static void cvt_f8e4m3_to_bf16(
    const at::Float8_e4m3fn* __restrict__ in, at::BFloat16* out, int64_t rows, int64_t cols, int64_t stride) {
  if (stride == cols) {
    // A contiguous buffer
    size_t len = rows * cols;
    size_t i = 0;
    for (; i < len; i += 32) {
      __m256i fp8_vec = _mm256_loadu_si256((__m256i*)&in[i]);
      __m512bh bf16_vec = cvt_e4m3_bf16_intrinsic_no_nan(fp8_vec);
      _mm512_storeu_si512((__m512i*)(out + i), (__m512i)bf16_vec);
    }
    for (; i < len; ++i) {
      out[i] = (at::BFloat16)in[i];
    }
  } else {
    // Non-contiguous. Access each row with stride
    TORCH_CHECK(stride > cols);
    for (int r = 0; r < rows; ++r) {
      size_t i = 0;
      size_t vec_len = cols / 32 * 32;
      for (; i < vec_len; i += 32) {
        __m256i fp8_vec = _mm256_loadu_si256((__m256i*)&in[r * stride + i]);
        __m512bh bf16_vec = cvt_e4m3_bf16_intrinsic_no_nan(fp8_vec);
        _mm512_storeu_si512((__m512i*)(out + r * cols + i), (__m512i)bf16_vec);
      }
      for (; i < cols; ++i) {
        out[r * cols + i] = (at::BFloat16)in[r * stride + i];
      }
    }
  }
}

// accumulate and store result to buffer
// if act/wei are per_group quantized, apply scales
template <bool accum, int64_t N, int act_quant_mode, int wei_quant_mode>
static void _accumulate_result(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int M,
    int ldi,
    int ldo,
    int ldsa = 1) {
  float a_scale, b_scale;
  __m512 va_scale;
  __m512 vb_scale;
  for (int m = 0; m < M; ++m) {
    if constexpr (act_quant_mode == PER_GROUP) {
      a_scale = *(scale_a + m * ldsa);
      va_scale = _mm512_set1_ps(a_scale);
    }
    constexpr int N_UNROLL = N / 16;
    c10::ForcedUnroll<N_UNROLL>{}([&](auto i) {
      constexpr int n = i * 16;
      __m512 vc_f = _mm512_loadu_ps(input + m * ldi + n);
      if constexpr (act_quant_mode == PER_GROUP) {
        vc_f = _mm512_mul_ps(vc_f, va_scale);
      }
      if constexpr (wei_quant_mode == PER_GROUP) {
        vb_scale = _mm512_loadu_ps(scale_b + n);
        vc_f = _mm512_mul_ps(vc_f, vb_scale);
      }
      if constexpr (accum) {
        __m512 vo = _mm512_loadu_ps(output + m * ldo + n);
        _mm512_storeu_ps(output + m * ldo + n, _mm512_add_ps(vo, vc_f));
      } else {
        _mm512_storeu_ps(output + m * ldo + n, vc_f);
      }
    });
    constexpr int tail_start = N / 16 * 16;
    for (int n = tail_start; n < N; ++n) {
      float dq_val = input[m * ldi + n];
      if constexpr (act_quant_mode == PER_GROUP) {
        dq_val = dq_val * a_scale;
      }
      if constexpr (wei_quant_mode == PER_GROUP) {
        b_scale = scale_b[n];
        dq_val = dq_val * b_scale;
      }
      if constexpr (accum) {
        output[m * ldo + n] += dq_val;
      } else {
        output[m * ldo + n] = dq_val;
      }
    }
  }
}

// Store result to output buffer with dtype conversion
// If act/wei are per_row or per_tensor quantized, apply scales
// If bias is not null, add bias
template <typename out_dtype, int64_t N, int act_quant_mode, int wei_quant_mode>
inline void store_out(
    const float* y_buf,
    out_dtype* c_ptr,
    int64_t M,
    int64_t lda,
    const float* scales_a,
    const float* scales_b,
    const float* bias) {
  float a_scale = 1.0, b_scale = 1.0;
  __m512 va_scale, vb_scale;
  if constexpr (act_quant_mode == PER_TENSOR) {
    a_scale = *scales_a;
  }
  if constexpr (wei_quant_mode == PER_TENSOR) {
    b_scale = *scales_b;
    vb_scale = _mm512_set1_ps(b_scale);
  }
  for (int i = 0; i < M; ++i) {
    if constexpr (act_quant_mode == PER_ROW) {
      a_scale = *(scales_a + i);
    }
    if constexpr (act_quant_mode != PER_GROUP) {
      va_scale = _mm512_set1_ps(a_scale);
    }
    constexpr int N_UNROLL = N / 16;
    c10::ForcedUnroll<N_UNROLL>{}([&](auto idx) {
      constexpr int j = idx * 16;
      __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
      __m512 bias_vec = bias ? _mm512_loadu_ps(bias + j) : _mm512_setzero_ps();
      if constexpr (act_quant_mode != PER_GROUP) {
        y_vec = _mm512_mul_ps(y_vec, va_scale);
      }
      if constexpr (wei_quant_mode == PER_ROW) {
        vb_scale = _mm512_loadu_ps(scales_b + j);
      }
      if constexpr (wei_quant_mode != PER_GROUP) {
        y_vec = _mm512_mul_ps(y_vec, vb_scale);
      }
      y_vec = _mm512_add_ps(y_vec, bias_vec);
      if constexpr (std::is_same<out_dtype, float>::value) {
        _mm512_storeu_ps(c_ptr + i * lda + j, y_vec);
      } else if constexpr (std::is_same<out_dtype, at::BFloat16>::value) {
        __m256i y_bf16_vec = at::vec::cvtfp32_bf16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_bf16_vec);
      } else if constexpr (std::is_same<out_dtype, at::Half>::value) {
        __m256i y_fp16_vec = at::vec::cvtfp32_fp16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_fp16_vec);
      } else {
        TORCH_CHECK(false, "Unsupported output dtype");
      }
    });
    constexpr int tail_start = N / 16 * 16;
    for (int j = tail_start; j < N; ++j) {
      if constexpr (wei_quant_mode == PER_ROW) {
        b_scale = scales_b[j];
      }
      c_ptr[i * lda + j] = static_cast<out_dtype>(y_buf[i * N + j] * a_scale * b_scale);
    }
  }  // for M
}

#else  // no AVX512

static void cvt_f8e4m3_to_bf16(
    const at::Float8_e4m3fn* __restrict__ in, at::BFloat16* out, int64_t rows, int64_t cols, int64_t stride) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out[r * cols + c] = (at::BFloat16)in[r * stride + c];
    }
  }
}

// Store result to output buffer with dtype conversion
// If act/wei are per_row or per_tensor quantized, apply scales
// If bias is not null, add bias
template <typename out_dtype, int64_t N, int act_quant_mode, int wei_quant_mode>
inline void store_out(
    const float* y_buf,
    out_dtype* c_ptr,
    int64_t M,
    int64_t lda,
    const float* scales_a,
    const float* scales_b,
    const float* bias) {
  float a_scale = 1.0, b_scale = 1.0;
  if constexpr (act_quant_mode == PER_TENSOR) {
    a_scale = *scales_a;
  }
  if constexpr (wei_quant_mode == PER_TENSOR) {
    b_scale = *scales_b;
  }
  for (int i = 0; i < M; ++i) {
    if constexpr (act_quant_mode == PER_ROW) {
      a_scale = *(scales_a + i);
    }
    for (int j = 0; j < N; ++j) {
      if constexpr (wei_quant_mode == PER_ROW) {
        b_scale = scales_b[j];
      }
      c_ptr[i * lda + j] = static_cast<out_dtype>(y_buf[i * N + j] * a_scale * b_scale);
    }
  }  // for M
}

#endif  // CPU_CAPABILITY_AVX512

template <bool cpublas_can_pack, int64_t N, int act_quant_mode, int wei_quant_mode>
void _micro_gemm(
    float* C,
    const at::Float8_e4m3fn* A,
    const float* scales_a,
    const at::Float8_e4m3fn* B,
    const float* scales_b,
    int64_t M,
    int64_t K,
    int64_t lda,
    int64_t ldc,
    int64_t ldsa,
    float* ukernel_buf,
    at::BFloat16* dqA_buf,
    at::BFloat16* dqB_buf) {
  // If FP8 brgemm is not available, convert A/B to bf16 for computation
  // Compute GEMM fp8 * fp8 -> fp32 (or bf16 * bf16 -> fp32)
  // If per_group quant, apply scales. Otherwise, don't apply scales here
  // Finally accumulate and store results
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (cpublas_can_pack) {
#ifdef CPUBLAS_BRGEMM_F8F8F32
    at::native::cpublas::brgemm(
        M, N, K, lda /*lda*/, N /*ldb*/, N /*ldc*/, false /* add_C */, A, B, ukernel_buf, true /* is_vnni */);
#else
    cvt_f8e4m3_to_bf16(A, dqA_buf, M, K, lda);
    cvt_f8e4m3_to_bf16(B, dqB_buf, K, N, N);
    at::native::cpublas::brgemm(
        M, N, K, K /*lda*/, N /*ldb*/, N /*ldc*/, false /* add_C */, dqA_buf, dqB_buf, ukernel_buf, true /* is_vnni */);
#endif
    _mm_prefetch(B + N * (K + 128), _MM_HINT_T0);
    _mm_prefetch(A + K + 128, _MM_HINT_T0);
    _accumulate_result<true, N, act_quant_mode, wei_quant_mode>(
        C, ukernel_buf, scales_a, scales_b, M, N /*ldi*/, ldc, ldsa);
  } else
#endif
  {
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        float sum = 0;
        for (int64_t k = 0; k < K; ++k) {
          sum += ((float)A[i * lda + k] * (float)B[k * N + j]);
        }
        if constexpr (act_quant_mode == PER_GROUP) {
          sum *= scales_a[i * ldsa];
        }
        if constexpr (wei_quant_mode == PER_GROUP) {
          sum *= scales_b[j];
        }
        C[i * ldc + j] += sum;
      }
    }
  }
}

template <typename out_dtype, bool cpublas_can_pack, int act_quant_mode, int wei_quant_mode>
void _float8_linear_impl(
    const at::Tensor& input,
    const at::Tensor& input_scales,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const std::optional<at::Tensor>& bias,
    at::Tensor& output) {
  // input shape = [..., K]
  // input is per token quantized
  int64_t K = input.size(-1);
  auto input_view = input.view({-1, K});
  int64_t M = input_view.size(0);

  // weight shape = [Nc, Kc, block_k, block_n]
  // scales shape = [Nc, G, block_n]
  int64_t Nc = weight.size(0);
  int64_t Kc = wei_quant_mode != PER_GROUP ? 1 : weight.size(1);
  int64_t block_k = wei_quant_mode != PER_GROUP ? weight.size(1) * weight.size(2) : weight.size(2);
  constexpr int64_t block_n = BLOCK_N;
  TORCH_CHECK(weight.size(3) == block_n, "Float8 linear: unexpected weight shape");
  int64_t N = Nc * block_n;
  TORCH_CHECK(K == Kc * block_k, "Float8 linear: weight and input shapes mismatch");
  auto [parallel_on_M, block_m, Mc, Mc_parallel] = get_m_blocking(M);

  // scales shape = [Nc, G, block_n]
  int64_t num_groups = wei_quant_mode == PER_TENSOR ? 1 : weight_scales.size(1);
  TORCH_CHECK(K % num_groups == 0, "K should be divisible by num_groups");
  int64_t group_size = K / num_groups;
  TORCH_CHECK(group_size % block_k == 0, "Float8 linear: group_size should be divisible by block_k");
  int64_t block_per_group = group_size / block_k;
  TORCH_CHECK(
      input_scales.numel() == 1 || input_scales.numel() == M || input_scales.numel() == M * num_groups,
      "Float8 linear: unexpected input scales shape");
  auto ldsa = act_quant_mode == PER_TENSOR ? 0 : act_quant_mode == PER_ROW ? 1 : num_groups;

  const at::Float8_e4m3fn* a_ptr = input_view.data_ptr<at::Float8_e4m3fn>();
  const float* a_scales_ptr = input_scales.data_ptr<float>();
  const at::Float8_e4m3fn* b_ptr = weight.data_ptr<at::Float8_e4m3fn>();
  const float* b_scales_ptr = weight_scales.data_ptr<float>();
  out_dtype* c_ptr = output.data_ptr<out_dtype>();
  const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

  int64_t block_size = block_m * block_n;
  int64_t num_thread = at::get_num_threads();
  at::Tensor y_buffer = at::empty({num_thread, block_size}, output.options().dtype(at::kFloat));
  // Create buffer for brgemm output and dqA/dqB (optional)
#if defined(CPU_CAPABILITY_AVX512)
  // buffer for brgemm output in float32
  int64_t buffer_size = block_size * 2;  // float32 = bfloat16 * 2
#ifndef CPUBLAS_BRGEMM_F8F8F32
  // buffers for dqA & dqB in bf16
  buffer_size += (block_k * block_n + block_m * block_k);
#endif
  at::Tensor micro_gemm_buffer = at::empty({num_thread, buffer_size}, output.options().dtype(at::kBFloat16));
#endif

  parallel_2d(Mc, Nc, [&](int64_t mc0, int64_t mc1, int64_t nc0, int64_t nc1) {
    int tid = get_thread_num();
    float* y_buf = y_buffer.data_ptr<float>() + tid * block_size;
    at::BFloat16 *dqA_buffer = nullptr, *dqB_buffer = nullptr;
    float* ukernel_buf = nullptr;
#if defined(CPU_CAPABILITY_AVX512)
    at::BFloat16* micro_gemm_buf = micro_gemm_buffer.data_ptr<at::BFloat16>() + tid * buffer_size;
    ukernel_buf = reinterpret_cast<float*>(micro_gemm_buf);
#ifndef CPUBLAS_BRGEMM_F8F8F32
    dqA_buffer = micro_gemm_buf;
    dqB_buffer = micro_gemm_buf + block_m * block_k;
    ukernel_buf = reinterpret_cast<float*>(micro_gemm_buf + block_m * block_k + block_k * block_n);
#endif
#endif

    loop_2d<at::Float8_e4m3fn>(mc0, mc1, nc0, nc1, block_n * K, [&](int64_t mci, int64_t nc, int64_t) {
      int64_t m_size = mci * block_m + block_m > M ? M - mci * block_m : block_m;
      zero_buffer(y_buf, m_size * block_n);
      for (int kci = 0; kci < Kc; ++kci) {
        auto scales_a = a_scales_ptr + mci * block_m * num_groups + kci / block_per_group;
        auto scales_b = b_scales_ptr + nc * block_n * num_groups + kci / block_per_group * block_n;
        _micro_gemm<cpublas_can_pack, block_n, act_quant_mode, wei_quant_mode>(
            /* C */ y_buf,
            /* A */ a_ptr + mci * block_m * K + kci * block_k,
            /* scales_a */ scales_a,
            /* B */ b_ptr + (nc * Kc + kci) * block_n * block_k,
            /* scales_b */ scales_b,
            /* M */ m_size,
            /* K */ block_k,
            /* lda */ K,
            /* ldc */ block_n,
            /* ldsa */ ldsa,
            /* ukernel_buf */ ukernel_buf,
            /* dqA_buf */ dqA_buffer,
            /* dqB_buf */ dqB_buffer);
      }
      auto scales_a = act_quant_mode == PER_TENSOR ? a_scales_ptr
                      : act_quant_mode == PER_ROW  ? a_scales_ptr + mci * block_m
                                                   : nullptr;
      auto scales_b = wei_quant_mode == PER_TENSOR ? b_scales_ptr
                      : wei_quant_mode == PER_ROW  ? b_scales_ptr + nc * block_n
                                                   : nullptr;
      auto bias_data = bias_ptr ? bias_ptr + nc * block_n : nullptr;
      store_out<out_dtype, block_n, act_quant_mode, wei_quant_mode>(
          y_buf, c_ptr + mci * block_m * N + nc * block_n, m_size, N /*lda*/, scales_a, scales_b, bias_data);
    });

    if constexpr (cpublas_can_pack) {
      at::native::cpublas::brgemm_release();
    }
  });
}

}  // anonymous namespace

void tinygemm_kernel(
    float* C,
    const at::Float8_e4m3fn* A,
    const float* scales_a,
    const at::Float8_e4m3fn* B,
    const float* scales_b,
    int64_t M,
    int64_t K,
    int64_t lda,
    int64_t ldc,
    int64_t ldsa,
    float* ukernel_buf,
    at::BFloat16* dqA_buf,
    at::BFloat16* dqB_buf) {
  // cpublas_can_pack = True, act_quant_mode = per row (2), wei_quant_mode = per group (3)
  _micro_gemm<true, BLOCK_N, 2, 3>(C, A, scales_a, B, scales_b, M, K, lda, ldc, ldsa, ukernel_buf, dqA_buf, dqB_buf);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE() \
  void tinygemm_kernel(                 \
      float* C,                         \
      const at::Float8_e4m3fn* A,       \
      const float* scales_a,            \
      const at::Float8_e4m3fn* B,       \
      const float* scales_b,            \
      int64_t M,                        \
      int64_t K,                        \
      int64_t lda,                      \
      int64_t ldc,                      \
      int64_t ldsa,                     \
      float* ukernel_buf,               \
      at::BFloat16* dqA_buf,            \
      at::BFloat16* dqB_buf)

INSTANTIATE_TINYGEMM_TEMPLATE();
/*
return: packed_weight, packed_scales
*/
std::tuple<at::Tensor, at::Tensor> float8_linear_prepack_impl(const at::Tensor& weight, const at::Tensor& scales) {
  // weight shape = [N, K]
  // scales shape = [N, G]
  TORCH_CHECK(weight.dim() == 2, "Float8 linear CPU: Weight should be a 2D tensor for packing");
  TORCH_CHECK(weight.size(1) % 2 == 0, "Float8 linear CPU: Weight should have even number of columns for packing");

  auto new_scales = scales;
  if (new_scales.dim() == 1) {
    new_scales.unsqueeze_(1);
  }
  new_scales = new_scales.to(at::kFloat);
  int N = weight.size(0);
  int K = weight.size(1);
  int G = scales.size(1);
  int group_size = K / G;
  int block_k = group_size > 128 ? 128 : group_size;
  while (K % block_k != 0) {
    block_k /= 2;
  }
  TORCH_CHECK(
      block_k > 0 && block_k <= group_size, "Float8 linear CPU: Invalid block_k size, should be in (0, group_size]");
  constexpr int block_n = BLOCK_N;
  int Nc = N / block_n;
  int Kc = K / block_k;

  // Reorder weight to [N/block_n, K/block_k, block_k, block_n]
  // Reorder scales to [N/block_n, G, block_n]
  auto weight_view = weight.view({Nc, block_n, Kc, block_k});
  at::Tensor weight_reordered = weight_view.permute({0, 2, 3, 1}).contiguous();
  at::Tensor blocked_weight;
  at::Tensor blocked_scales = new_scales.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();

#if defined(CPU_CAPABILITY_AVX512)
  if (cpublas_could_pack()) {
#ifdef CPUBLAS_BRGEMM_F8F8F32
    constexpr int vnni_size = 4;  // for fp8
#else
    constexpr int vnni_size = 2;  // for float16
#endif
    blocked_weight = at::empty({Nc, Kc, block_k, block_n}, weight.options());
    auto weight_ptr = reinterpret_cast<uint8_t*>(weight_reordered.data_ptr());
    auto blocked_weight_ptr = reinterpret_cast<uint8_t*>(blocked_weight.data_ptr());
    int64_t num_blocks = Nc * Kc;
    at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        auto in_ptr = weight_ptr + i * block_k * block_n;
        auto out_ptr = blocked_weight_ptr + i * block_k * block_n;

        // Reorder weight block to VNNI
        // plain shape = [block_k, block_n]
        // packed shape = [block_k / VNNI_SIZE, block_n, VNNI_SIZE] viewed as [block_k, block_n]
        constexpr int n_group_size = 8;
        constexpr int n_group = block_n / n_group_size;  // 4
        for (int nb = 0; nb < n_group; ++nb) {
          for (int k = 0; k < block_k; k += vnni_size) {
            for (int ni = 0; ni < n_group_size; ++ni) {
              for (int ki = 0; ki < vnni_size; ++ki) {
                int src_idx = nb * n_group_size + ni + (k + ki) * block_n;
                int dst_idx = (nb * n_group_size + ni) * vnni_size + k * block_n + ki;
                *(out_ptr + dst_idx) = *(in_ptr + src_idx);
              }
            }
          }
        }
      }
    });
  } else
#endif
  {
    blocked_weight = weight_reordered;
  }

  return std::make_tuple(std::move(blocked_weight), std::move(blocked_scales));
}

// AVX512 optimized channel-wise max computation
inline void
compute_channel_max_avx512(const float* data, float* max_vals, int64_t num_channels, int64_t elements_per_channel) {
  at::parallel_for(0, num_channels, 1, [&](int64_t start, int64_t end) {
    for (int64_t c = start; c < end; ++c) {
      const float* channel_data = data + c * elements_per_channel;
      float max_val = 0.0f;

      int64_t i = 0;
      // Process 16 elements at a time
      __m512 max_vec = _mm512_setzero_ps();

      for (; i <= elements_per_channel - 16; i += 16) {
        __m512 data_vec = _mm512_loadu_ps(&channel_data[i]);
        __m512 abs_vec = _mm512_abs_ps(data_vec);
        max_vec = _mm512_max_ps(max_vec, abs_vec);
      }

      // Horizontal max of the vector
      float hmax = _mm512_reduce_max_ps(max_vec);
      max_val = std::max(max_val, hmax);

      // Handle remaining elements
      for (; i < elements_per_channel; ++i) {
        max_val = std::max(max_val, std::abs(channel_data[i]));
      }

      max_vals[c] = max_val;
    }
  });
}

// AVX512 optimized scaling and clamping for channel-wise quantization
inline void scale_clamp_channelwise_avx512(
    const float* src,
    float* dst,
    const float* scales,
    int64_t num_channels,
    int64_t elements_per_channel,
    float quant_max,
    float neg_quant_max) {
  at::parallel_for(0, num_channels, 1, [&](int64_t start, int64_t end) {
    for (int64_t c = start; c < end; ++c) {
      float scale_val = scales[c];
      float scale_reciprocal = 1.0f / scale_val;

      const float* channel_src = src + c * elements_per_channel;
      float* channel_dst = dst + c * elements_per_channel;

      int64_t i = 0;
      const __m512 scale_recip_vec = _mm512_set1_ps(scale_reciprocal);
      const __m512 quant_max_vec = _mm512_set1_ps(quant_max);
      const __m512 neg_quant_max_vec = _mm512_set1_ps(neg_quant_max);

      for (; i <= elements_per_channel - 16; i += 16) {
        __m512 src_vec = _mm512_loadu_ps(&channel_src[i]);
        __m512 scaled_vec = _mm512_mul_ps(src_vec, scale_recip_vec);
        __m512 clamped_vec = _mm512_min_ps(_mm512_max_ps(scaled_vec, neg_quant_max_vec), quant_max_vec);
        _mm512_storeu_ps(&channel_dst[i], clamped_vec);
      }

      // Handle remaining elements
      for (; i < elements_per_channel; ++i) {
        float scaled = channel_src[i] * scale_reciprocal;
        channel_dst[i] = std::clamp(scaled, neg_quant_max, quant_max);
      }
    }
  });
}

// AVX512 optimized scaling and clamping for global quantization
inline void scale_clamp_global_avx512(
    const float* src, float* dst, int64_t size, float scale_reciprocal, float quant_max, float neg_quant_max) {
  if (size <= 0) return;

  const __m512 scale_recip_vec = _mm512_set1_ps(scale_reciprocal);
  const __m512 quant_max_vec = _mm512_set1_ps(quant_max);
  const __m512 neg_quant_max_vec = _mm512_set1_ps(neg_quant_max);

  at::parallel_for(0, size, 4096, [&](int64_t start, int64_t end) {
    int64_t block_size = end - start;
    const float* block_src = src + start;
    float* block_dst = dst + start;

    int64_t i = 0;
    for (; i <= block_size - 16; i += 16) {
      __m512 src_vec = _mm512_loadu_ps(&block_src[i]);
      __m512 scaled_vec = _mm512_mul_ps(src_vec, scale_recip_vec);
      __m512 clamped_vec = _mm512_min_ps(_mm512_max_ps(scaled_vec, neg_quant_max_vec), quant_max_vec);
      _mm512_storeu_ps(&block_dst[i], clamped_vec);
    }

    // Handle remaining elements in block
    for (; i < block_size; ++i) {
      float scaled = block_src[i] * scale_reciprocal;
      block_dst[i] = std::clamp(scaled, neg_quant_max, quant_max);
    }
  });
}

std::tuple<at::Tensor, at::Tensor>
_quantize_fp8e4m3(const at::Tensor& t, bool channelwise, c10::optional<at::Tensor> scale_opt = c10::nullopt) {
  constexpr float quant_max = 448.0f;  // torch.finfo(torch.float8_e4m3fn).max
  constexpr float eps = std::numeric_limits<float>::epsilon();

  // Ensure input is contiguous and in float32
  auto t_float = t.to(at::ScalarType::Float).contiguous();
  at::Tensor qt;
  at::Tensor scale_tensor;

  if (channelwise) {
    // Channel-wise quantization with AVX512 optimization
    int64_t num_channels = t_float.size(0);
    int64_t elements_per_channel = t_float.numel() / num_channels;

    if (elements_per_channel * num_channels != t_float.numel()) {
      throw std::runtime_error("Tensor must be divisible by number of channels for channel-wise quantization");
    }

    // Allocate scale tensor
    scale_tensor = at::empty({num_channels}, t_float.options());
    float* scale_data = scale_tensor.data_ptr<float>();

    // Compute channel-wise max using AVX512
    compute_channel_max_avx512(t_float.data_ptr<float>(), scale_data, num_channels, elements_per_channel);

    // Apply quant_max and EPS
    at::parallel_for(0, num_channels, 1, [&](int64_t start, int64_t end) {
      for (int64_t c = start; c < end; ++c) {
        scale_data[c] = std::max(scale_data[c] / quant_max, eps);
      }
    });

    // Create output tensor for quantized values
    qt = at::empty_like(t_float);

    // Scale and clamp using AVX512
    scale_clamp_channelwise_avx512(
        t_float.data_ptr<float>(),
        qt.data_ptr<float>(),
        scale_data,
        num_channels,
        elements_per_channel,
        quant_max,
        -quant_max);
  } else {
    // Global quantization with AVX512 optimization
    if (!scale_opt.has_value()) {
      throw std::runtime_error("Scale must be provided for non-channelwise quantization in AVX512 version");
    }

    scale_tensor = scale_opt.value().to(at::ScalarType::Float).contiguous();

    // Handle scalar scale case
    float scale_val;
    if (scale_tensor.numel() == 1) {
      scale_val = std::max(scale_tensor.item<float>(), eps);
    } else {
      // Compute max of scale tensor if it's not scalar
      float* scale_data = scale_tensor.data_ptr<float>();
      __m512 max_vec = _mm512_set1_ps(eps);

      int64_t i = 0;
      for (; i <= scale_tensor.numel() - 16; i += 16) {
        __m512 scale_vec = _mm512_loadu_ps(&scale_data[i]);
        max_vec = _mm512_max_ps(max_vec, scale_vec);
      }

      float hmax = _mm512_reduce_max_ps(max_vec);
      for (; i < scale_tensor.numel(); ++i) {
        hmax = std::max(hmax, scale_data[i]);
      }
      scale_val = std::max(hmax, eps);
    }

    scale_tensor = at::tensor({scale_val}, scale_tensor.options());
    float scale_reciprocal = 1.0f / scale_val;

    // Create output tensor
    qt = at::empty_like(t_float);

    // Scale and clamp using AVX512
    scale_clamp_global_avx512(
        t_float.data_ptr<float>(), qt.data_ptr<float>(), t_float.numel(), scale_reciprocal, quant_max, -quant_max);
  }

  // Final conversion to FP8 E4M3FN using ATen's native conversion
  qt = at::_to_copy(qt, at::ScalarType::Float8_e4m3fn);

  return std::make_tuple(qt, scale_tensor);
}

inline __m128i cvtfp32_fp8e4m3(__m512& src) {
#ifdef __AVX10_2__
  __m256i f16_vec =
      _mm512_cvt_roundps_ph(src, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return _mm256_cvtph_hf8(_mm256_castsi256_ph(f16_vec));
#else
  // cvt 16x32 from fp32 to fp8 e4m3
  const __m512i sign_mask = _mm512_set1_epi32(0x80000000);
  const __m512i fp8_max = _mm512_set1_epi32(UINT32_C(1087) << 20);
  const __m512i denorm_thresh = _mm512_set1_epi32(UINT32_C(121) << 23);
  const __m512i denorm_mask = _mm512_set1_epi32(UINT32_C(141) << 23);
  const __m512i bias_part1 = _mm512_set1_epi32((uint32_t)(7 - 127) << 23);
  const __m512i rounding_bias = _mm512_set1_epi32(0x7FFFF);
  __m512i f_bits = _mm512_castps_si512(src);
  // Extract and save sign
  __m512i sign = _mm512_and_epi32(f_bits, sign_mask);
  f_bits = _mm512_xor_epi32(f_bits, sign);

  // Prepare result containers
  __m512i result = _mm512_setzero_si512();

  // Step 1: Handle case of overflow
  // (f_bits >= fp8_max): set result = 0x7f
  __mmask16 overflow_mask = _mm512_cmpge_epu32_mask(f_bits, fp8_max);
  if (overflow_mask) {
    result = _mm512_mask_set1_epi32(result, overflow_mask, 0x7f);
  }

  // Step 2: Handle small numbers (denormals)
  // Small numbers (f_bits < denorm_thresh)
  __mmask16 denorm_thresh_mask = _mm512_cmplt_epu32_mask(f_bits, denorm_thresh);

  if (denorm_thresh_mask) {
    __m512 small_input = _mm512_castsi512_ps(f_bits);
    __m512 small_denorm = _mm512_add_ps(small_input, _mm512_castsi512_ps(denorm_mask));
    __m512i small_denorm_bits = _mm512_castps_si512(small_denorm);
    __m512i small_result = _mm512_sub_epi32(small_denorm_bits, denorm_mask);
    result = _mm512_mask_mov_epi32(result, denorm_thresh_mask, small_result);
  }

  // Step 3: Handle normal numbers
  __mmask16 normal_mask = ~(overflow_mask | denorm_thresh_mask);

  if (normal_mask) {
    // mant_odd = (f_bits >> 20) & 1
    __m512i mant_odd = _mm512_and_epi32(_mm512_srli_epi32(f_bits, 20), _mm512_set1_epi32(1));
    // f_bits += bias_part1 + rounding_bias
    __m512i rounded = _mm512_add_epi32(f_bits, bias_part1);
    rounded = _mm512_add_epi32(rounded, rounding_bias);
    // Add mant_odd
    rounded = _mm512_add_epi32(rounded, mant_odd);
    // Shift right by 20 bits
    __m512i normal_result = _mm512_srli_epi32(rounded, 20);
    result = _mm512_mask_mov_epi32(result, normal_mask, normal_result);
  }

  // Merge back the sign
  __m512i sign_shifted = _mm512_srli_epi32(sign, 24);
  result = _mm512_or_epi32(result, sign_shifted);

  // Now result is 16 x 32-bit integers, but we only need 8-bit for each
  __m512i packed = _mm512_and_si512(result, _mm512_set1_epi32(0xFF));

  // Narrow 32-bit integers to 8-bit
  return _mm512_cvtepi32_epi8(packed);
#endif
}

std::tuple<at::Tensor, at::Tensor> _quantize_fp8e4m3_bf16_per_tensor_no_scale(const at::Tensor& t) {
  constexpr float quant_max = 448.0f;  // torch.finfo(torch.float8_e4m3fn).max
  constexpr float eps = std::numeric_limits<float>::epsilon();

  // Input validation and preparation
  assert(t.scalar_type() == at::ScalarType::BFloat16);
  auto t_bf16 = t.contiguous();
  int64_t num_channels = t_bf16.size(0);
  int64_t elements_per_channel = t_bf16.numel() / num_channels;
  assert(elements_per_channel % 32 == 0);  // do not consider tile currently
  at::Tensor quant_t = at::empty(t_bf16.sizes(), t_bf16.options().dtype(at::kFloat8_e4m3fn));

  // Allocate output tensors
  at::Tensor scale_tensor = at::empty({num_channels}, t_bf16.options().dtype(at::ScalarType::Float));
  float* scale_data = scale_tensor.data_ptr<float>();

  // Unified processing: compute max, apply scale, and quantize in single pass
  at::parallel_for(0, num_channels, 1, [&](int64_t start, int64_t end) {
    for (int64_t c = start; c < end; ++c) {
      const at::BFloat16* channel_src = t_bf16.data_ptr<at::BFloat16>() + c * elements_per_channel;
      at::Float8_e4m3fn* quant_dst = quant_t.data_ptr<at::Float8_e4m3fn>() + c * elements_per_channel;

      // Step 1: Compute channel-wise max using AVX512
      float channel_max = 0.0f;
      int64_t i = 0;

      // Process 32 elements at a time for max computation
      for (; i <= elements_per_channel - 32; i += 32) {
        // Load 32 BF16 values (2x 256-bit vectors)
        __m256i src_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&channel_src[i]));
        __m256i src_vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&channel_src[i + 16]));

        // Convert BF16 to FP32
        __m512i fp32_int1 = _mm512_cvtepu16_epi32(src_vec1);
        __m512 fp32_vec1 = _mm512_castsi512_ps(_mm512_slli_epi32(fp32_int1, 16));

        __m512i fp32_int2 = _mm512_cvtepu16_epi32(src_vec2);
        __m512 fp32_vec2 = _mm512_castsi512_ps(_mm512_slli_epi32(fp32_int2, 16));

        // Compute absolute values
        __m512 abs_vec1 = _mm512_abs_ps(fp32_vec1);
        __m512 abs_vec2 = _mm512_abs_ps(fp32_vec2);

        // Find max in each vector
        float max1 = _mm512_reduce_max_ps(abs_vec1);
        float max2 = _mm512_reduce_max_ps(abs_vec2);
        channel_max = std::max(channel_max, std::max(max1, max2));
      }

      // Step 2: Apply quant_max and EPS to compute scale
      float scale_val = std::max(channel_max / quant_max, eps);
      float scale_reciprocal = 1.0f / scale_val;
      scale_data[c] = scale_val;

      // Step 3: Scale and clamp using AVX512 (reuse the same loop structure)
      i = 0;
      const __m512 scale_recip_vec = _mm512_set1_ps(scale_reciprocal);
      const __m512 quant_max_vec = _mm512_set1_ps(quant_max);
      const __m512 neg_quant_max_vec = _mm512_set1_ps(-quant_max);

      for (; i <= elements_per_channel - 32; i += 32) {
        // Load 32 BF16 values
        __m256i src_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&channel_src[i]));
        __m256i src_vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&channel_src[i + 16]));

        // Convert BF16 to FP32
        __m512 fp32_vec1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(src_vec1), 16));
        __m512 fp32_vec2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(src_vec2), 16));

        // Scale and clamp
        __m512 scaled_vec1 = _mm512_mul_ps(fp32_vec1, scale_recip_vec);
        __m512 clamped_vec1 = _mm512_min_ps(_mm512_max_ps(scaled_vec1, neg_quant_max_vec), quant_max_vec);

        __m512 scaled_vec2 = _mm512_mul_ps(fp32_vec2, scale_recip_vec);
        __m512 clamped_vec2 = _mm512_min_ps(_mm512_max_ps(scaled_vec2, neg_quant_max_vec), quant_max_vec);

        __m128i fp8_vec1 = cvtfp32_fp8e4m3(clamped_vec1);
        __m128i fp8_vec2 = cvtfp32_fp8e4m3(clamped_vec2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_dst + i), fp8_vec1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_dst + i + 16), fp8_vec2);
      }
    }
  });

  return std::make_tuple(quant_t, scale_tensor);
}

std::tuple<at::Tensor, at::Tensor>
_quantize_fp8e4m3_bf16_per_tensor_with_scale(const at::Tensor& t, at::Tensor& scale_tensor) {
  constexpr float quant_max = 448.0f;  // torch.finfo(torch.float8_e4m3fn).max
  constexpr float eps = std::numeric_limits<float>::epsilon();

  // Input validation
  assert(t.scalar_type() == at::ScalarType::BFloat16);

  auto t_bf16 = t.contiguous();
  auto scale_tensor_contig = scale_tensor.contiguous().to(at::ScalarType::Float);

  // Get scale value (handle scalar or tensor)
  float scale_val;
  if (scale_tensor_contig.numel() == 1) {
    scale_val = scale_tensor_contig.item<float>();
  } else {
    // Take max if scale is a tensor (though should be scalar for global quantization)
    scale_val = scale_tensor_contig.max().item<float>();
  }

  // Apply EPS to scale
  scale_val = std::max(scale_val, eps);
  float scale_reciprocal = 1.0f / scale_val;

  at::Tensor scale_output = at::tensor({scale_val}, scale_tensor_contig.options());

  // Apply scale and clamp using AVX512
  const at::BFloat16* src_data = t_bf16.data_ptr<at::BFloat16>();
  at::Tensor quant_t = at::empty(t_bf16.sizes(), t_bf16.options().dtype(at::kFloat8_e4m3fn));
  int64_t total_elements = t_bf16.numel();
  const __m512 scale_recip_vec = _mm512_set1_ps(scale_reciprocal);
  const __m512 quant_max_vec = _mm512_set1_ps(quant_max);
  const __m512 neg_quant_max_vec = _mm512_set1_ps(-quant_max);
  int64_t num_channels = t_bf16.size(0);
  int64_t elements_per_channel = total_elements / num_channels;
  assert(elements_per_channel % 32 == 0);  // do not consider tile currently
  // Process in parallel blocks
  at::parallel_for(0, num_channels, 1, [&](int64_t start, int64_t end) {
    for (int64_t c = start; c < end; ++c) {
      const at::BFloat16* src_data = t_bf16.data_ptr<at::BFloat16>() + c * elements_per_channel;
      at::Float8_e4m3fn* quant_t_data = quant_t.data_ptr<at::Float8_e4m3fn>() + c * elements_per_channel;

      int64_t i = 0;
      // Process 32 elements at a time using AVX512
      for (; i <= elements_per_channel - 32; i += 32) {
        // Load 32 BF16 values (2x 256-bit vectors)
        __m256i src_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src_data[i]));
        __m256i src_vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src_data[i + 16]));

        // Convert BF16 to FP32
        __m512 fp32_vec1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(src_vec1), 16));
        __m512 fp32_vec2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(src_vec2), 16));

        // Scale and clamp
        __m512 scaled_vec1 = _mm512_mul_ps(fp32_vec1, scale_recip_vec);
        __m512 clamped_vec1 = _mm512_min_ps(_mm512_max_ps(scaled_vec1, neg_quant_max_vec), quant_max_vec);

        __m512 scaled_vec2 = _mm512_mul_ps(fp32_vec2, scale_recip_vec);
        __m512 clamped_vec2 = _mm512_min_ps(_mm512_max_ps(scaled_vec2, neg_quant_max_vec), quant_max_vec);

        __m128i fp8_vec1 = cvtfp32_fp8e4m3(clamped_vec1);
        __m128i fp8_vec2 = cvtfp32_fp8e4m3(clamped_vec2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_t_data + i), fp8_vec1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_t_data + i + 16), fp8_vec2);
      }
    }
  });

  return std::make_tuple(quant_t, scale_output);
}

std::tuple<at::Tensor, at::Tensor>
_quantize_fp8e4m3_vec(const at::Tensor& t, bool channelwise, c10::optional<at::Tensor> scale_opt) {
  if (channelwise) {
    return _quantize_fp8e4m3_bf16_per_tensor_no_scale(t);
  } else {
    assert(scale_opt.has_value());
    return _quantize_fp8e4m3_bf16_per_tensor_with_scale(t, scale_opt.value());
  }
}

// Public wrapper for torch op registration
std::tuple<at::Tensor, at::Tensor>
quantize_fp8e4m3_vec(const at::Tensor& t, bool channelwise, c10::optional<at::Tensor> scale_opt) {
  return _quantize_fp8e4m3_vec(t, channelwise, scale_opt);
}

at::Tensor fp8_scaled_mm_with_quant(
    const at::Tensor& act,
    const std::optional<at::Tensor>& act_scales,
    bool channelwise,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const std::optional<at::Tensor>& bias,
    at::ScalarType output_dtype) {
  // TODO: refine here to use tensor ptr and in below dispatch api.
  std::tuple<at::Tensor, at::Tensor> quant_act = act.scalar_type() == at::ScalarType::BFloat16
                                                     ? _quantize_fp8e4m3_vec(act, channelwise, act_scales)
                                                     : _quantize_fp8e4m3(act, channelwise, act_scales);
  auto input = std::get<0>(quant_act);
  auto input_scales = std::get<1>(quant_act);
  int64_t N = weight.dim() == 4 ? weight.size(0) * weight.size(-1) : weight.size(0);
  int act_quant_mode = input_scales.numel() == 1                                ? PER_TENSOR
                       : input_scales.numel() == input.numel() / input.size(-1) ? PER_ROW
                                                                                : PER_GROUP;
  int wei_quant_mode = weight_scales.numel() == 1 ? PER_TENSOR : weight_scales.numel() == N ? PER_ROW : PER_GROUP;
  // Case to fall back
  if (weight.dim() == 2) {
    TORCH_CHECK(
        act_quant_mode != PER_GROUP && wei_quant_mode != PER_GROUP,
        "FP8 linear: Per-group quantization is not supported in the fallback path");
    auto y_fp32 = at::linear(input.to(at::kFloat).mul_(input_scales), weight.to(at::kFloat).mul_(weight_scales), bias);
    return y_fp32.to(output_dtype);
  }

  static bool cpublas_can_pack = cpublas_could_pack();
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  auto output = at::empty(out_sizes, input.options().dtype(output_dtype));

#define AT_DISPATCH_FP8_LINEAR_KERNEL(OUT_DTYPE, CAN_PACK, A_QUANT_MODE, B_QUANT_MODE, ...) \
  AT_DISPATCH_BOOL_NO_RETURN(                                                               \
      CAN_PACK,                                                                             \
      "cpublas_can_pack",                                                                   \
      can_pack,                                                                             \
      AT_DISPATCH_QUANT_MODE_NO_RETURN(                                                     \
          A_QUANT_MODE,                                                                     \
          "act_quant_mode",                                                                 \
          a_quant_mode,                                                                     \
          AT_DISPATCH_QUANT_MODE_NO_RETURN(                                                 \
              B_QUANT_MODE,                                                                 \
              "wei_quant_mode",                                                             \
              b_quant_mode,                                                                 \
              AT_DISPATCH_OUT_TYPES(OUT_DTYPE, "out_dtype", __VA_ARGS__))))

  AT_DISPATCH_FP8_LINEAR_KERNEL(output_dtype, cpublas_can_pack, act_quant_mode, wei_quant_mode, [&]() {
    _float8_linear_impl<out_t, can_pack, a_quant_mode, b_quant_mode>(
        input, input_scales, weight, weight_scales, bias, output);
  });
  return output;
}

at::Tensor float8_linear_impl(
    const at::Tensor& input,
    const at::Tensor& input_scales,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const std::optional<at::Tensor>& bias,
    at::ScalarType output_dtype) {
  int64_t N = weight.dim() == 4 ? weight.size(0) * weight.size(-1) : weight.size(0);
  int act_quant_mode = input_scales.numel() == 1                                ? PER_TENSOR
                       : input_scales.numel() == input.numel() / input.size(-1) ? PER_ROW
                                                                                : PER_GROUP;
  int wei_quant_mode = weight_scales.numel() == 1 ? PER_TENSOR : weight_scales.numel() == N ? PER_ROW : PER_GROUP;
  // Case to fall back
  if (weight.dim() == 2) {
    TORCH_CHECK(
        act_quant_mode != PER_GROUP && wei_quant_mode != PER_GROUP,
        "FP8 linear: Per-group quantization is not supported in the fallback path");
    auto y_fp32 = at::linear(input.to(at::kFloat).mul_(input_scales), weight.to(at::kFloat).mul_(weight_scales), bias);
    return y_fp32.to(output_dtype);
  }

  static bool cpublas_can_pack = cpublas_could_pack();
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  auto output = at::empty(out_sizes, input.options().dtype(output_dtype));

#define AT_DISPATCH_FP8_LINEAR_KERNEL(OUT_DTYPE, CAN_PACK, A_QUANT_MODE, B_QUANT_MODE, ...) \
  AT_DISPATCH_BOOL_NO_RETURN(                                                               \
      CAN_PACK,                                                                             \
      "cpublas_can_pack",                                                                   \
      can_pack,                                                                             \
      AT_DISPATCH_QUANT_MODE_NO_RETURN(                                                     \
          A_QUANT_MODE,                                                                     \
          "act_quant_mode",                                                                 \
          a_quant_mode,                                                                     \
          AT_DISPATCH_QUANT_MODE_NO_RETURN(                                                 \
              B_QUANT_MODE,                                                                 \
              "wei_quant_mode",                                                             \
              b_quant_mode,                                                                 \
              AT_DISPATCH_OUT_TYPES(OUT_DTYPE, "out_dtype", __VA_ARGS__))))

  AT_DISPATCH_FP8_LINEAR_KERNEL(output_dtype, cpublas_can_pack, act_quant_mode, wei_quant_mode, [&]() {
    _float8_linear_impl<out_t, can_pack, a_quant_mode, b_quant_mode>(
        input, input_scales, weight, weight_scales, bias, output);
  });
  return output;
}