// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

enum class ConvStateLayout { SD, DS };

struct ConvStateLayoutInfo {
  ConvStateLayout layout;
  int64_t state_len;
  int64_t conv_state_slot_stride;
};

ConvStateLayoutInfo validate_conv_state_layout(
    const at::Tensor& conv_states,
    int64_t dim,
    int64_t width,
    const char* op_name) {
  const int64_t state_len = conv_states.size(2);
  CHECK_GE(state_len, width - 1);

  ConvStateLayout layout = ConvStateLayout::SD;
  if (conv_states.stride(-2) == 1 && conv_states.stride(-1) == dim) {
    layout = ConvStateLayout::SD;
  } else if (
      conv_states.stride(-1) == 1 &&
      conv_states.stride(-2) == state_len) {
    layout = ConvStateLayout::DS;
  } else {
    TORCH_CHECK(
        false,
        op_name,
        ": conv_states must use SD "
        "(stride[-2]=1, stride[-1]=dim) or DS "
        "(stride[-2]=state_len, stride[-1]=1) layout; got strides ",
        conv_states.stride(-2),
        ", ",
        conv_states.stride(-1));
  }

  return {layout, state_len, conv_states.stride(0)};
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ y, const scalar_t* __restrict__ x, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  const bool is_padding = (x == nullptr);
  for (int64_t d = 0; d < size; d += Vec::size()) {
    Vec data_vec = is_padding ? Vec(0.f) : Vec::loadu(x + d);
    data_vec.store(y + d);
  }
}

// no remainder
template <typename scalar_t>
void inline update_conv_state(
    scalar_t* __restrict__ conv_states,
    const scalar_t* __restrict__ input,
    int64_t width,
    int64_t dim,
    int64_t seqlen,
    bool has_initial_states) {
  // width for `conv_states`
  int64_t width1 = width - 1;
  int64_t w = 0;
  for (; w < width1 - seqlen; ++w) {
    scalar_t* y = conv_states + w * dim;
    const scalar_t* x = has_initial_states ? conv_states + (w + seqlen) * dim : nullptr;
    copy_stub(y, x, dim);
  }
  for (; w < width1; ++w) {
    scalar_t* y = conv_states + w * dim;
    const scalar_t* x = input + (w + seqlen - width1) * dim;
    copy_stub(y, x, dim);
  }
}

// A : [M, BLOCK_N]
// B : [BLOCK_N, K], prepacked as [K/2, BLOCK_N, 2]
// C : [M, BLOCK_N]
// bias : [BLOCK_N]
//
// lda : leading dimension of `input` and `out`
//
template <typename scalar_t, int K, int BLOCK_N, bool has_bias, bool has_silu>
struct tinygemm_kernel {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const scalar_t* __restrict__ bias,
      const scalar_t* __restrict__ conv_states,
      bool has_initial_state,
      int64_t M,
      int64_t lda,
      bool is_first_token) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int K, int BLOCK_N, bool has_bias, bool has_silu>
struct tinygemm_kernel<at::BFloat16, K, BLOCK_N, has_bias, has_silu> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      at::BFloat16* __restrict__ C,
      const at::BFloat16* __restrict__ bias,
      const at::BFloat16* __restrict__ conv_states,
      bool has_initial_state,
      int64_t M,
      int64_t lda,
      bool is_first_token) {
    assert(K == 4);
    constexpr int ROWS = K;
    constexpr int COLS = BLOCK_N / block_size_n();

    // leading dimension size for b for next block [K/2, 32, 2]
    constexpr int ldb = block_size_n() * K;

    __m512bh va[ROWS * COLS];
    __m512bh vb[ROWS * COLS];
    __m512 vc[COLS * 2];

    // k: {-3, -2, -1} -> {0, 1, 2}
    auto set_conv_states = [&](int k, int col) -> __m512i {
      return has_initial_state ? _mm512_loadu_si512(conv_states + (k + K - 1) * lda + col * 32)
                               : _mm512_setzero_si512();
    };

#define MM512_LOAD_A(idx)                                                 \
  ((idx) < 0 && is_first_token) ? (__m512bh)(set_conv_states((idx), col)) \
                                : (__m512bh)(_mm512_loadu_si512(A + (idx) * lda + col * 32))

#define MM512_PACK_A(ap, bp, a, b)                       \
  do {                                                   \
    __m512i r0 = (__m512i)(a);                           \
    __m512i r1 = (__m512i)(b);                           \
    __m512i d0 = _mm512_unpacklo_epi16(r0, r1);          \
    __m512i d1 = _mm512_unpackhi_epi16(r0, r1);          \
    r0 = _mm512_shuffle_i32x4(d0, d1, 0x88);             \
    r1 = _mm512_shuffle_i32x4(d0, d1, 0xdd);             \
    (ap) = (__m512bh)_mm512_shuffle_i32x4(r0, r1, 0x88); \
    (bp) = (__m512bh)_mm512_shuffle_i32x4(r0, r1, 0xdd); \
  } while (0)

    // step 0 : preload a at time step [-3][-2][-1]
    auto preloada = [&](auto i) {
      constexpr int col = i;
      int64_t m = 0;
      va[1 * COLS + col] = MM512_LOAD_A(m - 3);
      va[2 * COLS + col] = MM512_LOAD_A(m - 2);
      va[3 * COLS + col] = MM512_LOAD_A(m - 1);
    };
    Unroll<COLS>{}(preloada);

    auto loada = [&](auto i, int64_t m) {
      constexpr int col = i;
      // update previous time step
      va[0 * COLS + col] = va[1 * COLS + col];
      va[1 * COLS + col] = va[2 * COLS + col];
      va[2 * COLS + col] = va[3 * COLS + col];
      // load current time step
      va[3 * COLS + col] = MM512_LOAD_A(m);
    };

    // step 1 : load weight for just once
    auto loadb = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      vb[row * COLS + col] = (__m512bh)(_mm512_loadu_si512(B + col * ldb + row * 32));
    };
    Unroll<ROWS * COLS>{}(loadb);

    // [NB] accumulates 4x32 bfloat16 blocks
    //
    //   +------------+------------+
    //   |    col0    |    col1    |
    //   +------------+------------+
    //   |  va0  va1  |  va0  va1  |
    //   |  va2  va3  |  va2  va3  |
    //   +------------+------------+
    //   |  vc0  vc1  |  vc0  vc1  |
    //   +------------+------------+
    //
    //  * va and vb shares the same memory layout
    //  * block_n 32 with 4 rows equals to 4 registers
    //  * 37 uops with avx512bf16 v.s. 57 uops with avx512f
    //
    auto compute = [&](auto i) {
      constexpr int col = i;

      // init accumulators
      if constexpr (has_bias) {
        __m512i b16 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(bias + col * 32));
        vc[col * 2 + 0] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(b16, 0));
        vc[col * 2 + 1] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(b16, 1));
      } else {
        vc[col * 2 + 0] = _mm512_set1_ps(0.f);
        vc[col * 2 + 1] = _mm512_set1_ps(0.f);
      }

      // convert to vnni2 format
      __m512bh va0, va1, va2, va3;
      MM512_PACK_A(va0, va1, va[0 * COLS + col], va[1 * COLS + col]);
      MM512_PACK_A(va2, va3, va[2 * COLS + col], va[3 * COLS + col]);

      // accumulate
      vc[col * 2 + 0] = _mm512_dpbf16_ps(vc[col * 2 + 0], va0, vb[0 * COLS + col]);
      vc[col * 2 + 0] = _mm512_dpbf16_ps(vc[col * 2 + 0], va2, vb[2 * COLS + col]);
      vc[col * 2 + 1] = _mm512_dpbf16_ps(vc[col * 2 + 1], va1, vb[1 * COLS + col]);
      vc[col * 2 + 1] = _mm512_dpbf16_ps(vc[col * 2 + 1], va3, vb[3 * COLS + col]);
    };

    using fVec = at::vec::Vectorized<float>;
    using bVec = at::vec::Vectorized<at::BFloat16>;
    auto storec = [&](auto i, int64_t m) {
      constexpr int col = i;
      fVec x0 = fVec(vc[col * 2 + 0]);
      fVec x1 = fVec(vc[col * 2 + 1]);
      if constexpr (has_silu) {
        x0 = fast_silu(x0);
        x1 = fast_silu(x1);
      }
      bVec out_vec = convert_from_float_ext<at::BFloat16>(x0, x1);
      out_vec.store(C + m * lda + col * 32);
    };

    for (int64_t m = 0; m < M; ++m) {
      // step 3.a : load a at current time step
      Unroll<COLS>{}(loada, m);
      // step 3.b : accumulate for window size (4)
      Unroll<COLS>{}(compute);
      // step 3.c : store c at current time step
      Unroll<COLS>{}(storec, m);
    }
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL(K, NB_SIZE)                                                   \
  tinygemm_kernel<scalar_t, K, NB_SIZE, has_bias, has_silu>::apply(                          \
      input + bs * seqlen * dim + mb_start * dim + nb_start,                                 \
      weight + nb_start * width,                                                             \
      out + bs * seqlen * dim + mb_start * dim + nb_start,                                   \
      has_bias ? bias + nb_start : nullptr,                                                  \
      has_conv_states ? conv_states + conv_state_index * conv_state_slot_stride + nb_start : nullptr, \
      has_initial_states_value,                                                              \
      mb_size,                                                                               \
      dim,                                                                                   \
      mb_start == 0);

template <typename scalar_t>
void causal_conv1d_fwd_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ conv_states,
    const int32_t* __restrict__ conv_indices,
    const bool* __restrict__ has_initial_state,
    bool silu_activation,
    int64_t batch,
    int64_t dim,
    int64_t seqlen,
    int64_t width,
    int64_t num_seq_blocks,
    int64_t conv_state_slot_stride) {
  // handle 32 x 64 per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n() * 2;
  const int64_t NB = div_up(dim, BLOCK_N);

  const int64_t num_blocks_per_seq = div_up(seqlen, BLOCK_M);
  const bool has_conv_states = conv_states != nullptr;
  const bool has_conv_indices = conv_indices != nullptr;

  // parallel on [batch, seq, NB]
  AT_DISPATCH_BOOL2(bias != nullptr, has_bias, silu_activation, has_silu, [&] {
    at::parallel_for(0, num_seq_blocks * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t mb{0}, nb{0};
      data_index_init(begin, mb, num_seq_blocks, nb, NB);

      for (int64_t i = begin; i < end; ++i) {
        int64_t bs = mb / num_blocks_per_seq;

        int64_t mb_start = (mb % num_blocks_per_seq) * BLOCK_M;
        int64_t mb_size = std::min(seqlen - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(dim - nb_start, BLOCK_N);

        const bool has_initial_states_value = has_conv_states ? has_initial_state[bs] : false;
        int32_t conv_state_index = has_conv_indices ? conv_indices[bs] : bs;

        switch (width << 4 | nb_size >> 4) {
          case 0x42:
            LAUNCH_TINYGEMM_KERNEL(4, 32);
            break;
          case 0x44:
            LAUNCH_TINYGEMM_KERNEL(4, 64);
            break;
          default:
            TORCH_CHECK(false, "Unexpected block size, ", width, " x ", nb_size);
        }

        // move to the next index
        data_index_step(mb, num_seq_blocks, nb, NB);
      }
    });
  });

  // update conv_states if necessary
  if (has_conv_states) {
    at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
      for (int64_t bs = begin; bs < end; ++bs) {
        int32_t conv_state_index = has_conv_indices ? conv_indices[bs] : bs;
        update_conv_state(
            conv_states + conv_state_index * conv_state_slot_stride,
            input + bs * seqlen * dim,
            width,
            dim,
            seqlen,
            has_initial_state[bs]);
      }
    });
  }
}

#define LAUNCH_TINYGEMM_VARLEN_KERNEL(K, NB_SIZE)                                                   \
  tinygemm_kernel<scalar_t, K, NB_SIZE, has_bias, has_silu>::apply(                                 \
      input + batch_offset * dim + mb_start * dim + nb_start,                                       \
      weight + nb_start * width,                                                                    \
      out + batch_offset * dim + mb_start * dim + nb_start,                                         \
      has_bias ? bias + nb_start : nullptr,                                                         \
      has_conv_states ? conv_states + conv_state_index * conv_state_slot_stride + nb_start : nullptr, \
      has_initial_states_value,                                                                     \
      mb_size,                                                                                      \
      dim,                                                                                          \
      mb_start == 0);

template <typename scalar_t>
void causal_conv1d_fwd_varlen_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ conv_states,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ conv_indices,
    const bool* __restrict__ has_initial_state,
    const int32_t* __restrict__ block_indices,
    bool silu_activation,
    int64_t batch,
    int64_t dim,
    int64_t width,
    int64_t num_seq_blocks,
    int64_t conv_state_slot_stride) {
  // handle 32 x 64 per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n() * 2;
  const int64_t NB = div_up(dim, BLOCK_N);

  const bool has_conv_states = conv_states != nullptr;
  const bool has_conv_indices = conv_indices != nullptr;

  // parallel on [batch, seq, NB]
  AT_DISPATCH_BOOL2(bias != nullptr, has_bias, silu_activation, has_silu, [&] {
    at::parallel_for(0, num_seq_blocks * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t mb{0}, nb{0};
      data_index_init(begin, mb, num_seq_blocks, nb, NB);

      for (int64_t i = begin; i < end; ++i) {
        int32_t bs = block_indices[mb * 2 + 0];
        int32_t batch_offset = query_start_loc[bs];
        int32_t seqlen = query_start_loc[bs + 1] - query_start_loc[bs];

        int64_t mb_start = block_indices[mb * 2 + 1] * BLOCK_M;
        int64_t mb_size = std::min(seqlen - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(dim - nb_start, BLOCK_N);

        const bool has_initial_states_value = has_conv_states ? has_initial_state[bs] : false;
        int32_t conv_state_index = has_conv_indices ? conv_indices[bs] : bs;

        switch (width << 4 | nb_size >> 4) {
          case 0x42:
            LAUNCH_TINYGEMM_VARLEN_KERNEL(4, 32);
            break;
          case 0x44:
            LAUNCH_TINYGEMM_VARLEN_KERNEL(4, 64);
            break;
          default:
            TORCH_CHECK(false, "Unexpected block size, ", width, " x ", nb_size);
        }

        // move to the next index
        data_index_step(mb, num_seq_blocks, nb, NB);
      }
    });
  });

  // update conv_states if necessary
  if (has_conv_states) {
    at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
      for (int64_t bs = begin; bs < end; ++bs) {
        int32_t conv_state_index = has_conv_indices ? conv_indices[bs] : bs;
        int32_t seqlen = query_start_loc[bs + 1] - query_start_loc[bs];
        int32_t batch_offset = query_start_loc[bs];
        update_conv_state(
            conv_states + conv_state_index * conv_state_slot_stride,
            input + batch_offset * dim,
            width,
            dim,
            seqlen,
            has_initial_state[bs]);
      }
    });
  }
}

template <typename scalar_t>
void causal_conv1d_update_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ conv_states,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int32_t* __restrict__ conv_indices,
    bool silu_activation,
    int64_t batch,
    int64_t dim,
    int64_t seqlen,
    int64_t width,
    int64_t conv_state_slot_stride) {
  // handle 32 x 64 per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n() * 2;
  const int64_t NB = div_up(dim, BLOCK_N);

  const bool has_conv_states = conv_states != nullptr;
  const bool has_conv_indices = conv_indices != nullptr;

  // parallel on [batch, NB]
  AT_DISPATCH_BOOL2(bias != nullptr, has_bias, silu_activation, has_silu, [&] {
    at::parallel_for(0, batch * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t bs{0}, nb{0};
      data_index_init(begin, bs, batch, nb, NB);

      for (int64_t i = begin; i < end; ++i) {
        int64_t mb_start = 0;
        int64_t mb_size = 1;
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(dim - nb_start, BLOCK_N);

        const bool has_initial_states_value = true;
        int32_t conv_state_index = has_conv_indices ? conv_indices[bs] : bs;

        switch (width << 4 | nb_size >> 4) {
          case 0x42:
            LAUNCH_TINYGEMM_KERNEL(4, 32);
            break;
          case 0x44:
            LAUNCH_TINYGEMM_KERNEL(4, 64);
            break;
          default:
            TORCH_CHECK(false, "Unexpected block size, ", width, " x ", nb_size);
        }

        // move to the next index
        data_index_step(bs, batch, nb, NB);
      }
    });
  });

#define CONV_STATE_INDEXR(w) conv_states + conv_state_index*conv_state_slot_stride + (w) * dim

  // update conv_states
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bs = begin; bs < end; ++bs) {
      // update old states, range [1, width - 1)
      int32_t conv_state_index = has_conv_indices ? conv_indices[bs] : bs;
      for (int64_t w = 1; w < width - 1; ++w) {
        std::memcpy(CONV_STATE_INDEXR(w - 1), CONV_STATE_INDEXR(w), dim * sizeof(scalar_t));
      }
      // copy new states
      std::memcpy(CONV_STATE_INDEXR(width - 2), input + bs * dim, dim * sizeof(scalar_t));
    }
  });
}

template <typename scalar_t>
void causal_conv1d_update_multi_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ conv_states,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int32_t* __restrict__ num_accepted_tokens,
    const int32_t* __restrict__ conv_indices,
    bool silu_activation,
    int64_t batch,
    int64_t dim,
    int64_t seqlen,
    int64_t width,
    int64_t state_len,
    int64_t conv_state_slot_stride) {
  constexpr int64_t BLOCK_N = block_size_n() * 2;
  const int64_t NB = div_up(dim, BLOCK_N);

  AT_DISPATCH_BOOL2(bias != nullptr, has_bias, silu_activation, has_silu, [&] {
    at::parallel_for(0, batch * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t bs{0}, nb{0};
      data_index_init(begin, bs, batch, nb, NB);

      for (int64_t i = begin; i < end; ++i) {
        const int64_t nb_start = nb * BLOCK_N;
        const int64_t nb_size = std::min(dim - nb_start, BLOCK_N);
        const int32_t conv_state_index = conv_indices[bs];
        const int32_t history_offset = num_accepted_tokens[bs] - 1;

        switch (width << 4 | nb_size >> 4) {
          case 0x42:
            tinygemm_kernel<scalar_t, 4, 32, has_bias, has_silu>::apply(
                input + bs * seqlen * dim + nb_start,
                weight + nb_start * width,
                out + bs * seqlen * dim + nb_start,
                has_bias ? bias + nb_start : nullptr,
                conv_states + conv_state_index * conv_state_slot_stride +
                    history_offset * dim + nb_start,
                true,
                seqlen,
                dim,
                true);
            break;
          case 0x44:
            tinygemm_kernel<scalar_t, 4, 64, has_bias, has_silu>::apply(
                input + bs * seqlen * dim + nb_start,
                weight + nb_start * width,
                out + bs * seqlen * dim + nb_start,
                has_bias ? bias + nb_start : nullptr,
                conv_states + conv_state_index * conv_state_slot_stride +
                    history_offset * dim + nb_start,
                true,
                seqlen,
                dim,
                true);
            break;
          default:
            TORCH_CHECK(false, "Unexpected block size, ", width, " x ", nb_size);
        }

        data_index_step(bs, batch, nb, NB);
      }
    });
  });

  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bs = begin; bs < end; ++bs) {
      const int32_t conv_state_index = conv_indices[bs];
      const int32_t num_accepted = num_accepted_tokens[bs];
      scalar_t* state = conv_states + conv_state_index * conv_state_slot_stride;

      std::memmove(
          state,
          state + num_accepted * dim,
          (state_len - seqlen) * dim * sizeof(scalar_t));
      std::memcpy(
          state + (state_len - seqlen) * dim,
          input + bs * seqlen * dim,
          seqlen * dim * sizeof(scalar_t));
    }
  });
}

at::Tensor stage_ds_conv_states(
    const at::Tensor& conv_states,
    const std::optional<at::Tensor>& conv_state_indices,
    int64_t batch,
    int64_t prefix_len) {
  // Narrow before selecting slots so a wide speculative tail is never copied.
  auto prefix_states = conv_states.narrow(2, 0, prefix_len);
  at::Tensor selected_states;
  bool identity_indices = true;
  if (conv_state_indices.has_value()) {
    const auto& indices = conv_state_indices.value();
    const int32_t* indices_data = indices.data_ptr<int32_t>();
    for (int64_t bs = 0; bs < batch; ++bs) {
      if (indices_data[bs] != bs) {
        identity_indices = false;
        break;
      }
    }
    selected_states = identity_indices
                         ? prefix_states.narrow(0, 0, batch)
                         : prefix_states.index_select(0, indices);
  } else {
    selected_states = prefix_states.narrow(0, 0, batch);
  }
  return selected_states.transpose(1, 2).contiguous();
}

at::Tensor identity_conv_state_indices(int64_t batch, const at::Tensor& conv_states) {
  return at::arange(
      batch,
      at::TensorOptions().dtype(at::kInt).device(conv_states.device()));
}

template <typename scalar_t>
void update_ds_conv_states_fwd(
    const at::Tensor& x,
    const at::Tensor& conv_states,
    const std::optional<at::Tensor>& query_start_loc,
    const std::optional<at::Tensor>& conv_state_indices,
    const at::Tensor& has_initial_state,
    int64_t batch,
    int64_t dim,
    int64_t width,
    int64_t conv_state_slot_stride) {
  const int64_t state_len = conv_states.size(2);
  const int64_t width1 = width - 1;
  const bool is_var_seqlen = query_start_loc.has_value();
  const int32_t* offsets =
      is_var_seqlen ? query_start_loc.value().data_ptr<int32_t>() : nullptr;
  const int32_t* indices =
      conv_state_indices.has_value() ? conv_state_indices.value().data_ptr<int32_t>()
                                     : nullptr;
  const bool* initial_state = has_initial_state.data_ptr<bool>();
  const scalar_t* input = x.data_ptr<scalar_t>();
  scalar_t* states = conv_states.data_ptr<scalar_t>();

  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bs = begin; bs < end; ++bs) {
      const int64_t slot = indices == nullptr ? bs : indices[bs];
      const int64_t sequence_start =
          is_var_seqlen ? offsets[bs] : bs * x.size(-1);
      const int64_t seqlen = is_var_seqlen
                                 ? offsets[bs + 1] - offsets[bs]
                                 : x.size(-1);
      scalar_t* state = states + slot * conv_state_slot_stride;
      const scalar_t* sequence = input + sequence_start * dim;

      int64_t w = 0;
      for (; w < width1 - seqlen; ++w) {
        for (int64_t d = 0; d < dim; ++d) {
          state[d * state_len + w] =
              initial_state[bs] ? state[d * state_len + w + seqlen]
                                : scalar_t(0);
        }
      }
      for (; w < width1; ++w) {
        const int64_t input_offset = w + seqlen - width1;
        for (int64_t d = 0; d < dim; ++d) {
          state[d * state_len + w] = sequence[input_offset * dim + d];
        }
      }
    }
  });
}

template <typename scalar_t>
void update_ds_conv_states_single(
    const at::Tensor& x,
    const at::Tensor& conv_states,
    const std::optional<at::Tensor>& conv_state_indices,
    int64_t batch,
    int64_t dim,
    int64_t width,
    int64_t conv_state_slot_stride) {
  const int64_t state_len = conv_states.size(2);
  const int64_t width1 = width - 1;
  const int32_t* indices =
      conv_state_indices.has_value() ? conv_state_indices.value().data_ptr<int32_t>()
                                     : nullptr;
  const scalar_t* input = x.data_ptr<scalar_t>();
  scalar_t* states = conv_states.data_ptr<scalar_t>();

  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bs = begin; bs < end; ++bs) {
      const int64_t slot = indices == nullptr ? bs : indices[bs];
      scalar_t* state = states + slot * conv_state_slot_stride;
      for (int64_t d = 0; d < dim; ++d) {
        scalar_t* row = state + d * state_len;
        for (int64_t w = 1; w < width1; ++w) {
          row[w - 1] = row[w];
        }
        row[width1 - 1] = input[bs * dim + d];
      }
    }
  });
}

template <typename scalar_t>
void update_ds_conv_states_multi(
    const at::Tensor& x,
    const at::Tensor& conv_states,
    const at::Tensor& num_accepted_tokens,
    const at::Tensor& conv_state_indices,
    int64_t batch,
    int64_t dim,
    int64_t seqlen,
    int64_t conv_state_slot_stride) {
  const int64_t state_len = conv_states.size(2);
  const int32_t* accepted = num_accepted_tokens.data_ptr<int32_t>();
  const int32_t* indices = conv_state_indices.data_ptr<int32_t>();
  const scalar_t* input = x.data_ptr<scalar_t>();
  scalar_t* states = conv_states.data_ptr<scalar_t>();

  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bs = begin; bs < end; ++bs) {
      const int64_t slot = indices[bs];
      const int64_t accepted_count = accepted[bs];
      scalar_t* state = states + slot * conv_state_slot_stride;
      for (int64_t d = 0; d < dim; ++d) {
        scalar_t* row = state + d * state_len;
        std::memmove(
            row,
            row + accepted_count,
            (state_len - seqlen) * sizeof(scalar_t));
        for (int64_t t = 0; t < seqlen; ++t) {
          row[state_len - seqlen + t] =
              input[(bs * seqlen + t) * dim + d];
        }
      }
    }
  });
}

bool any_initial_state(const at::Tensor& has_initial_state) {
  const bool* values = has_initial_state.data_ptr<bool>();
  for (int64_t i = 0; i < has_initial_state.size(0); ++i) {
    if (values[i]) {
      return true;
    }
  }
  return false;
}

}  // anonymous namespace

// from [dim, width] or [N, K]
// to [N/BLOCK_N, K/2, BLOCK_N, 2]
at::Tensor causal_conv1d_weight_pack(const at::Tensor& weight) {
  CHECK_INPUT(weight);

  int64_t dim = weight.size(0);
  int64_t width = weight.size(1);
  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(width == 4, "causal_conv1d_weight_pack: support only width of 4");
  TORCH_CHECK(dim % BLOCK_N == 0, "causal_conv1d_weight_pack: invalid dim size ", dim);

  const int64_t N = dim, K2 = width >> 1;
  const int64_t NB = div_up(N, BLOCK_N);

  auto packed_weight = at::empty_like(weight);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(weight.scalar_type(), "causal_conv1d_fwd_kernel_impl", [&] {
    // cast to float32 as vnni size is 2
    const float* w_data = reinterpret_cast<float*>(weight.data_ptr<scalar_t>());
    float* packed_data = reinterpret_cast<float*>(packed_weight.data_ptr<scalar_t>());

    at::parallel_for(0, NB * K2 * BLOCK_N, 0, [&](int64_t begin, int64_t end) {
      int64_t nb{0}, k2{0}, n{0};
      data_index_init(begin, nb, NB, k2, K2, n, BLOCK_N);

      // TODO: optimize this if we need to online prepacking.
      for (int64_t i = begin; i < end; ++i) {
        packed_data[i] = w_data[nb * BLOCK_N * K2 + n * K2 + k2];

        // move to the next index
        data_index_step(nb, NB, k2, K2, n, BLOCK_N);
      }
    });
  });
  return packed_weight;
}

#define CHECK_OPTIONAL_SHAPE_DTYPE(OPT, SIZE, DTYPE) \
  if (OPT.has_value()) {                             \
    const auto tensor = OPT.value();                 \
    CHECK_CONTIGUOUS(tensor);                        \
    CHECK_EQ(tensor.size(0), SIZE);                  \
    CHECK_EQ(tensor.scalar_type(), DTYPE);           \
  }

template <int BLOCK_M>
int64_t get_block_count(const std::optional<at::Tensor>& offsets, int64_t batch, int64_t seqlen) {
  if (offsets.has_value()) {
    const int32_t* offsets_data = offsets.value().data_ptr<int32_t>();
    int32_t num_seq_blocks = 0;
    for (int64_t row = 0; row < batch; ++row) {
      num_seq_blocks += div_up(offsets_data[row + 1] - offsets_data[row], BLOCK_M);
    }
    return num_seq_blocks;
  }
  return batch * div_up(seqlen, int64_t(BLOCK_M));
}

template <int BLOCK_M>
at::Tensor get_block_indices(const std::optional<at::Tensor>& offsets, int64_t num_seq_blocks) {
  if (!offsets.has_value()) {
    return at::Tensor();
  }

  const at::Tensor& offsets_ = offsets.value();
  at::Tensor indices = at::empty({num_seq_blocks, 2}, offsets_.options());

  int64_t batch = offsets_.size(0) - 1;

  const int32_t* offsets_data = offsets_.data_ptr<int32_t>();
  int32_t* indices_data = indices.data_ptr<int32_t>();

  int64_t idx = 0;
  for (int32_t row = 0; row < batch; ++row) {
    int32_t blocks = div_up(offsets_data[row + 1] - offsets_data[row], BLOCK_M);

    for (int32_t col = 0; col < blocks; ++col) {
      indices_data[idx * 2 + 0] = row;
      indices_data[idx * 2 + 1] = col;
      idx++;
    }
  }
  return indices;
}

// API aligned with GPUs
//
//   x: (batch, dim, seqlen) or (dim, cu_seq_len) for varlen
//   weight: (dim, width)
//   bias: (dim,)
//   query_start_loc: (batch + 1) int32
//   cache_indices: (batch)  int32
//   has_initial_state: (batch) bool
//   conv_states: (..., dim, state_len) itype, where state_len >= width - 1
//   activation: either None or "silu" or "swish"
//   pad_slot_id: int
//
at::Tensor causal_conv1d_fwd_cpu_sd(
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& query_start_loc,
    const std::optional<at::Tensor>& conv_state_indices,
    const std::optional<at::Tensor>& has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id,
    bool is_vnni) {
  CHECK_CONTIGUOUS(weight);
  auto packed_w = is_vnni ? weight : causal_conv1d_weight_pack(weight);

  const bool is_var_seqlen = query_start_loc.has_value();
  const int64_t input_ndim = is_var_seqlen ? 2 : 3;
  TORCH_CHECK(x.dim() == input_ndim, "causal_conv1d_fwd_cpu: expect x to be ", input_ndim, "D tensor.");
  TORCH_CHECK(x.stride(-2) == 1 && x.stride(-1) == x.size(-2), "causal_conv1d_fwd_cpu: expect x to be transposed.");

  const int64_t batch = is_var_seqlen ? query_start_loc.value().size(0) - 1 : x.size(0);
  const int64_t dim = x.size(-2);
  const int64_t seqlen = x.size(-1);
  const int64_t width = weight.size(-1);

  const auto scalar_type = x.scalar_type();
  CHECK_EQ(weight.scalar_type(), scalar_type);
  CHECK_OPTIONAL_SHAPE_DTYPE(bias, dim, scalar_type);
  CHECK_OPTIONAL_SHAPE_DTYPE(query_start_loc, batch + 1, at::kInt);
  CHECK_OPTIONAL_SHAPE_DTYPE(conv_state_indices, batch, at::kInt);
  CHECK_OPTIONAL_SHAPE_DTYPE(has_initial_state, batch, at::kBool);

  int64_t conv_state_slot_stride = 0;
  if (conv_states.has_value()) {
    const auto& conv_states_val = conv_states.value();
    int64_t padded_batch = conv_states_val.size(0);
    CHECK_EQ(conv_states_val.scalar_type(), scalar_type);
    CHECK_GE(padded_batch, batch);
    CHECK_EQ(conv_states_val.size(1), dim);
    CHECK_EQ(conv_states_val.stride(-2), 1);
    CHECK_EQ(conv_states_val.stride(-1), dim);
    // Preserve the physical per-slot stride used by the vLLM KV cache.
    conv_state_slot_stride = conv_states_val.stride(0);
  }

  // block size for sequence blocks, 32
  constexpr int64_t BLOCK_M = block_size_m();

  // total number of sequence blocks
  int64_t num_seq_blocks = get_block_count<BLOCK_M>(query_start_loc, batch, seqlen);

  at::Tensor out = at::empty_like(x);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(scalar_type, "causal_conv1d_fwd_kernel_impl", [&] {
    if (is_var_seqlen) {
      // record seq blocks in Coordinate format, aka [num_seq_blocks, 2]
      at::Tensor block_indices = get_block_indices<BLOCK_M>(query_start_loc, num_seq_blocks);

      causal_conv1d_fwd_varlen_kernel_impl(
          out.data_ptr<scalar_t>(),
          x.data_ptr<scalar_t>(),
          packed_w.data_ptr<scalar_t>(),
          conditional_data_ptr<scalar_t>(bias),
          conditional_data_ptr<scalar_t>(conv_states),
          conditional_data_ptr<int32_t>(query_start_loc),
          conditional_data_ptr<int32_t>(conv_state_indices),
          conditional_data_ptr<bool>(has_initial_state),
          block_indices.data_ptr<int32_t>(),
          silu_activation,
          batch,
          dim,
          width,
          num_seq_blocks,
          conv_state_slot_stride);
    } else {
      causal_conv1d_fwd_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          x.data_ptr<scalar_t>(),
          packed_w.data_ptr<scalar_t>(),
          conditional_data_ptr<scalar_t>(bias),
          conditional_data_ptr<scalar_t>(conv_states),
          conditional_data_ptr<int32_t>(conv_state_indices),
          conditional_data_ptr<bool>(has_initial_state),
          silu_activation,
          batch,
          dim,
          seqlen,
          width,
          num_seq_blocks,
          conv_state_slot_stride);
    }
  });
  return out;
}

// API aligned with GPUs
//
//   x: (batch, dim) or (batch, seqlen, dim)
//   conv_state: (..., dim, state_len), where state_len >= width - 1
//   weight: (dim, width)
//   bias: (dim,)
//   num_accepted_tokens: (batch,), dtype int32.
//   conv_state_indices: (batch,), dtype int32
//   pad_slot_id: int
//   out: (batch, dim) or (batch, seqlen, dim)
//
at::Tensor causal_conv1d_update_cpu_sd(
    const at::Tensor& x,
    const at::Tensor& conv_states,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    bool silu_activation,
    const std::optional<at::Tensor>& num_accepted_tokens,
    const std::optional<at::Tensor>& conv_state_indices,
    int64_t pad_slot_id,
    bool is_vnni) {
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight);
  auto packed_w = is_vnni ? weight : causal_conv1d_weight_pack(weight);

  TORCH_CHECK(
      x.dim() == 2 || x.dim() == 3,
      "causal_conv1d_update_cpu: expect x to be 2D or 3D tensor.");

  int64_t batch = x.size(0);
  int64_t dim = x.dim() == 2 ? x.size(1) : x.size(2);
  int64_t seqlen = x.dim() == 2 ? 1 : x.size(1);
  int64_t width = weight.size(-1);

  const auto scalar_type = x.scalar_type();
  CHECK_EQ(weight.scalar_type(), scalar_type);
  CHECK_OPTIONAL_SHAPE_DTYPE(bias, dim, scalar_type);
  CHECK_OPTIONAL_SHAPE_DTYPE(conv_state_indices, batch, at::kInt);

  CHECK_EQ(conv_states.scalar_type(), scalar_type);
  CHECK_EQ(conv_states.size(1), dim);
  CHECK_EQ(conv_states.stride(-2), 1);
  CHECK_EQ(conv_states.stride(-1), dim);
  const int64_t state_len = conv_states.size(2);
  CHECK_GE(state_len, width - 1);
  // Preserve the physical per-slot stride used by the vLLM KV cache.
  const int64_t conv_state_slot_stride = conv_states.stride(0);

  if (x.dim() == 3) {
    TORCH_CHECK(
        num_accepted_tokens.has_value(),
        "causal_conv1d_update_cpu: num_accepted_tokens is required for 3D x.");
    TORCH_CHECK(
        conv_state_indices.has_value(),
        "causal_conv1d_update_cpu: conv_state_indices is required for 3D x.");
    CHECK_OPTIONAL_SHAPE_DTYPE(num_accepted_tokens, batch, at::kInt);
    TORCH_CHECK(
        width == 4,
        "causal_conv1d_update_cpu: support only width of 4 for 3D x.");
    TORCH_CHECK(
        seqlen > 0,
        "causal_conv1d_update_cpu: expect non-empty sequence for 3D x.");
    TORCH_CHECK(
        state_len >= seqlen,
        "causal_conv1d_update_cpu: state_len must be >= seqlen for 3D x.");
    const int32_t* accepted_counts =
        num_accepted_tokens.value().data_ptr<int32_t>();
    const int32_t* indices = conv_state_indices.value().data_ptr<int32_t>();
    const int64_t num_slots = conv_states.size(0);
    for (int64_t bs = 0; bs < batch; ++bs) {
      const int32_t num_accepted = accepted_counts[bs];
      const int32_t conv_state_index = indices[bs];
      TORCH_CHECK(
          conv_state_index != pad_slot_id,
          "causal_conv1d_update_cpu: 3D x does not support pad slots.");
      TORCH_CHECK(
          conv_state_index >= 0 && conv_state_index < num_slots,
          "causal_conv1d_update_cpu: conv_state_indices out of range.");
      TORCH_CHECK(
          num_accepted >= 1 && num_accepted <= seqlen,
          "causal_conv1d_update_cpu: num_accepted_tokens must be in [1, "
          "seqlen].");
      TORCH_CHECK(
          num_accepted - 1 + width - 1 <= state_len,
          "causal_conv1d_update_cpu: history window exceeds conv_states.");
    }

    at::Tensor out = at::empty_like(x);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        scalar_type, "causal_conv1d_update_multi_kernel_impl", [&] {
          causal_conv1d_update_multi_kernel_impl<scalar_t>(
              out.data_ptr<scalar_t>(),
              x.data_ptr<scalar_t>(),
              conv_states.data_ptr<scalar_t>(),
              packed_w.data_ptr<scalar_t>(),
              conditional_data_ptr<scalar_t>(bias),
              accepted_counts,
              indices,
              silu_activation,
              batch,
              dim,
              seqlen,
              width,
              state_len,
              conv_state_slot_stride);
        });
    return out;
  }

  TORCH_CHECK(
      !num_accepted_tokens.has_value(),
      "causal_conv1d_update_cpu: num_accepted_tokens is only supported for 3D "
      "x.");

  at::Tensor out = at::empty_like(x);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(scalar_type, "causal_conv1d_update_kernel_impl", [&] {
    causal_conv1d_update_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        conv_states.data_ptr<scalar_t>(),
        packed_w.data_ptr<scalar_t>(),
        conditional_data_ptr<scalar_t>(bias),
        conditional_data_ptr<int32_t>(conv_state_indices),
        silu_activation,
        batch,
        dim,
        seqlen,
        width,
        conv_state_slot_stride);
  });
  return out;
}

at::Tensor causal_conv1d_fwd_cpu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& query_start_loc,
    const std::optional<at::Tensor>& conv_state_indices,
    const std::optional<at::Tensor>& has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id,
    bool is_vnni) {
  const bool is_var_seqlen = query_start_loc.has_value();
  const int64_t input_ndim = is_var_seqlen ? 2 : 3;
  TORCH_CHECK(
      x.dim() == input_ndim,
      "causal_conv1d_fwd_cpu: expect x to be ",
      input_ndim,
      "D tensor.");
  TORCH_CHECK(
      x.stride(-2) == 1 && x.stride(-1) == x.size(-2),
      "causal_conv1d_fwd_cpu: expect x to be transposed.");
  if (!is_var_seqlen) {
    TORCH_CHECK(
        x.stride(0) == x.size(-2) * x.size(-1),
        "causal_conv1d_fwd_cpu: expect the batch dimension to be dense.");
  } else {
    CHECK_CONTIGUOUS(query_start_loc.value());
    TORCH_CHECK(query_start_loc.value().dim() == 1);
    CHECK_EQ(query_start_loc.value().scalar_type(), at::kInt);
    TORCH_CHECK(query_start_loc.value().size(0) >= 1);
  }

  const int64_t batch =
      is_var_seqlen ? query_start_loc.value().size(0) - 1 : x.size(0);
  const int64_t dim = x.size(-2);
  const int64_t seqlen = x.size(-1);
  const int64_t width = weight.size(-1);
  if (!conv_states.has_value()) {
    return causal_conv1d_fwd_cpu_sd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        conv_state_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
        is_vnni);
  }

  const auto& states = conv_states.value();
  TORCH_CHECK(states.dim() == 3, "causal_conv1d_fwd_cpu: conv_states must be 3D.");
  CHECK_EQ(states.scalar_type(), x.scalar_type());
  CHECK_GE(states.size(0), batch);
  CHECK_EQ(states.size(1), dim);
  const auto layout_info =
      validate_conv_state_layout(states, dim, width, "causal_conv1d_fwd_cpu");
  TORCH_CHECK(
      has_initial_state.has_value(),
      "causal_conv1d_fwd_cpu: has_initial_state is required with conv_states.");
  CHECK_OPTIONAL_SHAPE_DTYPE(has_initial_state, batch, at::kBool);

  if (layout_info.layout == ConvStateLayout::SD) {
    return causal_conv1d_fwd_cpu_sd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        conv_state_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
        is_vnni);
  }

  const bool reads_initial_state = any_initial_state(has_initial_state.value());
  at::Tensor out;
  if (!reads_initial_state) {
    out = causal_conv1d_fwd_cpu_sd(
        x,
        weight,
        bias,
        std::nullopt,
        query_start_loc,
        std::nullopt,
        std::nullopt,
        silu_activation,
        pad_slot_id,
        is_vnni);
  } else {
    at::Tensor staged_storage =
        stage_ds_conv_states(states, conv_state_indices, batch, width - 1);
    at::Tensor staged_states = staged_storage.transpose(1, 2);
    out = causal_conv1d_fwd_cpu_sd(
        x,
        weight,
        bias,
        staged_states,
        query_start_loc,
        std::nullopt,
        has_initial_state,
        silu_activation,
        pad_slot_id,
        is_vnni);
  }

  const int64_t state_slot_stride = states.stride(0);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      x.scalar_type(), "update_ds_conv_states_fwd", [&] {
        update_ds_conv_states_fwd<scalar_t>(
            x,
            states,
            query_start_loc,
            conv_state_indices,
            has_initial_state.value(),
            batch,
            dim,
            width,
            state_slot_stride);
      });
  return out;
}

at::Tensor causal_conv1d_update_cpu(
    const at::Tensor& x,
    const at::Tensor& conv_states,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    bool silu_activation,
    const std::optional<at::Tensor>& num_accepted_tokens,
    const std::optional<at::Tensor>& conv_state_indices,
    int64_t pad_slot_id,
    bool is_vnni) {
  TORCH_CHECK(
      x.dim() == 2 || x.dim() == 3,
      "causal_conv1d_update_cpu: expect x to be 2D or 3D tensor.");
  const int64_t batch = x.size(0);
  const int64_t dim = x.dim() == 2 ? x.size(1) : x.size(2);
  const int64_t seqlen = x.dim() == 2 ? 1 : x.size(1);
  const int64_t width = weight.size(-1);

  TORCH_CHECK(conv_states.dim() == 3, "causal_conv1d_update_cpu: conv_states must be 3D.");
  CHECK_EQ(conv_states.scalar_type(), x.scalar_type());
  CHECK_EQ(conv_states.size(1), dim);
  const auto layout_info = validate_conv_state_layout(
      conv_states, dim, width, "causal_conv1d_update_cpu");
  const int64_t state_len = layout_info.state_len;
  if (x.dim() == 3) {
    TORCH_CHECK(
        num_accepted_tokens.has_value(),
        "causal_conv1d_update_cpu: num_accepted_tokens is required for 3D x.");
    TORCH_CHECK(
        conv_state_indices.has_value(),
        "causal_conv1d_update_cpu: conv_state_indices is required for 3D x.");
    CHECK_OPTIONAL_SHAPE_DTYPE(num_accepted_tokens, batch, at::kInt);
    TORCH_CHECK(
        width == 4,
        "causal_conv1d_update_cpu: support only width of 4 for 3D x.");
    TORCH_CHECK(
        seqlen > 0,
        "causal_conv1d_update_cpu: expect non-empty sequence for 3D x.");
    TORCH_CHECK(
        state_len >= seqlen,
        "causal_conv1d_update_cpu: state_len must be >= seqlen for 3D x.");
    const int32_t* accepted_counts =
        num_accepted_tokens.value().data_ptr<int32_t>();
    const int32_t* indices = conv_state_indices.value().data_ptr<int32_t>();
    for (int64_t bs = 0; bs < batch; ++bs) {
      const int32_t num_accepted = accepted_counts[bs];
      TORCH_CHECK(
          num_accepted >= 1 && num_accepted <= seqlen,
          "causal_conv1d_update_cpu: num_accepted_tokens must be in [1, "
          "seqlen].");
      TORCH_CHECK(
          num_accepted - 1 + width - 1 <= state_len,
          "causal_conv1d_update_cpu: history window exceeds conv_states.");
    }
  } else {
    TORCH_CHECK(
        !num_accepted_tokens.has_value(),
        "causal_conv1d_update_cpu: num_accepted_tokens is only supported "
        "for 3D x.");
  }

  if (layout_info.layout == ConvStateLayout::SD) {
    return causal_conv1d_update_cpu_sd(
        x,
        conv_states,
        weight,
        bias,
        silu_activation,
        num_accepted_tokens,
        conv_state_indices,
        pad_slot_id,
        is_vnni);
  }

  const int64_t scratch_len =
      x.dim() == 2
          ? width - 1
          : [&] {
              const int32_t* accepted =
                  num_accepted_tokens.value().data_ptr<int32_t>();
              int64_t max_accepted = 0;
              for (int64_t bs = 0; bs < batch; ++bs) {
                max_accepted = std::max<int64_t>(max_accepted, accepted[bs]);
              }
              return std::max<int64_t>(
                  seqlen, max_accepted + width - 2);
            }();
  at::Tensor staged_storage =
      stage_ds_conv_states(conv_states, conv_state_indices, batch, scratch_len);
  at::Tensor staged_states = staged_storage.transpose(1, 2);
  std::optional<at::Tensor> scratch_indices = std::nullopt;
  if (x.dim() == 3) {
    scratch_indices = identity_conv_state_indices(batch, conv_states);
  }
  at::Tensor out = causal_conv1d_update_cpu_sd(
      x,
      staged_states,
      weight,
      bias,
      silu_activation,
      num_accepted_tokens,
      scratch_indices,
      pad_slot_id,
      is_vnni);

  const int64_t state_slot_stride = conv_states.stride(0);
  if (x.dim() == 2) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        x.scalar_type(), "update_ds_conv_states_single", [&] {
          update_ds_conv_states_single<scalar_t>(
              x,
              conv_states,
              conv_state_indices,
              batch,
              dim,
              width,
              state_slot_stride);
        });
  } else {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        x.scalar_type(), "update_ds_conv_states_multi", [&] {
          update_ds_conv_states_multi<scalar_t>(
              x,
              conv_states,
              num_accepted_tokens.value(),
              conv_state_indices.value(),
              batch,
              dim,
              seqlen,
              state_slot_stride);
        });
  }
  return out;
}
