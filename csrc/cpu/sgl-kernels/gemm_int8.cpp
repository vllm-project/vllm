// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include "common.h"
#include "dense_int8_plan.h"
#include "../utils.hpp"
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "gemm.h"
#include "vec.h"

namespace {

template <typename scalar_t, bool has_bias, int BLOCK_N, bool stream_output = false>
struct scale_C {
  static inline void apply(
      scalar_t* __restrict__ C,
      const int32_t* __restrict__ Ctmp,
      const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias,
      float As,
      const float* __restrict__ Bs) {
    TORCH_CHECK(false, "scale_C: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <bool has_bias, int BLOCK_N, bool stream_output>
struct scale_C<at::BFloat16, has_bias, BLOCK_N, stream_output> {
  static inline void apply(
      at::BFloat16* __restrict__ C,
      const int32_t* __restrict__ Ctmp,
      const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias,
      float As,
      const float* __restrict__ Bs) {
    constexpr int COLS = BLOCK_N / 16;
    static_assert(COLS % 2 == 0);

    __m512 vc[COLS];
    __m512 vd0 = _mm512_set1_ps(As);

    auto compute = [&](auto col) {
      __m512 vd1 = _mm512_loadu_ps(Bs + col * 16);
      __m512i vcomp = _mm512_loadu_si512(Bcomp + col * 16);
      __m512i vc32 = _mm512_loadu_si512(Ctmp + col * 16);
      vc[col] = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc32, vcomp));
      if constexpr (has_bias) {
        __m512 vbias = _mm512_loadu_ps(bias + col * 16);
        vc[col] = _mm512_fmadd_ps(_mm512_mul_ps(vc[col], vd0), vd1, vbias);
      } else {
        vc[col] = _mm512_mul_ps(_mm512_mul_ps(vc[col], vd0), vd1);
      }
    };
    Unroll<COLS>{}(compute);

    auto storec = [&](auto col) {
      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        const __m512i result = (__m512i)(_mm512_cvtne2ps_pbh(vc[col + 1], vc[col + 0]));
        if constexpr (stream_output) {
          _mm512_stream_si512(reinterpret_cast<__m512i*>(C + col * 16), result);
        } else {
          _mm512_storeu_si512(reinterpret_cast<__m512i*>(C + col * 16), result);
        }
      }
    };
    Unroll<COLS>{}(storec);
  }
};
#endif

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const uint8_t* __restrict__ A,
      const int8_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const float* __restrict__ As,
      const float* __restrict__ Bs,
      const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const uint8_t* __restrict__ A,
      const int8_t* __restrict__ B,
      at::BFloat16* __restrict__ C,
      const float* __restrict__ As,
      const float* __restrict__ Bs,
      const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    static_assert(COLS % 2 == 0);

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512i va;
    __m512i vb[COLS];
    __m512i vc[ROWS * COLS];
    __m512i vcomp[COLS];
    __m512 vd0;
    __m512 vd1[COLS];

    // oops! 4x4 spills but we use 4x2
    __m512 vbias[COLS];

    // [NOTE]: s8s8 igemm compensation in avx512-vnni
    //
    // avx512-vnni has no s8s8, so we need to change s8s8 to u8s8 with compensate:
    //
    //   a * b = (a + 128) * b - 128 * b
    //   s   s       u       s    u    s
    //
    // 1) 128 * b is pre-computed when packing B to vnni formats
    // 2) a + 128 is fused when dynamically quantize A
    //
    auto loadc = [&](auto i) { vc[i] = _mm512_set1_epi32(0); };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K4 = K >> 2;
    const int64_t lda4 = lda >> 2;
    const int64_t ldb4 = ldb;  // ldb * 4 >> 2;
    const int32_t* a_ptr = reinterpret_cast<const int32_t*>(A);
    const int32_t* b_ptr = reinterpret_cast<const int32_t*>(B);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = _mm512_set1_epi32(a_ptr[row * lda4 + k]);
      }
      if constexpr (row == 0) {
        vb[col] = _mm512_loadu_si512(b_ptr + k * ldb4 + col * 16);
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb4 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
    };
    for (int64_t k = 0; k < K4; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      // load a scale
      if constexpr (col == 0) {
        vd0 = _mm512_set1_ps(As[row]);
      }
      // load b scale and vcomp per 2 vectors
      // also load bias if any
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          vd1[col + 0] = _mm512_loadu_ps(Bs + col * 16);
          vd1[col + 1] = _mm512_loadu_ps(Bs + col * 16 + 16);
          vcomp[col + 0] = _mm512_loadu_si512(Bcomp + col * 16);
          vcomp[col + 1] = _mm512_loadu_si512(Bcomp + col * 16 + 16);
          if constexpr (has_bias) {
            vbias[col + 0] = _mm512_loadu_ps(bias + col * 16);
            vbias[col + 1] = _mm512_loadu_ps(bias + col * 16 + 16);
          }
        }
      }

      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        __m512 vc0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc[row * COLS + col + 0], vcomp[col + 0]));
        __m512 vc1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc[row * COLS + col + 1], vcomp[col + 1]));
        if constexpr (has_bias) {
          vc0 = _mm512_fmadd_ps(_mm512_mul_ps(vc0, vd0), vd1[col + 0], vbias[col + 0]);
          vc1 = _mm512_fmadd_ps(_mm512_mul_ps(vc1, vd0), vd1[col + 1], vbias[col + 1]);
        } else {
          vc0 = _mm512_mul_ps(_mm512_mul_ps(vc0, vd0), vd1[col + 0]);
          vc1 = _mm512_mul_ps(_mm512_mul_ps(vc1, vd0), vd1[col + 1]);
        }

        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)), (__m512i)(_mm512_cvtne2ps_pbh(vc1, vc0)));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                \
  tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                          \
      B + nb_start * 4,                                            \
      C + mb_start * ldc + nb_start,                               \
      As + mb_start,                                               \
      Bs + nb_start,                                               \
      Bcomp + nb_start,                                            \
      has_bias ? bias + nb_start : nullptr,                        \
      K,                                                           \
      lda,                                                         \
      ldb,                                                         \
      ldc);

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int32_t* __restrict__ Ctmp,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  // B compensation
  const int32_t* Bcomp = reinterpret_cast<const int32_t*>(B + block_size_n() * K);

  if (brg) {
    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(M, N, K, lda, ldb, BLOCK_N, /* add_C */ false, A, B, Ctmp);

    // apply compensation and scale
    for (int64_t m = 0; m < M; ++m) {
      scale_C<scalar_t, has_bias, BLOCK_N>::apply(C + m * ldc, Ctmp + m * BLOCK_N, Bcomp, bias, As[m], Bs);
    }
    return;
  }

  // pattern: 1-4-16
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 64);
          break;
        // mb_size = 2
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 64);
          break;
        // mb_size = 3
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 64);
          break;
        // mb_size = 4
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 64);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t>
void int8_scaled_mm_kernel_impl(
    scalar_t* __restrict__ out,
    const uint8_t* __restrict__ mat1,
    const int8_t* __restrict__ mat2,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  const bool use_brgemm = can_use_brgemm<int8_t>(M);

  // K + 4 after compensation
  const int64_t packed_row_size = get_row_size<int8_t>(K);

  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
      // for brgemm, use int32_t for accumulate
      alignas(64) int32_t Ctmp[BLOCK_M * BLOCK_N];

      loop_2d<int8_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
        int mb_start = mb * BLOCK_M;
        int mb_size = std::min(M - mb_start, BLOCK_M);
        int nb_start = nb * BLOCK_N;
        int nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, has_bias>(
            /*   A */ mat1 + mb_start * K,
            /*   B */ mat2 + nb_start * packed_row_size /* nb * BLOCK_N * (K + 4) */,
            /*   C */ out + mb_start * N + nb_start,
            /* Ctmp*/ Ctmp,
            /*  As */ scales1 + mb_start,
            /*  Bs */ scales2 + nb_start,
            /* bias*/ bias + nb_start,
            /*   M */ mb_size,
            /*   N */ nb_size,
            /*   K */ K,
            /* lda */ K,
            /* ldb */ nb_size,
            /* ldc */ N,
            /* brg */ use_brgemm);
      });

      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });
}

#if defined(CPU_CAPABILITY_AVX512)

using sgl::dense_int8::DenseCoarsePlan;
using sgl::dense_int8::DenseSmallGridPlan;
using sgl::dense_int8::partition_begin;

template <typename T>
struct AlignedScratch {
  T* ptr = nullptr;
  size_t capacity = 0;

  ~AlignedScratch() { std::free(ptr); }

  T* get(size_t elements) {
    if (elements > capacity) {
      const size_t bytes = ((elements * sizeof(T) + 63) / 64) * 64;
      void* mem = nullptr;
      if (posix_memalign(&mem, 64, bytes) != 0 || mem == nullptr) {
        return nullptr;
      }
      std::free(ptr);
      ptr = static_cast<T*>(mem);
      capacity = bytes / sizeof(T);
    }
    return ptr;
  }
};

template <typename T>
T* worker_scratch(int64_t elements) {
  thread_local AlignedScratch<T> scratch;
  return scratch.get(static_cast<size_t>(elements));
}

inline void omp_team_barrier() {
#if defined(_OPENMP)
  #pragma omp barrier
#endif
}

template <typename Func>
inline void run_omp_team(int64_t team, const Func& fn) {
#if defined(_OPENMP)
  if (!at::in_parallel_region() && !omp_in_parallel() && team > 1) {
  #pragma omp parallel num_threads(team)
    {
      fn(omp_get_num_threads(), omp_get_thread_num());
    }
  } else
#endif
  {
    fn(1, 0);
  }
}

inline bool is_64b_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) % 64) == 0;
}

struct ActivationBuffer {
  static constexpr int64_t kMaxCachedBytes = 160LL * 1024 * 1024;
  static constexpr int kRingDepth = 2;

  struct Cache {
    // 2-slot alternating ring avoids cross-core L2 RFO snoop storms when
    // quantize_row_int8 overwrites the buffer right after all threads read it.
    std::array<at::Tensor, kRingDepth> aq;
    std::array<at::Tensor, kRingDepth> as;
    int cursor = 0;
    bool in_use = false;
  };

  at::Tensor transient_aq;
  at::Tensor transient_as;
  Cache* leased = nullptr;
  uint8_t* aq = nullptr;
  float* as = nullptr;

  ActivationBuffer(int64_t m, int64_t a_stride,
                   const at::TensorOptions& options) {
    static thread_local Cache cache;
    const bool use_cache =
        !cache.in_use && m > 0 && a_stride > 0 &&
        m <= kMaxCachedBytes / a_stride &&
        m * a_stride <= kMaxCachedBytes - m * int64_t(sizeof(float));
    if (!use_cache) {
      transient_aq = at::empty({m * a_stride}, options.dtype(at::kByte));
      transient_as = at::empty({m}, options.dtype(at::kFloat));
      aq = transient_aq.data_ptr<uint8_t>();
      as = transient_as.data_ptr<float>();
      return;
    }
    const int slot = cache.cursor;
    cache.cursor = (cache.cursor + 1) % kRingDepth;
    const int64_t need_aq = m * a_stride;
    auto& slot_aq = cache.aq[slot];
    auto& slot_as = cache.as[slot];
    if (!slot_aq.defined() || slot_aq.numel() < need_aq) {
      const int64_t cap = std::min(
          kMaxCachedBytes,
          std::max(need_aq, slot_aq.defined() ? slot_aq.numel() * 2 : need_aq));
      slot_aq = at::empty({cap}, options.dtype(at::kByte));
    }
    if (!slot_as.defined() || slot_as.numel() < m) {
      const int64_t cap =
          std::max(m, slot_as.defined() ? slot_as.numel() * 2 : m);
      slot_as = at::empty({cap}, options.dtype(at::kFloat));
    }
    cache.in_use = true;
    leased = &cache;
    aq = slot_aq.data_ptr<uint8_t>();
    as = slot_as.data_ptr<float>();
  }

  ~ActivationBuffer() {
    if (leased) {
      leased->in_use = false;
    }
  }
  ActivationBuffer(const ActivationBuffer&) = delete;
  ActivationBuffer& operator=(const ActivationBuffer&) = delete;
};

template <bool has_bias, bool stream_output>
void dense_brgemm_tile(
    const DenseCoarsePlan& plan, at::BFloat16* out, const uint8_t* A,
    const int8_t* B, const float* As, const float* Bs, const float* bias,
    int32_t* Ctmp, int64_t tile) {
  const auto& kernel = plan.kernel;
  const int64_t cs = kernel.c_stride();
  const auto bounds = plan.tile_bounds(tile);
  const int64_t mc = bounds.m, nc = bounds.n;
  const int64_t rows = bounds.rows, cols = bounds.cols;
  const int64_t block = kernel.k_block;

  for (int64_t kb = 0; kb < kernel.k; kb += block) {
    const int64_t depth = std::min(block, kernel.k - kb);
    const bool add_c = kb > 0;
    const uint8_t* ap = A + mc * kernel.a_stride + kb;
    for (int64_t n = 0; n < cols; n += 32) {
      const int8_t* bp = B + (nc + n) * kernel.b_stride + kb * 32;
      at::native::cpublas::brgemm(
          rows, 32, depth, kernel.a_stride, 32, cs, add_c,
          ap, bp, Ctmp + n);
    }
  }
  for (int64_t n = 0; n < cols; n += 32) {
    const auto* comp = reinterpret_cast<const int32_t*>(
        B + (nc + n) * kernel.b_stride + kernel.k * 32);
    for (int64_t m = 0; m < rows; ++m) {
      scale_C<at::BFloat16, has_bias, 32, stream_output>::apply(
          out + (mc + m) * kernel.output_stride + nc + n,
          Ctmp + m * cs + n, comp,
          has_bias ? bias + nc + n : nullptr, As[mc + m], Bs + nc + n);
    }
  }
}

enum class SmallGridResult { kNotPreferred, kExecuted, kFallbackToMain };

SmallGridResult dense_small_grid_gemm(
    at::Tensor& mat1, at::Tensor& packed_w, at::Tensor& scales2,
    const float* bias, at::Tensor& out,
    int64_t M, int64_t N, int64_t K, int64_t l2_bytes) {
  at::internal::lazy_init_num_threads();
  int64_t team = at::get_num_threads();

  for (int attempt = 0; attempt < 2; ++attempt) {
    if (!DenseSmallGridPlan::preferred(M, N, K, team, l2_bytes)) {
      return SmallGridResult::kNotPreferred;
    }
    const DenseSmallGridPlan plan =
        DenseSmallGridPlan::cached_select(M, N, K, team, l2_bytes);
    if (!plan.valid) {
      return SmallGridResult::kNotPreferred;
    }

    const at::BFloat16* src_a = mat1.data_ptr<at::BFloat16>();
    const int64_t lda = mat1.stride(0);
    const int8_t* B = packed_w.data_ptr<int8_t>();
    const float* Bs = scales2.data_ptr<float>();

    ActivationBuffer act_buf(M, plan.a_stride, mat1.options());
    uint8_t* aq_ptr = act_buf.aq;
    float* as_ptr = act_buf.as;

    out = at::empty({M, N}, mat1.options());
    at::BFloat16* out_ptr = out.data_ptr<at::BFloat16>();
    const bool use_stream_output = plan.stream_output && is_64b_aligned(out_ptr);
    int32_t* stack_slots[256];
    std::vector<int32_t*> heap_slots;
    int32_t** worker_slots = nullptr;
    if (plan.gk > 1) {
      if (plan.workers() <= 256) {
        worker_slots = stack_slots;
      } else {
        heap_slots.assign(plan.workers(), nullptr);
        worker_slots = heap_slots.data();
      }
    }
    std::atomic<bool> scratch_failed{false};
    int64_t observed_team = 0;

    auto execute = [&](int64_t actual_team, int64_t worker) {
      if (actual_team != team) {
        if (worker == 0) {
          observed_team = actual_team;
        }
        return;
      }

      const int64_t begin = partition_begin(M, actual_team, worker);
      const int64_t end = partition_begin(M, actual_team, worker + 1);
      for (int64_t m = begin; m < end; ++m) {
        quantize_row_int8<at::BFloat16>(
            aq_ptr + m * plan.a_stride, as_ptr[m], src_a + m * lda, K);
      }
      omp_team_barrier();

      if (worker >= plan.workers()) {
        if (plan.gk > 1) {
          omp_team_barrier();
        }
        return;
      }

      const auto b = plan.block(worker);
      const int64_t rows = b.r1 - b.r0;
      const int64_t panels = b.p1 - b.p0;
      const uint8_t* A_base = aq_ptr + b.r0 * plan.a_stride;
      const float* As_base = as_ptr + b.r0;

      at::internal::ThreadIdGuard thread_guard(worker);
      c10::ParallelGuard parallel_guard(true);

      int32_t* work = worker_scratch<int32_t>(plan.max_slot_elems());
      if (work == nullptr) {
        scratch_failed.store(true, std::memory_order_relaxed);
        if (plan.gk > 1) {
          omp_team_barrier();
        }
        return;
      }
      if (plan.gk > 1) {
        worker_slots[worker] = work;
      }

      const int64_t panel_stride = plan.panel_elems(rows);

      auto run_brgemm_chunk = [&](int64_t p_begin, int64_t p_end) {
        for (int64_t kb = b.k0; kb < b.k1; kb += plan.k_block) {
          const int64_t depth = std::min(plan.k_block, b.k1 - kb);
          const bool add_c = kb > b.k0;
          const uint8_t* ap = A_base + kb;
          for (int64_t p = p_begin; p < p_end; ++p) {
            const int64_t p_rel = p - p_begin;
            const int8_t* bp = B + p * 32 * plan.b_stride + kb * 32;
            at::native::cpublas::brgemm(
                rows, 32, depth, plan.a_stride, 32, 32, add_c,
                ap, bp, work + p_rel * panel_stride);
          }
        }
      };

      if (plan.gk == 1) {
        AT_DISPATCH_BOOL2(bias != nullptr, has_bias, use_stream_output, stream_output, [&] {
          for (int64_t p0 = b.p0; p0 < b.p1; p0 += plan.n_chunk) {
            const int64_t p1 = std::min(b.p1, p0 + plan.n_chunk);
            run_brgemm_chunk(p0, p1);
            for (int64_t p = p0; p < p1; ++p) {
              const int64_t p_rel = p - p0;
              const int64_t n0 = p * 32;
              const auto* comp = reinterpret_cast<const int32_t*>(
                  B + n0 * plan.b_stride + K * 32);
              const int32_t* panel_c = work + p_rel * panel_stride;
              for (int64_t r = 0; r < rows; ++r) {
                scale_C<at::BFloat16, has_bias, 32, stream_output>::apply(
                    out_ptr + (b.r0 + r) * N + n0, panel_c + r * 32, comp,
                    has_bias ? bias + n0 : nullptr, As_base[r], Bs + n0);
              }
            }
          }
        });
        at::native::cpublas::brgemm_release();
        if (use_stream_output) {
          _mm_sfence();
        }
        return;
      }

      run_brgemm_chunk(b.p0, b.p1);
      at::native::cpublas::brgemm_release();
      omp_team_barrier();
      if (scratch_failed.load(std::memory_order_relaxed)) {
        return;
      }

      const int64_t group_base = worker - b.ik;
      const int64_t total_out = panels * rows;
      const int64_t t0 = partition_begin(total_out, plan.gk, b.ik);
      const int64_t t1 = partition_begin(total_out, plan.gk, b.ik + 1);
      AT_DISPATCH_BOOL2(bias != nullptr, has_bias, use_stream_output, stream_output, [&] {
        alignas(64) int32_t sums[32];
        for (int64_t t = t0; t < t1; ++t) {
          const int64_t p_rel = t / rows;
          const int64_t r = t % rows;
          const int64_t n0 = (b.p0 + p_rel) * 32;
          const auto* comp = reinterpret_cast<const int32_t*>(
              B + n0 * plan.b_stride + K * 32);
          const int64_t elem_off = p_rel * panel_stride + r * 32;
          __m512i s0 = _mm512_setzero_si512();
          __m512i s1 = _mm512_setzero_si512();
          for (int64_t slot = 0; slot < plan.gk; ++slot) {
            const int32_t* src_c = worker_slots[group_base + slot] + elem_off;
            s0 = _mm512_add_epi32(s0, _mm512_loadu_si512(src_c + 0));
            s1 = _mm512_add_epi32(s1, _mm512_loadu_si512(src_c + 16));
          }
          _mm512_store_si512(sums + 0, s0);
          _mm512_store_si512(sums + 16, s1);
          scale_C<at::BFloat16, has_bias, 32, stream_output>::apply(
              out_ptr + (b.r0 + r) * N + n0, sums, comp,
              has_bias ? bias + n0 : nullptr, As_base[r], Bs + n0);
        }
      });
      if (use_stream_output) {
        _mm_sfence();
      }
    };

    run_omp_team(team, execute);
    if (observed_team > 0) {
      team = observed_team;
      continue;
    }
    return scratch_failed.load(std::memory_order_relaxed)
               ? SmallGridResult::kFallbackToMain
               : SmallGridResult::kExecuted;
  }
  return SmallGridResult::kNotPreferred;
}

bool dense_coarse_gemm(
    at::Tensor& mat1, at::Tensor& packed_w, at::Tensor& scales2,
    const std::optional<at::Tensor>& bias_tensor, at::Tensor& out,
    int64_t M, int64_t N, int64_t K) {
  const int64_t l2_bytes = 2 * cpu_utils::get_available_l2_size();
  if (mat1.scalar_type() != at::kBFloat16 ||
      at::in_parallel_region() ||
#if defined(_OPENMP)
      omp_in_parallel() ||
#endif
      !DenseCoarsePlan::l2_eligible(l2_bytes)) {
    return false;
  }
  const float* bias = nullptr;
  if (bias_tensor.has_value()) {
    CHECK_EQ(bias_tensor.value().size(0), N);
    bias = bias_tensor.value().data_ptr<float>();
  }
  if (brgemm_supported() && DenseSmallGridPlan::shape_eligible(M, N, K)) {
    const auto small_result = dense_small_grid_gemm(
        mat1, packed_w, scales2, bias, out, M, N, K, l2_bytes);
    if (small_result == SmallGridResult::kExecuted) {
      return true;
    }
    if (small_result == SmallGridResult::kFallbackToMain) {
      return false;
    }
  }
  if (!can_use_brgemm<int8_t>(M) || !DenseCoarsePlan::shape_eligible(M, N, K)) {
    return false;
  }

  at::internal::lazy_init_num_threads();
  int64_t team = at::get_num_threads();
  for (int attempt = 0; attempt < 2; ++attempt) {
    const DenseCoarsePlan plan =
        DenseCoarsePlan::cached_select(M, N, K, team, l2_bytes);
    if (!plan.valid) {
      return false;
    }

    ActivationBuffer act_buf(M, plan.kernel.a_stride, mat1.options());
    uint8_t* Aq = act_buf.aq;
    float* As = act_buf.as;

    out = at::empty({M, N}, mat1.options());
    at::BFloat16* out_ptr = out.data_ptr<at::BFloat16>();
    const bool use_stream_output =
        plan.kernel.stream_output && is_64b_aligned(out_ptr);
    const at::BFloat16* src_a = mat1.data_ptr<at::BFloat16>();
    const int64_t lda = mat1.stride(0);
    const int8_t* B = packed_w.data_ptr<int8_t>();
    const float* Bs = scales2.data_ptr<float>();

    std::atomic<bool> scratch_failed{false};
    int64_t observed_team = 0;

    auto execute = [&](int64_t actual_team, int64_t worker) {
      if (actual_team != team) {
        if (worker == 0) {
          observed_team = actual_team;
        }
        return;
      }
      int32_t* work = worker_scratch<int32_t>(plan.kernel.worker_scratch());
      if (work == nullptr) {
        scratch_failed.store(true, std::memory_order_relaxed);
      } else {
        const int64_t begin = partition_begin(M, actual_team, worker);
        const int64_t end = partition_begin(M, actual_team, worker + 1);
        for (int64_t m = begin; m < end; ++m) {
          quantize_row_int8<at::BFloat16>(
              Aq + m * plan.kernel.a_stride, As[m], src_a + m * lda, K);
        }
      }
      omp_team_barrier();
      if (scratch_failed.load(std::memory_order_relaxed)) {
        return;
      }

      at::internal::ThreadIdGuard thread_guard(worker);
      c10::ParallelGuard parallel_guard(true);
      AT_DISPATCH_BOOL2(bias != nullptr, has_bias, use_stream_output, stream_output, [&] {
        plan.each_macro_tile(worker, [&](int64_t tile) {
          dense_brgemm_tile<has_bias, stream_output>(
              plan, out_ptr, Aq, B, As, Bs, bias, work, tile);
        });
      });
      at::native::cpublas::brgemm_release();
      if (use_stream_output) {
        _mm_sfence();
      }
    };

    run_omp_team(team, execute);
    if (observed_team > 0) {
      team = observed_team;
      continue;
    }
    return !scratch_failed.load(std::memory_order_relaxed);
  }
  return false;
}

#endif

}  // anonymous namespace

// tinygemm interface
template <typename scalar_t>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int32_t* __restrict__ Ctmp,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  tinygemm_kernel<scalar_t, false>(A, B, C, Ctmp, As, Bs, nullptr, M, N, K, lda, ldb, ldc, brg);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE) \
  template void tinygemm_kernel<TYPE>(      \
      const uint8_t* __restrict__ A,        \
      const int8_t* __restrict__ B,         \
      TYPE* __restrict__ C,                 \
      int32_t* __restrict__ Ctmp,           \
      const float* __restrict__ As,         \
      const float* __restrict__ Bs,         \
      int64_t M,                            \
      int64_t N,                            \
      int64_t K,                            \
      int64_t lda,                          \
      int64_t ldb,                          \
      int64_t ldc,                          \
      bool brg)

INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

std::tuple<at::Tensor, at::Tensor> per_token_quant_int8_cpu(at::Tensor& A) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(A);
  CHECK_DIM(2, A);

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t lda = A.stride(0);

  const auto st = A.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf, "per_token_quant_int8: expect A to be bfloat16 or half.");

  auto Aq = at::empty({M, K}, A.options().dtype(at::kByte));
  auto As = at::empty({M}, A.options().dtype(at::kFloat));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "per_token_quant_int8", [&] {
    uint8_t* __restrict__ Aq_data = Aq.data_ptr<uint8_t>();
    float* __restrict__ As_data = As.data_ptr<float>();
    const scalar_t* __restrict__ A_data = A.data_ptr<scalar_t>();

    at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        quantize_row_int8<scalar_t>(Aq_data + m * K, As_data[m], A_data + m * lda, K);
      }
    });
  });
  return std::make_tuple(Aq, As);
}

// weight     :  static, per-channel, symmetric
// activation : dynamic,   per-token, symmetric
//
// mat1    : [M, K]
// mat2    : [N, K]
// scales1 : [M]
// scales2 : [N]
// bias    : [N]
// out     : [M, N]
//
at::Tensor int8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales1,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales1);
  CHECK_INPUT(scales2);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat1.size(1);

  // see [NOTE]: s8s8 igemm compensation in avx512-vnni
  CHECK_EQ(mat2.size(1), (int64_t)(is_vnni ? K + sizeof(int32_t) : K));
  CHECK_EQ(scales1.numel(), M);
  CHECK_EQ(scales2.numel(), N);

  TORCH_CHECK(mat1.scalar_type() == at::kByte, "int8_scaled_mm: expect mat1 to be uint8.");
  TORCH_CHECK(mat2.scalar_type() == at::kChar, "int8_scaled_mm: expect mat2 to be int8.");
  TORCH_CHECK(
      scales1.scalar_type() == at::kFloat && scales2.scalar_type() == at::kFloat,
      "int8_scaled_mm: expect scales to be float32.");

  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_dtype, "int8_scaled_mm_kernel_impl", [&] {
    int8_scaled_mm_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<uint8_t>(),
        packed_w.data_ptr<int8_t>(),
        scales1.data_ptr<float>(),
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K);
  });
  return out;
}

// fused `per_token_quant_int8_cpu` and `int8_scaled_mm_cpu`
at::Tensor int8_scaled_mm_with_quant(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales2);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat1.size(1);
  int64_t lda = mat1.stride(0);

  // see [NOTE]: s8s8 igemm compensation in avx512-vnni
  CHECK_EQ(mat2.size(1), (int64_t)(is_vnni ? K + sizeof(int32_t) : K));
  CHECK_EQ(scales2.numel(), N);

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf, "int8_scaled_mm_with_quant: expect A to be bfloat16 or half.");
  TORCH_CHECK(st == out_dtype, "int8_scaled_mm_with_quant: expect A has same dtype with out_dtype.");
  TORCH_CHECK(mat2.scalar_type() == at::kChar, "int8_scaled_mm_with_quant: expect mat2 to be int8.");
  TORCH_CHECK(scales2.scalar_type() == at::kFloat, "int8_scaled_mm_with_quant: expect scales to be float32.");

#if defined(CPU_CAPABILITY_AVX512)
  at::Tensor coarse_out;
  if (dense_coarse_gemm(mat1, packed_w, scales2, bias, coarse_out, M, N, K)) {
    return coarse_out;
  }
#endif

  const int64_t buffer_size = M * K + M * sizeof(float);
  auto buffer = at::empty({buffer_size}, mat1.options().dtype(at::kByte));
  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_dtype, "int8_scaled_mm_with_quant_kernel_impl", [&] {
    uint8_t* __restrict__ Aq_data = buffer.data_ptr<uint8_t>();
    float* __restrict__ As_data = (float*)((void*)(Aq_data + M * K));
    const scalar_t* __restrict__ A_data = mat1.data_ptr<scalar_t>();

    at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        quantize_row_int8<scalar_t>(Aq_data + m * K, As_data[m], A_data + m * lda, K);
      }
    });

    int8_scaled_mm_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        Aq_data,
        packed_w.data_ptr<int8_t>(),
        As_data,
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K);
  });
  return out;
}
