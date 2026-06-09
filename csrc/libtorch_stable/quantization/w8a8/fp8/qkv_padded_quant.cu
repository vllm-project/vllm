// FP8 quantization with head_dim padding for ViT attention.
// CUDA fast-path replacement for the Triton kernel in
// vllm/kernels/triton/qkv_padded_fp8_quant.py.
//
// Input  : 3D bf16/fp16 tensor (S, H, D) with arbitrary 3D strides
//          (caller must guarantee stride(-1) == 1 to take this fast path).
// Output : contiguous FP8 (S, H, padded_D), padded_D = round_up(D, 16),
//          padding region is filled with FP8 zero.
//
// Operation per element (when skip_scale == false):
//   y = clamp(x / scale, FP8_MIN, FP8_MAX).to(fp8)
// When skip_scale == true (cast-only path):
//   y = clamp(x, FP8_MIN, FP8_MAX).to(fp8)
//
// Notes
//  - Layout: each thread block handles BLOCK_M rows (one row == one
//    flattened (s, h) pair); within a row each thread handles VEC_SIZE
//    contiguous elements along head_dim.
//  - 16-byte vectorized load (uint4) for bf16/fp16 input when D is a
//    multiple of VEC_SIZE; tail handled by the last lane (last
//    block_n tile) with per-element bounds checks.
//  - Padding region (D <= d < padded_D) is written as FP8 zero in a
//    separate code path with no scale division and no clamp work.
//  - SKIP_SCALE is a template flag, eliminating the
//    runtime branch + scale load cost for cast-only invocations.

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "libtorch_stable/dispatch_utils.h"
#include "libtorch_stable/torch_utils.h"

namespace vllm {

// Vector size: 8 bf16/fp16 elements per thread = 16 bytes(128 bit) load
constexpr int kVecSize = 8;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v);

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}

// ---------------------------------------------------------------------------
// D-coalesced quant kernel (production path for padded_D <= 128).
//
// Layout:
//   PADDED_D <= 128 is a compile-time constant. We pick N_VEC = PADDED_D / 8
//   threads-per-row (10 for padded_D=80, 16 for padded_D=128) and BLOCK_M
//   rows per block, so block dims = (N_VEC, BLOCK_M) and total threads =
//   N_VEC * BLOCK_M (typically 128..256 to keep occupancy high).
//
// Why this beats the row-per-thread variant on H20:
//   * Threads inside one warp run along row, where col strides by 1
//     vec (16 bytes) -> consecutive threads access consecutive 8-byte
//     output addresses -> 256-byte coalesced HBM3 sectors, fully utilizing
//     the 4 TB/s peak.
//   * No "padding-lane" thread waste: N_VEC equals the exact number of
//     vector chunks per row, so every thread does useful work.
//
// Each thread additionally walks ROWS_PER_THREAD distinct rows in a
// strided fashion (a "grid-stride" along the row dimension). This
// amortizes the 1/scale reciprocal computation and the launch overhead
// over more output bytes -- crucial for large workloads (S*H >> #SMs)
// where launch overhead would otherwise dominate.
//
// Grid:
//   grid.x = cdiv(S*H, BLOCK_M * ROWS_PER_THREAD)
//   grid.y = 1
// Block:
//   block.x = N_VEC                    // along head_dim
//   block.y = BLOCK_M                  // along (s, h)
// ---------------------------------------------------------------------------
template <typename scalar_t, typename fp8_t, bool SKIP_SCALE, int PADDED_D,
          int BLOCK_M, int ROWS_PER_THREAD>
__global__ void qkv_padded_fp8_quant_dcoal_kernel(
    const scalar_t* __restrict__ input, fp8_t* __restrict__ output,
    const float* __restrict__ scale_ptr, int num_heads, int n_rows, int n_cols,
    int64_t in_stride_s, int64_t in_stride_h, float fp8_min, float fp8_max) {
  static_assert(PADDED_D % kVecSize == 0,
                "PADDED_D must be a multiple of VEC_SIZE");

  [[maybe_unused]] constexpr int N_VEC = PADDED_D / kVecSize;

  const int v = threadIdx.x;  // 0..N_VEC-1
  const int row_in_block = threadIdx.y;
  const int row_start = blockIdx.x * BLOCK_M * ROWS_PER_THREAD + row_in_block;

  float inv_scale = 1.0f;
  if constexpr (!SKIP_SCALE) {
    inv_scale = 1.0f / __ldg(scale_ptr);
  }

  const int col_base = v * kVecSize;                      // 0, 8, 16, ...
  const int n_cols_full_vecs = n_cols / kVecSize;         // floor(D / 8)
  const int tail = n_cols - n_cols_full_vecs * kVecSize;  // 0..7
  // Hoist the  tile decision out of the row loop.
  const bool is_full_vec = (v < n_cols_full_vecs);
  const bool is_mixed_vec = (v == n_cols_full_vecs && tail > 0);

#pragma unroll
  for (int r = 0; r < ROWS_PER_THREAD; ++r) {
    const int row = row_start + r * BLOCK_M;
    if (row >= n_rows) {
      return;
    }

    // Decompose flattened row -> (s, h).
    const int s = row / num_heads;
    const int h = row - s * num_heads;

    const scalar_t* __restrict__ in_row =
        input + static_cast<int64_t>(s) * in_stride_s +
        static_cast<int64_t>(h) * in_stride_h;
    fp8_t* __restrict__ out_row = output + static_cast<int64_t>(row) * PADDED_D;

    if (is_full_vec) {
      // Pure data vec: 16-byte vectorized load -> quantize -> 8-byte store.
      scalar_t reg[kVecSize];
      const uint4* src = reinterpret_cast<const uint4*>(in_row + col_base);
      *reinterpret_cast<uint4*>(&reg[0]) = __ldg(src);

      fp8_t out_reg[kVecSize];
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        float x = to_float<scalar_t>(reg[i]);
        if constexpr (!SKIP_SCALE) {
          x = x * inv_scale;
        }
        x = fminf(fmaxf(x, fp8_min), fp8_max);
        out_reg[i] = fp8_t(x);
      }
      *reinterpret_cast<uint64_t*>(out_row + col_base) =
          *reinterpret_cast<const uint64_t*>(&out_reg[0]);
    } else if (is_mixed_vec) {
      // Mixed vec: first `tail` elements real, rest are padding-zeros.
      fp8_t out_reg[kVecSize] = {fp8_t(0)};  // padding-zeros
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        if (i < tail) {
          float x = to_float<scalar_t>(in_row[col_base + i]);
          if constexpr (!SKIP_SCALE) {
            x = x * inv_scale;
          }
          x = fminf(fmaxf(x, fp8_min), fp8_max);
          out_reg[i] = fp8_t(x);
        }
      }
      *reinterpret_cast<uint64_t*>(out_row + col_base) =
          *reinterpret_cast<const uint64_t*>(&out_reg[0]);
    } else {
      // Pure-padding vec (only when PADDED_D > round_up_8(n_cols)).
      *reinterpret_cast<uint64_t*>(out_row + col_base) =
          static_cast<uint64_t>(0);
    }
  }
}

// ---------------------------------------------------------------------------
// Generic 2D-tiled quant kernel (large padded_D path, padded_D > 128).
//
//   grid  : (cdiv(S*H, BLOCK_M), cdiv(padded_D, BLOCK_N))
//   block : (BLOCK_N / VEC_SIZE, BLOCK_M)
//
// Each (threadIdx.y, threadIdx.x) pair handles one row tile of VEC_SIZE
// contiguous head_dim elements.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename fp8_t, bool SKIP_SCALE, int BLOCK_M,
          int BLOCK_N>
__global__ void qkv_padded_fp8_quant_kernel(
    const scalar_t* __restrict__ input,   // [S, H, D] strided
    fp8_t* __restrict__ output,           // [S, H, padded_D] contiguous
    const float* __restrict__ scale_ptr,  // scalar
    int num_heads,                        // H
    int n_rows,                           // S * H
    int n_cols,                           // D
    int n_cols_padded,                    // padded_D (multiple of 16)
    int64_t in_stride_s,                  // input.stride(0)
    int64_t in_stride_h,                  // input.stride(1)
    int64_t out_stride_row,               // output.stride(1) (== padded_D),
                                          // since output is contiguous:
                                          // out[row, d] at row * padded_D + d
    float fp8_min, float fp8_max) {
  static_assert(BLOCK_N % kVecSize == 0,
                "BLOCK_N must be a multiple of VEC_SIZE");

  const int row_in_block = threadIdx.y;
  const int tx = threadIdx.x;
  const int row = blockIdx.x * BLOCK_M + row_in_block;
  const int col_base = blockIdx.y * BLOCK_N + tx * kVecSize;

  if (row >= n_rows) {
    return;
  }

  // Decompose flattened row into (s, h) for 3D stride indexing.
  const int s = row / num_heads;
  const int h = row - s * num_heads;

  // Compute the input row pointer once.
  const scalar_t* in_row = input + static_cast<int64_t>(s) * in_stride_s +
                           static_cast<int64_t>(h) * in_stride_h;
  // Output row offset (output is contiguous).
  fp8_t* out_row = output + static_cast<int64_t>(row) * n_cols_padded;

  // Skip whole tiles outside output bounds.
  if (col_base >= n_cols_padded) {
    return;
  }

  // Read scale once per thread (compiler hoists; SKIP_SCALE branch
  // is fully eliminated at compile time).
  float inv_scale = 1.0f;
  if constexpr (!SKIP_SCALE) {
    // 1.0 / scale: the kernel divides by scale -> we precompute the
    // reciprocal so that the inner loop is a single multiply.
    inv_scale = 1.0f / __ldg(scale_ptr);
  }

  // Whole tile lies inside both input and output ranges?
  const bool full_tile_in =
      (col_base + kVecSize) <= n_cols;  // no input mask needed
  const bool full_tile_out =
      (col_base + kVecSize) <= n_cols_padded;         // no output mask needed
  const bool tile_is_padding = (col_base >= n_cols);  // entire tile is padding

  // --- Pure padding tile: write zeros (8 bytes per thread) ---------------
  if (tile_is_padding) {
    if (full_tile_out) {
      // 8 fp8 bytes packed into uint64 zero
      uint64_t zero = 0;
      *reinterpret_cast<uint64_t*>(out_row + col_base) = zero;
    } else {
      // Boundary tile: zero per-element with bounds check.
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        int d = col_base + i;
        if (d < n_cols_padded) {
          out_row[d] = fp8_t(0);
        }
      }
    }
    return;
  }

  // --- Mixed/full tile: load vector, quantize, store --------------------

  // Load up to VEC_SIZE bf16/fp16 elements from input. Use 16-byte
  // vectorized load when the tile is fully inside [0, n_cols);
  // otherwise per-element load with bounds check.
  scalar_t reg[kVecSize];
  if (full_tile_in) {
    // Aligned 16-byte load. in_stride_d == 1 is a precondition.
    const uint4* src = reinterpret_cast<const uint4*>(in_row + col_base);
    *reinterpret_cast<uint4*>(&reg[0]) = __ldg(src);
  } else {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      int d = col_base + i;
      reg[i] = (d < n_cols) ? in_row[d] : scalar_t(0);
    }
  }

  // Quantize -> fp8 per element.
  fp8_t out_reg[kVecSize];
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    float x = to_float<scalar_t>(reg[i]);
    if constexpr (!SKIP_SCALE) {
      x = x * inv_scale;
    }
    // Bounds check: padding elements always get exactly +0.0f.
    int d = col_base + i;
    if (d >= n_cols) {
      x = 0.0f;
    } else {
      x = fminf(fmaxf(x, fp8_min), fp8_max);
    }
    out_reg[i] = fp8_t(x);
  }

  // Store. fp8 is 1 byte each, kVecSize=8 -> 8 bytes per thread.
  if (full_tile_out) {
    *reinterpret_cast<uint64_t*>(out_row + col_base) =
        *reinterpret_cast<const uint64_t*>(&out_reg[0]);
  } else {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      int d = col_base + i;
      if (d < n_cols_padded) {
        out_row[d] = out_reg[i];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Launcher.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename fp8_t, bool SKIP_SCALE>
static void launch_qkv_padded_fp8_quant(const scalar_t* input, fp8_t* output,
                                        const float* scale_ptr, int H,
                                        int n_rows, int D, int padded_D,
                                        int64_t in_stride_s,
                                        int64_t in_stride_h, float fp8_min,
                                        float fp8_max, cudaStream_t stream) {
  // Tile-selection strategy.
  //
  // Two kernels are dispatched here:
  //
  //   1. D-coalesced kernel (padded_D <= 128, the production path):
  //      block dims = (N_VEC, BLOCK_M) where N_VEC = padded_D / 8;
  //      each thread emits exactly one 8-byte fp8 vec into one (row, vec)
  //      slot. Threads inside a warp stride along head_dim by 8 bytes
  //      (coalesced HBM access). Each thread also walks
  //      ROWS_PER_THREAD distinct rows so launch overhead is amortized
  //      on large workloads.
  //
  //   2. Generic 2D-tiled kernel (padded_D > 128): falls back to the
  //      classic (BLOCK_M, BLOCK_N/VEC) thread layout. Production ViTs
  //      never hit this branch, but it is kept for correctness on any
  //      future model with head_dim > 128 that still needs padding.
  //
  // BLOCK_M is chosen empirically per padded_D so that block size stays
  // within [128, 320] threads -- enough to hide HBM3 latency while
  // keeping multiple resident blocks per SM for occupancy.

  auto launch_dcoal = [&](auto padded_d_const, auto block_m_const,
                          auto rows_per_thread_const) {
    constexpr int PADDED_D = decltype(padded_d_const)::value;
    constexpr int BLOCK_M = decltype(block_m_const)::value;
    constexpr int ROWS_PER_THREAD = decltype(rows_per_thread_const)::value;
    constexpr int N_VEC = PADDED_D / kVecSize;
    dim3 block(N_VEC, BLOCK_M);
    constexpr int rows_per_block = BLOCK_M * ROWS_PER_THREAD;
    dim3 grid((n_rows + rows_per_block - 1) / rows_per_block);
    qkv_padded_fp8_quant_dcoal_kernel<scalar_t, fp8_t, SKIP_SCALE, PADDED_D,
                                      BLOCK_M, ROWS_PER_THREAD>
        <<<grid, block, 0, stream>>>(input, output, scale_ptr, H, n_rows, D,
                                     in_stride_s, in_stride_h, fp8_min,
                                     fp8_max);
  };

  auto launch_generic = [&](auto block_m_const, auto block_n_const) {
    constexpr int BLOCK_M = decltype(block_m_const)::value;
    constexpr int BLOCK_N = decltype(block_n_const)::value;
    dim3 block(BLOCK_N / kVecSize, BLOCK_M);
    dim3 grid((n_rows + BLOCK_M - 1) / BLOCK_M,
              (padded_D + BLOCK_N - 1) / BLOCK_N);
    qkv_padded_fp8_quant_kernel<scalar_t, fp8_t, SKIP_SCALE, BLOCK_M, BLOCK_N>
        <<<grid, block, 0, stream>>>(input, output, scale_ptr, H, n_rows, D,
                                     padded_D, in_stride_s, in_stride_h,
                                     padded_D, fp8_min, fp8_max);
  };

  // Helper: compile-time integral constant
#define IC(N) \
  std::integral_constant<int, N> {}

  // padded_D is always a multiple of 16; switch on it.
  // Block layout: (N_VEC, BLOCK_M); we aim for total threads ~= 256 to
  // give every SM at least 4 resident blocks at typical register usage.
  //
  // ROWS_PER_THREAD: each thread walks this many rows in a strided
  // fashion. For small n_rows we want ROWS_PER_THREAD=1 (full grid =
  // full parallelism); for large n_rows we want a bigger value so the
  // grid stays close to ~16x #SMs (H20 has 78 SMs) and per-block work
  // stays large enough to hide HBM3 latency. Threshold 4096 rows is
  // empirical: it matches the point where Triton transitions from
  // launch-bound to bandwidth-bound on H20.
  const bool large_workload = (n_rows >= 4096);

#define DISPATCH_PADDED_D(D_VAL, BLOCK_M_VAL)          \
  do {                                                 \
    if (large_workload) {                              \
      launch_dcoal(IC(D_VAL), IC(BLOCK_M_VAL), IC(4)); \
    } else {                                           \
      launch_dcoal(IC(D_VAL), IC(BLOCK_M_VAL), IC(1)); \
    }                                                  \
    return;                                            \
  } while (0)

  switch (padded_D) {
    case 16:  // N_VEC = 2
      DISPATCH_PADDED_D(16, 128);
    case 32:  // N_VEC = 4
      DISPATCH_PADDED_D(32, 64);
    case 48:  // N_VEC = 6
      DISPATCH_PADDED_D(48, 48);
    case 64:  // N_VEC = 8
      DISPATCH_PADDED_D(64, 32);
    case 80:  // N_VEC = 10  (Qwen3-VL hot path, D=72 -> padded=80)
      DISPATCH_PADDED_D(80, 32);
    case 96:  // N_VEC = 12
      DISPATCH_PADDED_D(96, 24);
    case 112:  // N_VEC = 14
      DISPATCH_PADDED_D(112, 16);
    case 128:  // N_VEC = 16
      DISPATCH_PADDED_D(128, 16);
    default:
      break;
  }

#undef DISPATCH_PADDED_D

  // padded_D > 128: tile along D with the generic 2D kernel.
  launch_generic(IC(8), IC(128));

#undef IC
}

}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry: stable-ABI op exposed as torch.ops._C.qkv_padded_fp8_quant.
//
// Computes the FP8 quantized output tensor for the given input. The
// returned tensor is always contiguous with shape
// (..., H, padded_D), where padded_D = round_up(D, 16). For 4D
// inputs (B, S, H, D) the output keeps the leading batch dimension.
// ---------------------------------------------------------------------------
torch::stable::Tensor qkv_padded_fp8_quant(const torch::stable::Tensor& input,
                                           const torch::stable::Tensor& scale,
                                           bool skip_scale) {
  const int64_t ndim = input.dim();
  STD_TORCH_CHECK(ndim == 3 || ndim == 4,
                  "qkv_padded_fp8_quant: input must be 3D (S,H,D) or "
                  "4D (B,S,H,D), got ndim=",
                  ndim);

  const int64_t H = input.size(ndim - 2);
  const int64_t D = input.size(ndim - 1);
  STD_TORCH_CHECK(D > 0, "qkv_padded_fp8_quant: D must be positive, got ", D);

  const int64_t padded_D = (D + 15) / 16 * 16;

  // Total flattened (S*H) row count.
  int64_t total_rows = 1;
  for (int64_t i = 0; i + 2 < ndim; ++i) {
    total_rows *= input.size(i);
  }
  const int64_t S_flat = total_rows;  // flattened S (incl. batch)
  const int64_t n_rows = S_flat * H;

  // Strides for the 3D view (s_flat, h, d).
  // For 4D input (B, S, H, D), we treat (B, S) as a single flattened S
  // dimension. This requires the (B, S) sub-block to be contiguous along
  // B->S, which is true for the .view(-1, H, D) reshape used by the
  // Python dispatcher. To avoid imposing any new constraint, we
  // require input.stride(-1) == 1 and input.stride(-2) == D-stride.
  STD_TORCH_CHECK(input.stride(ndim - 1) == 1,
                  "qkv_padded_fp8_quant: requires stride(-1) == 1, got ",
                  input.stride(ndim - 1));

  int64_t in_stride_h = input.stride(ndim - 2);
  // stride_s for the flattened leading dims: use stride of axis (ndim-3).
  // For 4D, callers should reshape first; we pick the innermost remaining
  // stride which is correct iff the leading dims are contiguous w.r.t.
  // the (H, D) sub-block.
  int64_t in_stride_s = (ndim >= 3) ? input.stride(ndim - 3) : 0;
  if (ndim == 4) {
    // Verify that batch is contiguous w.r.t. the flattened S*H*D
    // sub-block; if not, the caller should fall back to Triton.
    int64_t b_stride = input.stride(0);
    int64_t s_size = input.size(1);
    STD_TORCH_CHECK(b_stride == s_size * in_stride_s,
                    "qkv_padded_fp8_quant: 4D input must have batch "
                    "stride == S * stride(S); got batch_stride=",
                    b_stride, " S=", s_size, " s_stride=", in_stride_s);
  }

  // Build output shape: replace last dim with padded_D.
  std::vector<int64_t> out_shape;
  out_shape.reserve(ndim);
  for (int64_t i = 0; i + 1 < ndim; ++i) {
    out_shape.push_back(input.size(i));
  }
  out_shape.push_back(padded_D);

  auto out_dtype = torch::headeronly::ScalarType::Float8_e4m3fn;

  torch::stable::Tensor output =
      torch::stable::new_empty(input, out_shape, out_dtype);

  // FP8 e4m3 finite range: [-448, 448].
  const float fp8_max = 448.0f;
  const float fp8_min = -448.0f;

  cudaStream_t stream = get_current_cuda_stream();

  STD_TORCH_CHECK(scale.scalar_type() == torch::headeronly::ScalarType::Float,
                  "qkv_padded_fp8_quant: scale must be float32, got ",
                  static_cast<int>(scale.scalar_type()));

#define LAUNCH_KERNEL(scalar_t_macro, fp8_t_macro)                           \
  do {                                                                       \
    if (skip_scale) {                                                        \
      vllm::launch_qkv_padded_fp8_quant<scalar_t_macro, fp8_t_macro, true>(  \
          static_cast<const scalar_t_macro*>(input.data_ptr()),              \
          static_cast<fp8_t_macro*>(output.data_ptr()),                      \
          static_cast<const float*>(scale.data_ptr()), static_cast<int>(H),  \
          static_cast<int>(n_rows), static_cast<int>(D),                     \
          static_cast<int>(padded_D), in_stride_s, in_stride_h, fp8_min,     \
          fp8_max, stream);                                                  \
    } else {                                                                 \
      vllm::launch_qkv_padded_fp8_quant<scalar_t_macro, fp8_t_macro, false>( \
          static_cast<const scalar_t_macro*>(input.data_ptr()),              \
          static_cast<fp8_t_macro*>(output.data_ptr()),                      \
          static_cast<const float*>(scale.data_ptr()), static_cast<int>(H),  \
          static_cast<int>(n_rows), static_cast<int>(D),                     \
          static_cast<int>(padded_D), in_stride_s, in_stride_h, fp8_min,     \
          fp8_max, stream);                                                  \
    }                                                                        \
  } while (0)

  const auto in_dtype = input.scalar_type();
  if (in_dtype == torch::headeronly::ScalarType::BFloat16) {
    LAUNCH_KERNEL(__nv_bfloat16, __nv_fp8_e4m3);
  } else if (in_dtype == torch::headeronly::ScalarType::Half) {
    LAUNCH_KERNEL(__half, __nv_fp8_e4m3);
  } else {
    STD_TORCH_CHECK(false,
                    "qkv_padded_fp8_quant: input must be bfloat16 or "
                    "float16, got dtype=",
                    static_cast<int>(in_dtype));
  }

#undef LAUNCH_KERNEL

  return output;
}
