#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <algorithm>

#include "../cuda_compat.h"
#include "dispatch_utils.h"
#include "quantization/w8a8/fp8/common.cuh"
#include "core/batch_invariant.hpp"

// TODO(rasmith): The kernels in this file are susceptible to integer overflow
// issues, do not take strides, and are unable to handle PyTorch tensors that
// return is_contiguous() as False (the tensors may actually be contiguous
// in memory).
//
// However, it may be possible to fix these kernels to handle both issues.

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

// Combined RDNA macro (gfx11 + gfx12) - both use 32-wide wavefronts
#if defined(__GFX11__) || defined(__GFX12__)
  #define __HIP__GFX1X__
#endif

#if defined(__HIPCC__) && (defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__MI3XX__
#endif

#if defined(__gfx950__)
  #define LDS_SIZE 160 * 1024
#else
  #define LDS_SIZE 64 * 1024
#endif

int get_lds_size() {
  static const int result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    const std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx95") == std::string::npos ? 64 * 1024
                                                          : 160 * 1024;
  }();
  return result;
}

bool on_gfx1x() {
  static const bool result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    const std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx11") != std::string::npos ||
           device_arch.find("gfx12") != std::string::npos;
  }();
  return result;
}

bool on_gfx12() {
  static const bool result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    const std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx12") != std::string::npos;
  }();
  return result;
}

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

template <typename T>
struct scalar {};

template <typename T>
struct scalar2 {};

template <typename T>
__device__ __forceinline__ float2 __s22float2(T v);

template <typename T>
__device__ __forceinline__ T __float2s(float v);

template <typename T>
__device__ __forceinline__ T __float22s2_rn(float2 v);

// Definitions and cvt functions for fp16
template <>
struct scalar<c10::Half> {
  using type = half;
};

template <>
struct scalar2<c10::Half> {
  using type = __half2;
};

template <>
__device__ __forceinline__ half __float2s(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ float2 __s22float2(__half2 v) {
  return __half22float2(v);
}

template <>
__device__ __forceinline__ __half2 __float22s2_rn(float2 v) {
  return __float22half2_rn(v);
}

// Definitions and cvt functions for bf16
template <>
struct scalar<c10::BFloat16> {
  using type = __hip_bfloat16;
};

template <>
struct scalar2<c10::BFloat16> {
  using type = __hip_bfloat162;
};

template <>
__device__ __forceinline__ __hip_bfloat16 __float2s(float v) {
  return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ float2 __s22float2(__hip_bfloat162 v) {
  return __bfloat1622float2(v);
}

template <>
__device__ __forceinline__ __hip_bfloat162 __float22s2_rn(float2 v) {
  return __float22bfloat162_rn(v);
}

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ float4 load_ntmprl(const float4* addr) {
  auto addr_alias = reinterpret_cast<const float*>(addr);
  auto dat0 = loadnt(addr_alias);
  auto dat1 = loadnt(addr_alias + 1);
  auto dat2 = loadnt(addr_alias + 2);
  auto dat3 = loadnt(addr_alias + 3);
  return make_float4(dat0, dat1, dat2, dat3);
}

using swmmac_i16x8 = short __attribute__((ext_vector_type(8)));
using swmmac_i16x16 = short __attribute__((ext_vector_type(16)));
using swmmac_f16x8 = __fp16 __attribute__((ext_vector_type(8)));
using swmmac_f16x16 = __fp16 __attribute__((ext_vector_type(16)));
using swmmac_f32x8 = float __attribute__((ext_vector_type(8)));
using swmmac_i32x4 = int32_t __attribute__((__vector_size__(16)));

// TBlock fetches entire rows of A, and entire col of B (K dimension); assume
// N=1 for time being grid is M/A_NUM_ROWS blocks
template <typename scalar_t, int NUM_A_ROWS_PER_BLOCK>
__global__ void LLGemm1_kernel(const scalar_t* in_a, const scalar_t* in_b,
                               scalar_t* out_c, const int K) {
  using scalar2_t = typename scalar2<scalar_t>::type;
  auto af4 = reinterpret_cast<const float4*>(in_a);
  auto bf4 = reinterpret_cast<const scalar2_t*>(in_b);
  auto c = reinterpret_cast<scalar2_t*>(out_c);
  __shared__ float red_smem[NUM_A_ROWS_PER_BLOCK][WARP_SIZE];
  const int row_addr = blockIdx.x * NUM_A_ROWS_PER_BLOCK * K / 8;
  const int threadid = threadIdx.x;
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;
  const int qwarpid = threadid / 16;
  const int qthreadid = threadid % 16;
  float4 rowA_elem4[NUM_A_ROWS_PER_BLOCK];
  scalar2_t colB_elem4x, colB_elem4y, colB_elem4z, colB_elem4w;
  float acc[NUM_A_ROWS_PER_BLOCK];
  scalar2_t acch2;
  scalar2_t oval;

  // As we later use warp shuffle operations, we may have more threads in the
  // block than the actual available data, hence the if guard here.
  if (threadid * 8 < K) {
#pragma unroll
    for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
      // rowA_elem4[i] holds 8 * half numbers seen as a single float4.
      rowA_elem4[i] = load_ntmprl(&af4[row_addr + threadid + K / 8 * i]);
    }
    colB_elem4x = bf4[threadid * 4 + 0];
    colB_elem4y = bf4[threadid * 4 + 1];
    colB_elem4z = bf4[threadid * 4 + 2];
    colB_elem4w = bf4[threadid * 4 + 3];
  }

  scalar2_t Af2;
  float2 S;

  auto Ah2ptr = reinterpret_cast<scalar2_t*>(&rowA_elem4);
  scalar2_t* ah2lptr;

#pragma unroll
  for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
    // Multiply-add on 8 scalar_t.
    ah2lptr = Ah2ptr + i * 4;
    Af2 = *(ah2lptr);
    acch2 = __hmul2(Af2, colB_elem4x);
    Af2 = *(ah2lptr + 1);
    acch2 = __hfma2(Af2, colB_elem4y, acch2);
    Af2 = *(ah2lptr + 2);
    acch2 = __hfma2(Af2, colB_elem4z, acch2);
    Af2 = *(ah2lptr + 3);
    acch2 = __hfma2(Af2, colB_elem4w, acch2);
    S = __s22float2(acch2);

    // See comment above concerning the if guard.
    acc[i] = (threadid * 8 < K ? S.x + S.y : 0.f);
  }

// all reduce across warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
#pragma unroll
    for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
      acc[i] += __shfl_xor(acc[i], mask);
    }
  }

  // Warp leaders store the data to shared memory.
  if (lane < NUM_A_ROWS_PER_BLOCK) {
    red_smem[lane][warp] = acc[lane];
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  if (qwarpid < NUM_A_ROWS_PER_BLOCK) {
    acc[qwarpid] = qthreadid < num_warps ? red_smem[qwarpid][qthreadid] : 0.f;
#pragma unroll
    for (int mask = 16 / 2; mask >= 1; mask /= 2) {
      acc[qwarpid] += __shfl_xor(acc[qwarpid], mask);
    }
    float oval2 = __shfl_xor(acc[qwarpid], 16);

    if (lane % 32 == 0) {
      oval = __float22s2_rn<scalar2_t>(make_float2(acc[qwarpid], oval2));
      c[blockIdx.x * NUM_A_ROWS_PER_BLOCK / 2 + qwarpid / 2] = oval;
    }
  }
}

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block) {
  auto M = in_a.size(0);
  auto K = in_a.size(1);
  auto N = in_b.size(0);

  TORCH_CHECK(N == 1, "Row number of activation tensor must be 1.");
  TORCH_CHECK(in_a.dtype() == in_b.dtype());
  TORCH_CHECK(in_b.dtype() == torch::kFloat16 ||
              in_b.dtype() == torch::kBFloat16);

  auto out_c = torch::empty(
      {N, M}, torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  // NUM_TREADS need to be a multiple of WARP_SIZE, as we are using warp shuffle
  // operations.
  const int NUM_THREADS =
      max(rows_per_block * 16,
          K * 2 / 16 % WARP_SIZE == 0
              ? K * 2 / 16
              : K * 2 / 16 + (WARP_SIZE - K * 2 / 16 % WARP_SIZE));

  int NUM_BLOCKS = M / rows_per_block;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_b));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // call the kernel function...
  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_b.scalar_type(), "LLGemm1", [&] {
    auto a_ptr = in_a.data_ptr<scalar_t>();
    auto b_ptr = in_b.data_ptr<scalar_t>();
    auto c_ptr = out_c.data_ptr<scalar_t>();
    if (rows_per_block == 2) {
      LLGemm1_kernel<scalar_t, 2>
          <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else if (rows_per_block == 4) {
      LLGemm1_kernel<scalar_t, 4>
          <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else if (rows_per_block == 8) {
      LLGemm1_kernel<scalar_t, 8>
          <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else if (rows_per_block == 16) {
      LLGemm1_kernel<scalar_t, 16>
          <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else {
      NUM_BLOCKS = M / 4;
      LLGemm1_kernel<scalar_t, 4>
          <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    }
  });

  return out_c;
}

#if defined(__HIP__GFX9__) && !defined(__HIP__GFX1X__)
  #define DOT2C(V0, V2, V3)                                          \
    if constexpr (std::is_same_v<scalar_t, half>) {                  \
      asm("v_dot2c_f32_f16 %0, %2, %3"                               \
          : "=v"(V0)                                                 \
          : "0"(V0), "v"(V2), "v"(V3));                              \
    } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) { \
      float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *  \
                 __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));   \
      V0 += (s.x + s.y);                                             \
    }
#elif defined(__HIP__GFX1X__)
  // gfx1x: v_dot2_f32_f16 (VOP3-P, dot10-insts, available on gfx11+gfx12)
  #define DOT2C(V0, V2, V3)                                               \
    if constexpr (std::is_same_v<scalar_t, half>) {                       \
      asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(V0) : "v"(V2), "v"(V3)); \
    } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {      \
      float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *       \
                 __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));        \
      V0 += (s.x + s.y);                                                  \
    }
#endif

// To avoid LLVM silently upcasting to double
__device__ inline unsigned int min__(uint32_t a, uint32_t b) {
  return min(a, b);
}

#if defined(__GFX12__)

// Selectors used by the SWMMAC sparse A operand. Even/odd lanes choose
// different 16-bit pairs from the A fragment.
static constexpr int32_t kSwmmacSelect01 = static_cast<int32_t>(0x44444444u);
static constexpr int32_t kSwmmacSelect23 = static_cast<int32_t>(0xEEEEEEEEu);

template <typename scalar_t>
struct swmmac_traits;

template <>
struct swmmac_traits<__hip_bfloat16> {
  using a_frag_t = swmmac_i16x8;
  using b_frag_t = swmmac_i16x16;

  __device__ __forceinline__ static swmmac_f32x8 mma(const a_frag_t a,
                                                     const b_frag_t b,
                                                     const swmmac_f32x8 acc,
                                                     const int sparse_idx) {
    return __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a, b, acc, sparse_idx);
  }

  __device__ __forceinline__ static float to_float(const __hip_bfloat16 value) {
    return __bfloat162float(value);
  }

  __device__ __forceinline__ static __hip_bfloat16 from_float(
      const float value) {
    return __float2bfloat16(value);
  }
};

template <>
struct swmmac_traits<half> {
  using a_frag_t = swmmac_f16x8;
  using b_frag_t = swmmac_f16x16;

  __device__ __forceinline__ static swmmac_f32x8 mma(const a_frag_t a,
                                                     const b_frag_t b,
                                                     const swmmac_f32x8 acc,
                                                     const int sparse_idx) {
    return __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, acc, sparse_idx);
  }

  __device__ __forceinline__ static float to_float(const half value) {
    return __half2float(value);
  }

  __device__ __forceinline__ static half from_float(const float value) {
    return __float2half(value);
  }
};

// Build the A fragment for SWMMAC.
//
// B stays row-major. Each lane reads one contiguous 16-value half row from LDS:
//   lane_half 0 -> B[k +  0 : k + 16]
//   lane_half 1 -> B[k + 16 : k + 32]
template <typename scalar_t, int N, int K_PER_INSTRUCTION>
__device__ __forceinline__ typename swmmac_traits<scalar_t>::a_frag_t
load_swmmac_a_direct(const scalar_t* __restrict__ A, const int A_stride,
                     const int global_kblock, const int lane) {
  using a_frag_t = typename swmmac_traits<scalar_t>::a_frag_t;
  a_frag_t frag{};

  const int sparse_row = lane & 15;
  const int original_n = sparse_row >> 1;
  if (original_n >= N) {
    return frag;
  }

  const int lane_half = lane >> 4;
  const bool select_high_pair = (sparse_row & 1) != 0;

  const auto* src_aligned =
      reinterpret_cast<const int32_t*>(A) +
      (original_n * A_stride + global_kblock * K_PER_INSTRUCTION +
       lane_half * 16) /
          2;

  const auto* src_vec = reinterpret_cast<const swmmac_i32x4*>(src_aligned);

  const auto load0 = src_vec[0];
  const auto load1 = src_vec[1];

  auto* frag_i32 = reinterpret_cast<int32_t*>(&frag);

  if (select_high_pair) {
    frag_i32[0] = load0[1];
    frag_i32[1] = load0[3];
    frag_i32[2] = load1[1];
    frag_i32[3] = load1[3];
  } else {
    frag_i32[0] = load0[0];
    frag_i32[1] = load0[2];
    frag_i32[2] = load1[0];
    frag_i32[3] = load1[2];
  }

  return frag;
}

// B panels are streamed through the kernel, so use nontemporal vector loads.
template <typename frag_t>
__device__ __forceinline__ frag_t
load_swmmac_b(const frag_t* __restrict__ ptr) {
  return __builtin_nontemporal_load(ptr);
}

__device__ __forceinline__ int swmmac_bias_index(const int row, const int col,
                                                 const int N, const int M,
                                                 const int Bx, const int By) {
  if (Bx == 1 && By == 1) return 0;
  if (Bx == M && By == 1) return col;
  if (Bx == 1 && By == N) return row;
  if (Bx == M && By == N) return row * M + col;
  return (row % By) * Bx + (col % Bx);
}

__device__ __forceinline__ void swmmac_lds_wave_fence() {
  // Wait for wave-local LDS operations before reading from, or reusing, the
  // LDS panel.
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __syncwarp();
}

// Prefetch one row-major B panel from global memory into VGPRs.
//
// For one wave, the panel is:
//
//   B[m + 0 ][k_start : k_start + K_LOAD]
//   B[m + 1 ][k_start : k_start + K_LOAD]
//   ...
//   B[m + COLUMNS_PER_WAVE - 1][k_start : k_start + K_LOAD]
template <typename scalar_t, int K_LOAD, int COLUMNS_PER_WAVE,
          int THREADS_PER_WAVE>
__device__ __forceinline__ void prefetch_swmmac_b_panel(
    swmmac_i32x4* __restrict__ prefetched, const scalar_t* __restrict__ B,
    const int K, const int M, const int m, const int k_start,
    const int valid_kblocks) {
  constexpr int scalars_per_vec = 8;
  constexpr int vecs_per_row = K_LOAD / scalars_per_vec;
  constexpr int panel_vecs = COLUMNS_PER_WAVE * vecs_per_row;
  static_assert(panel_vecs % THREADS_PER_WAVE == 0);
  constexpr int vecs_per_lane = panel_vecs / THREADS_PER_WAVE;

  const int lane = static_cast<int>(threadIdx.x);
  const int valid_k = valid_kblocks * 32;

  #pragma unroll
  for (int v = 0; v < vecs_per_lane; ++v) {
    const int flat_vec = v * THREADS_PER_WAVE + lane;
    const int row = flat_vec / vecs_per_row;
    const int scalar_col = (flat_vec % vecs_per_row) * scalars_per_vec;

    swmmac_i32x4 value{};

    if (m + row < M && scalar_col + scalars_per_vec <= valid_k) {
      const auto* src = reinterpret_cast<const swmmac_i32x4*>(
          B + static_cast<size_t>(m + row) * K + k_start + scalar_col);
      value = load_swmmac_b(src);
    }

    prefetched[v] = value;
  }
}

// Commit the prefetched VGPR panel into LDS.
template <typename scalar_t, int K_LOAD, int K_LOAD_PAD, int COLUMNS_PER_WAVE,
          int THREADS_PER_WAVE>
__device__ __forceinline__ void commit_swmmac_b_panel(
    scalar_t* __restrict__ lds_panel,
    const swmmac_i32x4* __restrict__ prefetched) {
  constexpr int scalars_per_vec = 8;
  constexpr int vecs_per_row = K_LOAD / scalars_per_vec;
  constexpr int panel_vecs = COLUMNS_PER_WAVE * vecs_per_row;
  static_assert(panel_vecs % THREADS_PER_WAVE == 0);
  constexpr int vecs_per_lane = panel_vecs / THREADS_PER_WAVE;

  const int lane = static_cast<int>(threadIdx.x);

  #pragma unroll
  for (int v = 0; v < vecs_per_lane; ++v) {
    const int flat_vec = v * THREADS_PER_WAVE + lane;
    const int row = flat_vec / vecs_per_row;
    const int scalar_col = (flat_vec % vecs_per_row) * scalars_per_vec;

    auto* dst = reinterpret_cast<swmmac_i32x4*>(
        lds_panel + static_cast<size_t>(row) * K_LOAD_PAD + scalar_col);
    *dst = prefetched[v];
  }
}

// Read one SWMMAC B fragment from the row-major LDS panel. Each lane owns
// one output column and one 16-value half of the current K block.
template <typename scalar_t, int K_LOAD_PAD, int TILE_COLUMNS,
          int K_PER_INSTRUCTION>
__device__ __forceinline__ typename swmmac_traits<scalar_t>::b_frag_t
load_swmmac_b_lds_row_major(const scalar_t* __restrict__ lds_panel,
                            const int tile, const int u, const int lane) {
  using b_frag_t = typename swmmac_traits<scalar_t>::b_frag_t;

  const int row = tile * TILE_COLUMNS + (lane & 15);
  const int lane_half = lane >> 4;
  const int scalar_col = u * K_PER_INSTRUCTION + lane_half * 16;

  const auto* src = reinterpret_cast<const b_frag_t*>(
      lds_panel + static_cast<size_t>(row) * K_LOAD_PAD + scalar_col);
  return *src;
}

// Regular SWMMAC path. Each active wave computes one 16-column output tile.
//
// Two LDS panels are used as ping-pong buffers for B:
//   prefetch next B panel into VGPRs
//   compute current B panel from LDS
//   commit next B panel into the alternate LDS panel
template <typename scalar_t, int N, bool HAS_BIAS, int LOAD_GROUP,
          bool ALIGNED_K, int TILE_COLUMNS, int M_TILES_PER_WAVE,
          int K_PER_INSTRUCTION, int WV_PER_GROUP, int THREADS_PER_WAVE>
__global__ void __launch_bounds__(WV_PER_GROUP* THREADS_PER_WAVE)
    swmmacGemmRegular(const int K, const int A_stride, const int M,
                      const int Bx, const int By,
                      const scalar_t* __restrict__ A,
                      const scalar_t* __restrict__ B,
                      const scalar_t* __restrict__ BIAS,
                      scalar_t* __restrict__ C) {
  static_assert(N >= 5 && N <= 8);

  using traits_t = swmmac_traits<scalar_t>;
  using a_frag_t = typename traits_t::a_frag_t;
  using b_frag_t = typename traits_t::b_frag_t;

  constexpr int columns_per_wave = TILE_COLUMNS * M_TILES_PER_WAVE;
  constexpr int k_load = LOAD_GROUP * K_PER_INSTRUCTION;
  constexpr int k_load_pad = k_load + 8;
  constexpr int vecs_per_lane =
      (columns_per_wave * k_load) / (8 * THREADS_PER_WAVE);

  // Two B panels per wave. They are used as ping-pong buffers.
  __shared__ __align__(16)
      scalar_t b_lds[WV_PER_GROUP][2][columns_per_wave][k_load_pad];

  const int total_num_kblocks = K / K_PER_INSTRUCTION;
  const int lane = static_cast<int>(threadIdx.x);
  const int wave_id = static_cast<int>(threadIdx.y);
  const int j = lane & 15;
  const int row_base = (lane >> 4) * 4;
  const int32_t sparse_idx = (lane & 1) ? kSwmmacSelect23 : kSwmmacSelect01;

  // Some blocks need fewer than four waves when M is small.
  const int columns_per_grid_round =
      static_cast<int>(gridDim.x) * columns_per_wave;
  const int required_waves =
      (M + columns_per_grid_round - 1) / columns_per_grid_round;
  const int active_waves =
      required_waves < WV_PER_GROUP ? required_waves : WV_PER_GROUP;

  if (wave_id >= active_waves) return;

  uint32_t m =
      (static_cast<uint32_t>(blockIdx.x) * static_cast<uint32_t>(active_waves) +
       static_cast<uint32_t>(wave_id)) *
      columns_per_wave;

  while (m < static_cast<uint32_t>(M)) {
    // Accumulate even and odd K blocks separately, then merge before store.
    swmmac_f32x8 acc0[M_TILES_PER_WAVE]{};
    swmmac_f32x8 acc1[M_TILES_PER_WAVE]{};

    // Load the first B panel and place it in LDS buffer 0.
    swmmac_i32x4 initial_panel[vecs_per_lane];
    const int first_valid_kblocks =
        total_num_kblocks < LOAD_GROUP ? total_num_kblocks : LOAD_GROUP;

    prefetch_swmmac_b_panel<scalar_t, k_load, columns_per_wave,
                            THREADS_PER_WAVE>(
        initial_panel, B, K, M, static_cast<int>(m), 0, first_valid_kblocks);

    commit_swmmac_b_panel<scalar_t, k_load, k_load_pad, columns_per_wave,
                          THREADS_PER_WAVE>(&b_lds[wave_id][0][0][0],
                                            initial_panel);

    swmmac_lds_wave_fence();

    for (int group_base = 0; group_base < total_num_kblocks;
         group_base += LOAD_GROUP) {
      // Current buffer is consumed by SWMMAC; next buffer receives the next
      // prefetched panel.
      const int current_buffer = (group_base / LOAD_GROUP) & 1;
      const int next_buffer = current_buffer ^ 1;
      const int next_group_base = group_base + LOAD_GROUP;
      const bool has_next = next_group_base < total_num_kblocks;

      const int current_valid_kblocks =
          ALIGNED_K ? LOAD_GROUP
                    : ((total_num_kblocks - group_base) < LOAD_GROUP
                           ? (total_num_kblocks - group_base)
                           : LOAD_GROUP);

      // A is small, so keep the current group of A fragments in registers.
      a_frag_t frag_a[LOAD_GROUP]{};

      // Compute this group from the current LDS panel.
  #pragma unroll
      for (int u = 0; u < LOAD_GROUP; ++u) {
        if constexpr (!ALIGNED_K) {
          if (u >= current_valid_kblocks) continue;
        }

        frag_a[u] = load_swmmac_a_direct<scalar_t, N, K_PER_INSTRUCTION>(
            A, A_stride, group_base + u, lane);
      }

      // Prefetch the next B panel into VGPRs before computing this group.
      swmmac_i32x4 next_panel[vecs_per_lane];

      if (has_next) {
        const int next_valid_kblocks =
            ALIGNED_K ? LOAD_GROUP
                      : ((total_num_kblocks - next_group_base) < LOAD_GROUP
                             ? (total_num_kblocks - next_group_base)
                             : LOAD_GROUP);

        prefetch_swmmac_b_panel<scalar_t, k_load, columns_per_wave,
                                THREADS_PER_WAVE>(
            next_panel, B, K, M, static_cast<int>(m),
            next_group_base * K_PER_INSTRUCTION, next_valid_kblocks);
      }

  #pragma unroll
      for (int u = 0; u < LOAD_GROUP; ++u) {
        if constexpr (!ALIGNED_K) {
          if (u >= current_valid_kblocks) continue;
        }

        const int global_kblock = group_base + u;

  #pragma unroll
        for (int tile = 0; tile < M_TILES_PER_WAVE; ++tile) {
          const uint32_t tile_m =
              m + static_cast<uint32_t>(tile * TILE_COLUMNS);
          if (tile_m >= static_cast<uint32_t>(M)) continue;

          const b_frag_t frag_b =
              load_swmmac_b_lds_row_major<scalar_t, k_load_pad, TILE_COLUMNS,
                                          K_PER_INSTRUCTION>(
                  &b_lds[wave_id][current_buffer][0][0], tile, u, lane);

          if (global_kblock & 1) {
            acc1[tile] =
                traits_t::mma(frag_a[u], frag_b, acc1[tile], sparse_idx);
          } else {
            acc0[tile] =
                traits_t::mma(frag_a[u], frag_b, acc0[tile], sparse_idx);
          }
        }
      }

      // Commit the prefetched next panel into the alternate LDS buffer.
      if (has_next) {
        commit_swmmac_b_panel<scalar_t, k_load, k_load_pad, columns_per_wave,
                              THREADS_PER_WAVE>(
            &b_lds[wave_id][next_buffer][0][0], next_panel);

        swmmac_lds_wave_fence();
      }
    }

    // Merge even/odd K-block accumulators.
  #pragma unroll
    for (int tile = 0; tile < M_TILES_PER_WAVE; ++tile) {
  #pragma unroll
      for (int r = 0; r < 8; ++r) {
        acc0[tile][r] += acc1[tile][r];
      }
    }

    // Store the final C tile. Each lane owns one output column.
  #pragma unroll
    for (int tile = 0; tile < M_TILES_PER_WAVE; ++tile) {
      const int col = static_cast<int>(m) + tile * TILE_COLUMNS + j;
      if (col >= M) continue;

  #pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int row = row_base + r;
        if (row >= N) continue;

        float value = acc0[tile][2 * r] + acc0[tile][2 * r + 1];

        if constexpr (HAS_BIAS) {
          value += traits_t::to_float(
              BIAS[swmmac_bias_index(row, col, N, M, Bx, By)]);
        }

        C[row * M + col] = traits_t::from_float(value);
      }
    }

    m += static_cast<uint32_t>(gridDim.x) *
         static_cast<uint32_t>(active_waves) * columns_per_wave;
  }
}

// Split-K SWMMAC path. Compute different K shards for the same output
// tile, then reduce the partial sums through LDS.
template <typename scalar_t, int N, bool HAS_BIAS, int SPLIT_K, int LOAD_GROUP,
          bool ALIGNED_SHARDS, int TILE_COLUMNS, int M_TILES_PER_WAVE,
          int K_PER_INSTRUCTION, int WV_PER_GROUP, int THREADS_PER_WAVE>
__global__ void __launch_bounds__(WV_PER_GROUP* THREADS_PER_WAVE)
    swmmacGemmSplitK(const int K, const int A_stride, const int M, const int Bx,
                     const int By, const scalar_t* __restrict__ A,
                     const scalar_t* __restrict__ B,
                     const scalar_t* __restrict__ BIAS,
                     scalar_t* __restrict__ C) {
  static_assert(N >= 5 && N <= 8);

  using traits_t = swmmac_traits<scalar_t>;
  using a_frag_t = typename traits_t::a_frag_t;
  using b_frag_t = typename traits_t::b_frag_t;

  constexpr int columns_per_wave = TILE_COLUMNS * M_TILES_PER_WAVE;
  constexpr int k_load = LOAD_GROUP * K_PER_INSTRUCTION;
  constexpr int k_load_pad = k_load + 8;
  constexpr int vecs_per_lane =
      (columns_per_wave * k_load) / (8 * THREADS_PER_WAVE);

  // One partial C tile per K shard. The +1 column padding helps avoid LDS
  // bank conflicts during the reduction.
  __shared__ float partials_lds[SPLIT_K][N][columns_per_wave + 1];

  // Each K shard owns its own pair of B ping-pong panels.
  __shared__ __align__(16)
      scalar_t b_lds[SPLIT_K][2][columns_per_wave][k_load_pad];

  const int total_num_kblocks = K / K_PER_INSTRUCTION;

  // threadIdx.y selects which K shard this wave computes.
  const int shard_id = static_cast<int>(threadIdx.y);
  const int shard_begin_kblock = (total_num_kblocks * shard_id) / SPLIT_K;
  const int shard_end_kblock = (total_num_kblocks * (shard_id + 1)) / SPLIT_K;
  const int shard_num_kblocks = shard_end_kblock - shard_begin_kblock;

  const int lane = static_cast<int>(threadIdx.x);
  const int j = lane & 15;
  const int row_base = (lane >> 4) * 4;
  const int32_t sparse_idx = (lane & 1) ? kSwmmacSelect23 : kSwmmacSelect01;

  uint32_t m = static_cast<uint32_t>(blockIdx.x) * columns_per_wave;

  while (m < static_cast<uint32_t>(M)) {
    swmmac_f32x8 acc0[M_TILES_PER_WAVE]{};
    swmmac_f32x8 acc1[M_TILES_PER_WAVE]{};

    // Load this shard's first B panel into LDS buffer 0.
    swmmac_i32x4 initial_panel[vecs_per_lane];
    const int first_valid_kblocks =
        shard_num_kblocks < LOAD_GROUP ? shard_num_kblocks : LOAD_GROUP;

    prefetch_swmmac_b_panel<scalar_t, k_load, columns_per_wave,
                            THREADS_PER_WAVE>(
        initial_panel, B, K, M, static_cast<int>(m),
        shard_begin_kblock * K_PER_INSTRUCTION, first_valid_kblocks);

    commit_swmmac_b_panel<scalar_t, k_load, k_load_pad, columns_per_wave,
                          THREADS_PER_WAVE>(&b_lds[shard_id][0][0][0],
                                            initial_panel);

    swmmac_lds_wave_fence();

    for (int local_group_base = 0; local_group_base < shard_num_kblocks;
         local_group_base += LOAD_GROUP) {
      // Ping-pong between this shard's two B LDS buffers.
      const int current_buffer = (local_group_base / LOAD_GROUP) & 1;
      const int next_buffer = current_buffer ^ 1;
      const int next_local_group_base = local_group_base + LOAD_GROUP;
      const bool has_next = next_local_group_base < shard_num_kblocks;

      const int current_valid_kblocks =
          ALIGNED_SHARDS ? LOAD_GROUP
                         : ((shard_num_kblocks - local_group_base) < LOAD_GROUP
                                ? (shard_num_kblocks - local_group_base)
                                : LOAD_GROUP);

      // Keep this shard-local group of A fragments in registers.
      a_frag_t frag_a[LOAD_GROUP]{};

      // Compute this shard's partial sums from the current LDS panel.
  #pragma unroll
      for (int u = 0; u < LOAD_GROUP; ++u) {
        if constexpr (!ALIGNED_SHARDS) {
          if (u >= current_valid_kblocks) continue;
        }

        frag_a[u] = load_swmmac_a_direct<scalar_t, N, K_PER_INSTRUCTION>(
            A, A_stride, shard_begin_kblock + local_group_base + u, lane);
      }

      // Prefetch this shard's next B panel into VGPRs.
      swmmac_i32x4 next_panel[vecs_per_lane];

      if (has_next) {
        const int next_valid_kblocks =
            ALIGNED_SHARDS
                ? LOAD_GROUP
                : ((shard_num_kblocks - next_local_group_base) < LOAD_GROUP
                       ? (shard_num_kblocks - next_local_group_base)
                       : LOAD_GROUP);

        prefetch_swmmac_b_panel<scalar_t, k_load, columns_per_wave,
                                THREADS_PER_WAVE>(
            next_panel, B, K, M, static_cast<int>(m),
            (shard_begin_kblock + next_local_group_base) * K_PER_INSTRUCTION,
            next_valid_kblocks);
      }

  #pragma unroll
      for (int u = 0; u < LOAD_GROUP; ++u) {
        if constexpr (!ALIGNED_SHARDS) {
          if (u >= current_valid_kblocks) continue;
        }

        const int global_kblock = shard_begin_kblock + local_group_base + u;

  #pragma unroll
        for (int tile = 0; tile < M_TILES_PER_WAVE; ++tile) {
          const uint32_t tile_m =
              m + static_cast<uint32_t>(tile * TILE_COLUMNS);
          if (tile_m >= static_cast<uint32_t>(M)) continue;

          const b_frag_t frag_b =
              load_swmmac_b_lds_row_major<scalar_t, k_load_pad, TILE_COLUMNS,
                                          K_PER_INSTRUCTION>(
                  &b_lds[shard_id][current_buffer][0][0], tile, u, lane);

          if (global_kblock & 1) {
            acc1[tile] =
                traits_t::mma(frag_a[u], frag_b, acc1[tile], sparse_idx);
          } else {
            acc0[tile] =
                traits_t::mma(frag_a[u], frag_b, acc0[tile], sparse_idx);
          }
        }
      }

      // Commit the prefetched next panel into the alternate LDS buffer.
      if (has_next) {
        commit_swmmac_b_panel<scalar_t, k_load, k_load_pad, columns_per_wave,
                              THREADS_PER_WAVE>(
            &b_lds[shard_id][next_buffer][0][0], next_panel);

        swmmac_lds_wave_fence();
      }
    }

    // Merge even/odd K-block accumulators for this shard.
  #pragma unroll
    for (int tile = 0; tile < M_TILES_PER_WAVE; ++tile) {
  #pragma unroll
      for (int r = 0; r < 8; ++r) {
        acc0[tile][r] += acc1[tile][r];
      }
    }

    // Store this shard's partial C tile into LDS.
  #pragma unroll
    for (int tile = 0; tile < M_TILES_PER_WAVE; ++tile) {
      const int local_col = tile * TILE_COLUMNS + j;
      const int col = static_cast<int>(m) + local_col;
      if (col >= M) continue;

  #pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int row = row_base + r;
        if (row < N) {
          partials_lds[shard_id][row][local_col] =
              acc0[tile][2 * r] + acc0[tile][2 * r + 1];
        }
      }
    }

    // Wait for all K shards, then reduce partials and write final C.
    __syncthreads();

    const int epilogue_local_col = lane;
    const int epilogue_col = static_cast<int>(m) + epilogue_local_col;

    if (lane < columns_per_wave && epilogue_col < M) {
  #pragma unroll
      for (int r_step = 0; r_step < 2; ++r_step) {
        const int row = shard_id + r_step * SPLIT_K;
        if (row >= N) continue;

        float value = 0.0f;

  #pragma unroll
        for (int shard = 0; shard < SPLIT_K; ++shard) {
          value += partials_lds[shard][row][epilogue_local_col];
        }

        if constexpr (HAS_BIAS) {
          value += traits_t::to_float(
              BIAS[swmmac_bias_index(row, epilogue_col, N, M, Bx, By)]);
        }

        C[row * M + epilogue_col] = traits_t::from_float(value);
      }
    }

    __syncthreads();

    m += static_cast<uint32_t>(gridDim.x) * columns_per_wave;
  }
}

#else

// Matching host-compilation stubs. HIP-Clang parses kernel launches in the
// host pass as well, so the signatures must exactly match the device branch.

template <typename scalar_t, int N, bool HAS_BIAS, int LOAD_GROUP,
          bool ALIGNED_K, int TILE_COLUMNS, int M_TILES_PER_WAVE,
          int K_PER_INSTRUCTION, int WV_PER_GROUP, int THREADS_PER_WAVE>
__global__ void __launch_bounds__(WV_PER_GROUP* THREADS_PER_WAVE)
    swmmacGemmRegular(const int, const int, const int, const int, const int,
                      const scalar_t*, const scalar_t*, const scalar_t*,
                      scalar_t*) {
  UNREACHABLE_CODE
}

template <typename scalar_t, int N, bool HAS_BIAS, int SPLIT_K, int LOAD_GROUP,
          bool ALIGNED_SHARDS, int TILE_COLUMNS, int M_TILES_PER_WAVE,
          int K_PER_INSTRUCTION, int WV_PER_GROUP, int THREADS_PER_WAVE>
__global__ void __launch_bounds__(WV_PER_GROUP* THREADS_PER_WAVE)
    swmmacGemmSplitK(const int, const int, const int, const int, const int,
                     const scalar_t*, const scalar_t*, const scalar_t*,
                     scalar_t*){UNREACHABLE_CODE}

#endif

torch::Tensor swmmacGEMM(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t logical_M, const int64_t CuCount) {
  TORCH_CHECK(on_gfx12(), "swmmacGEMM is only supported on GFX12");
  TORCH_CHECK(in_b.scalar_type() == torch::kFloat16 ||
                  in_b.scalar_type() == torch::kBFloat16,
              "B must be FP16 or BF16");
  TORCH_CHECK(in_a.scalar_type() == in_b.scalar_type(),
              "A and B must use the same dtype");
  TORCH_CHECK(in_b.dim() == 2, "B must be a 2D tensor");
  TORCH_CHECK(in_a.dim() == 2, "A must be a 2D tensor");

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
    TORCH_CHECK(in_bias->scalar_type() == in_a.scalar_type(),
                "bias and A must use the same dtype");
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

#if defined(__HIP__GFX9__) || defined(__HIP__GFX1X__)
// This version targets cases where A[] fits LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_sml_(const int K, const int Kbp, const int Kap, const int M,
                     const int Bx, const int By, const scalar_t* B,
                     const scalar_t* __restrict__ A,
                     const scalar_t* __restrict__ BIAS, scalar_t* C,
                     const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  #if defined(__HIP__MI3XX__)
  constexpr bool use_mfma = (std::is_same_v<scalar_t, __hip_bfloat16>);
  #else
  constexpr bool use_mfma = false;
  #endif
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };

  //----------------------------------------------------
  // Reserving 64/160 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //     then this is not going to work!
  //----------------------------------------------------
  __shared__ scalar_t s[max_lds_len];

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  //----------------------------------------------------
  // Each wave works on a single column of weight matrix.
  // There are 16 waves per WG, and hence, each WG is
  // working on 16 columns of weight matrix. Moreover,
  // we tile in column direction by YTILE, so when YTILE=1
  // the above math is right, however, when YTILE=2 then
  // each wave  will be working on 2 columns and WG will
  // be working on 32 columns.
  //
  // Top level loop that makes WGs persistent!
  // - WGs iterates across columns of weight matrix
  // - Each wave within WG works on a given column(s)
  // - After completing first set of columns, WGs start
  //   working on the next set of available columns
  //----------------------------------------------------
  while (m < M) {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    float sum[N][YTILE] = {};
    scalar8 sum4[N][YTILE] = {};

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
        }
      }

      // Do the matrix multiplication in interleaved manner
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
              for (uint32_t b = 0; b < A_CHUNK / 4; b++)
                sum4[n][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
                    bigA[n][k2].h4[b], bigB[y][k2].h4[b], sum4[n][y], 0, 0, 0);
          }
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    if constexpr (!use_mfma) {
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf,
                                                1);  // row_shr8
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf,
                                                1);  // row_shr4
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf,
                                                1);  // row_shr2
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf,
                                                1);  // row_shr1
  #if defined(__HIP__GFX9__)
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x142, 0xf, 0xf,
                                                1);  // ROW_BCAST15
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x143, 0xf, 0xf,
                                                1);  // ROW_BCAST31
  #else
          sum[n][y] += __shfl_xor(sum[n][y], 16);
  #endif
        }
      }

      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (std::is_same_v<scalar_t, half>) {
              sum[n][y] += __half2float(biases[n][y]);
            } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
              sum[n][y] += __bfloat162float(biases[n][y]);
            }
            C[m + y + n * M] = __float2s<scalar_t>(sum[n][y]);
          }
        }
      }
    } else {
  #ifdef __HIP__GFX9__
    #pragma unroll
      for (int n = 0; n < N; n++) {
    #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          /*float accm1 = 0;
           for (int i=0; i<64; i++)
              accm1 += __shfl(sum4[n][y][i%4], i);
          sum4[n][y][0] = accm1;*/
          float accm = sum4[n][y][0];
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][1], 0x101, 0xf, 0xf,
                                           1);  // row_shl1
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][2], 0x102, 0xf, 0xf,
                                           1);  // row_shl2
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][3], 0x103, 0xf, 0xf,
                                           1);  // row_shl3
          accm += __builtin_amdgcn_mov_dpp(accm, 0x104, 0xf, 0xf,
                                           1);  // row_shl4
          accm += __builtin_amdgcn_mov_dpp(accm, 0x108, 0xf, 0xf,
                                           1);  // row_shl8
          accm = __builtin_amdgcn_mov_dpp(accm, 0x11f, 0xf, 0xf,
                                          1);  // row_shr15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x142, 0xf, 0xf,
                                           1);  // ROW_BCAST15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x143, 0xf, 0xf,
                                           1);  // ROW_BCAST31

          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            sum4[n][y][0] += __bfloat162float(biases[n][y]);
            C[m + y + n * M] = __float2bfloat16(sum4[n][y][0]);
          }
        }
      }
  #endif  // __HIP__GFX9__ (MFMA path)
    }
    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_hf_sml_(const int K, const int Kbp, const int Kap,
                                 const int M, const int Bx, const int By,
                                 const scalar_t* B,
                                 const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ BIAS, scalar_t* C,
                                 const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE
}
#endif

#if defined(__HIP__GFX9__) || defined(__HIP__GFX1X__)
// This version targets cases where A[] marginally exceeds LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_(const int K, const int Kbp, const int Kap, const int M,
                 const int Bx, const int By, const scalar_t* B,
                 const scalar_t* __restrict__ A,
                 const scalar_t* __restrict__ BIAS, scalar_t* C,
                 const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  #if defined(__HIP__MI3XX__)
  constexpr bool use_mfma = (std::is_same_v<scalar_t, __hip_bfloat16>);
  #else
  constexpr bool use_mfma = false;
  #endif

  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };

  __shared__ scalar_t s[max_lds_len];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  uint32_t m = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmentation!
  // This will happen only for the last wave!
  if (m < M && (m + YTILE) >= M) {
    uint32_t startColumn = M - YTILE;
    for (uint32_t i = 0; i < (m - startColumn); i++) {
      commitColumn[i] = 0;
    }
    m = startColumn;
  }

  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }

  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  while (m < M) {
    float sum[N][YTILE] = {};
    scalar8 sum4[N][YTILE] = {};

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          if (k_ + Kap * n < max_lds_len)
            bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
          else
            bigA[n][k2] = *((const bigType*)(&(A[k_ + Kap * n])));
        }
      }

      // Do the matrix multiplication in interleaved manner
      for (uint32_t n = 0; n < N; n++) {
        for (uint32_t k2 = 0; k2 < UNRL; k2++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
              for (uint32_t b = 0; b < A_CHUNK / 4; b++)
                sum4[n][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
                    bigA[n][k2].h4[b], bigB[y][k2].h4[b], sum4[n][y], 0, 0, 0);
          }
        }
      }
    }

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    if constexpr (!use_mfma) {
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf,
                                                1);  // row_shr8
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf,
                                                1);  // row_shr4
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf,
                                                1);  // row_shr2
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf,
                                                1);  // row_shr1
  #if defined(__HIP__GFX9__)
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x142, 0xf, 0xf,
                                                1);  // ROW_BCAST15
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x143, 0xf, 0xf,
                                                1);  // ROW_BCAST31
  #else
          sum[n][y] += __shfl_xor(sum[n][y], 16);
  #endif
        }
      }

      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if (commitColumn[y]) {
              if constexpr (std::is_same_v<scalar_t, half>) {
                sum[n][y] += __half2float(biases[n][y]);
              } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
                sum[n][y] += __bfloat162float(biases[n][y]);
              }
              C[m + y + n * M] = __float2s<scalar_t>(sum[n][y]);
            }
          }
        }
      }
    } else {
  #ifdef __HIP__GFX9__
    #pragma unroll
      for (int n = 0; n < N; n++) {
    #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          // float accm1 = 0;
          // for (int i=0; i<64; i++)
          //    accm1 += __shfl(sum4[n][y][i%4], i);
          float accm = sum4[n][y][0];
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][1], 0x101, 0xf, 0xf,
                                           1);  // row_shl1
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][2], 0x102, 0xf, 0xf,
                                           1);  // row_shl2
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][3], 0x103, 0xf, 0xf,
                                           1);  // row_shl3
          accm += __builtin_amdgcn_mov_dpp(accm, 0x104, 0xf, 0xf,
                                           1);  // row_shl4
          accm += __builtin_amdgcn_mov_dpp(accm, 0x108, 0xf, 0xf,
                                           1);  // row_shl8
          accm = __builtin_amdgcn_mov_dpp(accm, 0x11f, 0xf, 0xf,
                                          1);  // row_shr15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x142, 0xf, 0xf,
                                           1);  // ROW_BCAST15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x143, 0xf, 0xf,
                                           1);  // ROW_BCAST31
          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if (commitColumn[y]) {
              sum4[n][y][0] += __bfloat162float(biases[n][y]);
              C[m + y + n * M] = __float2bfloat16(sum4[n][y][0]);
            }
          }
        }
      }
  #endif  // __HIP__GFX9__ (MFMA path)
    }

    m += CuCount * _WvPrGrp * YTILE;

    // Check whether there will be fragmentation!
    // This will happen only for the last wave!
    if (m < M && (m + YTILE) >= M) {
      uint32_t startColumn = M - YTILE;
      for (uint32_t i = 0; i < (m - startColumn); i++) {
        commitColumn[i] = 0;
      }
      m = startColumn;
    }
  }
}

#else
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_hf_(const int K, const int Kbp, const int Kap,
                             const int M, const int Bx, const int By,
                             const scalar_t* B, const scalar_t* __restrict__ A,
                             const scalar_t* __restrict__ BIAS, scalar_t* C,
                             const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE
}
#endif

#if defined(__HIP__GFX9__) || defined(__HIP__GFX1X__)
// This version targets big A[] cases, where it is much larger than LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_big_(const int K, const int Kbp, const int Kap, const int M,
                     const int Bx, const int By, const scalar_t* B,
                     const scalar_t* __restrict__ A,
                     const scalar_t* __restrict__ BIAS, scalar_t* C,
                     const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  #if defined(__HIP__MI3XX__)
  constexpr bool use_mfma = (std::is_same_v<scalar_t, __hip_bfloat16>);
  #else
  constexpr bool use_mfma = false;
  #endif

  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };

  //----------------------------------------------------
  // Reserving 64/160 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //     then this is not going to work!
  //----------------------------------------------------
  __shared__ scalar_t s[max_lds_len];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  // int _WvPrGrp = mindiv(N, CuCount * YTILE, WvPrGrp);
  if (threadIdx.y >= _WvPrGrp) return;

  //----------------------------------------------------
  // Indexing function into the column of weight matrix B
  // Algorithm does 64 lane k-splitting / wave and uses
  // WG ID and Thread ID to find the index.
  //----------------------------------------------------
  uint32_t m = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmentation!
  // This will happen only for the last wave!
  if (m < M && (m + YTILE) >= M) {
    uint32_t startColumn = M - YTILE;
    for (uint32_t i = 0; i < (m - startColumn); i++) {
      commitColumn[i] = 0;
    }
    m = startColumn;
  }

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  #define PCML
  #ifndef PCML
  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
    #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
    #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
    #endif
  }
  __syncthreads();
  #endif

  #define TUC (THRDS * UNRL * A_CHUNK)
  uint32_t kBase = 0;
  // find biggest k size that fits in LDS
  uint32_t kFit = (max_lds_len) / N;
  // kFit = (kFit%TWC==0) ? kFit : (kFit-kFit%TWC+TWC); //round up to multiple
  // of TUC
  kFit = (kFit % TUC == 0)
             ? kFit
             : (kFit - kFit % TUC);  // round up to multiple of TUC
  // if (kFit == 0) kFit = TUC;
  kFit = min__(kFit, Kap);

  //----------------------------------------------------
  // Each wave works on a single column of weight matrix.
  // There are 16 waves per WG, and hence, each WG is
  // working on 16 columns of weight matrix. Moreover,
  // we tile in column direction by YTILE, so when YTILE=1
  // the above math is right, however, when YTILE=2 then
  // each wave  will be working on 2 columns and WG will
  // be working on 32 columns.
  //
  // Top level loop that makes WGs persistent!
  // - WGs iterates across columns of weight matrix
  // - Each wave within WG works on a given column(s)
  // - After completing first set of columns, WGs start
  //   working on the next set of available columns
  //----------------------------------------------------
  #ifdef PCML
  int YW = (YTILE * _WvPrGrp);
  uint32_t Mrndp = (M % YW == 0) ? M : (M - M % YW + YW);
  while (m < Mrndp) {
  #else
  while (m < M) {
  #endif
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    float sum[N][YTILE] = {};
    scalar8 sum4[N][YTILE] = {};

    //----------------------------------------------------
    // Fetch weight matrix B in interleaved K-split!
    // - Each thread (lane) is fetching 8 elements (A_Chunk)
    // - Each wave will fetch 64*8=> 512 elements (1024B)
    // - YTILE represents the number of column being serviced
    //   by wave
    // - Loop for fetching weight matrix (B) are unrolled
    //
    // Fetch activation matrix A from LDS
    // - Loop for fetching activation matrix (A) are unrolled
    //
    // Finally, do the matrix multiplication in an unrolled
    // fashion. This provides lot of food for compiler
    // scheduling.
    //
    // TODO: Logic below will only work when K is multiple of 8
    //----------------------------------------------------
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];

  #ifdef PCML
      if ((k1 == 0) || (k1 == kBase + kFit)) {  // load next chunk of A[] to LDS
        if (k1 != 0) kBase += kFit;
        __syncthreads();
        for (uint32_t k = 0; k < kFit; k += THRDS * _WvPrGrp * A_CHUNK) {
          uint32_t kOff = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);
          if (kBase + kOff >= Kap) break;
          if (kOff >= kFit) break;
          for (uint32_t n = 0; n < N; n++) {
            uint32_t k_in = kBase + n * Kap + kOff;
            uint32_t k_ot = n * kFit + kOff;
    #if defined(__gfx950__)
            __builtin_amdgcn_global_load_lds((int*)(&A[k_in]), (int*)(&s[k_ot]),
                                             16, 0, 0);
    #else
            *((bigType*)(&s[k_ot])) = *((bigType*)(&A[k_in]));
    #endif
          }
        }
        __syncthreads();
      }
      if (m >= M) continue;
  #endif

      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
  #ifdef PCML
          bigA[n][k2] = *((const bigType*)(&(s[k_ - kBase + kFit * n])));
  #else
          if (k_ + Kap * n < max_lds_len)
            bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
          else
            bigA[n][k2] = *((const bigType*)(&(A[k_ + Kap * n])));
  #endif
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
              for (uint32_t b = 0; b < A_CHUNK / 4; b++)
                sum4[n][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
                    bigA[n][k2].h4[b], bigB[y][k2].h4[b], sum4[n][y], 0, 0, 0);
          }
        }
      }
    }

  #ifdef PCML
    if (m >= M) {
      m += CuCount * _WvPrGrp * YTILE;
      kBase = 0;
      continue;
    }
  #endif

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    if constexpr (!use_mfma) {
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf,
                                                1);  // row_shr8
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf,
                                                1);  // row_shr4
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf,
                                                1);  // row_shr2
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf,
                                                1);  // row_shr1
  #if defined(__HIP__GFX9__)
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x142, 0xf, 0xf,
                                                1);  // ROW_BCAST15
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x143, 0xf, 0xf,
                                                1);  // ROW_BCAST31
  #else
          sum[n][y] += __shfl_xor(sum[n][y], 16);
  #endif
        }
      }

      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if (commitColumn[y]) {
              if constexpr (std::is_same_v<scalar_t, half>) {
                sum[n][y] += __half2float(biases[n][y]);
              } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
                sum[n][y] += __bfloat162float(biases[n][y]);
              }
              C[m + y + n * M] = __float2s<scalar_t>(sum[n][y]);
            }
          }
        }
      }
    } else {
  #ifdef __HIP__GFX9__
    #pragma unroll
      for (int n = 0; n < N; n++) {
    #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          float accm = sum4[n][y][0];
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][1], 0x101, 0xf, 0xf,
                                           1);  // row_shl1
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][2], 0x102, 0xf, 0xf,
                                           1);  // row_shl2
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][3], 0x103, 0xf, 0xf,
                                           1);  // row_shl3
          accm += __builtin_amdgcn_mov_dpp(accm, 0x104, 0xf, 0xf,
                                           1);  // row_shl4
          accm += __builtin_amdgcn_mov_dpp(accm, 0x108, 0xf, 0xf,
                                           1);  // row_shl8
          accm = __builtin_amdgcn_mov_dpp(accm, 0x11f, 0xf, 0xf,
                                          1);  // row_shr15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x142, 0xf, 0xf,
                                           1);  // ROW_BCAST15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x143, 0xf, 0xf,
                                           1);  // ROW_BCAST31
          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if (commitColumn[y]) {
              sum4[n][y][0] += __bfloat162float(biases[n][y]);
              C[m + y + n * M] = __float2bfloat16(sum4[n][y][0]);
            }
          }
        }
      }
  #endif  // __HIP__GFX9__ (MFMA path)
    }

    m += CuCount * _WvPrGrp * YTILE;
    kBase = 0;

    // Check whether there will be fragmentation!
    // This will happen only for the last wave!
    if (m < M && (m + YTILE) >= M) {
      uint32_t startColumn = M - YTILE;
      for (uint32_t i = 0; i < (m - startColumn); i++) {
        commitColumn[i] = 0;
      }
      m = startColumn;
    }
  }
}
#else
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_hf_big_(const int K, const int Kbp, const int Kap,
                                 const int M, const int Bx, const int By,
                                 const scalar_t* B,
                                 const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ BIAS, scalar_t* C,
                                 const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE
}
#endif

// Find the min val of div2 that doesn't increase N/(div1*div2)
int mindiv(int N, int div1, int div2) {
  int nPrRnd = div1 * div2;
  int rnds[13];
  for (int i = 0; i < 13; i++) {
    rnds[i] = (N + nPrRnd - 1) / nPrRnd;
    nPrRnd -= div1;
  }
  for (int i = 12; i >= 0; i--)
    if (rnds[0] == rnds[i]) return (div2 - i);
  return 0;
}

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

    const bool use_wave32 = on_gfx1x();
    switch (N_in) {
      case 1:
        if (use_wave32)
          WVSPLIT_TILE_CFG(32, 16, sYT, 1)
        else
          WVSPLIT_TILE_CFG(64, 16, sYT, 1)
        break;
      case 2:
        if (use_wave32)
          WVSPLIT_TILE_CFG(32, 16, sYT, 2)
        else
          WVSPLIT_TILE_CFG(64, 16, sYT, 2)
        break;
      case 3:
        if (use_wave32)
          WVSPLIT_TILE_CFG(32, 16, sYT, 3)
        else
          WVSPLIT_TILE_CFG(64, 16, sYT, 3)
        break;
      case 4:
        if (use_wave32)
          WVSPLIT_TILE_CFG(32, 16, sYT, 4)
        else
          WVSPLIT_TILE_CFG(64, 16, sYT, 4)
        break;
      case 5:
        if (use_wave32)
          WVSPLIT_TILE_CFG(32, 16, sYT, 5)
        else
          WVSPLIT_TILE_CFG(64, 16, sYT, 5)
        break;
      default:
        throw std::runtime_error(
            "Unsupported N value: " + std::to_string(M_in) + "," +
            std::to_string(K_in) + "," + std::to_string(N_in));
    }
  });
  return out_c;
}

// This version targets cases skinny where CUs are not filled
// Wave-SplitK is used with reduction done via atomics.
#if defined(__gfx950__)
  #define WVSPLITKRC_1KPASS
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GrpsShrB, int CHUNKK, int DTRMNSTC>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    __attribute__((amdgpu_waves_per_eu(1, 1)))
    wvSplitKrc_(const int actlN, const int K, const int Kap, const int M,
                const int Bx, const int By, const scalar_t* __restrict__ A,
                const scalar_t* __restrict__ B,
                const scalar_t* __restrict__ BIAS, float* glbl, int* cntr,
                scalar_t* C, const int CuCount) {
  constexpr int NTILE = 16;
  constexpr int APAD = 1;
  constexpr int ASTRD = 64;
  constexpr int BPAD = 1;
  constexpr int WVLDS_ = THRDS * A_CHUNK / CHUNKK;
  constexpr int WVLDS = ((WVLDS_ + A_CHUNK * BPAD)) * YTILE;

  constexpr int max_lds_len = LDS_SIZE / 2;

  using scalar16 =
      __attribute__((__vector_size__((A_CHUNK * 2) * sizeof(float)))) float;
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    unsigned int i[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    unsigned long l[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };
  using big4 = __attribute__((__vector_size__(4 * sizeof(bigType)))) __bf16;

  __shared__ scalar_t stg[WvPrGrp * WVLDS / GrpsShrB];
  unsigned int* myStg = (unsigned int*)(&stg[WVLDS * (threadIdx.y / GrpsShrB)]);
  __shared__ scalar_t s[max_lds_len - WvPrGrp * WVLDS / GrpsShrB];

  #ifndef WVSPLITKRC_1KPASS
  constexpr int TUC_ = (THRDS * UNRL * A_CHUNK);
  // find biggest k size that fits padded into LDS
  constexpr uint32_t kFit__ = (max_lds_len - WvPrGrp * WVLDS / GrpsShrB) / N;
  constexpr uint32_t kFit_ = (kFit__ * ASTRD) / (APAD + ASTRD);
  uint32_t kFit = kFit_ - (kFit_ % TUC_);
  uint32_t kfitsPerRdc = (K + kFit - 1) / kFit;

  // find best k split to fill the CUs
  if (((K + kfitsPerRdc * kFit - 1) / (kfitsPerRdc * kFit)) * numCuWithFullK <=
      CuCount)
    while (true) {
      while (kFit > TUC_) {
        uint32_t kFit_ = kFit - TUC_;
        if (((K + (kfitsPerRdc * kFit_ - 1)) / (kfitsPerRdc * kFit_)) *
                numCuWithFullK >
            CuCount)
          break;
        kFit = kFit_;
      }
      if (((K + ((kfitsPerRdc - 1) * kFit - 1)) / ((kfitsPerRdc - 1) * kFit)) *
              numCuWithFullK <=
          CuCount)
        kfitsPerRdc--;
      else
        break;
    }
  #else
  int constexpr kFit = 512 / CHUNKK;
  int constexpr kfitsPerRdc = 1;
  #endif

  bool doRdc = true;  // Assuming (kfitsPerRdc * kFit < K) is always true
  uint32_t numCuWithFullK =
      ((M + (WvPrGrp * YTILE / GrpsShrB) - 1) / (WvPrGrp * YTILE / GrpsShrB));
  uint32_t Mmod = numCuWithFullK * (WvPrGrp * YTILE / GrpsShrB);

  // given above k-split, find this wave's position
  uint32_t kFitPdd = kFit * CHUNKK + ((kFit * CHUNKK) / ASTRD) * APAD;
  uint32_t m0 = (blockIdx.x * WvPrGrp / GrpsShrB) * YTILE;
  uint32_t m1 = ((threadIdx.y % WvPrGrp) / GrpsShrB) * YTILE;
  uint32_t m = (m0 + m1) % Mmod;
  const uint32_t k_str = (m0 / Mmod) * kFit * kfitsPerRdc;
  uint32_t k_end = (m0 / Mmod + 1) * kFit * kfitsPerRdc;
  const uint32_t k_rnd = (K + kFit * kfitsPerRdc - 1) / (kFit * kfitsPerRdc);

  scalar8 sum4[N / NTILE / GrpsShrB][1] = {0};
  bigType bigB_[YTILE / GrpsShrB / CHUNKK][UNRL];
  const uint32_t bLoader = (threadIdx.y % GrpsShrB);
  uint32_t kBase = 0;
  if (k_str >= K) return;
  if (m >= Mmod) return;

  bool noreloada = false;
  constexpr bool FAST_UNSAFE_RDC_INIT = false;

  #ifdef WVSPLITKRC_1KPASS
  // Early glbl init, B[] loading, if 1KPASS
  if constexpr (FAST_UNSAFE_RDC_INIT) {
    if (m + (threadIdx.x % 16) < M)
      if (doRdc)
        if (k_str == 0) {
          int mindx = m + (threadIdx.x % 16);
          int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                       (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
          int adr_ = mindx + M * nindx_ / 4;
          __hip_atomic_store(&cntr[adr_], 0, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              int adr = mindx + M * nindx;
              __hip_atomic_store(&glbl[adr], 0, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
            }
          }
        }
  }

    // Load first B[] chunk
    #pragma unroll
  for (uint32_t k2 = 0; k2 < UNRL; k2++) {
    uint32_t k = k_str + k2 * THRDS * A_CHUNK;
    uint32_t k_ = k + (threadIdx.x % (THRDS / CHUNKK)) * A_CHUNK;
    const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
    #pragma unroll
    for (uint32_t y = 0; y < YTILE / GrpsShrB; y += CHUNKK)
      bigB_[y / CHUNKK][k2].h8 = (loadnt(
          (scalar8*)(&B_[min__((y + threadIdx.x / (THRDS / CHUNKK)) * GrpsShrB +
                                   bLoader + m,
                               M - 1) *
                         K])));
  }
  {
  #else
  while (m < Mmod) {
  #endif

  #ifndef WVSPLITKRC_1KPASS
    if constexpr (FAST_UNSAFE_RDC_INIT) {
      if (m + (threadIdx.x % 16) < M)
        if (doRdc)
          if (k_str == 0) {
            int mindx = m + (threadIdx.x % 16);
            int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                         (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
            int adr_ = mindx + M * nindx_ / 4;
            __hip_atomic_store(&cntr[adr_], 0, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
              for (uint32_t j = 0; j < 4; j++) {
                int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                            (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
                int adr = mindx + M * nindx;
                __hip_atomic_store(&glbl[adr], 0, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
              }
            }
          }
    }

  #endif

  #ifndef WVSPLITKRC_1KPASS
    for (uint32_t k1 = k_str; k1 < k_end; k1 += THRDS * A_CHUNK * UNRL) {
  #else
    const uint32_t k1 = k_str;
    {
  #endif
  #ifndef WVSPLITKRC_1KPASS
      const bool reloada = (!noreloada) &&
                           ((k1 == k_str) || (k1 == k_str + kBase + kFit)) &&
                           (k1 < k_end);
      // load next chunk of A[] to LDS
      if (reloada) {
        if (k1 != k_str) kBase += kFit;
        __syncthreads();
  #else
      const bool reloada = (!noreloada) &&
                           ((k1 == k_str) || (k1 == k_str + kBase + kFit)) &&
                           (k1 < k_end);
      if (reloada) {
  #endif
        constexpr int sprdN = 4;
        const uint32_t thrd = threadIdx.x % (THRDS / CHUNKK);

  #ifndef WVSPLITKRC_1KPASS
    #pragma unroll
        for (int k = 0; k < kFit;
             k += (THRDS * (WvPrGrp / sprdN) * A_CHUNK) / CHUNKK) {
  #else
        const unsigned int k = 0;
        {
  #endif
          unsigned int kOff = k + (thrd * A_CHUNK);
          unsigned int kOffcp = min__(K - A_CHUNK, k_str + kOff);
          for (unsigned int n = 0; n < N; n += CHUNKK * sprdN) {
            __builtin_amdgcn_global_load_lds(
                (int*)(&A[min__(Kap * actlN - A_CHUNK,
                                kOffcp + Kap * (n / CHUNKK +
                                                (N / CHUNKK) * (threadIdx.x /
                                                                (64 / CHUNKK)) +
                                                (threadIdx.y % sprdN)))]),
                (int*)(&s[(k +
                           kFitPdd * ((n / CHUNKK) + (threadIdx.y % sprdN)))]),
                16, 0, 0);
          }

          // Stage loaded B[] to LDS for MFMA swizzling...
          for (uint32_t k2 = 0; k2 < UNRL; k2++) {
            uint32_t k = k1 + k2 * THRDS * A_CHUNK;
            uint32_t k_ = k + (threadIdx.x % (THRDS / CHUNKK)) * A_CHUNK;
            const bool oob_k = (k_ >= K);
            for (uint32_t y = 0; y < YTILE / GrpsShrB; y += CHUNKK) {
              uint32_t idx =
                  (threadIdx.x % (THRDS / CHUNKK)) * 4 +
                  ((y + threadIdx.x / (THRDS / CHUNKK)) * GrpsShrB + bLoader) *
                      ((THRDS / CHUNKK + BPAD) * 4);
              // zero out if oob
              *((scalar8*)&myStg[idx]) =
                  (oob_k)  // TODO: ever necessary (y*GrpsShrB+bLoader+m>=M) ?
                      ? 0
                      : bigB_[y / CHUNKK][k2].h8;
            }
          }
        }
      }
    }
  #ifndef WVSPLITKRC_1KPASS
    // Fire load of next B[] chunk...
    if ((k1 + THRDS * A_CHUNK * UNRL < k_end) &&
        (k1 + THRDS * A_CHUNK * UNRL < K))
    #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + THRDS * A_CHUNK * UNRL + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
    #pragma unroll
        for (uint32_t y = 0; y < YTILE / GrpsShrB; y += CHUNKK)
          bigB_[y / CHUNKK][k2].h8 = (loadnt(
              (scalar8*)(&B_[min__((y + threadIdx.x / (THRDS / CHUNKK)) *
                                           GrpsShrB +
                                       bLoader + m,
                                   M - 1) *
                             K])));
      }
  #endif

    // B[] staging is cooperative across GrpsShrB, so sync here before reading
    // back. This wait is currently inserted by compiler, but not guaranteed.
    asm volatile("s_waitcnt 0");
    __syncthreads();

    // read back B[] swizzled for MFMA...
    bigType bigB[YTILE / CHUNKK][UNRL];
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      for (uint32_t y = 0; y < YTILE / CHUNKK; y++) {
        unsigned int idx =
            (threadIdx.x % YTILE) * ((THRDS / CHUNKK + BPAD) * 4) +
            (threadIdx.x / YTILE) * 4 + y * 16;
        bigB[y][k2].h8 = *((scalar8*)&myStg[idx]);
      }
    }

    // rReadback A[] swizzled for MFMA...
    bigType bigA[N / GrpsShrB / CHUNKK][UNRL];
  #pragma unroll
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      uint32_t k = k1 + k2 * THRDS * A_CHUNK - kBase - k_str;
  #pragma unroll
      for (uint32_t nt = 0; nt < N / GrpsShrB; nt += NTILE)
  #pragma unroll
        for (uint32_t n = 0; n < NTILE / CHUNKK; n++) {
          uint32_t idxa =
              ((nt + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) % (N / CHUNKK) +
               (threadIdx.x % NTILE)) *
                  kFitPdd +
              ((nt + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) /
               (N / CHUNKK)) *
                  A_CHUNK * (64 / CHUNKK) +
              A_CHUNK * ((threadIdx.x / NTILE) + n * 4) + k;
          bigA[nt / CHUNKK + n][k2] = *((const bigType*)(&(s[idxa])));
        }
    }

    // Do the MFMAs
  #pragma unroll
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
  #pragma unroll
      for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
  #pragma unroll
        for (uint32_t j = 0; j < YTILE / CHUNKK; j++) {
          if constexpr (std::is_same_v<scalar_t, half>) {
            sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                bigA[nt * (YTILE / CHUNKK) + j][k2].h8, bigB[j][k2].h8,
                sum4[nt][0], 0, 0, 0);
          } else {  // bf16
            sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                bigA[nt * (YTILE / CHUNKK) + j][k2].h8, bigB[j][k2].h8,
                sum4[nt][0], 0, 0, 0);
          }
        }
      }
    }
  }

  union flt4 {
    scalar8 s8;
    float2 f2[2];
    float4 f4;
  };
  if (m + (threadIdx.x % 16) < M) {
    int my_cntr;
    int mindx = m + (threadIdx.x % 16);
    int g_mindx = m * 4 + (threadIdx.x % 64);  // coalesced atomic reduction
    scalar_t biases[N / NTILE / GrpsShrB][4] = {};
    // Atomic add the output, read biases
    for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
      int g_nindx =
          (nt * NTILE + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) / 4;
      int g_adr = g_mindx * 4 + 0 + M * g_nindx * 4;
      if (DTRMNSTC) {
        flt4 flt4_ = {.s8 = sum4[nt][0]};
        __hip_atomic_store((float2*)&glbl[g_adr + M * N * (m0 / Mmod)],
                           flt4_.f2[0], __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store((float2*)&glbl[g_adr + 2 + M * N * (m0 / Mmod)],
                           flt4_.f2[1], __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      } else {
        for (uint32_t j = 0; j < 4; j++)
          atomicAdd((&glbl[g_adr + j]), sum4[nt][0][j]);
      }
    }

    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                 (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
    int adr_ = mindx + M * nindx_ / 4;
    my_cntr = atomicAdd(&cntr[adr_], 1);

    // make sure LDS is free for write out staging
    if (DTRMNSTC) __syncthreads();

    // Update the complete counter
    flt4 vals[N / NTILE / GrpsShrB] = {};
    // If we're the last k-shard, read back the value and convert...
    if (my_cntr + 1 == k_rnd) {
      cntr[adr_] = 0;  // clear for next round
      if constexpr (DTRMNSTC) {
  #pragma unroll
        for (int ks = 0; ks < k_rnd; ks++) {
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            int g_nindx =
                (nt * NTILE + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) / 4;
            int g_adr = g_mindx * 4 + 0 + M * g_nindx * 4;
            __builtin_amdgcn_global_load_lds(
                (float4*)(&glbl[g_adr + M * N * ks]),
                &(((float4*)s)[(threadIdx.y * THRDS) + ks * THRDS * 4 +
                               nt * THRDS * 4 * k_rnd]),
                16, 0, 0);
          }
        }
        if (BIAS)
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              biases[nt][j] = BIAS[(mindx % Bx) + (nindx % By) * Bx];
            }
          }
        asm volatile("s_waitcnt 0");
        for (int ks = 0; ks < k_rnd; ks++) {
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            float4 eval = ((float4*)s)[(threadIdx.x + threadIdx.y * THRDS) +
                                       ks * THRDS * 4 + nt * THRDS * 4 * k_rnd];
            vals[nt].f4 += eval;
          }
        }
      } else {
        for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
          int g_nindx =
              (nt * NTILE + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) / 4;
          int g_adr = g_mindx * 4 + 0 + M * g_nindx * 4;
          vals[nt].f4 = *(float4*)(&glbl[g_adr]);
          *(float4*)(&glbl[g_adr]) = {};  // clear out for next round
        }
        if (BIAS)
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              biases[nt][j] = BIAS[(mindx % Bx) + (nindx % By) * Bx];
            }
          }
      }
      __builtin_amdgcn_sched_barrier(0);
      for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
        for (uint32_t j = 0; j < 4; j++) {
          int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                      (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
          if (nindx < actlN) {
            int adr = mindx + M * nindx;
            if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
              vals[nt].s8[j] += __bfloat162float(biases[nt][j]);
              C[adr] = __float2bfloat16(vals[nt].s8[j]);
            } else {
              vals[nt].s8[j] += __half2float(biases[nt][j]);
              C[adr] = __float2half(vals[nt].s8[j]);
            }
          }
        }
      }
    }

  #ifndef WVSPLITKRC_1KPASS
    m0 += CuCount * WvPrGrp * YTILE / GrpsShrB;
    m = (m0 + m1) % Mmod;
    k_str = (m0 / Mmod) * kFit * kfitsPerRdc;
    k_end = (m0 / Mmod + 1) * kFit * kfitsPerRdc;
    if (k_str >= K) break;
    kBase = 0;
  #endif
  }
}
#else
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GrpsShrB, int CHUNKK, int DTRMNSTC>
__global__ void wvSplitKrc_(const int actlN, const int K, const int Kap,
                            const int M, const int Bx, const int By,
                            const scalar_t* B, const scalar_t* __restrict__ A,
                            const scalar_t* __restrict__ BIAS, float* glbl,
                            int* cntr, scalar_t* C,
                            const int CuCount){UNREACHABLE_CODE}
#endif  // defined(__HIP__GFX9__) TODO: Add NAVI support

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount) {
  int _DTRMNSTC = 1;  // vllm::vllm_is_batch_invariant();

  auto M_in = in_b.size(0);
  auto N_in = in_a.size(0);
  auto K_in = in_b.size(1);
  auto Kap_in = in_a.stride(0);

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

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_a.dtype()).device(in_a.device()));

  auto N_p2 = 1U << (32 - __builtin_clz(N_in - 1));

  dim3 grid(CuCount);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // const int max_lds_len = get_lds_size() / 2;

  // With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
  // and each working on a 512-shard of K, how many CUs would we need?
  int rndup_cus = ((M_in + 64 - 1) / 64) * ((K_in + 512 - 1) / 512);

  // How many of 4 waves in a group can work on same 16 Ms at same time? First
  // try to maximize this. This reduces the Ms each group works on, i.e.
  // increasing the number of CUs needed.
  int GrpsShrB = min(N_p2 / 16, 4);

  // Given the above, how many CUs would we need?
  int CuNeeded = rndup_cus * GrpsShrB;

  if (CuNeeded > CuCount) throw std::runtime_error("Invalid wvSplitKrc size");

  // Can we increase SplitK by shrinking the K-shared to 256?
  int chunkk = (CuNeeded * 2 <= CuCount) ? 2 : 1;

  static torch::Tensor axl_glbl =
      torch::zeros(
          128 * 1024 * (_DTRMNSTC ? 12 : 1),
          torch::TensorOptions().dtype(torch::kFloat32).device(in_a.device()))
          .detach();
  static torch::Tensor axl_cntr =
      torch::zeros(
          128 * 1024 * (_DTRMNSTC ? 12 : 1) / 4,
          torch::TensorOptions().dtype(torch::kInt).device(in_a.device()))
          .detach();
  auto glbl = axl_glbl.data_ptr<float>();
  auto cntr = axl_cntr.data_ptr<int>();

#define WVSPLITKrc(_N, _GrpsShrB, _CHUNKK)                                     \
  {                                                                            \
    dim3 block(64, 4);                                                         \
    if (_DTRMNSTC)                                                             \
      wvSplitKrc_<fptype, 64, 16, 4, 8, 1, _N, _GrpsShrB, _CHUNKK, 1>          \
          <<<grid, block, 0, stream>>>(N_in, K_in, Kap_in, M_in, Bx_in, By_in, \
                                       af4, bf4, biasf4, glbl, cntr, c,        \
                                       CuCount);                               \
    else                                                                       \
      wvSplitKrc_<fptype, 64, 16, 4, 8, 1, _N, _GrpsShrB, _CHUNKK, 0>          \
          <<<grid, block, 0, stream>>>(N_in, K_in, Kap_in, M_in, Bx_in, By_in, \
                                       af4, bf4, biasf4, glbl, cntr, c,        \
                                       CuCount);                               \
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_a.scalar_type(), "wvSplitKrc", [&] {
    using fptype = typename scalar<scalar_t>::type;
    const fptype* af4 = reinterpret_cast<const fptype*>(in_a.data_ptr());
    const fptype* bf4 = reinterpret_cast<const fptype*>(in_b.data_ptr());
    const fptype* biasf4 =
        (in_bias.has_value() && in_bias->numel() > 0)
            ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
            : nullptr;
    fptype* c = reinterpret_cast<fptype*>(out_c.data_ptr());

    switch (N_p2) {
      case 16:
        WVSPLITKrc(16, 1, 1) break;
      case 32:
        if (chunkk == 2) WVSPLITKrc(32, 2, 2) else WVSPLITKrc(32, 2, 1) break;
      case 64:
        if (chunkk == 2) WVSPLITKrc(64, 4, 2) else WVSPLITKrc(64, 4, 1) break;
      case 128:
        if (chunkk == 2) WVSPLITKrc(128, 4, 2) else WVSPLITKrc(128, 4, 1) break;
      default:
        throw std::runtime_error(
            "Unsupported N value: " + std::to_string(M_in) + "," +
            std::to_string(K_in) + "," + std::to_string(N_in));
    }
  });
  return out_c;
}

#if defined(__HIP__MI3XX__) || defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitKQ_hf_sml_(const int K, const int Kap, const int Kbp, const int M,
                      const int Bx, const int By, const fp8_t* B,
                      const fp8_t* __restrict__ A,
                      const scalar_t* __restrict__ BIAS, scalar_t* C,
                      const float* __restrict__ s_A,
                      const float* __restrict__ s_B, const int _WvPrGrp,
                      const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE;
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 4) * sizeof(float)))) float;
  using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
  using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  union bigType {
    char f8[A_CHUNK];
    char2 c2[A_CHUNK / 2];
    scalar_t h[A_CHUNK / 2];
    float f[A_CHUNK / 4];
    int i[A_CHUNK / 4];
    long l[A_CHUNK / 8];
    intx4 l2[A_CHUNK / 16];
    scalar8 h8;
  };

  __shared__ fp8_t s[max_lds_len];

  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  asm volatile("s_waitcnt vmcnt(0)");
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  float sA = *s_A;
  float sB = *s_B;

  while (m < M) {
  #ifdef __GFX12__
    // gfx12: per-lane scalar accumulation via v_dot4_f32_fp8_fp8
    float sum[N][YTILE] = {};
  #else
    // gfx9: MFMA accumulation
    scalar8 sum[N][YTILE] = {};
  #endif
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];

      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const fp8_t* B_ = &B[min__(k_, K - A_CHUNK)];
  #pragma unroll
        for (uint32_t y = 0; y < YTILE; ++y) {
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
        }
      }

  // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
        }
      }

  // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
  #ifdef __GFX12__
          // gfx12: 4 x dot4 per A_CHUNK=16 bytes (4 FP8 per dot4)
          for (int y = 0; y < YTILE; ++y) {
    #pragma unroll
            for (int i = 0; i < A_CHUNK / 4; i++) {
              sum[n][y] = __builtin_amdgcn_dot4_f32_fp8_fp8(
                  bigA[n][k2].i[i], bigB[y][k2].i[i], sum[n][y]);
            }
          }
  #else
          // gfx9: MFMA path
          for (int i = 0; i < A_CHUNK; i += 8) {
            for (int y = 0; y < YTILE; ++y) {
              sum[n][y] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                  bigA[n][k2].l[i / 8], bigB[y][k2].l[i / 8], sum[n][y], 0, 0,
                  0);
            }
          }
  #endif
        }
      }
    }

    // Final reduction
  #ifdef __GFX12__
    // gfx12 wave32: DPP row_shr within 16-lane rows + cross-row shuffle
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:1 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    }
  #else
    // gfx9 MFMA reduction
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        float accm0 = sum[n][y][0];
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][1], 0x101, 0xf, 0xf,
                                          1);  // row_shl1
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][2], 0x102, 0xf, 0xf,
                                          1);  // row_shl2
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][3], 0x103, 0xf, 0xf,
                                          1);  // row_shl3
        accm0 += __shfl_down(accm0, 20);
        accm0 += __shfl_down(accm0, 40);
        sum[n][y][0] = accm0;
      }
    }
  #endif

    const bool writeback_lane =
  #ifdef __GFX12__
        threadIdx.x == (THRDS - 1);
  #else
        threadIdx.x == 0;
  #endif
    if (writeback_lane) {
      scalar_t biases[N][YTILE] = {};
      if (BIAS)
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
          }
        }
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          if (y + m >= M) break;  // To avoid mem access fault.
  #ifdef __GFX12__
          float result = sum[n][y] * sA * sB;
  #else
          float result = sum[n][y][0] * sA * sB;
  #endif
          if constexpr (std::is_same_v<scalar_t, half>) {
            result += __half2float(biases[n][y]);
          } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
            result += __bfloat162float(biases[n][y]);
          }
          C[m + y + n * M] = __float2s<scalar_t>(result);
        }
      }
    }

    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__MI3XX__) && !defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void wvSplitKQ_hf_sml_(const int K, const int Kap, const int Kbp,
                                  const int M, const int Bx, const int By,
                                  const fp8_t* B, const fp8_t* __restrict__ A,
                                  const scalar_t* __restrict__ BIAS,
                                  scalar_t* C, const float* __restrict__ s_A,
                                  const float* __restrict__ s_B,
                                  const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__MI3XX__) || defined(__GFX12__)

#if defined(__HIP__MI3XX__) || defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitKQ_hf_(const int K, const int Kap, const int Kbp, const int M,
                  const int Bx, const int By, const fp8_t* B,
                  const fp8_t* __restrict__ A,
                  const scalar_t* __restrict__ BIAS, scalar_t* C,
                  const float* __restrict__ s_A, const float* __restrict__ s_B,
                  const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE;
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 4) * sizeof(float)))) float;
  using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
  using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  union bigType {
    char f8[A_CHUNK];
    char2 c2[A_CHUNK / 2];
    scalar_t h[A_CHUNK / 2];
    float f[A_CHUNK / 4];
    int i[A_CHUNK / 4];
    long l[A_CHUNK / 8];
    intx4 l2[A_CHUNK / 16];
    scalar8 h8;
  };

  __shared__ fp8_t s[max_lds_len];

  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  asm volatile("s_waitcnt vmcnt(0)");
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  float sA = *s_A;
  float sB = *s_B;

  while (m < M) {
  #ifdef __GFX12__
    // gfx12: per-lane scalar accumulation via v_dot4_f32_fp8_fp8
    float sum[N][YTILE] = {};
  #else
    // gfx9: MFMA accumulation
    scalar8 sum[N][YTILE] = {};
  #endif
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];

      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const fp8_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; ++y) {
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
        }
      }

  // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          if (k_ + Kap * n < max_lds_len)
            bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
          else
            bigA[n][k2] = *((const bigType*)(&(A[k_ + Kap * n])));
        }
      }

  // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
  #ifdef __GFX12__
          // gfx12: 4 x dot4 per A_CHUNK=16 bytes (4 FP8 per dot4)
          for (int y = 0; y < YTILE; ++y) {
    #pragma unroll
            for (int i = 0; i < A_CHUNK / 4; i++) {
              sum[n][y] = __builtin_amdgcn_dot4_f32_fp8_fp8(
                  bigA[n][k2].i[i], bigB[y][k2].i[i], sum[n][y]);
            }
          }
  #else
          // gfx9: MFMA path
          for (int i = 0; i < A_CHUNK; i += 8) {
            for (int y = 0; y < YTILE; ++y) {
              sum[n][y] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                  bigA[n][k2].l[i / 8], bigB[y][k2].l[i / 8], sum[n][y], 0, 0,
                  0);
            }
          }
  #endif
        }
      }
    }

    // Final reduction
  #ifdef __GFX12__
    // gfx12 wave32: DPP row_shr within 16-lane rows + cross-row shuffle
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:1 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    }
  #else
    // gfx9 MFMA reduction
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        float accm0 = sum[n][y][0];
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][1], 0x101, 0xf, 0xf,
                                          1);  // row_shl1
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][2], 0x102, 0xf, 0xf,
                                          1);  // row_shl2
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][3], 0x103, 0xf, 0xf,
                                          1);  // row_shl3
        accm0 += __shfl_down(accm0, 20);
        accm0 += __shfl_down(accm0, 40);
        sum[n][y][0] = accm0;
      }
    }
  #endif

    const bool writeback_lane =
  #ifdef __GFX12__
        threadIdx.x == (THRDS - 1);
  #else
        threadIdx.x == 0;
  #endif
    if (writeback_lane) {
      scalar_t biases[N][YTILE] = {};
      if (BIAS)
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
          }
        }
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          if (y + m >= M) break;  // To avoid mem access fault.
  #ifdef __GFX12__
          float result = sum[n][y] * sA * sB;
  #else
          float result = sum[n][y][0] * sA * sB;
  #endif
          if constexpr (std::is_same_v<scalar_t, half>) {
            result += __half2float(biases[n][y]);
          } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
            result += __bfloat162float(biases[n][y]);
          }
          C[m + y + n * M] = __float2s<scalar_t>(result);
        }
      }
    }

    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__MI3XX__) && !defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void wvSplitKQ_hf_(const int K, const int Kap, const int Kbp,
                              const int M, const int Bx, const int By,
                              const fp8_t* B, const fp8_t* __restrict__ A,
                              const scalar_t* __restrict__ BIAS, scalar_t* C,
                              const float* __restrict__ s_A,
                              const float* __restrict__ s_B, const int _WvPrGrp,
                              const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__MI3XX__) || defined(__GFX12__)

void wvSplitKQ(const at::Tensor& in_b, const at::Tensor& in_a,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;
  auto M_in = in_b.size(0);
  auto K_in = in_b.size(1);
  auto N_in = in_a.size(0);
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

  TORCH_CHECK(K_in % 16 == 0, "k % 16 == 0");
  TORCH_CHECK(in_a.dtype() == in_b.dtype() && in_a.dtype() == kFp8Type);
  TORCH_CHECK(out_c.dtype() == torch::kFloat16 ||
              out_c.dtype() == torch::kBFloat16);

  dim3 grid(CuCount);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int max_lds_len = get_lds_size();

#define WVSPLITKQ_IMPL(_THRDS, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N) \
  {                                                                            \
    dim3 block(_THRDS, _WvPrGrp);                                              \
    if ((Kap_in * N_in <= max_lds_len) && (M_in % _YTILEs == 0)) {             \
      int __wvPrGrp = min(_WvPrGrp, mindiv(M_in, CuCount * _YTILEs, 16));      \
      wvSplitKQ_hf_sml_<fptype, fp8_t, _THRDS, _YTILEs, _WvPrGrp, 16, _UNRLs,  \
                        _N><<<grid, block, 0, stream>>>(                       \
          K_in, Kap_in, Kbp_in, M_in, Bx_in, By_in, b_ptr, a_ptr, bias_ptr,    \
          c_ptr, s_a, s_b, __wvPrGrp, CuCount);                                \
    } else {                                                                   \
      int __wvPrGrp = min(_WvPrGrp, mindiv(M_in, CuCount * _YTILEm, 16));      \
      wvSplitKQ_hf_<fptype, fp8_t, _THRDS, _YTILEm, _WvPrGrp, 16, _UNRLm, _N>  \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,      \
                                       By_in, b_ptr, a_ptr, bias_ptr, c_ptr,   \
                                       s_a, s_b, __wvPrGrp, CuCount);          \
    }                                                                          \
  }

#define WVSPLITKQ(_WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N)      \
  if (on_gfx12())                                                      \
    WVSPLITKQ_IMPL(32, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N) \
  else                                                                 \
    WVSPLITKQ_IMPL(64, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N)

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_c.scalar_type(), "wvSplitKQ", [&] {
    using fptype = typename scalar<scalar_t>::type;
    auto c_ptr = reinterpret_cast<fptype*>(out_c.data_ptr());
    auto s_a = scale_a.data_ptr<float>();
    auto s_b = scale_b.data_ptr<float>();
    VLLM_DISPATCH_FP8_TYPES(in_a.scalar_type(), "wvSplitKQ", [&] {
      auto a_ptr = in_a.data_ptr<fp8_t>();
      auto b_ptr = in_b.data_ptr<fp8_t>();
      auto bias_ptr = (in_bias.has_value() && in_bias->numel() > 0)
                          ? reinterpret_cast<fptype*>(in_bias->data_ptr())
                          : nullptr;
      switch (N_in) {
        case 1:
          WVSPLITKQ(16, 2, 2, 2, 2, 1)
          break;
        case 2:
          WVSPLITKQ(16, 2, 2, 2, 2, 2)
          break;
        case 3:
          WVSPLITKQ(16, 2, 2, 1, 1, 3)
          break;
        case 4:
          WVSPLITKQ(16, 2, 2, 1, 1, 4)
          break;
        default:
          throw std::runtime_error(
              "Unsupported N value: " + std::to_string(M_in) + "," +
              std::to_string(K_in) + "," + std::to_string(N_in));
      }
    });
  });
}