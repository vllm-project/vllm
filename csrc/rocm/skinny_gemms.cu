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

template <typename scalar_t, typename scalar2_t, int ROWS_PER_BLOCK>
__device__ __forceinline__ void load_tile(
    const float4* mat_f4, const scalar2_t* vec_h2, int thread_id,
    int block_base_addr, int K, float4 mat_chunk[ROWS_PER_BLOCK],
    scalar2_t& vec_h2x, scalar2_t& vec_h2y, scalar2_t& vec_h2z,
    scalar2_t& vec_h2w) {
  constexpr int ELEMS_PER_FLOAT4 = sizeof(float4) / sizeof(scalar_t);
  constexpr int PAIRS_PER_FLOAT4 = sizeof(float4) / sizeof(scalar2_t);
  if (thread_id * ELEMS_PER_FLOAT4 < K) {
#pragma unroll
    for (int i = 0; i < ROWS_PER_BLOCK; i++) {
      // block_base_addr: first float4 of row 0 for this block
      // thread_id: this thread's column chunk within a row
      // K / ELEMS_PER_FLOAT4 * i: row stride to row i
      mat_chunk[i] = load_ntmprl(
          &mat_f4[block_base_addr + thread_id + K / ELEMS_PER_FLOAT4 * i]);
    }
    vec_h2x = vec_h2[thread_id * PAIRS_PER_FLOAT4 + 0];
    vec_h2y = vec_h2[thread_id * PAIRS_PER_FLOAT4 + 1];
    vec_h2z = vec_h2[thread_id * PAIRS_PER_FLOAT4 + 2];
    vec_h2w = vec_h2[thread_id * PAIRS_PER_FLOAT4 + 3];
  }
}

template <typename scalar2_t>
__device__ __forceinline__ float dot_row(scalar2_t* mat_row_ptr,
                                         scalar2_t vec_h2x, scalar2_t vec_h2y,
                                         scalar2_t vec_h2z, scalar2_t vec_h2w) {
  scalar2_t acc_h2 = __hmul2(*mat_row_ptr, vec_h2x);
  acc_h2 = __hfma2(*(mat_row_ptr + 1), vec_h2y, acc_h2);
  acc_h2 = __hfma2(*(mat_row_ptr + 2), vec_h2z, acc_h2);
  acc_h2 = __hfma2(*(mat_row_ptr + 3), vec_h2w, acc_h2);
  float2 sum = __s22float2(acc_h2);
  return sum.x + sum.y;
}

template <int ROWS_PER_BLOCK>
__device__ __forceinline__ void warp_reduce(float acc[ROWS_PER_BLOCK]) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
#pragma unroll
    for (int i = 0; i < ROWS_PER_BLOCK; i++) {
      acc[i] += __shfl_xor(acc[i], mask);
    }
  }
}

// Computes mat (M x K) * vec (1 x K) -> out (1 x M): one dot product per row of
// mat. Grid: (M / ROWS_PER_BLOCK) blocks; each block owns ROWS_PER_BLOCK
// consecutive rows.
//
// Four stages:
//   Stage 1 — partial dot products: each thread loads one float4 per row
//   (ROWS_PER_BLOCK rows
//             total) from mat and one float4 from vec, accumulating partial dot
//             products into acc[].
//   Stage 2 — intra-warp butterfly: warp_reduce gives all lanes the same acc[];
//   lanes
//             0..ROWS_PER_BLOCK-1 each write their row's partial sum to smem in
//             parallel.
//   Stage 3 — cross-warp reduction: ROWS_PER_BLOCK row groups each stride smem
//   columns
//             (one per warp) then butterfly-reduce so all threads in the group
//             hold the same scalar for that row.
//   Stage 4 — output write: adjacent row groups exchange scalars via shfl_xor;
//   the group
//             leader packs the pair into a half2 and writes to out.
template <typename scalar_t, int ROWS_PER_BLOCK>
__global__ void vecMatMul_kernel(const scalar_t* mat, const scalar_t* vec,
                                 scalar_t* out, const int K) {
  using scalar2_t = typename scalar2<scalar_t>::type;
  // Packing constants: data is stored as packed halves loaded via float4.
  constexpr int ELEMS_PER_FLOAT4 =
      sizeof(float4) / sizeof(scalar_t);  // 8 halves per float4
  constexpr int PAIRS_PER_FLOAT4 =
      sizeof(float4) / sizeof(scalar2_t);  // 4 half2 pairs per float4
  // Threads per row group: one group per output row in the second-stage
  // reduction.
  constexpr int THREADS_PER_ROW_GROUP = WARP_SIZE / ROWS_PER_BLOCK;
  auto mat_f4 = reinterpret_cast<const float4*>(mat);
  auto vec_h2 = reinterpret_cast<const scalar2_t*>(vec);
  auto out_h2 = reinterpret_cast<scalar2_t*>(out);
  __shared__ float reduction_smem[ROWS_PER_BLOCK][WARP_SIZE];
  // Base offset into mat_f4 for the first float4 of the first row this block
  // owns.
  const int block_base_addr =
      blockIdx.x * ROWS_PER_BLOCK * K / ELEMS_PER_FLOAT4;
  const int thread_id = threadIdx.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane = thread_id % WARP_SIZE;
  // Which output row group this thread belongs to (shared by all threads in the
  // group).
  const int row_group_id = thread_id / THREADS_PER_ROW_GROUP;
  // This thread's position within its row group; determines which smem columns
  // it reads.
  const int row_group_thread_id = thread_id % THREADS_PER_ROW_GROUP;
  scalar2_t vec_h2x, vec_h2y, vec_h2z, vec_h2w;
  float acc[ROWS_PER_BLOCK];
  float4 mat_chunk[ROWS_PER_BLOCK];
  scalar2_t out_val;

  // Stage 1: each thread loads one float4 per row (ROWS_PER_BLOCK rows total)
  // from mat and one float4 from vec, then computes a partial dot product per
  // row into acc[].
  //
  // Threads beyond K/8 diverge in load_tile's if guard (skipping the load) and
  // are zeroed by the ternary below; their dot_row result is discarded.
  load_tile<scalar_t, scalar2_t, ROWS_PER_BLOCK>(
      mat_f4, vec_h2, thread_id, block_base_addr, K, mat_chunk, vec_h2x,
      vec_h2y, vec_h2z, vec_h2w);

  auto mat_h2_ptr = reinterpret_cast<scalar2_t*>(&mat_chunk);

#pragma unroll
  for (int i = 0; i < ROWS_PER_BLOCK; i++) {
    float val = dot_row(mat_h2_ptr + i * PAIRS_PER_FLOAT4, vec_h2x, vec_h2y,
                        vec_h2z, vec_h2w);
    acc[i] = (thread_id * ELEMS_PER_FLOAT4 < K ? val : 0.f);
  }

  // Stage 2: intra-warp butterfly — reduces each acc[i] across all lanes in the
  // warp.
  warp_reduce<ROWS_PER_BLOCK>(acc);

  // All lanes hold the same acc[] after the butterfly; lanes
  // 0..ROWS_PER_BLOCK-1 each write their row's partial sum to smem in parallel
  // (lane i writes row i).
  if (lane < ROWS_PER_BLOCK) {
    reduction_smem[lane][warp_id] = acc[lane];
  }
  __syncthreads();

  // Stage 3: cross-warp reduction.
  // Threads are divided into ROWS_PER_BLOCK row groups of THREADS_PER_ROW_GROUP
  // threads. Each group strides smem columns (one per warp) accumulating
  // partial sums for its row, then butterfly-reduces within the group so all
  // threads hold the same scalar. Striding handles the case where num_warps >
  // THREADS_PER_ROW_GROUP (e.g. large K on WARP_SIZE=32 GPUs).
  const int num_warps = blockDim.x / WARP_SIZE;
  if (row_group_id < ROWS_PER_BLOCK) {
    float partial = 0.f;
    for (int w = row_group_thread_id; w < num_warps;
         w += THREADS_PER_ROW_GROUP) {
      partial += reduction_smem[row_group_id][w];
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW_GROUP / 2; mask >= 1; mask /= 2) {
      partial += __shfl_xor(partial, mask);
    }

    // Stage 4: adjacent row groups (row i, row i+1) exchange their scalars so
    // the group leader can pack both into a single half2 write.
    float out_val2 = __shfl_xor(partial, THREADS_PER_ROW_GROUP);
    if (lane % (2 * THREADS_PER_ROW_GROUP) == 0) {
      out_val = __float22s2_rn<scalar2_t>(make_float2(partial, out_val2));
      out_h2[blockIdx.x * ROWS_PER_BLOCK / 2 + row_group_id / 2] = out_val;
    }
  }
}

// Computes mat (M x K) * vec (1 x K) -> out (1 x M): one dot product per row of
// mat against the single vec vector. Optimized for the N=1 (single token) case.
torch::Tensor vecMatMul(at::Tensor& mat, at::Tensor& vec,
                        const int64_t rows_per_block) {
  TORCH_CHECK(vec.size(0) == 1, "Row number of activation tensor must be 1.");
  TORCH_CHECK(mat.size(1) == vec.size(1),
              "K dimension mismatch between mat and vec.");
  TORCH_CHECK(mat.dtype() == vec.dtype());
  TORCH_CHECK(vec.dtype() == torch::kFloat16 ||
              vec.dtype() == torch::kBFloat16);
  TORCH_CHECK(rows_per_block <= WARP_SIZE,
              "rows_per_block must not exceed WARP_SIZE (", WARP_SIZE, ").");

  auto M = mat.size(0);
  auto K = mat.size(1);

  auto out = torch::empty(
      {1, M}, torch::TensorOptions().dtype(vec.dtype()).device(vec.device()));

  // NUM_THREADS needs to be a multiple of WARP_SIZE, as we are using warp
  // shuffle operations.
  const int NUM_THREADS =
      max(rows_per_block * (int)sizeof(float4),
          K * 2 / (int)sizeof(float4) % WARP_SIZE == 0
              ? K * 2 / (int)sizeof(float4)
              : K * 2 / (int)sizeof(float4) +
                    (WARP_SIZE - K * 2 / (int)sizeof(float4) % WARP_SIZE));

  int NUM_BLOCKS = M / rows_per_block;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(vec.scalar_type(), "vecMatMul", [&] {
    auto mat_ptr = mat.data_ptr<scalar_t>();
    auto vec_ptr = vec.data_ptr<scalar_t>();
    auto out_ptr = out.data_ptr<scalar_t>();

    switch (rows_per_block) {
      case 2:
        vecMatMul_kernel<scalar_t, 2><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            mat_ptr, vec_ptr, out_ptr, K);
        break;
      case 4:
        vecMatMul_kernel<scalar_t, 4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            mat_ptr, vec_ptr, out_ptr, K);
        break;
      case 8:
        vecMatMul_kernel<scalar_t, 8><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            mat_ptr, vec_ptr, out_ptr, K);
        break;
      case 16:
        vecMatMul_kernel<scalar_t, 16><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            mat_ptr, vec_ptr, out_ptr, K);
        break;
      default:
        NUM_BLOCKS = M / 4;
        vecMatMul_kernel<scalar_t, 4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            mat_ptr, vec_ptr, out_ptr, K);
        break;
    }
  });

  return out;
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
  //	     then this is not going to work!
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
  //	     then this is not going to work!
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
