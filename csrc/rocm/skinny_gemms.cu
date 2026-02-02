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

#if defined(__HIPCC__) && (defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__MI3XX__
#endif

#if defined(__gfx950__)
  #define LDS_SIZE 160 * 1024
#else
  #define LDS_SIZE 64 * 1024
#endif

int get_lds_size() {
  static bool is_cached = false;
  static int result;
  if (is_cached == false) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    size_t substring = device_arch.find("gfx95");
    result = (substring == std::string::npos ? 64 * 1024 : 160 * 1024);
    is_cached = true;
  }
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

#define DOT2C(V0, V2, V3)                                                     \
  if constexpr (std::is_same_v<scalar_t, half>) {                             \
    asm("v_dot2c_f32_f16 %0, %2, %3" : "=v"(V0) : "0"(V0), "v"(V2), "v"(V3)); \
  } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {            \
    float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *             \
               __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));              \
    V0 += (s.x + s.y);                                                        \
  }

// To avoid LLVM silently upcasting to double
__device__ inline unsigned int min__(uint32_t a, uint32_t b) {
  return min(a, b);
}

#if defined(__HIP__GFX9__)  // TODO: Add NAVI support
// This version targets cases where A[] fits LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_sml_(const int K, const int M, const int Bx, const int By,
                     const scalar_t* B, const scalar_t* __restrict__ A,
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
  for (uint32_t k = 0; k < min__(K * N, max_lds_len);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    if (k_in >= min__(K * N, max_lds_len)) break;

    *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  float sum[N][YTILE];
  scalar8 sum4[N][YTILE];

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
    for (int i = 0; i < YTILE; i++)
      for (int n = 0; n < N; n++)
        if constexpr (!use_mfma)
          sum[n][i] = 0;
        else
          sum4[n][i] = {0, 0, 0, 0};

    bigType bigA[N][UNRL];
    bigType bigB[YTILE][UNRL];
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
    // for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        const scalar_t* B_ = &B[(m + 0) * K + k_];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[y * K])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigType*)(&(s[k_ + K * n])));
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        // Do the matrix multiplication of activation and weight matrix
        // - Remember the accumulation is happening for K-split of 64!
  #pragma unroll
        for (uint32_t n = 0; n < N; n++) {
  #pragma unroll
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
  #pragma unroll
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
  #pragma unroll
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
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        }
      }

      if (threadIdx.x == 63) {
        for (int n = 0; n < N; n++) {
          for (int i = 0; i < YTILE; i++) {
            if constexpr (std::is_same_v<scalar_t, half>) {
              if (BIAS)
                sum[n][i] += __half2float(BIAS[(m + i) % Bx + (n % By) * M]);
            } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
              if (BIAS)
                sum[n][i] +=
                    __bfloat162float(BIAS[(m + i) % Bx + (n % By) * M]);
            }
            C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
          }
        }
      }
    } else {
  #pragma unroll
      for (int n = 0; n < N; n++) {
  #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          // float accm1 = 0;
          // for (int i=0; i<64; i++)
          //    accm1 += __shfl(sum4[n][y][i%4], i);
          float accm = sum4[n][y][0];
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:1 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(sum4[n][y][1]), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:2 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(sum4[n][y][2]), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:3 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(sum4[n][y][3]), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:4 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:8 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));
          asm("s_nop 0\n\tv_mov_b32 %0, %2 row_shr:15 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));

          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == 63) {
        for (int n = 0; n < N; n++) {
          for (int i = 0; i < YTILE; i++) {
            if (BIAS)
              sum4[n][i][0] +=
                  __bfloat162float(BIAS[(m + i) % Bx + (n % By) * M]);
            C[m + i + n * M] = __float2bfloat16(sum4[n][i][0]);
          }
        }
      }
    }
    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__GFX9__) TODO: Add NAVI support
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_hf_sml_(const int K, const int M, const int Bx,
                                 const int By, const scalar_t* B,
                                 const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ BIAS, scalar_t* C,
                                 const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__GFX9__) TODO: Add NAVI support

#if defined(__HIP__GFX9__)  // TODO: Add NAVI support
// This version targets cases where A[] marginally exceeds LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_(const int K, const int M, const int Bx, const int By,
                 const scalar_t* B, const scalar_t* __restrict__ A,
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
  // Reserving 64 KB of LDS to have 1 WG / CU
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

  //----------------------------------------------------
  // Indexing function into the column of weight matrix B
  // Algorithm does 64 lane k-splitting / wave and uses
  // WG ID and Thread ID to find the index.
  //----------------------------------------------------
  // int _WvPrGrp = mindiv(N, CuCount * YTILE, WvPrGrp);
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
  for (uint32_t k = 0; k < min__(K * N, max_lds_len);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    if (k_in >= min__(K * N, max_lds_len)) break;

    *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
  }

  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  float sum[N][YTILE];
  scalar8 sum4[N][YTILE];

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
    for (int i = 0; i < YTILE; i++)
      for (int n = 0; n < N; n++)
        if constexpr (!use_mfma)
          sum[n][i] = 0;
        else
          sum4[n][i] = {0, 0, 0, 0};

    bigType bigA[N][UNRL];
    bigType bigB[YTILE][UNRL];
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
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        const scalar_t* B_ = &B[(m + 0) * K + k_];
        for (int b = 0; b < YTILE; b++)
          bigB[b][k2].h8 = (loadnt((scalar8*)(&B_[b * K])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int n = 0; n < N; n++) {
          if (k_ + K * n < max_lds_len)
            bigA[n][k2] = *((const bigType*)(&(s[k_ + K * n])));
          else
            bigA[n][k2] = *((const bigType*)(&(A[k_ + K * n])));
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t n = 0; n < N; n++) {
  #pragma unroll
        for (uint32_t k2 = 0; k2 < UNRL; k2++) {
          uint32_t k = k1 + k2 * THRDS * A_CHUNK;
          uint32_t k_ = k + threadIdx.x * A_CHUNK;
          if (k_ >= K) break;
          // Do the matrix multiplication of activation and weight matrix
          // - Remember the accumulation is happening for K-split of 64!
  #pragma unroll
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
  #pragma unroll
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
  #pragma unroll
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
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        }
      }

      if (threadIdx.x == 63) {
        for (int n = 0; n < N; n++) {
          for (int i = 0; i < YTILE; i++) {
            if (commitColumn[i]) {
              if constexpr (std::is_same_v<scalar_t, half>) {
                if (BIAS)
                  sum[n][i] += __half2float(BIAS[(m + i) % Bx + (n % By) * M]);
              } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
                if (BIAS)
                  sum[n][i] +=
                      __bfloat162float(BIAS[(m + i) % Bx + (n % By) * M]);
              }
              C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
            }
          }
        }
      }
    } else {
  #pragma unroll
      for (int n = 0; n < N; n++) {
  #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          // float accm1 = 0;
          // for (int i=0; i<64; i++)
          //    accm1 += __shfl(sum4[n][y][i%4], i);

          float accm = sum4[n][y][0];
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:1 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(sum4[n][y][1]), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:2 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(sum4[n][y][2]), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:3 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(sum4[n][y][3]), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:4 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:8 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));
          asm("s_nop 0\n\tv_mov_b32 %0, %2 row_shr:15 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));

          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == 63) {
        for (int n = 0; n < N; n++) {
          for (int i = 0; i < YTILE; i++) {
            if (commitColumn[i]) {
              if (BIAS)
                sum4[n][i][0] +=
                    __bfloat162float(BIAS[(m + i) % Bx + (n % By) * M]);
              C[m + i + n * M] = __float2bfloat16(sum4[n][i][0]);
            }
          }
        }
      }
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

#else   // !defined(__HIP__GFX9__) TODO: Add NAVI support
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_hf_(const int K, const int M, const int Bx,
                             const int By, const scalar_t* B,
                             const scalar_t* __restrict__ A,
                             const scalar_t* __restrict__ BIAS, scalar_t* C,
                             const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__GFX9__) TODO: Add NAVI support

#if defined(__HIP__GFX9__)  // TODO: Add NAVI support
// This version targets big A[] cases, where it is much larger than LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_big_(const int K, const int M, const int Bx, const int By,
                     const scalar_t* B, const scalar_t* __restrict__ A,
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
  for (uint32_t k = 0; k < min__(K * N, max_lds_len);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    if (k_in >= min__(K * N, max_lds_len)) break;

    *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
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
  kFit = min__(kFit, K);

  float sum[N][YTILE];
  scalar8 sum4[N][YTILE];

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
    for (int i = 0; i < YTILE; i++)
      for (int n = 0; n < N; n++)
        if constexpr (!use_mfma)
          sum[n][i] = 0;
        else
          sum4[n][i] = {0, 0, 0, 0};

    bigType bigA[N][UNRL];
    bigType bigB[YTILE][UNRL];
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
  #ifdef PCML
      if ((k1 == 0) || (k1 == kBase + kFit)) {  // load next chunk of A[] to LDS
        if (k1 != 0) kBase += kFit;
        __syncthreads();
        for (uint32_t k = 0; k < kFit; k += THRDS * _WvPrGrp * A_CHUNK) {
          uint32_t kOff = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);
          if (kBase + kOff >= K) break;
          if (kOff >= kFit) break;
          for (uint32_t n = 0; n < N; n++) {
            uint32_t k_in = kBase + n * K + kOff;
            uint32_t k_ot = n * kFit + kOff;
            *((bigType*)(&s[k_ot])) = *((bigType*)(&A[k_in]));
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
        if (k_ >= K) break;

        const scalar_t* B_ = &B[(m + 0) * K + k_];
        for (int b = 0; b < YTILE; b++)
          bigB[b][k2].h8 = (loadnt((scalar8*)(&B_[b * K])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int n = 0; n < N; n++) {
  #ifdef PCML
          bigA[n][k2] = *((const bigType*)(&(s[k_ - kBase + kFit * n])));
  #else
          if (k_ + K * n < 32 * 1024)
            bigA[n][k2] = *((const bigType*)(&(s[k_ + K * n])));
          else
            bigA[n][k2] = *((const bigType*)(&(A[k_ + K * n])));
  #endif
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
  #pragma unroll
        for (uint32_t n = 0; n < N; n++) {
          // Do the matrix multiplication of activation and weight matrix
          // - Remember the accumulation is happening for K-split of 64!
  #pragma unroll
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
  #pragma unroll
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
  #pragma unroll
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
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
              : "=v"(sum[n][y])
              : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        }
      }

      if (threadIdx.x == 63) {
        for (int n = 0; n < N; n++) {
          for (int i = 0; i < YTILE; i++) {
            if (commitColumn[i]) {
              if constexpr (std::is_same_v<scalar_t, half>) {
                if (BIAS)
                  sum[n][i] += __half2float(BIAS[(m + i) % Bx + (n % By) * M]);
              } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
                if (BIAS)
                  sum[n][i] +=
                      __bfloat162float(BIAS[(m + i) % Bx + (n % By) * M]);
              }
              C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
            }
          }
        }
      }
    } else {
  #pragma unroll
      for (int n = 0; n < N; n++) {
  #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          float accm = sum4[n][y][0];
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:1 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(sum4[n][y][1]), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:2 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(sum4[n][y][2]), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:3 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(sum4[n][y][3]), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:4 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shl:8 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));
          asm("s_nop 0\n\tv_mov_b32 %0, %2 row_shr:15 bound_ctrl:0 "
              : "=v"(accm)
              : "0"(accm), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
              : "=v"(accm)
              : "0"(accm), "v"(accm), "v"(accm));

          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == 63) {
        for (int n = 0; n < N; n++) {
          for (int i = 0; i < YTILE; i++) {
            if (commitColumn[i]) {
              if (BIAS)
                sum4[n][i][0] +=
                    __bfloat162float(BIAS[(m + i) % Bx + (n % By) * M]);
              C[m + i + n * M] = __float2bfloat16(sum4[n][i][0]);
            }
          }
        }
      }
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
#else   // !defined(__HIP__GFX9__) TODO: Add NAVI support
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_hf_big_(const int K, const int M, const int Bx,
                                 const int By, const scalar_t* B,
                                 const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ BIAS, scalar_t* C,
                                 const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__GFX9__) TODO: Add NAVI support

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

#define WVSPLITK(_YTILE, _UNRL, _N)                                        \
  {                                                                        \
    dim3 block(64, 16);                                                    \
    int __wvPrGrp = mindiv(M_in, CuCount * _YTILE, 16);                    \
    if ((K_in * N_in <= max_lds_len) && (M_in % _YTILE == 0))              \
      wvSplitK_hf_sml_<fptype, 64, _YTILE, 16, 8, _UNRL, _N>               \
          <<<grid, block, 0, stream>>>(K_in, M_in, Bx_in, By_in, af4, bf4, \
                                       biasf4, c, __wvPrGrp, CuCount);     \
    else if (K_in * N_in <= max_lds_len * 1.2)                             \
      wvSplitK_hf_<fptype, 64, _YTILE, 16, 8, _UNRL, _N>                   \
          <<<grid, block, 0, stream>>>(K_in, M_in, Bx_in, By_in, af4, bf4, \
                                       biasf4, c, __wvPrGrp, CuCount);     \
    else                                                                   \
      wvSplitK_hf_big_<fptype, 64, _YTILE, 16, 8, _UNRL, _N>               \
          <<<grid, block, 0, stream>>>(K_in, M_in, Bx_in, By_in, af4, bf4, \
                                       biasf4, c, __wvPrGrp, CuCount);     \
  }

#define WVSPLIT_TILE(_sYT, __N)                           \
  {                                                       \
    bool fit_lds = (K_in * N_in <= max_lds_len);          \
    if (_sYT <= 1)                                        \
      WVSPLITK(1, 4, __N)                                 \
    else if ((__N == 1) || (!fit_lds) || (_sYT <= 4 * 2)) \
      WVSPLITK(2, 2, __N)                                 \
    else if (_sYT <= 4 * 3)                               \
      WVSPLITK(3, 2, __N)                                 \
    else if (__N == 4)                                    \
      WVSPLITK(4, 1, __N)                                 \
    else                                                  \
      WVSPLITK(4, 2, __N)                                 \
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

    switch (N_in) {
      case 1:
        WVSPLIT_TILE(sYT, 1)
        break;
      case 2:
        WVSPLIT_TILE(sYT, 2)
        break;
      case 3:
        WVSPLIT_TILE(sYT, 3)
        break;
      case 4:
        WVSPLIT_TILE(sYT, 4)
        break;
      default:
        throw std::runtime_error(
            "Unsupported N value: " + std::to_string(M_in) + "," +
            std::to_string(K_in) + "," + std::to_string(N_in));
    }
  });
  return out_c;
}

#if defined(__gfx950__)  // TODO: Add NAVI support
  // This version targets big A[] cases, where it is much larger than LDS
  // capacity
  #define WVSPLITKRC_1KPASS
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GrpsShrB>

__global__ void __launch_bounds__(WvPrGrp* THRDS)
    __attribute__((amdgpu_waves_per_eu(1, 1)))
    wvSplitKrc_(const int actlN, const int K, const int M, const int Bx,
                const int By, const scalar_t* __restrict__ B,
                const scalar_t* __restrict__ A,
                const scalar_t* __restrict__ BIAS, float* glbl, scalar_t* C,
                const int CuCount) {
  // Use upper half of glbl buffer for atomic reduce counting
  int* cntr = (int*)(&glbl[M * N]);

  constexpr int NTILE = 16;
  constexpr int WVLDS_ = (NTILE * THRDS * A_CHUNK);
  constexpr int APAD = 1;
  constexpr int ASTRD = 64;
  constexpr int BPAD = 1;
  constexpr int BSTRD = 64;
  constexpr int WVLDS = ((WVLDS_ + (WVLDS_ / BSTRD) * 4 * BPAD));

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
  int constexpr kFit = 512;
  int constexpr kfitsPerRdc = 1;
  #endif

  bool doRdc = (kfitsPerRdc * kFit < K);
  uint32_t numCuWithFullK =
      ((M + (WvPrGrp * YTILE / GrpsShrB) - 1) / (WvPrGrp * YTILE / GrpsShrB));
  uint32_t Mmod = numCuWithFullK * (WvPrGrp * YTILE / GrpsShrB);

  // given above k-split, find this wave's position
  uint32_t kFitPdd = kFit + (kFit / ASTRD) * APAD;
  uint32_t m0 = (blockIdx.x * WvPrGrp / GrpsShrB) * YTILE;
  uint32_t m1 = ((threadIdx.y % WvPrGrp) / GrpsShrB) * YTILE;
  uint32_t m = (m0 + m1) % Mmod;
  const uint32_t k_str = (m0 / Mmod) * kFit * kfitsPerRdc;
  uint32_t k_end = (m0 / Mmod + 1) * kFit * kfitsPerRdc;
  const uint32_t k_rnd = (K + kFit * kfitsPerRdc - 1) / (kFit * kfitsPerRdc);

  scalar8 sum4[N / NTILE / GrpsShrB][1];
  bigType bigB_[YTILE / GrpsShrB][UNRL];
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
    uint32_t k_ = k + threadIdx.x * A_CHUNK;
    const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
    #pragma unroll
    for (uint32_t y = 0; y < YTILE / GrpsShrB; y++)
      bigB_[y][k2].h8 = (loadnt(
          (scalar8*)(&B_[min__(y * GrpsShrB + bLoader + m, M - 1) * K])));
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
        const uint32_t thrd = ((threadIdx.y / sprdN) * THRDS + threadIdx.x);

  #ifndef WVSPLITKRC_1KPASS
    #pragma unroll
        for (int k = 0; k < kFit; k += THRDS * (WvPrGrp / sprdN) * A_CHUNK) {
  #else
        const unsigned int k = 0;
        {
  #endif
          unsigned int kOff = k + (thrd * A_CHUNK);
          unsigned int kOffcp = min__(K - A_CHUNK, k_str + kOff);
          const unsigned int k_in = kOffcp + ((threadIdx.y % sprdN)) * K;
          const unsigned int k_ot = kOff + ((threadIdx.y % sprdN)) * kFitPdd;
          for (unsigned int n = 0; n < N / 2; n += sprdN) {
            __builtin_amdgcn_global_load_lds((int*)(&A[k_in + n * K]),
                                             (int*)(&s[(k_ot + n * kFitPdd)]),
                                             16, 0, 0);
            if (((threadIdx.y % sprdN)) + n + N / 2 >= actlN) continue;
            __builtin_amdgcn_global_load_lds(
                (int*)(&A[k_in + (n + N / 2) * K]),
                (int*)(&s[(k_ot + (n + N / 2) * kFitPdd)]), 16, 0, 0);
          }

          // Stage loaded B[] to LDS for MFMA swizzling...
          for (uint32_t k2 = 0; k2 < UNRL; k2++) {
            uint32_t k = k1 + k2 * THRDS * A_CHUNK;
            uint32_t k_ = k + threadIdx.x * A_CHUNK;
            const bool oob_k = (k_ >= K);
            for (uint32_t y = 0; y < YTILE / GrpsShrB; y++) {
              uint32_t idx = threadIdx.x * 4 +
                             (y * GrpsShrB + bLoader) * ((THRDS + BPAD) * 4);
              // zero out if oob
              *((scalar8*)&myStg[idx]) =
                  (oob_k || (y * GrpsShrB + bLoader + m >= M))
                      ? 0
                      : bigB_[y][k2].h8;
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
        for (uint32_t y = 0; y < YTILE / GrpsShrB; y++)
          bigB_[y][k2].h8 = (loadnt(
              (scalar8*)(&B_[min__(y * GrpsShrB + bLoader + m, M - 1) * K])));
      }
  #endif

    // B[] staging is cooperative across GrpsShrB, so sync here before reading
    // back
    __syncthreads();

    // read back B[] swizzled for MFMA...
    bigType bigB[YTILE][UNRL];
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      for (uint32_t y = 0; y < YTILE; y++) {
        unsigned int idx = (threadIdx.x % YTILE) * ((THRDS + BPAD) * 4) +
                           (threadIdx.x / YTILE) * 4 + y * 16;
        bigB[y][k2].h8 = *((scalar8*)&myStg[idx]);
      }
    }

    // rReadback A[] swizzled for MFMA...
    bigType bigA[N / GrpsShrB][UNRL];
  #pragma unroll
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      uint32_t k = k1 + k2 * THRDS * A_CHUNK - kBase - k_str;
  #pragma unroll
      for (uint32_t nt = 0; nt < N / GrpsShrB; nt += NTILE)
  #pragma unroll
        for (uint32_t n = 0; n < NTILE; n++) {
          uint32_t idxa = (nt + (threadIdx.x % NTILE) +
                           (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) *
                              kFitPdd +
                          A_CHUNK * ((threadIdx.x / NTILE) + n * 4) + k;
          bigA[nt + n][k2] = *((const bigType*)(&(s[idxa])));
        }
    }

    // Do the MFMAs
  #pragma unroll
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
  #pragma unroll
      for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
        if constexpr (std::is_same_v<scalar_t, half>) {
          sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x16f16(
              bigA[nt * NTILE + 0][k2].h4[0], bigB[0][k2].h4[0],
              (k1 == k_str) ? ((scalar8){0}) : sum4[nt][0], 0, 0, 0);
          sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x16f16(
              bigA[nt * NTILE + 0][k2].h4[1], bigB[0][k2].h4[1], sum4[nt][0], 0,
              0, 0);
        } else {  // bf16
          sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
              bigA[nt * NTILE + 0][k2].h4[0], bigB[0][k2].h4[0],
              (k1 == k_str) ? ((scalar8){0}) : sum4[nt][0], 0, 0, 0);
          sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
              bigA[nt * NTILE + 0][k2].h4[1], bigB[0][k2].h4[1], sum4[nt][0], 0,
              0, 0);
        }
  #pragma unroll
        for (uint32_t j = 1; j < YTILE; j++) {
          if constexpr (std::is_same_v<scalar_t, half>) {
            sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                bigA[nt * NTILE + j][k2].h4[0], bigB[j][k2].h4[0], sum4[nt][0],
                0, 0, 0);
            sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                bigA[nt * NTILE + j][k2].h4[1], bigB[j][k2].h4[1], sum4[nt][0],
                0, 0, 0);
          } else {  // bf16
            sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                bigA[nt * NTILE + j][k2].h4[0], bigB[j][k2].h4[0], sum4[nt][0],
                0, 0, 0);
            sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                bigA[nt * NTILE + j][k2].h4[1], bigB[j][k2].h4[1], sum4[nt][0],
                0, 0, 0);
          }
        }
      }
    }
  }

  if (!doRdc) {
    if (m + (threadIdx.x % 16) < M) {
      scalar_t biases[N / NTILE / GrpsShrB][4] = {0};
      if (BIAS)
        for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
          for (uint32_t j = 0; j < 4; j++) {
            int mindx = m + (threadIdx.x % 16);
            int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                        (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
            biases[nt][j] = BIAS[(mindx % Bx) + (nindx % By) * M];
          }
        }
      for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
        for (uint32_t j = 0; j < 4; j++) {
          int mindx = m + (threadIdx.x % 16);
          int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                      (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
          int adr = mindx + M * nindx;
          if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
            if (BIAS) sum4[nt][0][j] += __bfloat162float(biases[nt][j]);
            C[adr] = __float2bfloat16(sum4[nt][0][j]);
          } else {
            if (BIAS) sum4[nt][0][j] += __half2float(biases[nt][j]);
            C[adr] = __float2half(sum4[nt][0][j]);
          }
        }
      }
    }
  } else {
    if (m + (threadIdx.x % 16) < M) {
      int my_cntr;
      if (!BIAS) {
        int mindx = m + (threadIdx.x % 16);
        for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++)
          for (uint32_t j = 0; j < 4; j++) {
            int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                        (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
            int adr = mindx + M * nindx;
            atomicAdd(&glbl[adr], sum4[nt][0][j]);
          }
        int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                     (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
        int adr_ = mindx + M * nindx_ / 4;
        my_cntr = atomicAdd(&cntr[adr_], 1);
        float vals[N / NTILE / GrpsShrB][4] = {};
        if (my_cntr + 1 == k_rnd) {
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              int adr = mindx + M * nindx;
              vals[nt][j] = glbl[adr];
            }
          }
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              if (nindx >= actlN) break;
              int adr = mindx + M * nindx;
              if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
                C[adr] = __float2bfloat16(vals[nt][j]);
              } else {
                C[adr] = __float2half(vals[nt][j]);
              }
            }
          }
        }
      } else {
        int mindx = m + (threadIdx.x % 16);
        scalar_t biases[N / NTILE / GrpsShrB][4] = {};
        // Atomic add the output, read biases
        for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++)
          for (uint32_t j = 0; j < 4; j++) {
            int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                        (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
            int adr = mindx + M * nindx;
            atomicAdd(&glbl[adr], sum4[nt][0][j]);
            biases[nt][j] = BIAS[(mindx % Bx) + (nindx % By) * M];
          }
        int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                     (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
        int adr_ = mindx + M * nindx_ / 4;
        // Update the complete counter
        my_cntr = atomicAdd(&cntr[adr_], 1);
        float vals[N / NTILE / GrpsShrB][4] = {};
        // If we're the last k-shard, read back the value and convert...
        if (my_cntr + 1 == k_rnd) {
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              int adr = mindx + M * nindx;
              vals[nt][j] = glbl[adr];
            }
          }
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              if (nindx >= actlN) break;
              int adr = mindx + M * nindx;
              if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
                vals[nt][j] += __bfloat162float(biases[nt][j]);
                C[adr] = __float2bfloat16(vals[nt][j]);
              } else {
                vals[nt][j] += __half2float(biases[nt][j]);
                C[adr] = __float2half(vals[nt][j]);
              }
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
#else   // !defined(__HIP__GFX9__) TODO: Add NAVI support
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GrpsShrB>
__global__ void wvSplitKrc_(const int actlN, const int K, const int M,
                            const int Bx, const int By, const scalar_t* B,
                            const scalar_t* __restrict__ A,
                            const scalar_t* __restrict__ BIAS, float* glbl,
                            // int* cntr,
                            scalar_t* C, const int CuCount){UNREACHABLE_CODE}
#endif  // defined(__HIP__GFX9__) TODO: Add NAVI support

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount) {
  auto M_in = in_a.size(0);
  auto N_in = in_b.size(0);
  auto K_in = in_a.size(1);
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

  auto N_p2 = 1U << (32 - __builtin_clz(N_in - 1));
  auto axl_glbl = torch::empty(
      {N_p2 + N_p2 / 4, M_in + M_in / 4},
      torch::TensorOptions().dtype(torch::kFloat32).device(in_b.device()));
  axl_glbl.zero_();  // disable for FAST_UNSAFE_RDC_INIT

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // const int max_lds_len = get_lds_size() / 2;

#define WVSPLITKrc(_WvPrGrp, _YTILE, _UNRL, _N, _GrpsShrB)                     \
  {                                                                            \
    dim3 block(64, _WvPrGrp);                                                  \
    wvSplitKrc_<fptype, 64, _YTILE, _WvPrGrp, 8, _UNRL, _N, _GrpsShrB>         \
        <<<grid, block, 0, stream>>>(N_in, K_in, M_in, Bx_in, By_in, af4, bf4, \
                                     biasf4, glbl, c, CuCount);                \
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_b.scalar_type(), "wvSplitKrc", [&] {
    using fptype = typename scalar<scalar_t>::type;
    fptype* af4 = reinterpret_cast<fptype*>(in_a.data_ptr());
    const fptype* bf4 = reinterpret_cast<const fptype*>(in_b.data_ptr());
    const fptype* biasf4 =
        (in_bias.has_value() && in_bias->numel() > 0)
            ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
            : nullptr;
    fptype* c = reinterpret_cast<fptype*>(out_c.data_ptr());
    auto glbl = axl_glbl.data_ptr<float>();
    switch (N_p2) {
      case 16:
        WVSPLITKrc(4, 16, 1, 16, 1) break;
      case 32:
        WVSPLITKrc(4, 16, 1, 32, 2) break;
      case 64:
        WVSPLITKrc(4, 16, 1, 64, 2) break;
      case 128:
        WVSPLITKrc(4, 16, 1, 128, 4) break;
      default:
        throw std::runtime_error(
            "Unsupported N value: " + std::to_string(M_in) + "," +
            std::to_string(K_in) + "," + std::to_string(N_in));
    }
  });
  return out_c;
}

#if defined(__HIP__MI3XX__)  // TODO: Add NAVI support
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

  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  float sA = *s_A;
  float sB = *s_B;

  while (m < M) {
    floatx16 sum[N][YTILE] = {};

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
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[(y + m) * Kbp])));
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
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        if (k >= K) break;

        for (uint32_t n = 0; n < N; n++) {
          for (int i = 0; i < A_CHUNK; i += 8) {
            for (int y = 0; y < YTILE; ++y) {
              sum[n][y] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                  bigA[n][k2].l[i / 8], bigB[y][k2].l[i / 8], sum[n][y], 0, 0,
                  0);
            }
          }
        }
      }
    }

    // Final reduction
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        float accm0 = sum[n][y][0];
        float accm16 = sum[n][y][8];
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][1], 0x101, 0xf, 0xf,
                                          1);  // row_shl1
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][9], 0x101, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][2], 0x102, 0xf, 0xf,
                                          1);  // row_shl2
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][10], 0x102, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][3], 0x103, 0xf, 0xf,
                                          1);  // row_shl3
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][11], 0x103, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][4], 0x108, 0xf, 0xf,
                                          1);  // row_shl8
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][12], 0x108, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][5], 0x109, 0xf, 0xf,
                                          1);  // row_shl9
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][13], 0x109, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][6], 0x10a, 0xf, 0xf,
                                          1);  // row_shl10
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][14], 0x10a, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][7], 0x10b, 0xf, 0xf,
                                          1);  // row_shl11
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][15], 0x10b, 0xf, 0xf, 1);
        accm0 += __shfl(accm0, 36);
        accm16 += __shfl(accm16, 52);
        sum[n][y][0] = accm0 + __shfl(accm16, 16);
      }
    }

    scalar_t biases[N][YTILE] = {};
    if (BIAS)
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          biases[n][y] = BIAS[(m + y) % Bx + (n % By) * M];
        }
      }
    if (threadIdx.x == 0) {
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          if (y + m >= M) break;  // To avoid mem access fault.
          sum[n][y][0] *= sA * sB;
          if constexpr (std::is_same_v<scalar_t, half>) {
            sum[n][y][0] += __half2float(biases[n][y]);
          } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
            sum[n][y][0] += __bfloat162float(biases[n][y]);
          }
          C[m + y + n * M] = __float2s<scalar_t>(sum[n][y][0]);
        }
      }
    }

    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__MI3XX__) TODO: Add NAVI support
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
#endif  // defined(__HIP__MI3XX__) TODO: Add NAVI support

#if defined(__HIP__MI3XX__)  // TODO: Add NAVI support
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

  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  float sA = *s_A;
  float sB = *s_B;

  while (m < M) {
    floatx16 sum[N][YTILE] = {};

    bigType bigA[N][UNRL] = {};
    bigType bigB[YTILE][UNRL];

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const fp8_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; ++y) {
          // if (y + m >= M) break;  // To avoid mem access fault.
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
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        for (uint32_t n = 0; n < N; n++) {
          for (int i = 0; i < A_CHUNK; i += 8) {
            for (int y = 0; y < YTILE; ++y) {
              sum[n][y] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                  bigA[n][k2].l[i / 8], bigB[y][k2].l[i / 8], sum[n][y], 0, 0,
                  0);
            }
          }
        }
      }
    }

    // Final reduction
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        float accm0 = sum[n][y][0];
        float accm16 = sum[n][y][8];
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][1], 0x101, 0xf, 0xf,
                                          1);  // row_shl1
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][9], 0x101, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][2], 0x102, 0xf, 0xf,
                                          1);  // row_shl2
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][10], 0x102, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][3], 0x103, 0xf, 0xf,
                                          1);  // row_shl3
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][11], 0x103, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][4], 0x108, 0xf, 0xf,
                                          1);  // row_shl8
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][12], 0x108, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][5], 0x109, 0xf, 0xf,
                                          1);  // row_shl9
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][13], 0x109, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][6], 0x10a, 0xf, 0xf,
                                          1);  // row_shl10
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][14], 0x10a, 0xf, 0xf, 1);
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][7], 0x10b, 0xf, 0xf,
                                          1);  // row_shl11
        accm16 += __builtin_amdgcn_mov_dpp(sum[n][y][15], 0x10b, 0xf, 0xf, 1);
        accm0 += __shfl(accm0, 36);
        accm16 += __shfl(accm16, 52);
        sum[n][y][0] = accm0 + __shfl(accm16, 16);
      }
    }

    if (threadIdx.x == 0) {
      scalar_t biases[N][YTILE] = {};
      if (BIAS)
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            biases[n][y] = BIAS[(m + y) % Bx + (n % By) * M];
          }
        }
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          if (y + m >= M) break;  // To avoid mem access fault.
          sum[n][y][0] *= sA * sB;
          if constexpr (std::is_same_v<scalar_t, half>) {
            sum[n][y][0] += __half2float(biases[n][y]);
          } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
            sum[n][y][0] += __bfloat162float(biases[n][y]);
          }
          C[m + y + n * M] = __float2s<scalar_t>(sum[n][y][0]);
        }
      }
    }

    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__MI3XX__) TODO: Add NAVI support
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
#endif  // defined(__HIP__MI3XX__) TODO: Add NAVI support

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

#define WVSPLITKQ(_WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N)             \
  {                                                                           \
    dim3 block(64, _WvPrGrp);                                                 \
    if ((Kap_in * N_in <= max_lds_len) && (M_in % _YTILEs == 0)) {            \
      int __wvPrGrp = mindiv(M_in, CuCount * _YTILEs, _WvPrGrp);              \
      wvSplitKQ_hf_sml_<fptype, fp8_t, 64, _YTILEs, _WvPrGrp, 16, _UNRLs, _N> \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, b_ptr, a_ptr, bias_ptr, c_ptr,  \
                                       s_a, s_b, __wvPrGrp, CuCount);         \
    } else {                                                                  \
      int __wvPrGrp = mindiv(M_in, CuCount * _YTILEm, _WvPrGrp);              \
      wvSplitKQ_hf_<fptype, fp8_t, 64, _YTILEm, _WvPrGrp, 16, _UNRLm, _N>     \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, b_ptr, a_ptr, bias_ptr, c_ptr,  \
                                       s_a, s_b, __wvPrGrp, CuCount);         \
    }                                                                         \
  }

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
          WVSPLITKQ(12, 2, 2, 2, 2, 1)
          break;
        case 2:
          WVSPLITKQ(12, 2, 2, 2, 2, 2)
          break;
        case 3:
          WVSPLITKQ(8, 2, 2, 1, 1, 3)
          break;
        case 4:
          WVSPLITKQ(4, 2, 2, 1, 1, 4)
          break;
        default:
          throw std::runtime_error(
              "Unsupported N value: " + std::to_string(M_in) + "," +
              std::to_string(K_in) + "," + std::to_string(N_in));
      }
    });
  });
}
