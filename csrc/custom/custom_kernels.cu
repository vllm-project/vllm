#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <algorithm>
#include "cuda_compat.h"

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx940__) || \
                           defined(__gfx941__) || defined(__gfx942__))
  #define __HIP__MI300_MI250__
#endif

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

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
  // auto dat0 = *(addr_alias);
  // auto dat1 = *(addr_alias+1);
  // auto dat2 = *(addr_alias+2);
  // auto dat3 = *(addr_alias+3);
  return make_float4(dat0, dat1, dat2, dat3);
}

// TBlock fetches entire rows of A, and entire col of B (K dimension); assume
// N=1 for time being grid is M/A_NUM_ROWS blocks
template <int NUM_A_ROWS_PER_BLOCK>
__global__ void LLGemm1_kernel(float4* af4, __half2* bf4, __half2* c,
                               const int K) {
  __shared__ float red_smem[NUM_A_ROWS_PER_BLOCK][WARP_SIZE];
  const int row_addr = blockIdx.x * NUM_A_ROWS_PER_BLOCK * K / 8;
  const int threadid = threadIdx.x;
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;
  const int qwarpid = threadid / 16;
  const int qthreadid = threadid % 16;
  float4 rowA_elem4[NUM_A_ROWS_PER_BLOCK];
  __half2 colB_elem4x, colB_elem4y, colB_elem4z, colB_elem4w;
  float4 sum4;  //[NUM_A_ROWS_PER_BLOCK];
  float acc[NUM_A_ROWS_PER_BLOCK] = {0.0};
  __half2 acch2;
  __half2 oval;

  // As we later use warp shuffle operations, we may have more threads in the
  // block than the actual available data, hence the if guard here.
  if (threadid * 8 < K) {
#pragma unroll
    for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
      // rowA_elem4[i] holds 8 * half numbers seen as a single float4.
      rowA_elem4[i] = load_ntmprl(&af4[row_addr + threadid + K / 8 * i]);
    }
  }

  colB_elem4x = bf4[threadid * 4 + 0];
  colB_elem4y = bf4[threadid * 4 + 1];
  colB_elem4z = bf4[threadid * 4 + 2];
  colB_elem4w = bf4[threadid * 4 + 3];

  __half2 Af2;
  __half2 Bf2;
  float2 S;

  auto Ah2ptr = reinterpret_cast<__half2*>(&rowA_elem4);
  __half2* ah2lptr;

#pragma unroll
  for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
    // Multiply-add on 8 half.
    ah2lptr = Ah2ptr + i * 4;
    Af2 = *(ah2lptr);
    acch2 = __hmul2(Af2, colB_elem4x);
    Af2 = *(ah2lptr + 1);
    acch2 = __hfma2(Af2, colB_elem4y, acch2);
    Af2 = *(ah2lptr + 2);
    acch2 = __hfma2(Af2, colB_elem4z, acch2);
    Af2 = *(ah2lptr + 3);
    acch2 = __hfma2(Af2, colB_elem4w, acch2);
    S = __half22float2(acch2);

    // See comment above concerning the if guard.
    if (threadid * 8 < K) {
      acc[i] = S.x + S.y;  // accumulation on float
    }
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

    if (threadid % WARP_SIZE == 0 or threadid % WARP_SIZE == 32) {
      oval = __float22half2_rn(make_float2(acc[qwarpid], oval2));
      c[blockIdx.x * NUM_A_ROWS_PER_BLOCK / 2 + qwarpid / 2] = oval;
    }
  }
}

// define the kernel calling code:
// template <typename T>
void LLGemm1(void* in_a, void* in_b, void* out_c, const int M, const int K,
             cudaStream_t stream, const int rows_per_block = 4) {
  float4* af4 = reinterpret_cast<float4*>(in_a);
  auto* bf4 = reinterpret_cast<__half2*>(in_b);
  auto* c = reinterpret_cast<__half2*>(out_c);

  // NUM_TREADS need to be a multiple of WARP_SIZE, as we are using warp shuffle
  // operations.
  const int NUM_THREADS =
      K * 2 / 16 % WARP_SIZE == 0
          ? K * 2 / 16
          : K * 2 / 16 + (WARP_SIZE - K * 2 / 16 % WARP_SIZE);

  int NUM_BLOCKS = M / rows_per_block;

  if (rows_per_block == 2) {
    LLGemm1_kernel<2><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
  } else if (rows_per_block == 4) {
    LLGemm1_kernel<4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
  } else if (rows_per_block == 8) {
    LLGemm1_kernel<8><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
  } else if (rows_per_block == 16) {
    LLGemm1_kernel<16><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
  } else {
    NUM_BLOCKS = M / 4;
    LLGemm1_kernel<4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
  }

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

// instantiate the kernel template for T=float:
// template void AddGPUKernel<float>(float *in_a, float *in_b, float *out_c,
// const int M, const int K, cudaStream_t stream);

const unsigned int TILE_WIDTH = 32;

// Compute C = A * B
__global__ void matrixMultiplyShared(float* A, float* B, float* C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  __shared__ float sA[TILE_WIDTH][TILE_WIDTH];  // Tile size of 32x32
  __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

  int Row = blockDim.y * blockIdx.y + threadIdx.y;
  int Col = blockDim.x * blockIdx.x + threadIdx.x;
  float Cvalue = 0.0;
  sA[threadIdx.y][threadIdx.x] = 0.0;
  sB[threadIdx.y][threadIdx.x] = 0.0;

  for (int ph = 0; ph < (((numAColumns - 1) / TILE_WIDTH) + 1); ph++) {
    if ((Row < numARows) && (threadIdx.x + (ph * TILE_WIDTH)) < numAColumns) {
      sA[threadIdx.y][threadIdx.x] =
          A[(Row * numAColumns) + threadIdx.x + (ph * TILE_WIDTH)];
    } else {
      sA[threadIdx.y][threadIdx.x] = 0.0;
    }
    if (Col < numBColumns && (threadIdx.y + ph * TILE_WIDTH) < numBRows) {
      sB[threadIdx.y][threadIdx.x] =
          B[(threadIdx.y + ph * TILE_WIDTH) * numBColumns + Col];
    } else {
      sB[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    for (int j = 0; j < TILE_WIDTH; ++j) {
      Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
    }
  }
  if (Row < numCRows && Col < numCColumns) {
    C[Row * numCColumns + Col] = Cvalue;
  }
}

void MMGPUKernel(float* in_a, float* in_b, float* out_c, int numARows,
                 int numAColumns, int numBRows, int numBColumns, int numCRows,
                 int numCColumns, cudaStream_t stream) {
  // Initialize the grid and block dimensions
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((numCColumns / TILE_WIDTH) + 1, (numCRows / TILE_WIDTH) + 1, 1);
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(
      in_a, in_b, out_c, numARows, numAColumns, numBRows, numBColumns, numCRows,
      numCColumns);

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

template <int nThreads_per_row, int CTA, int MT0, int MT1>
__global__ __launch_bounds__(512) void HGEMV_WFPerRow(
    int m, int n, const _Float16* A, int lda, const _Float16* x, _Float16* y) {
  int num_row_per_block = CTA / nThreads_per_row;
  int row_id = (blockIdx.x * num_row_per_block + threadIdx.y) * MT0;
  int inc = (gridDim.x * num_row_per_block) * MT0;

  while (row_id < m) {
    float2 sum2[MT0];

#pragma unroll
    for (int i = 0; i < MT0; ++i) {
      sum2[i] = {0.0, 0.0};
    }

    for (int j = threadIdx.x; j < n; j += (nThreads_per_row * MT1)) {
      bool is_active = j < n;
      if (is_active) {
        float2 x2[MT1 >> 1];
#pragma unroll
        for (int offset = 0; offset < MT1; offset += 2) {
          x2[offset >> 1] = {x[j + nThreads_per_row * offset],
                             x[j + nThreads_per_row * (offset + 1)]};
        }
        float2 a2[MT0][MT1 >> 1];
#pragma unroll
        for (int i = 0; i < MT0; i++) {
#pragma unroll
          for (int offset = 0; offset < MT1; offset += 2) {
            a2[i][offset >> 1] = {
                A[(row_id + i) * n + j + nThreads_per_row * offset],
                A[(row_id + i) * n + j + nThreads_per_row * (offset + 1)]};
          }
        }

#pragma unroll
        for (int i = 0; i < MT0; i++) {
#pragma unroll
          for (int offset = 0; offset < (MT1 >> 1); offset++) {
            sum2[i] += a2[i][offset] * x2[offset];
          }
        }
      }
    }
    float sum[MT0];
#pragma unroll
    for (int i = 0; i < MT0; i++) {
      sum[i] = sum2[i].x + sum2[i].y;
    }

#pragma unroll
    for (int i = 0; i < MT0; i++) {
#pragma unroll
      for (int offset = nThreads_per_row >> 1; offset >= 1;
           offset = offset >> 1) {
        sum[i] += __shfl_down(sum[i], offset, nThreads_per_row);
      }
    }
    if (threadIdx.x == 0) {
#pragma unroll
      for (int i = 0; i < MT0; i++) {
        y[row_id + i] = sum[i];
      }
    }
    row_id += inc;
  }
}

void LLGemmZZ(void* in_a, void* in_b, void* out_c, const int M, const int K,
              cudaStream_t stream, const int solidx = 0) {
  // m -> M, n-> K
  dim3 grid(1024);
  dim3 block(64, 8);
  if (solidx == 0) {
    HGEMV_WFPerRow<64, 512, 4, 8><<<grid, block, 0, stream>>>(
        M, K, reinterpret_cast<const _Float16*>(in_a), K,
        reinterpret_cast<const _Float16*>(in_b),
        reinterpret_cast<_Float16*>(out_c));
  } else if (solidx == 1) {
    HGEMV_WFPerRow<64, 512, 2, 8><<<grid, block, 0, stream>>>(
        M, K, reinterpret_cast<const _Float16*>(in_a), K,
        reinterpret_cast<const _Float16*>(in_b),
        reinterpret_cast<_Float16*>(out_c));
  } else if (solidx == 2) {
    HGEMV_WFPerRow<64, 512, 1, 8><<<grid, block, 0, stream>>>(
        M, K, reinterpret_cast<const _Float16*>(in_a), K,
        reinterpret_cast<const _Float16*>(in_b),
        reinterpret_cast<_Float16*>(out_c));
  } else {
    HGEMV_WFPerRow<64, 512, 4, 8><<<grid, block, 0, stream>>>(
        M, K, reinterpret_cast<const _Float16*>(in_a), K,
        reinterpret_cast<const _Float16*>(in_b),
        reinterpret_cast<_Float16*>(out_c));
  }
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

/////////////////////////////////////////////

#define DTYPE half

__device__ __forceinline__ int mindiv(int N, int div1, int div2) {
  int nPrRnd = div1 * div2;
  int rnds0 = N / nPrRnd;
  nPrRnd -= div1 * 3;
  int rnds3 = N / nPrRnd;
  nPrRnd -= div1;
  int rnds4 = N / nPrRnd;
  nPrRnd -= div1;
  int rnds5 = N / nPrRnd;
  nPrRnd -= div1;
  int rnds6 = N / nPrRnd;
  nPrRnd -= div1;
  int rnds7 = N / nPrRnd;
  nPrRnd -= div1;
  int rnds8 = N / nPrRnd;
  nPrRnd -= div1;
  int rnds9 = N / nPrRnd;
  nPrRnd -= div1;
  int rtn = div2;
  if (rnds0 == rnds3) rtn = div2 - 3;
  if (rnds0 == rnds4) rtn = div2 - 4;
  if (rnds0 == rnds5) rtn = div2 - 5;
  if (rnds0 == rnds6) rtn = div2 - 6;
  if (rnds0 == rnds7) rtn = div2 - 7;
  if (rnds0 == rnds8) rtn = div2 - 8;
  if (rnds0 == rnds9) rtn = div2 - 9;
  return rtn;
}

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support
// This version targets cases where A[] fits LDS capacity
template <int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int M>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSpltK_hf_sml_(const int K, const int N, const DTYPE* B,
                    const DTYPE* __restrict__ A, DTYPE* C, const int CuCount) {
  using half8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  union bigType {
    DTYPE h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half8 h8;
  };

  //----------------------------------------------------
  // Reserving 64 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not goint to work!
  //----------------------------------------------------
  __shared__ half s[1024 * 32];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  // uint32_t commitColumn[YTILE];
  // for (uint32_t i = 0; i < YTILE; i++) {
  //  commitColumn[i] = 1;
  //}

  // It's worth trying to load-balance...
  int _WvPrGrp = mindiv(N, CuCount * YTILE, WvPrGrp);

  //----------------------------------------------------
  // Indexing function into the column of weight matrix B
  // Algorithm does 64 lane k-splitting / wave and uses
  // WG ID and Thread ID to find the index.
  //----------------------------------------------------
  uint32_t n = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmenation!
  // This will happen only for the last wave!
  // if (n < N && (n + YTILE) >= N) {
  //  uint32_t startColumn = N - YTILE;
  //  for (uint32_t i = 0; i < (n - startColumn); i++) {
  //    commitColumn[i] = 0;
  //  }
  //  n = startColumn;
  //}

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  for (uint32_t k = 0; k < min(K * M, 32 * 1024);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    // Transpose of A implementation
    // uint32_t k_ot = (k_in / M) + (k_in % M) * K; // transopse for
    // bank-conflict-free readback

    if (k_in >= min(K * M, 32 * 1024)) break;

    //((bigType*)(&s[k_in]))->b128 = ((bigType*)(&A[k_in]))->b128;
    *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
    //((bigType*)(&s[k_ot]))->b128 = ((bigType*)(&A[k_in]))->b128;
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  float sum[M][YTILE];

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
  while (n < N) {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    for (int i = 0; i < YTILE; i++)
      for (int m = 0; m < M; m++) sum[m][i] = 0;

    bigType bigA[M][UNRL];
    bigType bigB0[UNRL];
    bigType bigB1[UNRL];
    bigType bigB2[UNRL];
    bigType bigB3[UNRL];
    bigType bigB4[UNRL];
    bigType bigB5[UNRL];
    bigType bigB6[UNRL];
    bigType bigB7[UNRL];
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

        const half* B_ = &B[(n + 0) * K + k_];
        bigB0[k2].h8 = (loadnt((half8*)(&B_[0 * K])));
        //----------------------------------------------------
        // The following code with YTILE > 1 has to be deleted
        //----------------------------------------------------
        if (YTILE >= 2) bigB1[k2].h8 = (loadnt((half8*)(&B_[1 * K])));
        if (YTILE >= 3) bigB2[k2].h8 = (loadnt((half8*)(&B_[2 * K])));
        if (YTILE >= 4) bigB3[k2].h8 = (loadnt((half8*)(&B_[3 * K])));
        if (YTILE >= 5) bigB4[k2].h8 = (loadnt((half8*)(&B_[4 * K])));
        if (YTILE >= 6) bigB5[k2].h8 = (loadnt((half8*)(&B_[5 * K])));
        if (YTILE >= 7) bigB6[k2].h8 = (loadnt((half8*)(&B_[6 * K])));
        if (YTILE >= 8) bigB7[k2].h8 = (loadnt((half8*)(&B_[7 * K])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int m = 0; m < M; m++) {
          // if (k_ + K * m < 32 * 1024)
          bigA[m][k2] = *((const bigType*)(&(s[k_ + K * m])));
          // else
          //   bigA[m][k2] = *((const bigType*)(&(A[k_ + K * m])));
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
        for (uint32_t m = 0; m < M; m++) {
  #pragma unroll
          for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][0])
                : "0"(sum[m][0]), "v"(bigA[m][k2].f[b]), "v"(bigB0[k2].f[b]));

            //----------------------------------------------------
            // The following code with YTILE > 1
            //----------------------------------------------------
            if (YTILE >= 2)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][1])
                  : "0"(sum[m][1]), "v"(bigA[m][k2].f[b]), "v"(bigB1[k2].f[b]));
            if (YTILE >= 3)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][2])
                  : "0"(sum[m][2]), "v"(bigA[m][k2].f[b]), "v"(bigB2[k2].f[b]));
            if (YTILE >= 4)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][3])
                  : "0"(sum[m][3]), "v"(bigA[m][k2].f[b]), "v"(bigB3[k2].f[b]));
            if (YTILE >= 5)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][4])
                  : "0"(sum[m][4]), "v"(bigA[m][k2].f[b]), "v"(bigB4[k2].f[b]));
            if (YTILE >= 6)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][5])
                  : "0"(sum[m][5]), "v"(bigA[m][k2].f[b]), "v"(bigB5[k2].f[b]));
            if (YTILE >= 7)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][6])
                  : "0"(sum[m][6]), "v"(bigA[m][k2].f[b]), "v"(bigB6[k2].f[b]));
            if (YTILE >= 8)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][7])
                  : "0"(sum[m][7]), "v"(bigA[m][k2].f[b]), "v"(bigB7[k2].f[b]));
          }
        }
      }
    }

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    for (int m = 0; m < M; m++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
      }
    }
    if (threadIdx.x == 63) {
      for (int m = 0; m < M; m++) {
        for (int i = 0; i < YTILE; i++) {
          // if (commitColumn[i]) C[n + i + m * N] = __float2half(sum[m][i]);
          C[n + i + m * N] = __float2half(sum[m][i]);
        }
      }
    }

    n += CuCount * _WvPrGrp * YTILE;

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    // if (n < N && (n + YTILE) >= N) {
    //  uint32_t startColumn = N - YTILE;
    //  for (uint32_t i = 0; i < (n - startColumn); i++) {
    //    commitColumn[i] = 0;
    //  }
    //  n = startColumn;
    //}
  }
}
#else   // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support
template <int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int M>
__global__ void wvSpltK_hf_sml_(const int K, const int N, const DTYPE* B,
                                const DTYPE* __restrict__ A, DTYPE* C,
                                const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support
// This version targets cases where A[] marginally exceeds LDS capacity
template <int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int M>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSpltK_hf_(const int K, const int N, const DTYPE* B,
                const DTYPE* __restrict__ A, DTYPE* C, const int CuCount) {
  using half8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  union bigType {
    DTYPE h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half8 h8;
  };

  //----------------------------------------------------
  // Reserving 64 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not goint to work!
  //----------------------------------------------------
  __shared__ half s[1024 * 32];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  // It's worth trying to load-balance...
  int _WvPrGrp = mindiv(N, CuCount * YTILE, WvPrGrp);

  //----------------------------------------------------
  // Indexing function into the column of weight matrix B
  // Algorithm does 64 lane k-splitting / wave and uses
  // WG ID and Thread ID to find the index.
  //----------------------------------------------------
  uint32_t n = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmenation!
  // This will happen only for the last wave!
  if (n < N && (n + YTILE) >= N) {
    uint32_t startColumn = N - YTILE;
    for (uint32_t i = 0; i < (n - startColumn); i++) {
      commitColumn[i] = 0;
    }
    n = startColumn;
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
  for (uint32_t k = 0; k < min(K * M, 32 * 1024);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    // Transpose of A implementation
    // uint32_t k_ot = (k_in / M) + (k_in % M) * K; // transopse for
    // bank-conflict-free readback

    if (k_in >= min(K * M, 32 * 1024)) break;

    //((bigType*)(&s[k_in]))->b128 = ((bigType*)(&A[k_in]))->b128;
    *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
    //((bigType*)(&s[k_ot]))->b128 = ((bigType*)(&A[k_in]))->b128;
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  float sum[M][YTILE];

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
  while (n < N) {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    for (int i = 0; i < YTILE; i++)
      for (int m = 0; m < M; m++) sum[m][i] = 0;

    bigType bigA[M][UNRL];
    bigType bigB0[UNRL];
    bigType bigB1[UNRL];
    bigType bigB2[UNRL];
    bigType bigB3[UNRL];
    bigType bigB4[UNRL];
    bigType bigB5[UNRL];
    bigType bigB6[UNRL];
    bigType bigB7[UNRL];
    bigType bigB8[UNRL];
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

        const half* B_ = &B[(n + 0) * K + k_];
        bigB0[k2].h8 = (loadnt((half8*)(&B_[0 * K])));
        //----------------------------------------------------
        // The following code with YTILE > 1 has to be deleted
        //----------------------------------------------------
        if (YTILE >= 2) bigB1[k2].h8 = (loadnt((half8*)(&B_[1 * K])));
        if (YTILE >= 3) bigB2[k2].h8 = (loadnt((half8*)(&B_[2 * K])));
        if (YTILE >= 4) bigB3[k2].h8 = (loadnt((half8*)(&B_[3 * K])));
        if (YTILE >= 5) bigB4[k2].h8 = (loadnt((half8*)(&B_[4 * K])));
        if (YTILE >= 6) bigB5[k2].h8 = (loadnt((half8*)(&B_[5 * K])));
        if (YTILE >= 7) bigB6[k2].h8 = (loadnt((half8*)(&B_[6 * K])));
        if (YTILE >= 8) bigB7[k2].h8 = (loadnt((half8*)(&B_[7 * K])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int m = 0; m < M; m++) {
          if (k_ + K * m < 32 * 1024)
            bigA[m][k2] = *((const bigType*)(&(s[k_ + K * m])));
          else
            bigA[m][k2] = *((const bigType*)(&(A[k_ + K * m])));
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t m = 0; m < M; m++) {
  #pragma unroll
        for (uint32_t k2 = 0; k2 < UNRL; k2++) {
          uint32_t k = k1 + k2 * THRDS * A_CHUNK;
          uint32_t k_ = k + threadIdx.x * A_CHUNK;
          if (k_ >= K) break;
            // Do the matrix multiplication of activation and weight matrix
            // - Remember the accumulation is happening for K-split of 64!
  #pragma unroll
          for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][0])
                : "0"(sum[m][0]), "v"(bigA[m][k2].f[b]), "v"(bigB0[k2].f[b]));

            //----------------------------------------------------
            // The following code with YTILE > 1
            //----------------------------------------------------
            if (YTILE >= 2)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][1])
                  : "0"(sum[m][1]), "v"(bigA[m][k2].f[b]), "v"(bigB1[k2].f[b]));
            if (YTILE >= 3)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][2])
                  : "0"(sum[m][2]), "v"(bigA[m][k2].f[b]), "v"(bigB2[k2].f[b]));
            if (YTILE >= 4)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][3])
                  : "0"(sum[m][3]), "v"(bigA[m][k2].f[b]), "v"(bigB3[k2].f[b]));
            if (YTILE >= 5)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][4])
                  : "0"(sum[m][4]), "v"(bigA[m][k2].f[b]), "v"(bigB4[k2].f[b]));
            if (YTILE >= 6)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][5])
                  : "0"(sum[m][5]), "v"(bigA[m][k2].f[b]), "v"(bigB5[k2].f[b]));
            if (YTILE >= 7)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][6])
                  : "0"(sum[m][6]), "v"(bigA[m][k2].f[b]), "v"(bigB6[k2].f[b]));
            if (YTILE >= 8)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][7])
                  : "0"(sum[m][7]), "v"(bigA[m][k2].f[b]), "v"(bigB7[k2].f[b]));
          }
        }
      }
    }

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    for (int m = 0; m < M; m++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
      }
    }

    if (threadIdx.x == 63) {
      for (int m = 0; m < M; m++) {
        for (int i = 0; i < YTILE; i++) {
          if (commitColumn[i]) C[n + i + m * N] = __float2half(sum[m][i]);
        }
      }
    }

    n += CuCount * _WvPrGrp * YTILE;

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    if (n < N && (n + YTILE) >= N) {
      uint32_t startColumn = N - YTILE;
      for (uint32_t i = 0; i < (n - startColumn); i++) {
        commitColumn[i] = 0;
      }
      n = startColumn;
    }
  }
}

#else   // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support
template <int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int M>
__global__ void wvSpltK_hf_(const int K, const int N, const DTYPE* B,
                            const DTYPE* __restrict__ A, DTYPE* C,
                            const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support
// This version targets big A[] cases, where it is much larger than LDS capacity
template <int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int M>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSpltK_hf_big_(const int K, const int N, const DTYPE* B,
                    const DTYPE* __restrict__ A, DTYPE* C, const int CuCount) {
  using half8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;

  union bigType {
    DTYPE h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half8 h8;
  };

  //----------------------------------------------------
  // Reserving 64 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not goint to work!
  //----------------------------------------------------
  __shared__ half s[1024 * 32];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  // It's worth trying to load-balance...
  int _WvPrGrp = mindiv(N, CuCount * YTILE, WvPrGrp);
  if (threadIdx.y >= _WvPrGrp) return;

  //----------------------------------------------------
  // Indexing function into the column of weight matrix B
  // Algorithm does 64 lane k-splitting / wave and uses
  // WG ID and Thread ID to find the index.
  //----------------------------------------------------
  uint32_t n = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmenation!
  // This will happen only for the last wave!
  if (n < N && (n + YTILE) >= N) {
    uint32_t startColumn = N - YTILE;
    for (uint32_t i = 0; i < (n - startColumn); i++) {
      commitColumn[i] = 0;
    }
    n = startColumn;
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
  for (uint32_t k = 0; k < min(K * M, 32 * 1024);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    // Transpose of A implementation
    // uint32_t k_ot = (k_in / M) + (k_in % M) * K; // transopse for
    // bank-conflict-free readback

    if (k_in >= min(K * M, 32 * 1024)) break;

    //((bigType*)(&s[k_in]))->b128 = ((bigType*)(&A[k_in]))->b128;
    *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
    //((bigType*)(&s[k_ot]))->b128 = ((bigType*)(&A[k_in]))->b128;
  }
  __syncthreads();
  #endif

  #define TUC (THRDS * UNRL * A_CHUNK)
  uint32_t kBase = 0;
  // find biggest k size that fits in LDS
  uint32_t kFit = (32 * 1024) / M;
  // kFit = (kFit%TWC==0) ? kFit : (kFit-kFit%TWC+TWC); //round up to multiple
  // of TUC
  kFit = (kFit % TUC == 0)
             ? kFit
             : (kFit - kFit % TUC);  // round up to multiple of TUC
  // if (kFit == 0) kFit = TUC;
  kFit = min(kFit, K);

  float sum[M][YTILE];

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
  uint32_t Nrndp = (N % YW == 0) ? N : (N - N % YW + YW);
  while (n < Nrndp) {
  #else
  while (n < N) {
  #endif
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    for (int i = 0; i < YTILE; i++)
      for (int m = 0; m < M; m++) sum[m][i] = 0;

    bigType bigA[M][UNRL];
    bigType bigB0[UNRL];
    bigType bigB1[UNRL];
    bigType bigB2[UNRL];
    bigType bigB3[UNRL];
    bigType bigB4[UNRL];
    bigType bigB5[UNRL];
    bigType bigB6[UNRL];
    bigType bigB7[UNRL];
    bigType bigB8[UNRL];
    bigType bigB9[UNRL];
    bigType bigB10[UNRL];
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
          for (uint32_t m = 0; m < M; m++) {
            uint32_t k_in = kBase + m * K + kOff;
            uint32_t k_ot = m * kFit + kOff;
            *((bigType*)(&s[k_ot])) = *((bigType*)(&A[k_in]));
          }
        }
        __syncthreads();
      }
      if (n >= N) continue;
  #endif

        // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        const half* B_ = &B[(n + 0) * K + k_];
        bigB0[k2].h8 = (loadnt((half8*)(&B_[0 * K])));
        //----------------------------------------------------
        // The following code with YTILE > 1 has to be deleted
        //----------------------------------------------------
        if (YTILE >= 2) bigB1[k2].h8 = (loadnt((half8*)(&B_[1 * K])));
        if (YTILE >= 3) bigB2[k2].h8 = (loadnt((half8*)(&B_[2 * K])));
        if (YTILE >= 4) bigB3[k2].h8 = (loadnt((half8*)(&B_[3 * K])));
        if (YTILE >= 5) bigB4[k2].h8 = (loadnt((half8*)(&B_[4 * K])));
        if (YTILE >= 6) bigB5[k2].h8 = (loadnt((half8*)(&B_[5 * K])));
        if (YTILE >= 7) bigB6[k2].h8 = (loadnt((half8*)(&B_[6 * K])));
        if (YTILE >= 8) bigB7[k2].h8 = (loadnt((half8*)(&B_[7 * K])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int m = 0; m < M; m++) {
  #ifdef PCML
          bigA[m][k2] = *((const bigType*)(&(s[k_ - kBase + kFit * m])));
  #else
          if (k_ + K * m < 32 * 1024)
            bigA[m][k2] = *((const bigType*)(&(s[k_ + K * m])));
          else
            bigA[m][k2] = *((const bigType*)(&(A[k_ + K * m])));
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
        for (uint32_t m = 0; m < M; m++) {
          // Do the matrix multiplication of activation and weight matrix
          // - Remember the accumulation is happening for K-split of 64!
  #pragma unroll
          for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][0])
                : "0"(sum[m][0]), "v"(bigA[m][k2].f[b]), "v"(bigB0[k2].f[b]));

            //----------------------------------------------------
            // The following code with YTILE > 1
            //----------------------------------------------------
            if (YTILE >= 2)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][1])
                  : "0"(sum[m][1]), "v"(bigA[m][k2].f[b]), "v"(bigB1[k2].f[b]));
            if (YTILE >= 3)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][2])
                  : "0"(sum[m][2]), "v"(bigA[m][k2].f[b]), "v"(bigB2[k2].f[b]));
            if (YTILE >= 4)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][3])
                  : "0"(sum[m][3]), "v"(bigA[m][k2].f[b]), "v"(bigB3[k2].f[b]));
            if (YTILE >= 5)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][4])
                  : "0"(sum[m][4]), "v"(bigA[m][k2].f[b]), "v"(bigB4[k2].f[b]));
            if (YTILE >= 6)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][5])
                  : "0"(sum[m][5]), "v"(bigA[m][k2].f[b]), "v"(bigB5[k2].f[b]));
            if (YTILE >= 7)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][6])
                  : "0"(sum[m][6]), "v"(bigA[m][k2].f[b]), "v"(bigB6[k2].f[b]));
            if (YTILE >= 8)
              asm("v_dot2c_f32_f16 %0, %2, %3"
                  : "=v"(sum[m][7])
                  : "0"(sum[m][7]), "v"(bigA[m][k2].f[b]), "v"(bigB7[k2].f[b]));
          }
        }
      }
    }

  #ifdef PCML
    if (n >= N) {
      n += CuCount * _WvPrGrp * YTILE;
      kBase = 0;
      continue;
    }
  #endif

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    for (int m = 0; m < M; m++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
      }
    }

    if (threadIdx.x == 63) {
      for (int m = 0; m < M; m++) {
        for (int i = 0; i < YTILE; i++) {
          if (commitColumn[i]) C[n + i + m * N] = __float2half(sum[m][i]);
        }
      }
    }

    n += CuCount * _WvPrGrp * YTILE;
    kBase = 0;

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    if (n < N && (n + YTILE) >= N) {
      uint32_t startColumn = N - YTILE;
      for (uint32_t i = 0; i < (n - startColumn); i++) {
        commitColumn[i] = 0;
      }
      n = startColumn;
    }
  }
}
#else   // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support
template <int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int M>
__global__ void wvSpltK_hf_big_(const int K, const int N, const DTYPE* B,
                                const DTYPE* __restrict__ A, DTYPE* C,
                                const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

void wvSpltK_(void* in_a, void* in_b, void* out_c, const int M_in,
              const int K_in, const int N_in, cudaStream_t stream,
              const int CuCount = 0) {
  dim3 grid(CuCount);
  half* af4 = reinterpret_cast<half*>(in_a);
  const half* bf4 = reinterpret_cast<const half*>(in_b);
  auto* c = reinterpret_cast<half*>(out_c);

#define WVSPLTK(_WvPrGrp, _YTILEs, _YTILEm, _YTILEb, _UNRLs, _UNRLm, _UNRLb, \
                _N)                                                          \
  {                                                                          \
    dim3 block(64, _WvPrGrp);                                                \
    /*wvSpltK_hf:<int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL,  \
     * int M>*/                                                              \
    if ((K_in * N_in <= 32 * 1024) && (M_in % _YTILEs == 0)) {               \
      wvSpltK_hf_sml_<64, _YTILEs, _WvPrGrp, 8, _UNRLs, _N>                  \
          <<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c, CuCount);    \
    } else if (K_in * N_in <= 32 * 1024 * 1.2) {                             \
      wvSpltK_hf_<64, _YTILEm, _WvPrGrp, 8, _UNRLm, _N>                      \
          <<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c, CuCount);    \
    } else {                                                                 \
      wvSpltK_hf_big_<64, _YTILEb, _WvPrGrp, 8, _UNRLb, _N>                  \
          <<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c, CuCount);    \
    }                                                                        \
  }

  switch (N_in) {
    case 1:
      WVSPLTK(16, 2, 2, 2, 2, 2, 2, 1)  // MI308
      break;
    case 2:
      WVSPLTK(16, 2, 2, 2, 2, 2, 2, 2)  // MI308
      break;
    case 3:
      WVSPLTK(16, 4, 7, 7, 1, 1, 1, 3)  // MI308
      break;
    case 4:
      WVSPLTK(16, 4, 7, 7, 1, 1, 1, 4)  // MI308
      break;
    default:
      throw std::runtime_error("Unsupported N value: " + std::to_string(M_in) +
                               "," + std::to_string(K_in) + "," +
                               std::to_string(N_in));
  }

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
  }
}