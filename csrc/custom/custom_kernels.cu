#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <algorithm>

constexpr int WARP_SIZE = 64;

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
__global__ void LLGemm1_kernel(float4* af4, __half2* bf4, __half2* c) {
  __shared__ float red_smem[NUM_A_ROWS_PER_BLOCK][WARP_SIZE];
  const int row_addr = blockIdx.x * NUM_A_ROWS_PER_BLOCK * blockDim.x;
  // int row_addr_1 = row_addr + CUDA_NUM_THREADS;
  // int row_addr_2 = row_addr_1 + CUDA_NUM_THREADS;
  // int row_addr_3 = row_addr_2 + CUDA_NUM_THREADS;
  const int threadid = threadIdx.x;
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;
  const int qwarpid = threadid / 16;
  const int qthreadid = threadid % 16;
  float4 rowA_elem4[NUM_A_ROWS_PER_BLOCK];
  // float4 colB_elem4;
  __half2 colB_elem4x, colB_elem4y, colB_elem4z, colB_elem4w;
  float4 sum4;                      //[NUM_A_ROWS_PER_BLOCK];
  float acc[NUM_A_ROWS_PER_BLOCK];  //= 0.0;
  __half2 acch2;
  __half2 oval;

  // rowA_elem4 = af4[row_addr + threadid];
  //__syncthreads();
  // rowA_elem4_1 = af4[row_addr_1 + threadid];
  // rowA_elem4_2 = af4[row_addr_2 + threadid];
  // rowA_elem4_3 = af4[row_addr_3 + threadid];
#pragma unroll
  for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
    rowA_elem4[i] = load_ntmprl(&af4[row_addr + i * blockDim.x + threadid]);
    // rowA_elem4[i] = af4[row_addr + i*blockDim.x + threadid];
    //__syncthreads();
  }
  colB_elem4x = bf4[threadid * 4 + 0];
  colB_elem4y = bf4[threadid * 4 + 1];
  colB_elem4z = bf4[threadid * 4 + 2];
  colB_elem4w = bf4[threadid * 4 + 3];

  // __syncthreads();
  __half2 Af2;
  __half2 Bf2;
  float2 S;
  // auto Bh2ptr = reinterpret_cast<__half2 *>(&colB_elem4);
  // auto Bf2x = *Bh2ptr;
  // auto Bf2y = *(Bh2ptr+1);
  // auto Bf2z = *(Bh2ptr+2);
  // auto Bf2w = *(Bh2ptr+3);
  auto Ah2ptr = reinterpret_cast<__half2*>(&rowA_elem4);
  __half2* ah2lptr;
#pragma unroll
  for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
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
    acc[i] = S.x + S.y;
  }

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
#pragma unroll
    for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
      acc[i] += __shfl_xor(acc[i], mask);
    }
  }

  // Warp leaders store the data to shared memory.
  // if (lane == 0) {
  //   #pragma unroll
  //   for (int i=0; i<NUM_A_ROWS_PER_BLOCK; i++) {
  //     red_smem[i][warp] = acc[i];
  //   }
  // }

  if (lane < NUM_A_ROWS_PER_BLOCK) {
    red_smem[lane][warp] = acc[lane];
  }

  // Make sure the data is in shared memory.
  __syncthreads();
  if (qwarpid < NUM_A_ROWS_PER_BLOCK) {
    // if (threadid<64) {
    // #pragma unroll
    // for (int i=0; i<NUM_A_ROWS_PER_BLOCK/2; i++) {
    //     acc[i+2*qwarpid] = 0.0;
    // }
    ////acc[qwarpid] = 0.0;

    ////if (qthreadid<num_warps) {
    // #pragma unroll
    //   for (int i=0; i<NUM_A_ROWS_PER_BLOCK/2; i++) {
    //     acc[i+2*qwarpid] = red_smem[i+2*qwarpid][qthreadid];
    //   }
    ////acc[qwarpid] = red_smem[qwarpid][qthreadid];

    ////}
    acc[qwarpid] = qthreadid < num_warps ? red_smem[qwarpid][qthreadid] : 0.f;
    // if (threadid<32) {
#pragma unroll
    for (int mask = 16 / 2; mask >= 1; mask /= 2) {
      // #pragma unroll
      // for (int i=0; i<NUM_A_ROWS_PER_BLOCK/2; i++) {
      //   acc[i+2*qwarpid] += __shfl_xor(acc[i+2*qwarpid], mask);
      // }
      acc[qwarpid] += __shfl_xor(acc[qwarpid], mask);
    }
    float oval2 = __shfl_xor(acc[qwarpid], 16);
    // acc[1] = __shfl_xor(acc[1],16);
    // acc[3] = __shfl_xor(acc[3],16);
    //}
    //  __syncthreads();
    // if (threadid < NUM_A_ROWS_PER_BLOCK/2) {
    if (threadid % WARP_SIZE == 0 or threadid % WARP_SIZE == 32) {
      // oval =
      // __float22half2_rn(make_float2(acc[2*threadid],acc[2*threadid+1])); oval
      // = __float22half2_rn(make_float2(acc[2*qwarpid],acc[2*qwarpid+1])); oval
      // = __float22half2_rn(make_float2(acc[qwarpid],acc[qwarpid+1]));
      oval = __float22half2_rn(make_float2(acc[qwarpid], oval2));
      c[blockIdx.x * NUM_A_ROWS_PER_BLOCK / 2 + qwarpid / 2] = oval;
    }
  }  // threadid<WARP_SIZE

  // if (threadid < NUM_A_ROWS_PER_BLOCK/2) {
  //     acc[2*threadid] = 0.0;
  //     acc[2*threadid+1] = 0.0;
  //
  //     if (num_warps>8) {
  //       #pragma unroll
  //       for (int j=0; j<8; j++) {
  //         acc[2*threadid] += red_smem[2*threadid][j];
  //         acc[2*threadid+1] += red_smem[2*threadid+1][j];
  //       }
  //     }
  //       #pragma unroll
  //       for (int j=0; j<num_warps-8; j++) {
  //         acc[2*threadid] += red_smem[2*threadid][j+8];
  //         acc[2*threadid+1] += red_smem[2*threadid+1][j+8];
  //       }

  //      oval =
  //      __float22half2_rn(make_float2(acc[2*threadid],acc[2*threadid+1]));
  //      c[blockIdx.x*NUM_A_ROWS_PER_BLOCK/2+threadid] = oval;
  //}
}
// define the kernel calling code:
// template <typename T>
void LLGemm1(void* in_a, void* in_b, void* out_c, const int M, const int K,
             cudaStream_t stream, const int rows_per_block = 4) {
  float4* af4 = reinterpret_cast<float4*>(in_a);
  auto* bf4 = reinterpret_cast<__half2*>(in_b);
  auto* c = reinterpret_cast<__half2*>(out_c);
  // constexpr int A_ROWS_PER_BLOCK = 8;
  const int NUM_THREADS = K * 2 / 16;
  int NUM_BLOCKS = M / rows_per_block;
  if (rows_per_block == 2) {
    LLGemm1_kernel<2><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c);
  } else if (rows_per_block == 4) {
    LLGemm1_kernel<4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c);
  } else if (rows_per_block == 8) {
    LLGemm1_kernel<8><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c);
  } else if (rows_per_block == 16) {
    LLGemm1_kernel<16><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c);
  } else {
    NUM_BLOCKS = M / 4;
    LLGemm1_kernel<4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c);
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
