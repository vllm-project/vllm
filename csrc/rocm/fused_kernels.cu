#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <algorithm>

constexpr int WARP_SIZE = 64;

template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
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
  // auto dat0 = *(addr_alias);
  // auto dat1 = *(addr_alias+1);
  // auto dat2 = *(addr_alias+2);
  // auto dat3 = *(addr_alias+3);
  return make_float4(dat0, dat1, dat2, dat3);
}

// TBlock fetches entire rows of A, and entire col of B (K dimension); assume
// N=1 for time being grid is M/A_NUM_ROWS blocks
template <int NUM_A_ROWS_PER_BLOCK>
__global__ void LLGemm_Silu_kernel(float4* af4, __half2* bf4, _Float16* c,
                                   const int d) {
  __shared__ float red_smem[NUM_A_ROWS_PER_BLOCK][WARP_SIZE];
  const int row_addr = blockIdx.x * NUM_A_ROWS_PER_BLOCK / 2 * blockDim.x;
  const int row_addr_d = row_addr + d * blockDim.x;
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
  for (int i = 0; i < NUM_A_ROWS_PER_BLOCK / 2; i++) {
    rowA_elem4[2 * i] = load_ntmprl(&af4[row_addr + i * blockDim.x + threadid]);
    rowA_elem4[2 * i + 1] =
        load_ntmprl(&af4[row_addr_d + i * blockDim.x + threadid]);
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
    if (lane == 0 or lane == 32) {
      // oval = __float22half2_rn(make_float2(acc[qwarpid],oval2));
      // c[blockIdx.x*NUM_A_ROWS_PER_BLOCK/2+qwarpid/2] = oval;

      c[blockIdx.x * NUM_A_ROWS_PER_BLOCK / 2 + qwarpid / 2] =
          silu(acc[qwarpid]) * oval2;
    }
  }  // threadid<WARP_SIZE
}
// define the kernel calling code:
// template <typename T>
void LLGemm_Silu(void* in_a, void* in_b, void* out_c, const int M, const int K,
                 cudaStream_t stream, const int rows_per_block = 4) {
  float4* af4 = reinterpret_cast<float4*>(in_a);
  auto* bf4 = reinterpret_cast<__half2*>(in_b);
  auto* c = reinterpret_cast<_Float16*>(out_c);
  const int d = M / 2;
  const int NUM_THREADS = K * 2 / 16;
  int NUM_BLOCKS = M / rows_per_block;
  if (rows_per_block == 2) {
    LLGemm_Silu_kernel<2>
        <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, d);
  } else if (rows_per_block == 4) {
    LLGemm_Silu_kernel<4>
        <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, d);
  } else if (rows_per_block == 8) {
    LLGemm_Silu_kernel<8>
        <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, d);
  } else if (rows_per_block == 16) {
    LLGemm_Silu_kernel<16>
        <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, d);
  } else {
    NUM_BLOCKS = M / 4;
    LLGemm_Silu_kernel<4>
        <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, d);
  }

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}
