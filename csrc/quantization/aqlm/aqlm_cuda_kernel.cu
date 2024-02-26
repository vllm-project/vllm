#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>

__global__ void Code1x16MatVec(
  const int4* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;
  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;

  __shared__ int4 sh_b[32 * 9];
  float res = 0;

  int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 8; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8)
        sh_b[9 * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += 32 * 8;

    int b_sh_rd = 9 * (threadIdx.x % 32);
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[4];
        // We bypass the L1 cache to avoid massive amounts of memory streaming that doesn't
        // actually help us; this brings > 2x speedup.
        asm volatile (
          "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
          : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
          : "l"((void*) &codebook[enc[i]])
        );
        half2* a = reinterpret_cast<half2*>(&dec);
        half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
        half2 res2 = {};
        #pragma unroll
        for (int j = 0; j < 4; j++)
          res2 = __hfma2(a[j], b[j], res2);
        res += __half2float(res2.x) + __half2float(res2.y);
        b_sh_rd++;
      }
      a_gl_rd += 32;
    }
  }

  if (pred) {
    #pragma unroll
    for (int i = 16; i > 0; i /= 2)
      res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0)
      reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
  }
}

__global__ void Code2x8MatVec(
  const int4* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;
  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;
  int lane = threadIdx.x % 8;

  extern __shared__ int4 sh[];
  int4* sh_b = sh;
  int4* sh_code = sh_b + 32 * 9;
  int4* sh_code0 = sh_code;
  int4* sh_code1 = sh_code + 256 * 8;

  for (int i = threadIdx.x; i < 2 * 256; i += blockDim.x) {
    int4 dec = codebook[i];
    #pragma unroll
    for (int j = 0; j < 8; j++)
      sh_code[8 * i + (j + lane) % 8] = dec;
  }
  __syncthreads();

  float res = 0;

  int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 8; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8)
        sh_b[9 * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += 32 * 8;

    int b_sh_rd = 9 * (threadIdx.x % 32);
    if (pred && a_gl_rd < a_gl_end) {
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(&A[a_gl_rd]);
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        half2* a0 = reinterpret_cast<half2*>(&sh_code0[8 * enc[2 * i + 0] + lane]);
        half2* a1 = reinterpret_cast<half2*>(&sh_code1[8 * enc[2 * i + 1] + lane]);
        half2*  b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
        half2 res2 = {};
        #pragma unroll
        for (int j = 0; j < 4; j++)
          res2 = __hfma2(__hadd2(a0[j], a1[j]), b[j], res2);
        res += __half2float(res2.x) + __half2float(res2.y);
        b_sh_rd++;
      }
      a_gl_rd += 32;
    }
  }

  if (pred) {
    #pragma unroll
    for (int i = 16; i > 0; i /= 2)
      res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0)
      reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
  }
}

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

const int THREAD_M = 16;

void  code1x16_matvec_cuda(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Code1x16MatVec<<<blocks, threads, 16*32*9, stream>>>(
    (const int4*) A,
    (const int4*) B,
    (int4*) C,
    (const int4*) codebook,
    prob_m,
    prob_k
  );
}

void  code2x8_matvec_cuda(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  int shared = 16 * (2 * 256 * 8 + 32 * 9);
  cudaFuncSetAttribute(
    Code2x8MatVec, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
  );
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Code2x8MatVec<<<blocks, threads, shared, stream>>>(
    (const int4*) A,
    (const int4*) B,
    (int4*) C,
    (const int4*) codebook,
    prob_m,
    prob_k
  );
}
