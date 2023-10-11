#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// half-tensor
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDATensorMethods.cuh>

// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(
    double* address,
    double val
) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

const int BLOCKWIDTH  = 128;
const int BLOCKHEIGHT3 =  12;
const int BLOCKHEIGHT4 =  16;

__global__ void NUQ4MatMulKernel(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           half2* __restrict__ mul,
    const  __half* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
);

// 4-bit matvec kernel (LUT-based)
void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  NUQ4MatMulKernel<<<blocks, threads>>>(
    (half2*) vec.data<at::Half>(),
    mat.data_ptr<int>(),
    (half2*) mul.data<at::Half>(),
    (__half*) lookup_table.data<at::Half>(),
    height, width, batch, vec_height
  );
}


// 4-bit matvec kernel (LUT-based)
__global__ void NUQ4MatMulKernel(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           half2* __restrict__ mul,
    const  __half* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];

  __shared__ __half deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int column_offset = col * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  __half res;
  half2 res2;
  half2 tmp2;

  int i;
  int k;

  unsigned int tmp1;
  unsigned int lut_index1, lut_index2;

  for (int b = 0; b < batch; ++b){
    i = width * row + col;
    res = __int2half_rd(0);
    k = 0;

    __syncthreads();
    if (threadIdx.x < blockwidth2)
      blockvec[threadIdx.x] = vec[b * vec_height / 2 + (row / BLOCKHEIGHT4) * blockwidth2 + threadIdx.x];
    __syncthreads();

    while (k < blockwidth2) {
      tmp1 = as_unsigned(mat[i]);

      res2 = {};
      tmp2 = {};

      lut_index1 = tmp1 & 0xF;
      lut_index2 = (tmp1 >> 4) & 0xF;
      tmp2.x = deq2[lut_index1][off];
      tmp2.y = deq2[lut_index2][off];
      res2 = __hfma2(tmp2, blockvec[k + 0], res2);

      lut_index1 = (tmp1 >> 8) & 0xF;
      lut_index2 = (tmp1 >> 12) & 0xF;
      tmp2.x = deq2[lut_index1][off];
      tmp2.y = deq2[lut_index2][off];
      res2 = __hfma2(tmp2, blockvec[k + 1], res2);

      lut_index1 = (tmp1 >> 16) & 0xF;
      lut_index2 = (tmp1 >> 20) & 0xF;
      tmp2.x = deq2[lut_index1][off];
      tmp2.y = deq2[lut_index2][off];
      res2 = __hfma2(tmp2, blockvec[k + 2], res2);

      lut_index1 = (tmp1 >> 24) & 0xF;
      lut_index2 = (tmp1 >> 28) & 0xF;
      tmp2.x = deq2[lut_index1][off];
      tmp2.y = deq2[lut_index2][off];
      res2 = __hfma2(tmp2, blockvec[k + 3], res2);

      res = __hadd(__hadd(res2.x, res2.y), res);

      i += width;
      k += 4;
    }

    // col%2 -> only set one of the two values
    half2 res3 = {};
    if (col % 2 == 0) {
      res3.x = res;
    } else {
      res3.y = res;
    }

    atomicAdd(&mul[b * width / 2 + col / 2], res3);
  }
}
