#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compat.cuh"

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT =  32;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width
) {
    int h = BLOCKHEIGHT * blockIdx.x;
    int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
    int h_end = min(h + BLOCKHEIGHT, height);

    __shared__ scalar_t blockvec[BLOCKWIDTH];
    int i = width * h + w;
    int g_h = h * 8;
    int h_range = (h_end - h) * 8;
    int k;
    unsigned int g;
    scalar_t w_tmp;


    int z_w = w / 8;
    int z_mod = (w % 8) * 4;

    float weight[BLOCKWIDTH];

    if (w < width) {
        for (k = 0; k < h_range; ++k) {
    	      int k_w = (k / 8);
	          int k_bit = (k % 8) * 4;

            g = as_int(g_idx[g_h + k]);
            scalar_t scale = scales[g * width + w];
            scalar_t zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1);
            w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF);
            weight[k] = scale * (w_tmp - zero);
        }
    }

    scalar_t res;
    for (int b = 0; b < batch; ++b) {
	    res = 0;

        if (threadIdx.x < h_range) {
            blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
        }
        __syncthreads();
        if (w < width) {
	        for (k = 0; k < h_range; ++k){
	            res += weight[k] * blockvec[k];
            }
            atomicAdd(&mul[b * width + w], res);
        }
        __syncthreads();
    }
}

void vecquant4matmul_cuda(
    torch::Tensor vec,
    torch::Tensor mat,
    torch::Tensor mul,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor g_idx
) {
    int batch = vec.size(0);
    int vec_height = vec.size(1);
    int height = mat.size(0);
    int width = mat.size(1);
    int zero_width = zeros.size(1);

    dim3 blocks(
        (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
        (width + BLOCKWIDTH - 1) / BLOCKWIDTH
    );
    dim3 threads(BLOCKWIDTH);

    AT_DISPATCH_FLOATING_TYPES(
        vec.type(), "vecquant4matmul_cuda", ([&] {
            VecQuant4MatMulKernel<<<blocks, threads>>>(
                vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
                scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
                batch, vec_height, height, width, zero_width
            );
        })
    );
}

void gptq_descact_matmul(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}