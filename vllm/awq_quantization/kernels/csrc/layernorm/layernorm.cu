/*

Adapted from NVIDIA FasterTransformer:
https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/layernorm_kernels.cu

*/

#include <torch/extension.h>
#include <cuda_fp16.h>
#include "reduction.cuh"
#include "layernorm.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

static inline __device__ float to_float(half src)
{
    return __half2float(src);
}

static inline __device__ float to_float(float src)
{
    return src;
}

template<typename T>
__global__ void generalT5LayerNorm(
    const T* __restrict input, const T* __restrict gamma, T* output, const float layernorm_eps, int m, int n)
{
    // layernorm module in the T5 style No bias and no subtraction of mean.
    const int tid = threadIdx.x;

    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = to_float(__ldg(&input[blockIdx.x * n + i]));
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((to_float(input[blockIdx.x * n + i]) * s_variance) * to_float(__ldg(&gamma[i])));
    }
}


template<typename T>
void invokeGeneralT5LayerNorm(T*           out,
                              const T*     input,
                              const T*     gamma,
                              // const T*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0) {
        block.x = 1024;
    }

    block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision*/
    generalT5LayerNorm<T><<<grid, block>>>(input, gamma, out, layernorm_eps, m, n);  // For gpt-3
}

template void invokeGeneralT5LayerNorm(half*           out,
                              const half*     input,
                              const half*     gamma,
                              // const half*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n);

template void invokeGeneralT5LayerNorm(float*           out,
                              const float*     input,
                              const float*     gamma,
                              // const half*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n);



// input b, n, c
void layernorm_forward_cuda(
    torch::Tensor _input,
    torch::Tensor _gamma,
    torch::Tensor _out,
    float eps)
{
    int m = _input.size(0) * _input.size(1);
    int n = _input.size(2);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_input));

    auto input = reinterpret_cast<half*>(_input.data_ptr<at::Half>());
    auto gamma = reinterpret_cast<half*>(_gamma.data_ptr<at::Half>());
    auto out = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

    invokeGeneralT5LayerNorm(out, input, gamma, eps, m, n);
}
