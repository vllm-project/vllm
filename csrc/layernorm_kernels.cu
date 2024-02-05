#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "reduction_utils.cuh"

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void rms_norm_kernel(
  scalar_t* __restrict__ out,             // [..., hidden_size]
  const scalar_t* __restrict__ input,     // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size,
  bool use_shmem
  ) {
  __shared__ float s_variance;
  float variance = 0.0f;
  extern __shared__ __align__(sizeof(float)) char _shmem[];
  scalar_t* shmem = reinterpret_cast<scalar_t*>(_shmem);


  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = input[blockIdx.x * hidden_size + idx];
    if (use_shmem) {
      shmem[idx] = x;
    }
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = use_shmem?shmem[idx]:input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

// TODO: Further optimize this kernel.
template<typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
  scalar_t* __restrict__ input,           // [..., hidden_size]
  scalar_t* __restrict__ residual,        // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size,
  bool use_shmem
  ) {
  __shared__ float s_variance;
  float variance = 0.0f;
  extern __shared__ __align__(sizeof(float)) char _shmem[];
  scalar_t* shmem = reinterpret_cast<scalar_t*>(_shmem);

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * hidden_size + idx];
    x += (float) residual[blockIdx.x * hidden_size + idx];
    variance += x * x;
    if (use_shmem) {
      shmem[idx] = x;
    }
    residual[blockIdx.x * hidden_size + idx] = (scalar_t) x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = use_shmem?shmem[idx]:residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

} // namespace vllm


inline int getMaxSharedMemoryPerBlock(const torch::Tensor& input) {
  int max_shmem_size;
  cudaDeviceGetAttribute(&max_shmem_size, cudaDevAttrMaxSharedMemoryPerBlock, input.device().index());
  return max_shmem_size;
}


void rms_norm(
  torch::Tensor& out,      // [..., hidden_size]
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();



  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "rms_norm_kernel",
    [&] {
      bool use_shmem = true;
      //estimate the shared memory size
      int shmem_size = hidden_size * sizeof(scalar_t);

      if (shmem_size > getMaxSharedMemoryPerBlock(input)) {
        shmem_size = 0;
        use_shmem = false;
      }

      if (shmem_size >=48 * 1024) {
        VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(vllm::rms_norm_kernel<scalar_t>, shmem_size);
      }

      vllm::rms_norm_kernel<scalar_t><<<grid, block, shmem_size, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size,
        use_shmem
        );
    });
}

void fused_add_rms_norm(
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& residual, // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "fused_add_rms_norm_kernel",
    [&] {

      bool use_shmem = true;
      //estimate the shared memory size
      int shmem_size = hidden_size * sizeof(scalar_t);

      if (shmem_size > getMaxSharedMemoryPerBlock(input)) {
        shmem_size = 0;
        use_shmem = false;
      }
      if (shmem_size >=48 * 1024) {
        VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(vllm::fused_add_rms_norm_kernel<scalar_t>, shmem_size);
      }
      
      vllm::fused_add_rms_norm_kernel<scalar_t><<<grid, block, shmem_size, stream>>>(
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size,
        use_shmem
        );
    });
}