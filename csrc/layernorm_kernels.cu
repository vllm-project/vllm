#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "reduction_utils.cuh"
#include <c10/util/Half.h>
namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void rms_norm_kernel_impl_float(
  scalar_t* __restrict__ out,
  const scalar_t* __restrict__ input,
  const scalar_t* __restrict__ weight,
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}
  

__global__ void rms_norm_kernel_impl_half(
  __half* __restrict__ out,
  const __half* __restrict__ input,
  const __half* __restrict__ weight,
  const float epsilon,
  const int num_tokens,
  const int hidden_size)
{
  __shared__ float s_variance;
  float variance = 0.0f;

  const float4 *input_float4 = reinterpret_cast<const float4 *>(input) + blockIdx.x * hidden_size;
  float4 *out_float4 = reinterpret_cast<float4 *>(out)+ blockIdx.x * hidden_size;;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val_f4 = input_float4[idx];
    __half2 *val_h2 = (__half2 *)(&val_f4);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 val_f2 = __half22float2(val_h2[i]);
      variance += val_f2.x * val_f2.x + val_f2.y * val_f2.y;
    }
  }

  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = __frsqrt_rn(variance / (hidden_size * 8) + epsilon);
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 weight_f4 = __ldg(reinterpret_cast<const float4 *>(weight) + idx);
    __half2 *weight_h2 = reinterpret_cast<__half2 *>(&weight_f4);

    float4 val_f4 = input_float4[idx];
    __half2 *val_h2 = reinterpret_cast<__half2 *>(&val_f4);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 weight_f2 = __half22float2(weight_h2[i]);
    
      float2 val_f2 = __half22float2(val_h2[i]);
      val_f2.x = val_f2.x  * s_variance * weight_f2.x ;
      val_f2.y = val_f2.y * s_variance * weight_f2.y;
      val_h2[i] = __float22half2_rn(val_f2);
    }
    out_float4[idx] = val_f4;
  }
}



template<typename scalar_t>
void rms_norm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
  {
    dim3 grid(num_tokens);

    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if  (std::is_same<scalar_t, at::Half>::value) {
      dim3 block(min(((hidden_size / 8 + 31) / 32) * 32, 1024));
      rms_norm_kernel_impl_half<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(out),
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<const __half*>(weight),
        epsilon,
        num_tokens,
        hidden_size / 8);
    } else {
      dim3 block(std::min(hidden_size, 1024));
      rms_norm_kernel_impl_float<<<grid, block, 0, stream>>>(
        out,
        input,
        weight,
        epsilon,
        num_tokens,
        hidden_size);
    }
  }
  

} // namespace vllm

void rms_norm(
  torch::Tensor& out,      // [num_tokens, hidden_size]
  torch::Tensor& input,    // [num_tokens, hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    input.scalar_type(),
    "rms_norm_kernel",
    [&] {
       vllm::rms_norm_kernel(
          out.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          epsilon,
          input.size(0),
          input.size(1));
  }
  );
}

