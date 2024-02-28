#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "reduction_utils.cuh"
#include "attention/dtype_float16.cuh"

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void rms_norm_kernel(
  scalar_t* __restrict__ out,             // [..., hidden_size]
  const scalar_t* __restrict__ input,     // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
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

/* Helper struct to generate vectorized and packed FP16 ops
   for appropriate overloads of fused_add_rms_norm_kernel.
   Only special member functions and functions that are necessary
   in that kernel are implemented.
 */
template<int width>
struct _half2Vec {
  /* Not theoretically necessary that width is a power of 2 but should 
     almost always be the case for optimization purposes */ 
  static_assert(width > 0 && (width & (width - 1)) == 0,
                "Width is not a positive power of 2!");
  __half2 data[width];

  __device__ _half2Vec() = default;
  __device__ ~_half2Vec() = default;
  __device__ _half2Vec(const _half2Vec<width>&) = default;
  __device__ _half2Vec& operator=(const _half2Vec<width>&) = default;
  __device__ _half2Vec(_half2Vec<width>&&) = default;
  __device__ _half2Vec& operator=(_half2Vec<width>&&) = default;

  __device__ inline _half2Vec& operator+=(const _half2Vec<width>& other) {
    #pragma unroll
    for (int i = 0; i < width; ++i)
      data[i] += other.data[i];
    return *this;
  }

  __device__ inline _half2Vec& operator*=(const _half2Vec<width>& other) {
    #pragma unroll
    for (int i = 0; i < width; ++i)
      data[i] *= other.data[i];
    return *this;
  }

  __device__ inline _half2Vec& operator*=(const float scale) {
    #pragma unroll
    for (int i = 0; i < width; ++i)
      data[i] = __float22half2_rn(__half22float2(data[i]) * scale);
    return *this;
  }

  __device__ inline float sum_squares() const {
    float result = 0.0f;
    #pragma unroll
    for (int i = 0; i < width; ++i) {
      float2 z = __half22float2(data[i]);
      result += z.x * z.x + z.y * z.y;
    }
    return result; 
  }
};

/* Max blockSize to use for fused_add_rms_norm_kernel
   This kernel is memory-latency bound in many scenarios, so a smaller
   block size allows for increased block occupancy on CUs and better
   latency hiding on global mem ops. */
#define _FUSED_RMS_MAX_BLOCKSIZE    256

/* Function overload in the case of FP16 tensors.
   Additional optimizations we can make in this case are packed and
   vectorized operations, which help with the aforementioned memory
   latency bottleneck. */
template<typename scalar_t, int width>
__global__ typename std::enable_if<
  (width > 0) && std::is_same<scalar_t, c10::Half>::value,
  void>::type
fused_add_rms_norm_kernel(
  c10::Half* __restrict__ input,           // [..., hidden_size]
  c10::Half* __restrict__ residual,        // [..., hidden_size]
  const c10::Half* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size)
{
  static_assert(sizeof(_half2Vec<width>) == sizeof(c10::Half) * width * 2);
  const int vec_hidden_size = hidden_size / (width * 2);
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice */
  auto* __restrict__ input_v = reinterpret_cast<_half2Vec<width>*>(input);
  auto* __restrict__ residual_v = reinterpret_cast<_half2Vec<width>*>(residual);
  auto* __restrict__ weight_v = reinterpret_cast<const _half2Vec<width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _half2Vec<width> temp = residual_v[id];
    temp += input_v[id];
    residual_v[id] = temp;
    variance += temp.sum_squares();
  }
  variance = blockReduceSum<float, _FUSED_RMS_MAX_BLOCKSIZE>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _half2Vec<width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[id] = temp;
  }
}


/* Generic fused_add_rms_norm_kernel
   No optimizations in this case, the width field is not used
   but necessary for the correct overloading to occur in the
   FP16 case.
 */
template<typename scalar_t, int width>    // width is not used in this overload
__global__ void fused_add_rms_norm_kernel(
  scalar_t* __restrict__ input,           // [..., hidden_size]
  scalar_t* __restrict__ residual,        // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * hidden_size + idx];
    x += (float) residual[blockIdx.x * hidden_size + idx];
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = (scalar_t) x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

} // namespace vllm

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
      vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
    });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)              \
  VLLM_DISPATCH_FLOATING_TYPES(                       \
    input.scalar_type(),                              \
    "fused_add_rms_norm_kernel",                      \
    [&] {                                             \
      vllm::fused_add_rms_norm_kernel                 \
      <scalar_t, width><<<grid, block, 0, stream>>>(  \
        input.data_ptr<scalar_t>(),                   \
        residual.data_ptr<scalar_t>(),                \
        weight.data_ptr<scalar_t>(),                  \
        epsilon,                                      \
        num_tokens,                                   \
        hidden_size);                                 \
    });

void fused_add_rms_norm(
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& residual, // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, _FUSED_RMS_MAX_BLOCKSIZE));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16, try to use the optimized kernel
    with packed vectors. Max optimization is achieved with a width-4
    vector of 2-packed-FP16s (equivalent to a vector of 8 FP16s)
    since we can load at most 128 bits at once in a global memory op.
    However, we have to narrow the vectors if the hidden_size does
    not divide 8.
    
    Specifically, assuming hidden-size does not divide 8:
    If the hidden_size divides 4, we can use a width-2 packed vector
      (equivalent to a vector of 4 FP16s).
    If the hidden_size divides 2 or 6, we can use a width-1
      packed vector (equiv. to vector of 2 FP16s).
    If the hidden_size is odd, we cannot use packed vectors
      => cannot use the optimized kernel, which is signified
      by setting (packed vector) width = 0.
   */
  switch (hidden_size % 8) {
    case 0:
      LAUNCH_FUSED_ADD_RMS_NORM(4);
      break;
    case 2:
      [[fallthrough]];
    case 6:
      LAUNCH_FUSED_ADD_RMS_NORM(1);
      break;
    case 4:
      LAUNCH_FUSED_ADD_RMS_NORM(2);
      break;
    default:
      LAUNCH_FUSED_ADD_RMS_NORM(0);
      break;
  }
}
#undef _FUSED_RMS_MAX_BLOCKSIZE