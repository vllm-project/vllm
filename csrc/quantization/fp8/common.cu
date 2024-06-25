#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

#define FP8_E4M3_MAX std::numeric_limits<c10::Float8_e4m3fn>::max()

template <typename scalar_t>
__device__ __forceinline__ c10::Float8_e4m3fn scaled_fp8_conversion(
    const scalar_t val, const float inverted_scale) {
  float x = static_cast<float>(val) * inverted_scale;
  float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  return static_cast<c10::Float8_e4m3fn>(r);
}

// Compute the absolute maximum m of the input tensor and store
// m / float8_e4m3::max() in *scale. Each thread block performs a
// reduction tree and the memory in scale is atomically updated.
// So to get the right answer, *scale needs to be initialized to
// a value <= 0.0 and we need to wait for all thread blocks to
// finish before consuming *scale.
template <typename scalar_t>
__global__ void segmented_max_reduction(float* __restrict__ scale,
                                        const scalar_t* __restrict__ input,
                                        int64_t num_elems) {
  __shared__ float cache[1024];
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // First store maximum for all values processes by
  // the current thread in cache[threadIdx.x]
  scalar_t tmp = 0.0;
  while (i < num_elems) {
    float x = static_cast<float>(input[i]);
    tmp = max(tmp, fabs(x));
    i += blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = tmp;

  __syncthreads();

  // Now perform parallel reduction within the thread block
  int ib = blockDim.x / 2;
  while (ib != 0) {
    if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x]) {
      cache[threadIdx.x] = cache[threadIdx.x + ib];
    }
    __syncthreads();
    ib /= 2;
  }
  // Finally, since cache[0] contains the maximum for this thread block,
  // atomically write the max to the target location
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale,
                   cache[0] / std::numeric_limits<c10::Float8_e4m3fn>::max());
  }
}

template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

typedef struct __align__(4) {
  c10::Float8_e4m3fn x;
  c10::Float8_e4m3fn y;
  c10::Float8_e4m3fn z;
  c10::Float8_e4m3fn w;
}
float8x4_t;

template <typename scalar_t>
__global__ void scaled_fp8_quant_kernel(c10::Float8_e4m3fn* __restrict__ out,
                                        const scalar_t* __restrict__ input,
                                        const float* __restrict__ scale,
                                        int64_t num_elems) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.
  const float inverted_scale = 1.0f / (*scale);

  // Vectorized input/output to better utilize memory bandwidth.
  const vec4_t<scalar_t>* vectorized_in =
      reinterpret_cast<const vec4_t<scalar_t>*>(input);
  float8x4_t* vectorized_out = reinterpret_cast<float8x4_t*>(out);

  int num_vec_elems = num_elems >> 2;

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += blockDim.x * gridDim.x) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    float8x4_t out_vec;

    out_vec.x = scaled_fp8_conversion(in_vec.x, inverted_scale);
    out_vec.y = scaled_fp8_conversion(in_vec.y, inverted_scale);
    out_vec.z = scaled_fp8_conversion(in_vec.z, inverted_scale);
    out_vec.w = scaled_fp8_conversion(in_vec.w, inverted_scale);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int i = num_vec_elems * 4 + tid; i < num_elems;
       i += blockDim.x * gridDim.x) {
    out[i] = scaled_fp8_conversion(input[i], inverted_scale);
  }
}

__global__ void pack_fp8_to_int32_kernel(
    int32_t* __restrict__ out, const c10::Float8_e4m3fn* __restrict__ input,
    int64_t rows, int64_t cols) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int64_t num_packed_elems = (rows / 4) * cols;

  for (int64_t i = tid; i < num_packed_elems; i += stride) {
    int64_t out_row = i / cols;
    int64_t col = i % cols;

    uint32_t packed = 0;
    for (int j = 0; j < 4; ++j) {
      int64_t input_row = out_row * 4 + j;
      int64_t input_idx = input_row * cols + col;
      uint32_t fp8_bits = *reinterpret_cast<const uint8_t*>(&input[input_idx]);
      packed |= (fp8_bits << (j * 8));
    }
    out[i] = static_cast<int32_t>(packed);
  }
}

}  // namespace vllm

void static_scaled_fp8_quant(torch::Tensor& out,    // [..., d]
                             torch::Tensor& input,  // [..., d]
                             torch::Tensor& scale)  // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel", [&] {
        vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<c10::Float8_e4m3fn>(), input.data_ptr<scalar_t>(),
            scale.data_ptr<float>(), num_elems);
      });
}

void dynamic_scaled_fp8_quant(torch::Tensor& out,    // [..., d]
                              torch::Tensor& input,  // [..., d]
                              torch::Tensor& scale)  // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel", [&] {
        vllm::segmented_max_reduction<scalar_t><<<grid, block, 0, stream>>>(
            scale.data_ptr<float>(), input.data_ptr<scalar_t>(), num_elems);
        vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<c10::Float8_e4m3fn>(), input.data_ptr<scalar_t>(),
            scale.data_ptr<float>(), num_elems);
      });
}

void pack_fp8_to_int32(torch::Tensor& out,    // [d/4, ...]
                       torch::Tensor& input)  // [d, ...]
{
  TORCH_CHECK(input.scalar_type() == torch::kFloat8_e4m3fn,
              "Input tensor must be of type Float8_e4m3fn");
  TORCH_CHECK(out.scalar_type() == torch::kInt32,
              "Output tensor must be of type Int32");
  TORCH_CHECK(input.size(0) % 4 == 0,
              "First dimension of input tensor must be divisible by 4");
  TORCH_CHECK(input.size(0) / 4 == out.size(0),
              "First dimension of output tensor must be 1/4 of input tensor's "
              "first dimension");
  for (int64_t i = 1; i < input.dim(); ++i) {
    TORCH_CHECK(
        input.size(i) == out.size(i),
        "All dimensions except the first must match between input and output");
  }

  int64_t rows = input.size(0);
  int64_t cols = input.numel() / rows;

  int64_t num_threads = 1024;
  int64_t num_blocks = ((rows / 4) * cols + num_threads - 1) / num_threads;

  dim3 grid(num_blocks);
  dim3 block(num_threads);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  vllm::pack_fp8_to_int32_kernel<<<grid, block, 0, stream>>>(
      out.data_ptr<int32_t>(), input.data_ptr<c10::Float8_e4m3fn>(), rows,
      cols);
}
