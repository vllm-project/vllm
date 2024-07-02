#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#ifdef USE_ROCM
  #include "amd/quant_utils.cuh"
#else
  #include "nvidia/quant_utils.cuh"
#endif

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

template <typename Tout, typename Tin, int Vec_size>
__global__ void convert_fp8_kernel(const Tin* __restrict__ src_data,
                                   Tout* __restrict__ dst_data,
                                   const float* scale, size_t N) {
  const int64_t block_idx = blockIdx.x;

  using V_in_vec = typename Vec<Tin, Vec_size>::Type;
  using V_out_vec = typename Vec<Tout, Vec_size>::Type;
  auto dst_data_vec = reinterpret_cast<V_out_vec*>(dst_data);
  auto src_data_vec = reinterpret_cast<const V_in_vec*>(src_data);

  int64_t startIdx = (threadIdx.x + blockDim.x * blockIdx.x);
  auto idx = startIdx;
  if (idx >= N) {
    return;
  }
#if defined(ENABLE_FP8) && defined(USE_ROCM)
  dst_data_vec[idx] = fp8::scaled_vec_conversion<V_out_vec, V_in_vec>(
      src_data_vec[idx], *scale);
#else
  assert(false);
#endif
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

template <typename Tout, typename Tin, int Vec_size>
struct call_convert_fp8 {
  void operator()(torch::Tensor const& src_data, torch::Tensor& dst_data,
                  torch::Tensor const& scale) {
    const auto N = src_data.numel() / Vec_size;
    constexpr dim3 numThreads{1024, 1, 1};
    uint32_t numBlocks = (N + numThreads.x - 1) / numThreads.x;
    const dim3 grid{numBlocks, 1, 1};
    const auto stream = at::cuda::getCurrentCUDAStream();

    vllm::convert_fp8_kernel<Tout, Tin, Vec_size>
        <<<grid, numThreads, 0, stream>>>(
            reinterpret_cast<Tin*>(src_data.data_ptr()),
            reinterpret_cast<Tout*>(dst_data.data_ptr()),
            (float*)scale.data_ptr(), N);
  }
};

void convert_fp8(torch::Tensor& dst_data, torch::Tensor const& src_data,
                 torch::Tensor const& scale) {
  torch::Device src_device = src_data.device();
  torch::Device dst_device = dst_data.device();
  TORCH_CHECK(src_device.is_cuda(), "src must be on a GPU")
  TORCH_CHECK(dst_device.is_cuda(), "dst must be on a GPU")
  TORCH_CHECK(src_device.index() == dst_device.index(),
              "src and dst must be on the same GPU");
  at::cuda::OptionalCUDAGuard device_guard(src_device);
  auto t1 = src_data.dtype();
  auto t2 = dst_data.dtype();
  if (src_data.dtype() == at::ScalarType::Float) {
    call_convert_fp8<uint8_t, float, 2>{}(src_data, dst_data, scale);
  } else if (src_data.dtype() == at::ScalarType::Half) {
    call_convert_fp8<uint8_t, uint16_t, 2>{}(src_data, dst_data, scale);
  } else if (src_data.dtype() == at::ScalarType::BFloat16) {
    call_convert_fp8<uint8_t, __nv_bfloat16, 2>{}(src_data, dst_data, scale);
  } else if (dst_data.dtype() == at::ScalarType::Float) {
    call_convert_fp8<float, uint8_t, 2>{}(src_data, dst_data, scale);
  } else if (dst_data.dtype() == at::ScalarType::Half) {
    call_convert_fp8<uint16_t, uint8_t, 2>{}(src_data, dst_data, scale);
  } else if (dst_data.dtype() == at::ScalarType::BFloat16) {
    call_convert_fp8<__nv_bfloat16, uint8_t, 2>{}(src_data, dst_data, scale);
  }
}