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
  
template <typename Tout, typename Tin, int Vec_size, bool Scaled>
__global__ void convert_fp8_kernel(
    const Tin* __restrict__ src_data, Tout* __restrict__ dst_data, const float* scale, size_t N)
{
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
    dst_data_vec[idx] = fp8::scaled_vec_conversion<V_out_vec, V_in_vec>(src_data_vec[idx], *scale);
    //dst_data_vec[idx+1] = fp8_e4m3::vec_conversion<V_out_vec, V_in_vec, Scaled>(src_data_vec[idx+1], *scale);

    //for (int64_t i = 0; i < loopSize; ++i) {
    //    auto idx = startIdx + i;
    //    if (idx >= N) {
    //        return;
    //    }
    //    dst_data_vec[idx] = fp8_e4m3::vec_conversion<V_out_vec, V_in_vec, Scaled>(src_data_vec[idx], *scale);
    //}
}

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
    const scalar_t val, const float scale) {
  float x = static_cast<float>(val) / scale;
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
__global__ void scaled_fp8_quant_kernel(c10::Float8_e4m3fn* __restrict__ out,
                                        const scalar_t* __restrict__ input,
                                        const float* __restrict__ scale,
                                        int64_t num_elems) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  while (i < num_elems) {
    out[i] = scaled_fp8_conversion(input[i], *scale);
    i += blockDim.x * gridDim.x;
  }
}

} // namespace vllm

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
struct call_convert_fp8
{
    void operator()(torch::Tensor const& src_data, torch::Tensor& dst_data, torch::Tensor const& scale)
    {
        const auto N = src_data.numel() / 2;
        //std::cout << N << "\n";
        constexpr uint32_t loopSize = 1;//std::max(N / 50000000LL, 1);
        constexpr dim3 numThreads{1024, 1, 1};
        auto neededBlocks = (N + (numThreads.x * loopSize) - 1) / (numThreads.x * loopSize);
        uint32_t actualBlocks = neededBlocks;

        //static uint32_t maxBlocks = 0;
        //if (actualBlocks != maxBlocks) {
        //  maxBlocks = actualBlocks;
        //  std::cout << actualBlocks << "\n";
        //}

        const dim3 grid{actualBlocks, 1, 1};

        const auto stream = at::cuda::getCurrentCUDAStream();

        vllm::convert_fp8_kernel<Tout, Tin, Vec_size, true>
            <<<grid, numThreads, 0, stream>>>(reinterpret_cast<Tin*>(src_data.data_ptr()),
                reinterpret_cast<Tout*>(dst_data.data_ptr()), (float*)scale.data_ptr(), N);
    }
};

void convert_fp8(torch::Tensor& dst_data, torch::Tensor const& src_data, torch::Tensor const& scale)
{
    torch::Device src_device = src_data.device();
    torch::Device dst_device = dst_data.device();
    TORCH_CHECK(src_device.is_cuda(), "src must be on a GPU")
    TORCH_CHECK(dst_device.is_cuda(), "dst must be on a GPU")
    TORCH_CHECK(src_device.index() == dst_device.index(), "src and dst must be on the same GPU");
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