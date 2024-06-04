#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../attention/attention_dtypes.h"
#if defined(ENABLE_FP8_E5M2)
#include "../fp8_e5m2_kvcache/quant_utils.cuh"
#elif defined(ENABLE_FP8_E4M3)
#include "amd_detail/quant_utils.cuh"
#endif

namespace vllm
{

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
    dst_data_vec[idx] = fp8_e4m3::scaled_vec_conversion<V_out_vec, V_in_vec>(src_data_vec[idx], *scale);
    //dst_data_vec[idx+1] = fp8_e4m3::vec_conversion<V_out_vec, V_in_vec, Scaled>(src_data_vec[idx+1], *scale);

    //for (int64_t i = 0; i < loopSize; ++i) {
    //    auto idx = startIdx + i;
    //    if (idx >= N) {
    //        return;
    //    }
    //    dst_data_vec[idx] = fp8_e4m3::vec_conversion<V_out_vec, V_in_vec, Scaled>(src_data_vec[idx], *scale);
    //}
}

} // namespace vllm

template <typename Tout, typename Tin, int Vec_size>
struct call_convert_fp8
{
    void operator()(torch::Tensor& src_data, torch::Tensor& dst_data, torch::Tensor& scale)
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

void convert_fp8(torch::Tensor& src_data, torch::Tensor& dst_data, torch::Tensor& scale)
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