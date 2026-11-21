// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Hardware-converted NVFP4 dequant+add for the host-staged allreduce wire.
// Each e2m1 payload byte unpacks with the sm_120a-only F2FP hardware
// instruction (cvt.rn.f16x2.e2m1x2) -- one op per two elements where the
// Triton comparison-chain version costs ~15 ops per element and measures
// 0.180 ms/op vs 0.014 ms/op for the E4M3 dequant at 4096x5120. ptxas
// rejects the cvt on plain sm_120, so this extension builds for sm_120a
// (see _load_fp4_cuda in fp8_host_staged_all_reduce.py).

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kBlock = 1024;  // elements per block (KERNEL_BLOCK)
constexpr int kGroup = 16;   // elements per scale group (NVFP4_SCALE_BLOCK)
constexpr int kThreads = 256;  // 4 threads per group, 64 groups per block

__device__ __forceinline__ float e4m3_to_f32(unsigned char byte) {
    unsigned r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
        : "=r"(r)
        : "h"((unsigned short)byte));
    __half h = *reinterpret_cast<const __half*>(&r);  // low half
    return __half2float(h);
}

__device__ __forceinline__ unsigned e2m1x2_to_f16x2(unsigned char byte) {
    unsigned r;
    asm("{\n .reg .b8 t;\n"
        " cvt.u8.u16 t, %1;\n"
        " cvt.rn.f16x2.e2m1x2 %0, t;\n}"
        : "=r"(r)
        : "h"((unsigned short)byte));
    return r;
}

__global__ void dequant_add_nvfp4_kernel(
    const unsigned char* __restrict__ p0,
    const unsigned char* __restrict__ s0,
    const unsigned char* __restrict__ p1,
    const unsigned char* __restrict__ s1,
    __nv_bfloat16* __restrict__ out) {
    const int g = threadIdx.x >> 2;   // scale group within the block
    const int l = threadIdx.x & 3;    // 4-element lane within the group
    const size_t bb =
        (size_t)blockIdx.x * (kBlock / 2) + g * (kGroup / 2) + l * 2;
    const unsigned short y0 = *reinterpret_cast<const unsigned short*>(p0 + bb);
    const unsigned short y1 = *reinterpret_cast<const unsigned short*>(p1 + bb);
    const float f0 = e4m3_to_f32(s0[(size_t)blockIdx.x * (kBlock / kGroup) + g]);
    const float f1 = e4m3_to_f32(s1[(size_t)blockIdx.x * (kBlock / kGroup) + g]);

    float v0[4], v1[4];
    unsigned r;
    r = e2m1x2_to_f16x2((unsigned char)(y0 & 0xFF));
    __half2 h = *reinterpret_cast<const __half2*>(&r);
    v0[0] = __low2float(h) * f0;
    v0[1] = __high2float(h) * f0;
    r = e2m1x2_to_f16x2((unsigned char)(y0 >> 8));
    h = *reinterpret_cast<const __half2*>(&r);
    v0[2] = __low2float(h) * f0;
    v0[3] = __high2float(h) * f0;
    r = e2m1x2_to_f16x2((unsigned char)(y1 & 0xFF));
    h = *reinterpret_cast<const __half2*>(&r);
    v1[0] = __low2float(h) * f1;
    v1[1] = __high2float(h) * f1;
    r = e2m1x2_to_f16x2((unsigned char)(y1 >> 8));
    h = *reinterpret_cast<const __half2*>(&r);
    v1[2] = __low2float(h) * f1;
    v1[3] = __high2float(h) * f1;

    __nv_bfloat16 o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // Same three roundings as the Triton kernel: each side is
        // BF16-rounded before the FP32 add (cross-rank commutativity
        // protection), the sum is BF16-rounded once.
        o[i] = __float2bfloat16_rn(
            __bfloat162float(__float2bfloat16_rn(v0[i])) +
            __bfloat162float(__float2bfloat16_rn(v1[i])));
    }
    *reinterpret_cast<uint2*>(
        out + (size_t)blockIdx.x * kBlock + g * kGroup + l * 4) =
        *reinterpret_cast<const uint2*>(o);
}

void dequant_add_nvfp4(const torch::Tensor& p0, const torch::Tensor& s0,
                       const torch::Tensor& p1, const torch::Tensor& s1,
                       torch::Tensor& out) {
    const int64_t n = 2 * p0.numel();
    TORCH_CHECK(n % kBlock == 0, "n must be a multiple of ", kBlock);
    TORCH_CHECK(s0.numel() == n / kGroup && s1.numel() == n / kGroup,
                "scale numel must be n/", kGroup);
    TORCH_CHECK(out.numel() == n, "out numel must be n");
    const auto stream = at::cuda::getCurrentCUDAStream();
    dequant_add_nvfp4_kernel<<<n / kBlock, kThreads, 0, stream>>>(
        (const unsigned char*)p0.data_ptr(),
        (const unsigned char*)s0.data_ptr(),
        (const unsigned char*)p1.data_ptr(),
        (const unsigned char*)s1.data_ptr(),
        (__nv_bfloat16*)out.data_ptr());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_add_nvfp4", &dequant_add_nvfp4);
}
