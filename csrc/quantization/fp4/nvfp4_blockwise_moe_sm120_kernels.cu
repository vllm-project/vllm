/*
 * Minimal NVFP4 MoE Kernel for SM120 - Reference Implementation
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

// NVFP4 E2M1 dequantization lookup table
__device__ __forceinline__ float dequantize_nvfp4_e2m1(uint8_t fp4_val) {
    static const float e2m1_table[16] = {
        0.0f,   0.5f,   1.0f,   1.5f,
        2.0f,   3.0f,   4.0f,   6.0f,
        -0.0f,  -0.5f,  -1.0f,  -1.5f,
        -2.0f,  -3.0f,  -4.0f,  -6.0f
    };
    return e2m1_table[fp4_val & 0xF];
}

// E4M3 scale factor dequantization
__device__ __forceinline__ float dequantize_e4m3_scale(uint8_t e4m3_val) {
    if (e4m3_val == 0) return 1.0f;

    uint32_t sign = (e4m3_val >> 7) & 0x1;
    uint32_t exp = (e4m3_val >> 3) & 0xF;
    uint32_t mantissa = e4m3_val & 0x7;

    if (exp == 0xF) return sign ? -448.0f : 448.0f;

    float value = (exp == 0)
        ? ldexpf(mantissa / 8.0f, -6)
        : ldexpf(1.0f + mantissa / 8.0f, exp - 7);

    return sign ? -value : value;
}

// NVFP4 MoE kernel
template<typename OutType>
__global__ void nvfp4_moe_kernel(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    OutType* __restrict__ output,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ b_scales,
    const float* __restrict__ alphas,
    const int32_t* __restrict__ problem_sizes,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ sf_offsets,
    int M, int N, int K, int num_experts) {

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid_y >= M || tid_x >= N) return;

    // Find expert
    int expert_id = -1;
    for (int e = 0; e < num_experts; e++) {
        int start = expert_offsets[e];
        int end = (e == num_experts - 1) ? M : expert_offsets[e + 1];
        if (tid_y >= start && tid_y < end) {
            expert_id = e;
            break;
        }
    }

    if (expert_id < 0) return;

    // Compute dot product in 16-element blocks
    float sum = 0.0f;
    int k_packed = K / 2;
    int k_blocks = K / 16;

    for (int block_idx = 0; block_idx < k_blocks; block_idx++) {
        // Get scales (simplified indexing - complex swizzling logic preserved for correctness)
        float a_scale = 1.0f, b_scale = 1.0f;

        if (a_scales) {
            int local_row = tid_y - expert_offsets[expert_id];
            int global_row = sf_offsets[expert_id] + local_row;
            int k_tiles = (K + 63) / 64;
            int m_tile = global_row / 128;
            int row128 = global_row % 128;
            long long idx = (((((long long)m_tile * k_tiles + block_idx/4) * 32 + row128%32) * 4 + row128/32) * 4 + block_idx%4);
            a_scale = dequantize_e4m3_scale(a_scales[idx]);
        }

        if (b_scales) {
            int k_tiles = (K + 63) / 64;
            int n_tiles = (N + 127) / 128;
            long long base = (long long)expert_id * k_tiles * n_tiles * 512;
            int m_tile = tid_x / 128;
            int row128 = tid_x % 128;
            long long idx = base + (((((long long)m_tile * k_tiles + block_idx/4) * 32 + row128%32) * 4 + row128/32) * 4 + block_idx%4);
            b_scale = dequantize_e4m3_scale(b_scales[idx]);
        }

        // Process block
        float block_sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            int k = block_idx * 8 + i;
            if (k >= k_packed) break;

            uint8_t a_packed = a[tid_y * k_packed + k];
            uint8_t b_packed = b[((long long)expert_id * N + tid_x) * k_packed + k];

            float a0 = dequantize_nvfp4_e2m1(a_packed & 0x0F);
            float a1 = dequantize_nvfp4_e2m1(a_packed >> 4);
            float b0 = dequantize_nvfp4_e2m1(b_packed & 0x0F);
            float b1 = dequantize_nvfp4_e2m1(b_packed >> 4);

            block_sum += a0 * b0 + a1 * b1;
        }

        sum += block_sum * a_scale * b_scale;
    }

    if (alphas) sum *= alphas[expert_id];
    output[tid_y * N + tid_x] = static_cast<OutType>(sum);
}

// Main entry point
void cutlass_fp4_group_mm_sm120(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_blockscale,
    const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& sf_offsets) {

    // Basic validation
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == at::ScalarType::Byte);
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == at::ScalarType::Byte);
    TORCH_CHECK(output.is_cuda() && output.is_contiguous());

    // Check SM version
    int32_t major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    TORCH_CHECK(major * 10 + minor >= 120, "Requires SM120+");

    // Get dimensions
    int M = a.size(0);
    int N = b.size(1);
    int K = 2 * b.size(2);  // Unpacked K
    int num_experts = expert_offsets.size(0);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    auto stream = at::cuda::getCurrentCUDAStream(a.device().index());

    // Helper macro to avoid repetition
#define LAUNCH_KERNEL(TYPE) \
    nvfp4_moe_kernel<TYPE><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const uint8_t*>(a.data_ptr()), \
        reinterpret_cast<const uint8_t*>(b.data_ptr()), \
        reinterpret_cast<TYPE*>(output.data_ptr()), \
        reinterpret_cast<const uint8_t*>(a_blockscale.data_ptr()), \
        reinterpret_cast<const uint8_t*>(b_blockscales.data_ptr()), \
        reinterpret_cast<const float*>(alphas.data_ptr()), \
        reinterpret_cast<const int32_t*>(problem_sizes.data_ptr()), \
        reinterpret_cast<const int32_t*>(expert_offsets.data_ptr()), \
        reinterpret_cast<const int32_t*>(sf_offsets.data_ptr()), \
        M, N, K, num_experts)

    if (output.scalar_type() == torch::kBFloat16) {
        LAUNCH_KERNEL(at::BFloat16);
    } else if (output.scalar_type() == torch::kHalf) {
        LAUNCH_KERNEL(at::Half);
    } else {
        LAUNCH_KERNEL(float);
    }
#undef LAUNCH_KERNEL
}