// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../dispatch_utils.h"
#include "quant_conversions.cuh"
#include "../w8a8/fp8/common.cuh"

namespace vllm {

// =============================================================================
// Kernel 1: Original kernel for small num_groups (≤32)
// One block per token, processes groups sequentially
// =============================================================================

template <typename scalar_t, typename scalar_out_t, 
          bool is_scale_transposed, int32_t group_size, int32_t block_size>
__global__ void silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,       // Output: [num_tokens, hidden_size] in FP8/INT8
    float* __restrict__ scales,           // Output: [num_tokens, hidden_size / group_size]
                                          // or [hidden_size / group_size, num_tokens]
    scalar_t const* __restrict__ input,   // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,                // Optional scale upper bound
    int32_t const hidden_size             // Output hidden size (input is 2x this)
) {
    // Static assert to ensure block_size is power of 2 (required for reduction)
    static_assert((block_size & (block_size - 1)) == 0, 
                  "block_size must be a power of 2 for correct reduction");
    
    // Each thread block processes ONE token
    int const token_idx = blockIdx.x;
    int const tid = threadIdx.x;
    int const num_tokens = gridDim.x;
    
    // Input layout: [gate || up] concatenated along last dimension
    int const input_stride = hidden_size * 2;
    
    // Pointers to this token's data
    scalar_t const* token_input_gate = input + token_idx * input_stride;
    scalar_t const* token_input_up = token_input_gate + hidden_size;
    scalar_out_t* token_output = out + token_idx * hidden_size;
    
    // Scale pointer (depends on layout)
    int const num_groups = hidden_size / group_size;
    float* token_scales = is_scale_transposed 
        ? scales + token_idx
        : scales + token_idx * num_groups;
    
    // Dynamic shared memory: [shared_max (block_size floats)][shared_silu_results (group_size floats)]
    extern __shared__ char shared_mem[];
    float* shared_max = reinterpret_cast<float*>(shared_mem);
    float* shared_silu_results = shared_max + block_size;
    
    // Process elements in groups
    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
        int const group_start = group_idx * group_size;
        
        // =====================================================================
        // Step 1: Compute SiLU ONCE and find max
        // =====================================================================
        float local_max = 0.0f;
        
        for (int i = tid; i < group_size; i += blockDim.x) {
            int const elem_idx = group_start + i;
            
            // Load gate and up
            float gate = static_cast<float>(token_input_gate[elem_idx]);
            float up = static_cast<float>(token_input_up[elem_idx]);
            
            // Compute SiLU(gate) * up ONCE
            float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
            float silu_gate = gate * sigmoid_gate;
            float result = silu_gate * up;
            
            // Store in shared memory
            shared_silu_results[i] = result;
            
            // Track max absolute value for scale
            local_max = fmaxf(local_max, fabsf(result));
        }
        __syncthreads();
        
        // =====================================================================
        // Step 2: Reduce across threads to find group max
        // =====================================================================
        shared_max[tid] = local_max;
        __syncthreads();
        
        // Power-of-2 reduction (block_size guaranteed to be power of 2 by static_assert)
        #pragma unroll
        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
            }
            __syncthreads();
        }
        
        // =====================================================================
        // Step 3: Compute and store scale (thread 0 only)
        // =====================================================================
        float group_scale;
        if (tid == 0) {
            float group_max = shared_max[0];
            
            // Use vLLM's quantization constants
            float const quant_range = quant_type_max_v<scalar_out_t>;
            
            group_scale = group_max / quant_range;
            
            // Apply scale upper bound if provided
            if (scale_ub != nullptr) {
                group_scale = fminf(group_scale, *scale_ub);
            }
            
            // Use minimum safe scaling factor for the quant type
            group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());
            
            // Store scale with correct stride
            if constexpr (is_scale_transposed) {
                // Transposed layout: [num_groups, num_tokens]
                token_scales[group_idx * num_tokens] = group_scale;
            } else {
                // Normal layout: [num_tokens, num_groups]
                token_scales[group_idx] = group_scale;
            }
        }
        __syncthreads();
        
        // Broadcast scale to all threads
        if constexpr (is_scale_transposed) {
            group_scale = token_scales[group_idx * num_tokens];
        } else {
            group_scale = token_scales[group_idx];
        }
        
        // =====================================================================
        // Step 4: Quantize this group
        // =====================================================================
        for (int i = tid; i < group_size; i += blockDim.x) {
            int const elem_idx = group_start + i;
            
            // Read the SAME result we computed earlier
            float result = shared_silu_results[i];
            
            // Quantize using vLLM's helper
            token_output[elem_idx] = 
                vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result, group_scale);
        }
        __syncthreads();
    }
}

// =============================================================================
// Kernel 2: Large num_groups kernel (>32 groups)
// One block per (token, group) pair - fully parallel across groups
// =============================================================================

template <typename scalar_t, typename scalar_out_t, 
          bool is_scale_transposed, int32_t group_size>
__global__ void silu_and_mul_per_block_quant_kernel_large(
    scalar_out_t* __restrict__ out,       // Output: [num_tokens, hidden_size] in FP8/INT8
    float* __restrict__ scales,           // Output: [num_tokens, hidden_size / group_size]
                                          // or [hidden_size / group_size, num_tokens]
    scalar_t const* __restrict__ input,   // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,                // Optional scale upper bound
    int32_t const hidden_size             // Output hidden size (input is 2x this)
) {
    // Static assert: block_size == group_size, and must be power of 2 for reduction
    static_assert((group_size & (group_size - 1)) == 0, 
                  "group_size must be a power of 2 for correct reduction");
    
    // Grid: (num_tokens, num_groups)
    int const token_idx = blockIdx.x;
    int const group_idx = blockIdx.y;
    int const tid = threadIdx.x;  // tid in [0, group_size)
    int const num_tokens = gridDim.x;
    
    // Input layout: [gate || up] concatenated along last dimension
    int const input_stride = hidden_size * 2;
    int const group_start = group_idx * group_size;
    
    // Pointers to this token's data
    scalar_t const* token_input_gate = input + token_idx * input_stride + group_start;
    scalar_t const* token_input_up = token_input_gate + hidden_size;
    scalar_out_t* token_output = out + token_idx * hidden_size + group_start;
    
    // Scale pointer for this group
    int const num_groups = gridDim.y;
    float* group_scale_ptr = is_scale_transposed 
        ? scales + group_idx * num_tokens + token_idx
        : scales + token_idx * num_groups + group_idx;
    
    // Shared memory for reduction (compile-time sized)
    __shared__ float shared_max[group_size];
    
    // =========================================================================
    // Step 1: Each thread loads one element, computes SiLU, stores in register
    // =========================================================================
    float gate = static_cast<float>(token_input_gate[tid]);
    float up = static_cast<float>(token_input_up[tid]);
    
    // Compute SiLU(gate) * up
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    float silu_gate = gate * sigmoid_gate;
    float result = silu_gate * up;  // Keep in register
    
    // =========================================================================
    // Step 2: Reduce to find group max
    // =========================================================================
    shared_max[tid] = fabsf(result);
    __syncthreads();
    
    // Power-of-2 reduction (group_size guaranteed to be power of 2)
    #pragma unroll
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    
    // =========================================================================
    // Step 3: Compute scale (thread 0), broadcast via shared memory
    // =========================================================================
    if (tid == 0) {
        float group_max = shared_max[0];
        
        float const quant_range = quant_type_max_v<scalar_out_t>;
        float group_scale = group_max / quant_range;
        
        // Apply scale upper bound if provided
        if (scale_ub != nullptr) {
            group_scale = fminf(group_scale, *scale_ub);
        }
        
        // Use minimum safe scaling factor
        group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());
        
        // Store scale to global memory
        *group_scale_ptr = group_scale;
        
        // Reuse shared_max[0] to broadcast scale
        shared_max[0] = group_scale;
    }
    __syncthreads();
    
    float group_scale = shared_max[0];
    
    // =========================================================================
    // Step 4: Quantize and write output
    // =========================================================================
    token_output[tid] = vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result, group_scale);
}

}  // namespace vllm

// =============================================================================
// Dispatch function (host code)
// =============================================================================

// Threshold for switching to large kernel (num_groups > this value)
// constexpr int32_t LARGE_KERNEL_THRESHOLD = 32;
constexpr int32_t LARGE_KERNEL_WORK_THRESHOLD = 4096;

template <typename scalar_in_t>
void silu_and_mul_per_block_quant_dispatch(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int32_t group_size,
    std::optional<at::Tensor> const& scale_ub,
    bool is_scale_transposed
) {
    int32_t hidden_size = out.size(-1);
    auto num_tokens = input.size(0);
    int32_t num_groups = hidden_size / group_size;
    
    TORCH_CHECK(input.size(-1) == hidden_size * 2, 
                "input last dim must be 2x output hidden_size");
    TORCH_CHECK(hidden_size % group_size == 0,
                "hidden_size must be divisible by group_size");
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Decide which kernel to use based on num_groups
    //bool use_large_kernel = (num_groups > LARGE_KERNEL_THRESHOLD);
    bool use_large_kernel = (num_tokens * num_groups > LARGE_KERNEL_WORK_THRESHOLD);

    if (use_large_kernel) {
        // Large kernel: one block per (token, group) pair
        // Block size = group_size (64 or 128)
        dim3 grid(num_tokens, num_groups);
        dim3 block(group_size);
        
        VLLM_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_block_quant_kernel_large", [&] {
                using scalar_out_t = scalar_t;
                
                VLLM_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                    VLLM_DISPATCH_BOOL(is_scale_transposed, transpose_scale, [&] {
                        vllm::silu_and_mul_per_block_quant_kernel_large<
                            scalar_in_t, scalar_out_t, transpose_scale, gs
                        ><<<grid, block, 0, stream>>>(
                            out.data_ptr<scalar_out_t>(),
                            scales.data_ptr<float>(),
                            input.data_ptr<scalar_in_t>(),
                            scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                            hidden_size
                        );
                    });
                });
            });
    } else {
        // Original kernel: one block per token, sequential groups
        // Use power-of-2 block sizes for reduction correctness
        int block_size;
        if (hidden_size <= 256) {
            block_size = 256;
        } else if (hidden_size <= 512) {
            block_size = 512;
        } else {
            block_size = 1024;
        }
        
        dim3 grid(num_tokens);
        dim3 block(block_size);
        
        // Calculate shared memory size: shared_max[block_size] + shared_silu_results[group_size]
        size_t shared_mem_size = block_size * sizeof(float) + group_size * sizeof(float);
        
        VLLM_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_block_quant_kernel", [&] {
                using scalar_out_t = scalar_t;
                
                VLLM_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                    VLLM_DISPATCH_BOOL(is_scale_transposed, transpose_scale, [&] {
                        // Dispatch based on block_size (compile-time constant for static_assert)
                        if (block_size == 256) {
                            vllm::silu_and_mul_per_block_quant_kernel<
                                scalar_in_t, scalar_out_t, transpose_scale, gs, 256
                            ><<<grid, block, shared_mem_size, stream>>>(
                                out.data_ptr<scalar_out_t>(),
                                scales.data_ptr<float>(),
                                input.data_ptr<scalar_in_t>(),
                                scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                                hidden_size
                            );
                        } else if (block_size == 512) {
                            vllm::silu_and_mul_per_block_quant_kernel<
                                scalar_in_t, scalar_out_t, transpose_scale, gs, 512
                            ><<<grid, block, shared_mem_size, stream>>>(
                                out.data_ptr<scalar_out_t>(),
                                scales.data_ptr<float>(),
                                input.data_ptr<scalar_in_t>(),
                                scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                                hidden_size
                            );
                        } else {  // block_size == 1024
                            vllm::silu_and_mul_per_block_quant_kernel<
                                scalar_in_t, scalar_out_t, transpose_scale, gs, 1024
                            ><<<grid, block, shared_mem_size, stream>>>(
                                out.data_ptr<scalar_out_t>(),
                                scales.data_ptr<float>(),
                                input.data_ptr<scalar_in_t>(),
                                scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                                hidden_size
                            );
                        }
                    });
                });
            });
    }
}

void silu_and_mul_per_block_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int64_t group_size,
    std::optional<torch::Tensor> scale_ub,
    bool is_scale_transposed
) {
    static c10::ScalarType kFp8Type = is_fp8_ocp()
                                          ? c10::ScalarType::Float8_e4m3fn
                                          : c10::ScalarType::Float8_e4m3fnuz;
    
    TORCH_CHECK(out.dtype() == kFp8Type || out.dtype() == torch::kInt8);
    TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
    TORCH_CHECK(input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16,
                "Input must be FP16 or BF16");
    TORCH_CHECK(scales.dtype() == torch::kFloat32, "Scales must be FP32");
    TORCH_CHECK(group_size == 128 || group_size == 64,
                "Unsupported group size: ", group_size);
    
    if (scale_ub.has_value()) {
        TORCH_CHECK(out.dtype() == kFp8Type);
    }
    
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "silu_and_mul_per_block_quant_dispatch", [&] {
            silu_and_mul_per_block_quant_dispatch<scalar_t>(
                out, input, scales, group_size, scale_ub, is_scale_transposed
            );
        });
}
