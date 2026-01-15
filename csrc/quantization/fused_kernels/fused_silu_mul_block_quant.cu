// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../dispatch_utils.h"
#include "quant_conversions.cuh"

namespace vllm {

// =============================================================================
// SiLU+Mul+Block Quantization Kernel (Vectorized)
// =============================================================================

template <typename scalar_t, typename scalar_out_t, 
          bool is_scale_transposed = false, int32_t group_size = 0>
__global__ void silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,       // Output: [num_tokens, hidden_size] in FP8/INT8
    float* __restrict__ scales,           // Output: [num_tokens, hidden_size / group_size]
                                          // or [hidden_size / group_size, num_tokens]
    scalar_t const* __restrict__ input,   // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,                // Optional scale upper bound
    int32_t const hidden_size             // Output hidden size (input is 2x this)
) {
    // Each thread block processes ONE token
    int const token_idx = blockIdx.x;
    int const tid = threadIdx.x;
    int const num_tokens = gridDim.x;
    
    // Input layout: [gate || up] concatenated along last dimension
    // gate: input[token_idx, 0:hidden_size]
    // up:   input[token_idx, hidden_size:2*hidden_size]
    int const input_stride = hidden_size * 2;
    
    // Pointers to this token's data
    scalar_t const* token_input_gate = input + token_idx * input_stride;
    scalar_t const* token_input_up = token_input_gate + hidden_size;
    scalar_out_t* token_output = out + token_idx * hidden_size;
    
    // Scale pointer (depends on layout)
    int const num_groups = hidden_size / group_size;
    float* token_scales = is_scale_transposed 
        ? scales + token_idx  // Column-major: jump by 1, stride by num_tokens
        : scales + token_idx * num_groups;  // Row-major: contiguous
    int const scale_stride = is_scale_transposed ? num_tokens : 1;
    
    // Vectorization setup: process 4 elements at once for FP16
    constexpr int VEC_SIZE = 4;
    int const vec_elems = (group_size / VEC_SIZE) * VEC_SIZE;  // Aligned portion
    
    // Process elements in groups
    // Each thread processes multiple elements across groups
    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
        int const group_start = group_idx * group_size;
        
        // =====================================================================
        // Step 1: Compute max for this group (for scale computation)
        // =====================================================================
        
        float local_max = 0.0f;
        
        // Vectorized loop for FP16: Load and process 4 elements at once
        if constexpr (std::is_same_v<scalar_t, c10::Half>) {
            // Define half4 for vectorized loads (8-byte aligned)
            using half4 = struct __align__(8) { c10::Half x, y, z, w; };
            
            for (int i = tid * VEC_SIZE; i < vec_elems; i += blockDim.x * VEC_SIZE) {
                int const elem_idx = group_start + i;
                
                // Load 4 halfs at once (coalesced memory access)
                half4 gate4 = *reinterpret_cast<const half4*>(token_input_gate + elem_idx);
                half4 up4 = *reinterpret_cast<const half4*>(token_input_up + elem_idx);
                
                // Process all 4 elements
                #pragma unroll
                for (int j = 0; j < VEC_SIZE; j++) {
                    float gate = static_cast<float>((&gate4.x)[j]);
                    float up = static_cast<float>((&up4.x)[j]);
                    
                    // Compute SiLU(gate) * up
                    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
                    float silu_gate = gate * sigmoid_gate;
                    float result = silu_gate * up;
                    
                    // Track max absolute value for scale
                    local_max = fmaxf(local_max, fabsf(result));
                }
            }
        } else {
            // BFloat16 path: scalar loop (can be optimized later)
            for (int i = tid; i < group_size; i += blockDim.x) {
                int const elem_idx = group_start + i;
                
                // Load gate and up
                float gate = static_cast<float>(token_input_gate[elem_idx]);
                float up = static_cast<float>(token_input_up[elem_idx]);
                
                // Compute SiLU(gate) * up
                // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
                float silu_gate = gate * sigmoid_gate;
                float result = silu_gate * up;
                
                // Track max absolute value for scale
                local_max = fmaxf(local_max, fabsf(result));
            }
        }
        
        // Scalar loop for remainder elements (when group_size not divisible by 4)
        if constexpr (std::is_same_v<scalar_t, c10::Half>) {
            for (int i = vec_elems + tid; i < group_size; i += blockDim.x) {
                int const elem_idx = group_start + i;
                
                // Load gate and up
                float gate = static_cast<float>(token_input_gate[elem_idx]);
                float up = static_cast<float>(token_input_up[elem_idx]);
                
                // Compute SiLU(gate) * up
                // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
                float silu_gate = gate * sigmoid_gate;
                float result = silu_gate * up;
                
                // Track max absolute value for scale
                local_max = fmaxf(local_max, fabsf(result));
            }
        }
        
        // =====================================================================
        // Step 2: Reduce across threads to find group max
        // =====================================================================
        
        // Use shared memory for reduction
        __shared__ float shared_max[1024];  // Assuming max 1024 threads per block
        shared_max[tid] = local_max;
        __syncthreads();
        
        // Parallel reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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
            
            // Compute scale based on output type
            if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
                // INT8: range is [-127, 127]
                group_scale = group_max / 127.0f;
            } else {
                // FP8: range is approximately [-448, 448] for E4M3
                group_scale = group_max / 448.0f;
            }
            
            // Apply scale upper bound if provided
            if (scale_ub != nullptr) {
                group_scale = fminf(group_scale, *scale_ub);
            }
            
            // Avoid division by zero
            group_scale = fmaxf(group_scale, 1e-10f);
            
            // Store scale
            if constexpr (is_scale_transposed) {
                token_scales[group_idx * scale_stride] = group_scale;
            } else {
                token_scales[group_idx] = group_scale;
            }
        }
        __syncthreads();
        
        // Broadcast scale to all threads
        if constexpr (is_scale_transposed) {
            group_scale = token_scales[group_idx * scale_stride];
        } else {
            group_scale = token_scales[group_idx];
        }
        
        // =====================================================================
        // Step 4: Quantize this group
        // =====================================================================
        
        // Vectorized quantization for FP16
        if constexpr (std::is_same_v<scalar_t, c10::Half>) {
            using half4 = struct __align__(8) { c10::Half x, y, z, w; };
            
            for (int i = tid * VEC_SIZE; i < vec_elems; i += blockDim.x * VEC_SIZE) {
                int const elem_idx = group_start + i;
                
                // Load 4 halfs at once
                half4 gate4 = *reinterpret_cast<const half4*>(token_input_gate + elem_idx);
                half4 up4 = *reinterpret_cast<const half4*>(token_input_up + elem_idx);
                
                // Process and quantize all 4 elements
                #pragma unroll
                for (int j = 0; j < VEC_SIZE; j++) {
                    float gate = static_cast<float>((&gate4.x)[j]);
                    float up = static_cast<float>((&up4.x)[j]);
                    
                    // Compute SiLU(gate) * up
                    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
                    float silu_gate = gate * sigmoid_gate;
                    float result = silu_gate * up;
                    
                    // Quantize
                    if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
                        // INT8 quantization
                        float scaled = result / group_scale;
                        int32_t quantized = static_cast<int32_t>(roundf(scaled));
                        quantized = max(-127, min(127, quantized));
                        token_output[elem_idx + j] = static_cast<int8_t>(quantized);
                    } else {
                        // FP8 quantization
                        float scaled = result / group_scale;
                        token_output[elem_idx + j] = static_cast<scalar_out_t>(scaled);
                    }
                }
            }
        } else {
            // BFloat16 path: scalar loop
            for (int i = tid; i < group_size; i += blockDim.x) {
                int const elem_idx = group_start + i;
                
                // Load gate and up
                float gate = static_cast<float>(token_input_gate[elem_idx]);
                float up = static_cast<float>(token_input_up[elem_idx]);
                
                // Compute SiLU(gate) * up
                float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
                float silu_gate = gate * sigmoid_gate;
                float result = silu_gate * up;
                
                // Quantize
                if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
                    // INT8 quantization
                    float scaled = result / group_scale;
                    int32_t quantized = static_cast<int32_t>(roundf(scaled));
                    quantized = max(-127, min(127, quantized));
                    token_output[elem_idx] = static_cast<int8_t>(quantized);
                } else {
                    // FP8 quantization
                    float scaled = result / group_scale;
                    token_output[elem_idx] = static_cast<scalar_out_t>(scaled);
                }
            }
        }
        
        // Scalar loop for remainder elements
        if constexpr (std::is_same_v<scalar_t, c10::Half>) {
            for (int i = vec_elems + tid; i < group_size; i += blockDim.x) {
                int const elem_idx = group_start + i;
                
                // Load gate and up
                float gate = static_cast<float>(token_input_gate[elem_idx]);
                float up = static_cast<float>(token_input_up[elem_idx]);
                
                // Compute SiLU(gate) * up
                float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
                float silu_gate = gate * sigmoid_gate;
                float result = silu_gate * up;
                
                // Quantize
                if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
                    // INT8 quantization
                    float scaled = result / group_scale;
                    int32_t quantized = static_cast<int32_t>(roundf(scaled));
                    quantized = max(-127, min(127, quantized));
                    token_output[elem_idx] = static_cast<int8_t>(quantized);
                } else {
                    // FP8 quantization
                    float scaled = result / group_scale;
                    token_output[elem_idx] = static_cast<scalar_out_t>(scaled);
                }
            }
        }
        
        __syncthreads();
    }
}

// =============================================================================
// Dispatch function (host code)
// =============================================================================

template <typename scalar_in_t>
void silu_and_mul_per_block_quant_dispatch(
    torch::Tensor& out,           // [num_tokens, hidden_size]
    torch::Tensor const& input,   // [num_tokens, hidden_size * 2]
    torch::Tensor& scales,        // [num_tokens, hidden_size / group_size]
    int32_t group_size,
    std::optional<at::Tensor> const& scale_ub,
    bool is_scale_transposed
) {
    int32_t hidden_size = out.size(-1);  // Output hidden size
    auto num_tokens = input.size(0);
    
    // Validate dimensions
    TORCH_CHECK(input.size(-1) == hidden_size * 2, 
                "input last dim must be 2x output hidden_size");
    TORCH_CHECK(hidden_size % group_size == 0,
                "hidden_size must be divisible by group_size");
    
    // Launch configuration
    dim3 grid(num_tokens);
    const int max_block_size = (num_tokens <= 256) ? 512 : 256;
    dim3 block(std::min(hidden_size, max_block_size));
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Dispatch based on group size and scale layout
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "silu_and_mul_per_block_quant_kernel", [&] {
            using scalar_out_t = scalar_t;
            
            VLLM_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                VLLM_DISPATCH_BOOL(is_scale_transposed, transpose_scale, [&] {
                    vllm::silu_and_mul_per_block_quant_kernel<
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
}

}  // namespace vllm

// =============================================================================
// Main entry point (called from Python)
// =============================================================================

void silu_and_mul_per_block_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int64_t group_size,
    std::optional<torch::Tensor> scale_ub,
    bool is_scale_transposed
) {
    TORCH_CHECK(out.is_contiguous() && input.is_contiguous(),
                "Tensors must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16,
                "Input must be FP16 or BF16");
    TORCH_CHECK(scales.dtype() == torch::kFloat32,
                "Scales must be FP32");
    TORCH_CHECK(group_size == 128 || group_size == 64,
                "Unsupported group size: ", group_size, " (only 64 and 128 supported)");
    
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "silu_and_mul_per_block_quant_dispatch", [&] {
            vllm::silu_and_mul_per_block_quant_dispatch<scalar_t>(
                out, input, scales, group_size, scale_ub, is_scale_transposed
            );
        });
}