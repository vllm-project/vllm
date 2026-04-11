// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// CUDA-native GDN (Gated DeltaNet) single-token decode kernel.
// Adapted from the flash-moe project's gated_delta_net_step kernel.
// Provides a Triton-free decode path for Blackwell and other GPUs
// where Triton autotuning may fail or be slow on first run.

#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

namespace vllm {

// Warp-level reduction using __shfl_down_sync
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction: warp reduces, then lane 0 of each warp
// writes to shared memory, and final warp reduces those.
template <int BK>
__device__ float block_reduce_sum(float val, float* smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    constexpr int NUM_WARPS = BK / 32;

    val = warp_reduce_sum(val);

    if (lane_id == 0)
        smem[warp_id] = val;
    __syncthreads();

    // First warp reduces the warp sums
    val = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : 0.0f;
    if (warp_id == 0)
        val = warp_reduce_sum(val);

    // Broadcast result to all threads
    if (threadIdx.x == 0)
        smem[0] = val;
    __syncthreads();
    return smem[0];
}

// Single-token GDN decode step for one (v_head, batch) pair.
// Grid: (num_v_heads, batch_size)
// Block: (BK) threads — each thread handles one k-dimension element.
//
// The state matrix h is [V, K] per v-head. Each thread owns column k_idx
// of h and iterates over all V rows.
template <int BK>
__global__ void gdn_decode_step_kernel(
    const float* __restrict__ q,        // [B, H, K] (H = num_k_heads)
    const float* __restrict__ k,        // [B, H, K]
    const float* __restrict__ v,        // [B, HV, V] (HV = num_v_heads)
    const float* __restrict__ g_decay,  // [B, HV] (pre-computed exp(g))
    const float* __restrict__ beta,     // [B, HV] (pre-computed sigmoid(b))
    float* __restrict__ state,          // [num_slots, HV, V, K]
    float* __restrict__ output,         // [B, HV, V]
    const int32_t* __restrict__ state_indices, // [B]
    int num_k_heads,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    float scale,
    bool use_l2norm
) {
    const int hv_idx = blockIdx.x;  // which v-head
    const int b_idx = blockIdx.y;   // which sequence in batch
    const int k_idx = threadIdx.x;  // which k-dimension element

    // All threads must participate in __syncthreads, so use a mask
    // instead of early return for out-of-range threads
    const bool active = (k_idx < head_k_dim);

    // Check state index validity (PAD_SLOT_ID = -1)
    const int slot = state_indices[b_idx];
    if (slot < 0) return;  // safe: all threads return together per-block

    // Map v-head to k-head (GQA: multiple v-heads share one k-head)
    const int h_idx = hv_idx / (num_v_heads / num_k_heads);

    // Load per-head scalars
    const float g = g_decay[b_idx * num_v_heads + hv_idx];
    const float b = beta[b_idx * num_v_heads + hv_idx];

    // Load q[k_idx] and k[k_idx] for this head
    float q_val = active ? q[(b_idx * num_k_heads + h_idx) * head_k_dim + k_idx] : 0.0f;
    float k_val = active ? k[(b_idx * num_k_heads + h_idx) * head_k_dim + k_idx] : 0.0f;

    // Optional L2 norm on q and k
    __shared__ float smem[BK / 32 + 1];  // for warp-level reduction
    if (use_l2norm) {
        float q_norm = block_reduce_sum<BK>(q_val * q_val, smem);
        q_val *= rsqrtf(q_norm + 1e-6f);

        float k_norm = block_reduce_sum<BK>(k_val * k_val, smem);
        k_val *= rsqrtf(k_norm + 1e-6f);
    }

    q_val *= scale;

    // State base pointer for this (slot, v-head)
    // state layout: [num_slots, HV, V, K]
    float* h_base = state + ((int64_t)slot * num_v_heads + hv_idx)
                    * head_v_dim * head_k_dim;

    // v base pointer for this (batch, v-head)
    const float* v_base = v + (b_idx * num_v_heads + hv_idx) * head_v_dim;

    // output base pointer
    float* o_base = output + (b_idx * num_v_heads + hv_idx) * head_v_dim;

    // Process each v-dimension element
    for (int vi = 0; vi < head_v_dim; vi++) {
        float h_val = active ? h_base[vi * head_k_dim + k_idx] : 0.0f;

        // Decay
        h_val *= g;

        // kv_mem = sum_k(h[vi, k] * k[k]) — warp-level reduction
        float kv_mem = block_reduce_sum<BK>(h_val * k_val, smem);

        // Delta update
        float v_val = v_base[vi];
        float delta = (v_val - kv_mem) * b;

        // Update state
        h_val += k_val * delta;
        if (active)
            h_base[vi * head_k_dim + k_idx] = h_val;

        // output = sum_k(h[vi, k] * q[k]) — warp-level reduction
        float out_val = block_reduce_sum<BK>(h_val * q_val, smem);
        if (threadIdx.x == 0)
            o_base[vi] = out_val;
    }
}

void gdn_decode_step(
    torch::Tensor& q,              // [B, H, K]
    torch::Tensor& k,              // [B, H, K]
    torch::Tensor& v,              // [B, HV, V]
    torch::Tensor& g_decay,        // [B, HV] (pre-computed: exp(-A_log * softplus(a + dt_bias)))
    torch::Tensor& beta,           // [B, HV] (pre-computed: sigmoid(b))
    torch::Tensor& state,          // [num_slots, HV, V, K]
    torch::Tensor& output,         // [B, HV, V]
    torch::Tensor& state_indices,  // [B] int32
    float scale,
    bool use_l2norm
) {
    const int batch_size = q.size(0);
    const int num_k_heads = q.size(1);
    const int head_k_dim = q.size(2);
    const int num_v_heads = v.size(1);
    const int head_v_dim = v.size(2);

    TORCH_CHECK(head_k_dim <= 256, "head_k_dim must be <= 256");

    // All computation in fp32 — convert inputs if needed
    auto q_f = q.to(torch::kFloat32).contiguous();
    auto k_f = k.to(torch::kFloat32).contiguous();
    auto v_f = v.to(torch::kFloat32).contiguous();
    auto g_f = g_decay.to(torch::kFloat32).contiguous();
    auto b_f = beta.to(torch::kFloat32).contiguous();

    // Ensure output is fp32
    TORCH_CHECK(output.dtype() == torch::kFloat32,
                "output tensor must be float32");

    dim3 grid(num_v_heads, batch_size);

    auto stream = c10::cuda::getCurrentCUDAStream();

    // Dispatch based on head_k_dim (BK must be power-of-2 >= head_k_dim)
    if (head_k_dim <= 32) {
        gdn_decode_step_kernel<32><<<grid, 32, 0, stream>>>(
            q_f.data_ptr<float>(), k_f.data_ptr<float>(),
            v_f.data_ptr<float>(), g_f.data_ptr<float>(),
            b_f.data_ptr<float>(), state.data_ptr<float>(),
            output.data_ptr<float>(), state_indices.data_ptr<int32_t>(),
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            scale, use_l2norm);
    } else if (head_k_dim <= 64) {
        gdn_decode_step_kernel<64><<<grid, 64, 0, stream>>>(
            q_f.data_ptr<float>(), k_f.data_ptr<float>(),
            v_f.data_ptr<float>(), g_f.data_ptr<float>(),
            b_f.data_ptr<float>(), state.data_ptr<float>(),
            output.data_ptr<float>(), state_indices.data_ptr<int32_t>(),
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            scale, use_l2norm);
    } else if (head_k_dim <= 128) {
        gdn_decode_step_kernel<128><<<grid, 128, 0, stream>>>(
            q_f.data_ptr<float>(), k_f.data_ptr<float>(),
            v_f.data_ptr<float>(), g_f.data_ptr<float>(),
            b_f.data_ptr<float>(), state.data_ptr<float>(),
            output.data_ptr<float>(), state_indices.data_ptr<int32_t>(),
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            scale, use_l2norm);
    } else {
        gdn_decode_step_kernel<256><<<grid, 256, 0, stream>>>(
            q_f.data_ptr<float>(), k_f.data_ptr<float>(),
            v_f.data_ptr<float>(), g_f.data_ptr<float>(),
            b_f.data_ptr<float>(), state.data_ptr<float>(),
            output.data_ptr<float>(), state_indices.data_ptr<int32_t>(),
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            scale, use_l2norm);
    }
}

}  // namespace vllm

// Standalone pybind11 module for JIT compilation / testing.
// When built as part of vLLM's _C extension, the binding is in
// torch_bindings.cpp instead.
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_decode_step", &vllm::gdn_decode_step,
          "GDN decode step (CUDA)");
}
#endif
