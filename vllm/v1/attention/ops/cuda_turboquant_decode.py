# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA warp-per-head decode kernel for TurboQuant.

Uses warp shuffle for the Hadamard butterfly (no shared memory,
no inter-warp races). One warp (32 threads) per (slot, head) pair,
each thread handles 4 dimensions (128 / 32 = 4).

JIT-compiled via torch.utils.cpp_extension.load_inline.
"""

from functools import lru_cache

import torch

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Constants
constexpr int BLOCK_D = 128;
constexpr int WARP_SIZE = 32;
constexpr int ELEMS_PER_THREAD = BLOCK_D / WARP_SIZE;  // 4

__device__ __forceinline__ float warp_shuffle_xor(float val, int mask) {
    return __shfl_xor_sync(0xFFFFFFFF, val, mask);
}

// Warp-per-head decode kernel
// Grid: (N,)  where N = num_entries * block_size * num_kv_heads
// Block: (32,) — one warp per program
__global__ void turboquant_wph_decode_kernel(
    const uint8_t* __restrict__ slot_data,   // [N, slot_bytes]
    const float* __restrict__ sign_flips,     // [BLOCK_D]
    const float* __restrict__ codebook,       // [num_centroids]
    const half* __restrict__ norms,           // [N]
    const int64_t* __restrict__ normal_idx,   // [normal_size] or nullptr
    const int64_t* __restrict__ outlier_idx,  // [n_outliers] or nullptr
    __nv_bfloat16* __restrict__ output,       // [N, head_size]
    int normal_size,
    int head_size,
    int n_outliers,
    int outlier_u8_count,
    int packed_bytes,
    int slot_bytes,
    bool has_outliers
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;  // 0..31

    // Each thread handles 4 elements: [tid*4, tid*4+1, tid*4+2, tid*4+3]
    const int base_idx = tid * ELEMS_PER_THREAD;

    // ---- Unpack 4-bit indices ----
    float vals[ELEMS_PER_THREAD];
    const uint8_t* packed = slot_data + row * slot_bytes + outlier_u8_count;

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int dim = base_idx + e;
        if (dim < normal_size) {
            int byte_idx = dim / 2;
            int is_high = dim % 2;
            uint8_t packed_byte = packed[byte_idx];
            int idx = is_high ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
            vals[e] = codebook[idx];
        } else {
            vals[e] = 0.0f;
        }
    }

    // ---- Inverse Hadamard butterfly using warp shuffle ----
    // Level h=1: partners at distance 1 (within thread, elements 0↔1, 2↔3)
    {
        float t0 = vals[0], t1 = vals[1], t2 = vals[2], t3 = vals[3];
        vals[0] = t0 + t1;
        vals[1] = t0 - t1;
        vals[2] = t2 + t3;
        vals[3] = t2 - t3;
    }

    // Level h=2: partners at distance 2 (within thread, elements 0↔2, 1↔3)
    {
        float t0 = vals[0], t1 = vals[1], t2 = vals[2], t3 = vals[3];
        vals[0] = t0 + t2;
        vals[1] = t1 + t3;
        vals[2] = t0 - t2;
        vals[3] = t1 - t3;
    }

    // Levels h=4,8,16,32,64: cross-thread via warp shuffle
    #pragma unroll
    for (int h = 4; h < BLOCK_D; h *= 2) {
        int shuffle_mask = h / ELEMS_PER_THREAD;  // h/4
        bool is_lower = ((tid * ELEMS_PER_THREAD) & h) == 0;

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            float partner = warp_shuffle_xor(vals[e], shuffle_mask);
            if (is_lower) {
                vals[e] = vals[e] + partner;
            } else {
                vals[e] = partner - vals[e];
            }
        }
    }

    // ---- Scale by 1/sqrt(BLOCK_D) ----
    const float had_scale = 1.0f / sqrtf((float)BLOCK_D);
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        vals[e] *= had_scale;
    }

    // ---- Sign flips ----
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int dim = base_idx + e;
        if (dim < BLOCK_D) {
            vals[e] *= sign_flips[dim];
        }
    }

    // ---- Scale by norm ----
    float norm_val = __half2float(norms[row]);
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        vals[e] *= norm_val;
    }

    // ---- Write output ----
    __nv_bfloat16* out_row = output + row * head_size;

    if (has_outliers) {
        // Write normal channels to scattered positions
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int dim = base_idx + e;
            if (dim < normal_size) {
                int out_pos = normal_idx[dim];
                out_row[out_pos] = __float2bfloat16(vals[e]);
            }
        }

        // Copy outlier bf16 from slot to output (thread 0..n_outliers/4 handle)
        const uint8_t* outlier_bytes = slot_data + row * slot_bytes;
        for (int o = tid; o < n_outliers; o += WARP_SIZE) {
            // Read 2 bytes, interpret as bf16
            uint16_t lo = outlier_bytes[o * 2];
            uint16_t hi = outlier_bytes[o * 2 + 1];
            uint16_t bf16_bits = lo | (hi << 8);
            __nv_bfloat16 val;
            memcpy(&val, &bf16_bits, sizeof(val));
            int out_pos = outlier_idx[o];
            out_row[out_pos] = val;
        }
    } else {
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int dim = base_idx + e;
            if (dim < normal_size) {
                out_row[dim] = __float2bfloat16(vals[e]);
            }
        }
    }
}

// C++ wrapper
torch::Tensor turboquant_wph_decode(
    torch::Tensor slot_data,     // [N, slot_bytes] uint8
    torch::Tensor sign_flips,    // [128] float32
    torch::Tensor codebook,      // [num_centroids] float32
    torch::Tensor norms,         // [N] float16
    torch::Tensor normal_idx,    // [normal_size] int64
    torch::Tensor outlier_idx,   // [n_outliers] int64
    int normal_size,
    int head_size,
    int n_outliers,
    int outlier_u8_count,
    int packed_bytes,
    bool has_outliers
) {
    int N = slot_data.size(0);
    int slot_bytes = slot_data.size(1);

    auto output = torch::empty({N, head_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(slot_data.device()));

    // One warp (32 threads) per row
    dim3 grid(N);
    dim3 block(32);

    turboquant_wph_decode_kernel<<<grid, block, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        slot_data.data_ptr<uint8_t>(),
        sign_flips.data_ptr<float>(),
        codebook.data_ptr<float>(),
        reinterpret_cast<const half*>(norms.data_ptr<at::Half>()),
        has_outliers ? normal_idx.data_ptr<int64_t>() : nullptr,
        has_outliers ? outlier_idx.data_ptr<int64_t>() : nullptr,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        normal_size, head_size, n_outliers,
        outlier_u8_count, packed_bytes, slot_bytes,
        has_outliers
    );

    return output;
}
"""

CPP_SOURCE = """
torch::Tensor turboquant_wph_decode(
    torch::Tensor slot_data,
    torch::Tensor sign_flips,
    torch::Tensor codebook,
    torch::Tensor norms,
    torch::Tensor normal_idx,
    torch::Tensor outlier_idx,
    int normal_size,
    int head_size,
    int n_outliers,
    int outlier_u8_count,
    int packed_bytes,
    bool has_outliers
);
"""


@lru_cache(maxsize=1)
def _load_cuda_module():
    """JIT compile the CUDA warp-per-head decode kernel."""
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="turboquant_wph",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=["turboquant_wph_decode"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def cuda_wph_decode_from_slots(
    flat_slots: torch.Tensor,  # [N, slot_bytes] uint8
    sign_flips: torch.Tensor,  # [BLOCK_D] float32
    codebook: torch.Tensor,  # [num_centroids] float32
    normal_idx: torch.Tensor | None,
    outlier_idx: torch.Tensor | None,
    head_size: int,
    normal_size: int,
    n_outliers: int,
    packed_bytes: int,
) -> torch.Tensor:
    """CUDA warp-per-head decode: slot bytes → decoded bf16 head.

    Uses warp shuffle for Hadamard butterfly (no shared memory races).
    Returns: [N, head_size] bfloat16
    """
    mod = _load_cuda_module()

    N = flat_slots.shape[0]
    outlier_u8_count = n_outliers * 2
    norm_offset = outlier_u8_count + packed_bytes

    # Extract norms
    norms = (
        flat_slots[:, norm_offset : norm_offset + 2]
        .contiguous()
        .view(torch.float16)
        .reshape(N)
    )

    has_outliers = normal_idx is not None and n_outliers > 0
    if normal_idx is None:
        normal_idx = torch.empty(0, dtype=torch.int64, device=flat_slots.device)
    if outlier_idx is None:
        outlier_idx = torch.empty(0, dtype=torch.int64, device=flat_slots.device)

    return mod.turboquant_wph_decode(
        flat_slots,
        sign_flips,
        codebook,
        norms,
        normal_idx,
        outlier_idx,
        normal_size,
        head_size,
        n_outliers,
        outlier_u8_count,
        packed_bytes,
        has_outliers,
    )
