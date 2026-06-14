/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Fused kernel: K RoPE + static FP8 per-tensor KV cache write.
 * Supports NeoX and GPT-J rotation styles, both flash and non-flash
 * KV cache layouts, half and bfloat16 inputs.
 *
 * Works on both CUDA (sm80+) and AMD ROCm via USE_ROCM + hipify.
 *
 * Optimized v2: per-token block strategy with shared-memory RoPE peer
 * access and packed FP8 stores.
 */

// Dual-platform FP8 header pattern:
//   source: fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:32-36
#ifndef USE_ROCM
  #include <cuda_fp8.h>
#else
  #include <hip/hip_fp8.h>
#endif

// Standard CUDA/ATen includes:
//   source: fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:37-46
#include <cuda_runtime.h>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

// Wave/warp-width mask: 64-wide on ROCm (wavefront), 32-wide on CUDA (warp).
//   source: fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:48-54
#ifndef FINAL_MASK
  #ifdef USE_ROCM
    #define FINAL_MASK 0xffffffffffffffffULL
  #else
    #define FINAL_MASK 0xffffffffu
  #endif
#endif

// FP8 E4M3 clamp ceiling — mirrors kFp8ScaleDivisor in cache_kernels.cu:31-34.
//   AMD gfx942 uses FNUZ format whose max is 224 (not 240 which is the
//   theoretical bit-pattern max); OCP and NVIDIA use standard E4M3 max 448.
//   The HIP_FP8_TYPE_OCP guard follows fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:59.
//   source (values): cache_kernels.cu:31-34
//   source (OCP guard): fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:59
#ifdef USE_ROCM
  #if defined(HIP_FP8_TYPE_OCP)
    constexpr float kRopeFp8E4M3Max = 448.0f;
  #else
    constexpr float kRopeFp8E4M3Max = 224.0f;  // FNUZ (MI300X) — matches vLLM kFp8ScaleDivisor
  #endif
#else
  constexpr float kRopeFp8E4M3Max = 448.0f;
#endif

// ROCm FP8 float-to-byte helper — direct copy of rocm_cvt_float_to_fp8_e4m3.
//   source: fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:56-65
#ifdef USE_ROCM
__device__ __forceinline__ uint8_t rope_cvt_float_to_fp8_e4m3(float val) {
  #if defined(HIP_FP8_TYPE_OCP)
  __hip_fp8_e4m3 fp8_val(val);
  #else
  __hip_fp8_e4m3_fnuz fp8_val(val);
  #endif
  return reinterpret_cast<uint8_t&>(fp8_val);
}
#endif

// ── Inline FP8 conversion helper ──────────────────────────────────────────
// Unified cross-platform float → FP8 E4M3 byte conversion.
__device__ __forceinline__ uint8_t cvt_fp8(float val) {
#ifndef USE_ROCM
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return 0; // Volta/Turing stub fallback
  #else
    __nv_fp8_storage_t s =
        __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
    return static_cast<uint8_t>(s);
  #endif
#else
    return rope_cvt_float_to_fp8_e4m3(val);
#endif
}

namespace vllm {
namespace rope_fp8_kvcache {

// ─── Optimized Kernel v3 (Vectorized Parallel Heads) ───────────────────────
//
// Grid:  dim3(num_tokens) — one block per token
// Block: dim3(std::max(64, std::min((num_kv_heads * head_size) / 8, 256)))
//        — threads cooperatively process all KV heads in parallel
//
// This restructuring maps threads across all heads in parallel using 128-bit
// aligned vector loads and 64-bit vector stores, matching the unfused path's
// memory execution throughput.
//
// Each block:
//   1. Load entire K tensor of the token into shared memory (128-bit int4 loads)
//   2. __syncthreads()
//   3. Process 8 elements per thread: apply RoPE (using shared memory for peer access)
//      quantize to FP8, and store as 64-bit uint64_t to key_cache
//   4. Process 8 elements per thread for V: load from global, quantize, and store
//      as 64-bit uint64_t to value_cache (no RoPE/smem needed)
//
template <typename scalar_t, bool IS_NEOX>
__global__ void __launch_bounds__(256) fused_rope_fp8_kvcache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_kv_heads, head_size]
    uint8_t* __restrict__ key_cache,     // fp8, addressed via stride args
    uint8_t* __restrict__ value_cache,   // fp8, addressed via stride args
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]  int64
    const int64_t* __restrict__ positions,     // [num_tokens]  int64
    const float* __restrict__ cos_sin_cache,   // [max_pos, rot_dim]  fp32
    const float* __restrict__ k_scale,         // static per-tensor scale for K
    const float* __restrict__ v_scale,         // static per-tensor scale for V
    int64_t kc_stride_block,    // key_cache: elements per block
    int64_t kc_stride_page,     // key_cache: elements per position within block
    int64_t kc_stride_head,     // key_cache: elements per kv_head
    int64_t vc_stride_block,
    int64_t vc_stride_page,
    int64_t vc_stride_head,
    int num_kv_heads,
    int head_size,
    int rot_dim,
    int block_size)
{
    float k_scale_inv = 1.0f / (*k_scale);
    float v_scale_inv = 1.0f / (*v_scale);

    int token_idx = blockIdx.x;

    // Slot lookup: paged KV cache scatter addressing.
    int64_t slot_id = slot_mapping[token_idx];
    if (slot_id < 0) return;  // padded token — skip

    int64_t blk_idx    = slot_id / block_size;
    int64_t blk_offset = slot_id % block_size;

    // Position for cos/sin lookup.
    int64_t pos = positions[token_idx];

    int embed_dim = rot_dim / 2;
    const float* cos_ptr = cos_sin_cache + pos * rot_dim;
    const float* sin_ptr = cos_ptr + embed_dim;

    int total_elems = num_kv_heads * head_size;
    const int64_t token_offset = (int64_t)token_idx * total_elems;

    // Cache base pointers for this token's slot.
    uint8_t* kc_base = key_cache
                       + blk_idx    * kc_stride_block
                       + blk_offset * kc_stride_page;
    uint8_t* vc_base = value_cache
                       + blk_idx    * vc_stride_block
                       + blk_offset * vc_stride_page;

    // ── Shared memory for K token data (used for RoPE peer access) ────────
    extern __shared__ char smem[];
    scalar_t* smem_k = reinterpret_cast<scalar_t*>(smem);

    // Parallel load of K to shared memory (128-bit loads)
    int num_vecs_16b = total_elems / 8;
    const int4* k_src_v = reinterpret_cast<const int4*>(key + token_offset);
    int4* smem_k_v = reinterpret_cast<int4*>(smem_k);

    for (int idx = threadIdx.x; idx < num_vecs_16b; idx += blockDim.x) {
        smem_k_v[idx] = k_src_v[idx];
    }
    __syncthreads();

    // Parallel RoPE + Store K (8 elements per thread, 64-bit stores)
    int num_vecs_8 = total_elems / 8;
    for (int vec_idx = threadIdx.x; vec_idx < num_vecs_8; vec_idx += blockDim.x) {
        int idx_base = vec_idx * 8;
        int h = idx_base / head_size;
        int i = idx_base % head_size;

        alignas(16) scalar_t k_val[8];
        *reinterpret_cast<int4*>(k_val) = *reinterpret_cast<const int4*>(&smem_k[idx_base]);

        if (i < rot_dim) {
            if (IS_NEOX) {
                int peer_base = (i < embed_dim) ? idx_base + embed_dim : idx_base - embed_dim;
                alignas(16) scalar_t k_peer[8];
                *reinterpret_cast<int4*>(k_peer) = *reinterpret_cast<const int4*>(&smem_k[peer_base]);

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float val = static_cast<float>(k_val[j]);
                    float peer = static_cast<float>(k_peer[j]);
                    int rot_off = (i + j) % embed_dim;
                    float cos_v = VLLM_LDG(cos_ptr + rot_off);
                    float sin_v = VLLM_LDG(sin_ptr + rot_off);
                    if ((i + j) < embed_dim) {
                        val = val * cos_v - peer * sin_v;
                    } else {
                        val = peer * sin_v + val * cos_v;
                    }
                    k_val[j] = static_cast<scalar_t>(val);
                }
            } else {
                alignas(16) scalar_t k_val_orig[8];
                *reinterpret_cast<int4*>(k_val_orig) = *reinterpret_cast<int4*>(k_val);

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float val = static_cast<float>(k_val_orig[j]);
                    int rot_off = (i + j) / 2;
                    float cos_v = VLLM_LDG(cos_ptr + rot_off);
                    float sin_v = VLLM_LDG(sin_ptr + rot_off);
                    float peer = static_cast<float>(k_val_orig[j ^ 1]);
                    if (j % 2 == 0) {
                        val = val * cos_v - peer * sin_v;
                    } else {
                        val = peer * sin_v + val * cos_v;
                    }
                    k_val[j] = static_cast<scalar_t>(val);
                }
            }
        }

        uint8_t k_fp8[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = static_cast<float>(k_val[j]);
            float scaled = val * k_scale_inv;
            scaled = fminf(fmaxf(scaled, -kRopeFp8E4M3Max), kRopeFp8E4M3Max);
            k_fp8[j] = cvt_fp8(scaled);
        }

        uint8_t* k_dst = kc_base + h * kc_stride_head + i;
        *reinterpret_cast<uint64_t*>(k_dst) = *reinterpret_cast<uint64_t*>(k_fp8);
    }

    // Parallel Load + Store V (no RoPE, no shared memory)
    const int4* v_src_v = reinterpret_cast<const int4*>(value + token_offset);
    for (int vec_idx = threadIdx.x; vec_idx < num_vecs_8; vec_idx += blockDim.x) {
        int idx_base = vec_idx * 8;
        int h = idx_base / head_size;
        int i = idx_base % head_size;

        int4 v_val_raw = v_src_v[vec_idx];
        const scalar_t* v_val = reinterpret_cast<const scalar_t*>(&v_val_raw);

        uint8_t v_fp8[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = static_cast<float>(v_val[j]);
            float scaled = val * v_scale_inv;
            scaled = fminf(fmaxf(scaled, -kRopeFp8E4M3Max), kRopeFp8E4M3Max);
            v_fp8[j] = cvt_fp8(scaled);
        }

        uint8_t* v_dst = vc_base + h * vc_stride_head + i;
        *reinterpret_cast<uint64_t*>(v_dst) = *reinterpret_cast<uint64_t*>(v_fp8);
    }
}

// ─── Launch wrapper ───────────────────────────────────────────────────────────
template <typename scalar_t>
void launch_fused_rope_fp8_kvcache(
    const scalar_t* key, const scalar_t* value,
    uint8_t* key_cache, uint8_t* value_cache,
    const int64_t* slot_mapping, const int64_t* positions,
    const float* cos_sin_cache,
    const float* k_scale, const float* v_scale,
    int64_t kc_stride_block, int64_t kc_stride_page, int64_t kc_stride_head,
    int64_t vc_stride_block, int64_t vc_stride_page, int64_t vc_stride_head,
    int num_tokens, int num_kv_heads, int head_size, int rot_dim,
    int block_size, bool is_neox, cudaStream_t stream)
{
    if (num_tokens == 0) return;

    // Grid: one block per token
    // Block: dynamic based on total elements per token to maximize occupancy
    int total_elems = num_kv_heads * head_size;
    int num_threads = std::max(64, std::min(total_elems / 8, 256));

    dim3 grid(num_tokens);
    dim3 block(num_threads);

    // Dynamic shared memory: entire token K data for RoPE peer access
    size_t smem_bytes = total_elems * sizeof(scalar_t);

    if (is_neox) {
        fused_rope_fp8_kvcache_kernel<scalar_t, /*IS_NEOX=*/true>
            <<<grid, block, smem_bytes, stream>>>(
                key, value, key_cache, value_cache,
                slot_mapping, positions, cos_sin_cache,
                k_scale, v_scale,
                kc_stride_block, kc_stride_page, kc_stride_head,
                vc_stride_block, vc_stride_page, vc_stride_head,
                num_kv_heads, head_size, rot_dim, block_size);
    } else {
        fused_rope_fp8_kvcache_kernel<scalar_t, /*IS_NEOX=*/false>
            <<<grid, block, smem_bytes, stream>>>(
                key, value, key_cache, value_cache,
                slot_mapping, positions, cos_sin_cache,
                k_scale, v_scale,
                kc_stride_block, kc_stride_page, kc_stride_head,
                vc_stride_block, vc_stride_page, vc_stride_head,
                num_kv_heads, head_size, rot_dim, block_size);
    }
}

}  // namespace rope_fp8_kvcache
}  // namespace vllm

// Torch op wrapper 
//
// key:            [num_tokens, num_kv_heads, head_size]  fp16 or bf16
// value:          [num_tokens, num_kv_heads, head_size]  fp16 or bf16
// key_cache:      FP8 paged cache (uint8_t), flash or non-flash layout
// value_cache:    FP8 paged cache (uint8_t), flash or non-flash layout
// slot_mapping:   [num_tokens]  int64 — paged KV slot index per token
// positions:      [num_tokens]  int64 — position index for RoPE lookup
// cos_sin_cache:  [max_pos, rot_dim]  float32
// k_scale:        scalar float32 — static per-tensor scale for K
// v_scale:        scalar float32 — static per-tensor scale for V
// is_neox:        bool — True = NeoX rotation, False = GPT-J
// flash_layout:   bool — True = [blocks, block_size, heads, dim]
//                         False = [blocks, heads, block_size, dim]
//
void fused_rope_fp8_kvcache(
    torch::Tensor const& key,
    torch::Tensor const& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor const& slot_mapping,
    torch::Tensor const& positions,
    torch::Tensor const& cos_sin_cache,
    torch::Tensor const& k_scale,
    torch::Tensor const& v_scale,
    bool is_neox,
    bool flash_layout)
{
    TORCH_CHECK(key.is_cuda() && key.is_contiguous(),
                "key must be contiguous CUDA tensor");
    TORCH_CHECK(value.is_cuda() && value.is_contiguous(),
                "value must be contiguous CUDA tensor");
    TORCH_CHECK(key_cache.is_cuda(), "key_cache must be CUDA");
    TORCH_CHECK(value_cache.is_cuda(), "value_cache must be CUDA");
    TORCH_CHECK(slot_mapping.dtype() == torch::kInt64,
                "slot_mapping must be int64");
    TORCH_CHECK(positions.dtype() == torch::kInt64, "positions must be int64");
    TORCH_CHECK(cos_sin_cache.dtype() == torch::kFloat32,
                "cos_sin_cache must be float32");
    TORCH_CHECK(k_scale.dtype() == torch::kFloat32, "k_scale must be float32");
    TORCH_CHECK(v_scale.dtype() == torch::kFloat32, "v_scale must be float32");
    TORCH_CHECK(key_cache.dtype() == torch::kUInt8, "key_cache must be uint8");
    TORCH_CHECK(value_cache.dtype() == torch::kUInt8, "value_cache must be uint8");
    TORCH_CHECK(key.dim() == 3, "key must be [num_tokens, num_kv_heads, head_size]");
    TORCH_CHECK(value.dim() == 3, "value must be [num_tokens, num_kv_heads, head_size]");

    int num_tokens   = static_cast<int>(key.size(0));
    int num_kv_heads = static_cast<int>(key.size(1));
    int head_size    = static_cast<int>(key.size(2));
    int rot_dim      = static_cast<int>(cos_sin_cache.size(1));

    TORCH_CHECK(rot_dim % 2 == 0, "rot_dim must be even");
    TORCH_CHECK(rot_dim <= head_size, "rot_dim cannot exceed head_size");
    TORCH_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8");
    TORCH_CHECK(rot_dim % 16 == 0, "rot_dim must be a multiple of 16");
    TORCH_CHECK((num_kv_heads * head_size) % 8 == 0, "total KV elements must be a multiple of 8");

    // Stride extraction: read dim indices differently depending on cache layout.
    //   flash     [num_blocks, block_size, num_kv_heads, head_size] → strides (0,1,2)
    //   non-flash [num_blocks, num_kv_heads, block_size, head_size] → strides (0,2,1)
    int64_t kc_stride_block, kc_stride_page, kc_stride_head;
    int64_t vc_stride_block, vc_stride_page, vc_stride_head;
    int block_size;
    if (flash_layout) {
        kc_stride_block = key_cache.stride(0);
        kc_stride_page  = key_cache.stride(1);
        kc_stride_head  = key_cache.stride(2);
        vc_stride_block = value_cache.stride(0);
        vc_stride_page  = value_cache.stride(1);
        vc_stride_head  = value_cache.stride(2);
        block_size = static_cast<int>(key_cache.size(1));
    } else {
        kc_stride_block = key_cache.stride(0);
        kc_stride_head  = key_cache.stride(1);   // heads before positions
        kc_stride_page  = key_cache.stride(2);
        vc_stride_block = value_cache.stride(0);
        vc_stride_head  = value_cache.stride(1);
        vc_stride_page  = value_cache.stride(2);
        block_size = static_cast<int>(key_cache.size(2));
    }

    // Device guard + stream — standard pattern for all vLLM Torch op wrappers.
    //   source: pos_encoding_kernels.cu:170
    at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    auto stream = at::cuda::getCurrentCUDAStream();

    // Dispatch over float / half / bfloat16 — instantiates scalar_t at compile
    //   time for each supported dtype.
    //   source: pos_encoding_kernels.cu:171
    VLLM_DISPATCH_FLOATING_TYPES(
        key.scalar_type(), "fused_rope_fp8_kvcache", [&] {
            vllm::rope_fp8_kvcache::launch_fused_rope_fp8_kvcache<scalar_t>(
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                reinterpret_cast<uint8_t*>(key_cache.data_ptr()),
                reinterpret_cast<uint8_t*>(value_cache.data_ptr()),
                slot_mapping.data_ptr<int64_t>(),
                positions.data_ptr<int64_t>(),
                cos_sin_cache.data_ptr<float>(),
                reinterpret_cast<const float*>(k_scale.data_ptr()),
                reinterpret_cast<const float*>(v_scale.data_ptr()),
                kc_stride_block, kc_stride_page, kc_stride_head,
                vc_stride_block, vc_stride_page, vc_stride_head,
                num_tokens, num_kv_heads, head_size, rot_dim,
                block_size, is_neox, stream);
        });
}
