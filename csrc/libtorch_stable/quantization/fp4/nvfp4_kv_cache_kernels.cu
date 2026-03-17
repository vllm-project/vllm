// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// NVFP4 KV cache store kernel.
// Quantizes bf16 key/value to packed FP4 + FP8 block scales and writes them
// into the paged KV cache in HND layout.
//
// Reuses device functions from nvfp4_utils.cuh:
//   - cvt_warp_fp16_to_fp4()  for bf16 → fp4 quantization + block scale
//   - pack_fp4()              for packing float pairs to fp4
//   - reciprocal_approximate_ftz() for fast reciprocal

#define NVFP4_ENABLE_ELTS16 1
#include "nvfp4_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "../../dispatch_utils.h"

namespace vllm {

// Compute swizzled scale offset for SM100 trtllm-gen MHA kernel.
// The swizzle pattern for HND layout is:
//   [T//4, 4, 4, S//4] → permute(0, 2, 3, 1) → reshape to [T, S]
// where T = block_size (page_size), S = scale_dim = head_size // 16.
//
// For a linear (t, s) position, the swizzled position is:
//   swizzled_t = (t / 4) * 4 + (s / (S / 4))
//   swizzled_s = (s % (S / 4)) * 4 + (t % 4)
__device__ __forceinline__ int swizzle_scale_offset(int t, int s,
                                                    int scale_dim) {
  int s_group = scale_dim / 4;
  int swizzled_t = (t / 4) * 4 + (s / s_group);
  int swizzled_s = (s % s_group) * 4 + (t % 4);
  return swizzled_t * scale_dim + swizzled_s;
}

// Kernel: quantize bf16 key/value to NVFP4 and store in paged KV cache.
//
// Takes separate key_cache and value_cache pointers (matching the
// reshape_and_cache_flash interface where caller passes kv_cache[:, 0]
// and kv_cache[:, 1]).
//
// Cache layout per K or V (HND, packed data+scale):
//   [num_blocks, num_heads, block_size, head_size//2 + head_size//16]
//   The first head_size//2 bytes per slot are packed FP4 data (uint8).
//   The remaining head_size//16 bytes are FP8 E4M3 block scale factors,
//   stored in the SM100 4x4 swizzled layout.
//
// Threading: one CUDA block per token, threads process heads and
// groups of 16 elements within each head.
template <typename scalar_t>
__global__ void reshape_and_cache_nvfp4_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    uint8_t* __restrict__ key_cache,     // [num_blocks, num_heads, block_size,
                                         //  last_dim]
    uint8_t* __restrict__ value_cache,   // same as key_cache
    const int64_t* __restrict__ slot_mapping,  // [num_actual_tokens]
    const float k_global_scale,  // reciprocal of checkpoint k_scale
    const float v_global_scale,  // reciprocal of checkpoint v_scale
    const int64_t key_stride,    // key.stride(0) in elements
    const int64_t value_stride,  // value.stride(0) in elements
    const int num_heads, const int head_size, const int block_size,
    const int64_t block_stride,         // stride for block_idx in cache dim 0
    const int64_t head_stride,          // stride for each head in cache dim 1
    const int64_t block_offset_stride,  // stride for block_offset in cache
                                        // dim 2
    const int last_dim                  // head_size//2 + head_size//16
) {
  using CudaType = typename CUDATypeConverter<scalar_t>::Type;
  using PVec = PackedVec<CudaType, CVT_FP4_PACK16>;

  static constexpr int ELTS = CVT_FP4_ELTS_PER_THREAD;  // 16 or 8
  static constexpr int THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / ELTS;

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int block_offset = static_cast<int>(slot_idx % block_size);

  const int data_dim = head_size / 2;
  const int scale_dim = head_size / 16;
  const int groups_per_head = head_size / CVT_FP4_SF_VEC_SIZE;

  const int total_groups = num_heads * groups_per_head;
  const int tid = threadIdx.x;
  const int num_thread_groups = blockDim.x / THREADS_PER_SF;
  const int tg_id = tid / THREADS_PER_SF;
  const int tg_lane = tid % THREADS_PER_SF;

  // Process both K (kv=0) and V (kv=1)
#pragma unroll
  for (int kv = 0; kv < 2; kv++) {
    const scalar_t* __restrict__ src = (kv == 0) ? key : value;
    const float global_scale = (kv == 0) ? k_global_scale : v_global_scale;
    const int64_t src_stride = (kv == 0) ? key_stride : value_stride;
    uint8_t* __restrict__ cache = (kv == 0) ? key_cache : value_cache;

    // Source pointer for this token (use actual stride, not assumed contiguous)
    const CudaType* __restrict__ token_src =
        reinterpret_cast<const CudaType*>(src) + token_idx * src_stride;

    // Destination base in cache for this token's block
    uint8_t* __restrict__ cache_block = cache + block_idx * block_stride;

    for (int g = tg_id; g < total_groups; g += num_thread_groups) {
      const int head = g / groups_per_head;
      const int group_in_head = g % groups_per_head;

      // Load 16 (or 8) bf16 elements from source
      PVec in_vec;
      const CudaType* __restrict__ src_ptr =
          token_src + head * head_size + group_in_head * CVT_FP4_SF_VEC_SIZE +
          tg_lane * ELTS;

#pragma unroll
      for (int i = 0; i < ELTS / 2; i++) {
        in_vec.elts[i] = reinterpret_cast<
            const typename PackedTypeConverter<CudaType>::Type*>(src_ptr)[i];
      }

      // Quantize: produces packed fp4 and writes scale factor.
      uint8_t sf_val;
      uint8_t* sf_out_ptr = (tg_lane == 0) ? &sf_val : nullptr;

      fp4_packed_t packed = cvt_warp_fp16_to_fp4<CudaType, THREADS_PER_SF>(
          in_vec, global_scale, sf_out_ptr);

      // Write packed FP4 data to cache
      uint8_t* __restrict__ data_dst =
          cache_block + head * head_stride + block_offset * block_offset_stride;

#if CVT_FP4_PACK16
      {
        // 16 elements → 8 bytes (u32x2)
        int data_byte_offset = group_in_head * 8;
        reinterpret_cast<uint64_t*>(data_dst + data_byte_offset)[0] =
            (uint64_t(packed.hi) << 32) | uint64_t(packed.lo);
      }
#else
      {
        // 8 elements → 4 bytes (uint32_t)
        int data_byte_offset =
            group_in_head * CVT_FP4_SF_VEC_SIZE / 2 + tg_lane * ELTS / 2;
        reinterpret_cast<uint32_t*>(data_dst + data_byte_offset)[0] = packed;
      }
#endif

      // Write swizzled block scale (only one thread per group writes)
      if (sf_out_ptr != nullptr) {
        int scale_idx = group_in_head;
        int swizzled_offset =
            swizzle_scale_offset(block_offset, scale_idx, scale_dim);
        int swizzled_t = swizzled_offset / scale_dim;
        int swizzled_s = swizzled_offset % scale_dim;
        uint8_t* __restrict__ scale_dst = cache_block + head * head_stride +
                                          swizzled_t * block_offset_stride +
                                          data_dim + swizzled_s;
        *scale_dst = sf_val;
      }
    }
  }
}

}  // namespace vllm

// Non-template entry point callable from cache_kernels.cu
void reshape_and_cache_nvfp4_dispatch(torch::Tensor& key, torch::Tensor& value,
                                      torch::Tensor& key_cache,
                                      torch::Tensor& value_cache,
                                      torch::Tensor& slot_mapping,
                                      torch::Tensor& k_scale,
                                      torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int last_dim = head_size / 2 + head_size / 16;

  // key_cache logical shape is always [num_blocks, block_size, num_heads,
  // last_dim] regardless of NHD/HND layout (layout only affects strides via
  // permutation). So we always read sizes and strides in the logical order.
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be 4D");
  TORCH_CHECK(key_cache.size(3) == last_dim,
              "key_cache last dim must be head_size//2 + head_size//16");

  int block_size = key_cache.size(1);
  int64_t block_stride = key_cache.stride(0);
  int64_t block_offset_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);

  TORCH_CHECK(head_size % 16 == 0,
              "head_size must be divisible by 16 for NVFP4 KV cache");
  TORCH_CHECK(block_size % 4 == 0,
              "block_size must be divisible by 4 for NVFP4 KV cache swizzle");

  float k_global_scale = 1.0f / k_scale.item<float>();
  float v_global_scale = 1.0f / v_scale.item<float>();

  int groups_per_head = head_size / CVT_FP4_SF_VEC_SIZE;
  int total_groups = num_heads * groups_per_head;
  constexpr int THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  int num_threads = std::min(total_groups * THREADS_PER_SF, 512);
  num_threads = ((num_threads + 31) / 32) * 32;

  dim3 grid(num_tokens);
  dim3 block(num_threads);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      key.scalar_type(), "reshape_and_cache_nvfp4", [&] {
        vllm::reshape_and_cache_nvfp4_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
                slot_mapping.data_ptr<int64_t>(), k_global_scale,
                v_global_scale, key.stride(0), value.stride(0), num_heads,
                head_size, block_size, block_stride, head_stride,
                block_offset_stride, last_dim);
      });
}
