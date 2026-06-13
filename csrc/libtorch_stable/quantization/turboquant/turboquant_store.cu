#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "libtorch_stable/dispatch_utils.h"
#include "libtorch_stable/torch_utils.h"

namespace vllm {

template <typename scalar_t>
__global__ void turboquant_store_fp8_v4_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ slot_mapping,
    int64_t stride_block,
    int64_t stride_pos,
    int64_t stride_head,
    int D,
    int H,
    int block_size,
    int key_packed_size,
    int value_data_bytes) {
  int pid = blockIdx.x;
  int token_idx = pid / H;
  int head_idx = pid % H;

  int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }

  int64_t blk = slot / block_size;
  int64_t off = slot % block_size;
  int64_t slot_base =
      blk * stride_block + off * stride_pos + head_idx * stride_head;

  int64_t base = static_cast<int64_t>(pid) * D;
  int64_t value_base = slot_base + key_packed_size;
  int64_t value_meta_base = value_base + value_data_bytes;

  // FP8 key path: each CUDA thread owns one or more head dimensions.
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float k_val = static_cast<float>(key[base + d]);
    k_val = fminf(fmaxf(k_val, -448.0f), 448.0f);
    __nv_fp8_storage_t k_fp8 =
        __nv_cvt_float_to_fp8(k_val, __NV_SATFINITE, __NV_E4M3);
    uint8_t k_byte = static_cast<uint8_t>(k_fp8);
    kv_cache[slot_base + d] = k_byte;
  }

  float local_max = -FLT_MAX;
  float local_min = FLT_MAX;
  __shared__ float shared_min[128];
  __shared__ float shared_max[128];
  __shared__ float shared_scale;

  // Uniform value quantization uses one min/max range per token/head vector.
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float v_val = static_cast<float>(value[base + d]);
    if (v_val < local_min) {
      local_min = v_val;
    }
    if (v_val > local_max) {
      local_max = v_val;
    }
  }

  shared_min[threadIdx.x] = local_min;
  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int other = threadIdx.x + stride;
      if (shared_min[other] < shared_min[threadIdx.x]) {
        shared_min[threadIdx.x] = shared_min[other];
      }
      if (shared_max[other] > shared_max[threadIdx.x]) {
        shared_max[threadIdx.x] = shared_max[other];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float val_min = shared_min[0];
    float val_max = shared_max[0];
    float scale = (val_max - val_min) / 15.0f;
    shared_scale = scale > 1e-8f ? scale : 1e-8f;
  }
  __syncthreads();

  for (int pair = threadIdx.x; pair < value_data_bytes; pair += blockDim.x) {
    int d0 = pair * 2;
    int d1 = d0 + 1;

    float v0 = static_cast<float>(value[base + d0]);
    float v1 = d1 < D ? static_cast<float>(value[base + d1]) : 0.0f;

    int q0 = static_cast<int>((v0 - shared_min[0]) / shared_scale + 0.5f);
    int q1 = static_cast<int>((v1 - shared_min[0]) / shared_scale + 0.5f);
    q0 = q0 < 0 ? 0 : q0;
    q0 = q0 > 15 ? 15 : q0;
    q1 = q1 < 0 ? 0 : q1;
    q1 = q1 > 15 ? 15 : q1;

    uint8_t packed = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
    kv_cache[value_base + pair] = packed;
  }

  if (threadIdx.x == 0) {
    uint16_t scale_bits = __half_as_ushort(__float2half_rn(shared_scale));
    uint16_t min_bits = __half_as_ushort(__float2half_rn(shared_min[0]));

    kv_cache[value_meta_base] = static_cast<uint8_t>(scale_bits & 0xFF);
    kv_cache[value_meta_base + 1] = static_cast<uint8_t>(scale_bits >> 8);
    kv_cache[value_meta_base + 2] = static_cast<uint8_t>(min_bits & 0xFF);
    kv_cache[value_meta_base + 3] = static_cast<uint8_t>(min_bits >> 8);
  }
}

}  // namespace vllm

void turboquant_store_fp8_v4(
    const torch::stable::Tensor& key,
    const torch::stable::Tensor& value,
    torch::stable::Tensor& kv_cache,
    const torch::stable::Tensor& slot_mapping,
    int64_t stride_block,
    int64_t stride_pos,
    int64_t stride_head,
    int64_t num_heads,
    int64_t block_size,
    int64_t key_packed_size,
    int64_t value_data_bytes) {
  STD_TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  STD_TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  STD_TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be CUDA");
  STD_TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be CUDA");
  STD_TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
  STD_TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
  STD_TORCH_CHECK(kv_cache.is_contiguous(), "kv_cache must be contiguous");
  STD_TORCH_CHECK(slot_mapping.is_contiguous(),
                  "slot_mapping must be contiguous");
  STD_TORCH_CHECK(key.dim() == 2, "key must have shape [N * H, D]");
  STD_TORCH_CHECK(value.dim() == 2, "value must have shape [N * H, D]");
  STD_TORCH_CHECK(key.size(0) == value.size(0),
                  "key and value must have matching rows");
  STD_TORCH_CHECK(key.size(1) == value.size(1),
                  "key and value must have matching head dimensions");
  STD_TORCH_CHECK(kv_cache.scalar_type() == torch::headeronly::ScalarType::Byte,
                  "kv_cache must be uint8");
  STD_TORCH_CHECK(slot_mapping.scalar_type() ==
                      torch::headeronly::ScalarType::Int,
                  "slot_mapping must be int32");
  STD_TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                  "key and value must have the same dtype");
  STD_TORCH_CHECK(num_heads > 0, "num_heads must be positive");
  STD_TORCH_CHECK(block_size > 0, "block_size must be positive");

  const int64_t num_token_heads = key.size(0);
  const int64_t D = key.size(1);
  STD_TORCH_CHECK(num_token_heads % num_heads == 0,
                  "key rows must be divisible by num_heads");
  STD_TORCH_CHECK(slot_mapping.size(0) == num_token_heads / num_heads,
                  "slot_mapping length must match number of tokens");
  STD_TORCH_CHECK(D <= INT_MAX, "head dimension exceeds int range");
  STD_TORCH_CHECK(num_heads <= INT_MAX, "num_heads exceeds int range");
  STD_TORCH_CHECK(block_size <= INT_MAX, "block_size exceeds int range");
  STD_TORCH_CHECK(key_packed_size == D,
                  "FP8 key path expects key_packed_size == head dimension");
  STD_TORCH_CHECK(value_data_bytes == (D + 1) / 2,
                  "V4 value path expects value_data_bytes == ceil(D / 2)");

  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(key.get_device_index());

  constexpr int threads = 128;
  dim3 grid(num_token_heads);
  dim3 block(threads);

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "turboquant_store_fp8_v4", ([&] {
        vllm::turboquant_store_fp8_v4_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                key.const_data_ptr<scalar_t>(),
                value.const_data_ptr<scalar_t>(),
                static_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
                slot_mapping.const_data_ptr<int32_t>(), stride_block,
                stride_pos, stride_head, static_cast<int>(D),
                static_cast<int>(num_heads), static_cast<int>(block_size),
                static_cast<int>(key_packed_size),
                static_cast<int>(value_data_bytes));
      }));
}
