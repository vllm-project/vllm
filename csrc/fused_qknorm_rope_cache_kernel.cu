// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Step5 fused QK norm, RoPE, and KV-cache update kernel.

// This kernel is built into `_C_stable_libtorch` and uses the libtorch stable
// ABI (torch::stable::Tensor / STD_TORCH_CHECK) instead of ATen and c10.
#include "torch_utils.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"
#include "core/vec_dtypes/dtype_fp8.cuh"
#include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"

#include <torch/csrc/stable/macros.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>

namespace vllm {

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;

template <typename scalar_t, typename weight_t, typename rope_t, int HEAD_DIM,
          bool WIDE_192_ROTARY = false, bool FP8_CACHE = false>
__global__ void qknorm_rope_cache_kernel(
    const void* __restrict__ qkv_void, void* __restrict__ q_out_void,
    void* __restrict__ k_out_void, void* __restrict__ v_out_void,
    const void* __restrict__ q_weight_void,
    const void* __restrict__ k_weight_void, const void* __restrict__ cos_void,
    const void* __restrict__ sin_void, const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping, void* __restrict__ k_cache_void,
    void* __restrict__ v_cache_void, int64_t qkv_token_stride,
    int64_t q_token_stride, int64_t k_token_stride, int64_t v_token_stride,
    int64_t cos_token_stride, int64_t sin_token_stride,
    int64_t cache_block_stride, int64_t cache_page_stride,
    int64_t cache_head_stride, int num_tokens, int num_slots, int num_q_heads,
    int num_kv_heads, int rotary_pairs, int block_size, float epsilon,
    int block_size_shift, float norm_weight_bias, const float* k_scale,
    const float* v_scale) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
  if constexpr (std::is_same_v<scalar_t, torch::headeronly::BFloat16> ||
                std::is_same_v<weight_t, torch::headeronly::BFloat16> ||
                std::is_same_v<rope_t, torch::headeronly::BFloat16>) {
    return;
  } else {
#endif
    using ScalarConverter = _typeConvert<scalar_t>;
    using WeightConverter = _typeConvert<weight_t>;
    using RopeConverter = _typeConvert<rope_t>;
    using Scalar = typename ScalarConverter::hip_type;
    using Weight = typename WeightConverter::hip_type;
    using Rope = typename RopeConverter::hip_type;

    // Preserve the original per-head reduction and activation rounding.
    static_assert(!WIDE_192_ROTARY || HEAD_DIM == 192);
    constexpr int kThreadsPerHead =
        WIDE_192_ROTARY ? 32
                        : (HEAD_DIM <= 64 ? 8 : (HEAD_DIM <= 192 ? 16 : 32));
    constexpr int kValuesPerVector =
        WIDE_192_ROTARY ? 8 : (HEAD_DIM == 192 ? 4 : 8);
    constexpr int kValuesPerTile = kThreadsPerHead * kValuesPerVector;
    constexpr int kNumVectors =
        (HEAD_DIM + kValuesPerTile - 1) / kValuesPerTile;
    constexpr int kValuesPerThread = kValuesPerVector * kNumVectors;
    constexpr int kTileSize = kValuesPerTile * kNumVectors;
    constexpr int kHeadsPerBlock = kThreadsPerBlock / kThreadsPerHead;

    const auto* qkv = static_cast<const Scalar*>(qkv_void);
    auto* q_out = static_cast<Scalar*>(q_out_void);
    auto* k_out = static_cast<Scalar*>(k_out_void);
    auto* v_out = static_cast<Scalar*>(v_out_void);
    const auto* q_weight = static_cast<const Weight*>(q_weight_void);
    const auto* k_weight = static_cast<const Weight*>(k_weight_void);
    const auto* cos = static_cast<const Rope*>(cos_void);
    const auto* sin = static_cast<const Rope*>(sin_void);
    auto* k_cache = static_cast<Scalar*>(k_cache_void);
    auto* v_cache = static_cast<Scalar*>(v_cache_void);

    const int head_in_block = threadIdx.x / kThreadsPerHead;
    const int lane = threadIdx.x % kThreadsPerHead;
    const int total_heads = num_q_heads + 2 * num_kv_heads;
    int token;
    int local_head;
    const int64_t global_head =
        static_cast<int64_t>(blockIdx.x) * kHeadsPerBlock + head_in_block;
    token = global_head / total_heads;
    // Reuse the quotient instead of spelling this as global_head %
    // total_heads. nvcc otherwise emits a second runtime integer-division
    // sequence for the flattened token/head mapping.
    local_head = static_cast<int>(global_head -
                                  static_cast<int64_t>(token) * total_heads);

    if (token >= num_tokens) return;

    const bool is_q = local_head < num_q_heads;
    const bool is_k = !is_q && local_head < num_q_heads + num_kv_heads;
    const int head = is_q ? local_head
                          : (is_k ? local_head - num_q_heads
                                  : local_head - num_q_heads - num_kv_heads);

    int64_t input_offset = token * qkv_token_stride;
    Scalar* output;
    int64_t output_offset;
    if (is_q) {
      input_offset += static_cast<int64_t>(head) * HEAD_DIM;
      output = q_out;
      output_offset =
          token * q_token_stride + static_cast<int64_t>(head) * HEAD_DIM;
    } else if (is_k) {
      input_offset += static_cast<int64_t>(num_q_heads + head) * HEAD_DIM;
      output = k_out;
      output_offset =
          token * k_token_stride + static_cast<int64_t>(head) * HEAD_DIM;
    } else {
      input_offset +=
          static_cast<int64_t>(num_q_heads + num_kv_heads + head) * HEAD_DIM;
      output = v_out;
      output_offset =
          token * v_token_stride + static_cast<int64_t>(head) * HEAD_DIM;
    }

    bool write_cache = false;
    int64_t cache_offset = 0;
    if (!is_q && token < num_slots) {
      const int64_t slot = slot_mapping[token];
      if (slot >= 0) {
        int64_t block;
        int64_t page;
        if (block_size_shift >= 0) {
          // vLLM production cache blocks are powers of two. Avoid the costly
          // runtime int64 division on that path while retaining generic block
          // sizes through the fallback below.
          block = slot >> block_size_shift;
          page = slot & (block_size - 1);
        } else {
          block = slot / block_size;
          page = slot - block * block_size;
        }
        cache_offset = block * cache_block_stride + page * cache_page_stride +
                       static_cast<int64_t>(head) * cache_head_stride;
        write_cache = true;
      }
    }

    if (!is_q && !is_k) {
#pragma unroll
      for (int block = 0; block < kNumVectors; ++block) {
#pragma unroll
        for (int element = 0; element < kValuesPerVector; ++element) {
          const int dim = block * kThreadsPerHead * kValuesPerVector +
                          lane * kValuesPerVector + element;
          if constexpr (kTileSize > HEAD_DIM) {
            if (dim >= HEAD_DIM) continue;
          }
          const Scalar value = qkv[input_offset + dim];
          output[output_offset + dim] = value;
          if (write_cache) {
            if constexpr (FP8_CACHE) {
              reinterpret_cast<uint8_t*>(v_cache_void)[cache_offset + dim] =
                  fp8::scaled_convert<uint8_t, float,
                                      Fp8KVCacheDataType::kFp8E4M3>(
                      ScalarConverter::convert(value), *v_scale);
            } else {
              v_cache[cache_offset + dim] = value;
            }
          }
        }
      }
      return;
    }

    float values[kValuesPerThread];
    float sum_squares = 0.0f;
#pragma unroll
    for (int block = 0; block < kNumVectors; ++block) {
#pragma unroll
      for (int element = 0; element < kValuesPerVector; ++element) {
        const int value_index = block * kValuesPerVector + element;
        const int dim = block * kThreadsPerHead * kValuesPerVector +
                        lane * kValuesPerVector + element;
        float value = 0.0f;
        if constexpr (kTileSize > HEAD_DIM) {
          if (dim < HEAD_DIM) {
            value = ScalarConverter::convert(qkv[input_offset + dim]);
          }
        } else {
          value = ScalarConverter::convert(qkv[input_offset + dim]);
        }
        values[value_index] = value;
        // CuTeDSL lowers its fragment reduction to a sequential fma.rn
        // chain. Keep this explicit: changing it changes RMSNorm bits.
        sum_squares = __fmaf_rn(value, value, sum_squares);
      }
    }

    const int warp_lane = threadIdx.x % kWarpSize;
    const int subwarp_base = (warp_lane / kThreadsPerHead) * kThreadsPerHead;
    constexpr uint32_t kSubwarpBits =
        kThreadsPerHead == kWarpSize ? 0xffffffffu
                                     : ((uint32_t{1} << kThreadsPerHead) - 1);
    const uint32_t subwarp_mask = kSubwarpBits << subwarp_base;
#pragma unroll
    for (int offset = 1; offset < kThreadsPerHead; offset <<= 1) {
      // CuTeDSL's warp_reduce uses ascending XOR offsets (1, 2, 4, ...).
      sum_squares +=
          __shfl_xor_sync(subwarp_mask, sum_squares, offset, kWarpSize);
    }
    const float mean_square =
        __fdiv_rn(sum_squares, static_cast<float>(HEAD_DIM));
    const float inverse_rms = rsqrtf(mean_square + epsilon);
    const Weight* weight = is_q ? q_weight : k_weight;

    // CuTeDSL writes RMSNorm results to activation-typed shared memory and
    // reloads them for RoPE. This intermediate FP16/BF16 rounding boundary
    // is intentional and is necessary for bitwise output parity.
    __shared__ Scalar normalized[kHeadsPerBlock * HEAD_DIM];
    Scalar* normalized_head = normalized + head_in_block * HEAD_DIM;
#pragma unroll
    for (int block = 0; block < kNumVectors; ++block) {
#pragma unroll
      for (int element = 0; element < kValuesPerVector; ++element) {
        const int value_index = block * kValuesPerVector + element;
        const int dim = block * kThreadsPerHead * kValuesPerVector +
                        lane * kValuesPerVector + element;
        if constexpr (kTileSize > HEAD_DIM) {
          if (dim >= HEAD_DIM) continue;
        }
        const float x_hat = values[value_index] * inverse_rms;
        const float norm_weight =
            WeightConverter::convert(weight[dim]) + norm_weight_bias;
        normalized_head[dim] = ScalarConverter::convert(x_hat * norm_weight);
      }
    }

    __syncwarp(subwarp_mask);

    // Apply NeoX-style RoPE. CuTeDSL assigns four rotary pairs to each
    // participating lane and reloads activation-typed normalized values.
    constexpr int kRotaryPairsPerThread = 4;
    if (lane < rotary_pairs / kRotaryPairsPerThread) {
      const int64_t position = positions[token];
      const Rope* token_cos = cos + position * cos_token_stride;
      const Rope* token_sin = sin + position * sin_token_stride;
#pragma unroll
      for (int element = 0; element < kRotaryPairsPerThread; ++element) {
        const int pair = lane * kRotaryPairsPerThread + element;
        const int dim0 = pair;
        const int dim1 = pair + rotary_pairs;
        const float value0 = ScalarConverter::convert(normalized_head[dim0]);
        const float value1 = ScalarConverter::convert(normalized_head[dim1]);
        const float cos_value = RopeConverter::convert(token_cos[pair]);
        const float sin_value = RopeConverter::convert(token_sin[pair]);

        // Match CuTeDSL PTX exactly. The first half is mul + mul + sub; the
        // second is one mul followed by one fused multiply-add.
        const float rotated0 = __fsub_rn(__fmul_rn(value0, cos_value),
                                         __fmul_rn(value1, sin_value));
        const float rotated1 =
            __fmaf_rn(value0, sin_value, __fmul_rn(value1, cos_value));
        normalized_head[dim0] = ScalarConverter::convert(rotated0);
        normalized_head[dim1] = ScalarConverter::convert(rotated1);
      }
    }

    __syncwarp(subwarp_mask);

#pragma unroll
    for (int block = 0; block < kNumVectors; ++block) {
#pragma unroll
      for (int element = 0; element < kValuesPerVector; ++element) {
        const int dim = block * kThreadsPerHead * kValuesPerVector +
                        lane * kValuesPerVector + element;
        if constexpr (kTileSize > HEAD_DIM) {
          if (dim >= HEAD_DIM) continue;
        }
        const Scalar value = normalized_head[dim];
        output[output_offset + dim] = value;
        if (write_cache) {
          if constexpr (FP8_CACHE) {
            reinterpret_cast<uint8_t*>(k_cache_void)[cache_offset + dim] =
                fp8::scaled_convert<uint8_t, float,
                                    Fp8KVCacheDataType::kFp8E4M3>(
                    ScalarConverter::convert(value), *k_scale);
          } else {
            k_cache[cache_offset + dim] = value;
          }
        }
      }
    }

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
  }
#endif
}

template <typename scalar_t, typename weight_t, typename rope_t, bool FP8_CACHE>
void launch_qknorm_rope_cache(
    const torch::stable::Tensor& qkv, torch::stable::Tensor& q_out,
    torch::stable::Tensor& k_out, torch::stable::Tensor& v_out,
    const torch::stable::Tensor& q_weight,
    const torch::stable::Tensor& k_weight, const torch::stable::Tensor& cos,
    const torch::stable::Tensor& sin, const torch::stable::Tensor& positions,
    const torch::stable::Tensor& slot_mapping, torch::stable::Tensor& k_cache,
    torch::stable::Tensor& v_cache, int head_dim, int num_q_heads,
    int num_kv_heads, int rotary_pairs, float epsilon, float norm_weight_bias,
    cudaStream_t stream, const float* k_scale, const float* v_scale) {
  const int total_heads = num_q_heads + 2 * num_kv_heads;
  const int64_t total_work = qkv.size(0) * total_heads;
  const int block_size = static_cast<int>(k_cache.size(1));
  const int block_size_shift =
      (block_size & (block_size - 1)) == 0
          ? __builtin_ctz(static_cast<unsigned int>(block_size))
          : -1;

#define LAUNCH_QKNORM_ROPE_CACHE(HEAD_DIM, WIDE_192_ROTARY)                    \
  do {                                                                         \
    constexpr int threads_per_head =                                           \
        WIDE_192_ROTARY ? 32                                                   \
                        : (HEAD_DIM <= 64 ? 8 : (HEAD_DIM <= 192 ? 16 : 32));  \
    constexpr int heads_per_block = kThreadsPerBlock / threads_per_head;       \
    const dim3 grid((total_work + heads_per_block - 1) / heads_per_block);     \
    const dim3 block(kThreadsPerBlock);                                        \
    qknorm_rope_cache_kernel<scalar_t, weight_t, rope_t, HEAD_DIM,             \
                             WIDE_192_ROTARY, FP8_CACHE>                       \
        <<<grid, block, 0, stream>>>(                                          \
            qkv.data_ptr(), q_out.data_ptr(), k_out.data_ptr(),                \
            v_out.data_ptr(), q_weight.data_ptr(), k_weight.data_ptr(),        \
            cos.data_ptr(), sin.data_ptr(),                                    \
            reinterpret_cast<const int64_t*>(positions.data_ptr()),            \
            reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),         \
            k_cache.data_ptr(), v_cache.data_ptr(), qkv.stride(0),             \
            q_out.stride(0), k_out.stride(0), v_out.stride(0), cos.stride(0),  \
            sin.stride(0), k_cache.stride(0), k_cache.stride(1),               \
            k_cache.stride(2), static_cast<int>(qkv.size(0)),                  \
            static_cast<int>(slot_mapping.numel()), num_q_heads, num_kv_heads, \
            rotary_pairs, block_size, epsilon, block_size_shift,               \
            norm_weight_bias, k_scale, v_scale);                               \
  } while (false)

  switch (head_dim) {
    case 64:
      LAUNCH_QKNORM_ROPE_CACHE(64, false);
      break;
    case 128:
      LAUNCH_QKNORM_ROPE_CACHE(128, false);
      break;
    case 192:
      if (rotary_pairs > 64) {
        LAUNCH_QKNORM_ROPE_CACHE(192, true);
      } else {
        LAUNCH_QKNORM_ROPE_CACHE(192, false);
      }
      break;
    case 256:
      LAUNCH_QKNORM_ROPE_CACHE(256, false);
      break;
    default:
      STD_TORCH_CHECK(false, "Unsupported head_dim: ", head_dim);
  }
#undef LAUNCH_QKNORM_ROPE_CACHE
}

void check_cuda_tensor(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_same_device(const torch::stable::Tensor& reference,
                       const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.get_device_index() == reference.get_device_index(),
                  name, " must be on the same CUDA device as qkv");
}

}  // namespace

void fused_qknorm_rope_cache(
    torch::stable::Tensor& q_out, torch::stable::Tensor& k_out,
    torch::stable::Tensor& v_out, torch::stable::Tensor& qkv,
    torch::stable::Tensor& q_weight, torch::stable::Tensor& k_weight,
    torch::stable::Tensor& cos, torch::stable::Tensor& sin,
    torch::stable::Tensor& positions, torch::stable::Tensor& slot_mapping,
    torch::stable::Tensor& k_cache, torch::stable::Tensor& v_cache,
    int64_t head_dim, int64_t num_q_heads, int64_t num_kv_heads,
    int64_t rotary_pairs, double epsilon, double norm_weight_bias,
    const std::string& kv_cache_dtype,
    std::optional<torch::stable::Tensor> k_scale,
    std::optional<torch::stable::Tensor> v_scale) {
  check_cuda_tensor(qkv, "qkv");
  check_cuda_tensor(q_out, "q_out");
  check_cuda_tensor(k_out, "k_out");
  check_cuda_tensor(v_out, "v_out");
  check_cuda_tensor(q_weight, "q_weight");
  check_cuda_tensor(k_weight, "k_weight");
  check_cuda_tensor(cos, "cos");
  check_cuda_tensor(sin, "sin");
  check_cuda_tensor(positions, "positions");
  check_cuda_tensor(slot_mapping, "slot_mapping");
  check_cuda_tensor(k_cache, "k_cache");
  check_cuda_tensor(v_cache, "v_cache");
  check_same_device(qkv, q_out, "q_out");
  check_same_device(qkv, k_out, "k_out");
  check_same_device(qkv, v_out, "v_out");
  check_same_device(qkv, q_weight, "q_weight");
  check_same_device(qkv, k_weight, "k_weight");
  check_same_device(qkv, cos, "cos");
  check_same_device(qkv, sin, "sin");
  check_same_device(qkv, positions, "positions");
  check_same_device(qkv, slot_mapping, "slot_mapping");
  check_same_device(qkv, k_cache, "k_cache");
  check_same_device(qkv, v_cache, "v_cache");

  STD_TORCH_CHECK(qkv.dim() == 2 && qkv.stride(1) == 1,
                  "qkv must be a 2D tensor contiguous in its last dimension");
  STD_TORCH_CHECK(q_out.dim() == 2 && q_out.is_contiguous(),
                  "q_out must be contiguous and 2D");
  STD_TORCH_CHECK(k_out.dim() == 2 && k_out.is_contiguous(),
                  "k_out must be contiguous and 2D");
  STD_TORCH_CHECK(v_out.dim() == 2 && v_out.is_contiguous(),
                  "v_out must be contiguous and 2D");
  STD_TORCH_CHECK(q_weight.dim() == 1 && q_weight.is_contiguous(),
                  "q_weight must be contiguous and 1D");
  STD_TORCH_CHECK(k_weight.dim() == 1 && k_weight.is_contiguous(),
                  "k_weight must be contiguous and 1D");
  STD_TORCH_CHECK(cos.dim() == 2 && cos.stride(1) == 1,
                  "cos must be 2D and contiguous in its last dimension");
  STD_TORCH_CHECK(sin.dim() == 2 && sin.stride(1) == 1,
                  "sin must be 2D and contiguous in its last dimension");
  STD_TORCH_CHECK(
      positions.scalar_type() == torch::headeronly::ScalarType::Long,
      "positions must be an int64 tensor");
  STD_TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  STD_TORCH_CHECK(
      slot_mapping.scalar_type() == torch::headeronly::ScalarType::Long,
      "slot_mapping must be an int64 tensor");
  STD_TORCH_CHECK(slot_mapping.is_contiguous(),
                  "slot_mapping must be contiguous");
  STD_TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
                  "K/V cache must have shape [blocks, block_size, heads, dim]");
  for (int64_t d = 0; d < 4; ++d) {
    STD_TORCH_CHECK(k_cache.size(d) == v_cache.size(d) &&
                        k_cache.stride(d) == v_cache.stride(d),
                    "K/V cache must have identical shapes and strides");
  }
  STD_TORCH_CHECK(k_cache.stride(3) == 1,
                  "K/V cache head dimension must be contiguous");

  STD_TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0,
                  "Q/KV head counts must be positive");
  STD_TORCH_CHECK(
      head_dim == 64 || head_dim == 128 || head_dim == 192 || head_dim == 256,
      "Unsupported head_dim");
  STD_TORCH_CHECK(cos.size(0) == sin.size(0),
                  "cos/sin cache must have the same number of positions");
  const int64_t num_tokens = qkv.size(0);
  const int64_t packed_width = (num_q_heads + 2 * num_kv_heads) * head_dim;
  STD_TORCH_CHECK(qkv.size(1) == packed_width,
                  "qkv width must equal (num_q_heads + 2 * num_kv_heads) * "
                  "head_dim");
  STD_TORCH_CHECK(
      q_out.size(0) == num_tokens && q_out.size(1) == num_q_heads * head_dim,
      "q_out has an invalid shape");
  STD_TORCH_CHECK(
      k_out.size(0) == num_tokens && k_out.size(1) == num_kv_heads * head_dim,
      "k_out has an invalid shape");
  STD_TORCH_CHECK(
      v_out.size(0) == k_out.size(0) && v_out.size(1) == k_out.size(1),
      "v_out has an invalid shape");
  STD_TORCH_CHECK(q_weight.numel() == head_dim && k_weight.numel() == head_dim,
                  "Q/K norm weights must have head_dim elements");
  STD_TORCH_CHECK(positions.numel() == num_tokens,
                  "positions must have one entry per qkv token");
  STD_TORCH_CHECK(rotary_pairs >= 0 && rotary_pairs * 2 <= head_dim &&
                      rotary_pairs % 4 == 0,
                  "rotary_pairs must be a multiple of 4 in [0, head_dim / 2]");
  STD_TORCH_CHECK(cos.size(1) >= rotary_pairs && sin.size(1) >= rotary_pairs,
                  "cos/sin cache is smaller than rotary_pairs");
  STD_TORCH_CHECK(
      k_cache.size(2) == num_kv_heads && k_cache.size(3) == head_dim,
      "K/V cache head shape does not match the kernel arguments");
  STD_TORCH_CHECK(k_cache.size(1) > 0, "K/V cache block_size must be positive");
  STD_TORCH_CHECK(slot_mapping.numel() <= num_tokens,
                  "slot_mapping cannot contain more entries than qkv tokens");
  STD_TORCH_CHECK(qkv.scalar_type() == q_out.scalar_type() &&
                      qkv.scalar_type() == k_out.scalar_type() &&
                      qkv.scalar_type() == v_out.scalar_type(),
                  "qkv and Q/K/V outputs must have the same dtype");
  const bool fp8_cache =
      kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3";
  STD_TORCH_CHECK(fp8_cache || kv_cache_dtype == "auto" ||
                      kv_cache_dtype == "bfloat16" ||
                      kv_cache_dtype == "float16",
                  "Unsupported KV cache dtype: ", kv_cache_dtype);
  STD_TORCH_CHECK(k_cache.scalar_type() == v_cache.scalar_type(),
                  "K/V caches must have the same dtype");
  if (fp8_cache) {
    STD_TORCH_CHECK(
        k_cache.scalar_type() == torch::headeronly::ScalarType::Byte ||
            k_cache.scalar_type() ==
                torch::headeronly::ScalarType::Float8_e4m3fn,
        "FP8 cache requires uint8 or float8_e4m3fn storage");
    STD_TORCH_CHECK(k_scale.has_value() && v_scale.has_value(),
                    "FP8 cache requires K/V scales");
    for (const auto* scale : {&*k_scale, &*v_scale}) {
      check_cuda_tensor(*scale, "KV scale");
      check_same_device(qkv, *scale, "KV scale");
      STD_TORCH_CHECK(
          scale->numel() == 1 &&
              scale->scalar_type() == torch::headeronly::ScalarType::Float,
          "KV scales must be single FP32 values");
    }
  } else {
    STD_TORCH_CHECK(qkv.scalar_type() == k_cache.scalar_type(),
                    "Unquantized cache must match QKV dtype");
  }
  STD_TORCH_CHECK(q_weight.scalar_type() == k_weight.scalar_type(),
                  "Q/K norm weights must have the same dtype");
  STD_TORCH_CHECK(cos.scalar_type() == sin.scalar_type(),
                  "cos and sin must have the same dtype");

  if (num_tokens == 0) return;

  const torch::stable::accelerator::DeviceGuard device_guard(
      qkv.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(qkv.get_device_index());
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "qknorm_rope_cache_input", [&] {
        using input_t = scalar_t;
        VLLM_STABLE_DISPATCH_FLOATING_TYPES(
            q_weight.scalar_type(), "qknorm_rope_cache_weight", [&] {
              using norm_weight_t = scalar_t;
              VLLM_STABLE_DISPATCH_FLOATING_TYPES(
                  cos.scalar_type(), "qknorm_rope_cache_rope", [&] {
                    using rope_cache_t = scalar_t;
                    auto launch = [&](auto fp8_tag) {
                      launch_qknorm_rope_cache<input_t, norm_weight_t,
                                               rope_cache_t,
                                               decltype(fp8_tag)::value>(
                          qkv, q_out, k_out, v_out, q_weight, k_weight, cos,
                          sin, positions, slot_mapping, k_cache, v_cache,
                          static_cast<int>(head_dim),
                          static_cast<int>(num_q_heads),
                          static_cast<int>(num_kv_heads),
                          static_cast<int>(rotary_pairs),
                          static_cast<float>(epsilon),
                          static_cast<float>(norm_weight_bias), stream,
                          fp8_cache
                              ? static_cast<const float*>(k_scale->data_ptr())
                              : nullptr,
                          fp8_cache
                              ? static_cast<const float*>(v_scale->data_ptr())
                              : nullptr);
                    };
                    if (fp8_cache) {
                      launch(std::true_type{});
                    } else {
                      launch(std::false_type{});
                    }
                  });
            });
      });
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm
