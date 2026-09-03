/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#include <cstdint>
#include <string>
#ifdef USE_ROCM
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include "../torch_utils.h"
#ifndef USE_ROCM
#include "../../cuda_compat.h"
#endif

namespace {

#ifdef USE_ROCM
using bf16_t = __hip_bfloat16;
#else
using bf16_t = __nv_bfloat16;
#endif

__device__ __forceinline__ float bf16_to_float(bf16_t value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ bf16_t float_to_bf16(float value) {
  return __float2bfloat16(value);
}

template <typename StateT>
__device__ __forceinline__ void cp_async_16b(StateT* smem_ptr,
                                             const StateT* gmem_ptr) {
#ifdef USE_ROCM
  *reinterpret_cast<int4*>(smem_ptr) =
      *reinterpret_cast<const int4*>(gmem_ptr);
#else
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_addr), "l"(gmem_ptr));
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#ifndef USE_ROCM
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#ifndef USE_ROCM
  asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

template <typename StateT, int ChunkV, int DimK, int Stages>
__device__ __forceinline__ void copy_state_chunk(StateT* shared_state,
                                                 const StateT* state, int chunk,
                                                 int thread, int threads) {
  constexpr int kElementsPerCopy = 16 / sizeof(StateT);
  constexpr int kCopiesPerChunk = ChunkV * DimK / kElementsPerCopy;
  const int stage = chunk % Stages;
  for (int copy = thread; copy < kCopiesPerChunk; copy += threads) {
    const int element = copy * kElementsPerCopy;
    cp_async_16b(shared_state + stage * ChunkV * DimK + element,
                 state + chunk * ChunkV * DimK + element);
  }
  cp_async_commit();
}

template <typename StateT>
__device__ __forceinline__ float4 load_state4(const StateT* state);

template <>
__device__ __forceinline__ float4 load_state4<float>(const float* state) {
  return *reinterpret_cast<const float4*>(state);
}

template <>
__device__ __forceinline__ float4 load_state4<bf16_t>(const bf16_t* state) {
  return make_float4(bf16_to_float(state[0]), bf16_to_float(state[1]),
                     bf16_to_float(state[2]), bf16_to_float(state[3]));
}

template <typename StateT>
__device__ __forceinline__ void store_state4(StateT* state, float4 value);

template <>
__device__ __forceinline__ void store_state4<float>(float* state,
                                                    float4 value) {
  *reinterpret_cast<float4*>(state) = value;
}

template <>
__device__ __forceinline__ void store_state4<bf16_t>(bf16_t* state,
                                                     float4 value) {
  state[0] = float_to_bf16(value.x);
  state[1] = float_to_bf16(value.y);
  state[2] = float_to_bf16(value.z);
  state[3] = float_to_bf16(value.w);
}

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxMtpTokens = 8;
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head,
                                              int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) {
    return bf16_to_float(static_cast<const bf16_t*>(dt_bias)[head]);
  }
  if (dt_bias_type == kDtBiasFloat16) {
    return __half2float(static_cast<const __half*>(dt_bias)[head]);
  }
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#ifdef USE_ROCM
    value += __shfl_xor(value, offset, 32);
#else
    value += __shfl_xor_sync(0xffffffffu, value, offset);
#endif
  }
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ float warp_broadcast(float value, int source_lane) {
#ifdef USE_ROCM
  return __shfl(value, source_lane, 32);
#else
  return __shfl_sync(0xffffffffu, value, source_lane);
#endif
}

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#ifdef USE_ROCM
    x += __shfl_xor(x, offset, 32);
    y += __shfl_xor(y, offset, 32);
#else
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
#endif
  }
  return {x, y};
}

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
__global__ __launch_bounds__(kThreads, 2) void gdn_decode_post_conv_mtp_kernel(
    const bf16_t* __restrict__ mixed_qkv,
    const bf16_t* __restrict__ a, const bf16_t* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, StateT* __restrict__ state,
    const bf16_t* __restrict__ output_gate,
    const void* __restrict__ norm_weight, bf16_t* __restrict__ out,
    int H, int HV, int state_indices_width, int dt_bias_type,
    bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) {
    return;
  }

  const int accepted = num_accepted_tokens[request];
  const int source_slot =
      accepted > 0 && accepted <= state_indices_width
          ? state_indices[request * state_indices_width + accepted - 1]
          : 0;
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = float_to_bf16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ StateT shared_state[2][kChunkV][kDimK];
  __shared__ float shared_q[kMaxMtpTokens][kDimK];
  __shared__ float shared_k[kMaxMtpTokens][kDimK];
  __shared__ bf16_t shared_v[kMaxMtpTokens][kDimV];
  __shared__ bf16_t shared_out[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];

  StateT* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot +
      value_head * kDimV * kDimK;
  copy_state_chunk<StateT, kChunkV, kDimK, 2>(&shared_state[0][0][0],
                                              source_state, 0, tid, kThreads);

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] =
          bf16_to_float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = bf16_to_float(
          mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = warp_broadcast(
        lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = warp_broadcast(
        lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = bf16_to_float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = bf16_to_float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    rows[row] = warp + row * kWarps;
  }

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) {
      copy_state_chunk<StateT, kChunkV, kDimK, 2>(
          &shared_state[0][0][0], source_state, chunk + 1, tid, kThreads);
    }

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value =
          load_state4(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};

      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y,
                                              dot_hk_23.x, dot_hk_23.y};

      float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta =
            (bf16_to_float(shared_v[t][value]) - reduced_hk[row]) *
            shared_beta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] =
            float_to_bf16(dot_hq_01.x);
        shared_out[t][chunk * kChunkV + rows[1]] =
            float_to_bf16(dot_hq_01.y);
        shared_out[t][chunk * kChunkV + rows[2]] =
            float_to_bf16(dot_hq_23.x);
        shared_out[t][chunk * kChunkV + rows[3]] =
            float_to_bf16(dot_hq_23.y);
      }

      const int destination_slot =
          state_indices[request * state_indices_width + t];
      if (destination_slot > 0) {
        StateT* destination_state =
            state +
            static_cast<int64_t>(destination_slot) * strides.state_slot +
            value_head * kDimV * kDimK;
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
          const int value = chunk * kChunkV + rows[row];
          store_state4(destination_state + value * kDimK + k_base,
                       make_float4(h[row][0], h[row][1], h[row][2], h[row][3]));
        }
      }
    }
  }
  __syncthreads();

  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = bf16_to_float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd =
        rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input = bf16_to_float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row +
                      value_head * kDimV + value]);
      const float gate =
          SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight =
          norm_weight_is_bf16
              ? bf16_to_float(static_cast<const bf16_t*>(norm_weight)[value])
              : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] =
          float_to_bf16(output_values[i] * rstd * weight * gate);
    }
  }
}

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
void launch_gdn_decode_post_conv_mtp(
    torch::stable::Tensor const& mixed_qkv, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices,
    torch::stable::Tensor const& cu_seqlens,
    torch::stable::Tensor const& num_accepted_tokens,
    torch::stable::Tensor& state, torch::stable::Tensor const& norm_weight,
    torch::stable::Tensor& out, const bf16_t* a, const bf16_t* b,
    const bf16_t* output_gate, int num_key_heads, int num_value_heads,
    double scale, double norm_eps, GdnDecodeStrides strides) {
  using torch::headeronly::ScalarType;

  const auto dt_bias_scalar_type = dt_bias.scalar_type();
  const int dt_bias_type =
      dt_bias_scalar_type == ScalarType::Float
          ? kDtBiasFloat32
          : (dt_bias_scalar_type == ScalarType::BFloat16 ? kDtBiasBFloat16
                                                         : kDtBiasFloat16);
  torch::stable::accelerator::DeviceGuard const device_guard(
      mixed_qkv.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(mixed_qkv.get_device_index());
  const int num_requests = static_cast<int>(state_indices.size(0));
  const dim3 grid(num_requests, num_value_heads);
  gdn_decode_post_conv_mtp_kernel<StateT, ValueHeadsPerKeyHead, SigmoidGate>
      <<<grid, kThreads, 0, stream>>>(
          static_cast<const bf16_t*>(mixed_qkv.data_ptr()), a, b,
          static_cast<const float*>(a_log.data_ptr()), dt_bias.data_ptr(),
          static_cast<const int*>(state_indices.data_ptr()),
          static_cast<const int*>(cu_seqlens.data_ptr()),
          static_cast<const int*>(num_accepted_tokens.data_ptr()),
          static_cast<StateT*>(state.data_ptr()), output_gate,
          norm_weight.data_ptr(), static_cast<bf16_t*>(out.data_ptr()),
          num_key_heads, num_value_heads,
          static_cast<int>(state_indices.size(1)), dt_bias_type,
          norm_weight.scalar_type() == ScalarType::BFloat16,
          static_cast<float>(scale), static_cast<float>(norm_eps), strides);
#ifdef USE_ROCM
  const hipError_t error = hipGetLastError();
  STD_TORCH_CHECK(error == hipSuccess,
                  "GDN decode MTP post-conv kernel launch failed: ",
                  hipGetErrorString(error));
#else
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess,
                  "GDN decode MTP post-conv kernel launch failed: ",
                  cudaGetErrorString(error));
#endif
}

}  // namespace

void fused_gdn_decode_post_conv_mtp(
    torch::stable::Tensor const& mixed_qkv, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices,
    torch::stable::Tensor const& cu_seqlens,
    torch::stable::Tensor const& num_accepted_tokens,
    torch::stable::Tensor& state, torch::stable::Tensor const& output_gate,
    torch::stable::Tensor const& norm_weight, torch::stable::Tensor& out,
    double scale, double norm_eps, const std::string& output_gate_activation) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(
      mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == ScalarType::BFloat16,
      "mixed_qkv must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(a.is_cuda() && a.scalar_type() == ScalarType::BFloat16,
                  "a must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(b.is_cuda() && b.scalar_type() == ScalarType::BFloat16,
                  "b must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == ScalarType::Float,
                  "A_log must be a CUDA float32 tensor");
  const auto dt_bias_scalar_type = dt_bias.scalar_type();
  STD_TORCH_CHECK(
      dt_bias.is_cuda() && (dt_bias_scalar_type == ScalarType::Float ||
                            dt_bias_scalar_type == ScalarType::BFloat16 ||
                            dt_bias_scalar_type == ScalarType::Half),
      "dt_bias must be a CUDA float32, bfloat16, or float16 tensor");
  STD_TORCH_CHECK(
      state_indices.is_cuda() && state_indices.scalar_type() == ScalarType::Int,
      "state_indices must be a CUDA int32 tensor");
  STD_TORCH_CHECK(
      cu_seqlens.is_cuda() && cu_seqlens.scalar_type() == ScalarType::Int,
      "cu_seqlens must be a CUDA int32 tensor");
  STD_TORCH_CHECK(num_accepted_tokens.is_cuda() &&
                      num_accepted_tokens.scalar_type() == ScalarType::Int,
                  "num_accepted_tokens must be a CUDA int32 tensor");
  const auto state_scalar_type = state.scalar_type();
  STD_TORCH_CHECK(
      state.is_cuda() && (state_scalar_type == ScalarType::Float ||
                          state_scalar_type == ScalarType::BFloat16),
      "state must be a CUDA float32 or bfloat16 tensor");
  STD_TORCH_CHECK(output_gate.is_cuda() &&
                      output_gate.scalar_type() == ScalarType::BFloat16,
                  "output_gate must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(norm_weight.is_cuda() &&
                      (norm_weight.scalar_type() == ScalarType::Float ||
                       norm_weight.scalar_type() == ScalarType::BFloat16),
                  "norm_weight must be a CUDA float32 or bfloat16 tensor");
  STD_TORCH_CHECK(out.is_cuda() && out.scalar_type() == ScalarType::BFloat16,
                  "out must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(
      output_gate_activation == "silu" || output_gate_activation == "sigmoid",
      "output_gate_activation must be 'silu' or 'sigmoid'");

  STD_TORCH_CHECK(mixed_qkv.dim() == 2,
                  "mixed_qkv must have shape [L, 2 * H * 128 + HV * 128]");
  const int num_tokens = static_cast<int>(mixed_qkv.size(0));
  STD_TORCH_CHECK(num_tokens > 0,
                  "GDN decode MTP fusion requires at least one token");
  STD_TORCH_CHECK(
      state.dim() == 4 && state.size(2) == kDimV && state.size(3) == kDimK,
      "state must have shape [slots, HV, 128, 128]");
  const int num_value_heads = static_cast<int>(state.size(1));
  const int64_t key_width =
      mixed_qkv.size(1) - static_cast<int64_t>(num_value_heads) * kDimV;
  STD_TORCH_CHECK(key_width > 0 && key_width % (2 * kDimK) == 0,
                  "mixed_qkv width is inconsistent with state");
  const int num_key_heads = static_cast<int>(key_width / (2 * kDimK));
  const int value_heads_per_key_head = num_value_heads / num_key_heads;
  STD_TORCH_CHECK(
      num_value_heads % num_key_heads == 0 &&
          ((value_heads_per_key_head >= 1 && value_heads_per_key_head <= 4) ||
           value_heads_per_key_head == 8),
      "GDN decode MTP fusion requires HV/H in {1, 2, 3, 4, 8}");

  STD_TORCH_CHECK(state_indices.dim() == 2 && state_indices.size(0) > 0 &&
                      state_indices.size(1) > 0 &&
                      state_indices.size(1) <= kMaxMtpTokens,
                  "state_indices must have shape [N, S] with 1 <= S <= 8");
  const int num_requests = static_cast<int>(state_indices.size(0));
  STD_TORCH_CHECK(
      cu_seqlens.dim() == 1 && cu_seqlens.numel() == num_requests + 1,
      "cu_seqlens must have N + 1 elements");
  STD_TORCH_CHECK(num_accepted_tokens.dim() == 1 &&
                      num_accepted_tokens.numel() == num_requests,
                  "num_accepted_tokens must have N elements");
  STD_TORCH_CHECK(
      a.dim() == 2 && a.size(0) == num_tokens && a.size(1) == num_value_heads,
      "a must have shape [L, HV]");
  STD_TORCH_CHECK(
      b.dim() == 2 && b.size(0) == num_tokens && b.size(1) == num_value_heads,
      "b must have shape [L, HV]");
  STD_TORCH_CHECK(a_log.is_contiguous() && a_log.numel() == num_value_heads,
                  "A_log must be contiguous with HV elements");
  STD_TORCH_CHECK(dt_bias.is_contiguous() && dt_bias.numel() == num_value_heads,
                  "dt_bias must be contiguous with HV elements");
  STD_TORCH_CHECK(state_indices.is_contiguous(),
                  "state_indices must be contiguous");
  STD_TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
  STD_TORCH_CHECK(num_accepted_tokens.is_contiguous(),
                  "num_accepted_tokens must be contiguous");
  STD_TORCH_CHECK(output_gate.dim() == 3 && output_gate.size(0) == num_tokens &&
                      output_gate.size(1) == num_value_heads &&
                      output_gate.size(2) == kDimV,
                  "output_gate must have shape [L, HV, 128]");
  STD_TORCH_CHECK(norm_weight.is_contiguous() && norm_weight.numel() == kDimV,
                  "norm_weight must be contiguous with 128 elements");
  STD_TORCH_CHECK(out.dim() == 3 && out.size(0) == num_tokens &&
                      out.size(1) == num_value_heads && out.size(2) == kDimV,
                  "out must have shape [L, HV, 128]");
  STD_TORCH_CHECK(mixed_qkv.stride(1) == 1,
                  "mixed_qkv channels must be contiguous");
  STD_TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1,
                  "a and b heads must be contiguous");
  STD_TORCH_CHECK(state.stride(0) >= num_value_heads * kDimV * kDimK &&
                      state.stride(1) == kDimV * kDimK &&
                      state.stride(2) == kDimK && state.stride(3) == 1,
                  "state must have contiguous [HV, 128, 128] slot contents");
  const int state_elements_per_copy =
      state_scalar_type == ScalarType::Float ? 4 : 8;
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
                      state.stride(0) % state_elements_per_copy == 0,
                  "state slots must preserve 16-byte alignment");
  STD_TORCH_CHECK(output_gate.stride(2) == 1 && output_gate.stride(1) == kDimV,
                  "output_gate head rows must be contiguous");
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  STD_TORCH_CHECK(norm_eps >= 0.0, "norm_eps must be non-negative");

  const GdnDecodeStrides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0),
                                 output_gate.stride(0), state.stride(0)};
  const auto* a_ptr = static_cast<const bf16_t*>(a.data_ptr());
  const auto* b_ptr = static_cast<const bf16_t*>(b.data_ptr());
  const auto* output_gate_ptr =
      static_cast<const bf16_t*>(output_gate.data_ptr());
  const auto launch = [&]<typename StateT, int ValueHeadsPerKeyHead,
                          bool SigmoidGate>() {
    launch_gdn_decode_post_conv_mtp<StateT, ValueHeadsPerKeyHead, SigmoidGate>(
        mixed_qkv, a_log, dt_bias, state_indices, cu_seqlens,
        num_accepted_tokens, state, norm_weight, out, a_ptr, b_ptr,
        output_gate_ptr, num_key_heads, num_value_heads, scale, norm_eps,
        strides);
  };
  const auto dispatch_state_type = [&]<int ValueHeadsPerKeyHead>() {
    if (state_scalar_type == ScalarType::Float) {
      if (output_gate_activation == "sigmoid") {
        launch.template operator()<float, ValueHeadsPerKeyHead, true>();
      } else {
        launch.template operator()<float, ValueHeadsPerKeyHead, false>();
      }
    } else {
      if (output_gate_activation == "sigmoid") {
        launch.template operator()<bf16_t, ValueHeadsPerKeyHead, true>();
      } else {
        launch
            .template operator()<bf16_t, ValueHeadsPerKeyHead, false>();
      }
    }
  };
  switch (value_heads_per_key_head) {
    case 1:
      dispatch_state_type.template operator()<1>();
      break;
    case 2:
      dispatch_state_type.template operator()<2>();
      break;
    case 3:
      dispatch_state_type.template operator()<3>();
      break;
    case 4:
      dispatch_state_type.template operator()<4>();
      break;
    default:
      dispatch_state_type.template operator()<8>();
      break;
  }
}
