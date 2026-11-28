/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#include <cstdint>
#include <string>
#include <type_traits>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../torch_utils.h"
#include "core/cuda_compat.h"
#include "fused_gdn_decode.h"

namespace {

namespace cg = cooperative_groups;

template <typename StateT>
__device__ __forceinline__ void cp_async_16b(StateT* smem_ptr,
                                             const StateT* gmem_ptr) {
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
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
__device__ __forceinline__ float4
load_state4<__nv_bfloat16>(const __nv_bfloat16* state) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(state);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(state + 2);
  return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y),
                     __bfloat162float(hi.x), __bfloat162float(hi.y));
}

template <>
__device__ __forceinline__ float4 load_state4<__half>(const __half* state) {
  const __half2 lo = *reinterpret_cast<const __half2*>(state);
  const __half2 hi = *reinterpret_cast<const __half2*>(state + 2);
  return make_float4(__half2float(lo.x), __half2float(lo.y), __half2float(hi.x),
                     __half2float(hi.y));
}

template <typename StateT>
__device__ __forceinline__ void store_state4(StateT* state, float4 value);

template <>
__device__ __forceinline__ void store_state4<float>(float* state,
                                                    float4 value) {
  *reinterpret_cast<float4*>(state) = value;
}

template <>
__device__ __forceinline__ void store_state4<__nv_bfloat16>(
    __nv_bfloat16* state, float4 value) {
  *reinterpret_cast<__nv_bfloat162*>(state) =
      __floats2bfloat162_rn(value.x, value.y);
  *reinterpret_cast<__nv_bfloat162*>(state + 2) =
      __floats2bfloat162_rn(value.z, value.w);
}

template <>
__device__ __forceinline__ void store_state4<__half>(__half* state,
                                                     float4 value) {
  *reinterpret_cast<__half2*>(state) = __floats2half2_rn(value.x, value.y);
  *reinterpret_cast<__half2*>(state + 2) = __floats2half2_rn(value.z, value.w);
}

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kTokenPasses = (kMaxMtpTokens + kWarps - 1) / kWarps;

// Dynamic shared memory holds one per token of the launch's window.
struct alignas(16) TokenSmem {
  float q[kDimK];
  float k[kDimK];
  __nv_bfloat16 v[kDimV];
  __nv_bfloat16 out[kDimV];
  float decay;
  float beta;
  float sum_square;
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
    return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  }
  if (dt_bias_type == kDtBiasFloat16) {
    return __half2float(static_cast<const __half*>(dt_bias)[head]);
  }
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

// SplitV > 1 splits the V rows of one (request, value head) over a cluster of
// SplitV CTAs along blockIdx.z, so small batches fill more SMs. The gated
// RMSNorm over V then needs the partial sums of the other CTAs (via DSMEM).
template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate,
          int SplitV>
__global__ __launch_bounds__(kThreads, 2) void gdn_decode_post_conv_mtp_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, StateT* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate,
    const void* __restrict__ norm_weight, __nv_bfloat16* __restrict__ out,
    int H, int HV, int state_indices_width, int dt_bias_type,
    bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
  // Reduce binary size on architectures without cluster support.
  if constexpr (SplitV > 1) {
    __trap();
    return;
  }
#endif
  constexpr int kChunksPerCta = kNumChunks / SplitV;
  constexpr int kSliceV = kDimV / SplitV;
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int split_rank = SplitV == 1 ? 0 : static_cast<int>(blockIdx.z);
  const int first_chunk = split_rank * kChunksPerCta;
  const int slice_begin = split_rank * kSliceV;
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
  if (source_slot <= 0 || num_tokens > state_indices_width) {
    for (int linear = tid; linear < num_tokens * kSliceV; linear += kThreads) {
      const int token = bos + linear / kSliceV;
      const int value = slice_begin + linear % kSliceV;
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ StateT shared_state[2][kChunkV][kDimK];
  extern __shared__ TokenSmem tokens[];

  StateT* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot +
      value_head * kDimV * kDimK;
  copy_state_chunk<StateT, kChunkV, kDimK, 2>(
      &shared_state[0][0][0], source_state, first_chunk, tid, kThreads);

  // One warp per token. kMaxMtpTokens exceeds kWarps, so warps stride.
  // kTokenPasses bounds the trip count, so the compiler unrolls the passes.
#pragma unroll
  for (int pass = 0, t = warp; pass < kTokenPasses && t < num_tokens;
       ++pass, t += kWarps) {
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
          __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(
          mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      tokens[t].v[dim] =
          mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      tokens[t].q[dim] = q_values[i] * q_scale;
      tokens[t].k[dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      tokens[t].decay = __expf(g);
      tokens[t].beta = sigmoid_fast(b_value);
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
  for (int cta_chunk = 0; cta_chunk < kChunksPerCta; ++cta_chunk) {
    const int chunk = first_chunk + cta_chunk;
    cp_async_wait_all();
    __syncthreads();
    if (cta_chunk + 1 < kChunksPerCta) {
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
      const float4 q4 = *reinterpret_cast<const float4*>(&tokens[t].q[k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&tokens[t].k[k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};

      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= tokens[t].decay;
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
            (__bfloat162float(tokens[t].v[value]) - reduced_hk[row]) *
            tokens[t].beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        tokens[t].out[chunk * kChunkV + rows[0]] =
            __float2bfloat16(dot_hq_01.x);
        tokens[t].out[chunk * kChunkV + rows[1]] =
            __float2bfloat16(dot_hq_01.y);
        tokens[t].out[chunk * kChunkV + rows[2]] =
            __float2bfloat16(dot_hq_23.x);
        tokens[t].out[chunk * kChunkV + rows[3]] =
            __float2bfloat16(dot_hq_23.y);
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

  constexpr int kValuesPerLane = kSliceV / 32;
  const auto write_token = [&](int t, float sum_square) {
    const float rstd =
        rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      const int value = slice_begin + lane + i * 32;
      const float output_value = __bfloat162float(tokens[t].out[value]);
      const float gate_input = __bfloat162float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row +
                      value_head * kDimV + value]);
      const float gate =
          SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight =
          norm_weight_is_bf16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(norm_weight)[value])
              : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(output_value * rstd * weight * gate);
    }
  };

#pragma unroll
  for (int pass = 0, t = warp; pass < kTokenPasses && t < num_tokens;
       ++pass, t += kWarps) {
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      const float value =
          __bfloat162float(tokens[t].out[slice_begin + lane + i * 32]);
      sum_square += value * value;
    }
    sum_square = warp_reduce_sum(sum_square);
    if constexpr (SplitV == 1) {
      write_token(t, sum_square);
    } else if (lane == 0) {
      tokens[t].sum_square = sum_square;
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // Sum the partial sums of squares of all CTAs in the cluster via DSMEM.
  if constexpr (SplitV > 1) {
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    for (int t = warp; t < num_tokens; t += kWarps) {
      float sum_square = 0.0f;
  #pragma unroll
      for (int rank = 0; rank < SplitV; ++rank) {
        sum_square += *cluster.map_shared_rank(&tokens[t].sum_square, rank);
      }
      write_token(t, sum_square);
    }
    cluster.sync();  // keep sum_square alive for remote readers
  }
#endif
}

// Split V over a cluster while the split grid still fits in one wave (2 CTAs
// per SM, or 1 if two don't fit in its shared memory). On B300 this picks 4 up
// to 2 requests and 2 at 4 requests (value heads HV=32).
int choose_split_v(int num_blocks, int cta_smem_bytes, int device) {
  int major = 0, num_sms = 0, sm_smem_bytes = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&sm_smem_bytes,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
  if (major < 9) {
    return 1;
  }
  const int ctas_per_sm = 2 * cta_smem_bytes <= sm_smem_bytes ? 2 : 1;
  for (const int split : {4, 2}) {
    if (num_blocks * split <= ctas_per_sm * num_sms) {
      return split;
    }
  }
  return 1;
}

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate,
          int SplitV, typename... Args>
cudaError_t launch_split(dim3 grid, cudaStream_t stream, int token_smem_bytes,
                         Args... args) {
  const auto kernel =
      gdn_decode_post_conv_mtp_kernel<StateT, ValueHeadsPerKeyHead, SigmoidGate,
                                      SplitV>;
  // With the state chunks, a full window exceeds the default 48 KB per CTA.
  const cudaError_t error =
      cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kMaxMtpTokens * sizeof(TokenSmem));
  if (error != cudaSuccess) {
    return error;
  }
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim = {1, 1, SplitV};
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = token_smem_bytes;
  config.stream = stream;
  config.attrs = SplitV > 1 ? &attribute : nullptr;
  config.numAttrs = SplitV > 1 ? 1 : 0;
  return cudaLaunchKernelEx(&config, kernel, args...);
}

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
void launch_gdn_decode_post_conv_mtp(
    torch::stable::Tensor const& mixed_qkv, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices,
    torch::stable::Tensor const& cu_seqlens,
    torch::stable::Tensor const& num_accepted_tokens,
    torch::stable::Tensor& state, torch::stable::Tensor const& norm_weight,
    torch::stable::Tensor& out, const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* output_gate, int num_key_heads, int num_value_heads,
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
  const int token_smem_bytes =
      static_cast<int>(state_indices.size(1) * sizeof(TokenSmem));
  constexpr int kStateSmemBytes = 2 * kChunkV * kDimK * sizeof(StateT);
  const int split_v = choose_split_v(num_requests * num_value_heads,
                                     kStateSmemBytes + token_smem_bytes,
                                     mixed_qkv.get_device_index());
  // Without a V split, 16-bit state runs fused_gdn_decode_16bit_kernel.cu.
  if constexpr (!std::is_same_v<StateT, float>) {
    if (split_v == 1) {
      const cudaError_t error =
          launch_gdn_decode_post_conv_mtp_16bit<StateT, ValueHeadsPerKeyHead,
                                                SigmoidGate>(
              dim3(num_requests, num_value_heads), stream,
              static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()), a, b,
              static_cast<const float*>(a_log.data_ptr()), dt_bias.data_ptr(),
              static_cast<const int*>(state_indices.data_ptr()),
              static_cast<const int*>(cu_seqlens.data_ptr()),
              static_cast<const int*>(num_accepted_tokens.data_ptr()),
              static_cast<StateT*>(state.data_ptr()), output_gate,
              norm_weight.data_ptr(),
              static_cast<__nv_bfloat16*>(out.data_ptr()), num_key_heads,
              num_value_heads, static_cast<int>(state_indices.size(1)),
              dt_bias_type, norm_weight.scalar_type() == ScalarType::BFloat16,
              static_cast<float>(scale), static_cast<float>(norm_eps), strides);
      STD_TORCH_CHECK(error == cudaSuccess,
                      "GDN decode MTP post-conv kernel launch failed: ",
                      cudaGetErrorString(error));
      return;
    }
  }
  const auto launch = [&]<int SplitV>() {
    return launch_split<StateT, ValueHeadsPerKeyHead, SigmoidGate, SplitV>(
        dim3(num_requests, num_value_heads, SplitV), stream, token_smem_bytes,
        static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()), a, b,
        static_cast<const float*>(a_log.data_ptr()), dt_bias.data_ptr(),
        static_cast<const int*>(state_indices.data_ptr()),
        static_cast<const int*>(cu_seqlens.data_ptr()),
        static_cast<const int*>(num_accepted_tokens.data_ptr()),
        static_cast<StateT*>(state.data_ptr()), output_gate,
        norm_weight.data_ptr(), static_cast<__nv_bfloat16*>(out.data_ptr()),
        num_key_heads, num_value_heads, static_cast<int>(state_indices.size(1)),
        dt_bias_type, norm_weight.scalar_type() == ScalarType::BFloat16,
        static_cast<float>(scale), static_cast<float>(norm_eps), strides);
  };
  const cudaError_t error = split_v == 4   ? launch.template operator()<4>()
                            : split_v == 2 ? launch.template operator()<2>()
                                           : launch.template operator()<1>();
  STD_TORCH_CHECK(error == cudaSuccess,
                  "GDN decode MTP post-conv kernel launch failed: ",
                  cudaGetErrorString(error));
}

}  // namespace

bool fused_gdn_decode_kernel_available(
    torch::stable::Tensor const& device_tensor) {
  torch::stable::accelerator::DeviceGuard const device_guard(
      device_tensor.get_device_index());
  cudaFuncAttributes attributes;
  const cudaError_t error = cudaFuncGetAttributes(
      &attributes, gdn_decode_post_conv_mtp_kernel<float, 1, false>);
  return error == cudaSuccess;
}

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
                          state_scalar_type == ScalarType::BFloat16 ||
                          state_scalar_type == ScalarType::Half),
      "state must be a CUDA float32, bfloat16, or float16 tensor");
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

  STD_TORCH_CHECK(
      state_indices.dim() == 2 && state_indices.size(0) > 0 &&
          state_indices.size(1) > 0 && state_indices.size(1) <= kMaxMtpTokens,
      "state_indices must have shape [N, S] with 1 <= S <= ", kMaxMtpTokens);
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
  const auto* a_ptr = static_cast<const __nv_bfloat16*>(a.data_ptr());
  const auto* b_ptr = static_cast<const __nv_bfloat16*>(b.data_ptr());
  const auto* output_gate_ptr =
      static_cast<const __nv_bfloat16*>(output_gate.data_ptr());
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
    } else if (state_scalar_type == ScalarType::Half) {
      if (output_gate_activation == "sigmoid") {
        launch.template operator()<__half, ValueHeadsPerKeyHead, true>();
      } else {
        launch.template operator()<__half, ValueHeadsPerKeyHead, false>();
      }
    } else {
      if (output_gate_activation == "sigmoid") {
        launch.template operator()<__nv_bfloat16, ValueHeadsPerKeyHead, true>();
      } else {
        launch
            .template operator()<__nv_bfloat16, ValueHeadsPerKeyHead, false>();
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
