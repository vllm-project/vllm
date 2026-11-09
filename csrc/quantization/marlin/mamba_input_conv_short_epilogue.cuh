// SPDX-License-Identifier: Apache-2.0
// Conv semantics adapted from vLLM causal_conv1d.py;
// Copyright (c) 2024, Tri Dao.
// Adapted from mamba_input_conv_epilogue.cuh: actual T1..7 masks, odd tail,
// physical-history acceptance bounds and untouched history suffix.
#pragma once
#include "mamba_input_conv_epilogue.cuh"
namespace vllm_mamba_input_conv {
// Called only after the original raw BF16 scatter, original global stores and
// final CTA barrier. Only the actual short rows are read; physical history
// can be larger than the current draft window and its unused suffix is kept.
template <int ActualT>
__device__ __forceinline__ void epilogue_short(void* shared_b, int slice_col,
                                               const ConvParams& params,
                                               uint16_t* projected) {
  if (slice_col < 64 || slice_col >= 144)
    return;  // CTA-uniform z/dt exclusion.
  static_assert(ActualT >= 1 && ActualT < 8);
  const int tid = threadIdx.x;
  const int channel = (slice_col - 64) * kTileChannels + tid;
  const int slot = params.cache_indices[0];
  // Expected inactive slots never read accepted/history or modify history.
  // Both tests and all returns here are CTA-uniform, before any CTA barrier.
  if (slot == params.pad_slot_id || slot == params.null_block_id) return;
  if (slot < 0 || slot >= params.num_cache_pages ||
      params.reserved < ActualT + 2 || params.reserved > 10) {
    asm volatile("trap;");
  }
  // None is the public non-speculative T1/H3 path: carry starts at row0.
  // This indexes the existing history; it does not manufacture an RNG/metadata
  // tensor.
  const int accepted = params.accepted_tokens ? params.accepted_tokens[0] : 1;
  if ((!params.accepted_tokens && (ActualT != 1 || params.reserved != 3)) ||
      accepted < 1 || accepted > params.reserved - 2) {
    asm volatile("trap;");
  }
  auto* raw = reinterpret_cast<uint16_t*>(shared_b);
  auto* bytes = reinterpret_cast<unsigned char*>(shared_b);
  auto* history = reinterpret_cast<uint16_t*>(bytes + kHistoryOffset);
  auto* convolved = reinterpret_cast<uint16_t*>(bytes + kConvOffset);
  uint16_t* state =
      params.history + static_cast<int64_t>(slot) * params.history_stride_page +
      static_cast<int64_t>(channel) * params.history_stride_channel;
  const int64_t step = params.history_stride_token;
  uint16_t old0 = state[static_cast<int64_t>(accepted - 1) * step];
  uint16_t old1 = state[static_cast<int64_t>(accepted) * step];
  uint16_t old2 = state[static_cast<int64_t>(accepted + 1) * step];
  history[tid] = old1;
  history[kTileChannels + tid] = old2;
#pragma unroll
  for (int token = 0; token < ActualT; ++token)
    history[(token + 2) * kTileChannels + tid] = raw[token * kRawPitch + tid];
  // Four contiguous BF16 taps,8 bytes per channel; no16-byte tap overread.
  const uint64_t taps =
      *reinterpret_cast<const uint64_t*>(params.taps + channel * 4);
  const uint16_t w0 = static_cast<uint16_t>(taps);
  const uint16_t w1 = static_cast<uint16_t>(taps >> 16);
  const uint16_t w2 = static_cast<uint16_t>(taps >> 32);
  const uint16_t w3 = static_cast<uint16_t>(taps >> 48);
  const float bias =
      __uint_as_float(static_cast<uint32_t>(params.bias[channel]) << 16);
#pragma unroll
  for (int token = 0; token < ActualT; token += 2) {
    const uint16_t x0 = raw[token * kRawPitch + tid];
    const uint16_t x1 =
        token + 1 < ActualT ? raw[(token + 1) * kRawPitch + tid] : 0;
    const uint32_t p0 = mul_bf16_pair(pack(old0, old1), w0);
    const uint32_t p1 = mul_bf16_pair(pack(old1, old2), w1);
    const uint32_t p2 = mul_bf16_pair(pack(old2, x0), w2);
    const uint32_t p3 = mul_bf16_pair(pack(x0, x1), w3);
    float a0 = add_product(static_cast<uint16_t>(p0), bias);
    float a1 = add_product(static_cast<uint16_t>(p0 >> 16), bias);
    a0 = add_product(static_cast<uint16_t>(p1), a0);
    a1 = add_product(static_cast<uint16_t>(p1 >> 16), a1);
    a0 = add_product(static_cast<uint16_t>(p2), a0);
    a1 = add_product(static_cast<uint16_t>(p2 >> 16), a1);
    a0 = add_product(static_cast<uint16_t>(p3), a0);
    a1 = add_product(static_cast<uint16_t>(p3 >> 16), a1);
    const uint32_t result = round_pair(silu(a0), silu(a1));
    convolved[token * kRawPitch + tid] = static_cast<uint16_t>(result);
    if (token + 1 < ActualT)
      convolved[(token + 1) * kRawPitch + tid] =
          static_cast<uint16_t>(result >> 16);
    old0 = old2;
    old1 = x0;
    old2 = x1;
  }
  // No history is overwritten until every channel has loaded its old values.
  // Distinct shared tiles preserve all raw tokens needed by later taps.
  __syncthreads();
  const int tile_channel = (slice_col - 64) * kTileChannels;
#pragma unroll
  for (int vector = tid; vector < (ActualT + 2) * 16; vector += 128) {
    const int row = vector / 16;
    const int column = (vector % 16) * 8;
    auto* destination =
        params.history +
        static_cast<int64_t>(slot) * params.history_stride_page +
        static_cast<int64_t>(row) * params.history_stride_token + tile_channel +
        column;
    *reinterpret_cast<uint4*>(destination) =
        *reinterpret_cast<const uint4*>(history + row * kTileChannels + column);
  }
  const int row = tid / 16;
  const int column = (tid % 16) * 8;
  if (row < ActualT) {
    auto* destination = projected + static_cast<int64_t>(row) * kColumns +
                        8192 + tile_channel + column;
    *reinterpret_cast<uint4*>(destination) =
        *reinterpret_cast<const uint4*>(convolved + row * kRawPitch + column);
  }
  __syncthreads();
}

}  // namespace vllm_mamba_input_conv
