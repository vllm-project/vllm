// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Conv semantics adapted from vLLM causal_conv1d.py;
// Copyright (c) 2024, Tri Dao.
// Wider Marlin raw M16/M32 scatter, unchanged BF16 conv math.
// Adapted from the validated mamba_input_conv_short_epilogue.cuh in this
// session.
#pragma once
#include "mamba_input_conv_epilogue.cuh"

namespace vllm_mamba_input_conv {
// Raw matrix remains immutable; one request's history/conv scratch is reused.
// Caller must prove full-K/one-Ntile ownership and completed async pipelines.
template <int TileRows, int ActualT>
__device__ __forceinline__ void epilogue_wide_sequence(void* shared_b,
                                                       int slice_col,
                                                       const ConvParams& params,
                                                       uint16_t* projected,
                                                       int raw_row_start) {
  if (slice_col < 64 || slice_col >= 144)
    return;  // CTA-uniform z/dt exclusion.
  static_assert(TileRows == 16 || TileRows == 32);
  static_assert(ActualT >= 1 && ActualT <= 8);
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
  const auto* raw =
      reinterpret_cast<const uint16_t*>(shared_b) + raw_row_start * kRawPitch;
  auto* bytes = reinterpret_cast<unsigned char*>(shared_b);
  auto* history = reinterpret_cast<uint16_t*>(bytes + TileRows * kRawPitch * 2);
  auto* convolved = reinterpret_cast<uint16_t*>(
      bytes + TileRows * kRawPitch * 2 + kHistoryBytes);
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

// Same explicit new-family descriptor as C's M<=8 prototype:
// reserved low4bits=physicalH, positive upperbits=cache-index stride.
// Runtime rows=B*ActualT; all rows must fit ONE selected MMA tile.
template <int TileRows, int ActualT>
__device__ __forceinline__ void epilogue_wide(void* shared_b, int slice_col,
                                              const ConvParams& params,
                                              uint16_t* projected, int rows) {
  static_assert(TileRows == 16 || TileRows == 32);
  static_assert(ActualT >= 1 && ActualT <= 8);
  constexpr int scratch_bytes =
      TileRows * kRawPitch * 2 + kHistoryBytes + kConvBytes;
  constexpr int stage3_pipeline_bytes = 12288 + 1536 + 3 * TileRows * 64 * 2;
  static_assert(scratch_bytes <= stage3_pipeline_bytes);
  if (slice_col < 64 || slice_col >= 144) return;
  const int physical_h = params.reserved & 15;
  const int index_stride = params.reserved >> 4;
  if (params.reserved <= 0 || index_stride <= 0 || rows <= 0 ||
      rows > TileRows || rows % ActualT != 0 || physical_h < ActualT + 2 ||
      physical_h > 10) {
    asm volatile("trap;");
  }
  const int batches = rows / ActualT;
  if (batches < 2 || batches > 4) asm volatile("trap;");
  for (int batch = 0; batch < batches; ++batch) {
    ConvParams single = params;
    single.cache_indices =
        params.cache_indices + static_cast<int64_t>(batch) * index_stride;
    single.accepted_tokens =
        params.accepted_tokens ? params.accepted_tokens + batch : nullptr;
    single.reserved = physical_h;
    epilogue_wide_sequence<TileRows, ActualT>(
        shared_b, slice_col, single,
        projected + static_cast<int64_t>(batch) * ActualT * kColumns,
        batch * ActualT);
  }
}
}  // namespace vllm_mamba_input_conv
