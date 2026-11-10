// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vllm_mamba_input_conv {

// Explicit by-value native argument18. Projected output derives from C0.
struct alignas(8) ConvParams {
  const uint16_t* taps;
  const uint16_t* bias;
  uint16_t* history;
  const int32_t* cache_indices;
  const int32_t* accepted_tokens;
  int64_t history_stride_page;
  int64_t history_stride_token;
  int64_t history_stride_channel;
  int32_t num_cache_pages;
  int32_t pad_slot_id;
  int32_t null_block_id;
  int32_t reserved;  // Legacy T8:0. Short family: actual physical history
                     // length3..10.
};

static_assert(std::is_standard_layout_v<ConvParams>);
static_assert(std::is_trivially_copyable_v<ConvParams>);
static_assert(sizeof(void*) == 8 && alignof(ConvParams) == 8);
static_assert(sizeof(ConvParams) == 80);
static_assert(offsetof(ConvParams, taps) == 0);
static_assert(offsetof(ConvParams, bias) == 8);
static_assert(offsetof(ConvParams, history) == 16);
static_assert(offsetof(ConvParams, cache_indices) == 24);
static_assert(offsetof(ConvParams, accepted_tokens) == 32);
static_assert(offsetof(ConvParams, history_stride_page) == 40);
static_assert(offsetof(ConvParams, history_stride_token) == 48);
static_assert(offsetof(ConvParams, history_stride_channel) == 56);
static_assert(offsetof(ConvParams, num_cache_pages) == 64);
static_assert(offsetof(ConvParams, pad_slot_id) == 68);
static_assert(offsetof(ConvParams, null_block_id) == 72);
static_assert(offsetof(ConvParams, reserved) == 76);

constexpr int kTokens = 8;
constexpr int kColumns = 18560;
constexpr int kInputK = 4096;
constexpr int kChannels = 10240;
constexpr int kTileChannels = 128;
constexpr int kRawPitch = 136;
constexpr int kRawBytes = 2176;
constexpr int kHistoryOffset = 2176;
constexpr int kHistoryBytes = 2560;
constexpr int kConvOffset = 4736;
constexpr int kConvBytes = 2176;
constexpr int kSharedReuseBytes = 6912;
constexpr int kDepth3DeadBBytes = 12288;
static_assert(kRawBytes == kTokens * kRawPitch * 2);
static_assert(kHistoryBytes == 10 * kTileChannels * 2);
static_assert(kConvOffset == kHistoryOffset + kHistoryBytes);
static_assert(kSharedReuseBytes == kConvOffset + kConvBytes);
static_assert(kSharedReuseBytes <= kDepth3DeadBBytes);


enum class ConvFamily { Single, Batched, Wide };

cudaError_t launch_short(int tokens, const int4* input, const int4* weight,
                        int4* projected, int4* scratch, const int4* scales,
                        const float* global_scale, int* locks, ConvParams conv,
                        cudaStream_t stream);
cudaError_t launch_batch_small(int tokens, int rows, const int4* input,
                              const int4* weight, int4* projected, int4* scratch,
                              const int4* scales, const float* global_scale,
                              int* locks, ConvParams conv, cudaStream_t stream);
cudaError_t launch_batch_wide(int tokens, int rows, const int4* input,
                             const int4* weight, int4* projected, int4* scratch,
                             const int4* scales, const float* global_scale,
                             int* locks, ConvParams conv, cudaStream_t stream);

}  // namespace vllm_mamba_input_conv
