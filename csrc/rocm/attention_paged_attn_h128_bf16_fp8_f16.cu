// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Instantiates the ROCm paged attention decode kernels for a single
// (dtype, kv-cache dtype, head size) combination. Kept in its own translation
// unit so ninja can compile the instantiations in parallel; see
// attention_kernels.cuh.
#include "attention_kernels.cuh"

ATTN_ENTRY(paged_attention_h128_bf16_fp8_f16) {
  ATTN_LAUNCH(__hip_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3, 128,
              MFMAType::F16);
}
