// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Instantiates the ROCm paged attention decode kernels for a single
// (dtype, kv-cache dtype, head size) combination. Kept in its own translation
// unit so ninja can compile the instantiations in parallel; see
// attention_kernels.cuh.
#include "attention_kernels.cuh"

ATTN_ENTRY(paged_attention_h64_f16_auto) {
  ATTN_LAUNCH(_Float16, _Float16, vllm::Fp8KVCacheDataType::kAuto, 64,
              MFMAType::F16);
}
