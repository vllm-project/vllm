// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "attention_kernels.cuh"

ATTN_ENTRY(paged_attention_h128_bf16_auto) {
  ATTN_LAUNCH(__hip_bfloat16, __hip_bfloat16, vllm::Fp8KVCacheDataType::kAuto,
              128, MFMAType::F16);
}
