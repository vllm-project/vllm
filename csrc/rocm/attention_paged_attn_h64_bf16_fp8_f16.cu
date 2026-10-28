// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "attention_kernels.cuh"

ATTN_ENTRY(paged_attention_h64_bf16_fp8_f16) {
  ATTN_LAUNCH(__hip_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3, 64,
              MFMAType::F16);
}
