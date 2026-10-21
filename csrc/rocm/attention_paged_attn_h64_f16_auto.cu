// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "attention_kernels.cuh"

ATTN_ENTRY(paged_attention_h64_f16_auto) {
  ATTN_LAUNCH(_Float16, _Float16, vllm::Fp8KVCacheDataType::kAuto, 64,
              MFMAType::F16);
}
