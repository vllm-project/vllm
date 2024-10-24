/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "decoder_xqa_runner.h"

#include <assert.h>
#include <string.h>

#include <mutex>
#include <unordered_map>

#include "cubin/xqa_kernel_cubin.h"

DecoderXQARunner::DecoderXQARunner(const XQADataType data_type, int num_heads,
                                   int num_kv_heads, int head_size,
                                   bool multi_block_mode)
    : mDataType(data_type),
      mNumHeads(num_heads),
      mNumKVHeads(num_kv_heads),
      mHeadSize(head_size),
      mMultiBlockMode(multi_block_mode) {
  mMultiProcessorCount = getMultiProcessorCount();
  mPrecompiledImpl =
      DecoderXQAImpl::create(this, DecoderXQAImpl::ImplType::kPrecompiled);
}

DecoderXQARunner::~DecoderXQARunner() = default;

size_t DecoderXQARunner::getWorkspaceSize(int nb_semaphores) {
  // buffer for RoPE / output quantization.
  constexpr size_t kXQA_OUT_ELEM_SIZE = 2;  // fp16 or bf16.
  size_t workspace_size = 0;  // kXQA_OUT_ELEM_SIZE * mHeadSize * mNumHeads *
                              // max_num_tokens; // medusa

  if (mMultiBlockMode) {
    int workspaces[3];
    uint32_t const nbSubSeq = kXQA_MAX_NUM_SUB_SEQ;
    uint32_t const nbSeq = nbSubSeq / 2;
    int group_size = mNumHeads / mNumKVHeads;
    // workspaces[0] = sizeof(uint32_t) * nb_semaphores;//sizeof(uint32_t) *
    // nbSeq;                 // semaphores
    workspaces[0] =
        sizeof(float) * roundUp(group_size, 32) * nbSubSeq;  // rowMax
    workspaces[1] =
        sizeof(float) * roundUp(group_size, 32) * nbSubSeq;  // rowSum
    int32_t const multi_block_workspace_alignment = roundUp<int32_t>(
        kXQA_OUT_ELEM_SIZE * kMaxBeamWidth * group_size * mHeadSize, 128);
    workspaces[2] = multi_block_workspace_alignment * nbSubSeq;
    workspace_size =
        roundUp<size_t>(workspace_size, multi_block_workspace_alignment) +
        roundUp(workspaces[0], multi_block_workspace_alignment) +
        roundUp(workspaces[1], multi_block_workspace_alignment) +
        roundUp(workspaces[2], multi_block_workspace_alignment) +
        multi_block_workspace_alignment;  // extra space reserved for alignment
  }
  return workspace_size;
}

DecoderXQAImpl* DecoderXQARunner::getImplFromXQAParams(
    XQAParams const& xqaParams) {
  return mPrecompiledImpl.get();
}

void DecoderXQARunner::run(XQAParams const& xqa_params,
                           KVCacheListParams const& kv_cache_buffer,
                           cudaStream_t const& stream) {
  return getImplFromXQAParams(xqa_params)
      ->run(xqa_params, kv_cache_buffer, stream);
}

void DecoderXQARunner::run(XQAParams const& xqa_params,
                           KVCacheListParams const& kv_linear_buffer,
                           cudaStream_t const& stream);
