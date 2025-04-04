/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
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

#include "attention_kernels.cuh"

#define DISPATCH_BY_OUTPUT_DTYPE(OUTPUT_DTYPE, FN)                          \
  if (OUTPUT_DTYPE == at::ScalarType::Float) {                              \
    FN(float);                                                              \
  } else if (OUTPUT_DTYPE == at::ScalarType::Half) {                        \
    FN(uint16_t);                                                           \
  } else if (OUTPUT_DTYPE == at::ScalarType::BFloat16) {                    \
    FN(__nv_bfloat16);                                                      \
  } else {                                                                  \
    TORCH_CHECK(false, "Unsupported input type of output: ", OUTPUT_DTYPE); \
  }

template <typename scalar_t, int HEAD_SIZE, bool SHOULD_OUTPUT_LSE>
void merge_attn_states_launcher(
    torch::Tensor& output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const std::optional<torch::Tensor>& output_lse,  // [NUM_HEADS, NUM_TOKENS]
    torch::Tensor& prefix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    torch::Tensor& prefix_lse,     // [NUM_HEADS, NUM_TOKENS]
    torch::Tensor& suffix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    torch::Tensor& suffix_lse      // [NUM_HEADS, NUM_TOKENS]
) {
  const int num_tokens = output.size(0);
  const int num_heads = output.size(1);

  dim3 grid(num_tokens, num_heads);
  dim3 block(HEAD_SIZE);

  scalar_t* output_ptr = reinterpret_cast<scalar_t*>(output.data_ptr());
  float* output_lse_ptr =
      output_lse ? reinterpret_cast<float*>(output_lse.value().data_ptr())
                 : nullptr;
  const scalar_t* prefix_output_ptr =
      reinterpret_cast<const scalar_t*>(prefix_output.data_ptr());
  const float* prefix_lse_ptr =
      reinterpret_cast<const float*>(prefix_lse.data_ptr());
  const scalar_t* suffix_output_ptr =
      reinterpret_cast<const scalar_t*>(suffix_output.data_ptr());
  const float* suffix_lse_ptr =
      reinterpret_cast<const float*>(suffix_lse.data_ptr());

  vllm::merge_attn_states_kernel<scalar_t, HEAD_SIZE, SHOULD_OUTPUT_LSE>
      <<<grid, block>>>(output_ptr, output_lse_ptr, prefix_output_ptr,
                        prefix_lse_ptr, suffix_output_ptr, suffix_lse_ptr);
}

#define CALL_MERGE_ATTN_STATES_LAUNCHER(T, HEAD_SIZE, SHOULD_OUTPUT_LSE) \
  merge_attn_states_launcher<T, HEAD_SIZE, SHOULD_OUTPUT_LSE>(           \
      output, output_lse, prefix_output, prefix_lse, suffix_output,      \
      suffix_lse);

#define CALL_MERGE_ATTN_STATES_LAUNCHER_SHOULD_OUTPUT_LSE(T, HEAD_SIZE) \
  if (output_lse) {                                                     \
    CALL_MERGE_ATTN_STATES_LAUNCHER(T, HEAD_SIZE, true);                \
  } else {                                                              \
    CALL_MERGE_ATTN_STATES_LAUNCHER(T, HEAD_SIZE, false);               \
  }

#define CALL_MERGE_ATTN_STATES_LAUNCHER_HEADSIZE(T)              \
  switch (head_size) {                                           \
    case 32:                                                     \
      \   
      CALL_MERGE_ATTN_STATES_LAUNCHER_SHOULD_OUTPUT_LSE(T, 32);  \
      break;                                                     \
    case 64:                                                     \
      CALL_MERGE_ATTN_STATES_LAUNCHER_SHOULD_OUTPUT_LSE(T, 64);  \
      break;                                                     \
    case 128:                                                    \
      CALL_MERGE_ATTN_STATES_LAUNCHER_SHOULD_OUTPUT_LSE(T, 128); \
      break;                                                     \
    case 192:                                                    \
      CALL_MERGE_ATTN_STATES_LAUNCHER_SHOULD_OUTPUT_LSE(T, 192); \
      break;                                                     \
    case 256:                                                    \
      CALL_MERGE_ATTN_STATES_LAUNCHER_SHOULD_OUTPUT_LSE(T, 256); \
      \ 
      break;                                                     \
    default:                                                     \
      TORCH_CHECK(false, "Unsupported head size: ", head_size);  \
      break;                                                     \
  }

void merge_attn_states(
    torch::Tensor& output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const std::optional<torch::Tensor>& output_lse,  // [NUM_HEADS, NUM_TOKENS]
    torch::Tensor& prefix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    torch::Tensor& prefix_lse,     // [NUM_HEADS, NUM_TOKENS]
    torch::Tensor& suffix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    torch::Tensor& suffix_lse      // [NUM_HEADS, NUM_TOKENS]
) {
  int head_size = output.size(2);
  DISPATCH_BY_OUTPUT_DTYPE(output.dtype(),
                           CALL_MERGE_ATTN_STATES_LAUNCHER_HEADSIZE)
}