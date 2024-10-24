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
#pragma once
#include "decoder_xqa_common.h"
using XQADataType = Data_type;

struct XQAParams {
  XQADataType data_type = DATA_TYPE_FP16;
  XQADataType kv_cache_data_type = DATA_TYPE_FP16;
  void* output = nullptr;
  void const* qHeads = nullptr;
  float const* kv_scale_quant_orig = nullptr;
  uint32_t* semaphores = nullptr;
  void* workspaces = nullptr;
  uint32_t batch_size = 0;
  int32_t beam_width = 0;

  int32_t num_q_heads = 0;
  int32_t num_kv_heads = 0;
  int32_t head_size = 0;
  int timestep = 0;

  // Paged KV cache parameters.
  bool paged_kv_cache = true;  // always true
  int tokens_per_block;
  int max_blocks_per_sequence;
  bool multi_block_mode;
  bool multi_query_tokens = false;
};