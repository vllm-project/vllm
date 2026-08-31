/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "cutlass/bfloat16.h"
#include <cstdint>
#include <cuda_runtime.h>

struct SparseMlaQ8Kv8PrefillParams {
  int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
  float sm_scale_div_log2;

  const uint8_t* __restrict__ q;
  const uint8_t* __restrict__ kv;
  const int32_t* __restrict__ indices;
  const float* __restrict__ attn_sink;
  const int32_t* __restrict__ topk_length;

  const float* __restrict__ q_scale_ptr;
  const float* __restrict__ kv_scale_ptr;

  int stride_q_s_q;
  int stride_q_h_q;
  int64_t stride_kv_s_kv;
  int stride_kv_h_kv;
  int stride_indices_s_q;
  int stride_indices_h_kv;

  cutlass::bfloat16_t* __restrict__ out;
  float* __restrict__ max_logits;
  float* __restrict__ lse;

  cudaStream_t stream;
};
