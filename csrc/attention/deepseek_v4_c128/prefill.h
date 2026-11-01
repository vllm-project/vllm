// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cuda_runtime_api.h>

void c128_run(void* q, void* packed_q, void* kv, void* sink, void* lengths,
              void* out, void* raw_o, void* raw_m, void* raw_l, void* seq_lens,
              void* gather_lens, int num_query_tokens, int num_kv_rows,
              int compressed_width, int q_stride, int out_stride, float scale,
              cudaStream_t stream);
