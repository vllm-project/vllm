/*
 * Copyright (c) 2024, The vLLM team.
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

// Runtime dispatch for the ROCm paged attention kernels.
//
// The kernels themselves (and the expensive template instantiation they
// trigger) live in the generated attention_paged_attn_*.cu translation units,
// one per (dtype, kv-cache dtype, head size, block size) combination, so ninja
// compiles them in parallel instead of serializing the whole attention kernel
// set onto a single core. Those translation units, their prototypes and the
// dispatch below are all emitted from the variant list in
// csrc/rocm/generate_paged_attn.py.
#include <torch/all.h>

#include "attention_kernels.cuh"
#include "ops.h"

// One prototype per generated entry point.
#include "attention_paged_attn_entries.inc"

// gfx11/gfx12 (RDNA) need a different kernel set than gfx9 (CDNA).
bool is_navi_gpu() {
  static bool is_cached = false;
  static bool result;

  if (!is_cached) {
    int device_id;
    hipDeviceProp_t deviceProp;
    hipGetDevice(&device_id);
    hipGetDeviceProperties(&deviceProp, device_id);

    std::string arch = deviceProp.gcnArchName;
    result = arch.find("gfx11") == 0 || arch.find("gfx12") == 0;
    is_cached = true;
  }

  return result;
}

// clang-format off
void paged_attention(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& tmp_out,     // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor& value_cache, // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads, 
    double scale,
    torch::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens, // [num_seqs]
    const std::optional<torch::Tensor>& query_start_loc, // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type,
    bool use_interleaved_v_cache) {
  // clang-format on
  bool is_navi = is_navi_gpu();
  const int head_size = query.size(2);
  // clang-format off
  #include "attention_paged_attn_dispatch.inc"
  // clang-format on
}

#undef ATTN_ENTRY
#undef ATTN_FWD
#undef ATTN_PARAMS
