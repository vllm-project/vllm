// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cstdint>

#include "core/utils.hpp"

namespace vllm {

// One logical cache page of one request, as scheduled by the page-per-CTA
// cache gathers (cache_kernels.cu and the NVFP4 MLA reader).
struct GatherPageTask {
  int32_t req_id;
  int32_t logical_block;
  int32_t page_token_begin;
  int32_t page_token_end;
  int32_t output_token_begin;
};

template <bool has_terminal_start>
__device__ __forceinline__ bool map_gather_page_task(
    int32_t task, const int32_t* __restrict__ output_starts, int32_t num_reqs,
    int32_t total_tokens, int32_t block_size,
    const int32_t* __restrict__ seq_starts, GatherPageTask& page) {
  int32_t relative_page = task;
  for (int32_t req_id = 0; req_id < num_reqs; ++req_id) {
    const int32_t output_begin = min(output_starts[req_id], total_tokens);
    int32_t output_end;
    if constexpr (has_terminal_start) {
      output_end = min(output_starts[req_id + 1], total_tokens);
    } else {
      output_end =
          min(req_id + 1 < num_reqs ? output_starts[req_id + 1] : total_tokens,
              total_tokens);
    }
    const int32_t seq_len = max(output_end - output_begin, 0);
    const int32_t source_begin = seq_starts == nullptr ? 0 : seq_starts[req_id];
    const int32_t first_block = source_begin / block_size;
    const int32_t num_pages =
        div_ceil(source_begin + seq_len, block_size) - first_block;
    if (relative_page < num_pages) {
      page.req_id = req_id;
      page.logical_block = first_block + relative_page;
      const int32_t page_begin = page.logical_block * block_size;
      const int32_t copy_begin = max(source_begin, page_begin);
      const int32_t copy_end =
          min(source_begin + seq_len, page_begin + block_size);
      page.page_token_begin = copy_begin - page_begin;
      page.page_token_end = copy_end - page_begin;
      page.output_token_begin = output_begin + copy_begin - source_begin;
      return true;
    }
    relative_page -= num_pages;
  }
  return false;
}

}  // namespace vllm
