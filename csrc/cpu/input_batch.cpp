#include "cpu_types.hpp"

#include <cstring>

namespace cpu_utils {

void prepare_prefill_inputs_cpu(torch::Tensor& input_ids,
                                torch::Tensor& next_prefill_tokens,
                                const torch::Tensor& idx_mapping,
                                const torch::Tensor& query_start_loc,
                                const torch::Tensor& all_token_ids,
                                const torch::Tensor& prefill_len,
                                const torch::Tensor& num_computed_tokens) {
  int32_t* input_ids_ptr = input_ids.data_ptr<int32_t>();
  int32_t* next_prefill_tokens_ptr = next_prefill_tokens.data_ptr<int32_t>();
  const int32_t* idx_mapping_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* query_start_loc_ptr = query_start_loc.data_ptr<int32_t>();
  const int32_t* all_token_ids_ptr = all_token_ids.data_ptr<int32_t>();
  const int32_t* prefill_len_ptr = prefill_len.data_ptr<int32_t>();
  const int32_t* num_computed_tokens_ptr =
      num_computed_tokens.data_ptr<int32_t>();

  const int64_t num_reqs = idx_mapping.size(0);
  const int64_t num_lookahead = next_prefill_tokens.size(0);
  const int64_t next_prefill_tokens_stride = next_prefill_tokens.stride(0);
  const int64_t all_token_ids_stride = all_token_ids.stride(0);

  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int32_t req_state_idx = idx_mapping_ptr[batch_idx];
    const int32_t request_prefill_len = prefill_len_ptr[req_state_idx];
    const int32_t num_computed = num_computed_tokens_ptr[req_state_idx];
    if (num_computed >= request_prefill_len) {
      continue;
    }

    const int32_t query_start = query_start_loc_ptr[batch_idx];
    const int32_t query_end = query_start_loc_ptr[batch_idx + 1];
    const int32_t query_len = query_end - query_start;
    const int32_t* request_tokens =
        all_token_ids_ptr + req_state_idx * all_token_ids_stride;
    std::memcpy(input_ids_ptr + query_start, request_tokens + num_computed,
                query_len * sizeof(int32_t));

    const int64_t lookahead_start =
        static_cast<int64_t>(num_computed) + query_len;
    for (int64_t lookahead = 0; lookahead < num_lookahead; ++lookahead) {
      const int64_t pos = lookahead_start + lookahead;
      next_prefill_tokens_ptr[lookahead * next_prefill_tokens_stride +
                              req_state_idx] =
          pos < request_prefill_len ? request_tokens[pos] : 0;
    }
  }
}

}  // namespace cpu_utils
