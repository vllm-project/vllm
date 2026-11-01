#include "cpu_types.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

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
  const int64_t* idx_mapping_ptr = idx_mapping.data_ptr<int64_t>();
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
    const int64_t req_state_idx = idx_mapping_ptr[batch_idx];
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

torch::Tensor get_prompt_logprobs_token_ids_cpu(
    const int64_t num_tokens, const torch::Tensor& query_start_loc,
    const torch::Tensor& idx_mapping, const torch::Tensor& num_computed_tokens,
    const torch::Tensor& all_token_ids) {
  torch::Tensor token_ids =
      torch::empty({num_tokens}, idx_mapping.options().dtype(torch::kInt64));

  int64_t* token_ids_ptr = token_ids.data_ptr<int64_t>();
  const int32_t* query_start_loc_ptr = query_start_loc.data_ptr<int32_t>();
  const int64_t* idx_mapping_ptr = idx_mapping.data_ptr<int64_t>();
  const int32_t* num_computed_tokens_ptr =
      num_computed_tokens.data_ptr<int32_t>();
  const int32_t* all_token_ids_ptr = all_token_ids.data_ptr<int32_t>();
  const int64_t all_token_ids_stride = all_token_ids.stride(0);

  for (int64_t batch_idx = 0; batch_idx < idx_mapping.size(0); ++batch_idx) {
    const int64_t req_state_idx = idx_mapping_ptr[batch_idx];
    const int32_t query_start = query_start_loc_ptr[batch_idx];
    const int32_t query_end = query_start_loc_ptr[batch_idx + 1];
    const int32_t target_start = num_computed_tokens_ptr[req_state_idx] + 1;
    const int32_t* request_tokens =
        all_token_ids_ptr + req_state_idx * all_token_ids_stride;

    std::transform(request_tokens + target_start,
                   request_tokens + target_start + query_end - query_start,
                   token_ids_ptr + query_start, [](int32_t token_id) {
                     return static_cast<int64_t>(token_id);
                   });
  }
  return token_ids;
}

void prepare_pos_seq_lens_cpu(const torch::Tensor& idx_mapping,
                              const torch::Tensor& query_start_loc,
                              const torch::Tensor& num_computed_tokens,
                              torch::Tensor& pos, torch::Tensor& seq_lens) {
  const int64_t* idx_mapping_ptr = idx_mapping.data_ptr<int64_t>();
  const int32_t* query_start_loc_ptr = query_start_loc.data_ptr<int32_t>();
  const int32_t* num_computed_tokens_ptr =
      num_computed_tokens.data_ptr<int32_t>();
  int64_t* pos_ptr = pos.data_ptr<int64_t>();
  int32_t* seq_lens_ptr = seq_lens.data_ptr<int32_t>();

  const int64_t num_reqs = idx_mapping.size(0);
  std::fill(seq_lens_ptr + num_reqs, seq_lens_ptr + seq_lens.numel(), 0);

  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    const int64_t req_state_idx = idx_mapping_ptr[req_idx];
    const int32_t num_computed = num_computed_tokens_ptr[req_state_idx];
    const int64_t start = query_start_loc_ptr[req_idx];
    const int64_t end = query_start_loc_ptr[req_idx + 1];

    seq_lens_ptr[req_idx] = num_computed + static_cast<int32_t>(end - start);
    std::iota(pos_ptr + start, pos_ptr + end,
              static_cast<int64_t>(num_computed));
  }
}

void prepare_mrope_positions_cpu(torch::Tensor& positions,
                                 const torch::Tensor& prefill_positions,
                                 const torch::Tensor& prefill_delta,
                                 const torch::Tensor& idx_mapping,
                                 const torch::Tensor& query_start_loc,
                                 const torch::Tensor& prefill_lens,
                                 const torch::Tensor& num_computed_tokens) {
  int64_t* positions_ptr = positions.data_ptr<int64_t>();
  const int32_t* prefill_positions_ptr = prefill_positions.data_ptr<int32_t>();
  const int32_t* prefill_delta_ptr = prefill_delta.data_ptr<int32_t>();
  const int64_t* idx_mapping_ptr = idx_mapping.data_ptr<int64_t>();
  const int32_t* query_start_loc_ptr = query_start_loc.data_ptr<int32_t>();
  const int32_t* prefill_lens_ptr = prefill_lens.data_ptr<int32_t>();
  const int32_t* num_computed_tokens_ptr =
      num_computed_tokens.data_ptr<int32_t>();

  const int64_t num_reqs = idx_mapping.size(0);
  const int64_t num_dims = positions.size(0);
  const int64_t positions_stride0 = positions.stride(0);
  const int64_t positions_stride1 = positions.stride(1);
  const int64_t prefill_positions_stride0 = prefill_positions.stride(0);
  const int64_t prefill_positions_stride1 = prefill_positions.stride(1);

  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int64_t req_state_idx = idx_mapping_ptr[batch_idx];
    const int32_t query_start = query_start_loc_ptr[batch_idx];
    const int32_t query_end = query_start_loc_ptr[batch_idx + 1];
    const int32_t query_len = query_end - query_start;
    const int32_t num_computed = num_computed_tokens_ptr[req_state_idx];
    const int32_t prompt_len = std::min(
        query_len, std::max(prefill_lens_ptr[req_state_idx] - num_computed, 0));
    const int32_t delta = prefill_delta_ptr[req_state_idx];

    for (int64_t dim = 0; dim < num_dims; ++dim) {
      int64_t* output = positions_ptr + dim * positions_stride0 +
                        query_start * positions_stride1;
      const int32_t* prompt =
          prefill_positions_ptr +
          (req_state_idx * num_dims + dim) * prefill_positions_stride0;

      for (int32_t token_idx = 0; token_idx < prompt_len; ++token_idx) {
        output[token_idx * positions_stride1] =
            prompt[(num_computed + token_idx) * prefill_positions_stride1];
      }
      for (int32_t token_idx = prompt_len; token_idx < query_len; ++token_idx) {
        output[token_idx * positions_stride1] =
            num_computed + token_idx + delta;
      }
    }
  }
}

torch::Tensor combine_sampled_and_draft_tokens_cpu(
    torch::Tensor& input_ids, const torch::Tensor& idx_mapping,
    const torch::Tensor& last_sampled_tokens,
    const torch::Tensor& query_start_loc, const torch::Tensor& seq_lens,
    const torch::Tensor& prefill_len, const torch::Tensor& draft_tokens,
    const torch::Tensor& cu_num_logits, const int64_t num_logits,
    const int64_t num_new_sampled_tokens) {
  const int64_t* idx_mapping_ptr = idx_mapping.data_ptr<int64_t>();
  const int64_t* last_sampled_tokens_ptr =
      last_sampled_tokens.data_ptr<int64_t>();
  const int32_t* query_start_loc_ptr = query_start_loc.data_ptr<int32_t>();
  const int32_t* seq_lens_ptr = seq_lens.data_ptr<int32_t>();
  const int32_t* prefill_len_ptr = prefill_len.data_ptr<int32_t>();
  const int64_t* draft_tokens_ptr = draft_tokens.data_ptr<int64_t>();
  const int32_t* cu_num_logits_ptr = cu_num_logits.data_ptr<int32_t>();
  int32_t* input_ids_ptr = input_ids.data_ptr<int32_t>();

  torch::Tensor logits_indices =
      torch::empty({num_logits}, input_ids.options().dtype(torch::kInt64));
  int64_t* logits_indices_ptr = logits_indices.data_ptr<int64_t>();

  const int64_t num_reqs = idx_mapping.size(0);
  const int64_t draft_tokens_stride = draft_tokens.stride(0);
  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    const int64_t req_state_idx = idx_mapping_ptr[req_idx];
    const int64_t logits_offset = cu_num_logits_ptr[req_idx];
    const int64_t logits_offset_end = cu_num_logits_ptr[req_idx + 1];
    const int64_t request_num_logits = logits_offset_end - logits_offset;
    const int64_t num_draft_tokens =
        request_num_logits - num_new_sampled_tokens;
    const int64_t query_end = query_start_loc_ptr[req_idx + 1];
    const int64_t logits_start = query_end - request_num_logits;

    std::iota(logits_indices_ptr + logits_offset,
              logits_indices_ptr + logits_offset_end, logits_start);

    const int32_t seq_len = seq_lens_ptr[req_idx];
    const int32_t request_prefill_len = prefill_len_ptr[req_state_idx];
    if (seq_len <= request_prefill_len) {
      continue;
    }

    const int64_t first_logit_seq_pos = seq_len - request_num_logits;
    if (num_new_sampled_tokens > 0 &&
        first_logit_seq_pos >= request_prefill_len) {
      input_ids_ptr[logits_start] =
          static_cast<int32_t>(last_sampled_tokens_ptr[req_state_idx]);
    }

    const int64_t draft_start = query_end - num_draft_tokens;
    for (int64_t i = 0; i < num_draft_tokens; ++i) {
      input_ids_ptr[draft_start + i] = static_cast<int32_t>(
          draft_tokens_ptr[req_state_idx * draft_tokens_stride + i]);
    }
  }

  return logits_indices;
}

void post_update_cpu(const torch::Tensor& idx_mapping,
                     torch::Tensor& num_computed_tokens,
                     torch::Tensor& last_sampled_tokens,
                     const c10::optional<torch::Tensor>& output_bin_counts,
                     const torch::Tensor& sampled_tokens,
                     const torch::Tensor& num_sampled,
                     const torch::Tensor& num_rejected,
                     const c10::optional<torch::Tensor>& query_start_loc,
                     torch::Tensor& all_token_ids, torch::Tensor& total_len) {
  const int64_t* idx_mapping_ptr = idx_mapping.data_ptr<int64_t>();
  int32_t* num_computed_tokens_ptr = num_computed_tokens.data_ptr<int32_t>();
  int64_t* last_sampled_tokens_ptr = last_sampled_tokens.data_ptr<int64_t>();
  int32_t* output_bin_counts_ptr =
      output_bin_counts ? output_bin_counts->data_ptr<int32_t>() : nullptr;
  const int64_t* sampled_tokens_ptr = sampled_tokens.data_ptr<int64_t>();
  const int32_t* num_sampled_ptr = num_sampled.data_ptr<int32_t>();
  const int32_t* num_rejected_ptr = num_rejected.data_ptr<int32_t>();
  const int32_t* query_start_loc_ptr =
      query_start_loc ? query_start_loc->data_ptr<int32_t>() : nullptr;
  int32_t* all_token_ids_ptr = all_token_ids.data_ptr<int32_t>();
  int32_t* total_len_ptr = total_len.data_ptr<int32_t>();

  const int64_t num_reqs = idx_mapping.size(0);
  const int64_t last_sampled_tokens_stride = last_sampled_tokens.stride(0);
  const int64_t output_bin_counts_stride =
      output_bin_counts ? output_bin_counts->stride(0) : 0;
  const int64_t sampled_tokens_stride = sampled_tokens.stride(0);
  const int64_t all_token_ids_stride = all_token_ids.stride(0);

  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    const int64_t req_state_idx = idx_mapping_ptr[req_idx];
    if (req_state_idx < 0) {
      continue;
    }

    const int32_t request_total_len = total_len_ptr[req_state_idx];
    const int32_t request_num_sampled = num_sampled_ptr[req_idx];
    const int64_t* request_sampled_tokens =
        sampled_tokens_ptr + req_idx * sampled_tokens_stride;
    int32_t* request_all_token_ids =
        all_token_ids_ptr + req_state_idx * all_token_ids_stride;

    if (request_num_sampled > 0) {
      last_sampled_tokens_ptr[req_state_idx * last_sampled_tokens_stride] =
          request_sampled_tokens[request_num_sampled - 1];
      total_len_ptr[req_state_idx] = request_total_len + request_num_sampled;
    }

    for (int32_t i = 0; i < request_num_sampled; ++i) {
      const int64_t token_id = request_sampled_tokens[i];
      request_all_token_ids[request_total_len + i] =
          static_cast<int32_t>(token_id);
      if (output_bin_counts_ptr != nullptr) {
        ++output_bin_counts_ptr[req_state_idx * output_bin_counts_stride +
                                token_id];
      }
    }

    const int32_t query_len =
        query_start_loc_ptr == nullptr
            ? 0
            : query_start_loc_ptr[req_idx + 1] - query_start_loc_ptr[req_idx];
    const int32_t computed_delta = query_len - num_rejected_ptr[req_idx];
    if (computed_delta != 0) {
      num_computed_tokens_ptr[req_state_idx] += computed_delta;
    }
  }
}

}  // namespace cpu_utils
