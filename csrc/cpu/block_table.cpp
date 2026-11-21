#include "cpu_types.hpp"

#include <algorithm>
#include <cstring>

namespace cpu_utils {

void gather_block_table_cpu(const torch::Tensor& block_table,
                            torch::Tensor& output,
                            const torch::Tensor& idx_mapping,
                            const torch::Tensor& num_blocks,
                            const int64_t num_reqs_padded) {
  const int32_t* block_table_ptr = block_table.data_ptr<int32_t>();
  int32_t* output_ptr = output.data_ptr<int32_t>();
  const int64_t* idx_mapping_ptr = idx_mapping.data_ptr<int64_t>();
  const int32_t* num_blocks_ptr = num_blocks.data_ptr<int32_t>();

  const int64_t num_reqs = idx_mapping.size(0);
  const int64_t block_table_stride = block_table.stride(0);
  const int64_t output_stride = output.stride(0);
  const int64_t max_num_blocks_per_req =
      std::min(block_table.size(1), output.size(1));
  TORCH_CHECK(num_blocks.numel() >= block_table.size(0),
              "gather_block_table_cpu: num_blocks has ", num_blocks.numel(),
              " elements, expected at least block_table.size(0) = ",
              block_table.size(0));
  TORCH_CHECK(
      num_reqs >= 0 && num_reqs <= num_reqs_padded,
      "gather_block_table_cpu: idx_mapping.size(0) (", num_reqs,
      ") exceeds num_reqs_padded (", num_reqs_padded,
      "); at most num_reqs_padded rows fit in the output");
  TORCH_CHECK(num_reqs_padded <= output.size(0),
              "gather_block_table_cpu: num_reqs_padded (", num_reqs_padded,
              ") exceeds output.size(0) (", output.size(0), ")");
  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int64_t req_idx = idx_mapping_ptr[batch_idx];
    TORCH_CHECK(req_idx >= 0 && req_idx < block_table.size(0),
                "gather_block_table_cpu: idx_mapping[", batch_idx, "] = ",
                req_idx, " is out of range [0, ", block_table.size(0), ")");
    const int64_t request_num_blocks = num_blocks_ptr[req_idx];
    TORCH_CHECK(
        request_num_blocks >= 0 && request_num_blocks <= max_num_blocks_per_req,
        "gather_block_table_cpu: num_blocks[", req_idx, "] = ",
        request_num_blocks, " is out of range [0, ", max_num_blocks_per_req,
        "]");
    std::memcpy(output_ptr + batch_idx * output_stride,
                block_table_ptr + req_idx * block_table_stride,
                request_num_blocks * sizeof(int32_t));
  }

  for (int64_t batch_idx = num_reqs; batch_idx < num_reqs_padded; ++batch_idx) {
    std::memset(output_ptr + batch_idx * output_stride, 0,
                output.size(1) * sizeof(int32_t));
  }
}

void compute_slot_mapping_cpu(
    const torch::Tensor& block_table, torch::Tensor& slot_mapping,
    const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
    const torch::Tensor& positions, const int64_t block_size,
    const bool mapping_enabled, const int64_t pad_slot_id) {
  const int32_t* block_table_ptr = block_table.data_ptr<int32_t>();
  int64_t* slot_mapping_ptr = slot_mapping.data_ptr<int64_t>();
  const int64_t* idx_mapping_ptr = idx_mapping.data_ptr<int64_t>();
  const int32_t* query_start_loc_ptr = query_start_loc.data_ptr<int32_t>();
  const int64_t* positions_ptr = positions.data_ptr<int64_t>();

  const int64_t num_reqs = idx_mapping.size(0);
  TORCH_CHECK(query_start_loc.numel() == num_reqs + 1,
              "compute_slot_mapping_cpu: query_start_loc has ",
              query_start_loc.numel(),
              " elements, expected num_reqs + 1 = ", num_reqs + 1);
  TORCH_CHECK(query_start_loc_ptr[0] == 0,
              "compute_slot_mapping_cpu: query_start_loc[0] (",
              query_start_loc_ptr[0],
              ") must be 0 so the returned prefix starts at the first token");
  const int64_t num_tokens = query_start_loc_ptr[num_reqs];
  const int64_t block_table_stride = block_table.stride(0);
  TORCH_CHECK(block_size > 0, "compute_slot_mapping_cpu: block_size (",
              block_size, ") must be positive");
  TORCH_CHECK(positions.numel() >= num_tokens,
              "compute_slot_mapping_cpu: query_start_loc[", num_reqs, "] = ",
              num_tokens, " exceeds positions.numel() (", positions.numel(),
              ")");
  TORCH_CHECK(slot_mapping.numel() >= num_tokens,
              "compute_slot_mapping_cpu: query_start_loc[", num_reqs, "] = ",
              num_tokens, " exceeds slot_mapping.numel() (",
              slot_mapping.numel(), ")");
  int64_t prev_loc = 0;
  for (int64_t i = 0; i <= num_reqs; ++i) {
    const int64_t loc = query_start_loc_ptr[i];
    TORCH_CHECK(loc >= prev_loc && loc <= positions.numel(),
                "compute_slot_mapping_cpu: query_start_loc[", i, "] = ", loc,
                " is out of range [", prev_loc, ", ", positions.numel(), "]");
    prev_loc = loc;
  }
  std::fill(slot_mapping_ptr + num_tokens,
            slot_mapping_ptr + slot_mapping.numel(), pad_slot_id);

  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int64_t req_state_idx = idx_mapping_ptr[batch_idx];
    const int64_t start = query_start_loc_ptr[batch_idx];
    const int64_t end = query_start_loc_ptr[batch_idx + 1];
    if (!mapping_enabled || req_state_idx < 0) {
      std::fill(slot_mapping_ptr + start, slot_mapping_ptr + end, pad_slot_id);
      continue;
    }
    TORCH_CHECK(req_state_idx < block_table.size(0),
                "compute_slot_mapping_cpu: idx_mapping[", batch_idx, "] = ",
                req_state_idx, " is out of range [0, ", block_table.size(0),
                ")");

    for (int64_t token_idx = start; token_idx < end; ++token_idx) {
      const int64_t position = positions_ptr[token_idx];
      TORCH_CHECK(position >= 0, "compute_slot_mapping_cpu: positions[",
                  token_idx, "] = ", position, " is negative");
      const int64_t block_idx = position / block_size;
      TORCH_CHECK(block_idx < block_table.size(1),
                  "compute_slot_mapping_cpu: positions[", token_idx, "] = ",
                  position, " maps to block ", block_idx,
                  ", out of range [0, ", block_table.size(1), ")");
      const int64_t block_offset = position % block_size;
      const int32_t block_number =
          block_table_ptr[req_state_idx * block_table_stride + block_idx];
      slot_mapping_ptr[token_idx] =
          static_cast<int64_t>(block_number) * block_size + block_offset;
    }
  }
}

}  // namespace cpu_utils
