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
  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int64_t req_idx = idx_mapping_ptr[batch_idx];
    const int64_t request_num_blocks = num_blocks_ptr[req_idx];
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
    const bool mapping_enabled, const bool dcp_sharded, const int64_t cp_size,
    const int64_t cp_rank, const int64_t cp_interleave,
    const int64_t pad_slot_id) {
  const int32_t* block_table_ptr = block_table.data_ptr<int32_t>();
  int64_t* slot_mapping_ptr = slot_mapping.data_ptr<int64_t>();
  const int64_t* idx_mapping_ptr = idx_mapping.data_ptr<int64_t>();
  const int32_t* query_start_loc_ptr = query_start_loc.data_ptr<int32_t>();
  const int64_t* positions_ptr = positions.data_ptr<int64_t>();

  const int64_t num_reqs = idx_mapping.size(0);
  const int64_t num_tokens = query_start_loc_ptr[num_reqs];
  const int64_t block_table_stride = block_table.stride(0);
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

    for (int64_t token_idx = start; token_idx < end; ++token_idx) {
      const int64_t position = positions_ptr[token_idx];
      int64_t local_position = position;
      if (cp_size != 1 && dcp_sharded) {
        const int64_t virtual_block_size = block_size * cp_size;
        const int64_t virtual_block_idx = position / virtual_block_size;
        const int64_t virtual_block_offset = position % virtual_block_size;
        const bool is_local =
            virtual_block_offset / cp_interleave % cp_size == cp_rank;
        if (!is_local) {
          slot_mapping_ptr[token_idx] = pad_slot_id;
          continue;
        }
        const int64_t round = virtual_block_offset / (cp_interleave * cp_size);
        const int64_t remainder = virtual_block_offset % cp_interleave;
        const int64_t local_offset = round * cp_interleave + remainder;
        local_position = virtual_block_idx * block_size + local_offset;
      }

      const int64_t block_idx = local_position / block_size;
      const int64_t block_offset = local_position % block_size;
      const int32_t block_number =
          block_table_ptr[req_state_idx * block_table_stride + block_idx];
      slot_mapping_ptr[token_idx] =
          static_cast<int64_t>(block_number) * block_size + block_offset;
    }
  }
}

}  // namespace cpu_utils
