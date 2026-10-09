#include "cpu_types.hpp"

#include <cstring>

namespace cpu_utils {

void gather_block_table_cpu(const torch::Tensor& block_table,
                            torch::Tensor& output,
                            const torch::Tensor& idx_mapping,
                            const torch::Tensor& num_blocks,
                            const int64_t num_reqs_padded) {
  const int32_t* block_table_ptr = block_table.data_ptr<int32_t>();
  int32_t* output_ptr = output.data_ptr<int32_t>();
  const int32_t* idx_mapping_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* num_blocks_ptr = num_blocks.data_ptr<int32_t>();

  const int64_t num_reqs = idx_mapping.size(0);
  const int64_t block_table_stride = block_table.stride(0);
  const int64_t output_stride = output.stride(0);
  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int32_t req_idx = idx_mapping_ptr[batch_idx];
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

}  // namespace cpu_utils
