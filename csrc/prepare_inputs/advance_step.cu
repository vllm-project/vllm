/*
 * TODO: Add doc
 */

#include "advance_step.cuh"

namespace prepare_inputs {

template <int const num_threads>
__global__ void advance_step_kernel(int num_seqs, int block_size,
                                    long const* sampled_token_ids_ptr,
                                    long* input_positions_ptr,
                                    int* seq_lens_ptr, int* slot_mapping_ptr,
                                    int const* block_tables_ptr,
                                    int64_t const block_tables_stride) {
  int num_seq_blocks = div_ceil(num_seqs, num_threads);

  if (blockIdx.x >= num_seq_blocks) {
    return;
  }

  int cur_seq_id = blockIdx.x * num_threads + threadIdx.x;

  if (cur_seq_id >= num_seqs) {
    return;
  }

  int seq_len = seq_lens_ptr[cur_seq_id];
  int next_seq_len = seq_len + 1;
  int next_input_pos = next_seq_len - 1;

  seq_lens_ptr[cur_seq_id] = next_seq_len;
  input_positions_ptr[cur_seq_id] = next_input_pos;

  int const* seq_block_tables_ptr =
      block_tables_ptr + block_tables_stride * cur_seq_id;

  int block_index = next_input_pos / block_size;
  int block_offset = next_input_pos % block_size;

  int slot_num = seq_block_tables_ptr[block_index] * block_size + block_offset;
  slot_mapping_ptr[cur_seq_id] = slot_num;
}

inline void verify_tensor(std::string const& name, torch::Tensor& t,
                          int64_t const size_0, int64_t const size_1,
                          c10::ScalarType const type) {
  bool size_0_cond = true;
  if (size_0 != -1) {
    size_0_cond = t.size(0) == size_0;
  }

  bool size_1_cond = true;
  if (size_1 != -1) {
    size_1_cond = t.size(1) == size_1;
  }

  bool is_contiguous = t.is_contiguous();
  bool same_type = t.dtype() == type;

  bool pass = size_0_cond && size_1_cond && is_contiguous && same_type;
  if (!pass) {
    TORCH_CHECK(false, "tensor: name = ", name, ", shape = ", t.sizes(),
                " is_cont = ", t.is_contiguous(), ", type = ", t.dtype(),
                " is not as expected: shape = [", size_0, ", ", size_1,
                "], type = ", type);
  }
}

void advance_step(int num_seqs, int block_size,
                  torch::Tensor& sampled_token_ids,  // type: long
                  torch::Tensor& input_positions,    // type: long
                  torch::Tensor& seq_lens,           // type: int
                  torch::Tensor& slot_mapping,       // type: long
                  torch::Tensor& block_tables) {     // type: int

  if (logging) {
    printf("advance_step:\n");
    printf("  num_seqs = %d\n", num_seqs);
    printf("  block_size = %d\n", block_size);
  }
  // Verify all tensors
  verify_tensor("sampled_token_ids", sampled_token_ids, num_seqs, -1, at::kLong);
  verify_tensor("input_positions", input_positions, num_seqs, -1, at::kLong);
  verify_tensor("seq_lens", seq_lens, num_seqs, -1, at::kInt);
  verify_tensor("slot_mapping", slot_mapping, num_seqs, -1, at::kLong);
  verify_tensor("block_tables", block_tables, num_seqs, -1, at::kInt);

  int dev = sampled_token_ids.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);

  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);

  advance_step_kernel<max_threads><<<blocks, max_threads, 0, stream>>>(
      num_seqs, block_size,
      reinterpret_cast<long const*>(sampled_token_ids.data_ptr()),
      reinterpret_cast<long*>(input_positions.data_ptr()),
      reinterpret_cast<int*>(seq_lens.data_ptr()),
      reinterpret_cast<int*>(slot_mapping.data_ptr()),
      reinterpret_cast<int const*>(block_tables.data_ptr()),
      block_tables.stride(0));
}

}  // namespace prepare_inputs

void advance_step(int64_t num_seqs, int64_t block_size,
                  torch::Tensor& sampled_token_ids,
                  torch::Tensor& input_positions, torch::Tensor& seq_lens,
                  torch::Tensor& slot_mapping, torch::Tensor& block_tables) {
  prepare_inputs::advance_step(num_seqs, block_size, sampled_token_ids,
                               input_positions, seq_lens, slot_mapping,
                               block_tables);
}