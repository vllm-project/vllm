/*
 * The goal of this GPU kernel is to advance input tensors on the GPU directly
 * PR: https://github.com/vllm-project/vllm/pull/6338
 * Current restrictions:
 *     1. Specialized for DraftModelRunner
 *     2. Supports flash_attn only
 */

#include "advance_step.cuh"

namespace prepare_inputs {

__device__ void update_decode(int const cur_query_id,
                              long* input_tokens_ptr,
                              long* input_positions_ptr,
                              int* seq_lens_ptr,
                              long* slot_mapping_ptr,
                              int const* block_tables_ptr,
                              long const* sampled_token_ids_ptr,
                              int64_t const block_tables_stride,
                              int const block_size) {
  // Update input_tokens
  input_tokens_ptr[cur_query_id] = sampled_token_ids_ptr[cur_query_id];

  int const seq_len = seq_lens_ptr[cur_query_id];
  int const next_seq_len = seq_len + 1;
  int const next_input_pos = seq_len; 

  // Update seq_lens
  seq_lens_ptr[cur_query_id] = next_seq_len;
  // Update input_positions
  input_positions_ptr[cur_query_id] = next_input_pos;

  int const* seq_block_tables_ptr =
      block_tables_ptr + block_tables_stride * cur_query_id;

  int const block_index = next_input_pos / block_size;
  int const block_offset = next_input_pos % block_size;

  // TODO (varun) : CHeck if we can reuse this logic for filling prefill slot mapping
  // instead of passing it as an input
  int slot_num = seq_block_tables_ptr[block_index] * block_size + block_offset;
  // Update slot_mapping
  slot_mapping_ptr[cur_query_id] = slot_num;
}

__device__ void update_prefill(int const cur_query_id,
                               int* seq_lens_ptr,
                               int* context_lens_ptr,
                               int const token_chunk_size) {
  seq_lens_ptr[cur_query_id] += token_chunk_size;
  context_lens_ptr[cur_query_id] += token_chunk_size;
}

template <int const num_threads>
__global__ void advance_step_kernel(int num_prefill_tokens,
                                    int num_prefills,
                                    int num_seqs,
                                    int num_queries,
                                    int block_size,
                                    int token_chunk_size,
                                    long* input_tokens_ptr,
                                    long const* sampled_token_ids_ptr,
                                    long* input_positions_ptr,
                                    int* seq_lens_ptr, long* slot_mapping_ptr,
                                    int const* block_tables_ptr,
                                    int64_t const block_tables_stride,
                                    int* context_lens_ptr = nullptr,
                                    long const* prefill_steps_tokens = nullptr,
                                    long const* prefill_steps_slot_mapping = nullptr) {

  // copy prefills 
  if (num_prefill_tokens > 0 && blockIdx.x == 0) {
    // Update prefill input tokens and slot mapping
    for (int i = threadIdx.x; i < num_prefill_tokens; i += blockDim.x) {
      input_tokens_ptr[i] = prefill_steps_tokens[i];
      slot_mapping_ptr[i] = prefill_steps_slot_mapping[i];
      input_positions_ptr[i] += token_chunk_size;
    }
  }

  int num_query_blocks = div_ceil(num_queries, num_threads);
  if (blockIdx.x >= num_query_blocks) {
    return;
  }

  int cur_query_id = blockIdx.x * num_threads + threadIdx.x;
  if (cur_query_id >= num_queries) {
    return;
  }

  bool const is_prefill_query_id = cur_query_id < num_prefills;

  if (is_prefill_query_id) {
    // prefill update
    // Note that
    // - input tokens
    // - input positions and,
    // - slot mapping are already updated.
    update_prefill(cur_query_id,
                   seq_lens_ptr,
                   context_lens_ptr,
                   token_chunk_size);
  } else {
    // decode update
    update_decode(cur_query_id,
                  input_tokens_ptr,
                  input_positions_ptr,
                  seq_lens_ptr,
                  slot_mapping_ptr,
                  block_tables_ptr,
                  sampled_token_ids_ptr,
                  block_tables_stride,
                  block_size);
  }
}

inline void verify_tensor(std::string const& name,
                          torch::Tensor const& t,
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

inline void verify_tensor_ge(std::string const& name,
                             torch::Tensor const& t,
                             int64_t const size_0,
                             int64_t const size_1,
                             c10::ScalarType const type) {
  bool size_0_cond = true;
  if (size_0 != -1) {
    size_0_cond = t.size(0) >= size_0;
  }

  bool size_1_cond = true;
  if (size_1 != -1) {
    size_1_cond = t.size(1) >= size_1;
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

void advance_step(int const num_prefill_tokens,
                  int const num_prefills,
                  int const num_seqs,
                  int const num_queries,
                  int const block_size,
                  int const token_chunk_size,
                  torch::Tensor& input_tokens,       // type: long
                  torch::Tensor& sampled_token_ids,  // type: long
                  torch::Tensor& input_positions,    // type: long
                  torch::Tensor& seq_lens,           // type: int
                  torch::Tensor& slot_mapping,       // type: long
                  torch::Tensor& block_tables,       // type: int
                  c10::optional<torch::Tensor>& context_lens, // type: int
                  c10::optional<torch::Tensor> const& prefill_steps_tokens, // type long
                  c10::optional<torch::Tensor> const& prefill_steps_slot_mapping) { // type long 

  if (logging) {
    printf("advance_step:\n");
    printf("  num_prefill_tokens = %d\n", num_prefill_tokens);
    printf("  num_prefills = %d\n", num_prefills);
    printf("  num_seqs = %d\n", num_seqs);
    printf("  num_queries = %d\n", num_queries);
    printf("  block_size = %d\n", block_size);
    printf("  token_chunk_size = %d\n", token_chunk_size);
  }

  if (num_prefills > 0) {
    TORCH_CHECK(num_prefill_tokens > 0);
    TORCH_CHECK(token_chunk_size > 0);
    TORCH_CHECK(context_lens.has_value());
    TORCH_CHECK(prefill_steps_tokens.has_value());
    TORCH_CHECK(prefill_steps_slot_mapping.has_value());
  }

  // Verify all tensors
  verify_tensor("input_tokens", input_tokens, num_seqs, -1, at::kLong);
  verify_tensor("sampled_token_ids", sampled_token_ids, num_queries, 1,
                at::kLong);
  verify_tensor("input_positions", input_positions, num_seqs, -1, at::kLong);
  verify_tensor("seq_lens", seq_lens, num_seqs, -1, at::kInt);
  verify_tensor("slot_mapping", slot_mapping, num_seqs, -1, at::kLong);
  verify_tensor("block_tables", block_tables, num_seqs, -1, at::kInt);
  if (num_prefills > 0) {
    verify_tensor("context_lens", context_lens.value(), num_seqs, -1, at::kInt);
    verify_tensor_ge("prefill_steps_tokens", prefill_steps_tokens.value(), num_prefill_tokens, -1, at::kLong);
    verify_tensor_ge("prefill_steps_slot_mapping", prefill_steps_slot_mapping.value(), num_prefill_tokens, -1, at::kLong);
  }

  int dev = sampled_token_ids.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);

  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);

  advance_step_kernel<max_threads><<<blocks, max_threads, 0, stream>>>(
      num_prefill_tokens,
      num_prefills,
      num_seqs,
      num_queries,
      block_size,
      token_chunk_size,
      reinterpret_cast<long*>(input_tokens.data_ptr()),
      reinterpret_cast<long const*>(sampled_token_ids.data_ptr()),
      reinterpret_cast<long*>(input_positions.data_ptr()),
      reinterpret_cast<int*>(seq_lens.data_ptr()),
      reinterpret_cast<long*>(slot_mapping.data_ptr()),
      reinterpret_cast<int const*>(block_tables.data_ptr()),
      block_tables.stride(0),
      context_lens.has_value() ? reinterpret_cast<int*>(context_lens->data_ptr()) : nullptr,
      prefill_steps_tokens.has_value() ? reinterpret_cast<long const*>(prefill_steps_tokens->data_ptr()) : nullptr,
      prefill_steps_slot_mapping.has_value() ? reinterpret_cast<long const*>(prefill_steps_slot_mapping->data_ptr()) : nullptr);
}

}  // namespace prepare_inputs

void advance_step(int64_t num_prefill_tokens,
                  int64_t num_prefills,
                  int64_t num_seqs,
                  int64_t num_queries,
                  int64_t block_size,
                  int64_t token_chunk_size,
                  torch::Tensor& input_tokens, torch::Tensor& sampled_token_ids,
                  torch::Tensor& input_positions, torch::Tensor& seq_lens,
                  torch::Tensor& slot_mapping, torch::Tensor& block_tables,
                  c10::optional<torch::Tensor>& context_lens,
                  c10::optional<torch::Tensor> const& prefill_steps_tokens,
                  c10::optional<torch::Tensor> const& prefill_steps_slot_mapping) {
  prepare_inputs::advance_step(num_prefill_tokens, num_prefills, num_seqs, num_queries, block_size, token_chunk_size,
                               input_tokens,
                               sampled_token_ids, input_positions, seq_lens,
                               slot_mapping, block_tables, context_lens, prefill_steps_tokens,
                               prefill_steps_slot_mapping);
}