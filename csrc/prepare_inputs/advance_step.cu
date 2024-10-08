/*
 * The goal of this GPU kernel is to advance input tensors on the GPU directly
 * PR: https://github.com/vllm-project/vllm/pull/6338
 * Current restrictions:
 *     1. Specialized for DraftModelRunner
 *     2. Supports flash_attn only
 */

#include "advance_step.cuh"

namespace prepare_inputs {

//
template <int const num_threads>
__global__ void advance_step_flashattn_kernel(
    int num_seqs, int num_queries, int block_size, long* input_tokens_ptr,
    long const* sampled_token_ids_ptr, long* input_positions_ptr,
    int* seq_lens_ptr, long* slot_mapping_ptr, int const* block_tables_ptr,
    int64_t const block_tables_stride) {
  int const n_pad = num_seqs - num_queries;
  if (n_pad && blockIdx.x == 0) {
    // Handle cuda graph padding
    int const offset = num_queries;
    for (int i = threadIdx.x; i < n_pad; i += blockDim.x) {
      input_tokens_ptr[offset + i] = 0;
      input_positions_ptr[offset + i] = 0;
      slot_mapping_ptr[offset + i] = -1;
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

  // Update input_tokens
  input_tokens_ptr[cur_query_id] = sampled_token_ids_ptr[cur_query_id];

  int seq_len = seq_lens_ptr[cur_query_id];
  int next_seq_len = seq_len + 1;
  int next_input_pos = next_seq_len - 1;

  // Update seq_lens
  seq_lens_ptr[cur_query_id] = next_seq_len;
  // Update input_positions
  input_positions_ptr[cur_query_id] = next_input_pos;

  int const* seq_block_tables_ptr =
      block_tables_ptr + block_tables_stride * cur_query_id;

  int block_index = next_input_pos / block_size;
  int block_offset = next_input_pos % block_size;

  int slot_num = seq_block_tables_ptr[block_index] * block_size + block_offset;
  // Update slot_mapping
  slot_mapping_ptr[cur_query_id] = slot_num;
}

inline void verify_tensor(std::string const& name, torch::Tensor const& t,
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

__global__ void advance_step_flashinfer_kernel(
    int num_threads, int num_seqs, int num_queries, int block_size,
    long* input_tokens_ptr, long const* sampled_token_ids_ptr,
    long* input_positions_ptr, int* seq_lens_ptr, long* slot_mapping_ptr,
    int const* block_tables_ptr, int64_t const block_tables_stride,
    int* paged_kv_last_page_len_ptr, int* block_table_bound_ptr) {
  int num_query_blocks = div_ceil(num_queries, num_threads);

  if (blockIdx.x < num_query_blocks) {
    int cur_query_id = blockIdx.x * num_threads + threadIdx.x;

    if (cur_query_id < num_queries) {
      // Update input_tokens
      input_tokens_ptr[cur_query_id] = sampled_token_ids_ptr[cur_query_id];

      int seq_len = seq_lens_ptr[cur_query_id];
      int next_seq_len = seq_len + 1;
      int next_input_pos = next_seq_len - 1;

      // Update seq_lens
      seq_lens_ptr[cur_query_id] = next_seq_len;
      // Update input_positions
      input_positions_ptr[cur_query_id] = next_input_pos;

      int const* seq_block_tables_ptr =
          block_tables_ptr + block_tables_stride * cur_query_id;

      int block_index = next_input_pos / block_size;
      int block_offset = next_input_pos % block_size;

      // Update paged_kv_last_page_len
      paged_kv_last_page_len_ptr[cur_query_id] = block_offset + 1;

      int slot_num =
          seq_block_tables_ptr[block_index] * block_size + block_offset;
      // Update slot_mapping
      slot_mapping_ptr[cur_query_id] = slot_num;
      block_table_bound_ptr[cur_query_id] = div_ceil(next_seq_len, block_size);
    }
  }
}

__global__ void advance_step_flashinfer_indptr_kernel(
    int num_threads, int num_seqs, int num_queries, int* paged_kv_indptr_ptr,
    int* block_table_bound_ptr) {
  int idx = blockIdx.x * num_threads + threadIdx.x;

  // Update paged_kv_indptr
  if (idx < num_queries) {
    int sum = 0;
    for (int i = 0; i <= idx; ++i) {
      sum += block_table_bound_ptr[i];
    }
    paged_kv_indptr_ptr[idx + 1] = sum;
  }
}

__global__ void advance_step_flashinfer_indices_kernel(
    int num_threads, int num_seqs, int num_queries, int const* block_tables_ptr,
    int64_t const block_tables_stride, int* paged_kv_indices_ptr,
    int* paged_kv_indptr_ptr, int* block_table_bound_ptr) {
  int idx = blockIdx.x * num_threads + threadIdx.x;
  int row = idx / block_tables_stride;
  int col = idx % block_tables_stride;

  if (row < num_queries && col < block_table_bound_ptr[row]) {
    paged_kv_indices_ptr[paged_kv_indptr_ptr[row] + col] =
        block_tables_ptr[row * block_tables_stride + col];
  }
  // if cudagraph, fill padded seqs with the last valid seq's indptr
  if (num_queries < row && row <= num_seqs) {
    paged_kv_indptr_ptr[row] = paged_kv_indptr_ptr[num_queries];
  }
}

void advance_step_flashattn(int num_seqs, int num_queries, int block_size,
                            torch::Tensor& input_tokens,       // type: long
                            torch::Tensor& sampled_token_ids,  // type: long
                            torch::Tensor& input_positions,    // type: long
                            torch::Tensor& seq_lens,           // type: int
                            torch::Tensor& slot_mapping,       // type: long
                            torch::Tensor& block_tables) {     // type: int

  if (logging) {
    printf("advance_step_flashattn:\n");
    printf("  num_seqs = %d\n", num_seqs);
    printf("  num_queries = %d\n", num_queries);
    printf("  block_size = %d\n", block_size);
  }
  // Verify all tensors
  verify_tensor("input_tokens", input_tokens, num_seqs, -1, at::kLong);
  verify_tensor("sampled_token_ids", sampled_token_ids, num_queries, 1,
                at::kLong);
  verify_tensor("input_positions", input_positions, num_seqs, -1, at::kLong);
  verify_tensor("seq_lens", seq_lens, num_seqs, -1, at::kInt);
  verify_tensor("slot_mapping", slot_mapping, num_seqs, -1, at::kLong);
  verify_tensor("block_tables", block_tables, num_seqs, -1, at::kInt);

  int dev = sampled_token_ids.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);

  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);

  advance_step_flashattn_kernel<max_threads>
      <<<blocks, max_threads, 0, stream>>>(
          num_seqs, num_queries, block_size,
          reinterpret_cast<long*>(input_tokens.data_ptr()),
          reinterpret_cast<long const*>(sampled_token_ids.data_ptr()),
          reinterpret_cast<long*>(input_positions.data_ptr()),
          reinterpret_cast<int*>(seq_lens.data_ptr()),
          reinterpret_cast<long*>(slot_mapping.data_ptr()),
          reinterpret_cast<int const*>(block_tables.data_ptr()),
          block_tables.stride(0));
}

void advance_step_flashinfer(
    int num_seqs, int num_queries, int block_size,
    torch::Tensor& input_tokens,            // type: long
    torch::Tensor& sampled_token_ids,       // type: long
    torch::Tensor& input_positions,         // type: long
    torch::Tensor& seq_lens,                // type: int
    torch::Tensor& slot_mapping,            // type: long
    torch::Tensor& block_tables,            // type: int
    torch::Tensor& paged_kv_indices,        // type: int
    torch::Tensor& paged_kv_indptr,         // type: int
    torch::Tensor& paged_kv_last_page_len,  // type: int
    torch::Tensor& block_table_bound) {     // type: int

  if (logging) {
    printf("advance_step_flashinfer:\n");
    printf("  num_seqs = %d\n", num_seqs);
    printf("  num_queries = %d\n", num_queries);
    printf("  block_size = %d\n", block_size);
    printf("  block_tables.stride(0) = %zu\n", block_tables.stride(0));
  }
  // Verify all tensors
  verify_tensor("input_tokens", input_tokens, num_seqs, -1, at::kLong);
  // verify_tensor("sampled_token_ids", sampled_token_ids, num_queries, 1,
  //               at::kLong);
  verify_tensor("input_positions", input_positions, num_seqs, -1, at::kLong);
  verify_tensor("seq_lens", seq_lens, num_seqs, -1, at::kInt);
  verify_tensor("slot_mapping", slot_mapping, num_seqs, -1, at::kLong);
  verify_tensor("block_tables", block_tables, num_seqs, -1, at::kInt);

  verify_tensor("paged_kv_indices", paged_kv_indices, -1, -1, at::kInt);
  verify_tensor("paged_kv_indptr", paged_kv_indptr, num_seqs + 1, -1, at::kInt);
  verify_tensor("paged_kv_last_page_len", paged_kv_last_page_len, num_seqs, -1,
                at::kInt);

  verify_tensor("block_table_bound", block_table_bound, num_seqs, -1, at::kInt);

  int dev = sampled_token_ids.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);

  int blocks;
  int threads;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, dev);
  if (logging) {
    printf("launching kernel with %d blocks\n", blocks);
  }

  // TODO(will): support arbitrary block_tables stride
  if ((blocks * threads) / block_tables.stride(0) < num_queries) {
    TORCH_CHECK(false,
                "multi-step: not enough threads to map block_table to"
                "FlashInfer's paged_kv_indices on GPU. Try reducing the number "
                "of seqs,",
                " increasing the block size or take smaller steps.",
                " num_queries = ", num_queries,
                " block_tables.stride(0) = ", block_tables.stride(0),
                " blocks = ", blocks, " max_threads = ", threads);
  }

  advance_step_flashinfer_kernel<<<blocks, threads, 0, stream>>>(
      threads, num_seqs, num_queries, block_size,
      reinterpret_cast<long*>(input_tokens.data_ptr()),
      reinterpret_cast<long const*>(sampled_token_ids.data_ptr()),
      reinterpret_cast<long*>(input_positions.data_ptr()),
      reinterpret_cast<int*>(seq_lens.data_ptr()),
      reinterpret_cast<long*>(slot_mapping.data_ptr()),
      reinterpret_cast<int const*>(block_tables.data_ptr()),
      block_tables.stride(0),
      reinterpret_cast<int*>(paged_kv_last_page_len.data_ptr()),
      reinterpret_cast<int*>(block_table_bound.data_ptr()));

  advance_step_flashinfer_indptr_kernel<<<blocks, threads, 0, stream>>>(
      threads, num_seqs, num_queries,
      reinterpret_cast<int*>(paged_kv_indptr.data_ptr()),
      reinterpret_cast<int*>(block_table_bound.data_ptr()));

  advance_step_flashinfer_indices_kernel<<<blocks, threads, 0, stream>>>(
      threads, num_seqs, num_queries,
      reinterpret_cast<int const*>(block_tables.data_ptr()),
      block_tables.stride(0),
      reinterpret_cast<int*>(paged_kv_indices.data_ptr()),
      reinterpret_cast<int*>(paged_kv_indptr.data_ptr()),
      reinterpret_cast<int*>(block_table_bound.data_ptr()));
}

}  // namespace prepare_inputs

void advance_step_flashattn(int64_t num_seqs, int64_t num_queries,
                            int64_t block_size, torch::Tensor& input_tokens,
                            torch::Tensor& sampled_token_ids,
                            torch::Tensor& input_positions,
                            torch::Tensor& seq_lens,
                            torch::Tensor& slot_mapping,
                            torch::Tensor& block_tables) {
  prepare_inputs::advance_step_flashattn(
      num_seqs, num_queries, block_size, input_tokens, sampled_token_ids,
      input_positions, seq_lens, slot_mapping, block_tables);
}

void advance_step_flashinfer(
    int64_t num_seqs, int64_t num_queries, int64_t block_size,
    torch::Tensor& input_tokens, torch::Tensor& sampled_token_ids,
    torch::Tensor& input_positions, torch::Tensor& seq_lens,
    torch::Tensor& slot_mapping, torch::Tensor& block_tables,
    torch::Tensor& paged_kv_indices, torch::Tensor& paged_kv_indptr,
    torch::Tensor& paged_kv_last_page_len, torch::Tensor& block_table_bound) {
  prepare_inputs::advance_step_flashinfer(
      num_seqs, num_queries, block_size, input_tokens, sampled_token_ids,
      input_positions, seq_lens, slot_mapping, block_tables, paged_kv_indices,
      paged_kv_indptr, paged_kv_last_page_len, block_table_bound);
}
