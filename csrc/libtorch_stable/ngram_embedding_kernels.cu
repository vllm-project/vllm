// N-gram embedding index kernel for LongCat-Flash (n-gram embedding variant).
//
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/ngram_embedding.cuh
//
// For each position, computes the hashed n-gram embedding ids that index the
// concatenated embedder table. Integer tensors are int32 except ``row_indices``
// (int64); the token table is ``[max_running_reqs, max_context_len]`` int32,
// where a negative entry marks an ignored token (e.g. an EOS boundary).

#include "torch_utils.h"

#include "ops.h"

#include <cstdint>

namespace vllm::ngram_embedding {

constexpr int kBlockThreads = 256;

__global__ void ComputeNGramIdsKernel(
    int batch_size, int ne_n, int ne_k,
    int* ne_weights,                       // [ne_n-1, ne_k, ne_n]
    int* ne_mods,                          // [ne_n-1, ne_k]
    int* exclusive_ne_embedder_size_sums,  // [(ne_n-1)*ne_k + 1]
    int* exclusive_req_len_sums,           // [batch_size + 1]
    int* ne_token_table,  // [max_running_reqs, max_context_len]
    int max_context_len,
    const int64_t* __restrict__ row_indices,  // [batch_size]
    int* column_starts,                       // [batch_size]
    int* n_gram_ids                           // [token_num, (ne_n-1)*ne_k]
) {
  const int req_id = blockIdx.x % batch_size;
  const int config_id = (blockIdx.x - req_id) / batch_size;
  // n and k are offset from their physical meaning: n = real_n - 2, k = real_k
  // - 1 (they index into ne_weights / ne_mods).
  const int k = config_id % ne_k;
  const int n = (config_id - config_id % ne_k) / ne_k;
  const int ne_weight_base_idx = n * ne_k * ne_n + k * ne_n;
  const int ne_mod = ne_mods[n * ne_k + k];
  for (int i = exclusive_req_len_sums[req_id] + threadIdx.x;
       i < exclusive_req_len_sums[req_id + 1]; i += blockDim.x) {
    uint64_t n_gram_id = 0;
    const int64_t current_token_offset = i - exclusive_req_len_sums[req_id];
    const int64_t req_token_table_index =
        row_indices[req_id] * static_cast<int64_t>(max_context_len);
    const int64_t current_token_table_index =
        req_token_table_index + column_starts[req_id] + current_token_offset;
    for (int j = 0; j < n + 2; j++) {
      if (current_token_table_index - j < req_token_table_index) {
        break;  // outside this request's range
      }
      if (ne_token_table[current_token_table_index - j] < 0) {
        break;  // ignored token
      }
      const uint64_t term =
          (uint64_t)ne_token_table[current_token_table_index - j] *
          (uint64_t)ne_weights[ne_weight_base_idx + j];
      n_gram_id += term % ne_mod;
    }
    n_gram_id %= ne_mod;
    n_gram_id += exclusive_ne_embedder_size_sums[n * ne_k + k];
    n_gram_ids[i * (ne_n - 1) * ne_k + n * ne_k + k] = (int)(n_gram_id);
  }
}

}  // namespace vllm::ngram_embedding

void ngram_compute_n_gram_ids(
    int64_t ne_n, int64_t ne_k, torch::stable::Tensor& ne_weights,
    torch::stable::Tensor& ne_mods,
    torch::stable::Tensor& exclusive_ne_embedder_size_sums,
    torch::stable::Tensor& exclusive_req_len_sums,
    torch::stable::Tensor& ne_token_table, torch::stable::Tensor& row_indices,
    torch::stable::Tensor& column_starts, torch::stable::Tensor& n_gram_ids) {
  const int batch_size = static_cast<int>(exclusive_req_len_sums.size(0) - 1);
  const int max_context_len = static_cast<int>(ne_token_table.size(1));
  const int num_configs = (static_cast<int>(ne_n) - 1) * static_cast<int>(ne_k);
  const int grid_size = num_configs * batch_size;
  if (grid_size <= 0) return;

  const torch::stable::accelerator::DeviceGuard device_guard(
      ne_weights.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  vllm::ngram_embedding::ComputeNGramIdsKernel<<<
      grid_size, vllm::ngram_embedding::kBlockThreads, 0, stream>>>(
      batch_size, static_cast<int>(ne_n), static_cast<int>(ne_k),
      ne_weights.mutable_data_ptr<int32_t>(),
      ne_mods.mutable_data_ptr<int32_t>(),
      exclusive_ne_embedder_size_sums.mutable_data_ptr<int32_t>(),
      exclusive_req_len_sums.mutable_data_ptr<int32_t>(),
      ne_token_table.mutable_data_ptr<int32_t>(), max_context_len,
      row_indices.const_data_ptr<int64_t>(),
      column_starts.mutable_data_ptr<int32_t>(),
      n_gram_ids.mutable_data_ptr<int32_t>());
}
