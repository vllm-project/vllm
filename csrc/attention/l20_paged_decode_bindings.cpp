#include <torch/all.h>
#include <torch/library.h>

#include "core/registration.h"

void l20_paged_decode_split_out_cuda(
    torch::Tensor query, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor seq_lens,
    torch::Tensor partial_output, torch::Tensor partial_max,
    torch::Tensor partial_sum, torch::Tensor output, int64_t max_seq_len,
    int64_t split_size);

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "l20_paged_decode_split_out("
      "Tensor query, Tensor key_cache, Tensor value_cache, "
      "Tensor block_table, Tensor seq_lens, "
      "Tensor(a!) partial_output, Tensor(b!) partial_max, "
      "Tensor(c!) partial_sum, Tensor(d!) output, "
      "int max_seq_len, int split_size) -> ()");
}

TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("l20_paged_decode_split_out", &l20_paged_decode_split_out_cuda);
}

REGISTER_EXTENSION(_l20_C)
