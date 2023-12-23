#include "flash.h"
#include "flash_fwd_kernel.h"
#include "static_switch.h"
#include <torch/extension.h>
#include <cuda_runtime.h>

// void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream)
// {
//     FP16_SWITCH(!params.is_bf16, [&] {
//         FWD_HEADDIM_SWITCH(params.d, [&] { run_mha_fwd_<elem_type, kHeadDim>(params, stream); });
//     });
// }

// for now, assume that
// head_dim = 64
// block_size = 16
// num_heads = 12
void paged_flash_attention(
  torch::Tensor& out,             // [num_seqs, max_num_query, num_heads, head_size]
  torch::Tensor& query,           // [num_seqs, max_num_query, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len,
  int max_num_query,
  const c10::optional<torch::Tensor>& alibi_slopes) {
    int num_heads = query.size(2);
    int head_size = query.size(3);

    TORCH_CHECK(num_heads == 12, "only 12 heads are supported");
    TORCH_CHECK(head_size == 64, "only head size of 64 is supported");
    TORCH_CHECK(block_size == 16, "only block size of 16 is supported");
    TORCH_CHECK(num_kv_heads == num_heads, "MQA is not supported");
    TORCH_CHECK(query.dtype() == at::ScalarType::Half, "only half is supported");
    
    return;
}
