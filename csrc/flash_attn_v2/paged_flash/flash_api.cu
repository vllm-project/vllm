#include "flash.h"
#include "static_switch.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel_traits.h"
#include "flash_fwd_launch_template.h"

// void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream)
// {
//     FP16_SWITCH(!params.is_bf16, [&] {
//         FWD_HEADDIM_SWITCH(params.d, [&] { run_mha_fwd_<elem_type, kHeadDim>(params, stream); });
//     });
// }

#define M_LOG2E 1.4426950408889634074  // log_2 e
    
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
  torch::Tensor& draft_lens,      // [num_seqs]
  int block_size,
  int max_context_len,
  int max_num_query) {
    size_t num_heads = query.size(2);
    size_t head_size = query.size(3);
    size_t num_seqs = query.size(0);
    size_t max_num_blocks_per_seq = DIVIDE_ROUND_UP(max_context_len, block_size);

    TORCH_CHECK(num_heads == 12, "only 12 heads are supported");
    TORCH_CHECK(head_size == 64, "only head size of 64 is supported");
    TORCH_CHECK(block_size == 16, "only block size of 16 is supported");
    TORCH_CHECK(num_kv_heads == num_heads, "MQA is not supported");
    TORCH_CHECK(query.dtype() == at::ScalarType::Half, "only half is supported");

    // create params
    Flash_fwd_params params;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = reinterpret_cast<void *>(query.data_ptr());
    params.k_ptr = reinterpret_cast<void *>(key_cache.data_ptr());
    params.v_ptr = reinterpret_cast<void *>(value_cache.data_ptr());

    // Calculate batch_stride using cu_seq
    params.q_row_stride = num_heads * head_size;
    params.k_row_stride = num_kv_heads * head_size;
    params.v_row_stride = num_kv_heads * head_size;
    params.q_head_stride = head_size;
    params.k_head_stride = head_size;
    params.v_head_stride = head_size;

    params.h = num_heads;
    params.h_k = num_kv_heads;
    params.h_h_k_ratio = params.h / params.h_k;

    params.o_ptr = reinterpret_cast<void *>(out.data_ptr());

    params.o_row_stride = num_heads * head_size;
    params.o_head_stride = head_size;

    // Set the dimensions.
    params.d = params.d_rounded = head_size;

    params.scale_softmax = 1.0 / std::sqrt(head_size);
    params.scale_softmax_log2 = params.scale_softmax * M_LOG2E;

    params.is_bf16 = false;
    params.is_causal = true;

    params.num_seqs = num_seqs;
    params.max_num_query = max_num_query;
    params.max_context_len = max_context_len;
    params.block_size = block_size;
    params.max_num_blocks_per_seq = max_num_blocks_per_seq;

    params.block_tables = reinterpret_cast<Flash_fwd_params::index_t*>(block_tables.data_ptr());
    params.context_lens = reinterpret_cast<Flash_fwd_params::index_t*>(context_lens.data_ptr());
    params.draft_lens = reinterpret_cast<Flash_fwd_params::index_t*>(draft_lens.data_ptr());

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    run_flash_fwd<flash::Flash_fwd_kernel_traits<64, 128, 128, 4>, /*Is_causal=*/true>(
        params, stream);

    return;
}
