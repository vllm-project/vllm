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

void paged_flash_attention(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& fquery,           // [num_seqs, max_num_query, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& head_mapping,    // [num_heads]
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len,
  int max_num_query,
  const c10::optional<torch::Tensor>& alibi_slopes) {
    cutlass::half_t* out_ptr = reinterpret_cast<cutlass::half_t*>(out.data_ptr());
    typename cutlass::NumericConverter<cutlass::half_t, float> converter;
    cutlass::half_t data = converter.convert(0.5f);
    int error = cudaMemcpy(out_ptr, &data, 2, cudaMemcpyHostToDevice);
    TORCH_CHECK(error == 0, "an error ocurred in paged flash attention");
    return;
}
