
#include "decoder_xqa_impl_common.h"


// Overloading << operator for XQAKernelRuntimeHashKey
std::ostream& operator<<(std::ostream& os, const XQAKernelRuntimeHashKey& key) {
    os << "{kv_data_type: " << key.kv_data_type
       << ", head_size: " << key.head_size
       << ", beam_size: " << key.beam_size
       << ", num_q_heads_per_kv: " << key.num_q_heads_per_kv
       << ", m_tilesize: " << key.m_tilesize
       << ", tokens_per_page: " << key.tokens_per_page
       << ", paged_kv_cache: " << (key.paged_kv_cache ? "true" : "false")
       << ", multi_query_tokens: " << (key.multi_query_tokens ? "true" : "false")
       << "}";
    return os ;
}

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(XQAParams const& xqaParams)
{
    unsigned int head_size = xqaParams.head_size;
    unsigned int num_q_heads = xqaParams.num_q_heads;
    unsigned int num_kv_heads = xqaParams.num_kv_heads;
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "numQHeads should be multiple of numKVHeads.");
    unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
    unsigned int beam_width = xqaParams.beam_width;

    // Use mTileSize = 16 kernels when qSeqLen <= 16.vi
    unsigned int qSeqLen = static_cast<unsigned int>(xqaParams.generation_input_length);
    unsigned int mTileSize = qSeqLen <= 16 ? 16 : 32;
    // MultiQueryToken kernels can support any num_q_heads_over_kv that is power of 2.
    unsigned int kernel_num_q_heads_over_kv = xqaParams.multi_query_tokens ? 0 : num_q_heads_over_kv;
    // MultiQueryToken kernels can handle either 16/32 for M direction per CTA.
    unsigned int kernel_m_tilesize = xqaParams.multi_query_tokens ? mTileSize : num_q_heads_over_kv;
    
    return {xqaParams.kv_cache_data_type, head_size, beam_width, kernel_num_q_heads_over_kv, kernel_m_tilesize,
        xqaParams.paged_kv_cache ? static_cast<unsigned int>(xqaParams.tokens_per_block) : 0, xqaParams.paged_kv_cache,
        xqaParams.multi_query_tokens};
}

// Setup launch params and ioScratch. ioScratch is for RoPE and output type conversion. not used
void buildXQALaunchParams(XQALaunchParam& launchParams,  XQAParams const& params,
    KVCacheListParams kv_cache_buffer) {
    TORCH_CHECK(
        params.data_type == DATA_TYPE_FP16 || params.data_type == DATA_TYPE_BF16, "Only fp16 or bf16 supported now.");
    memset(&launchParams, 0, sizeof(XQALaunchParam));
    launchParams.num_k_heads = params.num_kv_heads;
    launchParams.output = static_cast<uint8_t*>(params.output);
    // launchParams.qkv = static_cast<uint8_t const*>(params.qkv);
    launchParams.batch_size = params.batch_size;
    launchParams.kv_scale_quant_orig = params.kv_scale_quant_orig;
    launchParams.semaphores = params.semaphores;

    // Workspace.
    int8_t* workspace = reinterpret_cast<int8_t*>(params.workspaces);
    

    // workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(
    //     workspace, 2 * params.head_size * params.num_q_heads * params.total_num_input_tokens);
    // unsigned int batch_beam_size = params.batch_size * params.beam_width;
    // const size_t cu_seqlens_size = sizeof(int) * (batch_beam_size + 0);
    // launchParams.cu_seq_lens (workspace);
    // launchParams.cu_seq_lens = launchParams.cu_seq_lens;
    // workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, cu_seqlens_size);
    // launchParams.rotary_inv_freq_buf = reinterpret_cast<float*>(workspace);
    // auto const multi_block_workspace_alignment = tensorrt_llm::common::roundUp(
    //     sizeof(half) * params.head_size * (params.num_q_heads / params.num_kv_heads) * params.beam_width, 128);
    // workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(
    //     workspace, rotary_inv_freq_size, multi_block_workspace_alignment);
  
    launchParams.scratch = reinterpret_cast<void*>(workspace);

    launchParams.kvCacheParams = kv_cache_buffer;
}