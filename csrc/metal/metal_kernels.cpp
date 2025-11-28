#include "metal_kernels.h"
#include "metal_context.h"
#include <stdexcept>
#include <sstream>

namespace vllm {
namespace metal {

// Helper to get Metal buffer pointer from PyTorch tensor
static void* get_metal_buffer(const torch::Tensor& tensor) {
    // For MPS tensors, get the underlying Metal buffer
    // The tensor must be contiguous for Metal operations
    if (!tensor.is_contiguous()) {
        throw std::runtime_error("Tensor must be contiguous for Metal operations");
    }

    // Get the data pointer - for MPS tensors, this should be the MTLBuffer
    void* buffer = const_cast<void*>(tensor.storage().data());

    if (!buffer) {
        throw std::runtime_error("Failed to get Metal buffer from tensor");
    }

    return buffer;
}

// Helper to construct kernel name
std::string get_paged_attention_kernel_name(
    const std::string& base_name,
    torch::ScalarType query_type,
    torch::ScalarType cache_type,
    int head_size,
    int block_size,
    int num_threads) {

    std::ostringstream oss;
    oss << base_name;

    // Add dtype suffix
    if (query_type == torch::kFloat32) {
        oss << "_float";
    } else if (query_type == torch::kFloat16) {
        oss << "_half";
    } else {
        throw std::runtime_error("Unsupported query dtype");
    }

    oss << "_";

    if (cache_type == torch::kFloat32) {
        oss << "float";
    } else if (cache_type == torch::kFloat16) {
        oss << "half";
    } else {
        throw std::runtime_error("Unsupported cache dtype");
    }

    oss << "_h" << head_size;
    oss << "_b" << block_size;
    oss << "_t" << num_threads;

    return oss.str();
}

void paged_attention_v1(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int num_kv_heads,
    float scale,
    int block_size,
    int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const c10::optional<torch::Tensor>& kv_cache_scales) {

    // Get Metal context
    MetalContext* ctx = get_metal_context();

    // Extract dimensions
    const int num_seqs = query.size(0);
    const int num_heads = query.size(1);
    const int head_size = query.size(2);
    const int max_num_blocks_per_seq = block_tables.size(1);

    // Vectorization factor
    const int x = 16;

    // Prepare kernel arguments
    PagedAttentionArgsV1 args;
    args.num_seqs = num_seqs;
    args.num_heads = num_heads;
    args.head_size = head_size;
    args.max_num_blocks_per_seq = max_num_blocks_per_seq;
    args.num_kv_heads = num_kv_heads;
    args.block_size = block_size;
    args.scale = scale;
    args.alibi_slope = 0.0f;  // TODO: support ALiBi
    args.kv_scale = kv_cache_scales.has_value() ? kv_cache_scales.value().item<float>() : 1.0f;

    // Query strides
    args.query_stride_0 = query.stride(0);
    args.query_stride_1 = query.stride(1);

    // Output strides (same as query)
    args.output_stride_0 = out.stride(0);
    args.output_stride_1 = out.stride(1);

    // K cache strides
    args.x = x;
    args.k_cache_stride_block = key_cache.stride(0);
    args.k_cache_stride_head = key_cache.stride(1);
    args.k_cache_stride_dim = key_cache.stride(2);
    args.k_cache_stride_token = key_cache.stride(3);

    // V cache strides
    args.v_cache_stride_block = value_cache.stride(0);
    args.v_cache_stride_head = value_cache.stride(1);
    args.v_cache_stride_dim = value_cache.stride(2);

    // Block tables stride
    args.block_tables_stride = block_tables.stride(0);

    // Get kernel name
    const int num_threads = 128;
    std::string kernel_name = get_paged_attention_kernel_name(
        "paged_attention_v1",
        query.scalar_type(),
        key_cache.scalar_type(),
        head_size,
        block_size,
        num_threads);

    // Get pipeline state
    void* pipeline = ctx->get_pipeline_state(kernel_name);
    if (!pipeline) {
        throw std::runtime_error("Failed to get pipeline for kernel: " + kernel_name);
    }

    // Create command buffer and encoder
    void* cmd_buffer = ctx->create_command_buffer();
    void* encoder = ctx->create_compute_encoder(cmd_buffer);

    // Set pipeline
    ctx->set_pipeline_state(encoder, pipeline);

    // Set arguments
    ctx->set_bytes(encoder, &args, sizeof(args), 0);
    ctx->set_buffer(encoder, get_metal_buffer(query), 0, 1);
    ctx->set_buffer(encoder, get_metal_buffer(key_cache), 0, 2);
    ctx->set_buffer(encoder, get_metal_buffer(value_cache), 0, 3);
    ctx->set_buffer(encoder, get_metal_buffer(block_tables), 0, 4);
    ctx->set_buffer(encoder, get_metal_buffer(seq_lens), 0, 5);
    ctx->set_buffer(encoder, get_metal_buffer(out), 0, 6);

    // Allocate threadgroup memory
    // shared_logits: needs space for both logits and reduction temporaries
    // The kernel uses indices [0, seq_len) for logits and [NUM_THREADS, NUM_THREADS + num_simd_groups] for reductions
    const int num_simd_groups = (num_threads + 31) / 32;
    const int shared_logits_count = std::max(max_seq_len, num_threads + num_simd_groups + 1);
    const size_t shared_logits_size = shared_logits_count * sizeof(float);
    ctx->set_threadgroup_memory_length(encoder, shared_logits_size, 0);

    // shared_output: [HEAD_SIZE] scalar_t (float or half)
    const size_t scalar_size = (query.scalar_type() == torch::kFloat32) ? sizeof(float) : sizeof(uint16_t);
    const size_t shared_output_size = head_size * scalar_size;
    ctx->set_threadgroup_memory_length(encoder, shared_output_size, 1);

    // Dispatch: Grid (num_heads, num_seqs, 1), Threadgroup (num_threads, 1, 1)
    ctx->dispatch_threadgroups(encoder, num_heads, num_seqs, 1, num_threads, 1, 1);

    // End encoding and execute
    ctx->end_encoding(encoder);
    ctx->commit(cmd_buffer);
    ctx->wait_until_completed(cmd_buffer);
}

void paged_attention_v2(
    torch::Tensor& out,
    torch::Tensor& exp_sums,
    torch::Tensor& max_logits,
    torch::Tensor& tmp_out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int num_kv_heads,
    float scale,
    int block_size,
    int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const c10::optional<torch::Tensor>& kv_cache_scales) {

    MetalContext* ctx = get_metal_context();

    const int num_seqs = query.size(0);
    const int num_heads = query.size(1);
    const int head_size = query.size(2);
    const int max_num_blocks_per_seq = block_tables.size(1);
    const int max_num_partitions = exp_sums.size(2);
    const int partition_size = 512;
    const int x = 16;

    // Prepare base arguments
    PagedAttentionArgsV2 v2_args;
    PagedAttentionArgsV1& args = v2_args.base;

    args.num_seqs = num_seqs;
    args.num_heads = num_heads;
    args.head_size = head_size;
    args.max_num_blocks_per_seq = max_num_blocks_per_seq;
    args.num_kv_heads = num_kv_heads;
    args.block_size = block_size;
    args.scale = scale;
    args.alibi_slope = 0.0f;
    args.kv_scale = kv_cache_scales.has_value() ? kv_cache_scales.value().item<float>() : 1.0f;

    args.query_stride_0 = query.stride(0);
    args.query_stride_1 = query.stride(1);
    args.output_stride_0 = out.stride(0);
    args.output_stride_1 = out.stride(1);

    args.x = x;
    args.k_cache_stride_block = key_cache.stride(0);
    args.k_cache_stride_head = key_cache.stride(1);
    args.k_cache_stride_dim = key_cache.stride(2);
    args.k_cache_stride_token = key_cache.stride(3);

    args.v_cache_stride_block = value_cache.stride(0);
    args.v_cache_stride_head = value_cache.stride(1);
    args.v_cache_stride_dim = value_cache.stride(2);

    args.block_tables_stride = block_tables.stride(0);

    // V2-specific args
    v2_args.max_num_partitions = max_num_partitions;
    v2_args.partition_size = partition_size;
    v2_args.exp_sums_stride = exp_sums.stride(0);
    v2_args.max_logits_stride = max_logits.stride(0);

    // Get kernel names
    const int num_threads = 128;
    std::ostringstream kernel_name_oss;
    kernel_name_oss << "paged_attention_v2_";
    if (query.scalar_type() == torch::kFloat32) {
        kernel_name_oss << "float_";
    } else {
        kernel_name_oss << "half_";
    }
    if (key_cache.scalar_type() == torch::kFloat32) {
        kernel_name_oss << "float";
    } else {
        kernel_name_oss << "half";
    }
    kernel_name_oss << "_h" << head_size << "_b" << block_size
                   << "_t" << num_threads << "_p" << partition_size;

    std::string kernel_name = kernel_name_oss.str();

    // Reduce kernel name
    std::ostringstream reduce_kernel_name_oss;
    reduce_kernel_name_oss << "paged_attention_v2_reduce_";
    if (query.scalar_type() == torch::kFloat32) {
        reduce_kernel_name_oss << "float";
    } else {
        reduce_kernel_name_oss << "half";
    }
    reduce_kernel_name_oss << "_h" << head_size << "_t" << num_threads;
    std::string reduce_kernel_name = reduce_kernel_name_oss.str();

    // Phase 1: Compute partitions
    void* pipeline = ctx->get_pipeline_state(kernel_name);
    if (!pipeline) {
        throw std::runtime_error("Failed to get pipeline for kernel: " + kernel_name);
    }

    void* cmd_buffer = ctx->create_command_buffer();
    void* encoder = ctx->create_compute_encoder(cmd_buffer);

    ctx->set_pipeline_state(encoder, pipeline);
    ctx->set_bytes(encoder, &v2_args, sizeof(v2_args), 0);
    ctx->set_buffer(encoder, get_metal_buffer(query), 0, 1);
    ctx->set_buffer(encoder, get_metal_buffer(key_cache), 0, 2);
    ctx->set_buffer(encoder, get_metal_buffer(value_cache), 0, 3);
    ctx->set_buffer(encoder, get_metal_buffer(block_tables), 0, 4);
    ctx->set_buffer(encoder, get_metal_buffer(seq_lens), 0, 5);
    ctx->set_buffer(encoder, get_metal_buffer(exp_sums), 0, 6);
    ctx->set_buffer(encoder, get_metal_buffer(max_logits), 0, 7);
    ctx->set_buffer(encoder, get_metal_buffer(tmp_out), 0, 8);

    // Allocate threadgroup memory for v2 partition kernel
    // shared_logits: needs space for partition size
    const int num_simd_groups = (num_threads + 31) / 32;
    const int shared_logits_count = std::max(partition_size, num_threads + num_simd_groups + 1);
    const size_t shared_logits_size = shared_logits_count * sizeof(float);
    ctx->set_threadgroup_memory_length(encoder, shared_logits_size, 0);

    // shared_output: [HEAD_SIZE] scalar_t (float or half)
    const size_t scalar_size = (query.scalar_type() == torch::kFloat32) ? sizeof(float) : sizeof(uint16_t);
    const size_t shared_output_size = head_size * scalar_size;
    ctx->set_threadgroup_memory_length(encoder, shared_output_size, 1);

    // Dispatch: Grid (num_heads, num_seqs, max_num_partitions)
    ctx->dispatch_threadgroups(encoder, num_heads, num_seqs, max_num_partitions, num_threads, 1, 1);
    ctx->end_encoding(encoder);

    // Phase 2: Reduce partitions
    void* reduce_pipeline = ctx->get_pipeline_state(reduce_kernel_name);
    if (!reduce_pipeline) {
        throw std::runtime_error("Failed to get pipeline for reduce kernel: " + reduce_kernel_name);
    }

    void* reduce_encoder = ctx->create_compute_encoder(cmd_buffer);
    ctx->set_pipeline_state(reduce_encoder, reduce_pipeline);
    ctx->set_bytes(reduce_encoder, &v2_args, sizeof(v2_args), 0);
    ctx->set_buffer(reduce_encoder, get_metal_buffer(exp_sums), 0, 1);
    ctx->set_buffer(reduce_encoder, get_metal_buffer(max_logits), 0, 2);
    ctx->set_buffer(reduce_encoder, get_metal_buffer(tmp_out), 0, 3);
    ctx->set_buffer(reduce_encoder, get_metal_buffer(seq_lens), 0, 4);
    ctx->set_buffer(reduce_encoder, get_metal_buffer(out), 0, 5);

    // Allocate threadgroup memory for v2 reduce kernel
    // shared_data: needs space for max_num_partitions
    const size_t shared_data_size = max_num_partitions * sizeof(float);
    ctx->set_threadgroup_memory_length(reduce_encoder, shared_data_size, 0);

    // Dispatch: Grid (num_heads, num_seqs, 1)
    ctx->dispatch_threadgroups(reduce_encoder, num_heads, num_seqs, 1, num_threads, 1, 1);
    ctx->end_encoding(reduce_encoder);

    ctx->commit(cmd_buffer);
    ctx->wait_until_completed(cmd_buffer);
}

void reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping) {

    MetalContext* ctx = get_metal_context();

    const int num_tokens = key.size(0);
    const int num_heads = key.size(1);
    const int head_size = key.size(2);
    const int block_size = 16;  // Default block size for MPS
    const int x = 16;

    ReshapeAndCacheArgs args;
    args.num_tokens = num_tokens;
    args.num_heads = num_heads;
    args.head_size = head_size;
    args.block_size = block_size;
    args.x = x;

    args.key_stride_token = key.stride(0);
    args.key_stride_head = key.stride(1);
    args.value_stride_token = value.stride(0);
    args.value_stride_head = value.stride(1);

    args.k_cache_stride_block = key_cache.stride(0);
    args.k_cache_stride_head = key_cache.stride(1);
    args.k_cache_stride_dim = key_cache.stride(2);
    args.k_cache_stride_token = key_cache.stride(3);

    args.v_cache_stride_block = value_cache.stride(0);
    args.v_cache_stride_head = value_cache.stride(1);
    args.v_cache_stride_dim = value_cache.stride(2);

    // Get kernel name
    std::ostringstream oss;
    oss << "reshape_and_cache_";
    if (key.scalar_type() == torch::kFloat32) {
        oss << "float_float";
    } else {
        oss << "half_half";
    }
    oss << "_h" << head_size << "_b" << block_size << "_t128";

    std::string kernel_name = oss.str();
    void* pipeline = ctx->get_pipeline_state(kernel_name);
    if (!pipeline) {
        throw std::runtime_error("Failed to get pipeline for kernel: " + kernel_name);
    }

    void* cmd_buffer = ctx->create_command_buffer();
    void* encoder = ctx->create_compute_encoder(cmd_buffer);

    ctx->set_pipeline_state(encoder, pipeline);
    ctx->set_bytes(encoder, &args, sizeof(args), 0);
    ctx->set_buffer(encoder, get_metal_buffer(key), 0, 1);
    ctx->set_buffer(encoder, get_metal_buffer(value), 0, 2);
    ctx->set_buffer(encoder, get_metal_buffer(key_cache), 0, 3);
    ctx->set_buffer(encoder, get_metal_buffer(value_cache), 0, 4);
    ctx->set_buffer(encoder, get_metal_buffer(slot_mapping), 0, 5);

    // Dispatch: Grid (num_tokens, 1, 1), Threadgroup (128, 1, 1)
    ctx->dispatch_threadgroups(encoder, num_tokens, 1, 1, 128, 1, 1);

    ctx->end_encoding(encoder);
    ctx->commit(cmd_buffer);
    ctx->wait_until_completed(cmd_buffer);
}

void copy_blocks(
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& src_to_dst) {

    MetalContext* ctx = get_metal_context();

    const int num_pairs = src_to_dst.size(0);

    // Calculate block size in bytes
    const int block_size_in_bytes = key_cache.stride(0) * key_cache.element_size();

    BlockOperationArgs args;
    args.num_pairs = num_pairs;
    args.block_size_in_bytes = block_size_in_bytes;

    // Get kernel name
    std::string dtype_suffix = (key_cache.scalar_type() == torch::kFloat32) ? "float" : "half";
    std::string kernel_name = "copy_blocks_" + dtype_suffix + "_t256";

    void* pipeline = ctx->get_pipeline_state(kernel_name);
    if (!pipeline) {
        throw std::runtime_error("Failed to get pipeline for kernel: " + kernel_name);
    }

    void* cmd_buffer = ctx->create_command_buffer();

    // Copy key cache blocks
    void* encoder = ctx->create_compute_encoder(cmd_buffer);
    ctx->set_pipeline_state(encoder, pipeline);
    ctx->set_bytes(encoder, &args, sizeof(args), 0);
    ctx->set_buffer(encoder, get_metal_buffer(key_cache), 0, 1);
    ctx->set_buffer(encoder, get_metal_buffer(key_cache), 0, 2);
    ctx->set_buffer(encoder, get_metal_buffer(src_to_dst), 0, 3);
    ctx->dispatch_threadgroups(encoder, num_pairs, 1, 1, 256, 1, 1);
    ctx->end_encoding(encoder);

    // Copy value cache blocks
    encoder = ctx->create_compute_encoder(cmd_buffer);
    ctx->set_pipeline_state(encoder, pipeline);
    ctx->set_bytes(encoder, &args, sizeof(args), 0);
    ctx->set_buffer(encoder, get_metal_buffer(value_cache), 0, 1);
    ctx->set_buffer(encoder, get_metal_buffer(value_cache), 0, 2);
    ctx->set_buffer(encoder, get_metal_buffer(src_to_dst), 0, 3);
    ctx->dispatch_threadgroups(encoder, num_pairs, 1, 1, 256, 1, 1);
    ctx->end_encoding(encoder);

    ctx->commit(cmd_buffer);
    ctx->wait_until_completed(cmd_buffer);
}

void swap_blocks(
    torch::Tensor& src_cache,
    torch::Tensor& dst_cache,
    torch::Tensor& src_to_dst) {

    MetalContext* ctx = get_metal_context();

    const int num_pairs = src_to_dst.size(0);
    const int block_size_in_bytes = src_cache.stride(0) * src_cache.element_size();

    BlockOperationArgs args;
    args.num_pairs = num_pairs;
    args.block_size_in_bytes = block_size_in_bytes;

    std::string dtype_suffix = (src_cache.scalar_type() == torch::kFloat32) ? "float" : "half";
    std::string kernel_name = "swap_blocks_" + dtype_suffix + "_t256";

    void* pipeline = ctx->get_pipeline_state(kernel_name);
    if (!pipeline) {
        throw std::runtime_error("Failed to get pipeline for kernel: " + kernel_name);
    }

    void* cmd_buffer = ctx->create_command_buffer();
    void* encoder = ctx->create_compute_encoder(cmd_buffer);

    ctx->set_pipeline_state(encoder, pipeline);
    ctx->set_bytes(encoder, &args, sizeof(args), 0);
    ctx->set_buffer(encoder, get_metal_buffer(src_cache), 0, 1);
    ctx->set_buffer(encoder, get_metal_buffer(dst_cache), 0, 2);
    ctx->set_buffer(encoder, get_metal_buffer(src_to_dst), 0, 3);

    ctx->dispatch_threadgroups(encoder, num_pairs, 1, 1, 256, 1, 1);

    ctx->end_encoding(encoder);
    ctx->commit(cmd_buffer);
    ctx->wait_until_completed(cmd_buffer);
}

} // namespace metal
} // namespace vllm
