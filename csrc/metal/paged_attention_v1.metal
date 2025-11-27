#include <metal_stdlib>
using namespace metal;

// Kernel argument structures (must match C++ definitions)
struct PagedAttentionArgsV1 {
    int32_t num_seqs;
    int32_t num_heads;
    int32_t head_size;
    int32_t max_num_blocks_per_seq;
    int32_t num_kv_heads;
    int32_t block_size;
    float scale;
    float alibi_slope;
    float kv_scale;
    int64_t query_stride_0;
    int64_t query_stride_1;
    int64_t output_stride_0;
    int64_t output_stride_1;
    int32_t x;
    int64_t k_cache_stride_block;
    int64_t k_cache_stride_head;
    int64_t k_cache_stride_dim;
    int64_t k_cache_stride_token;
    int64_t v_cache_stride_block;
    int64_t v_cache_stride_head;
    int64_t v_cache_stride_dim;
    int64_t block_tables_stride;
};

// Helper function to compute softmax
template<typename T>
inline T warp_reduce_sum(T val, uint simd_lane_id, uint simd_size) {
    for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

template<typename T>
inline T warp_reduce_max(T val, uint simd_lane_id, uint simd_size) {
    for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
        val = max(val, simd_shuffle_down(val, offset));
    }
    return val;
}

// Paged Attention V1 Kernel
// Grid: (num_heads, num_seqs, 1)
// Threadgroup: (NUM_THREADS, 1, 1)
template<typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
kernel void paged_attention_v1_kernel(
    // Kernel arguments
    constant PagedAttentionArgsV1 & args [[buffer(0)]],

    // Input tensors
    device const scalar_t * query [[buffer(1)]],           // [num_seqs, num_heads, head_size]
    device const cache_t * k_cache [[buffer(2)]],          // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    device const cache_t * v_cache [[buffer(3)]],          // [num_blocks, num_kv_heads, head_size, block_size]
    device const int32_t * block_tables [[buffer(4)]],     // [num_seqs, max_num_blocks_per_seq]
    device const int32_t * seq_lens [[buffer(5)]],         // [num_seqs]

    // Output tensor
    device scalar_t * output [[buffer(6)]],                // [num_seqs, num_heads, head_size]

    // Shared memory for reduction
    threadgroup float * shared_logits [[threadgroup(0)]],  // [MAX_SEQ_LEN]
    threadgroup scalar_t * shared_output [[threadgroup(1)]], // [HEAD_SIZE]

    // Thread indices
    uint3 tgpig [[threadgroup_position_in_grid]],          // (head_idx, seq_idx, 0)
    uint3 tpitg [[thread_position_in_threadgroup]])        // thread_id within group
{
    const int head_idx = tgpig.x;
    const int seq_idx = tgpig.y;
    const int thread_id = tpitg.x;
    const uint simd_lane_id = thread_id % 32;
    const uint simd_group_id = thread_id / 32;

    const int num_heads = args.num_heads;
    const int num_kv_heads = args.num_kv_heads;
    const int head_size = args.head_size;
    const int block_size = args.block_size;
    const int max_num_blocks_per_seq = args.max_num_blocks_per_seq;
    const float scale = args.scale;
    const int x = args.x;

    // Get sequence length
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Calculate KV head index (for grouped query attention)
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Load query vector into registers
    const int query_offset = seq_idx * args.query_stride_0 + head_idx * args.query_stride_1;
    device const scalar_t * q_ptr = query + query_offset;

    float q_vec[HEAD_SIZE];
    for (int i = 0; i < HEAD_SIZE; i++) {
        q_vec[i] = float(q_ptr[i]);
    }

    // Phase 1: Compute attention logits (Q * K^T)
    float max_logit = -INFINITY;

    // Get block table for this sequence
    device const int32_t * block_table = block_tables + seq_idx * args.block_tables_stride;

    // Number of blocks for this sequence
    const int num_blocks = (seq_len + block_size - 1) / block_size;

    // Each thread processes multiple tokens
    const int tokens_per_thread = (seq_len + NUM_THREADS - 1) / NUM_THREADS;
    const int start_token_idx = thread_id * tokens_per_thread;
    const int end_token_idx = min(start_token_idx + tokens_per_thread, seq_len);

    for (int token_idx = start_token_idx; token_idx < end_token_idx; token_idx++) {
        // Map token to block and position within block
        const int block_idx = token_idx / block_size;
        const int block_offset = token_idx % block_size;

        // Get physical block number from block table
        const int physical_block = block_table[block_idx];

        // Load key vector from paged cache
        // K cache layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        const int k_base_offset = physical_block * args.k_cache_stride_block +
                                   kv_head_idx * args.k_cache_stride_head;

        // Compute dot product Q · K
        float qk = 0.0f;
        for (int i = 0; i < head_size; i++) {
            const int dim_idx = i / x;
            const int x_idx = i % x;
            const int k_offset = k_base_offset +
                                dim_idx * args.k_cache_stride_dim +
                                block_offset * args.k_cache_stride_token +
                                x_idx;
            const float k_val = float(k_cache[k_offset]) * args.kv_scale;
            qk += q_vec[i] * k_val;
        }

        // Scale and apply ALiBi if needed
        float logit = qk * scale;
        if (args.alibi_slope != 0.0f) {
            logit += args.alibi_slope * float(token_idx - seq_len + 1);
        }

        // Store logit in shared memory
        shared_logits[token_idx] = logit;
        max_logit = max(max_logit, logit);
    }

    // Synchronize to ensure all logits are computed
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Reduce to find max logit across all threads
    max_logit = warp_reduce_max(max_logit, simd_lane_id, 32);

    // First thread in each SIMD group writes to shared memory
    if (simd_lane_id == 0) {
        shared_logits[NUM_THREADS + simd_group_id] = max_logit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first warp
    if (simd_group_id == 0) {
        const int num_simd_groups = (NUM_THREADS + 31) / 32;
        float local_max = (thread_id < num_simd_groups) ?
                         shared_logits[NUM_THREADS + thread_id] : -INFINITY;
        local_max = warp_reduce_max(local_max, simd_lane_id, 32);
        if (thread_id == 0) {
            shared_logits[NUM_THREADS] = local_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float global_max_logit = shared_logits[NUM_THREADS];

    // Phase 3: Compute exp(logit - max) and sum
    float exp_sum = 0.0f;
    for (int token_idx = start_token_idx; token_idx < end_token_idx; token_idx++) {
        float exp_logit = exp(shared_logits[token_idx] - global_max_logit);
        shared_logits[token_idx] = exp_logit;
        exp_sum += exp_logit;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce exp_sum across threads
    exp_sum = warp_reduce_sum(exp_sum, simd_lane_id, 32);
    if (simd_lane_id == 0) {
        shared_logits[NUM_THREADS + simd_group_id] = exp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        const int num_simd_groups = (NUM_THREADS + 31) / 32;
        float local_sum = (thread_id < num_simd_groups) ?
                         shared_logits[NUM_THREADS + thread_id] : 0.0f;
        local_sum = warp_reduce_sum(local_sum, simd_lane_id, 32);
        if (thread_id == 0) {
            shared_logits[NUM_THREADS] = local_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float global_exp_sum = shared_logits[NUM_THREADS];

    // Phase 4: Compute weighted sum of values (attention_weights * V)
    // Each thread handles a subset of output dimensions
    const int dims_per_thread = (head_size + NUM_THREADS - 1) / NUM_THREADS;
    const int start_dim = thread_id * dims_per_thread;
    const int end_dim = min(start_dim + dims_per_thread, head_size);

    for (int d = start_dim; d < end_dim; d++) {
        float acc = 0.0f;

        for (int token_idx = 0; token_idx < seq_len; token_idx++) {
            // Map token to block and position
            const int block_idx = token_idx / block_size;
            const int block_offset = token_idx % block_size;
            const int physical_block = block_table[block_idx];

            // Load value from paged cache
            // V cache layout: [num_blocks, num_kv_heads, head_size, block_size]
            const int v_offset = physical_block * args.v_cache_stride_block +
                                kv_head_idx * args.v_cache_stride_head +
                                d * args.v_cache_stride_dim +
                                block_offset;

            const float v_val = float(v_cache[v_offset]) * args.kv_scale;
            const float attention_weight = shared_logits[token_idx] / global_exp_sum;

            acc += attention_weight * v_val;
        }

        // Store output
        shared_output[d] = scalar_t(acc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output to global memory
    const int output_offset = seq_idx * args.output_stride_0 + head_idx * args.output_stride_1;
    device scalar_t * out_ptr = output + output_offset;

    for (int d = thread_id; d < head_size; d += NUM_THREADS) {
        out_ptr[d] = shared_output[d];
    }
}

// Kernel instantiations for different head sizes and data types
#define INSTANTIATE_PAGED_ATTENTION_V1(DTYPE, CACHE_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS) \
    template [[host_name("paged_attention_v1_" #DTYPE "_" #CACHE_DTYPE "_h" #HEAD_SIZE "_b" #BLOCK_SIZE "_t" #NUM_THREADS)]] \
    kernel void paged_attention_v1_kernel<DTYPE, CACHE_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>( \
        constant PagedAttentionArgsV1 &, \
        device const DTYPE *, device const CACHE_DTYPE *, device const CACHE_DTYPE *, \
        device const int32_t *, device const int32_t *, device DTYPE *, \
        threadgroup float *, threadgroup DTYPE *, \
        uint3, uint3);

// float32 variants
INSTANTIATE_PAGED_ATTENTION_V1(float, float, 64, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(float, float, 80, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(float, float, 96, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(float, float, 112, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(float, float, 128, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(float, float, 256, 16, 128)

// float16 variants
INSTANTIATE_PAGED_ATTENTION_V1(half, half, 64, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(half, half, 80, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(half, half, 96, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(half, half, 112, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(half, half, 128, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(half, half, 256, 16, 128)

// Mixed precision (float query, half cache)
INSTANTIATE_PAGED_ATTENTION_V1(float, half, 64, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(float, half, 80, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(float, half, 96, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(float, half, 112, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(float, half, 128, 16, 128)
INSTANTIATE_PAGED_ATTENTION_V1(float, half, 256, 16, 128)
