#include <metal_stdlib>
using namespace metal;

// Kernel argument structures
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

struct PagedAttentionArgsV2 {
    PagedAttentionArgsV1 base;
    int32_t max_num_partitions;
    int32_t partition_size;
    int64_t exp_sums_stride;
    int64_t max_logits_stride;
};

// Helper functions
template<typename T>
inline T warp_reduce_sum(T val, uint simd_lane_id) {
    for (uint offset = 16; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

template<typename T>
inline T warp_reduce_max(T val, uint simd_lane_id) {
    for (uint offset = 16; offset > 0; offset /= 2) {
        val = max(val, simd_shuffle_down(val, offset));
    }
    return val;
}

// Paged Attention V2 Kernel - Computes attention for one partition
// Grid: (num_heads, num_seqs, max_num_partitions)
// Threadgroup: (NUM_THREADS, 1, 1)
template<typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
         int NUM_THREADS, int PARTITION_SIZE>
kernel void paged_attention_v2_kernel(
    // Kernel arguments
    constant PagedAttentionArgsV2 & args [[buffer(0)]],

    // Input tensors
    device const scalar_t * query [[buffer(1)]],           // [num_seqs, num_heads, head_size]
    device const cache_t * k_cache [[buffer(2)]],          // [num_blocks, ...]
    device const cache_t * v_cache [[buffer(3)]],          // [num_blocks, ...]
    device const int32_t * block_tables [[buffer(4)]],     // [num_seqs, max_num_blocks_per_seq]
    device const int32_t * seq_lens [[buffer(5)]],         // [num_seqs]

    // Temporary buffers for partitioned results
    device float * exp_sums [[buffer(6)]],                 // [num_seqs, num_heads, max_num_partitions]
    device float * max_logits [[buffer(7)]],               // [num_seqs, num_heads, max_num_partitions]
    device scalar_t * tmp_output [[buffer(8)]],            // [num_seqs, num_heads, max_num_partitions, head_size]

    // Shared memory
    threadgroup float * shared_logits [[threadgroup(0)]],
    threadgroup scalar_t * shared_output [[threadgroup(1)]],

    // Thread indices
    uint3 tgpig [[threadgroup_position_in_grid]],          // (head_idx, seq_idx, partition_idx)
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const int head_idx = tgpig.x;
    const int seq_idx = tgpig.y;
    const int partition_idx = tgpig.z;
    const int thread_id = tpitg.x;
    const uint simd_lane_id = thread_id % 32;
    const uint simd_group_id = thread_id / 32;

    constant PagedAttentionArgsV1 & base = args.base;
    const int num_heads = base.num_heads;
    const int num_kv_heads = base.num_kv_heads;
    const int head_size = base.head_size;
    const int block_size = base.block_size;
    const float scale = base.scale;
    const int x = base.x;
    const int partition_size = args.partition_size;

    // Get sequence length
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Calculate partition bounds
    const int partition_start = partition_idx * partition_size;
    const int partition_end = min(partition_start + partition_size, seq_len);

    // Early exit if partition is out of bounds
    if (partition_start >= seq_len) {
        // Write default values
        const int tmp_idx = seq_idx * base.num_heads * args.max_num_partitions +
                           head_idx * args.max_num_partitions + partition_idx;
        if (thread_id == 0) {
            exp_sums[tmp_idx] = 0.0f;
            max_logits[tmp_idx] = -INFINITY;
        }
        return;
    }

    const int partition_len = partition_end - partition_start;

    // Calculate KV head index
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Load query vector
    const int query_offset = seq_idx * base.query_stride_0 + head_idx * base.query_stride_1;
    device const scalar_t * q_ptr = query + query_offset;

    float q_vec[HEAD_SIZE];
    for (int i = 0; i < HEAD_SIZE; i++) {
        q_vec[i] = float(q_ptr[i]);
    }

    // Get block table
    device const int32_t * block_table = block_tables + seq_idx * base.block_tables_stride;

    // Phase 1: Compute logits for this partition
    float max_logit = -INFINITY;

    const int tokens_per_thread = (partition_len + NUM_THREADS - 1) / NUM_THREADS;
    const int start_token_local = thread_id * tokens_per_thread;
    const int end_token_local = min(start_token_local + tokens_per_thread, partition_len);

    for (int local_idx = start_token_local; local_idx < end_token_local; local_idx++) {
        const int token_idx = partition_start + local_idx;

        // Map token to block
        const int block_idx = token_idx / block_size;
        const int block_offset = token_idx % block_size;
        const int physical_block = block_table[block_idx];

        // Load key and compute Q·K
        const int k_base_offset = physical_block * base.k_cache_stride_block +
                                   kv_head_idx * base.k_cache_stride_head;

        float qk = 0.0f;
        for (int i = 0; i < head_size; i++) {
            const int dim_idx = i / x;
            const int x_idx = i % x;
            const int k_offset = k_base_offset +
                                dim_idx * base.k_cache_stride_dim +
                                block_offset * base.k_cache_stride_token +
                                x_idx;
            const float k_val = float(k_cache[k_offset]) * base.kv_scale;
            qk += q_vec[i] * k_val;
        }

        // Scale and apply ALiBi
        float logit = qk * scale;
        if (base.alibi_slope != 0.0f) {
            logit += base.alibi_slope * float(token_idx - seq_len + 1);
        }

        shared_logits[local_idx] = logit;
        max_logit = max(max_logit, logit);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Reduce max logit
    max_logit = warp_reduce_max(max_logit, simd_lane_id);
    if (simd_lane_id == 0) {
        shared_logits[partition_size + simd_group_id] = max_logit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        const int num_simd_groups = (NUM_THREADS + 31) / 32;
        float local_max = (thread_id < num_simd_groups) ?
                         shared_logits[partition_size + thread_id] : -INFINITY;
        local_max = warp_reduce_max(local_max, simd_lane_id);
        if (thread_id == 0) {
            shared_logits[partition_size] = local_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partition_max_logit = shared_logits[partition_size];

    // Phase 3: Compute exp and sum
    float exp_sum = 0.0f;
    for (int local_idx = start_token_local; local_idx < end_token_local; local_idx++) {
        float exp_logit = exp(shared_logits[local_idx] - partition_max_logit);
        shared_logits[local_idx] = exp_logit;
        exp_sum += exp_logit;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    exp_sum = warp_reduce_sum(exp_sum, simd_lane_id);
    if (simd_lane_id == 0) {
        shared_logits[partition_size + simd_group_id] = exp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        const int num_simd_groups = (NUM_THREADS + 31) / 32;
        float local_sum = (thread_id < num_simd_groups) ?
                         shared_logits[partition_size + thread_id] : 0.0f;
        local_sum = warp_reduce_sum(local_sum, simd_lane_id);
        if (thread_id == 0) {
            shared_logits[partition_size] = local_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partition_exp_sum = shared_logits[partition_size];

    // Phase 4: Compute partial output (weighted values)
    const int dims_per_thread = (head_size + NUM_THREADS - 1) / NUM_THREADS;
    const int start_dim = thread_id * dims_per_thread;
    const int end_dim = min(start_dim + dims_per_thread, head_size);

    for (int d = start_dim; d < end_dim; d++) {
        float acc = 0.0f;

        for (int local_idx = 0; local_idx < partition_len; local_idx++) {
            const int token_idx = partition_start + local_idx;
            const int block_idx = token_idx / block_size;
            const int block_offset = token_idx % block_size;
            const int physical_block = block_table[block_idx];

            // Load value
            const int v_offset = physical_block * base.v_cache_stride_block +
                                kv_head_idx * base.v_cache_stride_head +
                                d * base.v_cache_stride_dim +
                                block_offset;

            const float v_val = float(v_cache[v_offset]) * base.kv_scale;
            const float attention_weight = shared_logits[local_idx];

            acc += attention_weight * v_val;
        }

        shared_output[d] = scalar_t(acc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write partition results to temporary buffers
    const int tmp_base_idx = seq_idx * base.num_heads * args.max_num_partitions +
                             head_idx * args.max_num_partitions + partition_idx;

    if (thread_id == 0) {
        exp_sums[tmp_base_idx] = partition_exp_sum;
        max_logits[tmp_base_idx] = partition_max_logit;
    }

    // Write partial output
    const int output_base = tmp_base_idx * head_size;
    for (int d = thread_id; d < head_size; d += NUM_THREADS) {
        tmp_output[output_base + d] = shared_output[d];
    }
}

// Reduction kernel to combine partition results
// Grid: (num_heads, num_seqs, 1)
// Threadgroup: (NUM_THREADS, 1, 1)
template<typename scalar_t, int HEAD_SIZE, int NUM_THREADS>
kernel void paged_attention_v2_reduce_kernel(
    constant PagedAttentionArgsV2 & args [[buffer(0)]],

    device const float * exp_sums [[buffer(1)]],           // [num_seqs, num_heads, max_num_partitions]
    device const float * max_logits [[buffer(2)]],         // [num_seqs, num_heads, max_num_partitions]
    device const scalar_t * tmp_output [[buffer(3)]],      // [num_seqs, num_heads, max_num_partitions, head_size]
    device const int32_t * seq_lens [[buffer(4)]],         // [num_seqs]

    device scalar_t * final_output [[buffer(5)]],          // [num_seqs, num_heads, head_size]

    threadgroup float * shared_data [[threadgroup(0)]],

    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const int head_idx = tgpig.x;
    const int seq_idx = tgpig.y;
    const int thread_id = tpitg.x;

    constant PagedAttentionArgsV1 & base = args.base;
    const int num_heads = base.num_heads;
    const int head_size = base.head_size;
    const int partition_size = args.partition_size;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int num_partitions = (seq_len + partition_size - 1) / partition_size;

    // Find global max logit across all partitions
    float global_max = -INFINITY;
    const int partitions_base = seq_idx * num_heads * args.max_num_partitions +
                                head_idx * args.max_num_partitions;

    for (int p = 0; p < num_partitions; p++) {
        global_max = max(global_max, max_logits[partitions_base + p]);
    }

    // Compute rescaled exp sums
    float global_exp_sum = 0.0f;
    for (int p = 0; p < num_partitions; p++) {
        const float partition_max = max_logits[partitions_base + p];
        const float partition_sum = exp_sums[partitions_base + p];
        global_exp_sum += partition_sum * exp(partition_max - global_max);
    }

    // Combine partition outputs with proper weighting
    const int dims_per_thread = (head_size + NUM_THREADS - 1) / NUM_THREADS;
    const int start_dim = thread_id * dims_per_thread;
    const int end_dim = min(start_dim + dims_per_thread, head_size);

    for (int d = start_dim; d < end_dim; d++) {
        float acc = 0.0f;

        for (int p = 0; p < num_partitions; p++) {
            const float partition_max = max_logits[partitions_base + p];
            const float partition_sum = exp_sums[partitions_base + p];
            const float partition_weight = partition_sum * exp(partition_max - global_max) / global_exp_sum;

            const int tmp_idx = partitions_base + p;
            const float partial_out = float(tmp_output[tmp_idx * head_size + d]);

            acc += partition_weight * partial_out;
        }

        shared_data[d] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write final output
    const int output_offset = seq_idx * base.output_stride_0 + head_idx * base.output_stride_1;
    device scalar_t * out_ptr = final_output + output_offset;

    for (int d = thread_id; d < head_size; d += NUM_THREADS) {
        out_ptr[d] = scalar_t(shared_data[d]);
    }
}

// Kernel instantiations
#define INSTANTIATE_PAGED_ATTENTION_V2(DTYPE, CACHE_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE) \
    template [[host_name("paged_attention_v2_" #DTYPE "_" #CACHE_DTYPE "_h" #HEAD_SIZE "_b" #BLOCK_SIZE "_t" #NUM_THREADS "_p" #PARTITION_SIZE)]] \
    kernel void paged_attention_v2_kernel<DTYPE, CACHE_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>( \
        constant PagedAttentionArgsV2 &, \
        device const DTYPE *, device const CACHE_DTYPE *, device const CACHE_DTYPE *, \
        device const int32_t *, device const int32_t *, \
        device float *, device float *, device DTYPE *, \
        threadgroup float *, threadgroup DTYPE *, \
        uint3, uint3);

#define INSTANTIATE_PAGED_ATTENTION_V2_REDUCE(DTYPE, HEAD_SIZE, NUM_THREADS) \
    template [[host_name("paged_attention_v2_reduce_" #DTYPE "_h" #HEAD_SIZE "_t" #NUM_THREADS)]] \
    kernel void paged_attention_v2_reduce_kernel<DTYPE, HEAD_SIZE, NUM_THREADS>( \
        constant PagedAttentionArgsV2 &, \
        device const float *, device const float *, device const DTYPE *, device const int32_t *, \
        device DTYPE *, threadgroup float *, uint3, uint3);

// V2 kernel instantiations (partition_size = 512)
INSTANTIATE_PAGED_ATTENTION_V2(float, float, 64, 16, 128, 512)
INSTANTIATE_PAGED_ATTENTION_V2(float, float, 128, 16, 128, 512)
INSTANTIATE_PAGED_ATTENTION_V2(float, float, 256, 16, 128, 512)
INSTANTIATE_PAGED_ATTENTION_V2(half, half, 64, 16, 128, 512)
INSTANTIATE_PAGED_ATTENTION_V2(half, half, 128, 16, 128, 512)
INSTANTIATE_PAGED_ATTENTION_V2(half, half, 256, 16, 128, 512)

// Reduce kernel instantiations
INSTANTIATE_PAGED_ATTENTION_V2_REDUCE(float, 64, 128)
INSTANTIATE_PAGED_ATTENTION_V2_REDUCE(float, 128, 128)
INSTANTIATE_PAGED_ATTENTION_V2_REDUCE(float, 256, 128)
INSTANTIATE_PAGED_ATTENTION_V2_REDUCE(half, 64, 128)
INSTANTIATE_PAGED_ATTENTION_V2_REDUCE(half, 128, 128)
INSTANTIATE_PAGED_ATTENTION_V2_REDUCE(half, 256, 128)
