#include <metal_stdlib>
using namespace metal;

// Kernel argument structures
struct ReshapeAndCacheArgs {
    int32_t num_tokens;
    int32_t num_heads;
    int32_t head_size;
    int32_t block_size;
    int32_t x;
    int64_t key_stride_token;
    int64_t key_stride_head;
    int64_t value_stride_token;
    int64_t value_stride_head;
    int64_t k_cache_stride_block;
    int64_t k_cache_stride_head;
    int64_t k_cache_stride_dim;
    int64_t k_cache_stride_token;
    int64_t v_cache_stride_block;
    int64_t v_cache_stride_head;
    int64_t v_cache_stride_dim;
};

struct BlockOperationArgs {
    int32_t num_pairs;
    int32_t block_size_in_bytes;
};

// Reshape and cache kernel
// Converts contiguous K/V tensors into paged cache format
// Grid: (num_tokens, 1, 1)
// Threadgroup: (NUM_THREADS, 1, 1)
template<typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
kernel void reshape_and_cache_kernel(
    constant ReshapeAndCacheArgs & args [[buffer(0)]],

    device const scalar_t * key [[buffer(1)]],         // [num_tokens, num_heads, head_size]
    device const scalar_t * value [[buffer(2)]],       // [num_tokens, num_heads, head_size]
    device cache_t * k_cache [[buffer(3)]],            // [num_blocks, num_heads, head_size/x, block_size, x]
    device cache_t * v_cache [[buffer(4)]],            // [num_blocks, num_heads, head_size, block_size]
    device const int32_t * slot_mapping [[buffer(5)]], // [num_tokens] - maps token to cache slot

    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const int token_idx = tgpig.x;
    const int thread_id = tpitg.x;

    if (token_idx >= args.num_tokens) return;

    // Get slot for this token
    const int slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) return;  // Invalid slot

    // Decompose slot into block and position within block
    const int block_idx = slot_idx / args.block_size;
    const int block_offset = slot_idx % args.block_size;

    const int num_heads = args.num_heads;
    const int head_size = args.head_size;
    const int x = args.x;

    // Each thread processes a subset of heads
    const int heads_per_thread = (num_heads + NUM_THREADS - 1) / NUM_THREADS;
    const int start_head = thread_id * heads_per_thread;
    const int end_head = min(start_head + heads_per_thread, num_heads);

    for (int head_idx = start_head; head_idx < end_head; head_idx++) {
        // Copy key
        const int key_offset = token_idx * args.key_stride_token + head_idx * args.key_stride_head;
        device const scalar_t * key_ptr = key + key_offset;

        for (int dim_idx = 0; dim_idx < head_size / x; dim_idx++) {
            for (int x_idx = 0; x_idx < x; x_idx++) {
                const int src_idx = dim_idx * x + x_idx;
                const int k_cache_idx = block_idx * args.k_cache_stride_block +
                                       head_idx * args.k_cache_stride_head +
                                       dim_idx * args.k_cache_stride_dim +
                                       block_offset * args.k_cache_stride_token +
                                       x_idx;
                k_cache[k_cache_idx] = cache_t(key_ptr[src_idx]);
            }
        }

        // Copy value
        const int value_offset = token_idx * args.value_stride_token + head_idx * args.value_stride_head;
        device const scalar_t * value_ptr = value + value_offset;

        for (int dim_idx = 0; dim_idx < head_size; dim_idx++) {
            const int v_cache_idx = block_idx * args.v_cache_stride_block +
                                   head_idx * args.v_cache_stride_head +
                                   dim_idx * args.v_cache_stride_dim +
                                   block_offset;
            v_cache[v_cache_idx] = cache_t(value_ptr[dim_idx]);
        }
    }
}

// Copy blocks kernel
// Copies cache blocks from source locations to destination locations
// Grid: (num_pairs, 1, 1)
// Threadgroup: (NUM_THREADS, 1, 1)
template<typename cache_t, int NUM_THREADS>
kernel void copy_blocks_kernel(
    constant BlockOperationArgs & args [[buffer(0)]],

    device const cache_t * src_cache [[buffer(1)]],      // Source cache
    device cache_t * dst_cache [[buffer(2)]],            // Destination cache
    device const int64_t * src_to_dst [[buffer(3)]],     // [num_pairs, 2] - mapping pairs

    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const int pair_idx = tgpig.x;
    const int thread_id = tpitg.x;

    if (pair_idx >= args.num_pairs) return;

    // Get source and destination block indices
    const int64_t src_block = src_to_dst[pair_idx * 2];
    const int64_t dst_block = src_to_dst[pair_idx * 2 + 1];

    // Calculate block size in elements
    const int block_size_elements = args.block_size_in_bytes / sizeof(cache_t);

    // Each thread copies a portion of the block
    const int elements_per_thread = (block_size_elements + NUM_THREADS - 1) / NUM_THREADS;
    const int start_elem = thread_id * elements_per_thread;
    const int end_elem = min(start_elem + elements_per_thread, block_size_elements);

    device const cache_t * src_ptr = src_cache + src_block * block_size_elements;
    device cache_t * dst_ptr = dst_cache + dst_block * block_size_elements;

    for (int i = start_elem; i < end_elem; i++) {
        dst_ptr[i] = src_ptr[i];
    }
}

// Swap blocks kernel
// Swaps cache blocks between two caches (e.g., GPU <-> GPU)
// Grid: (num_pairs, 1, 1)
// Threadgroup: (NUM_THREADS, 1, 1)
template<typename cache_t, int NUM_THREADS>
kernel void swap_blocks_kernel(
    constant BlockOperationArgs & args [[buffer(0)]],

    device cache_t * src_cache [[buffer(1)]],            // First cache
    device cache_t * dst_cache [[buffer(2)]],            // Second cache
    device const int64_t * src_to_dst [[buffer(3)]],     // [num_pairs, 2] - mapping pairs

    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const int pair_idx = tgpig.x;
    const int thread_id = tpitg.x;

    if (pair_idx >= args.num_pairs) return;

    // Get block indices
    const int64_t src_block = src_to_dst[pair_idx * 2];
    const int64_t dst_block = src_to_dst[pair_idx * 2 + 1];

    // Calculate block size in elements
    const int block_size_elements = args.block_size_in_bytes / sizeof(cache_t);

    // Each thread swaps a portion of the block
    const int elements_per_thread = (block_size_elements + NUM_THREADS - 1) / NUM_THREADS;
    const int start_elem = thread_id * elements_per_thread;
    const int end_elem = min(start_elem + elements_per_thread, block_size_elements);

    device cache_t * src_ptr = src_cache + src_block * block_size_elements;
    device cache_t * dst_ptr = dst_cache + dst_block * block_size_elements;

    for (int i = start_elem; i < end_elem; i++) {
        cache_t temp = src_ptr[i];
        src_ptr[i] = dst_ptr[i];
        dst_ptr[i] = temp;
    }
}

// Kernel instantiations for reshape_and_cache
#define INSTANTIATE_RESHAPE_AND_CACHE(DTYPE, CACHE_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS) \
    template [[host_name("reshape_and_cache_" #DTYPE "_" #CACHE_DTYPE "_h" #HEAD_SIZE "_b" #BLOCK_SIZE "_t" #NUM_THREADS)]] \
    kernel void reshape_and_cache_kernel<DTYPE, CACHE_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>( \
        constant ReshapeAndCacheArgs &, \
        device const DTYPE *, device const DTYPE *, device CACHE_DTYPE *, device CACHE_DTYPE *, \
        device const int32_t *, uint3, uint3);

// float32 variants
INSTANTIATE_RESHAPE_AND_CACHE(float, float, 64, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(float, float, 80, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(float, float, 96, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(float, float, 112, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(float, float, 128, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(float, float, 256, 16, 128)

// float16 variants
INSTANTIATE_RESHAPE_AND_CACHE(half, half, 64, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(half, half, 80, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(half, half, 96, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(half, half, 112, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(half, half, 128, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(half, half, 256, 16, 128)

// Mixed precision
INSTANTIATE_RESHAPE_AND_CACHE(float, half, 64, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(float, half, 80, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(float, half, 96, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(float, half, 112, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(float, half, 128, 16, 128)
INSTANTIATE_RESHAPE_AND_CACHE(float, half, 256, 16, 128)

// Kernel instantiations for block operations
#define INSTANTIATE_COPY_BLOCKS(DTYPE, NUM_THREADS) \
    template [[host_name("copy_blocks_" #DTYPE "_t" #NUM_THREADS)]] \
    kernel void copy_blocks_kernel<DTYPE, NUM_THREADS>( \
        constant BlockOperationArgs &, \
        device const DTYPE *, device DTYPE *, device const int64_t *, \
        uint3, uint3);

#define INSTANTIATE_SWAP_BLOCKS(DTYPE, NUM_THREADS) \
    template [[host_name("swap_blocks_" #DTYPE "_t" #NUM_THREADS)]] \
    kernel void swap_blocks_kernel<DTYPE, NUM_THREADS>( \
        constant BlockOperationArgs &, \
        device DTYPE *, device DTYPE *, device const int64_t *, \
        uint3, uint3);

INSTANTIATE_COPY_BLOCKS(float, 256)
INSTANTIATE_COPY_BLOCKS(half, 256)
INSTANTIATE_COPY_BLOCKS(char, 256)

INSTANTIATE_SWAP_BLOCKS(float, 256)
INSTANTIATE_SWAP_BLOCKS(half, 256)
INSTANTIATE_SWAP_BLOCKS(char, 256)
