
#pragma once

#include <cstdint>
#include "cuda.h"
#include "cute/pointer.hpp"

struct __align__(32) AttentionAtom {
    using index_t = uint32_t;

    index_t* block_idx_list;

    index_t q_start_idx;
    index_t q_len;
    index_t kv_blocks;
    index_t total_extent;
    index_t global_q_idx;
    index_t unused;

    template <int threads>
    __device__ void load_kv_block_idxs(cute::smem_ptr<int32_t> block_idx_list_shr, int tidx) const
    {
        for (int i = tidx; i < kv_blocks; i += threads) { block_idx_list_shr[i] = block_idx_list[i]; }
        // Aggressive (but safe) sync
        __syncthreads();
    }
};
