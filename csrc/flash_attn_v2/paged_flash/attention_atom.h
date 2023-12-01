
#pragma once

#include <cstdint>
#include "cuda.h"
#include "cute/pointer.hpp"

struct __align__(32) AttentionAtom {
    int32_t* block_idx_list;

    int32_t q_start_idx;
    int32_t q_len;
    int32_t kv_blocks;
    int32_t total_extent;
    int32_t global_q_idx;
    int32_t unused;

    template <int threads>
    __device__ void load_kv_block_idxs(cute::smem_ptr<int32_t> block_idx_list_shr, int tidx) const
    {
        for (int i = tidx; i < kv_blocks; i += threads) { block_idx_list_shr[i] = block_idx_list[i]; }
        // Aggressive (but safe) sync
        __syncthreads();
    }
};
