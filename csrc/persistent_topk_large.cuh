#ifndef PERSISTENT_TOPK_LARGE_CUH_
#define PERSISTENT_TOPK_LARGE_CUH_

#include "persistent_topk_common.cuh"
#include "persistent_topk_decode.cuh"
#include "persistent_topk_medium.cuh"

namespace vllm {
namespace persistent {

// ============================================================================
// Inter-CTA sync primitives (from FlashInfer)
// ============================================================================

__device__ __forceinline__ int ld_acquire(int* ptr) {
  int state = 0;
#if (__CUDA_ARCH__ >= 700)
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(state)
               : "l"(ptr));
#else
  asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif
  return state;
}

__device__ __forceinline__ void red_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
               :
               : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicAdd(ptr, val);
#endif
}

__device__ __forceinline__ void st_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicExch(ptr, val);
#endif
}

__device__ __forceinline__ void wait_ge(int* ptr, int target_val,
                                        int thread_idx) {
  if (thread_idx == 0) {
#pragma unroll 1
    while (ld_acquire(ptr) < target_val) {
    }
  }
  __syncthreads();
}

// ============================================================================
// Large path: multi-CTA radix select for sequences > 64K
//
// Each row is processed by a group of CTAs. Each CTA loads its chunk into
// shared memory as ordered uint32, then participates in 4 rounds of
// coordinated radix select via global-memory histograms and barriers.
// ============================================================================

template <uint32_t VEC_SIZE>
__device__ void large_topk_cuda(PersistentTopKParams params) {
  const uint32_t ctas_per_group = params.ctas_per_group;
  const uint32_t group_id = blockIdx.x / ctas_per_group;
  const uint32_t cta_in_group = blockIdx.x % ctas_per_group;
  const uint32_t num_groups = gridDim.x / ctas_per_group;
  const uint32_t tx = threadIdx.x;
  const uint32_t chunk_size = params.chunk_size;

  if (blockIdx.x >= num_groups * ctas_per_group) return;

  // -- Dynamic shared memory layout --
  extern __shared__ uint8_t smem_raw[];
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  uint32_t* shared_ordered =
      reinterpret_cast<uint32_t*>(smem_raw + kFixedSmemLarge);

  RadixRowState* state = &params.row_states[group_id];

  // -- Initialize state (CTA 0 of each group) --
  if (cta_in_group == 0) {
    for (uint32_t buf = 0; buf < 3; buf++) {
      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        state->histogram[buf][i] = 0;
      }
    }
    if (tx == 0) {
      state->remaining_k = 0;
      state->prefix = 0;
      state->arrival_counter = 0;
      state->output_counter = 0;
    }
  }
  __syncthreads();

  int barrier_phase = 0;
  const uint32_t total_iters = (params.num_rows + num_groups - 1) / num_groups;

  for (uint32_t iter = 0; iter < total_iters; iter++) {
    const uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= params.num_rows) break;

    const uint32_t seq_len = params.lengths[row_idx];
    int32_t* row_output = params.output + row_idx * TopK;
    const float* row_input = params.input + row_idx * params.stride;

    // -- Non-large rows: only CTA 0 processes, no barriers needed --
    if (seq_len <= LARGE_THRESHOLD) {
      if (cta_in_group == 0) {
        if (seq_len <= static_cast<uint32_t>(TopK)) {
          for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
               i += kThreadsPerBlock) {
            row_output[i] = (i < seq_len) ? static_cast<int32_t>(i) : -1;
          }
        } else if (seq_len <= static_cast<uint32_t>(DECODE_THRESHOLD)) {
          decode_topk_cuda(row_input, row_output, seq_len);
        } else {
          fast_topk_cuda_tl(row_input, row_output, 0, seq_len);
        }
      }
      // Clean histogram for potential next large row
      if (cta_in_group == 0) {
        uint32_t next_hist_idx = ((iter + 1) * 4) % 3;
        for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
          state->histogram[next_hist_idx][i] = 0;
        }
      }
      continue;
    }

    // -- Compute this CTA's chunk bounds --
    const uint32_t my_chunk_start = cta_in_group * chunk_size;
    const uint32_t my_chunk_end = (my_chunk_start + chunk_size < seq_len)
                                      ? my_chunk_start + chunk_size
                                      : seq_len;
    const uint32_t actual_chunk_size =
        (my_chunk_start < seq_len) ? (my_chunk_end - my_chunk_start) : 0;

    // -- Stage 1: Load chunk to shared memory as ordered uint32 --
    {
      const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

      for (uint32_t i = tx * VEC_SIZE; i < aligned_size;
           i += kThreadsPerBlock * VEC_SIZE) {
        const float* src = row_input + my_chunk_start + i;
        if constexpr (VEC_SIZE == 4) {
          float4 v = *reinterpret_cast<const float4*>(src);
          shared_ordered[i] = convert_to_uint32_v2(v.x);
          shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
          shared_ordered[i + 2] = convert_to_uint32_v2(v.z);
          shared_ordered[i + 3] = convert_to_uint32_v2(v.w);
        } else if constexpr (VEC_SIZE == 2) {
          float2 v = *reinterpret_cast<const float2*>(src);
          shared_ordered[i] = convert_to_uint32_v2(v.x);
          shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
        } else {
          shared_ordered[i] = convert_to_uint32_v2(*src);
        }
      }
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size;
           i += kThreadsPerBlock) {
        shared_ordered[i] = convert_to_uint32_v2(row_input[my_chunk_start + i]);
      }
    }
    __syncthreads();

    // -- Init radix select state --
    if (tx == 0) {
      shared_scalars[0] = 0;     // prefix
      shared_scalars[1] = TopK;  // remaining_k
    }
    __syncthreads();

    // -- Initial barrier --
    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    wait_ge(&state->arrival_counter,
            (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
    barrier_phase++;
    __syncthreads();

    if (cta_in_group == 0 && tx == 0) {
      st_release(&state->output_counter, 0);
    }

    // -- Stage 2: 4 rounds of radix select --
    for (uint32_t round = 0; round < 4; round++) {
      const uint32_t global_round = iter * 4 + round;
      const uint32_t shift = 24 - round * 8;
      const uint32_t prefix = shared_scalars[0];
      const uint32_t remaining_k = shared_scalars[1];

      uint32_t* current_hist = state->histogram[global_round % 3];
      uint32_t* next_hist = state->histogram[(global_round + 1) % 3];

      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        local_histogram[i] = 0;
      }
      __syncthreads();

      for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
        uint32_t ordered = shared_ordered[i];
        uint32_t mask = (round == 0) ? 0u : (~0u << (32 - round * 8));
        if ((ordered & mask) == prefix) {
          uint32_t bucket = (ordered >> shift) & 0xFF;
          atomicAdd(&local_histogram[bucket], 1);
        }
      }
      __syncthreads();

      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        if (local_histogram[i] > 0) {
          atomicAdd(&current_hist[i], local_histogram[i]);
        }
      }

      if (cta_in_group == 0) {
        for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
          next_hist[i] = 0;
        }
      }

      if (tx == 0) {
        red_release(&state->arrival_counter, 1);
      }
      wait_ge(&state->arrival_counter,
              (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
      barrier_phase++;
      __syncthreads();

      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        suffix_sum[i] = current_hist[i];
      }
      __syncthreads();

      for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
        uint32_t val = 0;
        if (tx < RADIX) {
          val = suffix_sum[tx];
          if (tx + stride < RADIX) val += suffix_sum[tx + stride];
        }
        __syncthreads();
        if (tx < RADIX) suffix_sum[tx] = val;
        __syncthreads();
      }

      if (tx == 0) {
        shared_scalars[2] = 0;
        shared_scalars[3] = remaining_k;
      }
      __syncthreads();

      if (tx < RADIX) {
        uint32_t count_ge = suffix_sum[tx];
        uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
        if (count_ge >= remaining_k && count_gt < remaining_k) {
          shared_scalars[2] = tx;
          shared_scalars[3] = remaining_k - count_gt;
        }
      }
      __syncthreads();

      if (tx == 0) {
        shared_scalars[0] = prefix | (shared_scalars[2] << shift);
        shared_scalars[1] = shared_scalars[3];
      }
      __syncthreads();
    }  // end 4 radix rounds

    // -- Count local > pivot elements --
    const uint32_t ordered_pivot = shared_scalars[0];

    if (tx == 0) suffix_sum[0] = 0;
    __syncthreads();

    uint32_t my_gt_count = 0;
    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      if (shared_ordered[i] > ordered_pivot) my_gt_count++;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      my_gt_count += __shfl_down_sync(0xffffffff, my_gt_count, offset);
    }
    if (tx % 32 == 0 && my_gt_count > 0) {
      atomicAdd(&suffix_sum[0], my_gt_count);
    }
    __syncthreads();
    const uint32_t local_gt_count = suffix_sum[0];

    // -- Stage 3: Collect top-k indices --
    if (tx == 0) {
      local_histogram[0] = 0;
      if (local_gt_count > 0) {
        local_histogram[1] =
            atomicAdd(&state->output_counter, static_cast<int>(local_gt_count));
      }
    }
    __syncthreads();

    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      if (shared_ordered[i] > ordered_pivot) {
        uint32_t local_pos = atomicAdd(&local_histogram[0], 1);
        int pos = static_cast<int>(local_histogram[1]) + local_pos;
        row_output[pos] = static_cast<int32_t>(my_chunk_start + i);
      }
    }

    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    wait_ge(&state->arrival_counter,
            (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
    barrier_phase++;
    __syncthreads();

    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      if (shared_ordered[i] == ordered_pivot) {
        int pos = atomicAdd(&state->output_counter, 1);
        if (pos < TopK) {
          row_output[pos] = static_cast<int32_t>(my_chunk_start + i);
        }
      }
    }
  }  // end row loop

  // -- Cleanup: CTA 0 resets state --
  if (cta_in_group == 0) {
    for (uint32_t buf = 0; buf < 3; buf++) {
      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        state->histogram[buf][i] = 0;
      }
    }
    if (tx == 0) {
      st_release(&state->arrival_counter, 0);
    }
  }
}

}  // namespace persistent
}  // namespace vllm

#endif  // PERSISTENT_TOPK_LARGE_CUH_
