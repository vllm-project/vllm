/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 #ifndef FLASHINFER_TOPK_CUH_
 #define FLASHINFER_TOPK_CUH_
 
 #include <cuda.h>
 #include <cub/block/block_reduce.cuh>
 
 #include <cstdlib>
 #include <cuda/std/limits>
 #include <numeric>
 
 #include "flashinfer_utils.cuh"
 #include "flashinfer_vec.cuh"
 
 namespace vllm {
 
 namespace sampling {
 
 // ============================================================================
 // RadixTopK Type Traits - supports float, half, and bfloat16
 // OrderedType: uint32_t for float, uint16_t for half/bf16
 // NUM_ROUNDS is computed as: sizeof(OrderedType) * 8 / RADIX_BITS
 // ============================================================================
 template <typename DType>
 struct RadixTopKTraits;
 
 // Specialization for float (32-bit)
 template <>
 struct RadixTopKTraits<float> {
   using OrderedType = uint32_t;
 
   // Compute number of rounds based on radix bits (not hardcoded)
   template <uint32_t RADIX_BITS>
   static __host__ __device__ constexpr uint32_t num_rounds() {
     return sizeof(OrderedType) * 8 / RADIX_BITS;
   }
 
   __device__ __forceinline__ static OrderedType ToOrdered(float val) {
     uint32_t bits = __float_as_uint(val);
     // For descending order: flip all bits if negative, else flip sign bit
     return (bits & 0x80000000) ? ~bits : (bits ^ 0x80000000);
   }
 
   __device__ __forceinline__ static float FromOrdered(OrderedType ordered) {
     uint32_t bits = (ordered & 0x80000000) ? (ordered ^ 0x80000000) : ~ordered;
     return __uint_as_float(bits);
   }
 
   __device__ __forceinline__ static float NegInf() {
     return -cuda::std::numeric_limits<float>::infinity();
   }
 };
 
 // Specialization for half (16-bit)
 template <>
 struct RadixTopKTraits<half> {
   using OrderedType = uint16_t;
 
   template <uint32_t RADIX_BITS>
   static __host__ __device__ constexpr uint32_t num_rounds() {
     return sizeof(OrderedType) * 8 / RADIX_BITS;
   }
 
   __device__ __forceinline__ static OrderedType ToOrdered(half val) {
     uint16_t bits = __half_as_ushort(val);
     return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits ^ 0x8000);
   }
 
   __device__ __forceinline__ static half FromOrdered(OrderedType ordered) {
     uint16_t bits = (ordered & 0x8000) ? static_cast<uint16_t>(ordered ^ 0x8000)
                                        : static_cast<uint16_t>(~ordered);
     return __ushort_as_half(bits);
   }
 
   __device__ __forceinline__ static half NegInf() {
     return __ushort_as_half(static_cast<uint16_t>(0xFC00));  // -inf in fp16
   }
 };
 
 // Specialization for nv_bfloat16 (16-bit)
 template <>
 struct RadixTopKTraits<nv_bfloat16> {
   using OrderedType = uint16_t;
 
   template <uint32_t RADIX_BITS>
   static __host__ __device__ constexpr uint32_t num_rounds() {
     return sizeof(OrderedType) * 8 / RADIX_BITS;
   }
 
   __device__ __forceinline__ static OrderedType ToOrdered(nv_bfloat16 val) {
     uint16_t bits = __bfloat16_as_ushort(val);
     return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits ^ 0x8000);
   }
 
   __device__ __forceinline__ static nv_bfloat16 FromOrdered(OrderedType ordered) {
     uint16_t bits = (ordered & 0x8000) ? static_cast<uint16_t>(ordered ^ 0x8000)
                                        : static_cast<uint16_t>(~ordered);
     return __ushort_as_bfloat16(bits);
   }
 
   __device__ __forceinline__ static nv_bfloat16 NegInf() {
     return __ushort_as_bfloat16(static_cast<uint16_t>(0xFF80));  // -inf in bf16
   }
 };
 // ==================== Multi-CTA Top-K Implementation ====================
 
 // Acquire/Release primitives for inter-CTA synchronization
 __device__ __forceinline__ int ld_acquire(int* ptr) {
   int state = 0;
 
 #if (__CUDA_ARCH__ >= 700)
   // SM70 and newer use memory consistency qualifiers
   // Acquire pattern using acquire modifier
   asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
 #else
   asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
 #endif
 
   return state;
 }
 
 __device__ __forceinline__ void red_release(int* ptr, int val) {
 #if (__CUDA_ARCH__ >= 700)
   // SM70 and newer use memory consistency qualifiers
   // Release pattern using acq_rel fence + relaxed modifier
   // (The fence also releases data that was weakly-written by other threads prior to the last
   // syncthreads)
   asm volatile("fence.acq_rel.gpu;\n");
   asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(ptr), "r"(val));
 #else
   __threadfence();
   atomicAdd(ptr, val);
 #endif
 }
 
 __device__ __forceinline__ void st_release(int* ptr, int val) {
 #if (__CUDA_ARCH__ >= 700)
   // SM70 and newer use memory consistency qualifiers
   // Release pattern: fence + release store
   asm volatile("fence.acq_rel.gpu;\n");
   asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
 #else
   __threadfence();
   atomicExch(ptr, val);
 #endif
 }
 
 // Wait until the value at ptr reaches target_val using acquire semantics
 // Only thread 0 spins, then all threads synchronize
 __device__ __forceinline__ void wait_ge(int* ptr, int target_val, int thread_idx) {
   if (thread_idx == 0) {
 #pragma unroll 1
     while (ld_acquire(ptr) < target_val) {
     }
   }
   __syncthreads();
 }
 
 // ==================== Multi-CTA Radix Top-K Mask Logits ====================
 
 // Global state for multi-CTA radix reduction (one per group)
 struct RadixRowState {
   uint32_t histogram[3][256];  // Triple-buffered histograms for 1-barrier-per-round
   uint32_t remaining_k;        // Remaining k after current round
   uint32_t prefix;             // Accumulated prefix (high bits of k-th element)
   int arrival_counter;         // For inter-CTA synchronization
   int output_counter;          // For collecting top-k indices (RadixTopK)
   float sum_topk;              // For RenormProb: sum of top-k elements
 };
 
 // ==================== Common Device Functions for Radix Top-K ====================
 
 /*!
  * \brief Compute suffix sum in shared memory using parallel reduction.
  *
  * After this function, suffix_sum[i] contains the count of elements >= bucket i.
  * This is computed by summing all histogram values from bucket i to 255.
  *
  * \param suffix_sum Shared memory array of size RADIX (256)
  * \param tx Thread index within the block
  */
 template <uint32_t BLOCK_THREADS>
 __device__ __forceinline__ void RadixSuffixSum(uint32_t* suffix_sum, uint32_t tx) {
   constexpr uint32_t RADIX = 256;
   // Parallel suffix sum: compute count of elements >= each bucket
   for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
     uint32_t val = 0;
     if (tx < RADIX) {
       val = suffix_sum[tx];
       if (tx + stride < RADIX) {
         val += suffix_sum[tx + stride];
       }
     }
     __syncthreads();
     if (tx < RADIX) {
       suffix_sum[tx] = val;
     }
     __syncthreads();
   }
 }
 
 /*!
  * \brief Find the threshold bucket that contains the k-th largest element.
  *
  * The threshold bucket satisfies: count_ge >= k && count_gt < k
  * where count_ge = suffix_sum[bucket] and count_gt = suffix_sum[bucket+1].
  *
  * \param suffix_sum Shared memory array containing suffix sums
  * \param remaining_k Number of top-k elements still to find
  * \param found_bucket Output: the found threshold bucket
  * \param found_remaining_k Output: remaining_k minus count of elements > threshold
  * \param tx Thread index within the block
  */
 __device__ __forceinline__ void RadixFindThresholdBucket(uint32_t* suffix_sum, uint32_t remaining_k,
                                                          uint32_t* found_bucket,
                                                          uint32_t* found_remaining_k, uint32_t tx) {
   constexpr uint32_t RADIX = 256;
   // Initialize (only thread 0)
   if (tx == 0) {
     *found_bucket = 0;
     *found_remaining_k = remaining_k;
   }
   __syncthreads();
 
   // All threads in RADIX range check their bucket
   if (tx < RADIX) {
     uint32_t count_ge = suffix_sum[tx];
     uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
     if (count_ge >= remaining_k && count_gt < remaining_k) {
       *found_bucket = tx;
       *found_remaining_k = remaining_k - count_gt;
     }
   }
   __syncthreads();
 }
 
 /*!
  * \brief Build local histogram for one round of radix select.
  *
  * Counts elements in shared_ordered that match the current prefix and bins them
  * by their byte at the current shift position.
  *
  * \tparam OrderedType The ordered integer type (uint16_t or uint32_t)
  * \param shared_ordered Shared memory containing ordered values
  * \param actual_chunk_size Number of elements in this CTA's chunk
  * \param local_histogram Output shared memory histogram
  * \param prefix Current prefix (high bits determined so far)
  * \param shift Bit shift for extracting current byte
  * \param round Current round (0 to NUM_ROUNDS-1)
  * \param tx Thread index
  */
 template <uint32_t BLOCK_THREADS, typename OrderedType>
 __device__ __forceinline__ void RadixBuildLocalHistogram(const OrderedType* shared_ordered,
                                                          uint32_t actual_chunk_size,
                                                          uint32_t* local_histogram, uint32_t prefix,
                                                          uint32_t shift, uint32_t round,
                                                          uint32_t tx) {
   constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;
   constexpr uint32_t RADIX_BITS = 8;
 
   for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
     OrderedType ordered = shared_ordered[i];
 
     // Check if this element matches the prefix (high bits determined so far)
     OrderedType mask =
         (round == 0)
             ? OrderedType(0)
             : static_cast<OrderedType>(~OrderedType(0) << (ORDERED_BITS - round * RADIX_BITS));
     if ((ordered & mask) == static_cast<OrderedType>(prefix)) {
       uint32_t bucket = (ordered >> shift) & 0xFF;
       atomicAdd(&local_histogram[bucket], 1);
     }
   }
 }
 
 /*!
  * \brief Perform one round of radix select with optional multi-CTA synchronization.
  *
  * This is the core radix select logic used by all TopK kernels.
  * It builds histogram, aggregates across CTAs (if multi-CTA), computes suffix sum,
  * and finds the threshold bucket.
  *
  * \tparam BLOCK_THREADS Number of threads per block
  * \tparam SINGLE_CTA True if single-CTA mode (no inter-CTA sync needed)
  * \tparam OrderedType The ordered integer type
  *
  * \param shared_ordered Shared memory containing ordered values
  * \param actual_chunk_size Number of elements in this CTA's chunk
  * \param local_histogram Shared memory for local histogram (size RADIX)
  * \param suffix_sum Shared memory for suffix sum computation (size RADIX)
  * \param state Pointer to RadixRowState for multi-CTA sync (nullptr if SINGLE_CTA)
  * \param prefix Current prefix value
  * \param remaining_k Current remaining k value
  * \param round Current round (0 to NUM_ROUNDS-1)
  * \param barrier_phase Reference to barrier phase counter
  * \param ctas_per_group Number of CTAs per group
  * \param tx Thread index
  * \param out_new_prefix Output: updated prefix after this round
  * \param out_new_remaining_k Output: updated remaining_k after this round
  */
 template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType>
 __device__ __forceinline__ void RadixSelectOneRound(
     const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t* local_histogram,
     uint32_t* suffix_sum, uint32_t* shared_scalars, RadixRowState* state, uint32_t prefix,
     uint32_t remaining_k, uint32_t round, uint32_t iter, int& barrier_phase,
     uint32_t ctas_per_group, uint32_t cta_in_group, uint32_t tx, uint32_t* out_new_prefix,
     uint32_t* out_new_remaining_k) {
   constexpr uint32_t RADIX = 256;
   constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;
   constexpr uint32_t RADIX_BITS = 8;
   constexpr uint32_t NUM_ROUNDS = ORDERED_BITS / RADIX_BITS;
   uint32_t shift = ORDERED_BITS - (round + 1) * RADIX_BITS;
   uint32_t global_round = iter * NUM_ROUNDS + round;
 
   // For multi-CTA: pointers to global histograms (triple buffer)
   uint32_t* current_hist = nullptr;
   uint32_t* next_hist = nullptr;
   if constexpr (!SINGLE_CTA) {
     current_hist = state->histogram[global_round % 3];
     next_hist = state->histogram[(global_round + 1) % 3];
   }
 
   // Clear local histogram only
   for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
     local_histogram[i] = 0;
   }
   __syncthreads();
 
   // Build local histogram from shared memory
   RadixBuildLocalHistogram<BLOCK_THREADS, OrderedType>(shared_ordered, actual_chunk_size,
                                                        local_histogram, prefix, shift, round, tx);
   __syncthreads();
 
   // For multi-CTA: write -> (leading CTA clears next) -> barrier -> read
   // For single-CTA: local_histogram is already the complete histogram
   if constexpr (!SINGLE_CTA) {
     // Accumulate local histogram to global
     for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
       if (local_histogram[i] > 0) {
         atomicAdd(&current_hist[i], local_histogram[i]);
       }
     }
 
     // Only leading CTA clears next round's histogram BEFORE barrier
     if (cta_in_group == 0) {
       for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
         next_hist[i] = 0;
       }
     }
 
     // Barrier: wait for all CTAs to finish atomicAdd and clearing
     if (tx == 0) {
       red_release(&state->arrival_counter, 1);
     }
     int target = (barrier_phase + 1) * ctas_per_group;
     wait_ge(&state->arrival_counter, target, tx);
     barrier_phase++;
     __syncthreads();
 
     // Read current histogram (after barrier, all atomicAdds are complete)
     for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
       suffix_sum[i] = current_hist[i];
     }
   } else {
     // Single-CTA: copy local histogram directly to suffix_sum
     for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
       suffix_sum[i] = local_histogram[i];
     }
   }
   __syncthreads();
 
   // Compute suffix sum
   RadixSuffixSum<BLOCK_THREADS>(suffix_sum, tx);
 
   // Find threshold bucket using shared_scalars for found_bucket and found_remaining_k
   // shared_scalars[0] = found_bucket, shared_scalars[1] = found_remaining_k
   RadixFindThresholdBucket(suffix_sum, remaining_k, &shared_scalars[0], &shared_scalars[1], tx);
 
   // Output new prefix and remaining_k
   *out_new_prefix = prefix | (shared_scalars[0] << shift);
   *out_new_remaining_k = shared_scalars[1];
 }
 
 /*!
  * \brief Load data from global memory to shared memory and convert to ordered representation.
  *
  * This is the common Stage 1 for all TopK kernels. It loads data using vectorized
  * memory access and converts to ordered representation for radix select.
  *
  * \tparam BLOCK_THREADS Number of threads per block
  * \tparam VEC_SIZE Vector size for memory access
  * \tparam DType Data type (float, half, nv_bfloat16)
  * \tparam Traits Type traits for DType
  *
  * \param input Pointer to input data row start (already offset by row)
  * \param shared_ordered Shared memory for ordered values
  * \param chunk_start Start index within the row for this CTA's chunk
  * \param actual_chunk_size Number of elements in this CTA's chunk
  * \param tx Thread index
  */
 template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType, typename Traits>
 __device__ __forceinline__ void LoadToSharedOrdered(const DType* input,
                                                     typename Traits::OrderedType* shared_ordered,
                                                     uint32_t chunk_start,
                                                     uint32_t actual_chunk_size, uint32_t tx) {
   using OrderedType = typename Traits::OrderedType;
   vec_t<DType, VEC_SIZE> input_vec;
   const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
 
 #pragma unroll 2
   for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
     input_vec.cast_load(input + chunk_start + i);
 #pragma unroll
     for (uint32_t j = 0; j < VEC_SIZE; ++j) {
       shared_ordered[i + j] = Traits::ToOrdered(input_vec[j]);
     }
   }
   // Handle tail
   for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
     shared_ordered[i] = Traits::ToOrdered(input[chunk_start + i]);
   }
   __syncthreads();
 }
 
 /*!
  * \brief Find the k-th largest element using radix select from pre-loaded shared memory.
  *
  * This function assumes data has already been loaded into shared_ordered.
  * It performs the complete radix select algorithm (initial barrier + NUM_ROUNDS)
  * and returns the ordered pivot value.
  *
  * \tparam BLOCK_THREADS Number of threads per block
  * \tparam SINGLE_CTA True if single-CTA mode
  * \tparam OrderedType The ordered integer type
  *
  * \param shared_ordered Shared memory containing ordered values (pre-loaded)
  * \param actual_chunk_size Number of elements in this CTA's chunk
  * \param k Number of top elements to select
  * \param local_histogram Shared memory for local histogram (size RADIX)
  * \param suffix_sum Shared memory for suffix sum (size RADIX)
  * \param shared_scalars Shared memory for scalars [prefix_cache, remaining_k_cache, found_bucket,
  * found_remaining_k, output_counter]
  * \param state RadixRowState pointer for multi-CTA sync (nullptr if SINGLE_CTA)
  * \param barrier_phase Reference to barrier phase counter
  * \param ctas_per_group Number of CTAs per group
  * \param cta_in_group CTA index within group
  * \param tx Thread index
  * \param iter Current iteration (for triple-buffer indexing)
  * \return The pivot value in ordered representation
  */
 template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType>
 __device__ __forceinline__ OrderedType RadixSelectFromSharedMemory(
     const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t k,
     uint32_t* local_histogram, uint32_t* suffix_sum, uint32_t* shared_scalars, RadixRowState* state,
     int& barrier_phase, uint32_t ctas_per_group, uint32_t cta_in_group, uint32_t tx, uint32_t iter,
     uint32_t& out_local_gt_count) {
   constexpr uint32_t RADIX = 256;
   constexpr uint32_t RADIX_BITS = 8;
   constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;
   constexpr uint32_t NUM_ROUNDS = ORDERED_BITS / RADIX_BITS;
 
 // Aliases for scalar shared variables
 #define prefix_cache shared_scalars[0]
 #define remaining_k_cache shared_scalars[1]
 #define found_bucket shared_scalars[2]
 #define found_remaining_k shared_scalars[3]
 #define shared_output_counter shared_scalars[4]
 
   // Initialize local caches
   if (tx == 0) {
     prefix_cache = 0;
     remaining_k_cache = k;
     if constexpr (SINGLE_CTA) {
       shared_output_counter = 0;
     }
   }
   __syncthreads();
 
   // Initial barrier (skip for single CTA)
   if constexpr (!SINGLE_CTA) {
     if (tx == 0) {
       red_release(&state->arrival_counter, 1);
     }
     int target = (barrier_phase + 1) * ctas_per_group;
     wait_ge(&state->arrival_counter, target, tx);
     barrier_phase++;
     __syncthreads();
 
     // CTA 0 clears output counter AFTER barrier
     if (cta_in_group == 0 && tx == 0) {
       st_release(&state->output_counter, 0);
     }
   }
 
   // NUM_ROUNDS of radix select
   for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
     uint32_t global_round = iter * NUM_ROUNDS + round;
     uint32_t shift = ORDERED_BITS - (round + 1) * RADIX_BITS;
     uint32_t prefix = prefix_cache;
     uint32_t remaining_k = remaining_k_cache;
 
     // For multi-CTA: pointers to global histograms (triple buffer)
     uint32_t* current_hist = nullptr;
     uint32_t* next_hist = nullptr;
     if constexpr (!SINGLE_CTA) {
       current_hist = state->histogram[global_round % 3];
       next_hist = state->histogram[(global_round + 1) % 3];
     }
 
     // Clear local histogram
     for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
       local_histogram[i] = 0;
     }
     __syncthreads();
 
     // Build local histogram
 #pragma unroll 2
     for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
       OrderedType ordered = shared_ordered[i];
       OrderedType mask =
           (round == 0)
               ? OrderedType(0)
               : static_cast<OrderedType>(~OrderedType(0) << (ORDERED_BITS - round * RADIX_BITS));
       if ((ordered & mask) == static_cast<OrderedType>(prefix)) {
         uint32_t bucket = (ordered >> shift) & 0xFF;
         atomicAdd(&local_histogram[bucket], 1);
       }
     }
     __syncthreads();
 
     // Multi-CTA: accumulate to global, barrier, read back
     if constexpr (!SINGLE_CTA) {
       for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
         if (local_histogram[i] > 0) {
           atomicAdd(&current_hist[i], local_histogram[i]);
         }
       }
       if (cta_in_group == 0) {
         for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
           next_hist[i] = 0;
         }
       }
       if (tx == 0) {
         red_release(&state->arrival_counter, 1);
       }
       int target = (barrier_phase + 1) * ctas_per_group;
       wait_ge(&state->arrival_counter, target, tx);
       barrier_phase++;
       __syncthreads();
 
       for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
         suffix_sum[i] = current_hist[i];
       }
     } else {
       for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
         suffix_sum[i] = local_histogram[i];
       }
     }
     __syncthreads();
 
     // Compute suffix sum
     RadixSuffixSum<BLOCK_THREADS>(suffix_sum, tx);
 
     // Find threshold bucket
     if (tx == 0) {
       found_bucket = 0;
       found_remaining_k = remaining_k;
     }
     __syncthreads();
 
     if (tx < RADIX) {
       uint32_t count_ge = suffix_sum[tx];
       uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
       if (count_ge >= remaining_k && count_gt < remaining_k) {
         found_bucket = tx;
         found_remaining_k = remaining_k - count_gt;
       }
     }
     __syncthreads();
 
     // Update caches
     if (tx == 0) {
       prefix_cache = prefix | (found_bucket << shift);
       remaining_k_cache = found_remaining_k;
     }
     __syncthreads();
   }
 
   OrderedType ordered_pivot = static_cast<OrderedType>(prefix_cache);
 
   // Count > pivot elements by scanning shared_ordered
   // This is needed because suffix_sum only tracks elements matching the current prefix,
   // not all elements > pivot (which includes elements with higher-order bits > pivot)
   if (tx == 0) {
     suffix_sum[0] = 0;
   }
   __syncthreads();
 
   uint32_t my_gt_count = 0;
 #pragma unroll 2
   for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
     if (shared_ordered[i] > ordered_pivot) {
       my_gt_count++;
     }
   }
 
   // Warp-level reduction
   for (int offset = 16; offset > 0; offset /= 2) {
     my_gt_count += __shfl_down_sync(0xffffffff, my_gt_count, offset);
   }
 
   // First thread of each warp atomics to shared
   int lane = tx % 32;
   if (lane == 0 && my_gt_count > 0) {
     atomicAdd(&suffix_sum[0], my_gt_count);
   }
   __syncthreads();
 
   out_local_gt_count = suffix_sum[0];
 
 #undef prefix_cache
 #undef remaining_k_cache
 #undef found_bucket
 #undef found_remaining_k
 #undef shared_output_counter
 
   return ordered_pivot;
 }
 
 /*!
  * \brief Find the k-th largest element pivot using radix select.
  *
  * This is the main entry point for the radix select algorithm.
  * It performs NUM_ROUNDS of radix select to find the exact pivot value.
  *
  * \tparam BLOCK_THREADS Number of threads per block
  * \tparam VEC_SIZE Vector size for memory access
  * \tparam SINGLE_CTA True if single-CTA mode
  * \tparam DType Data type (float, half, nv_bfloat16)
  *
  * \param input Input data pointer (for this row)
  * \param shared_ordered Shared memory for ordered values
  * \param local_histogram Shared memory for local histogram
  * \param suffix_sum Shared memory for suffix sum
  * \param shared_scalars Shared memory for temporary scalar values (size >= 5)
  * \param state RadixRowState pointer (nullptr if SINGLE_CTA)
  * \param chunk_start Start index in vocab for this CTA
  * \param actual_chunk_size Number of elements in this chunk
  * \param k Number of top elements to select
  * \param barrier_phase Reference to barrier phase counter
  * \param ctas_per_group Number of CTAs per group
  * \param cta_in_group CTA index within group
  * \param tx Thread index
  * \param iter Current iteration (for triple-buffer indexing)
  * \return The pivot value (k-th largest element)
  */
 template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType>
 __device__ __forceinline__ DType RadixSelectFindPivot(
     const DType* input, typename RadixTopKTraits<DType>::OrderedType* shared_ordered,
     uint32_t* local_histogram, uint32_t* suffix_sum, uint32_t* shared_scalars, RadixRowState* state,
     uint32_t chunk_start, uint32_t actual_chunk_size, uint32_t k, int& barrier_phase,
     uint32_t ctas_per_group, uint32_t cta_in_group, uint32_t tx, uint32_t iter = 0) {
   using Traits = RadixTopKTraits<DType>;
   using OrderedType = typename Traits::OrderedType;
 
   // Stage 1: Load and convert to ordered representation
   LoadToSharedOrdered<BLOCK_THREADS, VEC_SIZE, DType, Traits>(input, shared_ordered, chunk_start,
                                                               actual_chunk_size, tx);
 
   // Stage 2: Radix select to find pivot
   uint32_t local_gt_count = 0;  // Not used in this function
   OrderedType ordered_pivot = RadixSelectFromSharedMemory<BLOCK_THREADS, SINGLE_CTA, OrderedType>(
       shared_ordered, actual_chunk_size, k, local_histogram, suffix_sum, shared_scalars, state,
       barrier_phase, ctas_per_group, cta_in_group, tx, iter, local_gt_count);
 
   // Convert ordered representation back to DType pivot
   return Traits::FromOrdered(ordered_pivot);
 }
 
 /*!
  * \brief Collect top-k indices based on pivot value with custom output transform (Single Pass).
  *
  * This optimized version uses a single pass to write all elements:
  * - > pivot: use shared memory atomic for local offset within CTA's allocation
  * - == pivot: use global memory atomic, check if pos < k before writing
  *
  * The local_gt_count is computed during the last round of radix select, so we know
  * exactly how many > pivot elements each CTA has. This allows batched global atomic
  * (one per CTA) for > pivot elements.
  *
  * \tparam BLOCK_THREADS Number of threads per block
  * \tparam SINGLE_CTA True if single-CTA mode
  * \tparam OrderedType The ordered integer type
  * \tparam OutputFunc Functor type: void(uint32_t original_idx, OrderedType ordered_val, int
  * output_pos)
  *
  * \param shared_ordered Shared memory containing ordered values
  * \param actual_chunk_size Number of elements in this CTA's chunk
  * \param chunk_start Start index in input for this chunk
  * \param k Number of top elements to select
  * \param ordered_pivot The pivot value in ordered representation
  * \param local_gt_count Number of > pivot elements in this CTA (from radix select)
  * \param local_histogram Shared memory for counters
  * \param shared_output_counter Pointer to shared output counter (SINGLE_CTA mode)
  * \param state RadixRowState pointer for multi-CTA sync (nullptr if SINGLE_CTA)
  * \param barrier_phase Reference to barrier phase counter (unused in new implementation)
  * \param ctas_per_group Number of CTAs per group
  * \param tx Thread index
  * \param output_func Functor called as output_func(original_idx, ordered_val, output_pos) for each
  * element
  */
 template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType, typename OutputFunc>
 __device__ __forceinline__ void RadixCollectIndices(
     const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t chunk_start, uint32_t k,
     OrderedType ordered_pivot, uint32_t local_gt_count, uint32_t* local_histogram,
     uint32_t* shared_output_counter, RadixRowState* state, int& barrier_phase,
     uint32_t ctas_per_group, uint32_t tx, OutputFunc output_func) {
 // Use local_histogram for counters:
 // [0]: local_offset_gt (local offset for > pivot elements within CTA's allocation)
 // [1]: global_base_gt (global base position for > pivot)
 #define local_offset_gt local_histogram[0]
 #define global_base_gt local_histogram[1]
 
   // Get global base position for this CTA's > pivot elements (one atomic per CTA)
   if (tx == 0) {
     local_offset_gt = 0;
     if (local_gt_count > 0) {
       if constexpr (SINGLE_CTA) {
         global_base_gt = atomicAdd(shared_output_counter, local_gt_count);
       } else {
         global_base_gt = atomicAdd(&state->output_counter, local_gt_count);
       }
     }
   }
   __syncthreads();
 
   // Pass 1: Write elements > pivot
   // These are guaranteed to be in top-k, use local offset within CTA's allocation
 #pragma unroll 2
   for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
     OrderedType ordered_val = shared_ordered[i];
     if (ordered_val > ordered_pivot) {
       uint32_t local_pos = atomicAdd(&local_offset_gt, 1);
       int pos = global_base_gt + local_pos;
       output_func(chunk_start + i, ordered_val, pos);
     }
   }
 
   // Barrier to ensure all > pivot elements are collected first (only for multi-CTA)
   // This is critical: without this barrier, CTAs may write == pivot elements while
   // other CTAs are still writing > pivot elements, causing incorrect positions.
   if constexpr (!SINGLE_CTA) {
     if (tx == 0) {
       red_release(&state->arrival_counter, 1);
     }
     int target = (barrier_phase + 1) * ctas_per_group;
     wait_ge(&state->arrival_counter, target, tx);
     barrier_phase++;
   }
   __syncthreads();
 
   // Pass 2: Write elements == pivot
   // Use global atomic directly since we need cross-CTA coordination to respect
   // the k limit (some == pivot elements may be truncated).
 #pragma unroll 2
   for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
     OrderedType ordered_val = shared_ordered[i];
     if (ordered_val == ordered_pivot) {
       int pos;
       if constexpr (SINGLE_CTA) {
         pos = atomicAdd(shared_output_counter, 1);
       } else {
         pos = atomicAdd(&state->output_counter, 1);
       }
       if (pos < static_cast<int>(k)) {
         output_func(chunk_start + i, ordered_pivot, pos);
       }
     }
   }
 
 #undef local_offset_gt
 #undef global_base_gt
 }
 
 // ==================== Unified Radix Top-K Kernel with Epilogue Modes ====================
 
 /*!
  * \brief Epilogue mode for unified RadixTopK kernel.
  */
 enum class RadixTopKMode {
   Basic,               ///< Returns (indices, values) pairs
   PageTableTransform,  ///< Gathers indices through page table
   RaggedTransform,     ///< Adds offset to indices
 };
 
 /*!
  * \brief Unified Multi-CTA Radix Top-K kernel with mode-specific epilogues.
  *
  * This kernel unifies three top-k variants:
  * - Basic: Returns top-k indices and values
  * - PageTableTransform: Gathers top-k indices through a page table
  * - RaggedTransform: Adds per-row offset to top-k indices
  *
  * \tparam BLOCK_THREADS Number of threads per block
  * \tparam VEC_SIZE Vector size for memory access
  * \tparam SINGLE_CTA True if single-CTA mode
  * \tparam MODE Epilogue mode (Basic, PageTableTransform, or RaggedTransform)
  * \tparam DType Data type (float, half, nv_bfloat16)
  * \tparam IdType Index type
  */
 template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, RadixTopKMode MODE,
           typename DType, typename IdType>
 __global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKKernel_Unified(
     DType* input,            // [num_rows, stride]
     IdType* output_indices,  // [num_rows, top_k] - indices or page table entries
     DType* output_values,    // [num_rows, top_k] - only used in Basic mode, nullptr otherwise
     const IdType*
         aux_data,  // Mode-specific: top_k_arr (Basic), src_page_table (PageTable), offsets (Ragged)
     IdType* lengths,             // [num_rows] per-row lengths, nullptr for Basic (uses stride)
     const IdType* row_to_batch,  // [num_rows] batch mapping for PageTable, nullptr otherwise
     int64_t aux_stride,          // src_page_table stride for PageTable mode, 0 otherwise
     uint32_t top_k_val, uint32_t stride, uint32_t num_rows, RadixRowState* row_states,
     uint32_t chunk_size, uint32_t ctas_per_group) {
   using Traits = RadixTopKTraits<DType>;
   using OrderedType = typename Traits::OrderedType;
 
   constexpr uint32_t RADIX = 256;
 
   const uint32_t global_cta_id = blockIdx.x;
   const uint32_t group_id = global_cta_id / ctas_per_group;
   const uint32_t cta_in_group = global_cta_id % ctas_per_group;
   const uint32_t tx = threadIdx.x;
 
   extern __shared__ uint8_t smem[];
 
   constexpr size_t num_scalars = SINGLE_CTA ? 5 : 4;
   constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + num_scalars);
   uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
   uint32_t* suffix_sum = local_histogram + RADIX;
   uint32_t* shared_scalars = suffix_sum + RADIX;
 
   size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
   OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);
 
 #define shared_output_counter shared_scalars[4]
 
   RadixRowState* state = nullptr;
   if constexpr (!SINGLE_CTA) {
     state = &row_states[group_id];
   }
 
   uint32_t num_groups = gridDim.x / ctas_per_group;
   uint32_t total_iterations = (num_rows + num_groups - 1) / num_groups;
 
   int barrier_phase = 0;
 
   for (uint32_t iter = 0; iter < total_iterations; iter++) {
     uint32_t row_idx = group_id + iter * num_groups;
     if (row_idx >= num_rows) break;
 
     // Mode-specific: get row length and k value
     uint32_t length, k;
     if constexpr (MODE == RadixTopKMode::Basic) {
       length = stride;                                            // Fixed length for all rows
       k = (aux_data != nullptr) ? aux_data[row_idx] : top_k_val;  // aux_data = top_k_arr
     } else {
       length = lengths[row_idx];  // Per-row length
       k = top_k_val;              // Fixed k
     }
 
     // Mode-specific: output pointers and auxiliary data
     IdType* row_output = output_indices + row_idx * top_k_val;
 
     // Handle trivial cases
     if constexpr (MODE == RadixTopKMode::Basic) {
       if (k >= length) {
         // k >= vocab_size: return all indices
         const uint32_t chunk_start = cta_in_group * chunk_size;
         const uint32_t chunk_end = min(chunk_start + chunk_size, length);
         for (uint32_t i = tx; i < chunk_end - chunk_start; i += BLOCK_THREADS) {
           if (chunk_start + i < k) {
             row_output[chunk_start + i] = static_cast<IdType>(chunk_start + i);
             output_values[row_idx * top_k_val + chunk_start + i] =
                 input[row_idx * stride + chunk_start + i];
           }
         }
         // Clear histogram for next iteration (in case it's k < length)
         if constexpr (!SINGLE_CTA) {
           constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;
           uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
           if (cta_in_group == 0) {
             for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
               state->histogram[next_first_hist_idx][i] = 0;
             }
           }
         }
         continue;
       }
     } else if constexpr (MODE == RadixTopKMode::PageTableTransform) {
       uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[row_idx] : row_idx;
       const IdType* src_page_entry = aux_data + batch_idx * aux_stride;
       if (length <= top_k_val) {
         for (uint32_t i = tx; i < top_k_val; i += BLOCK_THREADS) {
           row_output[i] = (i < length) ? src_page_entry[i] : static_cast<IdType>(-1);
         }
         // Clear histogram for next iteration
         if constexpr (!SINGLE_CTA) {
           constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;
           uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
           if (cta_in_group == 0) {
             for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
               state->histogram[next_first_hist_idx][i] = 0;
             }
           }
         }
         continue;
       }
     } else {  // RaggedTransform
       if (length <= top_k_val) {
         for (uint32_t i = tx; i < top_k_val; i += BLOCK_THREADS) {
           row_output[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
         }
         // Clear histogram for next iteration
         if constexpr (!SINGLE_CTA) {
           constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;
           uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
           if (cta_in_group == 0) {
             for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
               state->histogram[next_first_hist_idx][i] = 0;
             }
           }
         }
         continue;
       }
     }
 
     const uint32_t chunk_start = cta_in_group * chunk_size;
     const uint32_t chunk_end = min(chunk_start + chunk_size, length);
     const uint32_t actual_chunk_size = (chunk_start < length) ? (chunk_end - chunk_start) : 0;

     // Stage 1: Load and convert to ordered representation
     LoadToSharedOrdered<BLOCK_THREADS, VEC_SIZE, DType, Traits>(
         input + row_idx * stride, shared_ordered, chunk_start, actual_chunk_size, tx);
 
     // Stage 2: Radix select to find k-th largest element (also computes local_gt_count)
     uint32_t local_gt_count = 0;
     OrderedType ordered_pivot = RadixSelectFromSharedMemory<BLOCK_THREADS, SINGLE_CTA, OrderedType>(
         shared_ordered, actual_chunk_size, k, local_histogram, suffix_sum, shared_scalars, state,
         barrier_phase, ctas_per_group, cta_in_group, tx, iter, local_gt_count);
 
     // Stage 3: Collect indices with mode-specific epilogue (single pass)
     if constexpr (MODE == RadixTopKMode::Basic) {
       DType* row_output_values = output_values + row_idx * top_k_val;
       RadixCollectIndices<BLOCK_THREADS, SINGLE_CTA, OrderedType>(
           shared_ordered, actual_chunk_size, chunk_start, k, ordered_pivot, local_gt_count,
           local_histogram, &shared_output_counter, state, barrier_phase, ctas_per_group, tx,
           [&](uint32_t original_idx, OrderedType ordered_val, int pos) {
             row_output[pos] = static_cast<IdType>(original_idx);
             row_output_values[pos] = Traits::FromOrdered(ordered_val);
           });
     } else if constexpr (MODE == RadixTopKMode::PageTableTransform) {
       uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[row_idx] : row_idx;
       const IdType* src_page_entry = aux_data + batch_idx * aux_stride;
 
       // Collect raw indices first
       RadixCollectIndices<BLOCK_THREADS, SINGLE_CTA, OrderedType>(
           shared_ordered, actual_chunk_size, chunk_start, k, ordered_pivot, local_gt_count,
           local_histogram, &shared_output_counter, state, barrier_phase, ctas_per_group, tx,
           [&](uint32_t original_idx, OrderedType /*ordered_val*/, int pos) {
             row_output[pos] = static_cast<IdType>(original_idx);
           });
 
       if constexpr (SINGLE_CTA) {
         __syncthreads();
         // Transform through page table with coalesced access
         for (uint32_t i = tx; i < k; i += BLOCK_THREADS) {
           IdType idx = row_output[i];
           row_output[i] = src_page_entry[idx];
         }
       } else {
         // Barrier to ensure all CTAs finished writing indices
         if (tx == 0) {
           red_release(&state->arrival_counter, 1);
         }
         int target = (barrier_phase + 1) * ctas_per_group;
         wait_ge(&state->arrival_counter, target, tx);
         barrier_phase++;
         __syncthreads();
 
         // All CTAs participate in page table transform (coalesced access)
         uint32_t elems_per_cta = (k + ctas_per_group - 1) / ctas_per_group;
         uint32_t my_start = cta_in_group * elems_per_cta;
         uint32_t my_end = min(my_start + elems_per_cta, k);
         for (uint32_t i = my_start + tx; i < my_end; i += BLOCK_THREADS) {
           IdType idx = row_output[i];
           row_output[i] = src_page_entry[idx];
         }
       }
     } else {  // RaggedTransform
       RadixCollectIndices<BLOCK_THREADS, SINGLE_CTA, OrderedType>(
           shared_ordered, actual_chunk_size, chunk_start, k, ordered_pivot, local_gt_count,
           local_histogram, &shared_output_counter, state, barrier_phase, ctas_per_group, tx,
           [&](uint32_t original_idx, OrderedType /*ordered_val*/, int pos) {
             row_output[pos] = static_cast<IdType>(original_idx);
           });
     }
   }
 
   // Clear histogram buffers and reset arrival counter
   if constexpr (!SINGLE_CTA) {
     if (cta_in_group == 0) {
       for (uint32_t buf = 0; buf < 3; ++buf) {
         for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
           state->histogram[buf][i] = 0;
         }
       }
       if (tx == 0) {
         st_release(&state->arrival_counter, 0);
       }
     }
   }
 
 #undef shared_output_counter
 }
 
 template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType,
           typename IdType>
 __global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKMaskLogitsKernel_MultiCTA(
     DType* logits,         // [batch, vocab_size]
     DType* masked_logits,  // [batch, vocab_size]
     IdType* top_k_arr,     // [batch] or nullptr
     uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
     RadixRowState* row_states,  // [num_groups] (nullptr if SINGLE_CTA)
     uint32_t chunk_size,        // elements per CTA
     uint32_t ctas_per_group)    // CTAs per row (1 if SINGLE_CTA)
 {
   // Type traits for FP16/BF16/FP32 support
   using Traits = RadixTopKTraits<DType>;
   using OrderedType = typename Traits::OrderedType;
 
   constexpr uint32_t RADIX = 256;  // 8-bit radix
 
   const uint32_t global_cta_id = blockIdx.x;
   const uint32_t group_id = global_cta_id / ctas_per_group;
   const uint32_t cta_in_group = global_cta_id % ctas_per_group;
   const uint32_t tx = threadIdx.x;
 
   // Shared memory layout: [fixed storage] [ordered values cache]
   extern __shared__ uint8_t smem[];
 
   // Fixed shared memory (at the beginning)
   // histogram[256] + suffix[256] + 5 scalars (for RadixSelectFromSharedMemory)
   constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 5);
   uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
   uint32_t* suffix_sum = local_histogram + RADIX;
   uint32_t* shared_scalars = suffix_sum + RADIX;
 
   // Align ordered values cache to 16 bytes
   size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
   OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);
 
   // State pointer only used when not SINGLE_CTA
   RadixRowState* state = nullptr;
   if constexpr (!SINGLE_CTA) {
     state = &row_states[group_id];
   }
 
   // Calculate total number of iterations for persistent loop
   uint32_t num_groups = gridDim.x / ctas_per_group;
   uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;
 
   int barrier_phase = 0;
 
   // Persistent loop over rows
   for (uint32_t iter = 0; iter < total_iterations; iter++) {
     uint32_t row_idx = group_id + iter * num_groups;
 
     if (row_idx >= batch_size) break;
 
     const uint32_t chunk_start = cta_in_group * chunk_size;
     const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
     const uint32_t actual_chunk_size = (chunk_start < vocab_size) ? (chunk_end - chunk_start) : 0;

     uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
 
     DType pivot = Traits::NegInf();
 
     if (k >= vocab_size) {
       // k >= vocab_size: no masking needed, just copy
       vec_t<DType, VEC_SIZE> logits_vec_copy;
       const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
 #pragma unroll 2
       for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
         logits_vec_copy.cast_load(logits + row_idx * vocab_size + chunk_start + i);
         logits_vec_copy.store(masked_logits + row_idx * vocab_size + chunk_start + i);
       }
       // Handle tail
       for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
         masked_logits[row_idx * vocab_size + chunk_start + i] =
             logits[row_idx * vocab_size + chunk_start + i];
       }
 
       // Clear histogram for next iteration (in case it's k < vocab_size)
       // Only needed for multi-CTA mode; single-CTA uses shared memory cleared each iteration
       if constexpr (!SINGLE_CTA) {
         constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;  // ORDERED_BITS / RADIX_BITS
         uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
         if (cta_in_group == 0) {
           for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
             state->histogram[next_first_hist_idx][i] = 0;
           }
         }
         // No sync needed - next iteration's barrier will ensure visibility
       }
       continue;
     }
 
     // ========== Stage 1: Load and convert to ordered representation ==========
     LoadToSharedOrdered<BLOCK_THREADS, VEC_SIZE, DType, Traits>(
         logits + row_idx * vocab_size, shared_ordered, chunk_start, actual_chunk_size, tx);
 
     // ========== Stage 2: Radix select to find pivot ==========
     uint32_t local_gt_count = 0;  // Not used in this kernel
     OrderedType ordered_pivot = RadixSelectFromSharedMemory<BLOCK_THREADS, SINGLE_CTA, OrderedType>(
         shared_ordered, actual_chunk_size, k, local_histogram, suffix_sum, shared_scalars, state,
         barrier_phase, ctas_per_group, cta_in_group, tx, iter, local_gt_count);
 
     pivot = Traits::FromOrdered(ordered_pivot);
 
     // ========== Stage 3: Final masking pass ==========
     const DType neg_inf = Traits::NegInf();
     const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
     vec_t<DType, VEC_SIZE> logits_vec;
 
 #pragma unroll 2
     for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
       logits_vec.cast_load(logits + row_idx * vocab_size + chunk_start + i);
 #pragma unroll
       for (uint32_t j = 0; j < VEC_SIZE; ++j) {
         logits_vec[j] = (logits_vec[j] >= pivot) ? logits_vec[j] : neg_inf;
       }
       logits_vec.store(masked_logits + row_idx * vocab_size + chunk_start + i);
     }
 
     // Handle tail
     for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
       DType val = logits[row_idx * vocab_size + chunk_start + i];
       masked_logits[row_idx * vocab_size + chunk_start + i] = (val >= pivot) ? val : neg_inf;
     }
   }
 
   // Clear histogram buffers and reset arrival counter for next kernel launch (only for multi-CTA)
   if constexpr (!SINGLE_CTA) {
     // Only leading CTA clears the buffers using release semantics
     if (cta_in_group == 0) {
       for (uint32_t buf = 0; buf < 3; ++buf) {
         for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
           state->histogram[buf][i] = 0;
         }
       }
 
       if (tx == 0) {
         st_release(&state->arrival_counter, 0);
       }
     }
   }
 }
 
 template <typename DType, typename IdType>
 cudaError_t RadixTopKMaskLogitsMultiCTA(DType* logits, DType* masked_logits, IdType* top_k_arr,
                                         uint32_t batch_size, uint32_t top_k_val,
                                         uint32_t vocab_size, RadixRowState* row_states_buffer,
                                         cudaStream_t stream = 0) {
   using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
   constexpr uint32_t BLOCK_THREADS = 1024;
   const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);
 
   // Get device properties
   int device;
   FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
   int num_sms;
   FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
   int max_smem_per_block;
   FLASHINFER_CUDA_CALL(
       cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
 
   // Fixed shared memory overhead: histogram[256] + suffix_sum[256] + 5 scalars
   constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 5);
   constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);
 
   // Calculate max chunk size that fits in shared memory
   const size_t available_for_ordered = max_smem_per_block - fixed_smem_aligned;
   uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
   max_chunk_elements = round_down(max_chunk_elements, vec_size);
   const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
   max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);
 
   uint32_t ctas_per_group = ceil_div(vocab_size, max_chunk_elements);
   uint32_t chunk_size = ceil_div(vocab_size, ctas_per_group);
   chunk_size = round_up(chunk_size, vec_size);
   chunk_size = std::min(chunk_size, max_chunk_elements);
 
   const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);
   const bool single_cta = (ctas_per_group == 1);
 
   // Calculate number of groups (how many rows to process concurrently)
   uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
   if (num_groups == 0) num_groups = 1;
   uint32_t total_ctas = num_groups * ctas_per_group;
 
   DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
     if (single_cta) {
       auto kernel =
           RadixTopKMaskLogitsKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, true, DType, IdType>;
       FLASHINFER_CUDA_CALL(
           cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
 
       dim3 nblks(total_ctas);
       dim3 nthrs(BLOCK_THREADS);
       void* args[] = {&logits,     &masked_logits,     &top_k_arr,  &top_k_val,     &vocab_size,
                       &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
       FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
     } else {
       auto kernel =
           RadixTopKMaskLogitsKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, false, DType, IdType>;
       FLASHINFER_CUDA_CALL(
           cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
 
       dim3 nblks(total_ctas);
       dim3 nthrs(BLOCK_THREADS);
       void* args[] = {&logits,     &masked_logits,     &top_k_arr,  &top_k_val,     &vocab_size,
                       &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
       FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
     }
   });
 
   return cudaSuccess;
 }
 
 // ==================== Multi-CTA Radix Top-K Renorm Probs ====================
 
 /*!
  * \brief Multi-CTA Radix Top-K RenormProb kernel with unified single/multi-CTA paths.
  *
  * Finds the k-th largest probability, then normalizes all probs >= pivot to sum to 1,
  * setting all others to 0. Uses the shared RadixSelectFindPivot function.
  */
 template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType,
           typename IdType>
 __global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKRenormProbKernel_MultiCTA(
     DType* probs,          // [batch, vocab_size]
     DType* renormed_prob,  // [batch, vocab_size]
     IdType* top_k_arr,     // [batch] or nullptr
     uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
     RadixRowState* row_states,  // [num_groups] (nullptr if SINGLE_CTA)
     uint32_t chunk_size,        // elements per CTA
     uint32_t ctas_per_group)    // CTAs per row (1 if SINGLE_CTA)
 {
   using Traits = RadixTopKTraits<DType>;
   using OrderedType = typename Traits::OrderedType;
 
   constexpr uint32_t RADIX = 256;
 
   const uint32_t global_cta_id = blockIdx.x;
   const uint32_t group_id = global_cta_id / ctas_per_group;
   const uint32_t cta_in_group = global_cta_id % ctas_per_group;
   const uint32_t tx = threadIdx.x;
 
   // Shared memory layout: [fixed storage] [ordered values cache]
   extern __shared__ uint8_t smem[];
 
   // Fixed shared memory (at the beginning)
   // histogram[256] + suffix[256] + scalars[4] + sum_local[1]
   constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 4) + sizeof(float);
   uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
   uint32_t* suffix_sum = local_histogram + RADIX;
   uint32_t* shared_scalars = suffix_sum + RADIX;
   float* shared_sum = reinterpret_cast<float*>(shared_scalars + 4);
 
   // Align ordered values cache to 16 bytes
   size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
   OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);
 
   // State pointer only used when not SINGLE_CTA
   RadixRowState* state = nullptr;
   if constexpr (!SINGLE_CTA) {
     state = &row_states[group_id];
   }
 
   // Calculate total number of iterations for persistent loop
   uint32_t num_groups = gridDim.x / ctas_per_group;
   uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;
 
   int barrier_phase = 0;
 
   // Persistent loop over rows
   for (uint32_t iter = 0; iter < total_iterations; iter++) {
     uint32_t row_idx = group_id + iter * num_groups;
 
     if (row_idx >= batch_size) break;
 
     const uint32_t chunk_start = cta_in_group * chunk_size;
     const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
     const uint32_t actual_chunk_size = (chunk_start < vocab_size) ? (chunk_end - chunk_start) : 0;

     uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
 
     // For RenormProb, pivot is compared with probs (must be non-negative)
     DType pivot = DType(0);
     float normalizer = 1.0f;
 
     if (k >= vocab_size) {
       // k >= vocab_size: no filtering needed, just compute sum and renormalize
       // Stage 1: Compute sum
       float thread_sum = 0.0f;
       vec_t<DType, VEC_SIZE> data_vec;
       const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
 
 #pragma unroll 2
       for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
         data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
 #pragma unroll
         for (uint32_t j = 0; j < VEC_SIZE; ++j) {
           thread_sum += float(data_vec[j]);
         }
       }
       // Handle tail
       for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
         thread_sum += float(probs[row_idx * vocab_size + chunk_start + i]);
       }
 
       // Block reduction for sum
       typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
       __shared__ typename BlockReduce::TempStorage temp_storage;
       float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
       __syncthreads();
 
       if constexpr (!SINGLE_CTA) {
         // Multi-CTA: atomic add to global sum
         if (tx == 0) {
           if (cta_in_group == 0) {
             state->sum_topk = 0.0f;  // First CTA initializes
           }
         }
         // Barrier for initialization
         if (tx == 0) {
           red_release(&state->arrival_counter, 1);
         }
         int target = (barrier_phase + 1) * ctas_per_group;
         wait_ge(&state->arrival_counter, target, tx);
         barrier_phase++;
         __syncthreads();
 
         if (tx == 0 && block_sum > 0) {
           atomicAdd(&state->sum_topk, block_sum);
         }
 
         // Barrier to ensure all CTAs have contributed
         if (tx == 0) {
           red_release(&state->arrival_counter, 1);
         }
         target = (barrier_phase + 1) * ctas_per_group;
         wait_ge(&state->arrival_counter, target, tx);
         barrier_phase++;
         __syncthreads();
 
         normalizer = math::ptx_rcp(max(state->sum_topk, 1e-8f));
       } else {
         // Single-CTA: use block_sum directly
         if (tx == 0) {
           *shared_sum = block_sum;
         }
         __syncthreads();
         normalizer = math::ptx_rcp(max(*shared_sum, 1e-8f));
       }
 
       // Normalize and store
 #pragma unroll 2
       for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
         data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
 #pragma unroll
         for (uint32_t j = 0; j < VEC_SIZE; ++j) {
           data_vec[j] = DType(float(data_vec[j]) * normalizer);
         }
         data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
       }
       for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
         renormed_prob[row_idx * vocab_size + chunk_start + i] =
             DType(float(probs[row_idx * vocab_size + chunk_start + i]) * normalizer);
       }
 
       // Clear histogram for next iteration (in case it's k < vocab_size)
       // Only needed for multi-CTA mode; single-CTA uses shared memory cleared each iteration
       // Next iteration (iter+1) will use histogram[((iter+1)*NUM_ROUNDS) % 3] for its first round
       if constexpr (!SINGLE_CTA) {
         constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;  // ORDERED_BITS / RADIX_BITS
         uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
         if (cta_in_group == 0) {
           for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
             state->histogram[next_first_hist_idx][i] = 0;
           }
         }
         // No sync needed - next iteration's barrier will ensure visibility
       }
       continue;
     }
 
     // ========== Stage 1: Find pivot using RadixSelectFindPivot ==========
     pivot = RadixSelectFindPivot<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, DType>(
         probs + row_idx * vocab_size, shared_ordered, local_histogram, suffix_sum, shared_scalars,
         state, chunk_start, actual_chunk_size, k, barrier_phase, ctas_per_group, cta_in_group, tx,
         iter);
 
     // ========== Stage 2: Compute sum of elements >= pivot ==========
     float thread_sum = 0.0f;
     vec_t<DType, VEC_SIZE> data_vec;
     const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
 
 #pragma unroll 2
     for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
       data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
 #pragma unroll
       for (uint32_t j = 0; j < VEC_SIZE; ++j) {
         if (data_vec[j] >= pivot) {
           thread_sum += float(data_vec[j]);
         }
       }
     }
     // Handle tail
     for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
       DType val = probs[row_idx * vocab_size + chunk_start + i];
       if (val >= pivot) {
         thread_sum += float(val);
       }
     }
 
     // Block reduction for sum
     typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
     __shared__ typename BlockReduce::TempStorage temp_storage;
     float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
     __syncthreads();
 
     if constexpr (!SINGLE_CTA) {
       // Multi-CTA: atomic add to global sum
       if (tx == 0) {
         if (cta_in_group == 0) {
           state->sum_topk = 0.0f;  // First CTA initializes
         }
       }
       // Barrier for initialization
       if (tx == 0) {
         red_release(&state->arrival_counter, 1);
       }
       int target = (barrier_phase + 1) * ctas_per_group;
       wait_ge(&state->arrival_counter, target, tx);
       barrier_phase++;
       __syncthreads();
 
       if (tx == 0 && block_sum > 0) {
         atomicAdd(&state->sum_topk, block_sum);
       }
 
       // Barrier to ensure all CTAs have contributed
       if (tx == 0) {
         red_release(&state->arrival_counter, 1);
       }
       target = (barrier_phase + 1) * ctas_per_group;
       wait_ge(&state->arrival_counter, target, tx);
       barrier_phase++;
       __syncthreads();
 
       normalizer = math::ptx_rcp(max(state->sum_topk, 1e-8f));
     } else {
       // Single-CTA: use block_sum directly
       if (tx == 0) {
         *shared_sum = block_sum;
       }
       __syncthreads();
       normalizer = math::ptx_rcp(max(*shared_sum, 1e-8f));
     }
 
     // ========== Stage 3: Normalize elements >= pivot, set others to 0 ==========
 #pragma unroll 2
     for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
       data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
 #pragma unroll
       for (uint32_t j = 0; j < VEC_SIZE; ++j) {
         data_vec[j] = (data_vec[j] >= pivot) ? DType(float(data_vec[j]) * normalizer) : DType(0);
       }
       data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
     }
     // Handle tail
     for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
       DType val = probs[row_idx * vocab_size + chunk_start + i];
       renormed_prob[row_idx * vocab_size + chunk_start + i] =
           (val >= pivot) ? DType(float(val) * normalizer) : DType(0);
     }
   }
 
   // Clear histogram buffers and reset arrival counter for next kernel launch (only for multi-CTA)
   if constexpr (!SINGLE_CTA) {
     // Only leading CTA clears the buffers using release semantics
     if (cta_in_group == 0) {
       for (uint32_t buf = 0; buf < 3; ++buf) {
         for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
           state->histogram[buf][i] = 0;
         }
       }
 
       if (tx == 0) {
         st_release(&state->arrival_counter, 0);
       }
     }
   }
 }
 
 template <typename DType, typename IdType>
 cudaError_t RadixTopKRenormProbMultiCTA(DType* probs, DType* renormed_prob, IdType* top_k_arr,
                                         uint32_t batch_size, uint32_t top_k_val,
                                         uint32_t vocab_size, RadixRowState* row_states_buffer,
                                         cudaStream_t stream = 0) {
   using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
   constexpr uint32_t BLOCK_THREADS = 1024;
   const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);
 
   // Get device properties
   int device;
   FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
   int num_sms;
   FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
   int max_smem_per_block;
   FLASHINFER_CUDA_CALL(
       cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
 
   // Fixed shared memory overhead: histogram[256] + suffix_sum[256] + 4 scalars + 1 float
   constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 4) + sizeof(float);
   constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);
 
   // Calculate max chunk size that fits in shared memory
   const size_t available_for_ordered = max_smem_per_block - fixed_smem_aligned;
   uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
   max_chunk_elements = round_down(max_chunk_elements, vec_size);
   const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
   max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);
 
   uint32_t ctas_per_group = ceil_div(vocab_size, max_chunk_elements);
   uint32_t chunk_size = ceil_div(vocab_size, ctas_per_group);
   chunk_size = round_up(chunk_size, vec_size);
   chunk_size = std::min(chunk_size, max_chunk_elements);
 
   const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);
   const bool single_cta = (ctas_per_group == 1);
 
   // Calculate number of groups (how many rows to process concurrently)
   uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
   if (num_groups == 0) num_groups = 1;
   uint32_t total_ctas = num_groups * ctas_per_group;
 
   DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
     if (single_cta) {
       auto kernel =
           RadixTopKRenormProbKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, true, DType, IdType>;
       FLASHINFER_CUDA_CALL(
           cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
 
       dim3 nblks(total_ctas);
       dim3 nthrs(BLOCK_THREADS);
       void* args[] = {&probs,      &renormed_prob,     &top_k_arr,  &top_k_val,     &vocab_size,
                       &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
       FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
     } else {
       auto kernel =
           RadixTopKRenormProbKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, false, DType, IdType>;
       FLASHINFER_CUDA_CALL(
           cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
 
       dim3 nblks(total_ctas);
       dim3 nthrs(BLOCK_THREADS);
       void* args[] = {&probs,      &renormed_prob,     &top_k_arr,  &top_k_val,     &vocab_size,
                       &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
       FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
     }
   });
 
   return cudaSuccess;
 }
 

 
 /*!
  * \brief Launch multi-CTA Radix Top-K with Ragged Index Transform kernel.
  *
  * Performs top-k selection and adds an offset to each index.
  * Used for sparse attention's second stage with ragged KV cache.
  *
  * \param input Input scores tensor [num_rows, max_len]
  * \param output_indices Output indices [num_rows, top_k]
  * \param offsets Offset to add per row [num_rows]
  * \param lengths Sequence lengths per row [num_rows]
  * \param num_rows Number of rows to process
  * \param top_k_val Number of top elements to select
  * \param max_len Maximum sequence length (input stride)
  * \param row_states_buffer Buffer for inter-CTA synchronization
  * \param stream CUDA stream
  */
 template <typename DType, typename IdType>
 cudaError_t RadixTopKRaggedTransformMultiCTA(DType* input, IdType* output_indices,
                                              IdType* lengths,
                                              uint32_t num_rows, uint32_t top_k_val,
                                              uint32_t max_len, RadixRowState* row_states_buffer,
                                              cudaStream_t stream = 0) {
   using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
   constexpr uint32_t BLOCK_THREADS = 1024;
   const uint32_t vec_size = std::gcd(16 / sizeof(DType), max_len);
 
   int device;
   FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
   int num_sms;
   FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
   int max_smem_per_block;
   FLASHINFER_CUDA_CALL(
       cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
 
   constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 5);
   constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);
 
   const size_t available_for_ordered = max_smem_per_block - fixed_smem_aligned;
   uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
   max_chunk_elements = round_down(max_chunk_elements, vec_size);
   const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
   max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);
 
   uint32_t ctas_per_group = ceil_div(max_len, max_chunk_elements);
   uint32_t chunk_size = ceil_div(max_len, ctas_per_group);
   chunk_size = round_up(chunk_size, vec_size);
   chunk_size = std::min(chunk_size, max_chunk_elements);
 
   const bool single_cta = (ctas_per_group == 1);
   const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);
 
   uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, num_rows);
   if (num_groups == 0) num_groups = 1;
   uint32_t total_ctas = num_groups * ctas_per_group;
 
   // Unified kernel parameters
   DType* output_values = nullptr;        // Not used in RaggedTransform mode
   const IdType* row_to_batch = nullptr;  // Not used in RaggedTransform mode
   const IdType* offsets = nullptr;       // Not used in RaggedTransform mode (offset-adding disabled)
   int64_t aux_stride = 0;                // Not used in RaggedTransform mode
 
   DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
     if (single_cta) {
       auto kernel = RadixTopKKernel_Unified<BLOCK_THREADS, VEC_SIZE, true,
                                             RadixTopKMode::RaggedTransform, DType, IdType>;
       FLASHINFER_CUDA_CALL(
           cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
       dim3 nblks(total_ctas);
       dim3 nthrs(BLOCK_THREADS);
       void* args[] = {&input,         &output_indices, &output_values,     &offsets,
                       &lengths,       &row_to_batch,   &aux_stride,        &top_k_val,
                       &max_len,       &num_rows,       &row_states_buffer, &chunk_size,
                       &ctas_per_group};
       FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
     } else {
       auto kernel = RadixTopKKernel_Unified<BLOCK_THREADS, VEC_SIZE, false,
                                             RadixTopKMode::RaggedTransform, DType, IdType>;
       FLASHINFER_CUDA_CALL(
           cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
       dim3 nblks(total_ctas);
       dim3 nthrs(BLOCK_THREADS);
       void* args[] = {&input,         &output_indices, &output_values,     &offsets,
                       &lengths,       &row_to_batch,   &aux_stride,        &top_k_val,
                       &max_len,       &num_rows,       &row_states_buffer, &chunk_size,
                       &ctas_per_group};
       FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
     }
   });
 
   return cudaSuccess;
 }
 

 // ==================== FilteredTopK Implementation ====================
 // Based on sgl-kernel's filter algorithm with multi-dtype support
 
 // FilteredTopK traits for different data types
 template <typename DType>
 struct FilteredTopKTraits;
 
 // Specialization for float (32-bit): coarse histogram uses FP16 high 8 bits, 4 refinement rounds
 template <>
 struct FilteredTopKTraits<float> {
   using OrderedType = uint32_t;
   static constexpr int NUM_REFINE_ROUNDS = 4;
   static constexpr int FIRST_REFINE_SHIFT = 24;
 
   __device__ __forceinline__ static uint8_t ToCoarseKey(float x) {
     // Convert to FP16 representation and extract high 8 bits
     __half h = __float2half_rn(x);
     uint16_t bits = __half_as_ushort(h);
     uint16_t key =
         (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
     return static_cast<uint8_t>(key >> 8);
   }
 
   __device__ __forceinline__ static OrderedType ToOrdered(float x) {
     uint32_t bits = __float_as_uint(x);
     return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
   }
 };
 
 // Specialization for half (16-bit): coarse histogram uses high 8 bits, only need low 8 bits for
 // refinement Since coarse key = high 8 bits, refinement only needs to look at low 8 bits (no
 // additional rounds needed if we can determine topk from coarse pass alone)
 template <>
 struct FilteredTopKTraits<half> {
   using OrderedType = uint16_t;
   static constexpr int NUM_REFINE_ROUNDS = 1;   // Only 1 round for low 8 bits
   static constexpr int FIRST_REFINE_SHIFT = 0;  // Start from bit 0 (low 8 bits)
 
   __device__ __forceinline__ static uint8_t ToCoarseKey(half x) {
     uint16_t bits = __half_as_ushort(x);
     uint16_t key =
         (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
     return static_cast<uint8_t>(key >> 8);
   }
 
   __device__ __forceinline__ static OrderedType ToOrdered(half x) {
     uint16_t bits = __half_as_ushort(x);
     return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
   }
 };
 
 // Specialization for nv_bfloat16 (16-bit): same as half
 template <>
 struct FilteredTopKTraits<nv_bfloat16> {
   using OrderedType = uint16_t;
   static constexpr int NUM_REFINE_ROUNDS = 1;
   static constexpr int FIRST_REFINE_SHIFT = 0;
 
   __device__ __forceinline__ static uint8_t ToCoarseKey(nv_bfloat16 x) {
     uint16_t bits = __bfloat16_as_ushort(x);
     uint16_t key =
         (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
     return static_cast<uint8_t>(key >> 8);
   }
 
   __device__ __forceinline__ static OrderedType ToOrdered(nv_bfloat16 x) {
     uint16_t bits = __bfloat16_as_ushort(x);
     return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
   }
 };
 
 // FilteredTopK constants
 constexpr uint32_t FILTERED_TOPK_MAX_K = 2048;
 constexpr uint32_t FILTERED_TOPK_BLOCK_THREADS = 1024;
 constexpr uint32_t FILTERED_TOPK_SMEM_INPUT_SIZE = 16 * 1024;  // 16K indices per buffer
 constexpr size_t FILTERED_TOPK_SMEM_DYNAMIC =
     sizeof(int) * 2 * FILTERED_TOPK_SMEM_INPUT_SIZE;  // 128KB
 
 // Output modes for unified FilteredTopK kernel
 enum class FilteredTopKMode { Plain, PageTable, Ragged };
 
 /*!
  * \brief Unified Filtered Top-K kernel supporting multiple output modes.
  *
  * \tparam DType Data type (float, half, nv_bfloat16)
  * \tparam IdType Index type (int32_t)
  * \tparam VEC_SIZE Vector size for input loads (1, 2, 4, or 8)
  * \tparam MODE Output mode (Plain, PageTable, Ragged)
  *
  * Parameters vary by mode:
  * - Plain: output = indices, aux_output = values, aux_input/aux_stride/row_to_batch unused
  * - PageTable: output = dst_page_table, aux_input = src_page_table, aux_stride = src_stride
  * - Ragged: output = indices, aux_input = offsets, aux_output/aux_stride/row_to_batch unused
  */
 template <typename DType, typename IdType, int VEC_SIZE, FilteredTopKMode MODE>
 __global__ void __launch_bounds__(FILTERED_TOPK_BLOCK_THREADS)
     FilteredTopKUnifiedKernel(const DType* __restrict__ input, IdType* __restrict__ output,
                               DType* __restrict__ aux_output,           // values for Plain mode
                               const IdType* __restrict__ aux_input,     // page_table or offsets
                               int64_t aux_stride,                       // src_stride for PageTable
                               const IdType* __restrict__ row_to_batch,  // for PageTable
                               const IdType* __restrict__ lengths, uint32_t num_rows, uint32_t top_k,
                               uint32_t max_len) {
   constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
   constexpr int RADIX = 256;
   constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;
 
   const uint32_t bid = blockIdx.x;
   const int tx = threadIdx.x;
 
   if (bid >= num_rows) return;
 
   const int length = (lengths != nullptr) ? lengths[bid] : static_cast<int>(max_len);
   const DType* score = input + bid * max_len;
   IdType* dst = output + bid * top_k;
 
   // Mode-specific setup
   [[maybe_unused]] const IdType* src_page_entry = nullptr;
   [[maybe_unused]] IdType offset_val = 0;
   [[maybe_unused]] DType* dst_values = nullptr;
 
   if constexpr (MODE == FilteredTopKMode::PageTable) {
     const uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[bid] : bid;
     src_page_entry = aux_input + batch_idx * aux_stride;
   } else if constexpr (MODE == FilteredTopKMode::Ragged) {
     offset_val = aux_input[bid];
   } else {  // Plain
     dst_values = aux_output + bid * top_k;
   }
 
   // Trivial case: length <= top_k
   if (length <= static_cast<int>(top_k)) {
     for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
       if constexpr (MODE == FilteredTopKMode::PageTable) {
         dst[i] = (i < length) ? src_page_entry[i] : static_cast<IdType>(-1);
       } else if constexpr (MODE == FilteredTopKMode::Ragged) {
         dst[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
       } else {  // Plain
         if (i < length) {
           dst[i] = static_cast<IdType>(i);
           dst_values[i] = score[i];
         } else {
           dst[i] = static_cast<IdType>(-1);
           dst_values[i] = DType(0);
         }
       }
     }
     return;
   }
 
   // Static shared memory
   alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
   alignas(128) __shared__ int s_counter;
   alignas(128) __shared__ int s_threshold_bin_id;
   alignas(128) __shared__ int s_num_input[2];
   alignas(128) __shared__ int s_indices[FILTERED_TOPK_MAX_K];
 
   auto& s_histogram = s_histogram_buf[0];
 
   // Dynamic shared memory for input double buffer
   extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];
 
   using Traits = FilteredTopKTraits<DType>;
   int topk = top_k;
 
   // Stage 1: 8-bit coarse histogram with vectorized loads
   if (tx < RADIX + 1) s_histogram[tx] = 0;
   __syncthreads();
 
   vec_t<DType, VEC_SIZE> score_vec;
 
   const int aligned_length = (length / VEC_SIZE) * VEC_SIZE;
 #pragma unroll 2
   for (int base = tx * VEC_SIZE; base < aligned_length; base += BLOCK_SIZE * VEC_SIZE) {
     score_vec.cast_load(&score[base]);
 #pragma unroll
     for (int j = 0; j < VEC_SIZE; ++j) {
       const auto bin = Traits::ToCoarseKey(score_vec[j]);
       atomicAdd(&s_histogram[bin], 1);
     }
   }
   // Handle tail
   for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
     const auto bin = Traits::ToCoarseKey(score[i]);
     atomicAdd(&s_histogram[bin], 1);
   }
   __syncthreads();
 
   // Suffix sum
   const auto run_cumsum = [&]() {
 #pragma unroll 8
     for (int i = 0; i < 8; ++i) {
       if (tx < RADIX) {
         const auto j = 1 << i;
         const auto k = i & 1;
         auto value = s_histogram_buf[k][tx];
         if (tx < RADIX - j) {
           value += s_histogram_buf[k][tx + j];
         }
         s_histogram_buf[k ^ 1][tx] = value;
       }
       __syncthreads();
     }
   };
 
   run_cumsum();
   if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
     s_threshold_bin_id = tx;
     s_num_input[0] = 0;
     s_counter = 0;
   }
   __syncthreads();
 
   const auto threshold_bin = s_threshold_bin_id;
   topk -= s_histogram[threshold_bin + 1];
 
   constexpr int NUM_ROUNDS = Traits::NUM_REFINE_ROUNDS;
   constexpr int FIRST_SHIFT = Traits::FIRST_REFINE_SHIFT;
 
   if (topk == 0) {
     // Collect indices where bin > threshold
 #pragma unroll 2
     for (int base = tx * VEC_SIZE; base < aligned_length; base += BLOCK_SIZE * VEC_SIZE) {
       score_vec.cast_load(&score[base]);
 #pragma unroll
       for (int j = 0; j < VEC_SIZE; ++j) {
         const auto bin = static_cast<int>(Traits::ToCoarseKey(score_vec[j]));
         if (bin > threshold_bin) {
           const auto pos = atomicAdd(&s_counter, 1);
           s_indices[pos] = base + j;
         }
       }
     }
     // Handle tail
     for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
       const auto bin = static_cast<int>(Traits::ToCoarseKey(score[i]));
       if (bin > threshold_bin) {
         const auto pos = atomicAdd(&s_counter, 1);
         s_indices[pos] = i;
       }
     }
     __syncthreads();
   } else {
     __syncthreads();
     if (tx < RADIX + 1) s_histogram[tx] = 0;
     __syncthreads();
 
     // Filter + histogram for refinement
     auto filter_and_add_to_histogram = [&](auto raw_input, int index) {
       const auto bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
       if (bin > threshold_bin) {
         const auto pos = atomicAdd(&s_counter, 1);
         s_indices[pos] = index;
       } else if (bin == threshold_bin) {
         const auto pos = atomicAdd(&s_num_input[0], 1);
         if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
           s_input_idx[0][pos] = index;
           const auto ordered = Traits::ToOrdered(raw_input);
           const auto sub_bin = (ordered >> FIRST_SHIFT) & 0xFF;
           atomicAdd(&s_histogram[sub_bin], 1);
         }
       }
     };
 #pragma unroll 2
     for (int base = tx * VEC_SIZE; base < aligned_length; base += BLOCK_SIZE * VEC_SIZE) {
       score_vec.cast_load(&score[base]);
 #pragma unroll
       for (int j = 0; j < VEC_SIZE; ++j) {
         filter_and_add_to_histogram(score_vec[j], base + j);
       }
     }
     // Handle tail
     for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
       filter_and_add_to_histogram(score[i], i);
     }
     __syncthreads();
 
     // Stage 2: refine with 8bit radix passes
 #pragma unroll
     for (int round = 0; round < NUM_ROUNDS; ++round) {
       __shared__ int s_last_remain;
       const auto r_idx = round % 2;
 
       const auto _raw_num_input = s_num_input[r_idx];
       const auto num_input = (_raw_num_input < SMEM_INPUT_SIZE) ? _raw_num_input : SMEM_INPUT_SIZE;
 
       run_cumsum();
       if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
         s_threshold_bin_id = tx;
         s_num_input[r_idx ^ 1] = 0;
         s_last_remain = topk - s_histogram[tx + 1];
       }
       __syncthreads();
 
       const auto threshold = s_threshold_bin_id;
       topk -= s_histogram[threshold + 1];
 
       const int offset = FIRST_SHIFT - round * 8;
       const bool is_last_round = (round == NUM_ROUNDS - 1);
 
       if (topk == 0) {
         for (int i = tx; i < num_input; i += BLOCK_SIZE) {
           const auto idx = s_input_idx[r_idx][i];
           const auto bin = (Traits::ToOrdered(score[idx]) >> offset) & 0xFF;
           if (static_cast<int>(bin) > threshold) {
             const auto pos = atomicAdd(&s_counter, 1);
             s_indices[pos] = idx;
           }
         }
         __syncthreads();
         break;
       } else {
         __syncthreads();
         if (tx < RADIX + 1) s_histogram[tx] = 0;
         __syncthreads();
         for (int i = tx; i < num_input; i += BLOCK_SIZE) {
           const auto idx = s_input_idx[r_idx][i];
           const auto raw_input = score[idx];
           const auto bin = (Traits::ToOrdered(raw_input) >> offset) & 0xFF;
           if (static_cast<int>(bin) > threshold) {
             const auto pos = atomicAdd(&s_counter, 1);
             s_indices[pos] = idx;
           } else if (static_cast<int>(bin) == threshold) {
             if (is_last_round) {
               const auto pos = atomicAdd(&s_last_remain, -1);
               if (pos > 0) {
                 s_indices[top_k - pos] = idx;
               }
             } else {
               const auto pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
               if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
                 s_input_idx[r_idx ^ 1][pos] = idx;
                 const auto bin32 = Traits::ToOrdered(raw_input);
                 const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
                 atomicAdd(&s_histogram[sub_bin], 1);
               }
             }
           }
         }
         __syncthreads();
       }
     }
   }
 
   // Output phase - mode-specific
 #pragma unroll 2
   for (int base = tx; base < static_cast<int>(top_k); base += BLOCK_SIZE) {
     const int idx = s_indices[base];
     if constexpr (MODE == FilteredTopKMode::PageTable) {
       dst[base] = src_page_entry[idx];
     } else if constexpr (MODE == FilteredTopKMode::Ragged) {
       dst[base] = static_cast<IdType>(idx);
     } else {  // Plain
       dst[base] = static_cast<IdType>(idx);
       dst_values[base] = score[idx];
     }
   }
 }
 
 // Helper to compute GCD for VEC_SIZE selection
 constexpr uint32_t gcd(uint32_t a, uint32_t b) {
   while (b != 0) {
     uint32_t t = b;
     b = a % b;
     a = t;
   }
   return a;
 }
 
 // Compute optimal VEC_SIZE based on max_len and dtype
 // Returns 1, 2, 4, or 8
 template <typename DType>
 constexpr int ComputeFilteredTopKVecSize(uint32_t max_len) {
   constexpr int MAX_VEC = 16 / sizeof(DType);  // 4 for float32, 8 for fp16/bf16
   // Use GCD to find largest power-of-2 divisor
   const uint32_t g = gcd(max_len, static_cast<uint32_t>(MAX_VEC));
   return static_cast<int>(g);
 }
 
 
 template <typename DType, typename IdType>
 cudaError_t FilteredTopKRaggedTransform(DType* input, IdType* output_indices,
                                         IdType* lengths, uint32_t num_rows, uint32_t top_k_val,
                                         uint32_t max_len, cudaStream_t stream = 0) {
   constexpr size_t smem_size = FILTERED_TOPK_SMEM_DYNAMIC;
   constexpr int MAX_VEC = 16 / sizeof(DType);
 
   dim3 grid(num_rows);
   dim3 block(FILTERED_TOPK_BLOCK_THREADS);
   DType* aux_output = nullptr;           // Not used for Ragged mode
   int64_t aux_stride = 0;                // Not used for Ragged mode
   const IdType* row_to_batch = nullptr;  // Not used for Ragged mode
   const IdType* offsets = nullptr;       // Not used for Ragged mode (offset-adding disabled)
   void* args[] = {&input,        &output_indices, &aux_output, &offsets,   &aux_stride,
                   &row_to_batch, &lengths,        &num_rows,   &top_k_val, &max_len};
 
   const int vec_size = ComputeFilteredTopKVecSize<DType>(max_len);
 
 #define DISPATCH_VEC_SIZE(VS)                                                                    \
   if (vec_size == VS) {                                                                          \
     auto kernel = FilteredTopKUnifiedKernel<DType, IdType, VS, FilteredTopKMode::Ragged>;        \
     FLASHINFER_CUDA_CALL(                                                                        \
         cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));   \
     FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, smem_size, stream)); \
     return cudaSuccess;                                                                          \
   }
 
   DISPATCH_VEC_SIZE(1)
   DISPATCH_VEC_SIZE(2)
   DISPATCH_VEC_SIZE(4)
   if constexpr (MAX_VEC >= 8) {
     DISPATCH_VEC_SIZE(8)
   }
 #undef DISPATCH_VEC_SIZE
 
   return cudaSuccess;
 }
 
 
 /*!
  * \brief Check if the GPU supports enough shared memory for FilteredTopK algorithm.
  *
  * FilteredTopK requires 128KB dynamic shared memory. This function checks if the
  * current GPU's max shared memory per SM is sufficient.
  *
  * \return true if GPU supports FilteredTopK, false otherwise
  */
 inline bool CanImplementFilteredTopK() {
   int device_id;
   if (cudaGetDevice(&device_id) != cudaSuccess) return false;
   int max_smem_per_sm;
   if (cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                              device_id) != cudaSuccess) {
     return false;
   }
   return static_cast<size_t>(max_smem_per_sm) >= FILTERED_TOPK_SMEM_DYNAMIC;
 }
 
 // Algorithm override for benchmarking (controlled by FLASHINFER_TOPK_ALGO env var)
 enum class TopKAlgoOverride { AUTO, FILTERED, MULTI_CTA };
 
 inline TopKAlgoOverride GetTopKAlgoOverride() {
   const char* env = std::getenv("FLASHINFER_TOPK_ALGO");
   if (env == nullptr) return TopKAlgoOverride::AUTO;
   if (std::strcmp(env, "filtered") == 0) return TopKAlgoOverride::FILTERED;
   if (std::strcmp(env, "multi_cta") == 0) return TopKAlgoOverride::MULTI_CTA;
   return TopKAlgoOverride::AUTO;
 }
 
 /*!
  * \brief Unified heuristic to decide whether to use FilteredTopK or Multi-CTA RadixTopK.
  *
  * \tparam DType Data type (affects threshold due to memory bandwidth considerations)
  * \param num_rows Number of rows (batch size)
  * \param top_k_val Number of top elements to select
  * \param max_len Maximum sequence length
  * \return true if FilteredTopK should be used, false for Multi-CTA RadixTopK
  *
  * Heuristics:
  * - 16-bit types (fp16/bf16): FilteredTopK for seq <= 16K
  * - 32-bit types (fp32): FilteredTopK for seq <= 32K, or larger seq with batch > seq/16K
  */
 template <typename DType>
 inline bool ShouldUseFilteredTopK(uint32_t num_rows, uint32_t top_k_val, uint32_t max_len) {
   // Check if GPU supports enough shared memory for FilteredTopK
   const bool gpu_supports_filtered = CanImplementFilteredTopK();
   const bool k_fits_filtered = (top_k_val <= FILTERED_TOPK_MAX_K) && (max_len > top_k_val);
 
   if (!gpu_supports_filtered || !k_fits_filtered) {
     return false;
   }
 
   // Check for algorithm override
   const TopKAlgoOverride algo_override = GetTopKAlgoOverride();
   if (algo_override == TopKAlgoOverride::FILTERED) return true;
   if (algo_override == TopKAlgoOverride::MULTI_CTA) return false;
 
   // Auto heuristics based on dtype
   if constexpr (sizeof(DType) <= 2) {
     // 16-bit types: simpler threshold at 16K
     return (max_len <= 16384);
   } else {
     // 32-bit types: must respect FILTERED_TOPK_SMEM_INPUT_SIZE (16K) limit
     if (max_len <= 16384) {
       return true;
     } else {
       const uint32_t batch_threshold = max_len / 16384;
       return (num_rows > batch_threshold);
     }
   }
 }
 
 // Dispatch functions with heuristics
 
 template <typename DType, typename IdType>
 cudaError_t TopKRaggedTransformDispatch(DType* input, IdType* output_indices,
                                         IdType* lengths, uint32_t num_rows, uint32_t top_k_val,
                                         uint32_t max_len, RadixRowState* row_states_buffer,
                                         cudaStream_t stream = 0) {
   if (ShouldUseFilteredTopK<DType>(num_rows, top_k_val, max_len)) {
     return FilteredTopKRaggedTransform<DType, IdType>(input, output_indices, lengths,
                                                       num_rows, top_k_val, max_len, stream);
   }
   return RadixTopKRaggedTransformMultiCTA<DType, IdType>(input, output_indices, lengths,
                                                          num_rows, top_k_val, max_len,
                                                          row_states_buffer, stream);
 }
 
 
 }  // namespace sampling
 
 }  // namespace vllm
 
 #endif  // FLASHINFER_TOPK_CUH_
 