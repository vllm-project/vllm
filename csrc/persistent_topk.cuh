/*
 * Persistent TopK Scheduler for DSA (DeepSeek Attention) Indexer
 *
 * Single persistent kernel with three paths selected by seq_len:
 *   - Decode path (seq_len <= 8K):  2048-bin histogram + CUB suffix sum
 *   - Medium path (8K < seq_len <= 64K): coarse FP16 histogram + FP32 radix
 *   - Large path  (seq_len > 64K): static CTA groups, multi-CTA radix
 *
 * Key optimizations vs standalone kernels:
 *   - __launch_bounds__(1024, 1) for more register headroom
 *   - Round-robin scheduling (no atomic counter, no cudaMemsetAsync)
 *   - Grid size = min(num_sms, num_rows) to avoid idle CTAs
 *   - __noinline__ on path functions for independent register allocation
 *   - Float4 vectorized loads with per-element predication in decode path
 *   - Warp-aggregated atomics + register-cached bins in decode path
 */

 #ifndef PERSISTENT_TOPK_CUH_
 #define PERSISTENT_TOPK_CUH_
 
 #include "persistent_topk_common.cuh"
 #include "persistent_topk_decode.cuh"
 #include "persistent_topk_medium.cuh"
 #include "persistent_topk_large.cuh"
 
 namespace vllm {
 namespace persistent {
 
 // ============================================================================
 // Persistent kernel
 // ============================================================================
 
 template <bool USE_LARGE_PATH, uint32_t VEC_SIZE = 1>
 __global__ void __launch_bounds__(kThreadsPerBlock, 1)
     persistent_topk_kernel(PersistentTopKParams params) {
 
   if constexpr (!USE_LARGE_PATH) {
     // Round-robin scheduling, 1 CTA per row
     for (uint32_t row_idx = blockIdx.x; row_idx < params.num_rows;
          row_idx += gridDim.x) {
       const int seq_len = params.lengths[row_idx];
       int32_t* output_indices = params.output + row_idx * TopK;
       const float* logits = params.input + row_idx * params.stride;
 
       if (seq_len <= TopK) {
         naive_topk_cuda(logits, output_indices, seq_len);
       } else if (seq_len <= static_cast<int>(DECODE_THRESHOLD)) {
         decode_topk_cuda(logits, output_indices, seq_len);
       } else {
         fast_topk_cuda_tl(logits, output_indices, 0, seq_len);
       }
     }
   } else {
     large_topk_cuda<VEC_SIZE>(params);
   }
 }
 
 }  // namespace persistent
 }  // namespace vllm
 
 #endif  // PERSISTENT_TOPK_CUH_
 