// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <ext/intel/esimd.hpp>

// clang-format on
#include <float.h>
#include <torch/extension.h>
#include <stdexcept>
#include "utils.h"
#include "xpu_types.h"
// #include "dtype_bfloat16.dp.hpp"
#include "dtype_float16.h"
#include "dtype_float32.h"
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
#include <c10/xpu/XPUStream.h>
#endif

#include <functional>
// #include <ipex.h>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))
using namespace sycl::ext::intel::esimd;

template<typename T>
static inline T attn_softcapping(T qk, float attn_logit_softcapping) {
    qk = qk / attn_logit_softcapping;
    qk = (sycl::exp(qk) - sycl::exp(-qk)) / (sycl::exp(qk) + sycl::exp(-qk));
    qk = qk * attn_logit_softcapping;
    return qk;
}

template <typename T>
struct Float_Trait {
  using Type = T;
};

template <>
struct Float_Trait<c10::Half> {
  using Type = uint16_t;
};

template <>
struct Float_Trait<c10::BFloat16> {
  using Type = sycl::ext::oneapi::bfloat16;
};

namespace vllm {

// Q*K^T operation.
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline float qk_dot_(
    const Vec* q,
    const Vec* k,
    const sycl::nd_item<3>& item_ct1) {
  using A_vec = typename FloatVec<Vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    
    qk += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template <typename Vec, int N>
  static inline float dot(
      const Vec* q,
      const Vec* k,
      const sycl::nd_item<3>& item_ct1) {
    return qk_dot_<THREAD_GROUP_SIZE, Vec, N>(q, k, item_ct1);
  }
};

template <int NUM_WARPS>
inline float block_sum(
    float* red_smem,
    float sum,
    const sycl::nd_item<3>& item_ct1) {
  // Decompose the thread index into warp / lane.
  int warp = item_ct1.get_local_id(2) / WARP_SIZE;
  int lane = item_ct1.get_local_id(2) % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    
    /*
    DPCT1096:42: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sum += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    
    /*
    DPCT1096:43: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sum += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), sum, mask);
  }

  // Broadcast to other threads.
  
  /*
  DPCT1096:44: The right-most dimension of the work-group used in the SYCL
  kernel that calls this function may be less than "32". The function
  "dpct::select_from_sub_group" may return an unexpected result on the CPU
  device. Modify the size of the work-group to ensure that the value of the
  right-most dimension is a multiple of "32".
  */
  return dpct::select_from_sub_group(
        item_ct1.get_sub_group(), sum, 0);
}

template <typename scalar_t, int GS, int HD>
void context_attention_kernel_v1_reshaped(
    void* query, void* key, void* value, const void* block_tables,
    const float scale, const void* query_start_loc, const void* seq_lens,
    const void* context_lens, const int block_size,
    // const int x,  // x in kv_cache
    void* out,    // output
    const int block_table_stride_batch, const int block_table_stride_seq,
    const int query_stride_bs, const int query_stride_head,
    const int query_stride_dim, const int k_cache_stride_tokens,
    const int k_cache_stride_head, const int k_cache_stride_block_size,
    const int k_cache_stride_dim,
    const int v_cache_stride_tokens, const int v_cache_stride_head,
    const int v_cache_stride_block_size, const int v_cache_stride_dim,
    const int out_stride_tokens, const int out_stride_head,
    const int num_queries_per_kv, const int max_input_length,
    const int batch_size, const int num_heads) {
  static_assert(GS * HD * sizeof(scalar_t) * 2 < 64 * 1024);

  const size_t key_slm_offset = 0;
  const size_t value_slm_offset = GS * HD * sizeof(scalar_t);
  sycl::queue& queue = vllm::xpu::vllmGetQueue();

  // Get the maximum seq_lens
  sycl::range<3> global_size(batch_size, num_heads,
                             (max_input_length + GS - 1) / GS * GS);
  sycl::range<3> local_size(1, 1, GS);

  auto cgf = [&](sycl::handler& handle) {
    handle.parallel_for(
        sycl::nd_range<3>(global_size, local_size),
        [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
          slm_init<GS * HD * sizeof(scalar_t) * 2>();

          const size_t bsz_idx = item.get_global_id(0);
          const size_t head_idx = item.get_global_id(1);
          // Assuming we have 32 query head and 8 kv_heads. Then
          // num_queries_per_group should be 4 For head_idx 13, then
          // kv_head_idx = 13 / 4 = 3, which is correct
          const size_t kv_head_idx = head_idx / num_queries_per_kv;
          const int32_t seq_idx = item.get_global_id(2);
          const size_t gid = item.get_group(2);
          const size_t tid = item.get_local_id(2);

          // const int64_t * seq_len = (const int64_t *) seq_lens;
          const int32_t* seq_len = (const int32_t*)seq_lens;
          int32_t seq_bound = seq_len[bsz_idx];

          const int32_t* query_loc = (const int32_t*)query_start_loc;
          // There is a possibility that the current token index pass
          // over the seq_len, therefore: token_idx is the position in
          // the query
          int32_t token_idx =
              query_loc[bsz_idx] + std::min(seq_idx, seq_bound - 1);

          const int32_t* context_len_pointer = (const int32_t*)context_lens;

          const int* block_tables_ptr = (const int*)block_tables;
          const int* block_table =
              block_tables_ptr + bsz_idx * block_table_stride_batch;
          // I guess this context_len should be 0...
          const int32_t context_len = context_len_pointer[bsz_idx];

          // Position in the sequence
          // context + seq_idx
          // const int32_t token_position =
          //     context_len + std::min(seq_idx, seq_bound - 1);
          const int32_t token_position = context_len + seq_idx;

          const scalar_t* query_head = (const scalar_t*)query +
                                       token_idx * query_stride_bs +
                                       head_idx * query_stride_head;
          // Target output
          scalar_t* out_head =
              (scalar_t*)out +
              (query_loc[bsz_idx] + seq_idx) * out_stride_tokens +
              head_idx * out_stride_head;

          int32_t context_groups = context_len / GS;

          // Each token load its query_row
          simd<scalar_t, HD> query_row =
              block_load<scalar_t, HD>(query_head) * scale;
          simd<scalar_t, HD> accv = 0;
          simd<scalar_t, GS> softmaxv = 0;
          scalar_t max_attn = -sycl::detail::max_v<scalar_t>();

          // ################# Handle n * GS context part ######################
          int32_t n = context_len / GS;
          int32_t context_offset = context_len % GS;

          for (int32_t group = 0; group < n; ++group) {
            size_t target_key_position = group * GS + tid;
            int which_block = target_key_position / block_size;
            int which_slot = target_key_position % block_size;

            int physical_block_number = block_table[which_block];
            // Now key shape is [num_blocks, num_heads, block_size, head_dim]
            const scalar_t* key_head =
                (const scalar_t*)key +
                physical_block_number * k_cache_stride_tokens +
                kv_head_idx * k_cache_stride_head +
                which_slot * k_cache_stride_block_size;
            simd<scalar_t, HD> key_row = block_load<scalar_t, HD>(key_head);
            slm_block_store(key_slm_offset + tid * HD * sizeof(scalar_t), key_row);

            const scalar_t* value_head =
                (const scalar_t*)value +
                physical_block_number * v_cache_stride_tokens +
                kv_head_idx * v_cache_stride_head + which_slot * v_cache_stride_block_size;
            simd<scalar_t, HD> value_row = block_load<scalar_t, HD>(value_head);
            slm_block_store(value_slm_offset + tid * HD * sizeof(scalar_t),
                            value_row);
            barrier();

            // Calculate QK^T for this group...
            simd<scalar_t, GS> attnv;
#pragma unroll
            for (size_t r = 0; r < GS; ++r) {
              simd<scalar_t, HD> key_row = slm_block_load<scalar_t, HD>(
                  key_slm_offset + r * HD * sizeof(scalar_t));
              scalar_t attn =
                  sycl::ext::intel::esimd::detail::sum<scalar_t, scalar_t, HD>(
                      query_row * key_row);
              attnv[r] = attn;
            }
            scalar_t new_max_attn =
                std::max(hmax<scalar_t, scalar_t, GS>(attnv), max_attn);
            scalar_t attn_exp = exp(max_attn - new_max_attn);
            accv = accv * attn_exp;
            softmaxv = softmaxv * attn_exp;
            max_attn = new_max_attn;
            const simd<scalar_t, GS> attn_expv = exp(attnv - max_attn);
#pragma unorll
            for (size_t r = 0; r < GS; ++r) {
              simd<scalar_t, HD> value_row = slm_block_load<scalar_t, HD>(
                  value_slm_offset + r * HD * sizeof(scalar_t));
              accv += value_row * attn_expv[r];
            }
            softmaxv += attn_expv;
            barrier();
          }

          // ########## End for handling context n * GS part ###########

          // ########## Handle n * GS ################
          for (size_t group = 0; group < gid; ++group) {
            // 1. begins to load each position's key and value
            size_t target_key_position = context_len + group * GS + tid;
            int which_block = target_key_position / block_size;
            int which_slot = target_key_position % block_size;

            int physical_block_number = block_table[which_block];
            const scalar_t* key_head =
                (const scalar_t*)key +
                physical_block_number * k_cache_stride_tokens +
                kv_head_idx * k_cache_stride_head +
                which_slot * k_cache_stride_block_size;
            simd<scalar_t, HD> key_row = block_load<scalar_t, HD>(key_head);
            slm_block_store(key_slm_offset + tid * HD * sizeof(scalar_t),
                            key_row);
            const scalar_t* value_head =
                (const scalar_t*)value +
                physical_block_number * v_cache_stride_tokens +
                kv_head_idx * v_cache_stride_head + which_slot * v_cache_stride_block_size;
            simd<scalar_t, HD> value_row = block_load<scalar_t, HD>(value_head);
            slm_block_store(value_slm_offset + tid * HD * sizeof(scalar_t),
                            value_row);
            barrier();
            simd<scalar_t, GS> attnv;
#pragma unroll
            for (size_t r = 0; r < GS; ++r) {
              simd<scalar_t, HD> key_row = slm_block_load<scalar_t, HD>(
                  key_slm_offset + r * HD * sizeof(scalar_t));
              scalar_t attn =
                  sycl::ext::intel::esimd::detail::sum<scalar_t, scalar_t, HD>(
                      query_row * key_row);
              attnv[r] = attn;
            }

            scalar_t new_max_attn =
                std::max(hmax<scalar_t, scalar_t, GS>(attnv), max_attn);
            scalar_t attn_exp = exp(max_attn - new_max_attn);
            accv = accv * attn_exp;

            softmaxv = softmaxv * attn_exp;
            max_attn = new_max_attn;
            const simd<scalar_t, GS> attn_expv = exp(attnv - max_attn);
#pragma unroll
            for (size_t r = 0; r < GS; ++r) {
              simd<scalar_t, HD> value_row = slm_block_load<scalar_t, HD>(
                  value_slm_offset + r * HD * sizeof(scalar_t));
              accv += value_row * attn_expv[r];
            }
            softmaxv += attn_expv;
            barrier();
          }

          // ######### End of handle n * GS part ##########

          // ################ Handle offset part ####################
          scalar_t softmax =
              sycl::ext::intel::esimd::detail::sum<scalar_t, scalar_t, GS>(
                  softmaxv);

          // ########### handle context offset ############
          if (tid < context_offset) {
            size_t target_key_position = n * GS + tid;
            int which_block = target_key_position / block_size;
            int which_slot = target_key_position % block_size;

            int physical_block_number = block_table[which_block];
            const scalar_t* key_head =
                (const scalar_t*)key +
                physical_block_number * k_cache_stride_tokens +
                kv_head_idx * k_cache_stride_head +
                which_slot * k_cache_stride_block_size;
            simd<scalar_t, HD> key_row = block_load<scalar_t, HD>(key_head);
            slm_block_store(key_slm_offset + tid * HD * sizeof(scalar_t),
                            key_row);

            const scalar_t* value_head =
                (const scalar_t*)value +
                physical_block_number * v_cache_stride_tokens +
                kv_head_idx * v_cache_stride_head +
                which_slot * v_cache_stride_block_size;
            simd<scalar_t, HD> value_row = block_load<scalar_t, HD>(value_head);
            slm_block_store(value_slm_offset + tid * HD * sizeof(scalar_t),
                            value_row);
          }

          barrier();

          if (token_position < seq_bound) {
#pragma unroll
            for (size_t r = 0; r < context_offset; ++r) {
              simd<scalar_t, HD> key_row = slm_block_load<scalar_t, HD>(
                  key_slm_offset + r * HD * sizeof(scalar_t));
              simd<scalar_t, HD> value_row = slm_block_load<scalar_t, HD>(
                  value_slm_offset + r * HD * sizeof(scalar_t));
              scalar_t attn =
                  sycl::ext::intel::esimd::detail::sum<scalar_t, scalar_t, HD>(
                      query_row * key_row);
              if (attn <= max_attn) {
                scalar_t attn_exp =
                    sycl::ext::intel::esimd::exp(attn - max_attn);
                accv += value_row * attn_exp;
                softmax += attn_exp;
              } else {
                scalar_t attn_exp =
                    sycl::ext::intel::esimd::exp(max_attn - attn);
                accv = accv * attn_exp + value_row;
                softmax = softmax * attn_exp + 1;
                max_attn = attn;
              }
            }
          }
          barrier();

          // ############## handle seq offset #################
          if (token_position < seq_bound) {
            const int64_t which_block =
                static_cast<int64_t>(token_position / block_size);
            const int64_t which_slot =
                static_cast<int64_t>(token_position % block_size);

            const int64_t physical_block_number =
                static_cast<int64_t>(block_table[which_block]);

            const scalar_t* key_head =
                (const scalar_t*)key +
                physical_block_number * k_cache_stride_tokens +
                kv_head_idx * k_cache_stride_head +
                which_slot * k_cache_stride_block_size;
            simd<scalar_t, HD> key_row = block_load<scalar_t, HD>(key_head);
            slm_block_store(key_slm_offset + tid * HD * sizeof(scalar_t),
                            key_row);

            // [num_blocks, num_kv_heads, head_size, block_size]
            const scalar_t* value_head =
                (const scalar_t*)value +
                physical_block_number * v_cache_stride_tokens +
                kv_head_idx * v_cache_stride_head +
                which_slot * v_cache_stride_block_size;
            simd<scalar_t, HD> value_row = block_load<scalar_t, HD>(value_head);
            slm_block_store(value_slm_offset + tid * HD * sizeof(scalar_t),
                            value_row);
          }
          barrier();

          if (token_position < seq_bound) {
            for (size_t r = 0; r <= tid; ++r) {
              simd<scalar_t, HD> key_row = slm_block_load<scalar_t, HD>(
                  key_slm_offset + r * HD * sizeof(scalar_t));
              simd<scalar_t, HD> value_row = slm_block_load<scalar_t, HD>(
                  value_slm_offset + r * HD * sizeof(scalar_t));
              scalar_t attn =
                  sycl::ext::intel::esimd::detail::sum<scalar_t, scalar_t, HD>(
                      query_row * key_row);
              if (attn <= max_attn) {
                scalar_t attn_exp =
                    sycl::ext::intel::esimd::exp(attn - max_attn);
                accv += value_row * attn_exp;
                softmax += attn_exp;
              } else {
                scalar_t attn_exp =
                    sycl::ext::intel::esimd::exp(max_attn - attn);
                accv = accv * attn_exp + value_row;
                softmax = softmax * attn_exp + 1;
                max_attn = attn;
              }
            }

            if (softmax > 0) {
              simd<scalar_t, HD> result = accv / softmax;
              block_store(out_head, result);
            } else {
              simd<scalar_t, HD> result = 0;
              block_store(out_head, result);
            }
          }
          // ######## Ending of handling seq offset ##########
        });
  };
  queue.submit(cgf);
}

// How about implement a first edition that can be used with non-chunked
// prefill requests, so that we can make sure the reference for heads is
// correct
template <typename scalar_t, int GS, int HD>
void context_attention_kernel_v1(
    void* query, void* key, void* value, const void* block_tables,
    const float scale, const void* query_start_loc, const void* seq_lens,
    const void* context_lens, const int block_size,
    const int x,  // x in kv_cache
    void* out,    // output
    const int block_table_stride_batch, const int block_table_stride_seq,
    const int query_stride_bs, const int query_stride_head,
    const int query_stride_dim, const int k_cache_stride_tokens,
    const int k_cache_stride_head, const int k_cache_stride_dim,
    const int k_cache_stride_block_size, const int k_cache_stride_x,
    const int v_cache_stride_tokens, const int v_cache_stride_head,
    const int v_cache_stride_dim, const int v_cache_stride_block_size,
    const int out_stride_tokens, const int out_stride_head,
    const int num_queries_per_kv, const int max_input_length,
    const int batch_size, const int num_heads) {
  static_assert(GS * HD * sizeof(scalar_t) * 2 < 64 * 1024);

  const size_t key_slm_offset = 0;
  const size_t value_slm_offset = GS * HD * sizeof(scalar_t);
  sycl::queue& queue = vllm::xpu::vllmGetQueue();

  // Get the maximum seq_lens
  sycl::range<3> global_size(batch_size, num_heads,
                             (max_input_length + GS - 1) / GS * GS);
  sycl::range<3> local_size(1, 1, GS);

  auto cgf = [&](sycl::handler& handle) {
    handle.parallel_for(
        sycl::nd_range<3>(global_size, local_size),
        [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
          slm_init<GS * HD * sizeof(scalar_t) * 2>();

          const size_t bsz_idx = item.get_global_id(0);
          const size_t head_idx = item.get_global_id(1);
          // Assuming we have 32 query head and 8 kv_heads. Then
          // num_queries_per_group should be 4 For head_idx 13, then
          // kv_head_idx = 13 / 4 = 3, which is correct
          const size_t kv_head_idx = head_idx / num_queries_per_kv;
          const int32_t seq_idx = item.get_global_id(2);
          const size_t gid = item.get_group(2);
          const size_t tid = item.get_local_id(2);

          // const int64_t * seq_len = (const int64_t *) seq_lens;
          const int32_t* seq_len = (const int32_t*)seq_lens;
          int32_t seq_bound = seq_len[bsz_idx];

          const int32_t* query_loc = (const int32_t*)query_start_loc;
          // There is a possibility that the current token index pass
          // over the seq_len, therefore: token_idx is the position in
          // the query
          int32_t token_idx =
              query_loc[bsz_idx] + std::min(seq_idx, seq_bound - 1);

          const int32_t* context_len_pointer = (const int32_t*)context_lens;

          const int* block_tables_ptr = (const int*)block_tables;
          const int* block_table =
              block_tables_ptr + bsz_idx * block_table_stride_batch;
          // I guess this context_len should be 0...
          const int32_t context_len = context_len_pointer[bsz_idx];

          // Position in the sequence
          // context + seq_idx
          // const int32_t token_position =
          //     context_len + std::min(seq_idx, seq_bound - 1);
          const int32_t token_position = context_len + seq_idx;

          // static const CONSTANT char FMT[] =
          //     "Invoke target function...\n ";

          // sycl::ext::oneapi::experimental::printf(FMT);
          // static const CONSTANT char FMT[] =
          //     "GroupID = %6d bsz_idx = %6d seq_len = %6d seq_idx =
          //     %6d" "local_id = "
          //     "%6d "
          //     "token_idx = %6d "
          //     "context_len = %6d "
          //     "v_cache_stride_head_dim = %6d "
          //     "token_position = %6d\n";
          // sycl::ext::oneapi::experimental::printf(
          //     FMT, gid, bsz_idx, seq_bound, seq_idx, tid,
          //     token_idx, context_len, v_cache_stride_dim,
          //     token_position);

          const scalar_t* query_head = (const scalar_t*)query +
                                       token_idx * query_stride_bs +
                                       head_idx * query_stride_head;
          // Target output
          scalar_t* out_head =
              (scalar_t*)out +
              (query_loc[bsz_idx] + seq_idx) * out_stride_tokens +
              head_idx * out_stride_head;

          int32_t context_groups = context_len / GS;

          // Each token load its query_row
          simd<scalar_t, HD> query_row =
              block_load<scalar_t, HD>(query_head) * scale;
          simd<scalar_t, HD> accv = 0;
          simd<scalar_t, GS> softmaxv = 0;
          scalar_t max_attn = -sycl::detail::max_v<scalar_t>();

          // ################# Handle n * GS context part ######################
          int32_t n = context_len / GS;
          int32_t context_offset = context_len % GS;

          for (int32_t group = 0; group < n; ++group) {
            size_t target_key_position = group * GS + tid;
            int which_block = target_key_position / block_size;
            int which_slot = target_key_position % block_size;

            int physical_block_number = block_table[which_block];
            const scalar_t* key_head =
                (const scalar_t*)key +
                physical_block_number * k_cache_stride_tokens +
                kv_head_idx * k_cache_stride_head +
                which_slot * k_cache_stride_block_size;
            for (int i = 0; i < HD / x; i++) {
              // Load 8 elements, decided by x
              simd<scalar_t, 8> key_row =
                  block_load<scalar_t, 8>(key_head + i * k_cache_stride_dim);
              slm_block_store(key_slm_offset + tid * HD * sizeof(scalar_t) +
                                  8 * i * sizeof(scalar_t),
                              key_row);
            }

            const scalar_t* value_head =
                (const scalar_t*)value +
                physical_block_number * v_cache_stride_tokens +
                kv_head_idx * v_cache_stride_head + which_slot;
            for (int i = 0; i < HD; i++) {
              scalar_t temp_value = value_head[i * v_cache_stride_dim];
              slm_scalar_store<scalar_t>(value_slm_offset +
                                             tid * HD * sizeof(scalar_t) +
                                             i * sizeof(scalar_t),
                                         temp_value);
            }
            barrier();

            // Calculate QK^T for this group...
            simd<scalar_t, GS> attnv;
#pragma unroll
            for (size_t r = 0; r < GS; ++r) {
              simd<scalar_t, HD> key_row = slm_block_load<scalar_t, HD>(
                  key_slm_offset + r * HD * sizeof(scalar_t));
              scalar_t attn =
                  sycl::ext::intel::esimd::detail::sum<scalar_t, scalar_t, HD>(
                      query_row * key_row);
              attnv[r] = attn;
            }
            scalar_t new_max_attn =
                std::max(hmax<scalar_t, scalar_t, GS>(attnv), max_attn);
            scalar_t attn_exp = exp(max_attn - new_max_attn);
            accv = accv * attn_exp;
            softmaxv = softmaxv * attn_exp;
            max_attn = new_max_attn;
            const simd<scalar_t, GS> attn_expv = exp(attnv - max_attn);
#pragma unorll
            for (size_t r = 0; r < GS; ++r) {
              simd<scalar_t, HD> value_row = slm_block_load<scalar_t, HD>(
                  value_slm_offset + r * HD * sizeof(scalar_t));
              accv += value_row * attn_expv[r];
            }
            softmaxv += attn_expv;
            barrier();
          }

          // ########## End for handling context n * GS part ###########

          // ########## Handle n * GS ################
          for (size_t group = 0; group < gid; ++group) {
            // 1. begins to load each position's key and value
            size_t target_key_position = context_len + group * GS + tid;
            int which_block = target_key_position / block_size;
            int which_slot = target_key_position % block_size;

            int physical_block_number = block_table[which_block];
            const scalar_t* key_head =
                (const scalar_t*)key +
                physical_block_number * k_cache_stride_tokens +
                kv_head_idx * k_cache_stride_head +
                which_slot * k_cache_stride_block_size;
            for (int i = 0; i < HD / x; i++) {
              // Load 8 elements
              simd<scalar_t, 8> key_row =
                  block_load<scalar_t, 8>(key_head + i * k_cache_stride_dim);
              slm_block_store(key_slm_offset + tid * HD * sizeof(scalar_t) +
                                  8 * i * sizeof(scalar_t),
                              key_row);
            }

            const scalar_t* value_head =
                (const scalar_t*)value +
                physical_block_number * v_cache_stride_tokens +
                kv_head_idx * v_cache_stride_head + which_slot;
            for (int i = 0; i < HD; i++) {
              scalar_t temp_value = value_head[i * v_cache_stride_dim];
              slm_scalar_store<scalar_t>(value_slm_offset +
                                             tid * HD * sizeof(scalar_t) +
                                             i * sizeof(scalar_t),
                                         temp_value);
            }
            barrier();
            simd<scalar_t, GS> attnv;
#pragma unroll
            for (size_t r = 0; r < GS; ++r) {
              simd<scalar_t, HD> key_row = slm_block_load<scalar_t, HD>(
                  key_slm_offset + r * HD * sizeof(scalar_t));
              scalar_t attn =
                  sycl::ext::intel::esimd::detail::sum<scalar_t, scalar_t, HD>(
                      query_row * key_row);
              attnv[r] = attn;
            }

            scalar_t new_max_attn =
                std::max(hmax<scalar_t, scalar_t, GS>(attnv), max_attn);
            scalar_t attn_exp = exp(max_attn - new_max_attn);
            accv = accv * attn_exp;

            softmaxv = softmaxv * attn_exp;
            max_attn = new_max_attn;
            const simd<scalar_t, GS> attn_expv = exp(attnv - max_attn);
#pragma unroll
            for (size_t r = 0; r < GS; ++r) {
              simd<scalar_t, HD> value_row = slm_block_load<scalar_t, HD>(
                  value_slm_offset + r * HD * sizeof(scalar_t));
              accv += value_row * attn_expv[r];
            }
            softmaxv += attn_expv;
            barrier();
          }

          // ######### End of handle n * GS part ##########

          // ################ Handle offset part ####################
          scalar_t softmax =
              sycl::ext::intel::esimd::detail::sum<scalar_t, scalar_t, GS>(
                  softmaxv);

          // ########### handle context offset ############
          if (tid < context_offset) {
            size_t target_key_position = n * GS + tid;
            int which_block = target_key_position / block_size;
            int which_slot = target_key_position % block_size;

            int physical_block_number = block_table[which_block];
            const scalar_t* key_head =
                (const scalar_t*)key +
                physical_block_number * k_cache_stride_tokens +
                kv_head_idx * k_cache_stride_head +
                which_slot * k_cache_stride_block_size;
            for (int i = 0; i < HD / x; i++) {
              // Load 8 elements
              simd<scalar_t, 8> key_row =
                  block_load<scalar_t, 8>(key_head + i * k_cache_stride_dim);
              slm_block_store(key_slm_offset + tid * HD * sizeof(scalar_t) +
                                  8 * i * sizeof(scalar_t),
                              key_row);
            }

            const scalar_t* value_head =
                (const scalar_t*)value +
                physical_block_number * v_cache_stride_tokens +
                kv_head_idx * v_cache_stride_head + which_slot;
            for (int i = 0; i < HD; i++) {
              // Seems to have an error here
              scalar_t temp_value = value_head[i * v_cache_stride_dim];
              slm_scalar_store<scalar_t>(value_slm_offset +
                                             tid * HD * sizeof(scalar_t) +
                                             i * sizeof(scalar_t),
                                         temp_value);
            }
          }

          barrier();

          if (token_position < seq_bound) {
#pragma unroll
            for (size_t r = 0; r < context_offset; ++r) {
              simd<scalar_t, HD> key_row = slm_block_load<scalar_t, HD>(
                  key_slm_offset + r * HD * sizeof(scalar_t));
              simd<scalar_t, HD> value_row = slm_block_load<scalar_t, HD>(
                  value_slm_offset + r * HD * sizeof(scalar_t));
              scalar_t attn =
                  sycl::ext::intel::esimd::detail::sum<scalar_t, scalar_t, HD>(
                      query_row * key_row);
              if (attn <= max_attn) {
                scalar_t attn_exp =
                    sycl::ext::intel::esimd::exp(attn - max_attn);
                accv += value_row * attn_exp;
                softmax += attn_exp;
              } else {
                scalar_t attn_exp =
                    sycl::ext::intel::esimd::exp(max_attn - attn);
                accv = accv * attn_exp + value_row;
                softmax = softmax * attn_exp + 1;
                max_attn = attn;
              }
            }
          }
          barrier();

          // ############## handle seq offset #################
          if (token_position < seq_bound) {
            const int64_t which_block =
                static_cast<int64_t>(token_position / block_size);
            const int64_t which_slot =
                static_cast<int64_t>(token_position % block_size);

            const int64_t physical_block_number =
                static_cast<int64_t>(block_table[which_block]);

            const scalar_t* key_head =
                (const scalar_t*)key +
                physical_block_number * k_cache_stride_tokens +
                kv_head_idx * k_cache_stride_head +
                which_slot * k_cache_stride_block_size;

            for (int i = 0; i < HD / x; i++) {
              // Load 8 elements
              simd<scalar_t, 8> key_row =
                  block_load<scalar_t, 8>(key_head + i * k_cache_stride_dim);
              slm_block_store(key_slm_offset + tid * HD * sizeof(scalar_t) +
                                  8 * i * sizeof(scalar_t),
                              key_row);
            }

            // [num_blocks, num_kv_heads, head_size, block_size]
            const scalar_t* value_head =
                (const scalar_t*)value +
                physical_block_number * v_cache_stride_tokens +
                kv_head_idx * v_cache_stride_head + which_slot;
            for (int i = 0; i < HD; i++) {
              scalar_t temp_value = value_head[i * v_cache_stride_dim];
              slm_scalar_store<scalar_t>(value_slm_offset +
                                             tid * HD * sizeof(scalar_t) +
                                             i * sizeof(scalar_t),
                                         temp_value);
            }
          }
          barrier();

          if (token_position < seq_bound) {
            for (size_t r = 0; r <= tid; ++r) {
              simd<scalar_t, HD> key_row = slm_block_load<scalar_t, HD>(
                  key_slm_offset + r * HD * sizeof(scalar_t));
              simd<scalar_t, HD> value_row = slm_block_load<scalar_t, HD>(
                  value_slm_offset + r * HD * sizeof(scalar_t));
              scalar_t attn =
                  sycl::ext::intel::esimd::detail::sum<scalar_t, scalar_t, HD>(
                      query_row * key_row);
              if (attn <= max_attn) {
                scalar_t attn_exp =
                    sycl::ext::intel::esimd::exp(attn - max_attn);
                accv += value_row * attn_exp;
                softmax += attn_exp;
              } else {
                scalar_t attn_exp =
                    sycl::ext::intel::esimd::exp(max_attn - attn);
                accv = accv * attn_exp + value_row;
                softmax = softmax * attn_exp + 1;
                max_attn = attn;
              }
            }

            if (softmax > 0) {
              simd<scalar_t, HD> result = accv / softmax;
              block_store(out_head, result);
            } else {
              simd<scalar_t, HD> result = 0;
              block_store(out_head, result);
            }
          }
          // ######## Ending of handling seq offset ##########
        });
  };
  queue.submit(cgf);
}

template <typename T, int GS, int HD>
void context_attention_kernel_v2(
    void* query, void* key, void* value, const void* block_tables,
    const float scale, const void* query_start_loc, const void* seq_lens,
    const void* context_lens, const int block_size,
    const int x,  // x in kv_cache
    void* out,    // output
    const int block_table_stride_batch, const int block_table_stride_seq,
    const int query_stride_bs, const int query_stride_head,
    const int query_stride_dim, const int k_cache_stride_tokens,
    const int k_cache_stride_head, const int k_cache_stride_dim,
    const int k_cache_stride_block_size, const int k_cache_stride_x,
    const int v_cache_stride_tokens, const int v_cache_stride_head,
    const int v_cache_stride_dim, const int v_cache_stride_block_size,
    const int out_stride_tokens, const int out_stride_head,
    const int num_queries_per_kv, const int max_input_length,
    const int batch_size, const int num_heads, const int num_tokens,
    const int max_context_len) {
  constexpr int BLOCK_SIZE = 8;
  constexpr int NUM_THREADS = 128;
  // Each wrap handles one context block, therefore, each thread_group_size is
  // this.
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  // Each query, and key thread_group loads 16 bytes
  // Assume TGS=4 then 16 / 4 / sizeof(half) = 2
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using sycl_t = vllm::xpu::SyclTypeTrait<T>::Type;
  using Q_Vec = typename Vec<sycl_t, VEC_SIZE>::Type;

  // Assuming HD = 128, TGS = 2, then 128 / 2 / 2 = 32
  int num_vecs_per_thread = HD / THREAD_GROUP_SIZE / VEC_SIZE;
  sycl_t* out_p = reinterpret_cast<sycl_t*>(out);
  sycl_t* query_ptr = reinterpret_cast<sycl_t*>(query);
  sycl_t* key_cache_ptr = reinterpret_cast<sycl_t*>(key);
  sycl_t* value_cache_ptr = reinterpret_cast<sycl_t*>(value);
  const int* query_loc_ptr = reinterpret_cast<const int*>(query_start_loc);
  const int* block_tables_ptr = reinterpret_cast<const int*>(block_tables);
  const int* context_lens_ptr = reinterpret_cast<const int*>(context_lens);
  const int* seq_lens_ptr = reinterpret_cast<const int*>(seq_lens);

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len =
      DIVIDE_ROUND_UP(max_context_len + 1 + max_input_length, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * HD * sizeof(float);
  // Python-side check in
  // vllm.worker.worker._check_if_can_support_max_seq_len Keep that in
  // sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);
  // WARN: we have changed this...
  sycl::range<3> grid(batch_size, num_heads, max_input_length);
  // One work-group that is executing on the device
  sycl::range<3> block(1, 1, NUM_THREADS);
  sycl::queue& queue = vllm::xpu::vllmGetQueue();

  auto cgf = [&](sycl::handler& handle) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(shared_mem_size), handle);
    sycl::local_accessor<Q_Vec, 1> q_vecs_acc_ct1(
        sycl::range<1>(THREAD_GROUP_SIZE * num_vecs_per_thread), handle);
    sycl::local_accessor<float, 1> red_smem_acc_ct1(
        sycl::range<1>(2 * NUM_WARPS), handle);

    handle.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          const int bsz_idx = item_ct1.get_group(0);
          const int seq_idx = item_ct1.get_group(2);
          constexpr bool USE_PARTITIONING = false;
          int context_len = context_lens_ptr[bsz_idx] + seq_idx;
          const int seq_len = seq_lens_ptr[bsz_idx];
          uint8_t* dpct_local = dpct_local_acc_ct1.get_pointer();
          Q_Vec* q_vecs = q_vecs_acc_ct1.get_pointer();
          float* red_smem = red_smem_acc_ct1.get_pointer();

          // output_stream << "Original context_len: " <<
          // context_lens_ptr[bsz_idx] << sycl::endl; output_stream <<
          // "Batch_idx: " << bsz_idx << " Seq_idx: " << seq_idx
          //     << " Context_len: " << context_len << " Original context_len: "
          //     << context_lens_ptr[bsz_idx] << " Seq_len: " << seq_len
          //     << " Max input length: " << max_input_length
          //     << sycl::endl;
          if (context_len >= seq_len) {
            return;
          }

          context_len = context_len + 1;

          const int num_context_blocks =
              DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
          const int num_blocks_per_partition = num_context_blocks;

          const int start_block_idx = 0;
          const int end_block_idx =
              MIN(start_block_idx + num_context_blocks, num_context_blocks);

          const int num_blocks = end_block_idx - start_block_idx;
          const int start_token_idx = start_block_idx * BLOCK_SIZE;
          const int end_token_idx =
              MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
          const int num_tokens = end_token_idx - start_token_idx;
          constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
          constexpr int NUM_THREAD_GROUPS =
              NUM_THREADS /
              THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE
          constexpr int NUM_TOKENS_PER_THREAD_GROUP =
              DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
          constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
          const int thread_idx = item_ct1.get_local_id(2);
          const int warp_idx = thread_idx / WARP_SIZE;
          const int lane = thread_idx % WARP_SIZE;
          const int head_idx = item_ct1.get_group(1);
          const int num_heads = item_ct1.get_group_range(1);
          const int kv_head_idx = head_idx / num_queries_per_kv;
          // TODO: consider alibi_slope later
          constexpr int NUM_ELEMS_PER_THREAD = HD / THREAD_GROUP_SIZE;
          constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;
          const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
          const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;
          const sycl_t* q_ptr =
              query_ptr + (query_loc_ptr[bsz_idx] + seq_idx) * query_stride_bs +
              head_idx * HD;

#pragma unroll
          for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
               i += NUM_THREAD_GROUPS) {
            const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
            q_vecs[thread_group_offset * NUM_VECS_PER_THREAD + i] =
                *reinterpret_cast<const Q_Vec*>(q_ptr + vec_idx * VEC_SIZE);
          }
          // Loaded q_vecs
          item_ct1.barrier(sycl::access::fence_space::local_space);
          auto shared_mem = (char*)dpct_local;
          float* logits = reinterpret_cast<float*>(shared_mem);
          constexpr int x = 16 / sizeof(sycl_t);
          float qk_max = -FLT_MAX;
          const int* block_table =
              block_tables_ptr + bsz_idx * block_table_stride_batch;

          // Loading key
          for (int block_idx = start_block_idx + warp_idx;
               block_idx < end_block_idx; block_idx += NUM_WARPS) {
            const int64_t physical_block_number =
                static_cast<int64_t>(block_table[block_idx]);
            for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
              const int physical_block_offset =
                  (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
              const int token_idx =
                  block_idx * BLOCK_SIZE + physical_block_offset;

              Q_Vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
              for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
                const sycl_t* k_ptr =
                    key_cache_ptr +
                    physical_block_number * k_cache_stride_tokens +
                    kv_head_idx * k_cache_stride_head +
                    physical_block_offset * x;

                const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
                const int offset1 = (vec_idx * VEC_SIZE) / x;
                const int offset2 = (vec_idx * VEC_SIZE) % x;
                k_vecs[j] = *reinterpret_cast<const Q_Vec*>(
                    k_ptr + offset1 * BLOCK_SIZE * x + offset2);
              }

              // Compute dot product.
              // This includes a reduction across the threads in the
              // same thread group. Q_Vec_t
              // q_vec_[NUM_VECS_PER_THREAD] = q_vecs +
              // thread_group_offset * THREAD_GROUP_SIZE;
              float qk = scale *
                         Qk_dot<sycl_t, THREAD_GROUP_SIZE>::template dot<
                             Q_Vec, NUM_VECS_PER_THREAD>(
                             q_vecs + thread_group_offset * NUM_VECS_PER_THREAD,
                             k_vecs, item_ct1);

              if (thread_group_offset == 0) {
                // Store the partial reductions to shared memory.
                // NOTE(woosuk): It is required to zero out the
                // masked logits.
                const bool mask = token_idx > context_len;
                logits[token_idx - start_token_idx] = mask ? 0.f : qk;
                qk_max = mask ? qk_max : sycl::fmax(qk_max, qk);
              }
            }
          }
#pragma unroll
          for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
            /*
            DPCT1096:38: The right-most dimension of the work-group used
            in the SYCL kernel that calls this function may be less than
            "32". The function "dpct::permute_sub_group_by_xor" may
            return an unexpected result on the CPU device. Modify the
            size of the work-group to ensure that the value of the
            right-most dimension is a multiple of "32".
            */
            qk_max =
                sycl::fmax(qk_max, dpct::permute_sub_group_by_xor(
                                       item_ct1.get_sub_group(), qk_max, mask));
          }
          if (lane == 0) {
            red_smem[warp_idx] = qk_max;
          }
          item_ct1.barrier(sycl::access::fence_space::local_space);
          // TODO(woosuk): Refactor this part.
          // Get the max qk value for the sequence.
          qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
          for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
            /*
            DPCT1096:39: The right-most dimension of the work-group used
            in the SYCL kernel that calls this function may be less than
            "32". The function "dpct::permute_sub_group_by_xor" may
            return an unexpected result on the CPU device. Modify the
            size of the work-group to ensure that the value of the
            right-most dimension is a multiple of "32".
            */
            qk_max =
                sycl::fmax(qk_max, dpct::permute_sub_group_by_xor(
                                       item_ct1.get_sub_group(), qk_max, mask));
          }
          qk_max =
              dpct::select_from_sub_group(item_ct1.get_sub_group(), qk_max, 0);

          // Get the sum of the exp values.
          float exp_sum = 0.f;
          for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
            float val = sycl::exp(logits[i] - qk_max);
            logits[i] = val;
            exp_sum += val;
          }
          exp_sum =
              block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum, item_ct1);
          // Compute softmax.
          const float inv_sum = 1.f / (exp_sum + 1e-6f);
#pragma unroll
          for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
            logits[i] *= inv_sum;
          }

          item_ct1.barrier(sycl::access::fence_space::local_space);
          constexpr int V_VEC_SIZE = MIN(16 / sizeof(sycl_t), BLOCK_SIZE);
          using V_vec = typename Vec<sycl_t, V_VEC_SIZE>::Type;
          using L_vec = typename Vec<sycl_t, V_VEC_SIZE>::Type;
          using Float_L_vec = typename FloatVec<L_vec>::Type;
          constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
          constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
          constexpr int NUM_ROWS_PER_THREAD =
              DIVIDE_ROUND_UP(HD, NUM_ROWS_PER_ITER);
          // NOTE(woosuk): We use FP32 for the accumulator for better
          // accuracy.
          float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
          for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            accs[i] = 0.f;
          }

          sycl_t zero_value;
          zero(zero_value);
          for (int block_idx = start_block_idx + warp_idx;
               block_idx < end_block_idx; block_idx += NUM_WARPS) {
            // NOTE(woosuk): The block number is stored in int32.
            // However, we cast it to int64 because int32 can lead to
            // overflow when this variable is multiplied by large
            // numbers (e.g., kv_block_stride).
            const int64_t physical_block_number =
                static_cast<int64_t>(block_table[block_idx]);
            const int physical_block_offset =
                (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
            const int token_idx =
                block_idx * BLOCK_SIZE + physical_block_offset;
            L_vec logits_vec;
            vllm::from_float(
                logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx -
                                                            start_token_idx));

            const sycl_t* v_ptr =
                value_cache_ptr +
                physical_block_number * v_cache_stride_tokens +
                kv_head_idx * v_cache_stride_head;
#pragma unroll
            for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
              const int row_idx =
                  lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
              if (row_idx < HD) {
                const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
                V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
                if (block_idx == num_context_blocks - 1) {
                  // NOTE(woosuk): When v_vec contains the tokens
                  // that are out of the context, we should
                  // explicitly zero out the values since they may
                  // contain NaNs. See
                  // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
                  sycl_t* v_vec_ptr = reinterpret_cast<sycl_t*>(&v_vec);
#pragma unroll
                  for (int j = 0; j < V_VEC_SIZE; j++) {
                    v_vec_ptr[j] =
                        token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
                  }
                }
                accs[i] += vllm::dot(logits_vec, v_vec);
              }
            }
          }
      // Perform reduction within each warp.
#pragma unroll
          for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            float acc = accs[i];
#pragma unroll
            for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
              /*
              DPCT1096:41: The right-most dimension of the work-group
              used in the SYCL kernel that calls this function may be
              less than "32". The function
              "dpct::permute_sub_group_by_xor" may return an
              unexpected result on the CPU device. Modify the size of
              the work-group to ensure that the value of the
              right-most dimension is a multiple of "32".
              */
              acc += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(),
                                                    acc, mask);
            }
            accs[i] = acc;
          }

          // NOTE(woosuk): A barrier is required because the shared memory
          // space for logits is reused for the output.

          item_ct1.barrier(sycl::access::fence_space::local_space);

          // Perform reduction across warps.
          float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
          for (int i = NUM_WARPS; i > 1; i /= 2) {
            int mid = i / 2;
            // Upper warps write to shared memory.
            if (warp_idx >= mid && warp_idx < i) {
              float* dst = &out_smem[(warp_idx - mid) * HD];
#pragma unroll
              for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
                const int row_idx =
                    lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
                if (row_idx < HD && lane % NUM_V_VECS_PER_ROW == 0) {
                  dst[row_idx] = accs[i];
                }
              }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            // Lower warps update the output.
            if (warp_idx < mid) {
              const float* src = &out_smem[warp_idx * HD];
#pragma unroll
              for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
                const int row_idx =
                    lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
                if (row_idx < HD && lane % NUM_V_VECS_PER_ROW == 0) {
                  accs[i] += src[row_idx];
                }
              }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);
          }

          // Write the final output.
          if (warp_idx == 0) {
            sycl_t* out_ptr =
                out_p + (query_loc_ptr[bsz_idx] + seq_idx) * out_stride_tokens +
                head_idx * out_stride_head;

#pragma unroll
            for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
              const int row_idx =
                  lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
              if (row_idx < HD && lane % NUM_V_VECS_PER_ROW == 0) {
                vllm::from_float(*(out_ptr + row_idx), accs[i]);
              }
            }
          }
        });
    // Each thread_group handles one token
  };
  queue.submit(cgf);
}

template <
    typename scalar_t,
    typename Q_Vec_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS,
    int VEC_SIZE,
    int PARTITION_SIZE = 0> // Zero means no partitioning.
void paged_attention_kernel(
    float* __restrict__ exp_sums, // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out, // [num_seqs, num_heads, max_num_partitions,
                                // head_size]
    const scalar_t* __restrict__ q, // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, // [num_heads]
    const float scale,
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float attn_logit_softcapping,
    const sycl::nd_item<3>& item_ct1,
    uint8_t* dpct_local,
    Q_Vec_t* q_vecs,
    float* red_smem) {
  const int seq_idx = item_ct1.get_group(1);
  const int partition_idx = item_ct1.get_group(0);
  const int max_num_partitions = item_ct1.get_group_range(0);
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE
                                       // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = item_ct1.get_local_id(2);
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = item_ct1.get_group(2);
  const int num_heads = item_ct1.get_group_range(2);
  const int num_queries_per_kv = num_heads / num_kv_heads;

  const int kv_head_idx = head_idx / num_queries_per_kv;
  ;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.

  // constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)),
  // 1);

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;

#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset * NUM_VECS_PER_THREAD + i] =
        *reinterpret_cast<const Q_Vec_t*>(q_ptr + vec_idx * VEC_SIZE);
  }
  /*
  DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space); // TODO(naed90): possible speedup if this is replaced with
                      // a memory wall right before we use q_vecs

  // Memory planning.
  auto shared_mem = (char*)dpct_local;
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in
    // the group has 0, 4, 8, ... th vectors of the key, and the second thread
    // has 1, 5, 9, ... th vectors of the key, and so on.

    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

      Q_Vec_t k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const scalar_t* k_ptr = k_cache +
            physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;

        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const Q_Vec_t*>(
            k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      // Q_Vec_t q_vec_[NUM_VECS_PER_THREAD] = q_vecs + thread_group_offset *
      // THREAD_GROUP_SIZE;
      float qk = scale *
          Qk_dot<scalar_t, THREAD_GROUP_SIZE>::
              template dot<Q_Vec_t, NUM_VECS_PER_THREAD>(
                     q_vecs + thread_group_offset * NUM_VECS_PER_THREAD,
                     k_vecs,
                     item_ct1);
      // Add the ALiBi bias if slopes are given.
      qk +=
          (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      // Add the attn_logit_softcapp if given.
      if (attn_logit_softcapping != 0.0) {
          qk = attn_softcapping(qk, attn_logit_softcapping);
      }
      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : sycl::fmax(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
  
    /*
    DPCT1096:38: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    qk_max = sycl::fmax(
        qk_max,
        dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    
    /*
    DPCT1096:39: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    qk_max = sycl::fmax(
        qk_max,
        dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  
  /*
  DPCT1096:40: The right-most dimension of the work-group used in the SYCL
  kernel that calls this function may be less than "32". The function
  "dpct::select_from_sub_group" may return an unexpected result on the CPU
  device. Modify the size of the work-group to ensure that the value of the
  right-most dimension is a multiple of "32".
  */
  qk_max = dpct::select_from_sub_group(
          item_ct1.get_sub_group(), qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = sycl::exp(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum, item_ct1);

  // Compute softmax.
  const float inv_sum = 1.f / (exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
        seq_idx * num_heads * max_num_partitions +
        head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
        head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    vllm::from_float(
        logits_vec,
        *reinterpret_cast<Float_L_vec*>(logits + token_idx - start_token_idx));

    const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
        kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        if (block_idx == num_context_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] =
                token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += vllm::dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
     
      /*
      DPCT1096:41: The right-most dimension of the work-group used in the SYCL
      kernel that calls this function may be less than "32". The function
      "dpct::permute_sub_group_by_xor" may return an unexpected result on the
      CPU device. Modify the size of the work-group to ensure that the value of
      the right-most dimension is a multiple of "32".
      */
      acc += dpct::permute_sub_group_by_xor(
          item_ct1.get_sub_group(), acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out +
        seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        vllm::from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <
    typename scalar_t,
    typename Q_Vec_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS,
    int VEC_SIZE>
void paged_attention_v1_kernel(
    scalar_t* __restrict__ out, // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q, // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, // [num_heads]
    const float scale,
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float attn_logit_softcapping,
    const sycl::nd_item<3>& item_ct1,
    uint8_t* dpct_local,
    Q_Vec_t* q_vecs,
    float* red_smem) {
  paged_attention_kernel<
      scalar_t,
      Q_Vec_t,
      HEAD_SIZE,
      BLOCK_SIZE,
      NUM_THREADS,
      VEC_SIZE>(
      /* exp_sums */ nullptr,
      /* max_logits */ nullptr,
      out,
      q,
      k_cache,
      v_cache,
      num_kv_heads,
      scale,
      block_tables,
      context_lens,
      max_num_blocks_per_seq,
      alibi_slopes,
      q_stride,
      kv_block_stride,
      kv_head_stride,
      attn_logit_softcapping,
      item_ct1,
      dpct_local,
      q_vecs,
      red_smem);
}

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE)      \
  paged_attention_xpu_v1_impl<T, HEAD_SIZE, BLOCK_SIZE>::call( \
      out_ptr,                                                 \
      query_ptr,                                               \
      key_cache_ptr,                                           \
      value_cache_ptr,                                         \
      num_kv_heads,                                            \
      scale,                                                   \
      block_tables_ptr,                                        \
      context_lens_ptr,                                        \
      max_num_blocks_per_seq,                                  \
      alibi_slopes_ptr,                                        \
      q_stride,                                                \
      kv_block_stride,                                         \
      kv_head_stride,                                          \
      num_seqs,                                                \
      num_heads,                                               \
      num_blocks);

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
  event = queue.submit([&](sycl::handler& cgh) {                            \
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(                    \
        sycl::range<1>(shared_mem_size), cgh);                              \
    sycl::local_accessor<Q_Vec, 1> q_vecs_acc_ct1(                          \
        sycl::range<1>(THREAD_GROUP_SIZE * num_vecs_per_thread), cgh);      \
    sycl::local_accessor<float, 1> red_smem_acc_ct1(                        \
        sycl::range<1>(2 * NUM_WARPS), cgh);                                \
                                                                            \
    auto out_ptr_ct0 = out_ptr;                                             \
    auto query_ptr_ct1 = query_ptr;                                         \
    auto key_cache_ptr_ct2 = key_cache_ptr;                                 \
    auto value_cache_ptr_ct3 = value_cache_ptr;                             \
    auto scale_ct5 = scale;                                                 \
    auto block_tables_ptr_ct6 = block_tables_ptr;                           \
    auto context_lens_ptr_ct7 = context_lens_ptr;                           \
    auto max_num_blocks_per_seq_ct8 = max_num_blocks_per_seq;               \
    auto alibi_slopes_ptr_ct9 = alibi_slopes_ptr;                           \
    auto q_stride_ct10 = q_stride;                                          \
    auto kv_block_stride_ct11 = kv_block_stride;                            \
    auto kv_head_stride_ct12 = kv_head_stride;                              \
    auto attn_logit_softcapping_ct13 = attn_logit_softcapping;              \
                                                                            \
    cgh.parallel_for(                                                       \
        sycl::nd_range<3>(grid * block, block),                             \
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] { \
          paged_attention_v1_kernel<                                        \
              sycl_t,                                                       \
              Q_Vec,                                                        \
              HEAD_SIZE,                                                    \
              BLOCK_SIZE,                                                   \
              NUM_THREADS,                                                  \
              VEC_SIZE>(                                                    \
              out_ptr_ct0,                                                  \
              query_ptr_ct1,                                                \
              key_cache_ptr_ct2,                                            \
              value_cache_ptr_ct3,                                          \
              num_kv_heads,                                                 \
              scale_ct5,                                                    \
              block_tables_ptr_ct6,                                         \
              context_lens_ptr_ct7,                                         \
              max_num_blocks_per_seq_ct8,                                   \
              alibi_slopes_ptr_ct9,                                         \
              q_stride_ct10,                                                \
              kv_block_stride_ct11,                                         \
              kv_head_stride_ct12,                                          \
              attn_logit_softcapping_ct13,                                  \
              item_ct1,                                                     \
              dpct_local_acc_ct1.get_pointer(),                             \
              q_vecs_acc_ct1.get_pointer(),                                 \
              red_smem_acc_ct1.get_pointer());                              \
        });                                                                 \
  });

template <typename T, int BLOCK_SIZE, int NUM_THREADS = 512>
void paged_attention_xpu_v1_impl_launcher(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const float attn_logit_softcapping) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using sycl_t = vllm::xpu::SyclTypeTrait<T>::Type;
  using Q_Vec = typename Vec<sycl_t, VEC_SIZE>::Type;

  int num_vecs_per_thread = head_size / THREAD_GROUP_SIZE / VEC_SIZE;
  assert(head_size % THREAD_GROUP_SIZE == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes
      ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
      : nullptr;

  sycl_t* out_ptr = reinterpret_cast<sycl_t*>(out.data_ptr());
  sycl_t* query_ptr = reinterpret_cast<sycl_t*>(query.data_ptr());
  sycl_t* key_cache_ptr = reinterpret_cast<sycl_t*>(key_cache.data_ptr());
  sycl_t* value_cache_ptr = reinterpret_cast<sycl_t*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len =
      DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
  
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  sycl::range<3> grid(1, num_seqs, num_heads);
  sycl::range<3> block(1, 1, NUM_THREADS);
  sycl::queue& queue = vllm::xpu::vllmGetQueue();
  sycl::event event;

  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v1", event);
#endif
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v1", event);
#endif
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v1", event);
#endif
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v1", event);
#endif
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v1", event);
#endif
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v1", event);
#endif
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
  // queue.wait();
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                  \
  vllm::paged_attention_xpu_v1_impl_launcher<T, BLOCK_SIZE>( \
      out,                                                   \
      query,                                                 \
      key_cache,                                             \
      value_cache,                                           \
      num_kv_heads,                                          \
      scale,                                                 \
      block_tables,                                          \
      context_lens,                                          \
      max_context_len,                                       \
      alibi_slopes,                                          \
      attn_logit_softcapping);

#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)                        \
  switch (block_size) {                                           \
    case 8:                                                      \
      CALL_KERNEL_LAUNCHER(T, 8);                                \
      break;                                                      \
    case 16:                                                      \
      CALL_KERNEL_LAUNCHER(T, 16);                                \
      break;                                                      \
    case 32:                                                      \
      CALL_KERNEL_LAUNCHER(T, 32);                                \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

// Grid: (num_heads, num_seqs).
template <
    typename scalar_t,
    int HEAD_SIZE,
    int NUM_THREADS,
    int PARTITION_SIZE>
void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out, // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums, // [num_seqs, num_heads,
                                        // max_num_partitions]
    const float* __restrict__ max_logits, // [num_seqs, num_heads,
                                          // max_num_partitions]
    const scalar_t* __restrict__ tmp_out, // [num_seqs, num_heads,
                                          // max_num_partitions, head_size]
    const int* __restrict__ context_lens, // [num_seqs]
    const int max_num_partitions,
    const sycl::nd_item<3>& item_ct1,
    uint8_t* dpct_local,
    float* red_smem) {
  const int num_heads = item_ct1.get_group_range(2);
  const int head_idx = item_ct1.get_group(2);
  const int seq_idx = item_ct1.get_group(1);
  const int context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr = tmp_out +
        seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = item_ct1.get_local_id(2); i < HEAD_SIZE;
         i += item_ct1.get_local_range(2)) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = item_ct1.get_local_id(2) / WARP_SIZE;
  const int lane = item_ct1.get_local_id(2) % WARP_SIZE;

  // Size: 2 * num_partitions.
  auto shared_mem = (char*)dpct_local;
  // Workspace for reduction.

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits +
      seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = item_ct1.get_local_id(2); i < num_partitions;
       i += item_ct1.get_local_range(2)) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = sycl::fmax(max_logit, (float)l);
  }
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    
    /*
    DPCT1096:45: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    max_logit = sycl::fmax(
        max_logit,
        dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  
  item_ct1.barrier(sycl::access::fence_space::local_space);
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    
    /*
    DPCT1096:46: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    max_logit = sycl::fmax(
        max_logit,
        dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), max_logit, mask));
  }
  // Broadcast the max value to all threads.
  
  /*
  DPCT1096:47: The right-most dimension of the work-group used in the SYCL
  kernel that calls this function may be less than "32". The function
  "dpct::select_from_sub_group" may return an unexpected result on the CPU
  device. Modify the size of the work-group to ensure that the value of the
  right-most dimension is a multiple of "32".
  */
  max_logit = dpct::select_from_sub_group(
      item_ct1.get_sub_group(), max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums =
      reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums +
      seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = item_ct1.get_local_id(2); i < num_partitions;
       i += item_ct1.get_local_range(2)) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * sycl::exp(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  
  item_ct1.barrier(sycl::access::fence_space::local_space);
  global_exp_sum =
      block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum, item_ct1);
  const float inv_global_exp_sum = 1.0f / (global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr = tmp_out +
      seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr =
      out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = item_ct1.get_local_id(2); i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
          inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <
    typename scalar_t,
    typename Q_Vec_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS,
    int VEC_SIZE,
    int PARTITION_SIZE>
void paged_attention_v2_kernel(
    float* __restrict__ exp_sums, // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ tmp_out, // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    const scalar_t* __restrict__ q, // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, // [num_heads]
    const float scale,
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float attn_logit_softcapping,
    const sycl::nd_item<3>& item_ct1,
    uint8_t* dpct_local,
    Q_Vec_t* q_vecs,
    float* red_smem) {
  paged_attention_kernel<
      scalar_t,
      Q_Vec_t,
      HEAD_SIZE,
      BLOCK_SIZE,
      NUM_THREADS,
      VEC_SIZE,
      PARTITION_SIZE>(
      exp_sums,
      max_logits,
      tmp_out,
      q,
      k_cache,
      v_cache,
      num_kv_heads,
      scale,
      block_tables,
      context_lens,
      max_num_blocks_per_seq,
      alibi_slopes,
      q_stride,
      kv_block_stride,
      kv_head_stride,
      attn_logit_softcapping,
      item_ct1,
      dpct_local,
      q_vecs,
      red_smem);
}

#define LAUNCH_PAGED_ATTENTION_V2_FIRST_HALF(HEAD_SIZE)                     \
  event = queue.submit([&](sycl::handler& cgh) {                            \
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(                    \
        sycl::range<1>(shared_mem_size), cgh);                              \
    sycl::local_accessor<Q_Vec, 1> q_vecs_acc_ct1(                          \
        sycl::range<1>(THREAD_GROUP_SIZE * num_vecs_per_thread), cgh);      \
    sycl::local_accessor<float, 1> red_smem_acc_ct1(                        \
        sycl::range<1>(2 * NUM_WARPS), cgh);                                \
                                                                            \
    auto exp_sums_ptr_ct0 = exp_sums_ptr;                                   \
    auto max_logits_ptr_ct1 = max_logits_ptr;                               \
    auto tmp_out_ptr_ct2 = tmp_out_ptr;                                     \
    auto query_ptr_ct3 = query_ptr;                                         \
    auto key_cache_ptr_ct4 = key_cache_ptr;                                 \
    auto value_cache_ptr_ct5 = value_cache_ptr;                             \
    auto scale_ct7 = scale;                                                 \
    auto block_tables_ptr_ct8 = block_tables_ptr;                           \
    auto context_lens_ptr_ct9 = context_lens_ptr;                           \
    auto max_num_blocks_per_seq_ct10 = max_num_blocks_per_seq;              \
    auto alibi_slopes_ptr_ct11 = alibi_slopes_ptr;                          \
    auto q_stride_ct12 = q_stride;                                          \
    auto kv_block_stride_ct13 = kv_block_stride;                            \
    auto kv_head_stride_ct14 = kv_head_stride;                              \
    auto attn_logit_softcapping_ct15 = attn_logit_softcapping;              \
                                                                            \
    cgh.parallel_for(                                                       \
        sycl::nd_range<3>(grid * block, block),                             \
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] { \
          vllm::paged_attention_v2_kernel<                                  \
              sycl_t,                                                       \
              Q_Vec,                                                        \
              HEAD_SIZE,                                                    \
              BLOCK_SIZE,                                                   \
              NUM_THREADS,                                                  \
              VEC_SIZE,                                                     \
              PARTITION_SIZE>(                                              \
              exp_sums_ptr_ct0,                                             \
              max_logits_ptr_ct1,                                           \
              tmp_out_ptr_ct2,                                              \
              query_ptr_ct3,                                                \
              key_cache_ptr_ct4,                                            \
              value_cache_ptr_ct5,                                          \
              num_kv_heads,                                                 \
              scale_ct7,                                                    \
              block_tables_ptr_ct8,                                         \
              context_lens_ptr_ct9,                                         \
              max_num_blocks_per_seq_ct10,                                  \
              alibi_slopes_ptr_ct11,                                        \
              q_stride_ct12,                                                \
              kv_block_stride_ct13,                                         \
              kv_head_stride_ct14,                                          \
              attn_logit_softcapping_ct15,                                  \
              item_ct1,                                                     \
              dpct_local_acc_ct1.get_pointer(),                             \
              q_vecs_acc_ct1.get_pointer(),                                 \
              red_smem_acc_ct1.get_pointer());                              \
        });                                                                 \
  });

#define LAUNCH_PAGED_ATTENTION_V2_SECOND_HALF(HEAD_SIZE)                    \
  event2 = queue.submit([&](sycl::handler& cgh) {                           \
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(                    \
        sycl::range<1>(reduce_shared_mem_size), cgh);                       \
    sycl::local_accessor<float, 1> red_smem_acc_ct1(                        \
        sycl::range<1>(2 * NUM_WARPS), cgh);                                \
                                                                            \
    auto out_ptr_ct0 = out_ptr;                                             \
    auto exp_sums_ptr_ct1 = exp_sums_ptr;                                   \
    auto max_logits_ptr_ct2 = max_logits_ptr;                               \
    auto tmp_out_ptr_ct3 = tmp_out_ptr;                                     \
    auto context_lens_ptr_ct4 = context_lens_ptr;                           \
    auto max_num_partitions_ct5 = max_num_partitions;                       \
                                                                            \
    cgh.parallel_for(                                                       \
        sycl::nd_range<3>(reduce_grid * block, block),                      \
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] { \
          vllm::paged_attention_v2_reduce_kernel<                           \
              sycl_t,                                                       \
              HEAD_SIZE,                                                    \
              NUM_THREADS,                                                  \
              PARTITION_SIZE>(                                              \
              out_ptr_ct0,                                                  \
              exp_sums_ptr_ct1,                                             \
              max_logits_ptr_ct2,                                           \
              tmp_out_ptr_ct3,                                              \
              context_lens_ptr_ct4,                                         \
              max_num_partitions_ct5,                                       \
              item_ct1,                                                     \
              dpct_local_acc_ct1.get_pointer(),                             \
              red_smem_acc_ct1.get_pointer());                              \
        });                                                                 \
  });

template <
    typename T,
    int BLOCK_SIZE,
    int NUM_THREADS = 512,
    int PARTITION_SIZE = 512>
void paged_attention_v2_launcher(
    torch::Tensor& out,
    torch::Tensor& exp_sums,
    torch::Tensor& max_logits,
    torch::Tensor& tmp_out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const float attn_logit_softcapping) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % THREAD_GROUP_SIZE == 0);
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using sycl_t = vllm::xpu::SyclTypeTrait<T>::Type;
  using Q_Vec = typename Vec<sycl_t, VEC_SIZE>::Type;

  int num_vecs_per_thread = head_size / THREAD_GROUP_SIZE / VEC_SIZE;
  assert(head_size % THREAD_GROUP_SIZE == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes
      ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
      : nullptr;

  sycl_t* out_ptr = reinterpret_cast<sycl_t*>(out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  sycl_t* tmp_out_ptr = reinterpret_cast<sycl_t*>(tmp_out.data_ptr());
  sycl_t* query_ptr = reinterpret_cast<sycl_t*>(query.data_ptr());
  sycl_t* key_cache_ptr = reinterpret_cast<sycl_t*>(key_cache.data_ptr());
  sycl_t* value_cache_ptr = reinterpret_cast<sycl_t*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

  // For paged attention v2 kernel.
  sycl::range<3> grid(max_num_partitions, num_seqs, num_heads);
  int shared_mem_size = std::max(logits_size, outputs_size);
  // For paged attention v2 reduce kernel.
  sycl::range<3> reduce_grid(1, num_seqs, num_heads);
  
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  sycl::range<3> block(1, 1, NUM_THREADS);
  sycl::queue& queue = vllm::xpu::vllmGetQueue();
  sycl::event event;
  sycl::event event2;
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V2_FIRST_HALF(64);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event);
#endif
      LAUNCH_PAGED_ATTENTION_V2_SECOND_HALF(64);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event2);
#endif
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V2_FIRST_HALF(80);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event);
#endif
      LAUNCH_PAGED_ATTENTION_V2_SECOND_HALF(80);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event2);
#endif
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V2_FIRST_HALF(96);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event);
#endif
      LAUNCH_PAGED_ATTENTION_V2_SECOND_HALF(96);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event2);
#endif
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V2_FIRST_HALF(112);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event);
#endif
      LAUNCH_PAGED_ATTENTION_V2_SECOND_HALF(112);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event2);
#endif
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V2_FIRST_HALF(128);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event);
#endif
      LAUNCH_PAGED_ATTENTION_V2_SECOND_HALF(128);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event2);
#endif
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V2_FIRST_HALF(256);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event);
#endif
      LAUNCH_PAGED_ATTENTION_V2_SECOND_HALF(256);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // xpu::profiler_record(event_desc, event);  // Uncomment when needed
#else
    ::xpu::profiler_record("paged attn v2", event2);
#endif
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V2_LAUNCHER(T, BLOCK_SIZE)             \
  vllm::paged_attention_v2_launcher<T, BLOCK_SIZE>( \
      out,                                          \
      exp_sums,                                     \
      max_logits,                                   \
      tmp_out,                                      \
      query,                                        \
      key_cache,                                    \
      value_cache,                                  \
      num_kv_heads,                                 \
      scale,                                        \
      block_tables,                                 \
      context_lens,                                 \
      max_context_len,                              \
      alibi_slopes,                                 \
      attn_logit_softcapping);

#define CALL_V2_LAUNCHER_BLOCK_SIZE(T)                            \
  switch (block_size) {                                           \
    case 8:                                                       \
      CALL_V2_LAUNCHER(T, 8);                                     \
      break;                                                      \
    case 16:                                                      \
      CALL_V2_LAUNCHER(T, 16);                                    \
      break;                                                      \
    case 32:                                                      \
      CALL_V2_LAUNCHER(T, 32);                                    \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

} // namespace vllm

void paged_attention_v1(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    const float kv_scale,
    const float attn_logit_softcapping) {
  VLLM_XPU_DISPATCH_FLOATING_TYPES_FLOAT_ONLY(
      query.scalar_type(), "paged_attention_xpu_v1_impl", [&] {
        CALL_KERNEL_LAUNCHER_BLOCK_SIZE(scalar_t);
      });
}

void paged_attention_v2(
    torch::Tensor& out,
    torch::Tensor& exp_sums,
    torch::Tensor& max_logits,
    torch::Tensor& tmp_out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    const float kv_scale,
    const float attn_logit_softcapping) {
  VLLM_XPU_DISPATCH_FLOATING_TYPES_FLOAT_ONLY(
      query.scalar_type(), "paged_attention_xpu_v2_impl", [&] {
        CALL_V2_LAUNCHER_BLOCK_SIZE(scalar_t);
      });
}

torch::Tensor context_attention_forward_v2(
    torch::Tensor query,  // [num_tokens, num_kv_head, head_dim]
    torch::Tensor key,    // [num_tokens, num_kv_heads * head_size]
    torch::Tensor value,  // [num_tokens, num_kv_heads * head_size]
    torch::Tensor block_tables, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, torch::Tensor context_lens, int max_input_length,
    int max_context_length) {
  // Currently, only support fp16 here
  int64_t num_tokens = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t head_dim = query.size(2);
  int64_t batch_size = seq_lens.size(0);
  int num_kv_heads = value.size(1);

  int key_dimension = key.dim();
  auto output = at::empty({query.size(0), query.size(1), query.size(2)},
                          at::device(query.device()).dtype(query.dtype()));

  assert(key_dimension == 5);
  assert(query.scalar_type() == key.scalar_type() &&
         query.scalar_type() == value.scalar_type());
  assert(head_dim == 128);
  assert(query.scalar_type() == at::ScalarType::Half);

  int query_stride_token = query.stride(0);
  int query_stride_head = query.stride(1);
  int query_stride_dim = query.stride(2);
  const float attn_scale = 1 / std::sqrt((float)head_dim);

  assert(num_heads % num_kv_heads == 0);
  int num_queries_per_kv = num_heads / num_kv_heads;


  // key: num_blocks, num_kv_heads, head_size // x, num_blocks, x)
  // value: [num_blocks, num_kv_heads, head_size, block_dim]
  int block_size = value.size(3);
  // Currently, only block_size 16 is supported...
  assert(block_size == 16);
  int x = key.size(4);
  int block_table_stride_bsz = block_tables.stride(0);
  int block_table_stride_seq = block_tables.stride(1);
  int k_cache_stride_token = key.stride(0);
  int k_cache_stride_head = key.stride(1);
  int k_cache_stride_head_dim = key.stride(2);
  int k_cache_stride_block = key.stride(3);
  int k_cache_stride_x = key.stride(4);

  int v_cache_stride_token = value.stride(0);
  int v_cache_stride_head = value.stride(1);
  int v_cache_stride_head_dim = value.stride(2);
  int v_cache_stride_block = value.stride(3);
  switch(head_dim) {
    case 128:
      vllm::context_attention_kernel_v2<sycl::half, 32, 128>(
        query.data_ptr(), key.data_ptr(), value.data_ptr(),
        block_tables.data_ptr(), attn_scale, query_start_loc.data_ptr(),
        seq_lens.data_ptr(), context_lens.data_ptr(), block_size, x,
        output.data_ptr(), block_table_stride_bsz, block_table_stride_seq,
        query_stride_token, query_stride_head, query_stride_dim,
        k_cache_stride_token, k_cache_stride_head, k_cache_stride_head_dim,
        k_cache_stride_block, k_cache_stride_x, v_cache_stride_token,
        v_cache_stride_head, v_cache_stride_head_dim, v_cache_stride_block,
        output.stride(0), output.stride(1), num_queries_per_kv,
        max_input_length, batch_size, num_heads, query.size(0),
        max_context_length);
      break;
    case 64:
      vllm::context_attention_kernel_v2<sycl::half, 32, 64>(
        query.data_ptr(), key.data_ptr(), value.data_ptr(),
        block_tables.data_ptr(), attn_scale, query_start_loc.data_ptr(),
        seq_lens.data_ptr(), context_lens.data_ptr(), block_size, x,
        output.data_ptr(), block_table_stride_bsz, block_table_stride_seq,
        query_stride_token, query_stride_head, query_stride_dim,
        k_cache_stride_token, k_cache_stride_head, k_cache_stride_head_dim,
        k_cache_stride_block, k_cache_stride_x, v_cache_stride_token,
        v_cache_stride_head, v_cache_stride_head_dim, v_cache_stride_block,
        output.stride(0), output.stride(1), num_queries_per_kv,
        max_input_length, batch_size, num_heads, query.size(0),
        max_context_length);
      break;
    case 80:
      vllm::context_attention_kernel_v2<sycl::half, 32, 80>(
        query.data_ptr(), key.data_ptr(), value.data_ptr(),
        block_tables.data_ptr(), attn_scale, query_start_loc.data_ptr(),
        seq_lens.data_ptr(), context_lens.data_ptr(), block_size, x,
        output.data_ptr(), block_table_stride_bsz, block_table_stride_seq,
        query_stride_token, query_stride_head, query_stride_dim,
        k_cache_stride_token, k_cache_stride_head, k_cache_stride_head_dim,
        k_cache_stride_block, k_cache_stride_x, v_cache_stride_token,
        v_cache_stride_head, v_cache_stride_head_dim, v_cache_stride_block,
        output.stride(0), output.stride(1), num_queries_per_kv,
        max_input_length, batch_size, num_heads, query.size(0),
        max_context_length);
      break;
    case 96:
      vllm::context_attention_kernel_v2<sycl::half, 32, 96>(
        query.data_ptr(), key.data_ptr(), value.data_ptr(),
        block_tables.data_ptr(), attn_scale, query_start_loc.data_ptr(),
        seq_lens.data_ptr(), context_lens.data_ptr(), block_size, x,
        output.data_ptr(), block_table_stride_bsz, block_table_stride_seq,
        query_stride_token, query_stride_head, query_stride_dim,
        k_cache_stride_token, k_cache_stride_head, k_cache_stride_head_dim,
        k_cache_stride_block, k_cache_stride_x, v_cache_stride_token,
        v_cache_stride_head, v_cache_stride_head_dim, v_cache_stride_block,
        output.stride(0), output.stride(1), num_queries_per_kv,
        max_input_length, batch_size, num_heads, query.size(0),
        max_context_length);
      break;
    default: throw std::runtime_error("unsupported head_dim");
  }
    return output;
}

torch::Tensor context_attention_forward_v1(
    torch::Tensor query,  // [num_tokens, num_kv_head, head_dim]
    torch::Tensor key,    // [num_tokens, num_kv_heads * head_size]
    torch::Tensor value,  // [num_tokens, num_kv_heads * head_size]
    torch::Tensor block_tables, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, torch::Tensor context_lens, int max_input_length,
    int max_context_length) {
  // Currently, only support fp16
  int64_t num_tokens = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t head_dim = query.size(2);
  int64_t batch_size = seq_lens.size(0);
  int num_kv_heads = value.size(1);

  int key_dimension = key.dim();
  auto output = at::empty({query.size(0), query.size(1), query.size(2)},
                          at::device(query.device()).dtype(query.dtype()));

  // key should be in shape:
  // 1. [num_blocks, num_heads, block_size, head_dim]
  // 2. [num_blocks, num_heads, head_dim / x, block_size, x]
  assert(key_dimension == 4 or key_dimension == 5);
  assert(query.scalar_type() == key.scalar_type() &&
         query.scalar_type() == value.scalar_type());
  assert(query.scalar_type() == at::ScalarType::Half);

  int query_stride_token = query.stride(0);
  int query_stride_head = query.stride(1);
  int query_stride_dim = query.stride(2);
  const float attn_scale = 1 / std::sqrt((float)head_dim);

  assert(num_heads % num_kv_heads == 0);
  int num_queries_per_kv = num_heads / num_kv_heads;
  int block_table_stride_bsz = block_tables.stride(0);
  int block_table_stride_seq = block_tables.stride(1);
  if (key_dimension == 4) {
    // key/value: num_blocks, num_kv_heads, num_blocks, head_dim)
    int block_size = value.size(2);
    int k_cache_stride_0 = key.stride(0);
    int k_cache_stride_1 = key.stride(1);
    int k_cache_stride_2 = key.stride(2);
    int k_cache_stride_3 = key.stride(3);

    int v_cache_stride_0 = value.stride(0);
    int v_cache_stride_1 = value.stride(1);
    int v_cache_stride_2 = value.stride(2);
    int v_cache_stride_3 = value.stride(3);
    switch (head_dim) {
      case 128:
        vllm::context_attention_kernel_v1_reshaped<sycl::half, 32, 128>(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            block_tables.data_ptr(), attn_scale, query_start_loc.data_ptr(),
            seq_lens.data_ptr(), context_lens.data_ptr(), block_size,
            output.data_ptr(), block_table_stride_bsz, block_table_stride_seq,
            query_stride_token, query_stride_head, query_stride_dim,
            k_cache_stride_0, k_cache_stride_1, k_cache_stride_2,
            k_cache_stride_3, v_cache_stride_0, v_cache_stride_1,
            v_cache_stride_2, v_cache_stride_3, output.stride(0),
            output.stride(1), num_queries_per_kv, max_input_length, batch_size,
            num_heads);
        break;
      case 64:
        vllm::context_attention_kernel_v1_reshaped<sycl::half, 32, 64>(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            block_tables.data_ptr(), attn_scale, query_start_loc.data_ptr(),
            seq_lens.data_ptr(), context_lens.data_ptr(), block_size,
            output.data_ptr(), block_table_stride_bsz, block_table_stride_seq,
            query_stride_token, query_stride_head, query_stride_dim,
            k_cache_stride_0, k_cache_stride_1, k_cache_stride_2,
            k_cache_stride_3, v_cache_stride_0, v_cache_stride_1,
            v_cache_stride_2, v_cache_stride_3, output.stride(0),
            output.stride(1), num_queries_per_kv, max_input_length, batch_size,
            num_heads);
        break;
      default:
        throw std::runtime_error("unsupported head_dim");
    }
  } else {
    int x = key.size(4);
    int block_size = value.size(3);
    int k_cache_stride_token = key.stride(0);
    int k_cache_stride_head = key.stride(1);
    int k_cache_stride_head_dim = key.stride(2);
    int k_cache_stride_block = key.stride(3);
    int k_cache_stride_x = key.stride(4);

    int v_cache_stride_token = value.stride(0);
    int v_cache_stride_head = value.stride(1);
    int v_cache_stride_head_dim = value.stride(2);
    int v_cache_stride_block = value.stride(3);
    switch (head_dim) {
      case 128:
        vllm::context_attention_kernel_v1<sycl::half, 32, 128>(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            block_tables.data_ptr(), attn_scale, query_start_loc.data_ptr(),
            seq_lens.data_ptr(), context_lens.data_ptr(), block_size, x,
            output.data_ptr(), block_table_stride_bsz, block_table_stride_seq,
            query_stride_token, query_stride_head, query_stride_dim,
            k_cache_stride_token, k_cache_stride_head, k_cache_stride_head_dim,
            k_cache_stride_block, k_cache_stride_x, v_cache_stride_token,
            v_cache_stride_head, v_cache_stride_head_dim, v_cache_stride_block,
            output.stride(0), output.stride(1), num_queries_per_kv,
            max_input_length, batch_size, num_heads);
        break;
      case 64:
        vllm::context_attention_kernel_v1<sycl::half, 32, 64>(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            block_tables.data_ptr(), attn_scale, query_start_loc.data_ptr(),
            seq_lens.data_ptr(), context_lens.data_ptr(), block_size, x,
            output.data_ptr(), block_table_stride_bsz, block_table_stride_seq,
            query_stride_token, query_stride_head, query_stride_dim,
            k_cache_stride_token, k_cache_stride_head, k_cache_stride_head_dim,
            k_cache_stride_block, k_cache_stride_x, v_cache_stride_token,
            v_cache_stride_head, v_cache_stride_head_dim, v_cache_stride_block,
            output.stride(0), output.stride(1), num_queries_per_kv,
            max_input_length, batch_size, num_heads);
        break;
      default:
        throw std::runtime_error("unsupported head_dim");
    }
  }
  return output;
}

template<typename IT, const int VS, const int HD>
void gqa_1_kernel(
    const void * query, // [num_seqs, num_heads, head_size]
    const void * key,   // [num_blocks, num_kv_heads, head_size, block_size]
    const void * value, // [num_blocks, num_kv_heads, head_size, block_size]
    const void* block_tables, // [num_seqs, max_num_blocks_per_seq]
    const void* context_lens, // [num_seqs]
    void * o_a_s,
    void * o_accs,
    const int64_t query_bsz_stride,
    const int64_t query_head_stride,
    const int64_t kv_token_stride,
    const int64_t kv_head_stride,
    const int64_t kv_block_stride,
    const int64_t block_table_stride_batch,
    const int64_t o_a_s_bsz_stride,
    const int64_t o_a_s_head_stride,
    const int64_t o_accs_bsz_stride,
    const int64_t o_accs_head_stride,
    const float scale,
    const int block_size,
    const int bsz,
    const int num_heads,
    const int num_kv_heads,
    const int block_num,
    const at::Device & device
) {
    const int group_size = num_heads / num_kv_heads;
    const int sub_rows = VS / group_size;
    const int rem_rows = VS % group_size;

    const float attn_scale = scale;

    sycl::range<3> global_size(bsz, num_heads, block_num);
    sycl::range<3> local_size(1, group_size, 1);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<3>(global_size, local_size),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                slm_init<VS * HD * sizeof(IT)>();

                const int bsz_idx = item.get_global_id(0);
                const int head_idx = item.get_global_id(1);
                const int kv_head_idx = item.get_group(1);
                const int tid = item.get_local_id(1);
                const int vid = item.get_global_id(2);

                const IT * query_head = (const IT *)query + bsz_idx * query_bsz_stride
                                                          + head_idx * query_head_stride;
                
                IT * o_accs_head = (IT *)o_accs + bsz_idx * o_accs_bsz_stride
                                                + head_idx * o_accs_head_stride;
                float * o_a_s_head = (float *)o_a_s + bsz_idx * o_a_s_bsz_stride
                                                    + head_idx * o_a_s_head_stride;

                const int* block_tables_ptr = (const int*)block_tables;
                const int* block_table =
                    block_tables_ptr + bsz_idx * block_table_stride_batch;

                const int* context_lens_ptr = (const int*)context_lens;
                const int context_length = context_lens_ptr[bsz_idx];

                simd<IT, HD> query_row = block_load<IT, HD>(query_head) * attn_scale;

                // copy k_cache to slm
                int start_row = std::min(vid * VS + tid * sub_rows + std::min(tid, rem_rows), context_length);
                int end_row = std::min(start_row + sub_rows + (tid < rem_rows), context_length);
                for (int r = start_row; r < end_row; ++r) {
                    int which_block = r / block_size;
                    int which_slot = r % block_size;
                    int physical_block_number = block_table[which_block];

                    const IT * key_head = (const IT *)key + physical_block_number * kv_token_stride +
                      kv_head_idx * kv_head_stride +
                      which_slot * kv_block_stride;

                    simd<IT, HD> key_row = block_load<IT, HD>(key_head);
                    slm_block_store<IT, HD>((r - vid * VS) * HD * sizeof(IT), key_row);
                }
                barrier();

                simd<float, VS> attns = -sycl::detail::max_v<float>();
                int row_num = (vid + 1) * VS > context_length ? context_length % VS : VS;
                // q @ k
                for (int r = 0; r < row_num; ++r) {
                    simd<IT, HD> key_row = slm_block_load<IT, HD>(r * HD * sizeof(IT));
                    float attn = sycl::ext::intel::esimd::detail::sum<float, IT, HD>(query_row * key_row);
                    attns[r] = attn;
                }

                float max_attn = hmax<float, float, VS>(attns);
                const simd<IT, VS> attn_exp = exp(attns - max_attn);
                barrier();

                // copy v_cache to slm
                for (int r = start_row; r < end_row; ++r) {
                    int which_block = r / block_size;
                    int which_slot = r % block_size;
                    int physical_block_number = block_table[which_block];

                    const IT * value_head = (const IT *)value + physical_block_number * kv_token_stride +
                      kv_head_idx * kv_head_stride +
                      which_slot * kv_block_stride;

                    simd<IT, HD> value_row = block_load<IT, HD>(value_head);
                    slm_block_store<IT, HD>((r - vid * VS) * HD * sizeof(IT), value_row);
                }
                barrier();

                // attn @ v
                simd<IT, HD> accs = 0;
                for (int r = 0; r < row_num; ++r) {
                    simd<IT, HD> value_row = slm_block_load<IT, HD>(r * HD * sizeof(IT));
                    accs = accs + value_row * attn_exp[r];
                }

                float softmax = sycl::ext::intel::esimd::detail::sum<float, float, VS>(attn_exp);

                block_store<IT, HD>(o_accs_head + vid * HD, accs);
                block_store<float, 1>(o_a_s_head + vid * 2, max_attn);
                block_store<float, 1>(o_a_s_head + vid * 2 + 1, softmax);
            }
        );
    };

    utils::submit_kernel(cgf, device, "gqa kernel 1/2");
}

template<typename IT, const int GS, const int HD>
void gqa_2_kernel(
    void * o_a_s,
    void * o_accs,
    void * output,
    const int64_t o_a_s_bsz_stride,
    const int64_t o_a_s_head_stride,
    const int64_t o_accs_bsz_stride,
    const int64_t o_accs_head_stride,
    const int64_t output_bsz_stride,
    const int64_t output_head_stride,
    const int bsz,
    const int num_heads,
    const int row_block_num,
    const at::Device & device
) {
    constexpr int SUB_HD = 8;
    static_assert(HD % SUB_HD == 0);
    static_assert(HD / SUB_HD <= GS);

    const int sub_rows = row_block_num / GS;
    const int rem_rows = row_block_num % GS;

    constexpr int accs_slm_offset = 0;
    constexpr int attn_slm_offset = GS * HD * sizeof(float);
    constexpr int softmax_slm_offset = attn_slm_offset + GS * sizeof(float);

    sycl::range<3> global_size(bsz, num_heads, GS);
    sycl::range<3> local_size(1, 1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<3>(global_size, local_size),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                slm_init<GS * HD * sizeof(float) + GS * 2 * sizeof(float)>();

                const int bsz_idx = item.get_global_id(0);
                const int head_idx = item.get_global_id(1);
                const int tid = item.get_global_id(2);

                const float * o_a_s_head = (const float *)o_a_s + bsz_idx * o_a_s_bsz_stride
                                                                + head_idx * o_a_s_head_stride;
                const IT * o_accs_head = (const IT *)o_accs + bsz_idx * o_accs_bsz_stride
                                                            + head_idx * o_accs_head_stride;
                IT * output_head = (IT *)output + bsz_idx * output_bsz_stride
                                                + head_idx * output_head_stride;

                int start_row = std::min(tid * sub_rows + std::min(tid, rem_rows), row_block_num);
                int end_row = std::min(start_row + sub_rows + (tid < rem_rows), row_block_num);

                float max_attn = -sycl::detail::max_v<float>();
                float softmax = 0;
                simd<float, HD> accs = 0;
                for (int r = start_row; r < end_row; ++r) {
                    float sub_attn = o_a_s_head[2 * r];
                    float sub_softmax = o_a_s_head[2 * r + 1];
                    simd<float, HD> sub_accs = block_load<IT, HD>(o_accs_head + r * HD);
                    float new_max_attn = std::max(max_attn, sub_attn);
                    float exp1 = exp(max_attn - new_max_attn);
                    float exp2 = exp(sub_attn - new_max_attn);
                    accs = accs * exp1 + sub_accs * exp2;
                    softmax = softmax * exp1 + sub_softmax * exp2;
                    max_attn = new_max_attn;
                }

                slm_block_store<float, HD>(accs_slm_offset + tid * HD * sizeof(float), accs);
                slm_block_store<float, 1>(attn_slm_offset + tid * sizeof(float), max_attn);
                slm_block_store<float, 1>(softmax_slm_offset + tid * sizeof(float), softmax);
                barrier();

                if (tid < HD / SUB_HD) {
                    simd<float, GS> max_attns = slm_block_load<float, GS>(attn_slm_offset);
                    const simd<float, GS> scales = exp(max_attns - hmax<float, float, GS>(max_attns));
                    simd<float, GS> softmaxs = slm_block_load<float, GS>(softmax_slm_offset);
                    float softmax_sum = sycl::ext::intel::esimd::detail::sum<float, float, GS>(softmaxs * scales);

                    simd<float, SUB_HD> result = 0;
                    #pragma unroll
                    for (int r = 0; r < GS; ++r) {
                        simd<float, SUB_HD> sub_accs = slm_block_load<float, SUB_HD>(
                            accs_slm_offset + (r * HD + tid * SUB_HD) * sizeof(float)
                        );
                        result = result + sub_accs * scales[r];
                    }
                    result = result / softmax_sum;
                    block_store<IT, SUB_HD>(output_head + tid * SUB_HD, result);
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "gqa kernel 2/2");
}

using AT = at::ScalarType;
using fp16 = sycl::half;
template<const int VS, const int GS, const int HD>
auto dispatch_gqa_kernel(AT it) {
    switch (it) {
        case AT::Float: return std::make_tuple(gqa_1_kernel<float, VS, HD>, gqa_2_kernel<float, GS, HD>);
        case AT::Half: return std::make_tuple(gqa_1_kernel<fp16, VS, HD>, gqa_2_kernel<fp16, GS, HD>);
        default: throw std::runtime_error("unsupported dtype, only fp32 and fp16 are supported");
    }
}

void paged_attention_gqa(
    torch::Tensor output,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t bsz,
    int64_t num_heads,
    int64_t num_kv_heads,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int block_size,
    int64_t head_dim,
    int max_seq_len
) {
    constexpr int VS = 32;
    constexpr int GS = 32;

    const int row_block_num = (max_seq_len + VS - 1) / VS;
    auto o_a_s = torch::empty({bsz, num_heads, 1, row_block_num * 2},
                              torch::device(query.device()).dtype(torch::kFloat32));
    auto o_accs = torch::empty({bsz, num_heads, 1, row_block_num * head_dim},
                               torch::device(query.device()).dtype(query.dtype()));

    auto [func1, func2] = [&](){
        switch (head_dim) {
            case 128: return dispatch_gqa_kernel<VS, GS, 128>(query.scalar_type());
            case 96: return dispatch_gqa_kernel<VS, GS, 96>(query.scalar_type());
            case 80: return dispatch_gqa_kernel<VS, GS, 80>(query.scalar_type());
            case 64: return dispatch_gqa_kernel<VS, GS, 64>(query.scalar_type());
            default: throw std::runtime_error("unsupported head_dim, only 128, 96, 80 and 64 are supported");
        }
    }();

    func1(
        query.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(),
        block_tables.data_ptr(), context_lens.data_ptr(), o_a_s.data_ptr(), o_accs.data_ptr(),
        query.stride(0), query.stride(1), key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), block_tables.stride(0),
        o_a_s.stride(0), o_a_s.stride(1), o_accs.stride(0), o_accs.stride(1),
        scale, block_size, bsz, num_heads, num_kv_heads, row_block_num,
        query.device()
    );

    func2(
        o_a_s.data_ptr(), o_accs.data_ptr(), output.data_ptr(),
        o_a_s.stride(0), o_a_s.stride(1),
        o_accs.stride(0), o_accs.stride(1),
        output.stride(0), output.stride(1),
        bsz, num_heads, row_block_num,
        query.device()
    );
}
