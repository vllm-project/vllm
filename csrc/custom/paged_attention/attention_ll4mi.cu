//TODO: add license terms
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define MAX_PARTITIONS 64
#define WARP_SIZE 64

#define GCN_MFMA_INSTR1 __builtin_amdgcn_mfma_f32_16x16x4f32
#define GCN_MFMA_INSTR __builtin_amdgcn_mfma_f32_4x4x4f16

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
typedef float16x4 _Half4;
typedef struct _Half8 { _Half4 xy[2]; } _Half8;
////// Non temporal load stores ///////

#if 1

template <typename T>
__device__ __forceinline__ T load(T* addr) {
  return addr[0];
}

template <typename T>
__device__ __forceinline__ void store(T value, T* addr) {
  addr[0] = value;
}

#else

template <typename T>
__device__ __forceinline__ T load(const T* addr) {
  return __builtin_nontemporal_load(addr);
}

template <>
__device__ __forceinline__
float2 load (const float2* addr) {
  auto addr_alias { reinterpret_cast<const uint64_t *>(addr) };
  auto result = __builtin_nontemporal_load(addr_alias);
  auto ret = reinterpret_cast<float2 *>(&result);
  return ret[0];
}

template <>
__device__ __forceinline__
float4 load (const float4* addr) {
  auto addr_alias { reinterpret_cast<const uint64_t *>(addr) };
  auto result1 = __builtin_nontemporal_load(addr_alias);
  auto result2 = __builtin_nontemporal_load(addr_alias + 1);
  float4 ret{};
  auto ret_alias = reinterpret_cast<float2 *>(&result1);
  ret.x = ret_alias->x;
  ret.y = ret_alias->y;
  ret_alias = reinterpret_cast<float2 *>(&result2);
  ret.z = ret_alias->x;
  ret.w = ret_alias->y;
  return ret;
}

template <>
__device__ __forceinline__
__half load (const __half* addr) {
  auto addr_alias { reinterpret_cast<const uint16_t *>(addr) };
  auto result = __builtin_nontemporal_load(addr_alias);
  auto ret = reinterpret_cast<__half *>(&result);
  return ret[0];
}

template <>
__device__ __forceinline__
__half2 load (const __half2* addr) {
  auto addr_alias { reinterpret_cast<const uint32_t *>(addr) };
  auto result = __builtin_nontemporal_load(addr_alias);
  auto ret = reinterpret_cast<__half2 *>(&result);
  return ret[0];
}

template <>
__device__ __forceinline__
vllm::Half4_ load (const vllm::Half4_* addr) {
  auto addr_alias { reinterpret_cast<const uint64_t *>(addr) };
  auto result = __builtin_nontemporal_load(addr_alias);
  auto ret = reinterpret_cast<vllm::Half4_ *>(&result);
  return ret[0];
}

template <>
__device__ __forceinline__
vllm::Half8_ load (const vllm::Half8_* addr) {
  auto addr_alias { reinterpret_cast<const uint64_t *>(addr) };
  auto result1 = __builtin_nontemporal_load(addr_alias);
  auto result2 = __builtin_nontemporal_load(addr_alias + 1);
  vllm::Half8_ ret {};
  auto ret_alias = reinterpret_cast<vllm::Half4_ *>(&result1);
  ret.x = ret_alias->x;
  ret.y = ret_alias->y;
  ret_alias = reinterpret_cast<vllm::Half4_ *>(&result2);
  ret.z = ret_alias->x;
  ret.w = ret_alias->y;
  return ret;
}

//// Not using nontemporal stores for now
template <typename T>
__device__ __forceinline__ void store(T value, T* addr) {
  return __builtin_nontemporal_store(value, addr);
}

#endif

///////////////////////////////////////

//grid (num_seqs, num_partitions,num_heads/gqa_ratio)
//block (partition size)
template <typename scalar_t, int BLOCK_SIZE, int HEAD_SIZE, int NUM_THREADS, int GQA_RATIO>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_kernel(
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
  const int num_kv_heads,
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes, // [num_heads]
  const int q_stride,
  const int kv_block_stride,
  const int kv_head_stride,
  float* __restrict__ exp_sums,           // [num_seqs, num_heads, max_num_partitions]
  float* __restrict__ max_logits,         // [num_seqs, num_heads, max_num_partitions]
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, max_num_partitions, head_size]
  scalar_t* __restrict__ final_out,             // [num_seqs, num_heads, head_size]
#if 0
  scalar_t* __restrict__ qk_out,             // [num_heads, num_seqs, max_ctx_blocks,block_size]
#endif
  int max_ctx_blocks
  ) {
      constexpr int NWARPS = NUM_THREADS/WARP_SIZE;
      const int warpid = threadIdx.x / WARP_SIZE;
      const int laneid = threadIdx.x % WARP_SIZE;
      const int lane4id = laneid%4;

      const int seq_idx = blockIdx.x;
      const int partition_idx = blockIdx.y;
      const int partition_size = blockDim.x;
      const int max_num_partitions = gridDim.y;

      const int context_len = context_lens[seq_idx];
      const int partition_start_token_idx = partition_idx * partition_size;
      //exit if partition is out of context for seq
      if (partition_start_token_idx >= context_len) {
          return;
      }
      constexpr int QHLOOP = DIVIDE_ROUND_UP(GQA_RATIO,4); // each 4 lanes fetch 4 different qheads, total qheads =8, so qhloop is 2
      constexpr int GQA_RATIO4 = 4*QHLOOP;
      __shared__ float shared_qk_max[NWARPS][GQA_RATIO4+1];
      __shared__ float shared_exp_sum[NWARPS][GQA_RATIO4+1];
      _Half8 Qlocal[QHLOOP];
      constexpr int x = 16 / sizeof(scalar_t);
      constexpr int HELOOP = HEAD_SIZE/x;
      _Half8 Klocal[HELOOP];
      constexpr int VHLOOP = HEAD_SIZE/WARP_SIZE; //v head_size dimension is distributed across lanes
      constexpr int VTLOOP = 8; //16 separate 4xtokens across warp -> 16/2 8xtokens
      _Half8 Vlocal[VHLOOP][VTLOOP];
      floatx4 dout[QHLOOP];
      float qk_max[QHLOOP];
      #pragma unroll
      for (int h=0; h<QHLOOP; h++) {
        dout[h] = {0};
        qk_max[h] = -FLT_MAX;
      }

      const int wg_start_head_idx = blockIdx.z * GQA_RATIO;
      const int wg_start_kv_head_idx = blockIdx.z;

      const int warp_start_token_idx = partition_start_token_idx + warpid*WARP_SIZE;

      if (warp_start_token_idx  >= context_len) { //warp out of context
        #pragma unroll
        for(int h=0;h<GQA_RATIO4;h++) {
            shared_qk_max[warpid][h] = -FLT_MAX;
            shared_exp_sum[warpid][h] = 0.0f;
        }
      }
      else {//warp within context

      const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
      const int last_ctx_block = num_context_blocks - 1;

      const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

      const int local_token_idx = threadIdx.x;
      const int global_token_idx = partition_start_token_idx + local_token_idx;

      const int block_idx = (global_token_idx < context_len) ? global_token_idx / BLOCK_SIZE : last_ctx_block;

      const int physical_block_number = block_table[block_idx];

      //constexpr int HELOOP = HEAD_SIZE/8;
      //each 4 lanes fetch 8 helems, so warp fetches 8*16 = 128 helems


      const scalar_t* q_ptr = q + seq_idx*q_stride + wg_start_head_idx*HEAD_SIZE;
      const _Half8*  q_ptrh8 = reinterpret_cast<const _Half8 *>(q_ptr);
      const int qhead_elemh8 = laneid/4;
      #pragma unroll
      for (int h=0; h<QHLOOP-1; h++) {
          const int qhead_idx = h*4 + lane4id;
          Qlocal[h] = q_ptrh8[qhead_idx*HEAD_SIZE/8 + qhead_elemh8];
      }
      const int final_qhead_idx = 4*(QHLOOP-1) + lane4id;
      if (final_qhead_idx < GQA_RATIO) {
          Qlocal[QHLOOP-1] = q_ptrh8[final_qhead_idx*HEAD_SIZE/8 + qhead_elemh8];
      } else {
          Qlocal[QHLOOP-1].xy[0] = {0};
          Qlocal[QHLOOP-1].xy[1] = {0};
      }

      //const int kv_head_idx = 0;
      const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                                        + wg_start_kv_head_idx * kv_head_stride;
      const _Half8* k_ptrh8 = reinterpret_cast<const _Half8 *>(k_ptr);

      const int physical_block_offset = local_token_idx%BLOCK_SIZE; //since x=half8, physical_block_offset is already cast as _H8


      #pragma unroll
      for (int d=0;d<HELOOP;d++) {
       Klocal[d] = k_ptrh8[d*BLOCK_SIZE + physical_block_offset];
      }

      float alibi_slope[QHLOOP];
      if (alibi_slopes != nullptr) {
        #pragma unroll
        for (int h=0; h<QHLOOP; h++) {
            const int qhead_idx = h*4 + lane4id;
            alibi_slope[h] = (qhead_idx < GQA_RATIO) ? alibi_slopes[wg_start_head_idx + qhead_idx] : 0.f;
        }
      }

#if 1

      const scalar_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride;
      const _Half8* v_ptrh8 = reinterpret_cast<const _Half8*>(v_ptr);
      const int warp_start_block_idx = warp_start_token_idx/BLOCK_SIZE;
      //iterate over each v block
      #pragma unroll
      for (int b=0;b<8*VTLOOP/BLOCK_SIZE;b++) {
	    const int vblock_idx = warp_start_block_idx + b;
        const int vblock_idx_ctx = (vblock_idx <= last_ctx_block) ? vblock_idx : last_ctx_block;
        const int vphysical_block_number = block_table[vblock_idx_ctx];
      	const _Half8* v_ptrh8b = v_ptrh8 + (vphysical_block_number * kv_block_stride)/8;
        //iterate over each head elem (within head_size)
        #pragma unroll
        for (int h=0;h<VHLOOP;h++) {
            const int head_size_elem = h*WARP_SIZE + laneid;
            const _Half8* v_ptrh8be = v_ptrh8b + head_size_elem*BLOCK_SIZE/8;
            //iterate over all velems within block
            #pragma unroll
            for (int d=0;d<BLOCK_SIZE/8;d++) {
                Vlocal[h][b*BLOCK_SIZE/8+d] = v_ptrh8be[d];
            }
        }
      }
#endif

      //dout[0] = {0};
      //dout[1] = {0};

        #pragma unroll
        for (int h=0;h<QHLOOP;h++) {
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[0].xy[0], dout[h], 4, 0, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[0].xy[1], dout[h], 4, 0, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[1].xy[0], dout[h], 4, 1, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[1].xy[1], dout[h], 4, 1, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[2].xy[0], dout[h], 4, 2, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[2].xy[1], dout[h], 4, 2, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[3].xy[0], dout[h], 4, 3, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[3].xy[1], dout[h], 4, 3, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[4].xy[0], dout[h], 4, 4, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[4].xy[1], dout[h], 4, 4, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[5].xy[0], dout[h], 4, 5, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[5].xy[1], dout[h], 4, 5, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[6].xy[0], dout[h], 4, 6, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[6].xy[1], dout[h], 4, 6, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[7].xy[0], dout[h], 4, 7, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[7].xy[1], dout[h], 4, 7, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[8].xy[0], dout[h], 4, 8, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[8].xy[1], dout[h], 4, 8, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[9].xy[0], dout[h], 4, 9, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[9].xy[1], dout[h], 4, 9, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[10].xy[0], dout[h], 4, 10, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[10].xy[1], dout[h], 4, 10, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[11].xy[0], dout[h], 4, 11, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[11].xy[1], dout[h], 4, 11, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[12].xy[0], dout[h], 4, 12, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[12].xy[1], dout[h], 4, 12, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[13].xy[0], dout[h], 4, 13, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[13].xy[1], dout[h], 4, 13, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[14].xy[0], dout[h], 4, 14, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[14].xy[1], dout[h], 4, 14, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[0], Klocal[15].xy[0], dout[h], 4, 15, 0);
          dout[h] = GCN_MFMA_INSTR(Qlocal[h].xy[1], Klocal[15].xy[1], dout[h], 4, 15, 0);
          dout[h]*=scale;
        }
        //transpose dout so that 4 token ids are in each lane, and 4 heads are across 4 lanes
        #pragma unroll
        for (int h=0;h<QHLOOP;h++) {
            floatx4 tmp={0};
            #pragma unroll
            for(int i=0; i<4; i++) {
                const float B = (lane4id==i)? 1.0f : 0.0f;
                //const float A = (global_token_idx < context_len) ? dout[h][i] : 0.0f;
                tmp = __builtin_amdgcn_mfma_f32_4x4x1f32(dout[h][i], B, tmp, 0, 0, 0);
                //tmp = __builtin_amdgcn_mfma_f32_4x4x1f32(A, B, tmp, 0, 0, 0);
            }
            dout[h] = tmp;
        }

        const int lane4_token_idx = 4*(global_token_idx>>2);
        const int alibi_offset = lane4_token_idx - context_len + 1;
        if (alibi_slopes != nullptr) {
          #pragma unroll
          for (int h=0;h<QHLOOP;h++) {
              #pragma unroll
              for(int i=0; i<4; i++) {
                  dout[h][i] += alibi_slope[h] * (alibi_offset + i);
              }
          }
        }

        #pragma unroll
        for (int h=0;h<QHLOOP;h++) {
            qk_max[h] = -FLT_MAX;
            #pragma unroll
            for(int i=0; i<4; i++) {
                qk_max[h] = (lane4_token_idx+i < context_len) ? fmaxf(qk_max[h], dout[h][i]) : qk_max[h];
            }
            #pragma unroll
            for(int mask=WARP_SIZE/2; mask>=4; mask/=2) {
                qk_max[h] = fmaxf(qk_max[h], __shfl_xor(qk_max[h],mask));
            }
        }

        float exp_sum[QHLOOP];
        #pragma unroll
        for (int h=0;h<QHLOOP;h++) {
            exp_sum[h] = 0.0f;
            #pragma unroll
            for(int i=0; i<4; i++) {
                dout[h][i] = (lane4_token_idx+i < context_len) ? __expf(dout[h][i] - qk_max[h]) : 0.0f;
                exp_sum[h] += dout[h][i];
            }
            #pragma unroll
            for(int mask=WARP_SIZE/2; mask>=4; mask/=2) {
                exp_sum[h] += __shfl_xor(exp_sum[h],mask);
            }
        }


        #pragma unroll
        for (int h=0;h<QHLOOP;h++) {
            const int head_idx = 4*h+lane4id;
            shared_qk_max[warpid][head_idx] = qk_max[h];
            shared_exp_sum[warpid][head_idx] = exp_sum[h];
        }
    }//warp within context

        __syncthreads();

    //float global_qk_max[QHLOOP];
    //float global_inv_sum_scale[QHLOOP];
    //float global_exp_scale[QHLOOP];
    const int num_heads = gridDim.z*GQA_RATIO;
    float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                       + partition_idx;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                   + partition_idx;
    #pragma unroll
    for (int h=0;h<QHLOOP;h++) {
        float global_qk_max = -FLT_MAX;
        float warp_qk_max[NWARPS];
        const int head_idx = 4*h+lane4id;
        #pragma unroll
        for (int w=0; w<NWARPS; w++) {
            warp_qk_max[w] = shared_qk_max[w][head_idx];
            global_qk_max = fmaxf(global_qk_max,warp_qk_max[w]);
        }
        //global_exp_scale[h] = __expf(qk_max[h] - global_qk_max);
        float global_exp_sum = 0.0f;
        #pragma unroll
        for (int w=0; w<NWARPS; w++) {
            global_exp_sum += shared_exp_sum[w][head_idx] * __expf(warp_qk_max[w] - global_qk_max);
        }
        if (head_idx < GQA_RATIO) {
          max_logits_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] = global_qk_max;
          exp_sums_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] = global_exp_sum;
        }
        //global_inv_sum[h] = 1.0f/global_exp_sum;
        const float global_inv_sum_scale = __fdividef(1.f, global_exp_sum + 1e-6f) * __expf(qk_max[h] - global_qk_max);
        dout[h] *= global_inv_sum_scale;
    }
    //logits[h] -> every 4 lanes hold 4 heads, each lane holds 4 tokens, there are 4x16 tokens across warp
    float16x4 logits[QHLOOP];
    #pragma unroll
    for (int h=0;h<QHLOOP;h++) {
        #pragma unroll
        for (int i=0;i<4;i++) {
            logits[h][i] = (scalar_t) dout[h][i];
        }
    }

    //float16x4 vout[QHLOOP][VHLOOP];
    __shared__ float16x4 vout_shared[QHLOOP][VHLOOP][WARP_SIZE][NWARPS+1];

    if (warp_start_token_idx  >= context_len) { //warp out of context
    #pragma unroll
    for (int qh=0; qh<QHLOOP; qh++) {
        #pragma unroll
        for (int vh=0; vh<VHLOOP; vh++) {
            vout_shared[qh][vh][laneid][warpid] = {0};
        }
    }
    }
    else{//warp in context
    //iterate across heads
    #pragma unroll
    for (int qh=0; qh<QHLOOP; qh++) {
        //iterate over each v head elem (within head_size)
        #pragma unroll
        for (int vh=0; vh<VHLOOP; vh++) {
            floatx4 acc = {0};
            //iterate over tokens
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][0].xy[0], acc, 4, 0, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][0].xy[1], acc, 4, 1, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][1].xy[0], acc, 4, 2, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][1].xy[1], acc, 4, 3, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][2].xy[0], acc, 4, 4, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][2].xy[1], acc, 4, 5, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][3].xy[0], acc, 4, 6, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][3].xy[1], acc, 4, 7, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][4].xy[0], acc, 4, 8, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][4].xy[1], acc, 4, 9, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][5].xy[0], acc, 4, 10, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][5].xy[1], acc, 4, 11, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][6].xy[0], acc, 4, 12, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][6].xy[1], acc, 4, 13, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][7].xy[0], acc, 4, 14, 0);
            acc = GCN_MFMA_INSTR(logits[qh], Vlocal[vh][7].xy[1], acc, 4, 15, 0);
            float16x4 tmp;
            #pragma unroll
            for(int i=0; i<4; i++) {
                tmp[i] = (scalar_t) acc[i];
            }
            vout_shared[qh][vh][laneid][warpid] = tmp;
        }
    }
    }//warp in context

    __syncthreads();

    if (warpid==0) {
        float16x4 vout[QHLOOP][VHLOOP];
        //iterate across heads
        //scalar_t* out_ptr = out + seq_idx*num_heads*HEAD_SIZE;
        scalar_t* out_ptr;
        int out_num_partitions;
        if (context_len > partition_size) {
            out_num_partitions = max_num_partitions;
            out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                            + partition_idx * HEAD_SIZE;
        } else {
            out_num_partitions = 1;
            out_ptr = final_out + seq_idx * num_heads * HEAD_SIZE;
        }
        #pragma unroll
        for (int qh=0; qh<QHLOOP; qh++) {
            //iterate over each v head elem (within head_size)
            #pragma unroll
            for (int vh=0; vh<VHLOOP; vh++) {
                vout[qh][vh] = {0};
                #pragma unroll
                for (int w=0; w<NWARPS; w++) {
                    vout[qh][vh] += vout_shared[qh][vh][laneid][w];
                }
                const int head_size_elem = vh*WARP_SIZE + laneid;
                #pragma unroll
                for (int i=0; i<4; i++) {
                    const int head_idx = 4*qh + i;
                    //out_ptr[head_idx*HEAD_SIZE + head_size_elem] = vout[qh][vh][i];
                    if (head_idx < GQA_RATIO) {
                      //out_ptr[(wg_start_head_idx + head_idx) * max_num_partitions * HEAD_SIZE + head_size_elem] = vout[qh][vh][i];
                      out_ptr[(wg_start_head_idx + head_idx) * out_num_partitions * HEAD_SIZE + head_size_elem] = vout[qh][vh][i];
                    }
                }
            }
        }
    }

#if 0
    const int num_seqs = gridDim.x;
    const int global_token4id = global_token_idx/4;
    #pragma unroll
    for (int t=0;t<4;t++) {
        #pragma unroll
        for (int h=0;h<QHLOOP;h++) {
          //const int head_idx = h*4 + t;
          const int head_idx = h*4 + lane4id;
	      //qk_out[head_idx*num_seqs*max_ctx_blocks*BLOCK_SIZE + seq_idx*max_ctx_blocks*BLOCK_SIZE + global_token_idx] = (scalar_t)dout[h][t];
	       qk_out[head_idx*num_seqs*max_ctx_blocks*BLOCK_SIZE + seq_idx*max_ctx_blocks*BLOCK_SIZE + 4*global_token4id + t] = logits[h][t];
	      //qk_out[head_idx*num_seqs*max_ctx_blocks*BLOCK_SIZE + seq_idx*max_ctx_blocks*BLOCK_SIZE + 4*global_token4id + t] = vout[h][t%2][t];
        }
    }
#endif

  }

// Grid: (num_heads, num_seqs).
template<
  typename scalar_t,
  int HEAD_SIZE,
  int NUM_THREADS,
  int PARTITION_SIZE>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const float* __restrict__ exp_sums,     // [num_seqs, num_heads, max_num_partitions]
  const float* __restrict__ max_logits,   // [num_seqs, num_heads, max_num_partitions]
  const scalar_t* __restrict__ tmp_out,   // [num_seqs, num_heads, max_num_partitions, head_size]
  const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    //scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    //const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
    //                                      + head_idx * max_num_partitions * HEAD_SIZE;
    //for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
    //  out_ptr[i] = tmp_out_ptr[i];
    //}
    // Terminate the thread block.
    //if num_partitions==1, main kernel will write to out directly, no work in reduction kernel
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];
  __shared__ float shared_global_exp_sum;
  //float reg_max_logits[MAX_PARTITIONS]; //dependent on max_num_partitions: assume 32K max context div 1K Partition size -> TODO: make this proper template param
  //Assume code below is optimized for MAX_PARTITIONS<=64 TODO: handle larger than warp size cases later
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  float* shared_exp_sums = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  //scalar_t tmp_outs[MAX_PARTITIONS];

  // Load max logits to shared memory.
  const float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                           + head_idx * max_num_partitions;
  ////float max_logit = -FLT_MAX;
  //for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
  ////for (int i = threadIdx.x; i < MAX_PARTITIONS; i += blockDim.x) {
    //const float l = max_logits_ptr[i];
    //shared_max_logits[i] = l;
    ////reg_max_logits[i] =  max_logits_ptr[i];  //TODO: review this -> right now num_partitions is very small <=32
    //max_logit = fmaxf(max_logit, l);
    ////max_logit = fmaxf(max_logit, reg_max_logits[i]);
  ////}
  //__syncthreads();
  float max_logit = (threadIdx.x < num_partitions) ? max_logits_ptr[threadIdx.x]:-FLT_MAX;
  float reg_max_logit =  max_logit;

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
  }
//  if (lane == 0) {
//    red_smem[warp_idx] = max_logit;
//  }

//  if (num_partitions >= WARP_SIZE) {
//  __syncthreads();
//  // Reduce across warps.
//  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
//#pragma unroll
//  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
//    max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
//  }
//  // Broadcast the max value to all threads.
//  //max_logit = __shfl(max_logit, 0);
//  }
  // Load rescaled exp sums to shared memory.
  const float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;

  //for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
  //  //float l = shared_max_logits[i];
  //  //float l = reg_max_logits[i];
  //  float rescaled_exp_sum = exp_sums_ptr[i] * expf(reg_max_logits[i] - max_logit);
  //  global_exp_sum += rescaled_exp_sum;
  //  shared_exp_sums[i] = rescaled_exp_sum;
  //}
  float rescaled_exp_sum = (threadIdx.x < num_partitions) ? exp_sums_ptr[threadIdx.x] * expf(reg_max_logit - max_logit) : 0.0f;
  global_exp_sum += rescaled_exp_sum;
  //if (threadIdx.x < num_partitions) {
    //shared_exp_sums[threadIdx.x] = (threadIdx.x < num_partitions) ? rescaled_exp_sum : 0.0f;
    shared_exp_sums[threadIdx.x] = rescaled_exp_sum;
  //}

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    global_exp_sum += __shfl_xor(global_exp_sum, mask);
  }
  if (threadIdx.x==0) {
    shared_global_exp_sum = global_exp_sum;
  }
  __syncthreads();

  //global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, shared_global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                        + head_idx * max_num_partitions * HEAD_SIZE;
//#pragma unroll
  //for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
  //if (threadIdx.x < HEAD_SIZE) { //TODO: assume HEAD_SIZE < NUM_THREADS, revisit this assumption later
    constexpr int MAX_NPAR = 64;
    scalar_t tmps[MAX_NPAR];
    int lastj=0;
    #pragma unroll
    for (int j = 0; j < MAX_NPAR; j++) {
        lastj = (j<num_partitions) ? j: lastj;
        tmps[j] = tmp_out_ptr[lastj * HEAD_SIZE + threadIdx.x];
    }

    //float tmpf[64];
    //#pragma unroll
    //for (int j = 0; j < 64; j++) {
    //    const float mult = (j<num_partitions) ? 1.0f: 0.0f;
    //    tmpf[j] = (float)tmps[j] * mult;
    //}

    float acc = 0.0f;
    //for (int j = 0; j < num_partitions; ++j) {
    #pragma unroll
    for (int j = 0; j < MAX_NPAR; j++) {
      //acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
      const float expsum = (j<num_partitions) ? shared_exp_sums[j] : 0.0f;
      //acc += (float)tmp_out_ptr[j * HEAD_SIZE + threadIdx.x] * shared_exp_sums[j] * inv_global_exp_sum;
      acc += (float)tmps[j] * expsum;
    }
    acc *= inv_global_exp_sum;
    //from_float(out_ptr[threadIdx.x], acc);
    out_ptr[threadIdx.x] = (scalar_t)acc;
  //}
}

#define CALL_CUSTOM_LAUNCHER(T)                             \
  paged_attention_custom_launcher<T>(                       \
    out,                                                            \
    exp_sums,                                                       \
    max_logits,                                                     \
    tmp_out,                                                        \
    query,                                                          \
    key_cache,                                                      \
    value_cache,                                                    \
    num_kv_heads,                                                   \
    scale,                                                          \
    block_tables,                                                   \
    context_lens,                                                   \
    max_context_len,\
    alibi_slopes);

#define LAUNCH_CUSTOM_ATTENTION(GQA_RATIO)                                                  \
  paged_attention_ll4mi_QKV_kernel<T,BLOCK_SIZE,HEAD_SIZE,NTHR,GQA_RATIO>      \
  <<<grid, block, 0, stream>>>(                                                 \
    query_ptr,                                                                                \
    key_cache_ptr,                                                                            \
    value_cache_ptr,                                                                          \
    num_kv_heads,                                                                         \
    scale,                                                                                    \
    block_tables_ptr,                                                                         \
    context_lens_ptr,                                                                         \
    max_num_blocks_per_seq,                                                                   \
    alibi_slopes_ptr,                                                                         \
    q_stride,                                                                                 \
    kv_block_stride,                                                                          \
    kv_head_stride, exp_sums_ptr, max_logits_ptr, tmp_out_ptr,out_ptr,max_ctx_blocks);

template<typename T, int BLOCK_SIZE=16, int HEAD_SIZE=128>
void paged_attention_custom_launcher(
  torch::Tensor& out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  const int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int max_context_len,
#if 0
  torch::Tensor& qk_out,
  torch::Tensor& softmax_out,
#endif
  const c10::optional<torch::Tensor>& alibi_slopes) {

  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  //int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  //assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes ?
    reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
    : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
  T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();
#if 0
  T* qk_out_ptr = reinterpret_cast<T*>(qk_out.data_ptr());
  T* softmax_out_ptr = reinterpret_cast<T*>(softmax_out.data_ptr());
#endif
  //constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  //int logits_size = PARTITION_SIZE * sizeof(float);
  //int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

  // For paged attention v2 kernel.
  //dim3 grid(num_heads, num_seqs, max_num_partitions);
  //int shared_mem_size = std::max(logits_size, outputs_size);
  //// For paged attention v2 reduce kernel.
  //assert(max_num_partitions <= MAX_PARTITIONS);
  //assert(MAX_PARTITIONS<=head_size);
  //dim3 reduce_grid(num_heads, num_seqs);
  //dim3 reduce_block(head_size); //TODO: assumes max_partitions < head_SIZE
  ////dim3 reduce_block(NUM_THREADS);
  //int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  int max_ctx_blocks = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE);

  //dim3 grid(num_seqs,BLOCK_RATIO_PER_WG*max_ctx_blocks);
  //dim3 block(num_heads*HEAD_SIZE*sizeof(T)/sizeof(float4));
  constexpr int NTHR = 256;
  const int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, NTHR);
  //constexpr int NPAR = 2;
  //constexpr int GQA_RATIO = 32;
  const int gqa_ratio = num_heads/num_kv_heads;
  //assert(gqa_ratio>=4);
  //assert(gqa_ratio%4==0);
  assert(num_heads%num_kv_heads==0);
  assert(head_size==HEAD_SIZE);
  dim3 grid(num_seqs,max_num_partitions,num_kv_heads);
  dim3 block(NTHR);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (gqa_ratio) {
      case 1: LAUNCH_CUSTOM_ATTENTION(1); break;
      case 2: LAUNCH_CUSTOM_ATTENTION(2); break;
      case 3: LAUNCH_CUSTOM_ATTENTION(3); break;
      case 4: LAUNCH_CUSTOM_ATTENTION(4); break;
      case 5: LAUNCH_CUSTOM_ATTENTION(5); break;
      case 6: LAUNCH_CUSTOM_ATTENTION(6); break;
      case 7: LAUNCH_CUSTOM_ATTENTION(7); break;
      case 8: LAUNCH_CUSTOM_ATTENTION(8); break;
      case 9: LAUNCH_CUSTOM_ATTENTION(9); break;
      case 10: LAUNCH_CUSTOM_ATTENTION(10); break;
      case 11: LAUNCH_CUSTOM_ATTENTION(11); break;
      case 12: LAUNCH_CUSTOM_ATTENTION(12); break;
      case 13: LAUNCH_CUSTOM_ATTENTION(13); break;
      case 14: LAUNCH_CUSTOM_ATTENTION(14); break;
      case 15: LAUNCH_CUSTOM_ATTENTION(15); break;
      case 16: LAUNCH_CUSTOM_ATTENTION(16); break;
      default:
        TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio);
        break;
  }
  //dim3 grid2(num_heads,num_seqs,head_size/HEAD_ELEMS_PER_WG);
  //dim3 block2(1024);
  // LAUNCH_CUSTOM_ATTENTION2;
  //constexpr int PARSIZE = 256;
  dim3 reduce_grid(num_heads, num_seqs);
  dim3 reduce_block(head_size); //TODO: assumes max_partitions < head_SIZE
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);
  paged_attention_ll4mi_reduce_kernel<T, HEAD_SIZE, HEAD_SIZE, NTHR>
  <<<reduce_grid, reduce_block, reduce_shared_mem_size, stream>>>(
    out_ptr,
    exp_sums_ptr,
    max_logits_ptr,
    tmp_out_ptr,
    context_lens_ptr,
    max_num_partitions);
  //switch (head_size) {
  //  // NOTE(woosuk): To reduce the compilation time, we only compile for the
  //  // head sizes that we use in the model. However, we can easily extend this
  //  // to support any head size which is a multiple of 16.
  //  case 64:
  //    LAUNCH_PAGED_ATTENTION_V2(64);
  //    break;
  //  case 80:
  //    LAUNCH_PAGED_ATTENTION_V2(80);
  //    break;
  //  case 96:
  //    LAUNCH_PAGED_ATTENTION_V2(96);
  //    break;
  //  case 112:
  //    LAUNCH_PAGED_ATTENTION_V2(112);
  //    break;
  //  case 128:
  //    LAUNCH_PAGED_ATTENTION_V2(128);
  //    break;
  //  case 256:
  //    LAUNCH_PAGED_ATTENTION_V2(256);
  //    break;
  //  default:
  //    TORCH_CHECK(false, "Unsupported head size: ", head_size);
  //    break;
  //}
}

void paged_attention_custom(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& exp_sums,        // [num_seqs, num_heads, max_num_partitions]
  torch::Tensor& max_logits,      // [num_seqs, num_heads, max_num_partitions]
  torch::Tensor& tmp_out,         // [num_seqs, num_heads, max_num_partitions, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len,
#if 0
  torch::Tensor& qk_out,
  torch::Tensor& softmax_out,
#endif
  const c10::optional<torch::Tensor>& alibi_slopes,
  const std::string& kv_cache_dtype) {
  assert(block_size==16);
  if (query.dtype() == at::ScalarType::Half) {
    //CALL_V2_LAUNCHER_BLOCK_SIZE(__half);
    CALL_CUSTOM_LAUNCHER(_Float16);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
