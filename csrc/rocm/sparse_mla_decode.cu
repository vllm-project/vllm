// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using bf16x8 = __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16;
using fx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

static constexpr int NOPE_DIM = 448;
static constexpr int ROPE_DIM = 64;
static constexpr int TOKEN_BYTES = 576;
static constexpr int SCALE_BYTES = 8;
static constexpr int HEAD_DIM = 512;
static constexpr int BLOCK_H = 16;
static constexpr int BLOCK_K = 32;
static constexpr int N_TILES = HEAD_DIM / 16;    // 32
static constexpr int QK_N_TILES = BLOCK_K / 16;  // 2

__device__ __forceinline__ fx4 mfma_16x16x32_bf16(bf16x8 a, bf16x8 b, fx4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ void gather_and_dequant_k_tile(
    int k_start, int k_len, const uint8_t* cache_base, int64_t cache_stride0,
    int num_rows, int block_size, const int32_t* idx_base, __bf16* k_lds,
    int8_t* kv_lds, int tid) {
  const int tok_id = tid >> 3;  // 0..31
  const int chunk = tid & 7;    // 0..7
  const int col0 = chunk * 64;

  int k_pos = k_start + tok_id;
  bool in_range = (k_pos < k_len);
  int slot = in_range ? idx_base[k_pos] : 0;
  bool valid = in_range && (slot >= 0) && (slot < num_rows);
  int safe_slot = valid ? slot : 0;
  int bi = safe_slot / block_size;
  int pib = safe_slot - bi * block_size;
  const uint8_t* block_ptr = cache_base + (int64_t)bi * cache_stride0;
  const uint8_t* token_ptr = block_ptr + pib * TOKEN_BYTES;

  __bf16* dst_row = &k_lds[tok_id * HEAD_DIM + col0];

  if (!valid) {
    int4 z;
    z.x = z.y = z.z = z.w = 0;
    int4* d4 = reinterpret_cast<int4*>(dst_row);
#pragma unroll
    for (int j = 0; j < 8; ++j) d4[j] = z;
  } else if (col0 < NOPE_DIM) {
    const uint8_t* scale_ptr =
        block_ptr + block_size * TOKEN_BYTES + pib * SCALE_BYTES;
    uint8_t scl_u = scale_ptr[chunk];
    union {
      uint32_t u;
      float fv;
    } sb;
    sb.u = ((uint32_t)scl_u) << 23;
    float scl_f = sb.fv;

    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(token_ptr + col0);
#pragma unroll
    for (int u32_i = 0; u32_i < 16; ++u32_i) {
      uint32_t word = src32[u32_i];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        uint8_t kb = (word >> (b * 8)) & 0xFF;
        uint32_t packed = (uint32_t)kb;
        float f = __builtin_amdgcn_cvt_f32_fp8(packed, 0) * scl_f;
        dst_row[u32_i * 4 + b] = (__bf16)f;
      }
    }
  } else {
    const int4* src4 = reinterpret_cast<const int4*>(token_ptr + NOPE_DIM);
    int4* d4 = reinterpret_cast<int4*>(dst_row);
#pragma unroll
    for (int j = 0; j < 8; ++j) d4[j] = src4[j];
  }

  if (tid < BLOCK_K) {
    int kp = k_start + tid;
    int sl = (kp < k_len) ? idx_base[kp] : -1;
    kv_lds[tid] = (kp < k_len) && (sl >= 0) && (sl < num_rows) ? 1 : 0;
  }
}

constexpr int N_TILES_PER_WAVE = 8;  // 32 N-tiles / 4 waves
__device__ __forceinline__ void process_k_tile(
    const __bf16* q_lds, const __bf16* k_lds, const int8_t* kv_lds,
    __bf16* p_lds, float* scores_lds, float* m_state, float* l_state, fx4* acc,
    float scale, int lane, int m_a, int kg, int n_b, int m_d_base, int n_d,
    int wave) {
  if (wave == 0) {
    fx4 qk[2] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; ++c) {
      bf16x8 q_reg;
      const __bf16* q_src = &q_lds[m_a * HEAD_DIM + c * 32 + kg * 8];
#pragma unroll
      for (int i = 0; i < 8; ++i) q_reg[i] = q_src[i];

#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        bf16x8 k_reg;
        const __bf16* k_src =
            &k_lds[(nt * 16 + n_b) * HEAD_DIM + c * 32 + kg * 8];
#pragma unroll
        for (int i = 0; i < 8; ++i) k_reg[i] = k_src[i];
        qk[nt] = mfma_16x16x32_bf16(q_reg, k_reg, qk[nt]);
      }
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int k_col = nt * 16 + n_d;
        float s = qk[nt][i] * scale;
        if (!kv_lds[k_col]) s = -3.4028234663852886e38f;
        scores_lds[(m_d_base + i) * BLOCK_K + nt * 16 + n_d] = s;
      }
    }
  }

  __syncthreads();

  fx4 qk_local[2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    qk_local[0][i] = scores_lds[(m_d_base + i) * BLOCK_K + n_d];
    qk_local[1][i] = scores_lds[(m_d_base + i) * BLOCK_K + 16 + n_d];
  }

  fx4 p[2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float row_max = fmaxf(qk_local[0][i], qk_local[1][i]);
    row_max = fmaxf(row_max, __shfl_xor(row_max, 1));
    row_max = fmaxf(row_max, __shfl_xor(row_max, 2));
    row_max = fmaxf(row_max, __shfl_xor(row_max, 4));
    row_max = fmaxf(row_max, __shfl_xor(row_max, 8));

    float m_new = fmaxf(m_state[i], row_max);
    float alpha =
        __builtin_amdgcn_exp2f((m_state[i] - m_new) * 1.4426950408889634f);

    float e0 =
        __builtin_amdgcn_exp2f((qk_local[0][i] - m_new) * 1.4426950408889634f);
    float e1 =
        __builtin_amdgcn_exp2f((qk_local[1][i] - m_new) * 1.4426950408889634f);

    float row_sum = e0 + e1;
    row_sum += __shfl_xor(row_sum, 1);
    row_sum += __shfl_xor(row_sum, 2);
    row_sum += __shfl_xor(row_sum, 4);
    row_sum += __shfl_xor(row_sum, 8);

    float l_new = l_state[i] * alpha + row_sum;
    p[0][i] = e0;
    p[1][i] = e1;

#pragma unroll
    for (int nt = 0; nt < N_TILES_PER_WAVE; ++nt) acc[nt][i] *= alpha;

    m_state[i] = m_new;
    l_state[i] = l_new;
  }

  if (wave == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p_lds[(m_d_base + i) * BLOCK_K + n_d] = (__bf16)p[0][i];
      p_lds[(m_d_base + i) * BLOCK_K + 16 + n_d] = (__bf16)p[1][i];
    }
  }

  __syncthreads();

  bf16x8 p_reg;
  const __bf16* p_src = &p_lds[m_a * BLOCK_K + kg * 8];
#pragma unroll
  for (int i = 0; i < 8; ++i) p_reg[i] = p_src[i];

#pragma unroll
  for (int nt_local = 0; nt_local < N_TILES_PER_WAVE; ++nt_local) {
    int n_tile = wave * N_TILES_PER_WAVE + nt_local;
    bf16x8 k_reg;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      k_reg[i] = k_lds[(kg * 8 + i) * HEAD_DIM + n_tile * 16 + n_b];
    }
    acc[nt_local] = mfma_16x16x32_bf16(p_reg, k_reg, acc[nt_local]);
  }
}

__device__ __forceinline__ void load_q(const __bf16* q, int64_t q_stride0,
                                       int64_t q_stride1, int query, int pid_h,
                                       int num_heads, __bf16* q_lds, int tid) {
  const int qh = tid >> 4;          // 0..15
  const int qc0 = (tid & 15) << 5;  // 0,32,...,480
  const int head_global = pid_h * BLOCK_H + qh;
  __bf16* dst = &q_lds[qh * HEAD_DIM + qc0];
  if (head_global < num_heads) {
    const __bf16* src = q + query * q_stride0 + head_global * q_stride1 + qc0;
    const int4* s4 = reinterpret_cast<const int4*>(src);
    int4* d4 = reinterpret_cast<int4*>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i) d4[i] = s4[i];
  } else {
    int4 z;
    z.x = z.y = z.z = z.w = 0;
    int4* d4 = reinterpret_cast<int4*>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i) d4[i] = z;
  }
}

template <bool HAS_ATTN_SINK, bool HAS_EXTRA>
__global__ __launch_bounds__(256, 2) void sparse_mla_decode_kernel(
    const __bf16* __restrict__ q, const uint8_t* __restrict__ main_cache,
    const int32_t* __restrict__ main_indices,
    const int32_t* __restrict__ main_indptr,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_indptr,
    const float* __restrict__ attn_sink, __bf16* __restrict__ output,
    int64_t q_stride0, int64_t q_stride1, int64_t out_stride0,
    int64_t out_stride1, int64_t main_cache_stride0,
    int64_t extra_cache_stride0, int main_num_rows, int extra_num_rows,
    int main_block_size, int extra_block_size, float scale, int num_heads) {
  const int query = blockIdx.x;
  const int pid_h = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int lane = tid & 63;

  const int m_a = lane & 15;
  const int kg = lane >> 4;
  const int n_b = lane & 15;
  const int m_d_base = (lane >> 4) * 4;
  const int n_d = lane & 15;

  __shared__ __bf16 q_lds[BLOCK_H * HEAD_DIM];
  __shared__ __bf16 k_lds[BLOCK_K * HEAD_DIM];
  __shared__ __bf16 p_lds[BLOCK_H * BLOCK_K];
  __shared__ float scores_lds[BLOCK_H * BLOCK_K];
  __shared__ int8_t kv_lds[BLOCK_K];
  __shared__ char force_1wg_per_cu[48 * 1024];  // pads LDS to ~96 KB
  (void)force_1wg_per_cu;

  load_q(q, q_stride0, q_stride1, query, pid_h, num_heads, q_lds, tid);

  float m_state[4], l_state[4];
  fx4 acc[N_TILES_PER_WAVE];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    m_state[i] = -3.4028234663852886e38f;
    l_state[i] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < N_TILES_PER_WAVE; ++i) {
    acc[i] = (fx4){0.f, 0.f, 0.f, 0.f};
  }

  __syncthreads();

  {
    int main_start = main_indptr[query];
    int main_end = main_indptr[query + 1];
    int main_len = main_end - main_start;
    for (int k_start = 0; k_start < main_len; k_start += BLOCK_K) {
      gather_and_dequant_k_tile(
          k_start, main_len, main_cache, main_cache_stride0, main_num_rows,
          main_block_size, main_indices + main_start, k_lds, kv_lds, tid);
      __syncthreads();
      process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds, m_state, l_state,
                     acc, scale, lane, m_a, kg, n_b, m_d_base, n_d, wave);
      __syncthreads();
    }
  }

  if (HAS_EXTRA) {
    int extra_start = extra_indptr[query];
    int extra_end = extra_indptr[query + 1];
    int extra_len = extra_end - extra_start;
    for (int k_start = 0; k_start < extra_len; k_start += BLOCK_K) {
      gather_and_dequant_k_tile(
          k_start, extra_len, extra_cache, extra_cache_stride0, extra_num_rows,
          extra_block_size, extra_indices + extra_start, k_lds, kv_lds, tid);
      __syncthreads();
      process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds, m_state, l_state,
                     acc, scale, lane, m_a, kg, n_b, m_d_base, n_d, wave);
      __syncthreads();
    }
  }

  {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int head_local = m_d_base + i;
      int head_global = pid_h * BLOCK_H + head_local;
      if (head_global >= num_heads) continue;

      float m_final = m_state[i];
      float l_final = l_state[i];
      float alpha_final = 1.f;
      if (HAS_ATTN_SINK) {
        float sink_val = attn_sink[head_global];
        m_final = fmaxf(m_state[i], sink_val);
        alpha_final = __builtin_amdgcn_exp2f((m_state[i] - m_final) *
                                             1.4426950408889634f);
        l_final =
            l_state[i] * alpha_final +
            __builtin_amdgcn_exp2f((sink_val - m_final) * 1.4426950408889634f);
      }
      float denom = fmaxf(l_final, 1.0e-30f);
      bool live = (l_final > 0.f);

      __bf16* out_row =
          output + query * out_stride0 + head_global * out_stride1;
#pragma unroll
      for (int nt_local = 0; nt_local < N_TILES_PER_WAVE; ++nt_local) {
        int n_tile = wave * N_TILES_PER_WAVE + nt_local;
        int col = n_tile * 16 + n_d;
        float v = live ? (acc[nt_local][i] * alpha_final) / denom : 0.f;
        out_row[col] = (__bf16)v;
      }
    }
  }
}

template <bool HAS_EXTRA, int SPLIT_K>
__global__ __launch_bounds__(256, 2) void sparse_mla_decode_partial_kernel(
    const __bf16* __restrict__ q, const uint8_t* __restrict__ main_cache,
    const int32_t* __restrict__ main_indices,
    const int32_t* __restrict__ main_indptr,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_indptr, float* __restrict__ scratch_m,
    float* __restrict__ scratch_l, __bf16* __restrict__ scratch_acc,
    int64_t q_stride0, int64_t q_stride1, int64_t main_cache_stride0,
    int64_t extra_cache_stride0, int main_num_rows, int extra_num_rows,
    int main_block_size, int extra_block_size, float scale, int num_heads,
    int num_head_blocks) {
  const int query = blockIdx.x;
  const int pid_hs = blockIdx.y;
  const int pid_split = pid_hs / num_head_blocks;
  const int pid_h = pid_hs - pid_split * num_head_blocks;
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int lane = tid & 63;

  const int m_a = lane & 15;
  const int kg = lane >> 4;
  const int n_b = lane & 15;
  const int m_d_base = (lane >> 4) * 4;
  const int n_d = lane & 15;

  __shared__ __bf16 q_lds[BLOCK_H * HEAD_DIM];
  __shared__ __bf16 k_lds[BLOCK_K * HEAD_DIM];
  __shared__ __bf16 p_lds[BLOCK_H * BLOCK_K];
  __shared__ float scores_lds[BLOCK_H * BLOCK_K];
  __shared__ int8_t kv_lds[BLOCK_K];
  load_q(q, q_stride0, q_stride1, query, pid_h, num_heads, q_lds, tid);

  float m_state[4], l_state[4];
  fx4 acc[N_TILES_PER_WAVE];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    m_state[i] = -3.4028234663852886e38f;
    l_state[i] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < N_TILES_PER_WAVE; ++i) {
    acc[i] = (fx4){0.f, 0.f, 0.f, 0.f};
  }

  __syncthreads();

  {
    int main_start = main_indptr[query];
    int main_end = main_indptr[query + 1];
    int main_len = main_end - main_start;
    for (int k_start = pid_split * BLOCK_K; k_start < main_len;
         k_start += BLOCK_K * SPLIT_K) {
      gather_and_dequant_k_tile(
          k_start, main_len, main_cache, main_cache_stride0, main_num_rows,
          main_block_size, main_indices + main_start, k_lds, kv_lds, tid);
      __syncthreads();
      process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds, m_state, l_state,
                     acc, scale, lane, m_a, kg, n_b, m_d_base, n_d, wave);
      __syncthreads();
    }
  }

  if (HAS_EXTRA) {
    int extra_start = extra_indptr[query];
    int extra_end = extra_indptr[query + 1];
    int extra_len = extra_end - extra_start;
    for (int k_start = pid_split * BLOCK_K; k_start < extra_len;
         k_start += BLOCK_K * SPLIT_K) {
      gather_and_dequant_k_tile(
          k_start, extra_len, extra_cache, extra_cache_stride0, extra_num_rows,
          extra_block_size, extra_indices + extra_start, k_lds, kv_lds, tid);
      __syncthreads();
      process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds, m_state, l_state,
                     acc, scale, lane, m_a, kg, n_b, m_d_base, n_d, wave);
      __syncthreads();
    }
  }

  const int triple = (query * num_head_blocks + pid_h) * SPLIT_K + pid_split;

  if (wave == 0 && n_d == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int idx = triple * BLOCK_H + m_d_base + i;
      scratch_m[idx] = m_state[i];
      scratch_l[idx] = l_state[i];
    }
  }

  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = m_d_base + i;
#pragma unroll
    for (int nt_local = 0; nt_local < N_TILES_PER_WAVE; ++nt_local) {
      int n_tile = wave * N_TILES_PER_WAVE + nt_local;
      int col = n_tile * 16 + n_d;
      k_lds[row * HEAD_DIM + col] = (__bf16)acc[nt_local][i];
    }
  }
  __syncthreads();

  {
    int my_row = tid >> 4;
    int my_col0 = (tid & 15) << 5;
    __bf16* dst = scratch_acc + (int64_t)triple * BLOCK_H * HEAD_DIM +
                  my_row * HEAD_DIM + my_col0;
    const int4* src4 =
        reinterpret_cast<const int4*>(&k_lds[my_row * HEAD_DIM + my_col0]);
    int4* dst4 = reinterpret_cast<int4*>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i) dst4[i] = src4[i];
  }
}

template <bool HAS_ATTN_SINK, int SPLIT_K>
__global__ __launch_bounds__(256, 4) void sparse_mla_decode_reduce_kernel(
    const float* __restrict__ scratch_m, const float* __restrict__ scratch_l,
    const __bf16* __restrict__ scratch_acc, const float* __restrict__ attn_sink,
    __bf16* __restrict__ output, int64_t out_stride0, int64_t out_stride1,
    int num_heads, int num_head_blocks) {
  const int query = blockIdx.x;
  const int pid_h = blockIdx.y;
  const int tid = threadIdx.x;

  const int my_row = tid >> 4;          // 0..15
  const int my_col0 = (tid & 15) << 5;  // 0,32,...,480
  const int head_global = pid_h * BLOCK_H + my_row;

  float m_merged = -3.4028234663852886e38f;
  float l_merged = 0.f;
  float acc_merged[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) acc_merged[i] = 0.f;

#pragma unroll
  for (int s = 0; s < SPLIT_K; ++s) {
    const int triple = (query * num_head_blocks + pid_h) * SPLIT_K + s;
    float m_s = scratch_m[triple * BLOCK_H + my_row];
    float l_s = scratch_l[triple * BLOCK_H + my_row];

    float m_new = fmaxf(m_merged, m_s);
    float alpha =
        __builtin_amdgcn_exp2f((m_merged - m_new) * 1.4426950408889634f);
    float beta = __builtin_amdgcn_exp2f((m_s - m_new) * 1.4426950408889634f);
    l_merged = l_merged * alpha + l_s * beta;
    m_merged = m_new;

    const __bf16* acc_base = scratch_acc +
                             (int64_t)triple * BLOCK_H * HEAD_DIM +
                             my_row * HEAD_DIM + my_col0;
    const int4* src4 = reinterpret_cast<const int4*>(acc_base);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int4 v = src4[i];
      __bf16 vbf[8];
      *reinterpret_cast<int4*>(vbf) = v;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float a_s = (float)vbf[j];
        acc_merged[i * 8 + j] = acc_merged[i * 8 + j] * alpha + a_s * beta;
      }
    }
  }

  if (head_global >= num_heads) return;

  float m_final = m_merged;
  float l_final = l_merged;
  float alpha_final = 1.f;
  if (HAS_ATTN_SINK) {
    float sink_val = attn_sink[head_global];
    m_final = fmaxf(m_merged, sink_val);
    alpha_final =
        __builtin_amdgcn_exp2f((m_merged - m_final) * 1.4426950408889634f);
    l_final =
        l_merged * alpha_final +
        __builtin_amdgcn_exp2f((sink_val - m_final) * 1.4426950408889634f);
  }
  float denom = fmaxf(l_final, 1.0e-30f);
  bool live = (l_final > 0.f);
  float inv_denom = live ? (alpha_final / denom) : 0.f;

  __bf16* out_row =
      output + query * out_stride0 + head_global * out_stride1 + my_col0;
  __bf16 out_buf[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) out_buf[i] = (__bf16)(acc_merged[i] * inv_denom);
  int4* dst4 = reinterpret_cast<int4*>(out_row);
  const int4* sb4 = reinterpret_cast<const int4*>(out_buf);
#pragma unroll
  for (int i = 0; i < 4; ++i) dst4[i] = sb4[i];
}

void sparse_mla_decode_single(
    torch::Tensor q, torch::Tensor main_cache, torch::Tensor main_indices,
    torch::Tensor main_indptr, torch::Tensor extra_cache,
    torch::Tensor extra_indices, torch::Tensor extra_indptr,
    c10::optional<torch::Tensor> attn_sink, torch::Tensor output,
    int64_t main_block_size, int64_t extra_block_size, int64_t main_num_rows,
    int64_t extra_num_rows, double scale_d, bool has_extra) {
  const int num_queries = q.size(0);
  const int num_heads = q.size(1);
  const int num_head_blocks = (num_heads + BLOCK_H - 1) / BLOCK_H;
  const float scale_f = (float)scale_d;
  const bool has_sink = attn_sink.has_value();

  dim3 grid(num_queries, num_head_blocks);
  dim3 block(256);

  const __bf16* q_ptr = reinterpret_cast<const __bf16*>(q.data_ptr());
  const uint8_t* mc_ptr =
      reinterpret_cast<const uint8_t*>(main_cache.data_ptr());
  const uint8_t* ec_ptr =
      reinterpret_cast<const uint8_t*>(extra_cache.data_ptr());
  const int32_t* mi_ptr = main_indices.data_ptr<int32_t>();
  const int32_t* mip_ptr = main_indptr.data_ptr<int32_t>();
  const int32_t* ei_ptr = extra_indices.data_ptr<int32_t>();
  const int32_t* eip_ptr = extra_indptr.data_ptr<int32_t>();
  __bf16* out_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
  const float* sink_ptr =
      has_sink ? attn_sink.value().data_ptr<float>() : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(HAS_S, HAS_E)                                                   \
  do {                                                                         \
    sparse_mla_decode_kernel<HAS_S, HAS_E><<<grid, block, 0, stream>>>(        \
        q_ptr, mc_ptr, mi_ptr, mip_ptr, ec_ptr, ei_ptr, eip_ptr, sink_ptr,     \
        out_ptr, q.stride(0), q.stride(1), output.stride(0), output.stride(1), \
        main_cache.stride(0), extra_cache.stride(0), main_num_rows,            \
        extra_num_rows, main_block_size, extra_block_size, scale_f,            \
        num_heads);                                                            \
  } while (0)

  if (has_sink && has_extra)
    LAUNCH(true, true);
  else if (has_sink)
    LAUNCH(true, false);
  else if (has_extra)
    LAUNCH(false, true);
  else
    LAUNCH(false, false);

#undef LAUNCH
}

void sparse_mla_decode_split(
    torch::Tensor q, torch::Tensor main_cache, torch::Tensor main_indices,
    torch::Tensor main_indptr, torch::Tensor extra_cache,
    torch::Tensor extra_indices, torch::Tensor extra_indptr,
    c10::optional<torch::Tensor> attn_sink, torch::Tensor output,
    torch::Tensor scratch_m, torch::Tensor scratch_l, torch::Tensor scratch_acc,
    int64_t main_block_size, int64_t extra_block_size, int64_t main_num_rows,
    int64_t extra_num_rows, double scale_d, bool has_extra, int64_t split_k) {
  const int num_queries = q.size(0);
  const int num_heads = q.size(1);
  const int num_head_blocks = (num_heads + BLOCK_H - 1) / BLOCK_H;
  const float scale_f = (float)scale_d;
  const bool has_sink = attn_sink.has_value();

  dim3 grid_p(num_queries, num_head_blocks * (int)split_k);
  dim3 grid_r(num_queries, num_head_blocks);
  dim3 block_p(256);
  dim3 block_r(256);

  const __bf16* q_ptr = reinterpret_cast<const __bf16*>(q.data_ptr());
  const uint8_t* mc_ptr =
      reinterpret_cast<const uint8_t*>(main_cache.data_ptr());
  const uint8_t* ec_ptr =
      reinterpret_cast<const uint8_t*>(extra_cache.data_ptr());
  const int32_t* mi_ptr = main_indices.data_ptr<int32_t>();
  const int32_t* mip_ptr = main_indptr.data_ptr<int32_t>();
  const int32_t* ei_ptr = extra_indices.data_ptr<int32_t>();
  const int32_t* eip_ptr = extra_indptr.data_ptr<int32_t>();
  __bf16* out_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
  float* sm_ptr = scratch_m.data_ptr<float>();
  float* sl_ptr = scratch_l.data_ptr<float>();
  __bf16* sa_ptr = reinterpret_cast<__bf16*>(scratch_acc.data_ptr());
  const float* sink_ptr =
      has_sink ? attn_sink.value().data_ptr<float>() : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_P(HAS_E, SK)                                                  \
  do {                                                                       \
    sparse_mla_decode_partial_kernel<HAS_E, SK>                              \
        <<<grid_p, block_p, 0, stream>>>(                                    \
            q_ptr, mc_ptr, mi_ptr, mip_ptr, ec_ptr, ei_ptr, eip_ptr, sm_ptr, \
            sl_ptr, sa_ptr, q.stride(0), q.stride(1), main_cache.stride(0),  \
            extra_cache.stride(0), main_num_rows, extra_num_rows,            \
            main_block_size, extra_block_size, scale_f, num_heads,           \
            num_head_blocks);                                                \
  } while (0)

#define LAUNCH_R(HAS_S, SK)                                              \
  do {                                                                   \
    sparse_mla_decode_reduce_kernel<HAS_S, SK>                           \
        <<<grid_r, block_r, 0, stream>>>(                                \
            sm_ptr, sl_ptr, sa_ptr, sink_ptr, out_ptr, output.stride(0), \
            output.stride(1), num_heads, num_head_blocks);               \
  } while (0)

#define DISPATCH_SK(SK)    \
  do {                     \
    if (has_extra)         \
      LAUNCH_P(true, SK);  \
    else                   \
      LAUNCH_P(false, SK); \
    if (has_sink)          \
      LAUNCH_R(true, SK);  \
    else                   \
      LAUNCH_R(false, SK); \
  } while (0)

  switch ((int)split_k) {
    case 2:
      DISPATCH_SK(2);
      break;
    case 4:
      DISPATCH_SK(4);
      break;
    case 8:
      DISPATCH_SK(8);
      break;
    case 16:
      DISPATCH_SK(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported SPLIT_K");
  }
#undef DISPATCH_SK
#undef LAUNCH_P
#undef LAUNCH_R
}

TORCH_LIBRARY_FRAGMENT(vllm_sparse_mla_hip, m) {
  m.def(
      "decode_single(Tensor q, Tensor main_cache, Tensor main_indices, "
      "Tensor main_indptr, Tensor extra_cache, Tensor extra_indices, "
      "Tensor extra_indptr, Tensor? attn_sink, Tensor output, "
      "int main_block_size, int extra_block_size, int main_num_rows, "
      "int extra_num_rows, float scale, bool has_extra) -> ()");
  m.def(
      "decode_split(Tensor q, Tensor main_cache, Tensor main_indices, "
      "Tensor main_indptr, Tensor extra_cache, Tensor extra_indices, "
      "Tensor extra_indptr, Tensor? attn_sink, Tensor output, "
      "Tensor scratch_m, Tensor scratch_l, Tensor scratch_acc, "
      "int main_block_size, int extra_block_size, int main_num_rows, "
      "int extra_num_rows, float scale, bool has_extra, int split_k) -> ()");
}
TORCH_LIBRARY_IMPL(vllm_sparse_mla_hip, CUDA, m) {
  m.impl("decode_single", &sparse_mla_decode_single);
  m.impl("decode_split", &sparse_mla_decode_split);
}
