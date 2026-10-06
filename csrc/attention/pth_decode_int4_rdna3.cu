// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HIP split-KV decode attention for INT4 per-token-head KV cache on RDNA3.
// Replaces the Triton packed split-KV kernel to eliminate ~40 µs/call of
// Python/JIT dispatch overhead. Fuses RHT butterfly on Q + inverse on output.
//
// Stage 1: each block handles one (query, head, kv_split) — scalar dot
//          product with wave reduction, online softmax, V accumulation.
// Stage 2: reduce partial outputs across splits.

#include <cstdint>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>

// Templated load/store helpers for fp16 and bf16 query/output dtypes.
// Both __half and __hip_bfloat16 are implicitly convertible to/from float.
template <typename T>
__device__ __forceinline__ float to_float(T x) {
  return (float)x;
}
template <>
__device__ __forceinline__ float to_float<half>(half x) {
  return __half2float(x);
}
template <typename T>
__device__ __forceinline__ T from_float(float x) {
  return (T)x;
}
template <>
__device__ __forceinline__ half from_float<half>(float x) {
  return __float2half(x);
}

// ---------------------------------------------------------------------------
// Stage 1: per-split attention
// Grid: (num_q, num_q_heads, NUM_SPLITS)
// Block: HEAD_SIZE threads. Each thread owns one head dimension.
// ---------------------------------------------------------------------------

template <int HEAD_SIZE, int BLOCK_KV, typename QT>
__global__ void decode_int4_stage1(
    const QT* __restrict__ Q,             // [num_q, num_q_heads, HEAD_SIZE]
    const uint8_t* __restrict__ K_cache,  // [blocks, block_size, kv_heads,
                                          // HEAD_SIZE/2]
    const uint8_t* __restrict__ V_cache,
    const float* __restrict__ K_scale,  // [blocks, block_size, kv_heads] stego
    const float* __restrict__ V_scale,
    const float* __restrict__ rht_signs,  // [HEAD_SIZE] D₁ signs
    const int* __restrict__ block_table,  // [num_reqs, max_blocks]
    const int* __restrict__ q_to_req, const int* __restrict__ q_to_klen,
    float* __restrict__ mid_o,  // [num_q, num_q_heads, NUM_SPLITS, HEAD_SIZE+2]
    float sm_scale, int num_q_heads, int num_kv_heads, int block_size,
    int max_blocks, int num_splits, int64_t sq0, int64_t sq1, int64_t skb,
    int64_t sks, int64_t skh, int64_t svb, int64_t svs, int64_t svh,
    int64_t ssb, int64_t sss, int64_t ssh, int64_t svsb, int64_t svss,
    int64_t svsh, int64_t smo, int64_t smh, int64_t sms) {
  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int si = blockIdx.z;
  const int tid = threadIdx.x;  // 0..HEAD_SIZE-1

  const int req = q_to_req[qi];
  const int kv_len = q_to_klen[qi];
  const int kvh = hi / (num_q_heads / num_kv_heads);

  // Split range
  const int tps = (kv_len + num_splits - 1) / num_splits;
  const int start = si * tps;
  const int end = min(start + tps, kv_len);
  if (start >= kv_len) {
    // Empty split — write zeros
    mid_o[qi * smo + hi * smh + si * sms + tid] = 0.0f;
    if (tid == 0) {
      mid_o[qi * smo + hi * smh + si * sms + HEAD_SIZE] = -INFINITY;
      mid_o[qi * smo + hi * smh + si * sms + HEAD_SIZE + 1] = 0.0f;
    }
    return;
  }

  // ---- Load Q + RHT butterfly ----
  __shared__ float qs[HEAD_SIZE];
  float qv = to_float<QT>(Q[qi * sq0 + hi * sq1 + tid]);
  // D₁ sign flip
  qs[tid] = qv * rht_signs[tid];
  __syncthreads();
  // Parallel Hadamard butterfly
  #pragma unroll
  for (int h = 1; h < HEAD_SIZE; h *= 2) {
    int partner = tid ^ h;
    float mine = qs[tid], other = qs[partner];
    __syncthreads();
    qs[tid] = (tid & h) ? (other - mine) : (mine + other);
    __syncthreads();
  }
  float q_rot = qs[tid];

  // ---- Online softmax + V accumulation ----
  constexpr int NW = HEAD_SIZE / 32;
  __shared__ float dot_lds[NW];  // partial sums from NW waves
  const int wave_id = tid / 32;
  const int lane = tid & 31;

  float m_state = -INFINITY;
  float l_state = 0.0f;
  float o_acc = 0.0f;

  for (int kv = start; kv < end; ++kv) {
    const int lb = kv / block_size;
    const int slot = kv - lb * block_size;
    const int pb = block_table[req * max_blocks + lb];

    // Load K nibble for my dimension
    const int packed_idx = tid / 2;
    uint8_t k_byte = K_cache[pb * skb + slot * sks + kvh * skh + packed_idx];
    int k_nib = (tid & 1) ? ((k_byte >> 4) & 0xF) : (k_byte & 0xF);

    // Extract scale + zp from steganographed float
    float k_raw = K_scale[pb * ssb + slot * sss + kvh * ssh];
    int k_bits;
    __builtin_memcpy(&k_bits, &k_raw, 4);
    int k_zp = k_bits & 0xF;
    int k_sb = k_bits & ~0xF;
    float k_sc;
    __builtin_memcpy(&k_sc, &k_sb, 4);

    float k_val = (float)(k_nib - k_zp);

    // Dot product: Q_rot[d] * K[d] summed over all d
    float partial = q_rot * k_val;
    // Wave-level reduction (32 lanes)
    partial += __shfl_xor(partial, 1);
    partial += __shfl_xor(partial, 2);
    partial += __shfl_xor(partial, 4);
    partial += __shfl_xor(partial, 8);
    partial += __shfl_xor(partial, 16);
    // Lane 0 of each wave has sum of 32 dims
    if (lane == 0) dot_lds[wave_id] = partial;
    __syncthreads();

    // Combine NW waves via tree reduction in first NW threads
    float score = 0;
    if (tid < NW) score = dot_lds[tid];
  #pragma unroll
    for (int s = 1; s < NW; s *= 2)
      if (tid < NW) score += __shfl_xor(score, s);
    // Broadcast score to all threads via LDS
    if (tid == 0) dot_lds[0] = score * k_sc * sm_scale;
    __syncthreads();
    score = dot_lds[0];

    // Online softmax
    float m_new = fmaxf(m_state, score);
    float alpha = (m_state == -INFINITY) ? 0.0f : __expf(m_state - m_new);
    float p = (m_new == -INFINITY) ? 0.0f : __expf(score - m_new);
    l_state = l_state * alpha + p;
    o_acc = o_acc * alpha;
    m_state = m_new;

    // Load V nibble and accumulate
    uint8_t v_byte = V_cache[pb * svb + slot * svs + kvh * svh + packed_idx];
    int v_nib = (tid & 1) ? ((v_byte >> 4) & 0xF) : (v_byte & 0xF);
    float v_raw = V_scale[pb * svsb + slot * svss + kvh * svsh];
    int v_bits;
    __builtin_memcpy(&v_bits, &v_raw, 4);
    int v_zp = v_bits & 0xF;
    int v_sb = v_bits & ~0xF;
    float v_sc;
    __builtin_memcpy(&v_sc, &v_sb, 4);
    float v_val = (float)(v_nib - v_zp) * v_sc;

    o_acc += p * v_val;
  }

  // Store partial output + (m, l) for reduce stage
  float* out_ptr = mid_o + qi * smo + hi * smh + si * sms;
  out_ptr[tid] = o_acc;
  if (tid == 0) {
    out_ptr[HEAD_SIZE] = m_state;
    out_ptr[HEAD_SIZE + 1] = l_state;
  }
}

// ---------------------------------------------------------------------------
// Stage 1 v2: 128 threads for HEAD_SIZE=256 (2 dims per thread, 4 waves).
// Q must be pre-rotated externally. Avoids 8-wave cross-wave barrier cost.
// Grid: (num_q, num_q_heads, NUM_SPLITS), Block: HEAD_SIZE/2 threads.
// ---------------------------------------------------------------------------

template <int HEAD_SIZE, typename QT>
__global__ void decode_int4_stage1_v2(
    const QT* __restrict__ Q,  // [num_q, num_q_heads, HEAD_SIZE] PRE-ROTATED
    const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    const float* __restrict__ K_scale, const float* __restrict__ V_scale,
    const int* __restrict__ block_table, const int* __restrict__ q_to_req,
    const int* __restrict__ q_to_klen, float* __restrict__ mid_o,
    float sm_scale, int num_q_heads, int num_kv_heads, int block_size,
    int max_blocks, int num_splits, int64_t sq0, int64_t sq1, int64_t skb,
    int64_t sks, int64_t skh, int64_t svb, int64_t svs, int64_t svh,
    int64_t ssb, int64_t sss, int64_t ssh, int64_t svsb, int64_t svss,
    int64_t svsh, int64_t smo, int64_t smh, int64_t sms) {
  constexpr int THREADS = HEAD_SIZE / 2;  // 128 for HS=256
  constexpr int NW = THREADS / 32;        // 4 waves

  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int si = blockIdx.z;
  const int tid = threadIdx.x;  // 0..THREADS-1

  const int req = q_to_req[qi];
  const int kv_len = q_to_klen[qi];
  const int kvh = hi / (num_q_heads / num_kv_heads);

  const int tps = (kv_len + num_splits - 1) / num_splits;
  const int start = si * tps;
  const int end = min(start + tps, kv_len);
  if (start >= kv_len) {
    mid_o[qi * smo + hi * smh + si * sms + 2 * tid] = 0.0f;
    mid_o[qi * smo + hi * smh + si * sms + 2 * tid + 1] = 0.0f;
    if (tid == 0) {
      mid_o[qi * smo + hi * smh + si * sms + HEAD_SIZE] = -INFINITY;
      mid_o[qi * smo + hi * smh + si * sms + HEAD_SIZE + 1] = 0.0f;
    }
    return;
  }

  // Load 2 Q values per thread (pre-rotated)
  const QT* qr = Q + qi * sq0 + hi * sq1;
  float q0 = to_float<QT>(qr[2 * tid]);
  float q1 = to_float<QT>(qr[2 * tid + 1]);

  __shared__ float dot_lds[NW];
  const int wave_id = tid / 32;
  const int lane = tid & 31;

  float m_state = -INFINITY;
  float l_state = 0.0f;
  float o0 = 0.0f, o1 = 0.0f;

  for (int kv = start; kv < end; ++kv) {
    const int lb = kv / block_size;
    const int slot = kv - lb * block_size;
    const int pb = block_table[req * max_blocks + lb];

    // Each thread loads 1 packed byte = 2 nibbles (dim 2*tid, 2*tid+1)
    uint8_t k_byte = K_cache[pb * skb + slot * sks + kvh * skh + tid];
    // Symmetric format: nibbles stored as offset-8 binary. Subtract 8
    // (constant).
    float k0 = (float)((int)(k_byte & 0xF) - 8);
    float k1 = (float)((int)((k_byte >> 4) & 0xF) - 8);

    // Scale is plain float32 (no steganography).
    float k_sc = K_scale[pb * ssb + slot * sss + kvh * ssh];

    // Dot: 2 terms per thread → wave sums 64 dims → 4 waves = 256 dims
    float partial = q0 * k0 + q1 * k1;
    partial += __shfl_xor(partial, 1);
    partial += __shfl_xor(partial, 2);
    partial += __shfl_xor(partial, 4);
    partial += __shfl_xor(partial, 8);
    partial += __shfl_xor(partial, 16);
    if (lane == 0) dot_lds[wave_id] = partial;
    __syncthreads();

    float score = 0;
    if (tid < NW) score = dot_lds[tid];
  #pragma unroll
    for (int s = 1; s < NW; s *= 2)
      if (tid < NW) score += __shfl_xor(score, s);
    if (tid == 0) dot_lds[0] = score * k_sc * sm_scale;
    __syncthreads();
    score = dot_lds[0];

    float m_new = fmaxf(m_state, score);
    float alpha = (m_state == -INFINITY) ? 0.0f : __expf(m_state - m_new);
    float p = (m_new == -INFINITY) ? 0.0f : __expf(score - m_new);
    l_state = l_state * alpha + p;
    o0 = o0 * alpha;
    o1 = o1 * alpha;
    m_state = m_new;

    uint8_t v_byte = V_cache[pb * svb + slot * svs + kvh * svh + tid];
    float v0 = (float)((int)(v_byte & 0xF) - 8);
    float v1 = (float)((int)((v_byte >> 4) & 0xF) - 8);
    float v_sc = V_scale[pb * svsb + slot * svss + kvh * svsh];

    o0 += p * v0 * v_sc;
    o1 += p * v1 * v_sc;
  }

  float* out_ptr = mid_o + qi * smo + hi * smh + si * sms;
  out_ptr[2 * tid] = o0;
  out_ptr[2 * tid + 1] = o1;
  if (tid == 0) {
    out_ptr[HEAD_SIZE] = m_state;
    out_ptr[HEAD_SIZE + 1] = l_state;
  }
}

// ---------------------------------------------------------------------------
// Stage 2 v2: reduce across splits, no RHT (done externally).
// 128 threads, 2 dims per thread.
// ---------------------------------------------------------------------------
// Stage 1 v3: 32 threads (1 wave) for HEAD_SIZE=256. ZERO __syncthreads.
// Each thread handles 8 dims (4 packed bytes). Reduction via __shfl_xor only.
// Symmetric format: nibble - 8, plain float scale (no steganography).
// Grid: (num_q, num_q_heads, NUM_SPLITS), Block: 32 threads.
// ---------------------------------------------------------------------------

template <int HEAD_SIZE, typename QT>
__global__ void decode_int4_stage1_v3(
    const QT* __restrict__ Q,  // [num_q, num_q_heads, HEAD_SIZE] PRE-ROTATED
    const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    const float* __restrict__ K_scale,  // plain float (symmetric)
    const float* __restrict__ V_scale, const int* __restrict__ block_table,
    const int* __restrict__ q_to_req, const int* __restrict__ q_to_klen,
    float* __restrict__ mid_o, float sm_scale, int num_q_heads,
    int num_kv_heads, int block_size, int max_blocks, int num_splits,
    int64_t sq0, int64_t sq1, int64_t skb, int64_t sks, int64_t skh,
    int64_t svb, int64_t svs, int64_t svh, int64_t ssb, int64_t sss,
    int64_t ssh, int64_t svsb, int64_t svss, int64_t svsh, int64_t smo,
    int64_t smh, int64_t sms) {
  constexpr int DIMS_PER_THREAD = HEAD_SIZE / 32;        // 8 for HS=256
  constexpr int BYTES_PER_THREAD = DIMS_PER_THREAD / 2;  // 4 packed bytes

  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int si = blockIdx.z;
  const int tid = threadIdx.x;  // 0..31

  const int req = q_to_req[qi];
  const int kv_len = q_to_klen[qi];
  const int kvh = hi / (num_q_heads / num_kv_heads);

  const int tps = (kv_len + num_splits - 1) / num_splits;
  const int start = si * tps;
  const int end = min(start + tps, kv_len);
  if (start >= kv_len) {
    float* out_ptr = mid_o + qi * smo + hi * smh + si * sms;
    for (int d = 0; d < DIMS_PER_THREAD; ++d)
      out_ptr[tid * DIMS_PER_THREAD + d] = 0.0f;
    if (tid == 0) {
      out_ptr[HEAD_SIZE] = -INFINITY;
      out_ptr[HEAD_SIZE + 1] = 0.0f;
    }
    return;
  }

  // Load Q values for this thread's dims (8 dims, pre-rotated)
  const QT* qr = Q + qi * sq0 + hi * sq1;
  float q_vals[DIMS_PER_THREAD];
  #pragma unroll
  for (int d = 0; d < DIMS_PER_THREAD; ++d)
    q_vals[d] = to_float<QT>(qr[tid * DIMS_PER_THREAD + d]);

  // Pre-scale to log2 space: exp(x) = exp2(x * log2e). By folding log2e
  // into sm_scale we use exp2f directly (saves 1 hidden mul per exp call).
  const float sm_scale_log2 = sm_scale * 1.4426950408889634f;

  float m_state = -INFINITY;
  float l_state = 0.0f;
  float o_vals[DIMS_PER_THREAD] = {};

  int prev_lb = -1;
  int pb = 0;
  for (int kv = start; kv < end; ++kv) {
    const int lb = kv / block_size;
    const int slot = kv - lb * block_size;
    if (lb != prev_lb) {
      pb = block_table[req * max_blocks + lb];
      prev_lb = lb;
    }

    // Vectorized load: 1 dword = 4 packed bytes = 8 nibbles for K
    uint32_t k_dw = *reinterpret_cast<const uint32_t*>(
        K_cache + pb * skb + slot * sks + kvh * skh + tid * BYTES_PER_THREAD);

    // Extract scale + zp from steganographed float
    float k_raw = K_scale[pb * ssb + slot * sss + kvh * ssh];
    int k_bits;
    __builtin_memcpy(&k_bits, &k_raw, 4);
    int k_zp = k_bits & 0xF;
    int k_sb = k_bits & ~0xF;
    float k_sc;
    __builtin_memcpy(&k_sc, &k_sb, 4);

    float partial = 0.0f;
  #pragma unroll
    for (int b = 0; b < BYTES_PER_THREAD; ++b) {
      int kb = (k_dw >> (b * 8)) & 0xFF;
      float k0 = (float)((kb & 0xF) - k_zp);
      float k1 = (float)(((kb >> 4) & 0xF) - k_zp);
      partial += q_vals[b * 2] * k0 + q_vals[b * 2 + 1] * k1;
    }

  // Wave-wide reduction (no sync needed — single wave)
  #pragma unroll
    for (int s = 16; s > 0; s >>= 1) partial += __shfl_xor(partial, s);

    float score = partial * k_sc * sm_scale_log2;

    // Online softmax in log2 space — exp2f avoids the hidden x*log2e
    // multiply inside __expf. IEEE: exp2(-inf)=0, so first iter correct.
    float m_new = fmaxf(m_state, score);
    float alpha = exp2f(m_state - m_new);
    float p = exp2f(score - m_new);
    l_state = l_state * alpha + p;
  #pragma unroll
    for (int d = 0; d < DIMS_PER_THREAD; ++d) o_vals[d] *= alpha;
    m_state = m_new;

    // V accumulation — vectorized dword load, same thread owns output dims
    uint32_t v_dw = *reinterpret_cast<const uint32_t*>(
        V_cache + pb * svb + slot * svs + kvh * svh + tid * BYTES_PER_THREAD);
    float v_raw = V_scale[pb * svsb + slot * svss + kvh * svsh];
    int v_bits;
    __builtin_memcpy(&v_bits, &v_raw, 4);
    int v_zp = v_bits & 0xF;
    int v_sb = v_bits & ~0xF;
    float v_sc;
    __builtin_memcpy(&v_sc, &v_sb, 4);
    float p_vs = p * v_sc;
  #pragma unroll
    for (int b = 0; b < BYTES_PER_THREAD; ++b) {
      int vb = (v_dw >> (b * 8)) & 0xFF;
      o_vals[b * 2] += p_vs * (float)((vb & 0xF) - v_zp);
      o_vals[b * 2 + 1] += p_vs * (float)(((vb >> 4) & 0xF) - v_zp);
    }
  }

  // Write output
  float* out_ptr = mid_o + qi * smo + hi * smh + si * sms;
  #pragma unroll
  for (int d = 0; d < DIMS_PER_THREAD; ++d)
    out_ptr[tid * DIMS_PER_THREAD + d] = o_vals[d];
  if (tid == 0) {
    out_ptr[HEAD_SIZE] = m_state;
    out_ptr[HEAD_SIZE + 1] = l_state;
  }
}

// ---------------------------------------------------------------------------

template <int HEAD_SIZE, typename OT>
__global__ void decode_int4_reduce_v2(const float* __restrict__ mid_o,
                                      OT* __restrict__ out, int num_splits,
                                      int64_t smo, int64_t smh, int64_t sms,
                                      int64_t soo, int64_t soh) {
  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int tid = threadIdx.x;  // 0..HEAD_SIZE/2-1

  float m_global = -INFINITY;
  for (int s = 0; s < num_splits; ++s) {
    float ms = mid_o[qi * smo + hi * smh + s * sms + HEAD_SIZE];
    m_global = fmaxf(m_global, ms);
  }

  float o0 = 0.0f, o1 = 0.0f, l_global = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float* sp = mid_o + qi * smo + hi * smh + s * sms;
    float ms = sp[HEAD_SIZE];
    float ls = sp[HEAD_SIZE + 1];
    // ms / m_global come from v3 stage1, which pre-scales scores by log2(e)
    // and stores m_state in log2 space. The correct rescale is exp2(delta),
    // NOT __expf(delta): __expf would compute e^(log2_delta) and shrink
    // non-max splits by a factor of e^(delta*(1 - 1/log2(e))) ≈
    // e^(-0.307*|delta|), biasing attention toward the max split and inducing
    // repetition loops on long generations.
    float alpha = (ms == -INFINITY) ? 0.0f : exp2f(ms - m_global);
    o0 += sp[2 * tid] * alpha;
    o1 += sp[2 * tid + 1] * alpha;
    l_global += ls * alpha;
  }
  float inv_l = 1.0f / (l_global + 1e-10f);
  // No RHT here — done externally by rht_rotate_inplace_rdna3
  out[qi * soo + hi * soh + 2 * tid] = from_float<OT>(o0 * inv_l);
  out[qi * soo + hi * soh + 2 * tid + 1] = from_float<OT>(o1 * inv_l);
}

// ---------------------------------------------------------------------------
// Stage 2: reduce across splits → final output + inverse RHT
// Grid: (num_q, num_q_heads)
// Block: HEAD_SIZE threads
// ---------------------------------------------------------------------------

template <int HEAD_SIZE, typename OT>
__global__ void decode_int4_reduce(
    const float* __restrict__ mid_o,  // [num_q, num_q_heads, NUM_SPLITS,
                                      // HEAD_SIZE+2]
    OT* __restrict__ out,             // [num_q, num_q_heads, HEAD_SIZE]
    const float* __restrict__ rht_signs, int num_splits, float inv_head_size,
    int64_t smo, int64_t smh, int64_t sms, int64_t soo, int64_t soh) {
  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int tid = threadIdx.x;

  // Find global max across splits
  float m_global = -INFINITY;
  for (int s = 0; s < num_splits; ++s) {
    float ms = mid_o[qi * smo + hi * smh + s * sms + HEAD_SIZE];
    m_global = fmaxf(m_global, ms);
  }

  // Combine outputs with rescaling
  float o_final = 0.0f;
  float l_global = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float* sp = mid_o + qi * smo + hi * smh + s * sms;
    float ms = sp[HEAD_SIZE];
    float ls = sp[HEAD_SIZE + 1];
    float os = sp[tid];
    float alpha = (ms == -INFINITY) ? 0.0f : __expf(ms - m_global);
    o_final += os * alpha;
    l_global += ls * alpha;
  }
  o_final /= (l_global + 1e-10f);

  // ---- Inverse RHT butterfly + scale ----
  __shared__ float rs[HEAD_SIZE];
  rs[tid] = o_final;
  __syncthreads();

  #pragma unroll
  for (int h = 1; h < HEAD_SIZE; h *= 2) {
    int partner = tid ^ h;
    float mine = rs[tid], other = rs[partner];
    __syncthreads();
    rs[tid] = (tid & h) ? (other - mine) : (mine + other);
    __syncthreads();
  }

  // D₁ sign flip + /head_size
  out[qi * soo + hi * soh + tid] =
      from_float<OT>(rs[tid] * rht_signs[tid] * inv_head_size);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

void pth_decode_int4_rdna3(
    torch::Tensor out,        // [num_q, num_q_heads, HEAD_SIZE]
    torch::Tensor query,      // [num_q, num_q_heads, HEAD_SIZE]
    torch::Tensor key_cache,  // [blocks, block_size, kv_heads, HEAD_SIZE/2]
    torch::Tensor value_cache, torch::Tensor k_scale_cache,
    torch::Tensor v_scale_cache,
    torch::Tensor rht_signs,  // [HEAD_SIZE]
    torch::Tensor block_table, torch::Tensor q_to_req, torch::Tensor q_to_klen,
    torch::Tensor mid_o_buf,  // [num_q, num_q_heads, NUM_SPLITS, HEAD_SIZE+2]
    double sm_scale, int64_t num_kv_splits) {
  const int num_q = query.size(0);
  const int num_q_heads = query.size(1);
  const int head_size = query.size(2);
  const int num_kv_heads = k_scale_cache.size(2);
  const int block_size = key_cache.size(1);
  const int max_blocks = block_table.size(1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(head_size == 128 || head_size == 256,
              "pth_decode_int4_rdna3: head_size must be 128 or 256, got ",
              head_size);
  TORCH_CHECK(query.dtype() == at::kHalf || query.dtype() == at::kBFloat16,
              "pth_decode_int4_rdna3: query must be fp16 or bf16");
  TORCH_CHECK(out.dtype() == query.dtype(),
              "pth_decode_int4_rdna3: out must match query dtype");

  constexpr int BKV = 1;
  int ns = (int)num_kv_splits;
  TORCH_CHECK(mid_o_buf.size(2) >= ns);
  TORCH_CHECK(mid_o_buf.size(3) >= head_size + 2);

  #define LAUNCH_HS128(QT, OT)                                                 \
    do {                                                                       \
      constexpr int HS = 128;                                                  \
      dim3 grid1(num_q, num_q_heads, ns);                                      \
      decode_int4_stage1<HS, BKV, QT><<<grid1, dim3(HS), 0, stream>>>(         \
          (const QT*)query.data_ptr(), (const uint8_t*)key_cache.data_ptr(),   \
          (const uint8_t*)value_cache.data_ptr(),                              \
          (const float*)k_scale_cache.data_ptr(),                              \
          (const float*)v_scale_cache.data_ptr(),                              \
          (const float*)rht_signs.data_ptr(),                                  \
          (const int*)block_table.data_ptr(), (const int*)q_to_req.data_ptr(), \
          (const int*)q_to_klen.data_ptr(), (float*)mid_o_buf.data_ptr(),      \
          (float)sm_scale, num_q_heads, num_kv_heads, block_size, max_blocks,  \
          ns, query.stride(0), query.stride(1), key_cache.stride(0),           \
          key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),     \
          value_cache.stride(1), value_cache.stride(2),                        \
          k_scale_cache.stride(0), k_scale_cache.stride(1),                    \
          k_scale_cache.stride(2), v_scale_cache.stride(0),                    \
          v_scale_cache.stride(1), v_scale_cache.stride(2),                    \
          mid_o_buf.stride(0), mid_o_buf.stride(1), mid_o_buf.stride(2));      \
      dim3 grid2(num_q, num_q_heads);                                          \
      constexpr float inv_hs = 1.0f / (float)HS;                               \
      decode_int4_reduce<HS, OT><<<grid2, dim3(HS), 0, stream>>>(              \
          (const float*)mid_o_buf.data_ptr(), (OT*)out.data_ptr(),             \
          (const float*)rht_signs.data_ptr(), ns, inv_hs, mid_o_buf.stride(0), \
          mid_o_buf.stride(1), mid_o_buf.stride(2), out.stride(0),             \
          out.stride(1));                                                      \
    } while (0)

  #define LAUNCH_HS256_V3(QT, OT)                                              \
    do {                                                                       \
      constexpr int HS = 256;                                                  \
      constexpr int TH = HS / 2;                                               \
      dim3 grid1(num_q, num_q_heads, ns);                                      \
      decode_int4_stage1_v3<HS, QT><<<grid1, dim3(32), 0, stream>>>(           \
          (const QT*)query.data_ptr(), (const uint8_t*)key_cache.data_ptr(),   \
          (const uint8_t*)value_cache.data_ptr(),                              \
          (const float*)k_scale_cache.data_ptr(),                              \
          (const float*)v_scale_cache.data_ptr(),                              \
          (const int*)block_table.data_ptr(), (const int*)q_to_req.data_ptr(), \
          (const int*)q_to_klen.data_ptr(), (float*)mid_o_buf.data_ptr(),      \
          (float)sm_scale, num_q_heads, num_kv_heads, block_size, max_blocks,  \
          ns, query.stride(0), query.stride(1), key_cache.stride(0),           \
          key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),     \
          value_cache.stride(1), value_cache.stride(2),                        \
          k_scale_cache.stride(0), k_scale_cache.stride(1),                    \
          k_scale_cache.stride(2), v_scale_cache.stride(0),                    \
          v_scale_cache.stride(1), v_scale_cache.stride(2),                    \
          mid_o_buf.stride(0), mid_o_buf.stride(1), mid_o_buf.stride(2));      \
      dim3 grid2(num_q, num_q_heads);                                          \
      decode_int4_reduce_v2<HS, OT><<<grid2, dim3(TH), 0, stream>>>(           \
          (const float*)mid_o_buf.data_ptr(), (OT*)out.data_ptr(), ns,         \
          mid_o_buf.stride(0), mid_o_buf.stride(1), mid_o_buf.stride(2),       \
          out.stride(0), out.stride(1));                                       \
    } while (0)

  const bool is_bf16 = (query.dtype() == at::kBFloat16);
  if (head_size == 128) {
    if (is_bf16) {
      LAUNCH_HS128(__hip_bfloat16, __hip_bfloat16);
    } else {
      LAUNCH_HS128(half, half);
    }
  } else {
    if (is_bf16) {
      LAUNCH_HS256_V3(__hip_bfloat16, __hip_bfloat16);
    } else {
      LAUNCH_HS256_V3(half, half);
    }
  }
  #undef LAUNCH_HS128
  #undef LAUNCH_HS256_V3
}

#else
void pth_decode_int4_rdna3(torch::Tensor out, torch::Tensor query,
                           torch::Tensor key_cache, torch::Tensor value_cache,
                           torch::Tensor k_scale_cache,
                           torch::Tensor v_scale_cache, torch::Tensor rht_signs,
                           torch::Tensor block_table, torch::Tensor q_to_req,
                           torch::Tensor q_to_klen, torch::Tensor mid_o_buf,
                           double sm_scale, int64_t num_kv_splits) {
  TORCH_CHECK(false, "requires ROCm");
}
#endif
