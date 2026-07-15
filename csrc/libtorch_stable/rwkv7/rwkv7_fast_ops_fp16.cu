// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from BlinkDL/Albatross faster3a_2605/cuda at commit
// 5e941fb1eeb7f735a562fb5bbb30fad19adc825b. Source:
// https://github.com/BlinkDL/Albatross/tree/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/faster3a_2605/cuda
// Upstream license: Apache-2.0
// (https://github.com/BlinkDL/Albatross/blob/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/LICENSE).
// Dense 3D mix layout follows faster3a_2607 at commit
// 63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b.

#include <assert.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include <limits>
#include <vector>

using dtype = at::Half;

namespace {

constexpr int HEAD_SIZE = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr float KK_NORMALIZE_EPS = 1.0e-12f;
constexpr float TMIX_LN_X_EPS = 64.0e-5f;
constexpr int FFN_SPMV_THREADS = 128;
constexpr int FFN_TILE = 128;
constexpr int MIX_3D_C = 4096;
constexpr int MIX_3D_MAX_GRID = 65535;

inline int64_t ceil_div(int64_t n, int64_t d) { return (n + d - 1) / d; }

bool use_3d_mix(int B, int T, int C) {
  return T > 1 && C == MIX_3D_C && B <= MIX_3D_MAX_GRID &&
         T <= MIX_3D_MAX_GRID &&
         (B >= 2 || T == 2 || T == 4 || T == 16 || T == 64 || T == 512);
}

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline float load_h1(const dtype* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

__device__ inline void store_h1(dtype* ptr, float value) {
  *reinterpret_cast<__half*>(ptr) = __float2half_rn(value);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ inline float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

template <bool UpdateShift, bool Grid3D = false>
__global__ void tmix_mix6_kernel(
    int T, int C, const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    const int* __restrict__ slot_indices, const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w, const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v, const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g, dtype* __restrict__ out_r,
    dtype* __restrict__ out_w, dtype* __restrict__ out_k,
    dtype* __restrict__ out_v, dtype* __restrict__ out_a,
    dtype* __restrict__ out_g, int64_t total_pairs) {
  const int c_pairs = C >> 1;
  int64_t idx;
  int c;
  int b;
  int t;
  if constexpr (Grid3D) {
    const int c_pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c_pair >= c_pairs) {
      return;
    }
    b = static_cast<int>(blockIdx.z);
    t = static_cast<int>(blockIdx.y);
    c = c_pair << 1;
    idx = (static_cast<int64_t>(b) * T + t) * C + c;
  } else {
    const int64_t pair_idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
      return;
    }
    const int64_t bt = pair_idx / c_pairs;
    c = static_cast<int>(pair_idx - bt * c_pairs) << 1;
    b = static_cast<int>(bt / T);
    t = static_cast<int>(bt - static_cast<int64_t>(b) * T);
    idx = bt * C + c;
  }
  const int state_b = slot_indices == nullptr ? b : slot_indices[b];

  const __half2 cur2 = load_h2(x + idx);
  __half2 prev2;
  if (t == 0) {
    prev2 = load_h2(shift_state + static_cast<int64_t>(state_b) * C + c);
  } else {
    prev2 = load_h2(x + idx - C);
  }

  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float dx0 = prev.x - cur.x;
  const float dx1 = prev.y - cur.y;

  const float2 xr = __half22float2(load_h2(x_r + c));
  const float2 xw = __half22float2(load_h2(x_w + c));
  const float2 xk = __half22float2(load_h2(x_k + c));
  const float2 xv = __half22float2(load_h2(x_v + c));
  const float2 xa = __half22float2(load_h2(x_a + c));
  const float2 xg = __half22float2(load_h2(x_g + c));

  store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
  store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
  store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
  store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
  store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
  store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);

  if constexpr (UpdateShift) {
    if (t == T - 1) {
      *reinterpret_cast<__half2*>(shift_state +
                                  static_cast<int64_t>(state_b) * C + c) = cur2;
    }
  }
}

__global__ void update_shift_state_last_kernel(
    int T, int C, const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    const int* __restrict__ slot_indices, int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int b = static_cast<int>(pair_idx / c_pairs);
  const int c = static_cast<int>(pair_idx - static_cast<int64_t>(b) * c_pairs)
                << 1;
  const int state_b = slot_indices == nullptr ? b : slot_indices[b];
  const int64_t src_idx = (static_cast<int64_t>(b) * T + (T - 1)) * C + c;
  *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(state_b) * C +
                              c) = load_h2(x + src_idx);
}

__global__ void tmix_mix6_varlen_kernel(
    int C, const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    const int* __restrict__ slot_indices, const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w, const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v, const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g, dtype* __restrict__ out_r,
    dtype* __restrict__ out_w, dtype* __restrict__ out_k,
    dtype* __restrict__ out_v, dtype* __restrict__ out_a,
    dtype* __restrict__ out_g, const int* __restrict__ query_start_loc,
    const int* __restrict__ req_id, int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int64_t token = pair_idx / c_pairs;
  const int c = static_cast<int>(pair_idx - token * c_pairs) << 1;
  const int b = req_id[token];
  const int state_b = slot_indices[b];
  const int token_start = query_start_loc[b];
  const int64_t idx = token * C + c;

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 =
      (token == token_start)
          ? load_h2(shift_state + static_cast<int64_t>(state_b) * C + c)
          : load_h2(x + idx - C);

  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float dx0 = prev.x - cur.x;
  const float dx1 = prev.y - cur.y;

  const float2 xr = __half22float2(load_h2(x_r + c));
  const float2 xw = __half22float2(load_h2(x_w + c));
  const float2 xk = __half22float2(load_h2(x_k + c));
  const float2 xv = __half22float2(load_h2(x_v + c));
  const float2 xa = __half22float2(load_h2(x_a + c));
  const float2 xg = __half22float2(load_h2(x_g + c));

  store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
  store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
  store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
  store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
  store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
  store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);
}

__global__ void update_shift_state_varlen_kernel(
    int B, int C, const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    const int* __restrict__ slot_indices,
    const int* __restrict__ query_start_loc, int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int b = static_cast<int>(pair_idx / c_pairs);
  if (b >= B) {
    return;
  }
  const int c = static_cast<int>(pair_idx - static_cast<int64_t>(b) * c_pairs)
                << 1;
  const int last_token = query_start_loc[b + 1] - 1;
  const int state_b = slot_indices[b];
  *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(state_b) * C +
                              c) =
      load_h2(x + static_cast<int64_t>(last_token) * C + c);
}

__global__ void tmix_kk_a_gate_kernel(
    int H, const dtype* __restrict__ k, const dtype* __restrict__ k_k,
    const dtype* __restrict__ a0, const dtype* __restrict__ a12,
    const dtype* __restrict__ k_a, dtype* __restrict__ new_k,
    dtype* __restrict__ neg_kk, dtype* __restrict__ kka, int64_t bth_size) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t bth = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (bth >= bth_size) {
    return;
  }

  const int64_t h = bth % H;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t c = h * HEAD_SIZE + static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + static_cast<int64_t>(lane) * 2;

  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 kk_scale = __half22float2(load_h2(k_k + c));
  const float u0 = kv.x * kk_scale.x;
  const float u1 = kv.y * kk_scale.y;

  float sum_sq = u0 * u0 + u1 * u1;
  sum_sq = warp_sum(sum_sq);
  const float total = __shfl_sync(0xffffffffu, sum_sq, 0);
  const float inv_d = 1.0f / fmaxf(sqrtf(total), KK_NORMALIZE_EPS);
  const float kk0 = u0 * inv_d;
  const float kk1 = u1 * inv_d;

  const float2 a0v = __half22float2(load_h2(a0 + c));
  const float2 a12v = __half22float2(load_h2(a12 + idx));
  const float av0 = sigmoid_fast(a0v.x + a12v.x);
  const float av1 = sigmoid_fast(a0v.y + a12v.y);
  const float2 ka = __half22float2(load_h2(k_a + c));
  store_h2(new_k + idx, kv.x * fmaf(av0, ka.x, 1.0f - ka.x),
           kv.y * fmaf(av1, ka.y, 1.0f - ka.y));
  store_h2(neg_kk + idx, -kk0, -kk1);
  store_h2(kka + idx, kk0 * av0, kk1 * av1);
}

__global__ void tmix_lnx_rkvres_xg_kernel(
    int C, int H, const dtype* __restrict__ x, const dtype* __restrict__ r,
    const dtype* __restrict__ k, const dtype* __restrict__ v,
    const dtype* __restrict__ r_k, const dtype* __restrict__ weight,
    const dtype* __restrict__ bias, const dtype* __restrict__ g,
    dtype* __restrict__ out, int64_t bth_size) {
  __shared__ float partial[2];
  const int bth = blockIdx.x;
  if (bth >= bth_size) {
    return;
  }
  const int lane = threadIdx.x;
  const int warp = lane >> 5;
  const int warp_lane = lane & 31;
  const int h = bth % H;
  const int64_t base = static_cast<int64_t>(bth) * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t idx = base + lane;
  const int64_t c = cbase + lane;

  const float xv = load_h1(x + idx);
  float sum = xv;
  sum = warp_sum(sum);
  if (warp_lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float mean = (partial[0] + partial[1]) * (1.0f / 64.0f);
  __syncthreads();

  const float d = xv - mean;
  float ss = d * d;
  ss = warp_sum(ss);
  if (warp_lane == 0) {
    partial[warp] = ss;
  }
  __syncthreads();
  const float var = (partial[0] + partial[1]) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);
  __syncthreads();

  const float rv = load_h1(r + idx);
  const float kv = load_h1(k + idx);
  const float vv = load_h1(v + idx);
  float dot = rv * kv * load_h1(r_k + c);
  dot = warp_sum(dot);
  if (warp_lane == 0) {
    partial[warp] = dot;
  }
  __syncthreads();
  const float rkv = partial[0] + partial[1];
  __syncthreads();

  const float y =
      (d * rstd * load_h1(weight + c) + load_h1(bias + c) + rkv * vv) *
      load_h1(g + idx);
  store_h1(out + idx, y);
}

__global__ void tmix_vres_gate_kernel(int C, const dtype* __restrict__ v,
                                      const dtype* __restrict__ v_first,
                                      const dtype* __restrict__ v0,
                                      const dtype* __restrict__ v12,
                                      dtype* __restrict__ out, int64_t total) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int c = static_cast<int>(idx % static_cast<int64_t>(C));
  const float vv = load_h1(v + idx);
  const float gate = sigmoid_fast(load_h1(v0 + c) + load_h1(v12 + idx));
  store_h1(out + idx, fmaf(load_h1(v_first + idx) - vv, gate, vv));
}

__global__ void zero_vec4_kernel(dtype* __restrict__ out, int64_t n_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
}

template <bool UpdateShift, bool Grid3D = false>
__global__ void cmix_mix_kernel(int T, int C, const dtype* __restrict__ x,
                                dtype* __restrict__ shift_state,
                                const int* __restrict__ slot_indices,
                                const dtype* __restrict__ x_k,
                                dtype* __restrict__ out, int64_t total_pairs) {
  const int c_pairs = C >> 1;
  int64_t idx;
  int c;
  int b;
  int t;
  if constexpr (Grid3D) {
    const int c_pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c_pair >= c_pairs) {
      return;
    }
    b = static_cast<int>(blockIdx.z);
    t = static_cast<int>(blockIdx.y);
    c = c_pair << 1;
    idx = (static_cast<int64_t>(b) * T + t) * C + c;
  } else {
    const int64_t pair_idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
      return;
    }
    const int64_t bt = pair_idx / c_pairs;
    c = static_cast<int>(pair_idx - bt * c_pairs) << 1;
    b = static_cast<int>(bt / T);
    t = static_cast<int>(bt - static_cast<int64_t>(b) * T);
    idx = bt * C + c;
  }
  const int state_b = slot_indices == nullptr ? b : slot_indices[b];

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 =
      (t == 0) ? load_h2(shift_state + static_cast<int64_t>(state_b) * C + c)
               : load_h2(x + idx - C);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float2 mix = __half22float2(load_h2(x_k + c));
  store_h2(out + idx, cur.x + (prev.x - cur.x) * mix.x,
           cur.y + (prev.y - cur.y) * mix.y);

  if constexpr (UpdateShift) {
    if (t == T - 1) {
      *reinterpret_cast<__half2*>(shift_state +
                                  static_cast<int64_t>(state_b) * C + c) = cur2;
    }
  }
}

__global__ void cmix_mix_varlen_kernel(
    int C, const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    const int* __restrict__ slot_indices, const dtype* __restrict__ x_k,
    dtype* __restrict__ out, const int* __restrict__ query_start_loc,
    const int* __restrict__ req_id, int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int64_t token = pair_idx / c_pairs;
  const int c = static_cast<int>(pair_idx - token * c_pairs) << 1;
  const int b = req_id[token];
  const int state_b = slot_indices[b];
  const int token_start = query_start_loc[b];
  const int64_t idx = token * C + c;

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 =
      (token == token_start)
          ? load_h2(shift_state + static_cast<int64_t>(state_b) * C + c)
          : load_h2(x + idx - C);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float2 mix = __half22float2(load_h2(x_k + c));
  store_h2(out + idx, cur.x + (prev.x - cur.x) * mix.x,
           cur.y + (prev.y - cur.y) * mix.y);
}

__global__ void relu_square_kernel(const dtype* __restrict__ x,
                                   dtype* __restrict__ out,
                                   int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  const float x0 = fmaxf(v.x, 0.0f);
  const float x1 = fmaxf(v.y, 0.0f);
  store_h2(out + idx, x0 * x0, x1 * x1);
}

__global__ void act_tanh_kernel(const dtype* __restrict__ x,
                                dtype* __restrict__ out, int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  store_h2(out + idx, tanhf(v.x), tanhf(v.y));
}

__global__ void act_sigmoid_kernel(const dtype* __restrict__ x,
                                   dtype* __restrict__ out,
                                   int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  store_h2(out + idx, sigmoid_fast(v.x), sigmoid_fast(v.y));
}

__global__ void add_vec_kernel(int C, const dtype* __restrict__ x,
                               const dtype* __restrict__ vec,
                               dtype* __restrict__ out, int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int c = static_cast<int>((pair_idx % (C >> 1)) << 1);
  const int64_t idx = pair_idx * 2;
  const float2 xv = __half22float2(load_h2(x + idx));
  const float2 vv = __half22float2(load_h2(vec + c));
  store_h2(out + idx, xv.x + vv.x, xv.y + vv.y);
}

__global__ __launch_bounds__(
    FFN_SPMV_THREADS,
    4) void cmix_sparse_spmv_relu_one_kernel(int C,
                                             const dtype* __restrict__ preact,
                                             const dtype* __restrict__ value_fc,
                                             dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;

  if (tid < FFN_TILE) {
    const float v = fmaxf(load_h1(preact + start_f + tid), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C +
        c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(reinterpret_cast<__half2*>(out + c_block * (2 * FFN_SPMV_THREADS) +
                                       tid * 2),
            acc);
}

__global__
__launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_rows_kernel(
    int C, int F, const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc, dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

  if (tid < FFN_TILE) {
    const float v = fmaxf(load_h1(pre_row + start_f + tid), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C +
        c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C +
                                 c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

__global__
__launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_kernel(
    int C, int F, const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc, dtype* __restrict__ out) {
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const float v = fmaxf(load_h1(pre_row + start_f + local_f), 0.0f);
    vec_slice[local_f] = __float2half_rn(v * v);
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    if (lane == 0) {
      warp_counts[warp_id + u * (THREADS / 32)] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * (THREADS / 32);
    if (nonzero) {
      nnz_ids[warp_prefix[group] + local_pos] = local_f;
    }
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int local_f = nnz_ids[i];
    const int actual_f = start_f + local_f;
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C +
        c_block * (2 * THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[local_f]), mat, acc);
  }
  atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C +
                                       c_block * (2 * THREADS) + tid * 2),
            acc);
}

}  // namespace

std::vector<at::Tensor> tmix_mix6_slot_cuda(int B, int T, int C, at::Tensor x,
                                            at::Tensor shift_state,
                                            at::Tensor slot_indices,
                                            at::Tensor x_r, at::Tensor x_w,
                                            at::Tensor x_k, at::Tensor x_v,
                                            at::Tensor x_a, at::Tensor x_g);

std::vector<at::Tensor> tmix_mix6_varlen_cuda(
    int B, int total_tokens, int C, at::Tensor x, at::Tensor shift_state,
    at::Tensor slot_indices, at::Tensor x_r, at::Tensor x_w, at::Tensor x_k,
    at::Tensor x_v, at::Tensor x_a, at::Tensor x_g, at::Tensor query_start_loc,
    at::Tensor req_id);

at::Tensor cmix_mix_slot_cuda(int B, int T, int C, at::Tensor x,
                              at::Tensor shift_state, at::Tensor slot_indices,
                              at::Tensor x_k);

at::Tensor cmix_mix_varlen_cuda(int B, int total_tokens, int C, at::Tensor x,
                                at::Tensor shift_state, at::Tensor slot_indices,
                                at::Tensor x_k, at::Tensor query_start_loc,
                                at::Tensor req_id);

std::vector<at::Tensor> tmix_mix6_cuda(int B, int T, int C, at::Tensor x,
                                       at::Tensor shift_state, at::Tensor x_r,
                                       at::Tensor x_w, at::Tensor x_k,
                                       at::Tensor x_v, at::Tensor x_a,
                                       at::Tensor x_g) {
  at::Tensor slot_indices;
  return tmix_mix6_slot_cuda(B, T, C, x, shift_state, slot_indices, x_r, x_w,
                             x_k, x_v, x_a, x_g);
}

std::vector<at::Tensor> tmix_mix6_slot_cuda(int B, int T, int C, at::Tensor x,
                                            at::Tensor shift_state,
                                            at::Tensor slot_indices,
                                            at::Tensor x_r, at::Tensor x_w,
                                            at::Tensor x_k, at::Tensor x_v,
                                            at::Tensor x_a, at::Tensor x_g) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* slot_ptr = slot_indices.defined() && slot_indices.numel() > 0
                            ? slot_indices.data_ptr<int>()
                            : nullptr;
  if (use_3d_mix(B, T, C)) {
    const dim3 grid(static_cast<unsigned int>(ceil_div(C / 2, threads)),
                    static_cast<unsigned int>(T), static_cast<unsigned int>(B));
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    if (T == 1) {
      tmix_mix6_kernel<true, true><<<grid, threads, 0, stream>>>(
          T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
          x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
          x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
          out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
          out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
          0);
    } else {
      tmix_mix6_kernel<false, true><<<grid, threads, 0, stream>>>(
          T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
          x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
          x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
          out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
          out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
          0);
      update_shift_state_last_kernel<<<
          static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
          T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
          state_pairs);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_r, out_w, out_k, out_v, out_a, out_g};
  }
  if (T == 1) {
    tmix_mix6_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)),
                             threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(), total_pairs);
  } else {
    tmix_mix6_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)),
                              threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(), total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<
        static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
        state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> tmix_mix6_varlen_cuda(
    int B, int total_tokens, int C, at::Tensor x, at::Tensor shift_state,
    at::Tensor slot_indices, at::Tensor x_r, at::Tensor x_w, at::Tensor x_k,
    at::Tensor x_v, at::Tensor x_a, at::Tensor x_g, at::Tensor query_start_loc,
    at::Tensor req_id) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(total_tokens) * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_mix6_varlen_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)),
                            threads, 0, stream>>>(
      C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      slot_indices.data_ptr<int>(), x_r.data_ptr<dtype>(),
      x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(),
      x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(), out_r.data_ptr<dtype>(),
      out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(),
      out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
      query_start_loc.data_ptr<int>(), req_id.data_ptr<int>(), total_pairs);
  const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
  update_shift_state_varlen_kernel<<<
      static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
      B, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      slot_indices.data_ptr<int>(), query_start_loc.data_ptr<int>(),
      state_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> tmix_kk_a_gate_cuda(int B, int T, int C, int H,
                                            at::Tensor k, at::Tensor k_k,
                                            at::Tensor a0, at::Tensor a12,
                                            at::Tensor k_a) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto new_k = at::empty_like(k);
  auto neg_kk = at::empty_like(k);
  auto kka = at::empty_like(k);
  const int64_t bth_size = static_cast<int64_t>(B) * T * H;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(
      ceil_div(bth_size, static_cast<int64_t>(WARPS_PER_BLOCK)));
  tmix_kk_a_gate_kernel<<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
      H, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(),
      a12.data_ptr<dtype>(), k_a.data_ptr<dtype>(), new_k.data_ptr<dtype>(),
      neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>(), bth_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {new_k, neg_kk, kka};
}

at::Tensor tmix_lnx_rkvres_xg_cuda(int B, int T, int C, int H, at::Tensor x,
                                   at::Tensor r, at::Tensor k, at::Tensor v,
                                   at::Tensor r_k, at::Tensor weight,
                                   at::Tensor bias, at::Tensor g) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto out = at::empty_like(x);
  const int64_t bth_size = static_cast<int64_t>(B) * T * H;
  TORCH_CHECK(bth_size <= static_cast<int64_t>(std::numeric_limits<int>::max()),
              "B*T*H (", bth_size, ") exceeds INT_MAX; cannot launch kernel");
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_lnx_rkvres_xg_kernel<<<static_cast<int>(bth_size), HEAD_SIZE, 0,
                              stream>>>(
      C, H, x.data_ptr<dtype>(), r.data_ptr<dtype>(), k.data_ptr<dtype>(),
      v.data_ptr<dtype>(), r_k.data_ptr<dtype>(), weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(), g.data_ptr<dtype>(), out.data_ptr<dtype>(),
      bth_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor tmix_vres_gate_cuda(int B, int T, int C, at::Tensor v,
                               at::Tensor v_first, at::Tensor v0,
                               at::Tensor v12) {
  auto out = at::empty_like(v);
  const int64_t total = static_cast<int64_t>(B) * T * C;
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_vres_gate_kernel<<<static_cast<int>(ceil_div(total, threads)), threads,
                          0, stream>>>(
      C, v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(),
      v12.data_ptr<dtype>(), out.data_ptr<dtype>(), total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_one_cuda(int C, int F, at::Tensor preact,
                                          at::Tensor value_fc) {
  auto out = at::empty({1, 1, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  zero_vec4_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(
      out.data_ptr<dtype>(), C / 8);
  cmix_sparse_spmv_relu_one_kernel<<<dim3(F / FFN_TILE,
                                          C / (2 * FFN_SPMV_THREADS), 1),
                                     FFN_SPMV_THREADS, 0, stream>>>(
      C, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_cuda(int B, int T, int C, int F,
                                           at::Tensor preact,
                                           at::Tensor value_fc) {
  const int rows = B * T;
  auto out = at::empty({B, T, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0,
                     stream>>>(out.data_ptr<dtype>(), out_vec4);
  cmix_sparse_spmv_relu_rows_kernel<<<dim3(F / FFN_TILE,
                                           C / (2 * FFN_SPMV_THREADS), rows),
                                      FFN_SPMV_THREADS, 0, stream>>>(
      C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_t512_cuda(int B, int T, int C, int F,
                                                at::Tensor preact,
                                                at::Tensor value_fc) {
  const int rows = B * T;
  auto out = at::empty({B, T, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0,
                     stream>>>(out.data_ptr<dtype>(), out_vec4);
  cmix_sparse_spmv_relu_rows_t512_kernel<<<dim3(F / 512, C / 512, rows), 256, 0,
                                           stream>>>(
      C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_mix_cuda(int B, int T, int C, at::Tensor x,
                         at::Tensor shift_state, at::Tensor x_k) {
  at::Tensor slot_indices;
  return cmix_mix_slot_cuda(B, T, C, x, shift_state, slot_indices, x_k);
}

at::Tensor cmix_mix_slot_cuda(int B, int T, int C, at::Tensor x,
                              at::Tensor shift_state, at::Tensor slot_indices,
                              at::Tensor x_k) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* slot_ptr = slot_indices.defined() && slot_indices.numel() > 0
                            ? slot_indices.data_ptr<int>()
                            : nullptr;
  if (use_3d_mix(B, T, C)) {
    const dim3 grid(static_cast<unsigned int>(ceil_div(C / 2, threads)),
                    static_cast<unsigned int>(T), static_cast<unsigned int>(B));
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    if (T == 1) {
      cmix_mix_kernel<true, true><<<grid, threads, 0, stream>>>(
          T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
          x_k.data_ptr<dtype>(), out.data_ptr<dtype>(), 0);
    } else {
      cmix_mix_kernel<false, true><<<grid, threads, 0, stream>>>(
          T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
          x_k.data_ptr<dtype>(), out.data_ptr<dtype>(), 0);
      update_shift_state_last_kernel<<<
          static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
          T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
          state_pairs);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
  if (T == 1) {
    cmix_mix_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)),
                            threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
        x_k.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  } else {
    cmix_mix_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)),
                             threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
        x_k.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<
        static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), slot_ptr,
        state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_mix_varlen_cuda(int B, int total_tokens, int C, at::Tensor x,
                                at::Tensor shift_state, at::Tensor slot_indices,
                                at::Tensor x_k, at::Tensor query_start_loc,
                                at::Tensor req_id) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(total_tokens) * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_mix_varlen_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)),
                           threads, 0, stream>>>(
      C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      slot_indices.data_ptr<int>(), x_k.data_ptr<dtype>(),
      out.data_ptr<dtype>(), query_start_loc.data_ptr<int>(),
      req_id.data_ptr<int>(), total_pairs);
  const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
  update_shift_state_varlen_kernel<<<
      static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
      B, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      slot_indices.data_ptr<int>(), query_start_loc.data_ptr<int>(),
      state_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor relu_square_cuda(at::Tensor x) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  relu_square_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)),
                       threads, 0, stream>>>(
      x.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor act_tanh_cuda(at::Tensor x) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  act_tanh_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads,
                    0, stream>>>(x.data_ptr<dtype>(), out.data_ptr<dtype>(),
                                 total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor act_sigmoid_cuda(at::Tensor x) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  act_sigmoid_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)),
                       threads, 0, stream>>>(
      x.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor add_vec_cuda(int C, at::Tensor x, at::Tensor vec) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_vec_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0,
                   stream>>>(C, x.data_ptr<dtype>(), vec.data_ptr<dtype>(),
                             out.data_ptr<dtype>(), total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
