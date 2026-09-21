// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// INT8 per-token-head decode attention for RDNA3, with the query heads that
// share a KV head batched into one wave.
//
// The v3 kernel gives each (query row, query head, split) its own wave, so with
// GQA every KV byte is read once per query head -- 6 times per rank at TP4 on
// Qwen3.5 (24 q heads / 4 kv heads) -- and each of those reads pays its own
// wave reduction. Measured in production at 167k context it spends 704 us per
// call moving 85 MiB of unique KV: 46 GB/s, far below the bus, because it is
// bound by instruction issue and not by memory.
//
// Here one wave carries HEADS_PER_WAVE query heads of the same KV head. K and V
// are loaded once for the group, so the per-token cost of loading and of the
// address arithmetic is amortised across the heads, and only the dot-product
// reduction stays per head.
//
// The mid_o layout is unchanged, so the existing stage-2 reduce still applies.

#include <cstdint>
#include <torch/all.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>

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

template <int CTRL>
__device__ __forceinline__ float dpp_add_f32(float v) {
  int y =
      __builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xF, 0xF, true);
  return v + __int_as_float(y);
}
__device__ __forceinline__ float wave_reduce_add32(float v) {
  v = dpp_add_f32<0xB1>(v);
  v = dpp_add_f32<0x4E>(v);
  v = dpp_add_f32<0x141>(v);
  v = dpp_add_f32<0x140>(v);
  // El paso cruzado-16 compila a ds_bpermute_b32 (via LDS) porque DPP16 no puede
  // cruzar filas de 16 carriles en gfx11.
  // ⛔ Probado y descartado: v_permlanex16_b32 hace el mismo intercambio en la ALU
  // y da el mismo resultado exacto, pero MEDIDO sale neutro o peor
  // (0,92x a 32k / 0,99x a 167k q=4 / 1,03x a 167k q=12). El acceso a LDS estaba
  // bien solapado; la reduccion pesa por los 4 DPP, no por el paso cruzado.
  v += __shfl_xor(v, 16);
  return v;
}

// Grid: (num_q, num_q_heads / HEADS_PER_WAVE, num_splits), Block: 32 threads.
template <int HEAD_SIZE, int HPW, typename QT, bool PREFETCH>
__global__ __launch_bounds__(32, 1) void decode_int8_stage1_gqa(
    const QT* __restrict__ Q, const int8_t* __restrict__ K_cache,
    const int8_t* __restrict__ V_cache, const float* __restrict__ K_scale,
    const float* __restrict__ V_scale, const int* __restrict__ block_table,
    const int* __restrict__ q_to_req, const int* __restrict__ q_to_klen,
    float* __restrict__ mid_o, float sm_scale, int num_q_heads,
    int num_kv_heads, int block_size, int max_blocks, int num_reqs,
    int num_phys_blocks, int num_splits, int64_t sq0, int64_t sq1, int64_t skb,
    int64_t sks, int64_t skh, int64_t svb, int64_t svs, int64_t svh,
    int64_t ssb, int64_t sss, int64_t ssh, int64_t svsb, int64_t svss,
    int64_t svsh, int64_t smo, int64_t smh, int64_t sms) {
  constexpr int DPT = HEAD_SIZE / 32;  // dims per thread, 8 for HS=256

  const int qi = blockIdx.x;
  const int h0 = blockIdx.y * HPW;  // first query head of this group
  const int si = blockIdx.z;
  const int tid = threadIdx.x;

  // Same defensive bounds as the v3 kernel: garbage metadata from a not-yet
  // landed H2D copy must stay harmless rather than page-fault or spin.
  int req = q_to_req[qi];
  int kv_len = q_to_klen[qi];
  if (req < 0 || req >= num_reqs) {
    req = 0;
    kv_len = 0;
  }
  const int max_kv = max_blocks * block_size;
  if (kv_len < 0) kv_len = 0;
  if (kv_len > max_kv) kv_len = max_kv;
  // Every head in the group shares this KV head by construction: HPW divides
  // num_q_heads / num_kv_heads, checked on the host.
  const int kvh = h0 / (num_q_heads / num_kv_heads);

  const int tps = (kv_len + num_splits - 1) / num_splits;
  const int start = si * tps;
  const int end = min(start + tps, kv_len);
  if (start >= kv_len) {
  #pragma unroll
    for (int g = 0; g < HPW; ++g) {
      float* out_ptr = mid_o + qi * smo + (h0 + g) * smh + si * sms;
      for (int d = 0; d < DPT; ++d) out_ptr[tid * DPT + d] = 0.0f;
      if (tid == 0) {
        out_ptr[HEAD_SIZE] = -INFINITY;
        out_ptr[HEAD_SIZE + 1] = 0.0f;
      }
    }
    return;
  }

  float q_vals[HPW][DPT];
  #pragma unroll
  for (int g = 0; g < HPW; ++g) {
    const QT* qr = Q + qi * sq0 + (h0 + g) * sq1;
  #pragma unroll
    for (int d = 0; d < DPT; ++d)
      q_vals[g][d] = to_float<QT>(qr[tid * DPT + d]);
  }

  const float sm_scale_log2 = sm_scale * 1.4426950408889634f;

  float m_state[HPW], l_state[HPW], o_vals[HPW][DPT];
  #pragma unroll
  for (int g = 0; g < HPW; ++g) {
    m_state[g] = -INFINITY;
    l_state[g] = 0.0f;
  #pragma unroll
    for (int d = 0; d < DPT; ++d) o_vals[g][d] = 0.0f;
  }

  int kv = start;
  while (kv < end) {
    const int lb = kv / block_size;
    const int slot0 = kv - lb * block_size;
    int pb = block_table[req * max_blocks + lb];
    if (pb < 0 || pb >= num_phys_blocks) pb = 0;
    int n = block_size - slot0;
    if (n > end - kv) n = end - kv;

    const int8_t* kp = K_cache + pb * skb + kvh * skh + tid * DPT +
                       (int64_t)slot0 * sks;
    const int8_t* vp = V_cache + pb * svb + kvh * svh + tid * DPT +
                       (int64_t)slot0 * svs;
    const float* ksp = K_scale + pb * ssb + kvh * ssh + (int64_t)slot0 * sss;
    const float* vsp = V_scale + pb * svsb + kvh * svsh + (int64_t)slot0 * svss;

    // Software pipeline: the loop reads 8 bytes of K and 8 of V per thread per
    // token, and consecutive tokens are `sks` apart, so every token is its own
    // dependent fetch. At hpw=6 there are only ~1024 waves for 192 SIMDs, not
    // enough to hide that latency by occupancy alone: measured 492 cycles per
    // token-iteration where the instruction count only accounts for ~160, i.e.
    // the SIMD idles two thirds of the time. So issue the NEXT token's loads
    // before computing the current one.
    int8_t nk[DPT], nv[DPT];
    float nks = 0.0f, nvs = 0.0f;
    if (PREFETCH) {
  #pragma unroll
      for (int d = 0; d < DPT; ++d) nk[d] = kp[d];
  #pragma unroll
      for (int d = 0; d < DPT; ++d) nv[d] = vp[d];
      nks = *ksp;
      nvs = *vsp;
    }
    for (int sl = 0; sl < n; ++sl) {
      // One load of K and V serves every head in the group: this is the whole
      // point of the kernel.
      float kv_k[DPT], kv_v[DPT];
      float ks, vs;
      if (PREFETCH) {
  #pragma unroll
        for (int d = 0; d < DPT; ++d) kv_k[d] = (float)nk[d];
  #pragma unroll
        for (int d = 0; d < DPT; ++d) kv_v[d] = (float)nv[d];
        ks = nks;
        vs = nvs;
        if (sl + 1 < n) {
          const int8_t* nkp = kp + sks;
          const int8_t* nvp = vp + svs;
  #pragma unroll
          for (int d = 0; d < DPT; ++d) nk[d] = nkp[d];
  #pragma unroll
          for (int d = 0; d < DPT; ++d) nv[d] = nvp[d];
          nks = *(ksp + sss);
          nvs = *(vsp + svss);
        }
      } else {
  #pragma unroll
        for (int d = 0; d < DPT; ++d) kv_k[d] = (float)kp[d];
  #pragma unroll
        for (int d = 0; d < DPT; ++d) kv_v[d] = (float)vp[d];
        ks = *ksp;
        vs = *vsp;
      }

  #pragma unroll
      for (int g = 0; g < HPW; ++g) {
        float partial = 0.0f;
  #pragma unroll
        for (int d = 0; d < DPT; ++d) partial += q_vals[g][d] * kv_k[d];
        partial = wave_reduce_add32(partial);

        float score = partial * ks * sm_scale_log2;
        // The running max only grows O(log n) times over n tokens, and `partial`
        // comes out of a wave reduction so `score` is wave-uniform: this branch
        // is uniform, never divergent, and it drops one v_exp_f32 (quarter rate
        // on RDNA3) plus the DPT rescale multiplies for every token that does
        // not raise the max -- almost all of them.
        if (score > m_state[g]) {
          float alpha = exp2f(m_state[g] - score);
          l_state[g] *= alpha;
  #pragma unroll
          for (int d = 0; d < DPT; ++d) o_vals[g][d] *= alpha;
          m_state[g] = score;
        }
        float p = exp2f(score - m_state[g]);
        l_state[g] += p;
        float p_vs = p * vs;
  #pragma unroll
        for (int d = 0; d < DPT; ++d) o_vals[g][d] += p_vs * kv_v[d];
      }

      kp += sks;
      vp += svs;
      ksp += sss;
      vsp += svss;
    }
    kv += n;
  }

  #pragma unroll
  for (int g = 0; g < HPW; ++g) {
    float* out_ptr = mid_o + qi * smo + (h0 + g) * smh + si * sms;
  #pragma unroll
    for (int d = 0; d < DPT; ++d) out_ptr[tid * DPT + d] = o_vals[g][d];
    if (tid == 0) {
      out_ptr[HEAD_SIZE] = m_state[g];
      out_ptr[HEAD_SIZE + 1] = l_state[g];
    }
  }
}

// Stage 2: same mid_o layout and same maths as the v3 reduce, but it stops at
// the last split that actually has data.
//
// Stage 1 gives each split ceil(kv_len / num_splits) tokens, so a short context
// leaves most splits empty: at kv_len = 40 with 256 splits only 40 carry
// anything, and the v3 reduce still walks all 256 twice. Measured in production
// at a 40-token context that reduce costs 44.9 us against 5.2 us for stage 1
// itself -- 8.6x the work it is reducing. The empty splits contribute exactly
// nothing (stage 1 writes -INFINITY, whose weight is zeroed), so skipping them
// changes no result.
template <int HEAD_SIZE, typename OT>
__global__ void decode_int8_reduce_gqa(const float* __restrict__ mid_o,
                                       const int* __restrict__ q_to_klen,
                                       OT* __restrict__ out, int num_splits,
                                       int num_reqs, int max_kv,
                                       const int* __restrict__ q_to_req,
                                       int64_t smo, int64_t smh, int64_t sms,
                                       int64_t soo, int64_t soh) {
  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int tid = threadIdx.x;

  // Same bounds as stage 1, for the same reason: garbage metadata must stay
  // harmless. A clamped kv_len only ever shortens the loop.
  int req = q_to_req[qi];
  int kv_len = q_to_klen[qi];
  if (req < 0 || req >= num_reqs) kv_len = 0;
  if (kv_len < 0) kv_len = 0;
  if (kv_len > max_kv) kv_len = max_kv;
  const int tps = (kv_len + num_splits - 1) / num_splits;
  int eff = tps > 0 ? (kv_len + tps - 1) / tps : 0;
  if (eff > num_splits) eff = num_splits;
  if (eff < 1) eff = 1;
  num_splits = eff;

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
    float a = (ms == -INFINITY) ? 0.0f : exp2f(ms - m_global);
    o0 += sp[2 * tid] * a;
    o1 += sp[2 * tid + 1] * a;
    l_global += ls * a;
  }
  float inv_l = 1.0f / (l_global + 1e-10f);
  out[qi * soo + hi * soh + 2 * tid] = from_float<OT>(o0 * inv_l);
  out[qi * soo + hi * soh + 2 * tid + 1] = from_float<OT>(o1 * inv_l);
}

void pth_decode_int8_gqa(torch::Tensor out, torch::Tensor query,
                         torch::Tensor key_cache, torch::Tensor value_cache,
                         torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
                         torch::Tensor block_table, torch::Tensor q_to_req,
                         torch::Tensor q_to_klen, torch::Tensor mid_o_buf,
                         double sm_scale, int64_t num_kv_splits,
                         int64_t heads_per_wave) {
  const int num_q = query.size(0);
  const int num_q_heads = query.size(1);
  const int head_size = query.size(2);
  const int num_kv_heads = k_scale_cache.size(2);
  const int block_size = key_cache.size(1);
  const int max_blocks = block_table.size(1);
  const int num_reqs = block_table.size(0);
  const int num_phys_blocks = key_cache.size(0);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(head_size == 256, "only head_size=256 supported");
  TORCH_CHECK(query.dtype() == at::kHalf || query.dtype() == at::kBFloat16);
  TORCH_CHECK(out.dtype() == query.dtype());
  TORCH_CHECK(key_cache.dtype() == at::kChar);

  const int ratio = num_q_heads / num_kv_heads;
  // heads_per_wave >= 100 pide la variante con prefetch (100 + hpw), para poder
  // medir las dos con el mismo binario.
  const bool want_pf = heads_per_wave >= 100;
  const int hpw = (int)(heads_per_wave % 100);
  TORCH_CHECK(hpw >= 1 && ratio % hpw == 0 && num_q_heads % hpw == 0,
              "heads_per_wave must divide num_q_heads/num_kv_heads (", ratio,
              "), got ", hpw);

  int ns = (int)num_kv_splits;
  constexpr int HS = 256;
  constexpr int TH = HS / 2;

  dim3 grid1(num_q, num_q_heads / hpw, ns);
  dim3 grid2(num_q, num_q_heads);

  #define LAUNCH(QT, OT, HPW)                                                  \
    decode_int8_stage1_gqa<HS, HPW, QT, PF><<<grid1, dim3(32), 0, stream>>>(       \
        (const QT*)query.data_ptr(), (const int8_t*)key_cache.data_ptr(),      \
        (const int8_t*)value_cache.data_ptr(),                                 \
        (const float*)k_scale_cache.data_ptr(),                                \
        (const float*)v_scale_cache.data_ptr(),                                \
        (const int*)block_table.data_ptr(), (const int*)q_to_req.data_ptr(),   \
        (const int*)q_to_klen.data_ptr(), (float*)mid_o_buf.data_ptr(),        \
        (float)sm_scale, num_q_heads, num_kv_heads, block_size, max_blocks,    \
        num_reqs, num_phys_blocks, ns, query.stride(0), query.stride(1),       \
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),         \
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),   \
        k_scale_cache.stride(0), k_scale_cache.stride(1),                      \
        k_scale_cache.stride(2), v_scale_cache.stride(0),                      \
        v_scale_cache.stride(1), v_scale_cache.stride(2), mid_o_buf.stride(0), \
        mid_o_buf.stride(1), mid_o_buf.stride(2));                             \
    decode_int8_reduce_gqa<HS, OT><<<grid2, dim3(TH), 0, stream>>>(            \
        (const float*)mid_o_buf.data_ptr(), (const int*)q_to_klen.data_ptr(),  \
        (OT*)out.data_ptr(), ns, num_reqs, max_blocks * block_size,            \
        (const int*)q_to_req.data_ptr(), mid_o_buf.stride(0),                  \
        mid_o_buf.stride(1), mid_o_buf.stride(2), out.stride(0),               \
        out.stride(1));

  #define DISPATCH_PF(QT, OT, HPW)                                             \
    if (want_pf) { constexpr bool PF = true; LAUNCH(QT, OT, HPW); }             \
    else { constexpr bool PF = false; LAUNCH(QT, OT, HPW); }

  #define DISPATCH_HPW(QT, OT)      \
    switch (hpw) {                  \
      case 1: DISPATCH_PF(QT, OT, 1); break;  \
      case 2: DISPATCH_PF(QT, OT, 2); break;  \
      case 3: DISPATCH_PF(QT, OT, 3); break;  \
      case 4: DISPATCH_PF(QT, OT, 4); break;  \
      case 6: DISPATCH_PF(QT, OT, 6); break;  \
      case 8: DISPATCH_PF(QT, OT, 8); break;  \
      default: TORCH_CHECK(false, "unsupported heads_per_wave ", hpw);         \
    }

  if (query.dtype() == at::kHalf) {
    DISPATCH_HPW(half, half);
  } else {
    DISPATCH_HPW(__hip_bfloat16, __hip_bfloat16);
  }
  #undef DISPATCH_HPW
  #undef DISPATCH_PF
  #undef LAUNCH
}

#else

void pth_decode_int8_gqa(torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, double, int64_t, int64_t) {
  TORCH_CHECK(false, "pth_decode_int8_gqa requires ROCm");
}

#endif

// Guardado para poder volcar el ensamblador con --offload-device-only, que no
// digiere el binding: -DSKIP_PYBIND.
#ifndef SKIP_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pth_decode_int8_gqa", &pth_decode_int8_gqa,
        "INT8 per-token-head decode attention with GQA head batching");
}
#endif
