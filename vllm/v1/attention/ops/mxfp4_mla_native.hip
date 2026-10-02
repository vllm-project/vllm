// Native FP8 x FP4 sparse MLA decode for gfx950.
//
// Both dots run on v_mfma_scale_f32_16x16x128_f8f6f4. K and V are the packed
// 272-byte rows as stored; Q and the softmax weights are E4M3. Accumulation is
// fp32 because every MFMA on this target accumulates in f32.
//
// Operand layout, measured by tests/v1/attention/hip_mfma_f8f6f4_probe.hip
// (lane l, c = l/16):
//   FP4 A: row l%16, k = 32c + n, low nibble first     (= our packed row)
//   FP8 B: col l%16, k = 16c + n%16 + 64*(n/16)
//   scale: lane (row|col) + 16*block supplies that 32-wide block's E8M0 byte
//   D:     lane holds D[4c + i][l%16]
//
// Three kernels per launch slice:
//   quantize_q  Q bf16 -> E4M3 + per-head scale, once per row
//   split       one workgroup = 4 waves = `chunks_per_split` consecutive
//               128-token chunks of a row's topk, all 16 heads, online softmax
//               across its chunks; writes an unnormalized partial (acc, max, sum)
//   reduce      log-sum-exp combine of a row's splits
// The split count adapts to the batch: 16 splits of one chunk each at decode
// sizes, where parallelism is short, down to one split of 16 chunks at prefill
// sizes, where 16 fp32 partials per row were more traffic than the KV read.
//
// Within a split workgroup, wave w does QK for token tiles 2w, 2w+1 and PV for
// latent groups 4w..4w+3 (tiles 8w..8w+7); the softmax statistics are combined
// across waves through LDS. Splitting PV by latent is what keeps each wave's
// accumulators at 32 registers instead of 128.
//
//   QK  A = K codes  [16 tokens x 128 latent], scale_a = K's E8M0 bytes
//       B = Q E4M3   [128 latent x 16 heads], identity scale
//   PV  A = V^T codes [16 latent x 128 tokens], by ds_read_b64_tr_b4
//       B = E4M3(p * s_V[:, g]) [128 tokens x 16 heads], one per latent group g
//
// PV sums over tokens but V's scales run along the latent, so they cannot use
// the scale slot; they are folded into p per group. E8M0 is a power of two, so
// the fold is exact up to E4M3 range.
//
// Every loop that indexes a register array is fully unrolled: one that was
// not put the accumulators in scratch (528 bytes/lane) and cost ~2x.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int v8i;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float v4f;
typedef __attribute__((__vector_size__(2 * sizeof(int)))) int v2i;
typedef __attribute__((address_space(3))) v2i lds_v2i;
typedef __attribute__((address_space(3))) void lds_void;
typedef __attribute__((address_space(1))) void glb_void;

#define LATENT 512
#define ROW_BYTES 272
#define SCALE_OFF 256
#define NGROUPS 16
#define HEADS 16
#define TOK 128
#define WAVES 4
#define THREADS (64 * WAVES)
#define ROW_U4 (ROW_BYTES / 16)

#define FMT_E4M3 0
#define FMT_E2M1 4
#define E8M0_ONE 127

__device__ inline float bf16_lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ inline float bf16_hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }

// 16 bf16 values at p -> the 16 floats, in order.
__device__ inline void load16(const __hip_bfloat16* p, float* v) {
  const uint4 a = *(const uint4*)p, b = *(const uint4*)(p + 8);
  const unsigned w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) { v[2 * i] = bf16_lo(w[i]); v[2 * i + 1] = bf16_hi(w[i]); }
}

// 16 floats -> 4 dwords of E4M3, 4 per dword, first value in the low byte.
__device__ inline void pack16(const float* v, float scale, unsigned* out) {
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    float x[4];
    // Correctly rounded, like torch's tensor / tensor: the default device
    // division is not, and one-step code flips in Q were measured to move a
    // sharply peaked head by 2.6%.
#pragma unroll
    for (int e = 0; e < 4; ++e)
      x[e] = fminf(fmaxf(__fdiv_rn(v[4 * d + e], scale), -448.f), 448.f);
    int w = __builtin_amdgcn_cvt_pk_fp8_f32(x[0], x[1], 0, false);
    out[d] = __builtin_amdgcn_cvt_pk_fp8_f32(x[2], x[3], w, true);
  }
}

// Q -> E4M3 with a per-head absmax scale, identical to
// mxfp4_mla_native.quantize_q, stored in latent order. One wave per row: lane
// (head r, block c) covers latents 128t + 64h + 16c .. +16.
extern "C" __global__ __launch_bounds__(64) void mxfp4_mla_native_quantize_q(
    const __hip_bfloat16* __restrict__ qin, int64_t q_stride_t, int64_t q_stride_h,
    uint8_t* __restrict__ q8, float* __restrict__ qscale) {
  const int q = blockIdx.x, l = threadIdx.x, r = l & 15, c = l >> 4;
  const __hip_bfloat16* qrow = qin + (int64_t)q * q_stride_t + (int64_t)r * q_stride_h;
  float v[8][16];
  float amax = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    load16(qrow + 64 * k + 16 * c, v[k]);
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[k][i]));
  }
  amax = fmaxf(amax, __shfl_xor(amax, 16));
  amax = fmaxf(amax, __shfl_xor(amax, 32));
  // torch evaluates `amax / 448.0` (tensor / python scalar) as a multiply by
  // the fp32 reciprocal -- measured: 100% equal to x * (1/448), 47% equal to
  // true division -- so the scale is formed the same way.
  const float scale = amax > 0.f ? amax * (1.0f / 448.0f) : 1.f;
  uint8_t* row = q8 + ((int64_t)q * HEADS + r) * LATENT;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    unsigned w[4];
    pack16(v[k], scale, w);
    *(uint4*)(row + 64 * k + 16 * c) = make_uint4(w[0], w[1], w[2], w[3]);
  }
  if (c == 0) qscale[q * HEADS + r] = scale;
}

extern "C" __global__ __launch_bounds__(THREADS, 2) void mxfp4_mla_native_split(
    const uint8_t* __restrict__ q8,       // [nq, 16, 512] E4M3, from quantize_q
    const float* __restrict__ qscale,     // [nq, 16]
    const uint8_t* __restrict__ kv,       // [num_slots, 272]
    const int32_t* __restrict__ indices,  // ragged, [nnz]
    const int32_t* __restrict__ indptr,   // [nq + 1]
    float* __restrict__ part_acc,         // [nq, splits, 16, 512]
    float* __restrict__ part_ml,          // [nq, splits, 2, 16]
    int num_slots, int num_splits, int chunks_per_split, float sm_scale) {
  const int q = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
  const int w = tid >> 6, l = tid & 63, r = l & 15, c = l >> 4;
  const int kv_start = indptr[q];
  const int kv_len = indptr[q + 1] - kv_start;
  const int first = s * chunks_per_split * TOK;

  float* pacc = part_acc + ((int64_t)(q * num_splits + s) * HEADS + r) * LATENT;
  float* pml = part_ml + (int64_t)(q * num_splits + s) * 2 * HEADS;
  if (first >= kv_len) {
    // Nothing to attend in this split: an empty partial the reduce ignores.
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) pacc[16 * (8 * w + j) + 4 * c + i] = 0.f;
    if (w == 0 && c == 0) { pml[r] = -INFINITY; pml[HEADS + r] = 0.f; }
    return;
  }

  // Two tile buffers: chunk k+1 is copied global -> LDS by the hardware
  // (global_load_lds; measured: lane i lands at base + 16*i) while chunk k
  // computes. Serialized, the gather was 40% of the time at prefill scale.
  __shared__ __attribute__((aligned(16))) uint8_t lds_kv_buf[2][TOK * ROW_BYTES];
  __shared__ float lds_p[HEADS * TOK];
  __shared__ int lds_slot_buf[2][TOK];
  __shared__ float lds_red[2][WAVES][HEADS];

  // Q as the FP8 B operand of QK, held in registers: a global load inside the
  // chunk loop would wait on the in-flight tile copy (vmcnt retires in order).
  v8i qb[4];
  const uint8_t* qrow = q8 + ((int64_t)q * HEADS + r) * LATENT;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const uint4 lo = *(const uint4*)(qrow + 128 * t + 16 * c);
    const uint4 hi = *(const uint4*)(qrow + 128 * t + 64 + 16 * c);
    qb[t] = (v8i){(int)lo.x, (int)lo.y, (int)lo.z, (int)lo.w,
                  (int)hi.x, (int)hi.y, (int)hi.z, (int)hi.w};
  }
  const float qs = qscale[q * HEADS + r] * sm_scale;

  // Online softmax across this split's chunks. Every lane with the same r
  // holds the same (m_run, l_run) for head r; o holds this wave's latents.
  float m_run = -INFINITY, l_run = 0.f;
  v4f o[8];
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) o[jj] = (v4f){0.f, 0.f, 0.f, 0.f};

  auto slot_of = [&](int t0) {
    const int pos = t0 + tid;
    const int slot = (tid < TOK && pos < kv_len) ? indices[kv_start + pos] : -1;
    return (slot >= 0 && slot < num_slots) ? slot : -1;
  };
  // Invalid tokens copy slot 0's row rather than zeros (the copy cannot write
  // zeros); their scores are masked and their weights forced to exactly 0.
  auto issue_copy = [&](int buf) {
#pragma unroll
    for (int i = 0; i < (TOK * ROW_U4 + THREADS - 1) / THREADS; ++i) {
      const int base_u = i * THREADS + w * 64;  // wave-uniform
      if (base_u < TOK * ROW_U4) {
        const int u = base_u + l;
        const int slot = lds_slot_buf[buf][u / ROW_U4];
        const uint8_t* src =
            kv + (int64_t)(slot < 0 ? 0 : slot) * ROW_BYTES + (u % ROW_U4) * 16;
        __builtin_amdgcn_global_load_lds((glb_void*)src,
                                         (lds_void*)(lds_kv_buf[buf] + base_u * 16), 16, 0, 0);
      }
    }
  };
  if (tid < TOK) lds_slot_buf[0][tid] = slot_of(first);
  __syncthreads();
  issue_copy(0);
  int next_slot = slot_of(first + TOK);

  for (int ch = 0; ch < chunks_per_split; ++ch) {
    const int t0 = first + ch * TOK;
    if (t0 >= kv_len) break;  // uniform across the workgroup
    const int cur = ch & 1, nxt = cur ^ 1;
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");  // this chunk's copy
    __syncthreads();  // every copy has landed; chunk ch-1 is done with nxt
    if (tid < TOK) lds_slot_buf[nxt][tid] = next_slot;
    __syncthreads();
    if (ch + 1 < chunks_per_split && t0 + TOK < kv_len) {
      issue_copy(nxt);
      next_slot = slot_of(t0 + 2 * TOK);
    }
    const uint8_t* lds_kv = lds_kv_buf[cur];
    const int* lds_slot = lds_slot_buf[cur];
    const uint8_t* vt = lds_kv + (32 * c + r) * ROW_BYTES;

    // QK for this wave's two token tiles. Lane holds head r's scores for
    // tokens 16m + 4c + i.
    float sc[2][4];
#pragma unroll
    for (int mm = 0; mm < 2; ++mm) {
      const int m = 2 * w + mm;
      const uint8_t* row = lds_kv + (16 * m + r) * ROW_BYTES;
      v4f d = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const uint4 a4 = *(const uint4*)(row + 64 * t + 16 * c);
        const v8i a = {(int)a4.x, (int)a4.y, (int)a4.z, (int)a4.w, 0, 0, 0, 0};
        const int sa = row[SCALE_OFF + 4 * t + c];
        d = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, qb[t], d, FMT_E2M1, FMT_E4M3,
                                                             0, sa, 0, E8M0_ONE);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
        sc[mm][i] = lds_slot[16 * m + 4 * c + i] >= 0 ? d[i] * qs : -INFINITY;
    }

    // Chunk max for head r: within the wave across lanes r + 16c, then across
    // waves through LDS; then the running max.
    float mx = -INFINITY;
#pragma unroll
    for (int mm = 0; mm < 2; ++mm)
#pragma unroll
      for (int i = 0; i < 4; ++i) mx = fmaxf(mx, sc[mm][i]);
    mx = fmaxf(mx, __shfl_xor(mx, 16));
    mx = fmaxf(mx, __shfl_xor(mx, 32));
    if (c == 0) lds_red[0][w][r] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(lds_red[0][0][r], lds_red[0][1][r]),
               fmaxf(lds_red[0][2][r], lds_red[0][3][r]));
    const float m_new = fmaxf(m_run, mx);
    const float alpha = m_run == -INFINITY ? 0.f : __expf(m_run - m_new);
    // Weights are relative to the running max, as in the Triton emulation.
    float sum = 0.f;
#pragma unroll
    for (int mm = 0; mm < 2; ++mm)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float x = sc[mm][i];
        const float p = (m_new == -INFINITY || x == -INFINITY) ? 0.f : __expf(x - m_new);
        lds_p[r * TOK + 16 * (2 * w + mm) + 4 * c + i] = p;
        sum += p;
      }
    sum += __shfl_xor(sum, 16);
    sum += __shfl_xor(sum, 32);
    if (c == 0) lds_red[1][w][r] = sum;
    __syncthreads();
    l_run = l_run * alpha +
            (lds_red[1][0][r] + lds_red[1][1][r] + lds_red[1][2][r] + lds_red[1][3][r]);
    m_run = m_new;

    // PV for this wave's groups 4w..4w+3. The lane's B tokens are
    // t(n) = 16c + n%16 + 64*(n/16). Their weights and scale bytes are read
    // from LDS inside each group's rebuild rather than held in 64 registers,
    // which buys a third wave per SIMD.
    const float* prow = lds_p + r * TOK + 16 * c;
    const uint8_t* srow = lds_kv + 16 * c * ROW_BYTES + SCALE_OFF + 4 * w;
    // D's column is the head, so the rescale is per lane: head r.
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) o[jj] *= alpha;
    // V^T by hardware transpose, measured by tests/v1/attention/hip_ds_read_tr4_probe.hip:
    // in a 16-lane group, lane i gets nibble i of each lane's 64-bit load, in
    // lane order. Lane (c, r) loads row 32c + r at the tile's 8 bytes, so it
    // receives latent 16j + r of tokens 32c..32c+15 -- A's k = 32c + n.
#pragma unroll
    for (int gg = 0; gg < 4; ++gg) {
      v8i b;
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int n = 4 * d + e;
          const int t = (n & 15) + 64 * (n >> 4);  // relative to 16c
          // E8M0 byte as a float exponent field. Byte 0 (2^-127) becomes 0.0
          // here, which changes nothing: E4M3(p * 2^-127) is 0 either way.
          const float sv = __uint_as_float((unsigned)srow[t * ROW_BYTES + gg] << 23);
          // Masked tokens have p == 0 exactly; select rather than multiply so
          // a stray E8M0 255 (inf) in a substituted row cannot make 0*inf.
          const float pv = prow[t];
          v[e] = pv == 0.f ? 0.f : fminf(pv * sv, 448.f);
        }
        // Word select must be a constant: low half then high half.
        int word = __builtin_amdgcn_cvt_pk_fp8_f32(v[0], v[1], 0, false);
        b[d] = __builtin_amdgcn_cvt_pk_fp8_f32(v[2], v[3], word, true);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int j = 8 * w + 2 * gg + h;
        const v2i lo = __builtin_amdgcn_ds_read_tr4_b64_v2i32((lds_v2i*)(vt + 8 * j));
        const v2i hi =
            __builtin_amdgcn_ds_read_tr4_b64_v2i32((lds_v2i*)(vt + 16 * ROW_BYTES + 8 * j));
        const v8i a = {lo[0], lo[1], hi[0], hi[1], 0, 0, 0, 0};
        o[2 * gg + h] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a, b, o[2 * gg + h], FMT_E2M1, FMT_E4M3, 0, E8M0_ONE, 0, E8M0_ONE);
      }
    }
  }

  // Lane holds latents 16j + 4c + i of head r, for this wave's tiles j.
#pragma unroll
  for (int jj = 0; jj < 8; ++jj)
    *(float4*)(pacc + 16 * (8 * w + jj) + 4 * c) =
        make_float4(o[jj][0], o[jj][1], o[jj][2], o[jj][3]);
  if (w == 0 && c == 0) { pml[r] = m_run; pml[HEADS + r] = l_run; }
}

// Combine the split partials: out = sum_s e^(m_s-M) acc_s / sum_s e^(m_s-M) l_s.
extern "C" __global__ __launch_bounds__(128) void mxfp4_mla_native_reduce(
    const float* __restrict__ part_acc, const float* __restrict__ part_ml,
    __hip_bfloat16* __restrict__ out, int num_splits, int64_t out_stride_t,
    int64_t out_stride_h) {
  const int q = blockIdx.x, h = blockIdx.y, l = threadIdx.x;
  const float* ml = part_ml + (int64_t)q * num_splits * 2 * HEADS;
  float M = -INFINITY;
  for (int s = 0; s < num_splits; ++s) M = fmaxf(M, ml[s * 2 * HEADS + h]);
  float denom = 0.f;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  if (M != -INFINITY) {
    for (int s = 0; s < num_splits; ++s) {
      const float ms = ml[s * 2 * HEADS + h];
      if (ms == -INFINITY) continue;
      const float wt = __expf(ms - M);
      denom += wt * ml[s * 2 * HEADS + HEADS + h];
      const float4 pa =
          *(const float4*)(part_acc + ((int64_t)(q * num_splits + s) * HEADS + h) * LATENT + 4 * l);
      acc.x += wt * pa.x; acc.y += wt * pa.y; acc.z += wt * pa.z; acc.w += wt * pa.w;
    }
  }
  const float inv = denom > 0.f ? 1.f / denom : 0.f;
  __hip_bfloat16* o = out + q * out_stride_t + h * out_stride_h + 4 * l;
  o[0] = __float2bfloat16(acc.x * inv);
  o[1] = __float2bfloat16(acc.y * inv);
  o[2] = __float2bfloat16(acc.z * inv);
  o[3] = __float2bfloat16(acc.w * inv);
}
