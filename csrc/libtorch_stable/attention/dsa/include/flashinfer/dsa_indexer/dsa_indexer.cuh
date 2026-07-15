// SPDX-License-Identifier: Apache-2.0
#ifndef FLASHINFER_DSA_INDEXER_CUH_
#define FLASHINFER_DSA_INDEXER_CUH_
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <stdexcept>
#include <limits>
#include <optional>
#include <tuple>
#include "dsa_indexer_kernels.cuh"
#ifndef DSA_CHECK
#define DSA_CHECK(cond, ...) do { if (!(cond)) throw std::runtime_error("dsa_indexer: " #cond); } while (0)
#endif
namespace {

static void* driver_handle() {
    static void* h = nullptr;
    if (!h) {
        h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        DSA_CHECK(h, "failed to load libcuda.so.1");
    }
    return h;
}

static CUresult enc_tiled(CUtensorMap* tm, CUtensorMapDataType dt, cuuint32_t rank,
                          void* addr, const cuuint64_t* dims, const cuuint64_t* strides,
                          const cuuint32_t* box, const cuuint32_t* estrides,
                          CUtensorMapInterleave il, CUtensorMapSwizzle sw,
                          CUtensorMapL2promotion l2, CUtensorMapFloatOOBfill oob) {
    using FT = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                            const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                            const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                            CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static FT f = nullptr;
    if (!f) {
        f = reinterpret_cast<FT>(dlsym(driver_handle(), "cuTensorMapEncodeTiled"));
        DSA_CHECK(f, "failed to load cuTensorMapEncodeTiled");
    }
    return f(tm, dt, rank, addr, dims, strides, box, estrides, il, sw, l2, oob);
}

static CUtensorMap make_2d(void* ptr, CUtensorMapDataType dt, int elem_size,
                           int gmem_inner, int gmem_outer,
                           int smem_inner, int smem_outer,
                           long gmem_outer_stride, int swizzle_mode) {
    if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
    CUtensorMap tm;
    const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
    const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    const cuuint64_t gstrides[1] = {(cuuint64_t)(gmem_outer_stride * elem_size)};
    const cuuint32_t estrides[2] = {1, 1};
    CUtensorMapSwizzle swizzle =
        swizzle_mode == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        swizzle_mode == 64  ? CU_TENSOR_MAP_SWIZZLE_64B :
        swizzle_mode == 32  ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE;
    CUresult r = enc_tiled(&tm, dt, 2, ptr, gdims, gstrides, sdims, estrides,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    DSA_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return tm;
}

static inline int align_up(int x, int a) { return (x + a - 1) / a * a; }

constexpr int NUM_HEADS = 32;
constexpr int HEAD_DIM = 128;
constexpr int BLOCK_Q = 4;         // 128 q*h rows per UMMA tile / 32 heads
constexpr int BLOCK_KV = 256;
constexpr int NUM_Q_STAGES = 1;   // one q-block per CTA
constexpr int NUM_KV_STAGES = 4;
constexpr int SPEC_THREADS = 128;
constexpr int MATH_THREADS = 256;  // 2 math warpgroups on SM100
constexpr int NUM_SMS = 148;       // B200

__global__ void refresh_threshold_from_bcount_kernel(
    int32_t* __restrict__ th_bucket,
    const int32_t* __restrict__ bcount,
    int R,
    int NB,
    int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= R) return;
    int old_th = th_bucket[row];
    int cum = 0;
    int new_th = old_th;
    bool found = false;
    for (int b = 0; b < NB; ++b) {
        cum += bcount[(size_t)row * NB + b];
        if (cum >= K) {
            new_th = b;
            found = true;
            break;
        }
    }
    if (found && new_th < old_th) th_bucket[row] = new_th;
}

// Fused seed/prep kernel (one block per row, all state in smem — borrows the
// vLLM top_k_per_row engineering): from the sample scores [Q, head] derive the
// per-row bucket params (origin, inv_delta), the initial gate threshold
// (bucket of the K-th best sample score), write the FULL sample histogram into
// bcount (a valid, conservative refresh base: counting genuine row elements
// can only tighten th safely), and emit every sample position with
// bucket <= th as initial candidates — a SUPERSET of the sample top-K, which
// the exact final select trims. Replaces: aminmax + torch.topk/radix seed +
// neg/contiguous copies + host seed copies + seed_bcount_kernel (~6 passes,
// ~10 launches) with 3 passes in 1 launch.
__global__ void seed_prep_kernel(
    const float* __restrict__ slog, const int64_t slog_stride,
    const int head, const int NB, const int K, const int cap,
    const int emit_limit,  // only columns j < emit_limit may emit seeds;
                           // probe columns beyond it are histogram/scale-only
    const int probe_stride_tok,  // >0: sample columns are strided probe pages;
                                 // emitted seed index j maps to original
                                 // position (j/64)*probe_stride_tok + (j%64)
    const int hist_stride,       // subsample factor for the minmax+histogram
                                 // passes (threshold estimation only; the
                                 // emit pass still reads everything). Caller
                                 // scales K to the subsample quantile.
    const float headroom,  // extend the bucket scale ABOVE the sample max by
                           // headroom*span (absolute, resolution-preserving
                           // when NB is scaled up with it): drifted scores
                           // land in real buckets instead of clamping to
                           // bucket 0 where refresh can never resolve them
    const int K_safe,      // certificate mode when >0: publish th (kq=K) as
                           // the PREDICTED gate, emit seeds up to the SAFE
                           // bucket (cum >= K_safe), and count seeds that
                           // qualify under the predicted bucket separately.
    float* __restrict__ origin, float* __restrict__ inv_delta,
    int32_t* __restrict__ th_bucket,
    int32_t* __restrict__ th_safe_out,      // [rows] (cert mode only)
    int32_t* __restrict__ seed_pred_cnt,    // [rows] (cert mode only)
    int32_t* __restrict__ bcount,
    float* __restrict__ cand_val, int32_t* __restrict__ cand_idx,
    int32_t* __restrict__ cand_cnt) {
    constexpr int BT = 1024;
    constexpr int NSUB = 4;  // sub-histograms to spread smem atomic conflicts
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const float* srow = slog + (size_t)row * slog_stride;

    // pass 1: min/max of the row's FINITE scores (vectorized). -inf appears
    // when the caller passes clean_logits=True full-row logits (dense-select
    // mode) for the out-of-range causal tail; it must not poison the range.
    __shared__ float s_mx[BT / 32];
    __shared__ float s_mn[BT / 32];
    float mx = -INFINITY, mn = INFINITY;
    const int head4 = head / 4 * 4;
    const auto acc = [&](const float s) {
        if (isfinite(s)) {
            mx = fmaxf(mx, s);
            mn = fminf(mn, s);
        }
    };
    for (int j = tid * 4; j < head4; j += BT * 4 * hist_stride) {
        const float4 s4 = *reinterpret_cast<const float4*>(srow + j);
        acc(s4.x); acc(s4.y); acc(s4.z); acc(s4.w);
    }
    if (hist_stride == 1)
        for (int j = head4 + tid; j < head; j += BT)
            acc(srow[j]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
    }
    if (lane == 0) { s_mx[tid >> 5] = mx; s_mn[tid >> 5] = mn; }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int wgi = 1; wgi < BT / 32; ++wgi) {
            s_mx[0] = fmaxf(s_mx[0], s_mx[wgi]);
            s_mn[0] = fminf(s_mn[0], s_mn[wgi]);
        }
    }
    __syncthreads();
    float o = -s_mx[0];         // min over x = -score
    const float hi = -s_mn[0];  // max over x
    const float span = fmaxf(hi - o, 1e-20f);
    o -= headroom * span;       // forward (above-max) drift headroom
    float inv = (NB - 1) / (span * (1.0f + headroom));

    // pass 2: histogram in [o, inv] bucket space, NSUB sub-histograms to cut
    // smem atomic conflicts, vectorized loads.
    extern __shared__ int s_hist[];  // NSUB * NB ints
    for (int b = tid; b < NSUB * NB; b += BT) s_hist[b] = 0;
    __syncthreads();
    int* my_hist = s_hist + (tid / (BT / NSUB)) * NB;
    const auto bucket_of = [&](const float s) -> int {
        const float x = -s;
        int b = static_cast<int>((x - o) * inv);
        return b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
    };
    for (int j = tid * 4; j < head4; j += BT * 4 * hist_stride) {
        const float4 s4 = *reinterpret_cast<const float4*>(srow + j);
        if (isfinite(s4.x)) atomicAdd(&my_hist[bucket_of(s4.x)], 1);
        if (isfinite(s4.y)) atomicAdd(&my_hist[bucket_of(s4.y)], 1);
        if (isfinite(s4.z)) atomicAdd(&my_hist[bucket_of(s4.z)], 1);
        if (isfinite(s4.w)) atomicAdd(&my_hist[bucket_of(s4.w)], 1);
    }
    if (hist_stride == 1)
        for (int j = head4 + tid; j < head; j += BT) {
            const float s = srow[j];
            if (isfinite(s)) atomicAdd(&my_hist[bucket_of(s)], 1);
        }
    __syncthreads();
    // merge sub-histograms into s_hist[0..NB)
    for (int b = tid; b < NB; b += BT) {
        int c = s_hist[b];
        #pragma unroll
        for (int g = 1; g < NSUB; ++g) c += s_hist[g * NB + b];
        s_hist[b] = c;
    }
    __syncthreads();
    // Coarse K-th estimate, then REBUILD the scale over just the useful
    // range: [~K-th value (+1 coarse bucket slack) .. sample max + drift
    // headroom]. The bottom of [min,max] is dead weight (the true global
    // threshold can only be TIGHTER than the sample's), and without top
    // headroom any score drifting above the sample max clamps into bucket 0
    // where refresh can never resolve past it (measured @512K/Q8192: 25% of
    // candidates above sample max, th floored at 0 for 43% of rows, 6.8xK
    // emission). Fine scale concentrates all NB buckets where the threshold
    // actually lives.
    // NOTE (C3, rejected by measurement 2026-07-09): rebuilding a FINE scale
    // over [K-th edge .. sample max + headroom] (fused into pass 3, no extra
    // read) did NOT fix the 512K@Q8192 drift shape (headroom proportional to
    // the narrow fine span is absolutely tiny — drift still clamps to bucket
    // 0) and cost 3-12% at every healthy shape (finer th moves more often ->
    // more gate fdiv reloads; extra smem atomic per value in pass 3).
    for (int b = tid; b < NB; b += BT)
        // Probe mode (emit_limit==0): scan-side refresh must start from zero
        // counts — write zeros here, saving the caller a separate memset.
        bcount[(size_t)row * NB + b] = (emit_limit == 0) ? 0 : s_hist[b];
    __shared__ int s_th, s_th_safe;
    if (tid == 0) {
        const int kk = K < head ? K : head;
        const int kk_safe = K_safe < head ? K_safe : head;
        int cum = 0, th = NB - 1, th2 = NB - 1;
        bool got1 = false;
        for (int b = 0; b < NB; ++b) {
            cum += s_hist[b];
            if (!got1 && cum >= kk) { th = b; got1 = true; if (K_safe <= 0) break; }
            if (K_safe > 0 && cum >= kk_safe) { th2 = b; break; }
        }
        s_th = th;
        s_th_safe = (K_safe > 0) ? th2 : th;
        th_bucket[row] = th;
        if (K_safe > 0) th_safe_out[row] = th2;
        origin[row] = o;
        inv_delta[row] = inv;
    }
    __syncthreads();
    const int th = s_th;
    const int th_emit = s_th_safe;  // == th when certificate mode is off

    if (emit_limit == 0) {  // threshold-probe mode: no seeds wanted; skip the
        if (tid == 0) {
            cand_cnt[row] = 0;  // whole pass-3 read of the sample
            if (K_safe > 0) seed_pred_cnt[row] = 0;
        }
        return;
    }

    // pass 3: emit every position with bucket <= th (compact, unordered).
    // Warp-aggregated: one shared-counter atomic per warp per pass-group.
    __shared__ int s_cnt, s_cnt_pred;
    if (tid == 0) { s_cnt = 0; s_cnt_pred = 0; }
    __syncthreads();
    float* vrow = cand_val + (size_t)row * cap;
    int32_t* irow = cand_idx + (size_t)row * cap;
    const auto emit_group = [&](const float s, const int j) {
        const int b_of = (isfinite(s) && j < emit_limit) ? bucket_of(s) : NB;
        const bool g = b_of <= th_emit;
        if (K_safe > 0) {
            const unsigned mp = __ballot_sync(0xffffffffu, b_of <= th);
            if (lane == 0 && mp) atomicAdd(&s_cnt_pred, __popc(mp));
        }
        const unsigned m = __ballot_sync(0xffffffffu, g);
        if (m != 0) {
            int base = 0;
            if (lane == 0) base = atomicAdd(&s_cnt, __popc(m));
            base = __shfl_sync(0xffffffffu, base, 0);
            if (g) {
                const int pos = base + __popc(m & ((lane == 0) ? 0u : ((1u << lane) - 1u)));
                if (pos < cap) {
                    // GATE4 build: candidate values live in BUCKET SPACE
                    // build-wide (the scan writes bq; select is rebased).
                    // Seeds must match: (x - o)*inv, same affine as the scan.
                    vrow[pos] = (-s - o) * inv;
                    irow[pos] = probe_stride_tok > 0
                        ? (j >> 6) * probe_stride_tok + (j & 63) : j;
                }
            }
        }
    };
    // Uniform trip counts so every lane always reaches the warp ballots.
    for (int j0 = 0; j0 < head4; j0 += BT * 4) {
        const int j = j0 + tid * 4;
        float4 s4 = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        if (j < head4) s4 = *reinterpret_cast<const float4*>(srow + j);
        emit_group(s4.x, j);
        emit_group(s4.y, j + 1);
        emit_group(s4.z, j + 2);
        emit_group(s4.w, j + 3);
    }
    for (int j0 = head4; j0 < head; j0 += BT) {
        const int j = j0 + tid;
        const float s = (j < head) ? srow[j] : -INFINITY;
        emit_group(s, j);
    }
    __syncthreads();
    if (tid == 0) {
        int c = s_cnt < cap ? s_cnt : cap;
        cand_cnt[row] = c;
        if (K_safe > 0) seed_pred_cnt[row] = s_cnt_pred;
    }
}

__device__ __forceinline__ uint32_t compact_enc_float(float v) {
    uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? (~bits) : (bits ^ 0x80000000u);
}

__global__ void compact_topk_min_thr_litetopk_kernel(
    const float* __restrict__ val,
    const int32_t* __restrict__ idx,
    const int32_t* __restrict__ cnt,
    const float* __restrict__ origin,
    const float* __restrict__ inv_delta,
    const int32_t* __restrict__ th_in,
    int R,
    int CAP,
    int K,
    int NB,
    float* __restrict__ out_val,
    int32_t* __restrict__ out_idx,
    // bulk-drain mode: scan-emitted indices (slot >= seed_base[row]) are in
    // COMPACTED space; map them back to original positions at output time
    // (only winners pay, K per row). Seeds (< seed_base) are already mapped.
    const uint32_t probe_group,
    const uint64_t probe_magic,
    const uint32_t probe_add_max,
    const int32_t* __restrict__ seed_base) {
    constexpr int BT = 256;
    constexpr int RADIX = 256;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= R) return;
    const float* vrow = val + (size_t)row * CAP;
    const int32_t* irow = idx + (size_t)row * CAP;
    float* ov = out_val + (size_t)row * K;
    int32_t* oi = out_idx + (size_t)row * K;
    int n = cnt[row];
    if (n > CAP) n = CAP;
    if (n < 0) n = 0;
    const int sbase = (probe_group != 0 && seed_base != nullptr)
                          ? seed_base[row] : 0x7fffffff;
    const auto out_map = [&](const int32_t raw, const int j) -> int32_t {
        if (j < sbase) return raw;
        const uint32_t kvo = static_cast<uint32_t>(raw);
        const uint32_t sup = (uint32_t)(((uint64_t)kvo * probe_magic) >> 42);
        return static_cast<int32_t>(kvo + min((sup + 1) * 64u, probe_add_max));
    };

    const float o = origin[row];
    const float inv = inv_delta[row];
    const int th = th_in[row];

    __shared__ uint32_t hist[RADIX];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int s_cnt_bucket_lt;
    __shared__ int s_cnt_bucket_eq;
    __shared__ int s_use_boundary;
    __shared__ int s_select_all;
    __shared__ int s_k_eq;
    __shared__ int s_cnt_lt;
    __shared__ int s_w_pre;
    __shared__ int s_w_lt;
    __shared__ int s_w_eq;

    if (n == 0) {
        for (int j = tid; j < K; j += BT) { ov[j] = INFINITY; oi[j] = 0; }
        return;
    }

    if (tid == 0) { s_cnt_bucket_lt = 0; s_cnt_bucket_eq = 0; }
    __syncthreads();
    int my_lt = 0, my_eq = 0;
    for (int j = tid; j < n; j += BT) {
        float v = vrow[j];
        if (!isfinite(v)) continue;
        int braw = static_cast<int>((v - o) * inv);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (b < th) my_lt++;
        else if (b == th) my_eq++;
    }
    atomicAdd(&s_cnt_bucket_lt, my_lt);
    atomicAdd(&s_cnt_bucket_eq, my_eq);
    __syncthreads();

    if (tid == 0) {
        int clt = s_cnt_bucket_lt;
        int ceq = s_cnt_bucket_eq;
        int k_eq = K - clt;
        s_use_boundary = (clt < K) && (k_eq > 0) && (k_eq <= ceq);
        s_select_all = (clt + ceq) < K;
        s_k_eq = s_use_boundary ? k_eq : (K < n ? K : n);
        desired = 0u;
        kfind = static_cast<uint32_t>(s_k_eq);
    }
    __syncthreads();

    #define DSA_IN_SELECT_SET(b) \
        (s_use_boundary ? ((b) == th) : (s_select_all ? true : ((b) <= th)))

    uint32_t mask = 0u;
    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - pass * 8;
        for (int b = tid; b < RADIX; b += BT) hist[b] = 0;
        __syncthreads();
        uint32_t d = desired;
        for (int j = tid; j < n; j += BT) {
            float v = vrow[j];
            if (!isfinite(v)) continue;
            int braw = static_cast<int>((v - o) * inv);
            int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
            if (!DSA_IN_SELECT_SET(b)) continue;
            uint32_t e = compact_enc_float(v);
            if ((e & mask) == (d & mask)) atomicAdd(&hist[(e >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = 0;
            uint32_t kf = kfind;
            for (int b = 0; b < RADIX; ++b) {
                uint32_t h = hist[b];
                if (acc < kf && kf <= acc + h) {
                    desired = d | (uint32_t(b) << shift);
                    kfind = kf - acc;
                    break;
                }
                acc += h;
            }
        }
        __syncthreads();
        mask |= 0xffu << shift;
    }
    uint32_t pivot = desired;

    if (tid == 0) { s_cnt_lt = 0; s_w_pre = 0; s_w_lt = 0; s_w_eq = 0; }
    __syncthreads();
    int lt = 0;
    for (int j = tid; j < n; j += BT) {
        float v = vrow[j];
        if (!isfinite(v)) continue;
        int braw = static_cast<int>((v - o) * inv);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (!DSA_IN_SELECT_SET(b)) continue;
        if (compact_enc_float(v) < pivot) lt++;
    }
    atomicAdd(&s_cnt_lt, lt);
    __syncthreads();
    int cnt_lt = s_cnt_lt;
    int pre_take = s_use_boundary ? s_cnt_bucket_lt : 0;
    int target_k = s_use_boundary ? s_k_eq : (K < n ? K : n);
    int eq_take = target_k - cnt_lt; if (eq_take < 0) eq_take = 0;

    for (int j = tid; j < n; j += BT) {
        float v = vrow[j];
        if (!isfinite(v)) continue;
        int braw = static_cast<int>((v - o) * inv);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (s_use_boundary && b < th) {
            int w = atomicAdd(&s_w_pre, 1);
            if (w < K) { ov[w] = v; oi[w] = out_map(irow[j], j); }
            continue;
        }
        if (!DSA_IN_SELECT_SET(b)) continue;
        uint32_t e = compact_enc_float(v);
        if (e < pivot) {
            int w = atomicAdd(&s_w_lt, 1);
            int out_pos = pre_take + w;
            if (out_pos < K) { ov[out_pos] = v; oi[out_pos] = out_map(irow[j], j); }
        } else if (e == pivot) {
            int oo = atomicAdd(&s_w_eq, 1);
            if (oo < eq_take) {
                int w = pre_take + cnt_lt + oo;
                if (w < K) { ov[w] = v; oi[w] = out_map(irow[j], j); }
            }
        }
    }
    #undef DSA_IN_SELECT_SET
    for (int j = tid + n; j < K; j += BT) {
        ov[j] = INFINITY;
        oi[j] = 0;
    }
}

static int compute_smem_bytes() {
    const int esz_fp8 = 1, esz_f32 = 4;
    const int smem_q  = BLOCK_Q * NUM_HEADS * HEAD_DIM * esz_fp8;
    const int smem_w  = BLOCK_Q * NUM_HEADS * esz_f32;
    const int smem_kv = BLOCK_KV * HEAD_DIM * esz_fp8;
    const int smem_ks = align_up(BLOCK_KV * esz_f32, 512);
    const int num_barriers = NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2 + (MATH_THREADS / 128) * 2;
    const int smem_barriers = num_barriers * 8;
    const int smem_slots = 4 * (int)sizeof(uint32_t);  // tmem ptr + daemon mailboxes
    const int smem_warpq = (MATH_THREADS / 32) * BLOCK_Q *
                           ((int)sizeof(int32_t) + DSA_WARP_QUEUE_CAP * ((int)sizeof(float) + (int)sizeof(int32_t)));
    const int smem_hist = BLOCK_Q * 256 * (int)sizeof(int32_t);  // per-CTA refresh
                                                                  // histogram (NB<=256)
    const int smem_safe = 0;
    return NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
           NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks +
           smem_barriers + smem_slots + smem_warpq + smem_hist + smem_safe;
}

}  // anonymous namespace
#endif  // FLASHINFER_DSA_INDEXER_CUH_
