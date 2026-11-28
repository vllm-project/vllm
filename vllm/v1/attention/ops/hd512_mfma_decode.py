# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-written HIP + MFMA (v_mfma_f32_16x16x16_bf16) split-KV decode
attention for head_size 512, GQA 8Q/2KV, bf16 KV, block_size 16 on gfx942
(MI300X/MI325X).

The stock Triton unified-attention decode is element-rate limited on this
geometry (~4.1 TB/s isolated for the c256 fleet distribution); this kernel
restructures the decode loop around a cooperative K->LDS stage (coalesced
1KB rows, XOR-swizzled 8B fragments) plus QK split-D (each warp dots only
its own 8 PV d-chunks, S^T partials reduced through a small LDS window) and
MFMA QK/PV. Isolated: ~13.3-13.6 ms vs stock ~13.1-13.25 ms at n=340 /
54.36 GB attended (parity within harness noise) and 2.20 ms vs 2.53 ms
(+15%) on the n=32 drain cohort of real c256 traces.

Correctness (worst max-abs-diff vs the stock kernel on real trace length
sets, fp32 accum both sides): <= 9.8e-4, within stock fp32-accum noise.

Enabled with VLLM_GEMMA4_HD512_DECODE=1; routed to from
TritonAttentionImpl.forward only for head_size 512 full-attention layers on
pure-decode batches. See vllm.v1.attention.backends.triton_attn.

Cache layout is read from the tensor strides: works on the default
(LBNHC) physical layout [B, N, H, 2*head_size] (K at content offset 0, V
at +head_size) and any layout with a stride-1 content dimension.
"""

import torch

_KERNEL = None

_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

using bf16 = __hip_bfloat16;

typedef __attribute__((ext_vector_type(4))) float f32x4;
typedef __attribute__((ext_vector_type(2))) int i32x2;

__device__ __forceinline__ uint pkbf16(float a, float b) {
    return (unsigned)__bfloat16_as_ushort(__float2bfloat16(a))
         | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(b)) << 16);
}

#define KDS_BYTES 16384

extern "C" __global__ void __launch_bounds__(256)
hd512_attn(const bf16* __restrict__ q,
        const bf16* __restrict__ kc,
        const int* __restrict__ bt,
        const int* __restrict__ seqk,
        float* __restrict__ part_o,
        float* __restrict__ part_ml,
        float scale,
        long v_delta, long k_blk, long k_slot, long k_hkv,
        int nseq, int segs, int bt_stride) {
    __shared__ char kds[KDS_BYTES];
    __shared__ float pofs[4][64];   // split-d partial S^T: [warp][16pos*4]
                     // NOTE: must NOT alias kds - warps can still be in their
                     // QK chunk loop (reading kds) when another warp writes.
    const int seq = blockIdx.x;
    const int kvh = blockIdx.y;
    const int seg = blockIdx.z;
    const int lane = threadIdx.x & 63;
    const int warp = threadIdx.x >> 6;
    const int gv = lane >> 4;          // 0..3, fragment group in both GEMMs
    const int nw = lane & 15;          // 0..15, q col (QK) / d mod16 (PV)

    const int slen = seqk[seq];
    const int nblk = (slen + 15) >> 4;
    const int tiles = (nblk + segs - 1) / segs;
    const int b0 = seg * tiles;
    const int b1 = min(nblk - b0, tiles) + b0;

    // partial offsets: [seq,kvh,seg,qh(4),(512|2)]
    const long tail = (((long)(seq * 2 + kvh) * segs + seg) * 4);
    if (b0 >= b1) {
        if (nw < 4) {
            if (gv == 0) {
                part_ml[(tail + lane) * 2] = -INFINITY;
                part_ml[(tail + lane) * 2 + 1] = 0.f;
            }
            float* po = part_o + (tail + nw) * 512 + 128 * warp + 4 * gv;
            #pragma unroll
            for (int c = 0; c < 8; ++c)
                *reinterpret_cast<float4*>(po + 16 * c) = float4{0, 0, 0, 0};
        }
        return;
    }

    const bf16* qrow = q + ((long)(seq * 8 + kvh * 4 + min(nw, 3))) * 512;
    const int* btrow = bt + (long)seq * bt_stride;

    float acc[8][4];
    #pragma unroll
    for (int c = 0; c < 8; ++c)
        #pragma unroll
        for (int t = 0; t < 4; ++t) acc[c][t] = 0.f;

    float m = -INFINITY, l = 1.0f;

    const int swz = 4 * ((lane & 15) & 3);   // fragment read: 4*(row&3)
    const int rld = warp;                     // loader: warp w+rows w, w+4, ...

    // load one block's K rows (coalesced, cooperative) into registers
    auto ldblk = [&](int b, uint4 raw[4]) {
        const bf16* kb = kc + (long)btrow[b] * k_blk + (long)kvh * k_hkv;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = rld + 4 * j;
            raw[j] = *reinterpret_cast<const uint4*>(
                kb + (long)r * k_slot + lane * 8);
        }
    };
    // write register block into LDS with the fragment swizzle
    auto stblk = [&](const uint4 raw[4]) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = rld + 4 * j;
            const int swj = 4 * (r & 3);
            char* dst = kds + r * 1024;
            *reinterpret_cast<uint2*>(dst + (((2 * lane + 0) ^ (swj & 15))) * 8)
                = uint2{raw[j].x, raw[j].y};
            *reinterpret_cast<uint2*>(dst + (((2 * lane + 1) ^ (swj & 15))) * 8)
                = uint2{raw[j].z, raw[j].w};
        }
    };

    {
        uint4 pf[4];
        ldblk(b0, pf);
        __syncthreads();
        stblk(pf);
    }

    for (int b = b0; b < b1; ++b) {
        __syncthreads();                      // stage visible / buffer free below
        uint4 pf[4];
        if (b + 1 < b1) ldblk(b + 1, pf);     // next stage issued before work
        const bf16* vbase = kc + (long)btrow[b] * k_blk + v_delta
                          + (long)kvh * k_hkv;

        // -- QK split-D: warp w covers chunks 8w..8w+7; partial S^T in LDS --
        f32x4 s = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int c2 = 0; c2 < 8; ++c2) {
            const int c = 8 * warp + c2;
            const int u = ((4 * c + gv) ^ (4 * ((lane & 15) & 3)));
            const uint2 ka = *reinterpret_cast<const uint2*>(
                kds + (lane & 15) * 1024 + u * 8);
            const uint2 qb = *reinterpret_cast<const uint2*>(
                qrow + 16 * c + 4 * gv);
            s = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                *(i32x2*)&ka, *(i32x2*)&qb,
                f32x4{s[0], s[1], s[2], s[3]}, 0, 0, 0);
        }
        if (nw < 4) {
            float* plo = pofs[warp] + (gv * 4 + nw) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) plo[i] = s[i];
        }
        __syncthreads();                      // kds reads done + partials staged
        if (b + 1 < b1) stblk(pf);            // stage b+1
        {
            float* pli = pofs[0] + (gv * 4 + nw) * 4;
            float a0 = pli[0], a1 = pli[1], a2 = pli[2], a3 = pli[3];
            #pragma unroll
            for (int w2 = 1; w2 < 4; ++w2) {
                const float* plw = pofs[w2] + (gv * 4 + nw) * 4;
                a0 += plw[0]; a1 += plw[1]; a2 += plw[2]; a3 += plw[3];
            }
            s[0] = a0; s[1] = a1; s[2] = a2; s[3] = a3;
        }

        // -- softmax over slots, valid-q lanes only ------------------------
        const int sl_lo = 4 * gv;             // slots b*16 + sl_lo + i
        const int lim = slen - 16 * b;        // valid slots in this block
        float sv[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            sv[i] = (sl_lo + i < lim) ? s[i] * scale : -INFINITY;
        float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
        mx = fmaxf(mx, __shfl_xor_sync(~0ULL, mx, 16));
        mx = fmaxf(mx, __shfl_xor_sync(~0ULL, mx, 32));
        const float m_new = fmaxf(m, mx);
        const float alpha = __expf(m - m_new);
        float psum = 0.f;
        i32x2 pk;
        {
            float p[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                p[i] = __expf(sv[i] - m_new);
                psum += p[i];
            }
            pk.x = pkbf16(p[0], p[1]);
            pk.y = pkbf16(p[2], p[3]);
        }
        psum += __shfl_xor_sync(~0ULL, psum, 16);
        psum += __shfl_xor_sync(~0ULL, psum, 32);
        l = l * alpha + psum;
        m = m_new;

        // -- PV: warp w owns chunks 8w..8w+7; A2 = V^T fragments ----------
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int d0 = 16 * (8 * warp + c) + nw;
            i32x2 a2;
            unsigned lo[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                lo[j] = (unsigned)*reinterpret_cast<const unsigned short*>(
                    vbase + (4 * gv + j) * k_slot + d0);
            }
            a2.x = lo[0] | (lo[1] << 16);
            a2.y = lo[2] | (lo[3] << 16);
            f32x4 a = {acc[c][0] * alpha, acc[c][1] * alpha,
                       acc[c][2] * alpha, acc[c][3] * alpha};
            a = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a2, pk, a, 0, 0, 0);
            acc[c][0] = a[0]; acc[c][1] = a[1];
            acc[c][2] = a[2]; acc[c][3] = a[3];
        }
    }

    // -- write partials -----------------------------------------------------
    if (nw < 4) {
        if (gv == 0) {
            part_ml[(tail + lane) * 2] = m;
            part_ml[(tail + lane) * 2 + 1] = l;
        }
        float* po = part_o + (tail + nw) * 512 + 128 * warp + 4 * gv;
        #pragma unroll
        for (int c = 0; c < 8; ++c)
            *reinterpret_cast<float4*>(po + 16 * c) =
                float4{acc[c][0], acc[c][1], acc[c][2], acc[c][3]};
    }
}

extern "C" __global__ void __launch_bounds__(64)
hd512_combine(bf16* __restrict__ out,
           const float* __restrict__ part_o,
           const float* __restrict__ part_ml,
           int nseq, int segs) {
    const int seq = blockIdx.x;
    const int head = blockIdx.y;
    const int kvh = head >> 2;
    const int warp = head & 3;
    const int lane = threadIdx.x;
    const long base = (((long)(seq * 2 + kvh) * segs) * 4 + warp);
    float mstar = -INFINITY;
    for (int sgm = 0; sgm < segs; ++sgm) {
        mstar = fmaxf(mstar, part_ml[(base + (long)sgm * 4) * 2]);
    }
    float acc[8] = {0.f};
    float lstar = 0.f;
    for (int sgm = 0; sgm < segs; ++sgm) {
        const long idx = base + (long)sgm * 4;
        float mm = part_ml[idx * 2];
        float ls = part_ml[idx * 2 + 1];
        if (mm == -INFINITY) continue;
        float w = __expf(mm - mstar);
        lstar += ls * w;
        const float* po = part_o + idx * 512 + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += w * po[i];
    }
    float inv = (lstar == 0.f) ? 0.f : 1.0f / lstar;  // zero-length row -> 0
    bf16* op = out + ((long)(seq * 8 + head)) * 512 + lane * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) op[i] = __float2bfloat16(acc[i] * inv);
}

void launch_hd512(at::Tensor q, at::Tensor kc, at::Tensor bt,
                 at::Tensor seqk, at::Tensor part_o, at::Tensor part_ml,
                 at::Tensor out, double scale_d, long v_delta, long k_blk,
                 long k_slot, long k_hkv, long segs, long stream) {
    long nseq = q.size(0);
    float scale = (float)scale_d;
    dim3 grid(nseq, 2, segs);
    int bt_stride = (int)bt.stride(0);
    hd512_attn<<<grid, 256, 0, (hipStream_t)stream>>>(
        (const bf16*)q.data_ptr(), (const bf16*)kc.data_ptr(),
        (const int*)bt.data_ptr(),
        (const int*)seqk.data_ptr(),
        (float*)part_o.data_ptr(), (float*)part_ml.data_ptr(),
        scale, v_delta, k_blk, k_slot, k_hkv,
        (int)nseq, (int)segs, bt_stride);
    dim3 cgrid(nseq, 8, 1);
    hd512_combine<<<cgrid, 64, 0, (hipStream_t)stream>>>(
        (bf16*)out.data_ptr(), (const float*)part_o.data_ptr(),
        (const float*)part_ml.data_ptr(), (int)nseq, (int)segs);
}


extern "C" __global__ void __launch_bounds__(256)
hd512_group_attn2(const bf16* __restrict__ q,
        const bf16* __restrict__ kc,
        const int* __restrict__ bt,         // [capacity, bt_stride] full
        const int* __restrict__ it_row,     // [n_items] -> bt row of member 0
        const int* __restrict__ it_mem,     // [n_items * 4] member rows, -1 pad
        const int* __restrict__ it_sblk,    // [n_items] shared (full) blocks
        float* __restrict__ part_o,
        float* __restrict__ part_ml,
        float scale,
        long v_delta, long k_blk, long k_slot, long k_hkv,
        int segs, int bt_stride) {
    // Shared-prefix cascade pass: CTAs (item, kvh, seg) each compute the
    // split-D QK/softmax/PV of one item's <=4 member requests over the
    // group's shared prefix blocks, packed as 16 MFMA columns
    // (member = nw>>2, qhead = nw&3). Same body as hd512_attn with d-quarter
    // striping per warp and a wider LDS partial window; partial slots are
    // written per member in the same [seq,kvh,seg,qh] layout (segs slots),
    // to be folded with the private-tail pass by hd512_combine_cascade.
    __shared__ char kds[KDS_BYTES];
    __shared__ float pofs2[4][256];  // split-d partial S^T: [warp][16slot*16col]
    const int item = blockIdx.x;
    const int kvh = blockIdx.y;
    const int seg = blockIdx.z;
    const int lane = threadIdx.x & 63;
    const int warp = threadIdx.x >> 6;
    const int gv = lane >> 4;
    const int nw = lane & 15;

    const int sblk = it_sblk[item];
    if (sblk <= 0) return;         // padded item: no members, no slots
    const int tiles = (sblk + segs - 1) / segs;
    const int b0 = min(sblk, seg * tiles);
    const int b1 = min(sblk, b0 + tiles);

    const int memc = it_mem[item * 4 + (nw >> 2)];
    if (b0 >= b1) {
        // empty segment: mark this member's slot so the combine skips it
        if (memc >= 0) {
            const long etail = (((long)(memc * 2 + kvh) * segs + seg) * 4
                               + (nw & 3));
            if (gv == 0) {
                part_ml[etail * 2] = -INFINITY;
                part_ml[etail * 2 + 1] = 0.f;
            }
            float* po = part_o + etail * 512 + 128 * warp + 4 * gv;
            #pragma unroll
            for (int c = 0; c < 8; ++c)
                *reinterpret_cast<float4*>(po + 16 * c) = float4{0, 0, 0, 0};
        }
        return;
    }

    const int slen = sblk * 16;    // all shared blocks are full
    const int mrowc = memc < 0 ? 0 : memc;
    const bf16* qrow = q + ((long)(mrowc * 8 + kvh * 4 + (nw & 3))) * 512;
    const int* btrow = bt + (long)it_row[item] * bt_stride;

    float acc[8][4];
    #pragma unroll
    for (int c = 0; c < 8; ++c)
        #pragma unroll
        for (int t = 0; t < 4; ++t) acc[c][t] = 0.f;

    float m = -INFINITY, l = 1.0f;

    const int rld = warp;

    auto ldblk = [&](int b, uint4 raw[4]) {
        const bf16* kb = kc + (long)btrow[b] * k_blk
                       + (long)kvh * k_hkv;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = rld + 4 * j;
            raw[j] = *reinterpret_cast<const uint4*>(
                kb + (long)r * k_slot + lane * 8);
        }
    };
    auto stblk = [&](const uint4 raw[4]) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = rld + 4 * j;
            const int swj = 4 * (r & 3);
            char* dst = kds + r * 1024;
            *reinterpret_cast<uint2*>(dst + (((2 * lane + 0) ^ (swj & 15))) * 8)
                = uint2{raw[j].x, raw[j].y};
            *reinterpret_cast<uint2*>(dst + (((2 * lane + 1) ^ (swj & 15))) * 8)
                = uint2{raw[j].z, raw[j].w};
        }
    };

    {
        uint4 pf[4];
        ldblk(b0, pf);
        __syncthreads();
        stblk(pf);
    }

    for (int b = b0; b < b1; ++b) {
        __syncthreads();
        uint4 pf[4];
        if (b + 1 < b1) ldblk(b + 1, pf);
        const bf16* vbase = kc + (long)btrow[b] * k_blk + v_delta
                          + (long)kvh * k_hkv;

        f32x4 s = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int c2 = 0; c2 < 8; ++c2) {
            const int c = 8 * warp + c2;
            const int u = ((4 * c + gv) ^ (4 * ((lane & 15) & 3)));
            const uint2 ka = *reinterpret_cast<const uint2*>(
                kds + (lane & 15) * 1024 + u * 8);
            const uint2 qb = *reinterpret_cast<const uint2*>(
                qrow + 16 * c + 4 * gv);
            s = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                *(i32x2*)&ka, *(i32x2*)&qb,
                f32x4{s[0], s[1], s[2], s[3]}, 0, 0, 0);
        }
        {
            float* plo = pofs2[warp] + (gv * 16 + nw) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) plo[i] = s[i];
        }
        __syncthreads();
        if (b + 1 < b1) stblk(pf);
        {
            float* pli = pofs2[0] + (gv * 16 + nw) * 4;
            float a0 = pli[0], a1 = pli[1], a2 = pli[2], a3 = pli[3];
            #pragma unroll
            for (int w2 = 1; w2 < 4; ++w2) {
                const float* plw = pofs2[w2] + (gv * 16 + nw) * 4;
                a0 += plw[0]; a1 += plw[1]; a2 += plw[2]; a3 += plw[3];
            }
            s[0] = a0; s[1] = a1; s[2] = a2; s[3] = a3;
        }

        const int sl_lo = 4 * gv;
        const int lim = slen - 16 * b;
        float sv[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            sv[i] = (sl_lo + i < lim) ? s[i] * scale : -INFINITY;
        float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
        mx = fmaxf(mx, __shfl_xor_sync(~0ULL, mx, 16));
        mx = fmaxf(mx, __shfl_xor_sync(~0ULL, mx, 32));
        const float m_new = fmaxf(m, mx);
        const float alpha = __expf(m - m_new);
        float psum = 0.f;
        i32x2 pk;
        {
            float p[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                p[i] = __expf(sv[i] - m_new);
                psum += p[i];
            }
            pk.x = pkbf16(p[0], p[1]);
            pk.y = pkbf16(p[2], p[3]);
        }
        psum += __shfl_xor_sync(~0ULL, psum, 16);
        psum += __shfl_xor_sync(~0ULL, psum, 32);
        l = l * alpha + psum;
        m = m_new;

        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int d0 = 16 * (8 * warp + c) + nw;
            i32x2 a2;
            unsigned lo[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                lo[j] = (unsigned)*reinterpret_cast<const unsigned short*>(
                    vbase + (long)(4 * gv + j) * k_slot + d0);
            }
            a2.x = lo[0] | (lo[1] << 16);
            a2.y = lo[2] | (lo[3] << 16);
            f32x4 a = {acc[c][0] * alpha, acc[c][1] * alpha,
                       acc[c][2] * alpha, acc[c][3] * alpha};
            a = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a2, pk, a, 0, 0, 0);
            acc[c][0] = a[0]; acc[c][1] = a[1];
            acc[c][2] = a[2]; acc[c][3] = a[3];
        }
    }

    if (memc < 0) return;
    const long tail = (((long)(memc * 2 + kvh) * segs + seg) * 4 + (nw & 3));
    if (gv == 0) {
        part_ml[tail * 2] = m;
        part_ml[tail * 2 + 1] = l;
    }
    float* po = part_o + tail * 512 + 128 * warp + 4 * gv;
    #pragma unroll
    for (int c = 0; c < 8; ++c)
        *reinterpret_cast<float4*>(po + 16 * c) =
            float4{acc[c][0], acc[c][1], acc[c][2], acc[c][3]};
}

extern "C" __global__ void __launch_bounds__(64)
hd512_combine_cascade(bf16* __restrict__ out,
           const int* __restrict__ grp_flag,   // [nseq] 1 = fold group A slots
           const float* __restrict__ part_oA,
           const float* __restrict__ part_mlA,
           const float* __restrict__ part_oB,
           const float* __restrict__ part_mlB,
           int nseq, int segsA, int segsB) {
    const int seq = blockIdx.x;
    const int head = blockIdx.y;
    const int kvh = head >> 2;
    const int warp = head & 3;
    const int lane = threadIdx.x;
    const long baseB = (((long)(seq * 2 + kvh) * segsB) * 4 + warp);
    const long baseA = (((long)(seq * 2 + kvh) * segsA) * 4 + warp);
    const int grp = grp_flag[seq];
    float mstar = -INFINITY;
    for (int sgm = 0; sgm < segsB; ++sgm)
        mstar = fmaxf(mstar, part_mlB[(baseB + (long)sgm * 4) * 2]);
    if (grp) {
        for (int sgm = 0; sgm < segsA; ++sgm)
            mstar = fmaxf(mstar, part_mlA[(baseA + (long)sgm * 4) * 2]);
    }
    float acc[8] = {0.f};
    float lstar = 0.f;
    for (int sgm = 0; sgm < segsB; ++sgm) {
        const long idx = baseB + (long)sgm * 4;
        float mm = part_mlB[idx * 2];
        float ls = part_mlB[idx * 2 + 1];
        if (mm == -INFINITY) continue;
        float w = __expf(mm - mstar);
        lstar += ls * w;
        const float* p = part_oB + idx * 512 + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += w * p[i];
    }
    if (grp) {
        for (int sgm = 0; sgm < segsA; ++sgm) {
            const long idx = baseA + (long)sgm * 4;
            float mm = part_mlA[idx * 2];
            float ls = part_mlA[idx * 2 + 1];
            if (mm == -INFINITY) continue;
            float w = __expf(mm - mstar);
            lstar += ls * w;
            const float* p = part_oA + idx * 512 + lane * 8;
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += w * p[i];
        }
    }
    float inv = (lstar == 0.f) ? 0.f : 1.0f / lstar;  // zero-length row -> 0
    bf16* op = out + ((long)(seq * 8 + head)) * 512 + lane * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) op[i] = __float2bfloat16(acc[i] * inv);
}

void launch_hd512_attn(at::Tensor q, at::Tensor kc, at::Tensor bt,
                 at::Tensor seqk, at::Tensor part_o, at::Tensor part_ml,
                 double scale_d, long v_delta, long k_blk,
                 long k_slot, long k_hkv, long segs, long stream) {
    long nseq = q.size(0);
    float scale = (float)scale_d;
    dim3 grid(nseq, 2, segs);
    int bt_stride = (int)bt.stride(0);
    hd512_attn<<<grid, 256, 0, (hipStream_t)stream>>>(
        (const bf16*)q.data_ptr(), (const bf16*)kc.data_ptr(),
        (const int*)bt.data_ptr(),
        (const int*)seqk.data_ptr(),
        (float*)part_o.data_ptr(), (float*)part_ml.data_ptr(),
        scale, v_delta, k_blk, k_slot, k_hkv,
        (int)nseq, (int)segs, bt_stride);
}

void launch_hd512_group2(at::Tensor q, at::Tensor kc, at::Tensor bt,
                         at::Tensor it_row, at::Tensor it_mem,
                         at::Tensor it_sblk,
                         at::Tensor part_o, at::Tensor part_ml,
                         double scale_d, long v_delta, long k_blk,
                         long k_slot, long k_hkv, long segs, long stream) {
    const long n_items = it_row.size(0);
    float scale = (float)scale_d;
    dim3 grid((unsigned)n_items, 2, (unsigned)segs);
    hd512_group_attn2<<<grid, 256, 0, (hipStream_t)stream>>>(
        (const bf16*)q.data_ptr(), (const bf16*)kc.data_ptr(),
        (const int*)bt.data_ptr(), (const int*)it_row.data_ptr(),
        (const int*)it_mem.data_ptr(), (const int*)it_sblk.data_ptr(),
        (float*)part_o.data_ptr(), (float*)part_ml.data_ptr(),
        scale, v_delta, k_blk, k_slot, k_hkv, (int)segs,
        (int)bt.stride(0));
}

void launch_hd512_combine_cascade(at::Tensor out, at::Tensor grp_flag,
                                  at::Tensor part_oA, at::Tensor part_mlA,
                                  at::Tensor part_oB, at::Tensor part_mlB,
                                  long segsA, long segsB, long stream) {
    long nseq = out.size(0);
    dim3 grid((unsigned)nseq, 8, 1);
    hd512_combine_cascade<<<grid, 64, 0, (hipStream_t)stream>>>(
        (bf16*)out.data_ptr(), (const int*)grp_flag.data_ptr(),
        (const float*)part_oA.data_ptr(), (const float*)part_mlA.data_ptr(),
        (const float*)part_oB.data_ptr(), (const float*)part_mlB.data_ptr(),
        (int)nseq, (int)segsA, (int)segsB);
}
"""

_CPP = """
#include <torch/extension.h>
void launch_hd512(at::Tensor q, at::Tensor kc, at::Tensor bt,
                 at::Tensor seqk, at::Tensor part_o, at::Tensor part_ml,
                 at::Tensor out, double scale, long v_delta, long k_blk,
                 long k_slot, long k_hkv, long segs, long stream);
void launch_hd512_attn(at::Tensor q, at::Tensor kc, at::Tensor bt,
                 at::Tensor seqk, at::Tensor part_o, at::Tensor part_ml,
                 double scale, long v_delta, long k_blk,
                 long k_slot, long k_hkv, long segs, long stream);
void launch_hd512_group2(at::Tensor q, at::Tensor kc, at::Tensor bt,
                         at::Tensor it_row, at::Tensor it_mem,
                         at::Tensor it_sblk,
                         at::Tensor part_o, at::Tensor part_ml,
                         double scale, long v_delta, long k_blk,
                         long k_slot, long k_hkv, long segs, long stream);
void launch_hd512_combine_cascade(at::Tensor out, at::Tensor grp_flag,
                                  at::Tensor part_oA, at::Tensor part_mlA,
                                  at::Tensor part_oB, at::Tensor part_mlB,
                                  long segsA, long segsB, long stream);
"""


def hd512_get_kernel():
    """Compile (or fetch from the torch-extensions cache) the HIP module."""
    global _KERNEL
    if _KERNEL is None:
        import os

        from torch.utils.cpp_extension import load_inline

        # The container presets PYTORCH_ROCM_ARCH to a 9-arch list including
        # RDNA targets, on which the CDNA-only MFMA builtin breaks the build;
        # compile for the local device only.
        os.environ["PYTORCH_ROCM_ARCH"] = torch.cuda.get_device_properties(
            0
        ).gcnArchName
        _KERNEL = load_inline(
            name="vllm_hd512_mfma_decode_i13c",
            cpp_sources=_CPP,
            cuda_sources=_SRC,
            functions=[
                "launch_hd512",
                "launch_hd512_attn",
                "launch_hd512_group2",
                "launch_hd512_combine_cascade",
            ],
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _KERNEL


#: Number of split-KV segments per sequence (fixed under CUDA graphs).
NUM_SEGMENTS = 64


def hd512_partial_scratch_shape(nseq: int) -> tuple[int, int]:
    """Elements of (part_o, part_ml) scratch for one decode batch."""
    return nseq * 2 * NUM_SEGMENTS * 4 * 512, nseq * 2 * NUM_SEGMENTS * 4 * 2


def hd512_decode_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    part_o: torch.Tensor,
    part_ml: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None:
    """Split-KV MFMA decode attention.

    Args:
        q: [num_reqs, 8, 512] bf16, one query token per request.
        kv_cache: logical [num_blocks, 2, 16, 2*512] per-layer cache view
            (any physical layout with a stride-1 content dim: K at content
            offset 0, V at +512).
        block_table: [num_reqs, max_blocks] int32, 16-token blocks.
        seq_lens: [num_reqs] int32.
        part_o: fp32 partial-output scratch, partial_scratch_shape(max_reqs).
        part_ml: fp32 max/sum scratch, partial_scratch_shape(max_reqs).
        out: [num_reqs, 8, 512] bf16.
        softmax_scale: attention scale.

    """
    strides = kv_cache.stride()
    assert kv_cache.dtype == torch.bfloat16 and strides[3] == 1, (
        "hd512_mfma_decode requires a bf16 cache with a stride-1 content dim"
    )
    stream = torch.cuda.current_stream().cuda_stream
    hd512_get_kernel().launch_hd512(
        q,
        kv_cache,
        block_table,
        seq_lens,
        part_o,
        part_ml,
        out,
        softmax_scale,
        512 * strides[3],  # V delta in elements within the content dim
        strides[0],  # block stride
        strides[2],  # slot (N) stride
        strides[1],  # kv-head (H) stride
        NUM_SEGMENTS,
        stream,
    )


class HD512MixedSplit:
    """Per-step plan for routing a mixed (chunked-prefill + decode) batch.

    Built once per step from the CPU-side query_start_loc and cached on the
    attention-metadata object, so the (up to 7) hd512 layers of one KV-cache
    group share the partition, its device index tensors and its packed
    block-table / seq-lens gathers. Decode rows (one scheduled query token
    per request) go to the split-KV kernel; chunk rows go to the stock
    unified kernel over a packed subset.
    """

    __slots__ = (
        "num_dec",
        "dec_tok",
        "bt_dec",
        "sl_dec",
        "num_chunk",
        "chunk_tok",
        "chunk_cu",
        "bt_chunk",
        "sl_chunk",
        "chunk_max_q",
    )


def hd512_build_split(
    query_start_loc_cpu: torch.Tensor,
    num_reqs: int,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> "HD512MixedSplit | None":
    """Partition the batch into decode rows (query_len == 1) and chunk rows.

    Returns None when there are no decode rows to divert (pure prefill
    step): the stock full-batch path handles it. Everything not returned
    here stays on the stock full-batch path by the caller's gate.
    """
    qsl = query_start_loc_cpu[: num_reqs + 1].to(torch.int64)
    lens = qsl[1:] - qsl[:-1]
    dec = (lens == 1).nonzero(as_tuple=True)[0]
    if dec.numel() == 0:
        return None
    device = block_table.device
    split = HD512MixedSplit()
    split.num_dec = int(dec.numel())
    dec_dev = dec.to(device, non_blocking=True)
    # Decode rows: a request's single query token sits at its
    # query_start_loc; block_table / seq_lens gather by request index.
    split.dec_tok = qsl[dec].to(device, non_blocking=True)
    split.bt_dec = block_table.index_select(0, dec_dev).contiguous()
    split.sl_dec = seq_lens.index_select(0, dec_dev).contiguous()
    chunk = (lens != 1).nonzero(as_tuple=True)[0]
    split.num_chunk = int(chunk.numel())
    if chunk.numel() == 0:
        split.chunk_tok = None
        split.chunk_cu = None
        split.bt_chunk = None
        split.sl_chunk = None
        split.chunk_max_q = 1
        return split
    chunk_dev = chunk.to(device, non_blocking=True)
    clens = lens[chunk]
    cu = clens.cumsum(0)
    # Global token rows of every chunk request in original order: the
    # decode rows are sparse singletons, so the chunk rows are the
    # complement of the decode rows within [0, total_tokens).
    num_tokens = int(qsl[num_reqs])
    seen = torch.zeros(num_tokens, dtype=torch.bool)
    seen[qsl[dec]] = True
    split.chunk_tok = (~seen).nonzero(as_tuple=True)[0].to(device, non_blocking=True)
    # Cumulative query offsets of the packed chunk subset, same dtype the
    # stock path hands the unified kernel.
    split.chunk_cu = (
        torch.cat([torch.zeros(1, dtype=torch.int64), cu])
        .to(torch.int32)
        .to(device, non_blocking=True)
    )
    split.bt_chunk = block_table.index_select(0, chunk_dev).contiguous()
    split.sl_chunk = seq_lens.index_select(0, chunk_dev).contiguous()
    split.chunk_max_q = int(clens.max())
    return split


def hd512_decode_rows(
    query: torch.Tensor,
    output: torch.Tensor,
    kv_cache: torch.Tensor,
    split: HD512MixedSplit,
    part_o: torch.Tensor,
    part_ml: torch.Tensor,
    softmax_scale: float,
) -> None:
    """Run this batch's decode rows through the split-KV kernel: gather the
    decode q rows into a packed batch (the kernel indexes rows by request),
    run the per-row-independent kernel, scatter back into the output."""
    q_dec = query.index_select(0, split.dec_tok)
    out_dec = torch.empty_like(q_dec)
    hd512_decode_attention(
        q=q_dec,
        kv_cache=kv_cache,
        block_table=split.bt_dec,
        seq_lens=split.sl_dec,
        part_o=part_o,
        part_ml=part_ml,
        out=out_dec,
        softmax_scale=softmax_scale,
    )
    output.index_copy_(0, split.dec_tok, out_dec)


# ----------------------------------------------------------------------
# Shared-prefix-group (cascade) extension (VLLM_GEMMA4_HD512_CASCADE).
#
# Agentic multi-turn traffic forms branching conversation trees: at c256/c512 the
# running batch's sibling requests share whole-plan prefixes through
# prefix-cache dedup, so 67.9% of the hd512 KV bytes a decode step reads
# are duplicated across requests (coding-agent traces). The cascade path
# computes one
# split-D attention+LSE pass over each sibling group's common prefix
# blocks per layer (hd512_group_attn2: 16 MFMA columns packed as
# (member, qhead)), runs the plain split-KV kernel segs=64 over every
# request's effective context (private tail for grouped rows, full context
# for singles), and folds the two partial sets per request in log-sum-exp
# space (hd512_combine_cascade). Singles' B-pass + combine math is ordered
# identically to hd512_attn + hd512_combine, so rows without a group are
# bit-identical to the plain path.

#: Segments of the shared-prefix (group) pass partial slots per request.
CASCADE_SEGS_A = 48


def hd512_cascade_scratch_shape(nseq: int) -> tuple[int, int]:
    """Elements of (grp_part_o, grp_part_ml) scratch for one decode batch."""
    return (
        nseq * 2 * CASCADE_SEGS_A * 4 * 512,
        nseq * 2 * CASCADE_SEGS_A * 4 * 2,
    )


def hd512_cascade_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    it_row: torch.Tensor,
    it_mem: torch.Tensor,
    it_sblk: torch.Tensor,
    grp_flag: torch.Tensor,
    bt_shifted: torch.Tensor,
    sq_shifted: torch.Tensor,
    grp_part_o: torch.Tensor,
    grp_part_ml: torch.Tensor,
    part_o: torch.Tensor,
    part_ml: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None:
    """Cascade decode attention: group shared pass (A) + per-request
    effective-context pass (B) + flagged LSE fold of A(48) + B(64) slots.

    Args:
        q: [num_reqs, 8, 512] bf16, one query token per request.
        out: [num_reqs, 8, 512] bf16 output.
        kv_cache: per-layer cache view, strides as in hd512_decode_attention.
        block_table: [capacity, max_blocks] int32 per-request table.
        it_row: group item rows; the item count is fixed per runner.
        it_mem: group item members (-1 on padded entries).
        it_sblk: shared prefix blocks per item (<= 0 on padded entries).
        grp_flag: [capacity] int32 per-row group-mark (0 = single).
        bt_shifted: per-request effective block table (grouped rows skip
            their shared prefix blocks).
        sq_shifted: per-request effective lengths, matching bt_shifted.
        grp_part_o: A-pass partial-output scratch (CASCADE_SEGS_A slots).
        grp_part_ml: A-pass max/sum scratch (CASCADE_SEGS_A slots).
        part_o: B-pass partial-output scratch (NUM_SEGMENTS slots), the
            plain path's buffer.
        part_ml: B-pass max/sum scratch, the plain path's buffer.
        softmax_scale: attention scale.

    """
    strides = kv_cache.stride()
    assert kv_cache.dtype == torch.bfloat16 and strides[3] == 1, (
        "hd512_mfma_decode requires a bf16 cache with a stride-1 content dim"
    )
    stream = torch.cuda.current_stream().cuda_stream
    kernel = hd512_get_kernel()
    v_args = (512 * strides[3], strides[0], strides[2], strides[1])
    kernel.launch_hd512_group2(
        q,
        kv_cache,
        block_table,
        it_row,
        it_mem,
        it_sblk,
        grp_part_o,
        grp_part_ml,
        softmax_scale,
        *v_args,
        CASCADE_SEGS_A,
        stream,
    )
    kernel.launch_hd512_attn(
        q,
        kv_cache,
        bt_shifted,
        sq_shifted,
        part_o,
        part_ml,
        softmax_scale,
        *v_args,
        NUM_SEGMENTS,
        stream,
    )
    kernel.launch_hd512_combine_cascade(
        out,
        grp_flag,
        grp_part_o,
        grp_part_ml,
        part_o,
        part_ml,
        CASCADE_SEGS_A,
        NUM_SEGMENTS,
        stream,
    )


def hd512_cascade_groups(
    bt,
    num_blocks_per_row,
    seq_lens,
    max_items: int,
):
    """Sibling-group detection for one decode step (CPU, numpy).

    Args:
        bt: [N, W] int32 numpy block table (full width).
        num_blocks_per_row: [N] numpy per-row valid block count.
        seq_lens: [N] numpy per-row token counts.
        max_items: capacity of the item descriptor rows.

    Returns:
        (it_row, it_mem, it_sblk, grp_flag, bt_shifted, sq_shifted,
         n_groups, items, max_sqb)
        int32 numpy arrays: item rows padded with it_sblk == 0 /
        it_mem == -1; bt_shifted/sq_shifted are the B-pass effective
        table/lengths (grouped rows skip their shared prefix).
        Rows missing from every group have grp_flag == 0 and identity
        shift (byte-identical B pass to the plain path). Whole groups
        are dropped to singles (never partially covered) when the item
        table would overflow.

    """
    import numpy as np

    n = len(seq_lens)
    it_row = np.zeros(max_items, dtype=np.int32)
    it_mem = np.full(max_items * 4, -1, dtype=np.int32)
    it_sblk = np.zeros(max_items, dtype=np.int32)
    grp_flag = np.zeros(n, dtype=np.int32)
    sblk_row = np.zeros(n, dtype=np.int32)

    nblk = num_blocks_per_row[:n].astype(np.int64)
    # Only columns up to the longest valid row matter for grouping and
    # for the shifted table; the kernel never reads further.
    w = int(max(int(nblk.max()), 1))
    t = bt[:n, :w]

    # Lexicographic sort of the block-id rows: void-compare over
    # big-endian int32 bytes equals lexicographic over the integers.
    be = t.astype(">i4")
    order = np.argsort(be.view(f"V{be.shape[1] * 4}").ravel(), kind="stable")
    ts = t[order]
    lens_sorted = nblk[order]
    # Adjacent longest-common-prefix, clipped to the shorter row.
    diff = ts[1:] != ts[:-1]
    adj = np.where(diff.any(axis=1), diff.argmax(axis=1), w)
    adj = np.minimum(np.minimum(adj, lens_sorted[:-1]), lens_sorted[1:])

    # Runs of sorted rows linked by a common first block.
    runs: list[list[int]] = []
    run = [0]
    for i in range(1, n):
        if adj[i - 1] >= 1:
            run.append(i)
        else:
            if len(run) >= 2:
                runs.append(run)
            run = [i]
    if len(run) >= 2:
        runs.append(run)

    # Assemble items whole-group (sub-batches of <=7 rows each; the
    # kernel packs 4 columns per item): sblk = min adjacent LCP inside
    # the chunk, clipped to each member's own block count.
    items = 0
    n_groups = 0
    for run in runs:
        chunks = [run[c : c + 7] for c in range(0, len(run), 7)]
        need = sum((len(c_) + 3) // 4 for c_ in chunks)
        if items + need > max_items:
            break  # whole-group granularity; later groups stay singles
        for chunk in chunks:
            rows = [int(order[j]) for j in chunk]
            with_adj = [int(adj[j]) for j in range(chunk[0], chunk[-1])]
            sblk = min(with_adj) if with_adj else 1 << 30
            sblk = int(min(sblk, min(lens_sorted[j] for j in chunk)))
            if sblk < 1:
                continue
            n_groups += 1
            for base in range(0, len(rows), 4):
                batch = rows[base : base + 4]
                it_mem[items * 4 : items * 4 + len(batch)] = batch
                it_row[items] = batch[0]
                it_sblk[items] = sblk
                items += 1
            for r_ in rows:
                grp_flag[r_] = 1
                sblk_row[r_] = sblk

    # Effective B-pass table/lengths: grouped rows skip sblk blocks.
    sq = seq_lens[:n].astype(np.int64)
    sq_shifted = np.maximum(sq - 16 * sblk_row.astype(np.int64), 1)
    max_sqb = int(((sq_shifted + 15) // 16).max())
    bt_shifted = t.copy()
    for r_ in range(n):
        s_ = int(sblk_row[r_])
        if s_ > 0:
            nb_ = int(nblk[r_])
            bt_shifted[r_, : nb_ - s_] = t[r_, s_:nb_]
    return (
        it_row,
        it_mem,
        it_sblk,
        grp_flag,
        bt_shifted.astype(np.int32),
        sq_shifted.astype(np.int32),
        n_groups,
        items,
        max_sqb,
    )
