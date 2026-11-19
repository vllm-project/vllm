# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-written HIP + MFMA (v_mfma_f32_16x16x16_bf16) chunk/prefill
attention for head_size 512, GQA 8Q/2KV, bf16 KV, block_size 16 on gfx942
(MI300X/MI325X).

The stock Triton unified-attention 2D chunk kernel runs the hd512 global
layers' prefill/chunk rows at ~103 TFLOPS = ~13% of the achievable BF16
compute peak across every context length (linear in context: 86 ms per
16K-token train at ~25K prefix, ~15% of what the roofline allows). This
kernel covers the same rows with the structure of the hd512 split-KV
decode kernel (hd512_mfma_decode): an NT-K + V^T coalesced re-layout
pre-pass (one 8B read per MFMA fragment, full cache lines), QK
split-D across the warp (each warp dots its own 8 PV d-chunks, S^T
partials reduced through a 16KB LDS window), MFMA
QK/PV with fp32 accumulators, split-KV over the context with the
hd512c_combine LSE merge, and __launch_bounds__(256, 3) (occupancy was the
binding constraint: 3 resident CTAs/CU).

Isolated on the burst-regime distribution (fresh Q [16280, 8, 512] vs
prefix 2K-88K): weighted mean 38.5 ms vs stock 85.9 ms (2.2x), i.e. ~28% of
the adopted achievable peak vs stock's ~13%. The readback+softmax phase is
a per-visit serial chain, CTA-count and LDS-instruction-count invariant
(falsifications: float4 pofs, pairs, col-owner, own-block warps) — the
floor for this structure was measured at ~38 ms without warp-specialized
overlap, which is the follow-up headroom item.

Correctness (isolated vs a tiled fp32 reference on the real shapes):
max-abs <= 2e-3, within stock fp32-accum noise (same values as stock's own
diff). Routed rows are written directly into the full batch's output rows
by the combine kernel, no gather/scatter of q.

Enabled with VLLM_GEMMA4_HD512_CHUNK=1; quiet below a prefix threshold
(VLLM_GEMMA4_HD512_CHUNK_PREFIX, default 7000 tokens: rows below it stay on
the stock unified kernel, whose cost there is under the noise of the
routed set). Routed from TritonAttentionImpl.forward only for head_size 512
full-attention DECODER layers; everything else (quantized KV, alibi/sinks/
softcap, sliding window, per-token-head quant) falls back to the stock
path unchanged. See vllm.v1.attention.backends.triton_attn.

Segment count per request (split-KV fan-out): VLLM_GEMMA4_HD512_CHUNK_SEGS,
default 8 (16 wins the long-context tail by ~6% there but loses ~1.5%
weighted; 8 is the default so the A/B/A has one personality).
"""

import os

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

// -- pre-pass: build the K-fragment and V^T-fragment tables for one
// (request-row, block, kvh). One 16 KB K tile + one 16 KB V^T tile per
// (r, b, kvh); the table is addressed by the LOCAL composite index
// (r * nblk_max + b), never by physical block id. K tile: 128 dim-quads x
// 16 tokens x 8B; tile[(u * 16 + tok) * 8] = K[tok][4u .. 4u+3].
// V^T tile: 32 dim-quads x 4 token-quads x 16 dims x 8B;
// vtile[((cv*4 + tq)*16 + dloc)*8 + j*2] = V[4tq+j][16cv+dloc], j=0..3.
extern "C" __global__ void __launch_bounds__(256)
hd512c_kt(const bf16* __restrict__ kc,
          const int* __restrict__ bt,
          const int* __restrict__ seqk,
          char* __restrict__ tab, char* __restrict__ vtab,
          long nblk_max, long k_blk, long k_slot, long k_hkv,
          int bt_stride, long v_delta) {
    const long b = blockIdx.x;                // logical block index
    const int kvh = blockIdx.y;
    const int r = blockIdx.z;                 // routed-request row
    const int nblk_r = (seqk[r] + 15) >> 4;
    if (b >= nblk_r) return;
    const bf16* kb = kc + (long)bt[(long)r * bt_stride + b] * k_blk
                   + (long)kvh * k_hkv;
    char* tile = tab + (((r * nblk_max + b) * 2 + kvh) * 16384);
    char* vtile = vtab + (((r * nblk_max + b) * 2 + kvh) * 16384);
    const int tok = threadIdx.x & 15;
    const int ch0 = threadIdx.x >> 4;         // 0..15 (16B chunk index/4)
    const bf16* row = kb + (long)tok * k_slot;
    const bf16* vrow = row + v_delta;
    const int tq = tok >> 2;
    const int tj = tok & 3;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int ch = ch0 + 16 * j;          // 16B chunk 0..63
        const uint4 v = *reinterpret_cast<const uint4*>(row + ch * 8);
        const int u0 = 2 * ch;
        *reinterpret_cast<uint2*>(tile + (u0 * 16 + tok) * 8)
            = uint2{v.x, v.y};
        *reinterpret_cast<uint2*>(tile + ((u0 + 1) * 16 + tok) * 8)
            = uint2{v.z, v.w};
        const ushort* vv = reinterpret_cast<const ushort*>(
            vrow + ch * 8);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int dim = ch * 8 + i;
            const int cv = dim >> 4;
            const int dloc = dim & 15;
            *reinterpret_cast<ushort*>(
                vtile + ((cv * 4 + tq) * 16 + dloc) * 8 + tj * 2) = vv[i];
        }
    }
}

// -- main kernel: 8-position super-tiles x 2 sub-tiles over the segments of
// the KV context; three resident CTAs per CU (occupancy measured binding).
// Partial super-tiles clamp the Q row index (their output rows are dropped
// by the combine), so arbitrary query lengths are safe.
extern "C" __global__ void __launch_bounds__(256, 3)
hd512c_attn(const bf16* __restrict__ q,
        const char* __restrict__ tab,
        const char* __restrict__ vtab,
        const int* __restrict__ seqk,
        const int* __restrict__ prefixes,
        const int* __restrict__ q_off,
        const int* __restrict__ tile_map,
        float* __restrict__ part_o,
        float* __restrict__ part_ml,
        float scale,
        long nblk_max,
        int segs) {
    __shared__ float pofs[2][4][2 * 256];  // ping-pong banks, 1 barrier/visit
    const int tile_blk = blockIdx.x;
    const int kvh = blockIdx.y;
    const int seg = blockIdx.z;
    const int lane = threadIdx.x & 63;
    const int warp = threadIdx.x >> 6;
    const int gv = lane >> 4;
    const int nw = lane & 15;

    const int seq = tile_map[2 * tile_blk];
    const int tile = tile_map[2 * tile_blk + 1];   // super-tile of 8 pos
    const int pos = nw >> 2;
    const int head = nw & 3;
    const int prefix = prefixes[seq];

    const int slen = seqk[seq];
    const int nblk = (slen + 15) >> 4;
    const int tiles_per_seg = (nblk + segs - 1) / segs;
    const int b0 = seg * tiles_per_seg;
    const int b1 = min(nblk - b0, tiles_per_seg) + b0;
    // last valid Q row of this request (partial super-tiles clamp to it)
    const int qmax = slen - prefix - 1;

    const long e = (long)tile_blk * 2 + kvh;
    const long tail = (e * segs + seg) * 32;
    // Causal visit cull: blocks at/above the
    // tile's max causal token carry no unmasked row; exact math, kept
    // blocks' accumulation order unchanged (bit-identical outputs).
    const int b_hi = min(b1, (prefix + 8 * tile + 23) >> 4);
    if (b0 >= b_hi) {
        if (gv == 0) {
            #pragma unroll
            for (int st = 0; st < 2; ++st) {
                part_ml[(tail + st * 16 + nw) * 2] = -INFINITY;
                part_ml[(tail + st * 16 + nw) * 2 + 1] = 0.f;
            }
        }
        #pragma unroll
        for (int st = 0; st < 2; ++st) {
            float* po0 = part_o + (tail + st * 16 + nw) * 512
                       + 128 * warp + 4 * gv;
            #pragma unroll
            for (int c = 0; c < 8; ++c)
                *reinterpret_cast<float4*>(po0 + 16 * c)
                    = float4{0, 0, 0, 0};
        }
        return;
    }

    const bf16* qseq = q + (long)q_off[seq] * 8 * 512;

    // causal caps per sub-tile (capped at slen for padded tail positions)
    int lim[2];
    #pragma unroll
    for (int st = 0; st < 2; ++st)
        lim[st] = min(prefix + 8 * tile + 4 * st + pos + 1,
                      slen - prefix > 0 ? slen : 0);

    float acc[2][8][4];
    #pragma unroll
    for (int st = 0; st < 2; ++st)
        #pragma unroll
        for (int c = 0; c < 8; ++c)
            #pragma unroll
            for (int t = 0; t < 4; ++t) acc[st][c][t] = 0.f;
    float m[2], l[2];
    #pragma unroll
    for (int st = 0; st < 2; ++st) { m[st] = -INFINITY; l[st] = 1.0f; }

    for (int b = b0; b < b_hi; ++b) {
        const int buf = (b - b0) & 1;   // ping-pong bank: 1 barrier/visit

        // -- V^T fragments from the coalesced V^T table: ONE uint2 per frag,
        // fully coalesced (16 consecutive dims per 128B line) --------------
        const char* vtbase =
            vtab + (((long)(seq * nblk_max + b) * 2 + kvh) * 16384);
        i32x2 a2v[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int cv = 8 * warp + c;
            a2v[c] = *reinterpret_cast<const i32x2*>(
                vtbase + ((cv * 4 + gv) * 16 + nw) * 8);
        }

        // -- QK split-D for both sub-tiles; A-fragments from the coalesced
        // table (16 tokens contiguous per 8B read); partials staged in LDS -
        const char* tbase_unc =
            tab + (((long)(seq * nblk_max + b) * 2 + kvh) * 16384);
        const char* tbase = tbase_unc + (lane & 15) * 8;
        #pragma unroll
        for (int st = 0; st < 2; ++st) {
            const int qpos = min(8 * tile + 4 * st + pos, qmax);
            const bf16* qrow = qseq + ((long)qpos * 8
                                       + kvh * 4 + head) * 512;
            f32x4 s = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int c2 = 0; c2 < 8; ++c2) {
                const int c = 8 * warp + c2;
                const uint2 ka = *reinterpret_cast<const uint2*>(
                    tbase + (4 * c + gv) * 128);
                const uint2 qb = *reinterpret_cast<const uint2*>(
                    qrow + 16 * c + 4 * gv);
                s = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    *(i32x2*)&ka, *(i32x2*)&qb,
                    f32x4{s[0], s[1], s[2], s[3]}, 0, 0, 0);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                pofs[buf][warp][st * 256 + nw * 16 + 4 * gv + i] = s[i];
        }
        __syncthreads();                      // partials staged

        // -- read-back, softmax and PV per sub-tile --------------------------
        #pragma unroll
        for (int st = 0; st < 2; ++st) {
            f32x4 s;
            {
                const float* pli = pofs[buf][0] + st * 256 + nw * 16
                                        + 4 * gv;
                float a0 = pli[0], a1 = pli[1], a2 = pli[2], a3 = pli[3];
                #pragma unroll
                for (int w2 = 1; w2 < 4; ++w2) {
                    const float* plw = pofs[buf][w2] + st * 256 + nw * 16
                                     + 4 * gv;
                    a0 += plw[0]; a1 += plw[1]; a2 += plw[2]; a3 += plw[3];
                }
                s[0] = a0; s[1] = a1; s[2] = a2; s[3] = a3;
            }
            const int sl_lo = 4 * gv;
            const int lim0 = lim[st] - 16 * b;
            float sv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sv[i] = (sl_lo + i < lim0) ? s[i] * scale : -INFINITY;
            float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
            mx = fmaxf(mx, __shfl_xor_sync(~0ULL, mx, 16));
            mx = fmaxf(mx, __shfl_xor_sync(~0ULL, mx, 32));
            const float m_new = fmaxf(m[st], mx);
            const float alpha = __expf(m[st] - m_new);
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
            l[st] = l[st] * alpha + psum;
            m[st] = m_new;
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                f32x4 a = {acc[st][c][0] * alpha, acc[st][c][1] * alpha,
                           acc[st][c][2] * alpha, acc[st][c][3] * alpha};
                a = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    a2v[c], pk,
                    a, 0, 0, 0);
                acc[st][c][0] = a[0]; acc[st][c][1] = a[1];
                acc[st][c][2] = a[2]; acc[st][c][3] = a[3];
            }
        }
    }

    // -- write partials -----------------------------------------------------
    if (gv == 0) {
        #pragma unroll
        for (int st = 0; st < 2; ++st) {
            part_ml[(tail + st * 16 + nw) * 2] = m[st];
            part_ml[(tail + st * 16 + nw) * 2 + 1] = l[st];
        }
    }
    #pragma unroll
    for (int st = 0; st < 2; ++st) {
        float* po0 = part_o + (tail + st * 16 + nw) * 512
                     + 128 * warp + 4 * gv;
        #pragma unroll
        for (int c = 0; c < 8; ++c)
            *reinterpret_cast<float4*>(po0 + 16 * c) =
                float4{acc[st][c][0], acc[st][c][1],
                       acc[st][c][2], acc[st][c][3]};
    }
}

extern "C" __global__ void __launch_bounds__(64)
hd512c_combine(bf16* __restrict__ out,
           const float* __restrict__ part_o,
           const float* __restrict__ part_ml,
           const int* __restrict__ seqk,
           const int* __restrict__ prefixes,
           const int* __restrict__ q_off,
           const int* __restrict__ tile_map,
           int segs) {
    const long e = blockIdx.x;                // super-tile*2 + kvh
    const int tile_blk = (int)(e >> 1);
    const int kvh = (int)(e & 1);
    const int col = blockIdx.y;               // 0..31
    const int lane = threadIdx.x;
    const int seq = tile_map[2 * tile_blk];
    const int tile = tile_map[2 * tile_blk + 1];
    const int qabs = 8 * tile + 4 * (col >> 4) + ((col & 15) >> 2);
    const int qlen = seqk[seq] - prefixes[seq];
    if (qabs >= qlen) return;                 // padded tail position
    const int head = col & 3;

    const long base = (e * (long)segs) * 32 + col;
    float mstar = -INFINITY;
    for (int sgm = 0; sgm < segs; ++sgm) {
        mstar = fmaxf(mstar, part_ml[(base + (long)sgm * 32) * 2]);
    }
    float acc[8] = {0.f};
    float lstar = 0.f;
    for (int sgm = 0; sgm < segs; ++sgm) {
        const long idx = base + (long)sgm * 32;
        float mm = part_ml[idx * 2];
        float ls = part_ml[idx * 2 + 1];
        if (mm == -INFINITY) continue;
        float w = __expf(mm - mstar);
        lstar += ls * w;
        const float* po = part_o + idx * 512 + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += w * po[i];
    }
    float inv = (lstar == 0.f) ? 0.f : 1.0f / lstar;
    bf16* op = out + ((long)(q_off[seq] + qabs) * 8
                      + kvh * 4 + head) * 512 + lane * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) op[i] = __float2bfloat16(acc[i] * inv);
}

void launch_hd512c(at::Tensor q, at::Tensor kc, at::Tensor tab,
                 at::Tensor vtab, at::Tensor bt,
                 at::Tensor seqk, at::Tensor prefixes, at::Tensor q_off,
                 at::Tensor tile_map,
                 at::Tensor part_o, at::Tensor part_ml, at::Tensor out,
                 double scale_d, long v_delta, long k_blk,
                 long k_slot, long k_hkv, long segs, long ntiles, long stream,
                 long nblk_max, long n_routed) {
    float scale = (float)scale_d;
    int bt_stride = (int)bt.stride(0);
    dim3 kgrid(nblk_max, 2, n_routed);
    hd512c_kt<<<kgrid, 256, 0, (hipStream_t)stream>>>(
        (const bf16*)kc.data_ptr(), (const int*)bt.data_ptr(),
        (const int*)seqk.data_ptr(),
        (char*)tab.data_ptr(), (char*)vtab.data_ptr(),
        nblk_max, k_blk, k_slot, k_hkv, bt_stride, v_delta);
    dim3 grid(ntiles, 2, segs);
    hd512c_attn<<<grid, 256, 0, (hipStream_t)stream>>>(
        (const bf16*)q.data_ptr(),
        (const char*)tab.data_ptr(), (const char*)vtab.data_ptr(),
        (const int*)seqk.data_ptr(),
        (const int*)prefixes.data_ptr(),
        (const int*)q_off.data_ptr(),
        (const int*)tile_map.data_ptr(),
        (float*)part_o.data_ptr(), (float*)part_ml.data_ptr(),
        scale, nblk_max, (int)segs);
    dim3 cgrid(ntiles * 2, 32, 1);
    hd512c_combine<<<cgrid, 64, 0, (hipStream_t)stream>>>(
        (bf16*)out.data_ptr(), (const float*)part_o.data_ptr(),
        (const float*)part_ml.data_ptr(),
        (const int*)seqk.data_ptr(),
        (const int*)prefixes.data_ptr(),
        (const int*)q_off.data_ptr(),
        (const int*)tile_map.data_ptr(),
        (int)segs);
}
"""

_CPP = """
#include <torch/extension.h>
void launch_hd512c(at::Tensor q, at::Tensor kc, at::Tensor tab,
                 at::Tensor vtab, at::Tensor bt,
                 at::Tensor seqk, at::Tensor prefixes, at::Tensor q_off,
                 at::Tensor tile_map,
                 at::Tensor part_o, at::Tensor part_ml, at::Tensor out,
                 double scale, long v_delta, long k_blk,
                 long k_slot, long k_hkv, long segs, long ntiles, long stream,
                 long nblk_max, long n_routed);
"""


def hd512c_get_kernel():
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
            name="vllm_hd512_mfma_chunk",
            cpp_sources=_CPP,
            cuda_sources=_SRC,
            functions=["launch_hd512c"],
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _KERNEL


#: Default split-KV segment count (overridable per launch).
NUM_SEGMENTS = 8


def hd512c_scratch_shape(ntiles: int, segs: int = NUM_SEGMENTS) -> tuple[int, int]:
    """Elements of (part_o, part_ml) scratch for ntiles super-tiles."""
    return ntiles * 2 * segs * 32 * 512, ntiles * 2 * segs * 32 * 2


def hd512_chunk_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    bt: torch.Tensor,
    seqk: torch.Tensor,
    prefixes: torch.Tensor,
    q_off: torch.Tensor,
    tile_map: torch.Tensor,
    part_o: torch.Tensor,
    part_ml: torch.Tensor,
    out: torch.Tensor,
    tab: torch.Tensor,
    vtab: torch.Tensor,
    softmax_scale: float,
    ntiles: int,
    nblk_max: int,
    n_routed: int,
    segs: int = NUM_SEGMENTS,
) -> None:
    """Chunk/prefill MFMA attention over the routed requests.

    Args:
        q: [num_tokens, 8, 512] bf16 (the FULL batch's query: rows are
            addressed through q_off).
        kv_cache: [num_blocks, num_kv_heads=2, block_size=16, 2*head_size]
            bf16 per-layer cache (any physical layout with a stride-1
            content dim: K at content offset 0, V at +512).
        bt: [n_routed, max_blocks] int32 block-table rows of the routed
            requests; tables are addressed by the local composite index
            (r * nblk_max + b), NOT by physical block id.
        seqk: [n_routed] int32 KV lengths.
        prefixes: [n_routed] int32 = seqk - query_len.
        q_off: [n_routed] int32 start row of each routed request in q.
        tile_map: [2 * ntiles] int32 (routed-row, super-tile) pairs.
        part_o: fp32 partial-output scratch of hd512c_scratch_shape(ntiles).
        part_ml: fp32 max/sum scratch of hd512c_scratch_shape(ntiles).
        out: [num_tokens, 8, 512] bf16; rows of the routed requests are
            written in place by the combine kernel.
        tab: uint8 K re-layout table, n_routed * nblk_max * 2 * 16384
            bytes.
        vtab: uint8 V^T re-layout table, the same size as tab.
        softmax_scale: attention scale.
        ntiles: number of (routed-row, super-tile) work items.
        nblk_max: maximum blocks per routed row in the re-layout tables.
        n_routed: number of routed requests.
        segs: split-KV segments per tile.

    """
    strides = kv_cache.stride()
    assert kv_cache.dtype == torch.bfloat16 and strides[3] == 1, (
        "hd512_mfma_chunk requires a bf16 cache with a stride-1 content dim"
    )
    stream = torch.cuda.current_stream().cuda_stream
    hd512c_get_kernel().launch_hd512c(
        q,
        kv_cache,
        tab,
        vtab,
        bt,
        seqk,
        prefixes,
        q_off,
        tile_map,
        part_o,
        part_ml,
        out,
        softmax_scale,
        512 * strides[3],  # V delta in elements within the content dim
        strides[0],  # block stride
        strides[2],  # slot (N) stride
        strides[1],  # kv-head (H) stride
        segs,
        ntiles,
        stream,
        nblk_max,
        n_routed,
    )


class HD512ChunkStepPlan:
    """Per-step routing plan for a mixed (chunked-prefill + decode) batch.

    Built once per step from the CPU-side query offsets and seq-lens (exact
    for prefill rows) and cached on the attention-metadata object, so the
    (up to 7) hd512 layers of one KV-cache group share one plan. Rows:
      - decode rows (query_len == 1) are handled by the split-KV decode
        kernel when the decode flag is on, else join the stock pack;
      - chunk rows with a prefix at/above the threshold go to this module's
        kernel (written directly into the full-batch output rows);
      - every other row is packed for the stock unified kernel, which keeps
        index-tuple parity with the mixed-batch decode-row split.
    """

    __slots__ = (
        "num_dec",
        "dec_tok",
        "bt_dec",
        "sl_dec",
        "stock_tok",
        "stock_cu",
        "bt_stock",
        "sl_stock",
        "stock_max_q",
        "routed",
        "rt_ntiles",
        "rt_tile_map",
        "rt_prefixes",
        "rt_q_off",
        "rt_seqk",
        "rt_bt",
        "rt_nblk_max",
    )


def hd512_build_step_plan(
    query_start_loc_cpu: torch.Tensor,
    num_reqs: int,
    seq_lens_cpu_ub: torch.Tensor | None,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    decode_rows_armed: bool,
    prefix_threshold: int,
) -> "HD512ChunkStepPlan | None":
    """Partition the step's rows into decode / routed-chunk / stock packs.

    Returns None when routing is impossible this step (no CPU seq-lens) or
    nothing would be diverted: the caller then takes the stock full-batch
    path unchanged.
    """
    if seq_lens_cpu_ub is None:
        return None
    qsl = query_start_loc_cpu[: num_reqs + 1].to(torch.int64)
    slub = seq_lens_cpu_ub[:num_reqs].to(torch.int64)
    lens = qsl[1:] - qsl[:-1]
    prefix = slub - lens
    device = block_table.device

    plan = HD512ChunkStepPlan()

    routed_idx = ((lens > 1) & (prefix >= prefix_threshold)).nonzero(as_tuple=True)[0]

    # Workspace shape caps. Per-call transient workspace is
    #   partials : ntiles * 1 MiB (fp32 partials per 8-position super-tile)
    #   tables   : sum(nblk_r) * 64 KiB (K and V^T re-layout tables)
    # The scheduler bounds scheduled tokens, which bounds ntiles to ~4K,
    # keeping partials under ~4.3 GiB, but sum(nblk_r) follows the
    # PREFIX lengths of the routed set — a deep-queue step (overload
    # concurrency) can hold many multi-hundred-block prefixes and would
    # otherwise try to allocate tens of GiB and OOM the engine. Cap the
    # routed set in input order; over-cap rows just stay on the stock
    # pack, which is the same correctness class at stock speed.
    max_blocks = int(os.environ.get("VLLM_GEMMA4_HD512_CHUNK_MAXBLOCKS", "40960"))
    tiles_cap = 4096
    capped: list[int] = []
    tot_blocks = 0
    tot_tiles = 0
    for r in routed_idx.tolist():
        nbl = int((int(slub[r]) + 15) // 16)
        nt = int((int(lens[r]) + 7) // 8)
        if tot_blocks + nbl > max_blocks or tot_tiles + nt > tiles_cap:
            continue
        capped.append(r)
        tot_blocks += nbl
        tot_tiles += nt
    routed_idx = torch.tensor(capped, dtype=torch.int64)
    plan.routed = int(routed_idx.numel()) > 0

    if decode_rows_armed:
        dec = (lens == 1).nonzero(as_tuple=True)[0]
        plan.num_dec = int(dec.numel())
    else:
        dec = torch.empty(0, dtype=torch.int64)
        plan.num_dec = 0

    # Stock rows: everything not routed and not decode (when decode armed).
    stock_mask = torch.ones(num_reqs, dtype=torch.bool)
    if plan.routed:
        stock_mask[routed_idx] = False
    if plan.num_dec:
        stock_mask[dec] = False
    stock = stock_mask.nonzero(as_tuple=True)[0]

    # Decode arm tensors (same layout as hd512_build_split's).
    if plan.num_dec:
        dec_dev = dec.to(device, non_blocking=True)
        plan.dec_tok = qsl[dec].to(device, non_blocking=True)
        plan.bt_dec = block_table.index_select(0, dec_dev).contiguous()
        plan.sl_dec = seq_lens.index_select(0, dec_dev).contiguous()
    else:
        plan.dec_tok = None
        plan.bt_dec = None
        plan.sl_dec = None

    n_stock = int(stock.numel())
    n_divert = plan.num_dec + int(routed_idx.numel())
    if n_divert == 0 or n_stock + n_divert != num_reqs:
        # Nothing diverted (all stock), or an inconsistent partition:
        # stock full-batch path.
        return None
    if n_stock == 0:
        plan.stock_tok = None
        plan.stock_cu = None
        plan.bt_stock = None
        plan.sl_stock = None
        plan.stock_max_q = 1
    else:
        stock_dev = stock.to(device)
        slens = lens[stock]
        cu = slens.cumsum(0)
        # Global token rows of every stock request in original order: the
        # complement of the diverted rows within [0, total_tokens).
        num_tokens = int(qsl[num_reqs])
        seen = torch.zeros(num_tokens, dtype=torch.bool)
        if plan.num_dec:
            seen[qsl[dec]] = True
        if plan.routed:
            for idx in routed_idx.tolist():
                s, e = int(qsl[idx]), int(qsl[idx + 1])
                seen[s:e] = True
        plan.stock_tok = (~seen).nonzero(as_tuple=True)[0].to(device, non_blocking=True)
        plan.stock_cu = (
            torch.cat([torch.zeros(1, dtype=torch.int64), cu])
            .to(torch.int32)
            .to(device, non_blocking=True)
        )
        plan.bt_stock = block_table.index_select(0, stock_dev).contiguous()
        plan.sl_stock = seq_lens.index_select(0, stock_dev).contiguous()
        plan.stock_max_q = int(slens.max())

    if not plan.routed:
        # Only the decode arm diverted rows; routing impossible is legal.
        if plan.num_dec == 0:
            return None
        plan.rt_ntiles = 0
        plan.rt_tile_map = None
        plan.rt_prefixes = None
        plan.rt_q_off = None
        plan.rt_seqk = None
        plan.rt_bt = None
        plan.rt_nblk_max = 0
    else:
        rl = routed_idx.tolist()
        rtiles = [int((int(lens[r]) + 7) // 8) for r in rl]
        ntiles = sum(rtiles)
        # Interleaved (seq, super-tile) pairs, one vectorized run per
        # routed request.
        parts = []
        for j, t in enumerate(rtiles):
            part = torch.empty(2 * t, dtype=torch.int32)
            part[0::2] = j
            part[1::2] = torch.arange(t, dtype=torch.int32)
            parts.append(part)
        tile_map_cpu = torch.cat(parts, 0)
        plan.rt_ntiles = ntiles
        rout_dev = routed_idx.to(device)
        plan.rt_tile_map = tile_map_cpu.to(device, non_blocking=True)
        plan.rt_prefixes = (
            prefix[routed_idx].to(torch.int32).to(device, non_blocking=True)
        )
        plan.rt_q_off = query_start_loc.index_select(0, rout_dev).contiguous()
        plan.rt_seqk = seq_lens.index_select(0, rout_dev).contiguous()
        plan.rt_bt = block_table.index_select(0, rout_dev).contiguous()
        plan.rt_nblk_max = int(((slub[routed_idx] + 15) // 16).max().item())
    return plan
