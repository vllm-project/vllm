// Milestone 1: single-kernel sparse MLA decode attention for DSV4.1-Flash (gfx950).
// Collapses the Triton partial + reduce kernel PAIR into ONE flash-style online-softmax
// kernel. NO split-K launch, NO separate reduce kernel: the split-over-tokens reduction
// is done IN-WORKGROUP via LDS to hide the serial-chain memory latency.
//
// MQA: one 512-dim KV latent per token shared across all H heads.
//   score_h = dot(q[h], K) * scale ; softmax over attended tokens ; out_h = sum_j p_hj*K_j
//   V == K (full 512-dim latent, nope-dequant + rope-bf16).
//
// fp8_ds_mla cache layout (matches _sparse_attn_decode_gfx950_partial_kernel):
//   page = 64 tokens. Per-page byte stride = 64*576 + 64*8 = 37376.
//   token t data:  [t*576 : t*576+448]     = 448 fp8_e4m3fn (nope), 7 blocks of 64
//                  [t*576+448 : t*576+576]  = 64 bf16 (rope), 128 bytes
//   token t scale: [64*576 + t*8 : +7]      = 7 e8m0 (ue8m0) bytes, one per 64-nope-block
//
// Grid: (B*H) workgroups, block = NWAVE waves. Wave w handles strided tokens {w,w+NWAVE,..}.
//   Lane owns dims [lane*8,+8); K,q,acc in REGISTERS. Per-wave chain is SW-PREFETCHED
//   (next token's global loads issued before current token's compute) to overlap the
//   K-load latency. NWAVE partial softmaxes merged via LDS by wave 0.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define HEAD_DIM   512
#define NOPE_DIM   448
#define PAGE_SIZE  64
#define PAGE_STRIDE 37376
#define TOK_DATA_STRIDE 576
#define SCALE_REGION_OFF 36864
#define DPT   8
#ifndef NWAVE
#define NWAVE 16
#endif
#define NEG   (-3.4028234663852886e38f)

__device__ __forceinline__ float fp8_e4m3fn_to_float(unsigned char b) {
    unsigned int s = (b >> 7) & 1u;
    unsigned int e = (b >> 3) & 0xFu;
    unsigned int m = b & 0x7u;
    if (e == 0u) { float v = (float)m * (1.0f / 512.0f); return s ? -v : v; }
    if (e == 0xFu && m == 0x7u) return 0.0f;
    unsigned int bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    return __uint_as_float(bits);
}
__device__ __forceinline__ float e8m0_to_float(unsigned char enc) {
    unsigned int bits = (enc == 0u) ? (1u << 22) : ((unsigned int)enc << 23);
    return __uint_as_float(bits);
}
__device__ __forceinline__ float bf16_round(float x) { return (float)(__hip_bfloat16)x; }

// (td, sc, valid) descriptor for token t.
__device__ __forceinline__ void tok_desc(
    int t, int swa_len, int b, int swa_maxlen, int topk_maxlen,
    const int* swa_indices, const int* topk_indices,
    const unsigned char* swa_cache, const unsigned char* kv_cache,
    int swa_num_rows, int kv_num_rows,
    const unsigned char*& td, const unsigned char*& sc, bool& valid)
{
    int slot; const unsigned char* cache; int num_rows;
    if (t < swa_len) { slot = swa_indices[b * swa_maxlen + t]; cache = swa_cache; num_rows = swa_num_rows; }
    else { slot = topk_indices[b * topk_maxlen + (t - swa_len)]; cache = kv_cache; num_rows = kv_num_rows; }
    valid = (slot >= 0) && (slot < num_rows);
    int safe = valid ? slot : 0;
    int page = safe / PAGE_SIZE, pos = safe % PAGE_SIZE;
    td = cache + (long)page * PAGE_STRIDE + (long)pos * TOK_DATA_STRIDE;
    sc = cache + (long)page * PAGE_STRIDE + SCALE_REGION_OFF + (long)pos * 8;
}

extern "C" __global__ void sparse_mla_decode_kernel(
    const __hip_bfloat16* __restrict__ q,
    const unsigned char*  __restrict__ swa_cache,
    const int*            __restrict__ swa_indices,
    const int*            __restrict__ swa_lens,
    const unsigned char*  __restrict__ kv_cache,
    const int*            __restrict__ topk_indices,
    const int*            __restrict__ topk_lens,
    __hip_bfloat16*       __restrict__ out,
    float*                __restrict__ lse,
    int B, int H, int swa_maxlen, int topk_maxlen,
    int swa_num_rows, int kv_num_rows, float scale)
{
    const int bid  = blockIdx.x;
    const int b    = bid / H;
    const int h    = bid % H;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int d0   = lane * DPT;
    const bool is_nope = (d0 < NOPE_DIM);
    const int  sblk    = lane >> 3;
    // Byte offset of this lane's 8 dims within a token's 576-byte data region:
    //   nope: 1 byte/dim (fp8) at d0 ; rope: 2 bytes/dim (bf16) at 448+(d0-448)*2.
    const int  boff    = is_nope ? d0 : (NOPE_DIM + (d0 - NOPE_DIM) * 2);

    __shared__ float sm[NWAVE];
    __shared__ float sl[NWAVE];
    __shared__ float sacc[NWAVE][HEAD_DIM];

    float qreg[DPT];
    const __hip_bfloat16* qrow = q + ((long)b * H + h) * HEAD_DIM + d0;
    #pragma unroll
    for (int i = 0; i < DPT; ++i) qreg[i] = (float)qrow[i];

    float acc[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] = 0.0f;
    float m_i = NEG, l_i = 0.0f;

    const int swa_len  = swa_lens[b];
    const int topk_len = topk_lens[b];
    const int ntok = swa_len + topk_len;

    // Software-pipelined chain over this wave's tokens: raw global loads for the NEXT
    // token are issued a full iteration ahead so the load latency overlaps compute.
    if (wave < ntok) {
        const unsigned char *td, *sc; bool valid;
        tok_desc(wave, swa_len, b, swa_maxlen, topk_maxlen, swa_indices, topk_indices,
                 swa_cache, kv_cache, swa_num_rows, kv_num_rows, td, sc, valid);
        uint2 raw[2];
        raw[0] = *reinterpret_cast<const uint2*>(td + boff);
        raw[1] = *reinterpret_cast<const uint2*>(td + boff + 8);
        unsigned char scb = sc[sblk];
        bool cur_valid = valid;

        for (int t = wave; t < ntok; t += NWAVE) {
            int tn = t + NWAVE;
            uint2 raw_n[2]; unsigned char scb_n = 0; bool nxt_valid = false;
            if (tn < ntok) {
                const unsigned char *tdn, *scn; bool vn;
                tok_desc(tn, swa_len, b, swa_maxlen, topk_maxlen, swa_indices, topk_indices,
                         swa_cache, kv_cache, swa_num_rows, kv_num_rows, tdn, scn, vn);
                raw_n[0] = *reinterpret_cast<const uint2*>(tdn + boff);       // issue next load
                raw_n[1] = *reinterpret_cast<const uint2*>(tdn + boff + 8);
                scb_n = scn[sblk];
                nxt_valid = vn;
            }

            // Dequant current token from registers.
            float K[DPT];
            if (!cur_valid) {
                #pragma unroll
                for (int i = 0; i < DPT; ++i) K[i] = 0.0f;
            } else if (is_nope) {
                const unsigned char* pb = reinterpret_cast<const unsigned char*>(raw);
                float esc = e8m0_to_float(scb);
                #pragma unroll
                for (int i = 0; i < DPT; ++i) K[i] = bf16_round(fp8_e4m3fn_to_float(pb[i]) * esc);
            } else {
                const __hip_bfloat16* rb = reinterpret_cast<const __hip_bfloat16*>(raw);
                #pragma unroll
                for (int i = 0; i < DPT; ++i) { float rv = (float)rb[i]; K[i] = (rv == rv) ? rv : 0.0f; }
            }

            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < DPT; ++i) part += qreg[i] * K[i];
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) part += __shfl_down(part, off, 64);
            float score = __shfl(part, 0, 64);

            float s = cur_valid ? (score * scale) : NEG;
            float m_new = fmaxf(m_i, s);
            float alpha = __expf(m_i - m_new);
            float p = cur_valid ? __expf(s - m_new) : 0.0f;
            float p_bf = bf16_round(p);
            l_i = l_i * alpha + p;
            #pragma unroll
            for (int i = 0; i < DPT; ++i) acc[i] = acc[i] * alpha + p_bf * K[i];
            m_i = m_new;

            raw[0] = raw_n[0]; raw[1] = raw_n[1]; scb = scb_n; cur_valid = nxt_valid;
        }
    }

    if (lane == 0) { sm[wave] = m_i; sl[wave] = l_i; }
    #pragma unroll
    for (int i = 0; i < DPT; ++i) sacc[wave][d0 + i] = acc[i];
    __syncthreads();

    if (wave == 0) {
        float mf = NEG;
        #pragma unroll
        for (int w = 0; w < NWAVE; ++w) mf = fmaxf(mf, sm[w]);
        float lf = 0.0f, wgt[NWAVE];
        #pragma unroll
        for (int w = 0; w < NWAVE; ++w) {
            float e = (sm[w] > NEG) ? __expf(sm[w] - mf) : 0.0f;
            wgt[w] = e; lf += e * sl[w];
        }
        float inv = (lf > 0.0f) ? (1.0f / lf) : 0.0f;
        __hip_bfloat16* orow = out + ((long)b * H + h) * HEAD_DIM + d0;
        #pragma unroll
        for (int i = 0; i < DPT; ++i) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWAVE; ++w) a += wgt[w] * sacc[w][d0 + i];
            orow[i] = (__hip_bfloat16)(a * inv);
        }
        if (lane == 0) lse[b * H + h] = mf + logf(lf > 0.0f ? lf : 1e-30f);
    }
}

extern "C" void sparse_mla_decode(
    const void* q, const void* swa_cache, const int* swa_indices, const int* swa_lens,
    const void* kv_cache, const int* topk_indices, const int* topk_lens,
    void* out, float* lse,
    int B, int H, int swa_maxlen, int topk_maxlen,
    int swa_num_rows, int kv_num_rows, float scale, void* stream)
{
    dim3 grid(B * H);
    dim3 block(NWAVE * 64);
    hipLaunchKernelGGL(sparse_mla_decode_kernel, grid, block, 0, (hipStream_t)stream,
        (const __hip_bfloat16*)q, (const unsigned char*)swa_cache, swa_indices, swa_lens,
        (const unsigned char*)kv_cache, topk_indices, topk_lens,
        (__hip_bfloat16*)out, lse,
        B, H, swa_maxlen, topk_maxlen, swa_num_rows, kv_num_rows, scale);
}
