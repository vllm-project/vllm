// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi-K3 DSpark draft MLA decode for gfx950 (MI350/MI355).
//
// The DSpark draft group is a NON-CAUSAL multi-token block: all `qlen` query
// positions of a request attend to the same committed KV prefix and never to a
// sibling block token.  vLLM's Triton MLA path handles that by flattening the
// block to one decode row per query token, so each of the `batch * qlen` rows
// re-reads the whole KV span -- qlen times the traffic.  This kernel keeps the
// block folded: all `qlen * num_heads` rows live in one 128-row workgroup tile
// and the KV span is read exactly once.
//
// The MLA latent (512) plus RoPE (64) is consumed as fp8 e4m3 through
// v_mfma_scale_f32_16x16x128_f8f6f4, which is why this is gfx950-only.
//
// Portions of this file and the bundled `opus/` headers are derived from AMD's
// aiter (https://github.com/ROCm/aiter), MIT licensed, and are redistributed
// here under those terms; see opus/opus.hpp for the original notice.

#include <cstdint>
#include <optional>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>

// Deliberately not ../torch_utils.h: it pulls <cuda_runtime.h>/<cublas_v2.h>,
// which only resolve for sources that go through hipify.  This file is
// HIP-native and is compiled as-is (hipify skips .hip), so it takes the two
// stable-ABI pieces it needs directly.
namespace {
inline hipStream_t vllm_current_hip_stream(int32_t device_index) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<hipStream_t>(stream_ptr);
}
}  // namespace

#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include "opus/hip_minimal.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cstdint>

#ifndef __launch_bounds__
#define __launch_bounds__(threads, blocks) \
    __attribute__((amdgpu_flat_work_group_size(1, threads), \
                   amdgpu_waves_per_eu(blocks)))
#endif

namespace kimi_k3_h40_draft {

#ifndef VLLM_K3_H40_PV_BATCH
#define VLLM_K3_H40_PV_BATCH 4
#endif
#ifndef VLLM_K3_H40_EP_PITCH
#define VLLM_K3_H40_EP_PITCH 520
#endif
#ifndef VLLM_K3_H40_Q_SUB
#define VLLM_K3_H40_Q_SUB 2
#endif
// Ablation probes -- ALL give WRONG results.  They exist to price a component
// by removing it and seeing how much time comes back.
// Stage KV through registers instead of buffer_load...lds.  A microbenchmark
// at the same 3.6 MB/WG footprint measures the direct-to-LDS DMA path at
// 14.4 GB/s/CU against 23.3 GB/s/CU for plain register loads at 8 in flight --
// the DMA path is latency-limited well below what the CU can do.  Since the
// kernel time equals its DMA time exactly (MFMA is fully hidden inside it),
// closing that gap should drop it onto the MFMA floor: 249.7 -> ~178 us.
// STAGE_BATCH = halves issued before their ds_writes, i.e. loads in flight
// = 5 * STAGE_BATCH, at 4 VGPRs each.
// Query-position masking mode, baked in at compile time so neither build pays
// for the other's branch.  1 = target verify: position p sees KV up to
// full_len-(qlen-1)+p.  0 = the DSpark draft group, which vLLM documents as a
// non-causal multi-token block -- every position sees the whole KV, so only
// the out-of-range tail guard remains and `qpos` (an integer divide per row)
// falls out as dead code.
// The draft block is non-causal; only the tail guard remains.
#define VLLM_K3_H40_CAUSAL 0
// Read a vLLM paged block table directly instead of a pre-expanded per-token
// index list.  Two reasons, both required for the in-engine draft path:
//   * The DSpark draft step runs under a FULL CUDA graph.  Expanding a block
//     table into token ids on the host side needs a data-dependent pack (the
//     batch is ragged), which is not capturable; reading the table is.
//   * A KV tile never straddles a page, because page_size is a multiple of
//     KV_TILE.  So one scalar page lookup per tile replaces KV_TILE gathered
//     index loads, and the 34 us/call expansion disappears entirely.
// The static_assert lives in the launcher: page_size % KV_TILE == 0.
// Always read vLLM's paged block table directly.
#define VLLM_K3_H40_BLOCK_TABLE 1
#ifndef VLLM_K3_H40_SCHED
#define VLLM_K3_H40_SCHED 0
#endif
#ifndef VLLM_K3_H40_QK_PIPE
#define VLLM_K3_H40_QK_PIPE 2
#endif
#ifndef VLLM_K3_H40_REDUCE_THREADS
#define VLLM_K3_H40_REDUCE_THREADS 128
#endif

struct kargs
{
    const void* __restrict__ q_ptr;       // [T, qlen*nhead, 576] fp8
    const void* __restrict__ kv_ptr;      // [rows, 576] fp8
    const int* __restrict__ kv_indptr;    // [T+1]
    const int* __restrict__ kv_indices;   // [sum(seq_len)]
    // Block-table mode only; both null otherwise.
    const int* __restrict__ kv_block_table; // [T, bt_stride]
    const int* __restrict__ kv_seq_lens;    // [T]
    int bt_stride;
    int page_size;
    float* __restrict__ part_m;           // [T, splits, rows_pad]
    float* __restrict__ part_l;           // [T, splits, rows_pad]
    float* __restrict__ part_acc;         // [T, splits, rows_pad, 512]
    void* __restrict__ out_ptr;           // [T, qlen*nhead, 512] bf16
    int T;
    int qlen;
    int nhead;
    int valid_rows;
    int rows_pad;
    int h_offset;
    int kv_rows;
    int num_splits;
    float qk_scale_log2;
    float output_scale;
};

struct traits
{
    static constexpr int Q_TILE = 16;
    static constexpr int Q_SUB = VLLM_K3_H40_Q_SUB;
    static constexpr int NUM_WARPS = 4;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int HEADS_PER_BLOCK = NUM_WARPS * Q_SUB * Q_TILE;

    static constexpr int KV_TILE = 128;
    static constexpr int D_QK = 576;
    static constexpr int D_V = 512;
    static constexpr int D_SLICES = 5; // 4x128 + a zero-padded 64 tail
    static constexpr int N_TILES = KV_TILE / 16;
    static constexpr int O_TILES = D_V / 16;
    static constexpr int K_BLOCKS = KV_TILE / 32;
    static constexpr int S_LEN = N_TILES * 4;

    // Same token permutation as pa_sparse_prefill_fp8_h40.  The first 32
    // 16-byte cells are V and the final four cells are K-only RoPE.
    static constexpr int ATOM = 128;
    static constexpr int PAD = 64;
    static constexpr int ROW = 36 * ATOM + PAD;
    static constexpr int TILE_BYTES = 16 * ROW;
    static constexpr int SLOTS_PER_WAVE = 16 / NUM_WARPS;
    static constexpr int SMEM_BYTES = 2 * TILE_BYTES;

    static constexpr int PV_BATCH = VLLM_K3_H40_PV_BATCH;
    static constexpr int QK_PIPE = VLLM_K3_H40_QK_PIPE;
    static_assert(QK_PIPE == 1 || QK_PIPE == 2);
    static_assert(O_TILES % PV_BATCH == 0);
    static constexpr int EP_PITCH =
        VLLM_K3_H40_EP_PITCH; // fp32 elements per staged head row
    static_assert(SMEM_BYTES <= 163840);
    static_assert(
        NUM_WARPS * Q_TILE * EP_PITCH * (int)sizeof(float)
        <= SMEM_BYTES);
};

__host__ __device__ inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

__global__ void partial_kernel(kargs args);
__global__ void reduce_kernel(kargs args);

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)

namespace device {

using namespace opus;
using fp8_t = _BitInt(8);
using u8_t = unsigned char;
using i32x2 = int __attribute__((ext_vector_type(2)));
using i32x4 = int __attribute__((ext_vector_type(4)));
using i32x8 = int __attribute__((ext_vector_type(8)));
using f32x4 = float __attribute__((ext_vector_type(4)));

constexpr float LOWEST = -3.4028234663852886e38f;

__device__ inline f32x4 mma_unit(const i32x8& a,
                                const i32x8& b,
                                const f32x4& c)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 0, 0, 0, 127, 0, 127);
}

__device__ inline f32x4 mma_p(const i32x8& v,
                             const i32x8& p,
                             const f32x4& c,
                             int p_scale)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        v, p, c, 0, 0, 0, 127, 0, p_scale);
}

__device__ inline i32x8 pack32(const i32x4& lo, const i32x4& hi)
{
    i32x8 out;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        out[i] = lo[i];
        out[4 + i] = hi[i];
    }
    return out;
}

__device__ inline float reduce_max_groups(float v)
{
    vector_t<u32_t, 2> r32 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    v = max(std::bit_cast<float>(r32.x), std::bit_cast<float>(r32.y));
    vector_t<u32_t, 2> r16 = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    return max(std::bit_cast<float>(r16.x), std::bit_cast<float>(r16.y));
}

__device__ inline float reduce_sum_groups(float v)
{
    vector_t<u32_t, 2> r32 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    v = std::bit_cast<float>(r32.x) + std::bit_cast<float>(r32.y);
    vector_t<u32_t, 2> r16 = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    return std::bit_cast<float>(r16.x) + std::bit_cast<float>(r16.y);
}

__device__ inline void lds_barrier()
{
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
}

} // namespace device

__global__ __launch_bounds__(traits::BLOCK_SIZE, 1)
void partial_kernel(kargs args)
{
    using namespace opus;
    using namespace device;
    using T = traits;

    const int req = (int)__builtin_amdgcn_workgroup_id_x();
    const int h_block = (int)__builtin_amdgcn_workgroup_id_y();
    const int split = (int)__builtin_amdgcn_workgroup_id_z();
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    int lane = tid & 63;
    asm volatile("" : "+v"(lane));
    const int warp = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int c = lane & 15;
    const int g = lane >> 4;

#if VLLM_K3_H40_BLOCK_TABLE
    const int full_begin = 0;
    const int full_len   = args.kv_seq_lens[req];
#else
    const int full_begin = args.kv_indptr[req];
    const int full_len = args.kv_indptr[req + 1] - full_begin;
#endif
    const int full_tiles = ceil_div(full_len, T::KV_TILE);
    const int tiles_per_split = ceil_div(full_tiles, args.num_splits);
    const int first_tile = split * tiles_per_split;
    const int last_tile = min(first_tile + tiles_per_split, full_tiles);
    const int segment_begin = first_tile * T::KV_TILE;
    const int segment_end = min(last_tile * T::KV_TILE, full_len);
    const int segment_len = max(0, segment_end - segment_begin);
    const int num_tiles = ceil_div(segment_len, T::KV_TILE);

    const int h_base =
        args.h_offset + h_block * T::HEADS_PER_BLOCK;
    const int h_wave = warp * (T::Q_SUB * T::Q_TILE);

    __shared__ char smem[T::SMEM_BYTES];

    auto g_q = make_gmem(
        reinterpret_cast<const fp8_t*>(args.q_ptr)
            + ((size_t)req * args.valid_rows + h_base) * T::D_QK,
        (unsigned)max(0, args.valid_rows - h_base) * T::D_QK);

    i32x8 v_q[T::Q_SUB][T::D_SLICES];
    const i32x4 zero4{0, 0, 0, 0};
#pragma unroll
    for(int qs = 0; qs < T::Q_SUB; ++qs)
    {
        const int q_row = (h_wave + qs * T::Q_TILE + c) * T::D_QK;
        const int q_off = q_row + 16 * g;
#pragma unroll
        for(int ds = 0; ds < T::D_SLICES; ++ds)
        {
            const i32x4 lo = __builtin_bit_cast(
                i32x4, load<16>(g_q, q_off + ds * 128));
            const i32x4 hi = ds == T::D_SLICES - 1
                ? zero4
                : __builtin_bit_cast(
                      i32x4, load<16>(g_q, q_off + ds * 128 + 64));
            v_q[qs][ds] = pack32(lo, hi);
        }
    }

    f32x4 v_o[T::Q_SUB][T::O_TILES];
#pragma unroll
    for(int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
        for(int ot = 0; ot < T::O_TILES; ++ot)
            v_o[qs][ot] = f32x4{0.f, 0.f, 0.f, 0.f};

    float m_row[T::Q_SUB];
    float m_ref[T::Q_SUB];
    float l_row[T::Q_SUB];
#pragma unroll
    for(int qs = 0; qs < T::Q_SUB; ++qs)
    {
        m_row[qs] = LOWEST;
        m_ref[qs] = LOWEST;
        l_row[qs] = 0.f;
    }

    if(num_tiles > 0)
    {
#if !VLLM_K3_H40_BLOCK_TABLE
        // Flat-index mode addresses the whole cache from one descriptor, so it
        // is limited to kv_rows * 576 < 2^32; the launcher rejects anything
        // bigger rather than aliasing silently.
        auto g_kv = make_gmem(
            reinterpret_cast<const fp8_t*>(args.kv_ptr),
            (unsigned)args.kv_rows * T::D_QK);
        auto g_idx = make_gmem(
            args.kv_indices + full_begin + segment_begin,
            (unsigned)segment_len * sizeof(int));
#endif

        const int qk_base =
            (c >> 2) * T::ROW + (c & 3) * 16 + g * T::ATOM;
        const int pv_base = g * T::ROW + c * 8;
        const int p_lane = 16 * ((lane & 7) >> 2) + (lane & 3);
        const int d_lane = (lane >> 3) * 16;

        auto load_rows = [&](int tile_idx,
                             int (&rows)[T::SLOTS_PER_WAVE],
                             int& page) {
#if VLLM_K3_H40_BLOCK_TABLE
            // segment_begin is a multiple of KV_TILE and page_size is a
            // multiple of KV_TILE, so the whole tile lives in one page and the
            // row ids are consecutive.  pg is clamped because the prologue
            // prefetches one tile past the end; that tile's data is never
            // consumed, but its address still has to be in range.
            //
            // rows[] are PAGE-LOCAL, and the page's byte offset is folded into
            // a 64-bit base pointer by the caller.  A global row index would
            // not fit: a production KV cache is 3846 pages x 3072 tokens, and
            // 11.8e6 * 576 = 6.8e9 overflows the 32-bit buffer offset -- 37% of
            // the cache aliases onto the wrong rows.  That is invisible at
            // benchmark sizes (528 pages = 0.93 GB) and fatal in the engine.
            const int tile_tok = segment_begin + tile_idx * T::KV_TILE;
            int pg             = tile_tok / args.page_size;
            pg                 = pg < args.bt_stride ? pg : args.bt_stride - 1;
            page               = __builtin_amdgcn_readfirstlane(
                args.kv_block_table[req * args.bt_stride + pg]);
            const int base     = tile_tok - pg * args.page_size;
#else
            page = 0;
#endif
#pragma unroll
            for(int half = 0; half < T::SLOTS_PER_WAVE; ++half)
            {
                const int s_hi = T::SLOTS_PER_WAVE * warp + half;
                const int p =
                    32 * (s_hi >> 2) + 4 * (s_hi & 3) + p_lane;
#if VLLM_K3_H40_BLOCK_TABLE
                rows[half] = base + p;
#else
                rows[half] =
                    load(g_idx, tile_idx * T::KV_TILE + p)[0];
#endif
            }
        };

        auto issue_copy = [&](const int (&rows)[T::SLOTS_PER_WAVE],
                              int page,
                              int buf) {
            char* const dst = smem + buf * T::TILE_BYTES;
#if VLLM_K3_H40_BLOCK_TABLE
            // One descriptor per tile, based at the page.  The 64-bit multiply
            // happens here, in scalar registers, so every offset the loads see
            // is page-local (< page_size * 576 = 1.77 MB) and cannot overflow.
            auto g_kv = make_gmem(
                reinterpret_cast<const fp8_t*>(args.kv_ptr)
                    + (size_t)page * (size_t)args.page_size * T::D_QK,
                (unsigned)args.page_size * T::D_QK);
#endif
#pragma unroll
            for(int half = 0; half < T::SLOTS_PER_WAVE; ++half)
            {
                const int s_hi = T::SLOTS_PER_WAVE * warp + half;
                const unsigned goff =
                    (unsigned)rows[half] * T::D_QK
                    + (unsigned)d_lane;
#pragma unroll
                for(int q = 0; q < 4; ++q)
                    g_kv.template async_load<16>(
                        dst + s_hi * T::ROW + q * 1024,
                        goff + (unsigned)q * 128u);
                if(d_lane < 64)
                    g_kv.template async_load<16>(
                        dst + s_hi * T::ROW + 4 * 1024,
                        goff + 4u * 128u);
            }
        };

        int rows0[T::SLOTS_PER_WAVE];
        int rows1[T::SLOTS_PER_WAVE];
        int page0 = 0, page1 = 0;
        load_rows(0, rows0, page0);
        issue_copy(rows0, page0, 0);
        load_rows(1, rows0, page0);

        int buf = 0;
        for(int tile = 0; tile < num_tiles; ++tile)
        {
            s_waitcnt_vmcnt(0_I);
            lds_barrier();

            const bool has_next = tile + 1 < num_tiles;
            if(has_next)
            {
                issue_copy(rows0, page0, buf ^ 1);
                load_rows(tile + 2, rows1, page1);
            }

            auto s_kv = make_smem(
                reinterpret_cast<fp8_t*>(
                    smem + buf * T::TILE_BYTES));

            float scores[T::Q_SUB][T::S_LEN];

            auto load_k = [&](auto i_nt,
                              i32x8 (&kk)[T::D_SLICES]) {
                constexpr int nt = decltype(i_nt)::value;
                constexpr int nt_off =
                    (nt >> 1) * 4 * T::ROW + (nt & 1) * 64;
                static_for<T::D_SLICES>([&](auto i_ds) {
                    constexpr int ds = i_ds.value;
                    constexpr int off0 = nt_off + ds * 1024;
                    const i32x4 lo = __builtin_bit_cast(
                        i32x4,
                        s_kv.template _load<16>(qk_base + off0));
                    const i32x4 hi =
                        ds == T::D_SLICES - 1
                        ? zero4
                        : __builtin_bit_cast(
                              i32x4,
                              s_kv.template _load<16>(
                                  qk_base + off0 + 4 * T::ATOM));
                    kk[ds] = pack32(lo, hi);
                });
            };

            i32x8 kbuf[T::QK_PIPE][T::D_SLICES];
            if constexpr(T::QK_PIPE == 2)
                load_k(number<0>{}, kbuf[0]);
            static_for<T::N_TILES>([&](auto i_nt) {
                constexpr int nt = i_nt.value;
                if constexpr(T::QK_PIPE == 2)
                {
                    if constexpr(nt + 1 < T::N_TILES)
                        load_k(number<nt + 1>{}, kbuf[(nt + 1) & 1]);
                }
                else
                    load_k(number<nt>{}, kbuf[0]);

                f32x4 acc[T::Q_SUB];
#pragma unroll
                for(int qs = 0; qs < T::Q_SUB; ++qs)
                    acc[qs] = f32x4{0.f, 0.f, 0.f, 0.f};
                static_for<T::D_SLICES>([&](auto i_ds) {
                    constexpr int ds = i_ds.value;
#pragma unroll
                    for(int qs = 0; qs < T::Q_SUB; ++qs)
                        acc[qs] = mma_unit(
                            kbuf[T::QK_PIPE == 2 ? (nt & 1) : 0][ds],
                            v_q[qs][ds],
                            acc[qs]);
                });
#pragma unroll
                for(int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                        scores[qs][nt * 4 + i] = acc[qs][i];
#if VLLM_K3_H40_SCHED
#pragma unroll
                for(int rep = 0; rep < 3; ++rep)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
                    __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);
                }
#endif
            });

            const int tile_global =
                segment_begin + tile * T::KV_TILE;
#if VLLM_K3_H40_CAUSAL
            const bool maybe_mask =
                tile_global + T::KV_TILE - 1 + args.qlen - 1
                >= full_len;
#else
            // Non-causal needs no (qlen-1) slack: the predicate is exact and
            // one fewer tile takes the masked path.
            const bool maybe_mask =
                tile_global + T::KV_TILE - 1 >= full_len;
#endif

            i32x8 v_p[T::Q_SUB];
            float bias[T::Q_SUB];
            int dexp[T::Q_SUB];
#pragma unroll
            for(int qs = 0; qs < T::Q_SUB; ++qs)
            {
                const int row =
                    h_base + h_wave + qs * T::Q_TILE + c;
#if VLLM_K3_H40_CAUSAL
                const int qpos = row / args.nhead;
                const int shift = args.qlen - 1 - qpos;
#else
                const int shift = 0;
#endif

                float local_max = LOWEST;
                // Only the tail tile can be masked, but maybe_mask was a plain
                // runtime scalar, so the per-element select was emitted on
                // every tile: ATT charged 64 v_cndmask 2.2% of the kernel for
                // work that matters on 1 tile in ~49.  Peel it onto a
                // compile-time flag so the common path has no kv_pos, no
                // compare, no select and no store-back.
                auto scan = [&]<bool MASK>() {
#pragma unroll
                    for(int a = 0; a < T::N_TILES; ++a)
#pragma unroll
                        for(int b = 0; b < 4; ++b)
                        {
                            float s = scores[qs][a * 4 + b];
                            if constexpr(MASK)
                            {
                                const int kv_pos =
                                    tile_global + 16 * a + 4 * g + b;
                                if(kv_pos + shift >= full_len)
                                    s = LOWEST;
                                scores[qs][a * 4 + b] = s;
                            }
                            local_max = max(local_max, s);
                        }
                };
                if(maybe_mask)
                    scan.template operator()<true>();
                else
                    scan.template operator()<false>();

                const float row_max =
                    reduce_max_groups(local_max)
                    * args.qk_scale_log2;
                const float new_m =
                    __builtin_floorf(max(m_row[qs], row_max));
                if(m_ref[qs] == LOWEST)
                    m_ref[qs] = new_m;
                m_row[qs] = new_m;
                bias[qs] = -m_ref[qs];
                dexp[qs] = (int)(new_m - m_ref[qs]);
                // P is scaled by 2^(7-dexp) to land in e4m3's range.  Folding
                // that into exp2's addend costs nothing -- it is already an fma
                // operand -- and removes one v_ldexp_f32 per score element.
                // dexp is exact (new_m and m_ref are both floored), so this is
                // an identity, not an approximation.  local_sum then carries
                // the factor and is unscaled once per row instead of 32 times.
                const float pbias = bias[qs] + (float)(7 - dexp[qs]);

                float local_sum = 0.f;
                i32x8 packed;
#pragma unroll
                for(int a = 0; a < T::N_TILES; ++a)
                {
                    vector_t<float, 4> p4;
#pragma unroll
                    for(int b = 0; b < 4; ++b)
                    {
                        const float p = __builtin_amdgcn_exp2f(
                            __builtin_fmaf(
                                scores[qs][a * 4 + b],
                                args.qk_scale_log2,
                                pbias));
                        local_sum += p;
                        p4[b] = p;
                    }
                    packed[a] = __builtin_bit_cast(
                        int, cast<fp8_t>(p4));
                }
                v_p[qs] = packed;
                l_row[qs] += __builtin_ldexpf(
                    reduce_sum_groups(local_sum), dexp[qs] - 7);
            }

            s_waitcnt_lgkmcnt(0_I);
            auto load_v = [&](auto i_ot) {
                constexpr int ot = decltype(i_ot)::value;
                i32x8 v;
                static_for<T::K_BLOCKS>([&](auto i_beta) {
                    constexpr int beta = i_beta.value;
                    const i32x2 r = __builtin_bit_cast(
                        i32x2,
                        s_kv.template _tr_load<
                            8,
                            ot * T::ATOM
                                + beta * 4 * T::ROW>(pv_base));
                    v[2 * beta] = r[0];
                    v[2 * beta + 1] = r[1];
                });
                return v;
            };

            asm volatile("" ::: "memory");
            s_waitcnt_lgkmcnt(0_I);
            static_for<T::O_TILES / T::PV_BATCH>([&](auto i_batch) {
                constexpr int first =
                    i_batch.value * T::PV_BATCH;
                i32x8 vv[T::PV_BATCH];
                static_for<T::PV_BATCH>([&](auto i) {
                    constexpr int j = i.value;
                    vv[j] = load_v(number<first + j>{});
                });
                s_waitcnt_lgkmcnt(0_I);
#pragma unroll
                for(int j = 0; j < T::PV_BATCH; ++j)
                    asm volatile("" : "+v"(vv[j]) ::);
                static_for<T::PV_BATCH>([&](auto i) {
                    constexpr int j = i.value;
#pragma unroll
                    for(int qs = 0; qs < T::Q_SUB; ++qs)
                        v_o[qs][first + j] = mma_p(
                            vv[j],
                            v_p[qs],
                            v_o[qs][first + j],
                            127 + dexp[qs]);
                });
            });
            if(has_next)
            {
#pragma unroll
                for(int i = 0; i < T::SLOTS_PER_WAVE; ++i)
                    rows0[i] = rows1[i];
                page0 = page1;
            }
            buf ^= 1;
        }
    }

    // The partial accumulator's native C layout scatters one wave over 16
    // head rows.  Reuse the now-dead KV LDS as a transpose buffer so global
    // stores are contiguous.  The barrier keeps an early wave from clobbering
    // the final tile while another wave is still in PV.
    lds_barrier();

    const size_t ml_base =
        ((size_t)req * args.num_splits + split)
        * args.rows_pad;
    if(g == 0)
    {
#pragma unroll
        for(int qs = 0; qs < T::Q_SUB; ++qs)
        {
            const int row =
                h_base + h_wave + qs * T::Q_TILE + c;
            if(row < args.valid_rows)
            {
                args.part_m[ml_base + row] = m_ref[qs];
                args.part_l[ml_base + row] = l_row[qs];
            }
        }
    }

    auto s_ep = make_smem(
        reinterpret_cast<float*>(smem)
        + warp * T::Q_TILE * T::EP_PITCH);
    const size_t acc_base = ml_base * T::D_V;
#pragma unroll
    for(int qs = 0; qs < T::Q_SUB; ++qs)
    {
#pragma unroll
        for(int ot = 0; ot < T::O_TILES; ++ot)
        {
            const f32x4 x = v_o[qs][ot] * 0.0078125f;
            s_ep.template store<4>(
                __builtin_bit_cast(vector_t<float, 4>, x),
                c * T::EP_PITCH + ot * 16 + g * 4);
        }
        s_waitcnt_lgkmcnt(0_I);

#pragma unroll
        for(int hh = 0; hh < T::Q_TILE; ++hh)
        {
            const int row =
                h_base + h_wave + qs * T::Q_TILE + hh;
            if(row < args.valid_rows)
            {
#pragma unroll
                for(int half = 0; half < 2; ++half)
                {
                    const auto x = s_ep.template load<4>(
                        hh * T::EP_PITCH + half * 256 + lane * 4);
                    float* dst =
                        args.part_acc + acc_base
                        + (size_t)row * T::D_V
                        + half * 256 + lane * 4;
                    *reinterpret_cast<vector_t<float, 4>*>(dst) = x;
                }
            }
        }
        s_waitcnt_lgkmcnt(0_I);
    }
}

__global__ __launch_bounds__(VLLM_K3_H40_REDUCE_THREADS, 1)
void reduce_kernel(kargs args)
{
    using device::f32x4;
    const int linear_row = (int)__builtin_amdgcn_workgroup_id_x();
    const int req = linear_row / args.valid_rows;
    const int row = linear_row - req * args.valid_rows;
    const int tid = (int)__builtin_amdgcn_workitem_id_x();

    __shared__ float weights[VLLM_K3_H40_REDUCE_THREADS];
    __shared__ float scratch[VLLM_K3_H40_REDUCE_THREADS];

    const size_t req_base =
        (size_t)req * args.num_splits * args.rows_pad;
    float l = 0.f;
    float m = device::LOWEST;
    if(tid < args.num_splits)
    {
        const size_t off =
            req_base + (size_t)tid * args.rows_pad + row;
        l = args.part_l[off];
        if(l > 0.f)
            m = args.part_m[off];
    }
    scratch[tid] = m;
    __builtin_amdgcn_s_barrier();
    for(int stride = VLLM_K3_H40_REDUCE_THREADS / 2;
        stride > 0;
        stride >>= 1)
    {
        if(tid < stride)
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        __builtin_amdgcn_s_barrier();
    }
    const float global_m = scratch[0];

    float w = 0.f;
    if(tid < args.num_splits && l > 0.f)
        w = __builtin_amdgcn_exp2f(m - global_m);
    weights[tid] = w;
    scratch[tid] = w * l;
    __builtin_amdgcn_s_barrier();
    for(int stride = VLLM_K3_H40_REDUCE_THREADS / 2;
        stride > 0;
        stride >>= 1)
    {
        if(tid < stride)
            scratch[tid] += scratch[tid + stride];
        __builtin_amdgcn_s_barrier();
    }
    const float denom = scratch[0];

    auto* out = reinterpret_cast<__bf16*>(args.out_ptr);
    const size_t req_acc =
        (size_t)req * args.num_splits
        * args.rows_pad * traits::D_V;
    for(int col = tid * 4;
        col < traits::D_V;
        col += VLLM_K3_H40_REDUCE_THREADS * 4)
    {
        f32x4 acc{0.f, 0.f, 0.f, 0.f};
        for(int s = 0; s < args.num_splits; ++s)
        {
            const size_t off =
                req_acc
                + ((size_t)s * args.rows_pad + row) * traits::D_V
                + col;
            const f32x4 x =
                *reinterpret_cast<const f32x4*>(args.part_acc + off);
            acc += x * weights[s];
        }
        const float scale =
            denom > 0.f ? args.output_scale / denom : 0.f;
        const f32x4 y = acc * scale;
        __bf16* dst =
            out + ((size_t)req * args.valid_rows + row) * traits::D_V + col;
#pragma unroll
        for(int i = 0; i < 4; ++i)
            dst[i] = (__bf16)y[i];
    }
}

#else

__global__ void partial_kernel(kargs) {}
__global__ void reduce_kernel(kargs) {}

#endif

} // namespace kimi_k3_h40_draft

#ifndef __HIP_DEVICE_COMPILE__

namespace {
// One workgroup per CU for exactly one round.  This depends only on shapes,
// never on seq_lens, so it stays valid under CUDA graph capture -- the draft
// step is captured as a FULL graph and a value-dependent launch would freeze.
constexpr int kNumComputeUnits = 256;
}  // namespace

void kimi_k3_h40_draft_mla_decode(
    torch::stable::Tensor const& q, torch::stable::Tensor const& kv_cache,
    torch::stable::Tensor const& block_table,
    torch::stable::Tensor const& seq_lens,
    torch::stable::Tensor& partial_max, torch::stable::Tensor& partial_sum,
    torch::stable::Tensor& partial_acc, torch::stable::Tensor& out,
    int64_t query_len, int64_t num_splits, double qk_scale_log2,
    double output_scale) {
  using namespace kimi_k3_h40_draft;
  using T = traits;

  STD_TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
                  "q must be fp8_e4m3 on device");
  STD_TORCH_CHECK(
      kv_cache.is_cuda() && kv_cache.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
      "kv_cache must be fp8_e4m3 on device");
  STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "out must be bfloat16");
  STD_TORCH_CHECK(block_table.scalar_type() == torch::headeronly::ScalarType::Int &&
                      seq_lens.scalar_type() == torch::headeronly::ScalarType::Int,
                  "block_table and seq_lens must be int32");
  STD_TORCH_CHECK(q.dim() == 3 && kv_cache.dim() == 3 && block_table.dim() == 2,
                  "expected q[S,H,D], kv_cache[blocks,page,D], bt[T,pages]");

  int64_t const total_rows = q.size(0);
  int64_t const num_heads = q.size(1);
  STD_TORCH_CHECK(q.size(2) == T::D_QK && out.size(2) == T::D_V,
                  "MLA head dim must be 576 (512 latent + 64 rope)");

  int64_t const num_reqs = seq_lens.size(0);
  STD_TORCH_CHECK(query_len >= 1 && total_rows == num_reqs * query_len,
                  "q rows must be num_reqs * query_len");

  int64_t const page_size = kv_cache.size(1);
  STD_TORCH_CHECK(page_size % T::KV_TILE == 0,
                  "page_size must be a multiple of ", T::KV_TILE,
                  " so a KV tile never straddles a page");
  STD_TORCH_CHECK(block_table.stride(1) == 1, "block_table rows must be dense");

  int64_t const rows = num_heads * query_len;
  int64_t const workgroups = num_reqs * num_splits;
  STD_TORCH_CHECK(partial_acc.numel() >= workgroups * rows * T::D_V &&
                      partial_max.numel() >= workgroups * rows &&
                      partial_sum.numel() >= workgroups * rows,
                  "workspace too small for ", workgroups, " workgroups");

  kargs args{};
  args.q_ptr = q.data_ptr();
  args.kv_ptr = kv_cache.data_ptr();
  args.kv_block_table = static_cast<int const*>(block_table.data_ptr());
  args.kv_seq_lens = static_cast<int const*>(seq_lens.data_ptr());
  args.bt_stride = static_cast<int>(block_table.stride(0));
  args.page_size = static_cast<int>(page_size);
  args.part_m = static_cast<float*>(partial_max.data_ptr());
  args.part_l = static_cast<float*>(partial_sum.data_ptr());
  args.part_acc = static_cast<float*>(partial_acc.data_ptr());
  args.out_ptr = out.data_ptr();
  args.T = static_cast<int>(num_reqs);
  args.qlen = static_cast<int>(query_len);
  args.nhead = static_cast<int>(num_heads);
  args.valid_rows = static_cast<int>(rows);
  args.rows_pad = args.valid_rows;
  // Page-local addressing keeps every buffer offset under 2^32; a global row
  // index would not (a 6.8 GB cache overflows it and aliases 37% of its rows).
  args.kv_rows = static_cast<int>(page_size);
  args.num_splits = static_cast<int>(num_splits);
  args.qk_scale_log2 = static_cast<float>(qk_scale_log2);
  args.output_scale = static_cast<float>(output_scale);

  torch::stable::accelerator::DeviceGuard const device_guard(
      static_cast<torch::stable::accelerator::DeviceIndex>(q.get_device()));
  hipStream_t const stream = vllm_current_hip_stream(q.get_device());

  dim3 const grid(static_cast<unsigned>(num_reqs),
                  static_cast<unsigned>(ceil_div(args.valid_rows,
                                                 T::HEADS_PER_BLOCK)),
                  static_cast<unsigned>(num_splits));
  partial_kernel<<<grid, dim3(T::BLOCK_SIZE), 0, stream>>>(args);
  reduce_kernel<<<dim3(static_cast<unsigned>(num_reqs * rows)),
                  dim3(VLLM_K3_H40_REDUCE_THREADS), 0, stream>>>(args);
}

#endif  // __HIP_DEVICE_COMPILE__
