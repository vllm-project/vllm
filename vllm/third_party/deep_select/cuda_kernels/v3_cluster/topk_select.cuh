/*
TopK select kernel bf16 cluster variant. Use multiple CTAs to calculate one sequence to avoid wave quantization

[SGLang] Algorithm:
- The row is split contiguously across the CTAs and streamed through a ring of bulk copies
- A slice of at most RAW_WINDOW elements is its own candidate set. A longer slice first takes a pivot from every
  CTA's leading RAW_WINDOW elements, then keeps only the elements above it while the rest streams in
- The cluster runs a two-pass (MSB, LSB) radix select over all candidates: every CTA pushes its 256-bin histogram
  to every CTA, so each one finds the same pivot on its own
- Each warp claims output slots with one atomic on CTA0's cursor and stores its selected indices directly. With
  `sorted_index`, each CTA instead sorts its selection by index and stores it after the lower CTAs' selections
*/

#pragma once

#include <cutlass/kernel_launch.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <kerutils/kerutils.cuh>
#include <utility>

#include "structs.h"
#include "cuda_kernels/utils.cuh"
#include "cuda_kernels/common_parts.cuh"

namespace topk_select_bf16_cluster {

template<typename Config>
class TopkSelectKernelBF16Cluster : public topk_select_common::TopkSelectKernelBF16Base<Config> {
    using BF16Base = topk_select_common::TopkSelectKernelBF16Base<Config>;
public:
    using ValueT = typename BF16Base::ValueT;
    using OutIdxT = typename BF16Base::OutIdxT;
    using TmaParams = typename BF16Base::TmaParams;
    using SharedMemoryPlanBase = typename BF16Base::SharedMemoryPlanBase;
    using EpilogueT = typename BF16Base::EpilogueT;
    using BF16Base::NUM_THREADS;
    using BF16Base::MAX_TOPK;

    static_assert(Config::target_occupancy == 1);
    static_assert(MAX_TOPK == 512 || MAX_TOPK == 1024);
    static_assert(Config::cluster_size == 2 || Config::cluster_size == 3 || Config::cluster_size == 4
                  || Config::cluster_size == 8 || Config::cluster_size == 16);
    static constexpr uint32_t CLUSTER_SIZE = Config::cluster_size;


    // A histogram travels as 256 u16 bins; the MSB one is followed by a 16-byte chunk whose first word is the
    // sender's NaN flag
    static constexpr uint32_t XCHG_MSB_WORDS = 128 + 4;
    static constexpr uint32_t XCHG_LSB_WORDS = 128;
    using RecvMsbT = uint32_t[Config::cluster_size][XCHG_MSB_WORDS];
    using RecvLsbT = uint32_t[Config::cluster_size][XCHG_LSB_WORDS];
    struct RecvPair {
        CUTE_ALIGNAS(16) RecvMsbT msb;
        CUTE_ALIGNAS(16) RecvLsbT lsb;
    };
    // A slice streams through the ring in stages of two unit rows, one 16-byte unit per thread per row
    static constexpr uint32_t UNIT_ROW = NUM_THREADS * 8;
    static constexpr uint32_t STAGE = 2 * UNIT_ROW;
    static constexpr uint32_t RAW_WINDOW = 32768;
    static constexpr uint32_t WINDOW_ROWS = RAW_WINDOW / UNIT_ROW;
    static constexpr uint32_t WINDOW_STAGES = RAW_WINDOW / STAGE;
    static_assert(NUM_THREADS == 512 && WINDOW_ROWS * 4 <= 32);
    // Room for a long slice's candidates; arbitrary, it only has to beat topk plus a typical stage's hits
    static constexpr uint32_t CAND_CAP = 4096;
    static_assert(CAND_CAP >= MAX_TOPK && CAND_CAP % NUM_THREADS == 0);
    struct StreamTail {
        // The window exchange is over before the first candidate is written
        union {
            RecvPair window_recv;
            uint64_t cand[CAND_CAP];    // (value bits << 32) | vocab index
        };
        RecvPair final_recv;
        CUTE_ALIGNAS(16) uint32_t xh_msb[256];
        CUTE_ALIGNAS(16) uint32_t xh_lsb[BF16Base::NUM_RECONSTRUCT_BUCKETS + 4];   // [256] is the LSB pass's sink
        CUTE_ALIGNAS(16) uint32_t xh_sum[256];
        uint32_t warp_cnt[BF16Base::NUM_WARPS];
        kerutils::transac_bar_t xbar[2];    // window exchange
        kerutils::transac_bar_t ybar[2];    // final exchange
        uint64_t xctr;      // CTA0's copy is the row's output cursor: (count above the pivot) << 32 | (ties)
        uint32_t cand_cnt;
        uint32_t eq_cnt;
        uint32_t sel_cnt;       // `sorted_index`: this CTA's selected elements, and the ties among them
        uint32_t sel_eq;
        uint32_t rank_gt[Config::cluster_size];     // `sorted_index`: every CTA's count above the pivot and at it
        uint32_t rank_eq[Config::cluster_size];
        uint32_t warp_pre[BF16Base::NUM_WARPS];     // `sorted_index`: selected elements before each warp's bitmap run
        uint32_t reconstruct_pivot_bucket;
        uint32_t reconstruct_num_should_select;
    };
    // The ring takes whatever the dynamic shared memory leaves; 225 KB keeps clear of the 227 KB opt-in limit
    static constexpr uint32_t SMEM_BUDGET = 225 * 1024;
    static constexpr uint32_t NUM_STAGES = cute::min(16u, (uint32_t)((SMEM_BUDGET - (sizeof(StreamTail) + 1023) / 1024 * 1024 - 16 * (sizeof(transac_bar_t) + 4)) / (STAGE * sizeof(ValueT))));
    static_assert(NUM_STAGES > WINDOW_STAGES);
    // Stages per streaming step; arbitrary, measured better than one stage per step
    static constexpr uint32_t STREAM_BATCH = 4;
    static_assert(STREAM_BATCH <= NUM_STAGES - WINDOW_STAGES);
    struct StreamPlan : StreamTail {
        CUTE_ALIGNAS(1024) ValueT ring[NUM_STAGES][STAGE];
        kerutils::transac_bar_t full[NUM_STAGES];
        uint32_t stage_off[NUM_STAGES];     // slice offset of the stage in each ring slot, published by its barrier
    };
    using PlanT = StreamPlan;
    // A row of at most topk elements takes the base kernel's epilogue, laid out as its plan
    static_assert(sizeof(SharedMemoryPlanBase) <= sizeof(StreamPlan));

    static __device__ __forceinline__ void st_async_v4(uint32_t dst, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t mbar) {
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(dst), "r"(a), "r"(b), "r"(c), "r"(d), "r"(mbar) : "memory");
    }

    static __device__ __forceinline__ void bulk_g2s(void *dst, const void *src, uint32_t bytes, transac_bar_t &bar) {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(cute::cast_smem_ptr_to_uint(dst)), "l"(src), "r"(bytes), "r"(cute::cast_smem_ptr_to_uint(&bar)) : "memory");
    }

    // Cluster-scope acquire: pairs with the release of the `st.async` complete-tx, which orders everything the
    // sender's CTA did before it (CTA0 zeroing `xctr` included) before this CTA's later accesses
    static __device__ __forceinline__ void wait_acquire_cluster(transac_bar_t &bar, uint32_t phase) {
        asm volatile(
            "{\n.reg .pred p;\n"
            "LAB_WAIT_%=:\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
            "@!p bra LAB_WAIT_%=;\n}\n"
            :: "r"(cute::cast_smem_ptr_to_uint(&bar)), "r"(phase) : "memory");
    }

    // +0 and -0 get distinct histogram keys but compare equal, so -0 becomes +0 before either sees it
    static __device__ __forceinline__ uint32_t canonical_zero_x2(uint32_t v) {
        uint32_t r;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(0u));
        return r;
    }

    static __device__ __forceinline__ bool has_nan_x2(uint32_t v) {
        return (v & 0x7FFFu) > 0x7F80u || ((v >> 16) & 0x7FFFu) > 0x7F80u;
    }

    // Pack this CTA's u32 bins into u16 and push them into slot `rank` of every CTA's `recv`
    template<uint32_t WORDS>
    static __device__ __forceinline__ void push_hist(const uint32_t *bins, uint32_t (&recv)[CLUSTER_SIZE][WORDS],
                                                     transac_bar_t &bar, uint32_t rank, uint32_t nan_flag) {
        constexpr uint32_t NCHUNK = WORDS / 4;
        const uint32_t recv_base = cute::cast_smem_ptr_to_uint(&recv[rank][0]);
        const uint32_t bar_addr = cute::cast_smem_ptr_to_uint(&bar);
        CUTE_UNROLL
        for (uint32_t i = threadIdx.x; i < NCHUNK * CLUSTER_SIZE; i += NUM_THREADS) {
            uint32_t c = i % NCHUNK, j = i / NCHUNK;
            uint32_t w0 = nan_flag, w1 = 0, w2 = 0, w3 = 0;
            if (c < 32) {
                uint4 lo = *reinterpret_cast<const uint4*>(bins + c * 8);
                uint4 hi = *reinterpret_cast<const uint4*>(bins + c * 8 + 4);
                w0 = lo.x | (lo.y << 16); w1 = lo.z | (lo.w << 16); w2 = hi.x | (hi.y << 16); w3 = hi.z | (hi.w << 16);
            }
            st_async_v4(cute::set_block_rank(recv_base + c * 16, j), w0, w1, w2, w3, cute::set_block_rank(bar_addr, j));
        }
    }

    template<uint32_t WORDS>
    static __device__ __forceinline__ uint32_t sum_bin(const uint32_t (&recv)[CLUSTER_SIZE][WORDS], uint32_t bin) {
        uint32_t all = 0;
        CUTE_UNROLL
        for (uint32_t j = 0; j < CLUSTER_SIZE; ++j) all += (recv[j][bin >> 1] >> (16 * (bin & 1))) & 0xFFFFu;
        return all;
    }

    static __device__ __forceinline__ void clear_local_hists(StreamPlan &smem) {
        for (uint32_t i = threadIdx.x; i < 256; i += NUM_THREADS) smem.xh_msb[i] = 0u;
        for (uint32_t i = threadIdx.x; i < BF16Base::NUM_RECONSTRUCT_BUCKETS + 4; i += NUM_THREADS) smem.xh_lsb[i] = 0u;
    }

    // Masks as in the main loop: two `set.*.bf16x2` + prmt + dp4a give a nibble per 4 elements, bit e for element e
    // (half e % 2 of word e / 2). GT_OP is `gt` or `gtu`; the invalid 0xFFFF is a NaN, so `gt` never takes it.
    template<uint32_t NU, bool UNORDERED>
    static __device__ __forceinline__ void pivot_masks(const uint32_t (&raw)[NU], uint32_t num_words, uint32_t p2, uint64_t &gt, uint64_t &eq) {
        static_assert(NU % 2 == 0 && NU <= 32);
        uint32_t gtw[2] = {0, 0}, eqw[2] = {0, 0};
        auto group = [&](uint32_t g) {
            if (2 * g >= num_words) return;
            if constexpr (UNORDERED) {
                asm("{\n.reg .b32 r0, r1, ind, nib;\n"
                    "set.gtu.s32.bf16x2 r0, %2, %4;\nset.gtu.s32.bf16x2 r1, %3, %4;\n"
                    "prmt.b32 ind, r0, r1, 0x7531;\ndp4a.s32.s32 nib, ind, 0xF8FCFEFF, 0;\nmad.lo.u32 %0, %0, 16, nib;\n"
                    "set.eq.s32.bf16x2 r0, %2, %4;\nset.eq.s32.bf16x2 r1, %3, %4;\n"
                    "prmt.b32 ind, r0, r1, 0x7531;\ndp4a.s32.s32 nib, ind, 0xF8FCFEFF, 0;\nmad.lo.u32 %1, %1, 16, nib;\n}\n"
                    : "+r"(gtw[g / 8]), "+r"(eqw[g / 8]) : "r"(raw[2 * g]), "r"(raw[2 * g + 1]), "r"(p2));
            } else {
                asm("{\n.reg .b32 r0, r1, ind, nib;\n"
                    "set.gt.s32.bf16x2 r0, %2, %4;\nset.gt.s32.bf16x2 r1, %3, %4;\n"
                    "prmt.b32 ind, r0, r1, 0x7531;\ndp4a.s32.s32 nib, ind, 0xF8FCFEFF, 0;\nmad.lo.u32 %0, %0, 16, nib;\n"
                    "set.eq.s32.bf16x2 r0, %2, %4;\nset.eq.s32.bf16x2 r1, %3, %4;\n"
                    "prmt.b32 ind, r0, r1, 0x7531;\ndp4a.s32.s32 nib, ind, 0xF8FCFEFF, 0;\nmad.lo.u32 %1, %1, 16, nib;\n}\n"
                    : "+r"(gtw[g / 8]), "+r"(eqw[g / 8]) : "r"(raw[2 * g]), "r"(raw[2 * g + 1]), "r"(p2));
            }
        };
        // Groups run from the top down; the upper 16 words are skipped as a whole when no thread has them
        if constexpr (NU > 16) {
            if (num_words > 16) {
                CUTE_UNROLL
                for (int32_t g = NU / 2 - 1; g >= 8; --g) group((uint32_t)g);
            }
        }
        CUTE_UNROLL
        for (int32_t g = (NU > 16 ? 8 : NU / 2) - 1; g >= 0; --g) group((uint32_t)g);
        gt = gtw[0] | ((uint64_t)gtw[1] << 32);
        eq = eqw[0] | ((uint64_t)eqw[1] << 32);
    }

    struct ClusterPivot { uint32_t key; uint32_t need_eq; bool nan; };

    // Four stage offsets as scalars: an array here gets its address taken and lands in local memory
    struct Quad {
        uint32_t v0, v1, v2, v3;
        __device__ __forceinline__ uint32_t at(uint32_t i) const {
            return i == 0 ? v0 : i == 1 ? v1 : i == 2 ? v2 : v3;
        }
    };
    static_assert(WINDOW_STAGES == 4 && STREAM_BATCH == 4);

    // The pivot of the top-k over every CTA's candidates. `raw` holds NU packed bf16x2 words, the first `num_words`
    // live; an invalid element holds 0xFFFF (distorted key 0) and is counted in `n_invalid`. `xh_msb` holds this
    // CTA's MSB histogram, invalid elements excluded. FIRST waits on the cluster barrier armed at kernel start
    // (every CTA's barriers are initialized); `on_synced` runs on thread 0 once the whole CTA is past its loads.
    template<bool FIRST, uint32_t NU, typename OnSyncedF>
    static __device__ __forceinline__ ClusterPivot cluster_pivot(
        StreamPlan &smem, RecvPair &recv, transac_bar_t (&bar)[2], uint32_t rank, uint32_t warp_idx, uint32_t lane_idx,
        uint32_t topk, bool nan_local, const uint32_t (&raw)[NU], uint32_t num_words, uint32_t n_invalid, OnSyncedF &&on_synced
    ) {
        const uint32_t tid = threadIdx.x;
        const nv_bfloat162 (&rawv)[NU] = reinterpret_cast<const nv_bfloat162 (&)[NU]>(raw);
        const uint32_t nan_cta = __syncthreads_or(nan_local) != 0;     // also publishes `xh_msb`
        if (tid == 0) on_synced();
        if constexpr (FIRST) ku::barrier_cluster_wait_acquire();
        push_hist(smem.xh_msb, recv.msb, bar[0], rank, nan_cta);
        wait_acquire_cluster(bar[0], 0);
        if (tid < 256) smem.xh_sum[tid] = sum_bin(recv.msb, tid);
        const uint32_t nan_any = tid < CLUSTER_SIZE ? recv.msb[tid][128] : 0u;
        if (__syncthreads_or(nan_any)) return {0u, 0u, true};
        if (warp_idx == 0) BF16Base::template find_pivot_in_histogram<false>(smem, smem.xh_sum, topk, lane_idx);
        __syncthreads();
        const uint32_t pivot_hi = smem.reconstruct_pivot_bucket;
        const uint32_t need_in_bucket = smem.reconstruct_num_should_select;

        BF16Base::template histogram_radix_lsb_for_pivot_msb<true, NU, (NU % 4 == 0 ? 4 : 1)>(smem.xh_lsb, pivot_hi, rawv, num_words);
        if (pivot_hi == 0 && n_invalid != 0) atomicAdd_block(&smem.xh_lsb[0], 0u - n_invalid);
        __syncthreads();
        push_hist(smem.xh_lsb, recv.lsb, bar[1], rank, 0u);
        wait_acquire_cluster(bar[1], 0);
        if (tid < 256) smem.xh_sum[tid] = sum_bin(recv.lsb, tid);   // warp 0 finished the MSB search before the barrier above
        __syncthreads();
        if (warp_idx == 0) BF16Base::template find_pivot_in_histogram<false>(smem, smem.xh_sum, need_in_bucket, lane_idx);
        __syncthreads();
        const ClusterPivot result{(pivot_hi << 8) | smem.reconstruct_pivot_bucket, smem.reconstruct_num_should_select, false};
        // Both pushes read their bins into registers, so the local histograms are free for the next pass
        clear_local_hists(smem);
        return result;
    }

    // Writes vocab index `idx` (and its value) to output slot `pos` of the row
    static __device__ __forceinline__ auto output_writer(const TopkSelectArgs &args, uint32_t batch_idx) {
        OutIdxT* result_indices = (OutIdxT*)args.output_index + (uint64_t)batch_idx * args.stride_output_index_batch;
        const int32_t idx_offset = args.output_idx_offset != nullptr ? __ldg(args.output_idx_offset + batch_idx) : 0;
        const int* page_table_row = Config::page_transform ? args.page_table + (uint64_t)batch_idx * args.stride_page_table_batch : nullptr;
        const uint32_t page_bits = args.page_bits;
        const ValueT* input_row = (const ValueT*)args.input + (uint64_t)batch_idx * args.stride_input_batch;
        ValueT* result_values = Config::return_value ? (ValueT*)args.output_value + (uint64_t)batch_idx * args.stride_output_value_batch : nullptr;
        return [=](uint32_t pos, uint32_t idx) {
            if constexpr (Config::page_transform) {
                result_indices[pos] = (OutIdxT)EpilogueT::page_transform_index(idx, true, 0, page_table_row, page_bits);
            } else {
                result_indices[pos] = (OutIdxT)idx + idx_offset;
            }
            if constexpr (Config::return_value) result_values[pos] = input_row[idx];
        };
    }

    // Store this CTA's share of the top-k given the cluster pivot: slots [0, topk - need_eq) take every candidate
    // above it, the rest the first `need_eq` ties to claim one. `index(j, b)` is the vocab index of element G * j + b.
    template<uint32_t NU, uint32_t G, typename IndexF>
    static __device__ __forceinline__ void cluster_emit(
        StreamPlan &smem, const TopkSelectArgs &args, uint32_t batch_idx, uint32_t lane_idx, ClusterPivot pivot,
        const uint32_t (&raw)[NU], uint32_t num_words, IndexF &&index
    ) {
        constexpr uint32_t NE = 2 * NU;
        static_assert(NE % G == 0 && G <= 32);
        const uint32_t pivot_raw = topk_select_common::un_distort((uint16_t)pivot.key);
        uint64_t gt, eq;
        pivot_masks<NU, false>(raw, num_words, pivot_raw | (pivot_raw << 16), gt, eq);

        const uint32_t my_gt = (uint32_t)__popcll(gt), my_eq = (uint32_t)__popcll(eq);
        const uint32_t packed = (my_gt << 16) | my_eq;     // a warp holds at most 32 * NE <= 2048 of each
        const uint32_t lane_off = warp_level_exclusive_prefix_sum(packed, lane_idx);
        const uint32_t warp_tot = __reduce_add_sync(0xFFFFFFFFu, packed);
        uint64_t base = 0;
        if (lane_idx == 0 && warp_tot != 0) {
            const uint32_t cursor = cute::set_block_rank(cute::cast_smem_ptr_to_uint(&smem.xctr), 0);
            const uint64_t inc = ((uint64_t)(warp_tot >> 16) << 32) | (warp_tot & 0xFFFFu);
            asm volatile("atom.shared::cluster.add.u64 %0, [%1], %2;" : "=l"(base) : "r"(cursor), "l"(inc) : "memory");
        }
        base = __shfl_sync(0xFFFFFFFFu, base, 0);
        // CTA0 must outlive every CTA's atomic on its `xctr`; having the result means this warp's is done
        ku::barrier_cluster_arrive_relaxed();
        uint32_t gt_pos = (uint32_t)(base >> 32) + (lane_off >> 16);
        const uint32_t eq_rank = (uint32_t)base + (lane_off & 0xFFFFu);
        const uint32_t eq_take = pivot.need_eq > eq_rank ? min(my_eq, pivot.need_eq - eq_rank) : 0u;
        while ((uint32_t)__popcll(eq) > eq_take) eq &= ~(1ull << (63 - __clzll(eq)));
        uint32_t eq_pos = args.topk - pivot.need_eq + eq_rank;

        auto store = output_writer(args, batch_idx);
        auto emit_unit = [&](uint32_t j) {
            uint32_t mg = (uint32_t)(gt >> (G * j)) & (uint32_t)((1ull << G) - 1);
            uint32_t me = (uint32_t)(eq >> (G * j)) & (uint32_t)((1ull << G) - 1);
            while (mg != 0) {
                uint32_t b = __ffs(mg) - 1;
                mg &= mg - 1;
                store(gt_pos++, index(j, b));
            }
            while (me != 0) {
                uint32_t b = __ffs(me) - 1;
                me &= me - 1;
                store(eq_pos++, index(j, b));
            }
        };
        constexpr uint32_t LOW_UNITS = (NE > 32 ? 32 : NE) / G;
        CUTE_UNROLL
        for (uint32_t j = 0; j < LOW_UNITS; ++j) emit_unit(j);
        if constexpr (NE > 32) {
            if (((gt | eq) >> 32) != 0) {
                CUTE_UNROLL
                for (uint32_t j = LOW_UNITS; j < NE / G; ++j) emit_unit(j);
            }
        }
        ku::barrier_cluster_wait_acquire();
    }

    // `sorted_index`: CTA r owns a contiguous slice past those of CTAs < r, so its selected elements start at the
    // count the lower CTAs select; ties go to the lowest CTAs first. Every CTA reads each CTA's counts off the
    // histograms of the last exchange (`recv`). Within the slice, an element's rank among the selected is a prefix
    // popcount over a bitmap of them, which the idle ring holds: quads [0, Q) of bits, then quads [Q, 2Q) of each
    // word's prefix within its warp's run.
    static constexpr uint32_t SORTED_MAX_SLICE = NUM_STAGES * STAGE * (uint32_t)sizeof(ValueT) / 8 * 32;

    template<uint32_t NU, uint32_t G, typename IndexF>
    static __device__ __forceinline__ void cluster_emit_sorted(
        StreamPlan &smem, const RecvPair &recv, const TopkSelectArgs &args, uint32_t batch_idx, uint32_t rank,
        uint32_t warp_idx, uint32_t lane_idx, ClusterPivot pivot, const uint32_t (&raw)[NU], uint32_t num_words,
        uint32_t start, uint32_t len, IndexF &&index
    ) {
        constexpr uint32_t NE = 2 * NU;
        static_assert(NE % G == 0 && NE <= 64);
        const uint32_t num_q = ku::ceil_div(len, 128u);
        uint4 *bits4 = reinterpret_cast<uint4*>(&smem.ring[0][0]);
        uint4 *prefix4 = bits4 + num_q;
        uint32_t *bits = reinterpret_cast<uint32_t*>(bits4), *prefix = reinterpret_cast<uint32_t*>(prefix4);
        // The ring is idle (every stage issued has been consumed)
        for (uint32_t i = threadIdx.x; i < num_q; i += NUM_THREADS) bits4[i] = make_uint4(0u, 0u, 0u, 0u);
        // Warp j counts CTA j's elements above the pivot key and at it; word w of a histogram holds bins 2w, 2w + 1
        if (warp_idx < CLUSTER_SIZE) {
            const uint32_t hi = pivot.key >> 8, lo = pivot.key & 0xFFu;
            uint32_t gt = 0, eq = 0;
            CUTE_UNROLL
            for (uint32_t w = lane_idx; w < 128; w += 32) {
                const uint32_t m = recv.msb[warp_idx][w], l = recv.lsb[warp_idx][w];
                CUTE_UNROLL
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t bin = 2 * w + h, mc = (m >> (16 * h)) & 0xFFFFu, lc = (l >> (16 * h)) & 0xFFFFu;
                    gt += (bin > hi ? mc : 0u) + (bin > lo ? lc : 0u);
                    eq += bin == lo ? lc : 0u;
                }
            }
            gt = __reduce_add_sync(0xFFFFFFFFu, gt);
            eq = __reduce_add_sync(0xFFFFFFFFu, eq);
            if (lane_idx == 0) {
                smem.rank_gt[warp_idx] = gt;
                smem.rank_eq[warp_idx] = eq;
            }
        }
        const uint32_t pivot_raw = topk_select_common::un_distort((uint16_t)pivot.key);
        uint64_t gt, eq;
        pivot_masks<NU, false>(raw, num_words, pivot_raw | (pivot_raw << 16), gt, eq);
        __syncthreads();
        uint32_t base = 0, eq_left = pivot.need_eq;
        CUTE_UNROLL
        for (uint32_t j = 0; j < CLUSTER_SIZE; ++j) {
            if (j < rank) {
                const uint32_t t = min(smem.rank_eq[j], eq_left);
                eq_left -= t;
                base += smem.rank_gt[j] + t;
            }
        }
        const uint32_t my_take = min(smem.rank_eq[rank], eq_left);

        // Any `my_take` of this CTA's ties will do: the bitmap orders whatever is kept
        const uint32_t my_eq = (uint32_t)__popcll(eq);
        const uint32_t warp_eq = __reduce_add_sync(0xFFFFFFFFu, my_eq);
        uint32_t eq_base = 0;
        if (lane_idx == 0 && warp_eq != 0) eq_base = atomicAdd_block(&smem.sel_eq, warp_eq);
        eq_base = __shfl_sync(0xFFFFFFFFu, eq_base, 0) + warp_level_exclusive_prefix_sum(my_eq, lane_idx);
        const uint32_t eq_take = my_take > eq_base ? min(my_eq, my_take - eq_base) : 0u;
        while ((uint32_t)__popcll(eq) > eq_take) eq &= ~(1ull << (63 - __clzll(eq)));
        const uint64_t keep = gt | eq;
        for (uint64_t m = keep; m != 0; m &= m - 1) {
            const uint32_t e = __ffsll(m) - 1;
            const uint32_t off = index(e / G, e % G) - start;
            atomicOr_block(&bits[off / 32], 1u << (off % 32));
        }
        __syncthreads();

        // One pass of exclusive prefixes: warp i takes a power-of-two run of quads, its lanes interleaved over it
        // (a thread-contiguous run would put a warp's accesses in one bank); the runs' bases are summed after
        constexpr uint32_t LOG_WARPS = 31 - __builtin_clz(BF16Base::NUM_WARPS);
        static_assert((1u << LOG_WARPS) == BF16Base::NUM_WARPS);
        const uint32_t log_run = max(5u, 32u - __clz(max((max(num_q, 1u) - 1) >> LOG_WARPS, 1u)));   // a CTA past the row end has len 0
        const uint32_t q_lo = min(warp_idx << log_run, num_q), q_hi = min(q_lo + (1u << log_run), num_q);
        uint32_t run = 0;
        for (uint32_t q0 = q_lo; q0 < q_hi; q0 += 32) {
            const uint32_t q = q0 + lane_idx;
            const uint4 v = q < q_hi ? bits4[q] : make_uint4(0u, 0u, 0u, 0u);
            const uint32_t c0 = __popc(v.x), c1 = __popc(v.y), c2 = __popc(v.z), c3 = __popc(v.w);
            const uint32_t local = c0 + c1 + c2 + c3;
            const uint32_t incl = warp_level_inclusive_prefix_sum(local, lane_idx);
            const uint32_t e0 = run + incl - local;
            if (q < q_hi) prefix4[q] = make_uint4(e0, e0 + c0, e0 + c0 + c1, e0 + c0 + c1 + c2);
            run += __shfl_sync(0xFFFFFFFFu, incl, 31);
        }
        if (lane_idx == 0) smem.warp_cnt[warp_idx] = run;
        __syncthreads();
        if (warp_idx == 0) {
            const uint32_t c = lane_idx < BF16Base::NUM_WARPS ? smem.warp_cnt[lane_idx] : 0u;
            const uint32_t e = warp_level_exclusive_prefix_sum(c, lane_idx);
            if (lane_idx < BF16Base::NUM_WARPS) smem.warp_pre[lane_idx] = e;
        }
        __syncthreads();

        auto store = output_writer(args, batch_idx);
        for (uint64_t m = keep; m != 0; m &= m - 1) {
            const uint32_t e = __ffsll(m) - 1;
            const uint32_t off = index(e / G, e % G) - start;
            const uint32_t w = off / 32;
            store(base + smem.warp_pre[w >> (log_run + 2)] + prefix[w] + __popc(bits[w] & ((1u << (off % 32)) - 1u)), start + off);
        }
    }

    // Load the window, ring stages 0..WINDOW_STAGES-1 (phase 0), into `vals` as WINDOW_ROWS units per thread,
    // building the MSB histogram as each row lands. Unit row r is the (r % 2)-th half of the slice stage at
    // element offset `win_off[r / 2]`; thread t holds unit t of it as words [4r, 4r + 4). Returns the number of live
    // unit rows of this thread (a row past the slice's end is never followed by a live one); words past them are
    // never read.
    static __device__ __forceinline__ uint32_t load_window(
        StreamPlan &smem, uint32_t len, uint32_t num_stages, Quad win_off,
        uint32_t (&vals)[WINDOW_ROWS * 4], uint32_t &n_invalid, bool &nan_local
    ) {
        const ValueT *buf = &smem.ring[0][0];
        uint32_t nan_acc = 0, num_units = 0;
        auto load_row = [&](auto row) {
            constexpr uint32_t r = decltype(row)::value;
            if (r / 2 >= num_stages) return;
            if (r % 2 == 0) smem.full[r / 2].wait(0);
            // Offset in the slice of this thread's unit, and in the window
            const uint32_t e0 = win_off.at(r / 2) + (r % 2) * UNIT_ROW + threadIdx.x * 8;
            if (e0 < len) {
                topk_select_common::ld_shared<4>(vals + r * 4, reinterpret_cast<const uint32_t*>(buf + r * UNIT_ROW + threadIdx.x * 8));
                const bool full = e0 + 8 <= len;
                CUTE_UNROLL
                for (uint32_t w = 0; w < 4; ++w) {
                    if (full) asm("min.NaN.bf16x2 %0, %0, %1;" : "+r"(nan_acc) : "r"(vals[r * 4 + w]));
                    vals[r * 4 + w] = canonical_zero_x2(vals[r * 4 + w]);
                }
                if (!full) {
                    // Only the last unit of the slice runs past its end
                    CUTE_UNROLL
                    for (uint32_t w = 0; w < 4; ++w) {
                        const bool ok0 = e0 + 2 * w < len, ok1 = e0 + 2 * w + 1 < len;
                        uint32_t v = vals[r * 4 + w];
                        nan_local |= (ok0 && (v & 0x7FFFu) > 0x7F80u) || (ok1 && ((v >> 16) & 0x7FFFu) > 0x7F80u);
                        n_invalid += (ok0 ? 0u : 1u) + (ok1 ? 0u : 1u);
                        v = ok0 ? v : (v | 0xFFFFu);
                        v = ok1 ? v : (v | 0xFFFF0000u);
                        vals[r * 4 + w] = v;
                    }
                }
                BF16Base::histogram_radix_msb(smem.xh_msb, reinterpret_cast<const nv_bfloat162 (&)[4]>(vals[r * 4]), 4u);
                num_units = r + 1;
            }
        };
        // The rows past the first half are live only for a long slice: one block-uniform branch skips them all
        constexpr uint32_t HALF = WINDOW_ROWS > 4 ? 4 : WINDOW_ROWS;
        [&]<uint32_t... R>(std::integer_sequence<uint32_t, R...>) {
            (load_row(std::integral_constant<uint32_t, R>{}), ...);
        }(std::make_integer_sequence<uint32_t, HALF>{});
        if constexpr (WINDOW_ROWS > HALF) {
            if (HALF / 2 < num_stages) {
                [&]<uint32_t... R>(std::integer_sequence<uint32_t, R...>) {
                    (load_row(std::integral_constant<uint32_t, HALF + R>{}), ...);
                }(std::make_integer_sequence<uint32_t, WINDOW_ROWS - HALF>{});
            }
        }
        uint32_t nan_flag;
        asm("set.nan.u32.bf16x2 %0, %1, %1;" : "=r"(nan_flag) : "r"(nan_acc));
        nan_local |= nan_flag != 0;
        if (n_invalid != 0) atomicAdd_block(&smem.xh_msb[0], 0u - n_invalid);
        return num_units;
    }

    // Slow path for a stage whose hits do not fit: replace the candidates [0, num_cand) and every element of the
    // stage with their top `topk` (a superset of this CTA's share), and return the new threshold (raw bf16 x2).
    // Kept out of line: it runs at most a few times per launch, and inlined copies only cost cold instruction fetch
    struct Reduced { uint32_t threshold; bool nan; };
    static __device__ __noinline__ Reduced reduce_candidates(
        StreamPlan &smem, uint32_t num_cand, const ValueT *stage_buf, uint32_t stage_slice_off, uint32_t len,
        uint32_t slice_start, uint32_t topk, uint32_t warp_idx, uint32_t lane_idx
    ) {
        bool nan_seen = false;
        constexpr uint32_t NC = CAND_CAP / NUM_THREADS;     // candidates per thread
        constexpr uint32_t NW = NC / 2 + 8;                 // their words, then the stage's two units
        static_assert(NC % 2 == 0);
        uint32_t vals[NW];
        uint32_t cidx[NC];
        uint32_t n_invalid = 0;
        CUTE_UNROLL
        for (uint32_t e = 0; e < NC; ++e) {
            const uint32_t c = threadIdx.x + e * NUM_THREADS;
            const uint64_t pair = c < num_cand ? smem.cand[c] : 0ull;
            cidx[e] = (uint32_t)pair;
            uint32_t v = (uint32_t)(pair >> 32) & 0xFFFFu;
            nan_seen |= c < num_cand && (v & 0x7FFFu) > 0x7F80u;
            v = v == 0x8000u ? 0u : v;
            v = c < num_cand ? v : 0xFFFFu;
            n_invalid += c < num_cand ? 0u : 1u;
            if (e % 2 == 0) vals[e / 2] = v; else vals[e / 2] |= v << 16;
        }
        CUTE_UNROLL
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t off = h * UNIT_ROW + threadIdx.x * 8;
            topk_select_common::ld_shared<4>(vals + NC / 2 + h * 4, reinterpret_cast<const uint32_t*>(stage_buf + off));
            CUTE_UNROLL
            for (uint32_t w = 0; w < 4; ++w) {
                uint32_t v = vals[NC / 2 + h * 4 + w];
                const bool ok0 = stage_slice_off + off + 2 * w < len, ok1 = stage_slice_off + off + 2 * w + 1 < len;
                nan_seen |= (ok0 && (v & 0x7FFFu) > 0x7F80u) || (ok1 && ((v >> 16) & 0x7FFFu) > 0x7F80u);
                v = canonical_zero_x2(v);
                v = ok0 ? v : (v | 0xFFFFu);
                v = ok1 ? v : (v | 0xFFFF0000u);
                n_invalid += (ok0 ? 0u : 1u) + (ok1 ? 0u : 1u);
                vals[NC / 2 + h * 4 + w] = v;
            }
        }
        const nv_bfloat162 (&rawv)[NW] = reinterpret_cast<const nv_bfloat162 (&)[NW]>(vals);
        BF16Base::template histogram_radix_msb<NW>(smem.xh_msb, rawv, NW);
        if (n_invalid != 0) atomicAdd_block(&smem.xh_msb[0], 0u - n_invalid);
        __syncthreads();
        if (warp_idx == 0) BF16Base::template find_pivot_in_histogram<false>(smem, smem.xh_msb, topk, lane_idx);
        __syncthreads();
        const uint32_t pivot_hi = smem.reconstruct_pivot_bucket;
        const uint32_t need_in_bucket = smem.reconstruct_num_should_select;
        BF16Base::template histogram_radix_lsb_for_pivot_msb<true, NW>(smem.xh_lsb, pivot_hi, rawv, NW);
        if (pivot_hi == 0 && n_invalid != 0) atomicAdd_block(&smem.xh_lsb[0], 0u - n_invalid);
        __syncthreads();
        if (warp_idx == 0) BF16Base::template find_pivot_in_histogram<false>(smem, smem.xh_lsb, need_in_bucket, lane_idx);
        __syncthreads();
        const uint32_t pivot_raw = topk_select_common::un_distort((uint16_t)((pivot_hi << 8) | smem.reconstruct_pivot_bucket));
        const uint32_t p2 = pivot_raw | (pivot_raw << 16);
        auto census = BF16Base::template get_census_counts<NW>(rawv, NW, p2);
        // Ends with a barrier, so every load of `cand` above is done before the stores below
        auto eqgt = BF16Base::compute_equal_quota_and_prefix(census.cnt_gt, census.cnt_eq, topk, warp_idx, lane_idx, smem.warp_cnt);
        uint32_t pos = eqgt.start_pos_in_collector, quota = eqgt.eq_quota;
        auto keep = [&](uint32_t w, uint32_t h, uint32_t idx) {
            const uint32_t v = (vals[w] >> (16 * h)) & 0xFFFFu;
            const uint32_t v2 = v | (v << 16);
            uint32_t g, q;
            asm("set.gt.u32.bf16x2 %0, %1, %2;" : "=r"(g) : "r"(v2), "r"(p2));
            asm("set.eq.u32.bf16x2 %0, %1, %2;" : "=r"(q) : "r"(v2), "r"(p2));
            const bool take = g != 0 || (q != 0 && quota > 0);
            if (take) {
                if (g == 0) --quota;
                smem.cand[pos++] = ((uint64_t)v << 32) | idx;
            }
        };
        CUTE_UNROLL
        for (uint32_t e = 0; e < NC; ++e) keep(e / 2, e % 2, cidx[e]);
        CUTE_UNROLL
        for (uint32_t h = 0; h < 2; ++h) {
            CUTE_UNROLL
            for (uint32_t b = 0; b < 8; ++b)
                keep(NC / 2 + h * 4 + b / 2, b % 2, slice_start + stage_slice_off + h * UNIT_ROW + threadIdx.x * 8 + b);
        }
        clear_local_hists(smem);
        if (threadIdx.x == 0) smem.cand_cnt = topk;
        __syncthreads();
        return {p2, nan_seen};
    }

    // A NaN in the cluster's windows: the stages [WINDOW_STAGES, num_issued) issued during the exchange must land
    // before the CTA exits. Out of line: inlined, this cold path slows the stream loop.
    static __device__ __noinline__ void window_nan_exit(
        StreamPlan &smem, const TopkSelectArgs &args, uint32_t batch_idx, uint32_t rank, uint32_t num_issued
    ) {
        for (uint32_t g = WINDOW_STAGES; g < num_issued; ++g) smem.full[g].wait(0);
        if (rank == 0) BF16Base::take_action_when_have_nan(args, batch_idx);
    }

    static __device__ __forceinline__ void select_devfunc(
        StreamPlan &smem, const TopkSelectArgs &args, uint32_t batch_idx, uint32_t end_vocab_idx,
        uint32_t rank, uint32_t warp_idx, uint32_t lane_idx
    ) {
        const uint32_t slice = ku::ceil_div(ku::ceil_div(end_vocab_idx, (uint32_t)CLUSTER_SIZE), 64u) * 64u;
        const uint32_t start = min(rank * slice, end_vocab_idx);
        const uint32_t len = min(start + slice, end_vocab_idx) - start;
        const uint32_t num_stages = ku::ceil_div(len, (uint32_t)STAGE);
        const ValueT *src = (const ValueT*)args.input + (uint64_t)batch_idx * args.stride_input_batch + start;
        // Visit order of the slice's stages. A slice longer than the window puts its
        // last stage (the row's tail on the last CTA, which tends to hold heavy scores, and the slice's only partial
        // stage) in the window, and visits the full stages in a pseudo-random order like the one-CTA-per-row kernel:
        // i -> (i * mul + base) % P, a bijection since mul is a residue of the prime 2^31 - 1, which exceeds P.
        // Offsets are computed once and then only advanced: inlined divisions at every use cost more in code size
        // (cold instruction fetch on every launch) than in instructions.
        const bool permuted = num_stages > WINDOW_STAGES;
        const uint32_t num_full = num_stages - 1;
        const uint32_t perm_mul = permuted ? 0x7FFFFFFFu % num_full : 1u;
        auto perm_next = [&](uint32_t cur) {
            cur += perm_mul;
            return cur >= num_full ? cur - num_full : cur;
        };
        // Window stages: the first WINDOW_STAGES - 1 visits of the permutation, then the last stage
        uint32_t perm_head = permuted ? (rank * 0x9E3779B1u + batch_idx * 0x85EBCA77u) % num_full : 0u;
        Quad win_off;
        win_off.v0 = (permuted ? perm_head : 0u) * STAGE;
        perm_head = perm_next(perm_head);
        win_off.v1 = (permuted ? perm_head : 1u) * STAGE;
        perm_head = perm_next(perm_head);
        win_off.v2 = (permuted ? perm_head : 2u) * STAGE;
        perm_head = perm_next(perm_head);
        win_off.v3 = (permuted ? num_full : 3u) * STAGE;
        // `perm_head` is now the permuted index of the first streamed stage
        auto issue_stage_at = [&](uint32_t g, uint32_t off) {
            // Rounded up to 16 bytes; the overrun stays inside the row, whose end is padded to 128 bytes
            const uint32_t bytes = (min((uint32_t)STAGE, len - off) * (uint32_t)sizeof(ValueT) + 15) & ~15u;
            transac_bar_t &bar = smem.full[g % NUM_STAGES];
            smem.stage_off[g % NUM_STAGES] = off;
            bar.arrive_and_expect_tx(bytes);
            bulk_g2s(smem.ring[g % NUM_STAGES], src + off, bytes, bar);
        };
        // Thread 0 issues the stages after the window in visit order from this pointer
        uint32_t perm_issue = perm_head;
        auto issue_next = [&](uint32_t g) {
            if (g < num_stages) issue_stage_at(g, perm_issue * STAGE);
            perm_issue = perm_next(perm_issue);
        };

        clear_local_hists(smem);
        if (warp_idx == 0 && cute::elect_one_sync()) {
            CUTE_UNROLL
            for (uint32_t s = 0; s < NUM_STAGES; ++s) smem.full[s].init(1);
            CUTE_UNROLL
            for (uint32_t i = 0; i < 2; ++i) {
                smem.xbar[i].init(1);
                smem.ybar[i].init(1);
            }
            smem.xbar[0].arrive_and_expect_tx(CLUSTER_SIZE * XCHG_MSB_WORDS * 4);
            smem.xbar[1].arrive_and_expect_tx(CLUSTER_SIZE * XCHG_LSB_WORDS * 4);
            smem.ybar[0].arrive_and_expect_tx(CLUSTER_SIZE * XCHG_MSB_WORDS * 4);
            smem.ybar[1].arrive_and_expect_tx(CLUSTER_SIZE * XCHG_LSB_WORDS * 4);
            smem.xctr = 0;
            smem.cand_cnt = 0;
            smem.eq_cnt = 0;
            smem.sel_cnt = 0;
            smem.sel_eq = 0;
            cutlass::arch::fence_barrier_init();
            // The window first: the stream behind it is useless until the window's pivot comes back, so it is
            // issued once the window has landed and arrives during the exchange instead of competing with it
            CUTE_UNROLL
            for (uint32_t g = 0; g < WINDOW_STAGES; ++g) {
                if (g < num_stages) issue_stage_at(g, win_off.at(g));
            }
        }
        __syncthreads();
        ku::barrier_cluster_arrive_relaxed();   // paired with the wait of the first exchange

        uint32_t vals[WINDOW_ROWS * 4];
        uint32_t n_invalid = 0;
        bool nan_local = false;
        const uint32_t num_units = load_window(smem, len, num_stages, win_off, vals, n_invalid, nan_local);
        // [vLLM] Programmatic dependent launch: the dependents (the attention that reads the indices) may start their
        // prologue once every CTA has its window; they wait for this grid's completion before reading the output
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        // `j` is a compile-time unit row in `cluster_emit`; a select chain keeps `win_off` in registers otherwise
        auto window_index = [&](uint32_t j, uint32_t b) {
            return start + win_off.at(j / 2) + (j % 2) * UNIT_ROW + threadIdx.x * 8 + b;
        };

        // Decided on `slice`, which every CTA of the cluster shares, so they all take the same exchanges
        if (slice <= RAW_WINDOW) {
            // The whole slice is in `vals`: it is the candidate set
            auto pivot = cluster_pivot<true>(smem, smem.window_recv, smem.xbar, rank, warp_idx, lane_idx, args.topk,
                                             nan_local, vals, num_units * 4, n_invalid, [] {});
            if (pivot.nan) {
                if (rank == 0) BF16Base::take_action_when_have_nan(args, batch_idx);
                return;
            }
            if constexpr (Config::sorted_index) {
                cluster_emit_sorted<WINDOW_ROWS * 4, 8>(smem, smem.window_recv, args, batch_idx, rank, warp_idx, lane_idx,
                                                        pivot, vals, num_units * 4, start, len, window_index);
            } else {
                cluster_emit<WINDOW_ROWS * 4, 8>(smem, args, batch_idx, lane_idx, pivot, vals, num_units * 4, window_index);
            }
            return;
        }

        // A long slice: the window's pivot over the whole cluster bounds every later candidate from below. The rest
        // of the ring fills while it is exchanged; the window stages stay until their candidates are copied out.
        const auto window_pivot = cluster_pivot<true>(smem, smem.window_recv, smem.xbar, rank, warp_idx, lane_idx, args.topk,
                                                      nan_local, vals, num_units * 4, n_invalid, [&] {
            CUTE_UNROLL
            for (uint32_t g = WINDOW_STAGES; g < NUM_STAGES; ++g) issue_next(g);
        });
        if (window_pivot.nan) {
            window_nan_exit(smem, args, batch_idx, rank, min((uint32_t)NUM_STAGES, num_stages));
            return;
        }
        const uint32_t pivot_raw = topk_select_common::un_distort((uint16_t)window_pivot.key);
        uint32_t threshold = pivot_raw | (pivot_raw << 16);
        {
            // Keep the window's elements above the pivot and up to `need_eq` ties: at least topk of the cluster's
            // window elements are at least the pivot, so every element of the top-k is kept or still to stream in
            uint64_t gt, eq;
            pivot_masks<WINDOW_ROWS * 4, false>(vals, num_units * 4, threshold, gt, eq);
            const uint32_t my_eq = (uint32_t)__popcll(eq);
            const uint32_t warp_eq = __reduce_add_sync(0xFFFFFFFFu, my_eq);
            uint32_t eq_base = 0;
            if (lane_idx == 0 && warp_eq != 0) eq_base = atomicAdd_block(&smem.eq_cnt, warp_eq);
            eq_base = __shfl_sync(0xFFFFFFFFu, eq_base, 0) + warp_level_exclusive_prefix_sum(my_eq, lane_idx);
            const uint32_t eq_take = window_pivot.need_eq > eq_base ? min(my_eq, window_pivot.need_eq - eq_base) : 0u;
            while ((uint32_t)__popcll(eq) > eq_take) eq &= ~(1ull << (63 - __clzll(eq)));
            const uint64_t keep = gt | eq;
            const uint32_t my_keep = (uint32_t)__popcll(keep);
            const uint32_t warp_keep = __reduce_add_sync(0xFFFFFFFFu, my_keep);
            uint32_t pos = 0;
            if (lane_idx == 0 && warp_keep != 0) pos = atomicAdd_block(&smem.cand_cnt, warp_keep);
            pos = __shfl_sync(0xFFFFFFFFu, pos, 0) + warp_level_exclusive_prefix_sum(my_keep, lane_idx);
            // The window is still in ring stages 0..WINDOW_STAGES-1, contiguous
            const ValueT *window_buf = &smem.ring[0][0];
            uint64_t m = keep;
            while (m != 0) {
                const uint32_t e = __ffsll(m) - 1;
                m &= m - 1;
                const uint32_t off = (e / 8) * UNIT_ROW + threadIdx.x * 8 + e % 8;
                uint32_t v;
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(v) : "r"(cute::cast_smem_ptr_to_uint(window_buf + off)));
                smem.cand[pos++] = ((uint64_t)v << 32) | (start + smem.stage_off[e / 16] + off % STAGE);
            }
        }
        bool nan_seen = false;
        __syncthreads();
        if (threadIdx.x == 0) {
            CUTE_UNROLL
            for (uint32_t g = NUM_STAGES; g < NUM_STAGES + WINDOW_STAGES; ++g) issue_next(g);
        }

        // Stream the rest: thread t takes unit t of both unit rows of a stage, 16 elements as 8 words. Stages go in
        // batches of STREAM_BATCH under one atomic per warp and one barrier: per stage, the dependent chain of
        // atomic, prefix and barrier costs more than the 16 elements' work.
        // Every streamed stage is full: the slice's partial stage is in the window
        auto load_stage = [&](uint32_t slot, uint32_t phase, uint32_t (&w)[8]) {
            const ValueT *stage_buf = smem.ring[slot];
            smem.full[slot].wait(phase);
            topk_select_common::ld_shared<4>(w, reinterpret_cast<const uint32_t*>(stage_buf + threadIdx.x * 8));
            topk_select_common::ld_shared<4>(w + 4, reinterpret_cast<const uint32_t*>(stage_buf + UNIT_ROW + threadIdx.x * 8));
        };
        // Bit e: element e of the thread's 16 (unit row e / 8, offset e % 8) beats the threshold. NaN counts as a
        // hit (`gtu`), so every NaN reaches the candidates and the final exchange reports it.
        auto hit_mask = [&](const uint32_t (&w)[8]) {
            uint32_t m = 0;
            CUTE_UNROLL
            for (int32_t q = 3; q >= 0; --q) {
                asm("{\n.reg .b32 r0, r1, ind, nib;\n"
                    "set.gtu.s32.bf16x2 r0, %1, %3;\nset.gtu.s32.bf16x2 r1, %2, %3;\n"
                    "prmt.b32 ind, r0, r1, 0x7531;\ndp4a.s32.s32 nib, ind, 0xF8FCFEFF, 0;\nmad.lo.u32 %0, %0, 16, nib;\n}\n"
                    : "+r"(m) : "r"(w[2 * q]), "r"(w[2 * q + 1]), "r"(threshold));
            }
            return m;
        };
        auto append_stage = [&](uint32_t stage_off, uint32_t slot, uint32_t hit, uint32_t &pos) {
            const ValueT *stage_buf = smem.ring[slot];
            while (hit != 0) {
                const uint32_t e = __ffs(hit) - 1;
                hit &= hit - 1;
                const uint32_t off = (e / 8) * UNIT_ROW + threadIdx.x * 8 + e % 8;
                uint32_t v;
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(v) : "r"(cute::cast_smem_ptr_to_uint(stage_buf + off)));
                smem.cand[pos++] = ((uint64_t)v << 32) | (start + stage_off + off);
            }
        };
        // Claims room for `my_hits` per thread with one atomic per warp; false when it does not fit
        auto claim = [&](uint32_t my_hits, uint32_t &pos) {
            if (!__any_sync(0xFFFFFFFFu, my_hits != 0)) return true;
            const uint32_t warp_hits = __reduce_add_sync(0xFFFFFFFFu, my_hits);
            pos = 0;
            if (lane_idx == 0) pos = atomicAdd_block(&smem.cand_cnt, warp_hits);
            pos = __shfl_sync(0xFFFFFFFFu, pos, 0) + warp_level_exclusive_prefix_sum(my_hits, lane_idx);
            return pos + my_hits <= CAND_CAP;
        };
        // After a failed claim the whole CTA is past the barrier and no atomic is in flight: rewind the count by
        // everything claimed since
        auto rewind = [&](uint32_t my_hits) {
            const uint32_t warp_hits = __reduce_add_sync(0xFFFFFFFFu, my_hits);
            if (lane_idx == 0) smem.warp_cnt[warp_idx] = warp_hits;
            __syncthreads();
            uint32_t claimed = 0;
            CUTE_UNROLL
            for (uint32_t i = 0; i < BF16Base::NUM_WARPS; ++i) claimed += smem.warp_cnt[i];
            const uint32_t before = smem.cand_cnt - claimed;
            __syncthreads();
            if (threadIdx.x == 0) smem.cand_cnt = before;
            __syncthreads();
            return before;
        };
        static_assert(STREAM_BATCH * 16 <= 64);
        for (uint32_t g0 = WINDOW_STAGES; g0 < num_stages; g0 += STREAM_BATCH) {
            // Slot and phase of each stage of the batch, without a division per stage
            uint32_t slot[STREAM_BATCH], phase[STREAM_BATCH];
            slot[0] = g0 % NUM_STAGES;
            phase[0] = (g0 / NUM_STAGES) & 1;
            CUTE_UNROLL
            for (uint32_t k = 1; k < STREAM_BATCH; ++k) {
                const bool wrap = slot[k - 1] + 1 == NUM_STAGES;
                slot[k] = wrap ? 0u : slot[k - 1] + 1;
                phase[k] = wrap ? phase[k - 1] ^ 1u : phase[k - 1];
            }
            uint32_t w[STREAM_BATCH][8];
            uint32_t hit[STREAM_BATCH];
            uint64_t hits = 0;
            CUTE_UNROLL
            for (uint32_t k = 0; k < STREAM_BATCH; ++k) {
                hit[k] = 0;
                if (g0 + k < num_stages) {
                    load_stage(slot[k], phase[k], w[k]);
                    hit[k] = hit_mask(w[k]);
                    hits |= (uint64_t)hit[k] << (16 * k);
                }
            }
            const uint32_t my_hits = (uint32_t)__popcll(hits);
            uint32_t pos;
            const bool fits = claim(my_hits, pos);
            if (fits) {
                // One loop over the whole batch's hits
                while (hits != 0) {
                    const uint32_t e = __ffsll(hits) - 1;
                    hits &= hits - 1;
                    const uint32_t k = e / 16, b = e % 16;
                    uint32_t sl = slot[0] + k;
                    sl = sl >= NUM_STAGES ? sl - NUM_STAGES : sl;
                    const uint32_t off = (b / 8) * UNIT_ROW + threadIdx.x * 8 + b % 8;
                    uint32_t v;
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(v) : "r"(cute::cast_smem_ptr_to_uint(smem.ring[sl] + off)));
                    smem.cand[pos++] = ((uint64_t)v << 32) | (start + smem.stage_off[sl] + off);
                }
            }
            if (__syncthreads_or(!fits)) {
                // Rare: redo the batch stage by stage, reducing the candidates whenever a stage does not fit. One copy
                // of this code (no unrolling), reloading each stage from the ring, which is not refilled yet.
                rewind(my_hits);
                #pragma unroll 1
                for (uint32_t g = g0; g < min(g0 + STREAM_BATCH, num_stages); ++g) {
                    const uint32_t sl = g % NUM_STAGES;
                    const uint32_t so = smem.stage_off[sl];
                    uint32_t ws[8];
                    load_stage(sl, (g / NUM_STAGES) & 1, ws);
                    const uint32_t h = hit_mask(ws);
                    const bool ok = claim(__popc(h), pos);
                    if (ok) append_stage(so, sl, h, pos);
                    if (__syncthreads_or(!ok)) {
                        const uint32_t before = rewind(__popc(h));
                        const Reduced red = reduce_candidates(smem, before, smem.ring[sl], so, len, start,
                                                              args.topk, warp_idx, lane_idx);
                        threshold = red.threshold;
                        nan_seen |= red.nan;
                    }
                }
            }
            if (threadIdx.x == 0) {
                CUTE_UNROLL
                for (uint32_t k = 0; k < STREAM_BATCH; ++k) issue_next(g0 + k + NUM_STAGES);
            }
        }
        // Every atomic on `cand_cnt` happened before the last barrier above (or the one after the window's appends)
        __syncthreads();
        const uint32_t num_cand = smem.cand_cnt;

        // The final exchange over the candidates
        constexpr uint32_t NC = CAND_CAP / NUM_THREADS;
        uint32_t cvals[NC / 2];
        CUTE_UNROLL
        for (uint32_t e = 0; e < NC; ++e) {
            const uint32_t c = threadIdx.x + e * NUM_THREADS;
            uint32_t v = c < num_cand ? (uint32_t)(smem.cand[c] >> 32) & 0xFFFFu : 0xFFFFu;
            nan_seen |= c < num_cand && (v & 0x7FFFu) > 0x7F80u;
            v = v == 0x8000u ? 0u : v;     // -0 to +0, see `canonical_zero_x2`
            if (e % 2 == 0) cvals[e / 2] = v; else cvals[e / 2] |= v << 16;
        }
        const uint32_t my_cand = num_cand > threadIdx.x ? ku::ceil_div(num_cand - threadIdx.x, (uint32_t)NUM_THREADS) : 0u;
        const uint32_t num_words = (my_cand + 1) / 2;
        const uint32_t c_invalid = my_cand % 2;     // the odd tail of the last live word
        BF16Base::template histogram_radix_msb<NC / 2>(smem.xh_msb, reinterpret_cast<const nv_bfloat162 (&)[NC / 2]>(cvals), num_words);
        if (c_invalid != 0) atomicAdd_block(&smem.xh_msb[0], 0u - c_invalid);
        auto pivot = cluster_pivot<false>(smem, smem.final_recv, smem.ybar, rank, warp_idx, lane_idx, args.topk,
                                          nan_seen, cvals, num_words, c_invalid, [] {});
        if (pivot.nan) {
            if (rank == 0) BF16Base::take_action_when_have_nan(args, batch_idx);
            return;
        }
        auto cand_index = [&](uint32_t j, uint32_t) { return (uint32_t)smem.cand[threadIdx.x + j * NUM_THREADS]; };
        if constexpr (Config::sorted_index) {
            cluster_emit_sorted<NC / 2, 1>(smem, smem.final_recv, args, batch_idx, rank, warp_idx, lane_idx,
                                           pivot, cvals, num_words, start, len, cand_index);
        } else {
            cluster_emit<NC / 2, 1>(smem, args, batch_idx, lane_idx, pivot, cvals, num_words, cand_index);
        }
    }

    static __device__ __forceinline__ void topk_select_kernel_devfunc(const TopkSelectArgs &args, const TmaParams &tma_params) {
        uint32_t rank_in_cluster = blockIdx.x;
        uint32_t batch_idx = blockIdx.y;
        // [vLLM] Programmatic dependent launch: `end` and the input come from the stream's previous kernels, and the
        // output may still be read by them
        asm volatile("griddepcontrol.wait;" ::: "memory");
        uint32_t end_vocab_idx = args.end_ptr == nullptr ? args.vocab_size : __ldg(args.end_ptr + batch_idx);

        extern __shared__ CUTE_ALIGNAS(1024) char wksp_buf[];
        SharedMemoryPlanBase &smem = *reinterpret_cast<SharedMemoryPlanBase*>(wksp_buf);

        uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
        uint32_t lane_idx = threadIdx.x % 32;

        if (end_vocab_idx <= args.topk) {
            if (rank_in_cluster != 0) {
                return;
            }
            EpilogueT::template topk_select_epilogue<true>(
                (ValueT*)smem.surviving_topk_pairs[0],
                (uint32_t*)(smem.surviving_topk_pairs[1]),
                args,
                batch_idx, end_vocab_idx, warp_idx,
                *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(smem.incoming_topk_pairs)
            );
            return;
        }

        select_devfunc(*reinterpret_cast<StreamPlan*>(wksp_buf), args, batch_idx, end_vocab_idx, rank_in_cluster, warp_idx, lane_idx);
    }
};

template<typename Kernel>
__launch_bounds__(Kernel::NUM_THREADS, Kernel::TARGET_OCCUPANCY, Kernel::CLUSTER_SIZE)
__global__ void topk_kernel(__grid_constant__ const TopkSelectArgs args, __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::topk_select_kernel_devfunc(args, tma_params);
}

template<typename Config>
void run_topk_select_kernel(const TopkSelectArgs &args) {
    KU_ASSERT(args.sorted_value == Config::sorted_value, "Dispatch failure");
    KU_ASSERT(args.sorted_index == Config::sorted_index, "Dispatch failure");
    KU_ASSERT(args.return_value == Config::return_value, "Dispatch failure");
    static_assert(cute::is_same_v<typename Config::ValueT, nv_bfloat16>);
    KU_ASSERT(args.vocab_size < MAX_VOCAB_SIZE, "`vocab_size` is too big");

    using Kernel = TopkSelectKernelBF16Cluster<Config>;
    KU_ASSERT(args.topk <= Kernel::MAX_TOPK, "topk is too large. Maximum allowed: %d\n", Kernel::MAX_TOPK);
    // `vocab_size` bounds every row's `end`
    KU_ASSERT(!Config::sorted_index || ku::ceil_div(args.vocab_size, (uint32_t)Kernel::CLUSTER_SIZE) <= Kernel::SORTED_MAX_SLICE - 63,
              "`sorted_index` slice too long for the cluster kernel");
    static_assert(INPUT_STRIDE_ALIGNMENT_REQUIREMENT % 16 == 0);

    auto kernel = topk_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(typename Kernel::PlanT);
    KU_ASSERT(smem_size * Kernel::TARGET_OCCUPANCY <= args.shared_memory_size_per_sm);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    if constexpr (Kernel::CLUSTER_SIZE > 8) {
        KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    }

    KU_ASSERT(args.stride_input_batch % 8 == 0, "stride_input_batch must be 16B-aligned");
    typename Kernel::TmaParams tma_params = {Kernel::make_topk_tensor_map(args)};

    ku::launch_kernel(ku::KernelLaunchConfig {
        dim3(Kernel::CLUSTER_SIZE, args.batch_size),
        dim3(Kernel::NUM_THREADS),
        smem_size,
        args.stream,
        dim3(Kernel::CLUSTER_SIZE, 1, 1),
        true    // [vLLM] programmatic dependent launch, see `topk_select_kernel_devfunc` and `select_devfunc`
    }, kernel, args, tma_params);
}

}   // topk_select_bf16_cluster
