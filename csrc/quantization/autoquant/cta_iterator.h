/*
 * Adapted from https://github.com/InternLM/lmdeploy
 * Copyright (c) OpenMMLab. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#pragma once

#include <cstdint>
#include "common.h"

namespace vllm {
namespace autoquant {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template<typename T>
__inline__ __device__ void cp_async_cg_A(uint32_t smem_int_ptr, const T* __restrict__ src, bool mask)
{
#if VLLM_ARCH_SM80
    constexpr int cp_size = sizeof(T);
    static_assert(cp_size == 16, "cp.async.cg requreis cp_size == 16");
    // clang-format off
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global" L2_CACHEHINT(256) " [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)mask),
                 "r"(smem_int_ptr),
                 "l"(src),
                 "n"(cp_size));
    // clang-format on
#else
    assert(VLLM_ARCH_SM80);
#endif
}

template<typename T>
__inline__ __device__ void cp_async_cg_B(uint32_t smem_int_ptr, const T* __restrict__ src, bool mask)
{
#if VLLM_ARCH_SM80
    constexpr int cp_size = sizeof(T);
    static_assert(cp_size == 16, "cp.async.cg requreis cp_size == 16");
    // clang-format off
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)mask),
                 "r"(smem_int_ptr),
                 "l"(src),
                 "n"(cp_size));
    // clang-format on
#else
    assert(VLLM_ARCH_SM80);
#endif
}

template<typename T>
__inline__ __device__ void cp_async_ca(uint32_t smem_int_ptr, const T* __restrict__ src, bool mask)
{
#if VLLM_ARCH_SM80
    constexpr int cp_size = sizeof(T);
    // clang-format off
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.ca.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)mask),
                 "r"(smem_int_ptr),
                 "l"(src),
                 "n"(cp_size));
    // clang-format on
#else
    assert(VLLM_ARCH_SM80);
#endif
}

template<int WARPS, int CTA_M, int CTA_N, int CTA_K, int STAGES, int SLICES>
struct IteratorA {
    static constexpr int SLICE_K = CTA_K / SLICES;

    using AccessType                 = uint4;
    static constexpr int kAccessSize = sizeof(AccessType);

    static_assert(CTA_M % 32 == 0 && CTA_K % 32 == 0, "A is pre-formatted as 32x32 tiles");

    // A is [K/32, M/32, WARP_SIZE] uint4

    static constexpr int kShapeM = CTA_M;
    static constexpr int kShapeK = SLICE_K / 32;

    // thread access shape
    static constexpr int kAccessM = 1;
    static constexpr int kAccessK = 1;

    // warp thread arrangement
    static constexpr int kWarpThreadC = 32;
    static constexpr int kWarpThreadS = 1;

    // warp shape per access
    static constexpr int kWarpAccessM = kWarpThreadC * kAccessM;  // 32
    static constexpr int kWarpAccessK = kWarpThreadS * kAccessK;  // 1

    // warp access iterations
    static constexpr int kWarpIterM = kShapeM / kWarpAccessM;
    static constexpr int kWarpIterK = kShapeK / kWarpAccessK;

    // warp arrangement
    static constexpr int kWarpM = kWarpIterM >= WARPS ? WARPS : kWarpIterM;
    static constexpr int kWarpK = WARPS > kWarpIterM ? (WARPS / kWarpM) : 1;

    // iterations
    static constexpr int kIterM = kWarpIterM / kWarpM;
    static constexpr int kIterK = kWarpIterK / kWarpK;

    static constexpr int kIterCount = kIterM * kIterK;
    static_assert(kIterCount > 0);

    // warp footprint
    static constexpr int kWarpFootprintM = kWarpAccessM * kIterM;
    static constexpr int kWarpFootprintK = kWarpAccessK * kIterK;

    static constexpr int kSizePerStage = kShapeK * kShapeM;
    static constexpr int kSmemByteSize = kAccessSize * STAGES * kSizePerStage;

    const uint* src_;
    AccessType* smem_;
    uint32_t    smem_int_ptr_;

    const int m_;
    const int k_;

    const int warp_id_;
    const int lane_id_;

    int src_offset_;
    int dst_offset_;

    int src_step_m_;
    int src_step_k_;
    int src_step_s_;

    int dst_step_m_;
    int dst_step_k_;
    int dst_step_s_;

    int iter_m_{0};

    IteratorA() = default;

    __device__ IteratorA(const uint* src, void* smem, int m, int k, int cta_m, int cta_k, int warp_id, int lane_id):
        src_(src),
        smem_((AccessType*)smem),
        smem_int_ptr_(cast_smem_ptr_to_uint(smem)),
        m_(m),
        k_(k),
        warp_id_(warp_id),
        lane_id_(lane_id)
    {
        const int warp_offset_m = warp_id_ % kWarpM;
        const int warp_offset_k = warp_id_ / kWarpM;

        const int warp_thread_offset_m = lane_id_ % kWarpThreadC;
        const int warp_thread_offset_k = lane_id_ / kWarpThreadC;

        const int cta_thread_offset_m = kWarpFootprintM * warp_offset_m + warp_thread_offset_m * kAccessM;
        const int cta_thread_offset_k = kWarpFootprintK * warp_offset_k + warp_thread_offset_k * kAccessK;

        const int src_offset_m = cta_thread_offset_m + cta_m;
        const int src_offset_k = cta_thread_offset_k + cta_k / 32;

        src_offset_ = src_offset_k * m_ + src_offset_m;
        src_step_m_ = kWarpAccessM;
        src_step_k_ = kWarpAccessK * m_ - kIterM * kWarpAccessM;
        src_step_s_ = CTA_K / 32 * m_ - kIterK * kWarpAccessK * m_;

        const int dst_offset_m = cta_thread_offset_m;
        const int dst_offset_k = cta_thread_offset_k;

        dst_offset_ = dst_offset_k * kShapeM + dst_offset_m;
        dst_step_m_ = kWarpAccessM;
        dst_step_k_ = kWarpAccessK * kShapeM - kIterM * kWarpAccessM;
        dst_step_s_ = SLICE_K / 32 * kShapeM - kIterK * kWarpAccessK * kShapeM;

        dst_offset_ *= kAccessSize;
        dst_step_m_ *= kAccessSize;
        dst_step_k_ *= kAccessSize;
        dst_step_s_ *= kAccessSize;
    }

    __device__ void prefetch_stage(bool mask)
    {
        PRAGMA_UNROLL
        for (int i = 0; i < kIterCount; ++i) {
            prefetch(mask);
            ++(*this);
        }
        next_stage();
    }

    __device__ void prefetch_batch(int batch_idx, int batch_size, bool mask)
    {
        PRAGMA_UNROLL
        for (int i = 0; i < batch_size; ++i) {
            if (batch_idx * batch_size + i < kIterCount) {
                prefetch(mask);
                ++(*this);
            }
        }
    }

    __device__ IteratorA& operator++()
    {
        src_offset_ += src_step_m_;
        dst_offset_ += dst_step_m_;
        ++iter_m_;
        if (iter_m_ < kIterM) {
            return *this;
        }
        iter_m_ = 0;
        src_offset_ += src_step_k_;
        dst_offset_ += dst_step_k_;

        return *this;
    }

    __device__ void next_stage()
    {
        src_offset_ += src_step_s_;
        dst_offset_ += dst_step_s_;

        if (dst_offset_ >= kSmemByteSize) {
            dst_offset_ -= kSmemByteSize;
        }
    }

    __device__ void prefetch(bool mask)
    {
        cp_async_cg_A(smem_int_ptr_ + dst_offset_, (const AccessType*)src_ + src_offset_, mask);
    }
};

template<int WARPS, int CTA_M, int CTA_N, int CTA_K, int STAGES, int SLICES, int GROUP_SIZE, typename T_Q>
struct IteratorQ {
    static constexpr int SLICE_K = CTA_K / SLICES;

    using AccessType                 = uint;
    static constexpr int kAccessSize = sizeof(AccessType);

    static constexpr int kAccessM = kAccessSize / sizeof(T_Q);
    static constexpr int kAccessK = GROUP_SIZE;

    // warp thread arrangement
    static constexpr int kWarpThreadC = 32;
    static constexpr int kWarpThreadS = 1;

    // warp shape per access
    static constexpr int kWarpAccessM = kWarpThreadC * kAccessM;  // 32
    static constexpr int kWarpAccessK = kWarpThreadS * kAccessK;  // GROUP_SIZE

    // warp access iterations
    static constexpr int kWarpIterM = CTA_M / kWarpAccessM;    // CTA_M / 32
    static constexpr int kWarpIterK = SLICE_K / kWarpAccessK;  // SLICE_K / GROUP_SIZE, maybe 0

    // kWarpIterK == 0 => SLICE_K < kWarpAccessK => kIterK == 1

    // warp arrangement
    static constexpr int kWarpM = kWarpIterM >= WARPS ? WARPS : kWarpIterM;
    static constexpr int kWarpK = WARPS > kWarpIterM ? WARPS / kWarpM : 1;

    // iterations
    static constexpr int kIterM     = kWarpIterM / kWarpM;
    static constexpr int kIterK     = kWarpIterK >= kWarpK ? kWarpIterK / kWarpK : 1;
    static constexpr int kIterCount = kIterM * kIterK;

    // warp footprint
    static constexpr int kWarpFootprintM = kWarpAccessM * kIterM;
    static constexpr int kWarpFootprintK = kWarpAccessK * kIterK;

    static constexpr int kSizePerStage = std::max(SLICE_K / GROUP_SIZE, 1) * CTA_M;
    static constexpr int kSmemByteSize = sizeof(uint) * STAGES * kSizePerStage;

    const T_Q* const src_;
    T_Q* const       smem_;
    uint32_t const     smem_int_ptr_;

    const int m_;
    const int k_;

    bool is_out_of_bound_;  // mask for out-of-bound warps

    int src_offset_k_;
    int src_offset_m_;

    int src_offset_;
    int src_step_m_;
    int src_step_k_;

    int dst_offset_;
    int dst_step_m_;
    int dst_step_k_;

    int tmp_src_offset_;
    int tmp_dst_offset_;

    int iter_m_{0};

    struct Storage {
        T_Q data[SLICES][STAGES * kSizePerStage];
    };

    IteratorQ() = default;

    __device__ IteratorQ(const T_Q* src, T_Q* smem, int m, int k, int cta_m, int cta_k, int warp_id, int lane_id):
        src_(src), smem_(smem), smem_int_ptr_(cast_smem_ptr_to_uint(smem)), m_(m), k_(k)
    {
        const int warp_offset_m = warp_id % kWarpM;
        const int warp_offset_k = warp_id / kWarpM;

        const int warp_thread_offset_m = lane_id % kWarpThreadC;
        const int warp_thread_offset_k = lane_id / kWarpThreadC;

        const int cta_thread_offset_m = kWarpFootprintM * warp_offset_m + warp_thread_offset_m * kAccessM;
        const int cta_thread_offset_k = kWarpFootprintK * warp_offset_k + warp_thread_offset_k * kAccessK;

        // mask out-of-bound warps
        is_out_of_bound_ = cta_thread_offset_k >= SLICE_K;

        src_offset_m_ = cta_thread_offset_m + cta_m;
        src_offset_k_ = cta_thread_offset_k + cta_k;

        src_offset_ = src_offset_k_ / GROUP_SIZE * m_ + src_offset_m_;
        src_step_m_ = kWarpAccessM;
        src_step_k_ = m_ - kIterM * kWarpAccessM;  // valid only when SLICE_K >= GROUP_SIZE

        const int dst_offset_m = cta_thread_offset_m;
        const int dst_offset_k = cta_thread_offset_k;

        dst_offset_ = dst_offset_k / GROUP_SIZE * CTA_M + dst_offset_m;
        dst_step_m_ = kWarpAccessM;
        dst_step_k_ = CTA_M - kIterM * kWarpAccessM;  // valid only when SLICE_K >= GROUP_SIZE

        dst_offset_ *= kAccessSize;
        dst_step_m_ *= kAccessSize;
        dst_step_k_ *= kAccessSize;

        tmp_src_offset_ = src_offset_;
        tmp_dst_offset_ = dst_offset_;
    }

    __device__ void prefetch_stage(bool mask)
    {
        if (is_out_of_bound_) {
            return;
        }

        PRAGMA_UNROLL
        for (int i = 0; i < kIterCount; ++i) {
            prefetch(mask);
            ++(*this);
        }
        next_stage();
    }

    __device__ void prefetch_batch(int batch_idx, int batch_size, bool mask)
    {
        if (is_out_of_bound_) {
            return;
        }

        PRAGMA_UNROLL
        for (int i = 0; i < batch_size; ++i) {
            if (batch_idx * batch_size + i < kIterCount) {
                prefetch(mask);
                ++(*this);
            }
        }
    }

    __device__ IteratorQ& operator++()
    {
        ++iter_m_;

        src_offset_ += src_step_m_;
        dst_offset_ += dst_step_m_;
        if (iter_m_ < kIterM) {
            return *this;
        }

        iter_m_ = 0;

        if constexpr (SLICE_K >= GROUP_SIZE) {
            src_offset_ += src_step_k_;
            dst_offset_ += dst_step_k_;
        }
        // else advnace offsets in `next_stage`

        return *this;
    }

    __device__ void next_stage()
    {
        if constexpr (SLICE_K >= GROUP_SIZE) {
            src_offset_ += (CTA_K / GROUP_SIZE - kIterK) * m_;
            dst_offset_ += kAccessSize * (SLICE_K / GROUP_SIZE - kIterK) * CTA_M;
        }
        else {  // SLICE_K < GROUP_SIZE, recompute `src_offset_`
            src_offset_k_ += CTA_K;
            src_offset_ = (src_offset_k_ / GROUP_SIZE) * m_ + src_offset_m_;
            dst_offset_ += dst_step_k_;
        }

        if (dst_offset_ >= kSmemByteSize) {
            dst_offset_ -= kSmemByteSize;
        }
    }

    __device__ void prefetch(bool mask)
    {
        cp_async_ca(smem_int_ptr_ + dst_offset_, (const AccessType*)src_ + src_offset_, mask);
    }
};

template<int WARPS, int CTA_M, int CTA_N, int CTA_K, int STAGES, int SLICES, typename T_BC>
struct IteratorB {

    static constexpr int SLICE_K      = CTA_K / SLICES;
    static constexpr int kElementSize = sizeof(T_BC);
    using AccessType                  = uint4;
    static constexpr int kAccessSize  = sizeof(AccessType);

    static constexpr int kShapeK = SLICE_K;
    static constexpr int kShapeN = CTA_N;

    static constexpr int kAccessK = kAccessSize / sizeof(T_BC);

    static_assert(kShapeK % kAccessSize == 0);

    // warp thread arrangement
    static constexpr int kWarpThreadC = std::max(kShapeK / kAccessK, 1);
    static constexpr int kWarpThreadS = WARP_SIZE / kWarpThreadC;

    // warp shape per access
    static constexpr int kWarpAccessK = kWarpThreadC * kAccessK;
    static constexpr int kWarpAccessN = kWarpThreadS;

    // warp access iterations
    static constexpr int kWarpIterK = kShapeK / kWarpAccessK;
    static constexpr int kWarpIterN = kShapeN / kWarpAccessN;

    // warp arrangement
    static constexpr int kWarpK = kWarpIterK >= WARPS ? WARPS : kWarpIterK;
    static constexpr int kWarpN = WARPS > kWarpIterK ? WARPS / kWarpK : 1;

    // iterations
    static constexpr int kIterK = kWarpIterK / kWarpK;
    static constexpr int kIterN = kWarpIterN >= kWarpN ? kWarpIterN / kWarpN : 1;

    static constexpr int kIterCount = kIterK * kIterN;
    static_assert(kIterCount > 0);

    // warp footprint
    static constexpr int kWarpFootprintK = kWarpAccessK * kIterK;
    static constexpr int kWarpFootprintN = kWarpAccessN * kIterN;

    // Eliminate bank-conflicts for 8x4 half2 tiles, watch out for misalignment
    static constexpr int kSmemPadCtaK  = SLICE_K + 8;
    static constexpr int kSizePerTile  = CTA_N * kSmemPadCtaK;
    static constexpr int kSmemByteSize = kElementSize * STAGES * kSizePerTile;

    const T_BC*       src_;
    AccessType* const smem_;  // [CTA_N, SLICE_K + 8]
    const uint32_t    smem_int_ptr_;
    const int         k_;
    const int         n_;
    const int         cta_n_;
    const int         warp_id_;
    const int         lane_id_;
    const int         c_;
    const int         s_;

    int src_offset_n_;

    int src_offset_;
    int dst_offset_;

    int  src_step_k_;
    int  src_step_n_;
    int  dst_step_k_;
    int  dst_step_n_;
    bool is_valid_n_;

    int tmp_src_offset_;
    int tmp_dst_offset_;
    int tmp_src_offset_n_;

    int iter_k_{0};
    int iter_n_{0};

    IteratorB() = default;

    __device__ IteratorB(const T_BC* src, void* smem, int k, int n, int cta_n, int cta_k, int warp_id, int lane_id):
        src_(src),
        smem_((AccessType*)smem),
        smem_int_ptr_(cast_smem_ptr_to_uint(smem)),
        k_(k),
        n_(n),
        cta_n_(cta_n),
        warp_id_(warp_id),
        lane_id_(lane_id),
        c_(lane_id_ % kWarpThreadC),
        s_(lane_id_ / kWarpThreadC)
    {

        const int warp_offset_k = warp_id_ % kWarpK;
        const int warp_offset_n = warp_id_ / kWarpK;

        const int warp_thread_offset_k = lane_id_ % kWarpThreadC;
        const int warp_thread_offset_n = lane_id_ / kWarpThreadC;

        const int cta_thread_offset_k = kWarpFootprintK * warp_offset_k + warp_thread_offset_k * kAccessK;
        const int cta_thread_offset_n = kWarpFootprintN * warp_offset_n + warp_thread_offset_n;

        const int src_offset_k = cta_thread_offset_k + cta_k;
        src_offset_n_          = cta_thread_offset_n + cta_n_;

        src_offset_ = src_offset_n_ * k_ + src_offset_k;

        const int dst_offset_k = cta_thread_offset_k;
        const int dst_offset_n = cta_thread_offset_n;

        dst_offset_ = dst_offset_n * kSmemPadCtaK + dst_offset_k;

        src_step_k_ = kWarpAccessK;
        src_step_n_ = kWarpAccessN * k_ - kIterK * kWarpAccessK;

        dst_step_k_ = kWarpAccessK;
        dst_step_n_ = kWarpAccessN * kSmemPadCtaK - kIterK * kWarpAccessK;

        dst_offset_ *= kElementSize;
        dst_step_k_ *= kElementSize;
        dst_step_n_ *= kElementSize;

        tmp_src_offset_   = src_offset_;
        tmp_dst_offset_   = dst_offset_;
        tmp_src_offset_n_ = src_offset_n_;
        is_valid_n_       = tmp_src_offset_n_ < n_;
    }

    __device__ void prefetch_stage(bool mask)
    {

        PRAGMA_UNROLL
        for (int i = 0; i < kIterCount; ++i) {
            prefetch(mask);
            ++(*this);
        }
        next_stage();
    }

    __device__ void prefetch_batch(int batch_idx, int batch_size, bool mask)
    {
        PRAGMA_UNROLL
        for (int i = 0; i < batch_size; ++i) {
            if (batch_idx * batch_size + i < kIterCount) {
                prefetch(mask);
                ++(*this);
            }
        }
    }

    __device__ IteratorB& operator++()
    {
        if (!is_valid_n_) {
            return *this;
        }

        // move to next k
        tmp_src_offset_ += src_step_k_;
        tmp_dst_offset_ += dst_step_k_;
        ++iter_k_;
        if (iter_k_ < kIterK) {
            return *this;
        }

        // move to next n
        iter_k_ = 0;
        tmp_src_offset_n_ += kWarpAccessN;
        tmp_src_offset_ += src_step_n_;
        tmp_dst_offset_ += dst_step_n_;
        is_valid_n_ = tmp_src_offset_n_ < n_;
        ++iter_n_;

        return *this;
    }

    __device__ void next_stage()
    {
        iter_n_ = 0;

        src_offset_ += CTA_K;
        dst_offset_ += kElementSize * kSizePerTile;
        if (dst_offset_ >= kSmemByteSize) {
            dst_offset_ -= kSmemByteSize;
        }

        tmp_src_offset_   = src_offset_;
        tmp_dst_offset_   = dst_offset_;
        tmp_src_offset_n_ = src_offset_n_;

        is_valid_n_ = tmp_src_offset_n_ < n_;
    }

    __device__ void prefetch(bool mask)
    {
        cp_async_cg_B(
            smem_int_ptr_ + tmp_dst_offset_, (const AccessType*)(src_ + tmp_src_offset_), is_valid_n_ && mask);
    }
};

}  // namespace autoquant
}  // namespace vllm
