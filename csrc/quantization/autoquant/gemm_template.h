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

#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>
#include "common.h"
#include "cta_iterator.h"
#include "warp_iterator.h"

namespace vllm {
namespace autoquant {

__inline__ __device__ void
mma_m16n8k16_row_col(Array<float, 4>& d, const Array<half, 8>& a, const Array<half, 4>& b, Array<float, 4>& c)
{
#if VLLM_ARCH_SM80
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    float const*    C = reinterpret_cast<float const*>(&c);
    float*          D = reinterpret_cast<float*>(&d);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#else
    assert(VLLM_ARCH_SM80);
#endif
}

__inline__ __device__ void
mma_m16n8k16_row_col(Array<float, 4>& d, const Array<__nv_bfloat16, 8>& a, const Array<__nv_bfloat16, 4>& b, Array<float, 4>& c)
{
#if VLLM_ARCH_SM80
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    float const*    C = reinterpret_cast<float const*>(&c);
    float*          D = reinterpret_cast<float*>(&d);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#else
    assert(VLLM_ARCH_SM80);
#endif
}

__inline__ __device__ uint transpose_m8n8_b16_warp_shuffle(uint value, int lane_id)
{
    int    src_lane = lane_id / 8 + lane_id % 4 * 8;
    uint   u0       = __shfl_sync(0xffffffff, value, src_lane);
    uint   u1       = __shfl_sync(0xffffffff, value, src_lane + 4);
    short2 r;

    if (lane_id % 8 < 4) {
        r.x = ((short2&)u0).x;
        r.y = ((short2&)u1).x;
    }
    else {
        r.x = ((short2&)u0).y;
        r.y = ((short2&)u1).y;
    }
    return (uint&)r;
}

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 8)
__inline__ __device__ uint transpose_m8n8_b16_movmatrix(uint a)
{
#if VLLM_ARCH_SM75
    uint d;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
    return d;
#else
    assert(VLLM_ARCH_SM75);
    return 0;
#endif
}
#endif

__inline__ __device__ uint transpose_m8n8_b16(uint a, int lane_id)
{

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 8)
    (void)lane_id;
    return transpose_m8n8_b16_movmatrix(a);
#else
    return transpose_m8n8_b16_warp_shuffle(a, lane_id);
#endif
}

namespace ops {

__inline__ __device__ float4 operator+(const float4& a, const float4& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__inline__ __device__ float2 operator+(const float2& a, const float2& b)
{
    return {a.x + b.x, a.y + b.y};
}

}  // namespace ops

template<int CTA_M,
         int CTA_N,
         int CTA_K,
         int WARP_M,
         int WARP_N,
         int WARP_K,
         int STAGES,
         int GROUP_SIZE,
         typename OutputOps,
         typename T_BC,
         typename T_Q>
struct Gemm {

    static constexpr int kWarpCountM = CTA_M / WARP_M;
    static constexpr int kWarpCountN = CTA_N / WARP_N;
    static constexpr int kWarpCountK = CTA_K / WARP_K;

    static constexpr int kWarpCountMN = kWarpCountM * kWarpCountN;
    static constexpr int kWarpCount   = kWarpCountMN * kWarpCountK;

    static constexpr int SLICES  = kWarpCountK;
    static constexpr int SLICE_K = CTA_K / SLICES;

    static_assert(SLICE_K % WARP_K == 0, "infeasible sliced-k setting");

    using IteratorA = vllm::autoquant::IteratorA<kWarpCountMN, CTA_M, CTA_N, CTA_K, STAGES, SLICES>;
    using IteratorQ = vllm::autoquant::IteratorQ<kWarpCountMN, CTA_M, CTA_N, CTA_K, STAGES, SLICES, GROUP_SIZE, T_Q>;
    using IteratorB = vllm::autoquant::IteratorB<kWarpCountMN, CTA_M, CTA_N, CTA_K, STAGES, SLICES, T_BC>;

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 8;
    static constexpr int OP_K = 16;

    using WarpIterA = vllm::autoquant::WarpIteratorA<CTA_M,
                                               CTA_K,
                                               WARP_M,
                                               WARP_K,
                                               OP_M,
                                               OP_K,
                                               GROUP_SIZE,
                                               STAGES,
                                               IteratorA::kSizePerStage,
                                               IteratorQ::kSizePerStage,
                                               T_BC,
                                               T_Q>;

    using WarpIterB =
        vllm::autoquant::WarpIteratorB<CTA_N, CTA_K, WARP_N, WARP_K, OP_N, OP_K, IteratorB::kSmemPadCtaK, STAGES, T_BC>;

    __device__ void warp_mma(IteratorA& iter_A,
                             IteratorQ& iter_Q,
                             IteratorB& iter_B,
                             WarpIterA& warp_iter_A,
                             WarpIterB& warp_iter_B,
                             float*     accum,
                             int        slice_id,
                             int&       gemm_iter)
    {

        constexpr int ITER_M = WARP_M / OP_M;
        constexpr int ITER_N = WARP_N / OP_N;
        constexpr int ITER_K = WARP_K / OP_K;

        constexpr int kBatchA = (IteratorA::kIterCount + ITER_K - 1) / ITER_K;
        constexpr int kBatchQ = (IteratorQ::kIterCount + ITER_K - 1) / ITER_K;
        constexpr int kBatchB = (IteratorB::kIterCount + ITER_K - 1) / ITER_K;

        auto frag_C_ptr = (Array<float, 4>*)accum;  // [ITER_N, ITER_M]

        PRAGMA_UNROLL
        for (int iter_k = 0; iter_k < ITER_K; ++iter_k) {

            warp_iter_A.load(warp_frag_A_[(iter_k + 1) % 2], (iter_k + 1) % ITER_K);
            warp_iter_B.load(warp_frag_B_[(iter_k + 1) % 2], (iter_k + 1) % ITER_K);

            auto warp_frag_A = warp_frag_A_[iter_k % 2];
            auto warp_frag_B = warp_frag_B_[iter_k % 2];

            PRAGMA_UNROLL
            for (int iter_m = 0; iter_m < ITER_M; ++iter_m) {
                PRAGMA_UNROLL
                for (int iter_n = 0; iter_n < ITER_N; ++iter_n) {
                    auto& frag_A = warp_frag_A[iter_m];
                    auto& frag_B = warp_frag_B[iter_n];
                    auto& frag_C = frag_C_ptr[iter_n * ITER_M + iter_m];
                    mma_m16n8k16_row_col(frag_C, frag_A, frag_B, frag_C);
                }
            }

            if (iter_k < ITER_K - 1) {
                iter_A.prefetch_batch(iter_k, kBatchA, gemm_iter > 0);
                iter_Q.prefetch_batch(iter_k, kBatchQ, gemm_iter > 0);
                iter_B.prefetch_batch(iter_k, kBatchB, gemm_iter > 0);
            }

            if (iter_k == ITER_K - 2) {
                iter_A.prefetch_batch(iter_k + 1, kBatchA, gemm_iter > 0);
                iter_Q.prefetch_batch(iter_k + 1, kBatchQ, gemm_iter > 0);
                iter_B.prefetch_batch(iter_k + 1, kBatchB, gemm_iter > 0);

                __pipeline_commit();
                __pipeline_wait_prior(STAGES - 2);
                sync_slice(slice_id);

                iter_A.next_stage();
                iter_Q.next_stage();
                iter_B.next_stage();

                warp_iter_A.next_stage();
                warp_iter_B.next_stage();

                --gemm_iter;
            }
        }
    }

    template<typename T, int N>
    __device__ static void copy(T (&dst)[N], const T (&src)[N])
    {
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            dst[i] = src[i];
        }
    }

    template<typename T, int N>
    __device__ static void clear(T (&dst)[N])
    {
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            dst[i] = T{};
        }
    }

    __device__ void sync_slice(int slice_id)
    {
        if constexpr (SLICES == 1) {
            __syncthreads();
        }
        else {
            constexpr int      SLICE_GROUP = (SLICES + 7) / 8;
            constexpr uint32_t num_threads = kWarpCountMN * WARP_SIZE;
            const uint32_t     barrier_id  = slice_id / SLICE_GROUP + 1;
            asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "n"(num_threads));
        }
    }

    __device__ void load_partial(float* tb_frag_C, const float* partial_C, int cta, int slice_id)
    {
        if (slice_id == 0) {
            PRAGMA_UNROLL
            for (int i = 0; i < CTA_N; ++i) {
                tb_frag_C[i] += partial_C[cta * CTA_N * CTA_M + i * CTA_M + threadIdx.x];
            }
        }
    }

    __device__ void store_partial(float* partial_C, const float* tb_frag_C, int cta, int slice_id)
    {
        if (slice_id == 0) {
            PRAGMA_UNROLL
            for (int i = 0; i < CTA_N; ++i) {
                partial_C[cta * CTA_N * CTA_M + i * CTA_M + threadIdx.x] = tb_frag_C[i];
            }
        }
    }

    template<int Index>
    __device__ void store_accum(float* tb_frag_C,
                                float* tb_smem_C,
                                T_BC*  C,
                                int    m,
                                int    n,
                                int    cta_m,
                                int    cta_n,
                                int    warp_id_m,
                                int    warp_id_n,
                                int    lane_id,
                                int    slice_id)
    {

        if (slice_id != 0) {
            return;
        }

        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16816-c
        PRAGMA_UNROLL
        for (int i = 0; i < WARP_N / OP_N; ++i) {
            const float2* frag_C = (float2*)&tb_frag_C[i * WARP_M / OP_M * 4];
            const int     nn     = cta_n + warp_id_n * WARP_N + i * OP_N + lane_id / 4;
            PRAGMA_UNROLL
            for (int j = 0; j < WARP_M / OP_M; ++j) {
                PRAGMA_UNROLL
                for (int x = 0; x < 2; ++x) {
                    const int mm = cta_m + warp_id_m * WARP_M + j * OP_M + x * 8 + lane_id % 4 * 2;
                    if(std::is_same<T_BC, half>::value){
                        // convert to half
                        float2 frag_c = frag_C[j * 2 + x];
                        frag_c.x = clamp_inf_for_half(frag_c.x);
                        frag_c.y = clamp_inf_for_half(frag_c.y);
                        half2 half_C = __float22half2_rn(frag_c);
                        // transpose 8x8 accum tile
                        uint trans_C = transpose_m8n8_b16((uint&)half_C, lane_id);
                        // store to global memory
                        OutputOps::template apply<Index>(trans_C, mm, nn, C, m, n);
                    }
                    else{
                        // convert to bfloat16
                        auto half_C = float22bfloat162_rn(frag_C[j * 2 + x]) ;
                        // transpose 8x8 accum tile
                        uint trans_C = transpose_m8n8_b16((uint&)half_C, lane_id);
                        // store to global memory
                        OutputOps::template apply<Index>(trans_C, mm, nn, C, m, n);
                    }
                }
            }
        }
    }

    __device__ void
    sum_slices(float* tb_frag_C, float* tb_smem_C, int warp_id_m, int warp_id_n, int lane_id, int slice_id)
    {

        int offset_m = warp_id_m * WARP_M / OP_M;
        int offset_n = warp_id_n * WARP_N / OP_N;

        PRAGMA_UNROLL
        for (int z = 0; z < SLICES; ++z) {
            if (slice_id == z) {
                PRAGMA_UNROLL
                for (int i = 0; i < WARP_N / OP_N; ++i) {
                    PRAGMA_UNROLL
                    for (int j = 0; j < WARP_M / OP_M; ++j) {
                        PRAGMA_UNROLL
                        for (int x = 0; x < 4; ++x) {
                            int src = (i * WARP_M / OP_M + j) * 4 + x;
                            int dst = ((i + offset_n) * CTA_M / OP_M + j + offset_m) * 4 + x;
                            if (z > 0) {
                                using namespace ops;
                                tb_frag_C[src] = tb_smem_C[dst * WARP_SIZE + lane_id] + tb_frag_C[src];
                            }
                            tb_smem_C[dst * WARP_SIZE + lane_id] = tb_frag_C[src];
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (slice_id == 0) {
            PRAGMA_UNROLL
            for (int i = 0; i < WARP_N / OP_N; ++i) {
                PRAGMA_UNROLL
                for (int j = 0; j < WARP_M / OP_M; ++j) {
                    PRAGMA_UNROLL
                    for (int x = 0; x < 4; ++x) {
                        int src = ((i + offset_n) * CTA_M / OP_M + j + offset_m) * 4 + x;
                        int dst = (i * WARP_M / OP_M + j) * 4 + x;

                        tb_frag_C[dst] = tb_smem_C[src * WARP_SIZE + lane_id];
                    }
                }
            }
        }
    }

    //Array<half, 8> warp_frag_A_[2][WARP_M / OP_M];
    //Array<half, 4> warp_frag_B_[2][WARP_N / OP_N];
    Array<T_BC, 8> warp_frag_A_[2][WARP_M / OP_M];
    Array<T_BC, 4> warp_frag_B_[2][WARP_N / OP_N];

    __device__ void run_v2(T_BC* __restrict__ C,
                           const uint* __restrict__ A,
                           const T_BC* __restrict__ B,
                           const T_Q* __restrict__ Q,
                           int M,
                           int N,
                           int K,
                           int output_op_idx)
    {
        static_assert(WARP_M % OP_N == 0);

        float tb_frag_C[(WARP_N / OP_N) * (WARP_M / OP_M) * 4];

        extern __shared__ uint8_t smem[];

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_id_m  = warp_id % kWarpCountM;
        const int warp_id_nk = warp_id / kWarpCountM;
        const int warp_id_n  = warp_id_nk % kWarpCountN;
        const int warp_id_k  = warp_id_nk / kWarpCountN;

        const int warp_id_mn = warp_id_n * kWarpCountM + warp_id_m;

        const int slice_id = warp_id_k;

        const int cta_k = slice_id * SLICE_K;  // sliced-k offset
        const int cta_m = blockIdx.x * CTA_M;
        const int cta_n = blockIdx.y * CTA_N;

        // each slice has its own partition of smem
        uint4* const tb_smem_A = (uint4*)(smem + IteratorA::kSmemByteSize * slice_id);
        T_BC* const tb_smem_B = (T_BC*)(smem + IteratorA::kSmemByteSize * SLICES + IteratorB::kSmemByteSize * slice_id);

        // [CTA_N / OP_N, CTA_M / OP_M, 4, WARP_SIZE], all mn fragments in CTA
        float* const tb_smem_C = (float*)smem;

        __shared__ typename IteratorQ::Storage tb_smem_Q_storage;

        auto tb_smem_Q = tb_smem_Q_storage.data[slice_id];

        IteratorA iter_A{A, tb_smem_A, M, K, cta_m, cta_k, warp_id_mn, lane_id};
        IteratorQ iter_Q{Q, tb_smem_Q, M, K, cta_m, cta_k, warp_id_mn, lane_id};
        IteratorB iter_B{B, tb_smem_B, K, N, cta_n, cta_k, warp_id_mn, lane_id};

        const int offset_m = warp_id_m * WARP_M + lane_id;

        WarpIterA warp_iter_A(iter_A.smem_, iter_Q.smem_, warp_id, lane_id, offset_m, cta_k);
        WarpIterB warp_iter_B(iter_B.smem_int_ptr_, warp_id_n, lane_id, 0);

        int gemm_iter = (K + CTA_K - 1) / CTA_K;

        PRAGMA_UNROLL
        for (int stage = 0; stage < STAGES - 1; ++stage, --gemm_iter) {
            iter_A.prefetch_stage(gemm_iter > 0);
            iter_Q.prefetch_stage(gemm_iter > 0);
            iter_B.prefetch_stage(gemm_iter > 0);
            __pipeline_commit();
        }

        clear(tb_frag_C);

        __pipeline_wait_prior(STAGES - 2);
        sync_slice(slice_id);

        warp_iter_A.load(warp_frag_A_[0], 0);
        warp_iter_B.load(warp_frag_B_[0], 0);

        PRAGMA_NO_UNROLL
        for (; gemm_iter > -STAGES + 1;) {
            warp_mma(iter_A, iter_Q, iter_B, warp_iter_A, warp_iter_B, tb_frag_C, slice_id, gemm_iter);
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        if constexpr (SLICES > 1) {
            sum_slices(tb_frag_C, tb_smem_C, warp_id_m, warp_id_n, lane_id, slice_id);
        }

        switch (output_op_idx) {
            case 0:
                store_accum<0>(tb_frag_C, tb_smem_C, C, M, N, cta_m, cta_n, warp_id_m, warp_id_n, lane_id, slice_id);
                break;
            case 1:
                store_accum<1>(tb_frag_C, tb_smem_C, C, M, N, cta_m, cta_n, warp_id_m, warp_id_n, lane_id, slice_id);
                break;
            default:
                return;
        }
    }
};

template<typename Gemm, typename T_BC, typename T_Q>
__global__ void gemm_s4_f16_nn(T_BC* __restrict__ C,
                               const uint* __restrict__ A,
                               const T_BC* __restrict__ B,
                               const T_Q* __restrict__ Q,
                               int M,
                               int N,
                               int K,
                               int output_op_idx)
{
    Gemm{}.run_v2(C, A, B, Q, M, N, K, output_op_idx);
}

}  // namespace autoquant
}  // namespace vllm