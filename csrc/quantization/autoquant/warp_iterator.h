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

#include "common.h"

namespace vllm {
namespace autoquant {

template<int CTA_M,
         int CTA_K,
         int WARP_M,
         int WARP_K,
         int OP_M,
         int OP_K,
         int GROUP_SIZE,
         int STAGES,
         int kSizePerStageA,
         int kSizePerStageQ,
         typename T_BC,
         typename T_Q>
struct WarpIteratorA {

    static_assert(WARP_K % GROUP_SIZE == 0 || GROUP_SIZE % WARP_K == 0);

    static constexpr int ITER_M = 32 / OP_M;
    static constexpr int ITER_X = WARP_M / 32;

    uint4 frag_A4_[ITER_X];    // 8 value per uint
    //half2 frag_Q_[ITER_X][4];  // 4 m8k8 tile along M, as WARP_M == 32
    T_Q frag_Q_[ITER_X][4];  // 4 m8k8 tile along M, as WARP_M == 32
    const uint4* smem_A_;
    const T_Q* smem_Q_;
    //const half2* smem_Q_;
    const int    offset_m_;
    const int    offset_m_Q_;

    int stage_{0};
    int offset_A_{0};
    int offset_Q_{0};

    //__device__ WarpIteratorA(uint4* smem_A, half2* smem_Q, int warp_id, int lane_id, int offset_m, int offset_k):
    __device__ WarpIteratorA(uint4* smem_A, T_Q* smem_Q, int warp_id, int lane_id, int offset_m, int offset_k):
        smem_A_(smem_A), smem_Q_(smem_Q), offset_m_(offset_m), offset_m_Q_(offset_m / 32 * 32 + lane_id / 4)
    {
    }

    // iter_k must be a compile tile constant
    __device__ void load(Array<T_BC, 8>* data, int iter_k)
    {
        // load A
        // smem_A uint4 [SLICE_K/32, CTA_M/32, WARP_SIZE], load as uint4 to avoid bank-conflicts
        if (iter_k % 2 == 0) {
            PRAGMA_UNROLL
            for (int x = 0; x < ITER_X; ++x) {
                frag_A4_[x] = smem_A_[offset_A_ + (iter_k / 2) * CTA_M + x * 32 + offset_m_];
            }
        }

        // load Q
        if (iter_k * OP_K % GROUP_SIZE == 0) {
            const int g = iter_k * OP_K / GROUP_SIZE;
            PRAGMA_UNROLL
            for (int x = 0; x < ITER_X; ++x) {
                PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) {
                    const int mm           = offset_m_Q_ + x * 32 + i * 8;  // stride of m8k8 tile
                    ((uint&)frag_Q_[x][i]) = ((uint&)smem_Q_[offset_Q_ + g * CTA_M + mm]);
                }
            }
        }

        PRAGMA_UNROLL
        for (int x = 0; x < ITER_X; ++x) {
            const uint* frag_A = (uint*)&frag_A4_[x];
            PRAGMA_UNROLL
            for (int iter_m = 0; iter_m < ITER_M; ++iter_m) {
                uint4 tmp;
                if(std::is_same<T_BC, half>::value){
                    tmp = dequantize_s4_to_fp16x2_v2(frag_A[iter_k % 2 * 2 + iter_m]);
                }
                else{
                    tmp = dequantize_s4_to_bf16x2_v2(frag_A[iter_k % 2 * 2 + iter_m]);
                }
                auto& vec = (Array<T_Q, 4>&)tmp;

                vec[0] = apply_Q(vec[0], frag_Q_[x][iter_m * 2]);
                vec[1] = apply_Q(vec[1], frag_Q_[x][iter_m * 2 + 1]);
                vec[2] = apply_Q(vec[2], frag_Q_[x][iter_m * 2]);
                vec[3] = apply_Q(vec[3], frag_Q_[x][iter_m * 2 + 1]);

                data[x * ITER_M + iter_m] = (Array<T_BC, 8>&)vec;
            }
        }
    }

    __device__ void next_stage()
    {
        ++stage_;
        if (stage_ >= STAGES) {
            stage_ = 0;
        }
        offset_A_ = stage_ * kSizePerStageA;
        offset_Q_ = stage_ * kSizePerStageQ;
    }
};

template<int CTA_N, int CTA_K, int WARP_N, int WARP_K, int OP_N, int OP_K, int SMEM_STRIDE, int STAGES, typename T_BC>
struct WarpIteratorB {

    static constexpr int kLdsmNum = WARP_N == 8 ? 2 : 4;
    static constexpr int ITER_N   = WARP_N / OP_N;
    static constexpr int ITER_K   = WARP_K / OP_K;

    static_assert(OP_N == 8 && OP_K == 16);

    const int warp_id_n_;
    const int lane_id_;

    const int ldsm_group_id_;

    const int offset_k_;
    int       offset_n_;

    const uint32_t smem_base_ptr_;

    uint32_t smem_ptr_;

    int stage_{0};

    __device__ WarpIteratorB(uint32_t smem_int_ptr, int warp_id_n, int lane_id, int offset_k):
        smem_base_ptr_(smem_int_ptr),
        smem_ptr_(smem_base_ptr_),
        warp_id_n_(warp_id_n),
        lane_id_(lane_id),
        ldsm_group_id_(lane_id / 8),
        offset_k_(ldsm_group_id_ % 2 * 8 + offset_k),
        offset_n_(ldsm_group_id_ / 2 * 8 + lane_id % 8)
    {
        if (kLdsmNum == 2) {
            offset_n_ -= ldsm_group_id_ / 2 * 8;
        }
        offset_n_ += warp_id_n_ * WARP_N;
    }

    __device__ void load(Array<T_BC, 4>* data, int iter_k)
    {
        const int kk  = iter_k * OP_K + offset_k_;
        auto      ptr = (uint*)data;
        PRAGMA_UNROLL
        for (int iter_n = 0; iter_n < ITER_N;) {
            const int nn  = offset_n_ + iter_n * OP_N;
            auto      src = smem_ptr_ + sizeof(T_BC) * (nn * SMEM_STRIDE + kk);
            if constexpr (kLdsmNum == 4) {
                ldmatrix_m8n8_x4_b16(ptr[0], ptr[1], ptr[2], ptr[3], src);
                ptr += 4;
                iter_n += 2;
            }
            else {
                ldmatrix_m8n8_x2_b16(ptr[0], ptr[1], src);
                ptr += 2;
                iter_n += 1;
            }
        }
    }

    __device__ void next_stage()
    {
        ++stage_;
        if (stage_ >= STAGES) {
            stage_ = 0;
        }
        smem_ptr_ = smem_base_ptr_ + stage_ * sizeof(half) * CTA_N * SMEM_STRIDE;
    }
};

}  // namespace autoquant
}  // namespace vllm
