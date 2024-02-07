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

#include <iostream>
#include <memory>
#include <sstream>
#include "gemm_template.h"
#include "metric.h"

namespace vllm {
namespace autoquant {

template<typename T_BC, typename T_Q>
struct IGemmKernel {

    virtual ~IGemmKernel() = default;

    virtual void GetMetric(Metric& metric, int m, int n, int k) = 0;

    virtual void Launch(T_BC*        C,
                        const uint*  A,
                        const T_BC*  B,
                        const T_Q*   Q,
                        int          M,
                        int          N,
                        int          K,
                        int          output_op_idx,
                        cudaStream_t) = 0;

    virtual void Dump(std::ostream& os) = 0;
};

template<typename CtaShape, typename WarpShape, int Stages, int GroupSize, typename OutputOps, typename T_BC, typename T_Q>
struct GemmKernel: public IGemmKernel<T_BC, T_Q> {

    static constexpr CtaShape  cta_shape{};
    static constexpr WarpShape warp_shape{};

    using GemmType = Gemm<cta_shape.m(),
                          cta_shape.n(),
                          cta_shape.k(),
                          warp_shape.m(),
                          warp_shape.n(),
                          warp_shape.k(),
                          Stages,
                          GroupSize,
                          OutputOps,
                          T_BC,
                          T_Q>;

    decltype(&gemm_s4_f16_nn<GemmType, T_BC, T_Q>) kernel_func_;
    std::shared_ptr<cudaDeviceProp>     props_;
    int                                 max_active_ctas_{};

    static constexpr int kSlices       = GemmType::SLICES;
    static constexpr int kSmemSizeA    = GemmType::IteratorA::kSmemByteSize * kSlices;
    static constexpr int kSmemSizeB    = GemmType::IteratorB::kSmemByteSize * kSlices;
    static constexpr int kSmemSizeC    = sizeof(float) * cta_shape.m() * cta_shape.n();
    static constexpr int kSmemByteSize = std::max(kSmemSizeA + kSmemSizeB, kSmemSizeC);

    // static shared memory size of Q
    static constexpr int kSmemSizeQ = sizeof(typename GemmType::IteratorQ::Storage);

    explicit GemmKernel(std::shared_ptr<cudaDeviceProp> props = {}): props_(std::move(props))
    {
        if (!props_) {
            props_        = std::make_shared<cudaDeviceProp>();
            int device_id = -1;
            cudaGetDevice(&device_id);
            cudaGetDeviceProperties(props_.get(), device_id);
        }

        kernel_func_ = gemm_s4_f16_nn<GemmType, T_BC, T_Q>;
        cudaFuncSetAttribute(kernel_func_, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemByteSize);

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_ctas_, kernel_func_, GemmType::kWarpCount * WARP_SIZE, kSmemByteSize);
    };

    bool is_feasible(int m, int n, int k)
    {
        return m % cta_shape.m() == 0 && k % cta_shape.k() == 0;
    }

    void GetMetric(Metric& metric, int m, int n, int k) override
    {
        metric.cta_shape  = {cta_shape.m(), cta_shape.n(), cta_shape.k()};
        metric.warp_shape = {warp_shape.m(), warp_shape.n(), warp_shape.k()};
        metric.warps      = GemmType::kWarpCount;
        metric.stages     = Stages;
        metric.smem       = (kSmemByteSize + kSmemSizeQ) / 1024.f;

        metric.feasible = is_feasible(m, n, k) && max_active_ctas_ > 0;

        metric.prefer = cta_shape.m() != 64 || m <= k;

        if (!metric.feasible) {
            return;
        }

        int grid_size    = ((m + cta_shape.m() - 1) / cta_shape.m()) * ((n + cta_shape.n() - 1) / cta_shape.n());
        metric.grid_size = grid_size;

        metric.max_active_ctas = max_active_ctas_;

        metric.active_ctas =
            std::min(max_active_ctas_, (grid_size + props_->multiProcessorCount - 1) / props_->multiProcessorCount);

        metric.waves     = (float)grid_size / (props_->multiProcessorCount * metric.active_ctas);
        metric.occupancy = (metric.active_ctas * GemmType::kWarpCount)
                           / (float)(props_->maxThreadsPerMultiProcessor / props_->warpSize);

        metric.cta_cnt_m  = (m + cta_shape.m() - 1) / cta_shape.m();
        metric.cta_cnt_n  = (n + cta_shape.n() - 1) / cta_shape.n();
        metric.cta_iter_k = (k + cta_shape.k() - 1) / cta_shape.k();

        metric.tile_efficiency = (float)n / (metric.cta_cnt_n * cta_shape.n());
        metric.wave_efficiency = metric.waves / std::ceil(metric.waves);

        const int m_pad = (m + cta_shape.m() - 1) / cta_shape.m() * cta_shape.m();
        const int n_pad = (n + cta_shape.n() - 1) / cta_shape.n() * cta_shape.n();

        metric.grid_a0  = 0.25f * m * n_pad / cta_shape.n();       // Ta0 *  M *  [N / ctaN]
        metric.grid_b0  = 1.00f * n * m_pad / cta_shape.m();       // Tb0 *  N *  [M / ctaM]
        metric.grid_a1  = 0.65f * m_pad * n_pad / warp_shape.n();  // Ta1 * [M] * [N] / warpN
        metric.grid_b1  = 0.25f * m_pad * n_pad / warp_shape.m();  // Tb1 * [M] * [N] / warpM
        metric.grid_mm  = 1.00f * m_pad * n_pad / 64;              // Tm * [M] * [N]
        metric.grid_sum = metric.grid_a0 + metric.grid_b0 + metric.grid_a1 + metric.grid_b1 + metric.grid_mm;

        metric.cta_sum = metric.grid_sum / grid_size;

        metric.waves1 = (float)grid_size / (props_->multiProcessorCount * metric.active_ctas);

        metric.cta_wave  = std::ceil(metric.waves1) * metric.active_ctas;
        metric.grid_norm = metric.cta_wave * metric.cta_sum;
    }

    void Launch(
        T_BC* C, const uint* A, const T_BC* B, const T_Q* Q, int M, int N, int K, int output_op_idx, cudaStream_t st)
        override
    {
        constexpr int block_size = GemmType::kWarpCount * WARP_SIZE;

        dim3 grid_size((M + cta_shape.m() - 1) / cta_shape.m(), (N + cta_shape.n() - 1) / cta_shape.n());

        kernel_func_<<<grid_size, block_size, kSmemByteSize, st>>>(C, A, B, Q, M, N, K, output_op_idx);
    }

    void Dump(std::ostream& os) override
    {
        {
            os << "[Gemm] CTA shape: " << cta_shape.m() << "x" << cta_shape.n() << "x" << cta_shape.k() << std::endl;
            os << "[Gemm] warp shape: " << warp_shape.m() << "x" << warp_shape.n() << "x" << warp_shape.k()
               << std::endl;
            os << "[Gemm] warp count: " << GemmType::kWarpCountM << "x" << GemmType::kWarpCountN << "x"
               << GemmType::kWarpCountK << " (" << GemmType::kWarpCount << ")" << std::endl;
            os << std::endl;
        }

        {
            using Iter = typename GemmType::IteratorA;
            os << "[A] shape: " << Iter::kShapeM << " " << Iter::kShapeK << std::endl;
            os << "[A] warp thread arrangement: " << Iter::kWarpThreadC << " " << Iter::kWarpThreadS << std::endl;
            os << "[A] warp shape per access: " << Iter::kWarpAccessM << " " << Iter::kWarpAccessK << std::endl;
            os << "[A] warp access iters: " << Iter::kWarpIterM << " " << Iter::kWarpIterK << std::endl;
            os << "[A] warp arrangement: " << Iter::kWarpM << " " << Iter::kWarpK << std::endl;
            os << "[A] iterations: " << Iter::kIterM << " " << Iter::kIterK << std::endl;
            os << "[A] iters per tile: " << Iter::kIterCount << std::endl;
            os << "[A] warp footprint: " << Iter::kWarpFootprintM << " " << Iter::kWarpFootprintK << std::endl;
            os << "[A] shared memory: " << Iter::kSmemByteSize << std::endl;
            os << std::endl;
        }
        {
            using Iter = typename GemmType::IteratorB;
            os << "[B] shape: " << Iter::kShapeK << " " << Iter::kShapeN << std::endl;
            os << "[B] warp thread arrangement: " << Iter::kWarpThreadC << " " << Iter::kWarpThreadS << std::endl;
            os << "[B] warp shape per access: " << Iter::kWarpAccessK << " " << Iter::kWarpAccessN << std::endl;
            os << "[B] warp access iters: " << Iter::kWarpIterK << " " << Iter::kWarpIterN << std::endl;
            os << "[B] warp arrangement: " << Iter::kWarpK << " " << Iter::kWarpN << std::endl;
            os << "[B] iterations: " << Iter::kIterK << " " << Iter::kIterN << std::endl;
            os << "[B] iters per tile: " << Iter::kIterCount << std::endl;
            os << "[B] warp footprint: " << Iter::kWarpFootprintK << " " << Iter::kWarpFootprintN << std::endl;
            os << "[B] shared memory: " << Iter::kSmemByteSize << std::endl;
            os << std::endl;
        }
        {

            using Iter = typename GemmType::IteratorQ;
            // os << "[Q] shape: " << CTA_M << " " << Iter::SLICE_K << std::endl;
            os << "[Q] warp thread arrangement: " << Iter::kWarpThreadC << " " << Iter::kWarpThreadS << std::endl;
            os << "[Q] warp shape per access: " << Iter::kWarpAccessM << " " << Iter::kWarpAccessK << std::endl;
            os << "[Q] warp access iters: " << Iter::kWarpIterM << " " << Iter::kWarpIterK << std::endl;
            os << "[Q] warp arrangement: " << Iter::kWarpM << " " << Iter::kWarpK << std::endl;
            os << "[Q] iterations: " << Iter::kIterM << " " << Iter::kIterK << std::endl;
            os << "[Q] iters per tile: " << Iter::kIterCount << std::endl;
            os << "[Q] warp footprint: " << Iter::kWarpFootprintM << " " << Iter::kWarpFootprintK << std::endl;
            os << "[Q] size per stage: " << Iter::kSizePerStage << std::endl;
            os << "[Q] shared memory: " << Iter::kSmemByteSize << std::endl;
            os << std::endl;
        }
        os << "Dynamic shared memory size: " << kSmemByteSize << std::endl;
    }
};

}  // namespace autoquant
}  // namespace vllm
