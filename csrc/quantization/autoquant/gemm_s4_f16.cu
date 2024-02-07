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

#include <algorithm>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "gemm_s4_f16.h"
#include "gemm_s4_f16_kernel.h"
#include "metric.h"
#include "common.h"

namespace vllm {
namespace autoquant {

bool g_dump_kernel_info_once = false;

namespace ops {

struct Identity {
    static __inline__ __device__ void apply(uint data, int m, int n, half* C, int M, int N)
    {
        if (n < N) {
            (uint&)C[n * M + m] = (uint&)data;
        }
    }

    static __inline__ __device__ void apply(uint data, int m, int n, __nv_bfloat16* C, int M, int N)
    {
        if (n < N) {
            (uint&)C[n * M + m] = (uint&)data;
        }
    }
};

struct SiluActivation {
    static __inline__ __device__ void apply(uint data, int m, int n, half* C, int M, int N)
    {
        auto  u    = __half22float2((half2&)data);
        float silu = u.x / (1.f + __expf(-u.x));
        half  val  = __float2half_rn(silu * u.y);

        if (n < N) {
            C[n * (M / 2) + m / 2] = val;
        }
    }

    static __inline__ __device__ void apply(uint data, int m, int n, __nv_bfloat16* C, int M, int N)
    {
        auto  u    = bfloat1622float2((__nv_bfloat162&)data);
        float silu = u.x / (1.f + __expf(-u.x));
        __nv_bfloat16  val  = __float2bfloat16_rn(silu * u.y);

        if (n < N) {
            C[n * (M / 2) + m / 2] = val;
        }
    }
};

}  // namespace ops

template<typename... Ts>
struct OutputOps {

    template<int index>
    static __inline__ __device__ void apply(uint data, int m, int n, half* C, int M, int N)
    {
        std::tuple_element_t<index, std::tuple<Ts...>>::apply(data, m, n, C, M, N);
    }

    template<int index>
    static __inline__ __device__ void apply(uint data, int m, int n, __nv_bfloat16* C, int M, int N)
    {
        std::tuple_element_t<index, std::tuple<Ts...>>::apply(data, m, n, C, M, N);
    }
};

template<typename T_BC, typename T_Q>
void Impl<T_BC, T_Q>::Generate(std::vector<Kernels>& kernels)
{
    // smem size (KB):
    // sm75: 64
    // sm80: 163
    // sm86: 99
    // sm89: 99
    // sm90: 227
    using Op = OutputOps<ops::Identity, ops::SiluActivation>;
    const int GS = 128;
    Kernels k;
    // 256
    k.emplace_back(new GemmKernel<Shape<256, 128, 32>, Shape<32, 128, 32>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<256, 64, 64>, Shape<64, 64, 32>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<256, 64, 32>, Shape<64, 64, 32>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<256, 32, 64>, Shape<64, 32, 32>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<256, 16, 256>, Shape<32, 16, 128>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<256, 8, 256>, Shape<32, 8, 128>, 3, GS, Op, T_BC, T_Q>{});
    // 128
    k.emplace_back(new GemmKernel<Shape<128, 128, 64>, Shape<32, 128, 32>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<128, 128, 32>, Shape<32, 128, 32>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<128, 96, 64>, Shape<32, 96, 32>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<128, 64, 64>, Shape<32, 64, 32>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<128, 64, 32>, Shape<32, 64, 32>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<128, 32, 128>, Shape<32, 32, 64>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<128, 16, 256>, Shape<32, 16, 64>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<128, 8, 512>, Shape<32, 8, 128>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<128, 8, 512>, Shape<32, 8, 128>, 2, GS, Op, T_BC, T_Q>{});  // for 86/89
    // 64
    k.emplace_back(new GemmKernel<Shape<64, 16, 256>, Shape<32, 16, 32>, 3, GS, Op, T_BC, T_Q>{});
    k.emplace_back(new GemmKernel<Shape<64, 8, 256>, Shape<32, 8, 32>, 3, GS, Op, T_BC, T_Q>{});
    kernels.push_back(std::move(k));
}

template<typename T_BC, typename T_Q>
void Impl<T_BC, T_Q>::Measure(T_BC*                 C,
                              const uint*           A,
                              const T_BC*           B,
                              const T_Q*            Q,
                              int                   m,
                              int                   n,
                              int                   k,
                              int                   group_size,
                              Type                  type,
                              std::vector<Metric>&  metrics,
                              cudaStream_t          st,
                              std::vector<Kernels>& _kernels)
{
    int gid = -1;
    for (size_t i = 0; i < group_sizes_.size(); ++i) {
        if (group_sizes_[i] == group_size) {
            gid = i;
            break;
        }
    }
    if (gid < 0) {
        throw std::runtime_error("unsupported group size");
    }
    const auto& kernels = _kernels[gid];
    metrics             = std::vector<Metric>(kernels.size());
    int best = 0;
    for (size_t i = 0; i < kernels.size(); ++i) {
        metrics[i].id = i;
        kernels[i]->GetMetric(metrics[i], m, n, k);
        if (!metrics[i].feasible) {
            metrics[i].time  = std::numeric_limits<float>::infinity();
            metrics[i].count = 1;
            continue;
        }
        if (Compare(metrics[i], metrics[best])) {
            best = i;
        }
        for (size_t j = 0; j < kWarmup + kMeasure; ++j) {
            if (j == kWarmup) {
                cudaEventRecord(ev_start_, st);
            }
            kernels[i]->Launch(C, A, B, Q, m, n, k, type, st);
        }
        cudaEventRecord(ev_end_, st);
        cudaEventSynchronize(ev_end_);
        float ms{};
        cudaEventElapsedTime(&ms, ev_start_, ev_end_);
        metrics[i].time  = ms;
        metrics[i].count = kMeasure;
    }
    metrics[best].best = 1;
    // sort metrics
    std::vector<int> indices(kernels.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(
        indices.begin(), indices.end(), [&](int i, int j) { return metrics[i].time < metrics[j].time; });
    if (g_dump_kernel_info_once) {
        DumpMetrics(std::cerr, metrics, indices);
        g_dump_kernel_info_once = 0;
    }
    std::vector<Metric> tmp;
    for (size_t i = 0; i < indices.size(); ++i) {
        tmp.push_back(metrics[indices[i]]);
    }
    metrics.swap(tmp);
}

static bool Compare(const Metric& a, const Metric& b)
{
    if (a.feasible != b.feasible) {
        return a.feasible > b.feasible;
    }
    if (a.prefer != b.prefer) {
        return a.prefer > b.prefer;
    }
    return a.grid_norm < b.grid_norm;
}

template<typename T_BC, typename T_Q>
int Impl<T_BC, T_Q>::Estimate(int m, int n, int k, Kernels& kernels)
{
    int                 best = 0;
    std::vector<Metric> metrics(kernels.size());
    for (size_t i = 0; i < kernels.size(); ++i) {
        metrics[i].id = i;
        kernels[i]->GetMetric(metrics[i], m, n, k);
        if (Compare(metrics[i], metrics[best])) {
            best = i;
        }
    }
    if (g_dump_kernel_info_once) {
        std::vector<int> indices(kernels.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(
            indices.begin(), indices.end(), [&](int i, int j) { return Compare(metrics[i], metrics[j]); });
        DumpMetrics(std::cerr, metrics, indices);
        g_dump_kernel_info_once = 0;
    }
    return best;
}

template<typename T_BC, typename T_Q>
void Impl<T_BC, T_Q>::Run(T_BC*                 C,
                          const uint*           A,
                          const T_BC*           B,
                          const T_Q*            Q,
                          int                   m,
                          int                   n,
                          int                   k,
                          int                   group_size,
                          Type                  type,
                          int                   algo_id,
                          cudaStream_t          st,
                          std::vector<Kernels>& kernels)
{
    for (size_t i = 0; i < group_sizes_.size(); ++i) {
        if (group_sizes_[i] == group_size) {
            if (algo_id < 0) {
                algo_id = Estimate(m, n, k, kernels[i]);
                //printf("**** m: %d, n: %d, k: %d, Run algo_id: %d \n", m, n, k, algo_id);
            }
            if (algo_id < 0) {
                throw std::runtime_error("no feasible kernel found");
            }
            kernels[i].at(algo_id)->Launch(C, A, B, Q, m, n, k, type, st);
            return;
        }
    }
    throw std::runtime_error("unsupported group size");
}

template<typename T_BC, typename T_Q>
Impl<T_BC, T_Q>::Impl()
{
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_end_);
    using Ops = OutputOps<ops::Identity, ops::SiluActivation>;
    /// TODO: add more group sizes
    //Generate<128, Ops>(kernels_);
    Generate(kernels_);
    group_sizes_.push_back(128);
}

template<typename T_BC, typename T_Q>
Impl<T_BC, T_Q>::~Impl()
{
    cudaEventDestroy(ev_end_);
    cudaEventDestroy(ev_start_);
}

template struct Impl<half, half2>;
template struct Impl<__nv_bfloat16, __nv_bfloat162>;

template<typename T_BC, typename T_Q>
GemmS4F16<T_BC, T_Q>::GemmS4F16(): impl_(std::make_unique<Impl<T_BC, T_Q>>()) {}

template<typename T_BC, typename T_Q>
GemmS4F16<T_BC, T_Q>::~GemmS4F16() = default;

template<typename T_BC, typename T_Q>
void GemmS4F16<T_BC, T_Q>::Measure(T_BC*                C,
                                   const uint*          A,
                                   const T_BC*          B,
                                   const T_Q*           Q,
                                   int                  m,
                                   int                  n,
                                   int                  k,
                                   int                  group_size,
                                   Type                 type,
                                   std::vector<Metric>& metrics,
                                   cudaStream_t         st)
{
    impl_->Measure(C, A, B, Q, m, n, k, group_size, type, metrics, st, impl_->kernels_);
}

template<typename T_BC, typename T_Q>
void GemmS4F16<T_BC, T_Q>::Run(T_BC*        C,
                               const uint*  A,
                               const T_BC*  B,
                               const T_Q*   Q,
                               int          m,
                               int          n,
                               int          k,
                               int          group_size,
                               Type         type,
                               int          algo_id,
                               cudaStream_t st)
{
    impl_->Run(C, A, B, Q, m, n, k, group_size, type, algo_id, st, impl_->kernels_);
}

template class GemmS4F16<half, half2>;
template class GemmS4F16<__nv_bfloat16, __nv_bfloat162>;

}  // namespace autoquant
}  // namespace vllm
