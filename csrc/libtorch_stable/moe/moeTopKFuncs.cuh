/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc2/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh
 * https://github.com/flashinfer-ai/flashinfer/blob/06400d062a2d51564bbe781f6f811d0b75ca593e/include/flashinfer/trtllm/fused_moe/RoutingKernelTopK.cuh
 * Copyright (c) 2026, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <type_traits>

namespace vllm {
namespace moe {
namespace reduce_topk {
namespace cg = cooperative_groups;
static constexpr int kWARP_SIZE = 32;

template <typename T_>
struct TopKRedType {
  using T = T_;
  static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, half> ||
          std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, int>,
      "Top K reduction only implemented for int, float, float16 and bfloat16");

  using TypeCmp = std::conditional_t<sizeof(T) == 4, uint64_t, uint32_t>;

  static constexpr int kMoveBits = (sizeof(T) == 4) ? 32 : 16;
  static constexpr int kMaxIdx = 65535;
  TypeCmp compVal;

  static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0) {
    auto valueBits = cub::Traits<T>::TwiddleIn(
        reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(val));
    TypeCmp compactTmp = valueBits;
    compactTmp = (compactTmp << kMoveBits) | (0xFFFF & (kMaxIdx - idx));
    // Use 65535 minus idx to give higher priority to elements with smaller
    // indices.
    return compactTmp;
  }

  static __host__ __device__ void unpack(T& value, int32_t& index,
                                         TypeCmp cmp) {
    // Since “65535-idx” is always smaller than 65536 and positive, we can
    // directly use it as the lower 16 bits
    index = kMaxIdx - static_cast<int32_t>((cmp & 0xFFFF));

    auto compactTmp = cmp >> kMoveBits;
    auto valueBits = cub::Traits<T>::TwiddleOut(
        reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(compactTmp));
    value = reinterpret_cast<T&>(valueBits);
  }

  __host__ __device__ TopKRedType() = default;

  __host__ __device__ TopKRedType(T val, int32_t idx)
      : compVal(makeCmpVal(val, idx)) {}

  __host__ __device__ operator TypeCmp() const noexcept { return compVal; }

  __device__ inline TypeCmp reduce(
      cg::thread_block_tile<kWARP_SIZE> const& warp) {
#ifdef __CUDA_ARCH__
    static constexpr bool kHAS_FAST_REDUX = (__CUDA_ARCH__ / 100) >= 10;
#else
    static constexpr bool kHAS_FAST_REDUX = false;
#endif
    if constexpr (!kHAS_FAST_REDUX) {
      return cg::reduce(warp, compVal, cg::greater<TypeCmp>{});
    } else if constexpr (sizeof(TypeCmp) == 8) {
      uint32_t hi = static_cast<uint32_t>(compVal >> 32);
      uint32_t lo = static_cast<uint32_t>(compVal & 0xffffffffu);
      uint32_t maxHi;
      asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;\n"
                   : "=r"(maxHi)
                   : "r"(hi));
      uint32_t loContrib = hi == maxHi ? lo : 0u;
      uint32_t maxLo;
      asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;\n"
                   : "=r"(maxLo)
                   : "r"(loContrib));
      return (static_cast<TypeCmp>(maxHi) << 32) | static_cast<TypeCmp>(maxLo);
    } else {
      TypeCmp result;
      asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;\n"
                   : "=r"(result)
                   : "r"(compVal));
      return result;
    }
  }
};

template <int N>
struct IsPowerOf2 {
  static constexpr bool value = N > 0 && (N & (N - 1)) == 0;
};

template <int N>
struct NextPow2 {
 private:
  static constexpr unsigned u = static_cast<unsigned>(N - 1);
  static constexpr unsigned s1 = u | (u >> 1);
  static constexpr unsigned s2 = s1 | (s1 >> 2);
  static constexpr unsigned s3 = s2 | (s2 >> 4);
  static constexpr unsigned s4 = s3 | (s3 >> 8);
  static constexpr unsigned s5 = s4 | (s4 >> 16);

 public:
  static constexpr int value = N <= 1 ? 1 : static_cast<int>(s5 + 1);
};

template <int A, int B, int Size, typename T>
__device__ __forceinline__ void topkCompareSwap(T* a) {
  if constexpr (A < Size && B < Size) {
    if (a[A] < a[B]) {
      T tmp = a[A];
      a[A] = a[B];
      a[B] = tmp;
    }
  } else {
    (void)a;
  }
}

template <int I, int End, int Step, int PairStride, int Size, typename T>
__device__ __forceinline__ void topkMergePairs(T* a) {
  if constexpr (I + Step < End) {
    topkCompareSwap<I, I + Step, Size, T>(a);
    topkMergePairs<I + PairStride, End, Step, PairStride, Size, T>(a);
  } else {
    (void)a;
  }
}

template <int Lo, int N, int R, int Size, typename T>
__device__ __forceinline__ void topkOEM(T* a) {
  constexpr int M = R * 2;
  if constexpr (M < N) {
    topkOEM<Lo, N, M, Size, T>(a);
    topkOEM<Lo + R, N - R, M, Size, T>(a);
    topkMergePairs<Lo + R, Lo + N, R, M, Size, T>(a);
  } else if constexpr (R < N) {
    topkCompareSwap<Lo, Lo + R, Size, T>(a);
  } else {
    (void)a;
  }
}

template <int Lo, int N, int Size, typename T>
__device__ __forceinline__ void topkSortBatcher(T* a) {
  if constexpr (N > 1) {
    constexpr int Half = N / 2;
    topkSortBatcher<Lo, Half, Size, T>(a);
    topkSortBatcher<Lo + Half, N - Half, Size, T>(a);
    topkOEM<Lo, N, 1, Size, T>(a);
  } else {
    (void)a;
  }
}

template <int N, typename RedType>
struct Sort {
  static_assert(N > 0 && N <= 64, "Sort only supports N in range [1, 64]");

  static __device__ void run(RedType* topK) {
    if constexpr (IsPowerOf2<N>::value) {
#pragma unroll
      for (int k = 2; k <= N; k *= 2) {
#pragma unroll
        for (int j = k / 2; j > 0; j /= 2) {
#pragma unroll
          for (int i = 0; i < N; ++i) {
            int ixj = i ^ j;
            if (ixj > i) {
              if ((i & k) == 0) {
                if (topK[i].compVal < topK[ixj].compVal) {
                  auto tmp = topK[i].compVal;
                  topK[i].compVal = topK[ixj].compVal;
                  topK[ixj].compVal = tmp;
                }
              } else {
                if (topK[i].compVal > topK[ixj].compVal) {
                  auto tmp = topK[i].compVal;
                  topK[i].compVal = topK[ixj].compVal;
                  topK[ixj].compVal = tmp;
                }
              }
            }
          }
        }
      }
    } else {
      constexpr int P = NextPow2<N>::value;
      topkSortBatcher<0, P, N, RedType>(topK);
    }
  }
};

template <typename RedType>
struct Sort<1, RedType> {
  static __device__ void run(RedType*) {}
};

template <typename RedType>
struct Sort<2, RedType> {
  static __device__ void run(RedType* topK) { topkCompareSwap<0, 1, 2>(topK); }
};

template <typename RedType>
struct Sort<3, RedType> {
  static __device__ void run(RedType* topK) {
    topkCompareSwap<0, 1, 3>(topK);
    topkCompareSwap<1, 2, 3>(topK);
    topkCompareSwap<0, 1, 3>(topK);
  }
};

template <typename RedType>
struct Sort<4, RedType> {
  static __device__ void run(RedType* topK) {
    topkCompareSwap<0, 2, 4>(topK);
    topkCompareSwap<1, 3, 4>(topK);
    topkCompareSwap<0, 1, 4>(topK);
    topkCompareSwap<2, 3, 4>(topK);
    topkCompareSwap<1, 2, 4>(topK);
  }
};

template <int K, typename Type>
__forceinline__ __device__ void reduceTopK(
    cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K],
    int32_t (&outIdx)[K], Type value, int32_t idx, Type const minValue,
    int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWARP_SIZE, "Top K must have K < kWARP_SIZE");
  using RedType = TopKRedType<Type>;
  RedType topK{value, idx};
  typename RedType::TypeCmp packedMax{};
#pragma unroll
  for (int kk = 0; kk < actualK; ++kk) {
    topK = kk > 0 && packedMax == topK.compVal ? RedType{minValue, idx} : topK;
    packedMax = topK.reduce(warp);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
};

template <int K, typename Type, int N>
__forceinline__ __device__ void reduceTopK(
    cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K],
    int32_t (&outIdx)[K], Type (&value)[N], int32_t (&idx)[N],
    Type const minValue, int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K <= kWARP_SIZE, "Top K must have K <= kWARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(N <= 64,
                "Only support candidates number less than or equal to "
                "64*32=2048");
  using RedType = TopKRedType<Type>;
  RedType topK[N];
#pragma unroll
  for (int nn = 0; nn < N; ++nn) {
    topK[nn] = RedType{value[nn], idx[nn]};
  }

  Sort<N, RedType>::run(topK);

  typename RedType::TypeCmp packedMax{};
  for (int kk = 0; kk < actualK; ++kk) {
    bool update = kk > 0 && packedMax == topK[0].compVal;
#pragma unroll
    for (int nn = 0; nn < N; ++nn) {
      topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]}
                 : update              ? topK[nn + 1]
                                       : topK[nn];
    }
    packedMax = topK[0].reduce(warp);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
};

template <int NumExperts, int NumTopExperts, int MinExperts, int MaxExperts,
          int MinTopExperts, int MaxTopExperts>
struct LaneOwnedTopKRange {
  static_assert(MinExperts > 0 && MinExperts <= MaxExperts);
  static_assert(MinTopExperts > 0 && MinTopExperts <= MaxTopExperts);
  static constexpr bool kEnabled =
      NumExperts >= MinExperts && NumExperts <= MaxExperts &&
      NumTopExperts >= MinTopExperts && NumTopExperts <= MaxTopExperts;
};

static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_EXPERTS = 512;
static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_EXPERTS = 1024;
static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_TOP_EXPERTS = 9;
static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_TOP_EXPERTS = 16;

template <int NumExperts, int NumTopExperts>
using HighExpertLaneOwnedTopKRange =
    LaneOwnedTopKRange<NumExperts, NumTopExperts,
                       kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_EXPERTS,
                       kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_EXPERTS,
                       kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_TOP_EXPERTS,
                       kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_TOP_EXPERTS>;

template <int K, typename Type, int N>
__forceinline__ __device__ void reduceTopKForLane(
    cg::thread_block_tile<kWARP_SIZE> const& warp, Type& out, int32_t& outIdx,
    Type (&value)[N], int32_t (&idx)[N], Type const minValue, int32_t laneIdx) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K <= kWARP_SIZE, "Top K must have K <= kWARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(N <= 64,
                "Only support candidates number less than or equal to "
                "64*32=2048");
  using RedType = TopKRedType<Type>;
  RedType topK[N];
#pragma unroll
  for (int nn = 0; nn < N; ++nn) {
    topK[nn] = RedType{value[nn], idx[nn]};
  }

  Sort<N, RedType>::run(topK);

  typename RedType::TypeCmp packedMax{};
  typename RedType::TypeCmp lanePacked{};
#pragma unroll
  for (int kk = 0; kk < K; ++kk) {
    bool update = kk > 0 && packedMax == topK[0].compVal;
#pragma unroll
    for (int nn = 0; nn < N; ++nn) {
      topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]}
                 : update              ? topK[nn + 1]
                                       : topK[nn];
    }
    packedMax = topK[0].reduce(warp);
    if (laneIdx == kk) {
      lanePacked = packedMax;
    }
  }

  if (laneIdx < K) {
    RedType::unpack(out, outIdx, lanePacked);
  } else {
    out = minValue;
    outIdx = -1;
  }
}

}  // namespace reduce_topk
}  // namespace moe
}  // namespace vllm
