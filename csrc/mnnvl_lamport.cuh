/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

// MNNVL Lamport all-reduce primitives adapted from FlashInfer's
// include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh (v0.7.0), itself from
// TensorRT-LLM. Workspace layout: three Lamport buffers per rank, initialized
// to FP32 -0.0, plus a uint32 flag array
// {current, dirty, bytes per buffer, dirty stages, bytes to clear[4], count}.

#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vllm {
namespace mnnvl {

template <typename T>
struct AllReduceKernelParams {
  T* outputPtr;
  T* prenormedPtr;
  T const* shardPtr;
  T const* residualInPtr;
  T const* gammaPtr;
  T** inputPtrs;
  T* mcastPtr;
  T* localBufferPtr;
  int nRanks;
  int rank;
  int numTokens;
  int tokenDim;
  float epsilon;
  uint32_t* bufferFlags;
};

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 ceil_div(const T1 x,
                                                          const T2 y) noexcept {
  return (x + y - 1) / y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 round_up(const T1 x,
                                                          const T2 y) noexcept {
  return ceil_div(x, y) * y;
}

namespace utils {

constexpr uint16_t kNEGZERO_FP16 = 0x8000U;
constexpr uint32_t kNEGZERO_FP32 = 0x80000000U;
constexpr int kMaxClusterSize = 8;

template <typename T>
union Fp16BitCast {
  T mFp;
  uint16_t mInt;

  constexpr Fp16BitCast() : mInt(0) {}

  constexpr Fp16BitCast(T val) : mFp(val) {}

  constexpr Fp16BitCast(uint16_t val) : mInt(val) {}
};

template <typename T>
inline __device__ float toFloat(T val) {
  return val;
}

template <>
inline __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
template <>
inline __device__ float toFloat<__nv_half>(__nv_half val) {
  return __half2float(val);
}

template <typename T>
inline __device__ T fromFloat(float val) {
  return val;
}

template <>
inline __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <>
inline __device__ __nv_half fromFloat<__nv_half>(float val) {
  return __float2half(val);
}

template <typename T>
static constexpr __device__ __host__ T negZero() {
  if constexpr (std::is_same_v<T, float>) {
    return -0.0F;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> ||
                       std::is_same_v<T, __nv_half>) {
    return Fp16BitCast<T>(kNEGZERO_FP16).mFp;
  } else {
    static_assert(sizeof(T) == 0, "negativeZero not specialized for this type");
  }
  return T{};
}

// The Lamport sentinel is a bit pattern. Compare bit-exact: an FTZ float
// compare would also match negative subnormals and deadlock the poll loop.
template <typename T>
static inline __device__ bool isNegZero(T val) {
  if constexpr (std::is_same_v<T, float>) {
    return __float_as_uint(val) == kNEGZERO_FP32;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> ||
                       std::is_same_v<T, __nv_half>) {
    return Fp16BitCast<T>(val).mInt == kNEGZERO_FP16;
  } else {
    static_assert(sizeof(T) == 0, "isNegZero not specialized for this type");
  }
  return false;
}

template <typename PackedType, typename T>
constexpr __device__ __host__ PackedType getPackedLamportInit() {
  static_assert(sizeof(PackedType) % sizeof(T) == 0,
                "PackedType size must be divisible by T size");
  constexpr int kNumElements = sizeof(PackedType) / sizeof(T);

  union PackedT {
    PackedType mPacked;
    std::array<T, kNumElements> mElements;

    constexpr PackedT() : mElements{} {
      for (int i = 0; i < kNumElements; i++) {
        mElements[i] = negZero<T>();
      }
    }
  };

  PackedT initValue{};
  return initValue.mPacked;
}

struct LamportBufferLayout {
  uint32_t numStages = 1;
  uint32_t bytesPerBuffer = 0;
  static constexpr uint32_t sNumLamportBuffers = 3;

  [[nodiscard]] __device__ __host__ size_t getTotalBytes() const {
    return numStages * static_cast<size_t>(bytesPerBuffer / numStages) *
           sNumLamportBuffers;
  }

  [[nodiscard]] __device__ __host__ void* getStagePtr(
      void* bufferBasePtr, uint32_t lamportIndex, uint32_t stageIndex) const {
    return reinterpret_cast<void*>(
        reinterpret_cast<char*>(bufferBasePtr) +
        static_cast<size_t>((lamportIndex * numStages + stageIndex) *
                            static_cast<size_t>(bytesPerBuffer / numStages)));
  }
};

namespace cg = cooperative_groups;

// PackedType is the Lamport buffer access width (LDG.128 or LDG.64).
template <typename PackedType = float4, bool UseCGA = false>
__device__ struct __attribute__((aligned(32))) LamportFlags {
 public:
  __device__ explicit LamportFlags(uint32_t* bufferFlags,
                                   uint32_t numStages = 1)
      : mBufferFlagsPtr(bufferFlags), mFlagAccessPtr(&bufferFlags[8]) {
    mCurBufferLayout.numStages = numStages;
    uint4 flag = reinterpret_cast<uint4*>(bufferFlags)[0];
    mCurrentIndex = flag.x;
    mDirtyIndex = flag.y;
    mCurBufferLayout.bytesPerBuffer = flag.z;
    mDirtyBufferLayout.bytesPerBuffer = flag.z;
    mDirtyBufferLayout.numStages = flag.w;
    *reinterpret_cast<uint4*>(&mBytesToClear) =
        reinterpret_cast<uint4*>(bufferFlags)[1];
  }

  [[nodiscard]] __device__ void* getCurLamportBuf(void* bufferBasePtr,
                                                  int stageIdx = 0) const {
    return mCurBufferLayout.getStagePtr(bufferBasePtr, mCurrentIndex, stageIdx);
  }

  // Refill a dirty stage (-1 for all) with the sentinel. Every thread of a
  // 1D-block grid must participate.
  __device__ void clearDirtyLamportBuf(void* bufferBasePtr, int stageIdx = -1) {
    uint32_t globalCtaIdx = blockIdx.x * gridDim.y + blockIdx.y;
    uint32_t globalTid = globalCtaIdx * blockDim.x + threadIdx.x;
    uint32_t numThreads = gridDim.x * gridDim.y * blockDim.x;

    if (stageIdx == -1) {
      for (uint32_t i = 0; i < mDirtyBufferLayout.numStages; i++) {
        clearPackedBuf(bufferBasePtr, globalTid, numThreads, mBytesToClear[i],
                       mDirtyIndex, i);
      }
    } else if (stageIdx < mDirtyBufferLayout.numStages) {
      clearPackedBuf(bufferBasePtr, globalTid, numThreads,
                     mBytesToClear[stageIdx], mDirtyIndex, stageIdx);
    }
  }

  __device__ void ctaArrive() {
    if constexpr (UseCGA) {
      cg::cluster_group cluster = cg::this_cluster();
      __cluster_barrier_arrive();
      if (cluster.block_rank() == 0 && threadIdx.x < 32) {
        __cluster_barrier_wait();
        arriveCounter(threadIdx.x);
      }
      return;
    }
    uint32_t const barrierThreads =
        round_up(static_cast<uint32_t>(blockDim.x), 32u);
    // A named barrier orders payload stores before the arrival counter
    // without a full __syncthreads().
    if (threadIdx.x < 32) {
      asm volatile("barrier.cta.sync 1, %0;" ::"r"(barrierThreads) : "memory");
      arriveCounter(threadIdx.x);
    } else {
      asm volatile("barrier.cta.arrive 1, %0;" ::"r"(barrierThreads)
                   : "memory");
    }
  }

  __device__ void waitAndUpdate(uint4 bytesToClearPerStage) {
    bool isLastCtaT0{false};
    int targetCount{0};
    cg::grid_group grid = cg::this_grid();
    // The last thread may exit early, so thread 0 updates the flags.
    isLastCtaT0 = grid.thread_rank() == 0;
    if constexpr (UseCGA) {
      cg::cluster_group cluster = cg::this_cluster();
      targetCount = gridDim.x * gridDim.y * gridDim.z / cluster.num_blocks();
    } else {
      targetCount = gridDim.x * gridDim.y * gridDim.z;
    }
    if (isLastCtaT0) {
      uint4* flagPtr = reinterpret_cast<uint4*>(mBufferFlagsPtr);
      while (*reinterpret_cast<uint32_t volatile*>(mFlagAccessPtr) <
             targetCount) {
      }
      // 'Current' becomes 'Dirty'.
      flagPtr[0] = {(mCurrentIndex + 1) % 3, mCurrentIndex,
                    mCurBufferLayout.bytesPerBuffer,
                    mCurBufferLayout.numStages};
      flagPtr[1] = bytesToClearPerStage;
      *mFlagAccessPtr = 0;
    }
  }

 private:
  uint32_t* mBufferFlagsPtr;
  uint32_t* mFlagAccessPtr;

  uint32_t mCurrentIndex, mDirtyIndex;
  alignas(16) std::array<uint32_t, 4> mBytesToClear;
  LamportBufferLayout mCurBufferLayout, mDirtyBufferLayout;

  __device__ void arriveCounter(int tid) {
    if (tid == 0) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
      asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(
                       mFlagAccessPtr),
                   "r"(1)
                   : "memory");
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
      asm volatile(
          "red.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr),
          "r"(1)
          : "memory");
#else
      atomicAdd(mFlagAccessPtr, 1);
#endif
    }
  }

  inline __device__ void clearPackedBuf(void* bufferBasePtr, uint32_t globalTid,
                                        uint32_t numThreads,
                                        uint32_t bytesToClear,
                                        uint8_t dirtyIndex, uint8_t stageIdx) {
    uint32_t clearBoundary =
        ceil_div<uint32_t>(bytesToClear, sizeof(PackedType));
    for (uint32_t packedIdx = globalTid; packedIdx < clearBoundary;
         packedIdx += numThreads) {
      reinterpret_cast<PackedType*>(mDirtyBufferLayout.getStagePtr(
          bufferBasePtr, dirtyIndex, stageIdx))[packedIdx] =
          getPackedLamportInit<PackedType, float>();
    }
  }
};

template <typename PackedType, typename T>
union PackedVec {
  PackedType packed;
  T elements[sizeof(PackedType) / sizeof(T)];
};

template <typename PackedType, typename T>
inline __device__ void sanitizeLamportPayload(PackedVec<PackedType, T>& value) {
#pragma unroll
  for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++) {
    if (isNegZero(value.elements[i])) {
      value.elements[i] = fromFloat<T>(0.f);
    }
  }
}

template <typename PackedType, typename T>
inline __device__ PackedType loadPacked(T* ptr) {
  return *reinterpret_cast<PackedType*>(ptr);
}

template <typename PackedType, typename T>
inline __device__ const PackedType loadPacked(T const* ptr) {
  return *reinterpret_cast<PackedType const*>(ptr);
}

template <typename PackedType>
union VolatilePackedLoad {
  PackedType packed;
  uint32_t words[sizeof(PackedType) / sizeof(uint32_t)];
};

template <typename PackedType>
inline __device__ VolatilePackedLoad<PackedType> loadPackedVolatile(
    void const* ptr) {
  static_assert(sizeof(PackedType) == 0, "Not implemented");
  return {};
}

template <>
inline __device__ VolatilePackedLoad<float4> loadPackedVolatile<float4>(
    void const* ptr) {
  VolatilePackedLoad<float4> returnValue;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(returnValue.words[0]), "=r"(returnValue.words[1]),
                 "=r"(returnValue.words[2]), "=r"(returnValue.words[3])
               : "l"(ptr)
               : "memory");
  return returnValue;
}

template <typename PackedType>
inline __device__ bool isLamportDirty(
    VolatilePackedLoad<PackedType> const& value) {
  // PTX only guarantees 64-bit atomicity, so every word must be checked.
  bool dirty = false;
#pragma unroll
  for (int i = 0; i < sizeof(PackedType) / sizeof(uint32_t); i++) {
    dirty |= value.words[i] == kNEGZERO_FP32;
  }
  return dirty;
}

template <int Rank, uint8_t WorldSize, uint8_t LocalRank, typename T,
          typename PackedType, int kELTS_PER_THREAD>
inline __device__ bool pollOneshotRemoteRank(
    PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token,
    int tokenDim, int packedIdx) {
  if constexpr (Rank == LocalRank) {
    return true;
  } else {
    auto loaded = loadPackedVolatile<PackedType>(
        &stagePtrLocal[token * tokenDim * WorldSize + Rank * tokenDim +
                       packedIdx * kELTS_PER_THREAD]);
    remoteValues[Rank].packed = loaded.packed;
    return !isLamportDirty(loaded);
  }
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType,
          int kELTS_PER_THREAD, int... Ranks>
inline __device__ bool pollOneshotRemoteRanks(
    PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token,
    int tokenDim, int packedIdx, std::integer_sequence<int, Ranks...>) {
  bool valid = true;
  ((valid &= pollOneshotRemoteRank<Ranks, WorldSize, LocalRank, T, PackedType,
                                   kELTS_PER_THREAD>(
        remoteValues, stagePtrLocal, token, tokenDim, packedIdx)),
   ...);
  return valid;
}

template <typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulatePacked(float (&accum)[kELTS_PER_THREAD],
                                        PackedVec<PackedType, T> const& value) {
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    accum[i] += toFloat<T>(value.elements[i]);
  }
}

template <int Rank, uint8_t LocalRank, typename T, typename PackedType,
          int kELTS_PER_THREAD>
inline __device__ void accumulateOneshotRank(
    float (&accum)[kELTS_PER_THREAD],
    PackedVec<PackedType, T> const* remoteValues,
    PackedVec<PackedType, T> const& localValue) {
  if constexpr (Rank == LocalRank) {
    accumulatePacked<T, PackedType, kELTS_PER_THREAD>(accum, localValue);
  } else {
    accumulatePacked<T, PackedType, kELTS_PER_THREAD>(accum,
                                                      remoteValues[Rank]);
  }
}

template <uint8_t LocalRank, typename T, typename PackedType,
          int kELTS_PER_THREAD, int... Ranks>
inline __device__ void accumulateOneshotRanks(
    float (&accum)[kELTS_PER_THREAD],
    PackedVec<PackedType, T> const* remoteValues,
    PackedVec<PackedType, T> const& localValue,
    std::integer_sequence<int, Ranks...>) {
  (accumulateOneshotRank<Ranks, LocalRank, T, PackedType, kELTS_PER_THREAD>(
       accum, remoteValues, localValue),
   ...);
}

// Polls every peer each iteration rather than waiting on them in turn.
template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType,
          int kELTS_PER_THREAD>
inline __device__ void waitOneshotRemoteRanks(
    PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token,
    int tokenDim, int packedIdx) {
  static_assert(LocalRank < WorldSize);
  while (1) {
    bool const valid = pollOneshotRemoteRanks<WorldSize, LocalRank, T,
                                              PackedType, kELTS_PER_THREAD>(
        remoteValues, stagePtrLocal, token, tokenDim, packedIdx,
        std::make_integer_sequence<int, WorldSize>{});
    if (valid) {
      break;
    }
  }
}

// Every rank sums in rank order, so all ranks produce identical results.
template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType,
          int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceOneshotDeterministic(
    PackedVec<PackedType, T> const* remoteValues,
    PackedVec<PackedType, T> const& localValue) {
  static_assert(LocalRank < WorldSize);
  float accum[kELTS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    accum[i] = 0.f;
  }
  accumulateOneshotRanks<LocalRank, T, PackedType, kELTS_PER_THREAD>(
      accum, remoteValues, localValue,
      std::make_integer_sequence<int, WorldSize>{});

  PackedVec<PackedType, T> packedAccum;
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    packedAccum.elements[i] = fromFloat<T>(accum[i]);
  }
  return packedAccum;
}

template <uint8_t WorldSize, int kRankChunk, typename T, typename PackedType,
          int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceLamportRanksChunked(
    T* buffer, int token, int tokenDim, int tokenOffset) {
  static_assert(kRankChunk > 0, "kRankChunk must be positive");
  static_assert(WorldSize % kRankChunk == 0,
                "WorldSize must be divisible by kRankChunk");
  // Chunking keeps large world sizes from holding every payload live.
  float accum[kELTS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    accum[i] = 0.f;
  }

#pragma unroll 1
  for (int rankBase = 0; rankBase < WorldSize; rankBase += kRankChunk) {
    float chunkAccum[kELTS_PER_THREAD];
    while (1) {
      bool valid = true;
#pragma unroll
      for (int i = 0; i < kELTS_PER_THREAD; i++) {
        chunkAccum[i] = 0.f;
      }
#pragma unroll
      for (int rr = 0; rr < kRankChunk; rr++) {
        int const r = rankBase + rr;
        auto loaded = loadPackedVolatile<PackedType>(
            &buffer[token * tokenDim * WorldSize + r * tokenDim + tokenOffset]);
        PackedVec<PackedType, T> value;
        value.packed = loaded.packed;
        valid &= !isLamportDirty(loaded);
        accumulatePacked<T, PackedType, kELTS_PER_THREAD>(chunkAccum, value);
      }
      if (valid) {
        break;
      }
    }
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      accum[i] += chunkAccum[i];
    }
  }

  PackedVec<PackedType, T> packedAccum;
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    packedAccum.elements[i] = fromFloat<T>(accum[i]);
  }
  return packedAccum;
}

uint32_t constexpr kWARP_SIZE = 32U;
uint32_t constexpr kLOG2_WARP_SIZE = 5U;
uint32_t constexpr kLANE_ID_MASK = 0x1f;
uint32_t constexpr kFINAL_MASK = 0xffffffff;

template <typename T>
inline __device__ T warpReduceSumFull(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(kFINAL_MASK, val, mask, kWARP_SIZE);
  }
  return val;
}

template <typename T>
inline __device__ T warpReduceSumPartial(T val) {
  int laneId = threadIdx.x & kLANE_ID_MASK;
  // Only the last warp calls this.
  int warpSize = blockDim.x - (threadIdx.x & ~(kWARP_SIZE - 1));
  unsigned int active_mask = (1U << warpSize) - 1;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    int targetLane = laneId ^ mask;
    auto tmp = __shfl_xor_sync(active_mask, val, mask, kWARP_SIZE);
    val += targetLane < warpSize ? tmp : 0;
  }
  return val;
}

// SYNC shares the sum with every thread; otherwise only thread 0 has it.
template <typename T, bool SYNC = false>
inline __device__ T blockReduceSumPartial(T val) {
  __shared__ T smem[kWARP_SIZE];
  int laneId = threadIdx.x & kLANE_ID_MASK;
  int warpId = threadIdx.x >> kLOG2_WARP_SIZE;
  int warpNum = (blockDim.x + kWARP_SIZE - 1) >> kLOG2_WARP_SIZE;

  val = (warpId == warpNum - 1) ? warpReduceSumPartial(val)
                                : warpReduceSumFull(val);
  if (laneId == 0) {
    smem[warpId] = val;
  }
  __syncthreads();

  if (warpId == 0) {
    val = (laneId < warpNum) ? smem[laneId] : (T)0.f;
    val = (warpNum == 1) ? warpReduceSumPartial(val) : warpReduceSumFull(val);

    if constexpr (SYNC) {
      if (laneId == 0) {
        smem[warpId] = val;
      }
    }
  }
  if constexpr (SYNC) {
    __syncthreads();
    val = smem[0];
  }
  return val;
}

template <typename T>
inline __device__ T blockReduceSumFull(T val) {
  __shared__ T smem[kWARP_SIZE];
  int lane_id = threadIdx.x & kLANE_ID_MASK;
  int warp_id = threadIdx.x >> kLOG2_WARP_SIZE;
  int warp_num = blockDim.x >> kLOG2_WARP_SIZE;

  val = warpReduceSumFull(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();

  val = (lane_id < warp_num) ? smem[lane_id] : (T)0.f;
  val = warpReduceSumFull(val);

  return val;
}

template <typename T, bool SYNC = false>
inline __device__ T blockReduceSum(T val) {
  bool hasPartialWarp = (blockDim.x & kLANE_ID_MASK) != 0;
  if (hasPartialWarp) {
    return blockReduceSumPartial<T, SYNC>(val);
  } else {
    return blockReduceSumFull<T>(val);
  }
}

// Grid configuration for the fused one-shot and RMSNorm kernels.
// Returns (block_size, cluster_size, loads_per_thread).
inline std::tuple<int, int, int> adjustGridConfig(int numTokens, int dim,
                                                  int eltsPerThread,
                                                  bool useCluster,
                                                  int smCount) {
  int clusterSize = useCluster ? kMaxClusterSize : 1;
  int blockSize = 128;
  int threadsNeeded = ceil_div(dim, eltsPerThread);
  int loadsPerThread = 1;

  blockSize = ceil_div(threadsNeeded, clusterSize);
  if (useCluster) {
    // Shrink the cluster until the row partitions cleanly across CTAs.
    while (threadsNeeded % clusterSize != 0 && clusterSize > 1) {
      clusterSize /= 2;
    }
    int const maxDivisibleClusterSize = clusterSize;
    blockSize = ceil_div(threadsNeeded, clusterSize);
    // Trade cluster width for CTAs of at least 128 threads.
    while (blockSize < 128 && clusterSize >= 2) {
      blockSize *= 2;
      clusterSize /= 2;
    }
    // Enough tokens already fill the GPU: prefer wider CTAs.
    while (numTokens * clusterSize > smCount && clusterSize > 1 &&
           blockSize <= 512) {
      blockSize *= 2;
      clusterSize /= 2;
    }
    // The grid still underfills the GPU: restore cluster width.
    while (clusterSize < maxDivisibleClusterSize) {
      int const candidateClusterSize = clusterSize * 2;
      int const candidateBlockSize =
          ceil_div(threadsNeeded, candidateClusterSize);
      if (candidateBlockSize < 64 ||
          numTokens * candidateClusterSize > smCount) {
        break;
      }
      clusterSize = candidateClusterSize;
      blockSize = candidateBlockSize;
    }
  }
  // Keep very wide rows within 1024 threads per CTA.
  while (blockSize > 1024) {
    if (useCluster && clusterSize < kMaxClusterSize) {
      clusterSize = clusterSize << 1;
    } else if (loadsPerThread < 8) {
      loadsPerThread += 1;
    } else {
      break;
    }
    blockSize = ceil_div(threadsNeeded, clusterSize * loadsPerThread);
  }
  return {blockSize, clusterSize, loadsPerThread};
}

}  // namespace utils

enum MNNVLTwoShotStage : uint8_t {
  SCATTER = 0,
  BROADCAST = 1,
  NUM_STAGES = 2,
};

// Scatters token rows to their owner rank, reduces owned rows in rank order,
// and broadcasts them through multicast. WaitForResults also gathers the
// reduced rows into outputPtr; otherwise a follow-up kernel consumes them.
template <uint8_t WorldSize, typename T, bool WaitForResults = true,
          typename PackedType = float4>
__global__ __launch_bounds__(128) void twoshotAllreduceKernel(
    AllReduceKernelParams<T> params) {
  using namespace utils;
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr uint32_t kELT_SIZE = sizeof(T);

  int const packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int const token = blockIdx.x;
  int const tokenOffset = packedIdx * kELTS_PER_THREAD;
  int const threadOffset = token * params.tokenDim + tokenOffset;
  bool const inBounds = tokenOffset < params.tokenDim;
  int const destRank = token % WorldSize;
  int const destTokenOffset = token / WorldSize;

  cudaGridDependencySynchronize();

  LamportFlags<PackedType> flag(params.bufferFlags,
                                MNNVLTwoShotStage::NUM_STAGES);
  T* scatterBufLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(
      params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER));
  T* scatterBufDest = reinterpret_cast<T*>(flag.getCurLamportBuf(
      params.inputPtrs[destRank], MNNVLTwoShotStage::SCATTER));
  T* broadcastBufW = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.mcastPtr, MNNVLTwoShotStage::BROADCAST));
  T* broadcastBufR = reinterpret_cast<T*>(flag.getCurLamportBuf(
      params.inputPtrs[params.rank], MNNVLTwoShotStage::BROADCAST));

  PackedVec<PackedType, T> val;
  if (inBounds) {
    val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
    sanitizeLamportPayload<PackedType, T>(val);

    reinterpret_cast<PackedType*>(
        &scatterBufDest[destTokenOffset * params.tokenDim * WorldSize +
                        params.rank * params.tokenDim])[packedIdx] = val.packed;
  }

  cudaTriggerProgrammaticLaunchCompletion();
  flag.clearDirtyLamportBuf(params.inputPtrs[params.rank],
                            MNNVLTwoShotStage::SCATTER);

  if (inBounds && destRank == params.rank) {
    int const localToken = token / WorldSize;
    constexpr int kRankChunk = WorldSize < 16 ? WorldSize : 16;
    PackedVec<PackedType, T> packedAccum =
        reduceLamportRanksChunked<WorldSize, kRankChunk, T, PackedType,
                                  kELTS_PER_THREAD>(
            scatterBufLocal, localToken, params.tokenDim, tokenOffset);
    // A reduced value can round to the sentinel.
    sanitizeLamportPayload<PackedType, T>(packedAccum);
    reinterpret_cast<PackedType*>(
        &broadcastBufW[token * params.tokenDim])[packedIdx] =
        packedAccum.packed;
  }

  flag.clearDirtyLamportBuf(params.inputPtrs[params.rank],
                            MNNVLTwoShotStage::BROADCAST);

  if constexpr (WaitForResults) {
    // Out-of-bounds threads still arrive so waitAndUpdate sees every CTA.
    flag.ctaArrive();

    if (inBounds) {
      auto loaded =
          loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
      while (isLamportDirty(loaded)) {
        loaded = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
      }
      reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] =
          loaded.packed;
    }

    flag.waitAndUpdate(
        {static_cast<uint32_t>(round_up(params.numTokens, WorldSize) *
                               params.tokenDim * kELT_SIZE),
         static_cast<uint32_t>(params.numTokens * params.tokenDim * kELT_SIZE),
         0, 0});
  }
}

}  // namespace mnnvl
}  // namespace vllm
