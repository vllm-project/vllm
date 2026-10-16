// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Communication helpers and launch geometry from NVIDIA/FlashInfer's MNNVL
// all-reduce (Apache-2.0); Beam supplies the normalization epilogue.

#include "flashinfer/comm/trtllm_mnnvl_allreduce.cuh"
#include "tvm_ffi_utils.h"

namespace beam {
namespace ar = flashinfer::trtllm_mnnvl_allreduce;
namespace cg = cooperative_groups;
using T = __nv_bfloat16;
using Vec = ar::utils::PackedVec<float4, T>;

struct Params {
  ar::AllReduceKernelParams<T> ar;
  T const* nextWeight;
  float nextEps;
  float xScale;
  float rScale;
  bool postNorm;
  bool nextNorm;
  bool scaled;
  T const* experts;
  void const* expertWeights;
  int const* permutation;
  int topK;
  bool floatWeights;
  float routedScale;
};

template <bool Finalize>
__device__ Vec loadInput(Params const& p, int token, int packedIdx) {
  Vec value;
  if constexpr (!Finalize) {
    value.packed = ar::utils::loadPacked<float4>(
        p.ar.shardPtr + int64_t(token) * p.ar.tokenDim + packedIdx * 8);
    return value;
  }
  float sum[8]{};
  for (int base = 0; base < p.topK; base += 8) {
    int rows[8];
    float weights[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      int index = token * p.topK + base + k;
      rows[k] = base + k < p.topK ? p.permutation[index] : -1;
      weights[k] = rows[k] < 0 ? 0.f
                   : p.floatWeights
                       ? static_cast<float const*>(p.expertWeights)[index]
                       : ar::utils::toFloat<T>(
                             static_cast<T const*>(p.expertWeights)[index]);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      Vec expert{};
      if (rows[k] >= 0)
        expert.packed = ar::utils::loadPacked<float4>(
            p.experts + int64_t(rows[k]) * p.ar.tokenDim + packedIdx * 8);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        sum[i] = __fadd_rn(
            sum[i],
            __fmul_rn(ar::utils::toFloat<T>(expert.elements[i]), weights[k]));
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    T rounded = ar::utils::fromFloat<T>(sum[i]);
    value.elements[i] = ar::utils::fromFloat<T>(
        __fmul_rn(ar::utils::toFloat<T>(rounded), p.routedScale));
  }
  return value;
}

template <bool UseCGA, int Stage>
__device__ float invRms(Vec const& value, int dim, float eps) {
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float v = ar::utils::toFloat<T>(value.elements[i]);
    sum = __fadd_rn(sum, __fmul_rn(v, v));
  }
  sum = ar::utils::blockReduceSum<float, true>(sum);
  if constexpr (UseCGA) {
    auto cluster = cg::this_cluster();
    __shared__ float partial[8];
    if (threadIdx.x < cluster.num_blocks()) {
      cluster.map_shared_rank(partial, threadIdx.x)[cluster.block_rank()] = sum;
    }
    cluster.sync();
    sum = 0.f;
    for (int i = 0; i < cluster.num_blocks(); ++i)
      sum = __fadd_rn(sum, partial[i]);
  }
  __syncthreads();
  return rsqrtf(sum / dim + eps);
}

template <bool UseCGA, bool Preloaded = false>
__device__ void epilogue(Params const& p, Vec value, int packedIdx, int offset,
                         Vec postWeight = {}, Vec residual = {},
                         Vec nextWeight = {}) {
  if (p.postNorm) {
    float rstd = invRms<UseCGA, 0>(value, p.ar.tokenDim, p.ar.epsilon);
    Vec weight = postWeight;
    if constexpr (!Preloaded) {
      if (p.ar.gammaPtr)
        weight.packed =
            ar::utils::loadPacked<float4>(p.ar.gammaPtr + packedIdx * 8);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = __fmul_rn(ar::utils::toFloat<T>(value.elements[i]), rstd);
      if (p.ar.gammaPtr)
        v = __fmul_rn(v, ar::utils::toFloat<T>(weight.elements[i]));
      value.elements[i] = ar::utils::fromFloat<T>(v);
    }
  }
  if constexpr (!Preloaded)
    residual.packed =
        ar::utils::loadPacked<float4>(p.ar.residualInPtr + offset);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    T x = value.elements[i], r = residual.elements[i];
    if (p.scaled) {
      x = ar::utils::fromFloat<T>(
          __fmul_rn(ar::utils::toFloat<T>(x), p.xScale));
      r = ar::utils::fromFloat<T>(
          __fmul_rn(ar::utils::toFloat<T>(r), p.rScale));
    }
    value.elements[i] = ar::utils::fromFloat<T>(
        __fadd_rn(ar::utils::toFloat<T>(x), ar::utils::toFloat<T>(r)));
  }
  *reinterpret_cast<float4*>(p.ar.prenormedPtr + offset) = value.packed;
  if (p.nextNorm) {
    float rstd = invRms<UseCGA, 1>(value, p.ar.tokenDim, p.nextEps);
    Vec weight = nextWeight;
    if constexpr (!Preloaded) {
      if (p.nextWeight)
        weight.packed =
            ar::utils::loadPacked<float4>(p.nextWeight + packedIdx * 8);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = __fmul_rn(ar::utils::toFloat<T>(value.elements[i]), rstd);
      if (p.nextWeight)
        v = __fmul_rn(v, ar::utils::toFloat<T>(weight.elements[i]));
      value.elements[i] = ar::utils::fromFloat<T>(v);
    }
    *reinterpret_cast<float4*>(p.ar.outputPtr + offset) = value.packed;
  }
}

template <int World, int Rank>
__device__ Vec reduceLocal(Vec value, T* buffer, int token, int dim,
                           int packedIdx) {
  Vec remote[World];
  ar::utils::waitOneshotRemoteRanks<World, Rank, T, float4, 8>(
      remote, buffer, token, dim, packedIdx);
  return ar::utils::reduceOneshotDeterministic<World, Rank, T, float4, 8>(
      remote, value);
}

template <int World, bool Finalize = false, bool SmallBatch = false>
__global__ void __launch_bounds__(1024) oneshot(Params p) {
  int packedIdx = threadIdx.x;
  if constexpr (!SmallBatch) packedIdx = cg::this_cluster().thread_rank();
  int token = blockIdx.x;
  int offset = token * p.ar.tokenDim + packedIdx * 8;
  cudaGridDependencySynchronize();
  ar::utils::LamportFlags<float4, !SmallBatch> flag(p.ar.bufferFlags, 1);
  auto multicast = static_cast<T*>(flag.getCurLamportBuf(p.ar.mcastPtr, 0));
  auto local =
      static_cast<T*>(flag.getCurLamportBuf(p.ar.inputPtrs[p.ar.rank], 0));
  Vec value = loadInput<Finalize>(p, token, packedIdx);
  ar::utils::sanitizeLamportPayload<float4, T>(value);
  reinterpret_cast<float4*>(multicast + token * p.ar.tokenDim * World +
                            p.ar.rank * p.ar.tokenDim)[packedIdx] =
      value.packed;
  Vec postWeight{}, residual{}, nextWeight{};
  if constexpr (SmallBatch) {
    // Overlap epilogue inputs with the peer payloads in flight.
    if (p.postNorm && p.ar.gammaPtr)
      postWeight.packed =
          ar::utils::loadPackedVolatile<float4>(p.ar.gammaPtr + packedIdx * 8)
              .packed;
    residual.packed =
        ar::utils::loadPackedVolatile<float4>(p.ar.residualInPtr + offset)
            .packed;
    if (p.nextNorm && p.nextWeight)
      nextWeight.packed =
          ar::utils::loadPackedVolatile<float4>(p.nextWeight + packedIdx * 8)
              .packed;
  }
  flag.ctaArrive();
  if constexpr (SmallBatch) {
    // This workspace uses at most two stages. Constant indices avoid a
    // dynamically indexed array spilling the Lamport state to local memory.
    flag.clearDirtyLamportBuf(p.ar.inputPtrs[p.ar.rank], 0);
    flag.clearDirtyLamportBuf(p.ar.inputPtrs[p.ar.rank], 1);
  } else {
    flag.clearDirtyLamportBuf(p.ar.inputPtrs[p.ar.rank], -1);
  }
#define REDUCE(R)                                                       \
  case R:                                                               \
    if constexpr (World > R)                                            \
      value = reduceLocal<World, R>(value, local, token, p.ar.tokenDim, \
                                    packedIdx);                         \
    break
  switch (p.ar.rank) {
    REDUCE(0);
    REDUCE(1);
    REDUCE(2);
    REDUCE(3);
    REDUCE(4);
    REDUCE(5);
    REDUCE(6);
    REDUCE(7);
  }
#undef REDUCE
  cudaTriggerProgrammaticLaunchCompletion();
  epilogue<!SmallBatch, SmallBatch>(p, value, packedIdx, offset, postWeight,
                                    residual, nextWeight);
  flag.waitAndUpdate({static_cast<uint32_t>(p.ar.numTokens * p.ar.tokenDim *
                                            World * sizeof(T)),
                      0, 0, 0});
}

template <int World>
__global__ void __launch_bounds__(128) finalizeScatter(Params p) {
  int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int token = blockIdx.x;
  int destRank = token % World;
  int localToken = token / World;
  cudaGridDependencySynchronize();
  ar::utils::LamportFlags<float4> flag(p.ar.bufferFlags, 2);
  auto local =
      static_cast<T*>(flag.getCurLamportBuf(p.ar.inputPtrs[p.ar.rank], 0));
  auto dest =
      static_cast<T*>(flag.getCurLamportBuf(p.ar.inputPtrs[destRank], 0));
  auto broadcast = static_cast<T*>(flag.getCurLamportBuf(p.ar.mcastPtr, 1));
  if (packedIdx * 8 < p.ar.tokenDim) {
    Vec value = loadInput<true>(p, token, packedIdx);
    ar::utils::sanitizeLamportPayload<float4, T>(value);
    reinterpret_cast<float4*>(dest + localToken * p.ar.tokenDim * World +
                              p.ar.rank * p.ar.tokenDim)[packedIdx] =
        value.packed;
  }
  cudaTriggerProgrammaticLaunchCompletion();
  flag.clearDirtyLamportBuf(p.ar.inputPtrs[p.ar.rank], 0);
  if (packedIdx * 8 < p.ar.tokenDim && destRank == p.ar.rank) {
    Vec value =
        ar::utils::reduceLamportRanksChunked<World, World, T, float4, 8>(
            local, localToken, p.ar.tokenDim, packedIdx * 8);
    ar::utils::sanitizeLamportPayload<float4, T>(value);
    reinterpret_cast<float4*>(broadcast + token * p.ar.tokenDim)[packedIdx] =
        value.packed;
  }
  flag.clearDirtyLamportBuf(p.ar.inputPtrs[p.ar.rank], 1);
}

template <bool UseCGA>
__global__ void __launch_bounds__(1024) finishTwoshot(Params p) {
  int packedIdx = threadIdx.x;
  if constexpr (UseCGA) packedIdx = cg::this_cluster().thread_rank();
  int offset = blockIdx.x * p.ar.tokenDim + packedIdx * 8;
  cudaTriggerProgrammaticLaunchCompletion();
  ar::utils::LamportFlags<float4, UseCGA> flag(p.ar.bufferFlags, 2);
  auto input = static_cast<T*>(flag.getCurLamportBuf(p.ar.localBufferPtr, 1));
  flag.ctaArrive();
  auto loaded = ar::utils::loadPackedVolatile<float4>(input + offset);
  while (ar::utils::isLamportDirty(loaded)) {
    loaded = ar::utils::loadPackedVolatile<float4>(input + offset);
  }
  Vec value{.packed = loaded.packed};
  epilogue<UseCGA>(p, value, packedIdx, offset);
  cudaGridDependencySynchronize();
  flag.waitAndUpdate(
      {static_cast<uint32_t>(flashinfer::round_up(p.ar.numTokens, p.ar.nRanks) *
                             p.ar.tokenDim * sizeof(T)),
       static_cast<uint32_t>(p.ar.numTokens * p.ar.tokenDim * sizeof(T)), 0,
       0});
}

__device__ Vec finalizeGatherRow(Params const& p, int token, int packedIdx) {
  int64_t inputOffset = int64_t(token) * p.topK * p.ar.tokenDim + packedIdx * 8;
  auto weights = static_cast<float const*>(p.expertWeights);
  Vec expert;
  expert.packed = ar::utils::loadPacked<float4>(p.experts + inputOffset);
  float weight = weights[int64_t(token) * p.topK];
  float sum[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    sum[i] = __fmul_rn(ar::utils::toFloat<T>(expert.elements[i]), weight);
  for (int k = 1; k < p.topK; ++k) {
    expert.packed = ar::utils::loadPacked<float4>(p.experts + inputOffset +
                                                  int64_t(k) * p.ar.tokenDim);
    weight = weights[int64_t(token) * p.topK + k];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sum[i] =
          __fmaf_rn(ar::utils::toFloat<T>(expert.elements[i]), weight, sum[i]);
  }
  Vec value;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    T reduced = ar::utils::fromFloat<T>(sum[i]);
    value.elements[i] = ar::utils::fromFloat<T>(
        __fmul_rn(ar::utils::toFloat<T>(reduced), p.routedScale));
  }
  if (p.postNorm) {
    float rstd = invRms<false, 0>(value, p.ar.tokenDim, p.ar.epsilon);
    Vec weight;
    if (p.ar.gammaPtr)
      weight.packed =
          ar::utils::loadPacked<float4>(p.ar.gammaPtr + packedIdx * 8);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float x = __fmul_rn(ar::utils::toFloat<T>(value.elements[i]), rstd);
      if (p.ar.gammaPtr)
        x = __fmul_rn(x, ar::utils::toFloat<T>(weight.elements[i]));
      value.elements[i] = ar::utils::fromFloat<T>(x);
    }
  }
  ar::utils::sanitizeLamportPayload<float4, T>(value);
  return value;
}

__device__ Vec waitGatherRow(T const* buffer, int64_t offset) {
  auto loaded = ar::utils::loadPackedVolatile<float4>(buffer + offset);
  while (ar::utils::isLamportDirty(loaded))
    loaded = ar::utils::loadPackedVolatile<float4>(buffer + offset);
  return Vec{.packed = loaded.packed};
}

template <int World>
__global__ void __launch_bounds__(1024) finalizeGatherOneshot(Params p) {
  int localTokens = flashinfer::ceil_div(p.ar.numTokens, World);
  int localToken = blockIdx.x;
  int token = p.ar.rank * localTokens + localToken;
  int packedIdx = threadIdx.x;
  cudaGridDependencySynchronize();
  ar::utils::LamportFlags<float4> flag(p.ar.bufferFlags, 1);
  auto broadcast = static_cast<T*>(flag.getCurLamportBuf(p.ar.mcastPtr));
  auto local = static_cast<T*>(flag.getCurLamportBuf(p.ar.localBufferPtr));
  Vec ownValue{};
  if (token < p.ar.numTokens)
    ownValue = finalizeGatherRow(p, localToken, packedIdx);
  // Empty ranks publish padding so no peer can outrun their readers.
  *reinterpret_cast<float4*>(broadcast + int64_t(token) * p.ar.tokenDim +
                             packedIdx * 8) = ownValue.packed;
  flag.ctaArrive();
  flag.clearDirtyLamportBuf(p.ar.localBufferPtr, -1);
  p.postNorm = false;
#pragma unroll
  for (int rank = 0; rank < World; ++rank) {
    token = rank * localTokens + localToken;
    int offset = token * p.ar.tokenDim + packedIdx * 8;
    Vec value = rank == p.ar.rank ? ownValue : waitGatherRow(local, offset);
    if (token < p.ar.numTokens) {
      epilogue<false>(p, value, packedIdx, offset);
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
  flag.waitAndUpdate({static_cast<uint32_t>(int64_t(localTokens) * World *
                                            p.ar.tokenDim * sizeof(T)),
                      0, 0, 0});
}

__global__ void __launch_bounds__(1024) finalizeGatherPublish(Params p) {
  int localTokens = flashinfer::ceil_div(p.ar.numTokens, p.ar.nRanks);
  int localToken = blockIdx.x;
  int token = p.ar.rank * localTokens + localToken;
  int packedIdx = threadIdx.x;
  cudaGridDependencySynchronize();
  ar::utils::LamportFlags<float4> flag(p.ar.bufferFlags, 1);
  auto broadcast = static_cast<T*>(flag.getCurLamportBuf(p.ar.mcastPtr));
  Vec value{};
  if (token < p.ar.numTokens)
    value = finalizeGatherRow(p, localToken, packedIdx);
  *reinterpret_cast<float4*>(broadcast + int64_t(token) * p.ar.tokenDim +
                             packedIdx * 8) = value.packed;
  cudaTriggerProgrammaticLaunchCompletion();
  flag.clearDirtyLamportBuf(p.ar.localBufferPtr, -1);
}

__global__ void __launch_bounds__(1024) finishGatherTwoshot(Params p) {
  int packedIdx = threadIdx.x;
  int offset = blockIdx.x * p.ar.tokenDim + packedIdx * 8;
  cudaTriggerProgrammaticLaunchCompletion();
  ar::utils::LamportFlags<float4> flag(p.ar.bufferFlags, 1);
  auto local = static_cast<T*>(flag.getCurLamportBuf(p.ar.localBufferPtr));
  flag.ctaArrive();
  Vec value = waitGatherRow(local, offset);
  p.postNorm = false;
  if (blockIdx.x < p.ar.numTokens) epilogue<false>(p, value, packedIdx, offset);
  // The producer must finish clearing the previous generation before reuse.
  cudaGridDependencySynchronize();
  flag.waitAndUpdate(
      {static_cast<uint32_t>(int64_t(gridDim.x) * p.ar.tokenDim * sizeof(T)), 0,
       0, 0});
}

void checkCuda(cudaError_t status) {
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

void launch(TensorView input, TensorView residual,
            tvm::ffi::Optional<TensorView> postWeight,
            tvm::ffi::Optional<TensorView> nextWeight, TensorView hidden,
            tvm::ffi::Optional<TensorView> normalized, int64_t pointers,
            TensorView flags, int64_t multicast, int64_t local, int64_t world,
            int64_t rank, double postEps, double nextEps, double xScale,
            double rScale, bool postNorm, bool nextNorm, bool scaled,
            bool oneShot, tvm::ffi::Optional<TensorView> experts,
            tvm::ffi::Optional<TensorView> expertWeights,
            tvm::ffi::Optional<TensorView> permutation, double routedScale,
            bool smallBatch) {
  ffi::CUDADeviceGuard guard(input.device().device_id);
  auto stream = get_stream(input.device());
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(input.dtype()), bfloat16_code);
  TVM_FFI_ICHECK_EQ(input.size(1) % 8, 0);
  Params p{};
  p.ar.outputPtr = normalized.has_value()
                       ? static_cast<T*>(normalized.value().data_ptr())
                       : nullptr;
  p.ar.prenormedPtr = static_cast<T*>(hidden.data_ptr());
  p.ar.shardPtr = static_cast<T const*>(input.data_ptr());
  p.ar.residualInPtr = static_cast<T const*>(residual.data_ptr());
  p.ar.gammaPtr = postWeight.has_value()
                      ? static_cast<T const*>(postWeight.value().data_ptr())
                      : nullptr;
  p.nextWeight = nextWeight.has_value()
                     ? static_cast<T const*>(nextWeight.value().data_ptr())
                     : nullptr;
  p.ar.inputPtrs = reinterpret_cast<T**>(pointers);
  p.ar.mcastPtr = reinterpret_cast<T*>(multicast);
  p.ar.localBufferPtr = reinterpret_cast<T*>(local);
  p.ar.bufferFlags = static_cast<uint32_t*>(flags.data_ptr());
  p.ar.nRanks = world;
  p.ar.rank = rank;
  p.ar.numTokens = input.size(0);
  p.ar.tokenDim = input.size(1);
  p.ar.epsilon = postEps;
  p.nextEps = nextEps;
  p.xScale = xScale;
  p.rScale = rScale;
  p.postNorm = postNorm;
  p.nextNorm = nextNorm;
  p.scaled = scaled;
  if (experts.has_value()) {
    TVM_FFI_ICHECK(expertWeights.has_value() && permutation.has_value());
    p.experts = static_cast<T const*>(experts.value().data_ptr());
    p.expertWeights = expertWeights.value().data_ptr();
    p.permutation = static_cast<int const*>(permutation.value().data_ptr());
    p.topK = permutation.value().size(1);
    p.floatWeights =
        encode_dlpack_dtype(expertWeights.value().dtype()) == float32_code;
    p.routedScale = routedScale;
  }
  auto [threads, clusters, loads] =
      ar::utils::adjustGridConfig(p.ar.numTokens, p.ar.tokenDim, 8, true);
  TVM_FFI_ICHECK(!smallBatch || (oneShot && !p.experts));
  if (smallBatch) {
    clusters = 1;
    threads = p.ar.tokenDim / 8;
  }
  TVM_FFI_ICHECK_EQ(loads, 1);
  TVM_FFI_ICHECK_EQ(threads * clusters * 8, p.ar.tokenDim);
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim = {1, static_cast<unsigned>(clusters), 1};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(p.ar.numTokens, clusters);
  config.blockDim = dim3(threads);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = smallBatch ? 1 : 2;
  if (oneShot) {
#define LAUNCH(W)                                                          \
  case W:                                                                  \
    if (smallBatch)                                                        \
      checkCuda(cudaLaunchKernelEx(&config, &oneshot<W, false, true>, p)); \
    else if (p.experts)                                                    \
      checkCuda(cudaLaunchKernelEx(&config, &oneshot<W, true>, p));        \
    else                                                                   \
      checkCuda(cudaLaunchKernelEx(&config, &oneshot<W>, p));              \
    break
    switch (world) {
      LAUNCH(2);
      LAUNCH(4);
      LAUNCH(8);
      default:
        TVM_FFI_LOG_AND_THROW(ValueError) << "Beam supports TP2/4/8";
    }
#undef LAUNCH
  } else {
    cudaLaunchConfig_t scatter{};
    scatter.gridDim =
        dim3(p.ar.numTokens, flashinfer::ceil_div(p.ar.tokenDim, 1024));
    scatter.blockDim = dim3(128);
    scatter.stream = stream;
    scatter.attrs = attrs;
    scatter.numAttrs = 1;
#define LAUNCH(W)                                                      \
  case W:                                                              \
    if (p.experts)                                                     \
      checkCuda(cudaLaunchKernelEx(&scatter, &finalizeScatter<W>, p)); \
    else                                                               \
      checkCuda(cudaLaunchKernelEx(                                    \
          &scatter, &ar::twoshotAllreduceKernel<W, T, false>, p.ar));  \
    break
    switch (world) {
      LAUNCH(2);
      LAUNCH(4);
      LAUNCH(8);
      default:
        TVM_FFI_LOG_AND_THROW(ValueError) << "Beam supports TP2/4/8";
    }
#undef LAUNCH
    if (clusters > 1) {
      checkCuda(cudaLaunchKernelEx(&config, &finishTwoshot<true>, p));
    } else {
      config.numAttrs = 1;
      checkCuda(cudaLaunchKernelEx(&config, &finishTwoshot<false>, p));
    }
  }
}

void launchGather(TensorView experts, TensorView expertWeights,
                  TensorView residual,
                  tvm::ffi::Optional<TensorView> postWeight,
                  tvm::ffi::Optional<TensorView> nextWeight, TensorView hidden,
                  tvm::ffi::Optional<TensorView> normalized, int64_t pointers,
                  TensorView flags, int64_t multicast, int64_t local,
                  int64_t world, int64_t rank, double postEps, double nextEps,
                  double xScale, double rScale, bool postNorm, bool nextNorm,
                  bool scaled, bool oneShot, double routedScale) {
  ffi::CUDADeviceGuard guard(residual.device().device_id);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(experts.dtype()), bfloat16_code);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(expertWeights.dtype()), float32_code);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(residual.dtype()), bfloat16_code);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(hidden.dtype()), bfloat16_code);
  TVM_FFI_ICHECK(world == 2 || world == 4 || world == 8);
  TVM_FFI_ICHECK(rank >= 0 && rank < world);
  int64_t tokens = residual.size(0);
  int64_t dim = residual.size(1);
  int64_t localTokens = flashinfer::ceil_div(tokens, world);
  TVM_FFI_ICHECK(dim >= 1024 && dim <= 8192 && dim % 1024 == 0);
  TVM_FFI_ICHECK(experts.size(0) >= localTokens);
  TVM_FFI_ICHECK(expertWeights.size(0) >= localTokens);
  TVM_FFI_ICHECK_EQ(experts.size(2), dim);
  TVM_FFI_ICHECK_EQ(expertWeights.size(1), experts.size(1));
  TVM_FFI_ICHECK(experts.size(1) >= 1 && experts.size(1) <= 32);
  TVM_FFI_ICHECK_EQ(hidden.size(0), tokens);
  TVM_FFI_ICHECK_EQ(hidden.size(1), dim);
  TVM_FFI_ICHECK(!nextNorm || normalized.has_value());
  TVM_FFI_ICHECK(localTokens * world * dim <= INT32_MAX);
  if (!tokens) return;

  Params p{};
  p.ar.outputPtr = normalized.has_value()
                       ? static_cast<T*>(normalized.value().data_ptr())
                       : nullptr;
  p.ar.prenormedPtr = static_cast<T*>(hidden.data_ptr());
  p.ar.residualInPtr = static_cast<T const*>(residual.data_ptr());
  p.ar.gammaPtr = postWeight.has_value()
                      ? static_cast<T const*>(postWeight.value().data_ptr())
                      : nullptr;
  p.nextWeight = nextWeight.has_value()
                     ? static_cast<T const*>(nextWeight.value().data_ptr())
                     : nullptr;
  p.ar.inputPtrs = reinterpret_cast<T**>(pointers);
  p.ar.mcastPtr = reinterpret_cast<T*>(multicast);
  p.ar.localBufferPtr = reinterpret_cast<T*>(local);
  p.ar.bufferFlags = static_cast<uint32_t*>(flags.data_ptr());
  p.ar.nRanks = world;
  p.ar.rank = rank;
  p.ar.numTokens = tokens;
  p.ar.tokenDim = dim;
  p.ar.epsilon = postEps;
  p.nextEps = nextEps;
  p.xScale = xScale;
  p.rScale = rScale;
  p.postNorm = postNorm;
  p.nextNorm = nextNorm;
  p.scaled = scaled;
  p.experts = static_cast<T const*>(experts.data_ptr());
  p.expertWeights = expertWeights.data_ptr();
  p.topK = experts.size(1);
  p.routedScale = routedScale;

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(localTokens);
  config.blockDim = dim3(dim / 8);
  config.stream = get_stream(residual.device());
  config.attrs = &attr;
  config.numAttrs = 1;
  if (oneShot) {
    switch (world) {
      case 2:
        checkCuda(cudaLaunchKernelEx(&config, &finalizeGatherOneshot<2>, p));
        break;
      case 4:
        checkCuda(cudaLaunchKernelEx(&config, &finalizeGatherOneshot<4>, p));
        break;
      case 8:
        checkCuda(cudaLaunchKernelEx(&config, &finalizeGatherOneshot<8>, p));
        break;
    }
  } else {
    checkCuda(cudaLaunchKernelEx(&config, &finalizeGatherPublish, p));
    config.gridDim = dim3(localTokens * world);
    checkCuda(cudaLaunchKernelEx(&config, &finishGatherTwoshot, p));
  }
}
}  // namespace beam
TVM_FFI_DLL_EXPORT_TYPED_FUNC(beam_allreduce, beam::launch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(beam_moe_gather, beam::launchGather);
