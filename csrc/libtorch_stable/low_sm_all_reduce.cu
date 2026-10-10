// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Low-occupancy in-place NVLS all-reduce.
// Ranks own vector tiles in a cyclic layout. Peer doorbells synchronize the
// local producer and multicast aliases without consuming NVLS atomics.

#include "torch_utils.h"

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>

namespace vllm::low_sm {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kNumBlocks = 4;
constexpr uint32_t kPipelineDepth = 4;
constexpr uint32_t kBarrierThreads = 32;
constexpr uint32_t kVectorBytes = 16;
constexpr uint32_t kDoorbellPhases = 2;
constexpr uint32_t kDoorbellHeaderWords = 128 / sizeof(uint64_t);

struct alignas(16) Bf16x8 {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
};
static_assert(sizeof(Bf16x8) == kVectorBytes);

enum class DoorbellPhase : uint32_t { Ready = 0, Done = 1 };

struct KernelParams {
  Bf16x8* input_mc;
  uint64_t* doorbells;
  int64_t const* peer_base_ptrs;
  uint64_t doorbell_offset;
  uint32_t rank;
  uint32_t world_size;
  uint64_t num_vectors;
};

__device__ __forceinline__ uint64_t
load_acquire_system(uint64_t const* address) {
  uint64_t value = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
#else
  asm volatile("trap;");
#endif
  return value;
}

__device__ __forceinline__ void store_release_system(uint64_t* address,
                                                     uint64_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ Bf16x8 multicast_sum(Bf16x8 const* address) {
  Bf16x8 value{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(address)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return value;
}

__device__ __forceinline__ void multicast_store(Bf16x8* address,
                                                Bf16x8 const& value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "multimem.st.relaxed.sys.global.v4.bf16x2 "
      "[%4], {%0, %1, %2, %3};"
      :
      : "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "l"(address)
      : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ uint64_t
next_generation(KernelParams const& params) {
  __shared__ uint64_t generation;
  if (threadIdx.x == 0) {
    generation = params.doorbells[0] + 1;
    params.doorbells[0] = generation;
  }
  __syncthreads();
  return generation;
}

__device__ __forceinline__ uint64_t
current_generation(KernelParams const& params) {
  __shared__ uint64_t generation;
  if (threadIdx.x == 0) {
    generation = params.doorbells[0];
  }
  __syncthreads();
  return generation;
}

template <DoorbellPhase Phase>
__device__ __forceinline__ void peer_barrier(KernelParams const& params,
                                             uint64_t generation) {
  uint32_t const phase = static_cast<uint32_t>(Phase);
  for (uint32_t peer = threadIdx.x; peer < params.world_size;
       peer += blockDim.x) {
    auto* remote = reinterpret_cast<uint64_t*>(
        static_cast<uintptr_t>(params.peer_base_ptrs[peer]) +
        params.doorbell_offset);
    store_release_system(
        &remote[kDoorbellHeaderWords + phase * params.world_size + params.rank],
        generation);
  }
  __syncthreads();
  for (uint32_t peer = threadIdx.x; peer < params.world_size;
       peer += blockDim.x) {
    auto const* local = &params.doorbells[kDoorbellHeaderWords +
                                          phase * params.world_size + peer];
    while (load_acquire_system(local) != generation) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void reduce_owned_vectors(
    KernelParams const& params) {
  uint64_t const tile = blockIdx.x + gridDim.x * params.rank;
  uint64_t vector = tile * kPipelineDepth * blockDim.x + threadIdx.x;
  uint64_t const stride = static_cast<uint64_t>(gridDim.x) * params.world_size *
                          kPipelineDepth * blockDim.x;
  for (; vector < params.num_vectors; vector += stride) {
    Bf16x8 values[kPipelineDepth];
#pragma unroll
    for (uint32_t i = 0; i < kPipelineDepth; ++i) {
      uint64_t const index = vector + i * blockDim.x;
      if (index < params.num_vectors) {
        values[i] = multicast_sum(params.input_mc + index);
      }
    }
#pragma unroll
    for (uint32_t i = 0; i < kPipelineDepth; ++i) {
      uint64_t const index = vector + i * blockDim.x;
      if (index < params.num_vectors) {
        multicast_store(params.input_mc + index, values[i]);
      }
    }
  }
}

template <DoorbellPhase Phase, bool AdvanceGeneration>
__global__ __launch_bounds__(kBarrierThreads,
                             1) void peer_barrier_kernel(KernelParams params) {
  uint64_t generation;
  if constexpr (AdvanceGeneration) {
    generation = next_generation(params);
  } else {
    generation = current_generation(params);
  }
  peer_barrier<Phase>(params, generation);
}

__global__ __launch_bounds__(
    kBlockSize,
    1) void vector_cyclic_nvls_all_reduce_kernel(KernelParams params) {
  reduce_owned_vectors(params);
}

void check_launch(char const* stage) {
  cudaError_t const error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, "low-SM all-reduce ", stage,
                  " launch failed: ", cudaGetErrorString(error));
}

void low_sm_all_reduce(torch::stable::Tensor& input, int64_t input_mc_ptr,
                       torch::stable::Tensor& doorbells, int64_t peer_base_ptrs,
                       int64_t doorbell_offset, int64_t rank,
                       int64_t world_size) {
  int const device = input.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  cudaDeviceProp const* properties = get_device_prop();

  STD_TORCH_CHECK(properties->major == 10 &&
                      (properties->minor == 0 || properties->minor == 3),
                  "low-SM all-reduce requires SM100 or SM103");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "low-SM all-reduce requires a bfloat16 input");
  STD_TORCH_CHECK(input.is_contiguous(),
                  "low-SM all-reduce requires contiguous input");
  int64_t const num_bytes = input.numel() * input.element_size();
  STD_TORCH_CHECK(num_bytes > 0 && num_bytes % kVectorBytes == 0,
                  "low-SM all-reduce requires a nonempty input whose size "
                  "in bytes is divisible by 16");
  STD_TORCH_CHECK(input_mc_ptr != 0,
                  "low-SM all-reduce requires an input multicast address");
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.mutable_data_ptr()) % alignof(Bf16x8) ==
              0 &&
          static_cast<uintptr_t>(input_mc_ptr) % alignof(Bf16x8) == 0,
      "low-SM all-reduce requires 16-byte-aligned input addresses");

  STD_TORCH_CHECK(
      doorbells.scalar_type() == torch::headeronly::ScalarType::Byte &&
          doorbells.is_contiguous(),
      "low-SM all-reduce requires contiguous uint8 doorbells");
  STD_TORCH_CHECK(doorbells.get_device_index() == device,
                  "low-SM all-reduce requires input and doorbells on the "
                  "same device");
  STD_TORCH_CHECK(peer_base_ptrs != 0,
                  "low-SM all-reduce requires a peer pointer table");
  STD_TORCH_CHECK(doorbell_offset >= 0 && doorbell_offset % 128 == 0,
                  "low-SM all-reduce requires a non-negative, 128-byte-aligned "
                  "doorbell offset");
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(doorbells.mutable_data_ptr()) % 128 == 0,
      "low-SM all-reduce requires 128-byte-aligned doorbells");
  STD_TORCH_CHECK(world_size > 1 && world_size <= UINT32_MAX,
                  "low-SM all-reduce requires at least two ranks");
  STD_TORCH_CHECK(rank >= 0 && rank < world_size,
                  "low-SM all-reduce received an invalid rank");
  uint64_t const doorbell_bytes =
      (kDoorbellHeaderWords + kDoorbellPhases * world_size) * sizeof(uint64_t);
  STD_TORCH_CHECK(static_cast<uint64_t>(doorbells.numel()) >= doorbell_bytes,
                  "low-SM all-reduce doorbell storage is too small");

  KernelParams params{};
  params.input_mc =
      reinterpret_cast<Bf16x8*>(static_cast<uintptr_t>(input_mc_ptr));
  params.doorbells = static_cast<uint64_t*>(doorbells.mutable_data_ptr());
  params.peer_base_ptrs =
      reinterpret_cast<int64_t const*>(static_cast<uintptr_t>(peer_base_ptrs));
  params.doorbell_offset = static_cast<uint64_t>(doorbell_offset);
  params.rank = static_cast<uint32_t>(rank);
  params.world_size = static_cast<uint32_t>(world_size);
  params.num_vectors = static_cast<uint64_t>(num_bytes) / kVectorBytes;

  cudaStream_t const stream = get_current_cuda_stream(device);
  peer_barrier_kernel<DoorbellPhase::Ready, true>
      <<<1, kBarrierThreads, 0, stream>>>(params);
  check_launch("ready barrier");

  vector_cyclic_nvls_all_reduce_kernel<<<kNumBlocks, kBlockSize, 0, stream>>>(
      params);
  check_launch("reduction");

  peer_barrier_kernel<DoorbellPhase::Done, false>
      <<<1, kBarrierThreads, 0, stream>>>(params);
  check_launch("done barrier");
}

}  // namespace vllm::low_sm

STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "low_sm_all_reduce_(Tensor! input, int input_mc_ptr, "
      "Tensor! doorbells, int peer_base_ptrs, int doorbell_offset, int rank, "
      "int world_size) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("low_sm_all_reduce_", TORCH_BOX(&vllm::low_sm::low_sm_all_reduce));
}
