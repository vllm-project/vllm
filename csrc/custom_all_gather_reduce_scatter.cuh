#pragma once

#include "custom_collective_common.cuh"

namespace vllm {

constexpr int kMnnvlLamportAgThreads = 128;
constexpr int kMnnvlLamportRsThreads = 256;
constexpr int kMnnvlLamportConcurrentPollMaxPacks = 8192;

using CopyPack = array_t<uint64_t, 2>;

template <int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_all_gather(RankData* _dp, RankSignals sg, Signal* self_sg,
                            CopyPack* __restrict__ result, int rank,
                            int size_per_rank) {
  auto dp = *_dp;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  barrier_at_start<ngpus>(sg, self_sg, rank);
#pragma unroll
  for (int src_rank = 0; src_rank < ngpus; ++src_rank) {
    auto src = reinterpret_cast<const CopyPack*>(dp.ptrs[src_rank]);
    auto dst = result + src_rank * size_per_rank;
    for (int idx = tid; idx < size_per_rank; idx += stride) {
      dst[idx] = src[idx];
    }
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_scatter(RankData* _dp, RankSignals sg, Signal* self_sg,
                                T* __restrict__ result, int rank,
                                int size_per_rank) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  auto offset = rank * size_per_rank;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size_per_rank;
       idx += gridDim.x * blockDim.x) {
    reinterpret_cast<P*>(result)[idx] =
        packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], offset + idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
union LamportPack {
  P packed;
  uint32_t words[sizeof(P) / sizeof(uint32_t)];
};

template <typename P>
DINLINE LamportPack<P> load_lamport_pack(const P* ptr) {
  static_assert(sizeof(P) == 16);
  LamportPack<P> value;
#if !defined(USE_ROCM)
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.words[0]), "=r"(value.words[1]),
                 "=r"(value.words[2]), "=r"(value.words[3])
               : "l"(ptr)
               : "memory");
#else
  const volatile uint32_t* src =
      reinterpret_cast<const volatile uint32_t*>(ptr);
  #pragma unroll
  for (int i = 0; i < sizeof(P) / sizeof(uint32_t); ++i) {
    value.words[i] = src[i];
  }
#endif
  return value;
}

template <typename P>
DINLINE bool is_lamport_dirty(const LamportPack<P>& value) {
#pragma unroll
  for (int i = 0; i < sizeof(P) / sizeof(uint32_t); ++i) {
    if (value.words[i] == 0x80000000U) return true;
  }
  return false;
}

template <typename P>
DINLINE P lamport_sentinel() {
  LamportPack<P> value;
#pragma unroll
  for (int i = 0; i < sizeof(P) / sizeof(uint32_t); ++i) {
    value.words[i] = 0x80000000U;
  }
  return value.packed;
}

template <typename P>
DINLINE P sanitize_lamport_payload(P packed) {
  LamportPack<P> value{.packed = packed};
#pragma unroll
  for (int i = 0; i < sizeof(P) / sizeof(uint32_t); ++i) {
    if (value.words[i] == 0x80000000U) value.words[i] = 0;
  }
  return value.packed;
}

__host__ __device__ constexpr int mnnvl_lamport_dirty_stage(int current_stage) {
  return (current_stage + 2) % 3;
}

__host__ __device__ constexpr int mnnvl_lamport_next_stage(int current_stage) {
  return (current_stage + 1) % 3;
}

template <typename P>
DINLINE void store_multimem_lamport_payload(P* ptr, P packed) {
  static_assert(sizeof(P) == 16);
  // multimem was introduced in PTX 8.1 (CUDA 12.1) for SM90 and newer.
#if !defined(USE_ROCM) && CUDA_VERSION >= 12010 && defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ >= 900)
  LamportPack<P> value{.packed = packed};
  // The local alias remains sentinel until the multicast payload becomes
  // observable; readers reject prior or partial values and retry.
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(ptr), "r"(value.words[0]), "r"(value.words[1]),
                 "r"(value.words[2]), "r"(value.words[3])
               : "memory");
#elif defined(USE_ROCM)
  __builtin_trap();
#else
  // Multicast mappings do not exist before SM90. Fail closed if this kernel is
  // ever dispatched for an unsupported target instead of issuing an undefined
  // ordinary store to a multicast address.
  asm volatile("trap;");
#endif
}

template <typename P>
DINLINE P wait_lamport_payload(const P* ptr) {
  auto value = load_lamport_pack(ptr);
  while (is_lamport_dirty(value)) value = load_lamport_pack(ptr);
  return value.packed;
}

template <typename P, int ngpus>
DINLINE void wait_lamport_payloads(const P* base, int rank, int rank_stride,
                                   P local_value, P (&values)[ngpus]) {
  bool ready[ngpus];
#pragma unroll
  for (int src_rank = 0; src_rank < ngpus; ++src_rank) {
    ready[src_rank] = src_rank == rank;
    if (src_rank == rank) values[src_rank] = local_value;
  }

  int remaining = ngpus - 1;
  while (remaining != 0) {
#pragma unroll
    for (int src_rank = 0; src_rank < ngpus; ++src_rank) {
      if (!ready[src_rank]) {
        auto value = load_lamport_pack(base + src_rank * rank_stride);
        if (!is_lamport_dirty(value)) {
          values[src_rank] = value.packed;
          ready[src_rank] = true;
          --remaining;
        }
      }
    }
  }
}

template <typename P, typename A, int ngpus>
DINLINE P reduce_lamport_payloads(const P* current_local, const P* packed_input,
                                  int rank, int size_per_rank, int idx) {
  P source_zero =
      rank == 0 ? packed_input[idx] : wait_lamport_payload(current_local + idx);
  A tmp = upcast(source_zero);
#pragma unroll
  for (int src_rank = 1; src_rank < ngpus; ++src_rank) {
    P value = src_rank == rank
                  ? packed_input[rank * size_per_rank + idx]
                  : wait_lamport_payload(current_local +
                                         src_rank * size_per_rank + idx);
    packed_assign_add(tmp, upcast(value));
  }
  return sanitize_lamport_payload(downcast<P>(tmp));
}

DINLINE void lamport_cta_arrive(uint32_t* counter) {
#if !defined(USE_ROCM)
  if (threadIdx.x < 32) {
    asm volatile("barrier.cta.sync 1, %0;" : : "r"(blockDim.x) : "memory");
    if (threadIdx.x == 0) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
      asm volatile("red.async.release.global.gpu.add.u32 [%0], 1;"
                   :
                   : "l"(counter)
                   : "memory");
  #elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile("red.release.global.gpu.add.u32 [%0], 1;"
                   :
                   : "l"(counter)
                   : "memory");
  #else
      atomicAdd(counter, 1);
  #endif
    }
  } else {
    asm volatile("barrier.cta.arrive 1, %0;" : : "r"(blockDim.x) : "memory");
  }
#else
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(counter, 1);
#endif
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(kMnnvlLamportAgThreads, 1)
    mnnvl_lamport_all_gather(RankData* _dp, const T* __restrict__ input,
                             T* __restrict__ result,
                             T* __restrict__ multicast_buffer,
                             uint32_t* __restrict__ epochs, int rank,
                             int size_per_rank, int stage_size) {
  using P = typename packed_t<T>::P;
#if !defined(USE_ROCM) && CUDA_VERSION >= 12000 && defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
  auto dp = *_dp;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  uint32_t epoch = epochs[0];
  int current_stage = epoch % 3;
  // A peer may start the next epoch after we publish but before we finish.
  // Clean the previous stage, which cannot be reused until two epochs later.
  int dirty_stage = mnnvl_lamport_dirty_stage(current_stage);
  int dirty_size = epochs[2 + dirty_stage];
  auto local_buffer = reinterpret_cast<P*>(const_cast<void*>(dp.ptrs[rank]));
  auto current_local = local_buffer + current_stage * stage_size;
  auto dirty_local = local_buffer + dirty_stage * stage_size;
  auto current_multicast =
      reinterpret_cast<P*>(multicast_buffer) + current_stage * stage_size;
  auto packed_input = reinterpret_cast<const P*>(input);
  auto packed_result = reinterpret_cast<P*>(result);

  int total_size = size_per_rank * ngpus;
  P local_value;
  if (tid < size_per_rank) {
    local_value = packed_input[tid];
    // A CUDA multicast mapping may only be accessed with multimem PTX;
    // ordinary global loads and stores have undefined behavior.
    store_multimem_lamport_payload(
        current_multicast + rank * size_per_rank + tid,
        sanitize_lamport_payload(local_value));
  }
#if !defined(USE_ROCM) && CUDA_VERSION >= 12000 && defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  lamport_cta_arrive(&epochs[1]);

  for (int idx = tid; idx < dirty_size; idx += stride) {
    dirty_local[idx] = lamport_sentinel<P>();
  }

  if (tid < size_per_rank) {
#pragma unroll
    for (int src_rank = 0; src_rank < ngpus; ++src_rank) {
      int output_idx = src_rank * size_per_rank + tid;
      P value = src_rank == rank
                    ? local_value
                    : wait_lamport_payload(current_local + output_idx);
      packed_result[output_idx] = value;
    }
  }

  if (tid == 0) {
    while (*reinterpret_cast<volatile uint32_t*>(&epochs[1]) < gridDim.x);
    epochs[2 + current_stage] = total_size;
    epochs[0] = mnnvl_lamport_next_stage(current_stage);
    epochs[1] = 0;
  }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(kMnnvlLamportRsThreads, 1)
    mnnvl_lamport_reduce_scatter_kernel(RankData* _dp,
                                        const T* __restrict__ input,
                                        T* __restrict__ result,
                                        uint32_t* __restrict__ epochs, int rank,
                                        int size_per_rank, int stage_size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
#if !defined(USE_ROCM) && CUDA_VERSION >= 12000 && defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
  auto dp = *_dp;
  int dst_rank = blockIdx.x % ngpus;
  int tile = blockIdx.x / ngpus;
  int idx = tile * blockDim.x + threadIdx.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  uint32_t epoch = epochs[0];
  int current_stage = epoch % 3;
  // A peer may start the next epoch after we publish but before we finish.
  // Clean the previous stage, which cannot be reused until two epochs later.
  int dirty_stage = mnnvl_lamport_dirty_stage(current_stage);
  int dirty_size = epochs[2 + dirty_stage];
  auto local_buffer = reinterpret_cast<P*>(const_cast<void*>(dp.ptrs[rank]));
  auto current_local = local_buffer + current_stage * stage_size;
  auto dirty_local = local_buffer + dirty_stage * stage_size;
  auto packed_input = reinterpret_cast<const P*>(input);

  if (idx < size_per_rank && dst_rank != rank) {
    auto dst = reinterpret_cast<P*>(const_cast<void*>(dp.ptrs[dst_rank])) +
               current_stage * stage_size + rank * size_per_rank;
    auto src = packed_input + dst_rank * size_per_rank;
    dst[idx] = sanitize_lamport_payload(src[idx]);
  }
#if !defined(USE_ROCM) && CUDA_VERSION >= 12000 && defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  lamport_cta_arrive(&epochs[1]);

  for (int idx = tid; idx < dirty_size; idx += stride) {
    dirty_local[idx] = lamport_sentinel<P>();
  }

  if (idx < size_per_rank && dst_rank == rank) {
    if constexpr (ngpus == 4) {
      if (size_per_rank > kMnnvlLamportConcurrentPollMaxPacks) {
        reinterpret_cast<P*>(result)[idx] =
            reduce_lamport_payloads<P, A, ngpus>(current_local, packed_input,
                                                 rank, size_per_rank, idx);
      } else {
        P values[ngpus];
        wait_lamport_payloads<P, ngpus>(
            current_local + idx, rank, size_per_rank,
            packed_input[rank * size_per_rank + idx], values);
        A tmp = upcast(values[0]);
#pragma unroll
        for (int src_rank = 1; src_rank < ngpus; ++src_rank) {
          packed_assign_add(tmp, upcast(values[src_rank]));
        }
        reinterpret_cast<P*>(result)[idx] =
            sanitize_lamport_payload(downcast<P>(tmp));
      }
    } else {
      reinterpret_cast<P*>(result)[idx] = reduce_lamport_payloads<P, A, ngpus>(
          current_local, packed_input, rank, size_per_rank, idx);
    }
  }

  if (tid == 0) {
    while (*reinterpret_cast<volatile uint32_t*>(&epochs[1]) < gridDim.x);
    epochs[2 + current_stage] = size_per_rank * ngpus;
    epochs[0] = mnnvl_lamport_next_stage(current_stage);
    epochs[1] = 0;
  }
}

}  // namespace vllm
