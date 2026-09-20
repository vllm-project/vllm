// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HIP all-reduce for RDNA3/4, TP=2/4, FP16/BF16.
// Input/output: [batch <= 128, hidden <= 8192], positive dimensions,
// weak-contiguous with identical strides, 16-byte aligned and numel % 8 == 0.
// All ranks must use identical shapes, dtypes, tuning settings and call order.
// A context must not execute concurrently on multiple streams.
//
// torch.ops._rdna_custom_ar lifecycle (on each rank's current device):
//   allocate_shared_buffer_and_handle(stride_bytes, world_size)
//   exchange handles; open_mem_handle only for get_required_peer_ranks(...)
//   init_custom_ar(signal_ptrs, rank_data_uint8, rank, stride_bytes)
//   register_buffer(context, [signal_ptr + meta_size(), ...])
//   all_reduce(context, input, output, local_payload_ptr, stride_bytes)
// Allocate/register before graph capture; kernel epochs advance on replay.
// After all uses/graphs finish: dispose, close peer handles, free local buffer.
// Kernel algorithms and tuning originate in
// tp{2,4}_rdna{3,4}_custom_all_reduce.

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;

typedef __hip_bfloat16 nv_bfloat16;

#define HIP_CHECK(cmd)                                             \
  do {                                                             \
    hipError_t error = (cmd);                                      \
    TORCH_CHECK(error == hipSuccess, "HIP call failed: ", #cmd,    \
                " returned error ", static_cast<int>(error), " (", \
                hipGetErrorString(error), ")");                    \
  } while (0)

namespace rdna_custom_ar {
constexpr int kMaxBlocks = 36;
constexpr int kLaunchBlockLimit = 24;
constexpr int kThreads = 256;
constexpr int64_t kMaxBatchSize = 128;
constexpr int64_t kMaxHiddenSize = 8192;
constexpr int64_t kMaxNumel = kMaxBatchSize * kMaxHiddenSize;
using FlagType = uint32_t;

struct Signal {
  alignas(128) FlagType start[kMaxBlocks][8];
  alignas(128) FlagType end[kMaxBlocks][8];
  alignas(128) FlagType flag[kMaxBlocks];
  // Advance payload parity once per call, independently of tile handshakes.
  alignas(128) FlagType epoch[kMaxBlocks];
};
static_assert(sizeof(Signal) % 128 == 0,
              "payload following Signal must remain cache-line aligned");

struct __align__(16) RankData {
  const void* ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

template <typename T, int size_>
struct __align__(alignof(T) * size_) array_t {
  T data[size_];
  using type = T;
  static constexpr int size = size_;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

template <typename P>
DINLINE P nontemporal_load(const P* pointer) {
  const auto value =
      __builtin_nontemporal_load(reinterpret_cast<const int32x4_t*>(pointer));
  return *reinterpret_cast<const P*>(&value);
}

template <typename P>
DINLINE void nontemporal_store(P* pointer, const P& value) {
  __builtin_nontemporal_store(*reinterpret_cast<const int32x4_t*>(&value),
                              reinterpret_cast<int32x4_t*>(pointer));
}

DINLINE float upcast_scalar(half value) { return __half2float(value); }
DINLINE float upcast_scalar(nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
DINLINE T downcast_scalar(float value);

template <>
DINLINE half downcast_scalar(float value) {
  return __float2half(value);
}

template <>
DINLINE nv_bfloat16 downcast_scalar(float value) {
  return __float2bfloat16(value);
}

template <typename T, int size>
DINLINE array_t<float, size> upcast(array_t<T, size> value) {
  array_t<float, size> output;
#pragma unroll
  for (int i = 0; i < size; ++i) {
    output.data[i] = upcast_scalar(value.data[i]);
  }
  return output;
}

template <typename Output>
DINLINE Output downcast(array_t<float, Output::size> value) {
  Output output;
#pragma unroll
  for (int i = 0; i < Output::size; ++i) {
    output.data[i] = downcast_scalar<typename Output::type>(value.data[i]);
  }
  return output;
}

template <int size>
DINLINE void packed_add(array_t<float, size>& lhs,
                        const array_t<float, size>& rhs) {
#pragma unroll
  for (int i = 0; i < size; ++i) {
    lhs.data[i] += rhs.data[i];
  }
}

// Keep every CTA on the same invocation epoch even when shape tuning changes
// the grid size and hence the ownership of payload addresses. Stream ordering
// completes these disjoint updates before the next invocation reads them.
DINLINE void advance_epoch(Signal* self_signal, FlagType epoch) {
  if (threadIdx.x == 0) {
    self_signal->epoch[blockIdx.x] = epoch + 1;
    if (blockIdx.x == 0) {
      for (int block = gridDim.x; block < kMaxBlocks; ++block) {
        self_signal->epoch[block] = epoch + 1;
      }
    }
  }
}

template <typename P>
DINLINE P* payload_slot(const RankData& data, int destination, int slot,
                        int64_t epoch_offset, int64_t buffer_stride) {
  return reinterpret_cast<P*>(
      reinterpret_cast<char*>(const_cast<void*>(data.ptrs[destination])) +
      epoch_offset + static_cast<int64_t>(slot) * buffer_stride);
}

struct Context {
  int rank;
  Signal* self_signal;
  RankSignals signals{};
  RankData* rank_data;
  int device;
  int arch;
  int world_size;
  int64_t buffer_stride;
  fptr_t local_payload = 0;
  at::Tensor rank_data_owner;
};

namespace rdna3_tp2 {
constexpr int kWorldSize = 2;
constexpr int64_t kChunkedThreshold = 128 * 1024;

DINLINE void barrier_after_push(const RankSignals& signals, Signal* self_signal,
                                int rank) {
  __syncthreads();
  const int peer = rank ^ 1;
  if (threadIdx.x == 0) {
    const FlagType expected = self_signal->flag[blockIdx.x] + 1;
    __scoped_atomic_store_n(&signals.signals[peer]->start[blockIdx.x][rank],
                            expected, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_signal->start[blockIdx.x][peer],
                                  __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < expected) {
    }
    // Publish before the CTA barrier so the next phase sees the new counter.
    self_signal->flag[blockIdx.x] = expected;
  }
  __syncthreads();
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    push_all_reduce(RankData* rank_data, RankSignals signals,
                    Signal* self_signal, const T* __restrict__ input,
                    T* __restrict__ output, int rank, int64_t buffer_stride,
                    int packed_size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  const auto data = *rank_data;
  // Every shape advances this epoch once, including chunked invocations.
  const FlagType epoch = self_signal->epoch[blockIdx.x];
  const int64_t buffer_offset =
      static_cast<int64_t>(epoch & 1u) * buffer_stride;
  const int peer = rank ^ 1;
  const P* local_input = reinterpret_cast<const P*>(input);
  const P* local_peer_data = reinterpret_cast<const P*>(
      reinterpret_cast<const char*>(data.ptrs[rank]) + buffer_offset);
  P* peer_buffer = reinterpret_cast<P*>(
      reinterpret_cast<char*>(const_cast<void*>(data.ptrs[peer])) +
      buffer_offset);

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < packed_size;
       index += gridDim.x * blockDim.x) {
    nontemporal_store(&peer_buffer[index], local_input[index]);
  }

  barrier_after_push(signals, self_signal, rank);

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < packed_size;
       index += gridDim.x * blockDim.x) {
    A sum = upcast(local_input[index]);
    packed_add(sum, upcast(nontemporal_load(local_peer_data + index)));
    reinterpret_cast<P*>(output)[index] = downcast<P>(sum);
  }
  // The push barrier has already made every participating lane read epoch.
  // The next invocation is ordered after this kernel on the same stream.
  if (threadIdx.x == 0) self_signal->epoch[blockIdx.x] = epoch + 1;
}

// Limit outstanding writes before consuming each tile from uncached memory.
// Tile addresses retain the same CTA ownership as the ordinary push path.
template <typename T, int Chunk>
__global__ void __launch_bounds__(kThreads, 1)
    chunked_push_all_reduce(RankData* rank_data, RankSignals signals,
                            Signal* self_signal, const T* __restrict__ input,
                            T* __restrict__ output, int rank,
                            int64_t buffer_stride, int packed_size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  const auto data = *rank_data;
  // Every shape advances this epoch once, including chunked invocations.
  const FlagType epoch = self_signal->epoch[blockIdx.x];
  const int64_t buffer_offset =
      static_cast<int64_t>(epoch & 1u) * buffer_stride;
  const int peer = rank ^ 1;
  const P* local_input = reinterpret_cast<const P*>(input);
  const P* local_peer_data = reinterpret_cast<const P*>(
      reinterpret_cast<const char*>(data.ptrs[rank]) + buffer_offset);
  P* peer_buffer = reinterpret_cast<P*>(
      reinterpret_cast<char*>(const_cast<void*>(data.ptrs[peer])) +
      buffer_offset);

  for (int base = 0; base < packed_size;
       base += gridDim.x * blockDim.x * Chunk) {
    const int tile_end =
        min(packed_size, base + int(gridDim.x * blockDim.x) * Chunk);
    for (int index = base + blockIdx.x * blockDim.x + threadIdx.x;
         index < tile_end; index += gridDim.x * blockDim.x) {
      nontemporal_store(&peer_buffer[index], local_input[index]);
    }

    barrier_after_push(signals, self_signal, rank);

    for (int index = base + blockIdx.x * blockDim.x + threadIdx.x;
         index < tile_end; index += gridDim.x * blockDim.x) {
      A sum = upcast(local_input[index]);
      packed_add(sum, upcast(nontemporal_load(local_peer_data + index)));
      reinterpret_cast<P*>(output)[index] = downcast<P>(sum);
    }
  }
  if (threadIdx.x == 0) self_signal->epoch[blockIdx.x] = epoch + 1;
}

static int block_limit(int packed_size) {
  static const int env_limit = [] {
    const char* setting = std::getenv("VLLM_RDNA3_TP2_BLOCKS");
    if (setting == nullptr) {
      return 0;
    }
    const int parsed = std::atoi(setting);
    TORCH_CHECK(parsed > 0 && parsed <= kMaxBlocks,
                "VLLM_RDNA3_TP2_BLOCKS must be in [1, ", kMaxBlocks, "]; got ",
                setting);
    return parsed;
  }();
  const int needed = (packed_size + kThreads - 1) / kThreads;
  const int limit = env_limit == 0 ? kLaunchBlockLimit : env_limit;
  return std::min(limit, needed);
}

std::vector<int64_t> get_required_peer_ranks(int64_t rank) {
  TORCH_CHECK(rank >= 0 && rank < kWorldSize, "invalid rank: ", rank);
  return {rank ^ 1};
}

template <typename T>
void launch(Context* context, hipStream_t stream, const T* input, T* output,
            int64_t numel, int64_t buffer_stride) {
  using P = typename packed_t<T>::P;
  TORCH_CHECK_EQ(numel % P::size, 0);
  const int packed_size = static_cast<int>(numel / P::size);
  if (numel > kChunkedThreshold) {
    constexpr int chunk = 4;
    chunked_push_all_reduce<T, chunk>
        <<<block_limit(packed_size), kThreads, 0, stream>>>(
            context->rank_data, context->signals, context->self_signal, input,
            output, context->rank, buffer_stride, packed_size);
    HIP_CHECK(hipGetLastError());
    return;
  }

  const int blocks = block_limit(packed_size);
  push_all_reduce<T><<<blocks, kThreads, 0, stream>>>(
      context->rank_data, context->signals, context->self_signal, input, output,
      context->rank, buffer_stride, packed_size);
  HIP_CHECK(hipGetLastError());
}

}  // namespace rdna3_tp2

namespace rdna3_tp4 {
constexpr int kWorldSize = 4;
constexpr int64_t kBulkRingThreshold = 256 * 1024;
constexpr int kRingThreads = 1024;
constexpr int64_t kRingMinNumel = 128 * 1024;
// Pairwise uses slots 0/1. Ring packets occupy slots 2/3 and 4/5: their
// 64-bit data+tag format needs up to twice the BF16/FP16 input byte count.
// Bulk ring uses slots 6/7, isolated from both smaller-message protocols.
constexpr int kSlotsPerEpoch = 8;

DINLINE void barrier_after_pair_push(const RankSignals& signals,
                                     Signal* self_signal, int rank, int peer) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const FlagType expected = self_signal->flag[blockIdx.x] + 1;
    __scoped_atomic_store_n(&signals.signals[peer]->start[blockIdx.x][rank],
                            expected, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_signal->start[blockIdx.x][peer],
                                  __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < expected) {
    }
    // Publish before the CTA barrier so the next phase sees the new counter.
    self_signal->flag[blockIdx.x] = expected;
  }
  __syncthreads();
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    pairwise_all_reduce(RankData* rank_data, RankSignals signals,
                        Signal* self_signal, const T* __restrict__ input,
                        T* __restrict__ output, int rank, int64_t buffer_stride,
                        int packed_size, int pair_mask, int cross_mask) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  const auto data = *rank_data;
  const int64_t epoch_offset =
      static_cast<int64_t>(self_signal->epoch[blockIdx.x] & 1u) *
      kSlotsPerEpoch * buffer_stride;
  const int pair_peer = rank ^ pair_mask;
  const int cross_peer = rank ^ cross_mask;
  const P* local_input = reinterpret_cast<const P*>(input);
  P* packed_output = reinterpret_cast<P*>(output);
  P* pair_destination =
      payload_slot<P>(data, pair_peer, 0, epoch_offset, buffer_stride);
  const P* pair_source =
      payload_slot<P>(data, rank, 0, epoch_offset, buffer_stride);
  P* cross_destination =
      payload_slot<P>(data, cross_peer, 1, epoch_offset, buffer_stride);
  const P* cross_source =
      payload_slot<P>(data, rank, 1, epoch_offset, buffer_stride);

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < packed_size;
       index += gridDim.x * blockDim.x) {
    nontemporal_store(&pair_destination[index], local_input[index]);
  }

  barrier_after_pair_push(signals, self_signal, rank, pair_peer);

  // Both ranks in each pair use the same rank order.
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < packed_size;
       index += gridDim.x * blockDim.x) {
    A pair_sum = rank < pair_peer
                     ? upcast(local_input[index])
                     : upcast(nontemporal_load(pair_source + index));
    packed_add(pair_sum, rank < pair_peer
                             ? upcast(nontemporal_load(pair_source + index))
                             : upcast(local_input[index]));
    const P reduced_pair = downcast<P>(pair_sum);
    packed_output[index] = reduced_pair;
    nontemporal_store(&cross_destination[index], reduced_pair);
  }

  barrier_after_pair_push(signals, self_signal, rank, cross_peer);

  // Accumulate the lower-ranked pair first for identical results on all ranks.
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < packed_size;
       index += gridDim.x * blockDim.x) {
    A sum = rank < cross_peer ? upcast(packed_output[index])
                              : upcast(nontemporal_load(cross_source + index));
    packed_add(sum, rank < cross_peer
                        ? upcast(nontemporal_load(cross_source + index))
                        : upcast(packed_output[index]));
    packed_output[index] = downcast<P>(sum);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    self_signal->epoch[blockIdx.x] += 1;
  }
}

// Each aligned 64-bit packet carries three BF16/FP16 values and a nonzero
// 16-bit tag. Polling the packet itself avoids a separate CTA handshake at
// each of the six ring steps. Uncached IPC memory is required on gfx1100.
DINLINE uint64_t receive_ring_packet(uint64_t* source, FlagType tag) {
#if defined(__gfx1100__)
  uint64_t packet;
  do {
    asm volatile(
        "global_load_b64 %0, %1, off glc slc dlc\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(packet)
        : "v"(source)
        : "memory");
  } while (static_cast<FlagType>(packet >> 48) != tag);

  // Retire every consumed tag, including when the next call uses a smaller
  // shape. This prevents stale packets from matching after 16-bit tag wrap.
  // Wait for the local clear before forwarding data: two invocation buffers
  // and separate step 0..3 / step 4..5 regions keep the producer from reusing
  // this address until the clear completes.
  *reinterpret_cast<volatile uint64_t*>(source) = 0;
  asm volatile("s_waitcnt_vscnt null, 0" ::: "memory");
  return packet & 0xffffffffffffULL;
#else
  __builtin_trap();
  return 0;
#endif
}

template <typename T>
DINLINE array_t<T, 4> load_ring_triplet(const T* input, int index, int numel) {
  array_t<T, 4> value{};
#pragma unroll
  for (int element = 0; element < 3; ++element) {
    if (3 * index + element < numel) {
      value.data[element] = input[3 * index + element];
    }
  }
  return value;
}

template <typename T>
DINLINE void store_ring_triplet(T* output, int index, int numel,
                                const array_t<T, 4>& value) {
#pragma unroll
  for (int element = 0; element < 3; ++element) {
    if (3 * index + element < numel) {
      output[3 * index + element] = value.data[element];
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(kRingThreads, 1)
    ring_all_reduce(RankData* rank_data, Signal* self_signal,
                    const T* __restrict__ input, T* __restrict__ output,
                    int rank, int64_t buffer_stride, int numel, int pair_mask,
                    int cross_mask) {
  using P = array_t<T, 4>;
  const auto data = *rank_data;
  const FlagType epoch = self_signal->epoch[blockIdx.x];
  const FlagType tag = epoch % 65535u + 1;
  const int64_t epoch_offset =
      static_cast<int64_t>(epoch & 1u) * kSlotsPerEpoch * buffer_stride;
  // The cycle is [0, pair_mask, pair_mask ^ cross_mask, cross_mask]. Each
  // GPU sends only to its successor, using the same two IPC peers as pairwise.
  const int position = rank == 0            ? 0
                       : rank == pair_mask  ? 1
                       : rank == cross_mask ? 3
                                            : 2;
  const int next = rank ^ ((position & 1) == 0 ? pair_mask : cross_mask);
  uint64_t* destination =
      payload_slot<uint64_t>(data, next, 2, epoch_offset, buffer_stride);
  uint64_t* source =
      payload_slot<uint64_t>(data, rank, 2, epoch_offset, buffer_stride);
  const int64_t second_region = 2 * buffer_stride / sizeof(uint64_t);
  const int packets = (numel + 2) / 3;
  const int tiles = (packets + 4 * kRingThreads - 1) / (4 * kRingThreads);

  // Fixed tiles and a shape-independent grid keep each IPC address on the
  // same CTA when shapes change. Each lane owns an independent ring.
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
#pragma unroll
    for (int step = 0; step < 6; ++step) {
      const int chunk = (position - step + 8) & 3;
      const int index = (tile * 4 + chunk) * kRingThreads + threadIdx.x;
      if (index < packets) {
        P value;
        if (step == 0) {
          value = load_ring_triplet(input, index, numel);
        } else {
          const uint64_t received = receive_ring_packet(
              source + index + ((step - 1) / 4) * second_region, tag);
          __builtin_memcpy(&value, &received, sizeof(value));
          if (step <= 3) {
            auto sum = upcast(value);
            packed_add(sum, upcast(load_ring_triplet(input, index, numel)));
            value = downcast<P>(sum);
          }
        }
        if (step >= 3) store_ring_triplet(output, index, numel, value);
        uint64_t bits;
        __builtin_memcpy(&bits, &value, sizeof(bits));
        const uint64_t packet =
            (static_cast<uint64_t>(tag) << 48) | (bits & 0xffffffffffffULL);
        *reinterpret_cast<volatile uint64_t*>(
            destination + index + (step / 4) * second_region) = packet;
      }
    }
    const int chunk = (position - 6 + 8) & 3;
    const int index = (tile * 4 + chunk) * kRingThreads + threadIdx.x;
    if (index < packets) {
      const uint64_t received =
          receive_ring_packet(source + index + second_region, tag);
      P value;
      __builtin_memcpy(&value, &received, sizeof(value));
      store_ring_triplet(output, index, numel, value);
    }
  }
  // All waves must have read the old epoch before thread 0 advances it.
  __syncthreads();
  if (threadIdx.x == 0) self_signal->epoch[blockIdx.x] = epoch + 1;
}

// Notify the successor and wait for the predecessor after each ring step.
DINLINE void barrier_after_ring_push(const RankSignals& signals,
                                     Signal* self_signal, int rank, int next,
                                     int previous) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const FlagType expected = self_signal->flag[blockIdx.x] + 1;
    __scoped_atomic_store_n(&signals.signals[next]->start[blockIdx.x][rank],
                            expected, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_signal->start[blockIdx.x][previous],
                                  __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < expected) {
    }
    self_signal->flag[blockIdx.x] = expected;
  }
  __syncthreads();
}

// Six ring stages transfer 1.5 input payloads without per-packet tags.
// Fixed tiles keep IPC addresses owned by the same CTA across shape changes.
template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    bulk_ring_all_reduce(RankData* rank_data, RankSignals signals,
                         Signal* self_signal, const T* __restrict__ input,
                         T* __restrict__ output, int rank,
                         int64_t buffer_stride, int packed_size, int pair_mask,
                         int cross_mask) {
  using P = typename packed_t<T>::P;
  const auto data = *rank_data;
  const int64_t epoch_offset =
      static_cast<int64_t>(self_signal->epoch[blockIdx.x] & 1u) *
      kSlotsPerEpoch * buffer_stride;
  const int position = rank == 0            ? 0
                       : rank == pair_mask  ? 1
                       : rank == cross_mask ? 3
                                            : 2;
  const int next = rank ^ ((position & 1) == 0 ? pair_mask : cross_mask);
  const int previous = rank ^ ((position & 1) == 0 ? cross_mask : pair_mask);
  const P* in = reinterpret_cast<const P*>(input);
  P* out = reinterpret_cast<P*>(output);
  const int quarter =
      (packed_size + 4 * kThreads - 1) / (4 * kThreads) * kThreads;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
#pragma unroll
  for (int step = 0; step < 6; ++step) {
    const int chunk = (position - step + 8) & 3;
    P* dst =
        payload_slot<P>(data, next, 6 + step / 4, epoch_offset, buffer_stride);
    const P* src = payload_slot<P>(data, rank, 6 + (step - 1) / 4, epoch_offset,
                                   buffer_stride);
    for (int q = first; q < quarter; q += stride) {
      const int index = (q / kThreads * 4 + chunk) * kThreads + q % kThreads;
      if (index < packed_size) {
        P value = step == 0 ? in[index] : nontemporal_load(src + index);
        if (step > 0 && step <= 3) {
          auto sum = upcast(value);
          packed_add(sum, upcast(in[index]));
          value = downcast<P>(sum);
        }
        if (step >= 3) out[index] = value;
        nontemporal_store(dst + index, value);
      }
    }
    barrier_after_ring_push(signals, self_signal, rank, next, previous);
  }
  const int chunk = (position - 6 + 8) & 3;
  const P* src =
      payload_slot<P>(data, rank, 6 + 1, epoch_offset, buffer_stride);
  for (int q = first; q < quarter; q += stride) {
    const int index = (q / kThreads * 4 + chunk) * kThreads + q % kThreads;
    if (index < packed_size) out[index] = nontemporal_load(src + index);
  }
  __syncthreads();
  if (threadIdx.x == 0) self_signal->epoch[blockIdx.x] += 1;
}

static int block_limit() {
  static const int value = [] {
    const char* setting = std::getenv("VLLM_RDNA3_TP4_BLOCKS");
    if (setting == nullptr) {
      return kLaunchBlockLimit;
    }
    const int parsed = std::atoi(setting);
    TORCH_CHECK(parsed > 0 && parsed <= kMaxBlocks,
                "VLLM_RDNA3_TP4_BLOCKS must be in [1, ", kMaxBlocks, "]; got ",
                setting);
    return parsed;
  }();
  return value;
}

static std::pair<int, int> pair_masks() {
  static const auto masks = [] {
    const char* pair_setting = std::getenv("VLLM_RDNA3_TP4_PAIR_MASK");
    const char* cross_setting = std::getenv("VLLM_RDNA3_TP4_CROSS_MASK");
    const int pair_mask = pair_setting == nullptr ? 1 : std::atoi(pair_setting);
    const int cross_mask =
        cross_setting == nullptr ? 3 : std::atoi(cross_setting);
    TORCH_CHECK(pair_mask >= 1 && pair_mask <= 3 && cross_mask >= 1 &&
                    cross_mask <= 3 && pair_mask != cross_mask,
                "TP=4 pair and cross masks must be distinct values in [1, 3]");
    return std::make_pair(pair_mask, cross_mask);
  }();
  return masks;
}

std::vector<int64_t> get_required_peer_ranks(int64_t rank) {
  TORCH_CHECK(rank >= 0 && rank < kWorldSize, "invalid rank: ", rank);
  const auto [pair_mask, cross_mask] = pair_masks();
  return {rank ^ pair_mask, rank ^ cross_mask};
}

template <typename T>
void launch(Context* context, hipStream_t stream, const T* input, T* output,
            int64_t numel, int64_t buffer_stride) {
  using P = typename packed_t<T>::P;
  TORCH_CHECK_EQ(numel % P::size, 0);
  const int packed_size = static_cast<int>(numel / P::size);
  // Select the same algorithm for both 16-bit floating-point types.
  if (numel > kBulkRingThreshold) {
    bulk_ring_all_reduce<T><<<block_limit(), kThreads, 0, stream>>>(
        context->rank_data, context->signals, context->self_signal, input,
        output, context->rank, buffer_stride, packed_size, pair_masks().first,
        pair_masks().second);
    HIP_CHECK(hipGetLastError());
    return;
  }

  const int blocks =
      std::min(block_limit(), (packed_size + kThreads - 1) / kThreads);
  const auto [pair_mask, cross_mask] = pair_masks();
  // Select the same algorithm for both 16-bit floating-point types.
  if (numel >= kRingMinNumel) {
    ring_all_reduce<T><<<block_limit(), kRingThreads, 0, stream>>>(
        context->rank_data, context->self_signal, input, output, context->rank,
        buffer_stride, static_cast<int>(numel), pair_mask, cross_mask);
    HIP_CHECK(hipGetLastError());
    return;
  }
  pairwise_all_reduce<T><<<blocks, kThreads, 0, stream>>>(
      context->rank_data, context->signals, context->self_signal, input, output,
      context->rank, buffer_stride, packed_size, pair_mask, cross_mask);
  HIP_CHECK(hipGetLastError());
}

}  // namespace rdna3_tp4

namespace rdna4_tp2 {
constexpr int kWorldSize = 2;

DINLINE void barrier_after_push(const RankSignals& signals, Signal* self_signal,
                                int rank) {
  __syncthreads();
  const int peer = rank ^ 1;
  if (threadIdx.x == 0) {
    const FlagType expected = self_signal->flag[blockIdx.x] + 1;
    __scoped_atomic_store_n(&signals.signals[peer]->start[blockIdx.x][rank],
                            expected, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_signal->start[blockIdx.x][peer],
                                  __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < expected) {
    }
    // Publish before the CTA barrier so the next phase sees the new counter.
    self_signal->flag[blockIdx.x] = expected;
  }
  __syncthreads();
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    push_all_reduce(RankData* rank_data, RankSignals signals,
                    Signal* self_signal, const T* __restrict__ input,
                    T* __restrict__ output, int rank, int64_t buffer_stride,
                    int packed_size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  const auto data = *rank_data;
  // Parity follows every invocation, independently of the active grid size.
  const FlagType epoch = self_signal->epoch[blockIdx.x];
  const int64_t buffer_offset =
      static_cast<int64_t>(epoch & 1u) * buffer_stride;
  const int peer = rank ^ 1;
  const P* local_input = reinterpret_cast<const P*>(input);
  const P* local_peer_data = reinterpret_cast<const P*>(
      reinterpret_cast<const char*>(data.ptrs[rank]) + buffer_offset);
  P* peer_buffer = reinterpret_cast<P*>(
      reinterpret_cast<char*>(const_cast<void*>(data.ptrs[peer])) +
      buffer_offset);

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < packed_size;
       index += gridDim.x * blockDim.x) {
    nontemporal_store(&peer_buffer[index], local_input[index]);
  }

  barrier_after_push(signals, self_signal, rank);

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < packed_size;
       index += gridDim.x * blockDim.x) {
    A sum = upcast(local_input[index]);
    packed_add(sum, upcast(nontemporal_load(local_peer_data + index)));
    reinterpret_cast<P*>(output)[index] = downcast<P>(sum);
  }
  advance_epoch(self_signal, epoch);
}

static int block_limit(int packed_size) {
  static const int env_limit = [] {
    const char* setting = std::getenv("VLLM_RDNA4_TP2_BLOCKS");
    if (setting == nullptr) {
      return 0;
    }
    const int parsed = std::atoi(setting);
    TORCH_CHECK(parsed > 0 && parsed <= kMaxBlocks,
                "VLLM_RDNA4_TP2_BLOCKS must be in [1, ", kMaxBlocks, "]; got ",
                setting);
    return parsed;
  }();
  const int needed = (packed_size + kThreads - 1) / kThreads;
  // Medium messages need fewer CTAs after removing redundant signal reads.
  // Larger messages retain enough concurrent writes to saturate PCIe.
  const int default_limit = packed_size > 32768 ? 32
                            : packed_size > 3072 && packed_size <= 12288
                                ? 8
                                : kLaunchBlockLimit;
  const int limit = env_limit == 0 ? default_limit : env_limit;
  return std::min(limit, needed);
}

std::vector<int64_t> get_required_peer_ranks(int64_t rank) {
  TORCH_CHECK(rank >= 0 && rank < kWorldSize, "invalid rank: ", rank);
  return {rank ^ 1};
}

template <typename T>
void launch(Context* context, hipStream_t stream, const T* input, T* output,
            int64_t numel, int64_t buffer_stride) {
  using P = typename packed_t<T>::P;
  TORCH_CHECK_EQ(numel % P::size, 0);
  const int packed_size = static_cast<int>(numel / P::size);
  const int blocks = block_limit(packed_size);
  push_all_reduce<T><<<blocks, kThreads, 0, stream>>>(
      context->rank_data, context->signals, context->self_signal, input, output,
      context->rank, buffer_stride, packed_size);
  HIP_CHECK(hipGetLastError());
}

}  // namespace rdna4_tp2

namespace rdna4_tp4 {
constexpr int kWorldSize = 4;
constexpr int64_t kBulkRingThreshold = 256 * 1024;
// Slots 0..3 serve one-shot/pairwise; 4/5 isolate bulk-ring traffic.
constexpr int kSlotsPerEpoch = 6;
constexpr int64_t kOneshotMaxNumel = 2048;

DINLINE void barrier_after_push(const RankSignals& signals, Signal* self_signal,
                                int rank) {
  __syncthreads();
  const FlagType expected = self_signal->flag[blockIdx.x] + 1;
  if (threadIdx.x < kWorldSize) {
    __scoped_atomic_store_n(
        &signals.signals[threadIdx.x]->start[blockIdx.x][rank], expected,
        __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_signal->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < expected) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    self_signal->flag[blockIdx.x] = expected;
  }
}

DINLINE void barrier_after_pair_push(const RankSignals& signals,
                                     Signal* self_signal, int rank, int peer) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const FlagType expected = self_signal->flag[blockIdx.x] + 1;
    __scoped_atomic_store_n(&signals.signals[peer]->start[blockIdx.x][rank],
                            expected, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_signal->start[blockIdx.x][peer],
                                  __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < expected) {
    }
    // Publish before the CTA barrier so the next phase sees the new counter.
    self_signal->flag[blockIdx.x] = expected;
  }
  __syncthreads();
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    oneshot_all_reduce(RankData* rank_data, RankSignals signals,
                       Signal* self_signal, const T* __restrict__ input,
                       T* __restrict__ output, int rank, int64_t buffer_stride,
                       int npack, int rows) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  const auto data = *rank_data;
  const FlagType epoch = self_signal->epoch[blockIdx.x];
  const int64_t epoch_offset =
      static_cast<int64_t>(epoch & 1u) * kSlotsPerEpoch * buffer_stride;
  const P* local_input = reinterpret_cast<const P*>(input);
  P* packed_output = reinterpret_cast<P*>(output);

  P* peer_dst[kWorldSize];
#pragma unroll
  for (int peer = 0; peer < kWorldSize; ++peer) {
    peer_dst[peer] =
        payload_slot<P>(data, peer, rank, epoch_offset, buffer_stride);
  }

  // Interleaved push: read each packet once and fan out to all peers so the
  // independent PCIe posted writes can overlap.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
      const int index = row * npack + p;
      const P value = local_input[index];
#pragma unroll
      for (int peer = 0; peer < kWorldSize; ++peer) {
        if (peer != rank) {
          peer_dst[peer][index] = value;
        }
      }
    }
  }

  barrier_after_push(signals, self_signal, rank);

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
      const int index = row * npack + p;
      // Every rank must accumulate in the same order, including its own input.
      const P* first = rank == 0 ? local_input
                                 : payload_slot<P>(data, rank, 0, epoch_offset,
                                                   buffer_stride);
      A sum = upcast(first[index]);
#pragma unroll
      for (int source = 1; source < kWorldSize; ++source) {
        const P* src = source == rank
                           ? local_input
                           : payload_slot<P>(data, rank, source, epoch_offset,
                                             buffer_stride);
        packed_add(sum, upcast(src[index]));
      }
      packed_output[index] = downcast<P>(sum);
    }
  }

  // The payload offset was consumed before the handshake; stream ordering
  // completes all readers before the next kernel can reuse this CTA's epoch.
  advance_epoch(self_signal, epoch);
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    pairwise_all_reduce(RankData* rank_data, RankSignals signals,
                        Signal* self_signal, const T* __restrict__ input,
                        T* __restrict__ output, int rank, int64_t buffer_stride,
                        int packed_size, int pair_mask, int cross_mask) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  const auto data = *rank_data;
  const FlagType epoch = self_signal->epoch[blockIdx.x];
  const int64_t epoch_offset =
      static_cast<int64_t>(epoch & 1u) * kSlotsPerEpoch * buffer_stride;
  const int pair_peer = rank ^ pair_mask;
  const int cross_peer = rank ^ cross_mask;
  const P* local_input = reinterpret_cast<const P*>(input);
  P* packed_output = reinterpret_cast<P*>(output);
  P* pair_destination =
      payload_slot<P>(data, pair_peer, 0, epoch_offset, buffer_stride);
  const P* pair_source =
      payload_slot<P>(data, rank, 0, epoch_offset, buffer_stride);
  P* cross_destination =
      payload_slot<P>(data, cross_peer, 1, epoch_offset, buffer_stride);
  const P* cross_source =
      payload_slot<P>(data, rank, 1, epoch_offset, buffer_stride);

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < packed_size;
       index += gridDim.x * blockDim.x) {
    nontemporal_store(&pair_destination[index], local_input[index]);
  }

  barrier_after_pair_push(signals, self_signal, rank, pair_peer);

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < packed_size;
       index += gridDim.x * blockDim.x) {
    A pair_sum = rank < pair_peer
                     ? upcast(local_input[index])
                     : upcast(nontemporal_load(pair_source + index));
    packed_add(pair_sum, rank < pair_peer
                             ? upcast(nontemporal_load(pair_source + index))
                             : upcast(local_input[index]));
    const P reduced_pair = downcast<P>(pair_sum);
    packed_output[index] = reduced_pair;
    nontemporal_store(&cross_destination[index], reduced_pair);
  }

  barrier_after_pair_push(signals, self_signal, rank, cross_peer);

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < packed_size;
       index += gridDim.x * blockDim.x) {
    A sum = rank < cross_peer ? upcast(packed_output[index])
                              : upcast(nontemporal_load(cross_source + index));
    packed_add(sum, rank < cross_peer
                        ? upcast(nontemporal_load(cross_source + index))
                        : upcast(packed_output[index]));
    packed_output[index] = downcast<P>(sum);
  }

  // The payload offset was consumed before the handshake; stream ordering
  // completes all readers before the next kernel can reuse this CTA's epoch.
  advance_epoch(self_signal, epoch);
}

// Notify the successor and wait for the predecessor after each ring step.
DINLINE void barrier_after_ring_push(const RankSignals& signals,
                                     Signal* self_signal, int rank, int next,
                                     int previous) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const FlagType expected = self_signal->flag[blockIdx.x] + 1;
    __scoped_atomic_store_n(&signals.signals[next]->start[blockIdx.x][rank],
                            expected, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_signal->start[blockIdx.x][previous],
                                  __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < expected) {
    }
    self_signal->flag[blockIdx.x] = expected;
  }
  __syncthreads();
}

// Six ring stages transfer 1.5 input payloads without per-packet tags.
// Fixed tiles keep IPC addresses owned by the same CTA across shape changes.
template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    bulk_ring_all_reduce(RankData* rank_data, RankSignals signals,
                         Signal* self_signal, const T* __restrict__ input,
                         T* __restrict__ output, int rank,
                         int64_t buffer_stride, int packed_size, int pair_mask,
                         int cross_mask) {
  using P = typename packed_t<T>::P;
  const auto data = *rank_data;
  const FlagType epoch = self_signal->epoch[blockIdx.x];
  const int64_t epoch_offset =
      static_cast<int64_t>(epoch & 1u) * kSlotsPerEpoch * buffer_stride;
  const int position = rank == 0            ? 0
                       : rank == pair_mask  ? 1
                       : rank == cross_mask ? 3
                                            : 2;
  const int next = rank ^ ((position & 1) == 0 ? pair_mask : cross_mask);
  const int previous = rank ^ ((position & 1) == 0 ? cross_mask : pair_mask);
  const P* in = reinterpret_cast<const P*>(input);
  P* out = reinterpret_cast<P*>(output);
  const int quarter =
      (packed_size + 4 * kThreads - 1) / (4 * kThreads) * kThreads;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
#pragma unroll
  for (int step = 0; step < 6; ++step) {
    const int chunk = (position - step + 8) & 3;
    P* dst =
        payload_slot<P>(data, next, 4 + step / 4, epoch_offset, buffer_stride);
    const P* src = payload_slot<P>(data, rank, 4 + (step - 1) / 4, epoch_offset,
                                   buffer_stride);
    for (int q = first; q < quarter; q += stride) {
      const int index = (q / kThreads * 4 + chunk) * kThreads + q % kThreads;
      if (index < packed_size) {
        P value = step == 0 ? in[index] : nontemporal_load(src + index);
        if (step > 0 && step <= 3) {
          auto sum = upcast(value);
          packed_add(sum, upcast(in[index]));
          value = downcast<P>(sum);
        }
        if (step >= 3) out[index] = value;
        nontemporal_store(dst + index, value);
      }
    }
    barrier_after_ring_push(signals, self_signal, rank, next, previous);
  }
  const int chunk = (position - 6 + 8) & 3;
  const P* src =
      payload_slot<P>(data, rank, 4 + 1, epoch_offset, buffer_stride);
  for (int q = first; q < quarter; q += stride) {
    const int index = (q / kThreads * 4 + chunk) * kThreads + q % kThreads;
    if (index < packed_size) out[index] = nontemporal_load(src + index);
  }
  __syncthreads();
  advance_epoch(self_signal, epoch);
}

static int block_limit(int packed_size) {
  static const int value = [] {
    const char* setting = std::getenv("VLLM_RDNA4_TP4_BLOCKS");
    if (setting == nullptr) {
      return 0;
    }
    const int parsed = std::atoi(setting);
    TORCH_CHECK(parsed > 0 && parsed <= kMaxBlocks,
                "VLLM_RDNA4_TP4_BLOCKS must be in [1, ", kMaxBlocks, "]; got ",
                setting);
    return parsed;
  }();
  // Reduce synchronization traffic for medium messages; keep enough CTAs
  // to saturate PCIe on larger transfers. An explicit override wins.
  const int default_limit =
      packed_size > 3072 && packed_size <= 12288 ? 8 : kLaunchBlockLimit;
  return value == 0 ? default_limit : value;
}

static std::pair<int, int> pair_masks() {
  static const auto masks = [] {
    const char* pair_setting = std::getenv("VLLM_RDNA4_TP4_PAIR_MASK");
    const char* cross_setting = std::getenv("VLLM_RDNA4_TP4_CROSS_MASK");
    const int pair_mask = pair_setting == nullptr ? 1 : std::atoi(pair_setting);
    const int cross_mask =
        cross_setting == nullptr ? 3 : std::atoi(cross_setting);
    TORCH_CHECK(pair_mask >= 1 && pair_mask <= 3 && cross_mask >= 1 &&
                    cross_mask <= 3 && pair_mask != cross_mask,
                "TP=4 pair and cross masks must be distinct values in [1, 3]");
    return std::make_pair(pair_mask, cross_mask);
  }();
  return masks;
}

static int forced_algorithm() {
  const char* value = std::getenv("VLLM_RDNA4_TP4_ALGO");
  if (value == nullptr || std::strcmp(value, "auto") == 0) {
    return 0;
  }
  if (std::strcmp(value, "oneshot") == 0) {
    return 1;
  }
  if (std::strcmp(value, "pairwise") == 0) {
    return 2;
  }
  TORCH_CHECK(false,
              "VLLM_RDNA4_TP4_ALGO must be auto, oneshot, or pairwise; got ",
              value);
  return 0;
}

static int64_t oneshot_max_numel() {
  static const int64_t value = [] {
    const char* setting = std::getenv("VLLM_RDNA4_ONESHOT_MAX_NUMEL");
    if (setting == nullptr) {
      return kOneshotMaxNumel;
    }
    const int64_t parsed = std::atoll(setting);
    TORCH_CHECK(parsed > 0 && parsed <= kMaxNumel,
                "VLLM_RDNA4_ONESHOT_MAX_NUMEL must be in (0, ", kMaxNumel,
                "]; got ", setting);
    return parsed;
  }();
  return value;
}

static bool use_pairwise(int64_t numel) {
  const int forced = forced_algorithm();
  if (forced != 0) {
    return forced == 2;
  }
  return numel > oneshot_max_numel();
}

std::vector<int64_t> get_required_peer_ranks(int64_t rank) {
  TORCH_CHECK(rank >= 0 && rank < kWorldSize, "invalid rank: ", rank);
  const int algo = forced_algorithm();
  if (algo == 2) {
    const auto [pair_mask, cross_mask] = pair_masks();
    return {rank ^ pair_mask, rank ^ cross_mask};
  }
  // auto and one-shot both need every peer (batch=1 one-shot path).
  std::vector<int64_t> peers;
  for (int peer = 0; peer < kWorldSize; ++peer) {
    if (peer != rank) {
      peers.push_back(peer);
    }
  }
  return peers;
}

template <typename T>
void launch(Context* context, hipStream_t stream, const T* input, T* output,
            int64_t numel, int64_t buffer_stride) {
  using P = typename packed_t<T>::P;
  TORCH_CHECK_EQ(numel % P::size, 0);
  const int packed_size = static_cast<int>(numel / P::size);
  // Select the same algorithm for both 16-bit floating-point types.
  if (numel > kBulkRingThreshold && forced_algorithm() == 0) {
    bulk_ring_all_reduce<T><<<block_limit(packed_size), kThreads, 0, stream>>>(
        context->rank_data, context->signals, context->self_signal, input,
        output, context->rank, buffer_stride, packed_size, pair_masks().first,
        pair_masks().second);
    HIP_CHECK(hipGetLastError());
    return;
  }

  if (use_pairwise(numel)) {
    const int blocks = std::min(block_limit(packed_size),
                                (packed_size + kThreads - 1) / kThreads);
    const auto [pair_mask, cross_mask] = pair_masks();
    pairwise_all_reduce<T><<<blocks, kThreads, 0, stream>>>(
        context->rank_data, context->signals, context->self_signal, input,
        output, context->rank, buffer_stride, packed_size, pair_mask,
        cross_mask);
  } else {
    constexpr int kOneshotRows = 1;
    const int npack = packed_size;
    const int grid = std::min(kLaunchBlockLimit, std::max(1, kOneshotRows));
    oneshot_all_reduce<T><<<grid, kThreads, 0, stream>>>(
        context->rank_data, context->signals, context->self_signal, input,
        output, context->rank, buffer_stride, npack, kOneshotRows);
  }
  HIP_CHECK(hipGetLastError());
}

}  // namespace rdna4_tp4

static hipStream_t current_stream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

static int current_arch() {
  int device;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  const std::string arch =
      std::string(props.gcnArchName)
          .substr(0, std::string(props.gcnArchName).find(':'));
  if (arch == "gfx1100") return 3;
  if (arch == "gfx1200" || arch == "gfx1201") return 4;
  TORCH_CHECK(false,
              "RDNA custom all-reduce supports gfx1100/gfx1200/gfx1201; got ",
              arch);
}

static void check_world_size(int64_t world_size) {
  TORCH_CHECK(world_size == 2 || world_size == 4,
              "RDNA custom all-reduce requires TP=2 or TP=4");
}

static void check_stride(int64_t stride) {
  TORCH_CHECK(stride > 0 && stride <= 2 * kMaxNumel && stride % 16 == 0,
              "buffer stride must be a positive multiple of 16, at most 2 MiB");
}

static std::vector<int64_t> required_peers(int arch, int world_size, int rank) {
  if (world_size == 2) return {rank ^ 1};
  if (arch == 3) return rdna3_tp4::get_required_peer_ranks(rank);
  return rdna4_tp4::get_required_peer_ranks(rank);
}

std::vector<int64_t> get_required_peer_ranks(int64_t rank, int64_t world_size) {
  check_world_size(world_size);
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  return required_peers(current_arch(), world_size, rank);
}

int64_t meta_size() { return sizeof(Signal); }
int64_t rank_data_size() { return sizeof(RankData); }

fptr_t init_custom_ar(const std::vector<fptr_t>& signal_ptrs,
                      at::Tensor& rank_data, int64_t rank,
                      int64_t buffer_stride) {
  const int world_size = signal_ptrs.size();
  check_world_size(world_size);
  check_stride(buffer_stride);
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  TORCH_CHECK(rank_data.is_cuda() && rank_data.scalar_type() == at::kByte &&
                  rank_data.is_contiguous() &&
                  rank_data.numel() >= static_cast<int64_t>(sizeof(RankData)),
              "rank_data must be a contiguous GPU uint8 buffer of "
              "rank_data_size bytes");
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(rank_data.device());
  const int arch = current_arch();
  auto peers = required_peers(arch, world_size, rank);
  peers.push_back(rank);
  for (int peer : peers) {
    TORCH_CHECK(
        signal_ptrs[peer] != 0 && signal_ptrs[peer] % alignof(Signal) == 0,
        "missing or misaligned signal pointer for rank ", peer);
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(rank_data.data_ptr()) % 16 == 0,
              "rank_data must be 16-byte aligned");
  auto context = std::make_unique<Context>();
  context->rank = rank;
  context->self_signal = reinterpret_cast<Signal*>(signal_ptrs[rank]);
  context->rank_data = reinterpret_cast<RankData*>(rank_data.data_ptr());
  context->device = rank_data.get_device();
  context->arch = arch;
  context->world_size = world_size;
  context->buffer_stride = buffer_stride;
  context->rank_data_owner = rank_data;
  for (int i = 0; i < world_size; ++i) {
    context->signals.signals[i] = reinterpret_cast<Signal*>(signal_ptrs[i]);
  }
  return reinterpret_cast<fptr_t>(context.release());
}

void register_buffer(fptr_t handle, const std::vector<fptr_t>& payload_ptrs) {
  auto* context = reinterpret_cast<Context*>(handle);
  TORCH_CHECK(context != nullptr, "invalid RDNA all-reduce context");
  TORCH_CHECK(payload_ptrs.size() == context->world_size,
              "wrong payload count");
  const at::hip::HIPGuardMasqueradingAsCUDA guard(context->device);
  RankData data{};
  for (int i = 0; i < context->world_size; ++i) {
    const auto signal = reinterpret_cast<fptr_t>(context->signals.signals[i]);
    TORCH_CHECK(payload_ptrs[i] == (signal == 0 ? 0 : signal + meta_size()),
                "payload pointer must immediately follow its signal metadata");
    data.ptrs[i] = reinterpret_cast<void*>(payload_ptrs[i]);
  }
  HIP_CHECK(hipMemcpy(context->rank_data, &data, sizeof(data),
                      hipMemcpyHostToDevice));
  context->local_payload = payload_ptrs[context->rank];
}

static bool is_weak_contiguous(const at::Tensor& tensor) {
  return tensor.is_contiguous() ||
         (tensor.is_non_overlapping_and_dense() &&
          tensor.storage().nbytes() -
                  tensor.storage_offset() * tensor.element_size() ==
              static_cast<size_t>(tensor.numel() * tensor.element_size()));
}

template <typename T>
static void launch(Context* context, const at::Tensor& input,
                   at::Tensor& output) {
  const hipStream_t stream = current_stream();
  const auto* in = reinterpret_cast<const T*>(input.data_ptr());
  auto* out = reinterpret_cast<T*>(output.data_ptr());
  const auto numel = input.numel();
  const auto stride = context->buffer_stride;
  if (context->arch == 3) {
    if (context->world_size == 2) {
      rdna3_tp2::launch(context, stream, in, out, numel, stride);
    } else {
      rdna3_tp4::launch(context, stream, in, out, numel, stride);
    }
  } else if (context->world_size == 2) {
    rdna4_tp2::launch(context, stream, in, out, numel, stride);
  } else {
    rdna4_tp4::launch(context, stream, in, out, numel, stride);
  }
}

void all_reduce(fptr_t handle, at::Tensor& input, at::Tensor& output,
                fptr_t registered_buffer, int64_t buffer_stride) {
  auto* context = reinterpret_cast<Context*>(handle);
  TORCH_CHECK(context != nullptr, "invalid RDNA all-reduce context");
  TORCH_CHECK(context->local_payload != 0 &&
                  registered_buffer == context->local_payload,
              "all_reduce requires its registered local IPC payload");
  TORCH_CHECK(buffer_stride == context->buffer_stride,
              "buffer stride mismatch");
  TORCH_CHECK(input.is_cuda() && output.is_cuda() &&
                  input.get_device() == context->device &&
                  output.get_device() == context->device,
              "input and output must be on the context device");
  TORCH_CHECK(input.dim() == 2 && input.size(0) > 0 &&
                  input.size(0) <= kMaxBatchSize && input.size(1) > 0 &&
                  input.size(1) <= kMaxHiddenSize,
              "expected [batch_size, hidden_size] with batch in [1, 128] "
              "and hidden in [1, 8192]");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "RDNA custom all-reduce supports FP16 and BF16 only");
  TORCH_CHECK(input.scalar_type() == output.scalar_type() &&
                  input.sizes() == output.sizes() &&
                  input.strides() == output.strides(),
              "input/output dtype, shape and strides must match");
  TORCH_CHECK(is_weak_contiguous(input) && is_weak_contiguous(output),
              "input and output must be weak-contiguous");
  TORCH_CHECK(
      input.numel() % 8 == 0 &&
          reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
      "16-byte packs require numel divisible by 8 and aligned pointers");
  TORCH_CHECK(input.numel() * input.element_size() <= buffer_stride,
              "input exceeds IPC slot capacity");
  at::assert_no_overlap(input, output);
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(input.device());
  if (input.scalar_type() == at::kHalf) {
    launch<half>(context, input, output);
  } else {
    launch<nv_bfloat16>(context, input, output);
  }
}

void dispose(fptr_t handle) { delete reinterpret_cast<Context*>(handle); }

std::tuple<fptr_t, at::Tensor> allocate_shared_buffer_and_handle(
    int64_t buffer_stride, int64_t world_size) {
  check_world_size(world_size);
  check_stride(buffer_stride);
  const int arch = current_arch();
  const int slots = world_size == 2 ? 2 : (arch == 3 ? 16 : 12);
  const int64_t bytes =
      (meta_size() + slots * buffer_stride + 4095) & ~int64_t{4095};
  const auto stream = current_stream();
  hipStreamCaptureStatus status;
  HIP_CHECK(hipStreamIsCapturing(stream, &status));
  TORCH_CHECK(status == hipStreamCaptureStatusNone,
              "allocate shared buffers before graph capture");
  auto handle =
      at::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))},
                at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  void* buffer = nullptr;
  HIP_CHECK(hipExtMallocWithFlags(&buffer, bytes, hipDeviceMallocUncached));
  try {
    HIP_CHECK(hipMemsetAsync(buffer, 0, bytes, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipIpcGetMemHandle(
        reinterpret_cast<hipIpcMemHandle_t*>(handle.data_ptr()), buffer));
  } catch (...) {
    (void)hipFree(buffer);
    throw;
  }
  return {reinterpret_cast<fptr_t>(buffer), handle};
}

fptr_t open_mem_handle(at::Tensor& handle) {
  TORCH_CHECK(handle.device().is_cpu() && handle.scalar_type() == at::kByte &&
                  handle.is_contiguous() &&
                  handle.numel() == sizeof(hipIpcMemHandle_t),
              "expected CPU uint8 HIP IPC handle");
  hipIpcMemHandle_t ipc_handle;
  std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(ipc_handle));
  void* pointer = nullptr;
  HIP_CHECK(
      hipIpcOpenMemHandle(&pointer, ipc_handle, hipIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(pointer);
}

void close_mem_handle(fptr_t pointer) {
  HIP_CHECK(hipIpcCloseMemHandle(reinterpret_cast<void*>(pointer)));
}

void free_shared_buffer(fptr_t pointer) {
  HIP_CHECK(hipFree(reinterpret_cast<void*>(pointer)));
}

}  // namespace rdna_custom_ar

TORCH_LIBRARY(_rdna_custom_ar, ops) {
  ops.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, int rank, "
      "int buffer_stride) -> int",
      &rdna_custom_ar::init_custom_ar);
  ops.def(
      "all_reduce(int fa, Tensor inp, Tensor(a!) out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()",
      &rdna_custom_ar::all_reduce);
  ops.def("dispose(int fa) -> ()", &rdna_custom_ar::dispose);
  ops.def("meta_size() -> int", &rdna_custom_ar::meta_size);
  ops.def("rank_data_size() -> int", &rdna_custom_ar::rank_data_size);
  ops.def("get_required_peer_ranks(int rank, int world_size) -> int[]",
          &rdna_custom_ar::get_required_peer_ranks);
  ops.def("register_buffer(int fa, int[] ipc_tensors) -> ()",
          &rdna_custom_ar::register_buffer);
  ops.def(
      "allocate_shared_buffer_and_handle(int size, int world_size) -> (int, "
      "Tensor)",
      &rdna_custom_ar::allocate_shared_buffer_and_handle);
  ops.def("open_mem_handle(Tensor mem_handle) -> int",
          &rdna_custom_ar::open_mem_handle);
  ops.def("close_mem_handle(int ptr) -> ()", &rdna_custom_ar::close_mem_handle);
  ops.def("free_shared_buffer(int ptr) -> ()",
          &rdna_custom_ar::free_shared_buffer);
}

#undef DINLINE
#undef HIP_CHECK
