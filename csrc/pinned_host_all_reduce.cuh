#pragma once

#include "custom_collective_common.cuh"

namespace vllm {

/**
 * All-reduce for two GPUs without peer access, staged through a host
 * shared-memory segment that both processes register as mapped pinned memory.
 * Each call is one kernel (plus a device-to-host copy in copy-engine mode) with
 * no host synchronization, so it can be captured in CUDA graphs.
 *
 * Modes by size:
 *  - LL: NCCL's LL lines {data, tag, data, tag}; one store publishes data and
 *    flag together and the reader polls the data itself.
 *  - flag: plain stores, a system fence, then one flag.
 *  - copy engine: cudaMemcpyAsync into this rank's slot, then a kernel raises a
 *    flag, waits for the peer's, reads the peer slot and sums. The slot is
 *    single-buffered, so the kernel ends with an ack handshake.
 *
 * The sequence number is 64-bit and never wraps. LL and flag buffers alternate
 * by its parity: a rank reaches call s + 2 only after the peer published
 * s + 1, which the peer does only after every read of the parity-s buffer.
 * The 32-bit LL tag repeats on a parity buffer every 2 * (2^32 - 1) calls, so
 * a stale line can match only if it was last written a multiple of that many
 * calls (about 8.6e9) earlier and the peer has not yet written it this call.
 *
 * Both ranks compute float(mine) + float(peer) rounded once, so results are
 * identical on both ranks and match NCCL's two-rank sum bit for bit.
 */
struct PinnedHostAllreduce {
  using Seq = unsigned long long;
  static constexpr int kLLThreads = 128;
  static constexpr int kThreads = 512;

  struct RankSignals {
    alignas(128) Seq flag;
    alignas(128) Seq ce_ready;
    alignas(128) Seq ce_ack;
  };

  // Device memory: the last finished call and a block counter for it.
  struct State {
    Seq seq;
    unsigned int blocks_done;
  };

  // Offsets into the shared host buffer:
  //   signals  RankSignals[rank]
  //   ll       [parity][rank] blocks of 2 * ll_max bytes (8 payload B per 16)
  //   flag     [parity][rank] blocks of flag_max bytes
  //   ce       [rank] blocks of max_size bytes
  struct Layout {
    size_t ll, flag, ce, total;

    static constexpr size_t align(size_t n) {
      return (n + 4095) & ~size_t{4095};
    }

    Layout(size_t ll_max, size_t flag_max, size_t max_size)
        : ll(align(2 * sizeof(RankSignals))),
          flag(ll + align(2 * 2 * 2 * ll_max)),
          ce(flag + align(2 * 2 * flag_max)),
          total(ce + align(2 * max_size)) {}
  };

  char* host;    // the buffer in this process' address space
  char* device;  // the same buffer as mapped for this device
  State* state;
  int rank;
  size_t ll_max, flag_max, max_size;
  Layout layout;

  template <typename T>
  cudaError_t allreduce(cudaStream_t stream, const T* input, T* output,
                        size_t bytes) const;
};

namespace pinned_host {

using Seq = PinnedHostAllreduce::Seq;
using RankSignals = PinnedHostAllreduce::RankSignals;
using State = PinnedHostAllreduce::State;

DINLINE void st_volatile_v4(uint4* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

DINLINE uint4 ld_volatile_v4(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

DINLINE void st_release(Seq* p, Seq v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u64 [%1], %0;" ::"l"(v), "l"(p)
               : "memory");
#else
  asm volatile("membar.sys; st.volatile.global.u64 [%1], %0;" ::"l"(v), "l"(p)
               : "memory");
#endif
}

DINLINE Seq ld_acquire(const Seq* p) {
  Seq v;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(p)
               : "memory");
#else
  asm volatile("ld.volatile.global.u64 %0, [%1]; membar.gl;"
               : "=l"(v)
               : "l"(p)
               : "memory");
#endif
  return v;
}

// The peer may already be one call ahead, so wait for >= seq.
DINLINE void wait_flag(const Seq* flag, Seq seq) {
  while (ld_acquire(flag) < seq);
}

// Never 0, so lines of the zero-filled buffer never match.
DINLINE uint32_t ll_tag(Seq seq) {
  return static_cast<uint32_t>(seq % 0xFFFFFFFFull) + 1;
}

// Elementwise sum of two 32-bit words of packed T in fp32, rounded once.
template <typename T>
DINLINE uint32_t packed_sum(uint32_t a, uint32_t b) {
  using P = array_t<T, sizeof(uint32_t) / sizeof(T)>;
  auto x = upcast(*reinterpret_cast<P*>(&a));
  const auto y = upcast(*reinterpret_cast<P*>(&b));
#pragma unroll
  for (int i = 0; i < P::size; i++) x.data[i] += y.data[i];
  P r = downcast<P>(x);
  return *reinterpret_cast<uint32_t*>(&r);
}

template <typename T>
DINLINE uint4 packed_sum(uint4 a, uint4 b) {
  return make_uint4(packed_sum<T>(a.x, b.x), packed_sum<T>(a.y, b.y),
                    packed_sum<T>(a.z, b.z), packed_sum<T>(a.w, b.w));
}

DINLINE Seq begin_call(const State* state) {
  __shared__ Seq seq;
  if (threadIdx.x == 0) seq = state->seq + 1;
  __syncthreads();
  return seq;
}

// The last block to finish publishes the call's sequence number for the next
// kernel on the stream.
DINLINE void finish_call(State* state, Seq seq) {
  state->blocks_done = 0;
  *reinterpret_cast<volatile Seq*>(&state->seq) = seq;
}

DINLINE bool last_block(State* state) {
  __threadfence();
  return atomicAdd(&state->blocks_done, 1u) == gridDim.x - 1;
}

template <typename T>
__global__ void __launch_bounds__(PinnedHostAllreduce::kLLThreads)
    ll_kernel(const uint2* __restrict__ input, uint2* __restrict__ output,
              uint4* ll, size_t ll_lines, int rank, size_t nlines,
              State* state) {
  const Seq seq = begin_call(state);
  const uint32_t tag = ll_tag(seq);
  const int parity = seq & 1;
  uint4* mine = ll + (parity * 2 + rank) * ll_lines;
  const uint4* peer = ll + (parity * 2 + (rank ^ 1)) * ll_lines;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t t0 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (size_t i = t0; i < nlines; i += stride) {
    const uint2 v = input[i];
    st_volatile_v4(mine + i, make_uint4(v.x, tag, v.y, tag));
  }
  // Poll up to four lines at a time so their PCIe round trips overlap.
  for (size_t i = t0; i < nlines; i += 4 * stride) {
    bool pending[4];
#pragma unroll
    for (int u = 0; u < 4; u++) pending[u] = i + u * stride < nlines;
    while (pending[0] || pending[1] || pending[2] || pending[3]) {
      uint4 p[4];
#pragma unroll
      for (int u = 0; u < 4; u++)
        if (pending[u]) p[u] = ld_volatile_v4(peer + i + u * stride);
#pragma unroll
      for (int u = 0; u < 4; u++) {
        if (pending[u] && p[u].y == tag && p[u].w == tag) {
          const size_t j = i + u * stride;
          const uint2 v = input[j];
          output[j] = make_uint2(packed_sum<T>(v.x, p[u].x),
                                 packed_sum<T>(v.y, p[u].z));
          pending[u] = false;
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && last_block(state)) finish_call(state, seq);
}

// One block.
template <typename T>
__global__ void __launch_bounds__(PinnedHostAllreduce::kThreads)
    flag_kernel(const uint4* __restrict__ input, uint4* __restrict__ output,
                RankSignals* signals, uint4* data, size_t data_vecs, int rank,
                size_t nvec, State* state) {
  const Seq seq = begin_call(state);
  const int parity = seq & 1;
  uint4* mine = data + (parity * 2 + rank) * data_vecs;
  const uint4* peer = data + (parity * 2 + (rank ^ 1)) * data_vecs;

  for (size_t i = threadIdx.x; i < nvec; i += blockDim.x) mine[i] = input[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release(&signals[rank].flag, seq);
    wait_flag(&signals[rank ^ 1].flag, seq);
  }
  __syncthreads();
  for (size_t i = threadIdx.x; i < nvec; i += blockDim.x)
    output[i] = packed_sum<T>(input[i], ld_volatile_v4(peer + i));
  __syncthreads();
  if (threadIdx.x == 0) finish_call(state, seq);
}

// The copy of this rank's input into its host slot finished before launch.
template <typename T>
__global__ void __launch_bounds__(PinnedHostAllreduce::kThreads)
    ce_kernel(const uint4* __restrict__ input, uint4* __restrict__ output,
              RankSignals* signals, const uint4* peer, int rank, size_t nvec,
              State* state) {
  const Seq seq = begin_call(state);
  RankSignals* mine = &signals[rank];
  RankSignals* other = &signals[rank ^ 1];
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) st_release(&mine->ce_ready, seq);
    wait_flag(&other->ce_ready, seq);
  }
  __syncthreads();
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < nvec; i += stride)
    output[i] = packed_sum<T>(input[i], ld_volatile_v4(peer + i));
  __syncthreads();
  if (threadIdx.x == 0 && last_block(state)) {
    // Every block has read the peer slot: ack, then wait for the peer's ack
    // before the next call's copy may overwrite this rank's slot.
    st_release(&mine->ce_ack, seq);
    wait_flag(&other->ce_ack, seq);
    finish_call(state, seq);
  }
}

}  // namespace pinned_host

template <typename T>
cudaError_t PinnedHostAllreduce::allreduce(cudaStream_t stream, const T* input,
                                           T* output, size_t bytes) const {
  using namespace pinned_host;
  auto* signals = reinterpret_cast<RankSignals*>(device);
  const auto* in = reinterpret_cast<const uint4*>(input);
  auto* out = reinterpret_cast<uint4*>(output);
  if (bytes <= ll_max) {
    const size_t nlines = bytes / 8;
    const int blocks = static_cast<int>((nlines + kLLThreads - 1) / kLLThreads);
    ll_kernel<T><<<blocks, kLLThreads, 0, stream>>>(
        reinterpret_cast<const uint2*>(input), reinterpret_cast<uint2*>(output),
        reinterpret_cast<uint4*>(device + layout.ll), 2 * ll_max / 16, rank,
        nlines, state);
  } else if (bytes <= flag_max) {
    flag_kernel<T><<<1, kThreads, 0, stream>>>(
        in, out, signals, reinterpret_cast<uint4*>(device + layout.flag),
        flag_max / 16, rank, bytes / 16, state);
  } else {
    const cudaError_t e =
        cudaMemcpyAsync(host + layout.ce + rank * max_size, input, bytes,
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return e;
    const int blocks = bytes <= (128 << 10) ? 2 : 4;
    ce_kernel<T><<<blocks, kThreads, 0, stream>>>(
        in, out, signals,
        reinterpret_cast<const uint4*>(device + layout.ce +
                                       (rank ^ 1) * max_size),
        rank, bytes / 16, state);
  }
  return cudaGetLastError();
}

}  // namespace vllm
