// Inkling2's sequence-parallel reduce-scatter over a symmetric (MNNVL) buffer:
// every rank's buffer starts with per-CTA barrier flags, followed by its
// partial-sum rows and the all-gather destination of the conv + mix kernel
// (conv_mix.cu).
//
// The reduce-scatter pulls one shard of rows (and the conv taps' halo rows
// before it) from every rank and sums them in rank order in fp32 (rounded
// once to bf16), the same arithmetic as the two-shot all-reduce.

#include <cuda_bf16.h>

#include "../torch_utils.h"

namespace vllm {
namespace inkling2 {
namespace {

// Inkling2's hidden size 4608 in 16-byte chunks.
constexpr int kRowVecs = 576;
constexpr int kMaxWorld = 8;
constexpr int kMaxCtas = 1024;

__device__ __forceinline__ void storeRelease(uint32_t* ptr, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(ptr), "r"(v)
               : "memory");
}

__device__ __forceinline__ uint32_t loadAcquire(uint32_t const* ptr) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
  return v;
}

struct Comm {
  char* const* bases;  // every rank's symmetric buffer
  uint32_t* epochs;    // this rank's barrier count per CTA
  int rank;
  int world;
};

// CTA b of every rank waits until CTA b of every other rank arrived; earlier
// writes of this rank (and of the kernels before it) are visible to them.
__device__ void barrier(Comm const& c) {
  __shared__ uint32_t epoch;
  if (threadIdx.x == 0) {
    epoch = c.epochs[blockIdx.x] + 1;
    c.epochs[blockIdx.x] = epoch;
  }
  __syncthreads();
  if (threadIdx.x < c.world) {
    auto flags = [&](int r) {
      return reinterpret_cast<uint32_t*>(c.bases[r]) + blockIdx.x * kMaxWorld;
    };
    __threadfence_system();
    storeRelease(flags(threadIdx.x) + c.rank, epoch);
    uint32_t const* mine = flags(c.rank) + threadIdx.x;
    while (static_cast<int>(loadAcquire(mine) - epoch) < 0) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ float2 unpack(uint32_t w) {
  return make_float2(__uint_as_float(w << 16),
                     __uint_as_float(w & 0xffff0000u));
}

__device__ __forceinline__ uint32_t pack(float2 v) {
  __nv_bfloat162 h = __float22bfloat162_rn(v);
  return *reinterpret_cast<uint32_t*>(&h);
}

// The ranks' chunks summed in rank order in fp32, rounded once.
template <int W>
__device__ __forceinline__ uint4 rankSum(uint4 const (&v)[W]) {
  uint32_t out[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 acc = unpack((&v[0].x)[k]);
#pragma unroll
    for (int r = 1; r < W; ++r) {
      float2 const x = unpack((&v[r].x)[k]);
      acc.x = __fadd_rn(acc.x, x.x);
      acc.y = __fadd_rn(acc.y, x.y);
    }
    out[k] = pack(acc);
  }
  return make_uint4(out[0], out[1], out[2], out[3]);
}

struct ReduceScatterParams {
  Comm comm;
  int64_t srcOffset;  // partial rows in the symmetric buffers
  int first;          // rows [first, first + rows) go to out
  int rows;
  uint4* out;
};

template <int W, int U>
__global__ void __launch_bounds__(512)
    reduceScatterKernel(ReduceScatterParams p) {
  barrier(p.comm);
  uint4 const* src[W];
#pragma unroll
  for (int r = 0; r < W; ++r)
    src[r] = reinterpret_cast<uint4 const*>(p.comm.bases[r] + p.srcOffset);

  int64_t const begin = int64_t(p.first) * kRowVecs;
  int64_t const n = int64_t(p.rows) * kRowVecs;
  int64_t const stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t base = blockIdx.x * blockDim.x + threadIdx.x; base < n;
       base += stride * U) {
    uint4 v[U][W];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      int64_t const i = base + u * stride;
      if (i < n) {
#pragma unroll
        for (int r = 0; r < W; ++r) v[u][r] = __ldcg(src[r] + begin + i);
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      int64_t const i = base + u * stride;
      if (i < n) p.out[i] = rankSum<W>(v[u]);
    }
  }
}

}  // namespace
}  // namespace inkling2
}  // namespace vllm

void inkling2_reduce_scatter(int64_t bases, torch::stable::Tensor& epochs,
                             int64_t rank, int64_t world, int64_t src_offset,
                             int64_t first, torch::stable::Tensor& out) {
  using namespace vllm::inkling2;
  int const device = out.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  STD_TORCH_CHECK(out.size(1) == kRowVecs * 8,
                  "Inkling2 reduce-scatter requires hidden size 4608");
  int const ctas = static_cast<int>(epochs.size(0));
  STD_TORCH_CHECK(ctas <= kMaxCtas);
  ReduceScatterParams p{};
  p.comm = Comm{reinterpret_cast<char* const*>(bases),
                static_cast<uint32_t*>(epochs.data_ptr()),
                static_cast<int>(rank), static_cast<int>(world)};
  p.srcOffset = src_offset;
  p.first = first;
  p.rows = out.size(0);
  p.out = static_cast<uint4*>(out.data_ptr());
  cudaStream_t const stream = get_current_cuda_stream(device);
  switch (world) {
    case 2:
      reduceScatterKernel<2, 4><<<ctas, 512, 0, stream>>>(p);
      break;
    case 4:
      reduceScatterKernel<4, 2><<<ctas, 512, 0, stream>>>(p);
      break;
    case 8:
      reduceScatterKernel<8, 1><<<ctas, 512, 0, stream>>>(p);
      break;
    default:
      STD_TORCH_CHECK(false, "Inkling2 reduce-scatter supports TP2/4/8");
  }
  cudaError_t const status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
