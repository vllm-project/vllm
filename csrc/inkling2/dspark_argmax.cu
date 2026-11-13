// Inkling2 DSpark greedy draft token: the first argmax of the softcapped
// logits softcap * tanh(bf16(base + bias) / softcap) over the TP-sharded
// vocabulary, in one kernel. Each CTA reduces 1024 columns of one row to
// (max, lowest global index) and multicasts the pair into its rank's slot of
// the row in every rank's Lamport buffer; the last local CTA of the row waits
// for every rank's slots of the row in its own buffer, merges them and resets
// them. A row's slots alternate between two buffers (per-row parity): a rank
// writes a row's next pairs only after it has read every rank's current ones.

#include "../torch_utils.h"
#include "../mnnvl_lamport.cuh"

namespace vllm {
namespace inkling2 {
namespace {

constexpr int kBlock = 1024;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kVec = kBlock / kThreads;
constexpr uint32_t kEmpty = vllm::mnnvl::utils::kNEGZERO_FP32;
constexpr uint32_t kParity = 1u << 16;

struct Params {
  uint4 const* base;
  uint4 const* bias;
  int64_t baseStride;  // in uint4
  int64_t biasStride;
  float softcap;
  int cols;
  int valid;
  int offset;
  uint2* mcast;
  uint2* local;
  uint32_t* rowState;  // arrivals | parity << 16
  int64_t bufferSlots;
  int rank;
  int world;
  int64_t* out;
};

__device__ __forceinline__ float f32(uint32_t bits) {
  return __uint_as_float(bits);
}

// The softcap as the Triton kernel of add_softcap compiles it: bf16 add,
// div.full, libdevice tanhf (its |x| < 0.6 polynomial or 1 - 2 / (e^2x + 1)
// with approximate ex2 and rcp), fp32 multiply, bf16 rounding.
__device__ __forceinline__ float softcapLogit(float x, float softcap) {
  float y, ax, small, x2, large, t, out;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(softcap));
  asm("abs.ftz.f32 %0, %1;" : "=f"(ax) : "f"(y));
  asm("mul.rn.f32 %0, %1, %1;" : "=f"(x2) : "f"(y));
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
      : "=f"(small)
      : "f"(f32(0x3C80F082)), "f"(x2), "f"(f32(0xBD563CAE)));
  asm("fma.rn.ftz.f32 %0, %0, %1, %2;"
      : "+f"(small)
      : "f"(x2), "f"(f32(0x3E085941)));
  asm("fma.rn.ftz.f32 %0, %0, %1, %2;"
      : "+f"(small)
      : "f"(x2), "f"(f32(0xBEAAA9ED)));
  asm("fma.rn.ftz.f32 %0, %0, %1, 0f00000000;" : "+f"(small) : "f"(x2));
  asm("fma.rn.ftz.f32 %0, %0, %1, %1;" : "+f"(small) : "f"(y));
  asm("mul.rn.f32 %0, %1, 0f4038AA3B;" : "=f"(t) : "f"(ax));
  asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(t));
  asm("add.rn.f32 %0, %0, 0f3F800000;" : "+f"(t));
  asm("rcp.approx.ftz.f32 %0, %0;" : "+f"(t));
  asm("fma.rn.ftz.f32 %0, %1, 0fC0000000, 0f3F800000;" : "=f"(large) : "f"(t));
  // The polynomial below 0.6 (and for NaN), 1.0 from 9.010913.
  uint32_t const ux = __float_as_uint(ax);
  uint32_t bits = __float_as_uint(small);
  if (ux >= 0x3F19999Au && ux <= 0x7F800000u)
    bits = (ux >= 0x41102CB4u ? 0x3F800000u : __float_as_uint(large)) |
           (__float_as_uint(y) & 0x80000000u);
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(f32(bits)), "f"(softcap));
  return __bfloat162float(__float2bfloat16_rn(out));
}

// (value, index) pairs: the larger value, the lower index on ties.
__device__ __forceinline__ void better(float& v, uint32_t& i, float ov,
                                       uint32_t oi) {
  if (ov > v || (ov == v && oi < i)) {
    v = ov;
    i = oi;
  }
}

__device__ __forceinline__ void ctaArgmax(float& v, uint32_t& i, float* sValue,
                                          uint32_t* sIndex) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1)
    better(v, i, __shfl_xor_sync(~0u, v, m), __shfl_xor_sync(~0u, i, m));
  int const warp = threadIdx.x / 32;
  if (threadIdx.x % 32 == 0) {
    sValue[warp] = v;
    sIndex[warp] = i;
  }
  __syncthreads();
  v = sValue[0];
  i = sIndex[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) better(v, i, sValue[w], sIndex[w]);
}

__global__ void __launch_bounds__(kThreads) dsparkArgmaxKernel(Params p) {
  __shared__ float sValue[kWarps];
  __shared__ uint32_t sIndex[kWarps];
  __shared__ uint32_t sState;
  int const row = blockIdx.y;
  int const blocks = gridDim.x;
  int const col0 = blockIdx.x * kBlock + threadIdx.x * kVec;

  float best = -INFINITY;
  uint32_t bestIndex = p.offset + col0;
  if (col0 < p.cols) {
    uint4 a = p.base[row * p.baseStride + col0 / kVec];
    uint4 const b = p.bias[row * p.biasStride + col0 / kVec];
    uint32_t* aw = reinterpret_cast<uint32_t*>(&a);
    uint32_t const* bw = reinterpret_cast<uint32_t const*>(&b);
#pragma unroll
    for (int k = 0; k < kVec / 2; ++k)
      asm("add.rn.bf16x2 %0, %0, %1;" : "+r"(aw[k]) : "r"(bw[k]));
    __nv_bfloat16 const* x = reinterpret_cast<__nv_bfloat16 const*>(&a);
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      float const v = softcapLogit(__bfloat162float(x[k]), p.softcap);
      // As greedy_argmax: NaN counts as -inf.
      if (col0 + k < p.valid && !isnan(v) && v > best) {
        best = v;
        bestIndex = p.offset + col0 + k;
      }
    }
  }
  ctaArgmax(best, bestIndex, sValue, sIndex);

  if (threadIdx.x == 0) {
    uint32_t const state = atomicAdd(&p.rowState[row], 1u);
    int64_t const slot = (state & kParity ? p.bufferSlots : 0) +
                         (int64_t(row) * p.world + p.rank) * blocks +
                         blockIdx.x;
    // An empty slot holds -0.0 words; indices are below 2^24.
    uint32_t const bits = __float_as_uint(best);
    p.mcast[slot] = make_uint2(bits == kEmpty ? 0u : bits, bestIndex);
    sState = state;
  }
  __syncthreads();
  uint32_t const state = sState;
  if ((state & (kParity - 1)) != uint32_t(blocks - 1)) return;

  // The row's last local CTA: merge every rank's blocks.
  uint2* slots = p.local + (state & kParity ? p.bufferSlots : 0) +
                 int64_t(row) * p.world * blocks;
  best = -INFINITY;
  bestIndex = ~0u;
  for (int s = threadIdx.x; s < p.world * blocks; s += kThreads) {
    uint2 pair;
    do {
      asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];"
                   : "=r"(pair.x), "=r"(pair.y)
                   : "l"(slots + s)
                   : "memory");
    } while (pair.x == kEmpty || pair.y == kEmpty);
    better(best, bestIndex, __uint_as_float(pair.x), pair.y);
    slots[s] = make_uint2(kEmpty, kEmpty);
  }
  ctaArgmax(best, bestIndex, sValue, sIndex);
  if (threadIdx.x == 0) {
    p.out[row] = bestIndex;
    p.rowState[row] = (state & kParity) ^ kParity;
  }
}

}  // namespace
}  // namespace inkling2
}  // namespace vllm

void inkling2_dspark_argmax(torch::stable::Tensor const& base,
                            torch::stable::Tensor const& bias, double softcap,
                            int64_t valid, int64_t offset,
                            int64_t multicast_ptr, int64_t local_ptr,
                            torch::stable::Tensor& row_state, int64_t rank,
                            int64_t world, torch::stable::Tensor& out) {
  using namespace vllm::inkling2;
  int const device = out.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  int const rows = base.size(0);
  int const cols = base.size(1);
  int64_t const maxRows = row_state.size(0);
  STD_TORCH_CHECK(rows <= maxRows && out.size(0) == rows);
  STD_TORCH_CHECK(base.stride(1) == 1 && bias.stride(1) == 1 &&
                  base.stride(0) % kVec == 0 && bias.stride(0) % kVec == 0 &&
                  cols % kVec == 0 &&
                  reinterpret_cast<uintptr_t>(base.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0);
  STD_TORCH_CHECK(0 < valid && valid <= cols && offset + cols < (1 << 24));
  if (rows == 0) return;
  int const blocks = (cols + kBlock - 1) / kBlock;
  STD_TORCH_CHECK(blocks < int(kParity));
  // Only the rows' last CTAs wait (on other ranks): every row's other CTAs
  // must be able to run beside them.
  static int const maxResident = [] {
    int perSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, dsparkArgmaxKernel,
                                                  kThreads, 0);
    return perSm * get_device_prop()->multiProcessorCount;
  }();
  STD_TORCH_CHECK(rows < maxResident);
  Params p{};
  p.base = static_cast<uint4 const*>(base.data_ptr());
  p.bias = static_cast<uint4 const*>(bias.data_ptr());
  p.baseStride = base.stride(0) / kVec;
  p.biasStride = bias.stride(0) / kVec;
  p.softcap = softcap;
  p.cols = cols;
  p.valid = valid;
  p.offset = offset;
  p.mcast = reinterpret_cast<uint2*>(multicast_ptr);
  p.local = reinterpret_cast<uint2*>(local_ptr);
  p.rowState = static_cast<uint32_t*>(row_state.data_ptr());
  p.bufferSlots = maxRows * world * blocks;
  p.rank = rank;
  p.world = world;
  p.out = static_cast<int64_t*>(out.data_ptr());
  dsparkArgmaxKernel<<<dim3(blocks, rows), kThreads, 0,
                       get_current_cuda_stream(device)>>>(p);
  cudaError_t const status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
