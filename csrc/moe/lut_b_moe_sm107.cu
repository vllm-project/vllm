// Grouped MoE GEMM for Rubin (SM107) LUT-B weights with MXFP8 activations.
//
// C[sid, n] = sum_k A[sid / top_k, k] * W[expert, n, k] (* topk_weight[sid])
//
// A is MXFP8 (E4M3 with a UE8M0 scale per 32 K). Each weight is a 3-bit index
// into an eight-entry E4M3 codebook shared by 8 output channels x 64 K, times
// a UE8M0 block scale. The decode happens inside the tensor core through
// tcgen05.mma.kind::mxf8f6f4.block_scale.decompress::lut::b; weights are never
// expanded in memory.
//
// This is a deliberately simple kernel: one CTA computes a 128 x 128 output
// tile of one expert using rows gathered through moe_align_block_size. All
// four warps load a 3-stage cp.async pipeline, one thread issues the MMAs, and
// all four warps run the epilogue. It favors clarity over peak throughput.
//
// Weights are pre-arranged by Python (see lut_b_moe.py) into the exact shared
// memory image per (expert, 128 N, 128 K) tile:
//   [6144 B packed indices][512 B B scales][512 B codebooks]

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <optional>
#include <type_traits>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "torch_utils.h"

namespace vllm {
namespace lut_b_moe {

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 128;
constexpr int kThreads = 128;
constexpr int kStages = 3;

constexpr int kABytes = kBlockM * kBlockK;               // E4M3 activations
constexpr int kMetaBytes = 512;                          // one 32x128b copy
constexpr int kBIndexBytes = kBlockN * kBlockK * 3 / 8;  // 3-bit indices
constexpr int kBTileBytes = kBIndexBytes + 2 * kMetaBytes;
constexpr int kStageBytes = kABytes + kMetaBytes + kBTileBytes;
constexpr int kSmemBytes = kStages * kStageBytes + 1024;  // + alignment slack
// TMEM: 128 accumulator columns, 4 A-scale, 4 B-scale, 4 codebook columns.
constexpr int kTmemCols = 256;
constexpr int kTmemAScale = kBlockN;
constexpr int kTmemBScale = kBlockN + 4;
constexpr int kTmemBook = kBlockN + 8;

static_assert(kStageBytes % 128 == 0);
static_assert(kStages * kStageBytes >= kBlockM * kBlockN * 2,
              "the epilogue reuses the pipeline buffers");

__device__ __forceinline__ uint32_t smem_addr(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Shared memory matrix descriptor, no swizzle.
__device__ __forceinline__ uint64_t smem_desc(void const* p, int lbo, int sbo) {
  return uint64_t(smem_addr(p) >> 4) | (uint64_t(lbo >> 4) << 16) |
         (uint64_t(sbo >> 4) << 32) | (uint64_t(1) << 46);
}

__device__ __forceinline__ void cp_async_16(void* dst, void const* src,
                                            bool valid) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_addr(dst)),
      "l"(src), "r"(valid ? 16 : 0)
      : "memory");
}

__device__ __forceinline__ void cp_async_4(void* dst, void const* src,
                                           bool valid) {
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_addr(dst)),
      "l"(src), "r"(valid ? 4 : 0)
      : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
  asm volatile(
      "{.reg .pred p; LAB_WAIT: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
      "@!p bra LAB_WAIT;}" ::"r"(smem_addr(bar)),
      "r"(phase)
      : "memory");
}

// Copy a 512-byte uint32[32][4] metadata block into four TMEM columns,
// replicated across all four 32-lane partitions.
__device__ __forceinline__ void copy_meta(uint32_t tmem, void const* src) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(tmem),
               "l"(smem_desc(src, 16, 128))
               : "memory");
}

// The two K64 halves of a K128 tile share one 48-byte compressed B row per
// output channel; the first fills the B collector and the second reuses it.
__device__ __forceinline__ void mma_lut_fill(uint32_t d, uint64_t a, uint64_t b,
                                             uint32_t book, uint32_t idesc,
                                             uint32_t sfa, uint32_t sfb,
                                             uint32_t accumulate) {
  asm volatile(
      "{.reg .pred p; setp.ne.b32 p, %7, 0; "
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b."
      "block32.collector::b::fill [%0], %1, %2, [%3], %4, [%5], [%6], p;}" ::
          "r"(d),
      "l"(a), "l"(b), "r"(book), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accumulate)
      : "memory");
}

__device__ __forceinline__ void mma_lut_lastuse(uint32_t d, uint64_t a,
                                                uint64_t b, uint32_t book,
                                                uint32_t idesc, uint32_t sfa,
                                                uint32_t sfb) {
  asm volatile(
      "{.reg .pred p; setp.ne.b32 p, %7, 0; "
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b."
      "block32.collector::b::lastuse [%0], %1, %2, [%3], %4, [%5], [%6], p;}" ::
          "r"(d),
      "l"(a), "l"(b), "r"(book), "r"(idesc), "r"(sfa), "r"(sfb), "r"(1u)
      : "memory");
}

template <typename OutT>
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
  } else {
    __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
  }
}

template <typename OutT>
__global__ void __launch_bounds__(kThreads)
    lut_b_moe_gemm_kernel(uint8_t const* __restrict__ a,
                          uint8_t const* __restrict__ a_scales,
                          uint8_t const* __restrict__ b, OutT* __restrict__ c,
                          float const* __restrict__ topk_weights,
                          int32_t const* __restrict__ sorted_token_ids,
                          int32_t const* __restrict__ expert_ids,
                          int32_t const* __restrict__ num_tokens_post_padded,
                          int num_valid_tokens, int top_k, int N, int K) {
  int const tid = threadIdx.x;
  int const m_block = blockIdx.x;
  int const n_tile = blockIdx.y;
  if (m_block * kBlockM >= *num_tokens_post_padded) return;

  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  __shared__ __align__(8) uint64_t mma_done[kStages];
  __shared__ int32_t row_ids[kBlockM];
  __shared__ uint32_t tmem_base;

  // Each thread owns one row of the M tile.
  int32_t const sid = sorted_token_ids[m_block * kBlockM + tid];
  bool const row_valid = sid < num_valid_tokens;
  row_ids[tid] = row_valid ? sid : -1;
  int const expert = expert_ids[m_block];
  int const k_tiles = K / kBlockK;
  int64_t const ldc = N;
  OutT* const c_tile = c + n_tile * kBlockN;

  if (expert < 0) {
    // Tokens routed to an expert that is not local to this rank.
    if (row_valid) {
      uint4* dst = reinterpret_cast<uint4*>(c_tile + sid * ldc);
      for (int i = 0; i < kBlockN * int(sizeof(OutT)) / 16; ++i)
        dst[i] = make_uint4(0, 0, 0, 0);
    }
    return;
  }

  if (tid == 0) {
    for (int s = 0; s < kStages; ++s)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(
          smem_addr(&mma_done[s])));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 32) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(smem_addr(&tmem_base)),
        "n"(kTmemCols)
        : "memory");
  }

  uint8_t const* a_row =
      a + int64_t(row_valid ? sid / top_k : 0) * K;  // E4M3 bytes
  uint8_t const* a_scale_row =
      a_scales + int64_t(row_valid ? sid / top_k : 0) * (K / 32);
  uint8_t const* b_tiles =
      b + (int64_t(expert) * (N / kBlockN) + n_tile) * k_tiles * kBTileBytes;

  auto load_stage = [&](int stage, int k_tile) {
    uint8_t* st = smem + stage * kStageBytes;
    // A in the canonical K-major no-swizzle layout: 8-row x 16-byte core
    // matrices, 128 bytes apart along K and 1024 bytes apart along M.
    uint8_t* a_dst = st + (tid / 8) * 1024 + (tid % 8) * 16;
    uint8_t const* a_src = a_row + k_tile * kBlockK;
#pragma unroll
    for (int chunk = 0; chunk < kBlockK / 16; ++chunk)
      cp_async_16(a_dst + chunk * 128, a_src + chunk * 16, row_valid);
    // A scales: row r's four K32 scales land in lane r % 32, column r / 32.
    cp_async_4(st + kABytes + (tid % 32) * 16 + (tid / 32) * 4,
               a_scale_row + k_tile * 4, row_valid);
    // B indices, scales and codebooks are already in their smem image.
    uint8_t* b_dst = st + kABytes + kMetaBytes;
    uint8_t const* b_src = b_tiles + int64_t(k_tile) * kBTileBytes;
    for (int i = tid; i < kBTileBytes / 16; i += kThreads)
      cp_async_16(b_dst + i * 16, b_src + i * 16, true);
  };

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < k_tiles) load_stage(s, s);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }

  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  uint32_t const tmem = tmem_base;

  // Instruction descriptor: E4M3 A and decoded B, FP32 accumulate, UE8M0
  // block32 scales, M = 128, N = kBlockN.
  uint32_t const idesc = (uint32_t(kBlockN / 8) << 17) | (1u << 23) |
                         (uint32_t(kBlockM / 16) << 24) | (1u << 31);
  // The second K64 half uses scale byte 2 of each 4-byte A/B scale word.
  uint32_t const idesc_hi = idesc | (2u << 29) | (2u << 4);

  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    int const stage = k_tile % kStages;
    asm volatile("cp.async.wait_group %0;" ::"n"(kStages - 2) : "memory");
    // Make this thread's cp.async writes visible to the tensor core.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    if (tid == 0) {
      uint8_t* st = smem + stage * kStageBytes;
      uint8_t* b_st = st + kABytes + kMetaBytes;
      // Metadata TMEM columns are reused every K tile; tcgen05.cp and
      // tcgen05.mma issued by one thread execute in order.
      copy_meta(tmem + kTmemAScale, st + kABytes);
      copy_meta(tmem + kTmemBScale, b_st + kBIndexBytes);
      copy_meta(tmem + kTmemBook, b_st + kBIndexBytes + kMetaBytes);
      uint64_t const a_lo = smem_desc(st, 128, 1024);
      uint64_t const a_hi = smem_desc(st + 512, 128, 1024);
      // Compressed B rows are 48 bytes; descriptor bit 53 selects the
      // second K64 half of each row.
      uint64_t const b_lo = smem_desc(b_st, 128, 384);
      uint64_t const b_hi = b_lo | (uint64_t(1) << 53);
      mma_lut_fill(tmem, a_lo, b_lo, tmem + kTmemBook, idesc,
                   tmem + kTmemAScale, tmem + kTmemBScale, k_tile != 0);
      mma_lut_lastuse(tmem, a_hi, b_hi, tmem + kTmemBook + 2, idesc_hi,
                      tmem + kTmemAScale, tmem + kTmemBScale);
      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(
              smem_addr(&mma_done[stage]))
          : "memory");
    }

    // Refill the stage read by the previous K tile once its MMAs finish.
    int const next = k_tile + kStages - 1;
    if (next < k_tiles) {
      if (k_tile > 0) {
        int const prev = k_tile - 1;
        mbar_wait(&mma_done[prev % kStages], (prev / kStages) & 1);
      }
      load_stage(next % kStages, next);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }

  // Completion of the last commit implies completion of all earlier MMAs.
  int const last = k_tiles - 1;
  mbar_wait(&mma_done[last % kStages], (last / kStages) & 1);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  // Epilogue: TMEM lane r holds output row r. Convert to the output dtype in
  // a swizzled smem tile, then write each 256-byte row contiguously.
  float const route_weight =
      (topk_weights != nullptr && row_valid) ? topk_weights[sid] : 1.0f;
  OutT* scratch = reinterpret_cast<OutT*>(smem);
  uint32_t const lane_base = uint32_t((tid / 32) * 32) << 16;
#pragma unroll
  for (int n = 0; n < kBlockN; n += 32) {
    uint32_t v[32];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,"
        "%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,"
        "%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]),
          "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]),
          "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]),
          "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]),
          "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]),
          "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]),
          "=r"(v[30]), "=r"(v[31])
        : "r"(tmem + lane_base + n)
        : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int j = 0; j < 32; j += 8) {
      uint4 packed;
      packed.x = pack2<OutT>(__uint_as_float(v[j + 0]) * route_weight,
                             __uint_as_float(v[j + 1]) * route_weight);
      packed.y = pack2<OutT>(__uint_as_float(v[j + 2]) * route_weight,
                             __uint_as_float(v[j + 3]) * route_weight);
      packed.z = pack2<OutT>(__uint_as_float(v[j + 4]) * route_weight,
                             __uint_as_float(v[j + 5]) * route_weight);
      packed.w = pack2<OutT>(__uint_as_float(v[j + 6]) * route_weight,
                             __uint_as_float(v[j + 7]) * route_weight);
      // XOR-swizzle 16-byte chunks to spread row-strided stores over banks.
      int const col = (n + j) ^ ((tid % 8) * 8);
      *reinterpret_cast<uint4*>(scratch + tid * kBlockN + col) = packed;
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();

  if (tid < 32) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1; "
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::"r"(
            tmem),
        "n"(kTmemCols)
        : "memory");
  }

  constexpr int kChunksPerRow = kBlockN * int(sizeof(OutT)) / 16;
  for (int i = tid; i < kBlockM * kChunksPerRow; i += kThreads) {
    int const row = i / kChunksPerRow;
    int const col = (i % kChunksPerRow) * 8;
    int32_t const out_row = row_ids[row];
    if (out_row < 0) continue;
    uint4 const packed = *reinterpret_cast<uint4 const*>(
        scratch + row * kBlockN + (col ^ ((row % 8) * 8)));
    *reinterpret_cast<uint4*>(c_tile + out_row * ldc + col) = packed;
  }
}

}  // namespace lut_b_moe
}  // namespace vllm

void lut_b_moe_gemm(torch::stable::Tensor const& a,
                    torch::stable::Tensor const& a_scales,
                    torch::stable::Tensor const& b, torch::stable::Tensor& c,
                    std::optional<torch::stable::Tensor> topk_weights,
                    torch::stable::Tensor const& sorted_token_ids,
                    torch::stable::Tensor const& expert_ids,
                    torch::stable::Tensor const& num_tokens_post_padded,
                    int64_t top_k) {
  using namespace vllm::lut_b_moe;
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(a.scalar_type() == ScalarType::Float8_e4m3fn,
                  "lut_b_moe_gemm: a must be float8_e4m3fn");
  STD_TORCH_CHECK(a_scales.scalar_type() == ScalarType::Byte,
                  "lut_b_moe_gemm: a_scales must be uint8 (UE8M0)");
  STD_TORCH_CHECK(b.scalar_type() == ScalarType::Byte,
                  "lut_b_moe_gemm: b must be uint8");
  STD_TORCH_CHECK(a.dim() == 2 && a.is_contiguous(),
                  "lut_b_moe_gemm: a must be a contiguous 2D tensor");
  STD_TORCH_CHECK(c.dim() == 2 && c.is_contiguous(),
                  "lut_b_moe_gemm: c must be a contiguous 2D tensor");
  int64_t const K = a.size(1);
  int64_t const N = c.size(1);
  STD_TORCH_CHECK(K % kBlockK == 0 && N % kBlockN == 0,
                  "lut_b_moe_gemm: N and K must be multiples of 128, got N=", N,
                  " K=", K);
  STD_TORCH_CHECK(a_scales.dim() == 2 && a_scales.is_contiguous() &&
                      a_scales.size(0) == a.size(0) &&
                      a_scales.size(1) == K / 32,
                  "lut_b_moe_gemm: a_scales must be [M, K / 32]");
  STD_TORCH_CHECK(
      b.dim() == 4 && b.is_contiguous() && b.size(1) == N / kBlockN &&
          b.size(2) == K / kBlockK && b.size(3) == kBTileBytes,
      "lut_b_moe_gemm: b must be [E, N / 128, K / 128, ", kBTileBytes, "]");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(a.const_data_ptr()) % 16 == 0 &&
                      reinterpret_cast<uintptr_t>(b.const_data_ptr()) % 16 == 0,
                  "lut_b_moe_gemm: a and b must be 16-byte aligned");
  STD_TORCH_CHECK(sorted_token_ids.scalar_type() == ScalarType::Int &&
                      expert_ids.scalar_type() == ScalarType::Int &&
                      num_tokens_post_padded.scalar_type() == ScalarType::Int,
                  "lut_b_moe_gemm: routing tensors must be int32");
  STD_TORCH_CHECK(expert_ids.size(0) * kBlockM >= sorted_token_ids.size(0),
                  "lut_b_moe_gemm: routing must be aligned to block size 128");
  float const* topk_weights_ptr = nullptr;
  if (topk_weights.has_value()) {
    STD_TORCH_CHECK(topk_weights->scalar_type() == ScalarType::Float &&
                        topk_weights->is_contiguous() &&
                        topk_weights->numel() == c.size(0),
                    "lut_b_moe_gemm: topk_weights must be contiguous float32 "
                    "with one entry per row of c");
    topk_weights_ptr = topk_weights->const_data_ptr<float>();
  }

  int const device = a.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  cudaDeviceProp const* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 10 && properties->minor == 7,
                  "lut_b_moe_gemm requires an SM107 (Rubin) GPU");
  if (expert_ids.size(0) == 0) return;

  // Blocks past num_tokens_post_padded exit immediately.
  dim3 const grid(expert_ids.size(0), N / kBlockN);
  cudaStream_t const stream = get_current_cuda_stream(device);
  auto launch = [&](auto* out) {
    using OutT = std::remove_pointer_t<decltype(out)>;
    auto kernel = lut_b_moe_gemm_kernel<OutT>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    kernel<<<grid, kThreads, kSmemBytes, stream>>>(
        static_cast<uint8_t const*>(a.const_data_ptr()),
        a_scales.const_data_ptr<uint8_t>(), b.const_data_ptr<uint8_t>(), out,
        topk_weights_ptr, sorted_token_ids.const_data_ptr<int32_t>(),
        expert_ids.const_data_ptr<int32_t>(),
        num_tokens_post_padded.const_data_ptr<int32_t>(),
        static_cast<int>(c.size(0)), static_cast<int>(top_k),
        static_cast<int>(N), static_cast<int>(K));
  };
  if (c.scalar_type() == ScalarType::BFloat16) {
    launch(reinterpret_cast<__nv_bfloat16*>(c.mutable_data_ptr()));
  } else if (c.scalar_type() == ScalarType::Half) {
    launch(reinterpret_cast<__half*>(c.mutable_data_ptr()));
  } else {
    STD_TORCH_CHECK(false, "lut_b_moe_gemm: c must be bfloat16 or float16");
  }
  cudaError_t const err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "lut_b_moe_gemm launch failed: ", cudaGetErrorString(err));
}
