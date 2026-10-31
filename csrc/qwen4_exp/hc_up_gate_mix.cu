// Fused HyperConnection up projection + gate mix for Qwen4Exp on SM100.
//
// out[m, c] = sum_s sigmoid(bf16(lowrank[m] . w_up[s * H + c])) *
//             xn[m, s * H + c] / HC
//
// with HC = 4, H = 2560 and rank R = 320. The [M, HC * H] gate is never
// written to global memory. Results are bitwise equal to the unfused
// F.linear + hc_gate_mix chain: the gate is rounded to bf16, the sigmoid and
// the stream sum (streams 0..3, fp32 FMAs) follow the Triton kernel.
//
// Persistent kernel, one CTA per SM. An output tile is 128 rows x BC hidden
// columns; its GEMM tile has N = HC * BC columns, gathered from the HC
// streams of w_up by HC TMA loads per stage, so all streams of an output
// column sit in one TMEM accumulator row.
//   warp 0: TMA producer (lowrank 128 x 64 and w_up HC x (BC x 64) per stage)
//   warp 1: tcgen05.mma issuer, fp32 accumulators in TMEM, 4 stages
//   warp 2: TMEM allocator
//   warps 4..11: epilogue, two warpgroups of one 16-column chunk each.
// BC = 32 with 4 accumulator stages was the fastest of BC 32 / 64, 1 / 2
// epilogue warpgroups, 3 / 4 / 6 smem stages and 2 / 4 accumulator stages on
// B300.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "../core/registration.h"
#include "../torch_utils.h"

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

#include <algorithm>
#include <limits>

namespace {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

constexpr uint32_t HC = 4, H = 2560, R = 320;
constexpr uint32_t BLOCK_M = 128, BLOCK_K = 64, UMMA_K = 16, BC = 32;
constexpr uint32_t BLOCK_N = HC * BC, NUM_K_BLOCKS = R / BLOCK_K;
constexpr uint32_t NUM_N_BLOCKS = H / BC;
constexpr uint32_t kStages = 4, kAccStages = 4,
                   kTmemCols = kAccStages * BLOCK_N;
constexpr uint32_t kEpiWarp0 = 4, kEpiWarpGroups = 2;
constexpr uint32_t kThreads = 128 + 128 * kEpiWarpGroups;
constexpr uint32_t kChunk = 16;  // output columns per epilogue step

struct SharedStorage {
  alignas(1024) cutlass::bfloat16_t a[kStages][BLOCK_M * BLOCK_K];
  alignas(1024) cutlass::bfloat16_t b[kStages][BLOCK_N * BLOCK_K];
  Barrier full[kStages], empty[kStages];
  Barrier tmem_full[kAccStages], tmem_empty[kAccStages];
  uint32_t tmem_base;
};

// K-major operand with 128B swizzle: 8-row atoms of 128 bytes (BLOCK_K bf16),
// stacked 1024 bytes apart along M/N, advanced by k elements along K.
__device__ __forceinline__ uint64_t make_sw128_desc(const void* smem,
                                                    uint32_t k) {
  cute::UMMA::SmemDescriptor desc;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ =
      static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  desc.start_address_ = static_cast<uint16_t>(
      (cute::cast_smem_ptr_to_uint(smem) + k * sizeof(cutlass::bfloat16_t)) >>
      4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = (8 * BLOCK_K * sizeof(cutlass::bfloat16_t)) >> 4;
  desc.leading_byte_offset_ = 0;
  return desc.desc_;
}

__device__ __forceinline__ void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

// The instructions Triton emits for hc_gate_mix: sigmoid as
// 1 / (1 + ex2(-x * log2(e))) and the mean as a division by HC.
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float sigmoid(float x) {
  float e;
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(x * -1.44269504088896341f));
  return div_full(1.0f, 1.0f + e);
}

__global__ void __launch_bounds__(kThreads, 1)
    hc_up_gate_mix_kernel(const __grid_constant__ CUtensorMap tm_lowrank,
                          const __grid_constant__ CUtensorMap tm_w_up,
                          const __nv_bfloat16* __restrict__ xn,
                          int64_t xn_stride, __nv_bfloat16* __restrict__ out,
                          int64_t out_stride, int M) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  auto& sm = *reinterpret_cast<SharedStorage*>(smem_raw);
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t num_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
  const uint32_t num_tiles = num_m_blocks * NUM_N_BLOCKS;

  if (warp == 0 && cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(&tm_lowrank);
    cute::prefetch_tma_descriptor(&tm_w_up);
  } else if (warp == 1 && cute::elect_one_sync()) {
    for (uint32_t i = 0; i < kStages; ++i) {
      sm.full[i].init(1);
      sm.empty[i].init(1);
    }
    for (uint32_t i = 0; i < kAccStages; ++i) {
      sm.tmem_full[i].init(1);
      sm.tmem_empty[i].init(128 * kEpiWarpGroups);
    }
    cutlass::arch::fence_barrier_init();
  } else if (warp == 2) {
    cute::TMEM::Allocator1Sm().allocate(kTmemCols, &sm.tmem_base);
  }
  __syncthreads();
  const uint32_t tmem_base = sm.tmem_base;
  // lowrank and xn are produced by the preceding kernels.
  cudaGridDependencySynchronize();

  // Tiles are M-fastest so that concurrent CTAs share a w_up tile.
  if (warp == 0) {
    if (cute::elect_one_sync()) {
      uint32_t stage = 0, phase = 0;
      for (uint32_t t = blockIdx.x; t < num_tiles; t += gridDim.x) {
        const int32_t m0 = (t % num_m_blocks) * BLOCK_M;
        const int32_t c0 = (t / num_m_blocks) * BC;
        for (uint32_t kb = 0; kb < NUM_K_BLOCKS; ++kb) {
          sm.empty[stage].wait(phase ^ 1);
          auto* full = reinterpret_cast<uint64_t*>(&sm.full[stage]);
          const int32_t k0 = kb * BLOCK_K;
          cute::SM90_TMA_LOAD_2D::copy(
              &tm_lowrank, full,
              static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
              sm.a[stage], k0, m0);
  #pragma unroll
          for (uint32_t s = 0; s < HC; ++s) {
            cute::SM90_TMA_LOAD_2D::copy(
                &tm_w_up, full,
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                sm.b[stage] + s * BC * BLOCK_K, k0, int32_t(s * H) + c0);
          }
          sm.full[stage].arrive_and_expect_tx(sizeof(sm.a[0]) +
                                              sizeof(sm.b[0]));
          stage = (stage + 1) % kStages;
          phase ^= stage == 0;
        }
      }
    }
    __syncwarp();
  } else if (warp == 1) {
    constexpr auto instr = cute::UMMA::make_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float, BLOCK_M, BLOCK_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    const uint64_t idesc = cute::UMMA::make_runtime_instr_desc<>(instr);
    uint32_t stage = 0, phase = 0, iter = 0;
    for (uint32_t t = blockIdx.x; t < num_tiles; t += gridDim.x, ++iter) {
      const uint32_t acc = iter % kAccStages;
      sm.tmem_empty[acc].wait(((iter / kAccStages) & 1) ^ 1);
      tcgen05_after_thread_sync();
      for (uint32_t kb = 0; kb < NUM_K_BLOCKS; ++kb) {
        sm.full[stage].wait(phase);
        tcgen05_after_thread_sync();
        // fma and umma_arrive are called by the whole warp and both elect
        // lane 0, so each commit tracks the MMAs issued before it.
  #pragma unroll
        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
          cute::SM100_MMA_F16BF16_SS<
              cutlass::bfloat16_t, cutlass::bfloat16_t, float, BLOCK_M, BLOCK_N,
              cute::UMMA::Major::K,
              cute::UMMA::Major::K>::fma(make_sw128_desc(sm.a[stage],
                                                         k * UMMA_K),
                                         make_sw128_desc(sm.b[stage],
                                                         k * UMMA_K),
                                         tmem_base + acc * BLOCK_N,
                                         (kb | k) > 0, idesc);
        }
        cutlass::arch::umma_arrive(
            reinterpret_cast<uint64_t*>(&sm.empty[stage]));
        if (kb == NUM_K_BLOCKS - 1) {
          cutlass::arch::umma_arrive(
              reinterpret_cast<uint64_t*>(&sm.tmem_full[acc]));
        }
        __syncwarp();
        stage = (stage + 1) % kStages;
        phase ^= stage == 0;
      }
    }
  } else if (warp >= kEpiWarp0) {
    // Each warpgroup mixes one 16-column chunk of every tile, one row per
    // thread. A warp may only access its own 32-lane quarter of TMEM.
    static_assert(BC == kChunk * kEpiWarpGroups);
    const uint32_t quarter = warp % 4;
    const uint32_t j = (warp - kEpiWarp0) / 4 * kChunk;
    uint32_t iter = 0;
    for (uint32_t t = blockIdx.x; t < num_tiles; t += gridDim.x, ++iter) {
      const uint32_t acc = iter % kAccStages;
      const int row = (t % num_m_blocks) * BLOCK_M + quarter * 32 + lane;
      const uint32_t c = (t / num_m_blocks) * BC + j;
      const bool live = row < M;
      // xn does not depend on the GEMM: load it before waiting.
      const __nv_bfloat16* xn_row =
          xn + int64_t(live ? row : 0) * xn_stride + c;
      uint4 x[HC][2];
  #pragma unroll
      for (uint32_t s = 0; s < HC; ++s) {
        const auto* p = reinterpret_cast<const uint4*>(xn_row + s * H);
        x[s][0] = p[0];
        x[s][1] = p[1];
      }
      sm.tmem_full[acc].wait((iter / kAccStages) & 1);
      tcgen05_after_thread_sync();
      uint32_t g[HC][kChunk];
  #pragma unroll
      for (uint32_t s = 0; s < HC; ++s) {
        uint32_t* v = g[s];
        cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x::copy(
            tmem_base + ((quarter * 32) << 16) + acc * BLOCK_N + s * BC + j,
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10],
            v[11], v[12], v[13], v[14], v[15]);
      }
      cutlass::arch::fence_view_async_tmem_load();
      if (live) {
        float o[kChunk] = {};
  #pragma unroll
        for (uint32_t s = 0; s < HC; ++s) {
          const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x[s]);
  #pragma unroll
          for (uint32_t i = 0; i < kChunk / 2; ++i) {
            const float2 xf = __bfloat1622float2(x2[i]);
            const float g0 = __bfloat162float(
                __float2bfloat16(__uint_as_float(g[s][2 * i])));
            const float g1 = __bfloat162float(
                __float2bfloat16(__uint_as_float(g[s][2 * i + 1])));
            o[2 * i] = fmaf(sigmoid(g0), xf.x, o[2 * i]);
            o[2 * i + 1] = fmaf(sigmoid(g1), xf.y, o[2 * i + 1]);
          }
        }
        uint4 y[2];
        auto* y2 = reinterpret_cast<__nv_bfloat162*>(y);
  #pragma unroll
        for (uint32_t i = 0; i < kChunk / 2; ++i) {
          y2[i] = __floats2bfloat162_rn(div_full(o[2 * i], HC),
                                        div_full(o[2 * i + 1], HC));
        }
        auto* p = reinterpret_cast<uint4*>(out + int64_t(row) * out_stride + c);
        p[0] = y[0];
        p[1] = y[1];
      }
      // Releasing the accumulator before the math above measured 2-4% slower.
      tcgen05_before_thread_sync();
      sm.tmem_empty[acc].arrive();
    }
  }
  __syncthreads();
  if (warp == 2) cute::TMEM::Allocator1Sm().free(tmem_base, kTmemCols);
#endif
}

PFN_cuTensorMapEncodeTiled_v12000 get_encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    const cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &status);
    STD_TORCH_CHECK(err == cudaSuccess && status == cudaDriverEntryPointSuccess,
                    "cuTensorMapEncodeTiled is unavailable");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

// K-major [rows, R] bf16 operand, row stride `ld` elements, box BLOCK_K x
// box_rows, 128B swizzle.
CUtensorMap make_tensor_map(const void* ptr, uint64_t rows, uint64_t ld,
                            uint32_t box_rows) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {R, rows};
  const cuuint64_t strides[1] = {ld * sizeof(__nv_bfloat16)};
  const cuuint32_t box[2] = {BLOCK_K, box_rows};
  const cuuint32_t elem_strides[2] = {1, 1};
  const CUresult res = get_encode_fn()(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims,
      strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(res == CUDA_SUCCESS,
                  "cuTensorMapEncodeTiled failed: ", static_cast<int>(res));
  return map;
}

}  // namespace

void qwen4_exp_hc_up_gate_mix(torch::stable::Tensor& out,
                              torch::stable::Tensor const& lowrank,
                              torch::stable::Tensor const& xn,
                              torch::stable::Tensor const& w_up) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(lowrank.dim() == 2 && xn.dim() == 2 && w_up.dim() == 2 &&
                  out.dim() == 2);
  const int64_t M = lowrank.size(0);
  STD_TORCH_CHECK(lowrank.size(1) == R && w_up.size(0) == HC * H &&
                      w_up.size(1) == R && xn.size(0) == M &&
                      xn.size(1) == HC * H && out.size(0) == M &&
                      out.size(1) == H,
                  "expected hc_count 4, hidden size 2560 and rank 320");
  STD_TORCH_CHECK(lowrank.scalar_type() == ScalarType::BFloat16 &&
                  xn.scalar_type() == ScalarType::BFloat16 &&
                  w_up.scalar_type() == ScalarType::BFloat16 &&
                  out.scalar_type() == ScalarType::BFloat16);
  STD_TORCH_CHECK(w_up.stride(0) == R && w_up.stride(1) == 1,
                  "w_up must be contiguous");
  // TMA needs 16-byte aligned rows; the epilogue uses 16-byte accesses.
  auto check_rows = [](const torch::stable::Tensor& t) {
    STD_TORCH_CHECK(t.stride(1) == 1 && t.stride(0) % 8 == 0 &&
                        reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                    "rows must be unit-stride and 16-byte aligned");
  };
  check_rows(lowrank);
  check_rows(xn);
  check_rows(out);
  STD_TORCH_CHECK(M <= std::numeric_limits<int>::max());
  if (M == 0) return;

  const torch::stable::accelerator::DeviceGuard device_guard(
      lowrank.get_device_index());
  const cudaDeviceProp* prop = get_device_prop();
  STD_TORCH_CHECK(prop->major == 10, "requires an SM100-family GPU");

  const CUtensorMap tm_lowrank =
      make_tensor_map(lowrank.data_ptr(), M, lowrank.stride(0), BLOCK_M);
  const CUtensorMap tm_w_up = make_tensor_map(w_up.data_ptr(), HC * H, R, BC);

  constexpr size_t smem = sizeof(SharedStorage) + 1024;
  STD_TORCH_CHECK(
      cudaFuncSetAttribute(hc_up_gate_mix_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem)) == cudaSuccess);
  const int64_t num_tiles = (M + BLOCK_M - 1) / BLOCK_M * NUM_N_BLOCKS;

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {};
  config.gridDim =
      dim3(std::min<int64_t>(num_tiles, prop->multiProcessorCount));
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = smem;
  config.stream = get_current_cuda_stream(lowrank.get_device_index());
  config.attrs = attrs;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &config, hc_up_gate_mix_kernel, tm_lowrank, tm_w_up,
      reinterpret_cast<const __nv_bfloat16*>(xn.data_ptr()), xn.stride(0),
      reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()), out.stride(0),
      static_cast<int>(M));
  STD_TORCH_CHECK(err == cudaSuccess,
                  "hc_up_gate_mix launch failed: ", cudaGetErrorString(err));
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen4_exp_hc_up_gate_mix", TORCH_BOX(&qwen4_exp_hc_up_gate_mix));
}
