// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Device helpers of the fused MQA top-k kernels, adapted from SGLang's
// sgl_kernel JIT headers (https://github.com/sgl-project/sglang/pull/47469,
// Apache-2.0). Helpers with an equivalent in DeepGEMM / CUTLASS / CuTe that
// generates the same SASS forward to it; the rest keeps the original PTX
// (each says why).
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
// The cute:: TMEM / UTCCP aliases deep_gemm/ptx/tcgen05.cuh needs.
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#define FMT_DEVICE __forceinline__ __device__

namespace fused_mqa_topk {

namespace device {

inline constexpr uint32_t kWarpThreads = 32u;

// volatile: deep_gemm::ptx::get_lane_idx() (not volatile) changes the SASS.
FMT_DEVICE uint32_t get_lane_id() {
  uint32_t lane_id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
  return lane_id;
}
template <std::integral T, std::integral U>
__forceinline__ __host__ __device__ constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

// Up to 16-byte vectors of 4-byte elements.
template <typename T, std::size_t N>
struct AlignedVector {
  static_assert((N & (N - 1)) == 0 && sizeof(T) == 4 && N <= 4);
  struct alignas(4 * N) Storage {
    uint32_t data[N];
  };
  FMT_DEVICE void load(const void* ptr, int64_t offset = 0) {
    m_storage = reinterpret_cast<const Storage*>(ptr)[offset];
  }
  FMT_DEVICE void store(void* ptr, int64_t offset = 0) const {
    reinterpret_cast<Storage*>(ptr)[offset] = m_storage;
  }
  FMT_DEVICE void fill(T value) {
#pragma unroll
    for (std::size_t i = 0; i < N; ++i)
      m_storage.data[i] = *reinterpret_cast<uint32_t*>(&value);
  }
  FMT_DEVICE T& operator[](std::size_t i) {
    return reinterpret_cast<T*>(&m_storage)[i];
  }
  FMT_DEVICE T operator[](std::size_t i) const {
    return reinterpret_cast<const T*>(&m_storage)[i];
  }
  Storage m_storage;
};

// f32x2 PTX: __ffma2_rn / __fmul2_rn / __fadd2_rn change the recovery's SASS.
namespace math {
FMT_DEVICE float2 fma_f32x2(float2 a, float2 b, float2 c) {
  float2 d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;"
      : "=l"(reinterpret_cast<uint64_t&>(d))
      : "l"(reinterpret_cast<const uint64_t&>(a)),
        "l"(reinterpret_cast<const uint64_t&>(b)),
        "l"(reinterpret_cast<const uint64_t&>(c)));
  return d;
}
FMT_DEVICE float2 mul_f32x2(float2 a, float2 b) {
  float2 d;
  asm("mul.rn.f32x2 %0, %1, %2;"
      : "=l"(reinterpret_cast<uint64_t&>(d))
      : "l"(reinterpret_cast<const uint64_t&>(a)),
        "l"(reinterpret_cast<const uint64_t&>(b)));
  return d;
}
FMT_DEVICE float2 add_f32x2(float2 a, float2 b) {
  float2 d;
  asm("add.rn.f32x2 %0, %1, %2;"
      : "=l"(reinterpret_cast<uint64_t&>(d))
      : "l"(reinterpret_cast<const uint64_t&>(a)),
        "l"(reinterpret_cast<const uint64_t&>(b)));
  return d;
}
}  // namespace math

namespace warp {
inline constexpr uint32_t kFullMask = 0xffffffffu;
using ::fused_mqa_topk::device::get_lane_id;
FMT_DEVICE bool elect_one_lane() { return cute::elect_one_sync(); }
template <typename T>
FMT_DEVICE T reduce_max(T v) {
  if constexpr (std::is_same_v<T, float>)
    return deep_gemm::ptx::reduce_max_sync(v);
  else
    return __reduce_max_sync(kFullMask, v);
}
// DeepGEMM has no redux.sync min.
FMT_DEVICE float reduce_min(float v) {
  float r;
  asm volatile("redux.sync.min.f32 %0, %1, %2;"
               : "=f"(r)
               : "f"(v), "r"(kFullMask));
  return r;
}
template <typename T>
FMT_DEVICE T inclusive_sum(T v, uint32_t lane = get_lane_id()) {
#pragma unroll
  for (uint32_t d = 1; d < kWarpThreads; d *= 2) {
    const T n = __shfl_up_sync(kFullMask, v, d, kWarpThreads);
    if (lane >= d) v += n;
  }
  return v;
}
}  // namespace warp

namespace ptx {
using Bar = cutlass::arch::ClusterTransactionBarrier;
template <typename T>
FMT_DEVICE uint32_t to_shared(T* p) {
  return cute::cast_smem_ptr_to_uint(p);
}
FMT_DEVICE void mbar_init(uint64_t* b, uint32_t n) { Bar::init(b, n); }
FMT_DEVICE void mbar_arrive(uint64_t* b) { Bar::arrive(b); }
FMT_DEVICE void mbar_arrive_expect_tx(uint64_t* b, uint32_t bytes) {
  Bar::arrive_and_expect_tx(b, bytes);
}
FMT_DEVICE void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  Bar::expect_transaction(b, bytes);
}
// No suspend hint, a one-instruction spin: a C++ loop around Bar::try_wait
// changes the SASS.
FMT_DEVICE void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tWAIT_%=: mbarrier.try_wait.parity.shared.b64 p, "
      "[%0], %1;\n\t@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity));
}
FMT_DEVICE void fence_barrier_init() { cutlass::arch::fence_barrier_init(); }
FMT_DEVICE void fence_proxy_async_shared() {
  cutlass::arch::fence_view_async_shared();
}
FMT_DEVICE void bar_sync(uint32_t id, uint32_t n) {
  deep_gemm::ptx::sync_aligned(n, id);
}
template <uint32_t kNumRegs>
FMT_DEVICE void setmaxnreg_inc() {
  cutlass::arch::warpgroup_reg_alloc<kNumRegs>();
}
template <uint32_t kNumRegs>
FMT_DEVICE void setmaxnreg_dec() {
  cutlass::arch::warpgroup_reg_dealloc<kNumRegs>();
}
FMT_DEVICE void prefetch_tensormap(const void* t) {
  cute::prefetch_tma_descriptor(static_cast<const cute::TmaDescriptor*>(t));
}
FMT_DEVICE void cp_async_bulk_g2s(void* dst, const void* src, uint32_t bytes,
                                  uint64_t* b) {
  cute::SM90_BULK_COPY_G2S::copy(src, b, dst, static_cast<int32_t>(bytes));
}
FMT_DEVICE void cp_async_bulk_prefetch_l2(const void* src, uint32_t bytes) {
  cute::SM90_BULK_COPY_G2S::PREFETCH::copy(src, static_cast<int32_t>(bytes));
}
// .shared::cta without a cache hint: cute::SM90_TMA_LOAD_{2,3}D
// (.shared::cluster, .L2::cache_hint) changes the SASS.
FMT_DEVICE void cp_async_bulk_tensor_2d(uint32_t dst, const CUtensorMap* t,
                                        int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(t), "r"(x), "r"(y), "r"(to_shared(bar))
      : "memory");
}
FMT_DEVICE void cp_async_bulk_tensor_3d(uint32_t dst, const CUtensorMap* t,
                                        int32_t x, int32_t y, int32_t z,
                                        uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(t), "r"(x), "r"(y), "r"(z), "r"(to_shared(bar))
      : "memory");
}

// clang-format off
#define TMEM_LD_REGS_4 "%0, %1, %2, %3"
#define TMEM_LD_REGS_16 TMEM_LD_REGS_4 ", %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15"
#define TMEM_LD_REGS_32 TMEM_LD_REGS_16 ", %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
#define TMEM_LD_REGS_64 TMEM_LD_REGS_32 ", %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
#define TMEM_LD_OPS_4(r, o) "=r"((r)[o]), "=r"((r)[(o) + 1]), "=r"((r)[(o) + 2]), "=r"((r)[(o) + 3])
#define TMEM_LD_OPS_16(r, o) TMEM_LD_OPS_4(r, o), TMEM_LD_OPS_4(r, (o) + 4), TMEM_LD_OPS_4(r, (o) + 8), TMEM_LD_OPS_4(r, (o) + 12)
#define TMEM_LD_OPS_32(r, o) TMEM_LD_OPS_16(r, o), TMEM_LD_OPS_16(r, (o) + 16)
#define TMEM_LD_OPS_64(r, o) TMEM_LD_OPS_32(r, o), TMEM_LD_OPS_32(r, (o) + 32)
#define TMEM_LD(n) if constexpr (N == n) asm volatile("tcgen05.ld.sync.aligned.32x32b.x" #n ".b32 {" TMEM_LD_REGS_##n "}, [%" #n "];" : TMEM_LD_OPS_##n(r, 0) : "r"(taddr));
// clang-format on

struct TCGen05 {
  FMT_DEVICE static void fence_before_thread_sync() {
    deep_gemm::ptx::tcgen05_before_thread_sync();
  }
  FMT_DEVICE static void fence_after_thread_sync() {
    deep_gemm::ptx::tcgen05_after_thread_sync();
  }
  FMT_DEVICE static void wait_ld() {
    cutlass::arch::fence_view_async_tmem_load();
  }
  // One tcgen05.ld per width (deep_gemm::ptx::tmem_load_32dp32b splits it).
  template <int N>
  FMT_DEVICE static void ld_32x32b(uint32_t taddr, void* dst) {
    static_assert(N == 4 || N == 16 || N == 32 || N == 64);
    const auto r = static_cast<uint32_t*>(dst);
    TMEM_LD(4) TMEM_LD(16) TMEM_LD(32) TMEM_LD(64)
  }
  FMT_DEVICE static void alloc(uint32_t* dst, uint32_t cols) {
    cute::TMEM::Allocator1Sm().allocate(static_cast<int>(cols), dst);
  }
  FMT_DEVICE static void dealloc(uint32_t taddr, uint32_t cols) {
    cute::TMEM::Allocator1Sm().free(taddr, static_cast<int>(cols));
  }
  // The caller is the elected lane (cutlass::arch::umma_arrive elects again).
  FMT_DEVICE static void commit(uint64_t* bar) {
    deep_gemm::ptx::umma_arrive_no_elect(*reinterpret_cast<Bar*>(bar));
  }
  FMT_DEVICE static void mma_f8_ss(uint32_t d, uint64_t a, uint64_t b,
                                   uint32_t idesc, uint32_t acc) {
    deep_gemm::ptx::SM100_MMA_F8F6F4_SS::fma(a, b, d, acc,
                                             uint64_t{idesc} << 32);
  }

  // UMMA descriptors with the bit layouts of cute::UMMA::SmemDescriptor /
  // InstrDescriptor; the kernels use .advance_bytes() / .bits on them.
  union SmemDesc {
    struct {
      uint64_t start_address : 14, : 2, leading_byte_offset : 14, : 2,
          stride_byte_offset : 14, version : 2, : 13, layout_type : 3;
    };
    struct {
      uint32_t lo_bits, hi_bits;
    };
    uint64_t bits;
    FMT_DEVICE SmemDesc advance_bytes(uint32_t bytes) const {
      SmemDesc desc = *this;
      desc.lo_bits += bytes >> 4;
      return desc;
    }
  };
  FMT_DEVICE static SmemDesc make_desc_kmajor_sw128(const void* smem) {
    SmemDesc desc{};
    desc.start_address = to_shared(smem) >> 4;
    desc.stride_byte_offset = 1024 >> 4;
    desc.version = 1;
    desc.layout_type = 2;
    return desc;
  }
  union InstrDesc {
    struct {
      uint32_t : 4, c_format : 2, : 1, a_format : 3, b_format : 3, : 4,
          n_dim : 6, : 1, m_dim : 5, : 3;
    };
    uint32_t bits;
  };
  template <uint32_t M, uint32_t N>
  FMT_DEVICE static InstrDesc make_instr_desc_f8() {  // e4m3 x e4m3 -> f32
    InstrDesc desc{};
    desc.c_format = 1;
    desc.n_dim = N >> 3;
    desc.m_dim = M >> 4;
    return desc;
  }
};

#undef TMEM_LD
#undef TMEM_LD_OPS_64
#undef TMEM_LD_OPS_32
#undef TMEM_LD_OPS_16
#undef TMEM_LD_OPS_4
#undef TMEM_LD_REGS_64
#undef TMEM_LD_REGS_32
#undef TMEM_LD_REGS_16
#undef TMEM_LD_REGS_4

}  // namespace ptx

}  // namespace device

}  // namespace fused_mqa_topk
