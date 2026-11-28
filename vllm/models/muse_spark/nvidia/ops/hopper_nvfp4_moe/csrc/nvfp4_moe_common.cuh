// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>

#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ < 900 || !defined(__CUDA_ARCH_FEAT_SM90_ALL))
  #error \
      "Muse Spark Hopper NVFP4 MoE kernels require Hopper SM90a (compute_90a / sm_90a)."
#endif

namespace vllm::muse_spark_hopper_nvfp4_moe {

constexpr uint32_t kFullWarpMask = 0xffffffffu;

// ---------------------------------------------------------------------------
// Shared-memory addressing, warp sync, and register-file reconfiguration
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool elect_one() {
  uint32_t pred;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "elect.sync _|p, 0xffffffff;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(pred));
  return pred != 0;
}

__device__ __forceinline__ void named_bar(int id, int count) {
  asm volatile("bar.sync %0, %1;" : : "r"(id), "r"(count) : "memory");
}

template <int Regs>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(Regs) : "memory");
}

template <int Regs>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(Regs) : "memory");
}

template <int Stages>
__device__ __forceinline__ void ring_next(int& slot, uint32_t& phase) {
  if (++slot == Stages) {
    slot = 0;
    phase ^= 1u;
  }
}

// ---------------------------------------------------------------------------
// Global and shared memory load/store and memory ordering primitives
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T ld_acquire(const T* ptr) {
  static_assert(sizeof(T) == 4, "ld_acquire expects a 32-bit scalar type");
  T val;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
               : "=r"(val)
               : "l"(ptr)
               : "memory");
  return val;
}

__device__ __forceinline__ void st_release(int* ptr, int val) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
}

__device__ __forceinline__ void red_add_u32(uint32_t* ptr, uint32_t val) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
}

__device__ __forceinline__ int ld_global_cg_s32(const int* ptr) {
  int val;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(val) : "l"(ptr));
  return val;
}

__device__ __forceinline__ float ld_global_cg_f32(const float* ptr) {
  float val;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
  return val;
}

__device__ __forceinline__ uint4 ld_global_cg_u4(const void* ptr) {
  uint4 val;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(ptr));
  return val;
}

__device__ __forceinline__ int4 ld_global_nc_i4(const int4* ptr) {
  int4 val;
  asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(ptr));
  return val;
}

__device__ __forceinline__ uint2 lds64(uint32_t addr) {
  uint2 val;
  asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];"
               : "=r"(val.x), "=r"(val.y)
               : "r"(addr)
               : "memory");
  return val;
}

__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 val;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "r"(addr)
               : "memory");
  return val;
}

__device__ __forceinline__ float ld_shared_f32(uint32_t addr) {
  float val;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(val) : "r"(addr) : "memory");
  return val;
}

__device__ __forceinline__ void st_shared_f32(uint32_t addr, float val) {
  asm volatile("st.shared.f32 [%0], %1;" : : "r"(addr), "f"(val) : "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
}

// ---------------------------------------------------------------------------
// Shared-memory mbarrier primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(bar), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(bar), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :
               : "r"(bar)
               : "memory");
}

__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"
               :
               : "r"(bar)
               : "memory");
}

__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  if (mbar_try(bar, parity)) {
    return;
  }
  while (!mbar_try(bar, parity)) {
  }
}

__device__ __forceinline__ void mbar_wait_spin(uint32_t bar, uint32_t parity) {
  while (!mbar_try(bar, parity)) {
  }
}

// Upper bound (ns) a thread may stay suspended inside one mbarrier.try_wait
// probe before re-polling the phase.
constexpr uint32_t kMbarSuspendHintNs = 10000000;

__device__ __forceinline__ bool mbar_try_hint(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(ok)
      : "r"(bar), "r"(parity), "n"(kMbarSuspendHintNs)
      : "memory");
  return ok != 0;
}

__device__ __forceinline__ void mbar_wait_hint(uint32_t bar, uint32_t parity) {
  if (!mbar_try_hint(bar, parity)) {
    while (!mbar_try_hint(bar, parity)) {
    }
  }
}

// ---------------------------------------------------------------------------
// L2 cache policies, TMA bulk copies, and cp.async helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
               : "=l"(pol));
  return pol;
}

__device__ __forceinline__ uint64_t policy_evict_normal() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;"
               : "=l"(pol));
  return pol;
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src,
                                         uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
      :
      : "r"(dst), "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void bulk_g2s_hint(uint32_t dst, const void* src,
                                              uint32_t bytes, uint32_t bar,
                                              uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
      "L2::cache_hint [%0], [%1], %2, [%3], %4;"
      :
      : "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(policy)
      : "memory");
}

__device__ __forceinline__ void cp_async_cg_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(dst), "l"(src)
               : "memory");
}

__device__ __forceinline__ void cp_async_cg_16_zfill(uint32_t dst,
                                                     const void* src,
                                                     uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(dst), "l"(src), "r"(src_bytes)
               : "memory");
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, const void* tmap,
                                            int c0, int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global."
      "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(dst), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1),
        "r"(bar)
      : "memory");
}

__device__ __forceinline__ void tma_store_2d(const void* tmap, uint32_t src,
                                             int col, int row) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
      "[%0, {%2, %3}], [%1];"
      :
      : "l"(reinterpret_cast<uint64_t>(tmap)), "r"(src), "r"(col), "r"(row)
      : "memory");
}

__device__ __forceinline__ void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;" : : : "memory");
}

__device__ __forceinline__ void tma_store_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;" : : : "memory");
}

__device__ __forceinline__ void tma_store_wait_all() {
  asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
}

// ---------------------------------------------------------------------------
// Programmatic Dependent Launch (PDL)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" : : : "memory");
}

__device__ __forceinline__ void pdl_launch() {
  asm volatile("griddepcontrol.launch_dependents;" : : : "memory");
}

template <typename Kernel, typename... Args>
inline cudaError_t launch_pdl(Kernel kernel, dim3 grid, dim3 block, size_t smem,
                              cudaStream_t stream, Args&&... args) {
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  const cudaLaunchConfig_t cfg = {grid, block, smem, stream, attr, 1};
  return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

// ---------------------------------------------------------------------------
// Shared-memory matrix descriptor, ldmatrix, and stmatrix
// ---------------------------------------------------------------------------

// Hopper GMMA shared-memory matrix descriptor: start address >> 4 in bits
// [0, 14), leading byte offset field 1 (unused for swizzled K-major tiles),
// stride byte offset >> 4 in bits [32, 46), base offset 0 (the swizzled tile
// base must be 1024-byte aligned), and swizzle mode in bits [62, 64).
template <int SBO, int Swizzle>
__device__ __forceinline__ uint64_t desc_swizzle(uint32_t addr) {
  return static_cast<uint64_t>((addr & 0x3FFFFu) >> 4) | (uint64_t{1} << 16) |
         (static_cast<uint64_t>(SBO >> 4) << 32) |
         (static_cast<uint64_t>(Swizzle) << 62);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t* regs) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
      : "r"(addr)
      : "memory");
}

__device__ __forceinline__ void stmatrix_x4_trans(uint32_t addr, uint32_t r0,
                                                  uint32_t r1, uint32_t r2,
                                                  uint32_t r3) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], "
      "{%1, %2, %3, %4};"
      :
      : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
      : "memory");
}

__device__ __forceinline__ void stmatrix_x2_trans(uint32_t addr, uint32_t r0,
                                                  uint32_t r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
               :
               : "r"(addr), "r"(r0), "r"(r1)
               : "memory");
}

// ---------------------------------------------------------------------------
// BF16 pack/unpack/arithmetic, SiLU-Mul, and FP4 E2M1 + E5M3 dequantization
// ---------------------------------------------------------------------------

__device__ __forceinline__ float bf16_lo_to_f32(uint32_t x) {
  return __uint_as_float(x << 16);
}

__device__ __forceinline__ float bf16_hi_to_f32(uint32_t x) {
  return __uint_as_float(x & 0xFFFF0000u);
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t broadcast_lo_u16(uint32_t v) {
  uint32_t d;
  asm("{\n"
      ".reg .b16 lo, hi;\n"
      "mov.b32 {lo, hi}, %1;\n"
      "mov.b32 %0, {lo, lo};\n"
      "}"
      : "=r"(d)
      : "r"(v));
  return d;
}

__device__ __forceinline__ uint32_t broadcast_hi_u16(uint32_t v) {
  uint32_t d;
  asm("{\n"
      ".reg .b16 lo, hi;\n"
      "mov.b32 {lo, hi}, %1;\n"
      "mov.b32 %0, {hi, hi};\n"
      "}"
      : "=r"(d)
      : "r"(v));
  return d;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ float silu_mul(float gate, float up) {
  return __fdividef(gate, 1.0f + __expf(-gate)) * up;
}

// Decode two packed E5M3 scales (code = e5 << 3 | m3, selected by hi_half in
// {0, 1}) into a BF16x2 pair 2^(e5 + 103) * (1 + m3 / 8). With the 2^-126
// placement of the FP4 codes in dequant_fp4x8 the product is the weight times
// 2^-7 (compensated by GS_FIX = 128 in the global scale). Only e5 <= 24 decodes
// to a finite BF16. Pass a runtime scale_mul=16 for single-IMAD codegen.
__device__ __forceinline__ uint32_t decode_scale_pair(
    uint32_t packed_scales, int hi_half, uint32_t scale_mul = 16u) {
  return __byte_perm(packed_scales, 0u, hi_half ? 0x4342u : 0x4140u) *
             scale_mul +
         0x73007300u;
}

__device__ __forceinline__ uint32_t
decode_scale_quad(uint2 packed_scales, int lane, uint32_t scale_mul = 16u) {
  return decode_scale_pair((lane & 2) ? packed_scales.y : packed_scales.x,
                           lane & 1, scale_mul);
}

__device__ __forceinline__ uint32_t scale_for_step(uint32_t lane_scales,
                                                   int step) {
  return __shfl_sync(kFullWarpMask, lane_scales, step, 4);
}

// Unpack 8 bit-permuted FP4 E2M1 weights and multiply by the two BF16 block
// scales in scale_pair, yielding 4 packed BF16x2 registers for MMA/WGMMA.
__device__ __forceinline__ uint4 dequant_fp4x8(uint32_t packed_fp4,
                                               uint32_t scale_pair) {
  constexpr uint32_t kFp4Bf16Mask = 0x81C081C0u;
  const uint32_t p0 = packed_fp4 & kFp4Bf16Mask;
  const uint32_t p1 =
      __funnelshift_l(packed_fp4, packed_fp4, 10) & kFp4Bf16Mask;
  const uint32_t p2 =
      __funnelshift_l(packed_fp4, packed_fp4, 13) & kFp4Bf16Mask;
  const uint32_t p3 = ((packed_fp4 & 0x001B001Bu) * 0x4108u) & kFp4Bf16Mask;
  const uint32_t sc_lo = broadcast_lo_u16(scale_pair);
  const uint32_t sc_hi = broadcast_hi_u16(scale_pair);
  return make_uint4(hmul2(p0, sc_lo), hmul2(p1, sc_hi), hmul2(p2, sc_lo),
                    hmul2(p3, sc_hi));
}

__device__ __forceinline__ void dequant_fp4x8(uint32_t packed_fp4,
                                              uint32_t scale_pair,
                                              uint32_t* out) {
  const uint4 v = dequant_fp4x8(packed_fp4, scale_pair);
  out[0] = v.x;
  out[1] = v.y;
  out[2] = v.z;
  out[3] = v.w;
}

// ---------------------------------------------------------------------------
// Warp MMA (m16n8k16) and Hopper SM90a RS-mode WGMMA (m64nNk16, f32.bf16.bf16)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" : : : "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" : : : "memory");
}

template <int Pending>
__device__ __forceinline__ void wgmma_wait() {
  static_assert(Pending >= 0 && Pending <= 7,
                "wgmma_wait pending group count must be in [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;" : : "n"(Pending) : "memory");
}

// RS-mode warpgroup MMA: D[64, N] += A_reg[64, 16] * B_smem[16, N] (K-major B).
// PTX RS operand list: {d0..d_{N/2-1}}, {a0..a3}, desc_b, 1, 1, 1, 0
// (scale_d=1, scale_a=1, scale_b=1, trans_b=0; no trans_a in RS mode).
template <int N>
__device__ __forceinline__ void wgmma_bf16_rs(float*, const uint32_t*,
                                              uint64_t) {
  static_assert(
      N != N,
      "Unsupported N for wgmma_bf16_rs<N> (supported: 16, 24, 32, 40, 48, 56, "
      "64, 80, 96, 112, 128, 144, 192, 224, 256)");
}

#define MUSE_HOPPER_NVFP4_WGMMA_D_8 "%0,%1,%2,%3,%4,%5,%6,%7"
#define MUSE_HOPPER_NVFP4_WGMMA_D_12 \
  MUSE_HOPPER_NVFP4_WGMMA_D_8 ",%8,%9,%10,%11"
#define MUSE_HOPPER_NVFP4_WGMMA_D_16 \
  MUSE_HOPPER_NVFP4_WGMMA_D_12 ",%12,%13,%14,%15"
#define MUSE_HOPPER_NVFP4_WGMMA_D_20 \
  MUSE_HOPPER_NVFP4_WGMMA_D_16 ",%16,%17,%18,%19"
#define MUSE_HOPPER_NVFP4_WGMMA_D_24 \
  MUSE_HOPPER_NVFP4_WGMMA_D_20 ",%20,%21,%22,%23"
#define MUSE_HOPPER_NVFP4_WGMMA_D_28 \
  MUSE_HOPPER_NVFP4_WGMMA_D_24 ",%24,%25,%26,%27"
#define MUSE_HOPPER_NVFP4_WGMMA_D_32 \
  MUSE_HOPPER_NVFP4_WGMMA_D_28 ",%28,%29,%30,%31"
#define MUSE_HOPPER_NVFP4_WGMMA_D_40 \
  MUSE_HOPPER_NVFP4_WGMMA_D_32 ",%32,%33,%34,%35,%36,%37,%38,%39"
#define MUSE_HOPPER_NVFP4_WGMMA_D_48 \
  MUSE_HOPPER_NVFP4_WGMMA_D_40 ",%40,%41,%42,%43,%44,%45,%46,%47"
#define MUSE_HOPPER_NVFP4_WGMMA_D_56 \
  MUSE_HOPPER_NVFP4_WGMMA_D_48 ",%48,%49,%50,%51,%52,%53,%54,%55"
#define MUSE_HOPPER_NVFP4_WGMMA_D_64 \
  MUSE_HOPPER_NVFP4_WGMMA_D_56 ",%56,%57,%58,%59,%60,%61,%62,%63"
#define MUSE_HOPPER_NVFP4_WGMMA_D_72 \
  MUSE_HOPPER_NVFP4_WGMMA_D_64 ",%64,%65,%66,%67,%68,%69,%70,%71"
#define MUSE_HOPPER_NVFP4_WGMMA_D_96 \
  MUSE_HOPPER_NVFP4_WGMMA_D_72       \
  ",%72,%73,%74,%75,%76,%77,%78,%79" \
  ",%80,%81,%82,%83,%84,%85,%86,%87" \
  ",%88,%89,%90,%91,%92,%93,%94,%95"
#define MUSE_HOPPER_NVFP4_WGMMA_D_112    \
  MUSE_HOPPER_NVFP4_WGMMA_D_96           \
  ",%96,%97,%98,%99,%100,%101,%102,%103" \
  ",%104,%105,%106,%107,%108,%109,%110,%111"
#define MUSE_HOPPER_NVFP4_WGMMA_D_128        \
  MUSE_HOPPER_NVFP4_WGMMA_D_112              \
  ",%112,%113,%114,%115,%116,%117,%118,%119" \
  ",%120,%121,%122,%123,%124,%125,%126,%127"

#define MUSE_HOPPER_NVFP4_WGMMA_O4(i) \
  "+f"(d[(i)]), "+f"(d[(i) + 1]), "+f"(d[(i) + 2]), "+f"(d[(i) + 3])
#define MUSE_HOPPER_NVFP4_WGMMA_O8(i) \
  MUSE_HOPPER_NVFP4_WGMMA_O4(i), MUSE_HOPPER_NVFP4_WGMMA_O4((i) + 4)
#define MUSE_HOPPER_NVFP4_WGMMA_O16(i) \
  MUSE_HOPPER_NVFP4_WGMMA_O8(i), MUSE_HOPPER_NVFP4_WGMMA_O8((i) + 8)
#define MUSE_HOPPER_NVFP4_WGMMA_O32(i) \
  MUSE_HOPPER_NVFP4_WGMMA_O16(i), MUSE_HOPPER_NVFP4_WGMMA_O16((i) + 16)

#define MUSE_HOPPER_NVFP4_WGMMA_OUT_8 MUSE_HOPPER_NVFP4_WGMMA_O8(0)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_12 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_8, MUSE_HOPPER_NVFP4_WGMMA_O4(8)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_16 MUSE_HOPPER_NVFP4_WGMMA_O16(0)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_20 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_16, MUSE_HOPPER_NVFP4_WGMMA_O4(16)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_24 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_16, MUSE_HOPPER_NVFP4_WGMMA_O8(16)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_28 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_24, MUSE_HOPPER_NVFP4_WGMMA_O4(24)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_32 MUSE_HOPPER_NVFP4_WGMMA_O32(0)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_40 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_32, MUSE_HOPPER_NVFP4_WGMMA_O8(32)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_48 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_32, MUSE_HOPPER_NVFP4_WGMMA_O16(32)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_56 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_48, MUSE_HOPPER_NVFP4_WGMMA_O8(48)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_64 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_32, MUSE_HOPPER_NVFP4_WGMMA_O32(32)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_72 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_64, MUSE_HOPPER_NVFP4_WGMMA_O8(64)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_96 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_64, MUSE_HOPPER_NVFP4_WGMMA_O32(64)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_112 \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_96, MUSE_HOPPER_NVFP4_WGMMA_O16(96)
#define MUSE_HOPPER_NVFP4_WGMMA_OUT_128                            \
  MUSE_HOPPER_NVFP4_WGMMA_OUT_64, MUSE_HOPPER_NVFP4_WGMMA_O32(64), \
      MUSE_HOPPER_NVFP4_WGMMA_O32(96)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && \
    defined(__CUDA_ARCH_FEAT_SM90_ALL)
  #define MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(N, M, A0, A1, A2, A3, B)      \
    template <>                                                                \
    __device__ __forceinline__ void wgmma_bf16_rs<N>(                          \
        float* d, const uint32_t* a, uint64_t desc_b) {                        \
      asm volatile("wgmma.mma_async.sync.aligned.m64n" #N                      \
                   "k16.f32.bf16.bf16 "                                        \
                   "{" MUSE_HOPPER_NVFP4_WGMMA_D_##M                           \
                   "},{"                                                       \
                   "%" #A0 ",%" #A1 ",%" #A2 ",%" #A3 "},%" #B ",1,1,1,0;\n"   \
                   : MUSE_HOPPER_NVFP4_WGMMA_OUT_##M                           \
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b)); \
    }
#else
  #define MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(N, M, A0, A1, A2, A3, B) \
    template <>                                                           \
    __device__ __forceinline__ void wgmma_bf16_rs<N>(                     \
        float* d, const uint32_t* a, uint64_t desc_b) {                   \
      (void)d;                                                            \
      (void)a;                                                            \
      (void)desc_b;                                                       \
    }
#endif

MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(16, 8, 8, 9, 10, 11, 12)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(24, 12, 12, 13, 14, 15, 16)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(32, 16, 16, 17, 18, 19, 20)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(40, 20, 20, 21, 22, 23, 24)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(48, 24, 24, 25, 26, 27, 28)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(56, 28, 28, 29, 30, 31, 32)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(64, 32, 32, 33, 34, 35, 36)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(80, 40, 40, 41, 42, 43, 44)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(96, 48, 48, 49, 50, 51, 52)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(112, 56, 56, 57, 58, 59, 60)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(128, 64, 64, 65, 66, 67, 68)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(144, 72, 72, 73, 74, 75, 76)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(192, 96, 96, 97, 98, 99, 100)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(224, 112, 112, 113, 114, 115, 116)
MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS(256, 128, 128, 129, 130, 131, 132)

#undef MUSE_HOPPER_NVFP4_DEFINE_WGMMA_BF16_RS
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_128
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_112
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_96
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_72
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_64
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_56
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_48
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_40
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_32
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_28
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_24
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_20
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_16
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_12
#undef MUSE_HOPPER_NVFP4_WGMMA_OUT_8
#undef MUSE_HOPPER_NVFP4_WGMMA_O32
#undef MUSE_HOPPER_NVFP4_WGMMA_O16
#undef MUSE_HOPPER_NVFP4_WGMMA_O8
#undef MUSE_HOPPER_NVFP4_WGMMA_O4
#undef MUSE_HOPPER_NVFP4_WGMMA_D_128
#undef MUSE_HOPPER_NVFP4_WGMMA_D_112
#undef MUSE_HOPPER_NVFP4_WGMMA_D_96
#undef MUSE_HOPPER_NVFP4_WGMMA_D_72
#undef MUSE_HOPPER_NVFP4_WGMMA_D_64
#undef MUSE_HOPPER_NVFP4_WGMMA_D_56
#undef MUSE_HOPPER_NVFP4_WGMMA_D_48
#undef MUSE_HOPPER_NVFP4_WGMMA_D_40
#undef MUSE_HOPPER_NVFP4_WGMMA_D_32
#undef MUSE_HOPPER_NVFP4_WGMMA_D_28
#undef MUSE_HOPPER_NVFP4_WGMMA_D_24
#undef MUSE_HOPPER_NVFP4_WGMMA_D_20
#undef MUSE_HOPPER_NVFP4_WGMMA_D_16
#undef MUSE_HOPPER_NVFP4_WGMMA_D_12
#undef MUSE_HOPPER_NVFP4_WGMMA_D_8

// Cached SM count per device (host side): grid sizes must not exceed it,
// because the persistent and split-K kernels spin on peer CTAs.
inline int get_device_sm_count(int dev_idx) {
  constexpr int kMaxCachedSmDevices = 64;
  static std::array<std::atomic<int>, kMaxCachedSmDevices> sm_counts{};
  const bool cacheable = dev_idx >= 0 && dev_idx < kMaxCachedSmDevices;
  if (cacheable) {
    const int cached = sm_counts[dev_idx].load(std::memory_order_relaxed);
    if (cached > 0) {
      return cached;
    }
  }
  int num_sms = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &num_sms, cudaDevAttrMultiProcessorCount, dev_idx));
  TORCH_CHECK(num_sms > 0, "invalid SM count ", num_sms, " on device ",
              dev_idx);
  if (cacheable) {
    sm_counts[dev_idx].store(num_sms, std::memory_order_relaxed);
  }
  return num_sms;
}

}  // namespace vllm::muse_spark_hopper_nvfp4_moe
