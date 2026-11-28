// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The inline-PTX wrappers and host-side driver helpers that the SM90 KDA
// chunked-prefill kernel (kda_prefill_sm90.cu) calls. vLLM has no shared SM90
// PTX layer (wgmma, TMA, mbarrier, bulk copies, griddepcontrol) outside
// CUTLASS, so these live next to the kernel. Only the instructions the kernel
// uses are here; each wrapper names its PTX instruction.
#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

namespace vllm {
namespace kda_sm90 {
namespace ptx {

// ---- shared-memory addresses, fences, grid dependency control ----

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// fence.proxy.async.shared::cta
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
// fence.proxy.async.global
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;\n" ::: "memory");
}
// fence.acq_rel.gpu
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
}
// fence.acquire.gpu (PTX ISA 8.6+)
__device__ __forceinline__ void fence_acquire_gpu() {
  asm volatile("fence.acquire.gpu;\n" ::: "memory");
}
// ld.relaxed.gpu.global.b32
__device__ __forceinline__ int ld_relaxed_gpu(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];\n"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
// st.release.gpu.global.b32
__device__ __forceinline__ void st_release_gpu(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v)
               : "memory");
}
// griddepcontrol.launch_dependents
__device__ __forceinline__ void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}
// griddepcontrol.wait
__device__ __forceinline__ void griddep_wait() {
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
}

// ---- math ----

// tanh.approx.f32
__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
// ex2.approx.ftz.f32
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// ---- mbarrier ----

// mbarrier.init.shared::cta.b64
__device__ __forceinline__ void mbarrier_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar),
               "r"(count)
               : "memory");
}
// fence.mbarrier_init.release.cluster
__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
// mbarrier.arrive.expect_tx.shared::cta.b64
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t bar,
                                                          uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar),
      "r"(bytes)
      : "memory");
}
// mbarrier.try_wait.parity.shared::cta.b64
__device__ __forceinline__ bool mbarrier_try_wait_parity(uint32_t bar,
                                                         uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], "
      "%2;\n selp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbarrier_wait_parity(uint32_t bar,
                                                     uint32_t parity) {
  while (!mbarrier_try_wait_parity(bar, parity)) {
  }
}

// ---- bulk and tensor (TMA) copies ----

// cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
__device__ __forceinline__ void bulk_copy_g2s(uint32_t dst_smem,
                                              const void* src, uint32_t bytes,
                                              uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n" ::"r"(dst_smem),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}
// cp.async.bulk.global.shared::cta.bulk_group
__device__ __forceinline__ void bulk_copy_s2g(void* dst, uint32_t src_smem,
                                              uint32_t bytes) {
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(
          dst),
      "r"(src_smem), "r"(bytes)
      : "memory");
}
// cp.async.bulk.commit_group
__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}
// cp.async.bulk.wait_group.read N
template <int N>
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
}
// cp.async.bulk.wait_group N
template <int N>
__device__ __forceinline__ void bulk_wait() {
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory");
}
// cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
__device__ __forceinline__ void tma_load_3d(uint32_t dst_smem, const void* tmap,
                                            int c0, int c1, int c2,
                                            uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
      "complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(dst_smem),
      "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
      : "memory");
}
// cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group
__device__ __forceinline__ void tma_store_3d(const void* tmap, int c0, int c1,
                                             int c2, uint32_t src_smem) {
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, "
      "%2, %3}], [%4];\n" ::"l"(tmap),
      "r"(c0), "r"(c1), "r"(c2), "r"(src_smem)
      : "memory");
}

// ---- warp-level matrix moves and MMA ----

// ldmatrix.sync.aligned.m8n8.x4.shared.b16
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4],
                                            uint32_t smem_addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr));
}
// ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4],
                                                  uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_addr));
}
// stmatrix.sync.aligned.m8n8.x4.shared.b16
__device__ __forceinline__ void stmatrix_x4(uint32_t smem_addr, uint32_t r0,
                                            uint32_t r1, uint32_t r2,
                                            uint32_t r3) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(
          smem_addr),
      "r"(r0), "r"(r1), "r"(r2), "r"(r3)
      : "memory");
}
// stmatrix.sync.aligned.m8n8.x4.trans.shared.b16
__device__ __forceinline__ void stmatrix_x4_trans(uint32_t smem_addr,
                                                  uint32_t r0, uint32_t r1,
                                                  uint32_t r2, uint32_t r3) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], "
      "{%1,%2,%3,%4};\n" ::"r"(smem_addr),
      "r"(r0), "r"(r1), "r"(r2), "r"(r3)
      : "memory");
}
// movmatrix.sync.aligned.m8n8.trans.b16
__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t a) {
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
               : "=r"(d)
               : "r"(a));
  return d;
}
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4],
                                                  const uint32_t (&a)[4],
                                                  uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 (fp32 bits read as tf32)
__device__ __forceinline__ void mma_m16n8k8_tf32(float (&d)[4],
                                                 const uint32_t (&a)[4],
                                                 uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---- wgmma ----

// wgmma.fence.sync.aligned
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
// wgmma.commit_group.sync.aligned
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// wgmma.wait_group.sync.aligned N
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
// Shared-memory matrix descriptor with 128B swizzle (PTX ISA, wgmma matrix
// descriptor format).
__device__ __forceinline__ uint64_t wgmma_desc_sw128(uint32_t smem_addr,
                                                     uint32_t lbo_bytes,
                                                     uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((smem_addr & 0x3FFFF) >> 4);
  d |= (uint64_t)((lbo_bytes & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((sbo_bytes & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
// Empty asm that pins a register value (keeps the compiler from moving an
// accumulator or a packed A operand across an async wgmma).
__device__ __forceinline__ void reg_fence(float& x) {
  asm volatile("" : "+f"(x)::"memory");
}
__device__ __forceinline__ void reg_fence_u32(uint32_t& x) {
  asm volatile("" : "+r"(x)::"memory");
}

// Operand lists of the wgmma wrappers below; undefined at the end.
#define KDA_WG_D8(c, i)                                                     \
  c(d[i]), c(d[i + 1]), c(d[i + 2]), c(d[i + 3]), c(d[i + 4]), c(d[i + 5]), \
      c(d[i + 6]), c(d[i + 7])
#define KDA_WG_D16(c) KDA_WG_D8(c, 0), KDA_WG_D8(c, 8)
#define KDA_WG_D32(c) KDA_WG_D16(c), KDA_WG_D8(c, 16), KDA_WG_D8(c, 24)
#define KDA_WG_D64(c)                                                  \
  KDA_WG_D32(c), KDA_WG_D8(c, 32), KDA_WG_D8(c, 40), KDA_WG_D8(c, 48), \
      KDA_WG_D8(c, 56)
#define KDA_WG_R16 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}"
#define KDA_WG_R32                                                          \
  "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19," \
  "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define KDA_WG_R64                                                          \
  "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19," \
  "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,"    \
  "%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,"    \
  "%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}"
#define KDA_WG_A "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])

// wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16. Template arguments:
// imm-trans-a TA / imm-trans-b TB (0: K-major, 1: MN-major). rs: A in
// registers; ss: A and B from shared-memory descriptors; scale_d 0 ignores the
// old accumulator; _z: scale-d 0 with an "=f" accumulator.

template <int TB = 0>
__device__ __forceinline__ void wgmma_m64n32k16_bf16_rs(float (&d)[16],
                                                        const uint32_t (&a)[4],
                                                        uint64_t bdesc,
                                                        int scale_d) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 " KDA_WG_R16
      ", {%16,%17,%18,%19}, %20, p, 1, 1, %22;\n}\n"
      : KDA_WG_D16("+f")
      : KDA_WG_A, "l"(bdesc), "r"(scale_d), "n"(TB));
}
template <int TB = 0>
__device__ __forceinline__ void wgmma_m64n32k16_bf16_rs_z(
    float (&d)[16], const uint32_t (&a)[4], uint64_t bdesc) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 " KDA_WG_R16
      ", {%16,%17,%18,%19}, %20, 0, 1, 1, %21;\n"
      : KDA_WG_D16("=f")
      : KDA_WG_A, "l"(bdesc), "n"(TB));
}
template <int TA = 0, int TB = 0>
__device__ __forceinline__ void wgmma_m64n32k16_bf16_ss(float (&d)[16],
                                                        uint64_t adesc,
                                                        uint64_t bdesc,
                                                        int scale_d) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 " KDA_WG_R16
      ", %16, %17, p, 1, 1, %19, %20;\n}\n"
      : KDA_WG_D16("+f")
      : "l"(adesc), "l"(bdesc), "r"(scale_d), "n"(TA), "n"(TB));
}
template <int TB = 0>
__device__ __forceinline__ void wgmma_m64n64k16_bf16_rs(float (&d)[32],
                                                        const uint32_t (&a)[4],
                                                        uint64_t bdesc,
                                                        int scale_d) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " KDA_WG_R32
      ", {%32,%33,%34,%35}, %36, p, 1, 1, %38;\n}\n"
      : KDA_WG_D32("+f")
      : KDA_WG_A, "l"(bdesc), "r"(scale_d), "n"(TB));
}
template <int TB = 0>
__device__ __forceinline__ void wgmma_m64n64k16_bf16_rs_z(
    float (&d)[32], const uint32_t (&a)[4], uint64_t bdesc) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " KDA_WG_R32
      ", {%32,%33,%34,%35}, %36, 0, 1, 1, %37;\n"
      : KDA_WG_D32("=f")
      : KDA_WG_A, "l"(bdesc), "n"(TB));
}
template <int TA = 0, int TB = 0>
__device__ __forceinline__ void wgmma_m64n64k16_bf16_ss(float (&d)[32],
                                                        uint64_t adesc,
                                                        uint64_t bdesc,
                                                        int scale_d) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " KDA_WG_R32
      ", %32, %33, p, 1, 1, %35, %36;\n}\n"
      : KDA_WG_D32("+f")
      : "l"(adesc), "l"(bdesc), "r"(scale_d), "n"(TA), "n"(TB));
}
template <int TB = 0>
__device__ __forceinline__ void wgmma_m64n128k16_bf16_rs(float (&d)[64],
                                                         const uint32_t (&a)[4],
                                                         uint64_t bdesc,
                                                         int scale_d) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " KDA_WG_R64
      ", {%64,%65,%66,%67}, %68, p, 1, 1, %70;\n}\n"
      : KDA_WG_D64("+f")
      : KDA_WG_A, "l"(bdesc), "r"(scale_d), "n"(TB));
}
template <int TB = 0>
__device__ __forceinline__ void wgmma_m64n128k16_bf16_rs_z(
    float (&d)[64], const uint32_t (&a)[4], uint64_t bdesc) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " KDA_WG_R64
      ", {%64,%65,%66,%67}, %68, 0, 1, 1, %69;\n"
      : KDA_WG_D64("=f")
      : KDA_WG_A, "l"(bdesc), "n"(TB));
}

#undef KDA_WG_D8
#undef KDA_WG_D16
#undef KDA_WG_D32
#undef KDA_WG_D64
#undef KDA_WG_R16
#undef KDA_WG_R32
#undef KDA_WG_R64
#undef KDA_WG_A

}  // namespace ptx

namespace cu {

// Raises a kernel's dynamic shared memory limit (launches above 48 KB).
template <typename Kernel>
inline cudaError_t max_dynamic_smem(Kernel* kernel, int bytes) {
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

// Launch with programmatic dependent launch (PDL); the kernel calls
// ptx::griddep_wait() before it reads the previous kernel's output.
template <typename... Params, typename... Args>
inline cudaError_t launch_pdl(void (*kernel)(Params...), dim3 grid, dim3 block,
                              size_t smem, cudaStream_t stream,
                              Args&&... args) {
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

// Driver entry points looked up through the runtime, so nothing links libcuda.
template <typename Fn>
inline Fn driver_entry_point(const char* name, unsigned version) {
  void* p = nullptr;
  cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
  cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, version,
                                                   cudaEnableDefault, &q);
#else
  (void)version;
  cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
  return e == cudaSuccess && q == cudaDriverEntryPointSuccess
             ? reinterpret_cast<Fn>(p)
             : nullptr;
}
inline PFN_cuLaunchKernel_v4000 launch_kernel_fn() {
  static const PFN_cuLaunchKernel_v4000 fn =
      driver_entry_point<PFN_cuLaunchKernel_v4000>("cuLaunchKernel", 4000);
  return fn;
}
using LaunchKernelExFn = CUresult (*)(const CUlaunchConfig*, CUfunction, void**,
                                      void**);
inline LaunchKernelExFn launch_kernel_ex_fn() {
  static const LaunchKernelExFn fn =
      driver_entry_point<LaunchKernelExFn>("cuLaunchKernelEx", 12000);
  return fn;
}
inline PFN_cuTensorMapEncodeTiled_v12000 encode_tiled_fn() {
  static const PFN_cuTensorMapEncodeTiled_v12000 fn =
      driver_entry_point<PFN_cuTensorMapEncodeTiled_v12000>(
          "cuTensorMapEncodeTiled", 12000);
  return fn;
}

// The driver function handle of a kernel in the current context.
template <typename Kernel>
inline CUfunction driver_function(Kernel* kernel) {
  cudaFunction_t f = nullptr;
  cudaGetFuncBySymbol(&f, reinterpret_cast<const void*>(kernel));
  return reinterpret_cast<CUfunction>(f);
}

// cuLaunchKernel of a one-parameter kernel (passed by address).
template <typename P>
inline CUresult launch_driver(CUfunction f, dim3 grid, dim3 block,
                              unsigned smem, cudaStream_t stream,
                              const P& param) {
  if (!f || !launch_kernel_fn()) return CUDA_ERROR_NOT_FOUND;
  void* args[1] = {const_cast<P*>(&param)};
  return launch_kernel_fn()(f, grid.x, grid.y, grid.z, block.x, block.y,
                            block.z, smem, reinterpret_cast<CUstream>(stream),
                            args, nullptr);
}

// cuLaunchKernelEx of a one-parameter kernel with programmatic stream
// serialization (PDL).
template <typename P>
inline CUresult launch_driver_pdl(CUfunction f, dim3 grid, dim3 block,
                                  unsigned smem, cudaStream_t stream,
                                  const P& param) {
  if (!f || !launch_kernel_ex_fn()) return CUDA_ERROR_NOT_FOUND;
  CUlaunchAttribute attr[1];
  attr[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  attr[0].value.programmaticStreamSerializationAllowed = 1;
  CUlaunchConfig cfg = {};
  cfg.gridDimX = grid.x;
  cfg.gridDimY = grid.y;
  cfg.gridDimZ = grid.z;
  cfg.blockDimX = block.x;
  cfg.blockDimY = block.y;
  cfg.blockDimZ = block.z;
  cfg.sharedMemBytes = smem;
  cfg.hStream = reinterpret_cast<CUstream>(stream);
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  void* args[1] = {const_cast<P*>(&param)};
  return launch_kernel_ex_fn()(&cfg, f, args, nullptr);
}

// Tiled TMA tensor map (rank 3 here), 256B L2 promotion, no OOB fill.
inline bool encode_tiled(CUtensorMap* map, CUtensorMapDataType type,
                         cuuint32_t rank, const void* base,
                         const cuuint64_t* dims, const cuuint64_t* strides,
                         const cuuint32_t* box, CUtensorMapSwizzle swizzle) {
  static const cuuint32_t ones[5] = {1, 1, 1, 1, 1};
  const PFN_cuTensorMapEncodeTiled_v12000 fn = encode_tiled_fn();
  return fn && fn(map, type, rank, const_cast<void*>(base), dims, strides, box,
                  ones, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// A tiled tensor map whose base address and outermost dimension change per
// call, rewritten on the host without the driver. probe_patch() encodes the map
// three times and finds the 8-byte address field and the 4-byte outermost
// dimension field; it returns false when the layout is not recognised (then the
// caller encodes per call). apply_patch() writes a copy with a new address and
// outermost dimension.
struct MapPatch {
  CUtensorMap tmpl;
  int addr_off, dim_off;
  cuuint32_t dim_bias;
};
template <typename Encode>
inline bool probe_patch(MapPatch* mp, const void* base, cuuint64_t outer,
                        Encode enc) {
  CUtensorMap m[3];
  const void* base2 = static_cast<const char*>(base) + (1 << 20);
  const cuuint64_t d = 13;
  if (!enc(&m[0], base, outer) || !enc(&m[1], base2, outer) ||
      !enc(&m[2], base, outer + d))
    return false;
  const unsigned char* a = reinterpret_cast<const unsigned char*>(&m[0]);
  const unsigned char* b = reinterpret_cast<const unsigned char*>(&m[1]);
  const unsigned char* c = reinterpret_cast<const unsigned char*>(&m[2]);
  const int n = static_cast<int>(sizeof(CUtensorMap));
  auto same_but = [&](const unsigned char* x, int off, int len) {
    for (int i = 0; i < n; ++i)
      if ((i < off || i >= off + len) && a[i] != x[i]) return false;
    return true;
  };
  mp->addr_off = mp->dim_off = -1;
  for (int o = 0; o + 8 <= n; ++o)
    if (!__builtin_memcmp(a + o, &base, 8) &&
        !__builtin_memcmp(b + o, &base2, 8) && same_but(b, o, 8))
      mp->addr_off = o;
  for (int o = 0; o + 4 <= n; ++o) {
    cuuint32_t va, vc;
    __builtin_memcpy(&va, a + o, 4);
    __builtin_memcpy(&vc, c + o, 4);
    if (vc - va == d && (va == outer || va + 1 == outer) && same_but(c, o, 4))
      mp->dim_off = o, mp->dim_bias = (cuuint32_t)outer - va;
  }
  mp->tmpl = m[0];
  return mp->addr_off >= 0 && mp->dim_off >= 0;
}
inline void apply_patch(const MapPatch& mp, CUtensorMap* dst, const void* base,
                        cuuint64_t outer) {
  *dst = mp.tmpl;
  const cuuint32_t v = (cuuint32_t)outer - mp.dim_bias;
  __builtin_memcpy(reinterpret_cast<unsigned char*>(dst) + mp.addr_off, &base,
                   8);
  __builtin_memcpy(reinterpret_cast<unsigned char*>(dst) + mp.dim_off, &v, 4);
}

}  // namespace cu
}  // namespace kda_sm90
}  // namespace vllm
