#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#include <utility>

namespace cu {

// Launches above 48 KB of dynamic shared memory need this limit raised.
template <typename Kernel>
inline cudaError_t max_dynamic_smem(Kernel* kernel, int bytes) {
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

// Kernels with the same carveout (percent of the maximum shared memory) can
// share an SM.
template <typename Kernel>
inline cudaError_t smem_carveout(Kernel* kernel, int percent) {
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, percent);
}

// Launches with programmatic dependent launch: the kernel may start while the
// previous kernel in the stream still runs, so it calls ptx::griddep_wait()
// before it reads that kernel's output.
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

}  // namespace cu

namespace ptx {

__device__ __forceinline__ void bulk_copy_g2s(uint32_t dst_smem,
                                              const void* src, uint32_t bytes,
                                              uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];\n" ::"r"(dst_smem),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void bulk_copy_s2g(void* dst, uint32_t src_smem,
                                              uint32_t bytes) {
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(
          dst),
      "r"(src_smem), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void cp_async_16(uint32_t dst_smem,
                                            const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_smem),
               "l"(src)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}

__device__ __forceinline__ void griddep_wait() {
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
}

// Global loads that are issued where they are written: the compiler otherwise
// moves a plain load down to the first use of its value.
__device__ __forceinline__ uint32_t ld_global_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint16_t ld_global_u16(const void* p) {
  uint16_t v;
  asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_global_v4(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void mbarrier_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar),
               "r"(count)
               : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t bar,
                                                          uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar),
      "r"(bytes)
      : "memory");
}

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

__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4],
                                                  const uint32_t (&a)[4],
                                                  uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// fp32 bits as tf32: the low mantissa bits are ignored.
__device__ __forceinline__ void mma_m16n8k8_tf32(float (&d)[4],
                                                 const uint32_t (&a)[4],
                                                 uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_m16n8k16_f16(float (&d)[4],
                                                 const uint32_t (&a)[4],
                                                 uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

}  // namespace ptx
