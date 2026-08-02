#pragma once

#include "kerutils/device/common.h"

namespace kerutils {

// st.async
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async)
template <typename T>
CUTE_DEVICE static void st_async(void* dst_ptr, const T& data,
                                 transac_bar_t& mbar) {
  static_assert(sizeof(T) == 16,
                "Data type must be 16 bytes (128 bits) for st_async.");
  long2 data_long2 = *reinterpret_cast<const long2*>(&data);
  uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
  uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar);
  asm volatile(
      "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 [%0], "
      "{%1, %2}, [%3]; \n"
      :
      : "r"(dst_addr), "l"(data_long2.x), "l"(data_long2.y), "r"(mbar_addr));
}

static constexpr int PEER_ADDR_MASK = 16777216;

// Given an address in the current CTA, return the corresponding address in the
// peer CTA
template <typename T>
CUTE_DEVICE T* get_peer_addr(const T* p) {
  return (T*)((int64_t)(p) ^ PEER_ADDR_MASK);
}

// Given an address in the current CTA, return the corresponding address in the
// peer CTA (if the current CTA_id%2 == 1) or the address itself (if CTA_id%2 ==
// 0)
template <typename T>
CUTE_DEVICE T* get_cta0_addr(const T* p) {
  constexpr int CTA0_ADDR_MASK = 0xFEFFFFFF;
  return (T*)((int64_t)(p)&CTA0_ADDR_MASK);
}

// TMA bulk reduce add (cp.reduce.async.bulk), shared to global, float32, add.
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk)
CUTE_DEVICE
void tma_bulk_reduce_add(void const* src_ptr, void* dst_ptr,
                         int32_t store_bytes) {
  uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(src_ptr);
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], "
      "%2;\n"
      :
      : "l"(dst_ptr), "r"(smem_int_ptr), "r"(store_bytes)
      : "memory");
}

// Cluster barrier arrive with .release modifier.
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster)
CUTE_DEVICE
void barrier_cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release;" : : : "memory");
}

// Cluster barrier arrive with .relaxed modifier.
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster)
CUTE_DEVICE
void barrier_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed;" : : :);
}

// Cluster barrier wait with .acquire modifier.
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster)
CUTE_DEVICE
void barrier_cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire;" : : : "memory");
}

// mbarrier.arrive with .relaxed.cluster qualifier
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive)
CUTE_DEVICE
void mbarrier_arrive_relaxed_cluster(transac_bar_t& mbar) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&mbar);
  asm volatile(
      "{\n\t"
      "mbarrier.arrive.relaxed.cluster.shared::cta.b64 _, [%0];\n\t"
      "}"
      :
      : "r"(smem_addr));
}

// AtomicAdd with v4.f32 type
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red)
CUTE_DEVICE
void atomicadd_f32x4_with_policy_and_pred(void* global_addr, const float4& data,
                                          int64_t cache_policy,
                                          uint32_t pred = true) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.eq.u32 p, %6, 1;\n\t"
      "@p red.relaxed.gpu.global.add.L2::cache_hint.v4.f32 [%4], {%0, %1, %2, "
      "%3}, %5; \n\t"
      "}"
      :
      : "f"(data.x), "f"(data.y), "f"(data.z), "f"(data.w),
        "l"((int64_t)global_addr), "l"(cache_policy), "r"(pred));
}

// cp.async.bulk, from .shared::cta to .shared::cluster
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk)
CUTE_DEVICE
void cp_async_bulk_shared_cta_to_shared_cluster(void* dst_ptr,
                                                const void* src_ptr,
                                                int32_t load_bytes,
                                                transac_bar_t& mbar) {
  uint32_t dst_smem_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
  uint32_t src_smem_addr = cute::cast_smem_ptr_to_uint(src_ptr);
  uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar);
  asm volatile(
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3]; \n"
      :
      : "r"(dst_smem_addr), "r"(src_smem_addr), "r"(load_bytes),
        "r"(mbar_addr));
}

}  // namespace kerutils
