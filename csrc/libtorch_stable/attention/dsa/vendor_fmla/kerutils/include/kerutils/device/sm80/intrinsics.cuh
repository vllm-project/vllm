#pragma once

#include "kerutils/device/common.h"

namespace kerutils {

// cp.async.cg (cache global) with prefetch and predicate
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)
template <PrefetchSize PREFETCH_SIZE = PrefetchSize::B128>
CUTE_DEVICE void cp_async_cacheglobal(const void* src, void* dst,
                                      bool pred = true) {
  uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
  if constexpr (PREFETCH_SIZE == PrefetchSize::B64) {
    asm volatile(
        "cp.async.cg.shared.global.L2::64B [%0], [%1], 16, %2;\n" ::"r"(
            dst_addr),
        "l"(src), "r"(pred ? 16 : 0));
  } else if constexpr (PREFETCH_SIZE == PrefetchSize::B128) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(
            dst_addr),
        "l"(src), "r"(pred ? 16 : 0));
  } else if constexpr (PREFETCH_SIZE == PrefetchSize::B256) {
    asm volatile(
        "cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" ::"r"(
            dst_addr),
        "l"(src), "r"(pred ? 16 : 0));
  } else {
    static_assert(PREFETCH_SIZE == PrefetchSize::B64 ||
                      PREFETCH_SIZE == PrefetchSize::B128 ||
                      PREFETCH_SIZE == PrefetchSize::B256,
                  "Unsupported prefetch size for cp_async_cacheglobal.");
  }
}

// Create fraction-based cache policy
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-createpolicy)
template <CacheHint PRIMARY_PRIORITY, CacheHint SECONDARY_PRIORITY>
CUTE_DEVICE int64_t create_fraction_based_cache_policy(float fraction = 1.0f) {
  int64_t result;
#define EMIT(PRIMARY_PRIORITY_STR, SECONDARY_PRIORITY_STR)         \
  asm volatile("createpolicy.fractional.L2::" PRIMARY_PRIORITY_STR \
               ".L2::" SECONDARY_PRIORITY_STR ".b64 %0, %1;\n"     \
               : "=l"(result)                                      \
               : "f"(fraction));
#define EMIT2(PRIMARY_PRIORITY_STR)                                          \
  {                                                                          \
    if constexpr (SECONDARY_PRIORITY == CacheHint::EVICT_FIRST) {            \
      EMIT(PRIMARY_PRIORITY_STR, "evict_first")                              \
    } else if constexpr (SECONDARY_PRIORITY == CacheHint::EVICT_UNCHANGED) { \
      EMIT(PRIMARY_PRIORITY_STR, "evict_unchanged")                          \
    } else {                                                                 \
      static_assert(SECONDARY_PRIORITY == CacheHint::EVICT_FIRST ||          \
                        SECONDARY_PRIORITY == CacheHint::EVICT_UNCHANGED,    \
                    "Unsupported secondary cache hint for "                  \
                    "create_fraction_based_cache_policy.");                  \
    }                                                                        \
  }
  if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_FIRST) {
    EMIT2("evict_first");
  } else if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_NORMAL) {
    EMIT2("evict_normal");
  } else if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_LAST) {
    EMIT2("evict_last");
  } else if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_UNCHANGED) {
    EMIT2("evict_unchanged");
  } else {
    static_assert(PRIMARY_PRIORITY == CacheHint::EVICT_FIRST ||
                      PRIMARY_PRIORITY == CacheHint::EVICT_NORMAL ||
                      PRIMARY_PRIORITY == CacheHint::EVICT_LAST ||
                      PRIMARY_PRIORITY == CacheHint::EVICT_UNCHANGED,
                  "Unsupported primary cache hint for "
                  "create_fraction_based_cache_policy.");
  }
#undef EMIT
#undef EMIT2
  return result;
}

// Create a simple cache policy (equivalent to
// create_fraction_based_cache_policy(1.0f)) The same as
// cute::TMA::CacheHintSmXX
template <CacheHint CACHE_HINT>
CUTE_DEVICE constexpr int64_t create_simple_cache_policy() {
  if constexpr (CACHE_HINT == CacheHint::EVICT_FIRST) {
    return 0x12F0000000000000;  // Result of
                                // createpolicy.fractional.L2::evict_first.b64
  } else if constexpr (CACHE_HINT == CacheHint::EVICT_NORMAL) {
    return 0x1000000000000000;  // Copied from CuTe. Unsure about the exact
                                // meaning. (TODO Change to 0x16F0000000000000?)
  } else if constexpr (CACHE_HINT == CacheHint::EVICT_LAST) {
    return 0x14F0000000000000;  // Result of
                                // createpolicy.fractional.L2::evict_last.b64
  } else {
    static_assert(CACHE_HINT == CacheHint::EVICT_FIRST ||
                      CACHE_HINT == CacheHint::EVICT_NORMAL ||
                      CACHE_HINT == CacheHint::EVICT_LAST,
                  "Unsupported cache hint for create_simple_cache_policy.");
  }
}

// AtomicAdd
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red)
CUTE_DEVICE
void atomicadd_f32_with_policy_and_pred(void* global_addr, const float& data,
                                        int64_t cache_policy,
                                        uint32_t pred = true) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.eq.u32 p, %3, 1;\n\t"
      "@p red.relaxed.gpu.global.add.L2::cache_hint.f32 [%1], %0, %2; \n\t"
      "}"
      :
      : "f"(data), "l"((int64_t)global_addr), "l"(cache_policy), "r"(pred));
}

// Get the id of the current SM
// About %smid
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-smid):
// PTX document says that %smid ranges from 0 to %nsmid-1, while "The SM
// identifier numbering is not guaranteed to be contiguous, so %nsmid may be
// larger than the physical number of SMs in the device.". However, result shows
// that, at least for sm90 and sm100f, %nsmid is the number of physical SMs - 1.
// For the sake of safety, I recommend you to check the return of get_sm_id
// manually or call `get_sm_id_with_range_check()` defined in
// `device/sm80/helpers.cuh`. Besides, PTX document also says that this number
// may change due to preemption, but currently this never happens according to
// [DATEN GELÖSCHT]
CUTE_DEVICE
uint32_t get_sm_id() {
  uint32_t ret;
  asm volatile("mov.u32 %0, %%smid;\n" : "=r"(ret));
  return ret;
}

// trap
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-trap)
CUTE_DEVICE
void trap() { asm volatile("trap;\n"); }

// LDG.128 or LDG.128 with non-coherent cache
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld)
// We use macro instead of function here, since we need a multi-level recursive
// dispatch based on template parameters if using function NC_STR should be
// either "" or ".nc" L1_CACHE_HINT_STR should be either "evict_first",
// "evict_normal", "evict_last", "evict_unchanged", or "no_allocate"
// L2_PREFETCH_SIZE_STR should be either "64B", "128B", or "256B"
// L2 cache hint is not supported since it's only supported for LDG.256
#define KU_LDG_128(global_addr, result, NC_STR, L1_CACHE_HINT_STR,        \
                   L2_PREFETCH_SIZE_STR)                                  \
  {                                                                       \
    static_assert(std::is_pointer_v<decltype(global_addr)> ||             \
                      std::is_array_v<decltype(global_addr)>,             \
                  "`global_addr` must be a pointer");                     \
    static_assert(std::is_pointer_v<decltype(result)> ||                  \
                      std::is_array_v<decltype(result)>,                  \
                  "`result` must be a pointer");                          \
    uint64_t* result_as_uint64_ptr = (uint64_t*)(result);                 \
    asm volatile("ld.global" NC_STR ".L1::" L1_CACHE_HINT_STR             \
                 ".L2::" L2_PREFETCH_SIZE_STR ".v2.u64 {%0, %1}, [%2];\n" \
                 : "=l"(result_as_uint64_ptr[0]),                         \
                   "=l"(result_as_uint64_ptr[1])                          \
                 : "l"(global_addr));                                     \
  }

}  // namespace kerutils
