#ifndef UTILS_HPP
#define UTILS_HPP

#include <atomic>
#include <cassert>
#include <cstdint>
#include <unistd.h>

#if defined(__APPLE__)
  #include <sys/sysctl.h>
#endif

#include "cpu_types.hpp"

namespace cpu_utils {
enum class ISA { AMX, VEC };

template <typename T>
struct VecTypeTrait {
  using vec_t = void;
};

template <>
struct VecTypeTrait<float> {
  using vec_t = vec_op::FP32Vec16;
};

#if !defined(__aarch64__) || defined(ARM_BF16_SUPPORT)
template <>
struct VecTypeTrait<c10::BFloat16> {
  using vec_t = vec_op::BF16Vec16;
};
#endif

template <>
struct VecTypeTrait<c10::Half> {
  using vec_t = vec_op::FP16Vec16;
};

struct Counter {
  std::atomic<int64_t> counter;
  char _padding[56];

  Counter() : counter(0) {}

  void reset_counter() { counter.store(0); }

  int64_t acquire_counter() { return counter++; }
};

inline int64_t get_l2_size() {
  static int64_t size = []() {
#if defined(__APPLE__)
    // macOS doesn't have _SC_LEVEL2_CACHE_SIZE. Use sysctlbyname.
    int64_t l2_cache_size = 0;
    size_t len = sizeof(l2_cache_size);
    if (sysctlbyname("hw.l2cachesize", &l2_cache_size, &len, NULL, 0) == 0 &&
        l2_cache_size > 0) {
      return l2_cache_size >> 1;  // use 50% of L2 cache
    }
    // Fallback if sysctlbyname fails
    return 128LL * 1024 >> 1;  // use 50% of 128KB
#else
    long l2_cache_size = sysconf(_SC_LEVEL2_CACHE_SIZE);
    assert(l2_cache_size != -1);
    return l2_cache_size >> 1;  // use 50% of L2 cache
#endif
  }();
  return size;
}
}  // namespace cpu_utils

#endif
