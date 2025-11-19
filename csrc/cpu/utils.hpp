#ifndef UTILS_HPP
#define UTILS_HPP

#include <atomic>
#include <cassert>
#include <cstdint>
#include <unistd.h>

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

template <>
struct VecTypeTrait<c10::BFloat16> {
  using vec_t = vec_op::BF16Vec16;
};

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
    long l2_cache_size = sysconf(_SC_LEVEL2_CACHE_SIZE);
    assert(l2_cache_size != -1);
    return l2_cache_size >> 1;  // use 50% of L2 cache
  }();
  return size;
}
}  // namespace cpu_utils

#endif
