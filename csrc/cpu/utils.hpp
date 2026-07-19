#ifndef UTILS_HPP
#define UTILS_HPP

#include <atomic>
#include <cstdint>
#include <string>
#include <unistd.h>
#include <ATen/cpu/Utils.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "cpu/cpu_types.hpp"

namespace cpu_utils {
enum class ISA { AMX, VEC, RVV, NEON };

inline ISA get_isa(const std::string& isa) {
  if (isa == "amx") {
    return ISA::AMX;
  } else if (isa == "vec") {
    return ISA::VEC;
  } else if (isa == "rvv") {
    return ISA::RVV;
  } else if (isa == "neon") {
    return ISA::NEON;
  } else {
    TORCH_CHECK(false, "Invalid isa type: " + isa);
  }
}

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

#if !defined(__powerpc__)
template <>
struct VecTypeTrait<c10::Half> {
  using vec_t = vec_op::FP16Vec16;
};
#endif

struct Counter {
  std::atomic<int64_t> counter;
  char _padding[56];

  Counter() : counter(0) {}

  void reset_counter() { counter.store(0); }

  int64_t acquire_counter() { return counter++; }
};

inline uint32_t get_l2_cache_size_or_default() {
  uint32_t l2_cache_size = 0;

#if defined(__APPLE__)
  uint64_t sys_l2 = 0;
  size_t value_size = sizeof(sys_l2);
  if (sysctlbyname("hw.l2cachesize", &sys_l2, &value_size, nullptr, 0) == 0 &&
      sys_l2 > 0) {
    l2_cache_size = static_cast<uint32_t>(sys_l2);
  }
#elif defined(__s390x__) || defined(__powerpc__)
  auto caps = at::cpu::get_cpu_capabilities();
  auto it = caps.find("l2_cache_size");
  if (it != caps.end()) {
    l2_cache_size = static_cast<uint32_t>(it->second.toInt());
  }
  if (l2_cache_size == 0) {
    long sys_l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (sys_l2 > 0) {
      l2_cache_size = static_cast<uint32_t>(sys_l2);
    }
  }
#else
  auto caps = at::cpu::get_cpu_capabilities();
  l2_cache_size = static_cast<uint32_t>(caps.at("l2_cache_size").toInt());
#endif

  if (l2_cache_size == 0) {
    l2_cache_size = 256 * 1024;
  }
  return l2_cache_size;
}

inline int64_t get_available_l2_size() {
  static int64_t size =
      static_cast<int64_t>(get_l2_cache_size_or_default()) >> 1;
  return size;
}

template <int32_t alignment_v, typename T>
inline T round_up(T size) {
  T alignment = alignment_v;
  return (((size + alignment - 1) / alignment) * alignment);
}

template <int32_t alignment_v, typename T>
inline T round_down(T size) {
  T alignment = alignment_v;
  return (size / alignment) * alignment;
}

template <typename T>
inline void print_logits(const char* name, T* ptr, int32_t row, int32_t col,
                         int32_t stride) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(5) << name << ": [\n";
  auto* curr_logits_buffer = ptr;
  for (int32_t m = 0; m < row; ++m) {
    for (int32_t n = 0; n < col; ++n) {
      ss << curr_logits_buffer[n] << ", ";
    }
    ss << "\n";
    curr_logits_buffer += stride;
  }
  ss << "]\n";
  std::printf("%s", ss.str().c_str());
}

class ScratchPadManager {
 public:
  static constexpr size_t allocation_unit = 4 * 1024;  // 4KB

  static ScratchPadManager* get_scratchpad_manager();

  ScratchPadManager();

  template <typename T>
  T* get_data() {
    return reinterpret_cast<T*>(ptr_);
  }

  static size_t round(size_t size) {
    return ((size + allocation_unit - 1) / allocation_unit) * allocation_unit;
  }

  void realloc(size_t new_size);

 private:
  size_t size_;
  void* ptr_;
};
}  // namespace cpu_utils

#endif
