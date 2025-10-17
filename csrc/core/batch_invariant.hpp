#pragma once
#include <cstdlib>
#include <string>
#include <cctype>

namespace vllm {

// vllm_is_batch_invariant(); returns true
// if env VLLM_BATCH_INVARIANT=1
inline bool vllm_is_batch_invariant() {
  static bool cached = []() {
    std::string env_key = "VLLM_BATCH_INVARIANT";
    const char* val = std::getenv(env_key.c_str());
    return (val && std::atoi(val) != 0) ? 1 : 0;
  }();
  return cached;
}

}  // namespace vllm
