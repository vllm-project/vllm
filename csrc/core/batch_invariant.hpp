#pragma once
#include <cstdlib>
#include <string>
#include <cctype>

namespace vllm {

// vllm_kernel_override_batch_invariant(); returns true
// if env VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT=1
inline bool vllm_kernel_override_batch_invariant() {
  std::string env_key = "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT";
  const char* val = std::getenv(env_key.c_str());
  return (val && std::atoi(val) != 0) ? 1 : 0;
}

}  // namespace vllm
