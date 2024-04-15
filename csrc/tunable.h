#pragma once

#include <torch/extension.h>
#if defined(__has_include) && defined(USE_ROCM)
#if __has_include(<ATen/cuda/tunable/Tunable.h>)
#include <ATen/cuda/tunable/Tunable.h>
#define VLLM_TUNABLEOP_AVAILABLE
#endif
#endif

namespace vllm::tunable {

#ifndef VLLM_TUNABLEOP_AVAILABLE

inline constexpr bool is_available() { return false; }
inline constexpr bool is_enabled() { return false; }
inline constexpr bool is_tuning_enabled() { return false; }

#else

inline constexpr bool is_available() { return true; }

inline void enable() {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  tuning_ctx->EnableTunableOp();
}

inline void disable() {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  tuning_ctx->DisableTunableOp();
}

inline bool is_enabled() {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  return tuning_ctx->IsTunableOpEnabled();
}

inline void enable_tuning() {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  tuning_ctx->EnableTuning();
}

inline void disable_tuning() {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  tuning_ctx->DisableTuning();
}

inline bool is_tuning_enabled() {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  return tuning_ctx->IsTuningEnabled();
}

inline void set_max_tuning_duration_ms(int max_duration_ms) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  tuning_ctx->SetMaxTuningDurationMs(max_duration_ms);
}

inline int get_max_tuning_duration_ms() {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  return tuning_ctx->GetMaxTuningDurationMs();
}

inline void set_max_tuning_iterations(int max_iterations) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  tuning_ctx->SetMaxTuningIterations(max_iterations);
}

inline int get_max_tuning_iterations() {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  return tuning_ctx->GetMaxTuningIterations();
}

inline void set_filename(const std::string& filename) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  tuning_ctx->SetFilename(filename);
}

inline std::string get_filename() {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  return tuning_ctx->GetFilename();
}

#endif // ifndef VLLM_TUNABLEOP_AVAILABLE

}
