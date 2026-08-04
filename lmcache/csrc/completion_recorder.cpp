// SPDX-License-Identifier: Apache-2.0

#include "completion_recorder.h"

#include <utility>

CompletionRecorder& CompletionRecorder::instance() {
  static CompletionRecorder recorder;
  return recorder;
}

void CompletionRecorder::push(std::unique_ptr<PendingCompletion> completion) {
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_.push_back(std::move(completion));
}

std::vector<std::unique_ptr<PendingCompletion>> CompletionRecorder::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::unique_ptr<PendingCompletion>> result;
  result.swap(buffer_);
  return result;
}

static void
#ifndef USE_ROCM
    CUDART_CB
#endif
    completion_host_callback(void* data) {
  // Adopt the raw pointer back into a unique_ptr.
  std::unique_ptr<PendingCompletion> completion(
      static_cast<PendingCompletion*>(data));
  CompletionRecorder::instance().push(std::move(completion));
}

void record_completion_on_stream(int64_t cuda_stream_ptr,
                                 const std::string& kind, std::string payload) {
  auto completion = std::make_unique<PendingCompletion>(
      PendingCompletion{kind, std::move(payload)});
  auto stream = reinterpret_cast<lmcache_completion_stream_t>(
      static_cast<uintptr_t>(cuda_stream_ptr));
  // Pass ownership through the driver as a raw pointer; the host callback
  // re-adopts it. Reclaim if the launch itself fails.
  PendingCompletion* raw = completion.release();
  auto err = LMCACHE_COMPLETION_LAUNCH_HOST_FUNC(stream,
                                                 completion_host_callback, raw);
  if (err != 0) {
    delete raw;
  }
}

CompletionDrainResult drain_recorded_completions() {
  auto completions = CompletionRecorder::instance().drain();
  CompletionDrainResult result;
  result.reserve(completions.size());
  for (auto& c : completions) {
    result.emplace_back(std::move(c->kind), std::move(c->payload));
  }
  return result;
}
