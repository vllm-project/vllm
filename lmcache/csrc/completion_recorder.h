// SPDX-License-Identifier: Apache-2.0
//
// CUDA/HIP host-callback buffer for stream-completion records.
// Mirrors event_recorder.h but carries an opaque bytes payload instead
// of timestamped metadata. The callback runs without the GIL; Python
// drains the buffer and dispatches to a handler keyed by ``kind``.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
using lmcache_completion_stream_t = hipStream_t;
  #define LMCACHE_COMPLETION_LAUNCH_HOST_FUNC hipLaunchHostFunc
#else
  #include <cuda_runtime.h>
using lmcache_completion_stream_t = cudaStream_t;
  #define LMCACHE_COMPLETION_LAUNCH_HOST_FUNC cudaLaunchHostFunc
#endif

struct PendingCompletion {
  std::string kind;     // dispatch key, e.g. "finish_write"
  std::string payload;  // opaque encoded bytes (e.g. msgpack)
};

class CompletionRecorder {
 public:
  static CompletionRecorder& instance();
  // Takes ownership of the heap-allocated PendingCompletion.
  void push(std::unique_ptr<PendingCompletion> completion);
  std::vector<std::unique_ptr<PendingCompletion>> drain();

 private:
  CompletionRecorder() = default;
  std::mutex mutex_;
  std::vector<std::unique_ptr<PendingCompletion>> buffer_;
};

// Schedule a completion record. Called WITHOUT the GIL.
void record_completion_on_stream(int64_t cuda_stream_ptr,
                                 const std::string& kind, std::string payload);

using CompletionDrainResult = std::vector<std::pair<std::string, std::string>>;

CompletionDrainResult drain_recorded_completions();
