// SPDX-License-Identifier: Apache-2.0

#include "event_recorder.h"

#include <chrono>
#include <utility>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Monotonic wall-clock reader.
//
// Why not just `system_clock::now()`?  `system_clock` maps to CLOCK_REALTIME,
// which NTP / chrony can slew backward by tens of microseconds to stay synced
// with upstream.  CUDA host callbacks fire at arbitrary wall moments, so two
// consecutive callbacks (e.g. MP_STORE_START then MP_STORE_END on the same
// stream) can straddle a backward slew and land with end_ts < start_ts.
// Jaeger then renders span duration as unsigned 64-bit subtraction, producing
// the characteristic ~213503982d span duration (= 2^64 microseconds).
//
// Fix: anchor once to the (system_clock, steady_clock) pair at first call.
// Every subsequent timestamp is computed as
//   epoch_sys + (steady_now - epoch_steady)
// which is monotonic (steady_clock == CLOCK_MONOTONIC on Linux) while
// remaining expressed in Unix-epoch seconds so downstream consumers don't
// notice any change.
static double wall_clock_time() {
  static const auto epoch_sys = std::chrono::system_clock::now();
  static const auto epoch_steady = std::chrono::steady_clock::now();
  auto now_steady = std::chrono::steady_clock::now();
  auto since_epoch = epoch_sys.time_since_epoch() + (now_steady - epoch_steady);
  return std::chrono::duration<double>(since_epoch).count();
}

// ---------------------------------------------------------------------------
// EventRecorder
// ---------------------------------------------------------------------------

EventRecorder& EventRecorder::instance() {
  static EventRecorder recorder;
  return recorder;
}

void EventRecorder::push(PendingEvent* event) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.push_back(std::move(*event));
  }
  delete event;
}

std::vector<PendingEvent> EventRecorder::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PendingEvent> result;
  result.swap(buffer_);
  return result;
}

// ---------------------------------------------------------------------------
// CUDA host callback — runs on a CUDA driver thread, no GIL.
// ---------------------------------------------------------------------------

static void
#ifndef USE_ROCM
    CUDART_CB
#endif
    event_host_callback(void* data) {
  auto* event = static_cast<PendingEvent*>(data);
  event->timestamp = wall_clock_time();
  EventRecorder::instance().push(event);
}

// ---------------------------------------------------------------------------
// Free functions for pybind11
// ---------------------------------------------------------------------------

void record_event_on_stream(
    int64_t cuda_stream_ptr, const std::string& event_type_name,
    const std::string& session_id,
    const std::unordered_map<std::string, std::string>& str_metadata,
    const std::unordered_map<std::string, int64_t>& int_metadata) {
  auto* event = new PendingEvent{
      event_type_name, session_id, 0.0, str_metadata, int_metadata,
  };

  auto stream = reinterpret_cast<lmcache_stream_t>(
      static_cast<uintptr_t>(cuda_stream_ptr));
  LMCACHE_LAUNCH_HOST_FUNC(stream, event_host_callback, event);
}

DrainResult drain_recorded_events() {
  auto events = EventRecorder::instance().drain();
  DrainResult result;
  result.reserve(events.size());
  for (auto& e : events) {
    result.emplace_back(std::move(e.event_type_name), std::move(e.session_id),
                        e.timestamp, std::move(e.str_metadata),
                        std::move(e.int_metadata));
  }
  return result;
}
