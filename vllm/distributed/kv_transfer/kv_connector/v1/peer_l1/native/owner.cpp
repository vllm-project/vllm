// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "replica.h"
#include <exception>
#include <string>
namespace {
thread_local std::string last_error;
template <class F>
int64_t checked(F&& fn) noexcept {
  try {
    return fn();
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}
}  // namespace
using namespace vllm::peer_l1;
extern "C" const char* l1_owner_error() { return last_error.c_str(); }
extern "C" {
int64_t l1_owner_start(const char* endpoint, uint64_t epoch, uint64_t rows,
                       uint64_t cap, uint64_t window, int lease_ms,
                       uint64_t log_capacity, uint64_t initial_window,
                       uint8_t* published, const Key* credential,
                       int64_t (*clock_ns)()) {
  return checked([&] {
    return reinterpret_cast<int64_t>(
        new Owner(endpoint, epoch, rows, cap, window, lease_ms, log_capacity,
                  initial_window, published, credential, clock_ns));
  });
}
int64_t l1_owner_stop(void* owner) {
  delete static_cast<Owner*>(owner);
  return 0;
}
int64_t l1_owner_describe(void* owner, const int64_t* hello, size_t n_hello,
                          const int64_t* members, size_t n_members) {
  return checked([&] {
    static_cast<Owner*>(owner)->describe(hello, n_hello, members, n_members);
    return 0;
  });
}
int64_t l1_owner_publish(void* owner, const Key* keys, const int64_t* rows,
                         size_t n) {
  return checked([&] {
    static_cast<Owner*>(owner)->publish(keys, rows, n);
    return 0;
  });
}
int64_t l1_owner_retire(void* owner, int64_t row, int withhold) {
  return checked(
      [&] { return static_cast<Owner*>(owner)->retire(row, withhold); });
}
int64_t l1_owner_absorb(void* owner, int64_t* out, size_t n, uint64_t* stats,
                        uint64_t free_blocks, const int64_t* in_use,
                        size_t n_in_use, uint64_t deferred_snapshot) {
  return checked([&] {
    return static_cast<Owner*>(owner)->absorb(
        out, n, stats, free_blocks, in_use, n_in_use, deferred_snapshot);
  });
}
int64_t l1_owner_deferred_rows(void* owner, int64_t* out, size_t n) {
  return checked(
      [&] { return static_cast<Owner*>(owner)->deferred_rows(out, n); });
}
int64_t l1_owner_reserve(void* owner) {
  return static_cast<Owner*>(owner)->reserve();
}
int64_t l1_owner_complete_absorb(void* owner, uint64_t free_blocks,
                                 uint64_t* stats) {
  static_cast<Owner*>(owner)->complete_absorb(free_blocks, stats);
  return 0;
}
int64_t l1_owner_pending(void* owner) {
  return static_cast<Owner*>(owner)->pending();
}
int64_t l1_owner_refs(void* owner) {
  return static_cast<Owner*>(owner)->refs();
}
int64_t l1_owner_wake_fd(void* owner) {
  return static_cast<Owner*>(owner)->wake_fd();
}
int64_t l1_owner_freeze(void* owner) {
  return static_cast<Owner*>(owner)->freeze();
}
int64_t l1_owner_thaw(void* owner, int clear) {
  static_cast<Owner*>(owner)->thaw(clear);
  return 0;
}
int64_t l1_replica_start(const char* const* endpoints, const uint64_t* epochs,
                         size_t n, const Key* credential) {
  return checked([&] {
    return reinterpret_cast<int64_t>(
        new Replica(endpoints, epochs, n, credential));
  });
}
int64_t l1_replica_stop(void* replica) {
  delete static_cast<Replica*>(replica);
  return 0;
}
int64_t l1_replica_generation(void* replica, size_t peer) {
  return checked(
      [&] { return static_cast<Replica*>(replica)->generation(peer); });
}
int64_t l1_replica_window(void* replica, size_t peer) {
  return checked([&] { return static_cast<Replica*>(replica)->window(peer); });
}
int64_t l1_replica_query(void* replica, const uint64_t* keys, size_t n,
                         uint8_t* out) {
  return checked([&] {
    static_cast<Replica*>(replica)->query(keys, n, out);
    return 0;
  });
}
int64_t l1_owner_rpc(const char* endpoint, uint64_t op, uint64_t epoch,
                     uint64_t reader, uint64_t seq,
                     const vllm::peer_l1::Key* keys, int64_t* out, size_t n,
                     int timeout_ms, const Key* credential) {
  vllm::peer_l1::Request request;
  request.op = op;
  request.epoch = epoch;
  request.reader = reader;
  request.seq = seq;
  request.n = n;
  if (credential) request.credential = *credential;
  return vllm::peer_l1::rpc(endpoint, request, keys, out, timeout_ms);
}
int64_t l1_cleanup_start(int timeout_ms) {
  return checked([&] {
    return reinterpret_cast<int64_t>(
        new vllm::peer_l1::CleanupQueue(timeout_ms));
  });
}
int64_t l1_cleanup_stop(void* queue) {
  delete static_cast<vllm::peer_l1::CleanupQueue*>(queue);
  return 0;
}
int64_t l1_cleanup_pending(void* queue) {
  return static_cast<vllm::peer_l1::CleanupQueue*>(queue)->pending();
}
int64_t l1_cleanup_create(void* queue, const char* endpoint, uint64_t op,
                          uint64_t epoch, uint64_t reader, uint64_t seq,
                          uint64_t lease_ms, void* next,
                          const Key* credential) {
  return checked([&] {
    vllm::peer_l1::Request request;
    request.op = op;
    request.epoch = epoch;
    request.reader = reader;
    request.seq = seq;
    request.n = 0;
    if (credential) request.credential = *credential;
    auto* item = new vllm::peer_l1::Cleanup{
        static_cast<vllm::peer_l1::CleanupQueue*>(queue), endpoint, request};
    item->batch_next = static_cast<vllm::peer_l1::Cleanup*>(next);
    if (lease_ms)
      item->expires =
          vllm::peer_l1::Clock::now() + std::chrono::milliseconds(lease_ms);
    return reinterpret_cast<int64_t>(item);
  });
}
void l1_cleanup_enqueue(void* item) { vllm::peer_l1::completed_copy(item); }
}
