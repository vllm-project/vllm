// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "replica.h"
#include <exception>
#include <cstdio>
namespace {
thread_local char last_error[512]{};
template <class F>
int64_t checked(F&& fn) noexcept {
  try {
    return fn();
  } catch (const std::exception& e) {
    std::snprintf(last_error, sizeof(last_error), "%s", e.what());
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "unknown peer L1 exception");
  }
  return -1;
}
template <class T>
T* handle(void* pointer) {
  if (!pointer) throw std::invalid_argument("peer L1 handle is closed");
  auto* result = static_cast<T*>(pointer);
  result->check_error();
  return result;
}
}  // namespace
using namespace vllm::peer_l1;
extern "C" const char* l1_owner_error() noexcept { return last_error; }
extern "C" {
const char* l1_owner_stat_schema() noexcept { return owner_stat_schema; }
int64_t l1_owner_start(const char* endpoint, uint64_t epoch, uint64_t rows,
                       uint64_t cap, uint64_t window, int lease_ms,
                       uint64_t log_capacity, uint64_t initial_window,
                       uint8_t* published, const Key* credential,
                       int64_t (*clock_ns)(), uint64_t protected_cap) noexcept {
  return checked([&] {
    return reinterpret_cast<int64_t>(new Owner(
        endpoint, epoch, rows, cap, window, lease_ms, log_capacity,
        initial_window, published, credential, clock_ns, protected_cap));
  });
}
int64_t l1_owner_stop(void* owner) noexcept {
  return checked([&] {
    delete static_cast<Owner*>(owner);
    return 0;
  });
}
int64_t l1_owner_gate(void* owner, int open) noexcept {
  return checked([&] {
    handle<Owner>(owner)->gate(open);
    return 0;
  });
}
int64_t l1_owner_describe(void* owner, const int64_t* hello, size_t n_hello,
                          const int64_t* members, size_t n_members) noexcept {
  return checked([&] {
    handle<Owner>(owner)->describe(hello, n_hello, members, n_members);
    return 0;
  });
}
int64_t l1_owner_publish(void* owner, const Key* keys, const int64_t* rows,
                         size_t n) noexcept {
  return checked([&] {
    handle<Owner>(owner)->publish(keys, rows, n);
    return 0;
  });
}
int64_t l1_owner_retire(void* owner, int64_t row, int withhold) noexcept {
  return checked([&] { return handle<Owner>(owner)->retire(row, withhold); });
}
int64_t l1_owner_absorb(void* owner, int64_t* out, size_t n, uint64_t* stats,
                        uint64_t free_blocks, const int64_t* in_use,
                        size_t n_in_use, uint64_t deferred_snapshot) noexcept {
  return checked([&] {
    return handle<Owner>(owner)->absorb(out, n, stats, free_blocks, in_use,
                                        n_in_use, deferred_snapshot);
  });
}
int64_t l1_owner_deferred_rows(void* owner, int64_t* out, size_t n) noexcept {
  return checked([&] { return handle<Owner>(owner)->deferred_rows(out, n); });
}
int64_t l1_owner_reserve(void* owner) noexcept {
  return checked([&] { return handle<Owner>(owner)->reserve(); });
}
int64_t l1_owner_complete_absorb(void* owner, uint64_t free_blocks,
                                 uint64_t* stats) noexcept {
  return checked([&] {
    handle<Owner>(owner)->complete_absorb(free_blocks, stats);
    return 0;
  });
}
int64_t l1_owner_pending(void* owner) noexcept {
  return checked([&] { return handle<Owner>(owner)->pending(); });
}
int64_t l1_owner_refs(void* owner) noexcept {
  return checked([&] { return handle<Owner>(owner)->refs(); });
}
int64_t l1_owner_wake_fd(void* owner) noexcept {
  return checked([&] { return handle<Owner>(owner)->wake_fd(); });
}
int64_t l1_owner_freeze(void* owner) noexcept {
  return checked([&] { return handle<Owner>(owner)->freeze(); });
}
int64_t l1_owner_thaw(void* owner, int clear) noexcept {
  return checked([&] {
    handle<Owner>(owner)->thaw(clear);
    return 0;
  });
}
int64_t l1_replica_start(const char* const* endpoints, const uint64_t* epochs,
                         const uint64_t* key_caps, size_t n,
                         const Key* credential) noexcept {
  return checked([&] {
    return reinterpret_cast<int64_t>(
        new Replica(endpoints, epochs, key_caps, n, credential));
  });
}
int64_t l1_replica_stop(void* replica) noexcept {
  return checked([&] {
    delete static_cast<Replica*>(replica);
    return 0;
  });
}
int64_t l1_replica_generation(void* replica, size_t peer) noexcept {
  return checked([&] { return handle<Replica>(replica)->generation(peer); });
}
int64_t l1_replica_window(void* replica, size_t peer) noexcept {
  return checked([&] { return handle<Replica>(replica)->window(peer); });
}
int64_t l1_replica_capacity(void* replica, size_t peer) noexcept {
  return checked([&] { return handle<Replica>(replica)->capacity(peer); });
}
int64_t l1_replica_query(void* replica, const uint64_t* keys, size_t n,
                         uint8_t* out) noexcept {
  return checked([&] {
    handle<Replica>(replica)->query(keys, n, out);
    return 0;
  });
}
int64_t l1_replica_query_any(void* replica, const uint64_t* keys, size_t n,
                             uint8_t* out) noexcept {
  return checked([&] {
    return handle<Replica>(replica)->try_query_any(keys, n, out) ? 1 : 0;
  });
}
int64_t l1_owner_rpc(const char* endpoint, uint64_t op, uint64_t epoch,
                     uint64_t reader, uint64_t seq,
                     const vllm::peer_l1::Key* keys, int64_t* out, size_t n,
                     int timeout_ms, const Key* credential) noexcept {
  return checked([&] {
    vllm::peer_l1::Request request;
    request.op = op;
    request.epoch = epoch;
    request.reader = reader;
    request.seq = seq;
    request.n = n;
    if (credential) request.credential = *credential;
    return vllm::peer_l1::rpc(endpoint, request, keys, out, timeout_ms);
  });
}
int64_t l1_cleanup_start(int timeout_ms) noexcept {
  return checked([&] {
    return reinterpret_cast<int64_t>(
        new vllm::peer_l1::CleanupQueue(timeout_ms));
  });
}
int64_t l1_cleanup_stop(void* queue) noexcept {
  return checked([&] {
    delete static_cast<vllm::peer_l1::CleanupQueue*>(queue);
    return 0;
  });
}
int64_t l1_cleanup_pending(void* queue) noexcept {
  return checked([&] { return handle<CleanupQueue>(queue)->pending(); });
}
int64_t l1_cleanup_create(void* queue, const char* endpoint, uint64_t op,
                          uint64_t epoch, uint64_t reader, uint64_t seq,
                          uint64_t lease_ms, void* next,
                          const Key* credential) noexcept {
  return checked([&] {
    vllm::peer_l1::Request request;
    request.op = op;
    request.epoch = epoch;
    request.reader = reader;
    request.seq = seq;
    request.n = 0;
    if (credential) request.credential = *credential;
    auto* item = new vllm::peer_l1::Cleanup{handle<CleanupQueue>(queue),
                                            endpoint, request};
    item->batch_next = static_cast<vllm::peer_l1::Cleanup*>(next);
    if (lease_ms)
      item->expires =
          vllm::peer_l1::Clock::now() + std::chrono::milliseconds(lease_ms);
    return reinterpret_cast<int64_t>(item);
  });
}
void l1_cleanup_enqueue(void* item) noexcept {
  vllm::peer_l1::completed_copy(item);
}
}
