// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "wire.h"
#include <map>
#include <memory>

namespace vllm::peer_l1 {
// Confined to the calling thread's connection cache; never shared across RPCs
// on different threads. A post-fork caller resets inherited descriptors.
class TcpConnection {
 public:
  int fd = -1;
  pid_t process = getpid();
  ~TcpConnection() { reset(); }
  void reset() {
    if (fd >= 0) close(fd);
    fd = -1;
    process = getpid();
  }
};

inline TcpConnection& connection(const char* endpoint) {
  thread_local std::map<std::string, std::unique_ptr<TcpConnection>> cache;
  auto found = cache.find(endpoint);
  if (found == cache.end()) {
    // Bound idle descriptors when a resolver visits many replaced owners.
    if (cache.size() >= 128) cache.erase(cache.begin());
    found = cache.emplace(endpoint, std::make_unique<TcpConnection>()).first;
  }
  auto& result = *found->second;
  if (result.process != getpid()) result.reset();
  return result;
}

// ABI result: nonnegative = reply words, -1 = unknown, -2 = changed owner
// epoch. The caller blocks only until the common connect/send/receive deadline.
inline int64_t rpc(const char* endpoint, Request request, const Key* keys,
                   int64_t* out, int timeout_ms) {
  SocketAddress addr;
  try {
    addr = address(endpoint);
  } catch (const std::exception&) {
    return -1;
  }
  auto* cached =
      addr.storage.ss_family == AF_UNIX ? nullptr : &connection(endpoint);
  int fd = cached ? cached->fd : -1;
  bool fresh = fd < 0;
  if (fresh)
    fd = socket(addr.storage.ss_family,
                SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) return -1;
  auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  int64_t result = -1;
  if (!fresh ||
      (no_delay(fd, addr.storage.ss_family) &&
       (connect(fd, addr.data(), addr.length) == 0 || errno == EINPROGRESS))) {
    Reply reply{};
    if (transfer_bytes(fd, &request, sizeof(request), true, deadline) &&
        transfer_bytes(fd, const_cast<Key*>(keys),
                       (request.op == Hello || request.op == Members
                            ? 0
                            : request.n * sizeof(Key)),
                       true, deadline) &&
        transfer_bytes(fd, &reply, sizeof(reply), false, deadline)) {
      if (reply.epoch != request.epoch &&
          !(request.op == Members && !request.epoch))
        result = -2;
      else if (reply.n <=
                   std::max<uint64_t>(
                       request.n, request.op == AcquireExact ||
                                          request.op == AcquireLeased ||
                                          request.op == AcquireDeferred ||
                                          request.op == AcquireDeferredLeased
                                      ? 2
                                      : 0) &&
               transfer_bytes(fd, out, reply.n * sizeof(int64_t), false,
                              deadline))
        result = reply.n;
    }
  }
  if (cached) {
    cached->fd = fd;
    // A failed/partial reply cannot be reused as the next RPC's frame.
    if (result < 0) cached->reset();
  } else {
    close(fd);
  }
  return result;
}

}  // namespace vllm::peer_l1
