// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "socket.h"
#include <array>
#include <chrono>
#include <poll.h>
#include <unistd.h>

namespace vllm::peer_l1 {
using Clock = std::chrono::steady_clock;
using Key = std::array<uint64_t, 4>;
enum Operation : uint64_t {
  Release = 3,
  Cancel = 4,
  Cleanups = 6,
  AcquireExact,
  Hello,
  AcquireLeased,
  Subscribe = 10,
  AcquireDeferred = 11
};
struct Request {
  uint64_t magic = 0x4c314f574e455231ULL;
  uint64_t op, epoch, reader, seq, n;
};
struct Reply {
  uint64_t epoch, n;
};

inline bool transfer_bytes(int fd, void* data, size_t size, bool send,
                           Clock::time_point deadline) {
  auto* p = static_cast<char*>(data);
  while (size) {
    auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - Clock::now())
                    .count();
    if (left <= 0) return false;
    pollfd descriptor{fd, static_cast<short>(send ? POLLOUT : POLLIN), 0};
    int ready = poll(&descriptor, 1, left);
    if (ready < 0 && errno == EINTR) continue;
    if (ready <= 0) return false;
    auto n = send ? ::send(fd, p, size, MSG_NOSIGNAL | MSG_DONTWAIT)
                  : recv(fd, p, size, MSG_DONTWAIT);
    if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (n <= 0) return false;
    p += n;
    size -= n;
  }
  return true;
}

}  // namespace vllm::peer_l1
