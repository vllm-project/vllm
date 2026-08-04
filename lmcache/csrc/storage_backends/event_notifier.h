// SPDX-License-Identifier: Apache-2.0
#pragma once

// Cross-platform event notification abstraction (C++ mirror of
// lmcache/v1/platform/event_notifier.py). Provides a unified
// EventNotifier interface for signaling between threads using a
// poll-able file descriptor.
//
// On Linux we use eventfd(2); on macOS / other POSIX we fall back
// to a self-pipe with non-blocking I/O. Connector code stays
// platform-agnostic — only this header has #if branches.

#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <memory>
#include <stdexcept>

#if defined(__linux__)
  #include <sys/eventfd.h>
#endif

namespace lmcache {
namespace connector {

// Abstract base: a binary signal that a listener can poll() on.
// notify() makes fileno() readable; consume() resets it. Multiple
// notify() calls before a consume() are coalesced.
class EventNotifier {
 public:
  virtual ~EventNotifier() = default;

  // Poll-able file descriptor. Becomes readable after notify().
  virtual int fileno() const = 0;

  // Signal the notifier (idempotent if already signaled).
  virtual void notify() = 0;

  // Consume any pending signal (non-blocking). EAGAIN/EWOULDBLOCK
  // is the only swallowed error; other OS errors propagate.
  virtual void consume() = 0;

  // Release underlying OS resources. Idempotent.
  virtual void close() = 0;

  EventNotifier() = default;
  EventNotifier(const EventNotifier&) = delete;
  EventNotifier& operator=(const EventNotifier&) = delete;
};

#if defined(__linux__)

// Linux eventfd-based notifier.
class EventfdNotifier final : public EventNotifier {
 public:
  EventfdNotifier() {
    efd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (efd_ < 0) {
      throw std::runtime_error("failed to create eventfd");
    }
  }

  ~EventfdNotifier() override { close(); }

  int fileno() const override { return efd_; }

  void notify() override {
    uint64_t one = 1;
    for (;;) {
      ssize_t w = ::write(efd_, &one, sizeof(one));
      if (w == static_cast<ssize_t>(sizeof(one))) return;
      if (w < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        throw std::runtime_error("eventfd write failed unexpectedly");
      }
      throw std::runtime_error("partial write to eventfd");
    }
  }

  void consume() override {
    for (;;) {
      uint64_t x;
      ssize_t r = ::read(efd_, &x, sizeof(x));
      if (r == static_cast<ssize_t>(sizeof(x))) continue;
      if (r < 0 && errno == EINTR) continue;
      break;  // drained or non-EINTR error (EAGAIN expected)
    }
  }

  void close() override {
    if (efd_ >= 0) {
      ::close(efd_);
      efd_ = -1;
    }
  }

 private:
  int efd_ = -1;
};

#endif  // __linux__

// Pipe-based fallback notifier (macOS / other POSIX).
class PipeNotifier final : public EventNotifier {
 public:
  PipeNotifier() {
    int fds[2];
    if (::pipe(fds) != 0) {
      throw std::runtime_error("failed to create pipe");
    }
    for (int i = 0; i < 2; ++i) {
      int flags = ::fcntl(fds[i], F_GETFL, 0);
      if (flags < 0 || ::fcntl(fds[i], F_SETFL, flags | O_NONBLOCK) < 0 ||
          ::fcntl(fds[i], F_SETFD, FD_CLOEXEC) < 0) {
        ::close(fds[0]);
        ::close(fds[1]);
        throw std::runtime_error("failed to configure pipe fd flags");
      }
    }
    read_fd_ = fds[0];
    write_fd_ = fds[1];
  }

  ~PipeNotifier() override { close(); }

  int fileno() const override { return read_fd_; }

  void notify() override {
    const uint8_t one = 1;
    for (;;) {
      ssize_t w = ::write(write_fd_, &one, sizeof(one));
      if (w == 1) return;
      if (w < 0) {
        if (errno == EINTR) continue;
        // Pipe full = signal already pending; readers will see it.
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        throw std::runtime_error("pipe write failed unexpectedly");
      }
    }
  }

  void consume() override {
    uint8_t buf[256];
    for (;;) {
      ssize_t r = ::read(read_fd_, buf, sizeof(buf));
      if (r > 0) continue;
      if (r < 0 && errno == EINTR) continue;
      break;  // drained, EOF, or EAGAIN
    }
  }

  void close() override {
    if (read_fd_ >= 0) {
      ::close(read_fd_);
      read_fd_ = -1;
    }
    if (write_fd_ >= 0) {
      ::close(write_fd_);
      write_fd_ = -1;
    }
  }

 private:
  int read_fd_ = -1;
  int write_fd_ = -1;
};

// Factory: returns the platform-appropriate notifier.
inline std::unique_ptr<EventNotifier> make_event_notifier() {
#if defined(__linux__)
  return std::unique_ptr<EventNotifier>(new EventfdNotifier());
#else
  return std::unique_ptr<EventNotifier>(new PipeNotifier());
#endif
}

}  // namespace connector
}  // namespace lmcache
