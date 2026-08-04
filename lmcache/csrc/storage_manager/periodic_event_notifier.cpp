// SPDX-License-Identifier: Apache-2.0

#include "periodic_event_notifier.h"

#include <cerrno>
#include <cstdio>
#include <algorithm>

namespace lmcache {
namespace storage_manager {

std::mutex PeriodicEventNotifier::create_mutex_;
std::unique_ptr<PeriodicEventNotifier> PeriodicEventNotifier::instance_;

PeriodicEventNotifier::PeriodicEventNotifier(int interval_ms, bool use_eventfd)
    : interval_ms_(std::max(1, interval_ms)), use_eventfd_(use_eventfd) {
  thread_ = std::thread(&PeriodicEventNotifier::thread_func, this);
}

PeriodicEventNotifier::~PeriodicEventNotifier() {
  stop_.store(true, std::memory_order_release);
  cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void PeriodicEventNotifier::create(int interval_ms, bool use_eventfd) {
  std::lock_guard<std::mutex> lock(create_mutex_);
  if (instance_) {
    return;
  }
  instance_.reset(new PeriodicEventNotifier(interval_ms, use_eventfd));
}

PeriodicEventNotifier* PeriodicEventNotifier::get() { return instance_.get(); }

void PeriodicEventNotifier::shutdown() {
  std::lock_guard<std::mutex> lock(create_mutex_);
  if (!instance_) {
    return;
  }
  instance_->stop_.store(true, std::memory_order_release);
  instance_->cv_.notify_all();
  if (instance_->thread_.joinable()) {
    instance_->thread_.join();
  }
  instance_.reset();
}

void PeriodicEventNotifier::register_fd(int fd) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    fds_.insert(fd);
  }
  cv_.notify_one();
}

void PeriodicEventNotifier::unregister_fd(int fd) {
  std::lock_guard<std::mutex> lock(mutex_);
  fds_.erase(fd);
}

void PeriodicEventNotifier::set_interval_ms(int interval_ms) {
  interval_ms_.store(std::max(1, interval_ms), std::memory_order_release);
  cv_.notify_one();
}

void PeriodicEventNotifier::signal_fd(int fd) {
  if (use_eventfd_) {
    uint64_t one = 1;
    for (;;) {
      ssize_t w = ::write(fd, &one, sizeof(one));
      if (w == static_cast<ssize_t>(sizeof(one))) return;
      if (w < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        return;
      }
    }
  } else {
    uint8_t one = 1;
    for (;;) {
      ssize_t w = ::write(fd, &one, sizeof(one));
      if (w == 1) return;
      if (w < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        return;
      }
    }
  }
}

void PeriodicEventNotifier::thread_func() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] {
        return stop_.load(std::memory_order_acquire) || !fds_.empty();
      });
      if (stop_.load(std::memory_order_acquire)) {
        return;
      }
    }

    while (!stop_.load(std::memory_order_acquire)) {
      std::vector<int> snapshot;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fds_.empty()) {
          break;
        }
        snapshot.assign(fds_.begin(), fds_.end());
      }

      for (int fd : snapshot) {
        signal_fd(fd);
      }

      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock,
                     std::chrono::milliseconds(
                         interval_ms_.load(std::memory_order_acquire)),
                     [this] { return stop_.load(std::memory_order_acquire); });
      }
    }
  }
}

}  // namespace storage_manager
}  // namespace lmcache
