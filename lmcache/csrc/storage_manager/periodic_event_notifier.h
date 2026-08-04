// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

namespace lmcache {
namespace storage_manager {

class PeriodicEventNotifier {
 public:
  static void create(int interval_ms, bool use_eventfd);
  static PeriodicEventNotifier* get();
  static void shutdown();

  void register_fd(int fd);
  void unregister_fd(int fd);
  void set_interval_ms(int interval_ms);

  ~PeriodicEventNotifier();

  PeriodicEventNotifier(const PeriodicEventNotifier&) = delete;
  PeriodicEventNotifier& operator=(const PeriodicEventNotifier&) = delete;

 private:
  PeriodicEventNotifier(int interval_ms, bool use_eventfd);

  void thread_func();
  void signal_fd(int fd);

  static std::mutex create_mutex_;
  static std::unique_ptr<PeriodicEventNotifier> instance_;

  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};

  std::atomic<int> interval_ms_;
  bool use_eventfd_;

  std::unordered_set<int> fds_;
};

}  // namespace storage_manager
}  // namespace lmcache
