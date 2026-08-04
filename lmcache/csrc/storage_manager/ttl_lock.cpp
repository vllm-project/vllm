// SPDX-License-Identifier: Apache-2.0

#include "ttl_lock.h"

namespace lmcache {
namespace storage_manager {

TTLLock::TTLLock(uint32_t ttl_sec)
    : counter_(0),
      expiration_ms_(0),
      ttl_ms_(static_cast<int64_t>(ttl_sec) * 1000) {}

void TTLLock::lock() {
  int64_t current_time = now_ms();
  int64_t new_expiration = current_time + ttl_ms_;

  // Use compare-and-swap loop to handle the TTL expiration case
  while (true) {
    int64_t old_expiration = expiration_ms_.load(std::memory_order_acquire);
    int64_t old_counter = counter_.load(std::memory_order_acquire);

    // Check if TTL has expired
    bool expired = (current_time >= old_expiration);

    if (expired) {
      // TTL expired, try to reset counter to 1 and set new expiration
      // First, try to update the expiration
      if (expiration_ms_.compare_exchange_strong(old_expiration, new_expiration,
                                                 std::memory_order_seq_cst)) {
        // Successfully updated expiration, now set counter to 1
        counter_.store(1, std::memory_order_seq_cst);
        return;
      }
      // Another thread updated expiration, retry
      continue;
    } else {
      // TTL not expired, try to increment counter
      if (counter_.compare_exchange_strong(old_counter, old_counter + 1,
                                           std::memory_order_seq_cst)) {
        // Successfully incremented counter, update expiration
        expiration_ms_.store(new_expiration, std::memory_order_seq_cst);
        return;
      }
      // Another thread modified counter, retry
      continue;
    }
  }
}

void TTLLock::unlock() {
  // Use compare-and-swap loop to ensure we don't go below 0
  while (true) {
    int64_t old_counter = counter_.load(std::memory_order_acquire);

    if (old_counter <= 0) {
      // Already at 0, nothing to do
      return;
    }

    if (counter_.compare_exchange_strong(old_counter, old_counter - 1,
                                         std::memory_order_seq_cst)) {
      return;
    }
    // Another thread modified counter, retry
  }
}

bool TTLLock::is_locked() const {
  int64_t current_time = now_ms();
  int64_t expiration = expiration_ms_.load(std::memory_order_acquire);
  int64_t counter = counter_.load(std::memory_order_acquire);

  // Lock is held if counter > 0 AND TTL not expired
  return (counter > 0) && (current_time < expiration);
}

void TTLLock::reset() {
  counter_.store(0, std::memory_order_seq_cst);
  expiration_ms_.store(0, std::memory_order_seq_cst);
}

int64_t TTLLock::now_ms() {
  auto now = Clock::now();
  return to_ms(now);
}

int64_t TTLLock::to_ms(const TimePoint& tp) {
  return static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          tp.time_since_epoch())
          .count());
}

}  // namespace storage_manager
}  // namespace lmcache
