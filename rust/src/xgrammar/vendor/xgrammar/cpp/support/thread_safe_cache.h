/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/thread_safe_cache.h
 * \brief The header for thread-safe caching functionality.
 */
#ifndef XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_
#define XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_

#include <atomic>
#include <chrono>  // IWYU pragma: keep
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

#include "container.h"

namespace xgrammar {

/*!
 * \brief Primary template for ThreadSafeCache
 * \details This class provides thread-safe caching functionality in two forms:
 * 1. Single value cache when only Value template parameter is provided
 * 2. Key-value cache when both Key and Value template parameters are provided
 */
template <typename... Args>
class ThreadSafeCache;

/*!
 * \brief Thread-safe cache for a single computed value
 * \tparam Value The type of value being cached
 * \details Specialization that provides:
 * - Thread-safe access to a single cached value
 * - Lazy computation on first access
 * - Reader-writer locking for concurrent reads
 */
template <typename Value>
class ThreadSafeCache<Value> {
 public:
  /*!
   * \brief Constructs a new single-value cache
   * \param compute The function that computes the cached value
   */
  explicit ThreadSafeCache(std::function<Value()> compute) : compute_(std::move(compute)) {}

  /*!
   * \brief Gets or computes the cached value
   * \return The cached or newly computed value
   */
  Value Get() {
    // First try reading from cache with shared lock
    {
      std::shared_lock<std::shared_mutex> cache_lock(cache_mutex_);
      if (cache_.has_value()) {
        return cache_.value();  // Cache hit
      }
    }

    // Acquire exclusive lock to compute value
    std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);

    // Double-check to prevent redundant computation
    if (cache_.has_value()) {
      return cache_.value();
    }

    Value value = compute_();
    XGRAMMAR_DCHECK(!cache_.has_value());
    cache_ = value;
    return value;
  }

  /*!
   * \brief Clears the cached value
   * This function removes the cached value, so the next call to Get() will recompute it.
   */
  void Clear() {
    std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
    cache_.reset();
  }

 private:
  /*! \brief Optional container holding the cached value */
  std::optional<Value> cache_;
  /*! \brief Function used to compute the value when not cached */
  std::function<Value()> compute_;
  /*! \brief Reader-writer lock protecting access to cache_ */
  std::shared_mutex cache_mutex_;
};

/*!
 * \brief A thread-safe key-value cache with on-demand computation
 * \tparam Key The type of keys used to lookup values. Should be hashable.
 * \tparam Value The type of values stored in the cache
 * \details This cache provides thread-safe access to computed values with the following features:
 * - Lazy computation: Values are only computed when first requested
 * - Thread safety: Uses reader-writer locks for concurrent reads
 * - Parallel computation: Different keys can be computed simultaneously
 * - Double-checked locking: Prevents redundant computation
 */
template <typename Key, typename Value>
class ThreadSafeCache<Key, Value> {
 public:
  /*!
   * \brief Constructs a new thread-safe cache
   * \param compute The function that computes values for uncached keys
   */
  explicit ThreadSafeCache(std::function<Value(const Key&)> compute)
      : compute_(std::move(compute)) {}

  /*!
   * \brief Gets or computes the value for a key
   * \param key The key to lookup
   * \return The cached or newly computed value of the key
   */
  Value Get(const Key& key) {
    // Why we need this:
    // - When adding new elements to a unordered_map, the map may be rehashed,
    // - which means all the iterators may be invalidated.
    // - However, cppreference says:
    // - "References and pointers to either key or data stored in the container are only invalidated
    // - by erasing that element, even when the corresponding iterator is invalidated."
    // - (See https://en.cppreference.com/w/cpp/container/unordered_map)
    // - Therefore, we should maintain 2 locks.
    // - When we add something to the cache, we should hold the cache_mutex_.
    // - When we erase something from the cache, we should hold the clear_mutex_.

    auto erase_lock = std::shared_lock(erase_mutex_);

    // First attempt to read from cache_
    {
      auto cache_lock = std::shared_lock(cache_mutex_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {    // Cache hit
        auto& entry = it->second;  // The iterator is invalidated after releasing the lock
        cache_lock.unlock();       // Therefore, we should hold the entry by reference first

        // We should not hold lock here, since this function may be blocking.
        return entry.get(compute_, key);
      }
    }

    // Acquire exclusive lock to compute value
    {
      auto cache_lock = std::unique_lock(cache_mutex_);
      auto& entry = cache_[key];  // Create a new entry
      cache_lock.unlock();        // Release the lock before blocking

      // We should not hold lock here, since this function may be blocking.
      return entry.get(compute_, key);
    }
  }

  /*!
   * \brief Clears all cached values and associated per-key mutexes
   * This function removes all cached key-value pairs, so subsequent calls to Get() will recompute
   * them.
   */
  void Clear() {
    auto erase_lock = std::unique_lock(erase_mutex_);
    cache_.clear();
  }

 private:
  struct Entry {
    Value value;
    std::once_flag flag;
    const Value& get(const std::function<Value(const Key&)>& f, const Key& key) {
      // block in this lambda until the value is computed
      std::call_once(flag, [&] { value = f(key); });
      return value;
    }
  };

  /*! \brief The cache mapping keys to computed values */
  std::unordered_map<Key, Entry> cache_;
  /*! \brief The function used to compute values for uncached keys */
  std::function<Value(const Key&)> compute_;
  /*! \brief Reader-writer lock protecting access to cache_ */
  std::shared_mutex cache_mutex_;
  /*! \brief Mutex protecting removing elements */
  std::shared_mutex erase_mutex_;
};

namespace details {

template <typename Key, typename Value>
class LRUCacheImpl {
 public:
  struct Entry {
    Value value;  // value of the node
    int index;    // node index
  };

  /*! \brief Visits the node and moves it to the back of the LRU list. Return its value. */
  const Value& LRUVisit(const std::pair<const Key, Entry>& pair) {
    const auto& entry = pair.second;
    lru_list_.MoveBack(entry.index);
    return entry.value;
  }

  /*! \brief Initializes the node with the given value and moves it to the back of the LRU list. */
  void LRUInit(std::pair<const Key, Entry>& pair, const Value& init) {
    auto& entry = pair.second;
    entry.value = init;
    entry.index = lru_list_.PushBack(&pair).Index();
  }

  /*!
   * \brief Evicts the least recently used nodes until the predicate returns false.
   * \param predicate The function that returns true if eviction should continue.
   * \param evict The function takes a value and returns true if the value can be evicted.
   * This will be only called when the predicate returns true.
   * If this function returns true, it should update the size information before return.
   * \details This function will evict the least recently used nodes until the predicate returns
   * false. The evict function will be called for each node to determine if it should be evicted.
   */
  template <typename Predicate, typename Evict>
  void LRUEvict(const Predicate& predicate, const Evict& evict) {
    if (!predicate()) return;

    auto iter = lru_list_.begin();
    if (iter == lru_list_.end()) return;

    do {
      auto& [key, entry] = **iter;
      if (evict(entry.value)) {
        iter = lru_list_.Erase(iter);
        map_.erase(key);
      } else {
        ++iter;  // simply skip those waiting for computation
      }
    } while (predicate() && iter != lru_list_.end());
  }

  std::unordered_map<Key, Entry>& GetMap() { return map_; }

 private:
  std::unordered_map<Key, Entry> map_;
  List<std::pair<const Key, Entry>*> lru_list_;
};

}  // namespace details

/**
 * \brief A thread-safe key-value cache with on-demand computation and LRU eviction
 * \tparam Key The type of keys used to lookup values. Should be hashable.
 * \tparam Value The type of values stored in the cache
 * \tparam Computer The functor that computes values for uncached keys
 * \tparam SizeEstimator The functor that estimates the size of a value in bytes
 * \details This cache provides thread-safe access to computed values with the following features:
 * - Lazy computation: Values are only computed when first requested
 * - LRU eviction: When the cache is full, the least recently used value is evicted
 * - Thread safety: Uses reader-writer locks for concurrent reads
 * \attention User should guarantee the following:
 * 1. The policy class should provide a compute method that takes a key and returns a value.
 * 2. The value type should have a MemorySize method that returns the size of the value in bytes.
 */
template <typename Key, typename Value, typename Computer, typename SizeEstimator>
class ThreadSafeLRUCache {
 private:
  struct SizedValue {
    Value value;
    std::size_t size;
  };

 public:
  inline static constexpr std::size_t kUnlimitedSize = static_cast<std::size_t>(-1);

  explicit ThreadSafeLRUCache(
      std::size_t max_size = kUnlimitedSize,
      const Computer& computer = Computer{},
      const SizeEstimator& size_estimator = SizeEstimator{}
  )
      : max_size_(max_size), computer_(computer), size_estimator_(size_estimator), cache_() {}

  std::size_t MaxMemorySize() const { return max_size_; }
  std::size_t MemorySize() const { return current_size_; }

  Value Get(const Key& key) {
    auto future = GetFuture(key);
    return future.get().value;
  }

  void Clear() {
    // Remove all the ready entries.
    const auto lock_map = std::lock_guard{map_mutex_};
    if (this->max_size_ == kUnlimitedSize)
      cache_.GetMap().clear();
    else
      cache_.LRUEvict(
          [] { return true; },
          [&](const std::shared_future<SizedValue>& value) {
            // always evict and block until the value is ready
            try {
              current_size_ -= value.get().size;
            } catch (...) {
              // fine, just ignore the exception, size is not updated
            }
            return true;
          }
      );
  }

 private:
  std::shared_future<SizedValue> GetFuture(const Key& key) {
    if (this->max_size_ == kUnlimitedSize) return GetFutureUnlimited(key);
    auto& map = cache_.GetMap();

    {
      auto lock_map = std::shared_lock{map_mutex_};
      auto it = map.find(key);
      if (it != map.end()) {
        // We only need to hold LRU lock when shared lock is held here.
        // When unique lock of map_mutex_ is held, only 1 thread can access the
        // LRU list at the same time, so we do not need to hold the LRU lock then.
        const auto lock_lru = std::lock_guard{lru_mutex_};
        return cache_.LRUVisit(*it);
      }
    }

    auto task = std::packaged_task<SizedValue()>{[this, &key] {
      auto value = computer_(key);
      auto result = SizedValue{value, size_estimator_(value)};
      current_size_ += result.size;
      return result;
    }};

    auto lock_map = std::unique_lock{map_mutex_};
    auto [it, success] = map.try_emplace(key);
    if (!success) return cache_.LRUVisit(*it);

    // in this case, we insert the task, and we need to compute the value
    auto future = task.get_future().share();

    // perform eviction if the cache is full
    cache_.LRUInit(*it, future);
    cache_.LRUEvict(
        [&] { return current_size_ > max_size_; },
        [&](const std::shared_future<SizedValue>& value) {
          using namespace std::chrono_literals;
          // if not ready, then do not wait and block here
          if (value.wait_for(0s) != std::future_status::ready) return false;
          try {
            current_size_ -= value.get().size;
          } catch (...) {
            // fine, just ignore the exception, size is not updated
          }
          return true;
        }
    );

    // perform the costly computation outside all locks
    lock_map.unlock();
    task();
    return future;
  }

  std::shared_future<SizedValue> GetFutureUnlimited(const Key& key) {
    auto& map = cache_.GetMap();

    {
      auto lock_map = std::shared_lock{map_mutex_};
      auto it = map.find(key);
      if (it != map.end()) return it->second.value;
    }

    auto task = std::packaged_task<SizedValue()>{[this, &key] {
      auto value = computer_(key);
      auto result = SizedValue{value, size_estimator_(value)};
      current_size_ += result.size;
      return result;
    }};

    auto lock_map = std::unique_lock{map_mutex_};
    auto [it, success] = map.try_emplace(key);
    if (!success) return it->second.value;

    auto future = task.get_future().share();
    it->second.value = future;

    // perform the costly computation outside all locks
    lock_map.unlock();
    task();
    return future;
  }

 private:
  const std::size_t max_size_;
  const Computer computer_;
  const SizeEstimator size_estimator_;
  details::LRUCacheImpl<Key, std::shared_future<SizedValue>> cache_;
  std::atomic_size_t current_size_{0};
  std::shared_mutex map_mutex_;
  std::mutex lru_mutex_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_
