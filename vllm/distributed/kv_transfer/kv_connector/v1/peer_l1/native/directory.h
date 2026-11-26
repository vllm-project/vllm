// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "wire.h"
#include <optional>
#include <utility>
#include <vector>

namespace vllm::peer_l1 {
// Stateless hasher; safe on any thread without locking.
struct KeyHash {
  size_t operator()(const Key& key) const { return key[0]; }
};
inline uint64_t mix(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
// Stateless hasher for replica keys; safe on any thread without locking.
struct ShortHash {
  size_t operator()(uint64_t key) const { return mix(key); }
};

template <class K, class V, class H>
// No internal locking; callers hold the owner mutex or replica shard lock.
class FlatMap {
  // Accessed only under the containing map's external lock.
  struct Slot {
    std::optional<std::pair<K, V>> item;
    bool tombstone = false;
  };
  std::vector<Slot> slots_ = std::vector<Slot>(16);
  size_t size_ = 0, used_ = 0;
  size_t position(const K& key) const {
    auto pos = mix(H{}(key)) & (slots_.size() - 1);
    while (slots_[pos].item || slots_[pos].tombstone) {
      if (slots_[pos].item && slots_[pos].item->first == key) return pos;
      pos = (pos + 1) & (slots_.size() - 1);
    }
    return pos;
  }
  void rebuild() {
    auto old = std::move(slots_);
    slots_ = std::vector<Slot>(std::max<size_t>(16, size_ * 4));
    // position() uses a mask, so capacity must remain a power of two.
    size_t capacity = 16;
    while (capacity < slots_.size()) capacity *= 2;
    slots_.resize(capacity);
    size_ = used_ = 0;
    for (auto& s : old)
      if (s.item) insert(s.item->first, std::move(s.item->second));
  }

 public:
  V* find(const K& key) {
    auto& slot = slots_[position(key)];
    return slot.item ? &slot.item->second : nullptr;
  }
  bool insert(const K& key, V value) {
    if ((used_ + 1) * 2 >= slots_.size()) rebuild();
    auto pos = position(key);
    if (slots_[pos].item) return false;
    slots_[pos].item.emplace(key, std::move(value));
    slots_[pos].tombstone = false;
    ++size_;
    ++used_;
    return true;
  }
  bool erase(const K& key) {
    auto& slot = slots_[position(key)];
    if (!slot.item) return false;
    slot.item.reset();
    slot.tombstone = true;
    --size_;
    return true;
  }
  template <class F>
  void each(F fn) const {
    for (const auto& slot : slots_)
      if (slot.item) fn(slot.item->first, slot.item->second);
  }
  void clear() { *this = FlatMap{}; }
};

// Scheduler and RPC threads access row state under Owner::mutex_.
struct Row {
  uint64_t pins = 0;
  bool absorbed = false, withheld = false, window = false, dirty = false;
  bool budgeted = false, budget_pending = false;
  std::vector<Key> aliases;
};
}  // namespace vllm::peer_l1
