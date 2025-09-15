// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <iterator>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

/*
 * A simple hash map with int32_t keys that's designed to be fast and compact:
 *   - Open addressing with triangular probing allows high load factors.
 *   - Iteration is very fast and cache-friendly to allow fast speculation.
 *   - int32_t should be all we need to store token and sequence IDs.
 */
template <class T>
class Int32Map {
public:
    using const_iterator_value = std::pair<int32_t, const T&>;

    Int32Map() = default;

    Int32Map(Int32Map&& o) noexcept
        : slots_(o.slots_), cap_(o.cap_), size_(o.size_), tombstones_(o.tombstones_) {
        o.slots_ = nullptr;
        o.cap_ = o.size_ = o.tombstones_ = 0;
    }

    Int32Map& operator=(Int32Map&& o) noexcept {
        if (this == &o) {
            return *this;
        }
        destroy_all_();
        delete[] slots_;
        slots_ = o.slots_;
        cap_ = o.cap_;
        size_ = o.size_;
        tombstones_ = o.tombstones_;
        o.slots_ = nullptr;
        o.cap_ = o.size_ = o.tombstones_ = 0;
        return *this;
    }

    Int32Map(const Int32Map&) = delete;

    Int32Map& operator=(const Int32Map&) = delete;

    ~Int32Map() {
        destroy_all_();
        delete[] slots_;
    }

    bool empty() const noexcept {
        return size_ == 0;
    }

    uint32_t size() const noexcept {
        return size_;
    }

    bool contains(int32_t key) const {
        if (key == KEY_EMPTY || key == KEY_TOMBSTONE) {
            throw std::invalid_argument("invalid key");
        }
        if (!slots_) {
            return false;
        }
        uint32_t idx;
        return probe_insert_or_find_(key, idx);
    }

    bool erase(int32_t key) {
        if (key == KEY_EMPTY || key == KEY_TOMBSTONE) {
            throw std::invalid_argument("invalid key");
        }
        if (!slots_) {
            return false;
        }
        uint32_t idx;
        if (!probe_insert_or_find_(key, idx)) {
            return false;
        }
        value_ptr_(slots_[idx])->~T();
        slots_[idx].key = KEY_TOMBSTONE;
        --size_;
        ++tombstones_;
        maybe_rehash_after_erase_();
        return true;
    }

    // Construct in-place if absent, otherwise return existing.
    template <class... Args>
    T& emplace(int32_t key, Args&&... args) {
        if (key == KEY_EMPTY || key == KEY_TOMBSTONE) {
            throw std::invalid_argument("invalid key");
        }

        // Allocate minimal table if needed.
        if (!slots_) {
            cap_ = MIN_CAPACITY;
            slots_ = new Slot[cap_];
            size_ = tombstones_ = 0;
        }

        // Probe once.
        uint32_t idx;
        if (probe_insert_or_find_(key, idx)) {
            return *value_ptr_(slots_[idx]);  // already present
        }

        // If we can reuse a tombstone, do it immediately without rehash.
        if (slots_[idx].key == KEY_TOMBSTONE) {
            --tombstones_;
            slots_[idx].key = key;
            ::new (static_cast<void*>(&slots_[idx].storage)) T(std::forward<Args>(args)...);
            ++size_;
            return *value_ptr_(slots_[idx]);
        }

        // We will use an EMPTY slot, which increases (size + tombstones).
        if (static_cast<uint64_t>(cap_) * MAX_LOAD_PCT <
                static_cast<uint64_t>(size_ + tombstones_ + 1) * 100) {
            // Will exceed max load factor after insert, we need to rehash.
            if (static_cast<uint64_t>(size_ + 1) * 100 <=
                    static_cast<uint64_t>(cap_) * MAX_LOAD_PCT) {
                // Load would fit without tombstones, prefer same-cap cleanup.
                rehash_(cap_);
            } else {
                // Grow capacity to the next power of 2.
                rehash_(cap_ * 2);
            }
            // Re-probe after rehash.
            bool found = probe_insert_or_find_(key, idx);
            assert(!found);  // Re-hashing should not change the key set.
        }

        assert(slots_[idx].key == KEY_EMPTY);  // Must have an empty slot now.

        slots_[idx].key = key;
        ::new (static_cast<void*>(&slots_[idx].storage)) T(std::forward<Args>(args)...);
        ++size_;
        return *value_ptr_(slots_[idx]);
    }

    // Default-construct in-place if absent, otherwise return existing.
    T& operator[](int32_t key) {
        return emplace(key);
    }

    size_t memory_usage() const noexcept {
        return sizeof(*this) + sizeof(Slot) * cap_;
    }

    class const_iterator {
    public:
        using value_type = const_iterator_value;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        const_iterator() : m_(nullptr), i_(0) {}

        const_iterator(const Int32Map* m, uint32_t i) : m_(m), i_(i) {
            advance_();
        }

        value_type operator*() const {
            const Slot& s = m_->slots_[i_];
            return { s.key, *m_->value_ptr_(s) };
        }

        const_iterator& operator++() {
            ++i_;
            advance_();
            return *this;
        }

        bool operator==(const const_iterator& o) const {
            return m_ == o.m_ && i_ == o.i_;
        }

        bool operator!=(const const_iterator& o) const {
            return !(*this == o);
        }

    private:
        void advance_() {
            const uint32_t c = m_ ? m_->cap_ : 0u;
            while (m_ && i_ < c && !m_->is_filled_(m_->slots_[i_].key)) {
                ++i_;
            }
        }
        const Int32Map* m_;
        uint32_t i_;
    };

    const_iterator begin() const {
        return const_iterator(this, 0);
    }

    const_iterator end() const {
        return const_iterator(this, cap_);
    }

    const_iterator cbegin() const {
        return begin();
    }

    const_iterator cend() const {
        return end();
    }

private:
    // Reserved key representing an empty slot.
    static constexpr int32_t KEY_EMPTY = INT32_MIN;

    // Reserved key representing a deleted slot (tombstone).
    static constexpr int32_t KEY_TOMBSTONE = INT32_MIN + 1;

    // Keep 2 * MIN_LOAD_PCT < MAX_LOAD_PCT with some buffer to avoid thrashing.
    static constexpr uint32_t MIN_LOAD_PCT = 25;
    static constexpr uint32_t MAX_LOAD_PCT = 75;

    // Capacity must be a power of 2 for triangular probing to cover all indices.
    static constexpr uint32_t MIN_CAPACITY = 2;

    struct Slot {
        int32_t key;
        typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;
        Slot() noexcept : key(KEY_EMPTY) {}
    };

    // ----- Data (1 pointer + 3x uint32_t) -----
    Slot* slots_ = nullptr;
    uint32_t cap_ = 0;  // capacity, power of two (0 == unallocated)
    uint32_t size_ = 0;  // number of FILLED
    uint32_t tombstones_ = 0;  // number of TOMBSTONE

    // ----- Helpers -----
    static bool is_filled_(int32_t k) noexcept {
        return k != KEY_EMPTY && k != KEY_TOMBSTONE;
    }

    static uint32_t mix_hash_(int32_t key) noexcept {
        // 32-bit mix (Murmur-inspired)
        uint32_t x = static_cast<uint32_t>(key);
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    static T* value_ptr_(Slot& s) noexcept {
        return std::launder(reinterpret_cast<T*>(&s.storage));
    }

    static const T* value_ptr_(const Slot& s) noexcept {
        return std::launder(reinterpret_cast<const T*>(&s.storage));
    }

    void destroy_all_() noexcept {
        if (!slots_) {
            return;
        }
        for (uint32_t i = 0; i < cap_; ++i) {
            if (is_filled_(slots_[i].key)) {
                value_ptr_(slots_[i])->~T();
                slots_[i].key = KEY_EMPTY;
            }
        }
        size_ = tombstones_ = 0;
    }

    void maybe_rehash_after_erase_() {
        if (!slots_) {
            return;
        }

        // If completely empty: free everything and return.
        if (size_ == 0) {
            delete[] slots_;
            slots_ = nullptr;
            cap_ = size_ = tombstones_ = 0;
            return;
        }

        // If too sparse, shrink by 1/2.
        if (static_cast<uint64_t>(size_) * 100 <
               static_cast<uint64_t>(cap_) * MIN_LOAD_PCT) {
            if (cap_ / 2 >= MIN_CAPACITY) {
                rehash_(cap_ / 2);
            }
        }
    }

    // Either finds existing (true, idx_out set) or returns best insert slot (false).
    // On "not found", idx_out is: first tombstone if any; else first empty.
    bool probe_insert_or_find_(int32_t key, uint32_t& idx_out) const {
        assert(slots_ && cap_ > 0 && "probe on uninitialized map");
        uint32_t idx  = mix_hash_(key) & (cap_ - 1);
        uint32_t step = 0;
        bool has_first_tomb = false;
        uint32_t first_tomb_idx = 0;
        for (uint32_t probes = 0; probes < cap_; ++probes) {
            int32_t k = slots_[idx].key;
            if (k == key) {
                idx_out = idx;
                return true;
            }
            if (k == KEY_EMPTY) {
                idx_out = has_first_tomb ? first_tomb_idx : idx;
                return false;
            }
            if (k == KEY_TOMBSTONE && !has_first_tomb) {
                first_tomb_idx = idx;
                has_first_tomb = true;
            }
            ++step;
            idx = (idx + step) & (cap_ - 1);  // triangular probing
        }
        if (!has_first_tomb) {
            // This should never happen if load factor is correctly maintained.
            throw std::runtime_error("Int32Map is full");
        }
        idx_out = first_tomb_idx;
        return false;
    }

    template <class U>
    static void place_new_(Slot* arr, uint32_t cap, int32_t key, U&& val) {
        uint32_t idx  = mix_hash_(key) & (cap - 1);
        uint32_t step = 0;
        for (uint32_t probes = 0; probes < cap; ++probes) {
            int32_t k = arr[idx].key;
            if (k == KEY_EMPTY || k == KEY_TOMBSTONE) {
                arr[idx].key = key;
                ::new (static_cast<void*>(&arr[idx].storage)) T(std::forward<U>(val));
                return;
            }
            ++step;
            idx = (idx + step) & (cap - 1);
        }
        assert(false && "rehash placement failed");
    }

    void rehash_(uint32_t new_cap) {
        assert((new_cap & (new_cap - 1)) == 0 && new_cap >= MIN_CAPACITY);
        Slot* fresh = new Slot[new_cap];  // keys default to KEY_EMPTY

        if (slots_) {
            for (uint32_t i = 0; i < cap_; ++i) {
                auto& s = slots_[i];
                if (!is_filled_(s.key)) {
                    continue;
                }
                int32_t k = s.key;
                T* v  = value_ptr_(s);
                place_new_(fresh, new_cap, k, std::move(*v));
                v->~T();
                s.key = KEY_EMPTY;
            }
            delete[] slots_;
        }

        slots_ = fresh;
        cap_ = new_cap;
        tombstones_ = 0;  // cleaned
        // size_ unchanged
    }
};
