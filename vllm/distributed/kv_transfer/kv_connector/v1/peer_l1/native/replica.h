// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "owner.h"
#include <shared_mutex>

namespace vllm::peer_l1 {
// A native subscription thread writes hints; scheduler queries take shared
// locks. Always acquire snapshot_mutex_ before shard mutexes; metadata is
// atomic.
class Replica {
  // Hint maps require this shard lock beneath the replica snapshot lock.
  struct Shard {
    std::shared_mutex mutex;
    FlatMap<uint64_t, uint64_t, ShortHash> bitmap;
    FlatMap<uint64_t, std::vector<size_t>, ShortHash> table;
  };
  // Socket and frame progress belong exclusively to the subscription thread.
  struct Peer {
    std::string endpoint;
    uint64_t epoch, key_cap, sequence = 0;
    int fd = -1;
    Request request{};
    Frame frame{};
    size_t sent = 0, header = 0, body = 0;
    bool has_snapshot = false;
    std::vector<Event> events;
    Clock::time_point retry{}, progress = Clock::now();
  };
  std::array<Shard, 64> shards_;
  std::shared_mutex snapshot_mutex_;
  std::vector<Peer> peers_;
  std::unique_ptr<std::atomic<uint64_t>[]> generations_;
  std::unique_ptr<std::atomic<uint64_t>[]> windows_;
  ThreadError error_;
  std::atomic<bool> stop_{false};
  std::thread thread_;

  void set(uint64_t key, size_t peer, bool present) {
    auto& shard = shards_[mix(key) % shards_.size()];
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    if (peers_.size() <= 64) {
      auto* owners = shard.bitmap.find(key);
      uint64_t bit = uint64_t{1} << peer;
      if (present) {
        if (owners)
          *owners |= bit;
        else
          shard.bitmap.insert(key, bit);
      } else if (owners) {
        *owners &= ~bit;
        if (!*owners) shard.bitmap.erase(key);
      }
      return;
    }
    auto* owners = shard.table.find(key);
    if (!owners && present) {
      shard.table.insert(key, {peer});
      return;
    }
    if (!owners) return;
    auto found = std::find(owners->begin(), owners->end(), peer);
    if (present && found == owners->end()) owners->push_back(peer);
    if (!present && found != owners->end()) {
      owners->erase(found);
      if (owners->empty()) shard.table.erase(key);
    }
  }
  void clear(size_t peer) {
    for (auto& shard : shards_) {
      std::unique_lock<std::shared_mutex> lock(shard.mutex);
      std::vector<uint64_t> empty;
      if (peers_.size() <= 64) {
        shard.bitmap.each([&](uint64_t key, auto&) {
          auto* owners = shard.bitmap.find(key);
          *owners &= ~(uint64_t{1} << peer);
          if (!*owners) empty.push_back(key);
        });
        for (auto key : empty) shard.bitmap.erase(key);
        continue;
      }
      shard.table.each([&](uint64_t key, const auto&) {
        auto& owners = *shard.table.find(key);
        owners.erase(std::remove(owners.begin(), owners.end(), peer),
                     owners.end());
        if (owners.empty()) empty.push_back(key);
      });
      for (auto key : empty) shard.table.erase(key);
    }
  }
  bool apply(Peer& peer, size_t id) {
    auto& frame = peer.frame;
    if (frame.epoch != peer.epoch) {
      std::unique_lock<std::shared_mutex> lock(snapshot_mutex_);
      clear(id);
      generations_[id].store(0, std::memory_order_release);
      windows_[id].store(0, std::memory_order_release);
      return false;
    }
    if (!frame.snapshot &&
        (!peer.has_snapshot || frame.seq != peer.sequence + peer.events.size()))
      return false;
    // The old same-epoch snapshot stays queryable until this replacement.
    std::unique_lock<std::shared_mutex> lock(snapshot_mutex_);
    if (frame.snapshot) clear(id);
    uint64_t seq = peer.sequence;
    for (auto& event : peer.events) {
      if (!frame.snapshot && event.seq != ++seq) return false;
      set(event.key, id, event.present);
    }
    peer.sequence = frame.seq;
    peer.has_snapshot = true;
    windows_[id].store(frame.window, std::memory_order_release);
    generations_[id].store(frame.generation, std::memory_order_release);
    return true;
  }
  void disconnect(Peer& peer) {
    if (peer.fd >= 0) close(peer.fd);
    peer.fd = -1;
    peer.retry = Clock::now() + std::chrono::milliseconds(50);
    peer.sent = peer.header = peer.body = 0;
    peer.events.clear();
  }
  bool receive(Peer& peer, size_t id) {
    size_t budget = 256 * 1024;
    while (budget) {
      void* target;
      size_t remaining;
      if (peer.header < sizeof(Frame)) {
        target = reinterpret_cast<char*>(&peer.frame) + peer.header;
        remaining = sizeof(Frame) - peer.header;
      } else {
        target = reinterpret_cast<char*>(peer.events.data()) + peer.body;
        remaining = peer.events.size() * sizeof(Event) - peer.body;
      }
      if (remaining) {
        auto n =
            recv(peer.fd, target, std::min(remaining, budget), MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) return true;
        if (n <= 0) return false;
        budget -= n;
        peer.progress = Clock::now();
        if (peer.header < sizeof(Frame)) {
          peer.header += n;
          if (peer.header == sizeof(Frame)) {
            if (peer.frame.magic != Frame{}.magic) return false;
            if (peer.frame.epoch != peer.epoch) return apply(peer, id);
            if (peer.frame.snapshot > 1 ||
                peer.frame.n >
                    (peer.frame.snapshot ? peer.key_cap : kMaxDeltaEvents))
              return false;
            peer.events.resize(peer.frame.n);
          }
        } else
          peer.body += n;
      }
      if (peer.header == sizeof(Frame) &&
          peer.body == peer.events.size() * sizeof(Event)) {
        if (!apply(peer, id)) return false;
        peer.header = peer.body = 0;
        peer.events.clear();
      }
    }
    return true;
  }
  void run() noexcept {
    try {
      while (!stop_.load()) {
        std::vector<pollfd> ready;
        for (auto& peer : peers_) {
          if (peer.fd < 0 && Clock::now() >= peer.retry) {
            try {
              auto addr = address(peer.endpoint.c_str());
              peer.fd = socket(addr.storage.ss_family,
                               SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
              if (peer.fd < 0 || (connect(peer.fd, addr.data(), addr.length) &&
                                  errno != EINPROGRESS))
                disconnect(peer);
              peer.progress = Clock::now();
            } catch (...) {
              disconnect(peer);
            }
          }
          ready.push_back({peer.fd,
                           static_cast<short>(
                               peer.sent < sizeof(Request) ? POLLOUT : POLLIN),
                           0});
        }
        poll(ready.data(), ready.size(), 10);
        for (size_t i = 0; i < peers_.size(); ++i) {
          auto& peer = peers_[i];
          if (peer.fd < 0) continue;
          bool keep = !(ready[i].revents & (POLLERR | POLLHUP | POLLNVAL));
          try {
            if (keep && ready[i].revents & POLLOUT) {
              auto n = send(
                  peer.fd, reinterpret_cast<char*>(&peer.request) + peer.sent,
                  sizeof(Request) - peer.sent, MSG_NOSIGNAL | MSG_DONTWAIT);
              if (n > 0) {
                peer.sent += n;
                peer.progress = Clock::now();
              } else if (n == 0 || (errno != EINTR && errno != EAGAIN))
                keep = false;
            }
            if (keep && ready[i].revents & POLLIN) keep = receive(peer, i);
            if ((peer.sent < sizeof(Request) || peer.header || peer.body) &&
                Clock::now() - peer.progress > std::chrono::milliseconds(100))
              keep = false;
          } catch (...) {
            keep = false;
          }
          if (!keep) disconnect(peer);
        }
      }
    } catch (...) {
      error_.capture();
    }
  }

 public:
  void check_error() const { error_.check(); }
  Replica(const char* const* endpoints, const uint64_t* epochs,
          const uint64_t* key_caps, size_t n, const Key* credential)
      : generations_(new std::atomic<uint64_t>[n] {}),
        windows_(new std::atomic<uint64_t>[n] {}) {
    for (size_t i = 0; i < n; ++i) {
      Peer peer;
      peer.endpoint = subscription_endpoint(endpoints[i]);
      peer.epoch = epochs[i];
      peer.key_cap = key_caps[i];
      peer.request.op = Subscribe;
      if (credential) peer.request.credential = *credential;
      peer.request.epoch = epochs[i];
      peer.request.reader = peer.request.seq = peer.request.n = 0;
      peers_.push_back(std::move(peer));
    }
    thread_ = std::thread([this] { run(); });
  }
  ~Replica() {
    stop_.store(true);
    thread_.join();
    for (auto& peer : peers_)
      if (peer.fd >= 0) close(peer.fd);
  }
  uint64_t generation(size_t peer) const {
    if (peer >= peers_.size()) throw std::out_of_range("peer L1 replica peer");
    return generations_[peer].load(std::memory_order_acquire);
  }
  uint64_t window(size_t peer) const {
    if (peer >= peers_.size()) throw std::out_of_range("peer L1 replica peer");
    return windows_[peer].load(std::memory_order_acquire);
  }
  void query(const uint64_t* keys, size_t n, uint8_t* out) {
    std::shared_lock<std::shared_mutex> snapshot(snapshot_mutex_);
    std::fill(out, out + n * peers_.size(), 0);
    for (size_t i = 0; i < n; ++i) {
      auto& shard = shards_[mix(keys[i]) % shards_.size()];
      std::shared_lock<std::shared_mutex> lock(shard.mutex);
      if (peers_.size() <= 64) {
        auto* owners = shard.bitmap.find(keys[i]);
        uint64_t bits = owners ? *owners : 0;
        while (bits) {
          auto peer = __builtin_ctzll(bits);
          out[i * peers_.size() + peer] = 1;
          bits &= bits - 1;
        }
        continue;
      }
      auto* owners = shard.table.find(keys[i]);
      if (owners)
        for (auto peer : *owners) out[i * peers_.size() + peer] = 1;
    }
  }
};
}  // namespace vllm::peer_l1
