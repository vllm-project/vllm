// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "directory.h"
#include "rpc.h"
#include "tcp.h"
#include <sys/eventfd.h>
#include <algorithm>
#include <deque>
#include <map>
#include <mutex>
#include <set>

namespace vllm::peer_l1 {
// Values copied under the owner mutex, then sent by the subscription thread.
struct Event {
  uint64_t seq, key, present;
};
// Subscription-thread frame, populated from an owner-mutex snapshot.
struct Frame {
  // The magic versions this host-endian subscription ABI.
  uint64_t magic = 0x4c31535542534336ULL;
  uint64_t epoch, seq, n, snapshot, generation, window, capacity;
};

// Counters come first: absorb clears every entry before owner_reserve.
#define PEER_L1_OWNER_STATS(X)                                                \
  X(owner_busy_window, "counter", "owner_rejections", "window", "immediate")  \
  X(owner_busy_cap, "counter", "owner_rejections", "cap", "immediate")        \
  X(owner_deferred_reject_frozen, "counter", "owner_rejections", "frozen",    \
    "deferred")                                                               \
  X(owner_deferred_reject_lease_expiry, "counter", "owner_rejections",        \
    "lease_expiry", "deferred")                                               \
  X(owner_deferred_reject_missing_key, "counter", "owner_rejections",         \
    "missing_key", "deferred")                                                \
  X(owner_deferred_reject_cap, "counter", "owner_rejections", "cap",          \
    "deferred")                                                               \
  X(owner_deferred_reject_budget, "counter", "owner_rejections", "budget",    \
    "deferred")                                                               \
  X(owner_busy_protected_cap, "counter", "owner_rejections", "protected_cap", \
    "immediate")                                                              \
  X(owner_busy_gate, "counter", "owner_rejections", "gate", "immediate")      \
  X(owner_deferred_reject_gate, "counter", "owner_rejections", "gate",        \
    "deferred")                                                               \
  X(owner_reserve, "internal", "", "", "")                                    \
  X(owner_effective_window, "internal", "", "", "")                           \
  X(owner_protected_blocks, "gauge", "owner_protected_blocks", "", "")        \
  X(owner_protected_high_water, "gauge", "owner_protected_high_water", "", "")
enum OwnerStat {
#define PEER_L1_STAT_ENUM(name, kind, family, reason, stage) name,
  PEER_L1_OWNER_STATS(PEER_L1_STAT_ENUM)
#undef PEER_L1_STAT_ENUM
      OwnerStatCount
};
inline constexpr const char* owner_stat_schema =
#define PEER_L1_STAT_SCHEMA(name, kind, family, reason, stage) \
  #name ":" kind ":" family ":" reason ":" stage "\n"
    PEER_L1_OWNER_STATS(PEER_L1_STAT_SCHEMA);
#undef PEER_L1_STAT_SCHEMA
#undef PEER_L1_OWNER_STATS

// Lives in the scheduler process. RPC, subscription and scheduler calls share
// mutex_; only the scheduler touches the block pool. Helpers require mutex_
// unless they acquire it themselves; socket I/O runs outside that lock.
class Owner {
  int64_t (*clock_ns_)() = nullptr;
  Clock::time_point lease_now() const {
    return clock_ns_ ? Clock::time_point(std::chrono::nanoseconds(clock_ns_()))
                     : Clock::now();
  }
  using Identity = std::pair<uint64_t, uint64_t>;
  // RPC and scheduler threads access grants only while holding mutex_.
  struct Grant {
    std::vector<int64_t> rows;
    std::vector<int64_t> unique_rows;
    Clock::time_point created;
    bool leased;
    std::vector<Key> pending_keys;
    uint64_t queued_at = 0;
    bool deferred_leased = false;
  };
  std::mutex mutex_;
  std::vector<Row> rows_;
  FlatMap<Key, int64_t, KeyHash> directory_;
  std::map<Identity, Grant> grants_;
  std::map<uint64_t, uint64_t> completed_;
  std::set<std::pair<Clock::time_point, Identity>> expirations_;
  std::vector<int64_t> dirty_;
  std::vector<int64_t> window_rows_;
  std::map<uint64_t, size_t> short_keys_;
  using EventChunk = std::array<Event, 64>;
  std::deque<std::shared_ptr<EventChunk>> events_;
  uint64_t first_chunk_ = 0;
  uint64_t epoch_, cap_, window_max_, window_, new_pins_ = 0, withheld_ = 0;
  uint64_t protected_cap_, budget_rows_ = 0;
  std::vector<int64_t> budget_releases_;
  bool gate_open_ = true;
  uint8_t* published_;
  uint64_t sequence_ = 0, refs_ = 0;
  int64_t generation_ = 0;
  std::array<uint64_t, OwnerStatCount> stats_{};
  uint64_t queued_keys_ = 0;
  uint64_t deferred_serial_ = 0;
  bool frozen_ = false;
  int lease_ms_, wake_, sub_fd_ = -1;
  size_t log_capacity_;
  std::vector<int64_t> description_;
  std::vector<int64_t> members_;
  Key credential_{};
  std::unique_ptr<TcpServer> server_;
  ThreadError subscription_error_;
  std::atomic<bool> stop_{false};
  std::thread subscription_thread_;
  std::string sub_endpoint_;

  bool budget_enabled() const { return protected_cap_ != ~uint64_t{0}; }
  bool budget_fits(uint64_t extra, uint64_t reserve) const {
    return !budget_enabled() ||
           (budget_rows_ <= protected_cap_ &&
            extra <= protected_cap_ - budget_rows_ &&
            reserve <= protected_cap_ - budget_rows_ - extra);
  }
  uint64_t desired_window(uint64_t free) const {
    auto desired = gate_open_ ? std::min(window_max_, free / 9) : 0;
    if (budget_enabled())
      desired = std::min(desired, (protected_cap_ - budget_rows_) / 2);
    return desired;
  }
  void budget_stats() {
    if (!budget_enabled()) return;
    stats_[owner_protected_blocks] = budget_rows_ + window_ - withheld_;
    stats_[owner_protected_high_water] = std::max(
        stats_[owner_protected_high_water], stats_[owner_protected_blocks]);
    if (stats_[owner_protected_blocks] > protected_cap_)
      throw std::logic_error("peer protected capacity overflow");
  }
  void protect(Row& row) {
    if (budget_enabled() && !row.budgeted) {
      row.budgeted = true;
      ++budget_rows_;
    }
  }
  void unprotect(Row& row) {
    if (row.budgeted && !row.pins && !row.absorbed && !row.withheld &&
        !row.budget_pending) {
      row.budgeted = false;
      --budget_rows_;
    }
  }

  void wake() {
    uint64_t one = 1;
    while (write(wake_, &one, sizeof(one)) < 0 && errno == EINTR) {
    }
  }
  void changed(int64_t row) {
    if (rows_[row].dirty) return;
    bool was_empty = dirty_.empty();
    rows_[row].dirty = true;
    dirty_.push_back(row);
    if (was_empty) wake();
  }
  void event(const Key& key, bool present) {
    auto& count = short_keys_[key[0]];
    if (present ? count++ != 0 : --count != 0) return;
    if (!present) short_keys_.erase(key[0]);
    auto index = sequence_++ % EventChunk{}.size();
    if (!index) events_.push_back(std::make_shared<EventChunk>());
    events_.back()->at(index) = {sequence_, key[0], present};
    if (sequence_ - first_chunk_ * 64 > log_capacity_ + 63) {
      events_.pop_front();
      ++first_chunk_;
    }
  }
  void finish(Identity id) {
    auto saved = grants_.find(id);
    if (saved != grants_.end()) {
      auto& grant = saved->second;
      queued_keys_ -= grant.pending_keys.size();
      published_[rows_.size()] = queued_keys_ != 0;
      if (grant.leased)
        expirations_.erase(
            {grant.created + std::chrono::milliseconds(lease_ms_), id});
      for (auto row : grant.unique_rows) {
        --rows_[row].pins;
        --refs_;
        if (!rows_[row].pins) changed(row);
        unprotect(rows_[row]);
      }
      grants_.erase(saved);
    }
    completed_[id.first] = std::max(completed_[id.first], id.second);
    budget_stats();
  }
  void expire() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = lease_now();
    while (!expirations_.empty() && expirations_.begin()->first <= now) {
      auto id = expirations_.begin()->second;
      finish(id);
    }
  }
  std::vector<int64_t> dispatch(const Request& request,
                                const std::vector<Key>& keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (request.op == Hello)
      return description_.size() <= request.n ? description_
                                              : std::vector<int64_t>{};
    if (request.op == Members)
      return members_.size() <= request.n ? members_ : std::vector<int64_t>{};
    Identity id{request.reader, request.seq};
    if (request.op == Cleanups) {
      for (auto& key : keys) finish({key[0], key[1]});
      return {};
    }
    if (request.op == Release || request.op == Cancel) {
      finish(id);
      return {};
    }
    std::vector<int64_t> result;
    auto saved = grants_.find(id);
    if (saved != grants_.end())
      return saved->second.pending_keys.empty()
                 ? saved->second.rows
                 : std::vector<int64_t>{-3 - generation_, -1};
    if (frozen_ || request.seq <= completed_[request.reader] || keys.empty() ||
        keys.size() > cap_)
      return {};
    if (!gate_open_) {
      ++stats_[owner_busy_gate];
      return {-3 - generation_, -2};
    }
    std::set<int64_t> unique;
    for (auto& key : keys) {
      auto* row = directory_.find(key);
      if (!row) return {};
      result.push_back(*row);
      unique.insert(*row);
    }
    size_t extra = 0;
    size_t extra_budget = 0;
    for (auto row : unique) extra += !rows_[row].absorbed && !rows_[row].window;
    for (auto row : unique) extra_budget += !rows_[row].budgeted;
    if ((request.op == AcquireDeferred ||
         request.op == AcquireDeferredLeased) &&
        (new_pins_ + extra > window_ || refs_ + unique.size() > cap_ ||
         !budget_fits(extra_budget, window_ - withheld_))) {
      if (queued_keys_ + keys.size() > cap_) {
        ++stats_[owner_busy_cap];
        return {-3 - generation_, static_cast<int64_t>(window_)};
      }
      auto created = lease_now();
      grants_.emplace(id, Grant{{},
                                {},
                                created,
                                false,
                                keys,
                                ++deferred_serial_,
                                request.op == AcquireDeferredLeased});
      queued_keys_ += keys.size();
      published_[rows_.size()] = 1;
      wake();
      return {-3 - generation_, -1};
    }
    // ABI control reply: [-3 - generation, window]; -1 detail means queued,
    // -2 means gated. Nonnegative rows are reserved for successful grants.
    if (new_pins_ + extra > window_ || refs_ + unique.size() > cap_) {
      ++stats_[new_pins_ + extra > window_ ? owner_busy_window
                                           : owner_busy_cap];
      return {-3 - generation_, static_cast<int64_t>(window_)};
    }
    if (!budget_fits(extra_budget, window_ - withheld_)) {
      ++stats_[owner_busy_protected_cap];
      return {-3 - generation_, static_cast<int64_t>(window_)};
    }
    auto created = lease_now();
    bool leased =
        request.op == AcquireLeased || request.op == AcquireDeferredLeased;
    auto inserted = grants_.emplace(
        id, Grant{result, {unique.begin(), unique.end()}, created, leased, {}});
    try {
      if (leased)
        expirations_.emplace(created + std::chrono::milliseconds(lease_ms_),
                             id);
    } catch (...) {
      grants_.erase(inserted.first);
      throw;
    }
    for (auto row : unique) {
      auto& state = rows_[row];
      protect(state);
      if (!state.absorbed && !state.window) {
        state.window = true;
        window_rows_.push_back(row);
        ++new_pins_;
      }
      ++state.pins;
      ++refs_;
      changed(row);
    }
    budget_stats();
    return result;
  }

  // Socket progress is confined to the subscription thread; owner data is
  // copied separately under mutex_ when preparing each frame.
  struct Subscriber {
    Request request{};
    size_t received = 0, sent = 0;
    bool subscribed = false;
    uint64_t sequence = 0;
    uint64_t generation = 0;
    uint64_t capacity = 0;
    std::vector<char> buffer;
    Clock::time_point progress = Clock::now();
  };
  bool prepare(Subscriber& client) {
    std::vector<Event> batch;
    std::vector<std::shared_ptr<const EventChunk>> chunks;
    Frame frame;
    frame.epoch = epoch_;
    frame.snapshot = !client.subscribed;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (client.request.credential != credential_) return false;
      if (!client.subscribed) {
        for (auto [key, count] : short_keys_)
          batch.push_back({sequence_, key, 1});
      } else {
        if (sequence_ - client.sequence > log_capacity_) return false;
        if (client.sequence < sequence_) {
          auto first = client.sequence / 64 - first_chunk_;
          chunks.assign(events_.begin() + first, events_.end());
        }
      }
      frame.seq = sequence_;
      frame.generation = generation_;
      frame.window = window_;
      frame.capacity =
          gate_open_
              ? (budget_enabled()
                     ? std::min(protected_cap_ - budget_rows_, cap_ - refs_)
                     : cap_)
              : 0;
    }
    // References keep chunks alive across log rotation. Appends write only
    // entries beyond frame.seq, so the selected entries are immutable.
    if (!chunks.empty()) {
      batch.reserve(frame.seq - client.sequence);
      auto first = client.sequence / 64;
      for (auto seq = client.sequence + 1; seq <= frame.seq; ++seq)
        batch.push_back(chunks[(seq - 1) / 64 - first]->at((seq - 1) % 64));
    }
    if (client.subscribed && batch.empty() &&
        client.generation == frame.generation &&
        client.capacity == frame.capacity)
      return true;
    frame.n = batch.size();
    client.sequence = frame.seq;
    client.generation = frame.generation;
    client.capacity = frame.capacity;
    client.subscribed = true;
    client.buffer.resize(sizeof(frame) + sizeof(Event) * batch.size());
    memcpy(client.buffer.data(), &frame, sizeof(frame));
    if (!batch.empty())
      memcpy(client.buffer.data() + sizeof(frame), batch.data(),
             sizeof(Event) * batch.size());
    client.sent = 0;
    client.progress = Clock::now();
    return true;
  }
  void subscriptions() noexcept {
    std::map<int, Subscriber> clients;
    try {
      while (!stop_.load()) {
        std::vector<pollfd> ready{{sub_fd_, POLLIN, 0}};
        for (auto& [fd, client] : clients)
          ready.push_back(
              {fd,
               static_cast<short>(client.received < sizeof(Request)
                                      ? POLLIN
                                      : (client.buffer.empty() ? 0 : POLLOUT)),
               0});
        poll(ready.data(), ready.size(), 2);
        if (ready[0].revents & POLLIN) {
          for (int i = 0; i < 16; ++i) {
            int fd = accept4(sub_fd_, nullptr, nullptr,
                             SOCK_NONBLOCK | SOCK_CLOEXEC);
            if (fd < 0) break;
            if (clients.size() >= 1024)
              close(fd);
            else {
              try {
                clients.try_emplace(fd);
              } catch (...) {
                close(fd);
                throw;
              }
            }
          }
        }
        for (size_t i = 1; i < ready.size(); ++i) {
          auto fd = ready[i].fd;
          auto& client = clients.at(fd);
          bool keep = !(ready[i].revents & (POLLERR | POLLHUP | POLLNVAL));
          try {
            if (keep && client.received < sizeof(Request)) {
              auto n = recv(
                  fd,
                  reinterpret_cast<char*>(&client.request) + client.received,
                  sizeof(Request) - client.received, MSG_DONTWAIT);
              if (n > 0) {
                client.received += n;
                client.progress = Clock::now();
              } else if (n == 0 || (errno != EAGAIN && errno != EINTR))
                keep = false;
              if (client.received == sizeof(Request))
                keep = client.request.magic == Request{}.magic &&
                       client.request.op == Subscribe && !client.request.n;
            }
            if (keep && client.received == sizeof(Request)) {
              if (client.buffer.empty()) keep = prepare(client);
              size_t budget = 256 * 1024;
              while (keep && client.sent < client.buffer.size() && budget) {
                auto n =
                    send(fd, client.buffer.data() + client.sent,
                         std::min(budget, client.buffer.size() - client.sent),
                         MSG_NOSIGNAL | MSG_DONTWAIT);
                if (n < 0 && (errno == EAGAIN || errno == EINTR)) break;
                if (n <= 0) {
                  keep = false;
                  break;
                }
                client.sent += n;
                budget -= n;
                client.progress = Clock::now();
              }
              if (client.sent == client.buffer.size()) client.buffer.clear();
            }
            if ((client.received < sizeof(Request) || !client.buffer.empty()) &&
                Clock::now() - client.progress > std::chrono::milliseconds(100))
              keep = false;
          } catch (...) {
            keep = false;
          }
          if (!keep) {
            close(fd);
            clients.erase(fd);
          }
        }
      }
      for (auto& [fd, client] : clients) close(fd);
    } catch (...) {
      subscription_error_.capture();
    }
    for (auto& [fd, client] : clients) close(fd);
  }

 public:
  Owner(const char* endpoint, uint64_t epoch, size_t rows, uint64_t cap,
        uint64_t window, int lease_ms, size_t log_capacity,
        uint64_t initial_window, uint8_t* published, const Key* credential,
        int64_t (*clock_ns)() = nullptr, uint64_t protected_cap = ~uint64_t{0})
      : clock_ns_(clock_ns),
        rows_(rows),
        epoch_(epoch),
        cap_(cap),
        window_max_(window),
        window_(std::min(window, initial_window)),
        protected_cap_(protected_cap),
        published_(published),
        lease_ms_(lease_ms),
        wake_(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)),
        log_capacity_(std::min(log_capacity, kMaxDeltaEvents)),
        sub_endpoint_(subscription_endpoint(endpoint)) {
    if (budget_enabled()) window_ = std::min(window_, protected_cap_ / 2);
    budget_stats();
    if (credential) credential_ = *credential;
    if (wake_ < 0) throw std::runtime_error("peer L1 eventfd failed");
    try {
      dirty_.reserve(rows);
      window_rows_.reserve(std::min<uint64_t>(window, rows));
      auto addr = address(sub_endpoint_.c_str());
      sub_fd_ = socket(addr.storage.ss_family,
                       SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
      int one = 1;
      if (sub_fd_ < 0 ||
          (addr.storage.ss_family != AF_UNIX &&
           setsockopt(sub_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one))) ||
          bind(sub_fd_, addr.data(), addr.length) || listen(sub_fd_, 128))
        throw std::system_error(errno, std::generic_category(),
                                "peer L1 subscription bind failed");
      server_ = std::make_unique<TcpServer>(
          endpoint, epoch,
          [this](const Request& r, const std::vector<Key>& k) {
            return dispatch(r, k);
          },
          credential_, cap_, [this] { expire(); });
      subscription_thread_ = std::thread([this] { subscriptions(); });
    } catch (...) {
      server_.reset();
      if (sub_fd_ >= 0) close(sub_fd_);
      close(wake_);
      throw;
    }
  }
  ~Owner() {
    stop_.store(true);
    if (subscription_thread_.joinable()) subscription_thread_.join();
    server_.reset();
    close(sub_fd_);
    close(wake_);
  }
  void check_error() const {
    subscription_error_.check();
    server_->check_error();
  }
  int wake_fd() const { return wake_; }
  void gate(bool open) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gate_open_ != open) {
      gate_open_ = open;
      ++generation_;
    }
  }
  void describe(const int64_t* hello, size_t n_hello, const int64_t* members,
                size_t n_members) {
    std::lock_guard<std::mutex> lock(mutex_);
    description_.assign(hello, hello + n_hello);
    members_.assign(members, members + n_members);
  }
  void publish(const Key* keys, const int64_t* rows, size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < n; ++i) {
      auto& row = rows_.at(rows[i]);
      if (row.withheld || directory_.find(keys[i])) continue;
      row.aliases.reserve(row.aliases.size() + 1);
      if (directory_.insert(keys[i], rows[i])) {
        row.aliases.push_back(keys[i]);
        published_[rows[i]] = 1;
        event(keys[i], true);
      }
    }
  }
  int retire(int64_t row_id, int mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& row = rows_.at(row_id);
    for (const auto& key : row.aliases) {
      directory_.erase(key);
      event(key, false);
    }
    row.aliases.clear();
    published_[row_id] = 0;
    // ABI retirement modes: 0 = aliases only, 1 = withhold, 2 = adopt pool ref.
    if (row.pins && mode == 1 && !row.withheld) {
      row.withheld = true;
      ++withheld_;
      if (withheld_ > window_)
        throw std::logic_error("peer L1 window overflow");
      changed(row_id);
      budget_stats();
    }
    if (row.pins && mode == 2 && !row.absorbed) {
      // Protect an in-use row before its local reference can be freed.
      row.absorbed = true;
      return 2;
    }
    return row.pins != 0;
  }
  size_t deferred_rows(int64_t* out, size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::set<int64_t> rows;
    for (auto& [id, grant] : grants_)
      for (auto& key : grant.pending_keys)
        if (auto* row = directory_.find(key)) rows.insert(*row);
    if (rows.size() + 1 > capacity)
      throw std::invalid_argument("deferred output too small");
    out[0] = deferred_serial_;
    std::copy(rows.begin(), rows.end(), out + 1);
    return rows.size() + 1;
  }
  size_t absorb(int64_t* out, size_t capacity, uint64_t* stats,
                uint64_t free_blocks, const int64_t* in_use, size_t n_in_use,
                uint64_t deferred_snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (capacity < rows_.size() * 2)
      throw std::invalid_argument("absorb output too small");
    // The scheduler supplies refcounts without intervening allocation. Queue
    // arrivals after that snapshot wait for the next absorb.
    std::set<int64_t> live(in_use, in_use + n_in_use);
    uint64_t fast_touches = 0;
    for (auto id : dirty_)
      fast_touches +=
          rows_[id].pins && !rows_[id].absorbed && !rows_[id].withheld;
    auto headroom = free_blocks > fast_touches ? free_blocks - fast_touches : 0;
    uint64_t free_budget = free_blocks == ~uint64_t{0} ? 0 : headroom / 2;
    // Discard the old admission window before granting queued reads, while
    // retaining reserve for withheld rows until their pool actions are applied.
    if (budget_enabled()) window_ = withheld_;
    for (auto it = grants_.begin(); queued_keys_ && it != grants_.end();) {
      auto current = it++;
      auto& [id, grant] = *current;
      if (grant.pending_keys.empty() || grant.queued_at > deferred_snapshot)
        continue;
      std::vector<int64_t> result;
      std::set<int64_t> unique;
      for (auto& key : grant.pending_keys) {
        auto* row = directory_.find(key);
        if (!row) break;
        result.push_back(*row);
        unique.insert(*row);
      }
      uint64_t cost = 0;
      uint64_t extra_budget = 0;
      for (auto row : unique)
        cost += !rows_[row].absorbed && !rows_[row].pins && !live.count(row);
      for (auto row : unique) extra_budget += !rows_[row].budgeted;
      int rejected =
          frozen_       ? owner_deferred_reject_frozen
          : !gate_open_ ? owner_deferred_reject_gate
          : grant.deferred_leased &&
                  lease_now() >=
                      grant.created + std::chrono::milliseconds(lease_ms_)
              ? owner_deferred_reject_lease_expiry
          : result.size() != grant.pending_keys.size()
              ? owner_deferred_reject_missing_key
          : refs_ + unique.size() > cap_ ? owner_deferred_reject_cap
          : cost > free_budget || !budget_fits(extra_budget, 0)
              ? owner_deferred_reject_budget
              : -1;
      if (rejected >= 0) {
        ++stats_[rejected];
        finish(id);
        continue;
      }
      if (grant.deferred_leased)
        expirations_.emplace(
            grant.created + std::chrono::milliseconds(lease_ms_), id);
      queued_keys_ -= grant.pending_keys.size();
      std::vector<Key>().swap(grant.pending_keys);
      grant.leased = grant.deferred_leased;
      grant.rows = std::move(result);
      grant.unique_rows.assign(unique.begin(), unique.end());
      free_budget -= cost;
      for (auto row : unique) {
        protect(rows_[row]);
        ++rows_[row].pins;
        ++refs_;
        changed(row);
      }
    }
    published_[rows_.size()] = queued_keys_ != 0;
    size_t n = 0;
    uint64_t touches = 0;
    for (auto id : dirty_) {
      auto& row = rows_[id];
      row.dirty = false;
      // ABI actions consumed by Python: 1 = touch, 2 = free, 3 = return
      // withheld.
      int action = 0;
      if (row.withheld) {
        if (!row.pins) {
          row.withheld = false;
          action = 3;
        }
      } else if (row.pins && !row.absorbed) {
        row.absorbed = true;
        ++touches;
        action = 1;
      } else if (!row.pins && row.absorbed) {
        row.absorbed = false;
        action = 2;
      }
      if (action) {
        if (budget_enabled() && (action == 2 || action == 3) &&
            !row.budget_pending) {
          row.budget_pending = true;
          budget_releases_.push_back(id);
        }
        out[n++] = action;
        out[n++] = id;
      }
    }
    dirty_.clear();
    for (auto id : window_rows_) rows_[id].window = false;
    window_rows_.clear();
    new_pins_ = withheld_ = 0;
    // A touch removes at most one free row; releases only add free rows.
    // Reserve <= 1/8 of the resulting admissible headroom, fixed per window.
    if (free_blocks != ~uint64_t{0}) {
      auto headroom = free_blocks > touches ? free_blocks - touches : 0;
      window_ = desired_window(headroom);
    } else if (budget_enabled())
      window_ = 0;
    stats_[owner_reserve] = window_;
    stats_[owner_effective_window] = window_;
    budget_stats();
    std::copy(stats_.begin(), stats_.end(), stats);
    std::fill(stats_.begin(), stats_.begin() + owner_reserve, 0);
    ++generation_;
    return n;
  }
  uint64_t reserve() {
    std::lock_guard<std::mutex> lock(mutex_);
    return window_ - withheld_;
  }
  void complete_absorb(uint64_t free_blocks, uint64_t* stats) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto id : budget_releases_) {
      rows_[id].budget_pending = false;
      unprotect(rows_[id]);
    }
    budget_releases_.clear();
    auto desired = desired_window(free_blocks);
    if (desired > window_) {
      window_ = desired;
      ++generation_;
    }
    stats[owner_reserve] = window_ - withheld_;
    stats[owner_effective_window] = window_;
    budget_stats();
    stats[owner_protected_blocks] = stats_[owner_protected_blocks];
    stats[owner_protected_high_water] = stats_[owner_protected_high_water];
  }
  bool pending() {
    std::lock_guard<std::mutex> lock(mutex_);
    return !dirty_.empty() || queued_keys_;
  }
  uint64_t refs() {
    std::lock_guard<std::mutex> lock(mutex_);
    return refs_;
  }
  bool freeze() {
    std::lock_guard<std::mutex> lock(mutex_);
    frozen_ = true;
    return !refs_ && !queued_keys_ && dirty_.empty() &&
           budget_releases_.empty() &&
           std::none_of(rows_.begin(), rows_.end(),
                        [](const Row& r) { return r.withheld || r.absorbed; });
  }
  void thaw(bool clear) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (clear) {
      for (auto& row : rows_) {
        for (auto& key : row.aliases) event(key, false);
        row.aliases.clear();
      }
      directory_.clear();
      std::fill(published_, published_ + rows_.size(), 0);
    }
    frozen_ = false;
  }
};

class CleanupQueue;
// Allocated by a caller, published once by a GPU callback/caller, then owned by
// the native sender. No caller may access a node after enqueue transfers it.
struct Cleanup {
  CleanupQueue* queue;
  std::string endpoint;
  Request request;
  Cleanup* next = nullptr;
  Cleanup* batch_next = nullptr;
  Clock::time_point retry{};
  int delay_ms = 1;
  Clock::time_point expires = Clock::time_point::max();
};

// Native sender owns retries/deletion; concurrent callbacks publish
// preallocated nodes through the lock-free head_. Join only after callbacks can
// no longer run.
class CleanupQueue {
 public:
  explicit CleanupQueue(int timeout_ms)
      : timeout_ms_(timeout_ms), wake_(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
    if (wake_ < 0) throw std::runtime_error("peer L1 eventfd failed");
    try {
      thread_ = std::thread([this] { run(); });
    } catch (...) {
      close(wake_);
      throw;
    }
  }
  ~CleanupQueue() {
    stop_.store(true);
    wake();
    thread_.join();
    close(wake_);
    for (auto* node : pending_nodes_) delete node;
    auto* node = head_.exchange(nullptr);
    while (node) {
      auto* next = node->next;
      delete node;
      node = next;
    }
  }
  void enqueue(Cleanup* node) noexcept {
    pending_.fetch_add(1);
    auto* head = head_.load(std::memory_order_relaxed);
    do {
      node->next = head;
    } while (!head_.compare_exchange_weak(head, node, std::memory_order_release,
                                          std::memory_order_relaxed));
    wake();
  }
  uint64_t pending() const { return pending_.load(); }
  void check_error() const { error_.check(); }

 private:
  void wake() noexcept {
    uint64_t one = 1;
    while (write(wake_, &one, sizeof(one)) < 0 && errno == EINTR) {
    }
  }
  void run() noexcept {
    try {
      auto& pending = pending_nodes_;
      while (!stop_.load()) {
        auto* node = head_.exchange(nullptr, std::memory_order_acquire);
        while (node) {
          auto* next = node->next;
          try {
            pending.push_back(node);
          } catch (...) {
            while (node) {
              auto* rest = node->next;
              enqueue(node);
              pending_.fetch_sub(1);
              node = rest;
            }
            throw;
          }
          node = next;
        }
        for (size_t i = 0; i < pending.size(); ++i) {
          auto* first = pending[i];
          if (first && first->expires != Clock::time_point::max() &&
              Clock::now() >= first->expires) {
            delete first;
            pending[i] = nullptr;
            pending_.fetch_sub(1);
            continue;
          }
          if (!first || Clock::now() < first->retry) continue;
          std::vector<size_t> positions;
          std::vector<Key> requests;
          for (size_t j = i;
               j < pending.size() && positions.size() < kMaxCleanupKeys; ++j) {
            auto* item = pending[j];
            if (item && item->endpoint == first->endpoint &&
                item->request.epoch == first->request.epoch &&
                Clock::now() >= item->retry) {
              positions.push_back(j);
              requests.push_back(
                  {item->request.reader, item->request.seq, 0, 0});
            }
          }
          Request request = first->request;
          request.op = Cleanups;
          request.n = requests.size();
          auto result = rpc(first->endpoint.c_str(), request, requests.data(),
                            nullptr, timeout_ms_);
          for (auto position : positions) {
            auto* item = pending[position];
            if (result == 0 || result == -2) {
              delete item;
              pending[position] = nullptr;
              pending_.fetch_sub(1);
            } else {
              item->retry =
                  Clock::now() + std::chrono::milliseconds(item->delay_ms);
              item->delay_ms = std::min(1000, item->delay_ms * 2);
            }
          }
        }
        pending.erase(std::remove(pending.begin(), pending.end(), nullptr),
                      pending.end());
        int timeout = -1;
        for (auto* item : pending) {
          auto left = std::max<int64_t>(
              0, std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::min(item->retry, item->expires) - Clock::now())
                     .count());
          timeout = timeout < 0 ? left : std::min<int64_t>(timeout, left);
        }
        pollfd descriptor{wake_, POLLIN, 0};
        if (poll(&descriptor, 1, timeout) > 0) {
          uint64_t count;
          while (read(wake_, &count, sizeof(count)) < 0 && errno == EINTR) {
          }
        }
      }
    } catch (...) {
      error_.capture();
    }
  }
  ThreadError error_;
  std::vector<Cleanup*> pending_nodes_;
  int timeout_ms_, wake_;
  std::atomic<bool> stop_{false};
  std::atomic<Cleanup*> head_{nullptr};
  std::atomic<uint64_t> pending_{0};
  std::thread thread_;
};
static_assert(std::atomic<Cleanup*>::is_always_lock_free);
inline void completed_copy(void* context) noexcept {
  auto* item = static_cast<Cleanup*>(context);
  while (item) {
    auto* next = item->batch_next;
    item->queue->enqueue(item);
    item = next;
  }
}
}  // namespace vllm::peer_l1
