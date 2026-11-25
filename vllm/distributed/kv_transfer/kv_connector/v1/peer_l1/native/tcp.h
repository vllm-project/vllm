// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "wire.h"
#include <atomic>
#include <functional>
#include <map>
#include <thread>
#include <system_error>
#include <vector>

namespace vllm::peer_l1 {
// One native I/O thread owns clients_; callbacks lock the owner's shared state.
// Destruction joins that thread before freeing client buffers.
class TcpServer {
 public:
  using Handler = std::function<std::vector<int64_t>(const Request&,
                                                     const std::vector<Key>&)>;
  TcpServer(const char* endpoint, uint64_t epoch, Handler handler,
            const Key& credential, uint64_t pin_cap,
            std::function<void()> maintenance = {})
      : epoch_(epoch),
        credential_(credential),
        key_cap_(std::min(pin_cap, kMaxAcquireKeys)),
        handler_(std::move(handler)),
        maintenance_(std::move(maintenance)) {
    auto addr = address(endpoint);
    family_ = addr.storage.ss_family;
    fd_ = socket(family_, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd_ < 0) throw std::runtime_error("peer L1 TCP socket failed");
    try {
      int one = 1;
      if ((family_ != AF_UNIX &&
           setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one))) ||
          bind(fd_, addr.data(), addr.length) || listen(fd_, 128))
        throw std::system_error(errno, std::generic_category(),
                                "peer L1 TCP bind failed");
      thread_ = std::thread([this] { run(); });
    } catch (...) {
      close(fd_);
      throw;
    }
  }
  ~TcpServer() {
    stop_.store(true);
    thread_.join();
    for (auto& [fd, state] : clients_) close(fd);
    close(fd_);
  }

  void check_error() const { error_.check(); }

 private:
  // Confined to the server I/O thread; no per-client mutex is needed.
  struct Client {
    Request request{};
    size_t header = 0, body = 0, sent = 0;
    std::vector<Key> keys;
    std::vector<char> response;
    Clock::time_point deadline = Clock::now() + std::chrono::milliseconds(100);
  };
  // Bound per-connection work so a large frame cannot monopolize a poll turn.
  static constexpr size_t kBudget = 256 * 1024;
  static constexpr size_t kMaxClients = 1024;
  bool read_part(int fd, void* data, size_t size, size_t& offset,
                 size_t& budget) {
    while (offset < size && budget) {
      auto n = recv(fd, static_cast<char*>(data) + offset,
                    std::min(size - offset, budget), MSG_DONTWAIT);
      if (n < 0 && errno == EINTR) continue;
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return true;
      if (n <= 0) return false;
      offset += n;
      budget -= n;
    }
    return true;
  }
  bool receive(int fd, Client& c) {
    size_t budget = kBudget;
    if (c.deadline == Clock::time_point::max())
      c.deadline = Clock::now() + std::chrono::milliseconds(100);
    if (!read_part(fd, &c.request, sizeof(Request), c.header, budget))
      return false;
    if (c.header < sizeof(Request)) return true;
    auto& r = c.request;
    if (r.magic != Request{}.magic || r.credential != credential_) return false;
    uint64_t limit = 0;
    switch (r.op) {
      case AcquireExact:
      case AcquireLeased:
      case AcquireDeferred:
      case AcquireDeferredLeased:
        limit = key_cap_;
        break;
      case Cleanups:
        limit = kMaxCleanupKeys;
        break;
      case Hello:
      case Members:
        limit = kMaxDescriptionWords;
        break;
      case Release:
      case Cancel:
        break;
      default:
        return false;
    }
    if (r.n > limit) return false;
    size_t body_keys = r.op == Hello || r.op == Members ? 0 : r.n;
    // Authenticate and bound the header before allocating a caller-sized body.
    c.keys.resize(body_keys);
    if (!read_part(fd, c.keys.data(), body_keys * sizeof(Key), c.body, budget))
      return false;
    if (c.body < body_keys * sizeof(Key)) return true;
    auto rows = (r.epoch == epoch_ || (r.op == Members && !r.epoch))
                    ? handler_(r, c.keys)
                    : std::vector<int64_t>{};
    Reply reply{epoch_, rows.size()};
    c.response.resize(sizeof(reply) + rows.size() * sizeof(int64_t));
    memcpy(c.response.data(), &reply, sizeof(reply));
    if (!rows.empty())
      memcpy(c.response.data() + sizeof(reply), rows.data(),
             rows.size() * sizeof(int64_t));
    return true;
  }
  bool send_reply(int fd, Client& c) {
    size_t budget = kBudget;
    while (c.sent < c.response.size() && budget) {
      auto n = send(fd, c.response.data() + c.sent,
                    std::min(c.response.size() - c.sent, budget),
                    MSG_NOSIGNAL | MSG_DONTWAIT);
      if (n < 0 && errno == EINTR) continue;
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return true;
      if (n <= 0) return false;
      c.sent += n;
      budget -= n;
    }
    if (c.sent == c.response.size()) {
      c.header = c.body = c.sent = 0;
      c.response.clear();
      c.deadline = Clock::time_point::max();
    }
    return true;
  }
  void accept_clients() {
    for (size_t i = 0; i < 16; ++i) {
      int fd = accept4(fd_, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
      if (fd < 0) break;
      if (clients_.size() >= kMaxClients || !no_delay(fd, family_)) {
        close(fd);
        continue;
      }
      try {
        clients_.try_emplace(fd);
      } catch (...) {
        close(fd);
        throw;
      }
    }
  }
  void run() noexcept {
    try {
      while (!stop_.load()) {
        if (maintenance_) maintenance_();
        std::vector<pollfd> ready{{fd_, POLLIN, 0}};
        for (auto& [fd, c] : clients_)
          ready.push_back(
              {fd, static_cast<short>(c.response.empty() ? POLLIN : POLLOUT),
               0});
        if (poll(ready.data(), ready.size(), 10) < 0) continue;
        for (size_t i = 1; i < ready.size(); ++i) {
          auto event = ready[i];
          auto& c = clients_.at(event.fd);
          bool keep = Clock::now() < c.deadline &&
                      !(event.revents & (POLLERR | POLLHUP | POLLNVAL));
          try {
            if (keep && (event.revents & POLLIN)) keep = receive(event.fd, c);
            if (keep && !c.response.empty()) keep = send_reply(event.fd, c);
          } catch (...) {
            keep = false;
          }
          if (!keep) {
            close(event.fd);
            clients_.erase(event.fd);
          }
        }
        if (ready[0].revents & POLLIN) accept_clients();
      }
    } catch (...) {
      error_.capture();
    }
  }
  int fd_, family_;
  uint64_t epoch_;
  Key credential_;
  uint64_t key_cap_;
  Handler handler_;
  std::function<void()> maintenance_;
  ThreadError error_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::map<int, Client> clients_;
};
}  // namespace vllm::peer_l1
