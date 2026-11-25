// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <charconv>
#include <cstring>
#include <stdexcept>
#include <string>

namespace vllm::peer_l1 {
// Caller-owned endpoint value; no shared mutable state or internal locking.
struct SocketAddress {
  sockaddr_storage storage{};
  socklen_t length;

  sockaddr* data() { return reinterpret_cast<sockaddr*>(&storage); }
};

inline SocketAddress address(const char* endpoint) {
  std::string value(endpoint);
  SocketAddress result;
  if (value.rfind("tcp://", 0) != 0) {
    if (value.rfind("unix:", 0) == 0) value.erase(0, 5);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (value.empty() || value.size() + 1 >= sizeof(addr.sun_path))
      throw std::invalid_argument("invalid peer L1 Unix endpoint");
    memcpy(addr.sun_path + 1, value.data(), value.size());
    result.length = offsetof(sockaddr_un, sun_path) + 1 + value.size();
    memcpy(&result.storage, &addr, result.length);
    return result;
  }
  auto separator = value.rfind(':');
  if (separator == std::string::npos || separator < 6)
    throw std::invalid_argument("peer L1 TCP endpoint needs an IP and port");
  auto host = value.substr(6, separator - 6);
  auto port_text = value.substr(separator + 1);
  unsigned port = 0;
  auto parsed = std::from_chars(port_text.data(),
                                port_text.data() + port_text.size(), port);
  if (parsed.ec != std::errc{} ||
      parsed.ptr != port_text.data() + port_text.size() || port > 65535)
    throw std::invalid_argument("invalid peer L1 TCP port");
  if (host.size() > 2 && host.front() == '[' && host.back() == ']') {
    host = host.substr(1, host.size() - 2);
    sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    addr.sin6_port = htons(port);
    if (inet_pton(AF_INET6, host.c_str(), &addr.sin6_addr) != 1)
      throw std::invalid_argument("invalid peer L1 IPv6 address");
    result.length = sizeof(addr);
    memcpy(&result.storage, &addr, sizeof(addr));
  } else {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
      throw std::invalid_argument("peer L1 TCP endpoints require a numeric IP");
    result.length = sizeof(addr);
    memcpy(&result.storage, &addr, sizeof(addr));
  }
  return result;
}

inline bool no_delay(int fd, int family) {
  int one = 1;
  return family == AF_UNIX ||
         setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0;
}

inline std::string subscription_endpoint(const std::string& endpoint) {
  if (endpoint.rfind("tcp://", 0) != 0) return endpoint + ".sub";
  auto addr = address(endpoint.c_str());
  unsigned port =
      addr.storage.ss_family == AF_INET6
          ? ntohs(reinterpret_cast<sockaddr_in6*>(&addr.storage)->sin6_port)
          : ntohs(reinterpret_cast<sockaddr_in*>(&addr.storage)->sin_port);
  if (!port || port == 65535)
    throw std::invalid_argument("peer L1 TCP control needs a port in 1..65534");
  return endpoint.substr(0, endpoint.rfind(':') + 1) + std::to_string(port + 1);
}

}  // namespace vllm::peer_l1
