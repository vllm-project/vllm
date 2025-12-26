#pragma once
#include <string>
#include <cstdint>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>
#include <stdexcept>

struct RpcClient {
  int fd{-1};
  explicit RpcClient(const std::string& path) {
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    sockaddr_un addr{}; addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) throw std::runtime_error("UDS path too long");
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
    if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { ::close(fd); throw std::runtime_error("connect failed"); }
  }
  ~RpcClient(){ if (fd>=0) ::close(fd); }
  void send_all(const void* p, size_t n) {
    const char* c = (const char*)p; size_t s=0;
    while (s<n) { ssize_t w = ::write(fd, c+s, n-s); if (w<=0) throw std::runtime_error("write"); s+=w; }
  }
  void recv_all(void* p, size_t n) {
    char* c = (char*)p; size_t s=0;
    while (s<n) { ssize_t r = ::read(fd, c+s, n-s); if (r<=0) throw std::runtime_error("read"); s+=r; }
  }
};

struct RpcServer {
  int fd{-1};
  explicit RpcServer(const std::string& path) {
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    ::unlink(path.c_str());
    sockaddr_un addr{}; addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
    if (bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) throw std::runtime_error("bind failed");
    if (listen(fd, 64) < 0) throw std::runtime_error("listen failed");
  }
  int accept_one() {
    int c = accept(fd, nullptr, nullptr);
    if (c < 0) throw std::runtime_error("accept failed");
    return c;
  }
};
