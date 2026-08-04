// SPDX-License-Identifier: Apache-2.0
//
// A subprocess that opens a POSIX shared memory segment and performs
// file I/O directly into/from it.
//
// Protocol (line-based, works over stdin/stdout OR TCP):
//   ATTACH <shm_name> <shm_size> <base_addr>
//     -> OK | ERROR <msg>
//   WRITE <file_path> <data_ptr> <length>
//     -> OK <bytes_written> | ERROR <msg>
//   READ <file_path> <data_ptr> <length>
//     -> OK <bytes_read> | ERROR <msg>
//   QUIT
//     -> OK
//
// Usage:
//   ./shm_file_worker                 # PIPE mode (stdin/stdout)
//   ./shm_file_worker --listen 0.0.0.0:9800  # TCP mode

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

static void* g_shm_ptr = nullptr;
static size_t g_shm_size = 0;
static uintptr_t g_base_addr = 0;

static bool handle_attach(const std::string& shm_name, size_t size,
                          uintptr_t base_addr, std::ostream& out) {
  std::cerr << "[LOG] ATTACH shm_name=" << shm_name << " size=" << size
            << " base_addr=" << base_addr << std::endl;
  int fd = shm_open(shm_name.c_str(), O_RDWR, 0600);
  if (fd < 0) {
    std::cerr << "[LOG] ATTACH failed: shm_open: " << strerror(errno)
              << std::endl;
    out << "ERROR shm_open failed: " << strerror(errno) << std::endl;
    return false;
  }
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    std::cerr << "[LOG] ATTACH failed: mmap: " << strerror(errno) << std::endl;
    out << "ERROR mmap failed: " << strerror(errno) << std::endl;
    return false;
  }
  g_shm_ptr = ptr;
  g_shm_size = size;
  g_base_addr = base_addr;
  std::cerr << "[LOG] ATTACH succeeded" << std::endl;
  out << "OK" << std::endl;
  return true;
}

static void handle_write(const std::string& file_path, uintptr_t data_ptr,
                         size_t length, std::ostream& out) {
  std::cerr << "[LOG] WRITE path=" << file_path << " data_ptr=" << data_ptr
            << " length=" << length << std::endl;
  if (!g_shm_ptr) {
    std::cerr << "[LOG] WRITE failed: not attached" << std::endl;
    out << "ERROR not attached" << std::endl;
    return;
  }
  if (data_ptr < g_base_addr) {
    std::cerr << "[LOG] WRITE failed: data_ptr is before shm base address"
              << std::endl;
    out << "ERROR data_ptr is before shm base address" << std::endl;
    return;
  }
  size_t offset = data_ptr - g_base_addr;
  if (offset >= g_shm_size || length > g_shm_size - offset) {
    std::cerr << "[LOG] WRITE failed: offset+length exceeds shm size"
              << std::endl;
    out << "ERROR offset+length exceeds shm size" << std::endl;
    return;
  }
  int fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    std::cerr << "[LOG] WRITE failed: open: " << strerror(errno) << std::endl;
    out << "ERROR open failed: " << strerror(errno) << std::endl;
    return;
  }
  const char* src = static_cast<const char*>(g_shm_ptr) + offset;
  size_t written = 0;
  while (written < length) {
    ssize_t n = write(fd, src + written, length - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      close(fd);
      std::cerr << "[LOG] WRITE failed: write: " << strerror(errno)
                << std::endl;
      out << "ERROR write failed: " << strerror(errno) << std::endl;
      return;
    }
    written += static_cast<size_t>(n);
  }
  close(fd);
  std::cerr << "[LOG] WRITE succeeded: " << written << " bytes" << std::endl;
  out << "OK " << written << std::endl;
}

static void handle_read(const std::string& file_path, uintptr_t data_ptr,
                        size_t length, std::ostream& out) {
  std::cerr << "[LOG] READ path=" << file_path << " data_ptr=" << data_ptr
            << " length=" << length << std::endl;
  if (!g_shm_ptr) {
    std::cerr << "[LOG] READ failed: not attached" << std::endl;
    out << "ERROR not attached" << std::endl;
    return;
  }
  if (data_ptr < g_base_addr) {
    std::cerr << "[LOG] READ failed: data_ptr is before shm base address"
              << std::endl;
    out << "ERROR data_ptr is before shm base address" << std::endl;
    return;
  }
  size_t offset = data_ptr - g_base_addr;
  if (offset >= g_shm_size || length > g_shm_size - offset) {
    std::cerr << "[LOG] READ failed: offset+length exceeds shm size"
              << std::endl;
    out << "ERROR offset+length exceeds shm size" << std::endl;
    return;
  }
  int fd = open(file_path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cerr << "[LOG] READ failed: open: " << strerror(errno) << std::endl;
    out << "ERROR open failed: " << strerror(errno) << std::endl;
    return;
  }
  // Get file size to clamp read
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    std::cerr << "[LOG] READ failed: fstat: " << strerror(errno) << std::endl;
    out << "ERROR fstat failed: " << strerror(errno) << std::endl;
    return;
  }
  size_t to_read = (static_cast<size_t>(st.st_size) < length)
                       ? static_cast<size_t>(st.st_size)
                       : length;

  char* dst = static_cast<char*>(g_shm_ptr) + offset;
  size_t total_read = 0;
  while (total_read < to_read) {
    ssize_t n = read(fd, dst + total_read, to_read - total_read);
    if (n < 0) {
      if (errno == EINTR) continue;
      close(fd);
      std::cerr << "[LOG] READ failed: read: " << strerror(errno) << std::endl;
      out << "ERROR read failed: " << strerror(errno) << std::endl;
      return;
    }
    if (n == 0) break;
    total_read += static_cast<size_t>(n);
  }
  close(fd);
  std::cerr << "[LOG] READ succeeded: " << total_read << " bytes" << std::endl;
  out << "OK " << total_read << std::endl;
}

// Process a single command line, write response to ostream.
static bool process_line(const std::string& line, std::ostream& out) {
  std::istringstream iss(line);
  std::string cmd;
  iss >> cmd;

  if (cmd == "ATTACH") {
    std::string shm_name;
    size_t size;
    uintptr_t base_addr;
    iss >> shm_name >> size >> base_addr;
    handle_attach(shm_name, size, base_addr, out);
  } else if (cmd == "WRITE") {
    std::string path;
    uintptr_t data_ptr;
    size_t length;
    iss >> path >> data_ptr >> length;
    handle_write(path, data_ptr, length, out);
  } else if (cmd == "READ") {
    std::string path;
    uintptr_t data_ptr;
    size_t length;
    iss >> path >> data_ptr >> length;
    handle_read(path, data_ptr, length, out);
  } else if (cmd == "QUIT") {
    std::cerr << "[LOG] QUIT received" << std::endl;
    out << "OK" << std::endl;
    return false;  // signal to exit
  } else {
    std::cerr << "[LOG] Unknown command: " << cmd << std::endl;
    out << "ERROR unknown command: " << cmd << std::endl;
  }
  return true;  // continue
}

// Read a line from a socket fd (up to newline).
static std::string read_line_from_fd(int fd) {
  std::string line;
  char ch;
  while (true) {
    ssize_t n = ::read(fd, &ch, 1);
    if (n <= 0) return "";
    if (ch == '\n') break;
    line += ch;
  }
  return line;
}

// Write a string to a socket fd.
static void write_to_fd(int fd, const std::string& data) {
  const char* ptr = data.c_str();
  size_t remaining = data.size();
  while (remaining > 0) {
    ssize_t n = ::write(fd, ptr, remaining);
    if (n <= 0) return;
    ptr += n;
    remaining -= static_cast<size_t>(n);
  }
}

// Run PIPE mode: read from stdin, write to stdout.
static int run_pipe_mode() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.tie(nullptr);

  std::string line;
  while (std::getline(std::cin, line)) {
    if (!process_line(line, std::cout)) break;
  }
  return 0;
}

// Parse "host:port" string.
static bool parse_addr(const std::string& addr, std::string& host, int& port) {
  auto pos = addr.rfind(':');
  if (pos == std::string::npos) return false;
  host = addr.substr(0, pos);
  port = std::stoi(addr.substr(pos + 1));
  return port > 0 && port < 65536;
}

// Run TCP mode: listen on host:port, accept one client.
static int run_tcp_mode(const std::string& addr) {
  std::string host;
  int port;
  if (!parse_addr(addr, host, port)) {
    std::cerr << "Invalid address: " << addr << std::endl;
    return 1;
  }

  int srv = socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) {
    std::cerr << "socket failed: " << strerror(errno) << std::endl;
    return 1;
  }
  int opt = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_port = htons(static_cast<uint16_t>(port));
  if (host == "0.0.0.0" || host.empty()) {
    // Default to localhost for security
    inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);
  } else {
    inet_pton(AF_INET, host.c_str(), &sa.sin_addr);
  }

  if (bind(srv, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
    std::cerr << "bind failed: " << strerror(errno) << std::endl;
    ::close(srv);
    return 1;
  }
  if (listen(srv, 1) < 0) {
    std::cerr << "listen failed: " << strerror(errno) << std::endl;
    ::close(srv);
    return 1;
  }

  std::cerr << "Listening on " << host << ":" << port << std::endl;

  // Accept one client connection
  struct sockaddr_in client_addr{};
  socklen_t client_len = sizeof(client_addr);
  int client_fd = accept(srv, (struct sockaddr*)&client_addr, &client_len);
  if (client_fd < 0) {
    std::cerr << "accept failed: " << strerror(errno) << std::endl;
    ::close(srv);
    return 1;
  }
  std::cerr << "Client connected" << std::endl;

  // Process commands from the client
  while (true) {
    std::string line = read_line_from_fd(client_fd);
    if (line.empty()) break;
    std::ostringstream oss;
    bool cont = process_line(line, oss);
    write_to_fd(client_fd, oss.str());
    if (!cont) break;
  }

  ::close(client_fd);
  ::close(srv);

  if (g_shm_ptr) {
    munmap(g_shm_ptr, g_shm_size);
  }
  return 0;
}

int main(int argc, char* argv[]) {
  // Parse arguments
  std::string listen_addr;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--listen" && i + 1 < argc) {
      listen_addr = argv[++i];
    }
  }

  int ret;
  if (!listen_addr.empty()) {
    ret = run_tcp_mode(listen_addr);
  } else {
    ret = run_pipe_mode();
  }

  if (g_shm_ptr) {
    munmap(g_shm_ptr, g_shm_size);
  }
  return ret;
}
