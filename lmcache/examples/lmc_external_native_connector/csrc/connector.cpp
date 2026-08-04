// SPDX-License-Identifier: Apache-2.0

#include "connector.h"
#include <fcntl.h>
#include <cerrno>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace example_connector {

// ---------------------------------------------------------------
// Shared I/O helpers
// ---------------------------------------------------------------

static void write_all(int fd, const void* data, size_t len) {
  size_t written = 0;
  const char* ptr = static_cast<const char*>(data);
  while (written < len) {
    ssize_t n = ::write(fd, ptr + written, len - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("write failed: " + std::string(strerror(errno)));
    }
    if (n == 0) throw std::runtime_error("write returned 0");
    written += static_cast<size_t>(n);
  }
}

static size_t read_all(int fd, void* buf, size_t len) {
  size_t total = 0;
  char* ptr = static_cast<char*>(buf);
  while (total < len) {
    ssize_t n = ::read(fd, ptr + total, len - total);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    }
    if (n == 0) break;
    total += static_cast<size_t>(n);
  }
  return total;
}

// ---------------------------------------------------------------
// ExampleFSConnector
// ---------------------------------------------------------------

// Key encoding constants — must match fs_l2_adapter.py
static constexpr char KEY_SEP = '@';
static constexpr const char* PATH_SLASH_REPLACEMENT = "-SEP-";
static constexpr const char* FILE_EXT = ".data";

std::string ExampleFSConnector::replace_all(const std::string& str,
                                            const std::string& from,
                                            const std::string& to) {
  std::string result = str;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.size(), to);
    pos += to.size();
  }
  return result;
}

std::string ExampleFSConnector::safe_filename(const std::string& key) {
  // Input key format (from _object_key_to_string):
  //   Unsalted: <model_name>@<kv_rank_hex>@<chunk_hash_hex>
  //   Salted  : <model_name>@<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>
  //
  // Output filename:
  //   Unsalted: <model_name_safe>@0x<kv_rank_hex>@<chunk_hash_hex>.data
  //   Salted  :
  //   <model_name_safe>@0x<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>.data
  //
  // Both model_name and cache_salt are forbidden from containing '@'
  // (invariant enforced on the Python side), so splitting on '@' is
  // unambiguous.

  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == KEY_SEP) {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 3 && parts.size() != 4) {
    throw std::runtime_error(
        "ExampleFSConnector: malformed key "
        "(expected 3 or 4 '@'-separated fields): " +
        key);
  }

  const std::string& model = parts[0];
  const std::string& kv_rank_hex = parts[1];
  const std::string& chunk_hash = parts[2];
  const std::string cache_salt = parts.size() == 4 ? parts[3] : std::string();

  // Replace '/' with '-SEP-' for filesystem safety
  std::string safe_model = replace_all(model, "/", PATH_SLASH_REPLACEMENT);

  std::string result;
  result.reserve(safe_model.size() + kv_rank_hex.size() + chunk_hash.size() +
                 cache_salt.size() + 32);
  result += safe_model;
  result += KEY_SEP;
  result += "0x";
  result += kv_rank_hex;
  result += KEY_SEP;
  result += chunk_hash;
  if (!cache_salt.empty()) {
    result += KEY_SEP;
    result += cache_salt;
  }
  result += FILE_EXT;
  return result;
}

ExampleFSConnector::ExampleFSConnector(std::string base_path, int num_workers)
    : ConnectorBase(num_workers), base_path_(std::move(base_path)) {
  std::filesystem::create_directories(base_path_);
  start_workers();
}

ExampleFSConnector::~ExampleFSConnector() { close(); }

WorkerFSConn ExampleFSConnector::create_connection() {
  WorkerFSConn conn;
  conn.base_path = base_path_;
  return conn;
}

void ExampleFSConnector::do_single_get(WorkerFSConn& conn,
                                       const std::string& key, void* buf,
                                       size_t len, size_t /*chunk_size*/) {
  auto path = conn.base_path / safe_filename(key);
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("open for read failed: " + path.string() + ": " +
                             strerror(errno));
  }
  size_t n = read_all(fd, buf, len);
  ::close(fd);
  if (n != len) {
    throw std::runtime_error("incomplete read for " + path.string());
  }
}

void ExampleFSConnector::do_single_set(WorkerFSConn& conn,
                                       const std::string& key, const void* buf,
                                       size_t len, size_t /*chunk_size*/) {
  auto file_path = conn.base_path / safe_filename(key);
  if (std::filesystem::exists(file_path)) return;

  auto tmp_path = file_path;
  tmp_path.replace_extension(".tmp");

  int fd = ::open(tmp_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) {
    throw std::runtime_error("open for write failed: " + tmp_path.string() +
                             ": " + strerror(errno));
  }
  try {
    write_all(fd, buf, len);
  } catch (...) {
    ::close(fd);
    std::filesystem::remove(tmp_path);
    throw;
  }
  ::close(fd);

  std::error_code ec;
  std::filesystem::rename(tmp_path, file_path, ec);
  if (ec) {
    std::error_code remove_ec;
    std::filesystem::remove(tmp_path, remove_ec);
    throw std::runtime_error("rename failed: " + ec.message());
  }
}

bool ExampleFSConnector::do_single_exists(WorkerFSConn& conn,
                                          const std::string& key) {
  auto path = conn.base_path / safe_filename(key);
  return std::filesystem::exists(path);
}

bool ExampleFSConnector::do_single_delete(WorkerFSConn& conn,
                                          const std::string& key) {
  auto path = conn.base_path / safe_filename(key);
  std::error_code ec;
  return std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------
// ExampleMemoryConnector
// ---------------------------------------------------------------

ExampleMemoryConnector::ExampleMemoryConnector(int num_workers)
    : ConnectorBase(num_workers),
      store_(std::make_shared<SharedMemoryStore>()) {
  start_workers();
}

ExampleMemoryConnector::~ExampleMemoryConnector() { close(); }

WorkerMemConn ExampleMemoryConnector::create_connection() {
  WorkerMemConn conn;
  conn.store = store_;
  return conn;
}

void ExampleMemoryConnector::do_single_get(WorkerMemConn& conn,
                                           const std::string& key, void* buf,
                                           size_t len, size_t /*chunk_size*/) {
  std::lock_guard<std::mutex> lk(conn.store->mu);
  auto it = conn.store->data.find(key);
  if (it == conn.store->data.end()) {
    throw std::runtime_error("key not found: " + key);
  }
  if (it->second.size() != len) {
    throw std::runtime_error("size mismatch for key: " + key);
  }
  std::memcpy(buf, it->second.data(), len);
}

void ExampleMemoryConnector::do_single_set(WorkerMemConn& conn,
                                           const std::string& key,
                                           const void* buf, size_t len,
                                           size_t /*chunk_size*/) {
  std::lock_guard<std::mutex> lk(conn.store->mu);
  auto& vec = conn.store->data[key];
  vec.resize(len);
  std::memcpy(vec.data(), buf, len);
}

bool ExampleMemoryConnector::do_single_exists(WorkerMemConn& conn,
                                              const std::string& key) {
  std::lock_guard<std::mutex> lk(conn.store->mu);
  return conn.store->data.count(key) > 0;
}

bool ExampleMemoryConnector::do_single_delete(WorkerMemConn& conn,
                                              const std::string& key) {
  std::lock_guard<std::mutex> lk(conn.store->mu);
  return conn.store->data.erase(key) > 0;
}

}  // namespace example_connector
