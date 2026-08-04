// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../connector_base.h"
#include <sys/socket.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <cstring>
#include <vector>

namespace lmcache {
namespace connector {

// a TCP session (one per thread) implementing RESP2
/*
key optimizations include:
1. preset batch_chunk_num_bytes (allows not parsing for \r\n byte-by-byte)
2. scatter/gather sending of data (with pre-allocated buffers)
3. zero copy (no bounce buffers)
*/
struct WorkerConn {
  int fd = -1;
  std::string host;
  int port;

  // authentication
  std::string username;  // optional; empty => auth password
  std::string password;  // required for auth
  bool authed = false;

  // pre-computed headers
  std::string get_prefix;
  std::string set_prefix;
  std::string exists_prefix;
  std::string del_prefix;

  // reusable buffers for building headers (avoids repeated dynamic allocations)
  std::string key_header_buf;
  std::string size_header_buf;

  // pre-computed constants (for comparisons)
  static constexpr std::string_view crlf = "\r\n";
  static constexpr size_t crlf_len = crlf.size();

  static constexpr std::string_view ok_response = "+OK\r\n";
  static constexpr size_t ok_response_len = ok_response.size();

  static constexpr std::string_view exists_one = ":1\r\n";
  static constexpr std::string_view exists_zero = ":0\r\n";
  static constexpr size_t exists_response_len = exists_one.size();

  WorkerConn()
      : get_prefix("*2\r\n$3\r\nGET\r\n"),
        set_prefix("*3\r\n$3\r\nSET\r\n"),
        exists_prefix("*2\r\n$6\r\nEXISTS\r\n"),
        del_prefix("*2\r\n$3\r\nDEL\r\n") {
    // pre-allocate key_header_buf to handle typical keys without reallocation
    // typical key format: model_name@world_size@worker_id@chunk_hash_hex@dtype
    // - model_name: 25-50 chars (e.g., "meta-llama/Llama-3-70b-instruct")
    // - world_size: 1-2 chars
    // - worker_id: 1-2 chars
    // - chunk_hash (SHA256): 64 chars hex
    // - dtype: 7-8 chars (e.g., "bfloat16")
    // - separators: 4 chars
    // total typical key: ~100-140 chars
    // resp header overhead: $<len>\r\n<key>\r\n = ~8 bytes
    // reserve 512 bytes to handle typical keys plus margin
    key_header_buf.reserve(512);

    // pre-allocate size_header_buf for chunk size headers
    // typical format: $<batch_chunk_num_bytes>\r\n
    // typical batch_chunk_num_bytes: 1MB-4MB = 7-8 digit number
    // reserve 32 bytes to handle up to 20+ digit numbers with margin
    size_header_buf.reserve(32);
  }

  ~WorkerConn() {
    if (fd >= 0) ::close(fd);
  }

  void connect(const std::string& host, int port);
  void authenticate_if_needed();
  void send_all(const void* data, size_t len);
  void send_multipart(const std::vector<std::pair<const void*, size_t>>& parts);
  void recv_exactly(void* buf, size_t len);
  std::string recv_line();
  const std::string& make_key_header(const std::string& key);
  const std::string& make_size_header(size_t batch_chunk_num_bytes);
};

class RedisConnector : public ConnectorBase<WorkerConn> {
 public:
  RedisConnector(std::string host, int port, int num_workers,
                 std::string username = "", std::string password = "");
  ~RedisConnector() override;

 protected:
  WorkerConn create_connection() override;
  void do_single_get(WorkerConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerConn& conn, const std::string& key, const void* buf,
                     size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerConn& conn, const std::string& key) override;
  bool do_single_delete(WorkerConn& conn, const std::string& key) override;
  void shutdown_connections() override;

 private:
  std::string host_;
  int port_;
  std::string username_;
  std::string password_;
  std::mutex worker_fds_mu_;
  std::vector<int> worker_fds_;
};

}  // namespace connector
}  // namespace lmcache
