// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../connector_base.h"

#include <aerospike/aerospike.h>
#include <aerospike/as_policy.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace lmcache {
namespace connector {

struct WorkerAerospikeConn {
  aerospike* client = nullptr;
  std::string ns;
  std::string set_name;
  as_policy_read read_policy;
  as_policy_write write_policy;
  as_policy_remove remove_policy;
};

struct ShardPlan {
  uint32_t nseg = 1;
  size_t seg_b = 0;
};

// Native Aerospike storage backend.
//
// Records use a meta + segment layout: every cache key maps to a meta record
// (``<key>|m``) carrying the shard plan, and payloads larger than the
// discovered Aerospike record-size cap are split across segment records
// (``<key>|s|<i>``). Payloads that fit a single record are stored inline in
// the meta record. The connector key is used verbatim as the Aerospike user
// key base (the framework's ObjectKey-to-string format).
class AerospikeNativeConnector : public ConnectorBase<WorkerAerospikeConn> {
 public:
  AerospikeNativeConnector(
      std::string hosts, std::string ns, std::string set_name, int num_workers,
      uint32_t read_timeout_ms = 1000, uint32_t write_timeout_ms = 2000,
      uint32_t default_ttl_seconds = 86400, size_t target_segment_bytes = 0,
      size_t max_record_bytes = 0, std::string username = "",
      std::string password = "");
  ~AerospikeNativeConnector() override;

  void close() override;

 protected:
  WorkerAerospikeConn create_connection() override;
  void do_single_get(WorkerAerospikeConn& conn, const std::string& key,
                     void* buf, size_t len, size_t chunk_size) override;
  void do_single_set(WorkerAerospikeConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerAerospikeConn& conn,
                        const std::string& key) override;
  bool do_single_delete(WorkerAerospikeConn& conn,
                        const std::string& key) override;
  void shutdown_connections() override;
  void on_workers_stopped() override;

 private:
  static std::vector<std::pair<std::string, int>> parse_hosts(
      const std::string& hosts);
  static std::string meta_user_key(const std::string& cache_key);
  static std::string segment_user_key(const std::string& cache_key,
                                      uint32_t index);
  static void throw_status(const char* op, as_status status,
                           const as_error& err);

  ShardPlan plan(size_t payload_bytes) const;
  size_t discover_record_cap();
  void configure_policies();
  void put_payload_record(WorkerAerospikeConn& conn,
                          const std::string& user_key, const void* buf,
                          size_t len);
  void put_meta_record(WorkerAerospikeConn& conn, const std::string& user_key,
                       const ShardPlan& plan, size_t total_bytes,
                       const void* inline_buf);
  bool read_payload_record(WorkerAerospikeConn& conn,
                           const std::string& user_key, void* buf, size_t len);

  std::string hosts_;
  std::string ns_;
  std::string set_name_;
  uint32_t read_timeout_ms_;
  uint32_t write_timeout_ms_;
  uint32_t default_ttl_seconds_;
  size_t target_segment_bytes_;
  size_t max_record_bytes_;
  size_t single_record_threshold_bytes_;

  aerospike as_;
  std::mutex close_mu_;
  bool connected_ = false;
  bool closed_native_ = false;
};

}  // namespace connector
}  // namespace lmcache
