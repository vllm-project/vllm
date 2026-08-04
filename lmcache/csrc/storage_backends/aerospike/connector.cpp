// SPDX-License-Identifier: Apache-2.0

#include "connector.h"

#include <aerospike/aerospike_info.h>
#include <aerospike/aerospike_key.h>
#include <aerospike/as_config.h>
#include <aerospike/as_record.h>
#include <aerospike/as_status.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sstream>
#include <stdexcept>

namespace lmcache {
namespace connector {
namespace {

constexpr size_t kSafetyMarginBytes = 64 * 1024;
constexpr size_t kDefaultRecordCapBytes = 1024 * 1024;
constexpr const char* kBinPayload = "b";
constexpr const char* kBinState = "state";
constexpr const char* kBinNseg = "nseg";
constexpr const char* kBinSegBytes = "seg_b";
constexpr const char* kBinTotalBytes = "tot_b";
constexpr const char* kBinVersion = "ver";
constexpr const char* kBinCreatedAt = "created_at";
constexpr const char* kBinPin = "pin";
constexpr const char* kReady = "ready";

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  for (size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == sep) {
      out.emplace_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  return out;
}

std::string status_message(as_status status, const as_error& err) {
  std::ostringstream oss;
  oss << "Aerospike status " << static_cast<int>(status);
  if (err.message[0] != '\0') {
    oss << ": " << err.message;
  }
  return oss.str();
}

// `as_record_get_int64` requires a default to return when a bin is absent or
// not an int64, so a fallback is unavoidable at the API level. This helper also
// treats non-positive values as "use the fallback" because the count/size bins
// written by put_meta_record() are always >= 1 on a healthy record. Integrity-
// critical fields (e.g. total size) are validated explicitly by the caller and
// must not rely on this fallback to mask a missing bin.
int64_t positive_int_bin(as_record* rec, const char* bin, int64_t fallback) {
  int64_t value = as_record_get_int64(rec, bin, fallback);
  return value > 0 ? value : fallback;
}

}  // namespace

AerospikeNativeConnector::AerospikeNativeConnector(
    std::string hosts, std::string ns, std::string set_name, int num_workers,
    uint32_t read_timeout_ms, uint32_t write_timeout_ms,
    uint32_t default_ttl_seconds, size_t target_segment_bytes,
    size_t max_record_bytes, std::string username, std::string password)
    : ConnectorBase(num_workers),
      hosts_(std::move(hosts)),
      ns_(std::move(ns)),
      set_name_(std::move(set_name)),
      read_timeout_ms_(read_timeout_ms),
      write_timeout_ms_(write_timeout_ms),
      default_ttl_seconds_(default_ttl_seconds) {
  as_config config;
  as_config_init(&config);
  config.thread_pool_size = static_cast<uint32_t>(std::max(num_workers, 1));

  for (const auto& [host, port] : parse_hosts(hosts_)) {
    as_config_add_host(&config, host.c_str(), port);
  }
  if (!username.empty()) {
    if (!as_config_set_user(&config, username.c_str(), password.c_str())) {
      throw std::runtime_error("invalid Aerospike username/password config");
    }
  }

  configure_policies();
  aerospike_init(&as_, &config);

  try {
    as_error err;
    if (aerospike_connect(&as_, &err) != AEROSPIKE_OK) {
      throw std::runtime_error("aerospike_connect failed: " +
                               status_message(err.code, err));
    }
    connected_ = true;

    size_t discovered =
        max_record_bytes == 0 ? discover_record_cap() : max_record_bytes;
    if (discovered <= kSafetyMarginBytes) {
      throw std::runtime_error("Aerospike record cap is too small");
    }
    max_record_bytes_ = discovered - kSafetyMarginBytes;
    target_segment_bytes_ =
        target_segment_bytes == 0
            ? max_record_bytes_
            : std::min(target_segment_bytes, max_record_bytes_);
    single_record_threshold_bytes_ = target_segment_bytes_;

    start_workers();
  } catch (...) {
    if (connected_) {
      as_error close_err;
      aerospike_close(&as_, &close_err);
      connected_ = false;
    }
    aerospike_destroy(&as_);
    throw;
  }
}

AerospikeNativeConnector::~AerospikeNativeConnector() { close(); }

void AerospikeNativeConnector::close() {
  {
    std::lock_guard<std::mutex> lk(close_mu_);
    if (closed_native_) {
      return;
    }
    closed_native_ = true;
  }

  ConnectorBase<WorkerAerospikeConn>::close();
}

WorkerAerospikeConn AerospikeNativeConnector::create_connection() {
  WorkerAerospikeConn conn;
  conn.client = &as_;
  conn.ns = ns_;
  conn.set_name = set_name_;

  as_policy_read_init(&conn.read_policy);
  conn.read_policy.base.total_timeout = read_timeout_ms_;
  conn.read_policy.base.socket_timeout = read_timeout_ms_;
  conn.read_policy.base.max_retries = 2;
  conn.read_policy.key = AS_POLICY_KEY_DIGEST;
  conn.read_policy.replica = AS_POLICY_REPLICA_SEQUENCE;

  as_policy_write_init(&conn.write_policy);
  conn.write_policy.base.total_timeout = write_timeout_ms_;
  conn.write_policy.base.socket_timeout = write_timeout_ms_;
  conn.write_policy.base.max_retries = 0;
  conn.write_policy.key = AS_POLICY_KEY_DIGEST;
  conn.write_policy.exists = AS_POLICY_EXISTS_IGNORE;
  conn.write_policy.gen = AS_POLICY_GEN_IGNORE;
  conn.write_policy.commit_level = AS_POLICY_COMMIT_LEVEL_ALL;
  conn.write_policy.ttl = default_ttl_seconds_;

  as_policy_remove_init(&conn.remove_policy);
  conn.remove_policy.base.total_timeout = write_timeout_ms_;
  conn.remove_policy.base.socket_timeout = write_timeout_ms_;
  conn.remove_policy.base.max_retries = 0;
  conn.remove_policy.key = AS_POLICY_KEY_DIGEST;

  return conn;
}

void AerospikeNativeConnector::do_single_get(WorkerAerospikeConn& conn,
                                             const std::string& key, void* buf,
                                             size_t len,
                                             size_t /*chunk_size*/) {
  std::string meta_key = meta_user_key(key);

  as_key as_meta_key;
  as_key_init_str(&as_meta_key, conn.ns.c_str(), conn.set_name.c_str(),
                  meta_key.c_str());

  as_error err;
  as_record* rec = nullptr;
  as_status status = aerospike_key_get(conn.client, &err, &conn.read_policy,
                                       &as_meta_key, &rec);
  if (status != AEROSPIKE_OK) {
    throw_status("get-meta", status, err);
  }
  if (rec == nullptr) {
    throw std::runtime_error("get-meta returned no record");
  }

  const char* state = as_record_get_str(rec, kBinState);
  if (state == nullptr || std::strcmp(state, kReady) != 0) {
    as_record_destroy(rec);
    throw std::runtime_error("meta record is not ready");
  }

  uint32_t nseg = static_cast<uint32_t>(positive_int_bin(rec, kBinNseg, 1));
  size_t seg_b = static_cast<size_t>(positive_int_bin(rec, kBinSegBytes, len));
  // Read the stored total directly with a sentinel so a missing or corrupt bin
  // fails the integrity check instead of silently matching `len`.
  int64_t total_raw = as_record_get_int64(rec, kBinTotalBytes, -1);
  if (total_raw < 0 || static_cast<size_t>(total_raw) != len) {
    as_record_destroy(rec);
    throw std::runtime_error("meta record total size mismatch");
  }
  bool ok = true;
  if (nseg == 1) {
    as_bytes* payload = as_record_get_bytes(rec, kBinPayload);
    if (payload == nullptr || payload->size != len) {
      ok = false;
    } else {
      std::memcpy(buf, payload->value, len);
    }
  }
  as_record_destroy(rec);

  if (!ok) {
    throw std::runtime_error("single-record payload size mismatch");
  }
  if (nseg == 1) {
    return;
  }

  size_t offset = 0;
  for (uint32_t i = 0; i < nseg; ++i) {
    std::string segment_key_i = segment_user_key(key, i);
    size_t chunk_len = std::min(seg_b, len - offset);
    if (!read_payload_record(conn, segment_key_i,
                             static_cast<char*>(buf) + offset, chunk_len)) {
      throw std::runtime_error("missing segment payload");
    }
    offset += chunk_len;
  }
  if (offset != len) {
    throw std::runtime_error("segment read size mismatch");
  }
}

void AerospikeNativeConnector::do_single_set(WorkerAerospikeConn& conn,
                                             const std::string& key,
                                             const void* buf, size_t len,
                                             size_t /*chunk_size*/) {
  ShardPlan shard = plan(len);

  if (shard.nseg == 1) {
    put_meta_record(conn, meta_user_key(key), shard, len, buf);
    return;
  }

  for (uint32_t i = 0; i < shard.nseg; ++i) {
    size_t start = static_cast<size_t>(i) * shard.seg_b;
    size_t chunk_len = std::min(shard.seg_b, len - start);
    put_payload_record(conn, segment_user_key(key, i),
                       static_cast<const char*>(buf) + start, chunk_len);
  }
  put_meta_record(conn, meta_user_key(key), shard, len, nullptr);
}

bool AerospikeNativeConnector::do_single_exists(WorkerAerospikeConn& conn,
                                                const std::string& key) {
  std::string user_key = meta_user_key(key);

  as_key as_meta_key;
  as_key_init_str(&as_meta_key, conn.ns.c_str(), conn.set_name.c_str(),
                  user_key.c_str());

  as_error err;
  as_record* rec = nullptr;
  as_status status = aerospike_key_exists(conn.client, &err, &conn.read_policy,
                                          &as_meta_key, &rec);
  if (rec != nullptr) {
    as_record_destroy(rec);
  }
  if (status == AEROSPIKE_OK) {
    return true;
  }
  if (status == AEROSPIKE_ERR_RECORD_NOT_FOUND) {
    return false;
  }
  throw_status("exists", status, err);
  return false;
}

bool AerospikeNativeConnector::do_single_delete(WorkerAerospikeConn& conn,
                                                const std::string& key) {
  std::string user_key = meta_user_key(key);
  uint32_t nseg = 1;

  as_key as_meta_key;
  as_key_init_str(&as_meta_key, conn.ns.c_str(), conn.set_name.c_str(),
                  user_key.c_str());

  as_error err;
  as_record* rec = nullptr;
  as_status status = aerospike_key_get(conn.client, &err, &conn.read_policy,
                                       &as_meta_key, &rec);
  if (status == AEROSPIKE_ERR_RECORD_NOT_FOUND) {
    return false;
  }
  if (status != AEROSPIKE_OK) {
    throw_status("delete-read-meta", status, err);
  }
  if (rec != nullptr) {
    nseg = static_cast<uint32_t>(positive_int_bin(rec, kBinNseg, 1));
    as_record_destroy(rec);
  }

  status = aerospike_key_remove(conn.client, &err, &conn.remove_policy,
                                &as_meta_key);
  if (status == AEROSPIKE_ERR_RECORD_NOT_FOUND) {
    return false;
  }
  if (status != AEROSPIKE_OK) {
    throw_status("delete-meta", status, err);
  }

  for (uint32_t i = 0; i < nseg; ++i) {
    as_key as_segment_key;
    std::string segment_key_i = segment_user_key(key, i);
    as_key_init_str(&as_segment_key, conn.ns.c_str(), conn.set_name.c_str(),
                    segment_key_i.c_str());
    as_error seg_err;
    aerospike_key_remove(conn.client, &seg_err, &conn.remove_policy,
                         &as_segment_key);
  }
  return true;
}

void AerospikeNativeConnector::shutdown_connections() {
  // Shared client must outlive worker threads until ConnectorBase drains.
}

void AerospikeNativeConnector::on_workers_stopped() {
  if (connected_) {
    as_error err;
    aerospike_close(&as_, &err);
    aerospike_destroy(&as_);
    connected_ = false;
  }
}

std::vector<std::pair<std::string, int>> AerospikeNativeConnector::parse_hosts(
    const std::string& hosts) {
  std::vector<std::pair<std::string, int>> out;
  for (const std::string& part : split(hosts, ',')) {
    size_t colon = part.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= part.size()) {
      throw std::runtime_error("hosts must be host:port[,host:port...]");
    }
    try {
      out.emplace_back(part.substr(0, colon),
                       std::stoi(part.substr(colon + 1)));
    } catch (...) {
      throw std::runtime_error("invalid port in hosts config: " + part);
    }
  }
  if (out.empty()) {
    throw std::runtime_error("hosts must not be empty");
  }
  return out;
}

std::string AerospikeNativeConnector::meta_user_key(
    const std::string& cache_key) {
  return cache_key + "|m";
}

std::string AerospikeNativeConnector::segment_user_key(
    const std::string& cache_key, uint32_t index) {
  return cache_key + "|s|" + std::to_string(index);
}

void AerospikeNativeConnector::throw_status(const char* op, as_status status,
                                            const as_error& err) {
  if (status == AEROSPIKE_ERR_RECORD_NOT_FOUND) {
    throw std::runtime_error(std::string(op) + ": record not found");
  }
  throw std::runtime_error(std::string(op) + ": " +
                           status_message(status, err));
}

ShardPlan AerospikeNativeConnector::plan(size_t payload_bytes) const {
  if (payload_bytes <= single_record_threshold_bytes_ &&
      payload_bytes <= max_record_bytes_) {
    return {1, payload_bytes};
  }
  uint32_t nseg = static_cast<uint32_t>(
      (payload_bytes + target_segment_bytes_ - 1) / target_segment_bytes_);
  size_t seg_b = (payload_bytes + nseg - 1) / nseg;
  if (seg_b > max_record_bytes_) {
    throw std::runtime_error("payload cannot be sharded within record cap");
  }
  return {nseg, seg_b};
}

size_t AerospikeNativeConnector::discover_record_cap() {
  std::string request = "namespace/" + ns_;
  char* response = nullptr;
  as_error err;
  as_status status =
      aerospike_info_any(&as_, &err, nullptr, request.c_str(), &response);
  if (status != AEROSPIKE_OK || response == nullptr) {
    if (response != nullptr) {
      std::free(response);
    }
    throw_status("info namespace", status, err);
  }

  std::string text(response);
  std::free(response);
  for (const char* field : {"max-record-size=", "write-block-size="}) {
    size_t pos = text.find(field);
    if (pos == std::string::npos) {
      continue;
    }
    pos += std::strlen(field);
    size_t end = text.find(';', pos);
    std::string value =
        text.substr(pos, end == std::string::npos ? end : end - pos);
    size_t cap = 0;
    try {
      cap = static_cast<size_t>(std::stoull(value));
    } catch (...) {
      continue;
    }
    if (cap > 0) {
      return cap;
    }
  }
  return kDefaultRecordCapBytes;
}

void AerospikeNativeConnector::configure_policies() {}

void AerospikeNativeConnector::put_payload_record(WorkerAerospikeConn& conn,
                                                  const std::string& user_key,
                                                  const void* buf, size_t len) {
  as_key key;
  as_key_init_str(&key, conn.ns.c_str(), conn.set_name.c_str(),
                  user_key.c_str());

  as_record rec;
  as_record_inita(&rec, 1);
  rec.ttl = AS_RECORD_CLIENT_DEFAULT_TTL;
  as_record_set_raw(&rec, kBinPayload, reinterpret_cast<const uint8_t*>(buf),
                    static_cast<uint32_t>(len));

  as_error err;
  as_status status =
      aerospike_key_put(conn.client, &err, &conn.write_policy, &key, &rec);
  as_record_destroy(&rec);
  if (status != AEROSPIKE_OK) {
    throw_status("put-payload", status, err);
  }
}

void AerospikeNativeConnector::put_meta_record(WorkerAerospikeConn& conn,
                                               const std::string& user_key,
                                               const ShardPlan& shard,
                                               size_t total_bytes,
                                               const void* inline_buf) {
  as_key key;
  as_key_init_str(&key, conn.ns.c_str(), conn.set_name.c_str(),
                  user_key.c_str());

  as_record rec;
  as_record_inita(&rec, inline_buf == nullptr ? 7 : 8);
  rec.ttl = AS_RECORD_CLIENT_DEFAULT_TTL;
  as_record_set_int64(&rec, kBinVersion, 1);
  as_record_set_str(&rec, kBinState, kReady);
  as_record_set_int64(&rec, kBinNseg, shard.nseg);
  as_record_set_int64(&rec, kBinSegBytes, static_cast<int64_t>(shard.seg_b));
  as_record_set_int64(&rec, kBinTotalBytes, static_cast<int64_t>(total_bytes));
  as_record_set_int64(&rec, kBinCreatedAt,
                      static_cast<int64_t>(std::time(nullptr)));
  as_record_set_bool(&rec, kBinPin, false);
  if (inline_buf != nullptr) {
    // The inline payload path is only taken for single-record writes, where
    // do_single_set() already guaranteed (via plan()) that
    // total_bytes <= max_record_bytes_ -- the discovered server record cap
    // minus a safety margin, which is far below UINT32_MAX. The narrowing cast
    // below is therefore safe; assert the invariant in case that ever changes.
    assert(total_bytes <= max_record_bytes_);
    as_record_set_raw(&rec, kBinPayload,
                      reinterpret_cast<const uint8_t*>(inline_buf),
                      static_cast<uint32_t>(total_bytes));
  }

  as_error err;
  as_status status =
      aerospike_key_put(conn.client, &err, &conn.write_policy, &key, &rec);
  as_record_destroy(&rec);
  if (status != AEROSPIKE_OK) {
    throw_status("put-meta", status, err);
  }
}

bool AerospikeNativeConnector::read_payload_record(WorkerAerospikeConn& conn,
                                                   const std::string& user_key,
                                                   void* buf, size_t len) {
  as_key key;
  as_key_init_str(&key, conn.ns.c_str(), conn.set_name.c_str(),
                  user_key.c_str());

  as_error err;
  as_record* rec = nullptr;
  as_status status =
      aerospike_key_get(conn.client, &err, &conn.read_policy, &key, &rec);
  if (status == AEROSPIKE_ERR_RECORD_NOT_FOUND) {
    return false;
  }
  if (status != AEROSPIKE_OK) {
    throw_status("get-payload", status, err);
  }
  if (rec == nullptr) {
    return false;
  }
  as_bytes* payload = as_record_get_bytes(rec, kBinPayload);
  bool ok = payload != nullptr && payload->size == len;
  if (ok) {
    std::memcpy(buf, payload->value, payload->size);
  }
  as_record_destroy(rec);
  return ok;
}

}  // namespace connector
}  // namespace lmcache
