// SPDX-License-Identifier: Apache-2.0

// Standard
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

// Third Party
#include "config.h"

// Local
#include "connector.h"

namespace lmcache {
namespace connector {

namespace {

template <typename T>
void ensure_batch_result_size(const std::vector<T>& results, size_t expected,
                              const char* op_name) {
  if (results.size() != expected) {
    throw std::runtime_error(std::string("Mooncake ") + op_name + " returned " +
                             std::to_string(results.size()) + " results for " +
                             std::to_string(expected) + " keys");
  }
}

}  // namespace

MooncakeConnector::MooncakeConnector(ConfigDict config, int num_workers,
                                     L1RegistrationConfig l1_registration,
                                     WorkerPoolConfig worker_pool_config)
    : ConnectorBase(num_workers, std::move(worker_pool_config)),
      config_(std::move(config)),
      l1_registration_(l1_registration) {
  // Create a RealClient via the static factory.
  client_ = mooncake::RealClient::create();
  if (!client_) {
    throw std::runtime_error("Failed to create mooncake RealClient");
  }

  // Forward the config dict to setup_internal().
  mooncake::ConfigDict mc_config(config_.begin(), config_.end());
  auto result = client_->setup_internal(mc_config);
  if (!result.has_value()) {
    throw std::runtime_error("Mooncake setup_internal failed");
  }

  if (l1_registration.is_valid()) {
    preregister_l1_memory(l1_registration.base, l1_registration.size);
  }

  start_workers();  // IMPORTANT: call at END of ctor
}

MooncakeConnector::~MooncakeConnector() {
  close();
  if (client_) {
    client_->tearDownAll();
    client_.reset();
  }
}

WorkerMooncakeConn MooncakeConnector::create_connection() {
  WorkerMooncakeConn conn;
  conn.client = client_.get();
  return conn;
}

void MooncakeConnector::do_single_get(WorkerMooncakeConn& conn,
                                      const std::string& key, void* buf,
                                      size_t len, size_t chunk_size) {
  (void)chunk_size;
  ensure_registered(buf, len);
  int64_t bytes_read = conn.client->get_into(key, buf, len);
  if (bytes_read <= 0) {
    throw std::runtime_error("Mooncake get_into failed for key: " + key);
  }
}

void MooncakeConnector::do_single_set(WorkerMooncakeConn& conn,
                                      const std::string& key, const void* buf,
                                      size_t len, size_t chunk_size) {
  (void)chunk_size;
  ensure_registered(buf, len);
  int rc = conn.client->put_from(key, const_cast<void*>(buf), len);
  if (rc != 0) {
    throw std::runtime_error("Mooncake put_from failed for key: " + key);
  }
}

bool MooncakeConnector::do_single_exists(WorkerMooncakeConn& conn,
                                         const std::string& key) {
  // isExist returns: 1=exists, 0=not, -1=error
  int result = conn.client->isExist(key);
  if (result < 0) {
    fprintf(stderr,
            "[LMCache EXISTS] key %s failed: Mooncake isExist returned %d\n",
            key.c_str(), result);
    return false;
  }
  return result == 1;
}

bool MooncakeConnector::do_single_delete(WorkerMooncakeConn& conn,
                                         const std::string& key) {
  int rc = conn.client->remove(key, /*force=*/true);
  if (rc != 0) {
    fprintf(stderr,
            "[LMCache DELETE] key %s failed: Mooncake remove returned %d\n",
            key.c_str(), rc);
    return false;
  }
  return true;
}

void MooncakeConnector::on_workers_stopped() { unregister_all_buffers(); }

size_t MooncakeConnector::choose_num_tiles(Op op, size_t num_items) const {
  (void)op;
  (void)num_items;
  // Mooncake's batch APIs already parallelize internally. Keep each LMCache
  // batch as one tile so we do not split a single backend batch across workers.
  return 1;
}

void MooncakeConnector::do_batch_get(WorkerMooncakeConn& conn,
                                     const Request& req) {
  // Keep GET tolerant at per-key granularity: registration failures should
  // only zero that key. Keep the no-failure path on Mooncake's batch API, and
  // fall back to ConnectorBase's singleton-style handling on error.
  for (size_t i = 0; i < req.buf_ptrs.size(); ++i) {
    try {
      ensure_registered(req.buf_ptrs[i], req.buf_lens[i]);
    } catch (const std::exception& e) {
      fprintf(stderr,
              "[LMCache GET] Mooncake buffer registration failed for key %zu, "
              "falling back to per-key get: %s\n",
              i, e.what());
      ConnectorBase<WorkerMooncakeConn>::do_batch_get(conn, req);
      return;
    }
  }

  auto results =
      conn.client->batch_get_into(req.keys, req.buf_ptrs, req.buf_lens);
  ensure_batch_result_size(results, req.keys.size(), "batch_get_into");

  for (size_t i = 0; i < results.size(); ++i) {
    if (results[i] <= 0) {
      req.batch->per_key_results[req.start_idx + i] = 0;
      fprintf(stderr,
              "[LMCache GET] key %s failed: Mooncake batch_get_into "
              "returned %lld\n",
              req.keys[i].c_str(), static_cast<long long>(results[i]));
      continue;
    }
    req.batch->per_key_results[req.start_idx + i] = 1;
  }
}

void MooncakeConnector::do_batch_set(WorkerMooncakeConn& conn,
                                     const Request& req) {
  for (size_t i = 0; i < req.buf_ptrs.size(); ++i) {
    ensure_registered(req.buf_ptrs[i], req.buf_lens[i]);
  }

  auto results =
      conn.client->batch_put_from(req.keys, req.buf_ptrs, req.buf_lens);
  ensure_batch_result_size(results, req.keys.size(), "batch_put_from");

  for (size_t i = 0; i < results.size(); ++i) {
    if (results[i] != 0) {
      throw std::runtime_error("Mooncake batch_put_from failed for key: " +
                               req.keys[i]);
    }
  }
}

void MooncakeConnector::do_batch_exists(WorkerMooncakeConn& conn,
                                        const Request& req) {
  auto results = conn.client->batchIsExist(req.keys);
  ensure_batch_result_size(results, req.keys.size(), "batchIsExist");

  for (size_t i = 0; i < results.size(); ++i) {
    if (results[i] < 0) {
      fprintf(stderr,
              "[LMCache EXISTS] key %s failed: Mooncake batchIsExist "
              "returned %d\n",
              req.keys[i].c_str(), results[i]);
      req.batch->per_key_results[req.start_idx + i] = 0;
      continue;
    }
    req.batch->per_key_results[req.start_idx + i] = results[i] == 1 ? 1 : 0;
  }
}

void MooncakeConnector::do_batch_delete(WorkerMooncakeConn& conn,
                                        const Request& req) {
  auto results = conn.client->batchRemove(req.keys, /*force=*/true);
  ensure_batch_result_size(results, req.keys.size(), "batchRemove");

  for (size_t i = 0; i < results.size(); ++i) {
    if (results[i] != 0) {
      fprintf(stderr,
              "[LMCache DELETE] key %s failed: Mooncake batchRemove "
              "returned %d\n",
              req.keys[i].c_str(), results[i]);
      req.batch->per_key_results[req.start_idx + i] = 0;
      continue;
    }
    req.batch->per_key_results[req.start_idx + i] = 1;
  }
}

void MooncakeConnector::ensure_registered(const void* buf, size_t len) {
  if (!l1_registration_.is_valid()) {
    return;
  }
  if (buf == nullptr) {
    throw std::runtime_error(
        "Mooncake buffer registration failed: null buffer");
  }
  if (len == 0) {
    throw std::runtime_error(
        "Mooncake buffer registration failed: zero length");
  }

  if (preregistered_block_size_ == 0) {
    throw std::runtime_error("Mooncake preregistered block size is invalid");
  }

  const auto registered_begin = l1_registration_.base;
  const auto registered_end = registered_begin + l1_registration_.size;
  const auto buf_begin = reinterpret_cast<std::uintptr_t>(buf);
  const auto buf_end = buf_begin + len;

  if (buf_begin < registered_begin || buf_end > registered_end) {
    throw std::runtime_error(
        "Buffer is outside of preregistered L1 region; Mooncake lazy "
        "registration is disabled");
  }

  const auto relative_begin = buf_begin - registered_begin;
  const auto relative_end = (buf_end - 1) - registered_begin;
  const size_t left_region = relative_begin / preregistered_block_size_;
  const size_t right_region = relative_end / preregistered_block_size_;

  if (left_region != right_region) {
    throw std::runtime_error("Buffer crosses preregistered Mooncake regions");
  }
}

void MooncakeConnector::preregister_l1_memory(std::uintptr_t base,
                                              size_t size) {
  if (base == 0 || size == 0) {
    return;
  }

  preregistered_block_size_ = mooncake::globalConfig().max_mr_size == 0
                                  ? size
                                  : mooncake::globalConfig().max_mr_size;
  const size_t max_registration_size = preregistered_block_size_;
  size_t remaining = size;
  auto current = base;

  while (remaining > 0) {
    const size_t segment_size = std::min(remaining, max_registration_size);
    void* segment_ptr = reinterpret_cast<void*>(current);
    const int register_rc = client_->register_buffer(segment_ptr, segment_size);
    if (register_rc != 0) {
      auto rollback_remaining = current - base;
      while (rollback_remaining > 0) {
        const auto rollback_size =
            std::min(rollback_remaining, max_registration_size);
        rollback_remaining -= rollback_size;
        void* rollback_ptr = reinterpret_cast<void*>(base + rollback_remaining);
        try {
          client_->unregister_buffer(rollback_ptr);
        } catch (...) {
          // Keep rolling back the remaining segments so a cleanup failure does
          // not mask the original registration error or strand later regions.
        }
      }
      preregistered_block_size_ = 0;
      throw std::runtime_error("Mooncake preregister_l1_memory failed");
    }

    current += segment_size;
    remaining -= segment_size;
  }
}

void MooncakeConnector::unregister_all_buffers() noexcept {
  if (client_ == nullptr || !l1_registration_.is_valid() ||
      preregistered_block_size_ == 0) {
    return;
  }

  auto current = l1_registration_.base;
  size_t remaining = l1_registration_.size;
  while (remaining > 0) {
    const auto segment_size = std::min(remaining, preregistered_block_size_);
    void* segment_ptr = reinterpret_cast<void*>(current);
    try {
      client_->unregister_buffer(segment_ptr);
    } catch (...) {
      // Preserve noexcept during teardown and keep attempting the remaining
      // segments so earlier failures do not strand later registrations.
    }
    current += segment_size;
    remaining -= segment_size;
  }

  preregistered_block_size_ = 0;
}

}  // namespace connector
}  // namespace lmcache
