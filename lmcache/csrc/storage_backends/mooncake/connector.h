// SPDX-License-Identifier: Apache-2.0
#pragma once

// Standard
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Third Party
#include "real_client.h"

// First Party
#include "../connector_base.h"

namespace lmcache {
namespace connector {

// ConfigDict mirrors mooncake::ConfigDict
// (std::unordered_map<std::string, std::string>).
using ConfigDict = std::unordered_map<std::string, std::string>;

// Per-worker connection state for the Mooncake connector.
// Each worker holds a raw pointer to the shared
// RealClient (owned by MooncakeConnector).
struct WorkerMooncakeConn {
  mooncake::RealClient* client{nullptr};
};

struct L1RegistrationConfig {
  bool enabled{false};
  std::uintptr_t base{0};
  size_t size{0};

  bool is_valid() const { return enabled && base != 0 && size != 0; }
};

class MooncakeConnector : public ConnectorBase<WorkerMooncakeConn> {
 public:
  MooncakeConnector(ConfigDict config, int num_workers,
                    L1RegistrationConfig l1_registration = {},
                    WorkerPoolConfig worker_pool_config = {});
  ~MooncakeConnector() override;

 protected:
  WorkerMooncakeConn create_connection() override;

  void do_single_get(WorkerMooncakeConn& conn, const std::string& key,
                     void* buf, size_t len, size_t chunk_size) override;

  void do_single_set(WorkerMooncakeConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;

  bool do_single_exists(WorkerMooncakeConn& conn,
                        const std::string& key) override;

  bool do_single_delete(WorkerMooncakeConn& conn,
                        const std::string& key) override;

  void on_workers_stopped() override;

  size_t choose_num_tiles(Op op, size_t num_items) const override;

  void do_batch_get(WorkerMooncakeConn& conn, const Request& req) override;

  void do_batch_set(WorkerMooncakeConn& conn, const Request& req) override;

  void do_batch_exists(WorkerMooncakeConn& conn, const Request& req) override;

  void do_batch_delete(WorkerMooncakeConn& conn, const Request& req) override;

 private:
  void ensure_registered(const void* buf, size_t len);
  void preregister_l1_memory(std::uintptr_t base, size_t size);
  void unregister_all_buffers() noexcept;

  // Shared Mooncake RealClient instance.
  std::shared_ptr<mooncake::RealClient> client_;

  // The original config dict (kept for diagnostics).
  ConfigDict config_;
  L1RegistrationConfig l1_registration_;
  size_t preregistered_block_size_{0};
};

}  // namespace connector
}  // namespace lmcache
