// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "connector_base.h"
#include <unistd.h>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace example_connector {

// ---------------------------------------------------------------
// FS strategy: per-worker connection state
// ---------------------------------------------------------------

struct WorkerFSConn {
  std::filesystem::path base_path;
};

class ExampleFSConnector
    : public lmcache::connector::ConnectorBase<WorkerFSConn> {
 public:
  ExampleFSConnector(std::string base_path, int num_workers);
  ~ExampleFSConnector() override;

 protected:
  WorkerFSConn create_connection() override;
  void do_single_get(WorkerFSConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerFSConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerFSConn& conn, const std::string& key) override;
  bool do_single_delete(WorkerFSConn& conn, const std::string& key) override;

 private:
  static std::string safe_filename(const std::string& key);
  static std::string replace_all(const std::string& str,
                                 const std::string& from,
                                 const std::string& to);
  std::string base_path_;
};

// ---------------------------------------------------------------
// Memory strategy: per-worker connection state (shared store)
// ---------------------------------------------------------------

// Shared in-memory store protected by a mutex.
// All workers share a single instance via shared_ptr.
struct SharedMemoryStore {
  std::mutex mu;
  std::unordered_map<std::string, std::vector<char>> data;
};

// Each worker holds a shared_ptr to the same store.
struct WorkerMemConn {
  std::shared_ptr<SharedMemoryStore> store;
};

class ExampleMemoryConnector
    : public lmcache::connector::ConnectorBase<WorkerMemConn> {
 public:
  ExampleMemoryConnector(int num_workers);
  ~ExampleMemoryConnector() override;

 protected:
  WorkerMemConn create_connection() override;
  void do_single_get(WorkerMemConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerMemConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerMemConn& conn, const std::string& key) override;
  bool do_single_delete(WorkerMemConn& conn, const std::string& key) override;

 private:
  std::shared_ptr<SharedMemoryStore> store_;
};

}  // namespace example_connector
