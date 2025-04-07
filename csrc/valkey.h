#pragma once

#include <stdio.h>
#include <stdbool.h>
#include <torch/all.h>
#include <string>
#include <valkey/valkey.h>
#include <valkey/rdma.h>

#include "queue.h"

using fptr_t = int64_t;

class SwapInfo {
 public:
  SwapInfo() {}
  SwapInfo(std::vector<int64_t> blocks, std::vector<int64_t> hashs)
      : blocks(blocks), hashs(hashs) {}
  SwapInfo(std::string req_id, std::vector<int64_t> blocks,
           std::vector<int64_t> hashs)
      : req_id(req_id), blocks(blocks), hashs(hashs) {}

  std::string req_id;
  std::vector<int64_t> blocks;
  std::vector<int64_t> hashs;
};

class ValkeyClient {
 public:
  ValkeyClient(std::string ip, int64_t port, bool enable_rdma, int64_t rank);
  ~ValkeyClient();

  void set(std::string key, torch::Tensor& value);
  void get(std::string key, torch::Tensor& value);
  void del(std::string key);
  bool exist(std::string key);
  void reg_mr(std::vector<torch::Tensor> kv_caches);
  void swap_in(std::string& req_id, std::vector<int64_t> hashs,
               std::vector<int64_t> blocks);
  void swap_out(std::vector<int64_t> blocks, std::vector<int64_t> hashs);
  std::vector<std::string> get_loaded_reqs();
  std::vector<int64_t> get_saved_blocks();
  void disconnect();

 private:
  int64_t rank;
  valkeyContext* valkey_ctx;
  torch::Tensor flag = torch::tensor(1);
  std::vector<torch::Tensor> kv_caches;
  Deque<SwapInfo> swap_in_queue;
  Deque<SwapInfo> swap_out_queue;
  std::mutex swap_in_mutex;
  std::mutex swap_out_mutex;
  std::vector<std::string> loaded_reqs;
  std::vector<int64_t> saved_blocks;

  void _handle_swap_in_loop();
  void _handle_swap_out_loop();
};

fptr_t valkey_connect(std::string ip, int64_t port, bool enable_rdma,
                      int64_t rank);
void valkey_set(fptr_t _fa, std::string key, torch::Tensor& val);
void valkey_get(fptr_t _fa, std::string key, torch::Tensor& val);
void valkey_delete(fptr_t _fa, std::string key);
bool valkey_exist(fptr_t _fa, std::string key);
void valkey_disconnect(fptr_t _fa);
void swap_in(fptr_t _fa, std::string req_id, std::vector<int64_t> hashs,
             std::vector<int64_t> blocks);
void swap_out(fptr_t _fa, std::vector<int64_t> blocks,
              std::vector<int64_t> hashs);
std::vector<std::string> get_loaded_reqs(fptr_t _fa);
std::vector<int64_t> get_saved_blocks(fptr_t _fa);
void reg_mr(fptr_t _fa, std::vector<torch::Tensor> kv_caches);
