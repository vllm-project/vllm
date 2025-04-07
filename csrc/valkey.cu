#include <valkey/valkey.h>
#include <valkey/rdma.h>
#include <stdio.h>
#include <stdbool.h>
#include <torch/all.h>
#include <string>
#include <sys/time.h>
#include <sstream>
#include <mutex>

#include "valkey.h"
#include "queue.h"

static std::once_flag rdma_init_flag;

static void assertReply(valkeyContext* context, valkeyReply* reply, int type) {
  if (reply == NULL) {
    std::stringstream ss;
    ss << "NULL reply from server (error: " << context->errstr << ")";
    throw std::runtime_error(ss.str());
  }

  if (reply->type != type) {
    if (reply->type == VALKEY_REPLY_ERROR) {
      std::stringstream ss;
      ss << "Server Error: " << reply->str;
      throw std::runtime_error(ss.str());
    }

    std::stringstream ss;
    ss << "Expected reply type " << type << " but got type " << reply->type;
    throw std::runtime_error(ss.str());
  }
}

static void assertReplyAndFree(valkeyContext* context, valkeyReply* reply,
                               int type) {
  assertReply(context, reply, type);
  freeReplyObject(reply);
}

ValkeyClient::ValkeyClient(std::string ip, int64_t port, bool enable_rdma,
                           int64_t rank) {
  this->rank = rank;
  valkeyOptions options = {0};
  const char* ip_str = ip.c_str();

  if (enable_rdma) {
    std::call_once(rdma_init_flag, []() { valkeyInitiateRdma(); });
    VALKEY_OPTIONS_SET_RDMA(&options, ip_str, port);
  } else {
    VALKEY_OPTIONS_SET_TCP(&options, ip_str, port);
  }

  this->valkey_ctx = valkeyConnectWithOptions(&options);
  if (!this->valkey_ctx || valkey_ctx->err) {
    throw std::runtime_error("Failed to connect Valkey Server");
  }

  this->swap_in_valkey_ctx = valkeyConnectWithOptions(&swap_in_options);
  if (!this->swap_in_valkey_ctx || swap_in_valkey_ctx->err) {
    throw std::runtime_error("Failed to connect Valkey Server");
  }

  this->swap_out_valkey_ctx = valkeyConnectWithOptions(&swap_out_options);
  if (!this->swap_out_valkey_ctx || swap_out_valkey_ctx->err) {
    throw std::runtime_error("Failed to connect Valkey Server");
  }

  std::thread swap_in_kv_cache_thread(
      [this]() { this->_handle_swap_in_loop(); });

  swap_in_kv_cache_thread.detach();

  std::thread swap_out_kv_cache_thread(
      [this]() { this->_handle_swap_out_loop(); });

  swap_out_kv_cache_thread.detach();
}

ValkeyClient::~ValkeyClient() {
  if (this->valkey_ctx) {
    valkeyFree(this->valkey_ctx);
  }

  if (this->swap_in_valkey_ctx) {
    valkeyFree(this->swap_in_valkey_ctx);
  }

  if (this->swap_out_valkey_ctx) {
    valkeyFree(this->swap_out_valkey_ctx);
  }
}

void ValkeyClient::set(std::string key, torch::Tensor& value) {
  torch::Tensor value_cpu;
  if (value.is_cuda()) {
    value_cpu = value.to(torch::kCPU);
  } else {
    value_cpu = value;
  }

  int64_t value_len = value_cpu.numel() * value_cpu.element_size();
  int64_t key_len = key.length();
  char* value_ptr = static_cast<char*>(value_cpu.data_ptr());

  void* reply = valkeyCommand(this->swap_out_valkey_ctx, "SET %b %b",
                              key.c_str(), key_len, value_ptr, value_len);
  valkeyReply* valkey_reply = static_cast<valkeyReply*>(reply);
  assertReplyAndFree(this->swap_out_valkey_ctx, valkey_reply,
                     VALKEY_REPLY_STATUS);
  return;
}

void ValkeyClient::get(std::string key, torch::Tensor& value) {
  bool is_gpu = value.is_cuda();
  torch::Tensor temp_tensor;
  if (is_gpu) {
    auto options =
        torch::TensorOptions().dtype(value.dtype()).device(torch::kCPU);
    temp_tensor = torch::empty_like(value, options);
  } else {
    temp_tensor = value;
  }

  char* value_ptr = static_cast<char*>(temp_tensor.data_ptr());
  void* reply = valkeyCommand(this->swap_in_valkey_ctx, "GET %s", key.c_str());
  valkeyReply* valkey_reply = static_cast<valkeyReply*>(reply);
  assertReply(this->swap_in_valkey_ctx, valkey_reply, VALKEY_REPLY_STRING);
  size_t expected_size = temp_tensor.numel() * temp_tensor.element_size();
  if (valkey_reply->len != expected_size) {
    freeReplyObject(valkey_reply);
    throw std::runtime_error("Redis data size does not match tensor size");
  }
  memcpy(value_ptr, valkey_reply->str, valkey_reply->len);
  freeReplyObject(valkey_reply);

  if (is_gpu) {
    value.copy_(temp_tensor);
  }
}

void ValkeyClient::del(std::string key) {
  void* reply = valkeyCommand(this->valkey_ctx, "DEL %s", key.c_str());
  valkeyReply* valkey_reply = static_cast<valkeyReply*>(reply);
  assertReplyAndFree(this->valkey_ctx, valkey_reply, VALKEY_REPLY_INTEGER);
}

bool ValkeyClient::exist(std::string key) {
  bool exists = false;
  void* reply = valkeyCommand(this->valkey_ctx, "EXISTS %s", key.c_str());
  valkeyReply* valkey_reply = static_cast<valkeyReply*>(reply);
  assertReply(this->valkey_ctx, valkey_reply, VALKEY_REPLY_INTEGER);
  if (valkey_reply->integer == 1) {
    exists = true;
  }

  freeReplyObject(valkey_reply);
  return exists;
}

void ValkeyClient::reg_mr(std::vector<torch::Tensor> kv_caches) {
  for (const auto& tensor : kv_caches) {
    this->kv_caches.push_back(tensor);
  }

  return;
}

void ValkeyClient::swap_in(std::string& req_id, std::vector<int64_t> hashs,
                           std::vector<int64_t> blocks) {
  this->swap_in_queue.push(SwapInfo(req_id, blocks, hashs));
  return;
}

void ValkeyClient::swap_out(std::vector<int64_t> blocks,
                            std::vector<int64_t> hashs) {
  this->swap_out_queue.push(SwapInfo(blocks, hashs));
  return;
}

std::vector<std::string> ValkeyClient::get_loaded_reqs() {
  std::vector<std::string> reqs;

  this->swap_in_mutex.lock();
  for (const auto& req : this->loaded_reqs) {
    reqs.push_back(req);
  }

  this->loaded_reqs.clear();
  this->swap_in_mutex.unlock();
  return reqs;
}

std::vector<int64_t> ValkeyClient::get_saved_blocks() {
  std::vector<int64_t> blocks;

  this->swap_out_mutex.lock();
  for (const auto& req : this->saved_blocks) {
    blocks.push_back(req);
  }

  this->saved_blocks.clear();
  this->swap_out_mutex.unlock();
  return blocks;
}

void ValkeyClient::disconnect() {
  if (this->valkey_ctx) {
    valkeyFree(this->valkey_ctx);
  }
}

void ValkeyClient::_handle_swap_in_loop() {
  while (true) {
    while (!this->swap_in_queue.empty()) {
      SwapInfo element;
      this->swap_in_queue.pop(element);
      std::vector<int64_t> hashs = element.hashs;
      std::vector<int64_t> blocks = element.blocks;
      std::string req_id = element.req_id;

      const int64_t num_blocks = hashs.size();
      for (size_t i = 0; i < num_blocks; i++) {
        int64_t hash = hashs[i];
        int64_t block = blocks[i];

        for (size_t i = 0; i < this->kv_caches.size(); i++) {
          auto& layer_kv_cache = this->kv_caches[i];
          torch::Tensor k_cache = layer_kv_cache[0];
          torch::Tensor v_cache = layer_kv_cache[1];

          torch::Tensor k_block_cache = k_cache[block];
          torch::Tensor v_block_cache = v_cache[block];

          std::string kcache_key = std::to_string(hash) + "/rank" +
                                   std::to_string(this->rank) + "/layer" +
                                   std::to_string(i) + "/key";
          std::string vcache_key = std::to_string(hash) + "/rank" +
                                   std::to_string(this->rank) + "/layer" +
                                   std::to_string(i) + "/val";

          this->get(kcache_key, k_block_cache);
          this->get(vcache_key, v_block_cache);
        }
      }

      this->swap_in_mutex.lock();
      this->loaded_reqs.push_back(req_id);
      this->swap_in_mutex.unlock();
    }
  }
}

void ValkeyClient::_handle_swap_out_loop() {
  while (true) {
    while (!this->swap_out_queue.empty()) {
      SwapInfo element;
      this->swap_out_queue.pop(element);
      std::vector<int64_t> hashs = element.hashs;
      std::vector<int64_t> blocks = element.blocks;
      std::string req_id = element.req_id;
      const int64_t num_blocks = hashs.size();
      for (size_t i = 0; i < num_blocks; i++) {
        int64_t block = blocks[i];
        int64_t hash = hashs[i];
        for (size_t i = 0; i < this->kv_caches.size(); i++) {
          auto& layer_kv_cache = this->kv_caches[i];
          torch::Tensor k_cache = layer_kv_cache[0];
          torch::Tensor v_cache = layer_kv_cache[1];

          torch::Tensor k_block_cache = k_cache[block];
          torch::Tensor v_block_cache = v_cache[block];

          std::string kcache_key = std::to_string(hash) + "/rank" +
                                   std::to_string(this->rank) + "/layer" +
                                   std::to_string(i) + "/key";
          std::string vcache_key = std::to_string(hash) + "/rank" +
                                   std::to_string(this->rank) + "/layer" +
                                   std::to_string(i) + "/val";

          this->set(kcache_key, k_block_cache);
          this->set(vcache_key, v_block_cache);
        }

        std::string hash_str = std::to_string(hash);
        this->set(hash_str, this->flag);
        this->swap_out_mutex.lock();
        this->saved_blocks.push_back(block);
        this->swap_out_mutex.unlock();
      }
    }
  }
}

fptr_t valkey_connect(std::string ip, int64_t port, bool enable_rdma,
                      int64_t rank) {
  return (fptr_t) new ValkeyClient(ip, port, enable_rdma, rank);
}

void valkey_set(fptr_t _fa, std::string key, torch::Tensor& val) {
  ValkeyClient* client = reinterpret_cast<ValkeyClient*>(_fa);
  client->set(key, val);
  return;
}

void valkey_get(fptr_t _fa, std::string key, torch::Tensor& val) {
  ValkeyClient* client = reinterpret_cast<ValkeyClient*>(_fa);
  return client->get(key, val);
}

void valkey_delete(fptr_t _fa, std::string key) {
  ValkeyClient* client = reinterpret_cast<ValkeyClient*>(_fa);
  return client->del(key);
}

bool valkey_exist(fptr_t _fa, std::string key) {
  ValkeyClient* client = reinterpret_cast<ValkeyClient*>(_fa);
  return client->exist(key);
}

void valkey_disconnect(fptr_t _fa) {
  ValkeyClient* client = reinterpret_cast<ValkeyClient*>(_fa);
  return client->disconnect();
}

void reg_mr(fptr_t _fa, std::vector<torch::Tensor> kv_caches) {
  ValkeyClient* client = reinterpret_cast<ValkeyClient*>(_fa);
  client->reg_mr(kv_caches);
}

void swap_in(fptr_t _fa, std::string req_id, std::vector<int64_t> hashs,
             std::vector<int64_t> blocks) {
  if (blocks.size() != hashs.size()) {
    throw std::runtime_error(
        "swap_out: blocks and hashs must have the same size");
  }
  ValkeyClient* client = reinterpret_cast<ValkeyClient*>(_fa);
  client->swap_in(req_id, hashs, blocks);
}

void swap_out(fptr_t _fa, std::vector<int64_t> blocks,
              std::vector<int64_t> hashs) {
  if (blocks.size() != hashs.size()) {
    throw std::runtime_error(
        "swap_out: blocks and hashs must have the same size");
  }
  ValkeyClient* client = reinterpret_cast<ValkeyClient*>(_fa);
  client->swap_out(blocks, hashs);
}

std::vector<std::string> get_loaded_reqs(fptr_t _fa) {
  ValkeyClient* client = reinterpret_cast<ValkeyClient*>(_fa);
  return client->get_loaded_reqs();
}

std::vector<int64_t> get_saved_blocks(fptr_t _fa) {
  ValkeyClient* client = reinterpret_cast<ValkeyClient*>(_fa);
  return client->get_saved_blocks();
}
