#pragma once

#include <torch/all.h>
#include <string>

int64_t valkey_init(std::string ip, int64_t port, bool enable_rdma);
int64_t valkey_set(std::string key, torch::Tensor& value);
int64_t valkey_get(std::string key, torch::Tensor& value);
int64_t valkey_delete(std::string key);
bool valkey_key_exists(std::string key);
