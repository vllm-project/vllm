#include <valkey/valkey.h>
#include <valkey/rdma.h>
#include <stdio.h>
#include <stdbool.h>
#include <torch/all.h>
#include <string>

static valkeyContext* valkey_ctx;

bool common_error_check(valkeyReply* reply) {
  if (reply == NULL || valkey_ctx->err || reply->type == VALKEY_REPLY_ERROR) {
    printf("Error: %s\n", reply->str);
    return true;
  }

  return false;
}

int64_t valkey_init(std::string ip, int64_t port, bool enable_rdma) {
  valkeyOptions options = {0};
  const char* ip_str = ip.c_str();

  if (enable_rdma) {
    valkeyInitiateRdma();
    VALKEY_OPTIONS_SET_RDMA(&options, ip_str, port);
  } else {
    VALKEY_OPTIONS_SET_TCP(&options, ip_str, port);
  }

  valkey_ctx = valkeyConnectWithOptions(&options);
  if (!valkey_ctx || valkey_ctx->err) {
    printf("Failed to connect valkey, exit ... \n");
    return -1;
  }

  return 0;
}

int64_t valkey_set(std::string key, torch::Tensor& value) {
  valkeyReply* reply;
  int64_t value_len = value.numel() * value.element_size();
  int64_t key_len = key.length();
  char* value_ptr = static_cast<char*>(value.data_ptr());

  void* valkey_reply = valkeyCommand(valkey_ctx, "SET %b %b", key.c_str(),
                                     key_len, value_ptr, value_len);
  reply = (valkeyReply*)valkey_reply;
  if (common_error_check(reply)) {
    printf("Failed to execute valkey set command \n");
    return -1;
  }

  freeReplyObject(reply);
  return 0;
}

int64_t valkey_get(std::string key, torch::Tensor& value) {
  valkeyReply* reply;
  char* value_ptr = static_cast<char*>(value.data_ptr());

  void* valkey_reply = valkeyCommand(valkey_ctx, "GET %s", key.c_str());
  reply = (valkeyReply*)valkey_reply;
  if (common_error_check(reply)) {
    printf("Failed to execute valkey get command \n");
    return -1;
  }

  memcpy(value_ptr, reply->str, reply->len);

  freeReplyObject(reply);
  return reply->len;
}

int64_t valkey_delete(std::string key) {
  valkeyReply* reply;

  void* valkey_reply = valkeyCommand(valkey_ctx, "DEL %s", key.c_str());
  reply = (valkeyReply*)valkey_reply;
  if (common_error_check(reply) || reply->type != VALKEY_REPLY_INTEGER) {
    printf("Failed to execute valkey delete command \n");
    return -1;
  }

  freeReplyObject(reply);
  return 0;
}

bool valkey_key_exists(std::string key) {
  valkeyReply* reply;

  void* valkey_reply = valkeyCommand(valkey_ctx, "EXISTS %s", key.c_str());
  reply = (valkeyReply*)valkey_reply;
  if (common_error_check(reply) || reply->type != VALKEY_REPLY_INTEGER) {
    printf("Failed to execute valkey key exists command \n");
    return false;
  }

  freeReplyObject(reply);
  return true;
}