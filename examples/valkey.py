import torch

from vllm import _custom_ops as ops


if __name__ == '__main__':
    ip = "127.0.0.1"
    port = 6379
    enable_rdma = True
    ops.valkey_init(ip, port, enable_rdma)

    key = "test_key1"
    value = torch.randn(10, 10)
    ops.valkey_set(key, value)

    exist = ops.valkey_key_exists(key)
    print(exist)

    value_get = torch.randn(10, 10)
    ops.valkey_get(key, value_get)

    print(value)
    print(value_get)

    ops.valkey_del(key)
    exist = ops.valkey_key_exists(key)
    print(exist)
