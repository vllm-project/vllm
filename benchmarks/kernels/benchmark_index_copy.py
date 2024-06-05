import os
import time

import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.dynamo_set_buffer_donor  # noqa: F401

os.environ["PJRT_DEVICE"] = "TPU"
device = xm.xla_device()

BATCH_SIZE = 1
SEQ_LEN = 1024
NUM_KV_HEADS = 16
HEAD_SIZE = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    torch.ops.xla.dynamo_set_buffer_donor_(k_cache, True)
    torch.ops.xla.dynamo_set_buffer_donor_(v_cache, True)
    key = key.flatten(0, 2)
    value = value.flatten(0, 2)
    k_cache = k_cache.flatten(0, 2)
    v_cache = v_cache.flatten(0, 2)

    k_cache.index_copy_(0, slot_mapping, key)
    v_cache.index_copy_(0, slot_mapping, value)


def benchmark(num_blocks: int):
    key = torch.randn(BATCH_SIZE,
                      SEQ_LEN,
                      NUM_KV_HEADS,
                      HEAD_SIZE,
                      device=device,
                      dtype=DTYPE)
    value = torch.randn_like(key)
    k_cache = torch.zeros(NUM_KV_HEADS,
                          num_blocks,
                          BLOCK_SIZE,
                          HEAD_SIZE,
                          device=device,
                          dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.randint(0,
                                 num_blocks * BLOCK_SIZE,
                                 (BATCH_SIZE, SEQ_LEN),
                                 device=device,
                                 dtype=torch.int64)
    xm.mark_step()

    num_kv_heads = k_cache.shape[0]
    numel_per_head = k_cache.numel() // num_kv_heads
    slot_mapping = slot_mapping.flatten()
    head_indicies = torch.arange(0,
                                 num_kv_heads,
                                 device=slot_mapping.device,
                                 dtype=slot_mapping.dtype)
    head_indicies = head_indicies * numel_per_head
    slot_mapping = slot_mapping.flatten()
    slot_mapping = slot_mapping.repeat_interleave(num_kv_heads).view(
        -1, num_kv_heads)
    slot_mapping = slot_mapping + head_indicies.view(1, -1)
    slot_mapping = slot_mapping.flatten()

    @torch.compile(backend="openxla")
    def f():
        for _ in range(100):
            write_to_kv_cache(key, value, k_cache, v_cache, slot_mapping)

    f()
    xm.wait_device_ops()

    f()
    xm.wait_device_ops()

    start = time.time()
    f()
    xm.wait_device_ops()
    end = time.time()
    op_time = (end - start) / 100
    print(f"# Blocks: {num_blocks} Time: {op_time * 1000 * 1000:.1f} us")


benchmark(8192)

# TPUv4 results:
# Blocks: 1 Time: 161.4 us
# Blocks: 1024 Time: 3201.9 us
# Blocks: 2048 Time: 3123.7 us
# Blocks: 4096 Time: 3112.3 us
# Blocks: 8192 Time: 3110.6 us
# Blocks: 16384 Time: 3105.4 us
