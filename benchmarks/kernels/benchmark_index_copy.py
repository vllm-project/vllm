import os
import time
os.environ["PJRT_DEVICE"] = "TPU"

import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.custom_kernel  # Required to register custom ops.
import torch_xla.experimental.dynamo_set_buffer_donor

device = xm.xla_device()

BATCH_SIZE = 1
SEQ_LEN = 128
NUM_KV_HEADS = 16
HEAD_SIZE = 256
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
    k_cache = k_cache.flatten(0, 1)
    key = key.flatten(0, 1)
    k_cache = k_cache.index_copy_(0, slot_mapping, key)
    v_cache = v_cache.flatten(0, 1)
    value = value.flatten(0, 1)
    v_cache = v_cache.index_copy_(0, slot_mapping, value)


def benchmark(num_blocks: int):
    key = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, device=device, dtype=DTYPE)
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, device=device, dtype=DTYPE)
    value = torch.randn_like(key)
    v_cache = torch.randn_like(k_cache)
    slot_mapping = torch.randint(0, num_blocks, (BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.int64)
    xm.mark_step()

    f = torch.compile(write_to_kv_cache, backend="openxla")
    f(key, value, k_cache, v_cache, slot_mapping.flatten())
    xm.wait_device_ops()

    for _ in range(10):
        for _ in range(10):
            f(key, value, k_cache, v_cache, slot_mapping.flatten())
    xm.wait_device_ops()

    start = time.time()
    for _ in range(100):
        for _ in range(10):
            f(key, value, k_cache, v_cache, slot_mapping.flatten())
    xm.wait_device_ops()
    end = time.time()
    op_time = (end - start) / 1000
    print(f"# Blocks: {num_blocks} Time: {op_time * 1000 * 1000:.1f} us")


for num_blocks in [1024, 2048, 4096, 8192, 16384]:
    benchmark(num_blocks)

# TPUv4 results:
# Blocks: 1024 Time: 306.2 us
# Blocks: 2048 Time: 307.0 us
# Blocks: 4096 Time: 308.7 us
# Blocks: 8192 Time: 313.9 us
# Blocks: 16384 Time: 313.6 us
