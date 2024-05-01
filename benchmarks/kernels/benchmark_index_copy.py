import os
import time
os.environ["PJRT_DEVICE"] = "TPU"

import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.custom_kernel  # Required to register custom ops.

device = xm.xla_device()

BATCH_SIZE = 1
SEQ_LEN = 128
NUM_KV_HEADS = 16
HEAD_SIZE = 256
BLOCK_SIZE = 16
DTYPE = torch.bfloat16


def benchmark(num_blocks: int):
    key = torch.randn(BATCH_SIZE * SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, device=device, dtype=DTYPE)
    k_cache = torch.randn(num_blocks * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, device=device, dtype=DTYPE)
    value = torch.randn_like(key)
    v_cache = torch.randn_like(k_cache)
    slot_mapping = torch.randint(0, num_blocks, (BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.int64)

    for _ in range(10):
        for _ in range(10):
            k_cache = k_cache.index_copy_(0, slot_mapping.flatten(), key)
            v_cache = v_cache.index_copy_(0, slot_mapping.flatten(), value)
        xm.mark_step()

    start = time.time()
    for _ in range(100):
        for _ in range(10):
            k_cache = k_cache.index_copy_(0, slot_mapping.flatten(), key)
            v_cache = v_cache.index_copy_(0, slot_mapping.flatten(), value)
        xm.mark_step()
    end = time.time()

    print(f"# Blocks: {num_blocks} Time: {(end - start) * 1000:.1f} us")


for num_blocks in [1024, 2048, 4096, 8192, 16384]:
    benchmark(num_blocks)

# TPUv4 results:
# Blocks: 1024 Time: 102.4 us
# Blocks: 2048 Time: 102.5 us
# Blocks: 4096 Time: 190.0 us
# Blocks: 8192 Time: 436.9 us
# Blocks: 16384 Time: 908.2 us
