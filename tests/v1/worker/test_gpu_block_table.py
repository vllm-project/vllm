import pytest
import random
import time

import torch

from vllm.v1.worker.gpu_block_table import BlockTable

MAX_NUM_REQS = 1024
MAX_MODEL_LEN = 128 * 1024
BLOCK_SIZE = 16
MAX_NUM_BLOCKS_PER_REQ = MAX_MODEL_LEN // BLOCK_SIZE


def test_block_table(do_wait: bool):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    block_table = BlockTable(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_blocks_per_req=MAX_NUM_BLOCKS_PER_REQ,
        pin_memory=True,
        device=torch.device(0),
    )

    num_blocks = random.randint(1, MAX_NUM_BLOCKS_PER_REQ - 1)
    block_ids = torch.randint(0, MAX_NUM_BLOCKS_PER_REQ, (num_blocks,), dtype=torch.int32, device="cpu")
    block_table.add_row(0, block_ids)
    num_blocks = random.randint(1, MAX_NUM_BLOCKS_PER_REQ - 100)
    block_ids = torch.randint(0, MAX_NUM_BLOCKS_PER_REQ, (num_blocks,), dtype=torch.int32, device="cpu")
    block_table.add_row(1, block_ids)
    block_table.commit(2)

    torch.cuda.synchronize()
    if do_wait:
        time.sleep(1)

    block_ids = torch.randint(0, MAX_NUM_BLOCKS_PER_REQ, (100,), dtype=torch.int32, device="cpu")
    block_table.append_row(1, num_blocks, block_ids)
    block_table.move_row(1, 0)
    block_table.commit(2)

    torch.cuda.synchronize()
    if do_wait:
        time.sleep(1)

    torch.testing.assert_close(block_table.block_table[:1].cpu(), block_table.block_table_cpu[:1])

if __name__ == "__main__":
    test_block_table(do_wait=False)
