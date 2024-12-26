import random

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_copy_subranges(seed, device):
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    num_rows = 1024
    num_cols = 1024
    src_matrix = torch.zeros(num_rows,
                             num_cols,
                             device=device,
                             dtype=torch.int32)
    dst_matrix = torch.zeros(num_rows,
                             num_cols,
                             device=device,
                             dtype=torch.int32)
    diff_matrix = torch.zeros(num_rows, 2, device=device, dtype=torch.int32)

    for i in range(num_rows):
        start_idx = random.randint(0, num_cols - 1)
        end_idx = random.randint(start_idx, num_cols - 1)
        num_diffs = end_idx - start_idx

        src_matrix[i, start_idx:end_idx] = torch.randint(0,
                                                         100, (num_diffs, ),
                                                         device=device,
                                                         dtype=torch.int32)

        diff_matrix[i, 0] = start_idx
        diff_matrix[i, 1] = num_diffs

    ops.copy_subranges(src_matrix, diff_matrix, dst_matrix, num_rows)
    assert torch.allclose(src_matrix, dst_matrix, rtol=0, atol=0)
