import random

import pytest
import torch

from vllm.model_executor.layers.ops.rand import seeded_uniform
from vllm.model_executor.utils import set_random_seed


@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_3d", [True, False])
def test_seeded_uniform(dtype: torch.dtype, use_3d: bool):
    device = "cuda"
    for seed in range(512):
        set_random_seed(seed)
        rows = random.randint(1, 512)
        cols = random.randint(1, 64000)
        if use_3d:
            third_dim = random.randint(2, 10)
            dims = [rows, third_dim, cols]
        else:
            dims = [rows, cols]
        seeds = torch.randint(torch.iinfo(torch.long).min,
                              torch.iinfo(torch.long).max, (rows, ),
                              device=device)

        # Test that the same seed produces the same output
        out = seeded_uniform(*dims, seeds=seeds, dtype=dtype, device=device)
        out2 = seeded_uniform(*dims, seeds=seeds, dtype=dtype, device=device)
        torch.testing.assert_close(out, out2)
        # del to save memory
        del out2

        out3 = seeded_uniform(*dims, seeds=seeds, dtype=dtype, device=device)
        torch.testing.assert_close(out, out3)
        # del to save memory
        del out3

        # Initialize out tensor with garbage to ensure that it is overwritten
        out_with_tensor = seeded_uniform(
            *dims,
            out=torch.full(
                (*dims, ),
                -1,
                dtype=dtype,
                device=device,
            ),
            seeds=seeds,
            dtype=dtype,
        )
        torch.testing.assert_close(out, out_with_tensor)
