# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding


def test_deepseek_mooncake_packs_and_releases_checkpoint_rows():
    published = []
    embedding = ParallelEngramEmbedding.__new__(ParallelEngramEmbedding)
    nn.Module.__init__(embedding)
    embedding.dim = 2
    embedding.block_size = 2
    embedding.mooncake_storage = SimpleNamespace(initialize=published.append)
    embedding.weight = nn.Parameter(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8).view(torch.float8_e4m3fn),
        requires_grad=False,
    )
    embedding.weight_scale_inv = nn.Parameter(
        torch.tensor([[5], [6]], dtype=torch.uint8),
        requires_grad=False,
    )

    embedding.initialize_mooncake()

    assert len(published) == 1
    assert torch.equal(
        published[0],
        torch.tensor([[1, 2, 5], [3, 4, 6]], dtype=torch.uint8),
    )
    assert embedding.weight.shape == (0, 2)
    assert embedding.weight_scale_inv.shape == (0, 1)


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_mooncake_dequantizes_cuda_byte_rows():
    values = torch.tensor(
        [[[1.0, -2.0], [0.5, 4.0]]],
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    scales = torch.full((1, 2, 1), 127, dtype=torch.uint8, device="cuda")
    packed = torch.cat((values.view(torch.uint8), scales), dim=-1)

    class FakeStorage:
        @staticmethod
        def lookup(ids, output, *, head_start):
            assert head_start == 0
            output.copy_(packed)

    embedding = ParallelEngramEmbedding.__new__(ParallelEngramEmbedding)
    nn.Module.__init__(embedding)
    embedding.dim = 2
    embedding.block_size = 2
    embedding.part_n_hash_cols = 2
    embedding.head_start = 0
    embedding.mooncake_storage = FakeStorage()
    embedding._mooncake_rows = torch.empty_like(packed)
    output = torch.empty((1, 2, 2), dtype=torch.bfloat16, device="cuda")

    embedding.lookup(torch.tensor([[0, 1]], device="cuda"), output)

    torch.testing.assert_close(output, values.to(torch.bfloat16))
