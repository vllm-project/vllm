# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The multicast LM head logits all-gather against the concatenated shards."""

import pytest
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.platforms import current_platform

from .tp import TP_SIZES, run_tp

VOCAB = 154880


def _gather(rank, device):
    from vllm.models.glm5next.nvidia.specialized_decode.logits_gather import (
        LogitsGather,
    )

    tp = get_tp_group().world_size
    columns = VOCAB // tp
    op = LogitsGather(columns=columns, capacity=64, device=device)
    g = torch.Generator(device=device).manual_seed(0)

    def logits(rows):
        full = torch.randn(rows, tp * columns, generator=g, device=device)
        full = full.bfloat16()
        # Bit patterns pass through untouched.
        full[0, :4] = torch.tensor([float("nan"), float("-inf"), -0.0, 0.0])
        shard = full[:, rank * columns : (rank + 1) * columns].contiguous()
        return full, shard

    for rows in (1, 3, 64, 16, 7, 64):
        full, shard = logits(rows)
        out = op(shard)
        assert out.shape == full.shape
        assert torch.equal(out.view(torch.int16), full.view(torch.int16))
    # Two gathers in flight: the second takes the other buffer.
    (a, sa), (b, sb) = logits(64), logits(32)
    out_a, out_b = op(sa), op(sb)
    assert torch.equal(out_a.view(torch.int16), a.view(torch.int16))
    assert torch.equal(out_b.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("tp", TP_SIZES)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)
def test_logits_gather_matches_concatenated_shards(tp):
    run_tp(_gather, tp)
