# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Hopper LM head shard GEMM against an FP64 reference and cuBLAS, and the
NVLink logits all-gather against the concatenated shards."""

import pytest
import torch

from .test_boundary import requires_sm90, run_tp

HIDDEN = 4096
VOCAB = 154880  # GLM-5.3-Flash


def _gather(rank, device):
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.logits import (
        HopperLogitsGather,
    )

    tp = get_tp_group().world_size
    columns = VOCAB // tp
    op = HopperLogitsGather(columns=columns, capacity=32, device=device)
    g = torch.Generator(device=device).manual_seed(0)

    def logits(rows):
        full = torch.randn(rows, tp * columns, generator=g, device=device)
        full = full.bfloat16()
        # Bit patterns pass through untouched.
        full[0, :4] = torch.tensor([float("nan"), float("-inf"), -0.0, 0.0])
        shard = full[:, rank * columns : (rank + 1) * columns].contiguous()
        return full, shard

    for rows in (1, 6, 32, 3, 24, 12, 18, 2, 4, 8):
        full, shard = logits(rows)
        out = op(shard)
        assert out.shape == full.shape
        assert torch.equal(out.view(torch.int16), full.view(torch.int16))
    # Two gathers in flight: the second takes the other buffer.
    (a, sa), (b, sb) = logits(32), logits(6)
    out_a, out_b = op(sa), op(sb)
    assert torch.equal(out_a.view(torch.int16), a.view(torch.int16))
    assert torch.equal(out_b.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("tp", [2, 4])
@requires_sm90
def test_logits_gather_matches_concatenated_shards(tp):
    run_tp(tp, _gather)


@pytest.mark.parametrize("tp", [2, 4])
@requires_sm90
def test_logits_head_matches_reference(tp):
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.logits import (
        SKINNY,
        HopperLogitsHead,
    )

    shard = VOCAB // tp
    weight = (torch.randn(shard, HIDDEN, device="cuda") * 0.02).bfloat16()
    rows = sorted(SKINNY.get((shard, HIDDEN), {}))
    if not rows:
        pytest.skip(f"No skinny head tactic for a {shard}-row shard")
    head = HopperLogitsHead(weight, rows)
    for m in rows:
        assert head.supports(m)
        x = torch.randn(m, HIDDEN, device="cuda").bfloat16()
        ref = x.double() @ weight.double().t()
        out = head(x)
        cublas = torch.nn.functional.linear(x, weight)
        assert out.shape == cublas.shape and out.dtype == cublas.dtype
        error = (out.double() - ref).norm() / ref.norm()
        assert error <= 1.25 * (cublas.double() - ref).norm() / ref.norm() + 1e-6
