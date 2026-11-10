# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of the Kimi-K3 ROCm AttnRes op (amd/ops/attn_res.py).

The op used to pick its tiling (BLOCK_L, num_warps) from the number of tokens
when more than one AttnRes block is valid, so the FP32 reduction order of a
token's row depended on the batch it shared a step with.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="The Kimi-K3 AttnRes Triton op is only used on ROCm",
)

HIDDEN_SIZE = 7168  # Kimi-K3 hidden size
MAX_BLOCKS = 4
NUM_TARGETS = 5
MAX_TOKENS = 1000 + NUM_TARGETS


def _inputs(device: torch.device):
    g = torch.Generator(device=device).manual_seed(0)

    def randn(*shape, scale=1.0, shift=0.0):
        x = torch.randn(*shape, device=device, generator=g) * scale + shift
        return x.to(torch.bfloat16)

    return dict(
        prefix=randn(MAX_TOKENS, HIDDEN_SIZE),
        delta=randn(MAX_TOKENS, HIDDEN_SIZE, scale=0.3),
        # Same layout as the model's block_residual: [num_tokens, blocks, hidden].
        blocks=randn(MAX_TOKENS, MAX_BLOCKS, HIDDEN_SIZE),
        norm_weight=randn(HIDDEN_SIZE, scale=0.1, shift=1.0),
        qk_weight=randn(HIDDEN_SIZE, scale=0.05),
        output_norm_weight=randn(HIDDEN_SIZE, scale=0.1, shift=1.0),
    )


def _run(inp, rows: torch.Tensor, num_blocks: int) -> torch.Tensor:
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    return attn_res(
        inp["prefix"][rows].contiguous(),
        inp["delta"][rows].contiguous(),
        inp["blocks"][rows].contiguous(),
        inp["norm_weight"],
        inp["qk_weight"],
        inp["output_norm_weight"],
        num_blocks,
        -1,
        1e-6,
        1e-6,
    )


@pytest.mark.parametrize("num_blocks", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [8, 64, 300, 1000])
def test_attn_res_rows_independent_of_batch(num_blocks: int, batch_size: int):
    """The same rows alone and at the end of a larger batch are bitwise equal."""
    device = torch.device(current_platform.device_type)
    inp = _inputs(device)
    targets = torch.arange(NUM_TARGETS, device=device)
    alone = _run(inp, targets, num_blocks)
    rows = torch.cat(
        (torch.arange(NUM_TARGETS, batch_size + NUM_TARGETS, device=device), targets)
    )
    batched = _run(inp, rows, num_blocks)[batch_size:]
    assert torch.equal(alone, batched), (
        f"num_blocks={num_blocks} batch_size={batch_size}: max abs diff "
        f"{(alone.float() - batched.float()).abs().max().item()}"
    )


def test_attn_res_prefill_numerics_unchanged(monkeypatch: pytest.MonkeyPatch):
    """Batch-invariant mode keeps the default large-batch (prefill) tiling."""
    device = torch.device(current_platform.device_type)
    inp = _inputs(device)
    rows = torch.arange(1000, device=device)
    invariant = _run(inp, rows, 3)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    default = _run(inp, rows, 3)
    assert torch.equal(invariant, default)
