# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM8x fused router (Humming expert path) against the separate routing
and alignment kernels."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(80):
    pytest.skip(
        "the fused router serves the SM8x Humming path", allow_module_level=True
    )

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.models.inkling2.nvidia.ops.moe_sm90 import (  # noqa: E402
    select_experts,
    select_experts_aligned,
)


@pytest.mark.parametrize("num_local, lo", [(64, 0), (64, 64), (128, 0)])
def test_select_experts_aligned_matches_separate_kernels(num_local: int, lo: int):
    """The one-launch router equals `select_experts` + `moe_align_block_size`
    for Inkling 2 Mini's router (both TP2 ranks' experts, and TP1), with
    padding tokens and tied scores. With 64 local experts every output is
    bitwise equal. With more, `moe_align_block_size` places rows by atomics,
    so the order of an expert's rows may differ and a row may sit in another
    block of the same expert: the expert ids per block, the padded count and
    each expert's set of rows (what its GEMM reads) are equal."""
    torch.manual_seed(0)
    experts, hidden, topk, block_size = 128, 2304, 8, 8
    weight = (torch.randn(experts, hidden, device="cuda") * 0.03).bfloat16()
    bias = torch.randn(experts, device="cuda") * 0.01
    # Ties: equal rows and biases make experts 1-3 score the same.
    weight[[2, 3]] = weight[1].clone()
    bias[[1, 2, 3]] = 1.0
    bounds = torch.tensor([lo, lo + num_local], dtype=torch.int32, device="cuda")
    counters = torch.zeros(64, dtype=torch.int32, device="cuda")
    for tokens in (1, 2, 3, 8, 9, 16, 24, 32):
        x = torch.randn(tokens, hidden, device="cuda").bfloat16()
        is_padding = torch.arange(tokens, device="cuda") >= tokens - tokens // 4
        selected = select_experts(x, weight, bias, topk, 8.0, bounds, is_padding)
        sorted_ids, expert_ids, num_padded = moe_align_block_size(
            selected[0], block_size, num_local, ignore_invalid_experts=True
        )[:3]
        fused, (f_sorted, f_expert_ids, f_num_padded) = select_experts_aligned(
            x,
            weight,
            bias,
            topk,
            8.0,
            bounds,
            num_local,
            counters,
            block_size,
            is_padding,
        )
        for ref, out in zip(selected, fused):
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
        torch.testing.assert_close(f_expert_ids, expert_ids, atol=0, rtol=0)
        torch.testing.assert_close(f_num_padded, num_padded, atol=0, rtol=0)
        if num_local <= 64:
            torch.testing.assert_close(f_sorted, sorted_ids, atol=0, rtol=0)
        else:
            rows = int(num_padded.item())
            block_experts = expert_ids[: rows // block_size]
            for e in block_experts.unique().tolist():
                in_e = (block_experts == e).repeat_interleave(block_size)
                torch.testing.assert_close(
                    f_sorted[:rows][in_e].sort().values,
                    sorted_ids[:rows][in_e].sort().values,
                    atol=0,
                    rtol=0,
                )
            torch.testing.assert_close(
                f_sorted[rows:], sorted_ids[rows:], atol=0, rtol=0
            )
        assert (counters == 0).all()
