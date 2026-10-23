# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused SiLU-and-mul + per-token FP8 quant used by `TritonExperts` must match
`silu_and_mul` followed by `scaled_fp8_quant(use_per_token_if_dynamic=True)` on
every row routed to a local expert."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts.silu_mul_token_quant import (
    silu_mul_token_quant,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() and current_platform.supports_fp8()),
    reason="needs a GPU with FP8 support",
)


def _reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    act = torch.empty((x.size(0), x.size(1) // 2), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(act, x)
    return ops.scaled_fp8_quant(act, None, use_per_token_if_dynamic=True)


@pytest.mark.parametrize("tokens", [1, 7, 64, 1000])
@pytest.mark.parametrize("hidden", [256, 512, 2048])
@pytest.mark.parametrize("ep_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_mul_token_quant(tokens, hidden, ep_size, dtype):
    set_random_seed(0)
    top_k, num_experts = 8, 64
    rows = tokens * top_k
    x = torch.randn(rows, 2 * hidden, device="cuda", dtype=dtype) * 3
    x[: rows // 4] *= 1e-3  # small rows exercise the minimum scale
    topk_ids = torch.randint(
        0, num_experts, (tokens, top_k), device="cuda", dtype=torch.int32
    )
    expert_map = None
    if ep_size > 1:
        local = num_experts // ep_size
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[local : 2 * local] = torch.arange(
            local, dtype=torch.int32, device="cuda"
        )

    ref_q, ref_s = _reference(x)
    q, s = silu_mul_token_quant(
        x, topk_ids if expert_map is not None else None, expert_map
    )
    assert q.shape == ref_q.shape and q.dtype == ref_q.dtype
    assert s.shape == ref_s.shape and s.dtype == torch.float32

    keep = torch.ones(rows, dtype=torch.bool, device="cuda")
    if expert_map is not None:
        keep = expert_map[topk_ids.reshape(-1).long()] >= 0
    torch.testing.assert_close(s[keep], ref_s[keep], rtol=1e-2, atol=0)
    match = (q[keep].view(torch.uint8) == ref_q[keep].view(torch.uint8)).float().mean()
    assert match > 0.999, f"only {match:.4%} of FP8 values match"
    torch.testing.assert_close(
        q[keep].float() * s[keep],
        ref_q[keep].float() * ref_s[keep],
        rtol=0.13,
        atol=1e-6,
    )
