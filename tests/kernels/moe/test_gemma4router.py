# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn
from torch.func import functional_call

from vllm.model_executor.models.gemma4 import (
    gemma4_fused_routing_kernel_triton,
    gemma4_routing_function_torch,
)


class _MinimalGemma4MoE(nn.Module):
    """Minimal module that reproduces the Gemma4MoE routing closure pattern."""

    def __init__(self, n_experts: int) -> None:
        super().__init__()
        self.per_expert_scale = nn.Parameter(torch.ones(n_experts))

        def routing_function(
            gating: torch.Tensor, topk: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return gemma4_routing_function_torch(gating, topk, self.per_expert_scale)

        self._route = routing_function

    def forward(
        self, gating: torch.Tensor, topk: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._route(gating, topk)


def test_gemma4_moe_routing_functional_call_substitution():
    """Verify that routing_function reads per_expert_scale from module state
    at call time, so torch.func.functional_call substitution is respected.

    Regression test for: https://github.com/vllm-project/vllm/issues/42239
    The old code captured `per_expert_scale` into a closure local variable,
    which prevented functional_call from reaching the routing function.
    """
    torch.manual_seed(0)
    n_experts, topk = 8, 2
    gating = torch.randn(4, n_experts, dtype=torch.float32)
    new_scale = torch.rand(n_experts, dtype=torch.float32) + 0.5

    m = _MinimalGemma4MoE(n_experts)

    # Baseline with the default scale (all-ones)
    orig_w, _orig_ids = m(gating, topk)

    # After functional_call substitution, routing must use new_scale
    sub_w, sub_ids = functional_call(m, {"per_expert_scale": new_scale}, (gating, topk))

    # Reference: same function called directly with new_scale
    ref_w, ref_ids = gemma4_routing_function_torch(gating, topk, new_scale)

    assert torch.allclose(sub_w, ref_w, atol=1e-5), (
        "functional_call substitution did not reach routing_function — "
        "per_expert_scale is likely still captured in a closure local"
    )
    assert (sub_ids == ref_ids).all(), (
        "functional_call substitution produced wrong expert ids"
    )
    # Sanity: different scales must produce different weights
    assert not torch.allclose(orig_w, sub_w), (
        "Expected different routing weights with different per_expert_scale"
    )


def sort_by_id(w, ids):
    order = ids.argsort(dim=-1)
    return w.gather(1, order), ids.gather(1, order)


# Gemma4 MoE Model has context length of 250K
# the minus 1 is to ensure that edge cases are tested
@pytest.mark.parametrize("num_tokens", [1, 2, 2048, 250000])
@pytest.mark.parametrize("num_experts", [128])  # gemma4 moe experts
@pytest.mark.parametrize("topk", [8])  # gemma4 topk
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_gemma4_routing_kernel_triton(
    num_tokens: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
):
    torch.manual_seed(0)

    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device="cuda")
    scales = torch.rand(num_experts, dtype=torch.float32, device="cuda")

    ref_w, ref_ids = gemma4_routing_function_torch(gating, topk, scales)
    tri_w, tri_ids = gemma4_fused_routing_kernel_triton(gating, topk, scales)

    # Sort by expert id — to remove tie-breaking differences
    ref_ws, ref_is = sort_by_id(ref_w, ref_ids)
    tri_ws, tri_is = sort_by_id(tri_w, tri_ids)

    ids_match = (ref_is == tri_is).all().item()
    weights_match = torch.allclose(ref_ws, tri_ws, atol=1e-2, rtol=1e-2)
    all_match = ids_match and weights_match
    max_err = (ref_ws - tri_ws).abs().max().item()
    print(
        f"T={num_tokens:5d} E={num_experts:4d} K={topk} "
        f"{str(dtype).split('.')[-1]:7s} ids={ids_match} max_Δweight={max_err:.2e}"
    )
    if not all_match:
        bad = (ref_is != tri_is).any(dim=-1).nonzero(as_tuple=True)[0]
        if len(bad):
            r = bad[0].item()
            print(
                f"  first bad row {r}: ref_ids={ref_ids[r].tolist()} "
                f"tri_ids={tri_ids[r].tolist()}"
            )
        assert all_match
