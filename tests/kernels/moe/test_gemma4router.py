# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.gemma4 import (
    gemma4_fused_routing_kernel_triton,
    gemma4_routing_function_torch,
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
