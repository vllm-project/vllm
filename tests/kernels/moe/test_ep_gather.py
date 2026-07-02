# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's ep_gather Triton kernel.

ep_gather is the inverse of ep_scatter: it reorders expert-sorted activations
back to token order, applying per-topk weights and accumulating:

    output[t] = sum_k  weight[t, k] * input[input_index[t, k]]

(slots whose expert id maps to -1 via expert_map are skipped). Compared against
a float32 PyTorch reference.

Source: vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_gather
from vllm.platforms import current_platform

DEVICE = current_platform.device_type


def ep_gather_ref(input_tensor, recv_topk_ids, recv_topk_weight, input_index,
                  expert_map=None):
    """Weighted gather reference (float32, CPU): expert-sorted -> token order.

    output[t] = sum_k weight[t, k] * input[input_index[t, k]]
    skipping slots whose expert maps to -1.
    """
    total_tokens, topk = recv_topk_ids.shape
    hidden_size = input_tensor.shape[1]
    output = torch.zeros(total_tokens, hidden_size, dtype=torch.float32)

    for t in range(total_tokens):
        for k_idx in range(topk):
            eid = recv_topk_ids[t, k_idx].item()
            if expert_map is not None and expert_map[eid].item() < 0:
                continue
            idx = input_index[t, k_idx].item()
            w = recv_topk_weight[t, k_idx].item()
            output[t] += w * input_tensor[idx].float()

    return output


def _run_gather(input_tensor, recv_topk_ids, recv_topk_weight, input_index,
                expert_map=None):
    total_tokens = recv_topk_ids.shape[0]
    hidden_size = input_tensor.shape[1]
    output_tensor = torch.zeros(
        total_tokens, hidden_size, device=DEVICE, dtype=input_tensor.dtype
    )
    ep_gather(
        input_tensor, recv_topk_ids, recv_topk_weight,
        input_index, expert_map, output_tensor,
    )
    return output_tensor


# (num_experts, total_tokens, topk, hidden_size, num_src)
CASES = [
    (4, 8, 1, 128, 512),
    (4, 16, 2, 256, 512),
    (8, 32, 1, 512, 1024),
    (4, 64, 1, 128, 512),
    (8, 32, 2, 256, 1024),
    (4, 128, 1, 1024, 512),
    (16, 64, 1, 256, 2048),
    (4, 16, 4, 128, 512),
]


def _build_expert_map(num_experts):
    """Map even experts to local ids, odd experts to -1 (dropped slots)."""
    expert_map = torch.full((num_experts,), -1, device=DEVICE, dtype=torch.int32)
    local = 0
    for glob in range(0, num_experts, 2):
        expert_map[glob] = local
        local += 1
    return expert_map


@pytest.mark.parametrize("use_expert_map", [False, True], ids=["no_map", "map"])
@pytest.mark.parametrize(
    "num_experts,total_tokens,topk,hidden_size,num_src",
    CASES,
    ids=[f"E{e}_T{t}_topk{k}_H{h}" for e, t, k, h, s in CASES],
)
@torch.inference_mode()
def test_ep_gather(num_experts, total_tokens, topk, hidden_size, num_src,
                   use_expert_map):
    """ep_gather must match the weighted-sum reference, with and without a map.

    Both expert_map paths matter: the kernel branches on HAS_EXPERT_MAP as a
    constexpr, so they compile to distinct kernels.
    """
    torch.manual_seed(0)
    expert_map = _build_expert_map(num_experts) if use_expert_map else None

    input_tensor = torch.randn(
        num_src, hidden_size, device=DEVICE, dtype=torch.bfloat16
    )
    recv_topk_ids = torch.randint(
        0, num_experts, (total_tokens, topk), device=DEVICE, dtype=torch.int32
    )
    recv_topk_weight = torch.rand(
        total_tokens, topk, device=DEVICE, dtype=torch.float32
    )
    recv_topk_weight = recv_topk_weight / recv_topk_weight.sum(dim=1, keepdim=True)
    input_index = torch.randint(
        0, num_src, (total_tokens, topk), device=DEVICE, dtype=torch.int32
    )

    out = _run_gather(
        input_tensor, recv_topk_ids, recv_topk_weight, input_index, expert_map
    )
    ref = ep_gather_ref(
        input_tensor.cpu(), recv_topk_ids.cpu(), recv_topk_weight.cpu(),
        input_index.cpu(), None if expert_map is None else expert_map.cpu(),
    )

    torch.testing.assert_close(out.cpu().float(), ref, atol=5e-2, rtol=5e-2)


@torch.inference_mode()
def test_ep_gather_topk1_unit_weight():
    """topk=1 with weight=1.0: each output row equals its gathered input row."""
    torch.manual_seed(0)
    total_tokens, hidden_size, num_src = 32, 256, 512

    input_tensor = torch.randn(
        num_src, hidden_size, device=DEVICE, dtype=torch.bfloat16
    )
    recv_topk_ids = torch.zeros(total_tokens, 1, device=DEVICE, dtype=torch.int32)
    recv_topk_weight = torch.ones(total_tokens, 1, device=DEVICE, dtype=torch.float32)
    input_index = torch.randint(
        0, num_src, (total_tokens, 1), device=DEVICE, dtype=torch.int32
    )

    out = _run_gather(input_tensor, recv_topk_ids, recv_topk_weight, input_index).cpu()

    for t in range(total_tokens):
        idx = input_index[t, 0].item()
        torch.testing.assert_close(
            out[t].float(), input_tensor[idx].cpu().float(), atol=1e-3, rtol=1e-3
        )
