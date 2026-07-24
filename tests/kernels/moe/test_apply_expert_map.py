# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's apply_expert_map Triton helper.

apply_expert_map is a @triton.jit device function (not a launchable kernel): it
remaps an expert id through a lookup table, leaving -1 (unrouted) untouched. It
is exercised here through ep_scatter, which calls it to decide where each token
is written. With a permuted expert_map, a token routed to expert `e` must land
in the output region of expert `expert_map[e]`, so checking the destination
regions verifies the remapping rather than just the data copy.

A token is *dropped* by the kernel whenever apply_expert_map resolves to -1 —
either because the routing entry was already -1 (unrouted) or because the map
sends that expert to -1 (expert not local to this rank). Both drop paths are
covered by test_apply_expert_map_unrouted.

Source: vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_scatter
from vllm.platforms import current_platform

# ep_scatter dispatches Triton kernels that require a GPU-class backend.
if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "ep_scatter Triton kernels require a CUDA-alike or XPU device",
        allow_module_level=True,
    )

DEVICE = current_platform.device_type

BLOCK_E = 128  # per-expert region alignment used by ep_scatter
HIDDEN = 128


def apply_expert_map_ref(expert_id, expert_map):
    """Python mirror of the device function: -1 passes through, else lookup."""
    return -1 if expert_id == -1 else int(expert_map[expert_id].item())


def _run_scatter(num_experts, recv_topk, expert_map):
    """Run ep_scatter and return (output_index, m_indices, output_tensor, recv_x).

    tokens_per_expert is counted per *mapped* expert so the kernel's per-expert
    regions are large enough for the atomic writes to stay in bounds. Tokens
    that map to -1 (unrouted or pruned expert) are dropped by the kernel and so
    contribute to no expert's count.
    """
    total_tokens, topk = recv_topk.shape
    tokens_per_expert = [0] * num_experts
    for t in range(total_tokens):
        for k in range(topk):
            mapped = apply_expert_map_ref(recv_topk[t, k].item(), expert_map)
            if mapped != -1:
                tokens_per_expert[mapped] += 1

    M_aligned = num_experts * BLOCK_E
    recv_x = torch.randn(total_tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    recv_x_scale = torch.ones(total_tokens, 1, device=DEVICE, dtype=torch.float32)
    output_tensor = torch.zeros(M_aligned, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    output_tensor_scale = torch.zeros(M_aligned, 1, device=DEVICE, dtype=torch.float32)
    # m_indices labels each output row with the expert whose region owns it
    # (written by ep_scatter's first kernel); -1 marks padding rows.
    m_indices = torch.full((M_aligned,), -1, device=DEVICE, dtype=torch.int32)
    output_index = torch.full(
        (total_tokens, topk), -1, device=DEVICE, dtype=torch.int32
    )

    ep_scatter(
        recv_x,
        recv_x_scale,
        recv_topk.to(DEVICE),
        torch.tensor(tokens_per_expert, device=DEVICE, dtype=torch.int32),
        None if expert_map is None else expert_map.to(DEVICE),
        torch.zeros(num_experts, device=DEVICE, dtype=torch.int32),  # expert_start_loc
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_index,
    )
    return output_index.cpu(), m_indices.cpu(), output_tensor.cpu(), recv_x.cpu()


def _check(recv_topk, expert_map, output_index, m_indices, output_tensor, recv_x):
    """Assert the kernel's per-token remap matches the reference.

    For a routed token the resolved expert is m_indices[output_index[t, k]] and
    must equal expert_map[recv_topk[t, k]]; the scattered row must also equal the
    source row. For a token that maps to -1 the kernel writes nothing, so
    output_index[t, k] must stay at its -1 sentinel.
    """
    total_tokens, topk = recv_topk.shape
    for t in range(total_tokens):
        for k in range(topk):
            ref_expert = apply_expert_map_ref(recv_topk[t, k].item(), expert_map)
            idx = output_index[t, k].item()
            if ref_expert == -1:
                assert idx == -1, (
                    f"token {t} slot {k}: unrouted token was scattered to row {idx}"
                )
                continue
            kernel_expert = m_indices[idx].item()
            assert kernel_expert == ref_expert, (
                f"token {t} slot {k}: kernel mapped to expert {kernel_expert}, "
                f"reference {ref_expert}"
            )
            # The scattered row must also equal the source row.
            torch.testing.assert_close(output_tensor[idx], recv_x[t], atol=0, rtol=0)


# (num_experts, total_tokens, topk)
SHAPES = [
    (4, 8, 1),
    (4, 8, 2),
    (8, 16, 2),
    (4, 32, 2),
    (8, 8, 4),
    (16, 32, 2),
]


@pytest.mark.parametrize("permuted", [False, True], ids=["identity", "permuted"])
@pytest.mark.parametrize(
    "num_experts,total_tokens,topk", SHAPES,
    ids=[f"E{e}_T{t}_topk{k}" for e, t, k in SHAPES],
)
@torch.inference_mode()
def test_apply_expert_map_via_scatter(num_experts, total_tokens, topk, permuted):
    """The kernel's expert remap must match the reference for every routed token."""
    if permuted:
        expert_map = torch.arange(num_experts - 1, -1, -1, dtype=torch.int32)
    else:
        expert_map = torch.arange(num_experts, dtype=torch.int32)

    torch.manual_seed(0)
    recv_topk = torch.randint(0, num_experts, (total_tokens, topk), dtype=torch.int32)
    output_index, m_indices, output_tensor, recv_x = _run_scatter(
        num_experts, recv_topk, expert_map
    )
    _check(recv_topk, expert_map, output_index, m_indices, output_tensor, recv_x)


@pytest.mark.parametrize(
    "drop", ["routing", "map"], ids=["unrouted_topk", "pruned_expert"]
)
@torch.inference_mode()
def test_apply_expert_map_unrouted(drop):
    """Tokens that resolve to -1 must be dropped, routed tokens still land.

    Two independent ways a token reaches the -1 (drop) branch of the helper:
      * ``routing`` — the routing entry in recv_topk is already -1 (the token was
        never assigned a top-k expert).
      * ``map``     — the routing entry is a valid expert, but expert_map sends it
        to -1 because that expert is not local to this rank.
    """
    num_experts, total_tokens, topk = 8, 16, 2

    torch.manual_seed(0)
    recv_topk = torch.randint(0, num_experts, (total_tokens, topk), dtype=torch.int32)

    if drop == "routing":
        # An identity map keeps routed experts in place; inject -1 routing entries.
        expert_map = torch.arange(num_experts, dtype=torch.int32)
        recv_topk[0, 0] = -1
        recv_topk[5, 1] = -1
        recv_topk[total_tokens - 1, :] = -1
    else:
        # Valid routing, but prune experts 2 and 5 out of the local rank.
        expert_map = torch.arange(num_experts, dtype=torch.int32)
        expert_map[2] = -1
        expert_map[5] = -1
        # Guarantee at least one token routes to a pruned expert.
        recv_topk[0, 0] = 2
        recv_topk[3, 1] = 5

    # The scenario is only meaningful if it actually exercises the drop branch.
    assert any(
        apply_expert_map_ref(recv_topk[t, k].item(), expert_map) == -1
        for t in range(total_tokens)
        for k in range(topk)
    ), "test setup failed to produce an unrouted token"

    output_index, m_indices, output_tensor, recv_x = _run_scatter(
        num_experts, recv_topk, expert_map
    )
    _check(recv_topk, expert_map, output_index, m_indices, output_tensor, recv_x)
