# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's apply_expert_map Triton helper.

apply_expert_map is a @triton.jit device function (not a launchable kernel): it
remaps an expert id through a lookup table, leaving -1 (unrouted) untouched. It
is exercised here through ep_scatter, which calls it to decide where each token
is written. With a permuted expert_map, a token routed to expert `e` must land
in the output region of expert `expert_map[e]`, so checking the destination
regions verifies the remapping rather than just the data copy.

Source: vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_scatter
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

BLOCK_E = 128  # per-expert region alignment used by ep_scatter
HIDDEN = 128


def apply_expert_map_ref(expert_id, expert_map):
    """Python mirror of the device function: -1 passes through, else lookup."""
    return -1 if expert_id == -1 else int(expert_map[expert_id].item())


def _run_scatter(num_experts, recv_topk, expert_map):
    """Run ep_scatter and return (output_index, m_indices, output_tensor, recv_x).

    tokens_per_expert is counted per *mapped* expert so the kernel's per-expert
    regions are large enough for the atomic writes to stay in bounds.
    """
    total_tokens, topk = recv_topk.shape
    tokens_per_expert = [0] * num_experts
    for t in range(total_tokens):
        for k in range(topk):
            mapped = apply_expert_map_ref(recv_topk[t, k].item(), expert_map)
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
    """The kernel's expert remap must match the reference for every token.

    ep_scatter writes each token to a row owned by its remapped expert, and
    m_indices records which expert owns each output row. So the expert the
    kernel resolved is m_indices[output_index[t, k]] — it must equal the
    reference expert_map[recv_topk[t, k]]. The mapping is read out of the kernel
    output (m_indices) and compared against an independent reference.
    """
    if permuted:
        expert_map = torch.arange(num_experts - 1, -1, -1, dtype=torch.int32)
    else:
        expert_map = torch.arange(num_experts, dtype=torch.int32)

    torch.manual_seed(0)
    recv_topk = torch.randint(0, num_experts, (total_tokens, topk), dtype=torch.int32)
    output_index, m_indices, output_tensor, recv_x = _run_scatter(
        num_experts, recv_topk, expert_map
    )

    for t in range(total_tokens):
        for k in range(topk):
            idx = output_index[t, k].item()
            kernel_expert = m_indices[idx].item()
            ref_expert = apply_expert_map_ref(recv_topk[t, k].item(), expert_map)
            assert kernel_expert == ref_expert, (
                f"token {t} slot {k}: kernel mapped to expert {kernel_expert}, "
                f"reference {ref_expert}"
            )
            # The scattered row must also equal the source row.
            torch.testing.assert_close(
                output_tensor[idx], recv_x[t], atol=0, rtol=0
            )
