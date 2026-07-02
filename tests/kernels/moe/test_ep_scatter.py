# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's ep_scatter Triton kernels.

ep_scatter runs two Triton kernels back to back:

  * _fwd_kernel_ep_scatter_1: from per-expert token counts it computes the
    per-expert region offsets and fills m_indices, labelling each output row with
    the expert that owns it (-1 for padding), with counts aligned to BLOCK_E=128.
  * _fwd_kernel_ep_scatter_2: copies each token's activation/scale into the
    row of its (optionally remapped) expert, recording the destination in
    output_index.

Both are validated against a float32 PyTorch reference.

Source: vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_scatter
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

BLOCK_E = 128  # per-expert region alignment used by ep_scatter


def _expert_start_loc(tokens_per_expert):
    """Region start per expert: exclusive cumsum of round_up_128(count)."""
    starts, acc = [], 0
    for n in tokens_per_expert:
        starts.append(acc)
        acc += ((n + BLOCK_E - 1) // BLOCK_E) * BLOCK_E
    return starts, acc  # (starts, total aligned size)


def _ref_m_indices(tokens_per_expert):
    """Phase-1 reference: m_indices labels each expert region, -1 for padding."""
    starts, total = _expert_start_loc(tokens_per_expert)
    m_indices = torch.full((total,), -1, dtype=torch.int32)
    for e, n in enumerate(tokens_per_expert):
        m_indices[starts[e] : starts[e] + n] = e
    return m_indices


def _counts_from_routing(recv_topk, num_experts, expert_map=None):
    """Tokens per (mapped) expert, matching how ep_scatter sizes its regions."""
    counts = [0] * num_experts
    for e in recv_topk.flatten().tolist():
        mapped = e if expert_map is None else int(expert_map[e].item())
        if mapped >= 0:
            counts[mapped] += 1
    return counts


def _run_scatter(num_experts, recv_topk, tokens_per_expert, expert_map=None, hidden=128):
    """Launch ep_scatter; return CPU (m_indices, output_index, output_tensor, recv_x)."""
    total_tokens, topk = recv_topk.shape
    _, M_aligned = _expert_start_loc(tokens_per_expert)
    M_aligned = max(M_aligned, BLOCK_E)

    # Scale has one column per 128-wide quant group, matching the kernel.
    scale_cols = hidden // BLOCK_E
    # Distinct per-token scales so the scatter of scales is actually checked.
    recv_x = torch.randn(total_tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
    recv_x_scale = (
        torch.rand(total_tokens, scale_cols, device=DEVICE, dtype=torch.float32) + 0.5
    )
    output_tensor = torch.zeros(M_aligned, hidden, device=DEVICE, dtype=torch.bfloat16)
    output_tensor_scale = torch.zeros(
        M_aligned, scale_cols, device=DEVICE, dtype=torch.float32
    )
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
    return (
        m_indices.cpu(),
        output_index.cpu(),
        output_tensor.cpu(),
        output_tensor_scale.cpu(),
        recv_x.cpu(),
        recv_x_scale.cpu(),
    )


# (num_experts, tokens_per_expert) — drives the phase-1 region layout.
COUNT_CASES = [
    (4, [2, 3, 1, 2]),
    (4, [0, 4, 0, 4]),
    (4, [8, 0, 0, 0]),
    (8, [1, 2, 3, 4, 5, 6, 7, 8]),
    (4, [128, 128, 128, 128]),
    (4, [1, 1, 1, 1]),
    (16, [10] * 16),
]


@pytest.mark.parametrize(
    "num_experts,tokens_per_expert",
    COUNT_CASES,
    ids=[f"E{e}_{'_'.join(map(str, c))}"[:40] for e, c in COUNT_CASES],
)
@torch.inference_mode()
def test_ep_scatter_phase1_m_indices(num_experts, tokens_per_expert):
    """Phase 1: m_indices must label each expert region and leave padding at -1."""
    total_tokens = max(sum(tokens_per_expert), 1)
    # Routing is irrelevant for phase-1 layout; identity-ish placeholder.
    recv_topk = torch.zeros(total_tokens, 1, dtype=torch.int32)

    m_indices = _run_scatter(num_experts, recv_topk, tokens_per_expert)[0]
    ref_m = _ref_m_indices(tokens_per_expert)

    torch.testing.assert_close(m_indices[: ref_m.shape[0]], ref_m, atol=0, rtol=0)


# (num_experts, total_tokens, topk, hidden)
SCATTER_CASES = [
    (4, 8, 1, 128),
    (4, 16, 2, 256),
    (8, 32, 1, 512),
    (8, 64, 2, 256),
    (4, 128, 1, 128),
    (16, 32, 1, 256),
    (4, 8, 4, 128),
]


@pytest.mark.parametrize(
    "num_experts,total_tokens,topk,hidden",
    SCATTER_CASES,
    ids=[f"E{e}_T{t}_topk{k}_H{h}" for e, t, k, h in SCATTER_CASES],
)
@torch.inference_mode()
def test_ep_scatter_phase2_scatter(num_experts, total_tokens, topk, hidden):
    """Phase 2: each token is copied into a row owned by its expert."""
    torch.manual_seed(0)
    recv_topk = torch.randint(0, num_experts, (total_tokens, topk), dtype=torch.int32)
    tokens_per_expert = _counts_from_routing(recv_topk, num_experts)

    m_indices, output_index, output_tensor, output_scale, recv_x, recv_scale = (
        _run_scatter(num_experts, recv_topk, tokens_per_expert, hidden=hidden)
    )

    for t in range(total_tokens):
        for k in range(topk):
            pos = output_index[t, k].item()
            # Destination row belongs to the routed expert (phase-1 label)...
            assert m_indices[pos].item() == recv_topk[t, k].item()
            # ...and holds this token's activation and scale (phase-2 copy).
            torch.testing.assert_close(output_tensor[pos], recv_x[t], atol=0, rtol=0)
            torch.testing.assert_close(output_scale[pos], recv_scale[t], atol=0, rtol=0)


@torch.inference_mode()
def test_ep_scatter_with_expert_map():
    """Phase 2 + remap: tokens land in the region of their mapped local expert."""
    num_global, num_local, total_tokens = 8, 4, 16
    # Global expert -> local id (-1 if not on this rank).
    expert_map = torch.full((num_global,), -1, dtype=torch.int32)
    for local, glob in enumerate([1, 3, 5, 7]):
        expert_map[glob] = local

    valid = [1, 3, 5, 7]
    recv_topk = torch.tensor(
        [[valid[i % 4]] for i in range(total_tokens)], dtype=torch.int32
    )
    tokens_per_expert = _counts_from_routing(recv_topk, num_local, expert_map)

    m_indices, output_index, output_tensor, output_scale, recv_x, recv_scale = (
        _run_scatter(num_local, recv_topk, tokens_per_expert, expert_map)
    )

    for t in range(total_tokens):
        pos = output_index[t, 0].item()
        mapped = int(expert_map[recv_topk[t, 0].item()].item())
        assert m_indices[pos].item() == mapped
        torch.testing.assert_close(output_tensor[pos], recv_x[t], atol=0, rtol=0)
        torch.testing.assert_close(output_scale[pos], recv_scale[t], atol=0, rtol=0)
