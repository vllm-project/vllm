# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ep_scatter Triton kernels in deep_gemm_utils."""

from typing import NamedTuple

import pytest
import torch
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_scatter
from vllm.platforms import current_platform

if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "ep_scatter requires a CUDA-alike or XPU device",
        allow_module_level=True,
    )

DEVICE = current_platform.device_type

BLOCK_E = 128  # per-expert region alignment


class ScatterOut(NamedTuple):
    m_indices: torch.Tensor
    output_index: torch.Tensor
    output_tensor: torch.Tensor
    output_scale: torch.Tensor
    recv_x: torch.Tensor
    recv_scale: torch.Tensor


def _expert_start_loc(tokens_per_expert):
    """Region start per expert (exclusive cumsum of counts rounded up to BLOCK_E)."""
    starts, acc = [], 0
    for n in tokens_per_expert:
        starts.append(acc)
        acc += ((n + BLOCK_E - 1) // BLOCK_E) * BLOCK_E
    return starts, acc


def _ref_m_indices(tokens_per_expert):
    starts, total = _expert_start_loc(tokens_per_expert)
    m_indices = torch.full((total,), -1, dtype=torch.int32)
    for e, n in enumerate(tokens_per_expert):
        m_indices[starts[e] : starts[e] + n] = e
    return m_indices


def _ref_pack_ue8m0(scale_row):
    """Pack 4 uint8 scale groups into one int32 (byte j = group 4*pk+j)."""
    scale_cols = scale_row.shape[0]
    packed_size = (scale_cols + 3) // 4
    out = torch.zeros(packed_size, dtype=torch.int32)
    for pk in range(packed_size):
        val = 0
        for j in range(4):
            g = 4 * pk + j
            if g < scale_cols:
                val |= int(scale_row[g].item()) << (8 * j)
        if val >= 2**31:
            val -= 2**32
        out[pk] = val
    return out


def _counts_from_routing(recv_topk, num_experts, expert_map=None):
    counts = [0] * num_experts
    for e in recv_topk.flatten().tolist():
        mapped = e if expert_map is None else int(expert_map[e].item())
        if mapped >= 0:
            counts[mapped] += 1
    return counts


def _run_scatter(
    num_experts,
    recv_topk,
    tokens_per_expert,
    expert_map=None,
    hidden=128,
    block_size=128,
    pack_ue8m0=False,
):
    total_tokens, topk = recv_topk.shape
    _, M_aligned = _expert_start_loc(tokens_per_expert)
    M_aligned = max(M_aligned, BLOCK_E)

    scale_cols = hidden // block_size
    recv_x = torch.randn(total_tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
    output_tensor = torch.zeros(M_aligned, hidden, device=DEVICE, dtype=torch.bfloat16)

    if pack_ue8m0:
        scale_packed = (scale_cols + 3) // 4
        recv_x_scale = torch.randint(
            0, 256, (total_tokens, scale_cols), device=DEVICE, dtype=torch.uint8
        )
        output_tensor_scale = torch.zeros(
            M_aligned, scale_packed, device=DEVICE, dtype=torch.int32
        )
    else:
        recv_x_scale = (
            torch.rand(total_tokens, scale_cols, device=DEVICE, dtype=torch.float32)
            + 0.5
        )
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
        block_size=block_size,
        pack_ue8m0=pack_ue8m0,
    )
    return ScatterOut(
        m_indices.cpu(),
        output_index.cpu(),
        output_tensor.cpu(),
        output_tensor_scale.cpu(),
        recv_x.cpu(),
        recv_x_scale.cpu(),
    )


def _assert_scattered(out, recv_topk, expert_map=None, pack_ue8m0=False):
    total_tokens, topk = recv_topk.shape
    scattered_any = False
    for t in range(total_tokens):
        expected_scale = (
            _ref_pack_ue8m0(out.recv_scale[t]) if pack_ue8m0 else out.recv_scale[t]
        )
        for k in range(topk):
            expert = recv_topk[t, k].item()
            if expert_map is not None:
                expert = int(expert_map[expert].item())
            if expert < 0:
                # off-rank token: nothing written, sentinel stays
                assert out.output_index[t, k].item() == -1
                continue
            scattered_any = True
            pos = out.output_index[t, k].item()
            assert out.m_indices[pos].item() == expert
            torch.testing.assert_close(
                out.output_tensor[pos], out.recv_x[t], atol=0, rtol=0
            )
            torch.testing.assert_close(
                out.output_scale[pos], expected_scale, atol=0, rtol=0
            )
    assert scattered_any


# (num_experts, tokens_per_expert)
COUNT_CASES = [
    (4, [2, 3, 1, 2]),
    (4, [0, 4, 0, 4]),
    (4, [8, 0, 0, 0]),
    (8, [1, 2, 3, 4, 5, 6, 7, 8]),
    (4, [128, 128, 128, 128]),
    (4, [1, 1, 1, 1]),
    (16, [10] * 16),
    (2, [200, 50]),  # experts crossing the 128-token tile boundary
    (3, [130, 10, 200]),
]


@pytest.mark.parametrize(
    "num_experts,tokens_per_expert",
    COUNT_CASES,
    ids=[f"E{e}_{'_'.join(map(str, c))}"[:40] for e, c in COUNT_CASES],
)
@torch.inference_mode()
def test_ep_scatter_phase1_m_indices(num_experts, tokens_per_expert):
    total_tokens = max(sum(tokens_per_expert), 1)
    recv_topk = torch.zeros(total_tokens, 1, dtype=torch.int32)

    m_indices = _run_scatter(num_experts, recv_topk, tokens_per_expert).m_indices
    ref_m = _ref_m_indices(tokens_per_expert)

    torch.testing.assert_close(m_indices[: ref_m.shape[0]], ref_m, atol=0, rtol=0)


# (num_experts, total_tokens, topk, hidden, block_size, pack_ue8m0)
SCATTER_CASES = [
    (4, 8, 1, 128, 128, False),
    (4, 16, 2, 256, 128, False),
    (8, 32, 1, 512, 128, False),
    (8, 64, 2, 256, 128, False),
    (4, 128, 1, 128, 128, False),
    (16, 32, 1, 256, 128, False),
    (4, 8, 4, 128, 128, False),
    (4, 8, 1, 128, 128, True),  # one group per int32
    (4, 16, 2, 256, 128, True),  # partial pack (2 groups)
    (8, 32, 1, 512, 32, True),  # 32-wide MXFP8 groups
    (4, 16, 1, 256, 32, True),
]


@pytest.mark.parametrize(
    "num_experts,total_tokens,topk,hidden,block_size,pack_ue8m0",
    SCATTER_CASES,
    ids=[
        f"E{e}_T{t}_topk{k}_H{h}_B{b}{'_ue8m0' if p else ''}"
        for e, t, k, h, b, p in SCATTER_CASES
    ],
)
@torch.inference_mode()
def test_ep_scatter_phase2_scatter(
    num_experts, total_tokens, topk, hidden, block_size, pack_ue8m0
):
    torch.manual_seed(0)
    recv_topk = torch.randint(0, num_experts, (total_tokens, topk), dtype=torch.int32)
    tokens_per_expert = _counts_from_routing(recv_topk, num_experts)

    out = _run_scatter(
        num_experts,
        recv_topk,
        tokens_per_expert,
        hidden=hidden,
        block_size=block_size,
        pack_ue8m0=pack_ue8m0,
    )
    _assert_scattered(out, recv_topk, pack_ue8m0=pack_ue8m0)


def _rank_expert_map(num_global=8):
    """Odd globals map to local ids, even globals map to -1 (off-rank)."""
    expert_map = torch.full((num_global,), -1, dtype=torch.int32)
    local_globals = list(range(1, num_global, 2))
    for local, glob in enumerate(local_globals):
        expert_map[glob] = local
    return expert_map, len(local_globals)


@pytest.mark.parametrize("pack_ue8m0", [False, True], ids=["plain", "ue8m0"])
@torch.inference_mode()
def test_ep_scatter_with_expert_map(pack_ue8m0):
    num_global, total_tokens = 8, 16
    expert_map, num_local = _rank_expert_map(num_global)

    # cycle every global so half the tokens are off-rank
    recv_topk = torch.tensor(
        [[i % num_global] for i in range(total_tokens)], dtype=torch.int32
    )
    tokens_per_expert = _counts_from_routing(recv_topk, num_local, expert_map)

    out = _run_scatter(
        num_local,
        recv_topk,
        tokens_per_expert,
        expert_map,
        hidden=256 if pack_ue8m0 else 128,
        block_size=32 if pack_ue8m0 else 128,
        pack_ue8m0=pack_ue8m0,
    )
    _assert_scattered(out, recv_topk, expert_map=expert_map, pack_ue8m0=pack_ue8m0)
