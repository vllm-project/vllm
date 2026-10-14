# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA DCP combine: copy-free LSE correction and fused output gate."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa_dcp
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

requires_cuda_triton = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA DCP combine kernels need CUDA and Triton",
)


def test_other_combines_are_called_with_base2_lse():
    """A combine other than cp_lse_ag_out_rs (for example the all-to-all
    one) gets the partial output and LSE unchanged, in base 2."""
    partial = torch.randn(5, 12, 16)
    lse = torch.randn(5, 12)
    calls = []

    def combine(out, out_lse, group, *, is_lse_base_on_e):
        calls.append((out, out_lse, group, is_lse_base_on_e))
        return out[:, :3]

    merged = qsa_dcp.qsa_dcp_combine(partial, lse, "group", combine, None)

    assert len(calls) == 1
    out, out_lse, group, is_lse_base_on_e = calls[0]
    assert out is partial and out_lse is lse and group == "group"
    assert is_lse_base_on_e is False
    assert torch.equal(merged, partial[:, :3])


@requires_cuda_triton
@pytest.mark.parametrize("rows", [80, 1024])
@pytest.mark.parametrize("rank", [0, 3])
def test_dcp_heads_first_correction_matches_upstream(rows, rank):
    """The head-major LSE correction writes the bits of upstream
    correct_attn_out (base 2), also for rows empty on one or all ranks."""
    from vllm.v1.attention.ops.dcp import CPTritonContext, correct_attn_out

    world, heads, head_dim = 4, 12, 256
    generator = torch.Generator().manual_seed(rows + rank)
    partial = torch.randn(rows, heads, head_dim, generator=generator).bfloat16()
    lses = torch.randn(world, rows, heads, generator=generator) * 4
    lses[:, ::11] = float("-inf")
    lses[rank, 1::7] = float("-inf")
    partial, lses = partial.cuda(), lses.cuda()
    expected, _ = correct_attn_out(
        partial.clone(), lses, rank, CPTritonContext(), is_lse_base_on_e=False
    )
    heads_first = torch.empty_like(partial).view(heads, rows, head_dim)

    qsa_dcp._correct_qsa_dcp_heads_first_kernel[(rows, heads)](
        partial,
        lses,
        heads_first,
        rows,
        rank,
        NUM_HEADS=heads,
        HEAD_DIM=head_dim,
        DCP_WORLD=world,
        num_warps=4,
    )

    assert torch.equal(heads_first.transpose(0, 1), expected)


@requires_cuda_triton
@pytest.mark.parametrize("rows", [80, 1024])
@pytest.mark.parametrize("heads_first", [True, False])
def test_dcp_fused_gate_matches_torch_gate(rows, heads_first):
    """The fused gate equals combined * sigmoid(gate) in fp32, for the
    head-major combine output and a row-major one."""
    heads, head_dim = 3, 256
    generator = torch.Generator().manual_seed(rows + heads_first)
    combined = torch.randn(heads, rows, head_dim, generator=generator) * 3
    combined = combined.bfloat16().cuda().transpose(0, 1)
    if not heads_first:
        combined = combined.contiguous()
    gate = torch.randn(rows, heads * head_dim, generator=generator) * 6
    gate = gate.bfloat16().cuda()
    out = torch.empty(rows, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    expected = torch.empty_like(out)
    expected.copy_(combined.float() * torch.sigmoid(gate.view_as(out).float()))

    qsa_dcp.gate_qsa_dcp_output(combined, gate, out)

    torch.testing.assert_close(out, expected, rtol=2**-7, atol=1e-6)


class _FakeDCPGroup:
    """One rank's view of W simulated ranks in one process.

    all_gather returns the dim-0 concatenation of every rank's LSE.
    reduce_scatter records its input and dim and returns the input; the
    test applies the sum and the split after all ranks have run.
    """

    def __init__(self, rank, lses, scattered):
        self.rank_in_group = rank
        self.world_size = len(lses)
        self.lses = lses
        self.scattered = scattered

    def all_gather(self, tensor, dim):
        assert dim == 0 and torch.equal(tensor, self.lses[self.rank_in_group])
        return torch.cat(self.lses, dim=0)

    def reduce_scatter(self, tensor, dim):
        self.scattered.append((tensor.clone(), dim))
        return tensor


def _reduce_scatter(scattered, rank):
    """The rank's chunk of the sum of every rank's input, on the recorded dim.

    The sum runs in float64, so it is exact and independent of the layout.
    """
    (dim,) = {dim for _, dim in scattered}
    total = torch.stack([tensor.double() for tensor, _ in scattered]).sum(dim=0)
    return total.chunk(len(scattered), dim=dim)[rank]


@requires_cuda_triton
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("rows", [5, 33])
def test_copy_free_combine_matches_upstream_cp_lse_ag_out_rs(world, rows):
    """qsa_dcp_lse_ag_out_rs gives every simulated rank the bits of upstream
    cp_lse_ag_out_rs (base 2): the gathered LSE is read as [W, rows, H], the
    head-major buffer is upstream's corrected output with heads first, and
    the dim-0 reduce-scatter gives rank r heads [r * H / W, (r + 1) * H / W).
    """
    from vllm.v1.attention.ops.dcp import cp_lse_ag_out_rs

    heads, head_dim = 12, 64
    generator = torch.Generator().manual_seed(world * 100 + rows)
    partials = [
        torch.randn(rows, heads, head_dim, generator=generator).bfloat16().cuda()
        for _ in range(world)
    ]
    lses = [
        (torch.randn(rows, heads, generator=generator) * 4).cuda() for _ in range(world)
    ]
    lses[0][1] = float("-inf")  # an empty owner on rank 0
    for lse in lses:
        lse[rows - 1] = float("-inf")  # a row empty on every rank

    ours, upstream = [], []
    for rank in range(world):
        heads_first = torch.empty_like(partials[rank]).view(-1)
        returned = qsa_dcp.qsa_dcp_lse_ag_out_rs(
            partials[rank],
            lses[rank],
            _FakeDCPGroup(rank, lses, ours),
            heads_first,
        )
        assert returned.shape == (heads, rows, head_dim)
        cp_lse_ag_out_rs(
            partials[rank].clone(),
            lses[rank],
            _FakeDCPGroup(rank, lses, upstream),
            is_lse_base_on_e=False,
        )

    assert [dim for _, dim in ours] == [0] * world
    assert [dim for _, dim in upstream] == [1] * world
    for (head_major, _), (row_major, _) in zip(ours, upstream):
        assert torch.equal(head_major.transpose(0, 1), row_major)
    local_heads = heads // world
    for rank in range(world):
        mine = _reduce_scatter(ours, rank)
        expected = _reduce_scatter(upstream, rank)
        assert mine.shape == (local_heads, rows, head_dim)
        assert torch.equal(mine.transpose(0, 1), expected)
        full = torch.stack([t.double() for t, _ in upstream]).sum(dim=0)
        heads_of_rank = slice(rank * local_heads, (rank + 1) * local_heads)
        assert torch.equal(expected, full[:, heads_of_rank])
