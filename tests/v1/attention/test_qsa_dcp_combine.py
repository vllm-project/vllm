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
