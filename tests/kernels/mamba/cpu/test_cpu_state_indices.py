# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.mamba.utils import selective_state_update_ref
from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not hasattr(torch.ops._C, "selective_state_update_cpu"),
    reason="requires the native CPU Mamba extension",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("separate_destinations", [False, True])
def test_speculative_state_tables_preserve_checkpoints(dtype, separate_destinations):
    """Ragged requests use accepted columns and strided, possibly aliased tables."""
    torch.manual_seed(7)
    backing = torch.zeros((2, 8), dtype=torch.int32)
    indices = backing[:, :6:2]
    indices[:] = torch.tensor([[1, 3, 5], [7, 9, 11]])
    destinations = indices + 1 if separate_destinations else indices
    state = torch.randn(16, 2, 2, 16, dtype=dtype)
    original = state.clone()
    expected_state = state.clone()
    x = torch.randn(5, 2, 2, dtype=dtype)
    dt = torch.full((5, 2), 0.125)
    a = torch.zeros(2)
    b = torch.randn(5, 1, 16, dtype=dtype)
    c = torch.randn_like(b)
    accepted = torch.tensor([2, 1], dtype=torch.int32)
    starts = torch.tensor([0, 3, 5], dtype=torch.int32)
    expected = torch.empty_like(x)
    for seq in range(2):
        recurrent = original[indices[seq, accepted[seq] - 1]].float()
        for step, token in enumerate(range(starts[seq], starts[seq + 1])):
            recurrent = (
                recurrent
                + b[token].float()[:, None, :]
                * x[token].float()[:, :, None]
                * dt[token, :, None, None]
            )
            expected[token] = (recurrent * c[token].float()[:, None, :]).sum(-1)
            expected_state[destinations[seq, step]] = recurrent.to(dtype)
    output = torch.empty_like(x)
    ops.selective_state_update_cpu(
        state,
        x,
        dt,
        a,
        b,
        c,
        None,
        None,
        None,
        False,
        indices,
        destinations if separate_destinations else None,
        -1,
        output,
        accepted,
        starts,
    )
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(state, expected_state)


@pytest.mark.parametrize("indexed", [False, True])
def test_regular_decode_preserves_mamba2_reference(indexed):
    torch.manual_seed(19)
    state = torch.randn(8, 2, 2, 16)
    indices = torch.tensor([1, 3], dtype=torch.int32) if indexed else None
    slots = indices if indexed else torch.arange(2)
    expected_state = state[slots].clone()
    original = state.clone()
    x = torch.randn(2, 2, 2)
    dt = torch.randn(2, 2, 1).expand_as(x)
    a = -torch.rand(2, 1, 1).expand(2, 2, 16)
    b = torch.randn(2, 1, 16)
    c = torch.randn_like(b)
    d = torch.randn(2, 1).expand(2, 2)
    bias = torch.randn(2, 1).expand(2, 2)
    z = torch.randn_like(x)
    expected = selective_state_update_ref(
        expected_state, x, dt, a, b, c, d, z, bias, True
    )
    original[slots] = expected_state
    output = torch.empty_like(x)
    ops.selective_state_update_cpu(
        state,
        x,
        dt,
        a,
        b,
        c,
        d,
        z,
        bias,
        True,
        indices,
        None,
        -1,
        output,
        None,
        None,
    )
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(state, original)
