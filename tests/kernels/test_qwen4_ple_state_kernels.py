# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness coverage for the self-developed Qwen4 PLE-state kernels.

Torch references live in this test module only. The production wrappers are
required to fail closed instead of dispatching a Torch compute fallback.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("triton")

from vllm.model_executor.layers.ple_state import (  # noqa: E402
    ple_state_gather,
    ple_state_scatter_,
)

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Qwen4 PLE-state Triton kernels require CUDA/ROCm.",
)


def _strided_state(cache_rows, hidden, width):
    storage = torch.randn(
        (cache_rows, width, hidden), device=DEVICE, dtype=torch.bfloat16
    )
    return storage.transpose(1, 2)


def test_qwen4_ple_gather_preserves_strides_and_null_is_not_row_zero():
    state = _strided_state(11, 3, 5)
    indices = torch.tensor([1, -1, 1, 99, 2], device=DEVICE, dtype=torch.int64)
    output = ple_state_gather(state, indices)
    valid = (indices >= 0) & (indices < state.shape[0])
    bounded = indices.clamp(0, state.shape[0] - 1)
    expected = torch.ops.aten.index_select.default(state, 0, bounded)
    expected = torch.where(valid.view(-1, 1, 1), expected, torch.zeros_like(expected))
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert output.stride(1) == 1
    assert torch.equal(output[1], torch.zeros_like(output[1]))
    assert not torch.equal(output[1], state[0])


def test_qwen4_ple_scatter_masked_null_and_duplicate_is_exact():
    state = _strided_state(11, 3, 5)
    baseline = state.clone()
    indices = torch.tensor([-1, 2, 2, 99, 4], device=DEVICE, dtype=torch.int64)
    rows = torch.randn_like(state[: indices.numel()])
    write_mask = torch.tensor(
        [False, False, True, True, True], device=DEVICE, dtype=torch.bool
    )
    expected = baseline.clone()
    for row, index in enumerate(indices.tolist()):
        if bool(write_mask[row]) and 0 <= index < state.shape[0]:
            expected[index].copy_(rows[row])
    ple_state_scatter_(state, indices, rows, write_mask=write_mask)
    torch.testing.assert_close(state, expected, atol=0, rtol=0)
    torch.testing.assert_close(state[0], baseline[0], atol=0, rtol=0)
    with pytest.raises(NotImplementedError):
        ple_state_scatter_(baseline, indices, rows)


def test_qwen4_ple_cpu_guards_fail_closed():
    state = torch.empty((2, 3, 5), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        ple_state_gather(state, torch.zeros((1,), dtype=torch.int64))
