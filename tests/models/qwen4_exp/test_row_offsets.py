# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp kernels on rows that start more than 2**31 - 1 elements in."""

import os

import pytest
import torch

from vllm.models.qwen4_exp.common.ops.ple import ple_gate
from vllm.models.qwen4_exp.nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

# TRITON_INTERPRET=1 runs the kernels on CPU tensors.
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"

pytestmark = pytest.mark.skipif(
    (DEVICE == "cuda" and not current_platform.is_cuda()) or not HAS_TRITON,
    reason="Qwen4Exp kernels require CUDA (or the Triton interpreter) and Triton",
)

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
EPS = 1e-6
# Row 2 starts at 2**31 + 8192 elements, like row 209,716 of a
# [num_tokens, HYPER_HIDDEN_SIZE] tensor.
ROW_STRIDE = 2**30 + 4096


def _strided_rows(*widths: int) -> list[torch.Tensor]:
    rows = torch.empty_strided(
        (3, sum(widths)), (ROW_STRIDE, 1), dtype=torch.bfloat16, device=DEVICE
    )
    rows.copy_(torch.randn(rows.shape, dtype=torch.bfloat16, device=DEVICE))
    return list(rows.split(widths, dim=-1))


def _assert_stride_invariant(fn, inputs: list[torch.Tensor]) -> None:
    expected = fn(*(t.contiguous() for t in inputs))
    for actual, want in zip(fn(*inputs), expected):
        torch.testing.assert_close(actual, want, atol=0, rtol=0)


def test_hc_ops_row_offsets_past_int32() -> None:
    torch.manual_seed(0)
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device=DEVICE)

    def run(x, gate, block_output, injection):
        return (
            grouped_gemma_rmsnorm(x, weight, EPS, HC),
            hc_silu(x, HC),
            hc_gate_mix(x, gate, HC),
            hc_combine(x, block_output, injection, HC),
            *hc_combine_norm(x, block_output, injection, weight, EPS, HC),
        )

    inputs = _strided_rows(HYPER_HIDDEN_SIZE, HYPER_HIDDEN_SIZE, HIDDEN_SIZE, HC)
    _assert_stride_invariant(run, inputs)


def test_ple_gate_row_offsets_past_int32() -> None:
    torch.manual_seed(0)
    shape = (3, HYPER_HIDDEN_SIZE)
    hidden = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    norm_weights = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)

    def run(key, value):
        return ple_gate(key, value, hidden, *norm_weights, EPS)

    _assert_stride_invariant(run, _strided_rows(HYPER_HIDDEN_SIZE, HIDDEN_SIZE))
