# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical equivalence check for the fused GDN input projection.

Qwen3.5 / Qwen3.8 (non-interleaved) fuse ``in_proj_qkvz`` + ``in_proj_ba``
into a single ``MergedColumnParallelLinear`` so the two input projections are
one GEMM instead of two. This test verifies that the fused GEMM, split back
into the qkvz and ba parts, is numerically identical to the two separate
GEMMs given the same weights.

Runs on CPU (a plain GEMM equivalence check, no AITER/GPU required).
"""

from __future__ import annotations

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import MergedColumnParallelLinear

# Small dims so the test is cheap.
H = 16  # hidden size
KEY_DIM = 32
VALUE_DIM = 64
NUM_V_HEADS = 8


def _make_vllm_config():
    return create_vllm_config(model_name="Qwen/Qwen3.5-0.8B", block_size=16)


def test_fused_in_proj_matches_unfused():
    vllm_config = _make_vllm_config()
    with set_current_vllm_config(vllm_config):
        # Fused: one 6-shard GEMM [q, k, v, z, b, a].
        fused = MergedColumnParallelLinear(
            input_size=H,
            output_sizes=[
                KEY_DIM,
                KEY_DIM,
                VALUE_DIM,
                VALUE_DIM,
                NUM_V_HEADS,
                NUM_V_HEADS,
            ],
            bias=False,
            prefix="linear_attn.in_proj_qkvzba",
        )
        # Unfused: the two separate GEMMs.
        qkvz = MergedColumnParallelLinear(
            input_size=H,
            output_sizes=[KEY_DIM, KEY_DIM, VALUE_DIM, VALUE_DIM],
            bias=False,
            prefix="linear_attn.in_proj_qkvz",
        )
        ba = MergedColumnParallelLinear(
            input_size=H,
            output_sizes=[NUM_V_HEADS, NUM_V_HEADS],
            bias=False,
            prefix="linear_attn.in_proj_ba",
        )

        # Same weights: the fused weight is the two unfused weights stacked.
        torch.manual_seed(0)
        qkvz.weight.data = torch.randn(*qkvz.weight.shape)
        ba.weight.data = torch.randn(*ba.weight.shape)
        fused.weight.data = torch.cat([qkvz.weight.data, ba.weight.data], dim=0)

        x = torch.randn(4, H)

        fused_out = fused(x)[0]
        qkvz_out = qkvz(x)[0]
        ba_out = ba(x)[0]

        # Split the fused output back into the qkvz and ba parts.
        ba_cols = NUM_V_HEADS * 2
        qkvz_part = fused_out[:, : fused_out.shape[1] - ba_cols]
        ba_part = fused_out[:, fused_out.shape[1] - ba_cols:]

        assert torch.allclose(qkvz_part, qkvz_out, atol=1e-5)
        assert torch.allclose(ba_part, ba_out, atol=1e-5)
