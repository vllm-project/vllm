# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GDP decode must apply every update and preserve unrelated request state."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.gated_delta_product import (
    fused_recurrent_gated_delta_product,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 3])
def test_decode_matches_delta_product_and_preserves_null_state(dtype, steps):
    torch.manual_seed(42)
    n, groups, heads, key_dim, value_dim = 3, 2, 4, 64, 32
    state = torch.randn(4, heads, value_dim, key_dim, device="cuda") * 0.1
    expected_state = state.clone()
    indices = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    a_log = torch.randn(heads, device="cuda")
    dt_bias = torch.randn(heads, device="cuda", dtype=dtype)
    width = steps * heads * value_dim + (steps + 1) * groups * key_dim
    for _ in range(2):
        # Column slices exercise the strides used by the merged projection.
        projected = torch.randn(
            n, width + (steps + 1) * heads, device="cuda", dtype=dtype
        )
        conv, b, a = projected.split([width, steps * heads, heads], dim=-1)
        output = torch.empty(n, heads, value_dim, device="cuda", dtype=dtype)
        fused_recurrent_gated_delta_product(
            conv,
            b,
            a,
            a_log,
            dt_bias,
            state,
            output,
            num_householder=steps,
            num_groups=groups,
            num_heads=heads,
            state_dim=key_dim,
            head_dim=value_dim,
            ssm_state_indices=indices,
        )
        values, keys, queries = conv.float().split(
            [steps * heads * value_dim, steps * groups * key_dim, groups * key_dim], -1
        )
        values = values.view(n, steps, heads, value_dim)
        keys = keys.view(n, steps, groups, key_dim).repeat_interleave(
            heads // groups, 2
        )
        queries = queries.view(n, groups, key_dim).repeat_interleave(heads // groups, 1)
        keys = keys * torch.rsqrt(keys.square().sum(-1, keepdim=True) + 1e-6)
        queries = queries * torch.rsqrt(queries.square().sum(-1, keepdim=True) + 1e-6)
        beta = b.float().view(n, steps, heads).sigmoid()
        decay = (-a_log.exp() * F.softplus(a.float() + dt_bias.float())).exp()
        expected_output = torch.zeros_like(output)
        for row, slot in enumerate((2, 0, 1)):
            if slot == 0:
                continue
            h = expected_state[slot] * decay[row, :, None, None]
            for step in range(steps):
                k = keys[row, step]
                error = values[row, step] - (h * k[:, None, :]).sum(-1)
                h = h + (beta[row, step, :, None] * error)[:, :, None] * k[:, None, :]
            expected_state[slot] = h
            expected_output[row] = (h * queries[row, :, None, :]).sum(
                -1
            ) * key_dim**-0.5
        torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=3e-6)
        torch.testing.assert_close(
            output,
            expected_output,
            rtol=1e-4 if dtype == torch.float32 else 1e-2,
            atol=3e-6,
        )
