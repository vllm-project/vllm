# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GDP decode must apply every update and preserve unrelated request state."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.model_executor.layers.mamba.ops.causal_conv1d_prefill import (
    causal_conv1d_prefill,
)
from vllm.model_executor.layers.mamba.ops.gated_delta_product import (
    fused_recurrent_gated_delta_product,
)
from vllm.model_executor.layers.mamba.ops.gdp_chunk_prefill import (
    gdp_chunk_prefill,
    gdp_chunk_prefill_supported,
)
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

requires_gdp_chunk_prefill = pytest.mark.skipif(
    not gdp_chunk_prefill_supported(1, 64, 32),
    reason="GDP chunk prefill is not supported on this platform",
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


def _virtual_inputs(conv, b, a, a_log, bias, m, heads=4, groups=2, kdim=64, vdim=32):
    t = conv.shape[0]
    pieces = conv.split([heads * vdim] * m + [groups * kdim] * (m + 1), dim=-1)
    v = torch.stack([x.view(t, heads, vdim) for x in pieces[:m]], 1)
    k = torch.stack(
        [x.view(t, groups, kdim) for x in pieces[m : 2 * m]], 1
    ).repeat_interleave(heads // groups, 2)
    q = pieces[-1].view(t, groups, kdim).repeat_interleave(heads // groups, 1)
    k = l2norm_fwd(k.reshape(t * m, heads, kdim).contiguous())
    q = l2norm_fwd(q.contiguous())
    vq = q.new_zeros(t, m, heads, kdim)
    vq[:, -1] = q
    g = -a_log.float().exp() * torch.nn.functional.softplus(a.float() + bias.float())
    vg = g.new_zeros(t, m, heads)
    vg[:, 0] = g
    return (
        vq.reshape(1, t * m, heads, kdim),
        k.unsqueeze(0),
        v.reshape(1, t * m, heads, vdim),
        vg.reshape(1, t * m, heads),
        b.reshape(1, t * m, heads).sigmoid(),
    )


@requires_gdp_chunk_prefill
@pytest.mark.parametrize("m", [1, 3, 4])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_direct_prefill_matches_fla_with_initial_and_checkpoint_states(m, state_dtype):
    torch.manual_seed(19)
    t, heads, groups, kdim, vdim = 195, 4, 2, 64, 32
    width = m * heads * vdim + (m + 1) * groups * kdim
    projection = torch.randn(
        t, width + (m + 1) * heads, device="cuda", dtype=torch.bfloat16
    )
    conv, b, a = projection.split([width, m * heads, heads], -1)
    a_log = torch.randn(heads, device="cuda")
    bias = torch.randn(heads, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 65, 194, 195, 195], device="cuda", dtype=torch.int32)
    indices = torch.tensor([1, 2, 0, 0], device="cuda", dtype=torch.int32)
    has = torch.tensor([True, False, False, False], device="cuda")
    offsets = torch.tensor([64, 64, 0, 0], device="cuda", dtype=torch.int32)
    checkpoints = torch.tensor([3, 4, 0, 0], device="cuda", dtype=torch.int32)
    state = torch.randn(6, heads, vdim, kdim, device="cuda", dtype=state_dtype) * 0.1
    before = state.clone()
    initial = state[indices].contiguous()
    initial[~has] = 0
    ckpt = torch.empty_like(initial)
    q, k, v, g, beta = _virtual_inputs(conv, b, a, a_log, bias, m)
    vcu = cu * m
    reference, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=vcu,
        chunk_indices=prepare_chunk_indices(vcu, 64),
        chunk_offsets=prepare_chunk_offsets(vcu, 64),
        checkpoint_state=ckpt,
        checkpoint_offsets=offsets * m,
    )
    out = gdp_chunk_prefill(
        conv,
        b,
        a,
        a_log,
        bias,
        state,
        cu,
        indices,
        has,
        num_householder=m,
        num_heads=heads,
        num_groups=groups,
        head_k_dim=kdim,
        head_v_dim=vdim,
        checkpoint_offsets=offsets,
        checkpoint_indices=checkpoints,
    )
    expected = reference.view(t, m, heads, vdim)[:, -1]
    # The two BF16 chunk decompositions accumulate/reorder products differently.
    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(
        state[1:3], final[:2].to(state_dtype), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(state[3:5], ckpt[:2], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state[[0, 5]], before[[0, 5]], atol=0, rtol=0)


@requires_gdp_chunk_prefill
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim_first", [True, False])
def test_direct_conv_matches_existing_conv_and_exports_windows(dtype, dim_first):
    torch.manual_seed(23)
    lengths = [1, 2, 3, 65, 129, 0]
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    t = sum(lengths)
    width = 96
    x = torch.randn(t, width + 16, device="cuda", dtype=dtype)[:, :width]
    weight = torch.randn(width, 4, device="cuda", dtype=dtype)
    # Match both cache layouts, including padding between cache pages.
    storage = torch.randn(9, width * 3 + 32, device="cuda", dtype=dtype)
    state = storage[:, : width * 3].view(9, width, 3)
    if not dim_first:
        state = storage[:, : width * 3].view(9, 3, width).transpose(-1, -2)
    initial = state.clone()
    ref_state = state.clone()
    indices = torch.tensor([1, 2, 3, 4, 5, 0], device="cuda", dtype=torch.int32)
    has = torch.tensor([True, False, True, False, True, False], device="cuda")
    offsets = torch.tensor([0, 0, 0, 64, 64, 0], device="cuda", dtype=torch.int32)
    ckpts = torch.tensor([0, 0, 0, 6, 7, 0], device="cuda", dtype=torch.int32)
    expected = causal_conv1d_fn(
        x.contiguous().T,
        weight,
        None,
        ref_state,
        cu,
        cache_indices=indices,
        has_initial_state=has,
        activation="silu",
    ).T
    actual = causal_conv1d_prefill(
        x,
        weight,
        state,
        cu,
        indices,
        has,
        checkpoint_offsets=offsets,
        checkpoint_indices=ckpts,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(state[:6], ref_state[:6], atol=0, rtol=0)
    for row, slot in ((3, 6), (4, 7)):
        end = int(cu[row]) + 64
        torch.testing.assert_close(state[slot], x[end - 3 : end].T, atol=0, rtol=0)
    torch.testing.assert_close(state[[0, 8]], initial[[0, 8]], atol=0, rtol=0)


@requires_gdp_chunk_prefill
@pytest.mark.parametrize("m", [1, 3, 4])
def test_checkpoint_at_sequence_end(m):
    torch.manual_seed(29)
    h, g, k, v, t = 4, 2, 64, 32, 64
    width = m * h * v + (m + 1) * g * k
    x = torch.randn(t, width, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(t, m * h, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(3, h, v, k, device="cuda", dtype=torch.bfloat16)
    gdp_chunk_prefill(
        x,
        b,
        a,
        torch.zeros(h, device="cuda"),
        torch.zeros(h, device="cuda", dtype=torch.bfloat16),
        state,
        torch.tensor([0, t], device="cuda", dtype=torch.int32),
        torch.tensor([1], device="cuda", dtype=torch.int32),
        torch.tensor([True], device="cuda"),
        num_householder=m,
        num_heads=h,
        num_groups=g,
        head_k_dim=k,
        head_v_dim=v,
        checkpoint_offsets=torch.tensor([t], device="cuda", dtype=torch.int32),
        checkpoint_indices=torch.tensor([2], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(state[1], state[2], atol=0, rtol=0)


@requires_gdp_chunk_prefill
def test_large_batch_grid_matches_recurrent():
    from vllm.model_executor.layers.mamba.ops.gated_delta_product import (
        fused_recurrent_gated_delta_product,
    )

    torch.manual_seed(31)
    n, h, g, k, v, m = 683, 96, 8, 64, 32, 3
    width = m * h * v + (m + 1) * g * k
    x = torch.randn(n, width, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, m * h, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(n, h, device="cuda", dtype=torch.bfloat16)
    a_log = torch.zeros(h, device="cuda")
    bias = torch.zeros(h, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(n + 1, h, v, k, device="cuda", dtype=torch.bfloat16) * 0.1
    reference = state.clone()
    expected = torch.empty(n, h, v, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
    actual = gdp_chunk_prefill(
        x,
        b,
        a,
        a_log,
        bias,
        state,
        torch.arange(n + 1, device="cuda", dtype=torch.int32),
        ids,
        torch.ones(n, device="cuda", dtype=torch.bool),
        num_householder=m,
        num_heads=h,
        num_groups=g,
        head_k_dim=k,
        head_v_dim=v,
    )
    fused_recurrent_gated_delta_product(
        x,
        b,
        a,
        a_log,
        bias,
        reference,
        expected,
        num_householder=m,
        num_heads=h,
        num_groups=g,
        state_dim=k,
        head_dim=v,
        ssm_state_indices=ids,
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state, reference, atol=2e-2, rtol=2e-2)
