# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The packed Q|K|V KDA prefill conv against three stock `causal_conv1d_fn`
calls: outputs and updated conv state bit for bit."""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.models.kimi_k3.nvidia.ops.kda_conv import (
    packed_causal_conv1d_fn,
    packed_causal_conv1d_supported,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# The bit-exact match holds for CUDA's code generation; ROCm's differs by ULPs.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="KDA prefill conv needs CUDA"
)


def _stock(x_parts, weights, bias_parts, states, qsl, cache_indices, has_init):
    outs = []
    for x, w, b, s in zip(x_parts, weights, bias_parts, states):
        outs.append(
            causal_conv1d_fn(
                x.transpose(0, 1),
                w,
                b,
                activation="silu",
                conv_states=s,
                has_initial_state=has_init,
                cache_indices=cache_indices,
                query_start_loc=qsl,
            ).transpose(0, 1)
        )
    return outs


@pytest.mark.parametrize(
    "seqlens",
    [
        [1],
        [2, 1, 3],
        [37],
        [5, 64, 1, 2, 33, 100],
        [4096, 3, 2047],
        [16384],
        [1] * 300 + [70] + [2] * 40 + [129],  # many sequences: the program search
    ],
)
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
# The KDA conv state holds width - 1 + num_spec columns; prefill uses the first 3.
@pytest.mark.parametrize("num_spec", [0, 3])
@torch.inference_mode()
def test_packed_causal_conv1d_matches_stock(seqlens, dim_first, with_bias, num_spec):
    width = 4
    state_len = width - 1 + num_spec
    torch.manual_seed(len(seqlens) * 131 + sum(seqlens))
    device = "cuda"
    part = 384  # 3 heads x 128 per Q/K/V: not a multiple of the 256 channel tile
    dim = 3 * part
    num_tokens = sum(seqlens)
    num_seqs = len(seqlens)
    num_lines = 2 * num_seqs + 2

    # Row-strided bf16 Q|K|V inside a wider in_proj output, as in the layer.
    proj = torch.randn(num_tokens, dim + 141, device=device, dtype=torch.bfloat16)
    x = proj[:, :dim]
    weight = torch.randn(dim, width, device=device, dtype=torch.float32) * 0.5
    bias = torch.randn(dim, device=device, dtype=torch.float32) if with_bias else None
    # Conv state values as written by the stock kernels: bf16 inputs held in FP32.
    if dim_first:  # DS: (lines, dim, state) contiguous
        state = torch.randn(num_lines, dim, state_len, device=device)
    else:  # SD: (lines, state, dim) storage, viewed as (lines, dim, state)
        state = torch.randn(num_lines, state_len, dim, device=device).transpose(-1, -2)
    state.copy_(state.to(torch.bfloat16).float())

    qsl = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    qsl[1:] = torch.tensor(seqlens, device=device).cumsum(0)
    # Distinct non-null lines; the last sequence sits on the null block when there
    # are several sequences (padding: neither output nor state is written).
    lines = torch.randperm(num_lines - 1, device=device)[:num_seqs] + 1
    if num_seqs > 2:
        lines[-1] = NULL_BLOCK_ID
    cache_indices = lines.to(torch.int32)
    has_init = torch.rand(num_seqs, device=device) < 0.6
    if num_seqs > 1:
        has_init[0] = True
        has_init[1] = False

    assert packed_causal_conv1d_supported(x, weight, state, part)

    ref_state = state.clone()
    ref = _stock(
        x.split(part, dim=-1),
        weight.split(part, dim=0),
        bias.split(part) if bias is not None else [None] * 3,
        ref_state.split(part, dim=-2),
        qsl,
        cache_indices,
        has_init,
    )
    new_state = state.clone()
    out = packed_causal_conv1d_fn(
        x, weight, bias, new_state, qsl, cache_indices, has_init, part
    )
    assert out.shape == (3, num_tokens, part) and out.dtype == torch.bfloat16
    assert out[0].is_contiguous()

    valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    for i in range(num_seqs):
        if int(cache_indices[i]) == NULL_BLOCK_ID:
            valid[int(qsl[i]) : int(qsl[i + 1])] = False
    for got, want in zip(out.unbind(0), ref):
        torch.testing.assert_close(got[valid], want[valid], atol=0, rtol=0)
    torch.testing.assert_close(new_state, ref_state, atol=0, rtol=0)


def test_packed_causal_conv1d_supported_shapes():
    x = torch.zeros(4, 3 * 128, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(3 * 128, 4, device="cuda")
    state = torch.zeros(2, 3 * 128, 3, device="cuda", dtype=torch.bfloat16)
    # The stock kernel computes in the state dtype: bf16 state is not FP32 math.
    assert not packed_causal_conv1d_supported(x, w, state, 128)
    assert packed_causal_conv1d_supported(x, w, state.float(), 128)
    assert not packed_causal_conv1d_supported(x, w[:, :3], state.float(), 128)
    # The state must be (cache lines, dim, >= 3) over the same channels.
    assert not packed_causal_conv1d_supported(x, w, state.float()[:, :128], 128)
    assert not packed_causal_conv1d_supported(x, w, state.float()[..., :2], 128)
    assert not packed_causal_conv1d_supported(x, w, state.float()[None], 128)


@torch.inference_mode()
def test_packed_causal_conv1d_cuda_graph_replay():
    """Captured once, replayed with other sequence lengths and initial states:
    the (sequence, chunk) map is read on the device."""
    device = "cuda"
    part, num_tokens = 256, 3000
    dim = 3 * part
    torch.manual_seed(5)
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, 4, device=device) * 0.5
    seed = torch.randn(8, dim, 3, device=device).to(torch.bfloat16).float()
    state = seed.clone()
    qsl = torch.tensor([0, 1000, 2999, 3000], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([3, 5, 7], dtype=torch.int32, device=device)
    has_init = torch.tensor([True, False, True], device=device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = packed_causal_conv1d_fn(
            x, weight, None, state, qsl, cache_indices, has_init, part
        )

    for lens, init in (
        ([1000, 1999, 1], [True, False, True]),
        ([2, 2900, 98], [False, True, True]),
    ):
        qsl[1:] = torch.tensor(lens, device=device).cumsum(0)
        has_init.copy_(torch.tensor(init, device=device))
        state.copy_(seed)
        graph.replay()
        ref_state = seed.clone()
        ref = _stock(
            x.split(part, dim=-1),
            weight.split(part, dim=0),
            [None] * 3,
            ref_state.split(part, dim=-2),
            qsl,
            cache_indices,
            has_init,
        )
        for got, want in zip(out.unbind(0), ref):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        torch.testing.assert_close(state, ref_state, atol=0, rtol=0)


@torch.inference_mode()
def test_packed_causal_conv1d_empty_sequences():
    """Zero-length sequences (padded query_start_loc entries) get no program: no
    output rows and an untouched conv state line; the others match stock."""
    device = "cuda"
    part = 128
    dim = 3 * part
    torch.manual_seed(11)
    seqlens = [70, 0, 1, 0, 200, 0, 0]
    qsl = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    qsl[1:] = torch.tensor(seqlens, device=device).cumsum(0)
    num_tokens = int(qsl[-1])
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, 4, device=device) * 0.5
    state = torch.randn(len(seqlens) + 1, dim, 3, device=device)
    state.copy_(state.to(torch.bfloat16).float())
    cache_indices = torch.arange(1, len(seqlens) + 1, dtype=torch.int32, device=device)
    has_init = torch.ones(len(seqlens), dtype=torch.bool, device=device)

    new_state = state.clone()
    out = packed_causal_conv1d_fn(
        x, weight, None, new_state, qsl, cache_indices, has_init, part
    )
    keep = [i for i, n in enumerate(seqlens) if n > 0]
    ref_qsl = torch.zeros(len(keep) + 1, dtype=torch.int32, device=device)
    ref_qsl[1:] = torch.tensor([seqlens[i] for i in keep], device=device).cumsum(0)
    ref_state = state.clone()
    ref = _stock(
        x.split(part, dim=-1),
        weight.split(part, dim=0),
        [None] * 3,
        ref_state.split(part, dim=-2),
        ref_qsl,
        cache_indices[keep],
        has_init[keep],
    )
    for got, want in zip(out.unbind(0), ref):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    torch.testing.assert_close(new_state, ref_state, atol=0, rtol=0)
