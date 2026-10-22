# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AITER fused conv1d is ROCm-only"
)

WIDTH = 4  # Kimi-K3 short_conv_kernel_size
STATE_LEN = WIDTH - 1
HEAD_DIM = 128
LOCAL_HEADS = 4  # per-TP-rank heads; projection = 512, packed dim = 1536


def _fused_fn():
    gdn = pytest.importorskip("aiter.ops.triton.gated_delta_net")
    return gdn.causal_conv1d_split_qkv_triton_tile_fn


def _make_inputs(seqlens, has_initial_state, seed=0, device="cuda"):
    torch.manual_seed(seed)
    proj = HEAD_DIM * LOCAL_HEADS
    dim = 3 * proj
    total = sum(seqlens)
    n_seqs = len(seqlens)
    # Token-major activations, as produced by the in_proj GEMM.
    mixed_qkv = torch.randn(total, dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(dim, WIDTH, dtype=torch.bfloat16, device=device) * 0.3
    # More cache lines than sequences, assigned in a shuffled order. Line 0 is
    # vLLM's null block (NULL_BLOCK_ID), which causal_conv1d_fn skips, so real
    # sequences never map to it.
    n_lines = n_seqs + 2
    conv_state = torch.randn(
        n_lines, dim, STATE_LEN, dtype=torch.bfloat16, device=device
    )
    cache_indices = (torch.randperm(n_lines - 1, device=device)[:n_seqs] + 1).to(
        torch.int32
    )
    query_start_loc = torch.tensor(
        [0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    has_initial = torch.full(
        (n_seqs,), has_initial_state, dtype=torch.bool, device=device
    )
    return mixed_qkv, weight, conv_state, cache_indices, query_start_loc, has_initial


def _reference(
    mixed_qkv, weight, conv_state, cache_indices, query_start_loc, has_initial
):
    """fp32 per-sequence depthwise causal conv + SiLU.

    Returns (out[T, dim], new_conv_state).
    """
    x = mixed_qkv.float().t()  # (dim, T)
    w = weight.float()
    dim = x.shape[0]
    out = torch.empty_like(x)
    new_state = conv_state.float().clone()
    qsl = query_start_loc.tolist()
    for s in range(len(qsl) - 1):
        start, end = qsl[s], qsl[s + 1]
        ci = int(cache_indices[s])
        xs = x[:, start:end]
        if bool(has_initial[s]):
            xin = torch.cat([conv_state[ci].float(), xs], dim=1)
        else:
            xin = F.pad(xs, (STATE_LEN, 0))
        y = F.conv1d(xin.unsqueeze(0), w.unsqueeze(1), None, groups=dim)[0]
        out[:, start:end] = F.silu(y)
        new_state[ci] = xin[:, -STATE_LEN:]
    return out.t(), new_state


def _three_launch(
    mixed_qkv, weight, conv_state, cache_indices, query_start_loc, has_initial
):
    """The pre-fusion KDA prefill path: one causal_conv1d_fn per projection."""
    proj = mixed_qkv.shape[1] // 3
    outs = []
    for x, state, w in zip(
        mixed_qkv.split(proj, dim=-1),
        conv_state.split(proj, dim=-2),
        weight.split(proj, dim=0),
    ):
        outs.append(
            causal_conv1d_fn(
                x.transpose(0, 1),
                w,
                None,
                activation="silu",
                conv_states=state,
                has_initial_state=has_initial,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            ).transpose(0, 1)
        )
    return outs


@pytest.mark.parametrize(
    "seqlens", [[1024], [8, 1, 300], [64, 64, 64], [513, 64, 1, 1200]]
)
@pytest.mark.parametrize("has_initial_state", [False, True])
@pytest.mark.parametrize("block_m", [8, 128])
def test_fused_split_qkv_conv1d_matches_reference(seqlens, has_initial_state, block_m):
    fused_fn = _fused_fn()
    mixed_qkv, weight, conv_state, cache_indices, qsl, has_initial = _make_inputs(
        seqlens, has_initial_state
    )
    proj = mixed_qkv.shape[1] // 3

    ref_out, ref_state = _reference(
        mixed_qkv, weight, conv_state, cache_indices, qsl, has_initial
    )

    state_3 = conv_state.clone()
    q3, k3, v3 = _three_launch(
        mixed_qkv, weight, state_3, cache_indices, qsl, has_initial
    )

    state_f = conv_state.clone()
    # Exactly the call made by KimiK3DeltaAttention._forward.
    qf, kf, vf = fused_fn(
        mixed_qkv.transpose(0, 1),
        weight,
        None,
        state_f,
        qsl,
        proj,
        proj,
        cache_indices=cache_indices,
        has_initial_state=has_initial,
        activation="silu",
        block_m=block_m,
        metadata=None,
        preserve_input_dtype=True,
    )

    fused = torch.cat([qf, kf, vf], dim=-1)
    assert fused.shape == mixed_qkv.shape and fused.dtype == mixed_qkv.dtype
    assert all(t.is_contiguous() for t in (qf, kf, vf))

    tol = dict(atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fused.float(), ref_out, **tol)
    torch.testing.assert_close(fused, torch.cat([q3, k3, v3], dim=-1), **tol)

    # conv_state writeback: assigned lines updated, the others untouched.
    torch.testing.assert_close(state_f.float(), ref_state, **tol)
    torch.testing.assert_close(state_f, state_3, **tol)
    untouched = torch.ones(conv_state.shape[0], dtype=torch.bool, device="cuda")
    untouched[cache_indices.long()] = False
    assert untouched[NULL_BLOCK_ID]
    assert torch.equal(state_f[untouched], conv_state[untouched])
