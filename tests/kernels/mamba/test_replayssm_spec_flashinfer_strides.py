# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stride contract for the FlashInfer ReplaySSM spec adapter.

The decode path hands FlashInfer views carved out of the packed conv output, so
a wider token stride is expected while the inner strides must match exactly.
The adapter asserts rather than calling `.contiguous()`, because a conditional
copy in the decode path would allocate during CUDA-graph capture -- these tests
reproduce the mixer's actual view construction rather than building tidy
contiguous tensors.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.replayssm_spec_flashinfer import (
    _validate_cache_spec,
    _validate_packed_inputs,
    _validate_tied_weights,
)
from vllm.v1.kv_cache_interface import MambaSpec

NHEADS = 8
HEAD_DIM = 64
DSTATE = 128
NGROUPS = 2
BUFFER_LEN = 16
MAX_SPEC_LEN = 4
RING_LEN = BUFFER_LEN + MAX_SPEC_LEN
NUM_TOKENS = 6


def _mixer_views(num_tokens: int = NUM_TOKENS):
    """Rebuild the views `MambaMixer2.conv_ssm_forward` passes to the SSU.

    Mirrors mamba_mixer2.py: split the packed (Q, conv_dim) conv output into
    x|B|C slices, `view` them into head/group shapes, and broadcast dt/dt_bias/D
    over head_dim.
    """
    conv_dim = NHEADS * HEAD_DIM + 2 * NGROUPS * DSTATE
    hidden_states_B_C = torch.zeros(num_tokens, conv_dim)
    x, b, c = torch.split(
        hidden_states_B_C,
        [NHEADS * HEAD_DIM, NGROUPS * DSTATE, NGROUPS * DSTATE],
        dim=-1,
    )
    x = x.view(-1, NHEADS, HEAD_DIM)
    b = b.view(-1, NGROUPS, DSTATE)
    c = c.view(-1, NGROUPS, DSTATE)

    dt = torch.zeros(num_tokens, NHEADS)[:, :, None].expand(-1, -1, HEAD_DIM)
    a = (
        torch.zeros(NHEADS)[:, None, ...][:, :, None]
        .expand(-1, HEAD_DIM, DSTATE)
        .to(dtype=torch.float32)
    )
    dt_bias = torch.ones(NHEADS)[:, None, ...].expand(-1, HEAD_DIM)
    out = torch.zeros(num_tokens, NHEADS * HEAD_DIM).view(num_tokens, -1, HEAD_DIM)

    return x, dt, a, b, c, out, dt_bias


def _cache_spec(
    ring_len: int = RING_LEN,
    dt_dtype: torch.dtype = torch.float32,
) -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=(
            (NHEADS * HEAD_DIM + 2 * NGROUPS * DSTATE, 3),
            (NHEADS, HEAD_DIM, DSTATE),
            (NHEADS, ring_len, HEAD_DIM),
            (NGROUPS, ring_len, DSTATE),
            (NHEADS, ring_len),
        ),
        dtypes=(
            torch.float32,
            torch.float32,
            torch.bfloat16,
            torch.bfloat16,
            dt_dtype,
        ),
    )


def test_real_mixer_views_satisfy_the_stride_contract():
    x, dt, a, b, c, out, dt_bias = _mixer_views()
    _validate_packed_inputs(
        x.unsqueeze(0),
        dt.unsqueeze(0),
        b.unsqueeze(0),
        c.unsqueeze(0),
        out.unsqueeze(0),
    )
    _validate_tied_weights(a, dt_bias)


def test_mixer_views_have_a_wider_token_stride_than_the_head_block():
    """The x/B/C views are slices of the conv output, so tokens are strided by
    conv_dim, not by nheads*head_dim. That is allowed; only inner strides are
    pinned.
    """
    x, _, _, b, _, _, _ = _mixer_views()
    conv_dim = NHEADS * HEAD_DIM + 2 * NGROUPS * DSTATE

    assert x.stride(0) == conv_dim > NHEADS * HEAD_DIM
    assert b.stride(0) == conv_dim
    # ...while the inner layout is exactly what the kernel indexes with.
    assert (x.stride(-2), x.stride(-1)) == (HEAD_DIM, 1)
    assert (b.stride(-2), b.stride(-1)) == (DSTATE, 1)


def test_rejects_untied_dt():
    x, _, _, b, c, out, _ = _mixer_views()
    untied_dt = torch.zeros(NUM_TOKENS, NHEADS, HEAD_DIM)
    with pytest.raises(AssertionError, match="dt must be tied"):
        _validate_packed_inputs(
            x.unsqueeze(0),
            untied_dt.unsqueeze(0),
            b.unsqueeze(0),
            c.unsqueeze(0),
            out.unsqueeze(0),
        )


def test_rejects_materialized_a():
    """A materialised A (e.g. if the parameter stopped being fp32) drops the tie."""
    _, _, a, _, _, _, dt_bias = _mixer_views()
    with pytest.raises(AssertionError, match="A must be tied"):
        _validate_tied_weights(a.contiguous(), dt_bias)


def test_rejects_transposed_inner_stride():
    """A head-major x (inner stride != head_dim) must fail, not be silently copied."""
    _, dt, _, b, c, out, _ = _mixer_views()
    bad_x = torch.zeros(NHEADS, NUM_TOKENS, HEAD_DIM).transpose(0, 1)
    assert bad_x.stride(-2) != HEAD_DIM
    with pytest.raises(AssertionError, match="x inner strides"):
        _validate_packed_inputs(
            bad_x.unsqueeze(0),
            dt.unsqueeze(0),
            b.unsqueeze(0),
            c.unsqueeze(0),
            out.unsqueeze(0),
        )


def test_cache_spec_accepted_at_exactly_b_plus_t():
    _validate_cache_spec(_cache_spec(), BUFFER_LEN, MAX_SPEC_LEN)


def test_rejects_power_of_two_padded_ring():
    """A next_pow2 ring (the Triton layout) would inflate FlashInfer's implicit
    max_window past replayssm_buffer_len, so it must be rejected outright.
    """
    with pytest.raises(ValueError, match="x_cache"):
        _validate_cache_spec(_cache_spec(ring_len=32), BUFFER_LEN, MAX_SPEC_LEN)


def test_rejects_non_fp32_dt_cache():
    with pytest.raises(ValueError, match="dt_cache must be fp32"):
        _validate_cache_spec(
            _cache_spec(dt_dtype=torch.bfloat16), BUFFER_LEN, MAX_SPEC_LEN
        )
