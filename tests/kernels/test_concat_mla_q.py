# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops

NUM_TOKENS = [1, 4, 16, 64, 128]
NUM_HEADS = [128]
NOPE_DIM = [512]
ROPE_DIM = [64]
DTYPES = [torch.bfloat16, torch.float16]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("nope_dim", NOPE_DIM)
@pytest.mark.parametrize("rope_dim", ROPE_DIM)
@pytest.mark.parametrize("dtype", DTYPES)
def test_concat_mla_q_contiguous(num_tokens, num_heads, nope_dim, rope_dim, dtype):
    """Test with contiguous inputs (standard layout)."""
    torch.manual_seed(42)
    ql_nope = torch.randn(num_tokens, num_heads, nope_dim, dtype=dtype, device="cuda")
    q_pe = torch.randn(num_tokens, num_heads, rope_dim, dtype=dtype, device="cuda")

    ref = torch.cat((ql_nope, q_pe), dim=-1)

    q_out = torch.empty(
        num_tokens, num_heads, nope_dim + rope_dim, dtype=dtype, device="cuda"
    )
    ops.concat_mla_q(ql_nope, q_pe, q_out)

    torch.testing.assert_close(q_out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [t for t in NUM_TOKENS if t > 1])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("nope_dim", NOPE_DIM)
@pytest.mark.parametrize("rope_dim", ROPE_DIM)
@pytest.mark.parametrize("dtype", DTYPES)
def test_concat_mla_q_transposed_nope(num_tokens, num_heads, nope_dim, rope_dim, dtype):
    """Test with transposed nope input (simulates BMM output after transpose).

    In the real code path, mqa_ql_nope is the result of:
        torch.bmm(q_nope, W_UK_T)  # [N, B, L]
        .transpose(0, 1)            # [B, N, L] — non-contiguous!
    """
    torch.manual_seed(42)
    nope_raw = torch.randn(num_heads, num_tokens, nope_dim, dtype=dtype, device="cuda")
    ql_nope = nope_raw.transpose(0, 1)  # [B, N, L], non-contiguous
    assert not ql_nope.is_contiguous()

    q_pe = torch.randn(num_tokens, num_heads, rope_dim, dtype=dtype, device="cuda")

    ref = torch.cat((ql_nope, q_pe), dim=-1)

    q_out = torch.empty(
        num_tokens, num_heads, nope_dim + rope_dim, dtype=dtype, device="cuda"
    )
    ops.concat_mla_q(ql_nope, q_pe, q_out)

    torch.testing.assert_close(q_out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_concat_mla_q_split_rope(num_tokens, num_heads, dtype):
    """Test with rope from a split (simulates the actual code path).

    In the real code path, q_pe comes from:
        mqa_q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    which creates a non-contiguous view with stride(1) != rope_dim.
    """
    torch.manual_seed(42)
    nope_dim = 512
    rope_dim = 64
    orig_dim = 128 + 64  # original q before absorption: [B, N, 192]

    # Simulate split from original q tensor
    q_orig = torch.randn(num_tokens, num_heads, orig_dim, dtype=dtype, device="cuda")
    q_nope_orig, q_pe = q_orig.split([128, 64], dim=-1)

    # q_pe is non-contiguous: stride(1) = 192, not 64
    assert q_pe.stride(1) == orig_dim
    assert q_pe.stride(2) == 1  # but innermost is fine

    # Simulate absorbed nope (contiguous, different size)
    ql_nope = torch.randn(num_tokens, num_heads, nope_dim, dtype=dtype, device="cuda")

    ref = torch.cat((ql_nope, q_pe), dim=-1)

    q_out = torch.empty(
        num_tokens, num_heads, nope_dim + rope_dim, dtype=dtype, device="cuda"
    )
    ops.concat_mla_q(ql_nope, q_pe, q_out)

    torch.testing.assert_close(q_out, ref, atol=0, rtol=0)


def test_concat_mla_q_zero_tokens():
    """Test with zero tokens (edge case)."""
    ql_nope = torch.empty(0, 128, 512, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.empty(0, 128, 64, dtype=torch.bfloat16, device="cuda")
    q_out = torch.empty(0, 128, 576, dtype=torch.bfloat16, device="cuda")

    ops.concat_mla_q(ql_nope, q_pe, q_out)


@pytest.mark.parametrize("num_tokens", [1, 64])
def test_concat_mla_q_values_preserved(num_tokens):
    """Verify exact bit-level preservation (no computation, pure copy).

    Compares raw int16 bits to avoid NaN != NaN issues from IEEE 754.
    """
    nope_dim, rope_dim = 512, 64

    # Use specific bit patterns (stay in int16 for bit-exact comparison)
    ql_nope_bits = torch.arange(
        num_tokens * 128 * nope_dim, dtype=torch.int16, device="cuda"
    ).view(num_tokens, 128, nope_dim)
    q_pe_bits = torch.arange(
        num_tokens * 128 * rope_dim, dtype=torch.int16, device="cuda"
    ).view(num_tokens, 128, rope_dim)

    ql_nope = ql_nope_bits.view(torch.bfloat16)
    q_pe = q_pe_bits.view(torch.bfloat16)

    q_out = torch.empty(
        num_tokens, 128, nope_dim + rope_dim, dtype=torch.bfloat16, device="cuda"
    )
    ops.concat_mla_q(ql_nope, q_pe, q_out)

    out_bits = q_out.view(torch.int16)

    assert torch.equal(out_bits[..., :nope_dim], ql_nope_bits)

    assert torch.equal(out_bits[..., nope_dim:], q_pe_bits)
