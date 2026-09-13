# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dispatch rules for the FlyDSL prefill attention backend.

Each case is a shape measured on MI350X against the AITER unified attention
kernel it falls back to; the speedup in the comment is that measurement. The
gate must admit every shape that wins and reject every shape that loses.
"""

import pytest

from vllm.v1.attention.backends.rocm_flydsl_attn import (
    FLYDSL_BLOCK_M,
    FLYDSL_MIN_QBLOCK_ROWS,
    FLYDSL_MIN_QUERY_LEN,
    FLYDSL_MIN_SEQ_LEN,
    _max_decode_rows,
)

NUM_HEADS = 16


def _eligible(query_lens, max_seq_len):
    """The shape-only half of RocmFlyDSLAttentionImpl._flydsl_eligible."""
    num_tokens = sum(query_lens)
    return (
        max(query_lens) >= FLYDSL_MIN_QUERY_LEN
        and NUM_HEADS * num_tokens >= FLYDSL_MIN_QBLOCK_ROWS * FLYDSL_BLOCK_M
        and max_seq_len >= FLYDSL_MIN_SEQ_LEN
        and sum(1 for q in query_lens if q == 1) <= _max_decode_rows(max_seq_len)
    )


def _mixed(prefill, num_decode):
    return [prefill] + [1] * num_decode


@pytest.mark.parametrize(
    "query_lens,max_seq_len",
    [
        ([2048], 8192),  # 1.51x
        ([8192], 8192),  # 1.79x
        ([32768], 61440),  # 1.50x
        ([4096] * 2, 8192),  # 1.68x
        ([1024] * 8, 8192),  # 1.59x
        ([1024] * 8, 1536),  # 1.09x, just past the crossover
        ([256] * 32, 1536),  # 1.20x
        (_mixed(8192, 8), 8192),  # 1.11x, at the cap
        (_mixed(32768, 15), 61440),  # 1.29x, cap scales with context
    ],
)
def test_admits_measured_wins(query_lens, max_seq_len):
    assert _eligible(query_lens, max_seq_len)


@pytest.mark.parametrize(
    "query_lens,max_seq_len",
    [
        ([1024] * 8, 1024),  # 0.75x
        ([512] * 16, 1024),  # 0.85x
        ([1024] * 8, 1408),  # 0.98x, last losing context
        ([1024], 1024),  # 0.34x, worst measured
        (_mixed(8192, 16), 8192),  # 0.82x, one past the cap
        (_mixed(8192, 32), 8192),  # 0.53x
        (_mixed(4096, 16), 8192),  # 0.85x
    ],
)
def test_rejects_measured_losses(query_lens, max_seq_len):
    assert not _eligible(query_lens, max_seq_len)


def test_decode_row_cap_scales_with_context():
    """The cap tracks context depth: a deeper KV walk amortises more rows."""
    assert _max_decode_rows(8192) == 8
    assert _max_decode_rows(16384) == 16
    assert _max_decode_rows(61440) == 56


def test_decode_row_cap_has_a_floor():
    """Contexts below one unit still tolerate the base count, never zero."""
    assert _max_decode_rows(FLYDSL_MIN_SEQ_LEN) == 8
