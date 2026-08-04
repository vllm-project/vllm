# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the multi-token decode flatten in ``TritonMLAImpl.forward_mqa``.

``decode_attention_fwd`` has no causal flag: it launches one program per row of
``q`` and reads that row's KV extent from ``B_Seqlen``. So a ``query_len``-token
decode block is flattened to one row per query token, and intra-block causality
has to be expressed through the per-row sequence lengths -- row ``t`` gets
``seq_len - (query_len - 1) + t``, i.e. the committed prefix plus block tokens
``0..t``. A non-causal draft block instead gives every row the same full extent.

The kernel is replaced by a spy, so these tests assert on the ``block_table``
and ``seq_lens`` the backend actually submits and need no GPU or Triton runtime.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import QueryLenSupport
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLAImpl,
    TritonMLAMetadataBuilder,
)

NUM_HEADS = 16
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
PAGE_SIZE = 16

# Committed context per request, including a fresh request at zero context
# where the whole KV extent is the block itself. A decode block's seq_len
# already counts the block, so seq_len = context + query_len.
CONTEXT_LENS = [0, 1, 37, 512]


def _seq_lens(query_len: int) -> list[int]:
    return [c + query_len for c in CONTEXT_LENS]


def _make_impl() -> TritonMLAImpl:
    """Only the attributes ``forward_mqa`` reads; __init__ wants a full config."""
    impl = object.__new__(TritonMLAImpl)
    impl.kv_lora_rank = KV_LORA_RANK
    impl.scale = HEAD_SIZE**-0.5
    impl._sm_count = 304
    return impl


def _make_metadata(seq_lens: list[int], query_len: int, causal: bool):
    num_decodes = len(seq_lens)
    return SimpleNamespace(
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes * query_len,
        max_seq_len=max(seq_lens),
        causal=causal,
        decode=SimpleNamespace(
            # arange page ids so each request's window is an identifiable slice.
            block_table=torch.arange(num_decodes * 64, dtype=torch.int32).reshape(
                num_decodes, 64
            ),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        ),
    )


def _run_forward_mqa(seq_lens: list[int], query_len: int, causal: bool, q_rows=None):
    """Drive forward_mqa with the decode kernel spied out.

    Returns the ``(block_table, seq_lens)`` handed to ``decode_attention_fwd``.
    """
    captured: dict = {}

    def spy(q, kv_cache, kv_c_cache, o, lse, block_table, b_seq_len, *args, **kwargs):
        captured["block_table"] = block_table.detach().clone()
        captured["seq_lens"] = b_seq_len.detach().clone()

    metadata = _make_metadata(seq_lens, query_len, causal)
    rows = metadata.num_decode_tokens if q_rows is None else q_rows
    q = torch.zeros(rows, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16)
    kv_cache = torch.zeros(
        len(seq_lens) * 64, PAGE_SIZE, HEAD_SIZE, dtype=torch.bfloat16
    )
    layer = SimpleNamespace(_k_scale=torch.ones(1), layer_name="test")

    with patch("vllm.v1.attention.backends.mla.triton_mla.decode_attention_fwd", spy):
        _make_impl().forward_mqa(q, kv_cache, metadata, layer)

    assert captured, "forward_mqa did not reach the decode kernel"
    return captured


def test_flag_pair_moves_together():
    """query_len_support gates supports_spec_decode, which raises
    reorder_batch_threshold; UNIFORM_BATCH capture probes max_query_len > 1."""
    if TritonMLAMetadataBuilder._cudagraph_support == AttentionCGSupport.UNIFORM_BATCH:
        assert TritonMLAMetadataBuilder.query_len_support != QueryLenSupport.SINGLE_ONLY


@pytest.mark.parametrize("query_len", [2, 3, 4, 5, 8])
def test_causal_block_rows_see_prefix_plus_own_position(query_len):
    """Causal verify: row t sees seq_len - (query_len - 1) + t KV entries.

    Regression guard: the causal branch used to be skipped entirely, so a verify
    token could attend the draft siblings it is supposed to be checking. Fails
    on unmodified upstream, where no flatten happens at all for causal blocks.
    """
    seq_lens = _seq_lens(query_len)
    captured = _run_forward_mqa(seq_lens, query_len, causal=True)

    got = captured["seq_lens"].tolist()
    want = [s - (query_len - 1) + t for s in seq_lens for t in range(query_len)]
    assert got == want, (
        "causal flatten is not causal: verify row r*query_len+t must get "
        f"seq_len_r - {query_len - 1} + t entries.\n"
        f"  seq_lens {seq_lens}\n  got      {got}\n  expected {want}"
    )

    # Per-request invariants the arithmetic must hold regardless of query_len.
    for r, seq_len in enumerate(seq_lens):
        rows = got[r * query_len : (r + 1) * query_len]
        assert rows == sorted(rows) and len(set(rows)) == query_len, (
            f"request {r} rows must strictly increase, got {rows}"
        )
        assert rows[-1] == seq_len, "last row must see the full sequence"
        assert rows[0] == seq_len - (query_len - 1)
        assert all(n > 0 for n in rows), f"non-positive KV extent in {rows}"


def test_cudagraph_padding_rows_clamp_to_empty():
    """Padding requests carry seq_len 0, so the causal tail drives their rows
    negative. The kernel skips a row whose extent is not positive
    (``if split_kv_end > split_kv_start``), so this is a no-op rather than a
    bad read -- but only as long as the padding stays in that branch."""
    query_len = 4
    seq_lens = _seq_lens(query_len) + [0]
    captured = _run_forward_mqa(seq_lens, query_len, causal=True)

    padding = captured["seq_lens"].tolist()[-query_len:]
    assert all(n <= 0 for n in padding), (
        f"padding rows must not present a positive KV extent, got {padding}"
    )


@pytest.mark.parametrize("query_len", [2, 4])
def test_non_causal_block_rows_all_see_the_full_prefix(query_len):
    """The DSpark draft block is generated in one pass, so no row is masked."""
    seq_lens = _seq_lens(query_len)
    captured = _run_forward_mqa(seq_lens, query_len, causal=False)

    want = [s for s in seq_lens for _ in range(query_len)]
    assert captured["seq_lens"].tolist() == want


@pytest.mark.parametrize("causal", [True, False])
def test_block_table_is_expanded_to_match_q_rows(causal):
    """The kernel indexes block_table by program_id, so it needs one row per
    query token -- a short block_table is read out of bounds."""
    query_len = 4
    captured = _run_forward_mqa(_seq_lens(query_len), query_len, causal=causal)

    num_rows = len(CONTEXT_LENS) * query_len
    assert captured["block_table"].shape[0] == num_rows
    assert captured["seq_lens"].shape[0] == num_rows
    # repeat_interleave, not repeat: a request's rows must stay adjacent.
    for r in range(len(CONTEXT_LENS)):
        block = captured["block_table"][r * query_len : (r + 1) * query_len]
        assert torch.equal(block, block[:1].expand_as(block))


@pytest.mark.parametrize("causal", [True, False])
def test_single_token_decode_is_untouched(causal):
    """query_len == 1 must short-circuit: ordinary decode sees no expansion."""
    seq_lens = _seq_lens(1)
    captured = _run_forward_mqa(seq_lens, query_len=1, causal=causal)

    assert captured["seq_lens"].tolist() == seq_lens
    assert captured["block_table"].shape[0] == len(seq_lens)


def test_row_count_mismatch_is_reported(caplog_vllm):
    """A short block_table surfaces as an opaque memory fault inside the kernel,
    so the mismatch is reported at the launch site while it is attributable."""
    # More q rows than the metadata accounts for, i.e. the shape that faulted.
    _run_forward_mqa(
        _seq_lens(1), query_len=1, causal=True, q_rows=len(CONTEXT_LENS) * 4
    )

    assert "[MQAOOB]" in caplog_vllm.text


def test_no_warning_when_rows_match(caplog_vllm):
    _run_forward_mqa(_seq_lens(4), query_len=4, causal=True)

    assert "[MQAOOB]" not in caplog_vllm.text
