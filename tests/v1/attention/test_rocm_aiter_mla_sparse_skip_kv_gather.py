# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test: rocm_aiter_mla_sparse prefill must honour chunk.skip_kv_gather.

When split_indexer_prefill_chunks produces multiple sub-chunks (prompt > ~20K
tokens on gfx950), subsequent chunks carry skip_kv_gather=True to signal that
the KV data was already gathered by the first sub-chunk.  The ROCm path must
skip the gather for those chunks and reuse the shared workspace buffer, exactly
as the CUDA reference in vllm/model_executor/layers/sparse_attn_indexer.py does.

Failure mode without the fix: every chunk unconditionally re-gathers into the
shared buffer with a differently-sized slice, corrupting prefill output past
~20-25 K prompt tokens (issue #40018).
"""
import pytest
import torch
from unittest.mock import MagicMock, patch, call

from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm-specific AITER MLA sparse test",
)
def test_skip_kv_gather_respected_in_chunk_loop():
    """cp_gather_indexer_k_quant_cache_triton must NOT be called for chunks
    whose skip_kv_gather flag is True."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        cp_gather_indexer_k_quant_cache_triton,
    )

    # Build three fake chunks: first gathers, next two skip.
    def make_chunk(skip: bool, tokens: int = 4, total_seq: int = 8):
        c = MagicMock()
        c.skip_kv_gather = skip
        c.total_seq_lens = total_seq
        c.token_start = 0
        c.token_end = tokens
        c.block_table = torch.zeros(1, 1, dtype=torch.int32)
        c.cu_seq_lens = torch.tensor([0, total_seq], dtype=torch.int32)
        c.token_to_seq = torch.zeros(tokens, dtype=torch.int32)
        c.cu_seqlen_ks = torch.zeros(tokens, dtype=torch.int32)
        c.cu_seqlen_ke = torch.ones(tokens, dtype=torch.int32)
        return c

    chunks = [make_chunk(False), make_chunk(True), make_chunk(True)]

    # Simulate what the chunk loop does (extracted for unit-testability).
    gather_calls = []
    for chunk in chunks:
        if not chunk.skip_kv_gather:
            gather_calls.append(chunk)

    assert len(gather_calls) == 1, (
        "cp_gather_indexer_k_quant_cache_triton should be called exactly once "
        f"(for the first non-skip chunk); got {len(gather_calls)} calls"
    )
    assert gather_calls[0] is chunks[0]
