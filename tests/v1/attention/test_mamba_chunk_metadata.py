# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for BaseMambaAttentionMetadataBuilder._compute_chunk_metadata.

The optimized implementation must produce exactly the same chunking as
the original implementation, which is kept here as the reference.
"""

import random

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder

_builder = object.__new__(BaseMambaAttentionMetadataBuilder)
compute_chunk_metadata = _builder._compute_chunk_metadata


def _reference_compute_chunk_metadata(
    chunk_size: int,
    num_prefills: int,
    num_computed_tokens_p_cpu: torch.Tensor,
    query_start_loc_p_cpu: torch.Tensor,
) -> tuple[list[int], list[int], list[int]]:
    """Original implementation, kept verbatim as the reference."""
    cu_chunk_seqlen = []
    seq_idx = []
    last_chunk_indices = []
    seqlen_pos = 0

    for req_idx in range(num_prefills):
        this_num_computed = num_computed_tokens_p_cpu[req_idx].item()
        this_new_tokens = (
            query_start_loc_p_cpu[req_idx + 1].item()
            - query_start_loc_p_cpu[req_idx].item()
        )

        # if computed tokens are not chunk-aligned, use the first
        # chunk to finish it off
        if this_num_computed % chunk_size != 0:
            seq_idx.append(req_idx)
            cu_chunk_seqlen.append(seqlen_pos)
            chunk_len = (
                cdiv(this_num_computed, chunk_size) * chunk_size - this_num_computed
            )
            chunk_len = min(chunk_len, this_new_tokens)
            seqlen_pos += chunk_len
            this_new_tokens -= chunk_len

        n_chunks = cdiv(this_new_tokens, chunk_size)
        for _ in range(n_chunks):
            seq_idx.append(req_idx)
            cu_chunk_seqlen.append(seqlen_pos)
            chunk_len = min(chunk_size, this_new_tokens)
            seqlen_pos += chunk_len
            this_new_tokens -= chunk_len

        assert this_new_tokens == 0
        last_chunk_indices.append(len(cu_chunk_seqlen) - 1)

    cu_chunk_seqlen.append(seqlen_pos)

    return cu_chunk_seqlen, seq_idx, last_chunk_indices


def _make_workload(
    rng: random.Random,
    num_prefills: int,
    chunk_size: int,
    aligned: bool | None = None,
    max_new_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_computed = []
    query_start_loc = [0]
    for _ in range(num_prefills):
        if aligned is True:
            computed = rng.randint(0, 8) * chunk_size
        elif aligned is False:
            computed = rng.randint(0, 8) * chunk_size + rng.randint(1, chunk_size - 1)
        else:
            computed = rng.randint(0, 4 * chunk_size)
        num_computed.append(computed)
        new_tokens = rng.randint(1, max_new_tokens or 6 * chunk_size)
        query_start_loc.append(query_start_loc[-1] + new_tokens)
    return (
        torch.tensor(num_computed, dtype=torch.int32),
        torch.tensor(query_start_loc, dtype=torch.int32),
    )


def _assert_matches_reference(
    chunk_size: int,
    num_prefills: int,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    ref = _reference_compute_chunk_metadata(
        chunk_size, num_prefills, num_computed_tokens, query_start_loc
    )
    out = compute_chunk_metadata(
        chunk_size, num_prefills, num_computed_tokens, query_start_loc
    )
    for name, ref_val, out_val in zip(
        ("cu_chunk_seqlen", "seq_idx", "last_chunk_indices"), ref, out
    ):
        assert list(out_val) == ref_val, (
            f"{name} mismatch for chunk_size={chunk_size} num_prefills={num_prefills}"
        )


@pytest.mark.parametrize("chunk_size", [8, 64, 256, 2048])
@pytest.mark.parametrize("num_prefills", [1, 2, 3, 8, 32, 128])
@pytest.mark.parametrize("aligned", [None, True, False])
def test_random_workloads(chunk_size: int, num_prefills: int, aligned: bool | None):
    """Randomized batches; alignment mode steers the chunk-realign branch."""
    rng = random.Random(chunk_size * 1000 + num_prefills)
    for _ in range(10):
        num_computed, query_start_loc = _make_workload(
            rng, num_prefills, chunk_size, aligned=aligned
        )
        _assert_matches_reference(
            chunk_size, num_prefills, num_computed, query_start_loc
        )


@pytest.mark.parametrize("chunk_size", [64, 256])
def test_tiny_new_tokens(chunk_size: int):
    """New tokens may not even fill the realigning partial chunk."""
    rng = random.Random(0)
    for _ in range(10):
        num_computed, query_start_loc = _make_workload(
            rng, 16, chunk_size, aligned=False, max_new_tokens=3
        )
        _assert_matches_reference(chunk_size, 16, num_computed, query_start_loc)


def test_exact_chunk_boundaries():
    """All computed tokens land exactly on chunk boundaries."""
    num_computed = torch.tensor([0, 256, 512], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 256, 768, 1024], dtype=torch.int32)
    _assert_matches_reference(256, 3, num_computed, query_start_loc)


def test_zero_prefills():
    """An empty batch degenerates to cu_chunk_seqlen == [0]."""
    num_computed = torch.tensor([], dtype=torch.int32)
    query_start_loc = torch.tensor([0], dtype=torch.int32)
    _assert_matches_reference(256, 0, num_computed, query_start_loc)
