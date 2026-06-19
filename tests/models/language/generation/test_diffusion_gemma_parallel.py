# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DiffusionGemma tensor-parallel self-conditioning.

Regression test for https://github.com/vllm-project/vllm/issues/45719: under
tensor parallelism the self-conditioning soft embedding
(``probs @ embed_tokens.weight``) multiplied a full-vocab ``probs`` by a
vocab-parallel-sharded ``embed_tokens.weight`` (``[vocab/tp, hidden]``), so the
matmul reduction dims mismatched and warmup crashed. The fix multiplies each
rank's vocab slice and all-reduces the partial products, never materializing
the full embedding weight.

These tests don't need the 26B checkpoint: the corrected computation is a pure
algebraic identity (sum of vocab-sharded matmuls == the full matmul) that runs
on CPU.
"""

import pytest
import torch

from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    pad_vocab_size,
    vocab_range_from_global_vocab_size,
)


def _org_vocab_shards(vocab_size: int, tp_size: int) -> list[tuple[int, int]]:
    """Reproduce VocabParallelEmbedding's per-rank org-vocab ranges.

    Mirrors ``VocabParallelEmbeddingShardIndices``: the padded vocab is tiled
    evenly across ranks, then each rank's range is clamped to the (unpadded)
    org vocab. Returns each rank's ``[org_vocab_start, org_vocab_end)`` — the
    exact slice ``DiffusionSampler`` passes as ``vocab_start``/``vocab_end``.
    """
    padded = pad_vocab_size(vocab_size, DEFAULT_VOCAB_PADDING_SIZE)
    shards = []
    for rank in range(tp_size):
        p_start, p_end = vocab_range_from_global_vocab_size(padded, rank, tp_size)
        shards.append((min(p_start, vocab_size), min(p_end, vocab_size)))
    return shards


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_self_conditioning_shards_tile_full_vocab(tp_size: int):
    # vocab deliberately not a multiple of tp_size to exercise padding/clamping.
    vocab_size = 262144 - 7
    shards = _org_vocab_shards(vocab_size, tp_size)

    # Shards must tile [0, vocab_size) contiguously with no gaps/overlaps, so
    # that summing per-rank slices of probs covers the full distribution.
    assert shards[0][0] == 0
    assert shards[-1][1] == vocab_size
    for (_, end), (next_start, _) in zip(shards, shards[1:]):
        assert end == next_start
    assert sum(e - s for s, e in shards) == vocab_size


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_self_conditioning_vocab_shard_matmul_matches_full(tp_size: int):
    """Sum of per-rank sharded matmuls == the full-vocab matmul.

    This is exactly the computation in ``_compiled_sample_step``:
    each rank computes ``probs[..., start:end] @ embed_weight[: end - start]``
    and the partials are summed (the all-reduce). The result must equal the
    full ``probs @ embed_weight`` regardless of ``tp_size`` (tp=1 is the
    original, unsharded path).
    """
    torch.manual_seed(0)
    num_tokens, hidden = 6, 32
    vocab_size = 1024 - 5  # non-multiple of tp_size on purpose

    probs = torch.randn(num_tokens, vocab_size).softmax(dim=-1)
    embed_weight = torch.randn(vocab_size, hidden)

    full = probs @ embed_weight

    reduced = torch.zeros(num_tokens, hidden)
    for start, end in _org_vocab_shards(vocab_size, tp_size):
        # embed_weight[: end - start] mirrors the rank-local shard rows.
        reduced = reduced + probs[..., start:end] @ embed_weight[start:end]

    torch.testing.assert_close(reduced, full, atol=1e-4, rtol=1e-4)
