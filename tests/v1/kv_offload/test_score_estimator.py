# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for attention score estimation and mapping.
"""
import torch

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.score_estimator import (
    compute_block_scores_from_hidden_states,
    map_scores_to_block_hashes,
)


def to_hash(i: int) -> BlockHash:
    return BlockHash(str(i).encode())


def to_hashes(ints: list[int]) -> list[BlockHash]:
    return [to_hash(i) for i in ints]


def test_compute_block_scores_basic():
    """Compute scores from uniform hidden states."""
    hidden_size = 64
    block_size = 4
    # 2 requests: req1 has 8 tokens (2 blocks), req2 has 4 tokens (1 block)
    total_tokens = 12
    hidden_states = torch.ones(total_tokens, hidden_size)

    num_scheduled = {"req1": 8, "req2": 4}

    scores = compute_block_scores_from_hidden_states(
        hidden_states, num_scheduled, block_size
    )

    assert "req1" in scores
    assert "req2" in scores
    assert len(scores["req1"]) == 2  # 8 tokens / 4 block_size
    assert len(scores["req2"]) == 1  # 4 tokens / 4 block_size

    # All ones with hidden_size=64: norm = sqrt(64) = 8.0
    expected_norm = (64 ** 0.5)
    for score in scores["req1"]:
        assert abs(score - expected_norm) < 0.01
    for score in scores["req2"]:
        assert abs(score - expected_norm) < 0.01


def test_compute_block_scores_varying_magnitude():
    """Blocks with higher magnitude hidden states get higher scores."""
    hidden_size = 64
    block_size = 4

    # 1 request, 8 tokens, 2 blocks
    # Block 0: low magnitude (0.1), Block 1: high magnitude (10.0)
    hidden_states = torch.zeros(8, hidden_size)
    hidden_states[:4] = 0.1  # block 0
    hidden_states[4:] = 10.0  # block 1

    scores = compute_block_scores_from_hidden_states(
        hidden_states, {"req1": 8}, block_size
    )

    assert len(scores["req1"]) == 2
    # Block 1 should have much higher score than block 0
    assert scores["req1"][1] > scores["req1"][0] * 10


def test_compute_block_scores_empty():
    """Handle edge cases gracefully."""
    # None hidden states
    scores = compute_block_scores_from_hidden_states(
        None, {"req1": 8}, 4
    )
    assert scores == {}

    # 1D tensor
    scores = compute_block_scores_from_hidden_states(
        torch.ones(8), {"req1": 8}, 4
    )
    assert scores == {}

    # Empty scheduled tokens
    scores = compute_block_scores_from_hidden_states(
        torch.ones(8, 64), {}, 4
    )
    assert scores == {}


def test_compute_block_scores_partial_block():
    """Handle requests with non-block-aligned token counts."""
    hidden_size = 64
    block_size = 4

    # 5 tokens = 1 full block + 1 partial block
    hidden_states = torch.ones(5, hidden_size)
    scores = compute_block_scores_from_hidden_states(
        hidden_states, {"req1": 5}, block_size
    )

    assert len(scores["req1"]) == 2  # ceil(5/4) = 2 blocks


def test_map_scores_to_block_hashes_basic():
    """Map positional scores to block hashes."""
    per_request_scores = {
        "req1": [1.0, 2.0, 3.0],
    }
    request_block_hashes = {
        "req1": to_hashes([10, 20, 30]),
    }

    result = map_scores_to_block_hashes(
        per_request_scores, request_block_hashes
    )

    assert result[to_hash(10)] == 1.0
    assert result[to_hash(20)] == 2.0
    assert result[to_hash(30)] == 3.0


def test_map_scores_max_aggregation():
    """Same block_hash from different requests takes max score."""
    per_request_scores = {
        "req1": [1.0, 5.0],
        "req2": [3.0, 2.0],
    }
    # Both requests share block hash 10
    request_block_hashes = {
        "req1": to_hashes([10, 20]),
        "req2": to_hashes([10, 30]),
    }

    result = map_scores_to_block_hashes(
        per_request_scores, request_block_hashes
    )

    # Block 10: max(1.0, 3.0) = 3.0
    assert result[to_hash(10)] == 3.0
    assert result[to_hash(20)] == 5.0
    assert result[to_hash(30)] == 2.0


def test_map_scores_with_block_size_factor():
    """Handle block_size_factor > 1 (multiple GPU blocks per offloaded block)."""
    per_request_scores = {
        "req1": [1.0, 2.0, 3.0, 4.0],  # 4 GPU blocks
    }
    # block_size_factor=2: GPU blocks 0,1 -> offloaded block 0 (hash 10)
    #                       GPU blocks 2,3 -> offloaded block 1 (hash 20)
    request_block_hashes = {
        "req1": to_hashes([10, 20]),
    }

    result = map_scores_to_block_hashes(
        per_request_scores, request_block_hashes, block_size_factor=2
    )

    # Block 10: max(1.0, 2.0) = 2.0
    # Block 20: max(3.0, 4.0) = 4.0
    assert result[to_hash(10)] == 2.0
    assert result[to_hash(20)] == 4.0


def test_map_scores_missing_request():
    """Requests not in block_hashes mapping are skipped."""
    per_request_scores = {
        "req1": [1.0],
        "req2": [2.0],  # not in block_hashes
    }
    request_block_hashes = {
        "req1": to_hashes([10]),
    }

    result = map_scores_to_block_hashes(
        per_request_scores, request_block_hashes
    )

    assert to_hash(10) in result
    assert len(result) == 1
