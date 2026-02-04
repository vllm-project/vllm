#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test PCP position computation using DualChunkSwap pattern."""

import numpy as np


def compute_pcp_positions(
    num_scheduled_tokens: np.ndarray,
    pcp_world_size: int,
    pcp_rank: int,
    reorder_batch_threshold: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCP positions for a given rank.

    Returns:
        pcp_tokens: tokens per request for this rank
        local_indices: indices into original batch for gathering
        positions: position values for this rank's tokens
    """
    num_reqs = len(num_scheduled_tokens)
    ws = pcp_world_size
    num_decode_reqs = int((num_scheduled_tokens <= reorder_batch_threshold).sum())

    # Pad prefill to 2*ws alignment; decode is duplicated
    padded = np.ceil(num_scheduled_tokens / (2 * ws)).astype(np.int32) * (2 * ws)
    padded[:num_decode_reqs] = num_scheduled_tokens[:num_decode_reqs] * ws

    pcp_tokens = padded // ws
    chunk = np.maximum(pcp_tokens // 2, 1)
    chunk[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]

    # Compute positions with DualChunkSwap
    positions = np.empty(int(pcp_tokens.sum()), dtype=np.int32)
    idx = 0
    for r in range(num_reqs):
        n, cs = pcp_tokens[r], chunk[r]
        if r < num_decode_reqs:
            positions[idx : idx + n] = np.arange(n)
        else:
            head = pcp_rank * cs
            tail = (2 * ws - pcp_rank - 1) * cs
            positions[idx : idx + cs] = np.arange(head, head + cs)
            positions[idx + cs : idx + n] = np.arange(tail, tail + cs)
        idx += n

    # Convert to indices into original batch
    cu_orig = np.cumsum(num_scheduled_tokens)
    orig_starts = np.concatenate([[0], cu_orig[:-1]])
    orig_lens = np.repeat(num_scheduled_tokens, pcp_tokens)
    orig_start = np.repeat(orig_starts, pcp_tokens)
    is_padding = positions >= orig_lens
    local_indices = np.where(is_padding, 0, orig_start + positions).astype(np.int64)

    return pcp_tokens[:num_reqs], local_indices, positions


def test_pcp_positions():
    """Test PCP position computation."""
    test_cases = [
        ("prefill_8_ws2", np.array([8]), 2),
        ("prefill_264_ws4", np.array([264]), 4),
        ("padded_260_ws4", np.array([260]), 4),
        ("multi_req_ws4", np.array([264, 128]), 4),
        ("decode_1_ws4", np.array([1]), 4),
        ("mixed_decode_prefill", np.array([1, 1, 64, 128]), 4),
    ]

    for name, num_scheduled, ws in test_cases:
        print(f"\n{name}: tokens={num_scheduled.tolist()}, ws={ws}")

        for rank in range(ws):
            pcp_tokens, local_indices, positions = compute_pcp_positions(
                num_scheduled, ws, rank
            )

            # Build per-request relative positions as global array
            global_positions = np.concatenate([np.arange(n) for n in num_scheduled])

            # Test: global_positions[local_indices] == positions (for non-padding)
            reconstructed = global_positions[local_indices]
            orig_lens = np.repeat(num_scheduled, pcp_tokens)
            non_padding = positions < orig_lens
            match = np.array_equal(reconstructed[non_padding], positions[non_padding])

            print(
                f"  rank={rank}: pcp_tokens={pcp_tokens.tolist()}, "
                f"positions[:5]={positions[:5].tolist()}, match={match}"
            )
            assert match, f"Position mismatch for {name} rank {rank}"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_pcp_positions()
