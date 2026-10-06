# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-block KV digest (checksum) helpers for the NIXL connector.

Prototype integrity check for pull-mode P/D disaggregation: the producer
digests the blocks it exposes, the consumer re-digests the destination
blocks after its NIXL READ completes and compares.
"""

import torch

_MASK64 = (1 << 64) - 1


def digest_block_pages(
    pages: torch.Tensor, block_ids: list[int]
) -> list[tuple[int, int]]:
    """Compute a 128-bit order-sensitive digest for each given block.

    Args:
        pages: uint8 view of one NIXL region with shape
            (num_blocks, block_len); row i must cover exactly the bytes NIXL
            transfers for block i.
        block_ids: Blocks to digest.

    Returns:
        One (h1, h2) pair per block. h1 is the wrapped uint64 sum of the
        block's int64 lanes; h2 is the wrapped uint64 sum of lane *
        lane_index. Bytes past the last full 8-byte lane are ignored.

    """
    if not block_ids:
        return []
    idx = torch.tensor(block_ids, device=pages.device, dtype=torch.long)
    rows = pages.index_select(0, idx)
    n_bytes = rows.shape[1] // 8 * 8
    lanes = rows[:, :n_bytes].contiguous().view(torch.int64)
    # int64 wrap-around arithmetic is bit-identical to uint64.
    h1 = lanes.sum(dim=1)
    weights = torch.arange(lanes.shape[1], device=pages.device, dtype=torch.int64)
    h2 = (lanes * weights).sum(dim=1)
    return [(a & _MASK64, b & _MASK64) for a, b in zip(h1.tolist(), h2.tolist())]


def merge_digests(
    parts: list[list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """Combine per-region digests block-wise into one digest per block."""
    acc_h1 = [0] * len(parts[0])
    acc_h2 = [0] * len(parts[0])
    for part in parts:
        for i, (p1, p2) in enumerate(part):
            acc_h1[i] = (acc_h1[i] + p1) & _MASK64
            acc_h2[i] = (acc_h2[i] + p2) & _MASK64
    return list(zip(acc_h1, acc_h2))


def serialize_digest(digest: tuple[int, int]) -> str:
    """Serialize a digest as a JSON-safe "h1_h2" hex string."""
    return f"{digest[0]:x}_{digest[1]:x}"
