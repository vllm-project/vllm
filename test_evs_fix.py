#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple harness to reason about EVS placeholder offsets.

The real implementation in ``iter_mm_grid_hw`` now relies on the
``is_embed`` mask stored in ``mm_position`` to recover the start offset of
each frame instead of scanning ``input_tokens``.  This script mirrors that
behaviour so we can validate that sparse EVS retention patterns (e.g. first
frame fully kept, other frames pruned unevenly) are still handled.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass
class MaskSimulationConfig:
    """Helper configuration for generating placeholder masks.

    ``prefix_tokens`` and ``suffix_tokens`` approximate the extra tokens that
    surround the `<|video_pad|>` sequence for each frame (timestamps,
    `<|vision_start|>`, `<|vision_end|>`, etc.).  ``tokens_per_frame`` encodes
    how many `<|video_pad|>` tokens survive EVS for each frame.  This lets us
    mimic both balanced and sparse retention distributions.
    """

    tokens_per_frame: list[int]
    prefix_tokens: int = 2
    suffix_tokens: int = 1


def build_is_embed_mask(cfg: MaskSimulationConfig) -> torch.Tensor:
    mask: list[int] = []
    for tokens in cfg.tokens_per_frame:
        mask.extend([0] * cfg.prefix_tokens)
        mask.extend([1] * tokens)
        mask.extend([0] * cfg.suffix_tokens)
    return torch.tensor(mask, dtype=torch.bool)


def extract_frame_offsets(
    offset_start: int, mask: torch.Tensor, expected_frames: int
) -> tuple[list[int], list[int]]:
    """Mimic the EVS branch in ``iter_mm_grid_hw``.

    We compute the first index of each contiguous run of ``True`` values,
    convert it back to an absolute offset using ``offset_start`` and return
    both the offsets and the corresponding run lengths.
    """

    flat_mask = mask.reshape(-1).to(torch.bool)
    true_indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
    if true_indices.numel() == 0:
        raise ValueError("Mask does not contain any embed tokens")

    if true_indices.numel() == 1:
        segments: Iterable[torch.Tensor] = (true_indices,)
    else:
        diffs = true_indices[1:] - true_indices[:-1]
        split_points = (
            torch.nonzero(diffs != 1, as_tuple=False).flatten().add(1).tolist()
        )
        segments = torch.tensor_split(true_indices, split_points)

    segments = list(segments)
    if len(segments) < expected_frames:
        raise ValueError(
            f"Expected {expected_frames} frame segments, got {len(segments)}"
        )

    offsets = [
        offset_start + int(segment[0].item()) for segment in segments[:expected_frames]
    ]
    lengths = [int(segment.numel()) for segment in segments[:expected_frames]]
    return offsets, lengths


def test_sparse_distribution() -> None:
    print("\n=== 测试场景 1: 稀疏分布 (真实 EVS 行为) ===")
    per_frame = [50176, 15000, 12000, 10000, 8000, 145668, 5000, 5000]
    cfg = MaskSimulationConfig(
        tokens_per_frame=per_frame, prefix_tokens=3, suffix_tokens=2
    )
    mask = build_is_embed_mask(cfg)
    offsets, lengths = extract_frame_offsets(128, mask, len(per_frame))

    for idx, (off, size, expected) in enumerate(zip(offsets, lengths, per_frame), 1):
        print(
            f"Frame {idx:02d}: offset={off:6d}, retained={size:6d} tokens "
            f"(expected {expected})"
        )
        assert size == expected

    print("✅ 稀疏分布模拟通过")


def test_uniform_distribution() -> None:
    print("\n=== 测试场景 2: 均匀分布 (处理器当前实现) ===")
    per_frame = [784 for _ in range(4)]
    cfg = MaskSimulationConfig(
        tokens_per_frame=per_frame, prefix_tokens=2, suffix_tokens=1
    )
    mask = build_is_embed_mask(cfg)
    offsets, lengths = extract_frame_offsets(42, mask, len(per_frame))

    expected_offsets: list[int] = []
    cursor = 42
    for tokens in per_frame:
        cursor += cfg.prefix_tokens
        expected_offsets.append(cursor)
        cursor += tokens + cfg.suffix_tokens

    for idx, (off, size, expected_offset) in enumerate(
        zip(offsets, lengths, expected_offsets), 1
    ):
        print(f"Frame {idx:02d}: offset={off:5d}, retained={size:4d} tokens")
        assert size == per_frame[idx - 1]
        assert off == expected_offset

    print("✅ 均匀分布模拟通过")


if __name__ == "__main__":
    test_sparse_distribution()
    test_uniform_distribution()
    print("\n所有 EVS 相关测试通过 ✅")
