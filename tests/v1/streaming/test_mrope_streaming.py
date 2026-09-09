# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only unit tests for the mRoPE streaming helpers.

These tests verify the position-arithmetic invariants of
`compute_chunk_mrope_positions` and `make_chunk_relative_mm_features`
without needing a real Qwen3-VL model: we drive them with a tiny stub
model that implements the `SupportsMRoPE` protocol with a deterministic,
trivially-checkable algorithm.

The parity properties exercised here:

1. `compute_chunk_mrope_positions` shifts positions by exactly
   `base_position` (its first column equals `base_position`).
2. Computing positions chunk-by-chunk produces a tensor identical to
   computing positions once over the cumulative prompt, as long as
   chunk-relative offsets are computed correctly and `base_position` is
   advanced by `prev_max + 1`. This is the load-bearing invariant for
   Piece 1's no-eviction case.
3. After eviction (Piece 3 territory but already simulatable here),
   surviving positions retain their original values, and new chunks get
   positions strictly greater than any surviving position.
"""

from dataclasses import dataclass, field

import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.streaming.mrope import (
    compute_chunk_mrope_positions,
    make_chunk_relative_mm_features,
)

# Synthetic token IDs. Anything works; the stub model below doesn't actually
# care about token semantics, only about mm_position offsets.
TEXT_TOKEN = 100
VIDEO_PAD_TOKEN = 200


@dataclass
class StubMRoPEModel:
    """Tiny model implementing `SupportsMRoPE` for tests.

    For each `MultiModalFeatureSpec`, we assign a 3x4 block of positions
    starting at the next available row. Text tokens get linear `(p, p, p)`
    positions, like Qwen3-VL's standard rope path. The algorithm is
    intentionally simpler than Qwen3-VL's but exercises the same call
    interface and the same prefix-deterministic property.
    """

    supports_mrope: bool = field(default=True, init=False)
    tokens_per_video: int = 4

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        # Walk tokens left-to-right. For each token, if it's inside any
        # mm_feature's range, assign a video-grid position; else a text one.
        # video positions: t advances per feature, h/w cycle 0..1 in a 2x2 grid.
        positions: list[tuple[int, int, int]] = []
        feature_for_index: dict[int, int] = {}
        for fi, feat in enumerate(mm_features):
            start = feat.mm_position.offset
            length = feat.mm_position.length
            for i in range(start, start + length):
                feature_for_index[i] = fi

        cursor = 0
        for idx, _tok in enumerate(input_tokens):
            if idx in feature_for_index:
                fi = feature_for_index[idx]
                # Stable 2x2 grid layout: (t=fi-offset-into-stream, h, w)
                within_feature = idx - mm_features[fi].mm_position.offset
                h = within_feature // 2
                w = within_feature % 2
                t = cursor
                positions.append((t, cursor + h, cursor + w))
                # Advance cursor by feature length at end of feature.
                if within_feature == mm_features[fi].mm_position.length - 1:
                    cursor += 2
            else:
                positions.append((cursor, cursor, cursor))
                cursor += 1
        tensor = torch.tensor(positions, dtype=torch.long).t().contiguous()
        max_val = int(tensor.max().item()) if tensor.numel() > 0 else -1
        return tensor, max_val + 1 - len(input_tokens)


def _make_video_feature(offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="video",
        identifier=f"vid-{offset}",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def test_compute_chunk_shifts_by_base_position():
    model = StubMRoPEModel()
    chunk_tokens = [TEXT_TOKEN, TEXT_TOKEN, TEXT_TOKEN]
    base = 50

    positions, new_max = compute_chunk_mrope_positions(
        model, chunk_tokens, [], base_position=base
    )

    assert positions.shape == (3, 3)
    # Text-only chunk: first column should be (base, base, base).
    assert positions[:, 0].tolist() == [base, base, base]
    assert positions[:, 1].tolist() == [base + 1, base + 1, base + 1]
    assert positions[:, 2].tolist() == [base + 2, base + 2, base + 2]
    assert new_max == base + 2


def test_compute_chunk_empty_tokens_returns_sentinel():
    model = StubMRoPEModel()
    positions, new_max = compute_chunk_mrope_positions(model, [], [], base_position=10)
    assert positions.shape == (3, 0)
    # `new_max` for an empty chunk should leave `max_cached_position` unchanged
    # when the caller does `max_cached_position = new_max`. Returning
    # `base_position - 1` achieves that.
    assert new_max == 9


def test_chunk_relative_mm_features_rewrites_offsets():
    cumulative = [
        _make_video_feature(offset=2, length=4),
        _make_video_feature(offset=20, length=4),
        _make_video_feature(offset=40, length=4),
    ]
    # Suppose prior chunk ended at token 24 (covers first two features).
    chunk_relative = make_chunk_relative_mm_features(cumulative, prev_num_tokens=24)
    assert len(chunk_relative) == 1
    assert chunk_relative[0].mm_position.offset == 40 - 24
    assert chunk_relative[0].mm_position.length == 4
    # Original features unchanged (we copy, not mutate).
    assert cumulative[2].mm_position.offset == 40


def test_chunk_relative_mm_features_includes_boundary_offset():
    """A feature whose offset == prev_num_tokens belongs to the new chunk."""
    cumulative = [_make_video_feature(offset=10, length=4)]
    chunk_relative = make_chunk_relative_mm_features(cumulative, prev_num_tokens=10)
    assert len(chunk_relative) == 1
    assert chunk_relative[0].mm_position.offset == 0


def test_per_chunk_compute_matches_full_recompute_no_eviction():
    """Load-bearing parity check.

    Computing positions chunk-by-chunk (with chunk-relative mm_features and
    `base_position = prev_max + 1`) must produce a tensor identical to a
    single-shot full recompute over the cumulative prompt, assuming no
    eviction. This is the equivalence the existing `_init_mrope_positions`
    relies on; we're verifying our per-chunk path stays equivalent.
    """
    model = StubMRoPEModel()

    # Chunk 1: [text, text, video(4), text]
    chunk1_tokens = [TEXT_TOKEN, TEXT_TOKEN] + [VIDEO_PAD_TOKEN] * 4 + [TEXT_TOKEN]
    chunk1_features = [_make_video_feature(offset=2, length=4)]

    # Chunk 2: [text, video(4), text, text]
    chunk2_tokens = [TEXT_TOKEN] + [VIDEO_PAD_TOKEN] * 4 + [TEXT_TOKEN, TEXT_TOKEN]
    # Chunk-relative offset for chunk 2's video: 1.
    chunk2_features_rel = [_make_video_feature(offset=1, length=4)]

    # Per-chunk path.
    pos1, max1 = compute_chunk_mrope_positions(
        model, chunk1_tokens, chunk1_features, base_position=0
    )
    pos2, max2 = compute_chunk_mrope_positions(
        model, chunk2_tokens, chunk2_features_rel, base_position=max1 + 1
    )
    per_chunk = torch.cat([pos1, pos2], dim=1)

    # Full-recompute path: build cumulative prompt and cumulative mm_features
    # with absolute offsets.
    cumulative_tokens = chunk1_tokens + chunk2_tokens
    cumulative_features = [
        _make_video_feature(offset=2, length=4),
        _make_video_feature(offset=len(chunk1_tokens) + 1, length=4),
    ]
    full, _ = model.get_mrope_input_positions(cumulative_tokens, cumulative_features)

    assert per_chunk.shape == full.shape
    assert torch.equal(per_chunk, full), (
        f"Per-chunk and full-recompute paths disagreed.\n"
        f"per-chunk:\n{per_chunk}\nfull:\n{full}"
    )


def test_eviction_then_new_chunk_assigns_strictly_greater_positions():
    """Piece 3 invariant exercised at the Piece 1 helper level.

    After eviction the surviving positions retain their values. The next
    chunk's positions must all be strictly greater than any surviving
    position. This is what `base_position = max_cached_position + 1`
    guarantees.
    """
    model = StubMRoPEModel()

    # Chunk 1: text + video + text.
    chunk1_tokens = [TEXT_TOKEN, TEXT_TOKEN] + [VIDEO_PAD_TOKEN] * 4 + [TEXT_TOKEN]
    chunk1_features = [_make_video_feature(offset=2, length=4)]
    pos1, max1 = compute_chunk_mrope_positions(
        model, chunk1_tokens, chunk1_features, base_position=0
    )

    # Simulate eviction of the video segment (tokens [2:6]).
    surviving = torch.cat([pos1[:, :2], pos1[:, 6:]], dim=1)
    surviving_max_pre = int(surviving.max().item())

    # Next chunk: just text. Crucially, `max_cached_position` carried over
    # from chunk 1 (NOT shrunk to the surviving max), so new positions are
    # strictly greater than max1, leaving a gap where the video used to be.
    next_chunk_tokens = [TEXT_TOKEN, TEXT_TOKEN]
    next_pos, next_max = compute_chunk_mrope_positions(
        model, next_chunk_tokens, [], base_position=max1 + 1
    )

    assert int(next_pos.min().item()) > surviving_max_pre, (
        "new-chunk positions must exceed every surviving position"
    )
    assert next_max == max1 + len(next_chunk_tokens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
