# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EncoderRunner.gather_mm_embeddings (model runner V2).

Covers the speculative-drafter encoder-cache handling: the drafter reads
``draft_lookahead`` positions past the target's processed boundary. The +1
look-ahead feature past that boundary is used when its encoder output is
present and tolerated (token-embedding fallback) when it is not, while a miss
within the processed range still fails loudly. Multi-module MTP looks further
still, and those positions land in a per-request grid appended to the mask.
"""

import numpy as np
import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

pytestmark = pytest.mark.cpu_test

HIDDEN = 4


def _feature(identifier: str, offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _make_runner(
    features: list[MultiModalFeatureSpec],
    cached: list[MultiModalFeatureSpec],
) -> EncoderRunner:
    cache = EncoderCache()
    cache.mm_features["req0"] = features
    for f in cached:
        length = f.mm_position.length
        cache.encoder_outputs[f.identifier] = torch.arange(
            length * HIDDEN, dtype=torch.float32
        ).reshape(length, HIDDEN)
    return EncoderRunner(
        model=None,  # unused by gather_mm_embeddings
        max_num_tokens=64,
        hidden_size=HIDDEN,
        encoder_cache=cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _gather(runner: EncoderRunner, *, num_scheduled: int, draft_lookahead: int):
    # Single prefilling request, num_computed_tokens=0, prefill_len large.
    return runner.gather_mm_embeddings(
        req_ids=["req0"],
        total_num_scheduled_tokens=num_scheduled,
        num_scheduled_tokens=np.array([num_scheduled]),
        query_start_loc=np.array([0]),
        prefill_lens=np.array([1000]),
        num_computed_tokens=np.array([0]),
        draft_lookahead=draft_lookahead,
    )


def test_draft_lookahead_uses_boundary_feature_when_cached():
    """The drafter's +1 look-ahead can reach the feature at offset ==
    processed_end (the next chunk). When its encoder output is already cached
    (the scheduler encoded it ahead), it is used for the look-ahead position
    rather than ignored."""
    f0 = _feature("h0", offset=0, length=8)
    f1 = _feature("h1", offset=8, length=8)  # starts exactly at processed_end
    runner = _make_runner([f0, f1], cached=[f0, f1])

    mm_embeds, is_mm_embed = _gather(runner, num_scheduled=8, draft_lookahead=1)

    # f0 covers positions 0..6 (+1 skew); f1's first embed covers position 7.
    assert len(mm_embeds) == 2
    assert [e.modality for e in mm_embeds] == ["image", "image"]
    assert bool(is_mm_embed[7])
    assert int(is_mm_embed.sum()) == 8


def test_draft_lookahead_tolerates_missing_boundary_feature():
    """When the +1 look-ahead feature past the processed boundary is not yet
    encoded, fall back to the token embedding (the draft token is verified by
    the target) instead of raising."""
    f0 = _feature("h0", offset=0, length=8)
    f1 = _feature("h1", offset=8, length=8)  # boundary feature, not cached
    runner = _make_runner([f0, f1], cached=[f0])

    mm_embeds, is_mm_embed = _gather(runner, num_scheduled=8, draft_lookahead=1)

    # Only f0 is gathered; f1's boundary position falls back silently.
    assert len(mm_embeds) == 1
    assert [e.modality for e in mm_embeds] == ["image"]
    assert not bool(is_mm_embed[7])
    assert int(is_mm_embed.sum()) == 7


def test_draft_lookahead_raises_on_interior_miss():
    """A miss for a feature within the processed range (not the look-ahead
    boundary) is a real invariant violation and must fail loudly, even on the
    drafter path."""
    f0 = _feature("h0", offset=0, length=8)  # interior, within processed range
    runner = _make_runner([f0], cached=[])

    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        _gather(runner, num_scheduled=8, draft_lookahead=1)


def test_target_path_raises_on_encoder_cache_miss():
    """On the target path (no look-ahead) a miss is a real invariant
    violation and must fail loudly."""
    f0 = _feature("h0", offset=0, length=8)
    runner = _make_runner([f0], cached=[])

    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        _gather(runner, num_scheduled=8, draft_lookahead=0)


@pytest.mark.parametrize("draft_lookahead", [0, 1])
def test_multi_request_batch_gathers_per_request(draft_lookahead):
    """Two prefilling requests in one batch: per-request query bounds must be
    indexed by request, not applied as whole arrays."""
    a0 = _feature("a0", offset=0, length=8)
    b0 = _feature("b0", offset=0, length=8)
    cache = EncoderCache()
    cache.mm_features["req0"] = [a0]
    cache.mm_features["req1"] = [b0]
    for f in (a0, b0):
        cache.encoder_outputs[f.identifier] = torch.arange(
            f.mm_position.length * HIDDEN, dtype=torch.float32
        ).reshape(f.mm_position.length, HIDDEN)
    runner = EncoderRunner(
        model=None,
        max_num_tokens=64,
        hidden_size=HIDDEN,
        encoder_cache=cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    mm_embeds, is_mm_embed = runner.gather_mm_embeddings(
        req_ids=["req0", "req1"],
        total_num_scheduled_tokens=16,
        num_scheduled_tokens=np.array([8, 8]),
        query_start_loc=np.array([0, 8]),
        prefill_lens=np.array([1000, 1000]),
        num_computed_tokens=np.array([0, 0]),
        draft_lookahead=draft_lookahead,
    )

    # Both requests contribute a feature; with the +1 skew each marks 7 of its
    # 8 positions (the skew drops one), otherwise all 8.
    assert len(mm_embeds) == 2
    assert [e.modality for e in mm_embeds] == ["image", "image"]
    assert int(is_mm_embed.sum()) == (14 if draft_lookahead else 16)


def test_extra_lookahead_extends_mask_with_per_request_grid():
    """Multi-module MTP reads ``draft_lookahead`` positions past the target's
    chunk end. Only the first has a query slot (the +1 skew); the rest land in
    a dense [num_reqs, draft_lookahead - 1] grid appended to the mask, and
    their embeddings are appended after every query-window one."""
    # req0 is mid-prefill and f1 straddles its chunk end, so f1 contributes to
    # both the query window and the grid. req1 contributes to the grid only,
    # pinning the per-request row stride. req2 is decoding, so its row stays
    # empty and must not shift the others.
    f0 = _feature("f0", offset=0, length=8)
    f1 = _feature("f1", offset=8, length=4)
    g0 = _feature("g0", offset=10, length=1)
    cache = EncoderCache()
    cache.mm_features["req0"] = [f0, f1]
    cache.mm_features["req1"] = [g0]
    cache.mm_features["req2"] = []
    for f in (f0, f1, g0):
        length = f.mm_position.length
        cache.encoder_outputs[f.identifier] = torch.arange(
            length * HIDDEN, dtype=torch.float32
        ).reshape(length, HIDDEN)
    runner = EncoderRunner(
        model=None,
        max_num_tokens=64,
        hidden_size=HIDDEN,
        encoder_cache=cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    mm_embeds, is_mm_embed = runner.gather_mm_embeddings(
        req_ids=["req0", "req1", "req2"],
        total_num_scheduled_tokens=24,
        num_scheduled_tokens=np.array([8, 8, 8]),
        query_start_loc=np.array([0, 8, 16]),
        prefill_lens=np.array([1000, 1000, 50]),
        num_computed_tokens=np.array([0, 0, 100]),
        draft_lookahead=3,
    )

    # 24 query slots, then one 2-wide grid row per request.
    assert is_mm_embed.shape == (30,)
    # req0's query window is positions 1..8 (its chunk 0..7, skewed by one):
    # f0 marks 1..7 and f1 marks 8. req1 has no feature in its window, and
    # req2 is decoding.
    assert bool(is_mm_embed[:8].all())
    assert not bool(is_mm_embed[8:24].any())
    # Grid rows: req0 covers positions 9..10 (both f1), req1 covers 9..10 of
    # which only 10 is g0, req2 contributes nothing.
    assert is_mm_embed[24:].tolist() == [True, True, False, True, False, False]

    # Query-window embeddings first, then the grid's:
    # f0[1:8], f1[0:1] | f1[1:3], g0[0:1].
    assert [tuple(e.shape) for e in mm_embeds] == [
        (7, HIDDEN),
        (1, HIDDEN),
        (2, HIDDEN),
        (1, HIDDEN),
    ]
    # The third entry is f1's grid slice (rows 1..2), not its query slice.
    assert float(mm_embeds[2][0, 0]) == 4.0


def test_gather_preserves_mixed_modalities():
    """Modalities must be attached on tensors in gather order."""
    video = MultiModalFeatureSpec(
        data=None,
        modality="video",
        identifier="v0",
        mm_position=PlaceholderRange(offset=0, length=4),
    )
    audio = MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier="a0",
        mm_position=PlaceholderRange(offset=4, length=4),
    )
    runner = _make_runner([video, audio], cached=[video, audio])

    mm_embeds, is_mm_embed = _gather(runner, num_scheduled=8, draft_lookahead=0)

    assert len(mm_embeds) == 2
    assert [e.modality for e in mm_embeds] == ["video", "audio"]
    assert int(is_mm_embed.sum()) == 8
