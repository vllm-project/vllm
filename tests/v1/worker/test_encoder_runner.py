# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EncoderRunner.gather_mm_embeddings (model runner V2).

Covers the speculative-drafter encoder-cache handling: the drafter reads one
position ahead of the target model (``draft_lookahead``), which must not
(1) select a not-yet-encoded next-chunk feature, nor (2) crash on a cache miss.
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
    # Single prefilling request, computed_prefill=0, prefill_len large.
    return runner.gather_mm_embeddings(
        req_ids=["req0"],
        total_num_scheduled_tokens=num_scheduled,
        num_scheduled_tokens=np.array([num_scheduled]),
        query_start_loc=np.array([0]),
        prefill_lens=np.array([1000]),
        computed_prefill_lens=np.array([0]),
        draft_lookahead=draft_lookahead,
    )


def test_draft_lookahead_excludes_unprocessed_next_chunk_feature():
    """The +1 drafter skew must not pull in the feature at offset ==
    processed_end (the next chunk, not yet encoded). Both features are cached
    here so the assertion isolates feature *selection* from the miss fallback:
    if the window clamp regressed, feature1 would be gathered and its position
    in is_mm_embed marked."""
    f0 = _feature("h0", offset=0, length=8)
    f1 = _feature("h1", offset=8, length=8)  # starts exactly at processed_end
    runner = _make_runner([f0, f1], cached=[f0, f1])

    mm_embeds, is_mm_embed = _gather(runner, num_scheduled=8, draft_lookahead=1)

    # Only the in-window feature (f0) is gathered; f1 is excluded entirely.
    assert len(mm_embeds) == 1
    # f1's first token would map to batch position 7; it must stay unset.
    assert not bool(is_mm_embed[7])
    assert int(is_mm_embed.sum()) == 7  # f0 contributes positions 0..6 (+1 skew)


def test_draft_lookahead_tolerates_encoder_cache_miss():
    """An evicted entry on the drafter path falls back to token embeddings
    (draft tokens are verified by the target), instead of raising."""
    f0 = _feature("h0", offset=0, length=8)
    runner = _make_runner([f0], cached=[])  # encoder output missing

    mm_embeds, is_mm_embed = _gather(runner, num_scheduled=8, draft_lookahead=1)

    assert mm_embeds == []
    assert int(is_mm_embed.sum()) == 0


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
        computed_prefill_lens=np.array([0, 0]),
        draft_lookahead=draft_lookahead,
    )

    # Both requests contribute a feature; with the +1 skew each marks 7 of its
    # 8 positions (the skew drops one), otherwise all 8.
    assert len(mm_embeds) == 2
    assert int(is_mm_embed.sum()) == (14 if draft_lookahead else 16)
