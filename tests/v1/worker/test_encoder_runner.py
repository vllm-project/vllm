# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EncoderRunner.gather_mm_embeddings (model runner V2).

Covers the speculative-drafter encoder-cache handling: the drafter reads one
position ahead of the target model (``draft_lookahead``). The +1 look-ahead
feature past the processed boundary is used when its encoder output is present
and tolerated (token-embedding fallback) when it is not, while a miss within
the processed range still fails loudly.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.model_states.interface import ModelState

pytestmark = pytest.mark.cpu_test

HIDDEN = 4


def _model_state(cache: EncoderCache) -> MagicMock:
    """A mock ModelState backed by a real EncoderCache."""
    state = MagicMock()
    state.encoder_cache = cache
    state.device = torch.device("cpu")
    return state


def _embeds_item(embeds: torch.Tensor) -> MultiModalKwargsItem:
    """A `prompt_embeds` kwargs item, as the HF renderer builds it."""
    return MultiModalKwargsItem(
        {
            "embedding": MultiModalFieldElem(
                data=embeds, field=MultiModalSharedField(batch_size=1)
            )
        }
    )


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


def test_execute_mm_encoder_caches_outputs_without_gathering():
    """An encoder instance encodes and publishes, and must stop there.

    `ModelState.execute_mm_encoder` is the half of `get_mm_embeddings` that an
    EPD encoder instance needs: it runs no language model, so gathering would
    build an `inputs_embeds` nobody reads -- and the gather raises
    `Encoder cache miss` for any scheduled item absent from the local cache,
    which on a producer takes the whole engine down (the scheduler hands it
    items the connector already holds, and a producer has no load path).
    """
    cache = EncoderCache()
    state = _model_state(cache)
    embedding = torch.ones(2, HIDDEN)
    # (mm_hashes, [(modality, kwargs item), ...]), as prepare_mm_inputs returns.
    state.encoder_runner.prepare_mm_inputs.return_value = (
        ["hash0"],
        [("image", MagicMock())],
    )
    state.encoder_runner.execute_mm_encoder.return_value = [embedding]

    ModelState.execute_mm_encoder(state, {"req0": [0]})

    assert cache.encoder_outputs == {"hash0": embedding}
    state.encoder_runner.gather_mm_embeddings.assert_not_called()


def test_execute_mm_encoder_is_a_noop_without_scheduled_items():
    """A step that schedules no encoder input must not touch the encoder."""
    cache = EncoderCache()
    state = _model_state(cache)
    state.encoder_runner.prepare_mm_inputs.return_value = ([], [])

    ModelState.execute_mm_encoder(state, {})

    assert not cache.encoder_outputs
    state.encoder_runner.execute_mm_encoder.assert_not_called()


def _pe_feature(identifier: str, embeds: torch.Tensor, offset: int = 0):
    return MultiModalFeatureSpec(
        data=_embeds_item(embeds),
        modality="prompt_embeds",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=embeds.shape[0]),
    )


def test_prepare_mm_inputs_passes_prompt_embeds_through():
    """`prompt_embeds` is already in embedding space, so no encoder may run.

    The renderer delivers prompt_embeds mixed with real media as an ordinary MM
    modality. prepare_mm_inputs must cache the tensor directly and keep it out
    of the encoder batch -- the vision encoder cannot consume it, and a missing
    cache entry makes the subsequent gather raise "Encoder cache miss".
    """
    prompt_embeds = torch.arange(2 * HIDDEN, dtype=torch.float32).view(2, HIDDEN)
    image_feature = MultiModalFeatureSpec(
        data=MagicMock(),
        modality="image",
        identifier="hash_img",
        mm_position=PlaceholderRange(offset=2, length=2),
    )
    runner = _make_runner(
        [_pe_feature("hash_pe", prompt_embeds), image_feature], cached=[]
    )

    mm_hashes, mm_kwargs = runner.prepare_mm_inputs({"req0": [0, 1]})

    # Only the image remains for the encoder; the embeds are already cached.
    assert mm_hashes == ["hash_img"]
    assert [modality for modality, _ in mm_kwargs] == ["image"]
    assert torch.equal(runner.encoder_cache.encoder_outputs["hash_pe"], prompt_embeds)


def test_prepare_mm_inputs_skips_cached_prompt_embeds():
    """A prompt_embeds item already in the cache must not be re-uploaded."""
    prompt_embeds = torch.ones(3, HIDDEN)
    feature = _pe_feature("hash_pe", prompt_embeds)
    runner = _make_runner([feature], cached=[feature])
    sentinel = runner.encoder_cache.encoder_outputs["hash_pe"]

    mm_hashes, mm_kwargs = runner.prepare_mm_inputs({"req0": [0]})

    assert mm_hashes == [] and mm_kwargs == []
    assert runner.encoder_cache.encoder_outputs["hash_pe"] is sentinel


def test_execute_mm_encoder_skips_encoder_for_prompt_embeds_only():
    """A batch of nothing but prompt_embeds must not invoke the encoder."""
    prompt_embeds = torch.ones(3, HIDDEN)
    runner = _make_runner([_pe_feature("hash_pe", prompt_embeds)], cached=[])
    state = _model_state(runner.encoder_cache)
    state.encoder_runner.prepare_mm_inputs.side_effect = runner.prepare_mm_inputs

    ModelState.execute_mm_encoder(state, {"req0": [0]})

    state.encoder_runner.execute_mm_encoder.assert_not_called()
    assert torch.equal(runner.encoder_cache.encoder_outputs["hash_pe"], prompt_embeds)


def test_encoder_timing_stats_registry():
    runner = _make_runner([], [])
    runner.enable_timing = True

    with runner.timed_encoder_operation({"r1"}):
        pass
    with runner.timed_encoder_operation({"r1"}):
        pass

    stats = runner.get_encoder_timing_stats()
    assert set(stats) == {"r1"}
    assert stats["r1"]["num_encoder_calls"] == 2
    assert stats["r1"]["encoder_forward_secs"] >= 0
    assert runner.get_encoder_timing_stats() == {}
