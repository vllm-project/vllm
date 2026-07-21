# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EncoderRunner.gather_mm_embeddings (model runner V2).

Covers the speculative-drafter encoder-cache handling: the drafter reads one
position ahead of the target model (``draft_lookahead``). The +1 look-ahead
feature past the processed boundary is used when its encoder output is present
and tolerated (token-embedding fallback) when it is not, while a miss within
the processed range still fails loudly.
"""

from types import SimpleNamespace

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


class _MergeRecordingModel:
    """Stands in for a multimodal model, recording the merge-path inputs."""

    def __init__(self):
        self.calls: list[tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]] = []

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal):
        self.calls.append((input_ids, list(multimodal_embeddings), is_multimodal))
        return torch.zeros(input_ids.size(0), HIDDEN)


def test_profile_encoder_cache_profiles_the_embedding_merge():
    """Startup profiling must run `embed_input_ids` on a full batch.

    Dummy runs hand the model a slice of the pre-allocated `inputs_embeds`
    buffer, so without this the merge path -- and temporaries that scale with
    `max_num_batched_tokens`, such as Qwen3-VL's deepstack embeddings -- is
    never allocated while profiling and first shows up on a live request,
    OOMing an engine that started up fine.
    """
    model = _MergeRecordingModel()
    runner = EncoderRunner(
        model=model,
        max_num_tokens=64,
        hidden_size=HIDDEN,
        encoder_cache=EncoderCache(),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    dummy_outputs = [torch.zeros(8, HIDDEN)]
    runner.execute_mm_encoder = lambda mm_kwargs: dummy_outputs
    budget = SimpleNamespace(
        get_encoder_budget=lambda: 64,
        mm_max_toks_per_item={"image": 8},
        model_config=SimpleNamespace(is_encoder_decoder=False),
    )

    runner.profile_encoder_cache([("image", None)], budget)

    assert model.calls, "the multimodal merge path was not profiled"
    input_ids, mm_embeds, is_mm_embed = model.calls[0]
    assert input_ids.size(0) == runner.max_num_tokens
    assert is_mm_embed.size(0) == runner.max_num_tokens
    # `_merge_multimodal_embeddings` needs one embedding row per masked
    # position; the mask must never claim more rows than were passed in.
    assert sum(e.size(0) for e in mm_embeds) == int(is_mm_embed.sum())
