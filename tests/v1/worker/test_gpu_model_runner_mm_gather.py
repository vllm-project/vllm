# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUModelRunner._gather_mm_embeddings (model runner V1).

Mirrors tests/v1/worker/test_encoder_runner.py (the V2 runner): the EAGLE/MTP
drafter reads one position ahead of the target (shift_computed_tokens=1). The
+1 look-ahead feature past the processed boundary is used when its encoder
output is present and tolerated (token-embedding fallback) when it is not,
while a miss within the processed range still fails loudly.

`_gather_mm_embeddings` only uses CPU-side state, so it is exercised against a
lightweight stub for `self` instead of a full (CUDA-only) runner.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test

HIDDEN = 4


def _feature(identifier: str, offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _gather(features, cached, *, num_scheduled, shift, num_computed=0):
    encoder_cache = {
        f.identifier: torch.arange(
            f.mm_position.length * HIDDEN, dtype=torch.float32
        ).reshape(f.mm_position.length, HIDDEN)
        for f in cached
    }
    req_state = SimpleNamespace(num_computed_tokens=num_computed, mm_features=features)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["req0"]),
        requests={"req0": req_state},
        encoder_cache=encoder_cache,
        _get_encoder_output_from_cache=lambda mm_hash: encoder_cache.get(mm_hash),
        is_multimodal_pruning_enabled=False,
        uses_mrope=False,
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=num_scheduled,
        num_scheduled_tokens={"req0": num_scheduled},
    )
    return GPUModelRunner._gather_mm_embeddings(
        runner, scheduler_output, shift_computed_tokens=shift
    )


def test_draft_shift_uses_boundary_feature_when_cached():
    """The drafter's +1 look-ahead reaches the feature at offset ==
    processed_end; when it is already cached it is used for the look-ahead
    position rather than ignored."""
    f0 = _feature("h0", offset=0, length=8)
    f1 = _feature("h1", offset=8, length=8)  # starts exactly at processed_end
    mm_embeds, is_mm_embed = _gather([f0, f1], [f0, f1], num_scheduled=8, shift=1)

    # f0 covers positions 0..6 (+1 skew); f1's first embed covers position 7.
    assert len(mm_embeds) == 2
    assert bool(is_mm_embed[7])
    assert int(is_mm_embed.sum()) == 8


def test_draft_shift_tolerates_missing_boundary_feature():
    """When the +1 look-ahead feature past the processed boundary is not yet
    encoded, fall back to the token embedding instead of raising."""
    f0 = _feature("h0", offset=0, length=8)
    f1 = _feature("h1", offset=8, length=8)  # boundary feature, not cached
    mm_embeds, is_mm_embed = _gather([f0, f1], [f0], num_scheduled=8, shift=1)

    assert len(mm_embeds) == 1  # only f0; f1's boundary position falls back
    assert not bool(is_mm_embed[7])
    assert int(is_mm_embed.sum()) == 7


def test_draft_shift_raises_on_interior_miss():
    """A miss for a feature within the processed range (not the look-ahead
    boundary) is a real invariant violation, even on the drafter path."""
    f0 = _feature("h0", offset=0, length=8)  # interior, within processed range
    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        _gather([f0], [], num_scheduled=8, shift=1)


def test_target_path_raises_on_encoder_cache_miss():
    """On the target path (no shift) a miss is a real invariant violation."""
    f0 = _feature("h0", offset=0, length=8)
    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        _gather([f0], [], num_scheduled=8, shift=0)


@pytest.mark.parametrize(
    ("actual_embeds", "expected_embeds"),
    [
        (3, 6),  # actual < expected (issue #55546: 138 vs 638)
        (6, 3),  # actual > expected (silent wrong-item slice)
    ],
)
def test_gather_isolates_embed_count_mismatch_from_cobatched_request(
    actual_embeds: int,
    expected_embeds: int,
):
    """A feature/placeholder mismatch aborts only the offending request, keeps
    co-batched multimodal requests mergeable, evicts only the mismatched cache
    entry, and prevents a follow-up speculative-drafter gather from crashing."""
    from vllm.model_executor.models.utils import _merge_multimodal_embeddings

    valid_co_item = _feature("valid_co", offset=0, length=2)
    bad_item = _feature("bad_hash", offset=2, length=expected_embeds)
    good_req_item = _feature("good_hash", offset=0, length=4)

    encoder_cache = {
        "valid_co": torch.ones((2, HIDDEN), dtype=torch.float32),
        "bad_hash": torch.full((actual_embeds, HIDDEN), 9.0, dtype=torch.float32),
        "good_hash": torch.full((4, HIDDEN), 5.0, dtype=torch.float32),
    }
    bad_scheduled = 2 + expected_embeds + 2
    good_scheduled = 6
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["req_bad", "req_dup", "req_good"]),
        requests={
            "req_bad": SimpleNamespace(
                num_computed_tokens=0,
                mm_features=[valid_co_item, bad_item],
            ),
            "req_dup": SimpleNamespace(
                num_computed_tokens=0,
                mm_features=[_feature("bad_hash", offset=0, length=expected_embeds)],
            ),
            "req_good": SimpleNamespace(
                num_computed_tokens=0,
                mm_features=[good_req_item],
            ),
        },
        encoder_cache=encoder_cache,
        _get_encoder_output_from_cache=lambda mm_hash: encoder_cache.get(mm_hash),
        is_multimodal_pruning_enabled=False,
        uses_mrope=False,
    )
    total_scheduled = bad_scheduled + expected_embeds + good_scheduled
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=total_scheduled,
        num_scheduled_tokens={
            "req_bad": bad_scheduled,
            "req_dup": expected_embeds,
            "req_good": good_scheduled,
        },
    )

    mm_embeds, is_mm_embed = GPUModelRunner._gather_mm_embeddings(
        runner, scheduler_output, shift_computed_tokens=0
    )

    expected_msg = (
        f"Attempted to assign {actual_embeds} = {actual_embeds} "
        f"multimodal tokens to {expected_embeds} placeholders"
    )
    mismatches = GPUModelRunner.take_mm_embed_mismatches(runner)
    assert mismatches == {
        "req_bad": (["bad_hash"], expected_msg),
        "req_dup": (["bad_hash"], expected_msg),
    }
    assert GPUModelRunner.take_mm_embed_mismatches(runner) is None

    # Valid co-item stays cached; poisoned bad_hash is evicted immediately.
    assert "valid_co" in encoder_cache
    assert "bad_hash" not in encoder_cache
    assert "good_hash" in encoder_cache

    # Offending requests fall back to text embeddings; co-batched req_good merges.
    bad_and_dup_tokens = bad_scheduled + expected_embeds
    assert not bool(is_mm_embed[:bad_and_dup_tokens].any())
    assert is_mm_embed[bad_and_dup_tokens:].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    inputs_embeds = torch.zeros((total_scheduled, HIDDEN), dtype=torch.float32)
    merged = _merge_multimodal_embeddings(inputs_embeds, mm_embeds, is_mm_embed)
    assert torch.all(merged[:bad_and_dup_tokens] == 0.0)
    assert torch.all(merged[bad_and_dup_tokens : bad_and_dup_tokens + 4] == 5.0)

    # Same-step speculative drafter look-ahead (+1) must not hit cache miss.
    draft_embeds, draft_mask = GPUModelRunner._gather_mm_embeddings(
        runner, scheduler_output, shift_computed_tokens=1
    )
    assert len(draft_embeds) == 1
    assert not bool(draft_mask[:bad_and_dup_tokens].any())
