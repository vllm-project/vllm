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
