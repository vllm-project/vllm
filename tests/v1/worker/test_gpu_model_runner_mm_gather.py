# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUModelRunner._gather_mm_embeddings (model runner V1).

Mirrors tests/v1/worker/test_encoder_runner.py (the V2 runner): the EAGLE/MTP
drafter reads one position ahead of the target (shift_computed_tokens=1), which
must not select a not-yet-encoded next-chunk feature, nor crash on a cache miss.

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


def test_draft_shift_excludes_unprocessed_next_chunk_feature():
    """The +1 drafter skew must not pull in the feature at offset ==
    processed_end. Both features are cached so the assertion isolates feature
    *selection* from the miss fallback."""
    f0 = _feature("h0", offset=0, length=8)
    f1 = _feature("h1", offset=8, length=8)  # starts exactly at processed_end
    mm_embeds, is_mm_embed = _gather([f0, f1], [f0, f1], num_scheduled=8, shift=1)

    assert len(mm_embeds) == 1  # only f0; f1 excluded
    assert not bool(is_mm_embed[7])  # f1's first token position stays unset
    assert int(is_mm_embed.sum()) == 7  # f0 positions 0..6 (+1 skew)


def test_draft_shift_tolerates_encoder_cache_miss():
    """An evicted entry on the drafter path falls back to token embeddings."""
    f0 = _feature("h0", offset=0, length=8)
    mm_embeds, is_mm_embed = _gather([f0], [], num_scheduled=8, shift=1)

    assert mm_embeds == []
    assert int(is_mm_embed.sum()) == 0


def test_target_path_raises_on_encoder_cache_miss():
    """On the target path (no shift) a miss is a real invariant violation."""
    f0 = _feature("h0", offset=0, length=8)
    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        _gather([f0], [], num_scheduled=8, shift=0)
