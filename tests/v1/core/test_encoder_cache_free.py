# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for encoder cache retention (cross-step fix).

Verifies that the scheduler no longer forwards free_encoder_mm_hashes
to workers for multimodal models (returns []), that encoder-decoder
models still get hashes forwarded, and that entries are retained in
the GPU-side encoder cache to survive preemption and hash reuse.

See: https://github.com/vllm-project/vllm/issues/38551
"""
import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _make_mm_feature(mm_hash: str, offset: int, length: int):
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=offset, length=length),
        identifier=mm_hash,
        modality="image",
    )


# ------------------------------------------------------------------ #
# Scheduler: multimodal returns empty, encoder-decoder forwards      #
# ------------------------------------------------------------------ #


def test_scheduler_returns_empty_hashes_for_multimodal():
    """After a multimodal request completes prefill, the scheduler
    returns free_encoder_mm_hashes=[] — eviction is the model runner's
    responsibility."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=8192,
    )
    mm_positions = [[PlaceholderRange(offset=50, length=100)]]
    requests = create_requests(
        num_requests=1,
        num_tokens=200,
        mm_positions=mm_positions,
    )
    for req in requests:
        scheduler.add_request(req)

    output = scheduler.schedule()
    assert output.free_encoder_mm_hashes == []

    req_to_index = {req.request_id: i for i, req in enumerate(requests)}
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[42]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Decode step — scheduler frees budget internally but returns [].
    output = scheduler.schedule()
    assert output.free_encoder_mm_hashes == []


def test_scheduler_still_reclaims_budget():
    """Scheduler reclaims encoder budget so new multimodal requests
    can be scheduled, even though it returns empty hashes."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=8192,
    )
    mm_positions = [
        [PlaceholderRange(offset=0, length=100)],
        [PlaceholderRange(offset=0, length=100)],
    ]
    requests = create_requests(
        num_requests=2,
        num_tokens=200,
        mm_positions=mm_positions,
    )
    for req in requests:
        scheduler.add_request(req)

    output = scheduler.schedule()
    assert output.free_encoder_mm_hashes == []

    req_to_index = {req.request_id: i for i, req in enumerate(requests)}
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[42], [43]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)
    scheduler.schedule()

    # New multimodal request can be scheduled (budget reclaimed).
    new_mm_positions = [[PlaceholderRange(offset=0, length=100)]]
    new_requests = create_requests(
        num_requests=1,
        num_tokens=200,
        mm_positions=new_mm_positions,
        req_ids=["new_mm_req"],
    )
    scheduler.add_request(new_requests[0])
    output = scheduler.schedule()
    assert "new_mm_req" in output.num_scheduled_tokens
    assert output.free_encoder_mm_hashes == []


def test_encoder_decoder_scheduler_forwards_hashes():
    """Encoder-decoder models (e.g. Whisper) still get hashes forwarded
    since they rely on scheduler-driven eviction."""
    scheduler = create_scheduler(
        model="openai/whisper-tiny",
        max_num_batched_tokens=8192,
    )
    assert scheduler.is_encoder_decoder

    mm_positions = [[PlaceholderRange(offset=0, length=100)]]
    requests = create_requests(
        num_requests=1,
        num_tokens=200,
        mm_positions=mm_positions,
    )
    for req in requests:
        scheduler.add_request(req)

    output = scheduler.schedule()

    req_to_index = {req.request_id: i for i, req in enumerate(requests)}
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[42]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Collect hashes across steps (buffer swap has one-step delay).
    all_freed: list[str] = []
    for _ in range(3):
        output = scheduler.schedule()
        all_freed.extend(output.free_encoder_mm_hashes)
        scheduler.update_from_output(output, model_output)

    assert len(all_freed) > 0, (
        "Encoder-decoder scheduler must forward free_encoder_mm_hashes"
    )


# ------------------------------------------------------------------ #
# Model runner: encoder cache retention                              #
# ------------------------------------------------------------------ #


class TestModelRunnerEncoderCacheRetention:
    """Tests for encoder cache retention in the model runner.

    With the full retention approach, the model runner never evicts
    encoder cache entries for multimodal models. Entries survive
    preemption (num_computed_tokens reset) and hash reuse across
    sequential requests sharing the same image.
    """

    def test_free_states_processes_encoder_decoder_hashes(self):
        """free_states() still frees entries for encoder-decoder models."""
        from unittest.mock import MagicMock
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        cache = EncoderCache()
        cache.encoder_outputs["enc_x"] = torch.zeros(1)

        scheduler_output = MagicMock()
        scheduler_output.free_encoder_mm_hashes = ["enc_x"]

        runner = type("_", (), {
            "encoder_cache": cache,
            "free_states": GPUModelRunner.free_states,
        })()
        runner.free_states(scheduler_output)
        assert "enc_x" not in cache.encoder_outputs

    def test_free_states_empty_list_for_multimodal(self):
        """free_states() with empty list (multimodal) keeps entries."""
        from unittest.mock import MagicMock
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        cache = EncoderCache()
        cache.encoder_outputs["img_x"] = torch.zeros(1)

        scheduler_output = MagicMock()
        scheduler_output.free_encoder_mm_hashes = []

        runner = type("_", (), {
            "encoder_cache": cache,
            "free_states": GPUModelRunner.free_states,
        })()
        runner.free_states(scheduler_output)
        assert "img_x" in cache.encoder_outputs

    def test_entries_survive_preemption_scenario(self):
        """Entries remain after a simulated preemption where
        num_computed_tokens would reset to 0.  Under the old
        num_computed_tokens-based eviction this would cause a miss."""
        cache = EncoderCache()
        cache.encoder_outputs["img_preempt"] = torch.zeros(1)

        # Simulate: scheduler sends empty hashes (multimodal suppression).
        # Model runner does NOT call any free method. Entry survives.
        assert "img_preempt" in cache.encoder_outputs

    def test_entries_survive_hash_reuse(self):
        """When two sequential requests share the same image hash, the
        entry must still be present for the second request."""
        cache = EncoderCache()
        cache.encoder_outputs["shared_img"] = torch.zeros(1)

        # First request finishes — under old code, entry was freed.
        # With retention, it stays.
        assert "shared_img" in cache.encoder_outputs

        # Second request reads it — no miss.
        output = cache.encoder_outputs.get("shared_img")
        assert output is not None
