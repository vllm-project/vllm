# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for request-scoped HLM state under the V2 runner batch contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.models.ncp_olmo.model import NCPOlmo3ForCausalLM
from vllm.model_executor.models.ncp_olmo.model_state import NCPOlmoModelState
from vllm.model_executor.models.ncp_olmo.state import (
    ConceptRequestStateStore,
    RequestStateError,
)


def make_input_batch(
    req_ids: list[str],
    scheduled_tokens: list[int],
    computed_tokens: list[int],
) -> SimpleNamespace:
    """Build the CPU metadata consumed from a V2 ``InputBatch``."""

    return SimpleNamespace(
        req_ids=req_ids,
        num_scheduled_tokens=np.asarray(scheduled_tokens, dtype=np.int32),
        num_computed_tokens_np=np.asarray(computed_tokens, dtype=np.int32),
        num_tokens=sum(scheduled_tokens),
    )


def make_store() -> ConceptRequestStateStore:
    return ConceptRequestStateStore(encoder_layers=2, hlm_layers=3)


def test_segments_follow_v2_batch_order_across_chunked_prefill() -> None:
    store = make_store()
    store.add_request("first", computed_tokens=0)
    store.add_request("second", computed_tokens=0)

    first_step = store.resolve_segments(
        make_input_batch(["first", "second"], [3, 2], [0, 0])
    )
    for segment in first_step:
        store.commit(segment)

    second_step = store.resolve_segments(
        make_input_batch(["second", "first"], [1, 2], [2, 3])
    )

    assert [segment.req_id for segment in second_step] == ["second", "first"]
    assert [(segment.flat_start, segment.flat_end) for segment in second_step] == [
        (0, 1),
        (1, 3),
    ]
    assert [segment.position_start for segment in second_step] == [2, 3]


def test_finished_slot_is_refilled_and_reordered_without_state_leakage() -> None:
    """Exercise a continuous batch that finishes, refills, and reorders a slot."""

    store = make_store()
    store.add_request("long", computed_tokens=0)
    store.add_request("short", computed_tokens=0)

    first_step = store.resolve_segments(
        make_input_batch(["long", "short"], [3, 1], [0, 0])
    )
    for segment in first_step:
        store.commit(segment)
    finished_state = store.states["short"]

    store.remove_request("short")
    store.add_request("replacement", computed_tokens=0)
    refill_step = store.resolve_segments(
        make_input_batch(["replacement", "long"], [2, 1], [0, 3])
    )

    assert "short" not in store.states
    assert finished_state is not store.states["replacement"]
    assert [segment.req_id for segment in refill_step] == ["replacement", "long"]
    assert [segment.position_start for segment in refill_step] == [0, 3]
    assert [(segment.flat_start, segment.flat_end) for segment in refill_step] == [
        (0, 2),
        (2, 3),
    ]


def test_batch_invariant_hlm_advances_requests_individually(monkeypatch) -> None:
    """Keep HLM arithmetic independent of incidental scheduler batching."""

    calls = []

    class RecordingHighLevel:
        def advance(self, state, encoder_chunk, layer_chunks) -> None:
            calls.append((state.req_id, encoder_chunk.item(), layer_chunks[0].item()))

        def advance_batch(self, *args, **kwargs) -> None:
            raise AssertionError("batch-invariant HLM must not use advance_batch")

    states = [
        SimpleNamespace(req_id="first", predicted_concepts=SimpleNamespace(length=0)),
        SimpleNamespace(req_id="second", predicted_concepts=SimpleNamespace(length=0)),
    ]
    advances = [
        (states[0], torch.tensor(1.0), (torch.tensor(2.0),)),
        (states[1], torch.tensor(3.0), (torch.tensor(4.0),)),
    ]
    model = SimpleNamespace(highlevel=RecordingHighLevel())
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    NCPOlmo3ForCausalLM._advance_hlm_batches(model, advances)

    assert calls == [("first", 1.0, 2.0), ("second", 3.0, 4.0)]


def test_normal_hlm_path_batches_equal_position_requests(monkeypatch) -> None:
    """Retain the throughput path when exact batch invariance is disabled."""

    calls = []

    class RecordingHighLevel:
        def advance(self, *args, **kwargs) -> None:
            raise AssertionError("normal HLM path should batch equal-position requests")

        def advance_batch(self, states, encoder_chunks, layer_chunks) -> None:
            calls.append(
                (
                    [state.req_id for state in states],
                    encoder_chunks.tolist(),
                    layer_chunks[0].tolist(),
                )
            )

    states = [
        SimpleNamespace(req_id="first", predicted_concepts=SimpleNamespace(length=0)),
        SimpleNamespace(req_id="second", predicted_concepts=SimpleNamespace(length=0)),
    ]
    advances = [
        (states[0], torch.tensor(1.0), (torch.tensor(2.0),)),
        (states[1], torch.tensor(3.0), (torch.tensor(4.0),)),
    ]
    model = SimpleNamespace(
        highlevel=RecordingHighLevel(),
        backend_config=SimpleNamespace(encoder_layers=1),
    )
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    NCPOlmo3ForCausalLM._advance_hlm_batches(model, advances)

    assert calls == [(["first", "second"], [1.0, 3.0], [2.0, 4.0])]


def test_remove_request_releases_state_and_requires_readmission() -> None:
    store = make_store()
    store.add_request("finished", computed_tokens=0)
    state = store.states["finished"]

    store.remove_request("finished")

    assert "finished" not in store.states
    with pytest.raises(RequestStateError, match="was not admitted"):
        store.resolve_segments(make_input_batch(["finished"], [1], [0]))
    assert state.req_id == "finished"


def test_preempted_request_restarts_from_empty_state_on_readmission() -> None:
    store = make_store()
    store.add_request("preempted", computed_tokens=0)
    original = store.states["preempted"]
    segment = store.resolve_segments(make_input_batch(["preempted"], [4], [0]))[0]
    store.commit(segment)

    store.remove_request("preempted")
    store.add_request("preempted", computed_tokens=0)

    replay = store.resolve_segments(make_input_batch(["preempted"], [2], [0]))[0]
    assert replay.state is not original
    assert replay.state.next_token_position == 0
    assert replay.position_end == 2


def test_full_replay_replaces_request_state() -> None:
    store = make_store()
    store.add_request("preempted", computed_tokens=0)
    original = store.states["preempted"]
    segment = store.resolve_segments(make_input_batch(["preempted"], [4], [0]))[0]
    store.commit(segment)

    replay = store.resolve_segments(make_input_batch(["preempted"], [2], [0]))[0]

    assert replay.state is not original
    assert replay.state.next_token_position == 0
    assert replay.position_end == 2


def test_partial_prefix_admission_fails_without_hlm_snapshot() -> None:
    store = make_store()

    with pytest.raises(RequestStateError, match="no matching HLM state snapshot"):
        store.add_request("cached", computed_tokens=16)
    assert "cached" not in store.states


def test_position_gap_fails_closed() -> None:
    store = make_store()
    store.add_request("gap", computed_tokens=0)

    with pytest.raises(RequestStateError, match="state ends at 0"):
        store.resolve_segments(make_input_batch(["gap"], [1], [3]))


def test_segment_total_must_match_v2_input_batch() -> None:
    store = make_store()
    store.add_request("bad-total", computed_tokens=0)
    input_batch = make_input_batch(["bad-total"], [2], [0])
    input_batch.num_tokens = 3

    with pytest.raises(RequestStateError, match="does not match total"):
        store.resolve_segments(input_batch)


def make_model_state() -> NCPOlmoModelState:
    """Construct the adapter without allocating runner device buffers."""

    model_state = object.__new__(NCPOlmoModelState)
    model_state.request_states = make_store()
    model_state.rope_state = None
    model_state.prompt_embeds_state = None
    return model_state


def test_model_state_distinguishes_dummy_profile_batch() -> None:
    model_state = make_model_state()
    input_batch = make_input_batch(["profile-only"], [4], [0])
    runner_state = SimpleNamespace(req_id_to_index={})

    prepared = model_state.prepare_inputs(input_batch, runner_state)

    assert prepared["request_segments"] == ()


def test_model_state_does_not_hide_missing_real_hlm_state() -> None:
    model_state = make_model_state()
    input_batch = make_input_batch(["real"], [2], [0])
    runner_state = SimpleNamespace(req_id_to_index={"real": 0})

    with pytest.raises(RequestStateError, match="was not admitted"):
        model_state.prepare_inputs(input_batch, runner_state)


def test_model_state_rejects_mixed_real_and_dummy_batch() -> None:
    model_state = make_model_state()
    model_state.request_states.add_request("real", computed_tokens=0)
    input_batch = make_input_batch(["real", "profile-only"], [1, 1], [0, 0])
    runner_state = SimpleNamespace(req_id_to_index={"real": 0})

    with pytest.raises(RuntimeError, match="mixed real/dummy"):
        model_state.prepare_inputs(input_batch, runner_state)
