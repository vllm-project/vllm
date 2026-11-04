# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for request-scoped HLM state under the V2 runner batch contract."""

from dataclasses import replace

import numpy as np
import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.models.ncp_olmo.model_state import NCPOlmoModelState
from vllm.model_executor.models.ncp_olmo.state import (
    ConceptRequestStateStore,
    HLMKVState,
    RequestStateError,
    active_tensor_buffer,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.states import RequestState


def make_input_batch(
    req_ids: list[str],
    scheduled_tokens: list[int],
    computed_tokens: list[int],
) -> InputBatch:
    """Build the CPU metadata consumed from a V2 ``InputBatch``."""
    batch = InputBatch.make_dummy(
        len(req_ids),
        sum(scheduled_tokens),
        InputBuffers(len(req_ids), sum(scheduled_tokens), torch.device("cpu")),
    )
    scheduled = np.asarray(scheduled_tokens, dtype=np.int32)
    computed = np.asarray(computed_tokens, dtype=np.int32)
    query_start_loc = np.concatenate(
        (np.zeros(1, dtype=np.int32), scheduled.cumsum(dtype=np.int32))
    )
    seq_lens = torch.from_numpy(computed + scheduled)
    return replace(
        batch,
        req_ids=req_ids,
        num_scheduled_tokens=scheduled,
        num_computed_tokens_np=computed,
        num_tokens=sum(scheduled_tokens),
        query_start_loc=torch.from_numpy(query_start_loc),
        query_start_loc_np=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        positions=torch.cat(
            [
                torch.arange(start, start + count)
                for start, count in zip(computed_tokens, scheduled_tokens)
            ]
        ),
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


def test_batch_invariant_hlm_advances_requests_individually(
    ncp_model, monkeypatch
) -> None:
    """Keep HLM arithmetic independent of incidental scheduler batching."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    _check_hlm_batch(ncp_model, exact=True)


def test_hlm_decode_preserves_output_without_host_sync(ncp_model) -> None:
    attention = ncp_model.highlevel.concept_predictor.hlm_block.layers[0].self_attention
    torch.manual_seed(42)
    hidden = torch.randn(
        2, ncp_model.backend_config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    reference_state, checked_state = HLMKVState(), HLMKVState()
    with torch.inference_mode():
        attention(0, hidden[:1], reference_state)
        attention(0, hidden[:1], checked_state)
        expected = attention(1, hidden[1:], reference_state)
        previous_mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            actual = attention(1, hidden[1:], checked_state)
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert checked_state.length == reference_state.length == 2


def test_batch_invariant_multichunk_prefill_advances_in_fixed_order(
    ncp_model,
    monkeypatch,
) -> None:
    """Make HLM results independent of chunked-prefill partitioning."""
    config = ncp_model.backend_config
    store = ncp_model.request_states
    store.add_request("request", computed_tokens=0)
    store.add_request("reference", computed_tokens=0)
    num_tokens = 2 * config.chunk_size
    segment = store.resolve_segments(make_input_batch(["request"], [num_tokens], [0]))[
        0
    ]
    torch.manual_seed(42)
    encoder_hidden = torch.randn(
        num_tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    encoder_raw_layers = (
        encoder_hidden + 1,
        encoder_hidden + 2,
    )
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    with torch.inference_mode():
        advance = ncp_model._advance_completed_chunks(
            segment, encoder_hidden, encoder_raw_layers
        )
        for start in range(0, num_tokens, config.chunk_size):
            chunk = slice(start, start + config.chunk_size)
            ncp_model.highlevel.advance(
                store.states["reference"],
                encoder_hidden[chunk].mean(dim=0).to(encoder_hidden.dtype),
                tuple(
                    layer[chunk].mean(dim=0).to(layer.dtype)
                    for layer in encoder_raw_layers
                ),
            )
    assert advance is None
    assert store.states["request"].predicted_concepts.length == 2
    assert torch.equal(
        active_tensor_buffer(store.states["request"].predicted_concepts),
        active_tensor_buffer(store.states["reference"].predicted_concepts),
    )


def test_normal_hlm_path_batches_equal_position_requests(
    ncp_model, monkeypatch
) -> None:
    """Retain the throughput path when exact batch invariance is disabled."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    _check_hlm_batch(ncp_model, exact=False)


def _check_hlm_batch(model, *, exact: bool) -> None:
    store = model.request_states
    for req_id in ("first", "second", "ref-first", "ref-second"):
        store.add_request(req_id, computed_tokens=0)
    torch.manual_seed(42)
    chunks = torch.randn(
        2, model.backend_config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    layers = (chunks + 1, chunks + 2)
    with torch.inference_mode():
        model._advance_hlm_batches(
            [
                (store.states[req_id], chunks[i], tuple(layer[i] for layer in layers))
                for i, req_id in enumerate(("first", "second"))
            ]
        )
        for i, req_id in enumerate(("first", "second")):
            model.highlevel.advance(
                store.states[f"ref-{req_id}"],
                chunks[i],
                tuple(layer[i] for layer in layers),
            )
            actual = active_tensor_buffer(store.states[req_id].predicted_concepts)
            expected = active_tensor_buffer(
                store.states[f"ref-{req_id}"].predicted_concepts
            )
            assert store.states[req_id].predicted_concepts.length == 1
            assert torch.isfinite(actual).all()
            if exact:
                assert torch.equal(actual, expected)
            else:
                torch.testing.assert_close(actual, expected)


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


@pytest.fixture
def model_state(ncp_config, ncp_model):
    return NCPOlmoModelState(ncp_config, ncp_model, None, torch.device("cuda"))


@pytest.fixture
def runner_state():
    return RequestState(4, 128, 128, 0, 128, torch.device("cuda"))


def test_model_state_distinguishes_dummy_profile_batch(
    model_state, runner_state
) -> None:
    input_batch = make_input_batch(["profile-only"], [4], [0])

    prepared = model_state.prepare_inputs(input_batch, runner_state)

    assert prepared["request_segments"] == ()


def test_model_state_does_not_hide_missing_real_hlm_state(
    model_state, runner_state
) -> None:
    input_batch = make_input_batch(["real"], [2], [0])
    runner_state.add_request("real", 2, [1, 2], 0, 4)

    with pytest.raises(RequestStateError, match="was not admitted"):
        model_state.prepare_inputs(input_batch, runner_state)


def test_model_state_rejects_mixed_real_and_dummy_batch(
    model_state, runner_state
) -> None:
    model_state.request_states.add_request("real", computed_tokens=0)
    input_batch = make_input_batch(["real", "profile-only"], [1, 1], [0, 0])
    runner_state.add_request("real", 1, [1], 0, 4)

    with pytest.raises(RuntimeError, match="mixed real/dummy"):
        model_state.prepare_inputs(input_batch, runner_state)
