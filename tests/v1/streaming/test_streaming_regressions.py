# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests from the streaming deep review (findings C2, C20, C26).

Scheduler-level tests drive a REAL `Scheduler` (CPU, mocked model_config —
same pattern as tests/v1/streaming_input/test_scheduler_streaming.py) through
`schedule()` / `update_from_output()` so the exact production interleavings
are exercised:

  - C2:  close-during-re-prefill must NOT assert-kill the engine — the
         finish sentinel popped on the phantom-discard path frees the
         request with a proper FINISHED_ABORTED status.
  - C26: the phantom-sample discard (`reprefill_discard_next_sample`) drops
         exactly one token and emits no EngineCoreOutput; and the re-prefill
         TRIGGER path inside `_handle_stopped_request` (idle and folded
         variants, including the C19 discard-flag contract).
  - C20: `StreamingRetentionParams` rejects reprefill_threshold <= 0 and
         NaN; `Scheduler.add_request` rejects retention whose
         max_session_tokens would immediately re-trigger re-prefill.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.streaming.retention import StreamingRetentionParams
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

STOP_TOKEN = 128001

# Trained position range for the fake model. should_trigger_reprefill fires
# when max_cached_position > reprefill_threshold * this (0.7 * 10000 = 7000).
MODEL_MAX_POSITION = 10_000


def _make_retention(**kw) -> StreamingRetentionParams:
    kw.setdefault("max_video_segments", 30)
    kw.setdefault("max_session_tokens", 4000)
    kw.setdefault("reprefill_threshold", 0.7)
    return StreamingRetentionParams(**kw)


def _create_scheduler() -> Scheduler:
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.model_config.is_multimodal_model = False
    # A truthy MagicMock here would trip the encoder-decoder assert in
    # Scheduler.__init__ (mm_budget is None for non-multimodal configs).
    vllm_config.model_config.is_encoder_decoder = False
    vllm_config.model_config.max_model_len = 1024
    vllm_config.model_config.enable_return_routed_experts = False
    vllm_config.model_config.uses_mrope = True
    # Retention sessions are rejected under async scheduling; this suite
    # models the deployed config (async scheduling off).
    vllm_config.scheduler_config.async_scheduling = False
    # Real int (a bare MagicMock would coerce via __int__ to a garbage
    # value); this is the operand of should_trigger_reprefill.
    vllm_config.model_config.hf_text_config.max_position_embeddings = MODEL_MAX_POSITION
    vllm_config.cache_config = MagicMock()
    vllm_config.cache_config.num_gpu_blocks = 1000
    vllm_config.cache_config.enable_prefix_caching = False
    kv_cache_config = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=16,
        hash_block_size=16,
    )


def _make_chunk(
    request_id: str,
    prompt_token_ids: list[int],
    *,
    resumable: bool = True,
    retention: StreamingRetentionParams | None = None,
    first_chunk: bool = True,
) -> Request:
    extra_args = {"streaming_retention": retention} if retention is not None else None
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            stop_token_ids=[STOP_TOKEN], max_tokens=16, extra_args=extra_args
        ),
        pooling_params=None,
        resumable=resumable,
        # The frontend stamps first_chunk on session-opening requests
        # (async_llm.handle_inputs); these tests construct them directly.
        first_chunk=first_chunk,
    )


def _mro(req_id: str, token: int, max_cached_position: int | None = None):
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[token]],
        logprobs=None,
        prompt_logprobs_dict={req_id: None},
        pooler_output=[],
        max_cached_positions=(
            {req_id: max_cached_position} if max_cached_position is not None else {}
        ),
    )


def _tokens_in_outputs(eco_dict, req_id: str) -> list[int]:
    """All token ids emitted for `req_id` across the returned client dict."""
    tokens: list[int] = []
    for engine_core_outputs in eco_dict.values():
        for output in engine_core_outputs.outputs:
            if output.request_id == req_id:
                tokens.extend(output.new_token_ids)
    return tokens


def _drive_session_to_reprefill(scheduler: Scheduler, session: Request):
    """Prefill + one decode step, then a stop with a high reported
    max_cached_position so `_handle_stopped_request` takes the idle
    re-prefill trigger path. Returns the stop step's outputs."""
    rid = session.request_id
    scheduler.add_request(session)

    # Prefill the prompt, sample one ordinary caption token.
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[rid] == len(session.prompt_token_ids)
    scheduler.update_from_output(out, _mro(rid, 10))
    assert session.status == RequestStatus.RUNNING

    # Decode step ends the chunk with a stop token; the worker reports a
    # position watermark above the 0.7 * MODEL_MAX_POSITION threshold, so
    # the stop handler triggers a re-prefill.
    out = scheduler.schedule()
    eco = scheduler.update_from_output(
        out, _mro(rid, STOP_TOKEN, max_cached_position=7500)
    )
    return eco


# ---------------------------------------------------------------------------
# C2: close-during-re-prefill must not assert-kill the engine
# ---------------------------------------------------------------------------


def test_close_during_reprefill_finishes_cleanly():
    """Guards C2: a session-finish sentinel (None) queued while a re-prefill
    is in flight is popped on the phantom-discard path of
    `update_from_output`; before the fix `_free_request` was reached with the
    request still RUNNING and its `assert request.is_finished()` killed the
    whole EngineCore. Now the request must finish as FINISHED_ABORTED, leave
    the registry, and free its blocks — with no output emitted."""
    scheduler = _create_scheduler()
    free_q = scheduler.kv_cache_manager.block_pool.free_block_queue
    free_blocks_initial = free_q.num_free_blocks

    session = _make_chunk("sess-c2", [1, 2, 3], retention=_make_retention())
    _drive_session_to_reprefill(scheduler, session)
    assert session.pending_reprefill is True
    assert session.reprefill_discard_next_sample is True
    assert session.status == RequestStatus.WAITING

    # Client closes the session while the re-prefill is pending: the
    # non-resumable final request becomes a None sentinel appended to
    # streaming_queue (status is WAITING, not WAITING_FOR_STREAMING_REQ).
    close_req = _make_chunk("sess-c2", [0], resumable=False)
    scheduler.add_request(close_req)
    assert list(session.streaming_queue) == [None]

    # Re-prefill executes and samples its one forced token. The discard path
    # pops the sentinel; the fixed code must set FINISHED_ABORTED before
    # freeing instead of tripping the is_finished() assert (engine death).
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["sess-c2"] == session.num_tokens
    eco = scheduler.update_from_output(out, _mro("sess-c2", 42))

    assert session.status == RequestStatus.FINISHED_ABORTED
    assert "sess-c2" not in scheduler.requests
    assert session not in scheduler.running
    assert session not in list(scheduler.waiting)
    # The phantom sample was discarded, not delivered.
    assert 42 not in session._all_token_ids
    assert _tokens_in_outputs(eco, "sess-c2") == []
    # All KV blocks returned to the pool.
    assert free_q.num_free_blocks == free_blocks_initial
    # A stale post-abort chunk (non-first, unknown id) cannot resurrect the
    # session: only first chunks may create sessions, by construction.
    stale = _make_chunk(
        "sess-c2", [7, 8], retention=_make_retention(), first_chunk=False
    )
    scheduler.add_request(stale)
    assert "sess-c2" not in scheduler.requests


def test_stale_chunk_after_close_during_reprefill_is_dropped():
    """Companion to C2: a chunk ADD racing behind the abort must not
    resurrect the freed session as a zombie."""
    scheduler = _create_scheduler()
    session = _make_chunk("sess-c2b", [1, 2, 3], retention=_make_retention())
    _drive_session_to_reprefill(scheduler, session)
    scheduler.add_request(_make_chunk("sess-c2b", [0], resumable=False))
    out = scheduler.schedule()
    scheduler.update_from_output(out, _mro("sess-c2b", 42))
    assert "sess-c2b" not in scheduler.requests

    stale = _make_chunk(
        "sess-c2b", [7, 8], retention=_make_retention(), first_chunk=False
    )
    scheduler.add_request(stale)
    assert "sess-c2b" not in scheduler.requests
    assert len(scheduler.waiting) == 0


# ---------------------------------------------------------------------------
# C26 (2): phantom-sample discard
# ---------------------------------------------------------------------------


def test_phantom_sample_discarded_without_output():
    """Guards C26 gap (2): after an idle-triggered re-prefill the one forced
    sample is a phantom — `update_from_output` must drop exactly that token,
    emit NO EngineCoreOutput, clear the flag, and park the session waiting
    for the next frame (a regression here shifts every caption by one
    frame; previously only a 3h soak surfaced it)."""
    scheduler = _create_scheduler()
    session = _make_chunk("sess-phantom", [1, 2, 3], retention=_make_retention())
    _drive_session_to_reprefill(scheduler, session)
    tokens_before = list(session._all_token_ids)

    # Re-prefill runs; the forced sample (42) must be discarded.
    out = scheduler.schedule()
    eco = scheduler.update_from_output(out, _mro("sess-phantom", 42))

    assert session.reprefill_discard_next_sample is False
    assert session._all_token_ids == tokens_before, "phantom token leaked in"
    assert _tokens_in_outputs(eco, "sess-phantom") == []
    # Session is alive, idle, waiting for the next frame.
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1
    assert "sess-phantom" in scheduler.requests

    # And the session still accepts the next frame normally.
    scheduler.add_request(
        _make_chunk("sess-phantom", [7, 8], retention=_make_retention())
    )
    assert session.status == RequestStatus.WAITING
    assert list(session._all_token_ids[-2:]) == [7, 8]
    assert scheduler.num_waiting_for_streaming_input == 0


# ---------------------------------------------------------------------------
# C26 (3): re-prefill trigger path inside _handle_stopped_request
# ---------------------------------------------------------------------------


def test_reprefill_trigger_idle_path_state():
    """Guards C26 gap (3): the trigger integration inside
    `_handle_stopped_request` (idle variant): kept output folded into the
    prompt, position state reset, request re-queued at the HEAD of waiting,
    the waiting-for-input counter balanced, and the phantom-discard flag set
    (idle => the forced sample is a phantom)."""
    scheduler = _create_scheduler()
    session = _make_chunk("sess-trigger", [1, 2, 3], retention=_make_retention())
    eco = _drive_session_to_reprefill(scheduler, session)

    # The chunk's stop was still delivered to the frontend.
    stop_tokens = _tokens_in_outputs(eco, "sess-trigger")
    assert stop_tokens and stop_tokens[-1] == STOP_TOKEN

    assert session.status == RequestStatus.WAITING
    assert list(scheduler.waiting)[0] is session
    assert scheduler.num_waiting_for_streaming_input == 0
    assert session.pending_reprefill is True
    assert session.reprefill_discard_next_sample is True
    assert session.reprefill_count == 1
    # Position state reset for the dense-from-0 re-prefill.
    assert session.num_computed_tokens == 0
    assert session.max_cached_position == -1
    # Kept output folded into the prompt; the stop token was discarded.
    assert session.prompt_token_ids == [1, 2, 3, 10]
    assert STOP_TOKEN not in session.prompt_token_ids
    # The folded caption was recorded as an evictable assistant_text segment.
    assert any(
        seg.segment_type == "assistant_text" and seg.token_range == (3, 4)
        for seg in session.session_history
    )


def test_reprefill_trigger_folded_path_keeps_first_sample():
    """Guards C26 gap (3) folded variant + the C19 contract: when the stop
    handler folds a QUEUED frame before triggering re-prefill, the re-prefilled
    prompt ends with that unanswered frame, so `reprefill_discard_next_sample`
    must be False and the first post-re-prefill sample must be delivered
    (discarding it wedges the session / shifts captions)."""
    scheduler = _create_scheduler()
    session = _make_chunk("sess-folded", [1, 2, 3], retention=_make_retention())
    rid = session.request_id
    scheduler.add_request(session)

    out = scheduler.schedule()
    scheduler.update_from_output(out, _mro(rid, 10))
    assert session.status == RequestStatus.RUNNING

    # Pipeline the next frame while the session is still decoding: it lands
    # in streaming_queue.
    scheduler.add_request(_make_chunk(rid, [7, 8], retention=_make_retention()))
    assert len(session.streaming_queue) == 1

    # Stop with the position watermark above threshold: the handler folds
    # the queued frame FIRST, then triggers re-prefill.
    out = scheduler.schedule()
    scheduler.update_from_output(out, _mro(rid, STOP_TOKEN, max_cached_position=7500))

    assert session.pending_reprefill is True
    assert session.reprefill_discard_next_sample is False, (
        "folded trigger must keep the first post-re-prefill sample: the "
        "prompt now ends with the unanswered queued frame"
    )
    assert session.status == RequestStatus.WAITING
    # Kept output (10) folded, stop discarded, queued frame appended.
    assert session.prompt_token_ids == [1, 2, 3, 10, 7, 8]

    # Re-prefill completes: the first sample is frame N+1's first REAL
    # caption token and must be appended + emitted.
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[rid] == session.num_tokens
    eco = scheduler.update_from_output(out, _mro(rid, 43))
    assert session._all_token_ids[-1] == 43
    assert _tokens_in_outputs(eco, rid) == [43]
    assert session.status == RequestStatus.RUNNING


# ---------------------------------------------------------------------------
# C20: retention-params validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_threshold", [0.0, -0.5, float("nan")])
def test_retention_rejects_nonpositive_or_nan_threshold(bad_threshold):
    """Guards C20: engine-side StreamingRetentionParams must reject
    reprefill_threshold <= 0 (immediate re-prefill livelock) and NaN (which
    silently disables re-prefill -> unbounded position growth)."""
    with pytest.raises(ValueError, match="reprefill_threshold"):
        StreamingRetentionParams(
            max_video_segments=30,
            max_session_tokens=4000,
            reprefill_threshold=bad_threshold,
        )


def test_retention_accepts_valid_thresholds():
    """C20 companion: in-range values construct; >= 1.0 means 'disable
    re-prefill' and stays a warning, not an error."""
    StreamingRetentionParams(
        max_video_segments=30, max_session_tokens=4000, reprefill_threshold=0.7
    )
    StreamingRetentionParams(
        max_video_segments=30, max_session_tokens=4000, reprefill_threshold=1.0
    )


def test_add_request_rejects_retention_that_would_loop():
    """Guards C20 fix (2): `Scheduler.add_request` must reject a retention
    whose max_session_tokens >= reprefill_threshold * model_max_position —
    the surviving prompt would re-cross the trigger right after every
    re-prefill (GPU-burn livelock). Checked engine-side with the scheduler's
    own _model_max_position, the exact trigger operand."""
    scheduler = _create_scheduler()
    # 4000 >= 0.2 * 10000 = 2000 -> reject at admission.
    bad = _make_chunk(
        "sess-loop",
        [1, 2, 3],
        retention=_make_retention(reprefill_threshold=0.2),
    )
    with pytest.raises(ValueError, match="max_session_tokens"):
        scheduler.add_request(bad)
    assert "sess-loop" not in scheduler.requests

    # The same budget with the default threshold is fine (4000 < 7000).
    ok = _make_chunk("sess-ok", [1, 2, 3], retention=_make_retention())
    scheduler.add_request(ok)
    assert "sess-ok" in scheduler.requests
