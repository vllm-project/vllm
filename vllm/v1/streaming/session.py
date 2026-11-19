# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler hooks for opt-in bounded streaming sessions."""

from dataclasses import fields
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.request import RequestStatus
from vllm.v1.streaming.eviction import (
    drop_old_segments_for_reprefill,
    maybe_evict_old_segments,
)
from vllm.v1.streaming.reprefill import (
    note_frame_turn,
    rebuild_transcript_for_reprefill,
)
from vllm.v1.streaming.retention import HistorySegment

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request, StreamingUpdate

logger = init_logger(__name__)

_PROMPT_SCORING_ERROR = "Bounded streaming sessions do not support prompt scoring"


def _frame_replay_tokens(params: "SamplingParams", num_tokens: int) -> int:
    count = (params.extra_args or {}).get("streaming_frame_replay_tokens", 0)
    if type(count) is not int or count not in (0, 1) or count > num_tokens:
        raise ValueError(
            "streaming_frame_replay_tokens must be 0 or 1 within the input chunk"
        )
    return count


def _replayable_images(features: list["MultiModalFeatureSpec"]) -> bool:
    return all(
        feature.modality == "image"
        and feature.data is not None
        and "address" not in feature.data
        for feature in features
    )


def validate_streaming_config(config: "VllmConfig") -> None:
    parallel = config.parallel_config
    if (
        not config.model_config.enforce_eager
        or config.scheduler_config.async_scheduling
        or config.speculative_config is not None
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or config.kv_transfer_config is not None
        or config.ec_transfer_config is not None
    ):
        raise ValueError(
            "Streaming retention requires --enforce-eager, "
            "--no-async-scheduling, PP/DP/CP size 1, and no speculative "
            "decoding or KV/encoder transfer connectors."
        )
    model = config.model_config
    mm_config = model.multimodal_config
    if (
        model.runner_type != "generate"
        or model.is_encoder_decoder
        or model.is_diffusion
        or model.is_hybrid
        or model.get_sliding_window() is not None
    ) or (mm_config is not None and mm_config.is_multimodal_pruning_enabled()):
        raise ValueError(
            "Streaming retention does not support encoder-decoder models "
            "or multimodal pruning."
        )


def _model_max_position(scheduler: "Scheduler") -> int:
    model = scheduler.vllm_config.model_config
    return int(
        getattr(model.hf_text_config, "max_position_embeddings", model.max_model_len)
    )


def validate_streaming_request(scheduler: "Scheduler", request: "Request") -> None:
    validate_streaming_config(scheduler.vllm_config)
    params = request.sampling_params
    if (
        params is None
        or params.n != 1
        or params.output_kind != RequestOutputKind.DELTA
        or params.stop
    ):
        raise ValueError(
            "Bounded sessions require DELTA output, n=1, and no stop strings"
        )
    if (
        params.prompt_logprobs is not None
        or params.prompt_logprob_token_ids is not None
    ):
        raise ValueError(_PROMPT_SCORING_ERROR)
    if _frame_replay_tokens(params, request.num_prompt_tokens):
        raise ValueError("The initial frame cannot replay a previous reply token")
    # Presence certifies image-only rendering; the value owns any replayed reply
    # token. Generic SDK turns cannot establish that their text is disposable.
    request.streaming_frames_uniform = (
        "streaming_frame_replay_tokens" in (params.extra_args or {})
        and len(request.mm_features) == 1
    )
    groups = scheduler.kv_cache_config.kv_cache_groups
    if len(groups) != 1 or groups[0].manager_cls is not FullAttentionManager:
        raise ValueError("Streaming retention requires one full-attention KV group")
    if any(type(spec) is not AttentionSpec for spec in groups[0].distinct_specs):
        raise ValueError("Streaming retention requires ordinary full-attention KV")
    if request.prompt_embeds is not None or request.prompt_is_token_ids is not None:
        raise ValueError("Streaming retention does not support prompt embeddings")
    if not _replayable_images(request.mm_features):
        raise ValueError(
            "Streaming retention requires text or replayable image payloads; "
            "render image chunks with skip_mm_cache=True"
        )
    retention = request.streaming_retention
    assert retention is not None
    if request.num_prompt_tokens > retention.max_session_tokens:
        raise ValueError("Initial prompt exceeds streaming max_session_tokens")
    if retention.max_session_tokens + request.max_tokens > scheduler.max_model_len:
        raise ValueError("Streaming token budgets exceed max_model_len")
    if (
        retention.max_session_tokens + request.max_tokens
        >= retention.reprefill_threshold * _model_max_position(scheduler)
    ):
        raise ValueError(
            "max_session_tokens + max_tokens must be below "
            "reprefill_threshold * the model position limit"
        )


def reset_streaming_positions(request: "Request") -> None:
    request.position_offset = 0
    request.max_cached_position = -1
    request.pending_evicted_token_ranges.clear()
    request.pending_reprefill = True


def admit_streaming_request(scheduler: "Scheduler", request: "Request") -> bool:
    try:
        validate_streaming_request(scheduler, request)
    except ValueError as error:
        logger.warning(
            "Rejecting bounded streaming session %s: %s", request.request_id, error
        )
        request.resumable = False
        request.status = RequestStatus.FINISHED_ERROR
        request.stop_reason = str(error)
        scheduler.dropped_streaming_updates.append(request)
        scheduler.finished_req_ids.add(request.request_id)
        if scheduler.finished_req_ids_dict is not None:
            scheduler.finished_req_ids_dict[request.client_index].add(
                request.request_id
            )
        return False
    return True


def update_streaming_session(
    scheduler: "Scheduler",
    session: "Request",
    update: "StreamingUpdate",
    prior_prompt_tokens: int,
    chunk_start: int,
    prior_sampling_params: "SamplingParams | None",
) -> bool:
    """Record a chunk, compact old KV, and consolidate before position overflow.

    False means the pinned prompt or newest chunk cannot fit the configured
    budget. The caller terminates this session without scheduling unsafe input.
    """
    params = session.sampling_params
    if params is None:
        session.stop_reason = "Bounded sessions require sampling parameters"
        return False
    try:
        replay_tokens = _frame_replay_tokens(params, len(update.prompt_token_ids or ()))
    except ValueError as error:
        session.stop_reason = str(error)
        return False
    if (
        "streaming_frame_replay_tokens" not in (params.extra_args or {})
        or len(update.mm_features or ()) != 1
    ):
        session.streaming_frames_uniform = False
    user_start = chunk_start + replay_tokens
    if replay_tokens and any(
        feature.mm_position.offset < user_start for feature in update.mm_features or ()
    ):
        session.stop_reason = "Replayed reply tokens cannot contain image placeholders"
        return False
    if not session.session_history and prior_prompt_tokens:
        session.session_history.append(
            HistorySegment("system_prompt", (0, prior_prompt_tokens), pinned=True)
        )
    for segment in session.session_history:
        segment.age_chunks += 1
    if user_start > prior_prompt_tokens:
        session.session_history.append(
            HistorySegment("assistant_text", (prior_prompt_tokens, user_start))
        )
    if session.num_prompt_tokens > user_start:
        session.session_history.append(
            HistorySegment(
                "video" if update.mm_features else "user_text",
                (user_start, session.num_prompt_tokens),
            )
        )
        if update.mm_features:
            note_frame_turn(session, user_start, session.num_prompt_tokens)

    retention = session.streaming_retention
    assert retention is not None
    if (
        params.prompt_logprobs is not None
        or params.prompt_logprob_token_ids is not None
    ):
        session.stop_reason = _PROMPT_SCORING_ERROR
        return False
    if prior_sampling_params is None or any(
        getattr(params, name) != getattr(prior_sampling_params, name)
        for name in (
            "n",
            "output_kind",
            "detokenize",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "min_tokens",
            "include_stop_str_in_output",
            "stop",
            "logprobs",
            "flat_logprobs",
            "stream_interval",
        )
    ):
        session.stop_reason = (
            "Bounded session output settings cannot change between turns"
        )
        return False
    old_output = prior_sampling_params.structured_outputs
    new_output = params.structured_outputs
    if (old_output is None) != (new_output is None):
        session.stop_reason = (
            "Bounded session output constraints cannot change between turns"
        )
        return False
    if (
        old_output is not None
        and new_output is not None
        and any(
            getattr(old_output, field.name) != getattr(new_output, field.name)
            for field in fields(old_output)
            if field.init
        )
    ):
        session.stop_reason = (
            "Bounded session output constraints cannot change between turns"
        )
        return False
    maybe_evict_old_segments(
        session,
        retention,
        scheduler.kv_cache_manager,
        scheduler.encoder_cache_manager,
        pending_encoder_input_ids=scheduler._pending_encoder_input_ids.get(
            session.request_id
        ),
    )
    force_reprefill = False
    if session.num_prompt_tokens > retention.max_session_tokens:
        force_reprefill = bool(
            drop_old_segments_for_reprefill(
                session,
                retention,
                scheduler.encoder_cache_manager,
                pending_encoder_input_ids=scheduler._pending_encoder_input_ids.get(
                    session.request_id
                ),
            )
        )
    if (
        session.num_prompt_tokens > retention.max_session_tokens
        or retention.max_session_tokens + session.max_tokens > scheduler.max_model_len
        or retention.max_session_tokens + session.max_tokens
        >= retention.reprefill_threshold * _model_max_position(scheduler)
        or not _replayable_images(update.mm_features or [])
    ):
        session.stop_reason = (
            "Bounded session exceeds its token/position budget or requires "
            "non-replayable image inputs"
        )
        return False

    highest_position = (
        max(
            session.max_cached_position + len(update.prompt_token_ids or ()),
            session.num_tokens - 1 + session.position_offset,
        )
        + session.max_tokens
    )
    if (
        force_reprefill
        or highest_position
        >= retention.reprefill_threshold * _model_max_position(scheduler)
    ):
        rebuild_transcript_for_reprefill(
            session,
            scheduler.encoder_cache_manager,
            pending_encoder_input_ids=scheduler._pending_encoder_input_ids.get(
                session.request_id
            ),
        )
        scheduler.kv_cache_manager.free(session)
        scheduler.encoder_cache_manager.free(session)
        scheduler._pending_encoder_input_ids.pop(session.request_id, None)
        session.num_computed_tokens = 0
        reset_streaming_positions(session)
        session.reprefill_count += 1
        logger.info(
            "Consolidating streaming session %s: surviving_tokens=%d count=%d",
            session.request_id,
            session.num_prompt_tokens,
            session.reprefill_count,
        )

    structured = session.structured_output_request
    if structured is not None:
        grammar = structured.grammar
        if isinstance(grammar, Exception):
            session.stop_reason = "Bounded session structured-output grammar failed"
            return False
        if grammar is not None:
            grammar.reset()
    return True
