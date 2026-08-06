# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import math
import json
import os
import re
import time
from collections import deque
from collections.abc import AsyncGenerator
from http import HTTPStatus
from uuid import uuid4

import numpy as np
import pybase64 as base64
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from vllm import envs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.serve.utils.api_utils import sanitize_message
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

from .protocol import (
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    RealtimeSessionConfig,
    SessionCreated,
    SessionUpdated,
    TranscriptionDelta,
    TranscriptionDone,
)
from .serving import OpenAIServingRealtime

logger = init_logger(__name__)

QWEN3_ASR_REALTIME_PATCH_ID = "fresh-segment-v2"
_AUDIO_HISTORY_MODE = "audio_history_kv"
_AUT_STABLE_WINDOW_MODE = "aut_stable_window_kv"
_ASR_HEADER_RE = re.compile(
    r"^\s*language\s+[^<]*?<\s*asr_text\s*>\s*",
    re.IGNORECASE,
)


def _normalize_asr_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "").replace("\n", "")
    return _ASR_HEADER_RE.sub("", text, count=1).strip()


def _normalize_realtime_candidate(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"(?:\n[ \t]*){2,}", "\n", text)


def _normalize_realtime_text(text: str) -> str:
    """Remove protocol markers and formatting artifacts before sending text."""
    if not text:
        return ""
    text = text.replace("\r", "").replace("\n", "")
    if "<asr_text>" in text:
        text = text.rsplit("<asr_text>", 1)[1]
    elif text.startswith("language "):
        text = ""
    return text.strip()


def _holdback_rollback(text: str, tokenizer, rollback_tokens: int) -> str:
    """Return *text* with the last *rollback_tokens* tokens removed.

    This is the mirror of ``_rollback_prefix`` in the model file: it
    determines the "stable" portion of a segment's generation that is
    safe to send to the client.  The held-back tail will be re-decided
    by the next segment (or flushed at stream end).
    """
    if not text or rollback_tokens <= 0:
        return text
    token_ids = tokenizer.encode(text)
    end_idx = max(0, len(token_ids) - rollback_tokens)
    if end_idx == 0:
        return ""
    stable = tokenizer.decode(token_ids[:end_idx])
    while "\ufffd" in stable and end_idx > 0:
        end_idx -= 1
        stable = tokenizer.decode(token_ids[:end_idx]) if end_idx > 0 else ""
    return stable


def _encode_text(tokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _decode_text(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def _decode_candidate_text(
    tokenizer,
    prefix_text: str,
    generated_token_ids: list[int],
    generated_text: str,
) -> str:
    """Decode one complete candidate without guessing word boundaries.

    English tokenizers commonly carry the separator space on the first token
    of a continuation.  Normalizing ``generated_text`` on its own used to
    strip that space before ``prefix + continuation`` was assembled, producing
    strings such as ``therewould``.  Decode the complete token sequence in one
    pass instead.  The same operation leaves Chinese text adjacent naturally.
    """
    if generated_token_ids:
        candidate_ids = _encode_text(tokenizer, prefix_text)
        candidate_ids.extend(generated_token_ids)
        candidate = _decode_text(tokenizer, candidate_ids)
    else:
        candidate = prefix_text + generated_text
    return _normalize_realtime_text(candidate)


def _token_lcp_len(left: list[int], right: list[int]) -> int:
    length = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        length += 1
    return length


def _repetition_reason(token_ids: list[int]) -> str | None:
    """Return a diagnostic reason for obvious decoder loops."""
    if not token_ids:
        return None

    run = 1
    for index in range(1, len(token_ids)):
        if token_ids[index] == token_ids[index - 1]:
            run += 1
            if run >= 8:
                return "same_token_run"
        else:
            run = 1

    for period in range(1, min(16, len(token_ids) // 4) + 1):
        repeated = token_ids[-period:]
        repeats = 1
        cursor = len(token_ids) - period
        while cursor >= period and token_ids[cursor - period:cursor] == repeated:
            repeats += 1
            cursor -= period
        if repeats >= 4 and repeats * period >= 16:
            return f"periodic_tokens_{period}"
    return None


def _candidate_reject_reason(
    candidate: str,
    token_ids: list[int],
    finish_reason: str | None,
    has_eos: bool,
) -> str | None:
    """Return why a completed realtime candidate must not become prefix state."""
    if not candidate:
        return "empty_candidate"
    if finish_reason == "length" and not has_eos:
        return "max_tokens_without_eos"
    return _repetition_reason(token_ids)


def _record_phase_timing(record: dict) -> None:
    """Log and optionally persist one realtime segment timing record."""
    logger.info(
        "QWEN3_ASR_RT_PHASE_TIMING request_id=%s segment_id=%s mode=%s "
        "ttft_s=%.6f segment_total_s=%.6f generated_tokens=%d "
        "prefill_proxy_prompt_tokens=%d decode_tokens=%d",
        record["request_id"],
        record["segment_id"],
        record["mode"],
        record["ttft_s"],
        record["segment_total_s"],
        record["generated_tokens"],
        record["prompt_tokens"],
        record["decode_tokens"],
    )
    output_path = os.environ.get("QWEN3_ASR_PHASE_TIMING_PATH", "").strip()
    if output_path:
        try:
            with open(output_path, "a", encoding="utf-8") as timing_file:
                timing_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to persist realtime phase timing: %s", exc)


class _CanonicalTranscript:
    """Append-only committed text plus a revisable last valid candidate."""

    def __init__(self, tokenizer, rollback_tokens: int, request_id: str):
        self.tokenizer = tokenizer
        self.rollback_tokens = max(0, rollback_tokens)
        self.request_id = request_id
        self.committed_ids: list[int] = []
        self.committed_text = ""
        self.previous_candidate_ids: list[int] | None = None
        self.latest_valid_ids: list[int] = []
        self.latest_valid_text = ""
        self.rejected_candidates = 0
        self.final_conflict = False
        self.stats = {
            "candidate_shorter": 0,
            "candidate_revision": 0,
            "final_conflicts": 0,
        }

    def reject(self, segment_id: int, reason: str) -> None:
        self.rejected_candidates += 1
        logger.warning(
            "QWEN3_ASR_RT_CANDIDATE_REJECT request_id=%s segment_id=%d "
            "reason=%s committed_chars=%d latest_valid_chars=%d",
            self.request_id,
            segment_id,
            reason,
            len(self.committed_text),
            len(self.latest_valid_text),
        )

    def observe(self, candidate_text: str, *, delta_turn: bool = False) -> str:
        if not candidate_text:
            return ""
        if delta_turn:
            self.committed_text += candidate_text
            self.committed_ids = _encode_text(self.tokenizer, self.committed_text)
            self.previous_candidate_ids = list(self.committed_ids)
            self.latest_valid_ids = list(self.committed_ids)
            self.latest_valid_text = self.committed_text
            return candidate_text

        candidate_ids = _encode_text(self.tokenizer, candidate_text)
        committed_len = len(self.committed_ids)
        if (
            candidate_ids[:committed_len] != self.committed_ids
            or not candidate_text.startswith(self.committed_text)
        ):
            # A streaming candidate may legitimately revise the rollback tail.
            # Do not poison the whole request on that revision: keep the
            # append-only committed prefix, remember the newest candidate, and
            # let the next segment re-establish a common prefix.  The previous
            # behavior set ``final_conflict`` here, which made ``finalize``
            # return an old committed prefix and silently truncated the tail.
            self.stats["final_conflicts"] += 1
            self.previous_candidate_ids = candidate_ids
            self.latest_valid_ids = candidate_ids
            self.latest_valid_text = candidate_text
            return ""

        delta = ""
        if self.previous_candidate_ids is not None:
            lcp_len = _token_lcp_len(self.previous_candidate_ids, candidate_ids)
            if len(candidate_ids) < len(self.previous_candidate_ids):
                self.stats["candidate_shorter"] += 1
            if lcp_len < min(len(self.previous_candidate_ids), len(candidate_ids)):
                self.stats["candidate_revision"] += 1
            holdback_boundary = max(0, len(candidate_ids) - self.rollback_tokens)
            commit_boundary = min(lcp_len, holdback_boundary)
            if commit_boundary > committed_len:
                new_ids = candidate_ids[:commit_boundary]
                new_text = _decode_text(self.tokenizer, new_ids)
                if "\ufffd" not in new_text and new_text.startswith(self.committed_text):
                    delta = new_text[len(self.committed_text):]
                    self.committed_ids = new_ids
                    self.committed_text = new_text

        self.previous_candidate_ids = candidate_ids
        self.latest_valid_ids = candidate_ids
        self.latest_valid_text = candidate_text
        return delta

    def fallback_tail_tokens(self, prefix_text: str) -> list[int]:
        if not self.latest_valid_text.startswith(prefix_text):
            return []
        return _encode_text(
            self.tokenizer,
            self.latest_valid_text[len(prefix_text):],
        )

    def finalize(self) -> tuple[str, str]:
        if self.final_conflict:
            return "", self.committed_text
        if not self.latest_valid_text:
            return "", self.committed_text
        if not self.latest_valid_text.startswith(self.committed_text):
            return "", self.committed_text
        delta = self.latest_valid_text[len(self.committed_text):]
        self.committed_text = self.latest_valid_text
        self.committed_ids = list(self.latest_valid_ids)
        return delta, self.committed_text


class RealtimeConnection:
    """Manages WebSocket lifecycle and state for realtime transcription.

    This class handles:
    - WebSocket connection lifecycle (accept, receive, send, close)
    - Event routing (session.update, append, commit)
    - Audio buffering via asyncio.Queue
    - Generation task management
    - Error handling and cleanup
    """

    def __init__(self, websocket: WebSocket, serving: OpenAIServingRealtime):
        self.websocket = websocket
        self.connection_id = f"ws-{uuid4()}"
        self.serving = serving
        self.audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self.generation_task: asyncio.Task | None = None

        self._is_connected = False
        self._is_model_validated = False
        self._session_config = RealtimeSessionConfig()

        self._max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

    async def handle_connection(self):
        """Main connection loop."""
        await self.websocket.accept()
        logger.debug("WebSocket connection accepted: %s", self.connection_id)
        self._is_connected = True

        # Send session created event
        await self.send(SessionCreated())

        try:
            while True:
                message = await self.websocket.receive_text()
                try:
                    event = json.loads(message)
                    await self.handle_event(event)
                except json.JSONDecodeError:
                    await self.send_error("Invalid JSON", "invalid_json")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self.send_error(sanitize_message(str(e)), "processing_error")
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.connection_id)
            self._is_connected = False
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            await self.cleanup()

    def _check_model(self, model: str | None) -> None | ErrorResponse:
        if self.serving._is_model_supported(model):
            return None

        return self.serving.create_error_response(
            message=f"The model `{model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def handle_event(self, event: dict):
        """Route events to handlers.

        Supported event types:
        - session.update: Configure model
        - input_audio_buffer.append: Add audio chunk to queue
        - input_audio_buffer.commit: Start transcription generation
        """
        event_type = event.get("type")
        if event_type == "session.update":
            logger.debug("Session updated: %s", event)
            model = event.get("model")
            if model is None:
                await self.send_error("Missing required field: model", "invalid_event")
                return
            err = self._check_model(model)
            if err is not None:
                await self.send_error(err.error.message, "model_not_found")
                return
            self._is_model_validated = True

            if event.get("language") is not None:
                self._session_config.language = event["language"]
            if event.get("prompt") is not None:
                self._session_config.prompt = event["prompt"]
            if event.get("segment_duration_s") is not None:
                segment_duration_s = float(event["segment_duration_s"])
                if not math.isfinite(segment_duration_s) or segment_duration_s <= 0:
                    await self.send_error(
                        "segment_duration_s must be a finite positive number",
                        "invalid_event",
                    )
                    return
                self._session_config.segment_duration_s = segment_duration_s
            if event.get("rollback_tokens") is not None:
                self._session_config.rollback_tokens = int(
                    event["rollback_tokens"]
                )
            if event.get("unfixed_chunks") is not None:
                self._session_config.unfixed_chunks = int(
                    event["unfixed_chunks"]
                )
            if event.get("max_prefix_tokens") is not None:
                max_prefix_tokens = int(event["max_prefix_tokens"])
                if max_prefix_tokens < 0:
                    await self.send_error(
                        "max_prefix_tokens must be >= 0; 0 keeps the full prefix",
                        "invalid_event",
                    )
                    return
                self._session_config.max_prefix_tokens = max_prefix_tokens
            if event.get("max_audio_s") is not None:
                self._session_config.max_audio_s = float(
                    event["max_audio_s"]
                )
            if event.get("realtime_max_tokens") is not None:
                realtime_max_tokens = int(event["realtime_max_tokens"])
                if realtime_max_tokens <= 0:
                    await self.send_error(
                        "realtime_max_tokens must be a positive integer",
                        "invalid_event",
                    )
                    return
                self._session_config.realtime_max_tokens = realtime_max_tokens

            await self.send(
                SessionUpdated(
                    model=model,
                    segment_duration_s=self._session_config.segment_duration_s,
                    runtime_patch_id=(
                        "aut-stable-window-kv-v1"
                        if os.environ.get("QWEN3_ASR_REALTIME_MODE", "")
                        .strip().lower() == _AUT_STABLE_WINDOW_MODE
                        else "audio-history-kv-full-prefix-v3"
                        if os.environ.get("QWEN3_ASR_REALTIME_MODE", "")
                        .strip().lower() == _AUDIO_HISTORY_MODE
                        else QWEN3_ASR_REALTIME_PATCH_ID
                    ),
                    max_prefix_tokens=self._session_config.max_prefix_tokens,
                    realtime_max_tokens=(
                        self._session_config.realtime_max_tokens
                        or self.serving.model_cls.realtime_max_tokens
                    ),
                    max_model_len=self.serving.model_config.max_model_len,
                )
            )
        elif event_type == "input_audio_buffer.append":
            append_event = InputAudioBufferAppend(**event)
            try:
                audio_bytes = base64.b64decode(append_event.audio)
                # Convert PCM16 bytes to float32 numpy array
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                if len(audio_array) / 1024**2 > self._max_audio_filesize_mb:
                    raise VLLMValidationError(
                        "Maximum file size exceeded",
                        parameter="audio_filesize_mb",
                        value=len(audio_array) / 1024**2,
                    )
                if len(audio_array) == 0:
                    raise VLLMValidationError("Can't process empty audio.")

                # Put audio chunk in queue
                self.audio_queue.put_nowait(audio_array)

            except Exception as e:
                logger.error("Failed to decode audio: %s", e)
                await self.send_error("Invalid audio data", "invalid_audio")

        elif event_type == "input_audio_buffer.commit":
            if not self._is_model_validated:
                err_msg = (
                    "Model not validated. Make sure to validate the"
                    " model by sending a session.update event."
                )
                await self.send_error(
                    err_msg,
                    "model_not_validated",
                )
                return

            commit_event = InputAudioBufferCommit(**event)
            # final signals that the audio is finished
            if commit_event.final:
                self.audio_queue.put_nowait(None)
            else:
                await self.start_generation()
        else:
            await self.send_error(f"Unknown event type: {event_type}", "unknown_event")

    async def audio_stream_generator(self) -> AsyncGenerator[np.ndarray, None]:
        """Generator that yields audio chunks from the queue."""
        while True:
            audio_chunk = await self.audio_queue.get()
            if audio_chunk is None:  # Sentinel value to stop
                break
            yield audio_chunk

    async def start_generation(self):
        """Start the transcription generation task."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        audio_stream = self.audio_stream_generator()
        input_stream = asyncio.Queue[list[int]]()
        prefix_texts: deque[str] = deque()
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        segment_traces: deque[dict] = deque()

        streaming_input_gen = self.serving.transcribe_realtime(
            audio_stream, input_stream, self._session_config,
            prefix_texts=prefix_texts,
            request_id=request_id,
            segment_traces=segment_traces,
        )

        mode = os.environ.get("QWEN3_ASR_REALTIME_MODE", "").strip().lower()
        generation_fn = (
            self._run_generation_legacy
            if mode in {_AUDIO_HISTORY_MODE, _AUT_STABLE_WINDOW_MODE}
            else self._run_generation
        )
        self.generation_task = asyncio.create_task(
            generation_fn(streaming_input_gen, input_stream,
                          prefix_texts, request_id, segment_traces)
        )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
        prefix_texts: deque[str],
        request_id: str | None = None,
        segment_traces: deque[dict] | None = None,
    ):
        """Run one independent engine request per cumulative audio segment.

        The WebSocket session remains long-lived, but engine/KV state does not.
        This is deliberate: vLLM resumable streaming requests append prompt and
        multimodal state, which is not equivalent to cumulative re-decoding.
        """
        prompt_tokens_total = 0
        completion_tokens_total = 0
        segments_started = 0
        transcript: _CanonicalTranscript | None = None

        try:
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            max_tokens = (
                self._session_config.realtime_max_tokens
                or self.serving.model_cls.realtime_max_tokens
            )
            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=max_tokens,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )
            tokenizer = self.serving.renderer.get_tokenizer()
            from vllm.model_executor.models.qwen3_asr_realtime import (
                _DEFAULT_ROLLBACK_TOKENS,
            )

            rollback_tokens = (
                self._session_config.rollback_tokens
                if self._session_config.rollback_tokens is not None
                else _DEFAULT_ROLLBACK_TOKENS
            )
            transcript = _CanonicalTranscript(
                tokenizer, rollback_tokens, request_id or "-"
            )

            eos_ids: set[int] = set()
            eos = getattr(self.serving.model_config.hf_config, "eos_token_id", None)
            if isinstance(eos, list):
                eos_ids.update(eos)
            elif eos is not None:
                eos_ids.add(eos)

            async for segment_input in streaming_input_gen:
                segments_started += 1
                segment_id = segments_started
                prefix_text = prefix_texts.popleft() if prefix_texts else ""
                trace = segment_traces.popleft() if segment_traces else None
                segment_request_id = (
                    f"{request_id or 'realtime'}:segment:{segment_id}"
                )
                if trace is not None:
                    logger.info(
                        "QWEN3_ASR_RT_FRESH_SEGMENT_INPUT request_id=%s "
                        "segment_request_id=%s segment_id=%d audio_samples=%d "
                        "audio_duration_s=%.6f prefix_chars=%d prompt_tokens=%d",
                        request_id or "-",
                        segment_request_id,
                        segment_id,
                        trace["audio_samples"],
                        trace["audio_duration_s"],
                        len(prefix_text),
                        len(segment_input.prompt.get("prompt_token_ids", [])),
                    )

                result_gen = self.serving.engine_client.generate(
                    prompt=segment_input.prompt,
                    sampling_params=sampling_params,
                    request_id=segment_request_id,
                )
                segment_phase_start = time.perf_counter()
                segment_first_output = None
                segment_text = ""
                segment_token_ids: list[int] = []
                segment_prompt_tokens = 0
                finish_reason = None
                has_eos = False
                async for output in result_gen:
                    if not output.outputs:
                        continue
                    now = time.perf_counter()
                    if segment_first_output is None:
                        segment_first_output = now
                    item = output.outputs[0]
                    segment_prompt_tokens = max(
                        segment_prompt_tokens,
                        len(output.prompt_token_ids or []),
                    )
                    segment_text += item.text or ""
                    token_ids = list(item.token_ids or [])
                    segment_token_ids.extend(token_ids)
                    completion_tokens_total += len(token_ids)
                    has_eos = has_eos or bool(eos_ids.intersection(token_ids))
                    if item.finish_reason is not None:
                        finish_reason = item.finish_reason

                segment_phase_end = time.perf_counter()
                _record_phase_timing(
                    {
                        "request_id": request_id or "-",
                        "segment_id": segment_id,
                        "mode": "fresh-segment",
                        "ttft_s": (
                            segment_first_output - segment_phase_start
                            if segment_first_output is not None else float("nan")
                        ),
                        "segment_total_s": segment_phase_end - segment_phase_start,
                        "generated_tokens": len(segment_token_ids),
                        "prompt_tokens": segment_prompt_tokens,
                        "decode_tokens": len(segment_token_ids),
                    }
                )

                prompt_tokens_total += segment_prompt_tokens
                if not segment_text and segment_token_ids:
                    segment_text = _decode_text(
                        tokenizer,
                        [token_id for token_id in segment_token_ids if token_id not in eos_ids],
                    )
                stream_mode = trace.get("stream_mode", "cumulative") if trace else "cumulative"
                candidate_token_ids = [
                    token_id
                    for token_id in segment_token_ids
                    if token_id not in eos_ids
                ]
                candidate = (
                    _normalize_realtime_text(segment_text)
                    if stream_mode == "delta_turn"
                    else _decode_candidate_text(
                        tokenizer,
                        prefix_text,
                        candidate_token_ids,
                        segment_text,
                    )
                )
                invalid_reason = _candidate_reject_reason(
                    candidate,
                    candidate_token_ids,
                    finish_reason,
                    has_eos,
                )

                if invalid_reason is not None:
                    transcript.reject(segment_id, invalid_reason)
                    accepted = False
                else:
                    delta = transcript.observe(
                        candidate,
                        delta_turn=(stream_mode == "delta_turn"),
                    )
                    accepted = (
                        stream_mode == "delta_turn"
                        or transcript.latest_valid_text == candidate
                    )
                    if delta:
                        await self.send(TranscriptionDelta(delta=delta))

                if accepted:
                    handshake_tokens = segment_token_ids
                elif stream_mode != "delta_turn":
                    handshake_tokens = transcript.fallback_tail_tokens(prefix_text)
                else:
                    handshake_tokens = []
                if handshake_tokens:
                    input_stream.put_nowait(handshake_tokens)
                input_stream.put_nowait([])
                logger.info(
                    "QWEN3_ASR_RT_FRESH_SEGMENT_OUTPUT request_id=%s "
                    "segment_request_id=%s segment_id=%d prompt_tokens=%d "
                    "generated_tokens=%d finish_reason=%r has_eos=%s "
                    "candidate_chars=%d accepted=%s reject_reason=%r "
                    "candidate=%r latest_valid=%r",
                    request_id or "-",
                    segment_request_id,
                    segment_id,
                    segment_prompt_tokens,
                    len(segment_token_ids),
                    finish_reason,
                    has_eos,
                    len(candidate),
                    accepted,
                    invalid_reason,
                    candidate,
                    transcript.latest_valid_text,
                )

                if not self._is_connected:
                    break

            final_delta, final_text = transcript.finalize()
            if final_delta:
                await self.send(TranscriptionDelta(delta=final_delta))
            logger.info(
                "QWEN3_ASR_RT_FRESH_REQUEST_DONE request_id=%s segments_started=%d "
                "prompt_tokens=%d completion_tokens=%d rejected_candidates=%d "
                "final_text=%r",
                request_id or "-",
                segments_started,
                prompt_tokens_total,
                completion_tokens_total,
                transcript.rejected_candidates,
                final_text,
            )
            usage = UsageInfo(
                prompt_tokens=prompt_tokens_total,
                completion_tokens=completion_tokens_total,
                total_tokens=prompt_tokens_total + completion_tokens_total,
            )
            await self.send(TranscriptionDone(text=final_text, usage=usage))
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
        except Exception as exc:
            logger.exception("Error in fresh-segment generation: %s", exc)
            await self.send_error(str(exc), "processing_error")

    async def _run_generation_legacy(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
        prefix_texts: deque[str],
        request_id: str | None = None,
        segment_traces: deque[dict] | None = None,
    ):
        """Run the generation and stream results back to the client.

        Each segment re-transcribes all accumulated audio.  The prompt
        includes a text *prefix* covering previously-confirmed text so
        the model only generates the *continuation*.  ``prefix_texts``
        carries the actual prefix string for each segment so we can
        reconstruct the full ``raw_decoded = prefix + gen_text`` and
        apply holdback identically to the model's rollback, keeping
        ``confirmed_text`` perfectly aligned.

        To avoid sending uncertain trailing text (e.g. "and." when the
        model doesn't yet have enough audio to know "and badly…"), we
        apply rollback holdback: only the *stable* portion of each
        segment's output is sent as a delta.  The trailing tokens that
        would be rolled back for the next segment are held back until
        either (a) the next segment confirms them, or (b) the stream
        ends and we flush everything.

        At segment end, a sentinel ``[]`` is pushed into ``input_stream``
        so ``buffer_realtime_audio`` can update its prefix for the next
        step.
        """
        confirmed_text = ""
        transcript: _CanonicalTranscript | None = None

        prompt_token_ids_len: int = 0
        completion_tokens_len: int = 0

        try:
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            max_tokens = (self._session_config.realtime_max_tokens
                          or self.serving.model_cls.realtime_max_tokens)
            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=max_tokens,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

            tokenizer = self.serving.renderer.get_tokenizer()

            from vllm.model_executor.models.qwen3_asr_realtime import (
                _DEFAULT_ROLLBACK_TOKENS,
            )
            rollback_tokens = (self._session_config.rollback_tokens
                               if self._session_config.rollback_tokens
                               is not None
                               else _DEFAULT_ROLLBACK_TOKENS)
            transcript = _CanonicalTranscript(
                tokenizer, rollback_tokens, request_id or "-"
            )

            eos_ids: set[int] = set()
            try:
                mc = self.serving.model_config
                eos = getattr(mc.hf_config, "eos_token_id", None)
                if isinstance(eos, list):
                    eos_ids.update(eos)
                elif eos is not None:
                    eos_ids.add(eos)
                if not eos_ids:
                    gen_cfg = mc.try_get_generation_config()
                    if isinstance(gen_cfg, dict):
                        eos = gen_cfg.get("eos_token_id")
                    else:
                        eos = getattr(gen_cfg, "eos_token_id", None)
                    if isinstance(eos, list):
                        eos_ids.update(eos)
                    elif eos is not None:
                        eos_ids.add(eos)
            except Exception as e:
                logger.warning("Failed to get EOS token IDs: %s", e)

            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            segment_gen_text = ""
            segment_gen_token_ids: list[int] = []
            segment_gen_tokens = 0
            prefix_text = ""
            segments_started = 0
            segment_signalled = True
            current_trace: dict | None = None
            request_phase_start = time.perf_counter()
            segment_phase_start = None
            segment_first_output = None

            async for output in result_gen:
                if not output.outputs or len(output.outputs) == 0:
                    continue

                cur_prompt_len = (len(output.prompt_token_ids)
                                  if output.prompt_token_ids else 0)
                # Prompt token count can grow during a single decode, so a
                # length change is not a segment boundary. The next output can
                # start a segment only after the previous one is complete.
                new_segment_starting = cur_prompt_len > 0 and segment_signalled

                if new_segment_starting:
                    prompt_token_ids_len = cur_prompt_len
                    segment_gen_text = ""
                    segment_gen_token_ids = []
                    segment_gen_tokens = 0
                    segment_signalled = False
                    segment_phase_start = time.perf_counter()
                    segment_first_output = None
                    segments_started += 1
                    prefix_text = (prefix_texts.popleft()
                                   if prefix_texts else "")
                    current_trace = (
                        segment_traces.popleft()
                        if segment_traces
                        else None
                    )
                    if current_trace is None:
                        logger.warning(
                            "QWEN3_ASR_RT_SEGMENT_TRACE_MISSING request_id=%s",
                            request_id or "-",
                        )
                    else:
                        logger.info(
                            "QWEN3_ASR_RT_SEGMENT_START request_id=%s "
                            "segment_id=%d engine_prompt_tokens=%d "
                            "audio_samples=%d audio_duration_s=%.6f "
                            "emit_reason=%s",
                            request_id or "-", current_trace["segment_id"],
                            cur_prompt_len, current_trace["audio_samples"],
                            current_trace["audio_duration_s"],
                            current_trace["emit_reason"],
                        )

                raw_delta = output.outputs[0].text
                tok_ids = list(output.outputs[0].token_ids)
                now = time.perf_counter()
                if segment_first_output is None:
                    segment_first_output = now

                completion_tokens_len += len(tok_ids)
                segment_gen_tokens += len(tok_ids)
                segment_gen_token_ids.extend(tok_ids)
                segment_gen_text += raw_delta

                has_eos = bool(eos_ids and eos_ids.intersection(tok_ids))
                if ((output.outputs[0].finish_reason is not None or has_eos)
                        and not segment_signalled):
                    segment_signalled = True

                    finish_reason = output.outputs[0].finish_reason
                    stream_mode = (
                        current_trace.get("stream_mode", "cumulative")
                        if current_trace else "cumulative"
                    )
                    candidate_token_ids = [
                        token_id
                        for token_id in segment_gen_token_ids
                        if token_id not in eos_ids
                    ]
                    candidate = (
                        _normalize_realtime_text(segment_gen_text)
                        if stream_mode == "delta_turn"
                        else _decode_candidate_text(
                            tokenizer,
                            prefix_text,
                            candidate_token_ids,
                            segment_gen_text,
                        )
                    )
                    invalid_reason = _candidate_reject_reason(
                        candidate,
                        candidate_token_ids,
                        finish_reason,
                        has_eos,
                    )
                    if invalid_reason == "max_tokens_without_eos":
                        logger.warning(
                            "QWEN3_ASR_RT_CANDIDATE_REJECT request_id=%s "
                            "segment_id=%s reason=max_tokens_without_eos "
                            "generated_tokens=%d prefix_chars=%d",
                            request_id or "-",
                            current_trace["segment_id"]
                            if current_trace else "-",
                            segment_gen_tokens,
                            len(prefix_text),
                        )
                        # Buffer tokens until validity is known. Release the
                        # model-side handshake without exposing this rejected
                        # candidate to raw_decoded, and stop the audio source
                        # before another resumable KV update can be built.
                        input_stream.put_nowait([])
                        while not self.audio_queue.empty():
                            self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(None)
                        await self.send_error(
                            "Realtime segment reached realtime_max_tokens "
                            "without EOS; the candidate was rejected and the "
                            "session was stopped before KV reuse.",
                            "max_tokens_without_eos",
                        )
                        return

                    if invalid_reason is not None:
                        segment_id = (
                            current_trace["segment_id"]
                            if current_trace else segments_started
                        )
                        transcript.reject(segment_id, invalid_reason)
                        fallback_tokens = (
                            transcript.fallback_tail_tokens(prefix_text)
                            if stream_mode != "delta_turn"
                            else []
                        )
                        if fallback_tokens:
                            input_stream.put_nowait(fallback_tokens)
                        raw_decoded = transcript.latest_valid_text
                        stable = transcript.committed_text
                        confirmed_text = stable
                        delta = ""
                    else:
                        # Only a completed, accepted segment may update the
                        # model-side text prefix for the next audio window.
                        input_stream.put_nowait(segment_gen_token_ids)
                        raw_decoded = candidate
                        if stream_mode == "delta_turn":
                            stable = raw_decoded
                            delta = raw_decoded
                            confirmed_text += delta
                            if delta:
                                await self.send(TranscriptionDelta(delta=delta))
                        else:
                            delta = transcript.observe(raw_decoded)
                            stable = transcript.committed_text
                            confirmed_text = stable
                            if delta:
                                await self.send(TranscriptionDelta(delta=delta))

                    logger.info(
                        "QWEN3_ASR_RT_SEGMENT_OUTPUT request_id=%s segment_id=%s "
                        "generated_tokens=%d finish_reason=%r has_eos=%s "
                        "prefix_text=%r generated_text=%r candidate_text=%r "
                        "accepted=%s reject_reason=%r latest_valid=%r "
                        "stable_text=%r emitted_delta=%r confirmed_text=%r",
                        request_id or "-",
                        current_trace["segment_id"] if current_trace else "-",
                        segment_gen_tokens, finish_reason, has_eos, prefix_text,
                        segment_gen_text, candidate, invalid_reason is None,
                        invalid_reason, transcript.latest_valid_text, stable,
                        delta, confirmed_text,
                    )
                    logger.info(
                        "QWEN3_ASR_RT_PHASE_TIMING request_id=%s segment_id=%s "
                        "ttft_s=%.6f segment_total_s=%.6f generated_tokens=%d "
                        "prefill_proxy_prompt_tokens=%d decode_tokens=%d",
                        request_id or "-",
                        current_trace["segment_id"] if current_trace else "-",
                        (segment_first_output - segment_phase_start)
                        if segment_first_output is not None and segment_phase_start is not None
                        else float("nan"),
                        (now - segment_phase_start)
                        if segment_phase_start is not None else float("nan"),
                        segment_gen_tokens,
                        cur_prompt_len,
                        segment_gen_tokens,
                    )
                    _record_phase_timing(
                        {
                            "request_id": request_id or "-",
                            "segment_id": current_trace["segment_id"]
                            if current_trace else -1,
                            "mode": (
                                "aut-stable-window-kv"
                                if stream_mode == _AUT_STABLE_WINDOW_MODE
                                else "audio-history-kv"
                            ),
                            "ttft_s": (
                                segment_first_output - segment_phase_start
                                if segment_first_output is not None
                                and segment_phase_start is not None
                                else float("nan")
                            ),
                            "segment_total_s": (
                                now - segment_phase_start
                                if segment_phase_start is not None
                                else float("nan")
                            ),
                            "generated_tokens": segment_gen_tokens,
                            "prompt_tokens": cur_prompt_len,
                            "decode_tokens": segment_gen_tokens,
                        }
                    )

                    input_stream.put_nowait([])

                if not self._is_connected:
                    break

            # Observe a partial final segment once, then flush the newest valid
            # candidate. Token-LCP state keeps English separator spaces and
            # Chinese adjacency without language-specific heuristics.
            final_stream_mode = (
                current_trace.get("stream_mode", "cumulative")
                if current_trace else "cumulative"
            )
            if (
                not segment_signalled
                and segment_gen_text
                and final_stream_mode != "delta_turn"
            ):
                candidate_token_ids = [
                    token_id
                    for token_id in segment_gen_token_ids
                    if token_id not in eos_ids
                ]
                candidate = _decode_candidate_text(
                    tokenizer,
                    prefix_text,
                    candidate_token_ids,
                    segment_gen_text,
                )
                invalid_reason = _candidate_reject_reason(
                    candidate,
                    candidate_token_ids,
                    None,
                    False,
                )
                if invalid_reason is not None:
                    segment_id = (
                        current_trace["segment_id"]
                        if current_trace else segments_started
                    )
                    transcript.reject(segment_id, invalid_reason)
                else:
                    raw_decoded = candidate
                    delta = transcript.observe(raw_decoded)
                    if delta:
                        await self.send(TranscriptionDelta(delta=delta))

            if final_stream_mode != "delta_turn":
                flush, confirmed_text = transcript.finalize()
                if flush:
                    await self.send(TranscriptionDelta(delta=flush))

            if not segment_signalled:
                input_stream.put_nowait([])

            logger.info(
                "QWEN3_ASR_RT_REQUEST_DONE request_id=%s segments_started=%d "
                "completion_tokens=%d total_elapsed_s=%.6f final_text=%r",
                request_id or "-", segments_started, completion_tokens_len,
                time.perf_counter() - request_phase_start, confirmed_text,
            )

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )

            await self.send(TranscriptionDone(text=confirmed_text, usage=usage))

            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(sanitize_message(str(e)), "processing_error")

    async def send(
        self,
        event: SessionCreated | SessionUpdated | TranscriptionDelta | TranscriptionDone,
    ):
        """Send event to client."""
        data = event.model_dump_json()
        await self.websocket.send_text(data)

    async def send_error(self, message: str, code: str | None = None):
        """Send error event to client."""
        error_event = ErrorEvent(error=message, code=code)
        await self.websocket.send_text(error_event.model_dump_json())

    async def cleanup(self):
        """Cleanup resources."""
        # Signal audio stream to stop
        self.audio_queue.put_nowait(None)

        # Cancel generation task if running
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()

        logger.debug("Connection cleanup complete: %s", self.connection_id)
