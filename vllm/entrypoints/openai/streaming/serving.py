# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving layer for the REST streaming API: session registry + config."""

import asyncio
import json
import time
from typing import TYPE_CHECKING
from uuid import uuid4

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.logger import init_logger
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.streaming.retention import StreamingRetentionParams

from .chunking import decode_frame
from .protocol import (
    CloseResponse,
    ConfigResponse,
    FrameResponse,
    SamplingConfig,
    SessionRequest,
    SessionResponse,
)
from .session import SessionBusyError, StreamingSession

if TYPE_CHECKING:
    from vllm.renderers import BaseRenderer

logger = init_logger(__name__)

# How often the background reaper scans for dead/idle sessions; small next to
# _IDLE_TIMEOUT_S so an abandoned session's KV is reclaimed promptly.
_REAPER_PERIOD_S = 60.0

# A session with no push_frame for this long is presumed abandoned and closed
# by the reaper, releasing its KV and encoder-cache state.
_IDLE_TIMEOUT_S = 900.0


class StreamingError(Exception):
    """Client-facing error; the router maps it to ``status_code``."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _build_structured_outputs(s: SamplingConfig) -> StructuredOutputsParams | None:
    """Build guided-decoding params from at most one guided_* field."""
    set_flags = [
        f for f in ("guided_json", "guided_choice", "guided_regex") if getattr(s, f)
    ]
    if not set_flags:
        return None
    if len(set_flags) > 1:
        raise ValueError(
            f"guided_json / guided_choice / guided_regex are mutually exclusive; "
            f"got {set_flags}"
        )
    if s.guided_json is not None:
        if isinstance(s.guided_json, str):
            schema = json.loads(s.guided_json)
        else:
            schema = s.guided_json
        return StructuredOutputsParams(json=schema)
    if s.guided_choice is not None:
        return StructuredOutputsParams(choice=list(s.guided_choice))
    return StructuredOutputsParams(regex=s.guided_regex)


class OpenAIServingStreaming:
    """Owns the shared engine + a registry of live caption sessions.

    The ``retention`` and ``sampling`` constructor args are only the DEFAULTS
    exposed by GET /config; each session's actual config is client-supplied in
    ``create_session`` (see ``SessionRequest``).
    """

    def __init__(
        self,
        engine_client: EngineClient,
        *,
        retention: StreamingRetentionParams | None = None,
        sampling: SamplingConfig | None = None,
        request_logger: RequestLogger | None = None,
        idle_timeout_s: float = _IDLE_TIMEOUT_S,
    ) -> None:
        self.engine = engine_client
        self.model_config = engine_client.model_config
        self.renderer: BaseRenderer | None = getattr(engine_client, "renderer", None)
        self.retention_config = retention or StreamingRetentionParams()
        self.sampling_config = sampling or SamplingConfig()
        self.request_logger = request_logger
        self.idle_timeout_s = idle_timeout_s
        self.sessions: dict[str, StreamingSession] = {}
        # Started lazily on the first create_session (no running loop is
        # guaranteed at construction time).
        self._reaper_task: asyncio.Task | None = None
        _build_structured_outputs(self.sampling_config)
        logger.info(
            "streaming serving initialized (default retention=%s; concurrent "
            "sessions capped at 1)",
            self.retention_config,
        )

    @property
    def model_name(self) -> str:
        """The served model's name (echoed in session/config responses)."""
        return self.model_config.model

    def config(self) -> ConfigResponse:
        """Return the default retention + sampling config (the descriptor)."""
        return ConfigResponse(
            model=self.model_name,
            retention=self.retention_config,
            sampling=self.sampling_config,
        )

    def _build_sampling_params(
        self,
        retention: StreamingRetentionParams,
        sampling: SamplingConfig,
    ) -> SamplingParams:
        """Build per-session SamplingParams carrying the retention policy."""
        return SamplingParams(
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            repetition_penalty=sampling.repetition_penalty,
            frequency_penalty=sampling.frequency_penalty,
            output_kind=RequestOutputKind.DELTA,
            structured_outputs=_build_structured_outputs(sampling),
            extra_args={"streaming_retention": retention},
        )

    def _engine_max_model_len(self) -> int | None:
        """The engine's configured context window, or None if unreachable."""
        try:
            return int(self.model_config.max_model_len)
        except Exception:
            logger.debug("max_model_len unavailable on model_config", exc_info=True)
            return None

    def _model_max_position(self) -> int | None:
        """The model's trained position range (what the re-prefill trigger
        compares against), falling back to max_model_len."""
        try:
            mp = getattr(
                self.model_config.hf_text_config, "max_position_embeddings", None
            )
            if mp:
                return int(mp)
        except Exception:
            logger.debug(
                "max_position_embeddings unavailable; falling back to max_model_len",
                exc_info=True,
            )
        return self._engine_max_model_len()

    def _check_capacity(
        self,
        retention: StreamingRetentionParams,
        sampling: SamplingConfig,
    ) -> None:
        """Admit a new session or raise ``StreamingError``.

        Retention-intrinsic bounds are enforced by
        ``StreamingRetentionParams.__post_init__``; this adds the
        engine-relative checks plus the single-session cap.
        """
        # One session at a time; multi-session sizing is future work.
        if self.sessions:
            raise StreamingError(
                "a streaming session is already active; this server hosts one "
                "session at a time. Close it (DELETE "
                "/v1/streaming/sessions/{id}) and retry.",
                status_code=503,
            )

        if retention.reprefill_threshold >= 1.0:
            raise StreamingError(
                f"retention.reprefill_threshold={retention.reprefill_threshold} "
                "would disable re-prefill; it must be < 1.0."
            )

        # Non-None is enforced by StreamingRetentionParams.__post_init__.
        max_session_tokens = retention.max_session_tokens
        assert max_session_tokens is not None

        # Retained budget + reply must fit under the context window.
        max_model_len = self._engine_max_model_len()
        if max_model_len is not None:
            reply = sampling.max_tokens
            if max_session_tokens + reply >= max_model_len:
                raise StreamingError(
                    f"retention.max_session_tokens="
                    f"{max_session_tokens} + sampling.max_tokens="
                    f"{reply} must stay below max_model_len={max_model_len} "
                    "with headroom for the next chunk's prefill. Lower "
                    "max_session_tokens (e.g. max_model_len - 1024)."
                )

        # Retained tokens must stay below the re-prefill position threshold,
        # else a re-prefill would re-trigger immediately and loop. The trigger
        # uses the trained position range, not max_model_len.
        model_max_position = self._model_max_position()
        if model_max_position is not None:
            pos_threshold = retention.reprefill_threshold * model_max_position
            if max_session_tokens >= pos_threshold:
                raise StreamingError(
                    f"retention.max_session_tokens="
                    f"{max_session_tokens} must be below "
                    f"reprefill_threshold*model_max_position={pos_threshold:.0f} "
                    f"(trained range {model_max_position}); otherwise a "
                    "re-prefill would re-trigger immediately."
                )

        # The engine's image budget must cover ~2x the retention window: each
        # retained frame is one image pinned in the encoder cache, which is
        # sized from limit_mm_per_prompt.
        try:
            mm_config = self.model_config.multimodal_config
            limit = (
                mm_config.get_limit_per_prompt("image")
                if mm_config is not None
                else None
            )
        except Exception:
            logger.debug(
                "image limit_mm_per_prompt unavailable; skipping the "
                "encoder-budget admission check",
                exc_info=True,
            )
            limit = None
        if limit:
            need = retention.max_video_segments * 2
            if need > limit:
                raise StreamingError(
                    f"retention.max_video_segments={retention.max_video_segments} "
                    f"needs limit_mm_per_prompt['image'] >= {need}, but the "
                    f"engine was launched with {limit}. Re-launch with a larger "
                    "image budget."
                )

    async def create_session(self, req: SessionRequest) -> SessionResponse:
        """Admit and start a session; returns its id + echoed config."""
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reaper_loop())
        # Awaited so a reaped session's engine state is fully released before
        # capacity is checked (no old/new KV + encoder-cache overlap).
        await self._reap_idle()
        self._check_capacity(req.retention, req.sampling)
        sp = self._build_sampling_params(req.retention, req.sampling)
        session_id = f"sess-{uuid4()}"
        # `EngineClient.renderer` is part of the protocol; the getattr in
        # __init__ only tolerates minimal stand-ins that never reach here.
        assert self.renderer is not None
        session = StreamingSession(
            session_id,
            self.engine,
            sp,
            self.renderer,
            system_prompt=req.system_prompt,
            question=req.question,
            fps=req.fps,
        )
        session.start()
        self.sessions[session_id] = session
        # Deregister when the driver task ends (engine failure, per-frame
        # timeout abort, or normal finish) so a dead session cannot hold the
        # single-session cap until the idle reaper fires.
        assert session._task is not None

        def _deregister(_task: asyncio.Task, sid: str = session_id) -> None:
            self.sessions.pop(sid, None)

        session._task.add_done_callback(_deregister)
        logger.info(
            "streaming session %s created (model=%s)", session_id, self.model_name
        )
        return SessionResponse(
            session_id=session_id,
            model=self.model_name,
            fps=req.fps,
            retention=req.retention,
        )

    async def push_frame(self, session_id: str, frame_bytes: bytes) -> FrameResponse:
        """Decode one frame's bytes and return the session's caption for it."""
        session = self.sessions.get(session_id)
        if session is None:
            raise StreamingError(f"unknown session {session_id}", status_code=404)
        if not frame_bytes:
            raise StreamingError("empty frame body")
        try:
            # Full JPEG/PNG decode is synchronous CPU work; keep it off the
            # event loop shared with the engine's output handler.
            frame = await asyncio.to_thread(decode_frame, frame_bytes)
        except Exception as e:
            raise StreamingError(f"could not decode frame image: {e}") from e
        try:
            return await session.push_frame(frame)
        except StreamingError:
            raise
        except SessionBusyError as e:
            raise StreamingError(str(e), status_code=409) from e
        except Exception as e:
            # Deregister sessions whose driver task has ended so they cannot
            # jam admission; 409 tells the client to recreate the session.
            dead = session._closed or (
                session._task is not None and session._task.done()
            )
            if dead:
                self.sessions.pop(session_id, None)
            raise StreamingError(str(e), status_code=409 if dead else 500) from e

    async def close_session(self, session_id: str) -> CloseResponse:
        """Close a session and free its engine-side state."""
        session = self.sessions.pop(session_id, None)
        if session is None:
            raise StreamingError(f"unknown session {session_id}", status_code=404)
        await session.close()
        logger.info(
            "streaming session %s closed (%d frames)",
            session_id,
            session.frames_done,
        )
        return CloseResponse(session_id=session_id, frames=session.frames_done)

    async def _reap_idle(self) -> None:
        """Close and drop dead sessions and sessions idle past the timeout so
        abandoned ones release KV. Awaited (not fire-and-forget) so callers
        observe the engine state fully released."""
        now = time.monotonic()
        stale = [
            sid
            for sid, s in self.sessions.items()
            if s._closed
            or (s._task is not None and s._task.done())
            or now - s.last_active > self.idle_timeout_s
        ]
        for sid in stale:
            session = self.sessions.pop(sid, None)
            if session is not None:
                logger.warning("reaping streaming session %s", sid)
                await session.close()

    async def _reaper_loop(self) -> None:
        """Reap periodically so an abandoned session releases its KV and
        encoder-cache state even if create_session is never called again."""
        while True:
            await asyncio.sleep(_REAPER_PERIOD_S)
            try:
                await self._reap_idle()
            except Exception:
                logger.exception("streaming session reaper failed")
