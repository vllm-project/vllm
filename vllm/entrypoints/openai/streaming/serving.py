# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission and lifecycle of streaming sessions."""

import asyncio
import time
from http import HTTPStatus
from uuid import uuid4

from vllm.config.model import get_served_model_name
from vllm.engine.protocol import EngineClient
from vllm.exceptions import GracefulHTTPError, VLLMError
from vllm.logger import init_logger
from vllm.renderers.params import ChatParams
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
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

logger = init_logger(__name__)

_REAPER_PERIOD_S = 60.0
_IDLE_TIMEOUT_S = 900.0


class OpenAIServingStreaming:
    """Owns at most one session in this API server process."""

    def __init__(
        self,
        engine_client: EngineClient,
        *,
        chat_params: ChatParams,
    ) -> None:
        self.engine = engine_client
        self.model_config = engine_client.model_config
        self.renderer = engine_client.renderer
        self.chat_params = chat_params
        self.sessions: dict[str, StreamingSession] = {}
        self._reaper_task: asyncio.Task | None = None

    @property
    def model_name(self) -> str:
        return get_served_model_name(
            self.model_config.model, self.model_config.served_model_name
        )

    def config(self) -> ConfigResponse:
        return ConfigResponse(
            model=self.model_name,
            retention=StreamingRetentionParams(),
            sampling=SamplingConfig(),
        )

    @staticmethod
    def _build_sampling_params(
        retention: StreamingRetentionParams,
        sampling: SamplingConfig,
    ) -> SamplingParams:
        return SamplingParams(
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            repetition_penalty=sampling.repetition_penalty,
            frequency_penalty=sampling.frequency_penalty,
            output_kind=RequestOutputKind.DELTA,
            structured_outputs=sampling.structured_outputs,
            extra_args={"streaming_retention": retention},
        )

    def _check_capacity(self, req: SessionRequest) -> None:
        if self.sessions:
            raise GracefulHTTPError(
                "a streaming session is already active; close it before creating "
                "another session",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        aliases = self.model_config.served_model_name or self.model_config.model
        if isinstance(aliases, str):
            aliases = [aliases]
        if req.model is not None and req.model not in aliases:
            raise GracefulHTTPError(f"unknown model {req.model}", HTTPStatus.NOT_FOUND)

        processor = self.renderer.mm_processor
        if processor is None or processor.info.allowed_mm_limits.get("image", 0) < 1:
            raise GracefulHTTPError(
                "the streaming frame API requires image inputs", HTTPStatus.BAD_REQUEST
            )

        retention = req.retention
        max_session_tokens = retention.max_session_tokens
        max_model_len = self.model_config.max_model_len
        if max_session_tokens + req.sampling.max_tokens > max_model_len:
            raise GracefulHTTPError(
                "retention.max_session_tokens + sampling.max_tokens must not "
                f"exceed max_model_len={max_model_len}",
                HTTPStatus.BAD_REQUEST,
            )
        model_max_position = getattr(
            self.model_config.hf_text_config, "max_position_embeddings", max_model_len
        )
        threshold = retention.reprefill_threshold * model_max_position
        if max_session_tokens + req.sampling.max_tokens >= threshold:
            raise GracefulHTTPError(
                "retention.max_session_tokens + sampling.max_tokens must be below "
                f"reprefill_threshold * model_max_position={threshold:.0f}",
                HTTPStatus.BAD_REQUEST,
            )

    async def create_session(self, req: SessionRequest) -> SessionResponse:
        await self._reap_idle()
        self._check_capacity(req)
        sp = self._build_sampling_params(req.retention, req.sampling)
        session_id = f"sess-{uuid4()}"
        session = StreamingSession(
            session_id,
            self.engine,
            sp,
            self.renderer,
            system_prompt=req.system_prompt,
            question=req.question,
            chat_params=self.chat_params,
        )
        task = session.start()
        self.sessions[session_id] = session
        task.add_done_callback(lambda _: self.sessions.pop(session_id, None))
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reaper_loop())
        return SessionResponse(
            session_id=session_id,
            model=self.model_name,
            retention=req.retention,
        )

    async def push_frame(self, session_id: str, frame_bytes: bytes) -> FrameResponse:
        session = self.sessions.get(session_id)
        if session is None:
            raise GracefulHTTPError(
                f"unknown session {session_id}", HTTPStatus.NOT_FOUND
            )
        if not frame_bytes:
            raise GracefulHTTPError("empty frame body", HTTPStatus.BAD_REQUEST)
        try:
            frame = await asyncio.to_thread(decode_frame, frame_bytes)
        except asyncio.CancelledError:
            await session.close()
            self.sessions.pop(session_id, None)
            raise
        except Exception as e:
            raise GracefulHTTPError(
                f"could not decode frame image: {e}", HTTPStatus.BAD_REQUEST
            ) from e
        try:
            return await session.push_frame(frame)
        except SessionBusyError as e:
            raise GracefulHTTPError(str(e), HTTPStatus.CONFLICT) from e
        except VLLMError:
            raise
        except Exception as e:
            dead = session.closed
            if dead:
                self.sessions.pop(session_id, None)
            raise GracefulHTTPError(
                str(e),
                HTTPStatus.CONFLICT if dead else HTTPStatus.INTERNAL_SERVER_ERROR,
            ) from e
        finally:
            frame.close()

    async def close_session(self, session_id: str) -> CloseResponse:
        session = self.sessions.pop(session_id, None)
        if session is None:
            raise GracefulHTTPError(
                f"unknown session {session_id}", HTTPStatus.NOT_FOUND
            )
        await session.close()
        return CloseResponse(session_id=session_id, frames=session.frames_done)

    async def shutdown(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            await asyncio.gather(self._reaper_task, return_exceptions=True)
            self._reaper_task = None
        sessions = list(self.sessions.values())
        self.sessions.clear()
        await asyncio.gather(*(session.close() for session in sessions))

    async def _reap_idle(self) -> None:
        now = time.monotonic()
        stale = [
            sid
            for sid, session in self.sessions.items()
            if session.closed
            or (not session.inflight and now - session.last_active > _IDLE_TIMEOUT_S)
        ]
        for sid in stale:
            session = self.sessions.pop(sid, None)
            if session is not None:
                await session.close()

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(_REAPER_PERIOD_S)
            try:
                await self._reap_idle()
            except Exception:
                logger.exception("streaming session reaper failed")
