# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One streaming session backed by a long-lived generate() call."""

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import uuid4

from PIL import Image

from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.logger import init_logger
from vllm.renderers.params import ChatParams
from vllm.sampling_params import SamplingParams

from .chunking import build_chunk
from .protocol import FrameResponse

if TYPE_CHECKING:
    from vllm.renderers import BaseRenderer

logger = init_logger(__name__)

_FRAME_TIMEOUT_S = 120.0


class SessionBusyError(RuntimeError):
    """A frame was pushed while the previous reply is still pending."""


class StreamingSession:
    def __init__(
        self,
        session_id: str,
        engine_client: EngineClient,
        sampling_params: SamplingParams,
        renderer: "BaseRenderer",
        *,
        system_prompt: str,
        question: str,
        chat_params: ChatParams,
    ) -> None:
        self.session_id = session_id
        self.request_id = f"stream-{session_id}-{uuid4()}"
        self.engine = engine_client
        self.sp = sampling_params
        self.renderer = renderer
        self.system_prompt = system_prompt
        self.question = question
        self.chat_params = chat_params
        self._in: asyncio.Queue[tuple[Image.Image, float]] = asyncio.Queue(maxsize=1)
        self._out: asyncio.Queue[FrameResponse | Exception | None] = asyncio.Queue()
        self._submit_times: deque[float] = deque()
        self._frames_yielded = 0
        self._replay_token: int | None = None
        self.frames_done = 0
        self._task: asyncio.Task | None = None
        self._closed = False
        self._inflight = False
        self.last_active = time.monotonic()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def inflight(self) -> bool:
        return self._inflight

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._run())
        return self._task

    async def push_frame(self, frame: Image.Image) -> FrameResponse:
        if self._closed:
            raise RuntimeError(f"session {self.session_id} is closed")
        if self._inflight:
            raise SessionBusyError(
                f"session {self.session_id} already has a frame in flight"
            )
        self._inflight = True
        self.last_active = time.monotonic()
        try:
            try:
                async with asyncio.timeout(_FRAME_TIMEOUT_S):
                    await self._in.put((frame, time.monotonic()))
                    item = await self._out.get()
            except TimeoutError as e:
                await self.close()
                raise RuntimeError(
                    f"session {self.session_id} timed out waiting for a reply"
                ) from e
            except asyncio.CancelledError:
                # A disconnected client cannot safely retry the same turn.
                await self.close()
                raise
            if isinstance(item, Exception):
                raise item
            if item is None:
                raise RuntimeError(f"session {self.session_id} ended before a reply")
            self.frames_done += 1
            self.last_active = time.monotonic()
            return item
        finally:
            self._inflight = False

    async def close(self) -> None:
        """Stop the input producer and release the engine request before returning."""
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        with contextlib.suppress(Exception):
            await self.engine.abort(self.request_id)

    async def _producer(self) -> AsyncGenerator[StreamingInput, None]:
        while True:
            frame, submit_t = await self._in.get()
            self._submit_times.append(submit_t)
            is_first = self._frames_yielded == 0
            self._frames_yielded += 1
            yield await build_chunk(
                self.renderer,
                frame,
                is_first=is_first,
                question=self.question,
                system_prompt=self.system_prompt,
                sp=self.sp,
                chat_params=self.chat_params,
                replay_token=self._replay_token,
            )

    async def _run(self) -> None:
        text = ""
        tokens = 0
        last_token: int | None = None
        ttft: float | None = None
        frame_index = 0
        try:
            async for output in self.engine.generate(
                prompt=self._producer(),
                sampling_params=self.sp,
                request_id=self.request_id,
            ):
                for completion in output.outputs:
                    submit_t = self._submit_times[0] if self._submit_times else None
                    tokens += len(completion.token_ids)
                    if completion.token_ids:
                        last_token = completion.token_ids[-1]
                    if completion.text:
                        if ttft is None and submit_t is not None:
                            ttft = time.monotonic() - submit_t
                        text += completion.text
                    if (
                        completion.finish_reason in ("stop", "length")
                        and frame_index < self._frames_yielded
                    ):
                        latency = (
                            time.monotonic() - submit_t
                            if submit_t is not None
                            else None
                        )
                        if self._submit_times:
                            self._submit_times.popleft()
                        # The scheduler drops the final sampled token at a fold.
                        # Keep visible content when a reply hit max_tokens.
                        self._replay_token = (
                            last_token if completion.finish_reason == "length" else None
                        )
                        self._out.put_nowait(
                            FrameResponse(
                                frame_index=frame_index,
                                text=text,
                                finish_reason=completion.finish_reason,
                                token_count=tokens,
                                ttft_s=ttft,
                                latency_s=latency,
                            )
                        )
                        frame_index += 1
                        text, tokens, ttft = "", 0, None
                        last_token = None
                if output.finished:
                    if frame_index < self._frames_yielded:
                        raise RuntimeError(
                            f"session {self.session_id} ended before replying "
                            "to an outstanding frame"
                        )
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("streaming session %s failed", self.session_id)
            self._out.put_nowait(e)
        finally:
            self._closed = True
            self._out.put_nowait(None)
