# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A single streaming session: one long-lived ``engine.generate()`` call."""

import asyncio
import contextlib
import time
from collections import deque
from typing import TYPE_CHECKING
from uuid import uuid4

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

from .chunking import build_chunk
from .protocol import FrameResponse

if TYPE_CHECKING:
    from vllm.renderers import BaseRenderer

logger = init_logger(__name__)

_DONE = ("stop", "length")

# push_frame enforces one frame in flight per session; a depth-1 queue means a
# bug elsewhere can never pipeline frames into the engine.
_MAX_INPUT_BACKLOG = 1

# Per-frame deadline for enqueueing a frame / receiving its reply; generous
# enough for a cold first-frame prefill or a streaming re-prefill, but bounded
# so a wedged engine fails the session instead of hanging the client forever.
_FRAME_TIMEOUT_S = 120.0


class SessionBusyError(RuntimeError):
    """A frame was pushed while the previous frame's reply is still pending."""


class StreamingSession:
    """One streaming session backed by a single long-lived generate() call."""

    def __init__(
        self,
        session_id: str,
        engine_client: EngineClient,
        sampling_params: SamplingParams,
        renderer: "BaseRenderer",
        *,
        system_prompt: str,
        question: str,
        fps: float,
    ) -> None:
        self.session_id = session_id
        self.request_id = f"stream-{session_id}-{uuid4()}"
        self.engine = engine_client
        self.sp = sampling_params
        self.renderer = renderer
        self.system_prompt = system_prompt
        self.question = question
        self.fps = fps
        self._in: asyncio.Queue = asyncio.Queue(maxsize=_MAX_INPUT_BACKLOG)
        self._out: asyncio.Queue = asyncio.Queue()
        self._submit_times: deque[float] = deque()
        self._frames_yielded = 0
        self.frames_done = 0
        self._task: asyncio.Task | None = None
        self._closed = False
        self._inflight = False
        self.last_active = time.monotonic()

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Launch the background engine-driving task."""
        self._task = asyncio.create_task(self._run())

    async def push_frame(self, frame) -> FrameResponse:
        """Enqueue one decoded ``(1,H,W,3)`` RGB frame; await and return its reply."""
        if self._closed:
            raise RuntimeError(f"session {self.session_id} is closed")
        if self._inflight:
            # Reject rather than queue: a second in-flight frame would desync
            # the reply<->frame pairing (e.g. after a client-side retry).
            raise SessionBusyError(
                f"session {self.session_id} already has a frame in flight"
            )
        self._inflight = True
        try:
            self.last_active = time.monotonic()

            try:
                await asyncio.wait_for(
                    self._in.put((frame, time.monotonic())),
                    timeout=_FRAME_TIMEOUT_S,
                )
            except asyncio.TimeoutError as e:
                await self._fail_closed()
                raise RuntimeError(
                    f"session {self.session_id} timed out after "
                    f"{_FRAME_TIMEOUT_S:.0f}s waiting to enqueue a frame"
                ) from e
            try:
                item = await asyncio.wait_for(self._out.get(), timeout=_FRAME_TIMEOUT_S)
            except asyncio.TimeoutError as e:
                await self._fail_closed()
                raise RuntimeError(
                    f"session {self.session_id} timed out after "
                    f"{_FRAME_TIMEOUT_S:.0f}s waiting for a reply"
                ) from e
            if isinstance(item, BaseException):
                raise item
            if item is None:
                raise RuntimeError(f"session {self.session_id} ended before a reply")
            self.frames_done += 1
            self.last_active = time.monotonic()
            return item
        finally:
            self._inflight = False

    async def _fail_closed(self) -> None:
        """Permanently close the session after a per-frame timeout."""
        self._closed = True
        # Stop the generate() driver first: its cancel handler cancels AND
        # awaits the engine-side input-stream task before aborting, so no
        # chunk ADD can be sent after the ABORT and resurrect the session.
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        with contextlib.suppress(Exception):
            await self.engine.abort(self.request_id)  # idempotent backstop
        while True:
            try:
                self._out.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close(self) -> None:
        """Close the session: end the generate() call and free engine state."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(asyncio.QueueFull):
            # producer returns -> generate() finishes; on QueueFull the
            # engine.abort backstop below still ends the request.
            self._in.put_nowait(None)
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10.0)
            except (asyncio.CancelledError, Exception):
                # CancelledError is a BaseException, so it needs its own entry.
                self._task.cancel()
        with contextlib.suppress(Exception):
            await self.engine.abort(self.request_id)  # backstop

    # -- engine driving --------------------------------------------------------

    async def _producer(self):
        """Yield one prompt chunk per pushed frame until the end sentinel."""
        while True:
            item = await self._in.get()
            if item is None:
                return
            frame, submit_t = item
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
            )

    async def _run(self) -> None:
        """Drive generate() and segment the DELTA stream into one reply per frame."""
        buf = ""
        tok = 0
        ttft: float | None = None
        frame_idx = 0
        try:
            async for output in self.engine.generate(
                prompt=self._producer(),
                sampling_params=self.sp,
                request_id=self.request_id,
            ):
                for completion in output.outputs:
                    submit_t = self._submit_times[0] if self._submit_times else None
                    if completion.token_ids:
                        tok += len(completion.token_ids)
                    if completion.text:
                        if ttft is None and submit_t is not None:
                            ttft = time.monotonic() - submit_t
                        buf += completion.text
                    fr = completion.finish_reason
                    # Emit one reply per outstanding frame: frame_idx counts
                    # emitted replies, _frames_yielded counts submitted frames.
                    # This also emits for single-output turns (e.g.
                    # max_tokens=1 or a first-token EOS) while dropping a
                    # spurious duplicate finish with no frame outstanding.
                    if fr in _DONE and frame_idx < self._frames_yielded:
                        latency = (
                            time.monotonic() - submit_t
                            if submit_t is not None
                            else None
                        )
                        if self._submit_times:
                            self._submit_times.popleft()
                        self._out.put_nowait(
                            FrameResponse(
                                frame_index=frame_idx,
                                text=buf.strip(),
                                finish_reason=fr,
                                token_count=tok,
                                ttft_s=ttft,
                                latency_s=latency,
                            )
                        )
                        frame_idx += 1
                        buf, tok, ttft = "", 0, None
                if output.finished:
                    if frame_idx < self._frames_yielded:
                        # Engine-initiated finish (abort/error/shutdown) with
                        # a frame outstanding is session-fatal; unblock the
                        # waiting push_frame instead of letting it time out.
                        self._out.put_nowait(
                            RuntimeError(
                                f"session {self.session_id} ended by engine "
                                "before replying to an outstanding frame"
                            )
                        )
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surface to the awaiting push_frame
            logger.exception("streaming session %s failed", self.session_id)
            self._out.put_nowait(e)
        finally:
            # Mark closed so retry pushes fail fast instead of enqueueing into
            # a queue with no consumer. Safe w.r.t. close()'s early return on
            # _closed (which skips its abort backstop): every path reaching
            # here has already ended the engine request.
            self._closed = True
            self._out.put_nowait(None)  # end sentinel for any waiter
