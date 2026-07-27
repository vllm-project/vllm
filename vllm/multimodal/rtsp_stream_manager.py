# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent DeepStream RTSP pipelines, one daemon thread per stream.

Each ``(uri, chunk_duration, num_frames)`` maps to one thread running
``stream_uri``, so consumers of the same stream share an NVDEC pipeline.
Segments cross the thread boundary as numpy arrays via a sync queue; an
asyncio bridge fans them out to bounded per-consumer queues, dropping the
oldest segment when a consumer falls behind.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import uuid
from contextlib import suppress
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_SENTINEL = None

# Shared by the serving layer, the manager and the decode generator.
DEFAULT_CHUNK_DURATION = 10.0
DEFAULT_NUM_FRAMES = 8

# Log every Nth drop so a slow consumer is visible without flooding the log.
_DROP_LOG_INTERVAL = 10

# Seconds to wait for the decode thread to notice ``stop_event`` and exit.
_THREAD_JOIN_TIMEOUT = 5.0


@dataclass
class _StreamState:
    thread: threading.Thread
    sync_queue: queue.Queue  # type: ignore[type-arg]
    stop_event: threading.Event
    consumers: dict[str, asyncio.Queue]
    bridge_task: asyncio.Task[None] | None = None
    dropped_segments: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)

    def add_consumer(self) -> tuple[str, asyncio.Queue]:
        cid = uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.consumers[cid] = q
        return cid, q

    def remove_consumer(self, cid: str) -> int:
        self.consumers.pop(cid, None)
        return len(self.consumers)


def _worker(
    uri: str,
    num_frames: int,
    chunk_duration: float,
    sync_q: queue.Queue,  # type: ignore[type-arg]
    stop_event: threading.Event,
) -> None:
    logger.info(
        "[stream worker] started for %s  num_frames=%d buffer_sec=%.1f",
        uri,
        num_frames,
        chunk_duration,
    )
    try:
        from vllm.multimodal.video import DeepStreamVideoBackendMixin

        stream_gen = DeepStreamVideoBackendMixin.stream_uri(
            uri,
            num_frames=num_frames,
            chunk_duration=chunk_duration,
        )

        try:
            for frames, metadata in stream_gen:
                if stop_event.is_set():
                    break
                try:
                    sync_q.put((frames, metadata), timeout=5)
                except queue.Full:
                    logger.warning("[stream worker] sync queue full, dropping segment")
        finally:
            # Release the GStreamer pipeline now, not at GC time.
            stream_gen.close()
    except Exception:
        logger.exception("[stream worker] error for %s", uri)
    finally:
        sync_q.put(_SENTINEL)
    logger.info("[stream worker] exiting for %s", uri)


class RTSPStreamManager:
    _instance: RTSPStreamManager | None = None

    def __new__(cls) -> RTSPStreamManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._streams = {}
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    _streams: dict[tuple[str, float, int], _StreamState]
    _lock: asyncio.Lock

    async def subscribe(
        self,
        uri: str,
        chunk_duration: float = DEFAULT_CHUNK_DURATION,
        num_frames: int = DEFAULT_NUM_FRAMES,
    ) -> tuple[str, asyncio.Queue]:
        key = (uri, chunk_duration, num_frames)

        async with self._lock:
            state = self._streams.get(key)
            if state is not None and state._closed:
                # Teardown began: this stream will never yield again.
                self._streams.pop(key, None)
                state = None
            if state is None:
                state = self._spawn(uri, num_frames, chunk_duration)
                self._streams[key] = state
                loop = asyncio.get_running_loop()
                state.bridge_task = loop.create_task(
                    self._bridge(key, state),
                    name=f"rtsp-bridge-{uri}",
                )
            cid, q = state.add_consumer()

        logger.info(
            "RTSP subscribe: consumer=%s uri=%s consumers=%d",
            cid,
            uri,
            len(state.consumers),
        )
        return cid, q

    async def unsubscribe(
        self,
        uri: str,
        chunk_duration: float,
        num_frames: int,
        consumer_id: str,
    ) -> None:
        key = (uri, chunk_duration, num_frames)
        async with self._lock:
            state = self._streams.get(key)
            if state is None:
                return
            remaining = state.remove_consumer(consumer_id)
            logger.info(
                "RTSP unsubscribe: consumer=%s remaining=%d",
                consumer_id,
                remaining,
            )
            if remaining == 0:
                await self._teardown(key, state)

    def _spawn(
        self,
        uri: str,
        num_frames: int,
        chunk_duration: float,
    ) -> _StreamState:
        stop_ev = threading.Event()
        sync_q: queue.Queue = queue.Queue(maxsize=8)
        t = threading.Thread(
            target=_worker,
            args=(uri, num_frames, chunk_duration, sync_q, stop_ev),
            daemon=True,
            name=f"rtsp-{uri[:40]}",
        )
        t.start()
        logger.info("Spawned RTSP thread for %s", uri)
        return _StreamState(
            thread=t,
            sync_queue=sync_q,
            stop_event=stop_ev,
            consumers={},
        )

    async def _bridge(
        self,
        key: tuple[str, float, int],
        state: _StreamState,
    ) -> None:
        loop = asyncio.get_running_loop()
        uri = key[0]
        seg = 0
        logger.info("[RTSP bridge] started for %s", uri)

        cumulative_frames: int = 0
        while True:
            item = await loop.run_in_executor(None, state.sync_queue.get)
            if item is _SENTINEL:
                logger.info("[RTSP bridge] sentinel received for %s", uri)
                for q in list(state.consumers.values()):
                    await q.put(None)
                break

            frames, raw_metadata = item
            fps = raw_metadata.get("fps", 0.0)
            n_frames = raw_metadata.get("total_num_frames", 0)
            pts_start = cumulative_frames / fps if fps > 0 else 0.0
            pts_end = (cumulative_frames + n_frames) / fps if fps > 0 else 0.0
            metadata = {
                **raw_metadata,
                "segment_index": seg,
                "pts_start": pts_start,
                "pts_end": pts_end,
                "duration": pts_end - pts_start,
            }
            cumulative_frames += n_frames

            for q in list(state.consumers.values()):
                try:
                    q.put_nowait((frames, metadata))
                except asyncio.QueueFull:
                    # Drop the oldest rather than stall the pipeline.
                    with suppress(asyncio.QueueEmpty):
                        q.get_nowait()
                    with suppress(asyncio.QueueFull):
                        q.put_nowait((frames, metadata))
                    state.dropped_segments += 1
                    if state.dropped_segments % _DROP_LOG_INTERVAL == 1:
                        logger.warning(
                            "[RTSP bridge] consumer behind on %s, dropped %d "
                            "segment(s) so far",
                            uri,
                            state.dropped_segments,
                        )
            seg += 1

        logger.info("[RTSP bridge] exiting for %s", uri)
        async with self._lock:
            if key in self._streams:
                await self._teardown(key, state)

    async def _teardown(self, key: tuple, state: _StreamState) -> None:
        if state._closed:
            return
        state._closed = True
        state.stop_event.set()

        # Cancelling ourselves would raise CancelledError at the join below.
        bridge_task = state.bridge_task
        state.bridge_task = None
        if (
            bridge_task is not None
            and bridge_task is not asyncio.current_task()
            and not bridge_task.done()
        ):
            bridge_task.cancel()

        # Drop the entry before any await: teardown runs from a finally block
        # in an already-cancelling task, so the await below may never return,
        # and a surviving entry would strand the next subscriber on a dead
        # stream.
        self._streams.pop(key, None)
        if state.dropped_segments:
            logger.info(
                "Torn down RTSP stream for %s (%d segment(s) dropped)",
                key[0],
                state.dropped_segments,
            )
        else:
            logger.info("Torn down RTSP stream for %s", key[0])

        # The worker only checks stop_event between segments, so this join can
        # take a full decode timeout. Off the loop, and best-effort only.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, state.thread.join, _THREAD_JOIN_TIMEOUT)
        if state.thread.is_alive():
            logger.warning(
                "RTSP worker thread still alive after %.0fs for %s",
                _THREAD_JOIN_TIMEOUT,
                key[0],
            )

    async def shutdown_all(self) -> None:
        async with self._lock:
            for key, state in list(self._streams.items()):
                await self._teardown(key, state)
