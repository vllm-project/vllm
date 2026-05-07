# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker→driver plumbing for ``location = "driver"`` capture consumers.

When a ``CaptureConsumer`` declares ``location = "driver"``, its
``on_capture`` callback must run in the driver (main) process rather
than inside the engine-core worker.  This module provides the
cross-process bridge:

- **``_DriverQueueShim``** — a ``CaptureSink``-compatible object that
  lives on the *worker* side.  It serializes chunk/finalize events and
  pushes them onto a ``torch.multiprocessing.Queue``.

- **``_DriverReceiver``** — a daemon thread in the *driver* process
  that pops events from the queue, feeds them through a
  ``_BatchedAdapter``, and pushes ``CaptureResult`` objects back to
  a result queue the shim can read.

- **``install_driver_consumer``** — convenience constructor that wires
  up both sides and returns the shim.

``torch.multiprocessing`` is used instead of ``multiprocessing`` to
ensure CUDA shared-memory tensors transfer correctly.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import ClassVar, Literal

import torch.multiprocessing as mp

from vllm.v1.capture.consumer import CaptureConsumer, _BatchedAdapter
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# _DriverQueueShim — worker side
# ---------------------------------------------------------------------------


class _DriverQueueShim:
    """Worker-side shim that sends capture events to a driver-side
    consumer via a multiprocessing queue.

    Implements the ``CaptureSink`` interface so the capture manager can
    treat it identically to an in-process ``_BatchedAdapter``.
    """

    location: ClassVar[Literal["worker"]] = "worker"

    def __init__(
        self,
        event_queue: mp.Queue,
        result_queue: mp.Queue,
        timeout: float = 30.0,
    ) -> None:
        self._event_queue = event_queue
        self._result_queue = result_queue
        self._timeout = timeout
        self._results: dict[CaptureKey, CaptureResult] = {}
        self._lock = threading.Lock()

    # -- helpers -----------------------------------------------------------

    def _drain_results(self) -> None:
        """Pull all available results from the result queue (non-blocking)."""
        while True:
            try:
                result: CaptureResult = self._result_queue.get_nowait()
            except queue.Empty:
                break
            self._results[result.key] = result

    # -- CaptureSink -------------------------------------------------------

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        try:
            self._event_queue.put(("chunk", chunk), timeout=self._timeout)
        except queue.Full:
            logger.warning(
                "Driver event queue full — marking key %s as partial_error",
                chunk.key,
            )
            with self._lock:
                self._results[chunk.key] = CaptureResult(
                    key=chunk.key,
                    status="partial_error",
                    error="event queue full during chunk submission",
                )

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        try:
            self._event_queue.put(("finalize", finalize), timeout=self._timeout)
        except queue.Full:
            logger.warning(
                "Driver event queue full — marking key %s as partial_error",
                finalize.key,
            )
            with self._lock:
                self._results[finalize.key] = CaptureResult(
                    key=finalize.key,
                    status="partial_error",
                    error="event queue full during finalize submission",
                )
            return

        # After a successful finalize put, drain any results the driver
        # receiver has posted so far.
        with self._lock:
            self._drain_results()

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        with self._lock:
            self._drain_results()
            return self._results.get(key)

    def wait_for_result(
        self,
        key: CaptureKey,
        timeout: float,
    ) -> CaptureResult | None:
        deadline = time.monotonic() + timeout

        with self._lock:
            self._drain_results()
            result = self._results.get(key)
            if result is not None:
                return result

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            try:
                result = self._result_queue.get(timeout=remaining)
            except queue.Empty:
                return None

            with self._lock:
                self._results[result.key] = result
                if result.key == key:
                    return result

    def shutdown(self, timeout: float = 30.0) -> None:
        """Send the sentinel to stop the driver receiver, then drain."""
        try:
            self._event_queue.put(None, timeout=timeout)
        except queue.Full:
            logger.warning("Could not send shutdown sentinel — event queue full")
        # Give the receiver time to flush any last results.
        with self._lock:
            self._drain_results()


# ---------------------------------------------------------------------------
# _DriverReceiver — driver side
# ---------------------------------------------------------------------------


class _DriverReceiver:
    """Driver-side thread that processes events from the worker queue
    and invokes the consumer's ``on_capture`` / ``on_error`` via a
    ``_BatchedAdapter``.

    Runs as a daemon thread so it does not prevent interpreter shutdown.
    """

    def __init__(
        self,
        consumer: CaptureConsumer,
        event_queue: mp.Queue,
        result_queue: mp.Queue,
    ) -> None:
        self._consumer = consumer
        self._event_queue = event_queue
        self._result_queue = result_queue
        self._adapter = _BatchedAdapter(consumer)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="capture-driver-receiver"
        )

    def start(self) -> None:
        self._thread.start()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        while True:
            try:
                event = self._event_queue.get()
            except EOFError:
                # Queue was closed (normal on process shutdown — the
                # daemon thread reads EOF once the owning subprocess
                # tears down the engine).  Exit quietly.
                logger.debug("Driver event queue closed (EOFError); exiting receiver")
                break
            except Exception:
                logger.exception("Unexpected error reading from driver event queue")
                break

            if event is None:
                # Sentinel — shut down cleanly.
                break

            event_type, payload = event

            if event_type == "chunk":
                self._adapter.submit_chunk(payload)
            elif event_type == "finalize":
                self._adapter.submit_finalize(payload)
                result = self._adapter.get_result(payload.key)
                if result is not None:
                    try:
                        self._result_queue.put(result)
                    except Exception:
                        logger.exception("Failed to put result for key %s", payload.key)
            else:
                logger.warning("Unknown event type %r in driver receiver", event_type)

    def join(self, timeout: float = 30.0) -> None:
        self._thread.join(timeout)


# ---------------------------------------------------------------------------
# install_driver_consumer
# ---------------------------------------------------------------------------


def install_driver_consumer(
    consumer: CaptureConsumer,
    *,
    queue_size: int = 1024,
    timeout: float = 30.0,
) -> _DriverQueueShim:
    """Set up the driver bridge for a ``location='driver'`` consumer.

    Creates the ``torch.multiprocessing.Queue`` pair, spawns the driver
    receiver thread, and returns the worker-side shim that the capture
    manager will install as its sink.

    Args:
        consumer: A ``CaptureConsumer`` instance with
            ``location = "driver"``.
        queue_size: Maximum capacity for each direction's queue.
        timeout: Default timeout (seconds) for the shim's put operations.

    Returns:
        A ``_DriverQueueShim`` implementing ``CaptureSink``.
    """
    event_queue: mp.Queue = mp.Queue(maxsize=queue_size)
    result_queue: mp.Queue = mp.Queue(maxsize=queue_size)

    receiver = _DriverReceiver(consumer, event_queue, result_queue)
    receiver.start()

    return _DriverQueueShim(event_queue, result_queue, timeout)
