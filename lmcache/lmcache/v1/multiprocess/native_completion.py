# SPDX-License-Identifier: Apache-2.0

"""Dispatch device-stream-ordered host callbacks without GIL contention.

Call graph::

    Server.store / Server.retrieve  (Python, holds GIL)
        |
        v
    submit_callback_to_stream(stream, kind, payload)
        |
        v  Python: msgspec.msgpack.encode(payload)
        v  C++:    record_completion_on_stream (gil_scoped_release)
        v  C++:    cudaLaunchHostFunc / hipLaunchHostFunc
        |
        v  -------- on the CUDA/HIP driver thread (no GIL) --------
        v  completion_host_callback: append PendingCompletion to buffer
        |
        v  -------- on DeviceHostFuncDispatcher drain thread (Python) --
        v  drain_recorded_completions  (returns list of (kind, bytes))
        v  msgspec.msgpack.decode(bytes) with the kind's registered type
        v  handler(decoded_payload)  ← finish_write / finish_read_prefetched

Payload round-trips as a single opaque blob. For multi-arg handlers, wrap
the args in a tuple/struct matching the registered ``payload_type``.

Extends PR #2952's AB-BA fix (GIL x driver lock) to Server.store/retrieve.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable
import threading

# Third Party
import msgspec
import torch  # noqa: F401 — must be imported before lmcache.c_ops

# First Party
from lmcache.logging import init_logger
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
)
import lmcache.c_ops as _lmc_ops

logger = init_logger(__name__)

# Runs on dispatcher thread with the decoded payload for one submission.
DeviceHostFunc = Callable[[Any], None]


class _Registration(msgspec.Struct):
    handler: DeviceHostFunc
    decoder: msgspec.msgpack.Decoder


class DeviceHostFuncDispatcher(PeriodicThread):
    """Drain buffered C++ completions and dispatch each payload to the handler
    registered for its ``kind``. One instance per process; owned by
    ``MPCacheServer``."""

    def __init__(self, drain_interval_seconds: float = 0.005) -> None:
        super().__init__(
            name="DeviceHostFuncDispatcher",
            interval=drain_interval_seconds,
            level=ThreadLevel.HIGH,
            init_wait=0.0,
        )
        self._registry: dict[str, _Registration] = {}
        self._registry_lock = threading.Lock()
        self._dispatched_count = 0
        self._exception_counts: dict[str, int] = {}
        PeriodicThreadRegistry.get_instance().register(self)

    def register(self, kind: str, handler: DeviceHostFunc, payload_type: Any) -> None:
        """Register *handler* for *kind*. ``payload_type`` is the msgspec
        decode type for the whole payload (e.g. ``list[ObjectKey]``)."""
        with self._registry_lock:
            self._registry[kind] = _Registration(
                handler=handler,
                decoder=msgspec.msgpack.Decoder(type=payload_type),
            )

    def start(self) -> None:  # type: ignore[override]
        if self.is_running:
            return
        super().start()

    def stop(self, timeout: float = 5.0) -> None:
        super().stop(timeout=timeout)
        PeriodicThreadRegistry.get_instance().unregister(self.name)
        self._drain_once()

    def dispatched_count(self) -> int:
        return self._dispatched_count  # single-writer; read is GIL-atomic

    def handler_exception_counts(self) -> dict[str, int]:
        with self._registry_lock:
            return dict(self._exception_counts)

    def _execute(self) -> ThreadRunSummary:
        before = self._dispatched_count
        self._drain_once()
        return ThreadRunSummary(
            success=True,
            message="dispatched=%d" % (self._dispatched_count - before),
        )

    def _drain_once(self) -> None:
        # Broad except keeps the drain thread alive across native/handler errors.
        try:
            completions = _lmc_ops.drain_recorded_completions()
        except Exception:
            logger.exception("DeviceHostFuncDispatcher: drain failed")
            return
        if not completions:
            return
        with self._registry_lock:
            registry = dict(self._registry)
        for kind, encoded_payload in completions:
            reg = registry.get(kind)
            if reg is None:
                logger.warning(
                    "DeviceHostFuncDispatcher: no handler for kind=%r (dropped)",
                    kind,
                )
                continue
            try:
                decoded = reg.decoder.decode(encoded_payload)
                reg.handler(decoded)
                self._dispatched_count += 1
            except Exception:
                with self._registry_lock:
                    self._exception_counts[kind] = (
                        self._exception_counts.get(kind, 0) + 1
                    )
                logger.exception(
                    "DeviceHostFuncDispatcher: handler for %r raised", kind
                )


def submit_callback_to_stream(stream: Any, kind: str, payload: Any) -> None:
    """Schedule a stream-ordered host callback for *kind* with *payload*.
    Runs on the dispatcher worker registered for *kind*; never acquires the
    GIL on the driver thread. ``payload`` is delivered to the handler as a
    single argument."""
    encoded = msgspec.msgpack.encode(payload)
    _lmc_ops.record_completion_on_stream(stream.ptr, kind, encoded)
