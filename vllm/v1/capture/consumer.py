# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``CaptureConsumer`` user-facing base class plus the internal
batched adapter that makes ``CaptureConsumer`` a ``CaptureSink``.

The adapter lives in this module (rather than under ``sink.py``)
because it is an implementation detail of how ``CaptureConsumer``
fulfills the ``CaptureSink`` contract — ``sink.py`` stays protocol-only.
"""

from __future__ import annotations

import contextlib
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.capture.types import CaptureContext


class CaptureConsumer(ABC):
    """User-facing base class for capture consumers.

    The default implementation accumulates ``CaptureChunk``s per
    ``CaptureKey`` in CPU memory until ``CaptureFinalize`` arrives,
    then invokes ``on_capture(key, tensor, sidecar)`` with the
    concatenated tensor. Subclasses override ``on_capture`` and
    usually nothing else.

    Subclasses set class-level metadata:

    - ``location``: where the consumer runs. ``"worker"`` (default)
      runs in the engine-core subprocess alongside the model runner
      with direct in-process access to the capture manager and no
      IPC overhead. ``"driver"`` runs in the main Python process
      where the ``LLM`` lives; vLLM transparently handles the
      worker→driver plumbing via ``torch.multiprocessing.Queue``
      with shared-memory tensor handoff.
    - ``required_sidecar_fields``: optional sidecar field names the
      framework must populate for this consumer.
      ``vllm_internal_request_id`` is always present.
    - ``reads_client_spec``: whether the consumer accepts per-request
      opt-in via ``SamplingParams.capture[consumer_name]``. Default
      ``False`` — most consumers have a global spec set at
      registration time.

    Override points, in order of necessity:

    - ``__init__(self, vllm_config, params)`` — called once at engine
      startup.
    - ``global_capture_spec()`` — the consumer's global capture spec,
      applied to every request. Default ``None``.
    - ``validate_client_spec(raw_spec, ctx)`` — if
      ``reads_client_spec = True``, called at admission time.
      Must return a ``CaptureSpec`` or raise
      ``CaptureValidationError``.
    - ``on_capture(key, tensor, sidecar)`` — the main override.
    - ``on_error(key, error)`` — called on capture failure for this
      key. Default ``pass``.
    - ``shutdown(timeout)`` — called on engine teardown. Default
      ``pass``.
    """

    location: ClassVar[Literal["worker", "driver"]] = "worker"
    required_sidecar_fields: ClassVar[frozenset[str]] = frozenset()
    reads_client_spec: ClassVar[bool] = False

    def __init__(  # noqa: B027 — intentional no-op default.
        self,
        vllm_config: VllmConfig,
        params: dict[str, Any],
    ) -> None:
        pass

    def global_capture_spec(self) -> CaptureSpec | None:
        return None

    def validate_client_spec(
        self,
        raw_spec: Any,
        ctx: CaptureContext,
    ) -> CaptureSpec:
        raise NotImplementedError(
            f"{type(self).__name__} has reads_client_spec=True but "
            f"did not override validate_client_spec()."
        )

    @abstractmethod
    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        """Called once per finalized capture key.

        ``tensor`` has shape ``(num_rows, hidden_size)`` in the dtype
        captured. ``sidecar`` is filtered to the consumer's
        ``required_sidecar_fields`` plus ``vllm_internal_request_id``.
        """

    def on_error(  # noqa: B027 — intentional no-op default.
        self,
        key: CaptureKey,
        error: str,
    ) -> None:
        pass

    def shutdown(self, timeout: float = 30.0) -> None:  # noqa: B027
        pass


class _BatchedAdapter:
    """Adapts a ``CaptureConsumer`` to the ``CaptureSink`` protocol.

    Accumulates every ``CaptureChunk`` for a key in CPU memory. When
    ``submit_finalize`` arrives for that key, concatenates the buffered
    tensors in ``row_offset`` order and invokes ``on_capture`` exactly
    once. Exceptions raised by ``on_capture`` are caught and surfaced
    via ``get_result(key).status == "error"`` — they never propagate
    into the manager (invariant 9: consumer isolation).

    The sort by ``row_offset`` is defensive. The manager guarantees
    in-order delivery per key (invariant 4), but sorting on finalize
    costs one ``sorted()`` call per key and keeps the adapter robust
    against test-injected disorder and any future manager changes.
    """

    def __init__(self, consumer: CaptureConsumer) -> None:
        self._consumer = consumer
        self.location: Literal["worker", "driver"] = consumer.location
        self._lock = threading.Lock()
        self._pending: dict[CaptureKey, list[tuple[int, torch.Tensor]]] = {}
        self._results: dict[CaptureKey, CaptureResult] = {}

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        with self._lock:
            self._pending.setdefault(chunk.key, []).append(
                (chunk.row_offset, chunk.tensor)
            )

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        key = finalize.key
        with self._lock:
            buffered = self._pending.pop(key, [])

        if not buffered:
            # No chunks were ever submitted for this key. Treat as an
            # empty capture — ``on_capture`` receives a zero-row tensor
            # so consumers don't have to special-case the empty case.
            tensor = torch.empty((0,))
        else:
            ordered = sorted(buffered, key=lambda pair: pair[0])
            tensors = [t for _, t in ordered]
            tensor = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)

        try:
            self._consumer.on_capture(key, tensor, finalize.sidecar)
        except Exception as exc:  # noqa: BLE001 — consumer isolation.
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._results[key] = CaptureResult(key=key, status="error", error=error)
            # Consumer's own ``on_error`` is best-effort; a bug there
            # must not break isolation for other consumers.
            with contextlib.suppress(Exception):
                self._consumer.on_error(key, error)
            return

        with self._lock:
            self._results[key] = CaptureResult(key=key, status="ok")

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        with self._lock:
            return self._results.get(key)

    def wait_for_result(
        self,
        key: CaptureKey,
        timeout: float,
    ) -> CaptureResult | None:
        return self.get_result(key)

    def shutdown(self, timeout: float = 30.0) -> None:
        self._consumer.shutdown(timeout)
