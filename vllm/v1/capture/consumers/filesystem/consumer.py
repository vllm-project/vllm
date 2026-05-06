# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filesystem capture consumer — streams captured activations to disk
via ``ActivationWriter``.

Implements ``CaptureSink`` directly (bypasses ``CaptureConsumer`` /
``_BatchedAdapter``) because long captures must not be buffered in
memory. Each ``CaptureChunk`` is converted to bytes and submitted as a
``WriteTask``; ``CaptureFinalize`` produces a ``FinalizeTask`` with
sidecar JSON.

See ``docs/design/capture_consumers.md`` for the rationale.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from vllm.v1.capture.consumers.filesystem.types import (
    FilesystemCaptureRequest,
    FilesystemConsumerParams,
)
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
)
from vllm.v1.capture.consumers.filesystem.writer import (
    ActivationWriter,
    FinalizeTask,
    WriteTask,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.capture.types import CaptureContext

logger = logging.getLogger(__name__)


def _parse_params(params: dict[str, Any]) -> FilesystemConsumerParams:
    """Parse a raw ``params`` dict into ``FilesystemConsumerParams``.

    Only extracts known fields; unknown keys are silently ignored so
    forward-compatible config additions don't break old consumers.
    """
    if "root" not in params:
        raise ValueError(
            "FilesystemConsumer requires a 'root' parameter pointing "
            "to the output directory."
        )
    return FilesystemConsumerParams(
        root=str(params["root"]),
        writer_threads=int(params.get("writer_threads", 4)),
        queue_size=int(params.get("queue_size", 1024)),
        timeout_seconds=float(params.get("timeout_seconds", 180.0)),
        on_collision=str(params.get("on_collision", "overwrite")),
        fd_cache_size=int(params.get("fd_cache_size", 256)),
    )


class FilesystemConsumer:
    """Capture consumer that streams activations to the filesystem.

    Implements the ``CaptureSink`` protocol directly. Wraps the
    in-package ``ActivationWriter`` thread pool.

    Path layout: ``{root}/{tag_slug}/{request_id_slug}/{layer}_{hook}.bin``
    """

    location: ClassVar[Literal["worker"]] = "worker"
    reads_client_spec: ClassVar[bool] = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        params: dict[str, Any],
    ) -> None:
        self._vllm_config = vllm_config
        self._params = _parse_params(params)
        self._root = pathlib.Path(self._params.root)
        self._writer = ActivationWriter(
            root=self._root,
            num_threads=self._params.writer_threads,
            queue_size=self._params.queue_size,
            timeout_seconds=self._params.timeout_seconds,
            on_collision=self._params.on_collision,  # type: ignore[arg-type]
            fd_cache_size=self._params.fd_cache_size,
        )
        # Lock protecting _key_paths which tracks the slug-based path
        # components per CaptureKey for use at finalize time.
        self._lock = threading.Lock()
        self._key_paths: dict[CaptureKey, tuple[str, str]] = {}
        # Per writer-key events set by _on_write_result when a result
        # goes terminal. Keyed by writer-side (str, int, str) tuples.
        self._wait_lock = threading.Lock()
        self._wait_events: dict[tuple[str, int, str], threading.Event] = {}
        self._writer.add_status_callback(self._on_write_result)

    # ------------------------------------------------------------------
    # CaptureSink protocol
    # ------------------------------------------------------------------

    def global_capture_spec(self) -> None:
        """Filesystem consumer has no global spec — capture is per-request."""
        return None

    def validate_client_spec(
        self,
        raw_spec: Any,
        ctx: CaptureContext,
    ) -> CaptureSpec:
        """Validate a per-request ``FilesystemCaptureRequest``.

        Accepts either a ``FilesystemCaptureRequest`` dataclass or a
        plain dict with the same fields.
        """
        if isinstance(raw_spec, dict):
            raw_spec = FilesystemCaptureRequest(**raw_spec)
        if not isinstance(raw_spec, FilesystemCaptureRequest):
            raise TypeError(
                f"Expected FilesystemCaptureRequest or dict, "
                f"got {type(raw_spec).__name__}"
            )
        # Lazy import to avoid pulling in pydantic (via vllm.config) at
        # module load time — keeps the consumer importable in lightweight
        # test environments.
        from vllm.v1.capture.consumers.filesystem.validation import (
            validate_filesystem_request,
        )

        return validate_filesystem_request(raw_spec, self._vllm_config, ctx)

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        """Convert a ``CaptureChunk`` to bytes and submit a ``WriteTask``."""
        key = chunk.key
        _request_id, layer_idx, hook_name = key

        # Compute path: {root}/{tag}/{request_id}/{layer}_{hook}.bin
        # We need tag and request_id slugs. The CaptureKey only has the
        # vllm_internal_request_id. For the filesystem layout we use
        # the key components directly — the request_id in the key IS the
        # vllm internal id, and we use it as the directory name.
        # The tag comes from the metadata on the first chunk, or we
        # fall back to "default".
        tag_slug = chunk.metadata.get("tag_slug", "default")
        request_id_slug = chunk.metadata.get("request_id_slug", str(_request_id))

        # Cache the slug paths for use at finalize time.
        with self._lock:
            if key not in self._key_paths:
                self._key_paths[key] = (tag_slug, request_id_slug)

        bin_path = (
            self._root / tag_slug / request_id_slug / f"{layer_idx}_{hook_name}.bin"
        )

        # Convert tensor to bytes (tensor is already on CPU).
        tensor_bytes = chunk.tensor.numpy().tobytes()

        writer_key = (str(_request_id), layer_idx, hook_name)
        self._writer.submit(
            WriteTask(
                path=bin_path,
                payload=tensor_bytes,
                append=True,
                key=writer_key,
            )
        )

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        """Build and submit a ``FinalizeTask`` for this key."""
        key = finalize.key
        _request_id, layer_idx, hook_name = key

        # Retrieve cached path slugs.
        with self._lock:
            path_info = self._key_paths.pop(key, None)

        tag_slug = "default"
        request_id_slug = str(_request_id)
        if path_info is not None:
            tag_slug, request_id_slug = path_info

        # Also check the finalize sidecar for slug overrides.
        tag_slug = finalize.sidecar.get("tag_slug", tag_slug)
        request_id_slug = finalize.sidecar.get("request_id_slug", request_id_slug)

        bin_path = (
            self._root / tag_slug / request_id_slug / f"{layer_idx}_{hook_name}.bin"
        )
        sidecar_path = bin_path.with_suffix(".json")

        # Build sidecar payload from the finalize sidecar.
        sidecar_payload: dict[str, Any] = {
            "request_id": str(_request_id),
            "layer": layer_idx,
            "hook": hook_name,
        }
        sidecar_payload.update(finalize.sidecar)

        writer_key = (str(_request_id), layer_idx, hook_name)
        self._writer.submit(
            FinalizeTask(
                bin_path=bin_path,
                sidecar_path=sidecar_path,
                sidecar_payload=sidecar_payload,
                key=writer_key,
            )
        )

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        """Map the writer's ``WriteResult`` to a ``CaptureResult``."""
        _request_id, layer_idx, hook_name = key
        writer_key = (str(_request_id), layer_idx, hook_name)
        write_result = self._writer.get_result(writer_key)
        if write_result is None:
            return None

        error_str: str | None = None
        if write_result.error is not None:
            error_str = str(write_result.error)

        payload: list[str] | None = None
        if write_result.bin_path is not None:
            paths = [str(write_result.bin_path)]
            if write_result.sidecar_path is not None:
                paths.append(str(write_result.sidecar_path))
            payload = paths

        return CaptureResult(
            key=key,
            status=write_result.status,
            error=error_str,
            payload=payload,
        )

    def wait_for_result(
        self,
        key: CaptureKey,
        timeout: float,
    ) -> CaptureResult | None:
        """Block up to ``timeout`` seconds for the terminal result for ``key``.

        Uses a per-key :class:`threading.Event` that
        :meth:`_on_write_result` sets when the underlying
        ``ActivationWriter`` worker transitions the result to a terminal
        status.  If the result is already terminal on entry the method
        returns immediately without waiting.
        """
        _request_id, layer_idx, hook_name = key
        writer_key = (str(_request_id), layer_idx, hook_name)

        with self._wait_lock:
            wr = self._writer.get_result(writer_key)
            if wr is not None and wr.status in ("ok", "error"):
                return self.get_result(key)
            event = self._wait_events.setdefault(writer_key, threading.Event())

        event.wait(timeout=timeout)
        return self.get_result(key)

    def _on_write_result(self, write_result: Any) -> None:
        """Status callback fired by ``ActivationWriter`` on terminal transitions.

        Called outside ``ActivationWriter._results_lock`` (see the
        ``fired`` list pattern in ``_record_ok`` / ``_record_error``).
        Sets the per-key event so any thread blocked in
        :meth:`wait_for_result` can wake up.
        """
        if write_result.status in ("ok", "error"):
            with self._wait_lock:
                event = self._wait_events.pop(write_result.key, None)
            if event is not None:
                event.set()

    def shutdown(self, timeout: float = 30.0) -> None:
        """Forward shutdown to the underlying ``ActivationWriter``."""
        self._writer.shutdown(timeout=timeout)
