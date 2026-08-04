# SPDX-License-Identifier: Apache-2.0

"""Trace recorder — writes TRACE_CALL events to a binary file.

Architecture:

* The recorder is an :class:`EventSubscriber` registered on the global
  EventBus.  Subscriber callbacks run on the EventBus drain thread, so
  they are already off the request path.
* Encoding (codec + msgpack) and disk I/O happen synchronously inside
  the callback.  Adding a second worker thread would be premature
  optimization; the EventBus drain thread already serves that role.
* Length-prefixed framing: each frame is written as a 4-byte
  big-endian length followed by msgpack bytes.  This keeps the reader
  simple and tolerates partial-write tail truncation gracefully.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any
import hashlib
import json
import os
import struct
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber
from lmcache.v1.mp_observability.trace import codecs
from lmcache.v1.mp_observability.trace.decorator import set_tracing_enabled
from lmcache.v1.mp_observability.trace.format import (
    FORMAT_VERSION,
    MAGIC,
    TRACE_SCHEMA_VERSION,
    Header,
    Record,
    encode_header,
    encode_record,
)

logger = init_logger(__name__)

#: Frame length prefix size (bytes).  Big-endian uint32 — 4 GiB cap
#: per frame which is far above any expected record size.
_LEN_PREFIX = 4
_LEN_STRUCT = struct.Struct(">I")


class TraceRecorder(EventSubscriber, ABC):
    """Base class for trace recorders.

    Concrete subclasses select which events they care about.  This
    base provides:

    * file management (open / write / fsync / close)
    * header emission
    * the trace-gate flip (on at construction, off at ``close()``)

    Subclasses implement :meth:`get_subscriptions`.
    """

    def __init__(self, output_path: str, level: str) -> None:
        self._output_path = output_path
        self._level = level
        self._fd = open(output_path, "wb", buffering=0)
        self._lock = threading.Lock()
        self._closed = False
        self._dropped_count = 0
        self._header_written = False
        self._t_mono_start = time.monotonic()
        self._t_wall_start = time.time()

        # The header is written lazily on the first of: a successful
        # ``attach_storage_config`` call, the first record, or
        # ``close()``.  Deferring avoids the in-place rewrite problem
        # that would otherwise corrupt the file when the placeholder
        # header and the final header have different byte lengths.
        # Flip the trace gate AFTER the file is open so a racing publish
        # cannot land on a half-initialized recorder.
        set_tracing_enabled(True)
        logger.info("trace recorder writing to %s (level=%s)", output_path, level)

    # ---- subclass extension points ------------------------------------

    @abstractmethod
    def get_subscriptions(self) -> dict[EventType, EventCallback]: ...

    # ---- public API ---------------------------------------------------

    @property
    def output_path(self) -> str:
        """Path of the trace file on disk."""
        return self._output_path

    @property
    def dropped_count(self) -> int:
        """Number of records that failed to encode/write."""
        return self._dropped_count

    def attach_storage_config(self, config: StorageManagerConfig) -> None:
        """Write the header populated from the StorageManagerConfig.

        Must be called before any records are written; the
        server lifecycle does this immediately after construction.
        Subsequent calls are silently ignored — the header is written
        once for the lifetime of the file.

        Args:
            config: The StorageManagerConfig in use.  Its dataclass
                form is JSON-serialized and SHA-256 hashed for the
                header digest, so a replay driver can detect
                mismatched configurations.
        """
        with self._lock:
            if self._closed or self._header_written:
                return
            sm_json = json.dumps(self._safe_config_dict(config), sort_keys=True)
            digest = hashlib.sha256(sm_json.encode("utf-8")).hexdigest()
            self._write_header(sm_json, digest)
            self._header_written = True

    def shutdown(self) -> None:
        """:class:`EventBus` shutdown hook — close the recorder."""
        self.close()

    def close(self) -> None:
        """Flush, fsync, and close the trace file.

        Idempotent.  Flips the trace gate off so any straggler
        publishes after this point are no-ops.  Writes a fallback
        empty-config header if neither ``attach_storage_config`` nor
        any record was written, so the resulting file is always
        readable.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            set_tracing_enabled(False)
            try:
                if not self._header_written:
                    self._write_header(sm_config_json="", sm_config_digest="")
                    self._header_written = True
                self._fd.flush()
                os.fsync(self._fd.fileno())
            except OSError:
                logger.exception("trace recorder: fsync failed")
            finally:
                self._fd.close()
        if self._dropped_count:
            logger.warning(
                "trace recorder closed; %d record(s) dropped", self._dropped_count
            )
        else:
            logger.info("trace recorder closed cleanly: %s", self._output_path)

    # ---- internal -----------------------------------------------------

    def _write_header(self, sm_config_json: str, sm_config_digest: str) -> None:
        header = Header(
            magic=MAGIC,
            format_version=FORMAT_VERSION,
            level=self._level,
            trace_schema_version=TRACE_SCHEMA_VERSION,
            t_mono_start=self._t_mono_start,
            t_wall_start=self._t_wall_start,
            sm_config_json=sm_config_json,
            sm_config_digest=sm_config_digest,
        )
        self._write_frame(encode_header(header))

    def _write_frame(self, frame: bytes) -> None:
        # Single write so the prefix and body land atomically for frames
        # below PIPE_BUF; two writes would let a concurrent appender
        # interleave between them and would also double the syscall
        # count on an unbuffered fd.  Caller holds ``self._lock`` (or is
        # in __init__ before the gate flips on).
        self._fd.write(_LEN_STRUCT.pack(len(frame)) + frame)

    def _on_trace_call(self, event: Event) -> None:
        """Encode and append one TRACE_CALL event.

        Errors are logged at WARNING and counted, but do not propagate
        — losing a record is preferable to taking down the EventBus
        drain thread.
        """
        try:
            qualname = event.metadata["qualname"]
            args = event.metadata["args"]
            # ``t_mono`` is stamped in the metadata at publish time
            # (see ``publish_call_event``) so the recorded value is
            # co-temporal with ``event.timestamp`` (wall-clock) instead
            # of picking up the drain-thread delay.
            publish_t_mono = event.metadata["t_mono"]
            encoded_args = codecs.encode_args(args)
            t_mono = max(0.0, publish_t_mono - self._t_mono_start)
            record = Record(
                t_mono=t_mono,
                t_wall=event.timestamp,
                qualname=qualname,
                args=encoded_args,
            )
            frame = encode_record(record)
        except Exception:
            self._dropped_count += 1
            logger.warning(
                "trace recorder: failed to encode TRACE_CALL event "
                "(qualname=%s); dropping",
                event.metadata.get("qualname", "<unknown>"),
                exc_info=True,
            )
            return

        with self._lock:
            if self._closed:
                self._dropped_count += 1
                return
            try:
                # Write a placeholder header on first record if the
                # caller never invoked ``attach_storage_config``.
                # Ensures the file is always readable.
                if not self._header_written:
                    self._write_header(sm_config_json="", sm_config_digest="")
                    self._header_written = True
                self._write_frame(frame)
            except OSError:
                self._dropped_count += 1
                logger.warning(
                    "trace recorder: write failed; dropping record",
                    exc_info=True,
                )

    @staticmethod
    def _safe_config_dict(config: StorageManagerConfig) -> dict[str, Any]:
        """Best-effort conversion of a StorageManagerConfig to a JSON
        dict.

        Delegates to the module-level
        :func:`safe_storage_config_dict` so the replay driver can
        reproduce the exact digest the recorder writes.  Kept as a
        staticmethod for backwards-compatible access.
        """
        return safe_storage_config_dict(config)


def safe_storage_config_dict(config: StorageManagerConfig) -> dict[str, Any]:
    """Best-effort JSON-serializable dict view of a StorageManagerConfig.

    Falls back to ``str(config)`` for fields that are not directly
    serializable.  The result is used only for the header digest and
    human inspection — it does not need to be replay-faithful.

    Exposed publicly so that the replay driver can compute a digest
    of its *own* config using the same algorithm the recorder used,
    for mismatch detection.

    Args:
        config: The StorageManagerConfig to serialize.

    Returns:
        A JSON-friendly dict.
    """
    if is_dataclass(config):
        try:
            return _coerce_jsonable(asdict(config))
        except Exception:
            pass
    return {"repr": str(config)}


def _coerce_jsonable(obj: Any) -> Any:
    """Recursively coerce ``obj`` into a JSON-friendly value.

    Anything not natively serializable falls through to ``str(obj)``.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_coerce_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _coerce_jsonable(v) for k, v in obj.items()}
    return str(obj)


class StorageTraceRecorder(TraceRecorder):
    """Records every ``TRACE_CALL`` event into a ``"storage"``-level file."""

    def __init__(self, output_path: str) -> None:
        super().__init__(output_path=output_path, level="storage")

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {EventType.TRACE_CALL: self._on_trace_call}
