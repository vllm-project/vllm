# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared object stores that deliver auxiliary outputs to other processes."""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from safetensors.numpy import load, save

from vllm.distributed.aux_output_connector.store import BlockObjectStoreError
from vllm.logger import init_logger

logger = init_logger(__name__)


class MooncakeAuxObjectStore:
    """Immutable objects in a standalone Mooncake store. Writing a key that
    exists succeeds and keeps the object written first."""

    def __init__(self, store: Any, replicate_config: Any) -> None:
        self._store = store
        self._replicate_config = replicate_config

    @classmethod
    def create(
        cls, config_path: str | None = None, hard_pin: bool = False
    ) -> MooncakeAuxObjectStore:
        """Connect with the Mooncake configuration file `config_path`, or the
        one at `MOONCAKE_CONFIG_PATH` if it is `None`. Objects are written
        hard-pinned if `hard_pin`."""
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
            rdma_utils,
        )
        from vllm.distributed.mooncake_store import (
            MooncakeStoreConfig,
            setup_mooncake_store,
        )
        from vllm.utils.network_utils import get_ip

        config = (
            MooncakeStoreConfig.load_from_config()
            if config_path is None
            else MooncakeStoreConfig.from_file(config_path)
        )
        if config.mode != "standalone-store":
            raise ValueError(
                "Auxiliary-output objects must outlive the clients that write "
                "and read them: set mode to 'standalone-store' in the Mooncake "
                "configuration."
            )
        store = MooncakeDistributedStore()
        setup_mooncake_store(
            store, config, rdma_utils.get_requester_local_hostname(get_ip())
        )
        replicate_config = ReplicateConfig()
        replicate_config.with_hard_pin = hard_pin
        return cls(store, replicate_config)

    def put_batch(self, keys: Sequence[str], values: Sequence[bytes]) -> None:
        """Write every object, or raise `BlockObjectStoreError`."""
        code = self._store.put_batch(list(keys), list(values), self._replicate_config)
        if code:
            raise BlockObjectStoreError(
                f"Mooncake failed to write {len(keys)} auxiliary-output "
                f"objects from {keys[0]!r}: {code}"
            )

    def get(self, key: str) -> bytes | None:
        """Return the object, or `None` if it does not exist."""
        return self._store.get(key) or None

    def close(self) -> None:
        self._store.close()


_REGEX_SYNTAX = re.compile(r"[\\^$.*+?()[\]{}|]")


def _escape(text: str) -> str:
    return _REGEX_SYNTAX.sub(r"\\\g<0>", text)


def aux_prefix_pattern(key_prefix: str) -> str:
    """Mooncake regex of every object under `key_prefix`."""
    return f"^{_escape(key_prefix)}/"


def aux_namespace_pattern(key_prefix: str, namespace: str) -> str:
    """Mooncake regex of every object of a namespace and of the namespaces
    nested in it."""
    return f"^{_escape(key_prefix)}/{_escape(namespace)}/"


def read_aux_rows(
    store: MooncakeAuxObjectStore,
    keys: Sequence[str],
    field: str,
    timeout_s: float = 600.0,
) -> np.ndarray:
    """Concatenate `field` of the objects `keys`; wait up to `timeout_s` in
    total for objects that are still being written."""
    deadline = time.monotonic() + timeout_s
    parts = []
    for key in keys:
        while (data := store.get(key)) is None:
            if time.monotonic() >= deadline:
                raise BlockObjectStoreError(
                    f"auxiliary-output object {key!r} is missing after {timeout_s}s"
                )
            time.sleep(0.05)
        parts.append(load(data)[field])
    return np.concatenate(parts) if parts else np.empty(0)


_PUT_RETRY_INTERVAL_S = 0.2
_PUT_CHUNK_SIZE = 64

AuxObjects = list[tuple[str, dict[str, np.ndarray]]]


class AuxObjectWriter:
    """Write objects in the background as safetensors with one array per
    field, at most `_PUT_CHUNK_SIZE` per put.

    A put the store refuses is retried every `_PUT_RETRY_INTERVAL_S` one
    object at a time, without the objects already written, and queuing waits
    while it is. Once objects have been pending for `stall_timeout_s` without
    a put succeeding, the rest of the batch is dropped, later batches get one
    attempt each until a put succeeds, and a batch that meets a full queue is
    dropped. Once a put has not returned for `stall_timeout_s`, `drain` stops
    waiting for it and drops the batches queued before it returned. A dropped
    object is missing from the store.
    """

    def __init__(
        self,
        store: MooncakeAuxObjectStore,
        *,
        max_pending_batches: int,
        stall_timeout_s: float,
    ) -> None:
        self._store = store
        self._stall_timeout_s = stall_timeout_s
        self._queue: queue.Queue[tuple[int, AuxObjects] | None] = queue.Queue(
            maxsize=max_pending_batches
        )
        self._last_put = time.monotonic()
        self._put_since: float | None = None
        self._num_dropped = 0
        self._num_batches = 0
        self._cut = 0
        self._cut_lock = threading.Lock()
        self._error: BaseException | None = None
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vllm-aux-object-writer"
        )
        self._thread.start()

    def _run(self) -> None:
        while (batch := self._queue.get()) is not None:
            try:
                if self._error is None:
                    self._write(*batch)
            except BaseException as error:
                self._error = error
            finally:
                self._queue.task_done()
        self._queue.task_done()

    def _stalled(self) -> bool:
        return time.monotonic() - self._last_put >= self._stall_timeout_s

    def _hung(self) -> bool:
        put_since = self._put_since
        return (
            put_since is not None
            and time.monotonic() - put_since >= self._stall_timeout_s
        )

    def _drop(self, keys: Sequence[str], reason: object) -> None:
        if not self._num_dropped:
            logger.error(
                "Dropping auxiliary output writes from %r on: %s", keys[0], reason
            )
        self._num_dropped += len(keys)

    def _write(self, sequence: int, objects: AuxObjects) -> None:
        keys = [key for key, _ in objects]
        values = [save(fields) for _, fields in objects]
        num_single = 0
        while keys:
            with self._cut_lock:
                if sequence <= self._cut:
                    self._drop(keys, "a put hung while they were queued")
                    return
                self._put_since = time.monotonic()
            step = 1 if num_single else _PUT_CHUNK_SIZE
            try:
                self._store.put_batch(keys[:step], values[:step])
            except BlockObjectStoreError as error:
                num_single = num_single or min(len(keys), _PUT_CHUNK_SIZE)
                if (
                    self._num_dropped
                    or self._stalled()
                    or self._closed.wait(_PUT_RETRY_INTERVAL_S)
                ):
                    self._drop(keys, error)
                    return
            else:
                keys, values = keys[step:], values[step:]
                num_single = max(num_single - 1, 0)
                self._last_put = time.monotonic()
                if self._num_dropped:
                    logger.warning(
                        "Auxiliary output objects are written again; %d were dropped.",
                        self._num_dropped,
                    )
                    self._num_dropped = 0
            finally:
                self._put_since = None

    def check(self) -> None:
        """Raise `BlockObjectStoreError` if writing failed other than in the
        store."""
        if self._error is not None:
            raise BlockObjectStoreError(
                "auxiliary output object write failed"
            ) from self._error

    def put(self, objects: AuxObjects) -> None:
        """Queue `(key, fields)` objects."""
        if not self._queue.unfinished_tasks:
            self._last_put = time.monotonic()
        self._num_batches += 1
        batch = (self._num_batches, objects)
        while True:
            wait = not (self._num_dropped or self._stalled())
            try:
                self._queue.put(batch, wait, _PUT_RETRY_INTERVAL_S)
                return
            except queue.Full:
                self.check()
                if not wait:
                    self._drop([key for key, _ in objects], "the queue is full")
                    return

    def drain(self) -> None:
        """Wait until the queued objects are written or dropped, or a put has
        not returned for the stall deadline; raise `BlockObjectStoreError` if
        writing failed other than in the store."""
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks and self._error is None:
                with self._cut_lock:
                    if self._hung():
                        self._cut = self._num_batches
                        break
                self._queue.all_tasks_done.wait(_PUT_RETRY_INTERVAL_S)
        self.check()

    def close(self) -> None:
        self._closed.set()
        self._queue.put(None)
        self._thread.join()
        self._store.close()
