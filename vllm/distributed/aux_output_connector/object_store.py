# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared object stores that deliver auxiliary outputs to other processes."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import regex as re
from safetensors.numpy import load, save

from vllm.distributed.aux_output_connector.store import BlockObjectStoreError
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


class AuxObjectStore(Protocol):
    """A store of immutable objects that outlive the processes writing and
    reading them. Keys are opaque strings. Writing a key that exists succeeds
    and keeps the object written first. A refusal that may succeed later
    raises `vllm.distributed.aux_output_connector.store.BlockObjectStoreError`
    and is retried until the batch's deadline;
    any other exception drops the batch. The class connects with the
    classmethod `create(**object_store_extra_config)`; the engine calls
    `put_batch` and `close` from one writer thread."""

    def put_batch(self, keys: Sequence[str], values: Sequence[bytes]) -> None: ...

    def get(self, key: str) -> bytes | None:
        """Return the object, or `None` if it does not exist (yet)."""

    def remove_prefix(self, prefix: str) -> None:
        """Remove the objects whose keys start with `prefix`, as built by
        `aux_prefix` or `aux_namespace_prefix`."""

    def close(self) -> None: ...


def create_aux_object_store(name: str, extra_config: dict[str, Any]) -> AuxObjectStore:
    """Connect the object store `name`: `"mooncake"`, or the qualified name of
    a class that implements `AuxObjectStore`, with `extra_config` as the
    keyword arguments of its `create`."""
    if name == "mooncake":
        return MooncakeAuxObjectStore.create(**extra_config)
    if not all(name.split(".")) or "." not in name:
        raise ValueError(
            f"object_store {name!r} is neither 'mooncake' nor a qualified class name."
        )
    return resolve_obj_by_qualname(name).create(**extra_config)


_NO_AVAILABLE_HANDLE = -200


class MooncakeAuxObjectStore:
    """`AuxObjectStore` on a standalone Mooncake store."""

    def __init__(self, store: Any, replicate_config: Any) -> None:
        self._store = store
        self._replicate_config = replicate_config

    @classmethod
    def create(
        cls, config_path: str | None = None, hard_pin: bool = False
    ) -> MooncakeAuxObjectStore:
        """Connect with the Mooncake configuration file `config_path`, or the
        one at `MOONCAKE_CONFIG_PATH` if it is `None`, with a master and pool
        separate from any Mooncake KV store. Objects are written hard-pinned if
        `hard_pin`, for a memory-only store; by default they are not pinned, for
        a store that offloads to SSD on eviction."""
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
        """Write every object; raise `BlockObjectStoreError` if Mooncake has no
        room for them (`NO_AVAILABLE_HANDLE`), `RuntimeError` on other errors."""
        code = self._store.put_batch(list(keys), list(values), self._replicate_config)
        if code:
            error = (
                BlockObjectStoreError if code == _NO_AVAILABLE_HANDLE else RuntimeError
            )
            raise error(
                f"Mooncake failed to write {len(keys)} auxiliary-output "
                f"objects from {keys[0]!r}: {code}"
            )

    def get(self, key: str) -> bytes | None:
        """Return the object, or `None` if it does not exist."""
        return self._store.get(key) or None

    def remove_prefix(self, prefix: str) -> None:
        if (code := self._store.remove_by_regex(f"^{_escape(prefix)}", True)) < 0:
            raise RuntimeError(
                f"Mooncake failed to remove the objects under {prefix!r}: {code}"
            )

    def close(self) -> None:
        self._store.close()


_REGEX_SYNTAX = re.compile(r"[\\^$.*+?()[\]{}|]")


def _escape(text: str) -> str:
    return _REGEX_SYNTAX.sub(r"\\\g<0>", text)


def aux_prefix(key_prefix: str) -> str:
    """Key prefix of every object under `key_prefix`."""
    return f"{key_prefix}/"


def aux_namespace_prefix(key_prefix: str, namespace: str) -> str:
    """Key prefix of every object of a namespace and of the namespaces nested
    in it."""
    return f"{key_prefix}/{namespace}/"


NAMESPACE_SALT_MARKER = "@"


def aux_namespace_cache_salt(
    key_prefix: str, namespace: str, cache_salt: str | None
) -> str:
    """The cache salt of a request with `aux_namespace`, shared only within its
    namespace and key prefix; no client salt may contain its leading marker."""
    return NAMESPACE_SALT_MARKER + json.dumps([key_prefix, namespace, cache_salt])


def read_aux_rows(
    store: AuxObjectStore,
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


_RETRY_INTERVAL_S = 0.2
_MAX_QUEUED_BATCHES = 256

AuxObjects = list[tuple[str, dict[str, np.ndarray]]]


@dataclass(slots=True)
class _Batch:
    # Dropped when the batch misses its deadline, so that a put that does not
    # return holds no more than the batches queued before their deadline.
    objects: AuxObjects | None
    deadline: float


class AuxObjectWriter:
    """Write objects in the background, as safetensors with one array per
    field. A batch not written within `timeout_s` of being queued is dropped:
    the store refused it until then (refusals are retried every
    `_RETRY_INTERVAL_S`), its put failed, or it was still queued behind a put
    that had not returned. Its keys are missing; nothing else changes.
    Queuing waits while `_MAX_QUEUED_BATCHES` batches are queued, at most
    until the first of them is written or misses its deadline. While a batch
    past its deadline is still queued, the store is not taking writes:
    queuing drops the batch without waiting."""

    def __init__(self, store: AuxObjectStore, *, timeout_s: float) -> None:
        self._store = store
        self._timeout_s = timeout_s
        self._queue: queue.SimpleQueue[_Batch | None] = queue.SimpleQueue()
        self._cond = threading.Condition()
        # The batches not written yet, in queue order.
        self._pending: deque[_Batch] = deque()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vllm-aux-object-writer"
        )
        self._thread.start()

    def put(self, objects: AuxObjects) -> None:
        """Queue `(key, fields)` objects as one batch: a caller that puts once
        per step waits at most once per step."""
        with self._cond:
            self._expire(time.monotonic())
            while len(self._pending) >= _MAX_QUEUED_BATCHES and not self._stuck():
                self._cond.wait(self._pending[0].deadline - time.monotonic())
                self._expire(time.monotonic())
            if self._stuck():
                _drop(objects, "an earlier write missed its deadline")
                return
            batch = _Batch(objects, time.monotonic() + self._timeout_s)
            self._pending.append(batch)
            self._queue.put(batch)

    def _stuck(self) -> bool:
        # Deadlines follow queue order, so dropped batches come first.
        return bool(self._pending) and self._pending[0].objects is None

    def _expire(self, now: float) -> None:
        """Drop the pending batches past their deadline."""
        for batch in self._pending:
            if batch.deadline > now:
                break
            if batch.objects is not None:
                _drop(batch.objects, f"not written within {self._timeout_s:g} s")
                batch.objects = None

    def _run(self) -> None:
        try:
            while (batch := self._queue.get()) is not None:
                error = None
                if (objects := batch.objects) is not None:
                    try:
                        error = self._write(batch.deadline, objects)
                    except Exception as e:
                        error = e
                with self._cond:
                    if error is not None and batch.objects is not None:
                        _drop(batch.objects, error)
                        batch.objects = None
                    self._expire(time.monotonic())
                    self._pending.popleft()
                    self._cond.notify_all()
        finally:
            self._store.close()

    def _write(self, deadline: float, objects: AuxObjects) -> object:
        keys = [key for key, _ in objects]
        values = [save(fields) for _, fields in objects]
        error: object = "its deadline passed"
        while (wait := deadline - time.monotonic()) > 0:
            try:
                self._store.put_batch(keys, values)
                return None
            except BlockObjectStoreError as e:
                error = e
                time.sleep(min(_RETRY_INTERVAL_S, wait))
        return error

    def close(self) -> None:
        """Write what is queued, waiting until the last deadline at most; the
        writer thread closes the store once it is done."""
        with self._cond:
            pending = self._pending[-1].deadline if self._pending else None
        deadline = pending or time.monotonic() + self._timeout_s
        self._queue.put(None)
        self._thread.join(max(deadline - time.monotonic(), 0.0))


def _drop(objects: AuxObjects, reason: object) -> None:
    logger.error(
        "%d auxiliary-output objects from %r are not written: %s",
        len(objects),
        objects[0][0],
        reason,
    )
