# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared object stores that deliver auxiliary outputs to other processes."""

from __future__ import annotations

import json
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
    total for objects still being written, unless their namespace has a drop."""
    deadline = time.monotonic() + timeout_s
    parts = []
    for key in keys:
        while (data := store.get(key)) is None:
            if (reason := store.get(_drop_marker(key))) is not None:
                raise BlockObjectStoreError(
                    f"auxiliary-output object {key!r} is missing, and the engine "
                    f"dropped objects of its namespace: {reason.decode()}"
                )
            if time.monotonic() >= deadline:
                raise BlockObjectStoreError(
                    f"auxiliary-output object {key!r} is missing after {timeout_s}s"
                )
            time.sleep(0.05)
        parts.append(load(data)[field])
    return np.concatenate(parts) if parts else np.empty(0)


def _drop_marker(key: str) -> str:
    return f"{key.rpartition('/')[0]}/dropped"


_RETRY_INTERVAL_S = 0.2
_MAX_QUEUED_BATCHES = 256

AuxObjects = list[tuple[str, dict[str, np.ndarray]]]


@dataclass(slots=True)
class _Batch:
    objects: AuxObjects
    deadline: float


class AuxObjectWriter:
    """Write objects in the background, as safetensors with one array per
    field. A batch not written within `timeout_s` of being queued is dropped:
    the store refused it until then (refusals are retried every
    `_RETRY_INTERVAL_S`), or it was still queued behind a put that had not
    returned. A batch whose put fails otherwise is dropped at once. Its
    namespaces get a drop marker that readers stop at; nothing else changes.
    Queuing waits while `_MAX_QUEUED_BATCHES` batches are queued or being
    written, at most until the first of them is written or misses its deadline.
    Once a batch has missed its deadline, until a put succeeds, queuing never
    waits: a full queue drops the batch."""

    def __init__(self, store: AuxObjectStore, *, timeout_s: float) -> None:
        self._store = store
        self._timeout_s = timeout_s
        self._cond = threading.Condition()
        # The batches not written yet, in queue order, and the one being written.
        self._pending: deque[_Batch] = deque()
        self._active: _Batch | None = None
        self._closing = False
        self._failing = False
        self._markers: dict[str, bytes] = {}
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vllm-aux-object-writer"
        )
        self._thread.start()

    def put(self, objects: AuxObjects) -> None:
        """Queue `(key, fields)` objects as one batch: a caller that puts once
        per step waits at most once per step."""
        with self._cond:
            self._expire(time.monotonic())
            while self._num_queued() >= _MAX_QUEUED_BATCHES and not self._failing:
                head = self._active or self._pending[0]
                self._cond.wait(head.deadline - time.monotonic())
                self._expire(time.monotonic())
            if self._num_queued() >= _MAX_QUEUED_BATCHES:
                self._drop(
                    objects, "the queue is full after a write missed its deadline"
                )
                return
            self._pending.append(_Batch(objects, time.monotonic() + self._timeout_s))
            self._cond.notify_all()

    def _num_queued(self) -> int:
        return len(self._pending) + (self._active is not None)

    def _expire(self, now: float) -> None:
        """Drop the queued batches past their deadline; the batch being written
        past its deadline is settled by its put."""
        if self._active is not None and self._active.deadline <= now:
            self._failing = True
        while self._pending and self._pending[0].deadline <= now:
            batch = self._pending.popleft()
            self._drop(batch.objects, f"not written within {self._timeout_s:g} s")
            self._failing = True

    def _run(self) -> None:
        try:
            while True:
                with self._cond:
                    self._expire(time.monotonic())
                    while not self._pending and not self._closing:
                        self._cond.wait()
                        self._expire(time.monotonic())
                    if not self._pending:
                        break
                    batch = self._active = self._pending.popleft()
                error: object = None
                try:
                    error = self._write(batch.deadline, batch.objects)
                except Exception as e:
                    error = e
                late = time.monotonic() >= batch.deadline
                with self._cond:
                    self._active = None
                    if error is None:
                        self._failing = False
                    else:
                        self._drop(batch.objects, error)
                        self._failing |= late
                    self._expire(time.monotonic())
                    self._cond.notify_all()
                    markers, self._markers = self._markers, {}
                if markers:
                    self._write_markers(markers)
        finally:
            self._store.close()

    def _drop(self, objects: AuxObjects, reason: object) -> None:
        logger.error(
            "%d auxiliary-output objects from %r are not written: %s",
            len(objects),
            objects[0][0],
            reason,
        )
        text = str(reason).encode() or b"?"
        for key, _ in objects:
            self._markers.setdefault(_drop_marker(key), text)

    def _write_markers(self, markers: dict[str, bytes]) -> None:
        try:
            self._store.put_batch(list(markers), list(markers.values()))
        except Exception as e:
            logger.error("Drop markers %s are not written: %s", list(markers), e)

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
            last = self._pending[-1] if self._pending else self._active
            deadline = last.deadline if last else time.monotonic() + self._timeout_s
            self._closing = True
            self._cond.notify_all()
        self._thread.join(max(deadline - time.monotonic(), 0.0))
