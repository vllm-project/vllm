# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side identities and keys of session objects."""

from __future__ import annotations

import itertools
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.aux_output_connector.session_objects import aux_object_key

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.request import Request


@dataclass(slots=True)
class _RequestObjects:
    session: str | None
    emit_start: int
    next_object: int
    ids: dict[int, str] = field(default_factory=dict)
    worker_sent: int = 0
    published: int = 0
    keys_sent: int = 0
    tail_start: int = 0
    tail: np.ndarray | None = None
    keys: list[str] | None = None


class SessionObjects:
    """Name every KV computation of an object and track which objects each
    session holds in the store.

    An object is one KV cache block of positions. Its identity is the
    computation that produced the KV of that block: computing the block again,
    for example after an eviction or a weight update, gives a new identity, so
    a key never names two different computations. A session request hits an
    object only while the cached KV block that a hit reads carries the
    identity the session holds, and it lists the identities it took at
    admission or computed itself.
    """

    def __init__(
        self,
        object_size: int,
        hash_block_size: int,
        key_prefix: str,
        idle_timeout_s: float,
        cached_block: Callable[[BlockHash], int | None],
        request_block: Callable[[Request, int], int | None],
    ) -> None:
        self._object_size = object_size
        self._hashes_per_object = object_size // hash_block_size
        self._key_prefix = key_prefix
        self._idle_timeout_s = idle_timeout_s
        self._cached_block = cached_block
        self._request_block = request_block
        self._id_prefix = secrets.token_hex(4)
        self._next_id = itertools.count()
        self._block_ids: dict[int, str] = {}
        self._published: dict[str, dict[bytes, str]] = {}
        self._requests: dict[str, _RequestObjects] = {}
        self._session_requests: dict[str, set[str]] = {}
        self._last_used: dict[str, float] = {}
        self._released: list[str] = []
        self._next_sweep = 0.0

    def _block_hash(self, request: Request, index: int) -> BlockHash:
        return request.block_hashes[(index + 1) * self._hashes_per_object - 1]

    def max_cache_hit(self, request: Request, session: str) -> int:
        """Tokens of the prefix whose cached KV `session` holds the rows of."""
        if self._expired(session, time.monotonic()):
            self._forget([session])
        published = self._published.get(session, {})
        num_objects = 0
        num_hashed = len(request.block_hashes) // self._hashes_per_object
        for index in range(num_hashed):
            block_hash = self._block_hash(request, index)
            block_id = self._cached_block(block_hash)
            identity = published.get(bytes(block_hash))
            if (
                identity is None
                or block_id is None
                or identity != self._block_ids.get(block_id)
            ):
                break
            num_objects += 1
        last = (request.num_tokens - 1) // self._object_size
        return min(num_objects, last) * self._object_size

    def schedule(
        self, request: Request, session: str | None, end: int
    ) -> tuple[int, list[str]] | None:
        """Admit or resume `request` and name the objects it computes up to
        position `end`.

        Returns, for a session request, the identities the worker has not
        received yet as `(first index, identities)`.
        """
        objects = self._requests.get(request.request_id)
        computed = request.num_computed_tokens
        if objects is None:
            objects = self._admit(request, session)
        elif computed < objects.next_object * self._object_size:
            objects.next_object = computed // self._object_size
            objects.published = min(objects.published, objects.next_object)
            objects.worker_sent = objects.next_object
        num_hashed = len(request.block_hashes) // self._hashes_per_object
        end_object = min(num_hashed, end // self._object_size)
        for index in range(objects.next_object, end_object):
            identity = f"{self._id_prefix}{next(self._next_id):x}"
            block_id = self._request_block(request, index)
            if block_id is not None:
                self._block_ids[block_id] = identity
            if session is not None:
                objects.ids[index] = identity
        objects.next_object = max(objects.next_object, end_object)
        if session is None:
            return None
        first = objects.worker_sent
        objects.worker_sent = objects.next_object
        return first, [objects.ids[i] for i in range(first, objects.next_object)]

    def _admit(self, request: Request, session: str | None) -> _RequestObjects:
        computed = request.num_computed_tokens
        first = -(-computed // self._object_size)
        objects = _RequestObjects(session, computed, first, published=first)
        self._requests[request.request_id] = objects
        if session is not None:
            published = self._published[session] if first else {}
            for index in range(first):
                objects.ids[index] = published[bytes(self._block_hash(request, index))]
            objects.worker_sent = first
            objects.tail_start = computed
            self._session_requests.setdefault(session, set()).add(request.request_id)
        return objects

    def emit_start(self, request: Request) -> int:
        """First position whose rows a session request returns."""
        if request.num_output_tokens:
            return request.num_tokens - 1
        return self._requests[request.request_id].emit_start

    def take_output(
        self, request: Request, token_start: int, rows: np.ndarray
    ) -> np.ndarray:
        """Publish the objects that the committed rows of a session request
        complete.

        Returns the rows to deliver; `take_keys` then returns the keys of the
        objects completed since the last output. Rows follow the last listed
        object, and an output that lists keys carries every row after them.
        """
        objects = self._requests[request.request_id]
        session = objects.session
        assert session is not None
        if objects.tail is None:
            objects.tail = rows[:0]
            objects.tail_start = token_start
        assert token_start == objects.tail_start + len(objects.tail), (
            "auxiliary output rows of a session request are not contiguous"
        )
        tail = np.concatenate((objects.tail, rows))
        token_end = token_start + len(rows)
        num_objects = min(token_end // self._object_size, objects.next_object)
        published = self._published.setdefault(session, {})
        for index in range(objects.published, num_objects):
            block_hash = self._block_hash(request, index)
            if self._cached_block(block_hash) == self._request_block(request, index):
                published[bytes(block_hash)] = objects.ids[index]
        objects.published = max(objects.published, num_objects)
        keys = [
            aux_object_key(
                self._key_prefix,
                session,
                bytes(self._block_hash(request, index)).hex(),
                objects.ids[index],
            )
            for index in range(objects.keys_sent, num_objects)
        ]
        objects.keys_sent = max(objects.keys_sent, num_objects)
        boundary = objects.keys_sent * self._object_size
        if boundary > objects.tail_start:
            tail = tail[boundary - objects.tail_start :]
            objects.tail_start = boundary
        objects.tail = tail
        objects.keys = keys or None
        return tail if keys else rows

    def take_keys(self, request: Request) -> list[str] | None:
        objects = self._requests[request.request_id]
        keys, objects.keys = objects.keys, None
        return keys

    def finish(self, request: Request) -> None:
        if not request.is_finished():
            return
        objects = self._requests.pop(request.request_id, None)
        if objects is None or objects.session is None:
            return
        session_requests = self._session_requests.get(objects.session)
        if session_requests is not None:
            session_requests.discard(request.request_id)
        self._last_used[objects.session] = time.monotonic()

    def take_released(self) -> tuple[str, ...]:
        """Sessions forgotten since the last call, after expiring idle ones."""
        now = time.monotonic()
        if now >= self._next_sweep:
            self._next_sweep = now + 1.0
            self._forget([s for s in self._last_used if self._expired(s, now)])
        released = tuple(self._released)
        self._released.clear()
        return released

    def reset(self) -> None:
        """Forget every session after a prefix-cache reset."""
        self._forget(list(set(self._published) | set(self._last_used)))
        self._block_ids.clear()

    def _expired(self, session: str, now: float) -> bool:
        last_used = self._last_used.get(session)
        return (
            last_used is not None
            and last_used < now - self._idle_timeout_s
            and not self._session_requests.get(session)
        )

    def _forget(self, sessions: list[str]) -> None:
        for session in sessions:
            self._last_used.pop(session, None)
            self._session_requests.pop(session, None)
            self._published.pop(session, None)
            self._released.append(session)
