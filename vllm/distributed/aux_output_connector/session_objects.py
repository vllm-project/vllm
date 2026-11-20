# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write auxiliary outputs of session requests to a shared object store."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from safetensors.numpy import save

from vllm.distributed.aux_output_connector.object_store import (
    _NOT_FOUND,
    MooncakeAuxObjectStore,
)
from vllm.distributed.aux_output_connector.store import BlockObjectStoreError
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.utils import length_from_prompt_token_ids_or_embeds

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.aux_output_connector.worker import _WorkerRequestState
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


def _valid_version_segments(segments: object) -> bool:
    if not isinstance(segments, (list, tuple)) or not segments:
        return False
    previous: tuple[object, object] = (-1, None)
    for index, segment in enumerate(segments):
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            return False
        start, version = segment
        last = index == len(segments) - 1
        if (
            type(start) is not int
            or start <= previous[0]
            or (index == 0 and start != 0)
            or (version is None) != last
            or not (last or (isinstance(version, str) and version))
            or (index and version == previous[1])
        ):
            return False
        previous = (start, version)
    return True


def validate_aux_request(
    vllm_config: VllmConfig, sampling_params: SamplingParams, resumable: bool
) -> None:
    """Check the auxiliary-output parameters of a request."""
    session = sampling_params.aux_session
    segments = sampling_params.aux_version_segments
    held = sampling_params.aux_held
    if session is None:
        if segments is not None or held is not None:
            raise VLLMValidationError(
                "aux_version_segments or aux_held requires aux_session.",
                parameter="aux_version_segments",
            )
        return
    if held is not None and not (isinstance(held, int) and held >= 0):
        raise VLLMValidationError(
            "aux_held must be a non-negative integer.", parameter="aux_held"
        )
    if vllm_config.aux_output_config.object_key_prefix is None:
        reason = "requires AuxOutputConfig.object_key_prefix"
    elif not session or "/" in session or session == "tail":
        reason = "must be a non-empty string without '/', other than 'tail'"
    elif sampling_params.stop:
        reason = "does not support stop strings; use stop_token_ids"
    elif resumable:
        reason = "does not support streaming input"
    elif segments is not None and not _valid_version_segments(segments):
        raise VLLMValidationError(
            "aux_version_segments must be [start, version] pairs whose starts "
            "increase from 0, whose adjacent versions differ and whose last pair, "
            "and only that one, has a null version.",
            parameter="aux_version_segments",
        )
    else:
        return
    raise VLLMValidationError(f"aux_session {reason}.", parameter="aux_session")


_SESSION_SALT = "aux_session:"
_PLAIN_SALT = "aux_salt:"


def aux_segment_id(request_id: str, weight_version: str) -> str:
    """Name of the positions a request computes while `weight_version` is
    current."""
    return f"{request_id}@{weight_version}"


def session_version_segments(
    client_segments: Sequence[Sequence] | None,
    num_prompt_tokens: int,
    block_size: int,
    segment_id: str,
) -> tuple[int, list[tuple[int, str]]]:
    """The end of the prompt positions a session request's client
    `aux_version_segments` salt, rounded down to `block_size` before the last
    prompt token, and the cache salt segments: the client's segments below
    it, then `segment_id`."""
    end = min(client_segments[-1][0], num_prompt_tokens - 1) if client_segments else 0
    end = end // block_size * block_size
    segments = [
        (start, version) for start, version in client_segments or () if start < end
    ]
    segments.append((end, segment_id))
    return end, segments


def salt_aux_session(
    vllm_config: VllmConfig,
    request: EngineCoreRequest,
    weight_version: str,
    hash_block_size: int,
) -> None:
    """Fold a request's `aux_session` into its cache salt, so that only
    requests of the session compute or hit its KV blocks, and salt its
    positions with the segments that name them."""
    if vllm_config.aux_output_config.object_key_prefix is None:
        return
    params = request.sampling_params
    salt = request.cache_salt
    if params is not None and params.aux_session is not None:
        prefix = vllm_config.aux_output_config.object_key_prefix
        salt = _SESSION_SALT + json.dumps([prefix, params.aux_session, salt])
        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )
        _, request.cache_salt_segments = session_version_segments(
            params.aux_version_segments,
            num_prompt_tokens,
            hash_block_size,
            aux_segment_id(request.request_id, weight_version),
        )
    elif salt is not None and salt.startswith((_SESSION_SALT, _PLAIN_SALT)):
        salt = _PLAIN_SALT + salt
    else:
        return
    request.cache_salt = salt
    request.block_hashes = None


def aux_owned_keys(
    request_id: str,
    keys: Sequence[str],
    segments: Sequence[Sequence],
) -> list[str]:
    """The keys of a session response whose objects its request `request_id`
    wrote: the full objects from its first own segment on, and the tail."""
    full = [key for key in keys if key.rsplit("/", 2)[-2] != "tail"]
    end = segments[-1][0]
    start = next(
        (
            begin
            for begin, salt in segments
            if salt and salt.rpartition("@")[0] == request_id
        ),
        end,
    )
    first = -(-start * len(full) // end) if full else 0
    return full[first:] + list(keys[len(full) :])


def aux_object_key(key_prefix: str, session: str, block_hash_hex: str) -> str:
    """Key of the object that ends with the hash block `block_hash_hex`."""
    return f"{key_prefix}/{session}/{block_hash_hex}"


def aux_tail_key(key_prefix: str, session: str, tail_hash_hex: str) -> str:
    """Key of the object of the rows after a request's last full object."""
    return f"{key_prefix}/{session}/tail/{tail_hash_hex}"


_PUT_RETRY_INTERVAL_S = 0.2


class SessionObjectWriter:
    """Write the objects of session requests in the background, one array per
    output field. A put the store refuses is retried every
    `_PUT_RETRY_INTERVAL_S`, without the objects already written, and queuing
    and draining wait while it is. Once writes have been pending for
    `put_stall_timeout_s` without a successful put, the rest of the batch is
    dropped, and later batches get one attempt each until a put succeeds. A
    dropped object is absent from the store; the response that lists it is not
    changed. While batches are dropped, or no put succeeded for
    `put_stall_timeout_s`, queuing does not wait: a put that meets a full queue
    is dropped, and a removal is sent after the batches queued before it. Once
    a store call has not returned for `put_stall_timeout_s`, draining stops
    waiting for it and drops the puts queued behind it."""

    def __init__(
        self,
        store: MooncakeAuxObjectStore,
        *,
        key_prefix: str,
        max_pending_batches: int,
        fields: Callable[[np.ndarray], dict[str, np.ndarray]],
        put_stall_timeout_s: float = 600.0,
    ) -> None:
        self._store = store
        self._unremoved: list[str] = []
        self._deferred: deque[tuple[int, list[str]]] = deque()
        self._key_prefix = key_prefix
        self._fields = fields
        self._put_stall_timeout_s = put_stall_timeout_s
        self._last_progress = time.monotonic()
        self._busy_since: float | None = None
        self._dropped = 0
        self._enqueued = 0
        self._taken = 0
        self._abandon_upto = 0
        self._closed = threading.Event()
        self._queue: queue.Queue[tuple[list[str], list[bytes] | None] | None] = (
            queue.Queue(maxsize=max_pending_batches)
        )
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vllm-aux-object-writer"
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            batch = self._queue.get()
            if batch is None:
                self._queue.task_done()
                return
            self._taken += 1
            try:
                if self._error is None:
                    self._write(batch)
            except BaseException as error:
                self._fail(error)
                logger.exception("Auxiliary output object write failed.")
            finally:
                self._queue.task_done()

    def _fail(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error

    def _stalled(self) -> bool:
        return time.monotonic() - self._last_progress >= self._put_stall_timeout_s

    def _hung(self) -> bool:
        busy_since = self._busy_since
        return (
            busy_since is not None
            and time.monotonic() - busy_since >= self._put_stall_timeout_s
        )

    def _call(self, method: Callable, *args):
        self._busy_since = time.monotonic()
        try:
            return method(*args)
        finally:
            self._busy_since = None

    def _drop(self, keys: list[str], reason: object) -> None:
        if not self._dropped:
            logger.error(
                "Dropping auxiliary output writes from %s on: %s", keys[:1], reason
            )
        self._dropped += len(keys)

    def _write(self, batch: tuple[list[str], list[bytes] | None]) -> None:
        keys, values = batch
        if values is not None and self._deferred and self._deferred[0][0] < self._taken:
            self._remove([])
        if values is None:
            self._remove(keys)
            if self._unremoved:
                logger.warning(
                    "%d auxiliary output objects are not removed yet; first: %s",
                    len(self._unremoved),
                    self._unremoved[0],
                )
            self._last_progress = time.monotonic()
            return
        attempts = 0
        while keys:
            if self._taken <= self._abandon_upto:
                self._drop(keys, "a store call hung while they were queued")
                return
            try:
                self._call(self._store.put_batch, keys[:1], values[:1])
            except BlockObjectStoreError as error:
                attempts += 1
                if attempts & (attempts - 1) == 0 or attempts % 32 == 0:
                    self._remove([])
                if (
                    self._dropped
                    or self._stalled()
                    or self._closed.wait(_PUT_RETRY_INTERVAL_S)
                ):
                    self._drop(keys, error)
                    return
            else:
                keys, values = keys[1:], values[1:]
                attempts = 0
                self._last_progress = time.monotonic()
                if self._dropped:
                    logger.warning(
                        "Auxiliary output objects are written again; %d were dropped.",
                        self._dropped,
                    )
                    self._dropped = 0

    def _remove(self, keys: list[str]) -> None:
        deferred = []
        while self._deferred and self._deferred[0][0] < self._taken:
            deferred += self._deferred.popleft()[1]
        keys = self._unremoved + deferred + keys
        codes = self._call(self._store.remove, keys) if keys else []
        self._unremoved = [
            key
            for key, code in zip(keys, codes, strict=True)
            if code and code != _NOT_FOUND
        ]

    def check(self) -> None:
        """Raise `BlockObjectStoreError` if writing failed other than in the
        store."""
        if self._error is not None:
            raise BlockObjectStoreError(
                "auxiliary output object write failed"
            ) from self._error

    def _put(self, batch: tuple[list[str], list[bytes] | None]) -> None:
        if not self._queue.unfinished_tasks:
            self._last_progress = time.monotonic()
        while True:
            wait = not (self._dropped or self._stalled() or self._hung())
            try:
                self._queue.put(batch, wait, _PUT_RETRY_INTERVAL_S)
                self._enqueued += 1
                return
            except queue.Full:
                self.check()
                if not wait:
                    keys, values = batch
                    if values is None:
                        self._deferred.append((self._enqueued, keys))
                    else:
                        self._drop(keys, "the queue is full while the store fails")
                    return

    def drain(self, timeout_s: float | None = None) -> None:
        """Wait until the queued writes are written or dropped, or a store call
        hangs; raise `BlockObjectStoreError` if writing failed other than in
        the store, or `timeout_s` passed."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        if self._unremoved or self._deferred:
            self._put(([], None))
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks and self._error is None:
                if self._hung():
                    self._abandon_upto = self._enqueued
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise BlockObjectStoreError(
                        "auxiliary output objects were not written in time"
                    )
                self._queue.all_tasks_done.wait(_PUT_RETRY_INTERVAL_S)
        self.check()

    def add_blocks(
        self,
        state: _WorkerRequestState,
        blocks: Sequence[tuple[int, np.ndarray]],
        object_size: int,
    ) -> None:
        """Stage the next full hash blocks of a request; write completed objects."""
        keys: list[str] = []
        values: list[bytes] = []
        pieces = list(state.staged)
        first = 0
        for index, (start, rows) in enumerate(blocks):
            pieces.append((start, rows))
            end = start + len(rows)
            if end % object_size:
                continue
            assert state.session is not None
            low = max(end - object_size, state.write_from)
            if end <= low:
                self._drop_fields(state, end)
            else:
                key = aux_object_key(
                    self._key_prefix,
                    state.session,
                    state.block_hashes[end // len(rows) - 1 - state.hash_start].hex(),
                )
                keys.append(key)
                fields = self._take_fields(state, low, end)
                data = np.concatenate(
                    [
                        piece[max(low - begin, 0) :]
                        for begin, piece in pieces
                        if begin + len(piece) > low
                    ]
                )
                values.append(save({**self._fields(data), **fields}))
            pieces = []
            first = index + 1
            state.staged.clear()
        state.staged.extend((start, np.array(rows)) for start, rows in blocks[first:])
        if keys:
            self._put((keys, values))

    def add_tail(
        self, key: str, rows: np.ndarray, fields: dict[str, np.ndarray]
    ) -> None:
        """Write the rows after a request's last full object."""
        fields = {**self._fields(np.array(rows)), **fields}
        self._put(([key], [save(fields)]))

    def remove(self, keys: Sequence[str]) -> None:
        """Remove objects of a request that it never returned, after the
        writes queued before."""
        self._put((list(keys), None))

    @staticmethod
    def field_rows(
        state: _WorkerRequestState, start: int, end: int
    ) -> dict[str, np.ndarray]:
        """Rows `[start, end)` of the staged fields of a request."""
        fields = {}
        for name, (first, stop, chunks) in state.fields.items():
            assert first <= start and stop >= end, (
                f"{name} rows of positions [{start}, {end}) are incomplete"
            )
            pieces = []
            position = first
            for chunk in chunks:
                if position >= end:
                    break
                if position + len(chunk) > start:
                    pieces.append(chunk[max(start - position, 0) : end - position])
                position += len(chunk)
            fields[name] = np.concatenate(pieces)
        return fields

    @staticmethod
    def _drop_fields(state: _WorkerRequestState, end: int) -> None:
        """Drop the staged field rows before `end`. A remainder of less than
        half the buffer it views is copied so that it does not keep that
        buffer alive."""
        for name, (position, stop, chunks) in state.fields.items():
            while chunks and position + len(chunks[0]) <= end:
                position += len(chunks.pop(0))
            if chunks and position < end:
                rest = chunks[0][end - position :]
                buffer = rest if rest.base is None else rest.base
                chunks[0] = rest.copy() if 2 * rest.nbytes < buffer.nbytes else rest
            state.fields[name] = end, stop, chunks

    @staticmethod
    def _take_fields(
        state: _WorkerRequestState, start: int, end: int
    ) -> dict[str, np.ndarray]:
        """Rows `[start, end)` of the staged fields; later rows stay staged."""
        fields = SessionObjectWriter.field_rows(state, start, end)
        SessionObjectWriter._drop_fields(state, end)
        return fields

    def close(self) -> None:
        self._closed.set()
        self._queue.put(None)
        self._thread.join()
        self._store.close()
