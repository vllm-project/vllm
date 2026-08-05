# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend boundary for immutable execution-artifact objects."""

import queue
import threading
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ArtifactObject:
    """One immutable, self-describing object."""

    key: str
    payload: bytes


class ArtifactStoreError(RuntimeError):
    """Base class for artifact-store failures."""


class ArtifactCapacityError(ArtifactStoreError):
    """The artifact store cannot retain another object."""


class ArtifactCorruptionError(ArtifactStoreError):
    """An artifact object failed structural or checksum validation."""


class ArtifactNotFoundError(ArtifactStoreError):
    """A requested artifact object is not present."""


class ArtifactReader(Protocol):
    """Opaque byte-object reads used to materialize terminal artifacts."""

    def get(self, keys: list[str]) -> list[bytes]: ...

    def close(self) -> None: ...


class ArtifactStore(ArtifactReader, Protocol):
    """Artifact reader that can publish immutable objects."""

    def put(self, objects: list[ArtifactObject]) -> None: ...


class BackgroundArtifactStore:
    """Publish objects off the scheduler thread while preserving read order."""

    def __init__(self, store: ArtifactStore, *, max_pending_batches: int) -> None:
        if max_pending_batches <= 0:
            raise ValueError("max_pending_batches must be positive")
        self._store = store
        self._queue: queue.Queue[list[ArtifactObject] | None] = queue.Queue(
            maxsize=max_pending_batches
        )
        self._error: BaseException | None = None
        self._closed = False
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="vllm-artifact-writer",
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            objects = self._queue.get()
            num_batches = 1
            try:
                if objects is None:
                    return
                pending = objects
                while True:
                    try:
                        next_objects = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    assert next_objects is not None
                    pending.extend(next_objects)
                    num_batches += 1
                if self._error is None:
                    self._store.put(pending)
            except BaseException as error:
                self._error = error
            finally:
                for _ in range(num_batches):
                    self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise ArtifactStoreError("artifact publication failed") from self._error

    def put(self, objects: list[ArtifactObject]) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("artifact store is closed")
            self._raise_if_failed()
            if objects:
                self._queue.put(list(objects))
                self._raise_if_failed()

    def get(self, keys: list[str]) -> list[bytes]:
        # Internal reads must observe publications issued earlier by the same
        # connector. Independent readers still see not-found until publication.
        self._queue.join()
        self._raise_if_failed()
        return self._store.get(keys)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.join()
            self._queue.put(None)
        self._thread.join()
        try:
            self._raise_if_failed()
        finally:
            self._store.close()
