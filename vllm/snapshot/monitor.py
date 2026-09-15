# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Callable
from types import TracebackType
from typing import Protocol


class _Event(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...

    def clear(self) -> None: ...


class _Lock(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class SnapshotMonitor:
    """Shared snapshot lifecycle gates and completion states.

    The ``is_*ing`` events are execution gates that prevent duplicate work.
    The ``*_done`` events record successful completion for later requests and
    snapshot health checks. Factories allow the same state machine to use
    thread primitives in one process or multiprocessing primitives when API
    workers need to share it.
    """

    def __init__(
        self,
        event_factory: Callable[[], _Event] = threading.Event,
        lock_factory: Callable[[], _Lock] = threading.Lock,
    ) -> None:
        # The lock makes each gate check-and-set operation atomic across all
        # threads or processes sharing this monitor.
        self._lock = lock_factory()
        self._is_suspending = event_factory()
        self._is_unlocking = event_factory()
        self._is_resuming = event_factory()
        self._suspend_done = event_factory()
        self._unlock_done = event_factory()
        self._resume_done = event_factory()

    @property
    def is_suspending(self) -> bool:
        return self._is_suspending.is_set()

    @property
    def is_unlocking(self) -> bool:
        return self._is_unlocking.is_set()

    @property
    def is_resuming(self) -> bool:
        return self._is_resuming.is_set()

    @property
    def is_suspend_done(self) -> bool:
        return self._suspend_done.is_set()

    @property
    def is_unlock_done(self) -> bool:
        return self._unlock_done.is_set()

    @property
    def is_resume_done(self) -> bool:
        return self._resume_done.is_set()

    def try_start_suspending(self) -> bool:
        with self._lock:
            if self._is_suspending.is_set() or self._suspend_done.is_set():
                return False
            self._is_suspending.set()
            return True

    def mark_suspend_done(self) -> None:
        with self._lock:
            self._suspend_done.set()
            self._is_suspending.clear()

    def mark_suspend_failed(self) -> None:
        with self._lock:
            self._is_suspending.clear()

    def try_start_unlocking(self) -> bool:
        with self._lock:
            if (
                not self._suspend_done.is_set()
                or self._is_unlocking.is_set()
                or self._unlock_done.is_set()
                or self._is_resuming.is_set()
                or self._resume_done.is_set()
            ):
                return False
            self._is_unlocking.set()
            return True

    def mark_unlock_done(self) -> None:
        with self._lock:
            self._unlock_done.set()
            self._is_unlocking.clear()

    def mark_unlock_failed(self) -> None:
        with self._lock:
            self._is_unlocking.clear()

    def try_start_resuming(self) -> bool:
        with self._lock:
            if (
                not self._suspend_done.is_set()
                or self._is_unlocking.is_set()
                or self._is_resuming.is_set()
                or self._resume_done.is_set()
            ):
                return False
            self._is_resuming.set()
            return True

    def mark_resume_done(self) -> None:
        with self._lock:
            self._resume_done.set()
            self._is_resuming.clear()

    def mark_resume_failed(self) -> None:
        with self._lock:
            self._is_resuming.clear()
