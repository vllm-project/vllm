# SPDX-License-Identifier: Apache-2.0
"""Cross-platform event notification abstraction.

Provides a unified ``EventNotifier`` interface for signaling between
threads using poll-able file descriptors.  On Linux (Python 3.10+),
uses ``os.eventfd``; on other platforms (macOS, etc.), falls back to
``os.pipe`` with non-blocking I/O.

Usage::

    from lmcache.v1.platform import create_event_notifier

    notifier = create_event_notifier()
    notifier.notify()          # signal
    notifier.consume()         # drain
    fd = notifier.fileno()     # for select.poll()
    notifier.close()           # release resources

Or as a context manager::

    with create_event_notifier() as notifier:
        notifier.notify()
        notifier.consume()

Exception contract:
    ``consume()`` / ``consume_fd()`` only swallow
    :class:`BlockingIOError` (i.e. "no data pending").  All other
    :class:`OSError` subclasses - most importantly ``EBADF`` from a
    stale / closed fd - propagate to the caller so bugs surface
    instead of being silently absorbed.
"""

# Standard
from abc import ABC, abstractmethod
from types import TracebackType
import os

#: True when :func:`os.eventfd` is available (Linux + Python 3.10+).
#: Single source of truth for platform capability detection; exposed
#: via :mod:`lmcache.v1.platform` for external callers.
HAS_EVENTFD: bool = hasattr(os, "eventfd")


class EventNotifier(ABC):
    """Abstract base class for cross-platform event notification.

    An ``EventNotifier`` models a **binary signal**: calling
    ``notify()`` makes the notifier readable via ``poll()``/
    ``select()``, and ``consume()`` resets it.  Multiple
    ``notify()`` calls before a ``consume()`` are coalesced.
    """

    @abstractmethod
    def fileno(self) -> int:
        """Return a poll-able file descriptor.

        The fd becomes readable after ``notify()`` is called.
        """

    @abstractmethod
    def notify(self) -> None:
        """Signal the notifier (idempotent if already signaled)."""

    @abstractmethod
    def consume(self) -> None:
        """Consume the pending signal (non-blocking).

        Only :class:`BlockingIOError` (no data pending) is
        swallowed; other OS errors propagate.
        """

    @abstractmethod
    def notify_fileno(self) -> int:
        """Return the fd to write to for signaling.

        For eventfd this is the same as ``fileno()``.  For pipes
        this is the *write* end (``fileno()`` returns the read end).
        """

    @abstractmethod
    def close(self) -> None:
        """Release underlying OS resources.  Idempotent."""

    # Context manager support

    def __enter__(self) -> "EventNotifier":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


if HAS_EVENTFD:

    class EventfdNotifier(EventNotifier):
        """Linux eventfd-based notifier (Python 3.10+).

        Invariants:
            ``self._efd`` holds the live eventfd while open and is
            set to ``-1`` once :meth:`close` has released it.  All
            other methods treat ``-1`` as "already closed".
        """

        def __init__(self) -> None:
            self._efd: int = os.eventfd(
                0,
                os.EFD_NONBLOCK | os.EFD_CLOEXEC,
            )

        def fileno(self) -> int:
            return self._efd

        def notify(self) -> None:
            os.eventfd_write(self._efd, 1)

        def notify_fileno(self) -> int:
            return self._efd

        def consume(self) -> None:
            try:
                os.eventfd_read(self._efd)
            except BlockingIOError:
                pass  # no pending signal

        def close(self) -> None:
            if self._efd >= 0:
                try:
                    os.close(self._efd)
                except OSError:
                    pass
                self._efd = -1
else:

    class EventfdNotifier(EventNotifier):  # type: ignore[no-redef]
        """Placeholder - eventfd unavailable on this platform."""

        def __init__(self) -> None:
            raise RuntimeError(
                "EventfdNotifier requires os.eventfd (Linux + "
                "Python 3.10+); use PipeNotifier instead."
            )

        def fileno(self) -> int:  # pragma: no cover
            raise RuntimeError("unreachable")

        def notify(self) -> None:  # pragma: no cover
            raise RuntimeError("unreachable")

        def notify_fileno(self) -> int:  # pragma: no cover
            raise RuntimeError("unreachable")

        def consume(self) -> None:  # pragma: no cover
            raise RuntimeError("unreachable")

        def close(self) -> None:  # pragma: no cover
            raise RuntimeError("unreachable")


class PipeNotifier(EventNotifier):
    """Pipe-based fallback notifier for non-Linux platforms.

    Invariants:
        ``self._read_fd`` / ``self._write_fd`` hold the live pipe
        ends while open and are both set to ``-1`` once
        :meth:`close` has released them.
    """

    def __init__(self) -> None:
        # ``os.pipe()`` on Python 3.4+ creates descriptors with
        # ``FD_CLOEXEC`` already set (PEP 446), so we only need to
        # flip them to non-blocking.  ``os.set_blocking`` is the
        # portable way to do this and improves future portability
        # (e.g. to Windows pipes).
        r, w = os.pipe()
        try:
            os.set_blocking(r, False)
            os.set_blocking(w, False)
        except OSError:
            os.close(r)
            os.close(w)
            raise
        self._read_fd: int = r
        self._write_fd: int = w

    def fileno(self) -> int:
        return self._read_fd

    def notify(self) -> None:
        try:
            os.write(self._write_fd, b"\x01")
        except BlockingIOError:
            pass  # pipe buffer full - signal already pending

    def notify_fileno(self) -> int:
        return self._write_fd

    def consume(self) -> None:
        while True:
            try:
                # ``os.read`` returns ``b''`` at EOF (write end
                # closed); treat that as drained so background
                # loops cannot spin forever on a dead pipe.
                if not os.read(self._read_fd, 4096):
                    return
            except BlockingIOError:
                return  # drained

    def close(self) -> None:
        for attr in ("_read_fd", "_write_fd"):
            fd = getattr(self, attr, -1)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attr, -1)


def create_event_notifier() -> EventNotifier:
    """Create a platform-appropriate EventNotifier.

    On Linux (Python 3.10+), returns an ``EventfdNotifier``.
    On other platforms, returns a ``PipeNotifier``.
    """
    if HAS_EVENTFD:
        return EventfdNotifier()
    return PipeNotifier()


def consume_fd(fd: int) -> None:
    """Consume a pending signal from a raw file descriptor.

    This is a convenience function for code that only has a
    raw fd (e.g., obtained from ``adapter.get_store_event_fd()``)
    and needs to drain it after ``poll()`` reports it readable.

    On Linux, uses ``os.eventfd_read()``; on other platforms,
    drains all bytes via ``os.read()``.

    Only :class:`BlockingIOError` (no data pending) is swallowed;
    other OS errors propagate.
    """
    if HAS_EVENTFD:
        try:
            os.eventfd_read(fd)
        except BlockingIOError:
            pass
        return
    while True:
        try:
            if not os.read(fd, 4096):
                return  # EOF -- write end closed; stop draining
        except BlockingIOError:
            return
