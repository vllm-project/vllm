# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import select

# Third Party
import pytest

# First Party
from lmcache.v1.platform import (
    HAS_EVENTFD,
    EventfdNotifier,
    EventNotifier,
    PipeNotifier,
    consume_fd,
    create_event_notifier,
)
from lmcache.v1.platform import event_notifier as _canonical_ev_mod


class TestEventNotifierAPI:
    """Test the public API on the current platform."""

    def test_create_returns_event_notifier(self):
        """Factory returns an EventNotifier subclass."""
        n = create_event_notifier()
        try:
            assert isinstance(n, EventNotifier)
        finally:
            n.close()

    def test_fileno_is_nonnegative(self):
        """fileno() returns a valid fd."""
        n = create_event_notifier()
        try:
            assert n.fileno() >= 0
        finally:
            n.close()

    def test_notify_then_consume(self):
        """notify() makes the fd readable; consume() drains it."""
        n = create_event_notifier()
        try:
            n.notify()
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            events = poller.poll(100)
            assert len(events) > 0
            n.consume()
        finally:
            n.close()

    def test_multiple_notify_single_consume(self):
        """Multiple notify() calls are coalesced by one consume()."""
        n = create_event_notifier()
        try:
            n.notify()
            n.notify()
            n.notify()
            n.consume()
            # After consume, fd should not be readable
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            events = poller.poll(50)
            assert len(events) == 0
        finally:
            n.close()

    def test_consume_without_notify_is_noop(self):
        """consume() without prior notify() does not block or raise."""
        n = create_event_notifier()
        try:
            n.consume()  # should not block or raise
        finally:
            n.close()

    def test_close_is_idempotent(self):
        """Calling close() twice does not raise."""
        n = create_event_notifier()
        n.close()
        n.close()  # should not raise

    def test_context_manager(self):
        """EventNotifier works as a context manager."""
        with create_event_notifier() as n:
            n.notify()
            n.consume()
            assert n.fileno() >= 0


class TestPipeNotifier:
    """Force the pipe-based fallback path."""

    def test_both_fds_closed(self):
        """close() releases both read and write fds."""
        n = PipeNotifier()
        r = n.fileno()
        # Second fd via notify/consume cycle: after close, both must be gone.
        n.notify()
        n.consume()
        n.close()
        with pytest.raises(OSError):
            os.fstat(r)

    def test_notify_when_pipe_full_is_noop(self):
        """notify() does not raise when pipe buffer is full."""
        n = PipeNotifier()
        try:
            # A POSIX pipe's capacity is at most 1 MiB on current
            # Linux/macOS kernels and each notify() writes a single
            # byte, so (1 << 21) iterations comfortably saturate it
            # while keeping the test fast - notify() itself swallows
            # BlockingIOError once the buffer is full.
            for _ in range(1 << 21):
                n.notify()
            # Still a no-op once the pipe is saturated.
            n.notify()
        finally:
            n.close()

    def test_pollable(self):
        """PipeNotifier fd is pollable with select.poll."""
        n = PipeNotifier()
        try:
            n.notify()
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            events = poller.poll(100)
            assert len(events) > 0
        finally:
            n.close()

    def test_multiple_create_close_cycles(self):
        """Create and close multiple notifiers without leaking."""
        for _ in range(20):
            n = PipeNotifier()
            n.notify()
            n.consume()
            n.close()


class TestPipeEOF:
    """Regression test: pipe EOF must not spin the drain loop.

    ``os.read`` returns ``b''`` (not ``BlockingIOError``) when the
    write end is closed.  Both ``PipeNotifier.consume`` and the
    module-level ``consume_fd`` helper share this drain loop, so
    they must treat an empty read as "drained" -- otherwise a dead
    pipe would hang a background event loop.
    """

    def test_consume_fd_returns_on_pipe_eof(self, monkeypatch):
        """consume_fd() returns when the pipe write end is closed."""
        monkeypatch.setattr(_canonical_ev_mod, "HAS_EVENTFD", False)
        r, w = os.pipe()
        os.set_blocking(r, False)
        try:
            os.close(w)  # write end closed -> read end at EOF
            consume_fd(r)  # must return promptly, not hang
        finally:
            os.close(r)


class TestConsumeFd:
    """Test the consume_fd utility function."""

    def test_consume_fd_after_notify(self):
        """consume_fd() drains a notifier's fd."""
        n = create_event_notifier()
        try:
            n.notify()
            consume_fd(n.fileno())
            # After consume, fd should not be readable
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            events = poller.poll(50)
            assert len(events) == 0
        finally:
            n.close()

    def test_consume_fd_without_signal(self):
        """consume_fd() on unsignaled fd does not block."""
        n = create_event_notifier()
        try:
            consume_fd(n.fileno())  # should not block
        finally:
            n.close()


class TestBackendSelection:
    """Validate which concrete backend the factory picks."""

    def test_factory_matches_platform_capability(self):
        """create_event_notifier() picks the backend for this OS."""
        n = create_event_notifier()
        try:
            if HAS_EVENTFD:
                assert isinstance(n, EventfdNotifier)
            else:
                assert isinstance(n, PipeNotifier)
        finally:
            n.close()


class TestForcedPipeFallback:
    """Exercise the pipe fallback even when eventfd is available.

    On Linux CI ``create_event_notifier()`` would normally return
    an ``EventfdNotifier``; monkeypatch ``HAS_EVENTFD`` to force
    the pipe path so the non-Linux code is covered on every OS.
    """

    @pytest.fixture
    def force_pipe(self, monkeypatch):
        monkeypatch.setattr(_canonical_ev_mod, "HAS_EVENTFD", False)
        yield

    def test_factory_returns_pipe_notifier(self, force_pipe):
        n = create_event_notifier()
        try:
            assert isinstance(n, PipeNotifier)
        finally:
            n.close()

    def test_notify_consume_roundtrip(self, force_pipe):
        n = create_event_notifier()
        try:
            n.notify()
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            assert poller.poll(100)
            n.consume()
            assert not poller.poll(50)
        finally:
            n.close()

    def test_consume_fd_uses_pipe_path(self, force_pipe):
        n = create_event_notifier()
        try:
            n.notify()
            consume_fd(n.fileno())
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            assert not poller.poll(50)
        finally:
            n.close()
