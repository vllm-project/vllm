# SPDX-License-Identifier: Apache-2.0

# Standard
import os
import select
import time

# Third Party
import pytest

pytest.importorskip(
    "lmcache.native_storage_ops",
    reason="native_storage_ops extension not built",
)

# First Party
from lmcache.native_storage_ops import PeriodicEventNotifier
from lmcache.v1.platform import HAS_EVENTFD


@pytest.fixture(autouse=True)
def cleanup():
    yield
    PeriodicEventNotifier.shutdown()


def _make_notifier_fd():
    """Create a test fd suitable for the periodic notifier to write to.

    Returns (poll_fd, write_fd, close_func) where poll_fd is for
    select.poll() and write_fd is what gets registered with the notifier.
    For eventfd they are the same; for pipes they differ.
    """
    if HAS_EVENTFD:
        fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        return fd, fd, lambda: os.close(fd)
    else:
        r, w = os.pipe()
        os.set_blocking(r, False)
        os.set_blocking(w, False)

        def close_both():
            os.close(r)
            os.close(w)

        return r, w, close_both


def _consume_fd(fd):
    if HAS_EVENTFD:
        try:
            os.eventfd_read(fd)
        except BlockingIOError:
            pass
    else:
        while True:
            try:
                if not os.read(fd, 4096):
                    return
            except BlockingIOError:
                return


def _poll_fd(fd, timeout_ms=500):
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    return bool(poller.poll(timeout_ms))


class TestLifecycle:
    def test_get_before_create_returns_none(self):
        assert PeriodicEventNotifier.get() is None

    def test_create_and_get(self):
        PeriodicEventNotifier.create(interval_ms=50, use_eventfd=HAS_EVENTFD)
        assert PeriodicEventNotifier.get() is not None

    def test_shutdown_clears_instance(self):
        PeriodicEventNotifier.create(interval_ms=50, use_eventfd=HAS_EVENTFD)
        PeriodicEventNotifier.shutdown()
        assert PeriodicEventNotifier.get() is None

    def test_double_create_idempotent(self):
        PeriodicEventNotifier.create(interval_ms=50, use_eventfd=HAS_EVENTFD)
        PeriodicEventNotifier.create(interval_ms=100, use_eventfd=HAS_EVENTFD)
        assert PeriodicEventNotifier.get() is not None

    def test_double_shutdown_idempotent(self):
        PeriodicEventNotifier.create(interval_ms=50, use_eventfd=HAS_EVENTFD)
        PeriodicEventNotifier.shutdown()
        PeriodicEventNotifier.shutdown()

    def test_shutdown_before_create(self):
        PeriodicEventNotifier.shutdown()


class TestSignaling:
    def test_register_fd_triggers_signals(self):
        PeriodicEventNotifier.create(interval_ms=10, use_eventfd=HAS_EVENTFD)
        poll_fd, write_fd, close_fn = _make_notifier_fd()
        try:
            notifier = PeriodicEventNotifier.get()
            notifier.register_fd(write_fd)
            assert _poll_fd(poll_fd, timeout_ms=1000)
            _consume_fd(poll_fd)
        finally:
            notifier.unregister_fd(write_fd)
            close_fn()

    def test_multiple_fds_all_signaled(self):
        PeriodicEventNotifier.create(interval_ms=10, use_eventfd=HAS_EVENTFD)
        fds = [_make_notifier_fd() for _ in range(3)]
        try:
            notifier = PeriodicEventNotifier.get()
            for _, write_fd, _ in fds:
                notifier.register_fd(write_fd)
            time.sleep(0.05)
            for poll_fd, _, _ in fds:
                assert _poll_fd(poll_fd, timeout_ms=500)
                _consume_fd(poll_fd)
        finally:
            for _, write_fd, close_fn in fds:
                notifier.unregister_fd(write_fd)
                close_fn()

    def test_unregister_stops_signals(self):
        PeriodicEventNotifier.create(interval_ms=10, use_eventfd=HAS_EVENTFD)
        poll_fd, write_fd, close_fn = _make_notifier_fd()
        try:
            notifier = PeriodicEventNotifier.get()
            notifier.register_fd(write_fd)
            assert _poll_fd(poll_fd, timeout_ms=1000)
            _consume_fd(poll_fd)

            notifier.unregister_fd(write_fd)
            time.sleep(0.05)
            _consume_fd(poll_fd)
            assert not _poll_fd(poll_fd, timeout_ms=100)
        finally:
            close_fn()


class TestDormancy:
    def test_dormancy_and_wake(self):
        PeriodicEventNotifier.create(interval_ms=10, use_eventfd=HAS_EVENTFD)
        poll_fd, write_fd, close_fn = _make_notifier_fd()
        try:
            notifier = PeriodicEventNotifier.get()

            notifier.register_fd(write_fd)
            assert _poll_fd(poll_fd, timeout_ms=1000)
            _consume_fd(poll_fd)

            notifier.unregister_fd(write_fd)
            time.sleep(0.05)
            _consume_fd(poll_fd)
            assert not _poll_fd(poll_fd, timeout_ms=100)

            notifier.register_fd(write_fd)
            assert _poll_fd(poll_fd, timeout_ms=1000)
            _consume_fd(poll_fd)
        finally:
            notifier.unregister_fd(write_fd)
            close_fn()


class TestInterval:
    def test_set_interval_ms(self):
        PeriodicEventNotifier.create(interval_ms=10, use_eventfd=HAS_EVENTFD)
        poll_fd, write_fd, close_fn = _make_notifier_fd()
        try:
            notifier = PeriodicEventNotifier.get()
            notifier.register_fd(write_fd)

            # Drain initial burst and wait for steady state.
            time.sleep(0.05)
            _consume_fd(poll_fd)

            # Count signals at 10ms interval over 200ms → expect ~20.
            count_fast = 0
            deadline = time.monotonic() + 0.2
            while time.monotonic() < deadline:
                if _poll_fd(poll_fd, timeout_ms=50):
                    _consume_fd(poll_fd)
                    count_fast += 1

            notifier.set_interval_ms(200)
            time.sleep(0.25)
            _consume_fd(poll_fd)

            # Count signals at 200ms interval over 1s → expect ~5.
            count_slow = 0
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if _poll_fd(poll_fd, timeout_ms=250):
                    _consume_fd(poll_fd)
                    count_slow += 1

            assert count_fast > count_slow
        finally:
            notifier.unregister_fd(write_fd)
            close_fn()


class TestEdgeCases:
    def test_register_same_fd_twice(self):
        PeriodicEventNotifier.create(interval_ms=10, use_eventfd=HAS_EVENTFD)
        poll_fd, write_fd, close_fn = _make_notifier_fd()
        try:
            notifier = PeriodicEventNotifier.get()
            notifier.register_fd(write_fd)
            notifier.register_fd(write_fd)
            assert _poll_fd(poll_fd, timeout_ms=1000)
            _consume_fd(poll_fd)
        finally:
            notifier.unregister_fd(write_fd)
            close_fn()

    def test_unregister_nonexistent_fd(self):
        PeriodicEventNotifier.create(interval_ms=10, use_eventfd=HAS_EVENTFD)
        notifier = PeriodicEventNotifier.get()
        notifier.unregister_fd(999)

    def test_closed_fd_does_not_crash(self):
        PeriodicEventNotifier.create(interval_ms=10, use_eventfd=HAS_EVENTFD)
        poll_fd, write_fd, close_fn = _make_notifier_fd()
        notifier = PeriodicEventNotifier.get()
        notifier.register_fd(write_fd)
        close_fn()
        time.sleep(0.05)
        notifier.unregister_fd(write_fd)
