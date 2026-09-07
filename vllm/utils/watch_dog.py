# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import faulthandler
import os
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from logging import Logger

from vllm.config import WatchdogConfig


class WatchDog(ABC):
    @abstractmethod
    def feed(self):
        return NotImplementedError

    @abstractmethod
    def dump_stack(self, reason):
        return NotImplementedError

    @contextmanager
    def disable(self):
        """Context in which timeout detection is suspended. No-op by default."""
        yield


class WatchDogNoop(WatchDog):
    def feed(self):
        pass

    def dump_stack(self, reason):
        pass


class WatchDogRaw(WatchDog):
    def __init__(
        self,
        name: str,
        watchdog_config: WatchdogConfig,
        logger: Logger,
    ):
        """Initialize the watchdog."""
        self._name = name
        self._timeout = watchdog_config.timeout
        self._check_interval = watchdog_config.check_interval
        self._dump_dir = watchdog_config.dump_dir
        self._dump_file = os.path.join(
            self._dump_dir,
            f"VLLM_STACK_DUMP_for_{self._name}_{os.getpid()}.log",
        )
        self._logger = logger

        # Initialize feed time to current time to avoid immediate timeout on startup
        self._last_feed_time = time.monotonic()
        # Last timeout log time, initialized to 0 to ensure the first
        # timeout is always logged
        self._last_timeout_log = 0.0

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._dump_seq = 1
        self._dump_lock = threading.Lock()
        self._active = True

    def feed(self):
        """Feed interface, external callers use this to update last feed time"""
        # Single float assignment is atomic in CPython
        self._last_feed_time = time.monotonic()

    @contextmanager
    def disable(self):
        """Temporarily disable timeout detection during the wrapped block."""
        assert self._active, "WatchDogRaw.disable() is not re-entrant"
        self._active = False
        try:
            yield
        finally:
            assert not self._active
            self._active = True

    def dump_stack(self, reason):
        """Dump all thread stack traces to the dump file for the given reason."""
        try:
            with self._dump_lock:
                with open(self._dump_file, "a") as f:
                    f.write(
                        f"\n>>> Call stack dump #{self._dump_seq} at "
                        f"{time.ctime()} due to {reason}\n\n"
                    )
                    faulthandler.dump_traceback(file=f, all_threads=True)
                self._dump_seq += 1
                self._logger.info(
                    "[Watchdog]Dumped stack to %s due to %s",
                    self._dump_file,
                    reason,
                )
        except Exception as e:
            self._logger.warning(
                "[Watchdog]Failed to dump stack trace to %s: %s",
                self._dump_file,
                e,
            )

    def _check_loop(self):
        """Main loop of the background check thread"""
        while not self._stop_event.is_set():
            # Wait on the stop event so a stop request wakes the loop
            # immediately instead of sleeping the full check interval.
            if self._stop_event.wait(self._check_interval):
                break

            now = time.monotonic()
            # While disabled, treat the watchdog as continuously fed so the
            # timeout window restarts once the watchdog is re-enabled.
            if not self._active:
                self._last_feed_time = now
                continue
            # Calculate time elapsed since last feed
            if (
                now - self._last_feed_time > self._timeout
                and self._last_timeout_log < self._last_feed_time
            ):
                # If last timeout log time is earlier than last feed time,
                # the timeout hasn't been logged since the last feed; update
                # the log timestamp to avoid duplicate logs.
                self._last_timeout_log = now
                self.dump_stack("feed timeout")
            # If not timed out, do nothing and keep _last_timeout_log unchanged

    def start(self):
        """Start the watchdog background thread (daemon thread)"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self.feed()
        self._thread.start()
        self._logger.info(
            "[Watchdog]Started thread for %s (pid=%d)",
            self._name,
            os.getpid(),
        )

    def stop(self):
        """Stop the watchdog background thread"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if not self._thread.is_alive():
                # Only clear the reference once the thread has actually
                # exited; keeping it otherwise prevents a later start()
                # from creating a duplicate live monitor thread.
                self._thread = None


_watch_dog: WatchDog = WatchDogNoop()


def get_watch_dog() -> WatchDog:
    """Return the process-wide WatchDog singleton."""
    return _watch_dog


def start_watch_dog(
    name: str,
    watchdog_config: WatchdogConfig,
    logger: Logger,
) -> WatchDog:
    """Start the process-wide WatchDog background thread if enabled.

    The watchdog is enabled only when ``watchdog_config.dump_dir`` is set;
    otherwise it stays dormant. Returns the WatchDog singleton either way.
    """
    logger.info(
        "[Watchdog]Start with timeout=%s check_interval=%s dump_dir=%s",
        watchdog_config.timeout,
        watchdog_config.check_interval,
        watchdog_config.dump_dir,
    )

    global _watch_dog
    if not watchdog_config.dump_dir:
        return _watch_dog

    faulthandler.enable(all_threads=True)
    os.makedirs(watchdog_config.dump_dir, exist_ok=True)

    _watch_dog = WatchDogRaw(name, watchdog_config, logger)
    _watch_dog.start()
    return _watch_dog
